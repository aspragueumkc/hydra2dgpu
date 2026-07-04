"""
swe2d_backend.py
Python bridge for the native 2D SWE GPU solver (hydra_swe2d).

Usage example:
    from swe2d_backend import SWE2DBackend, BCType
    import numpy as np

    backend = SWE2DBackend()
    backend.build_mesh(node_x, node_y, node_z, cell_nodes)
    backend.initialize(h0, n_mann=0.030, cfl=0.45)
    diags = backend.run(t_end=3600.0, dt_request=1.0)
    h, hu, hv = backend.get_state()
"""

from __future__ import annotations

import logging
import os
import sys
import importlib
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

from swe2d.runtime.native_binding_compat import (
    call_solver_create_compat,
    log_feature_unavailable,
)

from swe2d import units as _u
from swe2d.extensions.extension_models import (
    BedFrictionModel,
    GodunovSolverMode,
    SolverModelOptions,
    SpatialDiscretization,
    TemporalScheme,
    TurbulenceModel,
)

# Ensure the native extension (.so) built under ./build/ is findable regardless
# of how Python was launched (QGIS, standalone terminal, pytest, etc.).
_here = os.path.dirname(os.path.abspath(__file__))
_plugin_root = os.path.abspath(os.path.join(_here, "..", ".."))
for _candidate in (
    os.path.join(_plugin_root, "build"),
    os.path.join(_plugin_root, "build", "Release"),
    os.path.join(_plugin_root, "build", "Debug"),
    os.path.join(_plugin_root, "lib"),
    os.path.join(_here, "build"),
    os.path.join(_here, "build", "Release"),
    os.path.join(_here, "build", "Debug"),
    os.path.join(_here, "lib"),
):
    if os.path.isdir(_candidate) and _candidate not in sys.path:
        sys.path.insert(0, _candidate)


def _platform_tag() -> str:
    """Return a platform tag matching the release ZIP naming convention."""
    import platform as _plat
    system = _plat.system().lower()
    machine = _plat.machine().lower()
    tag_map = {
        "x86_64": "x86_64",
        "amd64": "x86_64",
        "aarch64": "aarch64",
        "arm64": "arm64",
    }
    arch = tag_map.get(machine, machine)
    if system == "windows":
        system = "windows"
    return f"{system}-{arch}"

# ─────────────────────────────────────────────────────────────────────────────
# BCType constants (mirrored from native module; imported after module load)
# ─────────────────────────────────────────────────────────────────────────────
class BCType:
    """Boundary-condition type constants (mirrored from the native C++ module)."""
    INTERIOR = 0
    WALL     = 1
    INFLOW_Q = 2
    STAGE    = 3
    OPEN     = 4
    REFLECT  = 5
    NORMAL_DEPTH = 6


# ─────────────────────────────────────────────────────────────────────────────
# Module loader (GPU only — always loads hydra_swe2d)
# ─────────────────────────────────────────────────────────────────────────────
_swe2d_mod = None
_swe2d_load_error: Optional[str] = None
_swe2d_last_load_error: Optional[str] = None


def _load_swe2d_module():
    """Load and return the hydra_swe2d native module.

    Search order:
      1. Already loaded (cached).
      2. Standard Python import paths (including build/ dirs added above).
      3. Pre-compiled binary in ``lib/`` (release ZIP layout).
    """
    global _swe2d_last_load_error, _swe2d_mod, _swe2d_load_error
    if _swe2d_mod is not None:
        _swe2d_last_load_error = None
        _swe2d_load_error = None
        return _swe2d_mod

    # Build search path candidates (release layout places .so/.pyd in lib/)
    for _candidate in (
        os.path.join(_plugin_root, "lib"),
        os.path.join(_here, "lib"),
    ):
        if os.path.isdir(_candidate) and _candidate not in sys.path:
            sys.path.insert(0, _candidate)

    # ponytail: removed — project-root .so takes priority over build/ .so
    # and can silently load stale binaries without the latest bindings.

    # ── Windows: add plugin root to DLL search path ──────────────────────
    # This lets Python find cudart64_12.dll bundled alongside the .pyd files
    # without requiring a system-wide CUDA installation or manual PATH juggling.
    # Also check QSettings for a user-specified custom CUDA DLL path.
    if sys.platform == "win32":
        _custom_dll_dir = None
        try:
            from qgis.PyQt.QtCore import QSettings
            _s = QSettings("HYDRA2DGPU", "HYDRA2DGPU")
            _custom = _s.value("cuda_dll_path", "")
            if _custom and os.path.isdir(_custom):
                _custom_dll_dir = _custom
            elif _custom and os.path.isfile(_custom):
                _custom_dll_dir = os.path.dirname(_custom)
        except Exception:
            logger.warning("Unexpected error silently caught", exc_info=True)

        _dll_dirs_to_add = []
        if _custom_dll_dir:
            _dll_dirs_to_add.append(_custom_dll_dir)
        else:
            # Default: plugin root where cudart64_*.dll is bundled
            _dll_dirs_to_add.append(_plugin_root)
            # Fallback: lib/ dir (source-build layout)
            _lib_dir = os.path.join(_plugin_root, "lib")
            if os.path.isdir(_lib_dir):
                _dll_dirs_to_add.append(_lib_dir)

        for _d in _dll_dirs_to_add:
            if os.path.isdir(_d):
                if hasattr(os, "add_dll_directory"):
                    try:
                        os.add_dll_directory(_d)
                    except (OSError, FileNotFoundError):
                        pass
                else:
                    # Fallback for older Python: prepend to PATH
                    _old_path = os.environ.get("PATH", "")
                    if _d not in _old_path:
                        os.environ["PATH"] = _d + os.pathsep + _old_path

    try:
        mod = importlib.import_module("hydra_swe2d")
        _swe2d_mod = mod
        _swe2d_last_load_error = None
        _swe2d_load_error = None
        return mod
    except ImportError as e:
        err = (
            f"hydra_swe2d native module not found ({e}). "
            f"Platform: {_platform_tag()}. "
            "Either build from source (see README.md) or download the "
            "pre-compiled binary for your platform from "
            "https://github.com/aspragueumkc/hydra2dgpu/releases"
        )
        _swe2d_last_load_error = err
        _swe2d_load_error = err
        return None


def swe2d_available() -> bool:
    """Return True if the native 2D solver module is importable."""
    return _load_swe2d_module() is not None


def load_swe2d_native_module():
    """Load and return the hydra_swe2d native module, or None on failure."""
    return _load_swe2d_module()


def swe2d_gpu_available() -> bool:
    """Return True if the native module is loaded AND a CUDA device is present."""
    mod = _load_swe2d_module()
    if mod is None:
        return False
    try:
        return mod.swe2d_gpu_available()
    except Exception as exc:
        logger.warning("[BACKEND] GPU availability check failed: %s", exc)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# SWE2DBackend
# ─────────────────────────────────────────────────────────────────────────────
class SWE2DBackend:
    """
    High-level Python interface to the native hydra_swe2d GPU module.

    Lifecycle:
        1. Construct.
        2. build_mesh(...)  — must be called before initialize().
        3. initialize(...)  — creates native solver with initial conditions.
        4. step() or run()  — advance in time.
        5. get_state()      — retrieve current (h, hu, hv) numpy arrays.
        6. destroy()        — free native resources (or let GC handle it).
    """

    def __init__(self):
        mod = _load_swe2d_module()
        if mod is None:
            raise RuntimeError(
                f"{_swe2d_last_load_error} "
                "Build from source or download the pre-compiled binary "
                "from https://github.com/aspragueumkc/hydra2dgpu/releases"
            )
        self._mod = mod
        self._mesh_h   = None   # PyMesh handle
        self._solver_h = None   # PySolver handle
        self._n_cells  = 0
        self._boundary_edge_index_by_nodes = {}
        self._supports_solver_bc_update = log_feature_unavailable(
            self._mod, "swe2d_solver_set_boundary_values", logger,
        )
        self._supports_solver_hydrographs = log_feature_unavailable(
            self._mod, "swe2d_solver_set_boundary_hydrographs", logger,
        )
        self._supports_solver_rain_cn = log_feature_unavailable(
            self._mod, "swe2d_solver_set_rain_cn_forcing", logger,
        )
        self._supports_solver_external_sources = log_feature_unavailable(
            self._mod, "swe2d_solver_set_external_sources", logger,
        )
        self._h_min = 1.0e-6
        self._cell_area = np.empty(0, dtype=np.float64)
        self._cell_zb = np.empty(0, dtype=np.float64)
        self._mesh_node_x = np.empty(0, dtype=np.float64)
        self._mesh_node_y = np.empty(0, dtype=np.float64)
        self._mesh_node_z = np.empty(0, dtype=np.float64)
        self._mesh_cell_nodes = np.empty(0, dtype=np.int32)
        self._mesh_face_offsets = None
        self._bc_n0 = np.empty(0, dtype=np.int32)
        self._bc_n1 = np.empty(0, dtype=np.int32)
        self._bc_tp = np.empty(0, dtype=np.int32)
        self._bc_vl = np.empty(0, dtype=np.float64)
        self._tiny_mode = 1
        self._tiny_persistent_chunk_substeps = 8
        self._cell_perm = np.empty(0, dtype=np.int32)
        self._inv_cell_perm = np.empty(0, dtype=np.int32)

        # Last step diagnostics
        self._last_diag: Optional[dict] = None

    # ── Mesh ─────────────────────────────────────────────────────────────────

    def build_mesh(
        self,
        node_x: np.ndarray,
        node_y: np.ndarray,
        node_z: np.ndarray,
        cell_nodes: np.ndarray,
        bc_edge_node0: Optional[np.ndarray] = None,
        bc_edge_node1: Optional[np.ndarray] = None,
        bc_edge_type:  Optional[np.ndarray] = None,
        bc_edge_val:   Optional[np.ndarray] = None,
        cell_face_offsets: Optional[np.ndarray] = None,
    ) -> None:
        """
        Build the unstructured mesh.

        Parameters
        ----------
        node_x, node_y, node_z : array_like, shape (N,)
            Node coordinates and bed elevations (m).
        cell_nodes : array_like
            Either triangular node triplets (shape (M*3,) or (M, 3)) or,
            for polygon meshes, concatenated cell node rings referenced by
            `cell_face_offsets`.
        bc_edge_node0, bc_edge_node1 : array_like int32, shape (E,), optional
            Endpoint node indices for boundary edges with explicit BC.
        bc_edge_type : array_like int32, shape (E,), optional
            BCType value per specified boundary edge.
        bc_edge_val : array_like float64, shape (E,), optional
            Prescribed BC value (h or q) per boundary edge.
        cell_face_offsets : array_like int32, shape (M+1,), optional
            CSR offsets into `cell_nodes` for variable-vertex polygon cells.
            If provided, native polygon-cell build path is used.
        """
        node_x = np.ascontiguousarray(node_x, dtype=np.float64)
        node_y = np.ascontiguousarray(node_y, dtype=np.float64)
        node_z = np.ascontiguousarray(node_z, dtype=np.float64)

        cell_nodes_flat = np.ascontiguousarray(cell_nodes, dtype=np.int32).ravel()
        self._cell_area = np.empty(0, dtype=np.float64)
        self._cell_zb = np.empty(0, dtype=np.float64)

        # Empty BC arrays if not provided
        bc_n0  = np.empty(0, dtype=np.int32)
        bc_n1  = np.empty(0, dtype=np.int32)
        bc_tp  = np.empty(0, dtype=np.int32)
        bc_vl  = np.empty(0, dtype=np.float64)

        if bc_edge_node0 is not None:
            bc_n0 = np.ascontiguousarray(bc_edge_node0, dtype=np.int32)
            bc_n1 = np.ascontiguousarray(bc_edge_node1, dtype=np.int32)
            bc_tp = np.ascontiguousarray(bc_edge_type,  dtype=np.int32)
            bc_vl = np.ascontiguousarray(bc_edge_val,   dtype=np.float64)

        if cell_face_offsets is not None:
            face_offsets = np.ascontiguousarray(cell_face_offsets, dtype=np.int32).ravel()
            if face_offsets.size < 2:
                raise ValueError("cell_face_offsets must contain at least 2 entries")
            if int(face_offsets[-1]) != int(cell_nodes_flat.size):
                raise ValueError("cell_face_offsets[-1] must equal len(cell_nodes)")
            self._cell_area = np.zeros(face_offsets.size - 1, dtype=np.float64)
            self._cell_zb = np.zeros(face_offsets.size - 1, dtype=np.float64)
            for i in range(face_offsets.size - 1):
                s = int(face_offsets[i])
                e = int(face_offsets[i + 1])
                ids = cell_nodes_flat[s:e]
                if ids.size < 3:
                    continue
                xx = node_x[ids]
                yy = node_y[ids]
                self._cell_area[i] = 0.5 * abs(float(np.dot(xx, np.roll(yy, -1)) - np.dot(yy, np.roll(xx, -1))))
                self._cell_zb[i] = float(np.min(node_z[ids]))
            self._mesh_h = self._mod.swe2d_build_mesh_poly(
                node_x,
                node_y,
                node_z,
                face_offsets,
                cell_nodes_flat,
                bc_n0,
                bc_n1,
                bc_tp,
                bc_vl,
            )
        else:
            tris = cell_nodes_flat.reshape((-1, 3))
            x0 = node_x[tris[:, 0]]
            y0 = node_y[tris[:, 0]]
            x1 = node_x[tris[:, 1]]
            y1 = node_y[tris[:, 1]]
            x2 = node_x[tris[:, 2]]
            y2 = node_y[tris[:, 2]]
            self._cell_area = 0.5 * np.abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))
            self._cell_zb = np.min(node_z[cell_nodes_flat.reshape((-1, 3))], axis=1)
            self._mesh_h = self._mod.swe2d_build_mesh(
                node_x, node_y, node_z, cell_nodes_flat,
                bc_n0, bc_n1, bc_tp, bc_vl)

        info = self._mod.swe2d_mesh_info(self._mesh_h)
        self._n_cells = info["n_cells"]

        # Expose cell permutation (RCMK renumbering applied in C++ build_mesh).
        # cell_perm[c_new] = c_old. Empty if no renumbering.
        # Used in get_state() / get_max_tracking() to un-permute results.
        perm_arr = self._mod.swe2d_get_cell_perm(self._mesh_h)
        if perm_arr.size == self._n_cells:
            self._cell_perm = np.asarray(perm_arr, dtype=np.int32).ravel()
            # Inverse: inv_perm[c_old] = c_new
            self._inv_cell_perm = np.zeros(self._n_cells, dtype=np.int32)
            self._inv_cell_perm[self._cell_perm] = np.arange(self._n_cells, dtype=np.int32)
        else:
            self._cell_perm = np.empty(0, dtype=np.int32)
            self._inv_cell_perm = np.empty(0, dtype=np.int32)

        self._boundary_edge_index_by_nodes = {}
        self._boundary_edge_cells: Optional[np.ndarray] = None
        try:
            edge_idx, n0, n1, _, _, cell0 = self._mod.swe2d_boundary_edges(self._mesh_h)
            self._boundary_edge_cells = np.asarray(cell0, dtype=np.int32)
            for i in range(edge_idx.size):
                a = int(n0[i])
                b = int(n1[i])
                key = (a, b) if a < b else (b, a)
                self._boundary_edge_index_by_nodes[key] = int(edge_idx[i])
        except Exception:
            # Older binaries may not expose boundary-edge query; dynamic BC updates
            # will be unavailable in that case.
            self._boundary_edge_index_by_nodes = {}
            self._boundary_edge_cells = None

        self._mesh_node_x = np.asarray(node_x, dtype=np.float64)
        self._mesh_node_y = np.asarray(node_y, dtype=np.float64)
        self._mesh_node_z = np.asarray(node_z, dtype=np.float64)
        self._mesh_cell_nodes = np.asarray(cell_nodes_flat, dtype=np.int32)
        self._mesh_face_offsets = None
        if cell_face_offsets is not None:
            self._mesh_face_offsets = np.asarray(face_offsets, dtype=np.int32)
        self._bc_n0 = bc_n0
        self._bc_n1 = bc_n1
        self._bc_tp = bc_tp
        self._bc_vl = bc_vl

    def build_mesh_from_baked(
        self,
        baked_blob: bytes,
    ) -> None:
        """Build the mesh from a serialized baked BLOB (skip C++ builder).

        This is the headless-runner path: load a previously serialized mesh
        without re-running swe2d_build_mesh_poly, avoiding RCMK permutation
        differences and edge reordering variance.

        Parameters
        ----------
        baked_blob : bytes
            Serialized mesh blob from swe2d_serialize_mesh().
        """
        self._mesh_h = self._mod.swe2d_deserialize_mesh(baked_blob)
        info = self._mod.swe2d_mesh_info(self._mesh_h)
        self._n_cells = info["n_cells"]

        # Restore cell permutation
        perm_arr = self._mod.swe2d_get_cell_perm(self._mesh_h)
        if perm_arr.size == self._n_cells:
            self._cell_perm = np.asarray(perm_arr, dtype=np.int32).ravel()
            self._inv_cell_perm = np.zeros(self._n_cells, dtype=np.int32)
            self._inv_cell_perm[self._cell_perm] = np.arange(self._n_cells, dtype=np.int32)
        else:
            self._cell_perm = np.empty(0, dtype=np.int32)
            self._inv_cell_perm = np.empty(0, dtype=np.int32)

        # Restore mesh geometry arrays from accessor properties
        pm = self._mesh_h
        self._mesh_node_x = np.asarray(pm.node_x, dtype=np.float64)
        self._mesh_node_y = np.asarray(pm.node_y, dtype=np.float64)
        self._mesh_node_z = np.asarray(pm.node_z, dtype=np.float64)
        self._cell_zb = np.asarray(pm.cell_zb, dtype=np.float64)
        self._cell_area = np.asarray(pm.cell_area, dtype=np.float64)

        # Restore cell topology
        cfn = pm.cell_face_nodes
        self._mesh_cell_nodes = np.asarray(cfn, dtype=np.int32) if cfn is not None else np.empty(0, dtype=np.int32)
        cfo = pm.cell_face_offsets
        self._mesh_face_offsets = np.asarray(cfo, dtype=np.int32) if cfo is not None else None

        # Restore boundary edge index
        self._boundary_edge_index_by_nodes = {}
        self._boundary_edge_cells = None
        try:
            edge_idx, n0, n1, _, _, cell0 = self._mod.swe2d_boundary_edges(self._mesh_h)
            self._boundary_edge_cells = np.asarray(cell0, dtype=np.int32)
            for i in range(edge_idx.size):
                a = int(n0[i])
                b = int(n1[i])
                key = (a, b) if a < b else (b, a)
                self._boundary_edge_index_by_nodes[key] = int(edge_idx[i])
        except Exception:
            self._boundary_edge_index_by_nodes = {}
            self._boundary_edge_cells = None

    def boundary_edge_cells(self) -> Optional[np.ndarray]:
        """Return interior cell index for each boundary edge, or None."""
        return self._boundary_edge_cells

    def supports_dynamic_boundary_update(self) -> bool:
        """Check if dynamic boundary update is supported."""
        return bool(self._boundary_edge_index_by_nodes)

    def set_boundary_conditions(
        self,
        bc_edge_node0: np.ndarray,
        bc_edge_node1: np.ndarray,
        bc_edge_type: np.ndarray,
        bc_edge_val: np.ndarray,
    ) -> None:
        """Set boundary conditions."""
        if self._mesh_h is None:
            raise RuntimeError("build_mesh() must be called before set_boundary_conditions().")
        if not self._boundary_edge_index_by_nodes:
            raise RuntimeError("Dynamic boundary update not supported by current native module.")

        n0 = np.ascontiguousarray(bc_edge_node0, dtype=np.int32).ravel()
        n1 = np.ascontiguousarray(bc_edge_node1, dtype=np.int32).ravel()
        tp = np.ascontiguousarray(bc_edge_type, dtype=np.int32).ravel()
        vl = np.ascontiguousarray(bc_edge_val, dtype=np.float64).ravel()
        if not (n0.size == n1.size == tp.size == vl.size):
            raise ValueError("bc edge arrays must have the same length")

        edge_index = np.empty(n0.size, dtype=np.int32)
        for i in range(n0.size):
            a = int(n0[i])
            b = int(n1[i])
            key = (a, b) if a < b else (b, a)
            if key not in self._boundary_edge_index_by_nodes:
                raise ValueError(f"Boundary edge ({a}, {b}) not found in mesh")
            edge_index[i] = self._boundary_edge_index_by_nodes[key]

        if self._solver_h is not None and self._supports_solver_bc_update:
            self._mod.swe2d_solver_set_boundary_values(self._solver_h, edge_index, tp, vl)
        else:
            self._mod.swe2d_set_boundary_values(self._mesh_h, edge_index, tp, vl)

    def set_boundary_hydrographs_native(
        self,
        edge_index: np.ndarray,
        bc_type: np.ndarray,
        offsets: np.ndarray,
        time_s: np.ndarray,
        value: np.ndarray,
    ) -> None:
        """Set boundary hydrographs native."""
        if self._solver_h is None:
            raise RuntimeError("initialize() must be called before set_boundary_hydrographs_native().")
        if not self._supports_solver_hydrographs:
            raise RuntimeError("Native boundary hydrograph API not supported by current module.")
        e = np.ascontiguousarray(edge_index, dtype=np.int32).ravel()
        t = np.ascontiguousarray(bc_type, dtype=np.int32).ravel()
        o = np.ascontiguousarray(offsets, dtype=np.int32).ravel()
        ts = np.ascontiguousarray(time_s, dtype=np.float64).ravel()
        v = np.ascontiguousarray(value, dtype=np.float64).ravel()
        if e.size != t.size:
            raise ValueError("edge_index and bc_type must have same length")
        if o.size != e.size + 1:
            raise ValueError("offsets length must be n_edges + 1")
        if ts.size != v.size:
            raise ValueError("time_s and value must have same length")
        self._mod.swe2d_solver_set_boundary_hydrographs(self._solver_h, e, t, o, ts, v)

    def set_progressive_bc_data(
        self,
        n_groups: int,
        n_edges_total: int,
        group_offsets: np.ndarray,
        edge_hg_idx: np.ndarray,
        edge_len: np.ndarray,
        edge_cum_len: np.ndarray,
        group_peak_q: np.ndarray,
        group_total_len: np.ndarray,
    ) -> None:
        """Set progressive bc data."""
        if self._solver_h is None:
            raise RuntimeError("initialize() must be called before set_progressive_bc_data().")
        if not self._supports_solver_hydrographs:
            raise RuntimeError("Native boundary hydrograph API not supported by current module.")
        if not hasattr(self._mod, "swe2d_solver_set_progressive_bc_data"):
            raise RuntimeError("Progressive BC data API not supported by current module.")
        go = np.ascontiguousarray(group_offsets, dtype=np.int32).ravel()
        ehi = np.ascontiguousarray(edge_hg_idx, dtype=np.int32).ravel()
        el = np.ascontiguousarray(edge_len, dtype=np.float64).ravel()
        ecl = np.ascontiguousarray(edge_cum_len, dtype=np.float64).ravel()
        gpq = np.ascontiguousarray(group_peak_q, dtype=np.float64).ravel()
        gtl = np.ascontiguousarray(group_total_len, dtype=np.float64).ravel()
        if go.size != n_groups + 1:
            raise ValueError("group_offsets length must be n_groups + 1")
        if ehi.size != n_edges_total or el.size != n_edges_total or ecl.size != n_edges_total:
            raise ValueError("edge arrays must have n_edges_total elements")
        if gpq.size != n_groups or gtl.size != n_groups:
            raise ValueError("group arrays must have n_groups elements")
        self._mod.swe2d_solver_set_progressive_bc_data(
            self._solver_h, n_groups, n_edges_total,
            go, ehi, el, ecl, gpq, gtl,
        )

    def set_rain_cn_forcing_native(
        self,
        cell_gage_idx: np.ndarray,
        gage_offsets: np.ndarray,
        hg_time_s: np.ndarray,
        hg_cum_mm: np.ndarray,
        cn: np.ndarray,
        ia_ratio: float = 0.2,
        mm_to_model_depth: float = 1.0e-3,
        rain_update_interval_s: float = 60.0,
    ) -> None:
        """Set rain cn forcing native."""
        if self._solver_h is None:
            raise RuntimeError("initialize() must be called before set_rain_cn_forcing_native().")
        if not self._supports_solver_rain_cn:
            raise RuntimeError("Native rain+CN forcing API not supported by current module.")
        cg = np.ascontiguousarray(cell_gage_idx, dtype=np.int32).ravel()
        go = np.ascontiguousarray(gage_offsets, dtype=np.int32).ravel()
        ts = np.ascontiguousarray(hg_time_s, dtype=np.float64).ravel()
        cr = np.ascontiguousarray(hg_cum_mm, dtype=np.float64).ravel()
        cna = np.ascontiguousarray(cn, dtype=np.float64).ravel()
        if cg.size != cna.size:
            raise ValueError("cell_gage_idx and cn must have same length")
        if go.size < 2:
            raise ValueError("gage_offsets must have at least two entries")
        if ts.size != cr.size:
            raise ValueError("hg_time_s and hg_cum_mm must have same length")
        self._mod.swe2d_solver_set_rain_cn_forcing(
            self._solver_h,
            cg,
            go,
            ts,
            cr,
            cna,
            float(ia_ratio),
            float(mm_to_model_depth),
            float(rain_update_interval_s),
        )

    def set_external_sources_native(self, source_rate_mps: Optional[np.ndarray]) -> None:
        """Set external per-cell depth source rates [m/s] directly on native solver.

        This API is designed for device-resident coupling workflows: source
        terms are uploaded once per step and consumed inside the native update
        kernel, avoiding full h/hu/hv host->device state round-trips.
        Passing None clears the external source field.
        """
        if self._solver_h is None:
            raise RuntimeError("initialize() must be called before set_external_sources_native().")
        if not self._supports_solver_external_sources:
            raise RuntimeError("Native external source API not supported by current module.")
        if source_rate_mps is None:
            self._mod.swe2d_solver_set_external_sources(self._solver_h, None)
            return
        src = np.ascontiguousarray(source_rate_mps, dtype=np.float64).ravel()
        if src.size != self._n_cells:
            raise ValueError("source_rate_mps length must equal n_cells")
        self._mod.swe2d_solver_set_external_sources(self._solver_h, src)

    def accumulate_external_sources_native(self, source_rate_mps: np.ndarray) -> None:
        """Accumulate per-cell depth source rates into the on-device source buffer.

        Unlike set_external_sources_native which overwrites, this ADDS the given
        rates to whatever is already in d_external_source_mps on the GPU.
        Uses a H2D upload + GPU kernel — no D2H readback.
        """
        if self._solver_h is None:
            raise RuntimeError("initialize() must be called before accumulate_external_sources_native().")
        if not self._supports_solver_external_sources:
            raise RuntimeError("Native external source API not supported by current module.")
        src = np.ascontiguousarray(source_rate_mps, dtype=np.float64).ravel()
        if src.size != self._n_cells:
            raise ValueError("source_rate_mps length must equal n_cells")
        self._mod.swe2d_gpu_accumulate_external_source(self._solver_h, src)

    # ── Snapshot ring buffer ─────────────────────────────────────────────────
    # Device ring buffer: snapshots stay on GPU during the run.
    # Auto-dump to host when device free memory drops below a safety margin
    # (to prevent OOM crashes).  Bulk readback on explicit request.
    # The memory margin is 4× the per-snapshot size plus a 256 MB buffer.

    _snap_auto_dump_margin_mult = 4       # dump when free_mem < 4 * snap_size + margin_bytes
    _snap_auto_dump_margin_bytes = 256 * 1024 * 1024  # 256 MB headroom
    _snap_host_buffer: Optional[Dict[str, list]] = None  # auto-dumped host data

    def _snap_per_snapshot_bytes(self) -> int:
        """Size of one snapshot on device (3 × n_cells × 8 bytes + 8 bytes for time)."""
        nc = int(getattr(self, "_n_cells", 0))
        return (3 * nc + 1) * 8 if nc > 0 else 0

    def _snap_should_auto_dump(self) -> bool:
        """Check actual GPU free memory — dump if below safety margin."""
        if not hasattr(self._mod, "swe2d_gpu_device_memory_info"):
            return False
        try:
            info = self._mod.swe2d_gpu_device_memory_info()
            free_bytes = int(info.get("free_bytes", 0))
            if free_bytes <= 0:
                return False
            snap_sz = self._snap_per_snapshot_bytes()
            if snap_sz <= 0:
                return False
            threshold = self._snap_auto_dump_margin_mult * snap_sz + self._snap_auto_dump_margin_bytes
            return free_bytes < threshold
        except Exception:
            return False

    def store_snapshot(self, t_s: float) -> None:
        """Copy current h/hu/hv to the next snapshot slot on the device ring buffer.

        No D2H transfer during normal operation — device-only D2D copy on the
        compute stream.  When GPU free memory drops below the safety margin,
        an automatic bulk D2H readback + reset is triggered to prevent OOM.
        Call at each output interval instead of get_state().
        """
        if self._solver_h is None:
            raise RuntimeError("initialize() must be called before store_snapshot().")
        if not hasattr(self._mod, "swe2d_gpu_store_snapshot"):
            raise RuntimeError("swe2d_gpu_store_snapshot not available in native module.")
        # Auto-dump: check actual GPU memory pressure, not snapshot count.
        if self._snap_should_auto_dump():
            self._auto_dump_snapshots()
        self._mod.swe2d_gpu_store_snapshot(self._solver_h, float(t_s))

    def _auto_dump_snapshots(self) -> None:
        """Drain device snapshots to host buffer to prevent OOM.

        Stores data in solver (RCMK) order.  The permutation to original
        (pre-RCMK) order is applied once in :meth:`read_snapshots`.
        """
        raw = self._mod.swe2d_gpu_read_snapshots(self._solver_h)
        if not raw or "t_s" not in raw:
            return
        ts = np.asarray(raw["t_s"], dtype=np.float64)
        h_arr  = np.asarray(raw["h"],  dtype=np.float64)
        hu_arr = np.asarray(raw["hu"], dtype=np.float64)
        hv_arr = np.asarray(raw["hv"], dtype=np.float64)
        if self._snap_host_buffer is None:
            self._snap_host_buffer = {"t_s": [], "h": [], "hu": [], "hv": []}
        buf = self._snap_host_buffer
        for si in range(ts.shape[0]):
            buf["t_s"].append(float(ts[si]))
            buf["h"].append(np.ascontiguousarray(h_arr[si, :]))
            buf["hu"].append(np.ascontiguousarray(hu_arr[si, :]))
            buf["hv"].append(np.ascontiguousarray(hv_arr[si, :]))

    def read_snapshots(self) -> Optional[Dict[str, np.ndarray]]:
        """Read all accumulated snapshots from device + host to host.

        Returns a dict with keys 't_s' (shape [N]), 'h'/'hu'/'hv'
        (shape [N, n_cells]) or None if no snapshots accumulated.
        Consumes both the host auto-dump buffer and the device ring
        buffer — subsequent calls only return snapshots written after
        this read. Call free_snapshot_buf() explicitly when a hard reset
        is needed (e.g. before starting a new simulation).
        """
        if self._solver_h is None:
            return None
        if not hasattr(self._mod, "swe2d_gpu_read_snapshots"):
            return None
        # Read device-side snapshots
        raw = self._mod.swe2d_gpu_read_snapshots(self._solver_h)
        dev_ts = np.asarray(raw["t_s"], dtype=np.float64) if raw and "t_s" in raw else np.empty(0, dtype=np.float64)
        dev_h  = np.asarray(raw["h"],  dtype=np.float64)  if raw and "h" in raw  else np.empty((0, 0), dtype=np.float64)
        dev_hu = np.asarray(raw["hu"], dtype=np.float64)  if raw and "hu" in raw else np.empty((0, 0), dtype=np.float64)
        dev_hv = np.asarray(raw["hv"], dtype=np.float64)  if raw and "hv" in raw else np.empty((0, 0), dtype=np.float64)
        # Merge with host buffer
        hb = self._snap_host_buffer
        self._snap_host_buffer = None  # consume
        if hb is None and dev_ts.size == 0:
            return None
        n_dev = int(dev_ts.shape[0])
        n_host = len(hb["t_s"]) if hb else 0
        n_total = n_dev + n_host
        if n_total == 0:
            return None
        n_cells = int(dev_h.shape[1]) if dev_h.ndim >= 2 and dev_h.shape[1] > 0 else \
                  (hb["h"][0].shape[0] if hb and hb["h"] else 0)
        out_ts  = np.empty(n_total, dtype=np.float64)
        out_h   = np.empty((n_total, n_cells), dtype=np.float64)
        out_hu  = np.empty((n_total, n_cells), dtype=np.float64)
        out_hv  = np.empty((n_total, n_cells), dtype=np.float64)
        idx = 0
        if hb:
            for si in range(n_host):
                out_ts[idx]  = float(hb["t_s"][si])
                out_h[idx]   = np.asarray(hb["h"][si],  dtype=np.float64)
                out_hu[idx]  = np.asarray(hb["hu"][si], dtype=np.float64)
                out_hv[idx]  = np.asarray(hb["hv"][si], dtype=np.float64)
                idx += 1
        if n_dev > 0:
            out_ts[idx:]  = dev_ts[:]
            out_h[idx:]   = dev_h[:]
            out_hu[idx:]  = dev_hu[:]
            out_hv[idx:]  = dev_hv[:]
        # Per the baked BLOB spec (§5.12): data stays in solver (RCMK) order.
        # Mesh geometry from swe2d_deserialize_mesh and results from
        # this function share the same ordering — no permutation needed.
        result = {"t_s": out_ts, "h": out_h, "hu": out_hu, "hv": out_hv}
        self.free_snapshot_buf()
        return result

    def free_snapshot_buf(self) -> None:
        """Free the device snapshot ring buffer and host buffer."""
        if self._solver_h is None:
            return
        if hasattr(self._mod, "swe2d_gpu_free_snapshot_buf"):
            self._mod.swe2d_gpu_free_snapshot_buf(self._solver_h)
        self._snap_host_buffer = None

    # ── Solver init ──────────────────────────────────────────────────────────

    def initialize(
        self,
        h0: np.ndarray,
        hu0: Optional[np.ndarray] = None,
        hv0: Optional[np.ndarray] = None,
        n_mann_cell: Optional[np.ndarray] = None,
        g:        float = _u.gravity(),
        k_mann:   float = 1.0,
        n_mann:   float = 0.035,
        h_min:    float = 1.0e-6,
        cfl:      float = 0.45,
        dt_max:   float = 10.0,
        dt_fixed: float = -1.0,
        dt_initial: float = -1.0,
        max_inv_area: float = 1.0e6,
        cfl_lambda_cap: float = 1.0e6,
        momentum_cap_min_speed: float = 50.0,
        momentum_cap_celerity_mult: float = 20.0,
        depth_cap: float = 1.0e6,
        max_rel_depth_increase: float = 2.0,
        shallow_damping_depth: float = 1.0e-4,
        extreme_rain_mode: bool = False,
        source_cfl_beta: float = 0.25,
        source_max_substeps: int = 16,
        source_rate_cap: float = 0.0,
        source_depth_step_cap: float = 0.0,
        source_true_subcycling: bool = False,
        source_imex_split: bool = False,
        enable_shallow_front_recon_fallback: bool = True,
        gpu_diag_sync_interval_steps: int = 50,  # Production-friendly: reduce host sync overhead. Set 1 for high-frequency monitoring.
        tiny_mode: int = 1,
        tiny_cell_threshold: int = 8000,
        tiny_edge_threshold: int = 24000,
        tiny_wet_cell_threshold: int = 2000,
        tiny_persistent_chunk_substeps: int = 8,
        tiny_active_compaction_stride_steps: int = 8,
        tiny_enable_active_compaction: bool = True,
        n_threads: int  = 0,
        temporal_scheme: TemporalScheme = TemporalScheme.SSP_RK2,
        spatial_discretization: SpatialDiscretization = SpatialDiscretization.FV_FIRST_ORDER,
        turbulence_model: TurbulenceModel = TurbulenceModel.NONE,
        bed_friction_model: BedFrictionModel = BedFrictionModel.MANNING,
        model_options: Optional[SolverModelOptions] = None,
        degen_mode: int = 0,
        front_flux_damping: float = 0.5,
        active_set_hysteresis: bool = True,
        friction_substep_enabled: bool = True,
        friction_target_courant: float = 1.0,
        friction_max_substeps: int = 64,
        shallow_friction_correction: bool = True,
        shallow_friction_depth_alpha: float = 5.0,
        shallow_friction_exponent: float = 0.4,
    ) -> None:
        """
        Create the solver with initial conditions.

        Parameters
        ----------
        h0 : array_like float64, shape (M,)
            Initial water depth per cell (m).  Must be >= 0.
        hu0, hv0 : array_like float64, shape (M,), optional
            Initial x- and y-momentum per cell (m²/s).  Default zeros.
        n_mann_cell : array_like float64, shape (M,), optional
            Spatial Manning roughness values per cell.  If provided, it overrides
            global n_mann on a per-cell basis.
        g : float
            Gravitational acceleration (m/s²).
        n_mann : float
            Global Manning's roughness coefficient (m^{-1/3} s).
        h_min : float
            Wet/dry threshold (m).
        cfl : float
            CFL safety factor for explicit timestep.
        dt_max : float
            Maximum timestep (s).
        dt_fixed : float
            If > 0, override CFL with this fixed dt.
        dt_initial : float
            If > 0, use this dt for the first step only (cold-start override).
            Useful for CFL adaptive stepping on dry domains where lambda_max=0
            causes compute_cfl_dt() to return dt_max.
        max_inv_area : float
            Cap on 1/area used by GPU flux/update kernels for tiny cells.
        cfl_lambda_cap : float
            Cap on local CFL lambda used for diagnostic and dt reduction.
        momentum_cap_min_speed : float
            Minimum speed bound used for momentum clipping.
        momentum_cap_celerity_mult : float
            Multiplier for sqrt(g*h) in momentum clipping speed bound.
        depth_cap : float
            Absolute depth ceiling for robustness.
        max_rel_depth_increase : float
            Per-step limiter on depth increase: h <= h_old + rel*max(h_old,h_min).
        shallow_damping_depth : float
            Depth below which momentum is smoothly damped toward zero.
        enable_shallow_front_recon_fallback : bool
            If True, force first-order reconstruction on shallow edge pairs near
            advancing wet/dry fronts as a stability control.
        gpu_diag_sync_interval_steps : int
            GPU host-sync diagnostics cadence. 1=every step, N=every N steps,
            <=0 disables per-step host diagnostic sync.
        n_threads : int
            CPU thread count (0 = auto).
        temporal_scheme : TemporalScheme
            Temporal integrator selection (default SSP_RK2).
        spatial_discretization : SpatialDiscretization
            Spatial scheme selector (currently scaffolded).
        godunov_mode : GodunovSolverMode
            Selects the current GPU path or the Godunov rollout mode.
        turbulence_model : TurbulenceModel
            Turbulence closure selector (currently scaffolded).
        bed_friction_model : BedFrictionModel
            Bed friction law selector (currently scaffolded).
        model_options : SolverModelOptions, optional
            Composite extension config (rain, drainage, hydraulic structures).
        """
        if self._mesh_h is None:
            raise RuntimeError("build_mesh() must be called before initialize().")

        h0_arr = np.ascontiguousarray(h0, dtype=np.float64)
        hu0_arr = np.ascontiguousarray(hu0, dtype=np.float64) if hu0 is not None else None
        hv0_arr = np.ascontiguousarray(hv0, dtype=np.float64) if hv0 is not None else None
        n_mann_cell_arr = np.ascontiguousarray(n_mann_cell, dtype=np.float64) if n_mann_cell is not None else None

        if self._solver_h is not None:
            self._mod.swe2d_destroy(self._solver_h)

        native_opts: Dict[str, object] = {
            "temporal_order": int(temporal_scheme),
            "spatial_scheme": int(spatial_discretization),
            "turbulence_model": int(turbulence_model),
            "bed_friction_model": int(bed_friction_model),
            "enable_rain_module": False,
            "enable_pipe_network_module": False,
            "enable_hydraulic_structures": False,
            "friction_substep_enabled": friction_substep_enabled,
            "friction_target_courant": friction_target_courant,
            "friction_max_substeps": friction_max_substeps,
            "shallow_friction_correction": shallow_friction_correction,
            "shallow_friction_depth_alpha": shallow_friction_depth_alpha,
            "shallow_friction_exponent": shallow_friction_exponent,
        }
        if model_options is not None:
            native_opts.update(model_options.to_native_dict())

        self._solver_h = call_solver_create_compat(self._mod,
            self._mesh_h,
            h0_arr, hu0_arr, hv0_arr, n_mann_cell_arr,
            g=g, k_mann=k_mann, n_mann=n_mann, h_min=h_min,
            cfl=cfl, dt_max=dt_max, dt_fixed=dt_fixed, dt_initial=dt_initial,
            max_inv_area=max_inv_area,
            cfl_lambda_cap=cfl_lambda_cap,
            momentum_cap_min_speed=momentum_cap_min_speed,
            momentum_cap_celerity_mult=momentum_cap_celerity_mult,
            depth_cap=depth_cap,
            max_rel_depth_increase=max_rel_depth_increase,
            shallow_damping_depth=shallow_damping_depth,
            extreme_rain_mode=bool(extreme_rain_mode),
            source_cfl_beta=float(source_cfl_beta),
            source_max_substeps=int(source_max_substeps),
            source_rate_cap=float(source_rate_cap),
            source_depth_step_cap=float(source_depth_step_cap),
            source_true_subcycling=bool(source_true_subcycling),
            source_imex_split=bool(source_imex_split),
            enable_shallow_front_recon_fallback=bool(enable_shallow_front_recon_fallback),
            gpu_diag_sync_interval_steps=int(gpu_diag_sync_interval_steps),
            tiny_mode=int(tiny_mode),
            tiny_cell_threshold=int(tiny_cell_threshold),
            tiny_edge_threshold=int(tiny_edge_threshold),
            tiny_wet_cell_threshold=int(tiny_wet_cell_threshold),
            tiny_persistent_chunk_substeps=int(tiny_persistent_chunk_substeps),
            tiny_active_compaction_stride_steps=int(tiny_active_compaction_stride_steps),
            tiny_enable_active_compaction=bool(tiny_enable_active_compaction),
            use_gpu=True, n_threads=n_threads,
            temporal_order=int(native_opts["temporal_order"]),
            spatial_scheme=int(native_opts["spatial_scheme"]),
            turbulence_model=int(native_opts["turbulence_model"]),
            bed_friction_model=int(native_opts["bed_friction_model"]),
            enable_rain_module=bool(native_opts["enable_rain_module"]),
            enable_pipe_network_module=bool(native_opts["enable_pipe_network_module"]),
            enable_hydraulic_structures=bool(native_opts["enable_hydraulic_structures"]),
            friction_substep_enabled=bool(native_opts["friction_substep_enabled"]),
            friction_target_courant=float(native_opts["friction_target_courant"]),
            friction_max_substeps=int(native_opts["friction_max_substeps"]),
            shallow_friction_correction=bool(native_opts["shallow_friction_correction"]),
            shallow_friction_depth_alpha=float(native_opts["shallow_friction_depth_alpha"]),
            shallow_friction_exponent=float(native_opts["shallow_friction_exponent"]),
            degen_mode=int(degen_mode),
            front_flux_damping=float(front_flux_damping),
            active_set_hysteresis=bool(active_set_hysteresis),
        )
        self._tiny_mode = int(tiny_mode)
        self._tiny_persistent_chunk_substeps = int(tiny_persistent_chunk_substeps)
        self._h_min = float(h_min)

    # ── Stepping ─────────────────────────────────────────────────────────────

    def step(self, dt_request: float = -1.0) -> dict:
        """
        Advance one timestep.

        Parameters
        ----------
        dt_request : float
            Requested timestep (s).  Pass -1 (default) for CFL-controlled dt.

        Returns
        -------
        dict with keys: dt, wet_cells, max_depth, min_depth, mass_total,
        max_courant, max_depth_residual, max_wse_elev_error, gpu_active
        """
        if self._solver_h is None:
            raise RuntimeError("initialize() must be called before step().")
        diag = self._mod.swe2d_step(self._solver_h, dt_request)
        self._last_diag = diag
        return diag

    def run(
        self,
        t_end: float,
        dt_request: float = -1.0,
        progress_callback: Optional[Callable[[float, dict], None]] = None,
        cancel_check:      Optional[Callable[[], bool]] = None,
        source_rate_callback: Optional[Callable[[float, float, np.ndarray, np.ndarray, np.ndarray], Optional[np.ndarray]]] = None,
        use_native_source_injection: bool = True,  # Production default: keep state device-resident.
    ) -> List[dict]:
        """
        Run the solver to t_end.

        Parameters
        ----------
        t_end : float
            Simulation end time (s).
        dt_request : float
            Requested timestep per step (-1 = CFL-controlled).
        progress_callback : callable(t, diag), optional
            Called after each step with current time and diagnostics.
        cancel_check : callable() -> bool, optional
            If provided, called each step; run stops early if True returned.
        source_rate_callback : callable(t, dt, h, hu, hv) -> ndarray, optional
            Optional coupled-source hook called after each native step.
            Return per-cell depth source rates [L/T]. Returned array must have
            length n_cells. Positive values add depth, negative values remove
            depth. Depth is clipped to >=0 and momentum is zeroed in dry cells.
        use_native_source_injection : bool, default False
            If True, source_rate_callback output is uploaded via
            set_external_sources_native() and consumed directly by native step
            updates. This keeps state device-resident (no per-step set_state
            round-trip). In adaptive-CFL mode this applies with one-step lag
            because dt is only known after each native step.

        Returns
        -------
        List of per-step diagnostic dicts.
        """
        if self._solver_h is None:
            raise RuntimeError("initialize() must be called before run().")

        # Determine if we can use the native run-to-time API.
        # Native run is available when: no source_rate_callback (uncoupled case).
        has_native_run = hasattr(self._mod, "swe2d_run_to_time") and source_rate_callback is None

        diags: List[dict] = []

        if has_native_run:
            diag_batch_size = 0
            if int(self._tiny_mode) == 3:
                diag_batch_size = max(1, int(self._tiny_persistent_chunk_substeps))

            # Use native run-to-time API: eliminates per-step Python orchestration.
            # Returns dict with keys: 'diags', 'steps_completed', 'cancelled', 'final_time'
            result = self._mod.swe2d_run_to_time(
                self._solver_h,
                t_end,
                dt_request,
                diag_batch_size
            )

            # Result is a dict with 'diags' (list of batched diagnostics), 'steps_completed',
            # 'cancelled', and 'final_time'. For the zero batch case, diags will typically be empty
            # but we get significant speedup from eliminating Python loop overhead.
            diags = result.get("diags") or []
            self._last_diag = None

            # Call progress_callback once at the end if provided.
            if progress_callback and result.get("steps_completed", 0) > 0:
                final_time = result.get("final_time", t_end)

                # Prefer the last real diagnostic if available to avoid synthesizing
                # an incomplete or misleading diag.
                final_diag = None
                if diags:
                    # Flatten possible batches and use the last diagnostic entry
                    last_batch = diags[-1]
                    if isinstance(last_batch, (list, tuple)) and last_batch:
                        final_diag = last_batch[-1]
                    else:
                        final_diag = last_batch

                if final_diag is None:
                    # No real diag available; provide a clearly marked summary object
                    # so callers can distinguish it from a normal diagnostic entry.
                    final_diag = {
                        "summary": True,
                        "type": "final_run_summary",
                        "time": final_time,
                        "steps_completed": result.get("steps_completed", 0),
                        "cancelled": result.get("cancelled", False),
                        "dt_request": dt_request,
                    }

                progress_callback(final_time, final_diag)
            # Result is a dict with 'diags' (list of batched diagnostics), 'steps_completed',
            # 'cancelled', and 'final_time'. For the zero batch case, diags will be empty
            # but we get significant speedup from eliminating Python loop overhead.
            diags = result.get("diags", [])
            self._last_diag = None
            
            # Call progress_callback once at the end if provided.
            if progress_callback and result.get("steps_completed", 0) > 0:
                # Construct a final diagnostic dict with cumulative information.
                final_time = result.get("final_time", t_end)
                final_diag = {"dt": dt_request, "time": final_time}
                progress_callback(final_time, final_diag)
            
            return diags

        # GPU-only orchestration loop (coupling applied on device per step)
        coupling_controller = None
        if source_rate_callback is not None:
            coupling_controller = getattr(source_rate_callback, "__self__", None)
            if not hasattr(coupling_controller, "apply_native_device_sources"):
                raise RuntimeError(
                    "source_rate_callback.__self__ must have apply_native_device_sources. "
                    "CPU fallback removed — all coupling goes through GPU."
                )

        t = 0.0
        while t < t_end:
            if cancel_check and cancel_check():
                break
            diag = self.step(dt_request)
            dt = float(diag["dt"])
            if coupling_controller is not None and dt > 0.0:
                coupling_controller.apply_native_device_sources(t, dt)
            t += dt
            diags.append(diag)
            if progress_callback:
                progress_callback(t, diag)

        return diags

    # ── State retrieval ───────────────────────────────────────────────────────

    def get_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return current (h, hu, hv) numpy arrays, each shape (M,) float64.

        Results are un-permuted back to original (pre-RCMK) cell order.
        """
        if self._solver_h is None:
            raise RuntimeError("initialize() must be called before get_state().")
        h, hu, hv = self._mod.swe2d_get_state(self._solver_h)
        if self._inv_cell_perm.size > 0:
            h = h[self._inv_cell_perm]
            hu = hu[self._inv_cell_perm]
            hv = hv[self._inv_cell_perm]
        return (h, hu, hv)

    def set_state(self, h: np.ndarray, hu: np.ndarray, hv: np.ndarray) -> None:
        """Overwrite current (h, hu, hv) state arrays."""
        if self._solver_h is None:
            raise RuntimeError("initialize() must be called before set_state().")
        h_arr = np.ascontiguousarray(h, dtype=np.float64)
        hu_arr = np.ascontiguousarray(hu, dtype=np.float64)
        hv_arr = np.ascontiguousarray(hv, dtype=np.float64)
        if h_arr.size != self._n_cells or hu_arr.size != self._n_cells or hv_arr.size != self._n_cells:
            raise ValueError("h/hu/hv lengths must all equal n_cells")
        self._mod.swe2d_set_state(self._solver_h, h_arr, hu_arr, hv_arr)

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def sync_device(self) -> None:
        """Full CUDA device sync + error clear.  Use before Python timers that
        need to separate GPU execution time from Python overhead."""
        if hasattr(self._mod, "swe2d_gpu_device_sync"):
            self._mod.swe2d_gpu_device_sync()

    def gpu_active(self) -> bool:
        """True if the last completed step ran on the GPU."""
        if self._last_diag is None:
            return False
        return bool(self._last_diag.get("gpu_active", False))

    def get_max_tracking(self) -> Optional[Dict[str, np.ndarray]]:
        """Return per-cell max (h, hu, hv) across the whole simulation.

        Returns None if the native module doesn't support max tracking.
        """
        if self._solver_h is None:
            raise RuntimeError("initialize() must be called before get_max_tracking().")
        if not hasattr(self._mod, "swe2d_get_max_tracking"):
            logger.warning(
                "[BACKEND] get_max_tracking unavailable: "
                "swe2d_get_max_tracking not in native module. "
                "Rebuild hydra_swe2d with the latest max-tracking kernels."
            )
            return None
        h_max, hu_max, hv_max = self._mod.swe2d_get_max_tracking(self._solver_h)
        h_max = np.asarray(h_max, dtype=np.float64)
        hu_max = np.asarray(hu_max, dtype=np.float64)
        hv_max = np.asarray(hv_max, dtype=np.float64)
        if self._inv_cell_perm.size > 0:
            h_max = h_max[self._inv_cell_perm]
            hu_max = hu_max[self._inv_cell_perm]
            hv_max = hv_max[self._inv_cell_perm]
        h_safe = np.maximum(h_max, self._h_min)
        return {
            "max_h": h_max,
            "max_hu": hu_max,
            "max_hv": hv_max,
            "max_wse": h_max + np.asarray(self._cell_zb, dtype=np.float64),
            "max_vel": np.sqrt(hu_max**2 + hv_max**2) / h_safe,
        }

    @property
    def n_cells(self) -> int:
        """Number of cells in the mesh."""
        return self._n_cells

    def cell_areas(self) -> np.ndarray:
        """Return cached per-cell planform areas [L^2] from the input mesh."""
        return self._cell_area.copy()

    def export_mesh_data(self) -> Dict[str, np.ndarray]:
        """Return copy of all mesh arrays for serialization (host memory)."""
        out: Dict[str, np.ndarray] = {
            "node_x": self._mesh_node_x.copy(),
            "node_y": self._mesh_node_y.copy(),
            "node_z": self._mesh_node_z.copy(),
            "cell_nodes": self._mesh_cell_nodes.copy(),
        }
        if self._mesh_face_offsets is not None:
            out["cell_face_offsets"] = self._mesh_face_offsets.copy()
        if self._bc_n0.size > 0:
            out["bc_edge_node0"] = self._bc_n0.copy()
            out["bc_edge_node1"] = self._bc_n1.copy()
            out["bc_edge_type"] = self._bc_tp.copy()
            out["bc_edge_val"] = self._bc_vl.copy()
        return out

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def __del__(self):
        try:
            if getattr(self, "_solver_h", None) is not None:
                self._mod.swe2d_destroy(self._solver_h)
        except Exception as _e:

            logger.warning(f"[ERROR] Exception in backend.py: {_e}")

    def destroy(self) -> None:
        """Explicitly free native solver resources."""
        if self._solver_h is not None:
            self._mod.swe2d_destroy(self._solver_h)
            self._solver_h = None


# ── Shared mesh-build helper (CLI + workbench) ──────────────────────────
# Single code path for the face_offsets/face_nodes polygon-mesh logic so
# both the headless runner and the QGIS workbench call the same method.


def build_mesh(
    backend: SWE2DBackend,
    *,
    node_x: np.ndarray,
    node_y: np.ndarray,
    node_z: np.ndarray,
    cell_nodes: np.ndarray,
    cell_face_offsets: Optional[np.ndarray] = None,
    cell_face_nodes: Optional[np.ndarray] = None,
    **kwargs,
) -> None:
    """Build mesh, handling polygon meshes via face_offsets/face_nodes.

    When *both* ``cell_face_offsets`` and ``cell_face_nodes`` are
    provided they are passed as polygon args; otherwise ``cell_nodes``
    is used alone (no offsets).

    ``load_baked_mesh`` restores ``cell_face_nodes`` from the serialized
    C++ mesh BLOB when offsets exist AND ``offsets[-1] == len(face_nodes)``.
    If the BLOB is inconsistent (triangulated cell_nodes with orphan
    offsets), the alias is omitted and the triangle path is used.

    Remaining ``**kwargs`` are forwarded to ``SWE2DBackend.build_mesh``
    (e.g. ``bc_edge_node0``, ...).
    """
    if cell_face_offsets is not None and cell_face_nodes is not None:
        backend.build_mesh(
            node_x, node_y, node_z, cell_face_nodes,
            cell_face_offsets=cell_face_offsets,
            **kwargs,
        )
    else:
        backend.build_mesh(
            node_x, node_y, node_z, cell_nodes,
            **kwargs,
        )


