"""
swe2d_backend.py
Python bridge for the native 2D SWE hybrid GPU/CPU solver (backwater_swe2d).

Usage example:
    from swe2d_backend import SWE2DBackend, BCType
    import numpy as np

    backend = SWE2DBackend(use_gpu=True)
    backend.build_mesh(node_x, node_y, node_z, cell_nodes)
    backend.initialize(h0, n_mann=0.030, cfl=0.45)
    diags = backend.run(t_end=3600.0, dt_request=1.0)
    h, hu, hv = backend.get_state()
"""

from __future__ import annotations

import os
import sys
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple

from swe2d_extensions import (
    BedFrictionModel,
    SolverModelOptions,
    SpatialDiscretization,
    TemporalScheme,
    TurbulenceModel,
)

# Ensure the native extension (.so) built under ./build/ is findable regardless
# of how Python was launched (QGIS, standalone terminal, pytest, etc.).
_here = os.path.dirname(os.path.abspath(__file__))
for _candidate in (
    os.path.join(_here, "build"),
    os.path.join(_here, "build", "Release"),
    os.path.join(_here, "build", "Debug"),
):
    if os.path.isdir(_candidate) and _candidate not in sys.path:
        sys.path.insert(0, _candidate)

# ─────────────────────────────────────────────────────────────────────────────
# BCType constants (mirrored from native module; imported after module load)
# ─────────────────────────────────────────────────────────────────────────────
class BCType:
    INTERIOR = 0
    WALL     = 1
    INFLOW_Q = 2
    STAGE    = 3
    OPEN     = 4
    REFLECT  = 5
    NORMAL_DEPTH = 6


# ─────────────────────────────────────────────────────────────────────────────
# Module loader (lazy, with fallback messaging)
# ─────────────────────────────────────────────────────────────────────────────
_swe2d_mod = None
_swe2d_load_error: Optional[str] = None


def _load_swe2d_module():
    global _swe2d_mod, _swe2d_load_error
    if _swe2d_mod is not None:
        return _swe2d_mod
    if _swe2d_load_error is not None:
        return None
    try:
        import backwater_swe2d as mod
        _swe2d_mod = mod
        return mod
    except ImportError as e:
        _swe2d_load_error = str(e)
        return None


def swe2d_available() -> bool:
    """Return True if the native 2D solver module is importable."""
    return _load_swe2d_module() is not None


def swe2d_gpu_available() -> bool:
    """Return True if the native module is loaded AND a CUDA device is present."""
    mod = _load_swe2d_module()
    if mod is None:
        return False
    try:
        return mod.swe2d_gpu_available()
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# SWE2DBackend
# ─────────────────────────────────────────────────────────────────────────────
class SWE2DBackend:
    """
    High-level Python interface to the native backwater_swe2d module.

    Lifecycle:
        1. Construct (optionally pass use_gpu=False to force CPU path).
        2. build_mesh(...)  — must be called before initialize().
        3. initialize(...)  — creates native solver with initial conditions.
        4. step() or run()  — advance in time.
        5. get_state()      — retrieve current (h, hu, hv) numpy arrays.
        6. destroy()        — free native resources (or let GC handle it).
    """

    def __init__(self, use_gpu: bool = True):
        mod = _load_swe2d_module()
        if mod is None:
            raise RuntimeError(
                f"backwater_swe2d native module not available: {_swe2d_load_error}. "
                "Build the native module first (cmake --build build)."
            )
        self._mod = mod

        # Override GPU if env var requests CPU-only
        env_gpu = os.environ.get("BACKWATER_SWE2D_GPU", "").strip()
        if env_gpu == "0":
            use_gpu = False

        # Let the native layer decide final GPU activation/fallback at solver
        # creation time. Python-side availability probes can be conservative in
        # embedded environments (e.g., QGIS-launched interpreter).
        self._use_gpu = bool(use_gpu)
        self._mesh_h   = None   # PyMesh handle
        self._solver_h = None   # PySolver handle
        self._n_cells  = 0
        self._boundary_edge_index_by_nodes = {}
        self._supports_solver_bc_update = hasattr(self._mod, "swe2d_solver_set_boundary_values")
        self._supports_solver_hydrographs = hasattr(self._mod, "swe2d_solver_set_boundary_hydrographs")
        self._supports_solver_rain_cn = hasattr(self._mod, "swe2d_solver_set_rain_cn_forcing")
        self._h_min = 1.0e-6
        self._cell_area = np.empty(0, dtype=np.float64)

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
            for i in range(face_offsets.size - 1):
                s = int(face_offsets[i])
                e = int(face_offsets[i + 1])
                ids = cell_nodes_flat[s:e]
                if ids.size < 3:
                    continue
                xx = node_x[ids]
                yy = node_y[ids]
                self._cell_area[i] = 0.5 * abs(float(np.dot(xx, np.roll(yy, -1)) - np.dot(yy, np.roll(xx, -1))))
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
            self._mesh_h = self._mod.swe2d_build_mesh(
                node_x, node_y, node_z, cell_nodes_flat,
                bc_n0, bc_n1, bc_tp, bc_vl)

        info = self._mod.swe2d_mesh_info(self._mesh_h)
        self._n_cells = info["n_cells"]

        self._boundary_edge_index_by_nodes = {}
        try:
            edge_idx, n0, n1, _, _ = self._mod.swe2d_boundary_edges(self._mesh_h)
            for i in range(edge_idx.size):
                a = int(n0[i])
                b = int(n1[i])
                key = (a, b) if a < b else (b, a)
                self._boundary_edge_index_by_nodes[key] = int(edge_idx[i])
        except Exception:
            # Older binaries may not expose boundary-edge query; dynamic BC updates
            # will be unavailable in that case.
            self._boundary_edge_index_by_nodes = {}

    def supports_dynamic_boundary_update(self) -> bool:
        return bool(self._boundary_edge_index_by_nodes)

    def set_boundary_conditions(
        self,
        bc_edge_node0: np.ndarray,
        bc_edge_node1: np.ndarray,
        bc_edge_type: np.ndarray,
        bc_edge_val: np.ndarray,
    ) -> None:
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

    def set_rain_cn_forcing_native(
        self,
        cell_gage_idx: np.ndarray,
        gage_offsets: np.ndarray,
        hg_time_s: np.ndarray,
        hg_cum_mm: np.ndarray,
        cn: np.ndarray,
        ia_ratio: float = 0.2,
        mm_to_model_depth: float = 1.0e-3,
    ) -> None:
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
        )

    # ── Solver init ──────────────────────────────────────────────────────────

    def initialize(
        self,
        h0: np.ndarray,
        hu0: Optional[np.ndarray] = None,
        hv0: Optional[np.ndarray] = None,
        n_mann_cell: Optional[np.ndarray] = None,
        g:        float = 9.81,
        n_mann:   float = 0.035,
        h_min:    float = 1.0e-6,
        cfl:      float = 0.45,
        dt_max:   float = 10.0,
        dt_fixed: float = -1.0,
        max_inv_area: float = 1.0e6,
        cfl_lambda_cap: float = 1.0e6,
        momentum_cap_min_speed: float = 50.0,
        momentum_cap_celerity_mult: float = 20.0,
        depth_cap: float = 1.0e6,
        max_rel_depth_increase: float = 2.0,
        shallow_damping_depth: float = 1.0e-4,
        gpu_diag_sync_interval_steps: int = 1,
        n_threads: int  = 0,
        temporal_scheme: TemporalScheme = TemporalScheme.SSP_RK2,
        spatial_discretization: SpatialDiscretization = SpatialDiscretization.FV_FIRST_ORDER,
        turbulence_model: TurbulenceModel = TurbulenceModel.NONE,
        bed_friction_model: BedFrictionModel = BedFrictionModel.MANNING,
        model_options: Optional[SolverModelOptions] = None,
        degen_mode: int = 0,
        front_flux_damping: float = 0.5,
        active_set_hysteresis: bool = True,
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
        gpu_diag_sync_interval_steps : int
            GPU host-sync diagnostics cadence. 1=every step, N=every N steps,
            <=0 disables per-step host diagnostic sync.
        n_threads : int
            CPU thread count (0 = auto).
        temporal_scheme : TemporalScheme
            Temporal integrator selection (default SSP_RK2).
        spatial_discretization : SpatialDiscretization
            Spatial scheme selector (currently scaffolded).
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
        }
        if model_options is not None:
            native_opts.update(model_options.to_native_dict())

        self._solver_h = self._mod.swe2d_create_solver(
            self._mesh_h,
            h0_arr, hu0_arr, hv0_arr, n_mann_cell_arr,
            g=g, n_mann=n_mann, h_min=h_min,
            cfl=cfl, dt_max=dt_max, dt_fixed=dt_fixed,
            max_inv_area=max_inv_area,
            cfl_lambda_cap=cfl_lambda_cap,
            momentum_cap_min_speed=momentum_cap_min_speed,
            momentum_cap_celerity_mult=momentum_cap_celerity_mult,
            depth_cap=depth_cap,
            max_rel_depth_increase=max_rel_depth_increase,
            shallow_damping_depth=shallow_damping_depth,
            gpu_diag_sync_interval_steps=int(gpu_diag_sync_interval_steps),
            use_gpu=self._use_gpu, n_threads=n_threads,
            temporal_order=int(native_opts["temporal_order"]),
            spatial_scheme=int(native_opts["spatial_scheme"]),
            turbulence_model=int(native_opts["turbulence_model"]),
            bed_friction_model=int(native_opts["bed_friction_model"]),
            enable_rain_module=bool(native_opts["enable_rain_module"]),
            enable_pipe_network_module=bool(native_opts["enable_pipe_network_module"]),
            enable_hydraulic_structures=bool(native_opts["enable_hydraulic_structures"]),
            degen_mode=int(degen_mode),
            front_flux_damping=float(front_flux_damping),
            active_set_hysteresis=bool(active_set_hysteresis),
        )
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

        Returns
        -------
        List of per-step diagnostic dicts.
        """
        if self._solver_h is None:
            raise RuntimeError("initialize() must be called before run().")

        diags: List[dict] = []
        t = 0.0
        while t < t_end:
            if cancel_check and cancel_check():
                break
            diag = self.step(dt_request)
            dt = float(diag["dt"])
            if source_rate_callback is not None and dt > 0.0:
                h, hu, hv = self.get_state()
                src = source_rate_callback(t, dt, h, hu, hv)
                if src is not None:
                    src_arr = np.ascontiguousarray(src, dtype=np.float64).ravel()
                    if src_arr.size != self._n_cells:
                        raise ValueError("source_rate_callback must return an array with length n_cells")
                    h = np.maximum(0.0, h + dt * src_arr)
                    dry = h < self._h_min
                    hu = np.where(dry, 0.0, hu)
                    hv = np.where(dry, 0.0, hv)
                    self.set_state(h, hu, hv)
            t += dt
            diags.append(diag)
            if progress_callback:
                progress_callback(t, diag)

        return diags

    # ── State retrieval ───────────────────────────────────────────────────────

    def get_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return current (h, hu, hv) numpy arrays, each shape (M,) float64.
        """
        if self._solver_h is None:
            raise RuntimeError("initialize() must be called before get_state().")
        return self._mod.swe2d_get_state(self._solver_h)

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

    def gpu_active(self) -> bool:
        """True if the last completed step ran on the GPU."""
        if self._last_diag is None:
            return False
        return bool(self._last_diag.get("gpu_active", False))

    @property
    def n_cells(self) -> int:
        """Number of cells in the mesh."""
        return self._n_cells

    def cell_areas(self) -> np.ndarray:
        """Return cached per-cell planform areas [L^2] from the input mesh."""
        return self._cell_area.copy()

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def destroy(self) -> None:
        """Explicitly free native solver resources."""
        if self._solver_h is not None:
            self._mod.swe2d_destroy(self._solver_h)
            self._solver_h = None

    def __del__(self):
        try:
            self.destroy()
        except Exception:
            pass
