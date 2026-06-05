"""
swe2d_backend.py
Python bridge for the native 2D SWE hybrid GPU/CPU solver (hydra_swe2d).

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
import inspect
import importlib
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union

from swe2d import units as _u
from swe2d.extensions.extension_models import (
    BedFrictionModel,
    GodunovSolverMode,
    SWE2DEquationSet,
    SWE2DThreeDCouplingMode,
    SWE2DThreeDSolverModel,
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
_swe2d_mod_cache: Dict[str, object] = {}
_swe2d_load_errors: Dict[str, str] = {}
_swe2d_last_load_error: Optional[str] = None
# Backward-compatible globals used by older tests and monkeypatches.
_swe2d_mod = None
_swe2d_load_error: Optional[str] = None


def _env_openmp_enabled_default() -> bool:
    raw = str(os.environ.get("BACKWATER_SWE2D_OPENMP", "1") or "1").strip().lower()
    return raw not in {"0", "false", "off", "no"}


def _module_name_for_openmp_enabled(openmp_enabled: Optional[bool] = None) -> str:
    use_openmp = _env_openmp_enabled_default() if openmp_enabled is None else bool(openmp_enabled)
    return "hydra_swe2d" if use_openmp else "hydra_swe2d_serial"


def _load_swe2d_module(openmp_enabled: Optional[bool] = None):
    global _swe2d_last_load_error, _swe2d_mod, _swe2d_load_error
    if _swe2d_mod is not None:
        _swe2d_last_load_error = None
        _swe2d_load_error = None
        return _swe2d_mod
    module_name = _module_name_for_openmp_enabled(openmp_enabled)
    if module_name in _swe2d_mod_cache:
        _swe2d_last_load_error = None
        _swe2d_load_error = None
        return _swe2d_mod_cache[module_name]
    for loaded_name in _swe2d_mod_cache.keys():
        if loaded_name != module_name:
            _swe2d_last_load_error = (
                "Cannot switch SWE2D native module variant in the same Python process "
                f"(loaded={loaded_name}, requested={module_name}). "
                "Restart QGIS/Python and run again with the desired OpenMP setting."
            )
            _swe2d_load_error = _swe2d_last_load_error
            return None
    try:
        mod = importlib.import_module(module_name)
        _swe2d_mod_cache[module_name] = mod
        if module_name == "hydra_swe2d":
            _swe2d_mod = mod
        _swe2d_last_load_error = None
        _swe2d_load_error = None
        return mod
    except ImportError as e:
        err = f"{module_name}: {e}"
        _swe2d_load_errors[module_name] = err
        _swe2d_last_load_error = err
        _swe2d_load_error = err
        return None


def swe2d_available(openmp_enabled: Optional[bool] = None) -> bool:
    """Return True if the native 2D solver module is importable."""
    return _load_swe2d_module(openmp_enabled=openmp_enabled) is not None


def load_swe2d_native_module(openmp_enabled: Optional[bool] = None):
    """Load and return the selected SWE2D native module, or None on failure."""
    return _load_swe2d_module(openmp_enabled=openmp_enabled)


def swe2d_gpu_available(openmp_enabled: Optional[bool] = None) -> bool:
    """Return True if the native module is loaded AND a CUDA device is present."""
    mod = _load_swe2d_module(openmp_enabled=openmp_enabled)
    if mod is None:
        return False
    try:
        return mod.swe2d_gpu_available()
    except Exception:
        return False


_SWE3D_ADAPTIVE_DT_METHOD_MAP = {
    "advective": 0,
    "advective_only": 0,
    "gravity": 1,
    "advective_gravity": 1,
    "advective_plus_gravity_wave": 1,
    "projection": 2,
    "advective_gravity_projection": 2,
    "advective_gravity_plus_projection": 2,
}


def set_swe3d_adaptive_dt_method(method: Union[int, str]) -> int:
    """
    Configure SWE3D adaptive dt mode via environment variable.

    Modes:
      0 / "advective"                    : advective CFL only
      1 / "advective_gravity"            : advective + gravity-wave CFL
      2 / "advective_gravity_projection" : advective + gravity-wave + projection-aware dt
    """
    mode_val: int
    if isinstance(method, str):
        key = method.strip().lower()
        if key.isdigit() or (key.startswith("-") and key[1:].isdigit()):
            mode_val = int(key)
        else:
            if key not in _SWE3D_ADAPTIVE_DT_METHOD_MAP:
                raise ValueError(
                    "Unknown SWE3D adaptive dt method. "
                    "Use 0/1/2 or one of: "
                    f"{sorted(_SWE3D_ADAPTIVE_DT_METHOD_MAP.keys())}"
                )
            mode_val = int(_SWE3D_ADAPTIVE_DT_METHOD_MAP[key])
    else:
        mode_val = int(method)

    if mode_val < 0 or mode_val > 2:
        raise ValueError("SWE3D adaptive dt mode must be 0, 1, or 2")

    os.environ["BACKWATER_SWE3D_ADAPTIVE_DT_MODE"] = str(mode_val)
    return mode_val


def set_swe3d_vof_max_substeps(max_substeps: int) -> int:
    """Set SWE3D VOF transport max substeps cap (>= 1)."""
    cap = int(max_substeps)
    if cap < 1:
        raise ValueError("SWE3D VOF max substeps must be >= 1")
    os.environ["BACKWATER_SWE3D_VOF_MAX_SUBSTEPS"] = str(cap)
    return cap


def set_swe3d_predictor_damping_coeff(coeff: float) -> float:
    """Set SWE3D predictor damping coefficient (>= 0)."""
    value = float(coeff)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("SWE3D predictor damping coefficient must be a non-negative finite number")
    os.environ["BACKWATER_SWE3D_PREDICTOR_DAMPING_COEFF"] = f"{value:.17g}"
    return value


def set_swe3d_free_surface_gauge_tolerance_pa(tolerance_pa: float) -> float:
    """Set ZMAX free-surface pressure band tolerance in pressure units (Pa-equivalent)."""
    value = float(tolerance_pa)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("SWE3D free-surface gauge tolerance must be a non-negative finite number")
    os.environ["BACKWATER_SWE3D_FREE_SURFACE_GAUGE_TOLERANCE_PA"] = f"{value:.17g}"
    return value


def configure_swe3d_runtime(
    adaptive_dt_method: Optional[Union[int, str]] = None,
    vof_max_substeps: Optional[int] = None,
    predictor_damping_coeff: Optional[float] = None,
    free_surface_gauge_tolerance_pa: Optional[float] = None,
    gravity_wave_cfl: Optional[float] = None,
    projection_residual_target: Optional[float] = None,
    projection_reject_enable: Optional[bool] = None,
    projection_fail_fast: Optional[bool] = None,
    projection_divergence_gate_enable: Optional[bool] = None,
    projection_divergence_ratio_target: Optional[float] = None,
    projection_dt_reduction: Optional[float] = None,
    projection_max_retries: Optional[int] = None,
    projection_min_dt_factor: Optional[float] = None,
    state_reject_enable: Optional[bool] = None,
    state_vof_bounds_tol: Optional[float] = None,
    state_max_abs_velocity: Optional[float] = None,
    state_max_abs_pressure: Optional[float] = None,
    geometry_gate_strict: Optional[bool] = None,
    geometry_gate_max_solid_fraction: Optional[float] = None,
    geometry_gate_max_seed_leak_fallbacks: Optional[int] = None,
    outflow_policy: Optional[int] = None,
    free_surface_vent_bias: Optional[float] = None,
    q_inflow_area_policy: Optional[int] = None,
    open_bc_damping: Optional[float] = None,
    projection_boundary_policy: Optional[int] = None,
) -> Dict[str, object]:
    """
    Convenience helper for configuring SWE3D runtime controls from Python/QGIS console.

    All values are applied as process environment variables and are consumed
    by the CUDA path each step.
    """
    applied: Dict[str, object] = {}

    if adaptive_dt_method is not None:
        applied["adaptive_dt_mode"] = set_swe3d_adaptive_dt_method(adaptive_dt_method)
    if vof_max_substeps is not None:
        applied["vof_max_substeps"] = set_swe3d_vof_max_substeps(vof_max_substeps)
    if predictor_damping_coeff is not None:
        applied["predictor_damping_coeff"] = set_swe3d_predictor_damping_coeff(predictor_damping_coeff)
    if free_surface_gauge_tolerance_pa is not None:
        applied["free_surface_gauge_tolerance_pa"] = (
            set_swe3d_free_surface_gauge_tolerance_pa(free_surface_gauge_tolerance_pa)
        )
    if gravity_wave_cfl is not None:
        gw_cfl = float(gravity_wave_cfl)
        if not np.isfinite(gw_cfl) or gw_cfl <= 0.0:
            raise ValueError("gravity_wave_cfl must be a positive finite number")
        os.environ["BACKWATER_SWE3D_GRAVITY_WAVE_CFL"] = f"{gw_cfl:.17g}"
        applied["gravity_wave_cfl"] = gw_cfl
    if projection_residual_target is not None:
        target = float(projection_residual_target)
        if not np.isfinite(target) or target <= 0.0:
            raise ValueError("projection_residual_target must be a positive finite number")
        os.environ["BACKWATER_SWE3D_PROJECTION_RESIDUAL_TARGET"] = f"{target:.17g}"
        applied["projection_residual_target"] = target
    if projection_reject_enable is not None:
        enabled = bool(projection_reject_enable)
        os.environ["BACKWATER_SWE3D_PROJECTION_REJECT_ENABLE"] = "1" if enabled else "0"
        applied["projection_reject_enable"] = enabled
    if projection_fail_fast is not None:
        fail_fast = bool(projection_fail_fast)
        os.environ["BACKWATER_SWE3D_PROJECTION_FAIL_FAST"] = "1" if fail_fast else "0"
        applied["projection_fail_fast"] = fail_fast
    if projection_divergence_gate_enable is not None:
        enabled = bool(projection_divergence_gate_enable)
        os.environ["BACKWATER_SWE3D_PROJECTION_DIVERGENCE_GATE_ENABLE"] = "1" if enabled else "0"
        applied["projection_divergence_gate_enable"] = enabled
    if projection_divergence_ratio_target is not None:
        div_ratio = float(projection_divergence_ratio_target)
        if not np.isfinite(div_ratio) or div_ratio <= 0.0:
            raise ValueError("projection_divergence_ratio_target must be a positive finite number")
        os.environ["BACKWATER_SWE3D_PROJECTION_DIVERGENCE_RATIO_TARGET"] = f"{div_ratio:.17g}"
        applied["projection_divergence_ratio_target"] = div_ratio
    if projection_dt_reduction is not None:
        reduction = float(projection_dt_reduction)
        if not np.isfinite(reduction) or reduction <= 0.0 or reduction >= 1.0:
            raise ValueError("projection_dt_reduction must be in (0, 1)")
        os.environ["BACKWATER_SWE3D_PROJECTION_DT_REDUCTION"] = f"{reduction:.17g}"
        applied["projection_dt_reduction"] = reduction
    if projection_max_retries is not None:
        retries = int(projection_max_retries)
        if retries < 0:
            raise ValueError("projection_max_retries must be >= 0")
        os.environ["BACKWATER_SWE3D_PROJECTION_MAX_RETRIES"] = str(retries)
        applied["projection_max_retries"] = retries
    if projection_min_dt_factor is not None:
        min_fac = float(projection_min_dt_factor)
        if not np.isfinite(min_fac) or min_fac <= 0.0:
            raise ValueError("projection_min_dt_factor must be > 0")
        os.environ["BACKWATER_SWE3D_PROJECTION_MIN_DT_FACTOR"] = f"{min_fac:.17g}"
        applied["projection_min_dt_factor"] = min_fac

    if state_reject_enable is not None:
        enabled = bool(state_reject_enable)
        os.environ["BACKWATER_SWE3D_STATE_REJECT_ENABLE"] = "1" if enabled else "0"
        applied["state_reject_enable"] = enabled
    if state_vof_bounds_tol is not None:
        tol = float(state_vof_bounds_tol)
        if not np.isfinite(tol) or tol < 0.0:
            raise ValueError("state_vof_bounds_tol must be >= 0")
        os.environ["BACKWATER_SWE3D_STATE_VOF_BOUNDS_TOL"] = f"{tol:.17g}"
        applied["state_vof_bounds_tol"] = tol
    if state_max_abs_velocity is not None:
        vmax = float(state_max_abs_velocity)
        if not np.isfinite(vmax) or vmax <= 0.0:
            raise ValueError("state_max_abs_velocity must be > 0")
        os.environ["BACKWATER_SWE3D_STATE_MAX_ABS_VELOCITY"] = f"{vmax:.17g}"
        applied["state_max_abs_velocity"] = vmax
    if state_max_abs_pressure is not None:
        pmax = float(state_max_abs_pressure)
        if not np.isfinite(pmax) or pmax <= 0.0:
            raise ValueError("state_max_abs_pressure must be > 0")
        os.environ["BACKWATER_SWE3D_STATE_MAX_ABS_PRESSURE"] = f"{pmax:.17g}"
        applied["state_max_abs_pressure"] = pmax

    if geometry_gate_strict is not None:
        strict = bool(geometry_gate_strict)
        os.environ["BACKWATER_SWE3D_GEOM_STRICT"] = "1" if strict else "0"
        applied["geometry_gate_strict"] = strict
    if geometry_gate_max_solid_fraction is not None:
        frac = float(geometry_gate_max_solid_fraction)
        if not np.isfinite(frac) or frac < 0.0 or frac > 1.0:
            raise ValueError("geometry_gate_max_solid_fraction must be in [0, 1]")
        os.environ["BACKWATER_SWE3D_GEOM_MAX_SOLID_FRACTION"] = f"{frac:.17g}"
        applied["geometry_gate_max_solid_fraction"] = frac
    if geometry_gate_max_seed_leak_fallbacks is not None:
        leaks = int(geometry_gate_max_seed_leak_fallbacks)
        if leaks < 0:
            raise ValueError("geometry_gate_max_seed_leak_fallbacks must be >= 0")
        os.environ["BACKWATER_SWE3D_GEOM_MAX_SEED_LEAK_FALLBACKS"] = str(leaks)
        applied["geometry_gate_max_seed_leak_fallbacks"] = leaks

    if outflow_policy is not None:
        policy = int(outflow_policy)
        if policy not in (0, 1):
            raise ValueError("outflow_policy must be 0 (legacy_passive) or 1 (characteristic_nonreflecting)")
        os.environ["BACKWATER_SWE3D_OUTFLOW_POLICY"] = str(policy)
        applied["outflow_policy"] = policy
    if free_surface_vent_bias is not None:
        vent_bias = float(free_surface_vent_bias)
        if not np.isfinite(vent_bias):
            raise ValueError("free_surface_vent_bias must be a finite number")
        os.environ["BACKWATER_SWE3D_FREE_SURFACE_VENT_BIAS"] = f"{vent_bias:.17g}"
        applied["free_surface_vent_bias"] = vent_bias
    if q_inflow_area_policy is not None:
        area_policy = int(q_inflow_area_policy)
        if area_policy not in (0, 1):
            raise ValueError("q_inflow_area_policy must be 0 (total_face_area) or 1 (dynamic_wet_open_area)")
        os.environ["BACKWATER_SWE3D_Q_INFLOW_AREA_POLICY"] = str(area_policy)
        applied["q_inflow_area_policy"] = area_policy
    if open_bc_damping is not None:
        damping = float(open_bc_damping)
        if not np.isfinite(damping) or damping < 0.0 or damping > 1.0:
            raise ValueError("open_bc_damping must be in [0, 1]")
        os.environ["BACKWATER_SWE3D_OPEN_BC_DAMPING"] = f"{damping:.17g}"
        applied["open_bc_damping"] = damping
    if projection_boundary_policy is not None:
        proj_policy = int(projection_boundary_policy)
        if proj_policy not in (0, 1):
            raise ValueError("projection_boundary_policy must be 0 (legacy_zmax_only) or 1 (face_aware)")
        os.environ["BACKWATER_SWE3D_PROJECTION_BOUNDARY_POLICY"] = str(proj_policy)
        applied["projection_boundary_policy"] = proj_policy

    return applied


# ─────────────────────────────────────────────────────────────────────────────
# SWE2DBackend
# ─────────────────────────────────────────────────────────────────────────────
class SWE2DBackend:
    """
    High-level Python interface to the native hydra_swe2d module.

    Lifecycle:
        1. Construct (optionally pass use_gpu=False to force CPU path).
        2. build_mesh(...)  — must be called before initialize().
        3. initialize(...)  — creates native solver with initial conditions.
        4. step() or run()  — advance in time.
        5. get_state()      — retrieve current (h, hu, hv) numpy arrays.
        6. destroy()        — free native resources (or let GC handle it).
    """

    def __init__(self, use_gpu: bool = True, openmp_enabled: Optional[bool] = None):
        self._openmp_enabled = _env_openmp_enabled_default() if openmp_enabled is None else bool(openmp_enabled)
        self._native_module_name = _module_name_for_openmp_enabled(self._openmp_enabled)
        mod = _load_swe2d_module(openmp_enabled=self._openmp_enabled)
        if mod is None:
            raise RuntimeError(
                f"{self._native_module_name} native module not available: {_swe2d_last_load_error}. "
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
        self._supports_solver_external_sources = hasattr(self._mod, "swe2d_solver_set_external_sources")
        self._h_min = 1.0e-6
        self._cell_area = np.empty(0, dtype=np.float64)
        self._tiny_mode = 1
        self._tiny_persistent_chunk_substeps = 8

        # Last step diagnostics
        self._last_diag: Optional[dict] = None

    def _create_solver_compat(self, *args, **kwargs):
        """Call swe2d_create_solver with kwargs compatible with loaded extension."""
        try:
            return self._mod.swe2d_create_solver(*args, **kwargs)
        except TypeError as exc:
            # pybind11 emits this for signature mismatches (often when Python
            # passes newer kwargs to an older compiled extension).
            if "incompatible function arguments" not in str(exc):
                raise

        sig = None
        try:
            sig = inspect.signature(self._mod.swe2d_create_solver)
        except (TypeError, ValueError):
            sig = None

        if sig is not None:
            allowed = {
                name
                for name, param in sig.parameters.items()
                if param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
            }
            filtered = {k: v for k, v in kwargs.items() if k in allowed}
            return self._mod.swe2d_create_solver(*args, **filtered)

        # Conservative fallback for environments where signature introspection
        # on the pybind function is unavailable.
        filtered = dict(kwargs)
        for key in (
            "extreme_rain_mode",
            "source_cfl_beta",
            "source_max_substeps",
            "source_rate_cap",
            "source_depth_step_cap",
            "source_true_subcycling",
            "source_imex_split",
            "enable_shallow_front_recon_fallback",
            "tiny_mode",
            "tiny_cell_threshold",
            "tiny_edge_threshold",
            "tiny_wet_cell_threshold",
            "tiny_persistent_chunk_substeps",
            "tiny_active_compaction_stride_steps",
            "tiny_enable_active_compaction",
            "godunov_mode",
            "equation_set",
            "coupling_mode",
            "three_d_solver_model",
            "enforce_gpu_only_advanced_modes",
            "three_d_single_phase_free_surface",
            "dt_initial",
        ):
            filtered.pop(key, None)
        return self._mod.swe2d_create_solver(*args, **filtered)

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

    def boundary_edge_cells(self) -> Optional[np.ndarray]:
        """Return interior cell index for each boundary edge, or None."""
        return self._boundary_edge_cells

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
        godunov_mode: GodunovSolverMode = GodunovSolverMode.CURRENT_GPU_STEP,
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
            "godunov_mode": int(godunov_mode),
            "turbulence_model": int(turbulence_model),
            "bed_friction_model": int(bed_friction_model),
            "equation_set": int(SWE2DEquationSet.HYDROSTATIC_2D),
            "coupling_mode": int(SWE2DThreeDCouplingMode.OFF),
            "three_d_solver_model": int(SWE2DThreeDSolverModel.DISABLED),
            "enforce_gpu_only_advanced_modes": True,
            "three_d_single_phase_free_surface": True,
            "enable_rain_module": False,
            "enable_pipe_network_module": False,
            "enable_hydraulic_structures": False,
        }
        if model_options is not None:
            native_opts.update(model_options.to_native_dict())

        equation_set = int(native_opts.get("equation_set", int(SWE2DEquationSet.HYDROSTATIC_2D)))
        coupling_mode = int(native_opts.get("coupling_mode", int(SWE2DThreeDCouplingMode.OFF)))
        advanced_mode_requested = (
            equation_set != int(SWE2DEquationSet.HYDROSTATIC_2D)
            or coupling_mode != int(SWE2DThreeDCouplingMode.OFF)
        )
        enforce_gpu_only = bool(native_opts.get("enforce_gpu_only_advanced_modes", True))
        if advanced_mode_requested and enforce_gpu_only:
            if not self._use_gpu:
                raise ValueError(
                    "Nonhydrostatic/coupled solver modes are GPU-only. "
                    "Initialize backend with use_gpu=True."
                )
            try:
                if not bool(self._mod.swe2d_gpu_available()):
                    raise RuntimeError(
                        "Nonhydrostatic/coupled solver modes require an active CUDA device "
                        "and GPU-enabled native build."
                    )
            except AttributeError:
                raise RuntimeError(
                    "Loaded native solver does not expose GPU capability checks; "
                    "rebuild hydra_swe2d with CUDA support."
                )

        # Apply conservative SWE3D runtime guardrails for interactive workbench
        # runs. Use setdefault so explicit user/env tuning still takes priority.
        three_d_model_enabled = (
            int(native_opts.get("three_d_solver_model", int(SWE2DThreeDSolverModel.DISABLED)))
            != int(SWE2DThreeDSolverModel.DISABLED)
        )
        if self._use_gpu and three_d_model_enabled:
            os.environ.setdefault("BACKWATER_SWE3D_ADAPTIVE_DT_MODE", "2")
            os.environ.setdefault("BACKWATER_SWE3D_PROJECTION_REJECT_ENABLE", "1")
            os.environ.setdefault("BACKWATER_SWE3D_STATE_REJECT_ENABLE", "1")
            os.environ.setdefault("BACKWATER_SWE3D_STATE_MAX_ABS_VELOCITY", "50")
            os.environ.setdefault("BACKWATER_SWE3D_PROJECTION_FAIL_FAST", "1")
            os.environ.setdefault("BACKWATER_SWE3D_VOF_TRANSPORT_DEBUG", "0")

        self._solver_h = self._create_solver_compat(
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
            use_gpu=self._use_gpu, n_threads=n_threads,
            temporal_order=int(native_opts["temporal_order"]),
            spatial_scheme=int(native_opts["spatial_scheme"]),
            godunov_mode=int(native_opts["godunov_mode"]),
            turbulence_model=int(native_opts["turbulence_model"]),
            bed_friction_model=int(native_opts["bed_friction_model"]),
            equation_set=int(native_opts["equation_set"]),
            coupling_mode=int(native_opts["coupling_mode"]),
            three_d_solver_model=int(native_opts["three_d_solver_model"]),
            enforce_gpu_only_advanced_modes=bool(native_opts["enforce_gpu_only_advanced_modes"]),
            three_d_single_phase_free_surface=bool(native_opts["three_d_single_phase_free_surface"]),
            enable_rain_module=bool(native_opts["enable_rain_module"]),
            enable_pipe_network_module=bool(native_opts["enable_pipe_network_module"]),
            enable_hydraulic_structures=bool(native_opts["enable_hydraulic_structures"]),
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

        # Fallback: Python loop (when source_rate_callback is used or native API unavailable)
        emulate_native_lag = False
        pending_source_arr: Optional[np.ndarray] = None

        if use_native_source_injection and source_rate_callback is not None:
            if not self._supports_solver_external_sources:
                # Compatibility fallback for extensions built without native
                # external-source API support.
                use_native_source_injection = False
                emulate_native_lag = True
            # Start from zero external source on solver.
            if use_native_source_injection:
                self.set_external_sources_native(None)

        t = 0.0
        while t < t_end:
            if cancel_check and cancel_check():
                break
            diag = self.step(dt_request)
            dt = float(diag["dt"])
            if source_rate_callback is not None and dt > 0.0:
                native_device_applied = False
                if use_native_source_injection:
                    owner = getattr(source_rate_callback, "__self__", None)
                    apply_native_device_sources = getattr(owner, "apply_native_device_sources", None)
                    if callable(apply_native_device_sources):
                        try:
                            native_device_applied = bool(apply_native_device_sources(t, dt))
                        except Exception:
                            native_device_applied = False
                if native_device_applied:
                    src = None
                else:
                    h, hu, hv = self.get_state()
                    src = source_rate_callback(t, dt, h, hu, hv)
                if use_native_source_injection:
                    if not native_device_applied and src is not None:
                        self.set_external_sources_native(src)
                elif emulate_native_lag:
                    if src is not None:
                        src_arr = np.ascontiguousarray(src, dtype=np.float64).ravel()
                        if src_arr.size != self._n_cells:
                            raise ValueError("source_rate_callback must return an array with length n_cells")
                    else:
                        src_arr = None

                    if pending_source_arr is not None:
                        h = np.maximum(0.0, h + dt * pending_source_arr)
                        dry = h < self._h_min
                        hu = np.where(dry, 0.0, hu)
                        hv = np.where(dry, 0.0, hv)
                        self.set_state(h, hu, hv)
                    pending_source_arr = src_arr
                elif src is not None:
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

        if use_native_source_injection and source_rate_callback is not None:
            # Prevent stale external sources from affecting future runs.
            self.set_external_sources_native(None)

        return diags

    # ── State retrieval ───────────────────────────────────────────────────────

    def get_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return current (h, hu, hv) numpy arrays, each shape (M,) float64.
        """
        if self._solver_h is None:
            raise RuntimeError("initialize() must be called before get_state().")
        return self._mod.swe2d_get_state(self._solver_h)

    def supports_3d_patch_observation(self) -> bool:
        """Return True when native 3D patch observation APIs are available."""
        if self._solver_h is None:
            return False
        return bool(
            hasattr(self._mod, "swe2d_get_3d_patch_stats")
            and hasattr(self._mod, "swe2d_get_3d_patch_vof")
        )

    def get_3d_patch_stats(self) -> Dict[str, object]:
        """Return aggregate diagnostics for the active 3D Cartesian patch."""
        if self._solver_h is None:
            raise RuntimeError("initialize() must be called before get_3d_patch_stats().")
        if not hasattr(self._mod, "swe2d_get_3d_patch_stats"):
            raise RuntimeError("Native module does not expose swe2d_get_3d_patch_stats().")
        return dict(self._mod.swe2d_get_3d_patch_stats(self._solver_h))

    def get_3d_patch_vof(self) -> np.ndarray:
        """Download full 3D patch VoF field as a flat float64 array."""
        if self._solver_h is None:
            raise RuntimeError("initialize() must be called before get_3d_patch_vof().")
        if not hasattr(self._mod, "swe2d_get_3d_patch_vof"):
            raise RuntimeError("Native module does not expose swe2d_get_3d_patch_vof().")
        return np.asarray(self._mod.swe2d_get_3d_patch_vof(self._solver_h), dtype=np.float64)

    def get_3d_patch_velocity(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Download full 3D patch velocity fields (u, v, w) as flat float64 arrays."""
        if self._solver_h is None:
            raise RuntimeError("initialize() must be called before get_3d_patch_velocity().")
        if not hasattr(self._mod, "swe2d_get_3d_patch_velocity"):
            raise RuntimeError("Native module does not expose swe2d_get_3d_patch_velocity().")
        raw = self._mod.swe2d_get_3d_patch_velocity(self._solver_h)
        if not isinstance(raw, (tuple, list)) or len(raw) != 3:
            raise RuntimeError("Invalid 3D patch velocity payload returned by native module.")
        u = np.asarray(raw[0], dtype=np.float64).ravel()
        v = np.asarray(raw[1], dtype=np.float64).ravel()
        w = np.asarray(raw[2], dtype=np.float64).ravel()
        return u, v, w

    def get_3d_patch_pressure(self) -> np.ndarray:
        """Download full 3D patch pressure field as a flat float64 array."""
        if self._solver_h is None:
            raise RuntimeError("initialize() must be called before get_3d_patch_pressure().")
        if not hasattr(self._mod, "swe2d_get_3d_patch_pressure"):
            raise RuntimeError("Native module does not expose swe2d_get_3d_patch_pressure().")
        return np.asarray(self._mod.swe2d_get_3d_patch_pressure(self._solver_h), dtype=np.float64).ravel()

    def supports_3d_patch_geometry_upload(self) -> bool:
        """Return True when native 3D geometry tensor upload API is available."""
        if self._solver_h is None:
            return False
        return bool(hasattr(self._mod, "swe2d_set_3d_patch_geometry"))

    def supports_3d_patch_state_upload(self) -> bool:
        """Return True when native 3D patch state upload APIs are available."""
        if self._solver_h is None:
            return False
        return bool(
            hasattr(self._mod, "swe2d_set_3d_patch_state")
            or hasattr(self._mod, "swe2d_set_3d_patch_vof")
        )

    def supports_3d_patch_face_bc_upload(self) -> bool:
        """Return True when native per-face 3D BC runtime API is available."""
        if self._solver_h is None:
            return False
        return bool(hasattr(self._mod, "swe2d_set_3d_patch_face_bc"))

    def set_3d_patch_face_bc(
        self,
        *,
        face: int,
        mode: int,
        u: float = 0.0,
        v: float = 0.0,
        w: float = 0.0,
        q: float = 0.0,
        vof: float = 1.0,
        p: float = 0.0,
    ) -> None:
        """Update a single 3D patch face BC definition at runtime."""
        if self._solver_h is None:
            raise RuntimeError("initialize() must be called before set_3d_patch_face_bc().")
        if not hasattr(self._mod, "swe2d_set_3d_patch_face_bc"):
            raise RuntimeError("Native module does not expose swe2d_set_3d_patch_face_bc().")

        face_i = int(face)
        mode_i = int(mode)
        if face_i < 0 or face_i > 5:
            raise ValueError("face must be in [0..5] (XMIN,XMAX,YMIN,YMAX,ZMIN,ZMAX)")
        mode_i = max(0, min(4, mode_i))

        vof_f = float(vof)
        if not np.isfinite(vof_f):
            vof_f = 1.0
        vof_f = max(0.0, min(1.0, vof_f))

        self._mod.swe2d_set_3d_patch_face_bc(
            self._solver_h,
            face=face_i,
            mode=mode_i,
            u=float(u),
            v=float(v),
            w=float(w),
            q=float(q),
            vof=vof_f,
            p=float(p),
        )

    def set_3d_patch_vof(self, vof: np.ndarray) -> None:
        """Upload a full VoF array to the active 3D patch."""
        if self._solver_h is None:
            raise RuntimeError("initialize() must be called before set_3d_patch_vof().")
        if not hasattr(self._mod, "swe2d_set_3d_patch_vof"):
            raise RuntimeError("Native module does not expose swe2d_set_3d_patch_vof().")
        vof_arr = np.ascontiguousarray(vof, dtype=np.float64).ravel()
        self._mod.swe2d_set_3d_patch_vof(self._solver_h, vof_arr)

    def set_3d_patch_state(
        self,
        u: Optional[np.ndarray] = None,
        v: Optional[np.ndarray] = None,
        w: Optional[np.ndarray] = None,
        p: Optional[np.ndarray] = None,
        vof: Optional[np.ndarray] = None,
    ) -> None:
        """Upload any subset of per-cell 3D patch state arrays."""
        if self._solver_h is None:
            raise RuntimeError("initialize() must be called before set_3d_patch_state().")
        if hasattr(self._mod, "swe2d_set_3d_patch_state"):
            u_arr = None if u is None else np.ascontiguousarray(u, dtype=np.float64).ravel()
            v_arr = None if v is None else np.ascontiguousarray(v, dtype=np.float64).ravel()
            w_arr = None if w is None else np.ascontiguousarray(w, dtype=np.float64).ravel()
            p_arr = None if p is None else np.ascontiguousarray(p, dtype=np.float64).ravel()
            vof_arr = None if vof is None else np.ascontiguousarray(vof, dtype=np.float64).ravel()
            self._mod.swe2d_set_3d_patch_state(
                self._solver_h,
                u=u_arr,
                v=v_arr,
                w=w_arr,
                p=p_arr,
                vof=vof_arr,
            )
            return

        if vof is not None and hasattr(self._mod, "swe2d_set_3d_patch_vof"):
            self.set_3d_patch_vof(vof)
            return

        raise RuntimeError(
            "Native module does not expose swe2d_set_3d_patch_state() or compatible vof upload fallback."
        )

    def set_3d_patch_geometry(
        self,
        phi: Optional[np.ndarray] = None,
        ax: Optional[np.ndarray] = None,
        ay: Optional[np.ndarray] = None,
        az: Optional[np.ndarray] = None,
        sanitize: bool = False,
        phi_snap_min: float = 0.005,
        area_snap_min: float = 0.01,
    ) -> None:
        """
        Upload static 3D geometry tensors (phi/ax/ay/az) for sub-grid solids.

        Parameters
        ----------
        sanitize : bool, optional
            If True, clamp arrays to [0, 1] and snap very small values to zero.
            This is intended as a numerical-stability guard for sliver cut-cells.
        phi_snap_min : float, optional
            Cells with phi < phi_snap_min are snapped to fully solid (phi=0), and
            corresponding face-open fractions are also zeroed.
        area_snap_min : float, optional
            Face-open fractions ax/ay/az below this threshold are snapped to 0.
        """
        if self._solver_h is None:
            raise RuntimeError("initialize() must be called before set_3d_patch_geometry().")
        if not hasattr(self._mod, "swe2d_set_3d_patch_geometry"):
            raise RuntimeError("Native module does not expose swe2d_set_3d_patch_geometry().")

        phi_arr = None if phi is None else np.ascontiguousarray(phi, dtype=np.float64).ravel()
        ax_arr = None if ax is None else np.ascontiguousarray(ax, dtype=np.float64).ravel()
        ay_arr = None if ay is None else np.ascontiguousarray(ay, dtype=np.float64).ravel()
        az_arr = None if az is None else np.ascontiguousarray(az, dtype=np.float64).ravel()

        if phi_arr is None and ax_arr is None and ay_arr is None and az_arr is None:
            return

        if sanitize:
            try:
                phi_thr = float(phi_snap_min)
            except Exception:
                phi_thr = 0.0
            try:
                area_thr = float(area_snap_min)
            except Exception:
                area_thr = 0.0

            if not np.isfinite(phi_thr):
                phi_thr = 0.0
            if not np.isfinite(area_thr):
                area_thr = 0.0
            phi_thr = max(0.0, min(1.0, phi_thr))
            area_thr = max(0.0, min(1.0, area_thr))

            # Work on private copies so caller-owned arrays are never mutated.
            if phi_arr is not None:
                phi_arr = phi_arr.copy()
            if ax_arr is not None:
                ax_arr = ax_arr.copy()
            if ay_arr is not None:
                ay_arr = ay_arr.copy()
            if az_arr is not None:
                az_arr = az_arr.copy()

            def _sanitize_tensor(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
                if arr is None:
                    return None
                np.nan_to_num(arr, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
                np.clip(arr, 0.0, 1.0, out=arr)
                return arr

            phi_arr = _sanitize_tensor(phi_arr)
            ax_arr = _sanitize_tensor(ax_arr)
            ay_arr = _sanitize_tensor(ay_arr)
            az_arr = _sanitize_tensor(az_arr)

            if phi_arr is not None and phi_thr > 0.0:
                tiny_phi = phi_arr < phi_thr
                if np.any(tiny_phi):
                    phi_arr[tiny_phi] = 0.0
                    if ax_arr is not None:
                        ax_arr[tiny_phi] = 0.0
                    if ay_arr is not None:
                        ay_arr[tiny_phi] = 0.0
                    if az_arr is not None:
                        az_arr[tiny_phi] = 0.0

            if area_thr > 0.0:
                if ax_arr is not None:
                    ax_arr[ax_arr < area_thr] = 0.0
                if ay_arr is not None:
                    ay_arr[ay_arr < area_thr] = 0.0
                if az_arr is not None:
                    az_arr[az_arr < area_thr] = 0.0

            # If a cell has no open faces after snapping, treat it as solid.
            if phi_arr is not None and (ax_arr is not None or ay_arr is not None or az_arr is not None):
                area_sum = np.zeros_like(phi_arr)
                if ax_arr is not None:
                    area_sum += ax_arr
                if ay_arr is not None:
                    area_sum += ay_arr
                if az_arr is not None:
                    area_sum += az_arr
                isolated = (phi_arr > 0.0) & (area_sum <= 0.0)
                if np.any(isolated):
                    phi_arr[isolated] = 0.0

        self._mod.swe2d_set_3d_patch_geometry(
            self._solver_h,
            phi=phi_arr,
            ax=ax_arr,
            ay=ay_arr,
            az=az_arr,
        )

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

    # ─────────────────────────────────────────────────────────────────────────────
    # Phase 7: 2D-3D interface contract API
    # ─────────────────────────────────────────────────────────────────────────────

    def create_interface_contract(
        self,
        cell2d: np.ndarray,
        face_area: np.ndarray,
        face_nx: np.ndarray,
        face_ny: np.ndarray,
        face_nz: np.ndarray,
    ):
        """
        Create a 2D-3D interface contract from geometry arrays.
        
        Args:
            cell2d: int32 array of 2D cell indices (length N_FACES)
            face_area: float64 array of interface face areas (length N_FACES)
            face_nx, face_ny, face_nz: float64 arrays of outward face normals (length N_FACES)
        
        Returns:
            Contract handle (opaque object); pass to upload_interface_contract().
            Returns None on validation failure.
        """
        cell2d_arr = np.asarray(cell2d, dtype=np.int32, order='C')
        face_area_arr = np.asarray(face_area, dtype=np.float64, order='C')
        face_nx_arr = np.asarray(face_nx, dtype=np.float64, order='C')
        face_ny_arr = np.asarray(face_ny, dtype=np.float64, order='C')
        face_nz_arr = np.asarray(face_nz, dtype=np.float64, order='C')

        if not hasattr(self._mod, 'swe2d_contract_create'):
            raise RuntimeError(
                "Native module does not expose swe2d_contract_create. "
                "Rebuild with Phase 7 support (cmake --build build)."
            )

        return self._mod.swe2d_contract_create(
            cell2d_arr, face_area_arr, face_nx_arr, face_ny_arr, face_nz_arr
        )

        

    def is_interface_contract_valid(self, contract) -> bool:
        """
        Validate contract consistency before upload.
        
        Args:
            contract: Handle returned by create_interface_contract()
        
        Returns:
            True if contract is valid (all arrays same length > 0).
        """
        if contract is None:
            return False
        try:
            if not hasattr(self._mod, 'swe2d_contract_is_valid'):
                return False
            return self._mod.swe2d_contract_is_valid(contract)
        except Exception:
            return False

    def upload_interface_contract(self, contract) -> bool:
        """
        Upload contract geometry and allocate device buffers for 2D-3D exchange.
        
        Args:
            contract: Handle returned by create_interface_contract()
        
        Returns:
            True on success; False if allocation failed.
        """
        if self._solver_h is None:
            raise RuntimeError("Solver not initialized; call initialize() first.")
        if contract is None:
            raise ValueError("null contract handle")
        
        try:
            if not hasattr(self._mod, 'swe2d_gpu_contract_upload'):
                raise RuntimeError(
                    "Native module does not expose swe2d_gpu_contract_upload. "
                    "Rebuild with Phase 7 support (cmake --build build)."
                )
            return self._mod.swe2d_gpu_contract_upload(self._solver_h, contract)
        except Exception as e:
            print(f"Failed to upload interface contract: {e}")
            return False

    def clear_interface_contract(self) -> None:
        """
        Free device-side contract buffers (flux, head-loss, etc).
        
        Safe to call even if no contract is currently uploaded.
        """
        if self._solver_h is None:
            return
        
        try:
            if hasattr(self._mod, 'swe2d_gpu_contract_clear'):
                self._mod.swe2d_gpu_contract_clear(self._solver_h)
        except Exception as e:
            print(f"Warning: failed to clear interface contract: {e}")

    def is_interface_contract_uploaded(self) -> bool:
        """
        Query: is a 2D-3D interface contract currently uploaded?
        
        Returns:
            True if contract is uploaded to GPU; False otherwise.
        """
        if self._solver_h is None:
            return False
        
        try:
            if not hasattr(self._mod, 'swe2d_gpu_is_contract_uploaded'):
                return False
            return self._mod.swe2d_gpu_is_contract_uploaded(self._solver_h)
        except Exception:
            return False

    def __del__(self):
        try:
            self.destroy()
        except Exception:
            pass
