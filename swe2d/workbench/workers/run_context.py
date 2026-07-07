from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


def _noop(*_args, **_kwargs):
    return None


@dataclass(frozen=True)
class RunContext:
    """Immutable snapshot of everything needed to execute a run off the main thread."""

    # Identity / bookkeeping
    run_id: str
    run_wallclock_start: str
    run_log_start_idx: int
    results_gpkg_path: str = ""
    model_gpkg_path: str = ""
    mesh_name: str = ""
    mesh_crs_wkt: str = ""

    # Simulation parameters
    run_duration_s: float = 0.0
    output_interval_s: float = 1.0
    dt_cfg: float = 0.05
    dt_request: float = 0.05
    dt_fixed: float = -1.0
    initial_dt: float = 0.0
    adaptive_cfl_dt: bool = False
    reconstruction_mode: int = 0
    reconstruction_name: str = ""
    temporal_scheme: Any = None
    temporal_scheme_name: str = ""
    solver_backend_mode: str = "gpu"
    coupling_loop_mode: str = "cuda"
    drainage_solver_backend_mode: str = "gpu"
    drainage_gpu_method_mode: str = "step"
    culvert_solver_mode: int = 0
    cuda_graphs_enabled: bool = False
    bridge_cuda_coupling: bool = False
    bridge_stacked_coupling_mode: str = "phase3_spatial"
    culvert_face_flux_mode: str = "off"

    # Solver numerics
    gravity: float = 9.81
    k_mann: float = 1.0
    n_mann: float = 0.035
    cfl: float = 0.45
    h_min: float = 1.0e-4
    max_inv_area: float = 0.0
    cfl_lambda_cap: float = 0.0
    momentum_cap_min_speed: float = 0.0
    momentum_cap_celerity_mult: float = 0.0
    depth_cap: float = 0.0
    max_rel_depth_increase: float = 0.0
    shallow_damping_depth: float = 0.0
    source_cfl_beta: float = 0.0
    source_max_substeps: int = 1
    source_rate_cap: float = 0.0
    source_depth_step_cap: float = 0.0
    source_true_subcycling: bool = False
    source_imex_split: bool = False
    gpu_diag_sync_interval_steps: int = 0
    tiny_mode: int = 0
    tiny_wet_cell_threshold: int = 0
    degen_mode: int = 0
    front_flux_damping: float = 0.0
    open_bc_relaxation: float = 0.0
    active_set_hysteresis: bool = False
    use_redistribution: bool = False
    inflow_progressive: bool = False
    uniform_inflow_enabled: bool = False
    rain_update_interval_s: float = 60.0

    # Mesh / state arrays
    node_x: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    node_y: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    node_z: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    cell_nodes: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.int32))
    face_offsets: Optional[np.ndarray] = None
    face_nodes: Optional[np.ndarray] = None
    bc_n0: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    bc_n1: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    bc_tp: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    bc_vl: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    bc_relax: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    side_hydrographs: Dict[str, Any] = field(default_factory=dict)
    edge_hydrographs: Dict[Tuple[int, int], Any] = field(default_factory=dict)
    edge_group_overrides: Dict[int, str] = field(default_factory=dict)
    h0: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    hu0: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    hv0: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    n_mann_cell: Optional[np.ndarray] = None
    cell_areas: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    cell_centroids: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.float64))

    # Forcing / coupling
    rain_rate_model: Any = 0.0
    internal_flow_forcing: Any = None
    cell_source_model: Any = None
    thiessen_forcing: Any = None
    pipe_network_cfg: Any = None
    hydraulic_structures_cfg: Any = None
    bridge_stacked_plans: List[Any] = field(default_factory=list)
    coupling_soa: Any = None

    # Storage flags
    save_mesh_results: bool = False
    save_line_results: bool = False
    save_coupling_results: bool = False
    save_run_log: bool = False
    save_max_only: bool = False

    # Unit system
    length_unit_name: str = "m"
    length_scale_si_to_model: float = 1.0
    rain_mm_to_model_depth: float = 1.0
    rain_rate_si_to_model: float = 1.0
    flow_si_to_model: float = 1.0

    # Runtime callbacks that do not touch Qt widgets
    apply_timeseries_bc_values: Callable = field(default=_noop)
    distribute_total_flow_to_unit_q: Callable = field(default=_noop)
    apply_external_sources: Callable = field(default=lambda *a, **k: None)
    build_line_sampling_map: Callable = field(default=lambda: None)
    mesh_cell_areas: Callable = field(default=lambda: np.empty(0))
    mesh_cell_min_bed: Callable = field(default=lambda: np.empty(0))
    mesh_cell_centroids: Callable = field(default=lambda: np.empty((0, 2)))
    internal_flow_source_cms_at_time: Callable = field(default=lambda *a, **k: None)

    # Plain-data values captured on the main thread before the worker starts
    # (so per-step callbacks don't touch Qt widgets from worker thread).
    sample_map_data: List[Dict[str, object]] = field(default_factory=list)
    inflow_progressive_enabled: bool = False
    edge_groups: Dict[int, str] = field(default_factory=dict)

    # Cancel signal
    cancel_event: threading.Event = field(default_factory=threading.Event)
