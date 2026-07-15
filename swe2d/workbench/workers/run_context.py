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

    def to_replay_json(self) -> Dict[str, Any]:
        """Serialize to the canonical replay JSON shape.

        Mesh arrays and callable callbacks are NOT included.
        """
        return {
            "schema_version": "swe2d-replay/1",
            "run_id": self.run_id,
            "mesh": {
                "gpkg_path": self.model_gpkg_path,
                "mesh_name": self.mesh_name,
                "crs_wkt": self.mesh_crs_wkt,
            },
            "params": {
                "run_duration_s": self.run_duration_s,
                "output_interval_s": self.output_interval_s,
                "dt_cfg": self.dt_cfg,
                "dt_request": self.dt_request,
                "dt_fixed": self.dt_fixed,
                "initial_dt": self.initial_dt,
                "adaptive_cfl_dt": self.adaptive_cfl_dt,
                "reconstruction_mode": self.reconstruction_mode,
                "reconstruction_name": self.reconstruction_name,
                "temporal_scheme": self._scalar_val(self.temporal_scheme),
                "temporal_scheme_name": self.temporal_scheme_name,
                "solver_backend_mode": self.solver_backend_mode,
                "coupling_loop_mode": self.coupling_loop_mode,
                "drainage_solver_backend_mode": self.drainage_solver_backend_mode,
                "drainage_gpu_method_mode": self.drainage_gpu_method_mode,
                "culvert_solver_mode": self.culvert_solver_mode,
                "cuda_graphs_enabled": self.cuda_graphs_enabled,
                "bridge_cuda_coupling": self.bridge_cuda_coupling,
                "bridge_stacked_coupling_mode": self.bridge_stacked_coupling_mode,
                "culvert_face_flux_mode": self.culvert_face_flux_mode,
                "gravity": self.gravity,
                "k_mann": self.k_mann,
                "n_mann": self.n_mann,
                "cfl": self.cfl,
                "h_min": self.h_min,
                "max_inv_area": self.max_inv_area,
                "cfl_lambda_cap": self.cfl_lambda_cap,
                "momentum_cap_min_speed": self.momentum_cap_min_speed,
                "momentum_cap_celerity_mult": self.momentum_cap_celerity_mult,
                "depth_cap": self.depth_cap,
                "max_rel_depth_increase": self.max_rel_depth_increase,
                "shallow_damping_depth": self.shallow_damping_depth,
                "source_cfl_beta": self.source_cfl_beta,
                "source_max_substeps": self.source_max_substeps,
                "source_rate_cap": self.source_rate_cap,
                "source_depth_step_cap": self.source_depth_step_cap,
                "source_true_subcycling": self.source_true_subcycling,
                "source_imex_split": self.source_imex_split,
                "gpu_diag_sync_interval_steps": self.gpu_diag_sync_interval_steps,
                "tiny_mode": self.tiny_mode,
                "tiny_wet_cell_threshold": self.tiny_wet_cell_threshold,
                "degen_mode": self.degen_mode,
                "front_flux_damping": self.front_flux_damping,
                "open_bc_relaxation": self.open_bc_relaxation,
                "active_set_hysteresis": self.active_set_hysteresis,
                "use_redistribution": self.use_redistribution,
                "inflow_progressive": self.inflow_progressive,
                "uniform_inflow_enabled": self.uniform_inflow_enabled,
                "rain_update_interval_s": self.rain_update_interval_s,
            },
            "data_sources": {},
            "results": {
                "results_gpkg_path": self.results_gpkg_path,
                "save_line_results": self.save_line_results,
                "save_coupling_results": self.save_coupling_results,
                "save_mesh_results": self.save_mesh_results,
                "save_run_log": self.save_run_log,
                "save_max_only": self.save_max_only,
            },
            "units": {
                "length_unit_name": self.length_unit_name,
                "length_scale_si_to_model": self.length_scale_si_to_model,
                "rain_mm_to_model_depth": self.rain_mm_to_model_depth,
                "rain_rate_si_to_model": self.rain_rate_si_to_model,
                "flow_si_to_model": self.flow_si_to_model,
            },
            "coupling_soa_blob_b64": None,
            "bridge_stacked_plans_b64": None,
            "h0": None,
            "side_hydrographs": None,
            "edge_hydrographs": None,
            "edge_group_overrides": None,
            "id": self.run_id or "current_setup",
        }

    @staticmethod
    def _scalar_val(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        if hasattr(value, "value"):
            return value.value
        if hasattr(value, "name"):
            return value.name
        return str(value)

    @staticmethod
    def from_replay_json(payload: Dict[str, Any]) -> "RunContext":
        """Build a RunContext from canonical replay JSON (params only, no mesh arrays)."""
        mesh = payload.get("mesh", {})
        params = payload.get("params", {})
        results = payload.get("results", {})
        units = payload.get("units", {})

        run_id = payload.get("run_id", payload.get("id", "replay"))

        return RunContext(
            run_id=run_id,
            run_wallclock_start="",
            run_log_start_idx=0,
            model_gpkg_path=mesh.get("gpkg_path", ""),
            mesh_name=mesh.get("mesh_name", ""),
            mesh_crs_wkt=mesh.get("crs_wkt", ""),
            results_gpkg_path=results.get("results_gpkg_path", ""),

            run_duration_s=float(params.get("run_duration_s", 0.0)),
            output_interval_s=float(params.get("output_interval_s", 1.0)),
            dt_cfg=float(params.get("dt_cfg", 0.05)),
            dt_request=float(params.get("dt_request", 0.05)),
            dt_fixed=float(params.get("dt_fixed", -1.0)),
            adaptive_cfl_dt=bool(params.get("adaptive_cfl_dt", False)),
            reconstruction_mode=int(params.get("reconstruction_mode", 0)),
            reconstruction_name=str(params.get("reconstruction_name", "")),
            temporal_scheme=params.get("temporal_scheme"),
            temporal_scheme_name=str(params.get("temporal_scheme_name", "")),
            cuda_graphs_enabled=bool(params.get("cuda_graphs_enabled", False)),
            bridge_cuda_coupling=bool(params.get("bridge_cuda_coupling", False)),
            bridge_stacked_coupling_mode=str(params.get("bridge_stacked_coupling_mode", "phase3_spatial")),
            culvert_face_flux_mode=str(params.get("culvert_face_flux_mode", "off")),
            culvert_solver_mode=int(params.get("culvert_solver_mode", 0)),
            drainage_gpu_method_mode=str(params.get("drainage_gpu_method_mode", "step")),
            use_redistribution=bool(params.get("use_redistribution", False)),
            inflow_progressive=bool(params.get("inflow_progressive", False)),
            uniform_inflow_enabled=bool(params.get("uniform_inflow_enabled", False)),
            rain_update_interval_s=float(params.get("rain_update_interval_s", 60.0)),

            gravity=float(params.get("gravity", 9.81)),
            k_mann=float(params.get("k_mann", 1.0)),
            n_mann=float(params.get("n_mann", 0.035)),
            cfl=float(params.get("cfl", 0.45)),
            h_min=float(params.get("h_min", 1.0e-4)),
            max_inv_area=float(params.get("max_inv_area", 0.0)),
            cfl_lambda_cap=float(params.get("cfl_lambda_cap", 0.0)),
            momentum_cap_min_speed=float(params.get("momentum_cap_min_speed", 0.0)),
            momentum_cap_celerity_mult=float(params.get("momentum_cap_celerity_mult", 0.0)),
            depth_cap=float(params.get("depth_cap", 0.0)),
            max_rel_depth_increase=float(params.get("max_rel_depth_increase", 0.0)),
            shallow_damping_depth=float(params.get("shallow_damping_depth", 0.0)),
            source_cfl_beta=float(params.get("source_cfl_beta", 0.0)),
            source_max_substeps=int(params.get("source_max_substeps", 1)),
            source_rate_cap=float(params.get("source_rate_cap", 0.0)),
            source_depth_step_cap=float(params.get("source_depth_step_cap", 0.0)),
            source_true_subcycling=bool(params.get("source_true_subcycling", False)),
            source_imex_split=bool(params.get("source_imex_split", False)),
            gpu_diag_sync_interval_steps=int(params.get("gpu_diag_sync_interval_steps", 0)),
            tiny_mode=int(params.get("tiny_mode", 0)),
            tiny_wet_cell_threshold=int(params.get("tiny_wet_cell_threshold", 0)),
            degen_mode=int(params.get("degen_mode", 0)),
            front_flux_damping=float(params.get("front_flux_damping", 0.0)),
            open_bc_relaxation=float(params.get("open_bc_relaxation", 0.0)),
            active_set_hysteresis=bool(params.get("active_set_hysteresis", False)),

            length_unit_name=str(units.get("length_unit_name", "m")),
            length_scale_si_to_model=float(units.get("length_scale_si_to_model", 1.0)),
            rain_mm_to_model_depth=float(units.get("rain_mm_to_model_depth", 1.0)),

            save_mesh_results=bool(results.get("save_mesh_results", False)),
            save_line_results=bool(results.get("save_line_results", False)),
            save_coupling_results=bool(results.get("save_coupling_results", False)),
            save_run_log=bool(results.get("save_run_log", False)),
            save_max_only=bool(results.get("save_max_only", False)),
        )

    @staticmethod
    def from_widget_params(widget_params: dict, mesh_name: str = "", mesh_gpkg: str = "", results_gpkg: str = "") -> "RunContext":
        """Build a minimal RunContext from widget param dicts (used by batch dialog snapshot).

        Uses the existing _widget_params_to_run_params helper for name mapping.
        """
        from swe2d.workbench.dialogs.batch_simulation_dialog import _widget_params_to_run_params
        run_params = _widget_params_to_run_params(widget_params)
        flat = run_params.get("params", run_params)
        return RunContext(
            run_id="current_setup",
            run_wallclock_start="",
            run_log_start_idx=0,
            model_gpkg_path=mesh_gpkg,
            mesh_name=mesh_name,
            results_gpkg_path=results_gpkg,

            run_duration_s=float(flat.get("run_duration_s", 0.0)),
            output_interval_s=float(flat.get("output_interval_s", 1.0)),
            dt_cfg=float(flat.get("dt_cfg", 0.05)),
            dt_request=float(flat.get("dt_request", 0.05)),
            dt_fixed=float(flat.get("dt_fixed", -1.0)),
            adaptive_cfl_dt=bool(flat.get("adaptive_cfl_dt", False)),
            reconstruction_mode=int(flat.get("reconstruction_mode", 0)),
            reconstruction_name=str(flat.get("reconstruction_name", "")),
            temporal_scheme=flat.get("temporal_scheme"),
            cuda_graphs_enabled=bool(flat.get("cuda_graphs_enabled", False)),
            bridge_cuda_coupling=bool(flat.get("bridge_cuda_coupling", False)),
            bridge_stacked_coupling_mode=str(flat.get("bridge_stacked_coupling_mode", "phase3_spatial")),
            culvert_face_flux_mode=str(flat.get("culvert_face_flux_mode", "off")),
            culvert_solver_mode=int(flat.get("culvert_solver_mode", 0)),
            drainage_gpu_method_mode=str(flat.get("drainage_gpu_method_mode", "step")),

            gravity=float(flat.get("gravity", 9.81)),
            k_mann=float(flat.get("k_mann", 1.0)),
            n_mann=float(flat.get("n_mann", 0.035)),
            cfl=float(flat.get("cfl", 0.45)),
            h_min=float(flat.get("h_min", 1.0e-4)),
            max_inv_area=float(flat.get("max_inv_area", 0.0)),
            cfl_lambda_cap=float(flat.get("cfl_lambda_cap", 0.0)),
            momentum_cap_min_speed=float(flat.get("momentum_cap_min_speed", 0.0)),
            momentum_cap_celerity_mult=float(flat.get("momentum_cap_celerity_mult", 0.0)),
            depth_cap=float(flat.get("depth_cap", 0.0)),
            max_rel_depth_increase=float(flat.get("max_rel_depth_increase", 0.0)),
            shallow_damping_depth=float(flat.get("shallow_damping_depth", 0.0)),
            source_cfl_beta=float(flat.get("source_cfl_beta", 0.0)),
            source_max_substeps=int(flat.get("source_max_substeps", 1)),
            source_rate_cap=float(flat.get("source_rate_cap", 0.0)),
            source_depth_step_cap=float(flat.get("source_depth_step_cap", 0.0)),
            source_true_subcycling=bool(flat.get("source_true_subcycling", False)),
            source_imex_split=bool(flat.get("source_imex_split", False)),
            gpu_diag_sync_interval_steps=int(flat.get("gpu_diag_sync_interval_steps", 0)),
            tiny_mode=int(flat.get("tiny_mode", 0)),
            tiny_wet_cell_threshold=int(flat.get("tiny_wet_cell_threshold", 0)),
            degen_mode=int(flat.get("degen_mode", 0)),
            front_flux_damping=float(flat.get("front_flux_damping", 0.0)),
            open_bc_relaxation=float(flat.get("open_bc_relaxation", 0.0)),
            active_set_hysteresis=bool(flat.get("active_set_hysteresis", False)),
            use_redistribution=bool(flat.get("use_redistribution", False)),
            inflow_progressive=bool(flat.get("inflow_progressive", False)),
            uniform_inflow_enabled=bool(flat.get("uniform_inflow_enabled", False)),
            rain_update_interval_s=float(flat.get("rain_update_interval_s", 60.0)),

            length_unit_name=str(flat.get("length_unit_name", "m")),
            length_scale_si_to_model=float(flat.get("length_scale_si_to_model", 1.0)),
            rain_mm_to_model_depth=float(flat.get("rain_mm_to_model_depth", 1.0)),

            save_mesh_results=bool(flat.get("save_mesh_results", False)),
            save_line_results=bool(flat.get("save_line_results", False)),
            save_coupling_results=bool(flat.get("save_coupling_results", False)),
            save_run_log=bool(flat.get("save_run_log", False)),
            save_max_only=bool(flat.get("save_max_only", False)),
        )
