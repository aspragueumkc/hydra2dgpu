from __future__ import annotations

import datetime
import logging
import os
import threading
import time
import traceback
from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from qgis.PyQt.QtCore import QThread, pyqtSignal

from swe2d import units as _u
from swe2d.boundary_and_forcing.runtime_source_logic import (
    permute_internal_flow_forcing,
    permute_thiessen_forcing,
)
from swe2d.runtime.backend import SWE2DBackend
from swe2d.runtime.backend_initializer import SWE2DBackendInitializer
from swe2d.runtime.coupling import build_coupling_controller
from swe2d.runtime.native_bc_forcing import SWE2DNativeBoundaryHydrographConfigurator
from swe2d.runtime.runtime_reporting import SWE2DRuntimeReporter
from swe2d.runtime.runtime_setup_configurator import SWE2DRunSetupConfigurator
from swe2d.runtime.runtime_sources import SWE2DRuntimeSourceManager
from swe2d.runtime.runtime_step_executor import SWE2DRuntimeStepExecutor
from swe2d.workbench.services.constants_service import BC_INFLOW_Q as _BC_INFLOW_Q
from swe2d.workbench.services.constants_service import BC_TS_FLOW as _BC_TS_FLOW
from swe2d.workbench.services.constants_service import BC_TS_STAGE as _BC_TS_STAGE
from swe2d.workbench.services.mesh_service import apply_cell_permutation, classify_boundary_edges
from swe2d.workbench.services.non_gui_runtime_service import (
    build_coupling_keys,
    build_pipe_cell_keys,
    execute_run_timestep_loop as _execute_run_timestep_loop_runtime_logic,
    _sample_coupling_object_metrics,
)
from swe2d.results.data import SWE2DResultsData
from swe2d.workbench.workers.run_context import RunContext


logger = logging.getLogger(__name__)


class SnapshotData:
    """Data emitted when a device snapshot is ready for UI sync."""

    def __init__(
        self,
        t_s: float = 0.0,
        h: Optional[np.ndarray] = None,
        hu: Optional[np.ndarray] = None,
        hv: Optional[np.ndarray] = None,
        timesteps: Any = None,
        line_ts: Any = None,
        line_profiles: Any = None,
        coupling_data: Any = None,
        pipe_cell_data: Any = None,
    ):
        self.timesteps = timesteps
        self.t_s = t_s
        self.h = h
        self.hu = hu
        self.hv = hv
        self.line_ts = line_ts
        self.line_profiles = line_profiles
        self.coupling_data = coupling_data
        self.pipe_cell_data = pipe_cell_data


class ComputeResult:
    """Result emitted when the simulation worker finishes compute."""

    def __init__(
        self,
        *,
        ok: bool,
        h: np.ndarray,
        hu: np.ndarray,
        hv: np.ndarray,
        final_sim_time_s: float,
        n_area: int,
        area_model: np.ndarray,
        storage_start_model: float,
        source_budget_model: Dict[str, float],
        source_step_rows_model: List[Dict[str, float]],
        run_duration_s: float,
        boundary_flux_budget_model: Dict[str, float],
        boundary_flux_step_rows_model: List[Dict[str, float]],
        run_id: str,
        mesh_name: str,
        output_interval_s: float,
        run_perf_start: float,
        run_wallclock_start: str,
        run_log_start_idx: int,
        thiessen_forcing: Any,
        rain_stats_acc: Dict[str, float],
        max_tracking: Optional[Dict[str, np.ndarray]],
        snapshot_timesteps: List[Tuple[float, np.ndarray, np.ndarray, np.ndarray]],
        coupling_snapshots: Dict[Tuple[str, str, str], Dict[str, Any]],
        precomputed_line_results: Any,
        cancelled: bool = False,
        error_message: str = "",
        save_line_results: bool = False,
        save_coupling_results: bool = False,
        save_mesh_results: bool = False,
        save_run_log: bool = False,
        save_max_only: bool = False,
        h_min: float = 1.0e-4,
    ):
        self.ok = ok
        self.h = h
        self.hu = hu
        self.hv = hv
        self.final_sim_time_s = final_sim_time_s
        self.n_area = n_area
        self.area_model = area_model
        self.storage_start_model = storage_start_model
        self.source_budget_model = source_budget_model
        self.source_step_rows_model = source_step_rows_model
        self.run_duration_s = run_duration_s
        self.boundary_flux_budget_model = boundary_flux_budget_model
        self.boundary_flux_step_rows_model = boundary_flux_step_rows_model
        self.run_id = run_id
        self.mesh_name = mesh_name
        self.output_interval_s = output_interval_s
        self.run_perf_start = run_perf_start
        self.run_wallclock_start = run_wallclock_start
        self.run_log_start_idx = run_log_start_idx
        self.thiessen_forcing = thiessen_forcing
        self.rain_stats_acc = rain_stats_acc
        self.max_tracking = max_tracking
        self.snapshot_timesteps = snapshot_timesteps
        self.coupling_snapshots = coupling_snapshots
        self.precomputed_line_results = precomputed_line_results
        self.cancelled = cancelled
        self.error_message = error_message
        self.save_line_results = save_line_results
        self.save_coupling_results = save_coupling_results
        self.save_mesh_results = save_mesh_results
        self.save_run_log = save_run_log
        self.save_max_only = save_max_only
        self.h_min = h_min



class _WorkbenchShim:
    """Adapts the worker/context to the ``wb`` object expected by the runtime loop."""

    def __init__(
        self,
        worker: "SimulationWorker",
        ctx: RunContext,
        results_data: SWE2DResultsData,
        mesh_data: Dict[str, Any],
    ) -> None:
        from swe2d.workbench.services.runtime_source_application_service import (
            _apply_external_sources_logic,
            _distribute_total_flow_to_unit_q_logic,
        )

        self._worker = worker
        self._ctx = ctx
        self._results_data = results_data
        self._mesh_data = mesh_data
        # Per-step callbacks must not touch Qt widgets.  Bind the pure
        # *_logic helpers to the values captured on the main thread into
        # the RunContext, so the runtime loop runs entirely off the GUI
        # thread without any silent UI cross-thread access.
        node_x = ctx.node_x
        node_y = ctx.node_y
        node_z = ctx.node_z
        edge_groups = ctx.edge_groups

        from swe2d.boundary_and_forcing.bc_logic import apply_timeseries_bc_values as _apply_timeseries_bc_values_logic
        from swe2d.boundary_and_forcing.boundary_runtime_logic import classify_boundary_edges as _bc_side_classifier
        _side_idx = _bc_side_classifier(ctx.bc_n0, ctx.bc_n1, node_x, node_y)

        def _apply_timeseries_bc_values(
            edge_n0, edge_n1, bc_type, bc_val, side_hydrographs, t_sec, edge_hydrographs=None,
        ):
            return _apply_timeseries_bc_values_logic(
                edge_n0=edge_n0, edge_n1=edge_n1, bc_type=bc_type, bc_val=bc_val,
                side_hydrographs=side_hydrographs,
                node_x=node_x, node_y=node_y,
                t_sec=t_sec,
                ts_flow_code=1, ts_stage_code=2,
                edge_hydrographs=edge_hydrographs,
                _side_idx=_side_idx,
            )

        self._apply_timeseries_bc_values = _apply_timeseries_bc_values
        mesh_snapshot = {
            "node_x": node_x,
            "node_y": node_y,
            "node_z": node_z,
            "cell_nodes": mesh_data.get("cell_nodes", np.empty((0, 3), dtype=np.int32)),
        }
        cfo = mesh_data.get("cell_face_offsets")
        cfn = mesh_data.get("cell_face_nodes")
        if cfo is not None:
            mesh_snapshot["cell_face_offsets"] = cfo
        if cfn is not None:
            mesh_snapshot["cell_face_nodes"] = cfn

        max_source_rate = float(ctx.source_rate_cap)
        h_min = float(ctx.h_min)
        max_rel_depth_increase = float(ctx.max_rel_depth_increase)
        max_source_depth_step = float(ctx.source_depth_step_cap)
        shallow_damping_depth = float(ctx.shallow_damping_depth)
        momentum_cap_min_speed = float(ctx.momentum_cap_min_speed)
        momentum_cap_celerity_mult = float(ctx.momentum_cap_celerity_mult)
        progressive = bool(ctx.inflow_progressive_enabled)

        def _apply_external_sources(
            backend,
            dt_step,
            rain_rate_model,
            cell_source_model=None,
            coupled_source_rate=None,
        ):
            if cell_source_model is not None:
                _cell_areas = backend.cell_areas()
            else:
                _cell_areas = None
            _apply_external_sources_logic(
                backend=backend,
                dt_step=dt_step,
                rain_rate_model=rain_rate_model,
                cell_source_model=cell_source_model,
                coupled_source_rate=coupled_source_rate,
                mesh_cell_areas=_cell_areas,
                max_source_rate=max_source_rate,
                h_min=h_min,
                max_rel_depth_increase=max_rel_depth_increase,
                max_source_depth_step=max_source_depth_step,
                shallow_damping_depth=shallow_damping_depth,
                momentum_cap_min_speed=momentum_cap_min_speed,
                momentum_cap_celerity_mult=momentum_cap_celerity_mult,
            )

        def _distribute_total_flow_to_unit_q(
            edge_n0,
            edge_n1,
            bc_type_step,
            bc_val_step,
            bc_type_template,
            side_hydrographs,
            edge_hydrographs=None,
            edge_groups_arg=None,
        ):
            eg = edge_groups_arg if edge_groups_arg is not None else edge_groups
            return _distribute_total_flow_to_unit_q_logic(
                edge_n0=edge_n0,
                edge_n1=edge_n1,
                bc_type_step=bc_type_step,
                bc_val_step=bc_val_step,
                bc_type_template=bc_type_template,
                side_hydrographs=side_hydrographs,
                node_x=node_x,
                node_y=node_y,
                node_z=node_z,
                progressive=progressive,
                edge_hydrographs=edge_hydrographs,
                edge_groups=eg,
            )

        self._apply_external_sources = _apply_external_sources
        self._distribute_total_flow_to_unit_q = _distribute_total_flow_to_unit_q
        self._length_unit_name = ctx.length_unit_name
        self._log = worker.log_message.emit

    @property
    def _cancel_requested(self) -> bool:
        return self._ctx.cancel_event.is_set()


class _PermutationResult:
    """Thread-safe holder for main-thread mesh permutation results."""

    def __init__(self):
        self.event = threading.Event()
        self.sample_map: List[Dict[str, Any]] = []
        self.error: str = ""


class SimulationWorker(QThread):
    """Background worker that owns the SWE2D backend and runs the timestep loop."""

    log_message = pyqtSignal(str)
    progress_percent = pyqtSignal(int)
    snapshot_ready = pyqtSignal(object)
    compute_finished = pyqtSignal(object)
    compute_failed = pyqtSignal(str)
    mesh_permutation_ready = pyqtSignal(object, object)

    def __init__(self, context: RunContext, parent=None):
        super().__init__(parent)
        self._context = context
        self._runtime_reporter: Optional[SWE2DRuntimeReporter] = None

    def request_snapshot(self):
        """Request a snapshot readback on the next reporter step."""
        reporter = self._runtime_reporter
        if reporter is not None:
            reporter.request_snapshot_readback()

    def request_cancel(self):
        """Signal the worker thread to stop at the next timestep check."""
        self._context.cancel_event.set()

    def run(self):
        try:
            result = self._execute()
            self.compute_finished.emit(result)
        except Exception as exc:
            self.log_message.emit(f"[ERROR] Simulation worker failed: {exc}")
            self.log_message.emit(traceback.format_exc())
            self.compute_failed.emit(str(exc))

    def _execute(self) -> ComputeResult:
        ctx = self._context
        log = self.log_message.emit
        run_perf_start = time.perf_counter()
        run_id = ctx.run_id
        run_wallclock_start = ctx.run_wallclock_start
        run_log_start_idx = ctx.run_log_start_idx

        node_x = ctx.node_x
        node_y = ctx.node_y
        node_z = ctx.node_z
        cell_nodes = ctx.cell_nodes
        face_offsets = ctx.face_offsets
        face_nodes = ctx.face_nodes
        bc_n0 = ctx.bc_n0
        bc_n1 = ctx.bc_n1
        bc_tp = ctx.bc_tp
        bc_vl = ctx.bc_vl
        side_hydrographs = ctx.side_hydrographs
        edge_hydrographs = ctx.edge_hydrographs
        edge_group_overrides = ctx.edge_group_overrides
        h0 = ctx.h0
        hu0 = ctx.hu0
        hv0 = ctx.hv0
        n_mann_cell = ctx.n_mann_cell

        dt_cfg = ctx.dt_cfg
        dt_request = ctx.dt_request
        dt_fixed = ctx.dt_fixed
        initial_dt = ctx.initial_dt
        run_duration_s = ctx.run_duration_s
        output_interval_s = ctx.output_interval_s
        reconstruction_mode = ctx.reconstruction_mode
        temporal_scheme = ctx.temporal_scheme

        dynamic_bc = bool(
            np.any((bc_tp == _BC_TS_FLOW) | (bc_tp == _BC_TS_STAGE)) or edge_hydrographs
        )

        mesh_data: Dict[str, Any] = {
            "node_x": node_x,
            "node_y": node_y,
            "node_z": node_z,
            "cell_nodes": cell_nodes,
            "mesh_name": ctx.mesh_name,
            "crs_wkt": ctx.mesh_crs_wkt,
        }
        if face_offsets is not None:
            mesh_data["cell_face_offsets"] = face_offsets
        if face_nodes is not None:
            mesh_data["cell_face_nodes"] = face_nodes

        if not mesh_data.get("mesh_name"):
            _gpkg = ctx.results_gpkg_path or ctx.model_gpkg_path or ""
            _stem = os.path.splitext(os.path.basename(_gpkg))[0] if _gpkg else "mesh"
            mesh_data["mesh_name"] = (
                f"{_stem}_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            )

        from swe2d.boundary_and_forcing.bc_logic import apply_timeseries_bc_values as _apply_timeseries_bc_values_logic
        from swe2d.boundary_and_forcing.boundary_runtime_logic import classify_boundary_edges as _bc_side_classifier
        from swe2d.workbench.services.runtime_source_application_service import _distribute_total_flow_to_unit_q_logic as _distribute_total_flow_to_unit_q_logic_fn
        _side_idx_init = _bc_side_classifier(bc_n0, bc_n1, node_x, node_y)
        _edge_groups_init = dict(ctx.edge_groups)
        _progressive_init = bool(ctx.inflow_progressive_enabled)

        def _apply_bc_init(edge_n0, edge_n1, bc_type, bc_val, side_hydrographs, t_sec, edge_hydrographs=None):
            return _apply_timeseries_bc_values_logic(
                edge_n0=edge_n0, edge_n1=edge_n1, bc_type=bc_type, bc_val=bc_val,
                side_hydrographs=side_hydrographs,
                node_x=node_x, node_y=node_y,
                t_sec=t_sec,
                ts_flow_code=1, ts_stage_code=2,
                edge_hydrographs=edge_hydrographs,
                _side_idx=_side_idx_init,
            )

        def _distribute_bc_init(edge_n0, edge_n1, bc_type_step, bc_val_step, bc_type_template, side_hydrographs, edge_hydrographs=None, edge_groups_arg=None):
            eg = edge_groups_arg if edge_groups_arg is not None else _edge_groups_init
            return _distribute_total_flow_to_unit_q_logic_fn(
                edge_n0=edge_n0, edge_n1=edge_n1,
                bc_type_step=bc_type_step, bc_val_step=bc_val_step,
                bc_type_template=bc_type_template,
                side_hydrographs=side_hydrographs,
                node_x=node_x, node_y=node_y, node_z=node_z,
                progressive=_progressive_init,
                edge_hydrographs=edge_hydrographs,
                edge_groups=eg,
            )

        backend = None
        try:
            backend_initializer = SWE2DBackendInitializer(
                apply_timeseries_bc_values_callback=_apply_bc_init,
                distribute_total_flow_to_unit_q_callback=_distribute_bc_init,
            )

            def _build_and_initialize_backend() -> SWE2DBackend:
                return backend_initializer.build_and_initialize(
                    backend_cls=SWE2DBackend,
                    dynamic_bc=dynamic_bc,
                    node_x=node_x,
                    node_y=node_y,
                    node_z=node_z,
                    cell_nodes=cell_nodes,
                    face_offsets=face_offsets,
                    face_nodes=face_nodes,
                    bc_n0=bc_n0,
                    bc_n1=bc_n1,
                    bc_tp=bc_tp,
                    bc_vl=bc_vl,
                    side_hydrographs=side_hydrographs,
                    edge_hydrographs=edge_hydrographs,
                    h0=h0,
                    hu0=hu0,
                    hv0=hv0,
                    n_mann_cell=n_mann_cell,
                    dt_fixed=dt_fixed,
                    dt_max=dt_cfg,
                    dt_initial=initial_dt,
                    reconstruction_mode=reconstruction_mode,
                    temporal_scheme=temporal_scheme,
                    gravity=ctx.gravity,
                    k_mann=ctx.k_mann,
                    n_mann=ctx.n_mann,
                    cfl=ctx.cfl,
                    h_min=ctx.h_min,
                    max_inv_area=ctx.max_inv_area,
                    cfl_lambda_cap=ctx.cfl_lambda_cap,
                    momentum_cap_min_speed=ctx.momentum_cap_min_speed,
                    momentum_cap_celerity_mult=ctx.momentum_cap_celerity_mult,
                    depth_cap=ctx.depth_cap,
                    max_rel_depth_increase=ctx.max_rel_depth_increase,
                    shallow_damping_depth=ctx.shallow_damping_depth,
                    source_cfl_beta=ctx.source_cfl_beta,
                    source_max_substeps=ctx.source_max_substeps,
                    source_rate_cap=ctx.source_rate_cap,
                    source_depth_step_cap=ctx.source_depth_step_cap,
                    source_true_subcycling=ctx.source_true_subcycling,
                    source_imex_split=ctx.source_imex_split,
                    enable_shallow_front_recon_fallback=False,
                    gpu_diag_sync_interval_steps=ctx.gpu_diag_sync_interval_steps,
                    tiny_mode=ctx.tiny_mode,
                    tiny_wet_cell_threshold=ctx.tiny_wet_cell_threshold,
                    degen_mode=ctx.degen_mode,
                    front_flux_damping=ctx.front_flux_damping,
                    open_bc_relaxation=ctx.open_bc_relaxation,
                    bc_relax=ctx.bc_relax,
                    active_set_hysteresis=ctx.active_set_hysteresis,
                    gpkg_path=ctx.results_gpkg_path or ctx.model_gpkg_path or "",
                    mesh_name=mesh_data.get("mesh_name", "") or "",
                    mesh_crs_wkt=mesh_data.get("crs_wkt", "") or "",
                )

            cuda_graphs_enabled = bool(ctx.cuda_graphs_enabled)
            os.environ["BACKWATER_ENABLE_CUDA_GRAPHS"] = "1" if cuda_graphs_enabled else "0"
            try:
                backend = _build_and_initialize_backend()
            except Exception as init_exc:
                err_l = str(init_exc).lower()
                is_illegal_mem = "illegal memory access" in err_l
                if cuda_graphs_enabled and is_illegal_mem:
                    log(
                        "CUDA solver init failed with illegal memory access while graph replay was enabled; "
                        "retrying once with CUDA graph replay disabled."
                    )
                    cuda_graphs_enabled = False
                    os.environ["BACKWATER_ENABLE_CUDA_GRAPHS"] = "0"
                    backend = _build_and_initialize_backend()
                    log("CUDA graph replay fallback at solver init succeeded.")
                else:
                    raise

            cp = getattr(backend, "_cell_perm", None)
            # Capture coupling geometry in pre-RCMK order before the
            # permutation signal mutates view._mesh_data.  The coupling
            # controller expects original-order arrays — structure cell
            # indices come from the config in original order.
            _coupling_cell_area = ctx.mesh_cell_areas()
            _coupling_cell_bed = ctx.mesh_cell_min_bed()
            _coupling_cell_centroids = ctx.mesh_cell_centroids()
            if cp is not None and cp.size > 0:
                if self.receivers(self.mesh_permutation_ready) > 0:
                    result_holder = _PermutationResult()
                    self.mesh_permutation_ready.emit(np.asarray(cp, dtype=np.int32), result_holder)
                    if not result_holder.event.wait(timeout=60.0):
                        raise RuntimeError("Timed out waiting for main-thread mesh permutation.")
                    if result_holder.error:
                        raise RuntimeError(f"Mesh permutation failed: {result_holder.error}")
                    sample_map = list(result_holder.sample_map or [])
                else:
                    sample_map = list(ctx.sample_map_data or [])
                apply_cell_permutation(mesh_data, cp)
                if ctx.internal_flow_forcing is not None:
                    ctx = replace(
                        ctx,
                        internal_flow_forcing=permute_internal_flow_forcing(
                            ctx.internal_flow_forcing, np.asarray(cp, dtype=np.int32)
                        ),
                    )
                if ctx.thiessen_forcing is not None:
                    ctx = replace(
                        ctx,
                        thiessen_forcing=permute_thiessen_forcing(
                            ctx.thiessen_forcing, np.asarray(cp, dtype=np.int32)
                        ),
                    )
            else:
                sample_map = list(ctx.sample_map_data or [])

            # ── GPU line sampling setup ──
            import logging as _lg
            _lg.warning("[LINE_DIAG] worker: sample_map=%d items, backend.has_line_sampling=%s",
                        len(sample_map) if sample_map else 0, backend.has_line_sampling if backend else False)
            line_names_by_id: Dict[int, str] = {}
            line_ids_ordered: List[int] = []
            if sample_map and backend.has_line_sampling:
                station_offsets_list = [0]
                cell_idx_parts: List[np.ndarray] = []
                weights_parts: List[np.ndarray] = []
                normal_x_parts: List[np.ndarray] = []
                normal_y_parts: List[np.ndarray] = []
                station_m_parts: List[np.ndarray] = []
                for sm in sample_map:
                    ci = np.asarray(sm.get("cell_idx", []), dtype=np.int32).ravel()
                    wt = np.asarray(sm.get("weights", []), dtype=np.float64).ravel()
                    nx = float(sm.get("normal_x", 0.0))
                    ny = float(sm.get("normal_y", 1.0))
                    st = np.asarray(sm.get("station_m", []), dtype=np.float64).ravel()
                    lid = int(sm.get("line_id", len(station_offsets_list) - 1))
                    lname = str(sm.get("line_name", f"line_{lid}"))
                    line_names_by_id[lid] = lname
                    line_ids_ordered.append(lid)
                    n = ci.size
                    station_offsets_list.append(station_offsets_list[-1] + n)
                    cell_idx_parts.append(ci)
                    weights_parts.append(wt)
                    normal_x_parts.append(np.full(n, nx, dtype=np.float64))
                    normal_y_parts.append(np.full(n, ny, dtype=np.float64))
                    station_m_parts.append(st)
                station_offsets = np.array(station_offsets_list, dtype=np.int32)
                cell_idx_arr = np.concatenate(cell_idx_parts).astype(np.int32) if cell_idx_parts else np.empty(0, dtype=np.int32)
                weights_arr = np.concatenate(weights_parts).astype(np.float64) if weights_parts else np.empty(0, dtype=np.float64)
                normal_x_arr = np.concatenate(normal_x_parts).astype(np.float64) if normal_x_parts else np.empty(0, dtype=np.float64)
                normal_y_arr = np.concatenate(normal_y_parts).astype(np.float64) if normal_y_parts else np.empty(0, dtype=np.float64)
                station_m_arr = np.concatenate(station_m_parts).astype(np.float64) if station_m_parts else np.empty(0, dtype=np.float64)
                try:
                    backend.configure_line_sampling(
                        station_offsets=station_offsets,
                        cell_idx=cell_idx_arr,
                        weights=weights_arr,
                        normal_x=normal_x_arr,
                        normal_y=normal_y_arr,
                        station_m=station_m_arr,
                        gravity=float(ctx.gravity),
                        h_min=float(ctx.h_min),
                    )
                    log(f"GPU line sampling configured: {len(sample_map)} lines, {int(station_offsets[-1])} stations.")
                except Exception as exc:
                    log(f"[ERROR] GPU line sampling setup failed: {exc}")
                    raise

            coupling_controller = None
            if ctx.pipe_network_cfg is not None or ctx.hydraulic_structures_cfg is not None:
                coupling_controller = build_coupling_controller(
                    pipe_network_cfg=ctx.pipe_network_cfg,
                    hydraulic_structures_cfg=ctx.hydraulic_structures_cfg,
                    cell_area=_coupling_cell_area,
                    cell_bed=_coupling_cell_bed,
                    length_scale_si_to_model=float(ctx.length_scale_si_to_model),
                    bridge_cuda_coupling=bool(ctx.bridge_cuda_coupling),
                    bridge_stacked_coupling_mode=str(ctx.bridge_stacked_coupling_mode),
                    culvert_face_flux_mode=str(ctx.culvert_face_flux_mode),
                    culvert_solver_mode=ctx.culvert_solver_mode,
                    drainage_gpu_method_mode=ctx.drainage_gpu_method_mode,
                    use_redistribution=bool(ctx.use_redistribution),
                    cell_centroids=_coupling_cell_centroids,
                    log_fn=log,
                )

            if coupling_controller is not None:
                _inv_perm = getattr(backend, "_inv_cell_perm", None)
                if _inv_perm is not None and _inv_perm.size > 0:
                    try:
                        coupling_controller._inv_cell_perm = np.asarray(_inv_perm, dtype=np.int32).copy()
                    except Exception as sync_exc:
                        log(f"[WARNING] Failed to sync inv_cell_perm to coupling controller: {sync_exc}")

            area_model_pre = np.asarray(ctx.cell_areas, dtype=np.float64).ravel()
            if area_model_pre.size == 0:
                area_model_pre = np.asarray(ctx.mesh_cell_areas(), dtype=np.float64).ravel()
            n_area = int(area_model_pre.size)
            if cp is not None and cp.size > 0:
                area_model = area_model_pre[cp]
                h0_model = np.asarray(h0, dtype=np.float64).ravel()[cp]
            else:
                area_model = area_model_pre
                h0_model = np.asarray(h0, dtype=np.float64).ravel()
            n_store = min(n_area, int(h0_model.size))
            storage_start_model = (
                float(np.sum(h0_model[:n_store] * area_model[:n_store])) if n_store > 0 else 0.0
            )

            edge_len_bc, side_idx_bc, side_names_bc = classify_boundary_edges(
                mesh_data["node_x"], mesh_data["node_y"], bc_n0, bc_n1
            )
            edge_group_labels: List[str] = []
            for ei in range(int(bc_n0.size)):
                if ei in edge_group_overrides:
                    edge_group_labels.append(str(edge_group_overrides[ei]))
                else:
                    edge_group_labels.append(str(side_names_bc[int(side_idx_bc[ei])]))

            if dynamic_bc and not backend.supports_dynamic_boundary_update():
                raise RuntimeError("Native module does not support dynamic boundary updates. Rebuild hydra_swe2d.")

            native_bc_forcing = False
            native_rain_cn_forcing = False

            if dynamic_bc and hasattr(backend, "set_boundary_hydrographs_native"):
                try:
                    native_bc_cfg = SWE2DNativeBoundaryHydrographConfigurator()
                    native_bc_res = native_bc_cfg.configure(
                        backend=backend,
                        bc_n0=bc_n0,
                        bc_n1=bc_n1,
                        bc_tp=bc_tp,
                        side_hydrographs=side_hydrographs,
                        edge_hydrographs=edge_hydrographs,
                        node_x=mesh_data["node_x"],
                        node_y=mesh_data["node_y"],
                        node_z=mesh_data["node_z"],
                        inflow_q_bc_type=int(_BC_INFLOW_Q),
                        progressive=bool(ctx.inflow_progressive),
                        ts_flow_code=int(_BC_TS_FLOW),
                        ts_stage_code=int(_BC_TS_STAGE),
                    )
                    if bool(native_bc_res.get("native_bc_forcing", False)):
                        native_bc_forcing = True
                        log(
                            f"Native BC hydrograph forcing configured for "
                            f"{int(native_bc_res.get('configured_edges', 0))} boundary edges."
                        )
                        if bool(native_bc_res.get("progressive_uploaded", False)):
                            log(
                                f"Progressive inflow data uploaded for "
                                f"{int(native_bc_res.get('n_prog_edges', 0))} edges."
                            )
                    elif bool(native_bc_res.get("skipped_progressive", False)):
                        log("Native BC hydrographs skipped: progressive inflow activation is enabled for flow hydrographs.")
                except Exception as exc:
                    log(f"Native BC hydrograph forcing unavailable: {exc}")

            if hasattr(backend, "set_rain_cn_forcing_native"):
                try:
                    run_setup_configurator = SWE2DRunSetupConfigurator()
                    if ctx.thiessen_forcing is not None:
                        native_rain_res = run_setup_configurator.configure_native_rain_cn_forcing(
                            backend=backend,
                            thiessen_forcing=ctx.thiessen_forcing,
                            mm_to_model_depth=float(ctx.rain_mm_to_model_depth),
                            rain_update_interval_s=ctx.rain_update_interval_s,
                        )
                    elif float(np.asarray(ctx.rain_rate_model, dtype=np.float64)) > 0.0:
                        native_rain_res = run_setup_configurator.configure_constant_rain_rate_native(
                            backend=backend,
                            rate_model_mps=float(np.asarray(ctx.rain_rate_model, dtype=np.float64)),
                            mm_to_model_depth=float(ctx.rain_mm_to_model_depth),
                        )
                    else:
                        native_rain_res = {"configured": False}
                    if bool(native_rain_res.get("configured", False)):
                        native_rain_cn_forcing = True
                        log(
                            "Native rainfall forcing configured for GPU timestep evaluation "
                            f"(groups={int(native_rain_res.get('groups', 0))})."
                        )
                except Exception as exc:
                    log(f"[WARNING] Native rain forcing unavailable: {exc}")

            native_source_injection_mode = hasattr(backend, "set_external_sources_native")
            if native_source_injection_mode:
                try:
                    native_src_res = SWE2DRunSetupConfigurator().configure_native_source_injection(backend=backend)
                    native_source_injection_mode = bool(native_src_res.get("native_source_injection_mode", False))
                    if bool(native_src_res.get("configured", False)):
                        log("Native external source injection enabled (device-resident coupling path).")
                except Exception as exc:
                    native_source_injection_mode = False
                    log(f"Native external source injection unavailable: {exc}")

            rain_stats_acc = {"rain_mm": 0.0, "excess_mm": 0.0, "samples": 0}
            perf_mode = str(os.environ.get("BACKWATER_SWE2D_PERF_MODE", "0")).strip().lower() in (
                "1", "true", "yes", "on",
            )

            runtime_source_manager = SWE2DRuntimeSourceManager(
                rain_rate_model=ctx.rain_rate_model,
                thiessen_forcing=ctx.thiessen_forcing,
                native_rain_cn_forcing=native_rain_cn_forcing,
                internal_flow_forcing=ctx.internal_flow_forcing,
                rain_stats_acc=rain_stats_acc,
                area_model=area_model,
                edge_len_bc=edge_len_bc,
                edge_group_labels=edge_group_labels,
                inflow_q_bc_type=int(_BC_INFLOW_Q),
                rain_rate_si_to_model_callback=lambda rr: float(np.asarray(rr)) * _u.rain_si_to_model(1.0),
                internal_flow_source_cms_at_time_callback=ctx.internal_flow_source_cms_at_time,
                flow_si_to_model_callback=lambda q: np.asarray(q, dtype=np.float64),
                enable_source_volume_accounting=(not perf_mode),
                enable_boundary_flux_accounting=(not perf_mode),
                record_source_step_rows=(not perf_mode),
                record_boundary_flux_step_rows=(not perf_mode),
            )

            results_data = SWE2DResultsData()
            # Store on self so headless executor can access via adapter after _execute() returns
            self._results_data = results_data
            results_data.clear_live_snapshots()
            if coupling_controller is not None:
                coupling_keys, coupling_object_names = build_coupling_keys(coupling_controller)
                if coupling_keys:
                    results_data.init_coupling_storage(coupling_keys, coupling_object_names)
                # Initialize per-pipe-cell storage (drainage_cell rows)
                pipe_cell_keys = build_pipe_cell_keys(coupling_controller)
                log(f"[PipeCell] initialized {len(pipe_cell_keys)} pipe-cell keys")
                if pipe_cell_keys:
                    results_data.init_pipe_cell_storage(pipe_cell_keys)
                # t=0 coupling snapshot (node depths, link flows, cell indices)
                t0_rows = list(_sample_coupling_object_metrics(coupling_controller, 0.0, None, None))
                log(f"[PipeCell] t=0 snapshot produced {len(t0_rows)} coupling rows")
                for row in t0_rows:
                    results_data.append_coupling_snapshot(row)
                log(f"[PipeCell] _live_pipe_cell now has {len(results_data._live_pipe_cell)} keys")

            runtime_step_executor = SWE2DRuntimeStepExecutor()
            runtime_reporter = SWE2DRuntimeReporter()
            self._runtime_reporter = runtime_reporter

            wb = _WorkbenchShim(self, ctx, results_data, mesh_data)

            def _on_snapshot_readback() -> None:
                import logging as _lg
                timesteps = results_data.get_live_snapshot_timesteps()
                if not timesteps:
                    _lg.warning("[LINE_DIAG] _on_snapshot_readback: no timesteps, skipping emit")
                    return
                line_ts = dict(results_data._live_line_ts)
                line_profiles = dict(results_data._live_line_profile)
                _lg.warning("[LINE_DIAG] _on_snapshot_readback: emitting %d timesteps, %d line_ts keys, %d line_profiles keys",
                            len(timesteps), len(line_ts), len(line_profiles))
                coupling_data = {
                    key: {"object_name": d["object_name"],
                          "t_s": list(d["t_s"]),
                          "values": list(d["values"])}
                    for key, d in results_data._live_coupling.items()
                }
                pipe_cell_data = None
                if results_data._live_pipe_cell:
                    pipe_cell_data = {
                        key: {
                            "cell_invert": float(d.get("cell_invert", 0.0)),
                            "cell_width": float(d.get("cell_width", 1.0)),
                            "cell_height": float(d.get("cell_height", d.get("cell_width", 1.0))),
                            "cell_shape_type": int(d.get("cell_shape_type", 0)),
                            "times": list(d.get("times", [])),
                            "values": list(d.get("values", [])),
                        }
                        for key, d in results_data._live_pipe_cell.items()
                    }
                    _lg.warning("[PipeCell] emitting %d pipe-cell keys in snapshot", len(pipe_cell_data))
                self.snapshot_ready.emit(
                    SnapshotData(
                        timesteps=timesteps,
                        line_ts=line_ts,
                        line_profiles=line_profiles,
                        coupling_data=coupling_data,
                        pipe_cell_data=pipe_cell_data,
                    )
                )

            runtime_reporter.set_post_readback_callback(_on_snapshot_readback)

            log("The numbers go UP! They go UP UP UP!!!")

            last_diag = None
            t_accum = 0.0
            i = 0
            last_valid_cmax = float("nan")
            last_valid_wse_res = float("nan")
            _PROCESS_EVENTS_INTERVAL_S = 0.10
            _last_process_events_wall = time.perf_counter()
            timing_totals_ms = {
                "wall": 0.0,
                "step": 0.0,
                "coupling": 0.0,
                "source": 0.0,
                "state": 0.0,
                "bc": 0.0,
                "ui": 0.0,
            }
            timing_samples = 0
            run_span_s = max(float(run_duration_s), 1.0e-9)
            _next_snap_t = min(output_interval_s, run_span_s)

            loop_result = _execute_run_timestep_loop_runtime_logic(
                wb=wb,
                backend=backend,
                runtime_step_executor=runtime_step_executor,
                runtime_reporter=runtime_reporter,
                run_duration_s=run_duration_s,
                t_accum=t_accum,
                i=i,
                last_diag=last_diag,
                last_valid_cmax=last_valid_cmax,
                last_valid_wse_res=last_valid_wse_res,
                dt_cfg=dt_cfg,
                dt_request=dt_request,
                coupling_controller=coupling_controller,
                dynamic_bc=dynamic_bc,
                native_bc_forcing=native_bc_forcing,
                bc_n0=bc_n0,
                bc_n1=bc_n1,
                bc_tp=bc_tp,
                bc_vl=bc_vl,
                side_hydrographs=side_hydrographs,
                edge_hydrographs=edge_hydrographs,
                rain_source_for_window_callback=runtime_source_manager.rain_source_for_window,
                cell_source_model_at_time_callback=runtime_source_manager.cell_source_model_at_time,
                accumulate_source_volume_model_callback=runtime_source_manager.accumulate_source_volume_model,
                native_source_injection_mode=native_source_injection_mode,
                accumulate_boundary_flux_volume_model_callback=runtime_source_manager.accumulate_boundary_flux_volume_model,
                sample_map=sample_map,
                timing_totals_ms=timing_totals_ms,
                timing_samples=timing_samples,
                next_snap_t=_next_snap_t,
                output_interval_s=output_interval_s,
                process_events_interval_s=_PROCESS_EVENTS_INTERVAL_S,
                last_process_events_wall=_last_process_events_wall,
                process_events_callback=lambda: None,
                h_min=ctx.h_min,
                uniform_enabled=ctx.uniform_inflow_enabled,
                progress_callback=self.progress_percent.emit,
                perf_mode=perf_mode,
                line_names_by_id=line_names_by_id,
                line_ids_ordered=line_ids_ordered,
            )

            snap_data = backend.read_snapshots()
            snapshot_timesteps: List[Tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []
            if snap_data and "t_s" in snap_data:
                try:
                    ts_arr = np.asarray(snap_data["t_s"], dtype=np.float64)
                    h_arr = np.asarray(snap_data["h"], dtype=np.float64)
                    hu_arr = np.asarray(snap_data["hu"], dtype=np.float64)
                    hv_arr = np.asarray(snap_data["hv"], dtype=np.float64)
                    n_snaps = int(ts_arr.shape[0])
                    for si in range(n_snaps):
                        snapshot_timesteps.append((
                            float(ts_arr[si]),
                            np.ascontiguousarray(h_arr[si, :]),
                            np.ascontiguousarray(hu_arr[si, :]),
                            np.ascontiguousarray(hv_arr[si, :]),
                        ))
                except Exception as exc:
                    log(f"[SnapReadback] Device snapshot readback failed: {exc}")
            # Restore pre-threading pattern: merge final device snapshots
            # with any earlier readbacks (read_snapshots may be destructive),
            # then compute line TS + profile for the full snapshot set so
            # the viewer and GPKG persistence both have line data.
            if snapshot_timesteps:
                try:
                    # Device ring buffer is non-destructive — replace live data
                    # with the full current device state (no merge needed).
                    results_data.set_live_snapshot_timesteps(
                        snapshot_timesteps, t_sec=float(t_accum))
                    snapshot_timesteps = list(results_data.get_live_snapshot_timesteps())
                    if sample_map and backend.has_line_sampling:
                        lm = backend.read_line_metrics()
                        if lm:
                            results_data.populate_live_line_metrics_from_gpu(
                                lm, line_names_by_id=line_names_by_id,
                                line_ids_ordered=line_ids_ordered,
                            )
                except Exception as exc:
                    log(f"[SnapReadback] Final line metrics readback failed: {exc}")

            # Push the final full state to the viewer so it has every
            # accumulated timestep even if no snapshot button was pressed.
            _on_snapshot_readback()

            h, hu, hv = backend.get_state()
            if hasattr(backend, "_cell_perm") and backend._cell_perm is not None and backend._cell_perm.size > 0:
                cp = backend._cell_perm
                h = h[cp]
                hu = hu[cp]
                hv = hv[cp]

            if native_source_injection_mode:
                try:
                    backend.set_external_sources_native(None)
                except Exception:
                    logger.warning("Unexpected error resetting native sources", exc_info=True)

            max_results = backend.get_max_tracking() if ctx.save_mesh_results else None

            coupling_snapshots: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
            if ctx.save_coupling_results:
                for key, d in results_data._live_coupling.items():
                    t_s = d.get("t_s")
                    values = d.get("values")
                    if not t_s or not values:
                        continue
                    n = min(len(t_s), len(values))
                    if n <= 0:
                        continue
                    coupling_snapshots[key] = {
                        "object_name": d.get("object_name", key[1]),
                        "times": np.asarray(t_s[:n], dtype=np.float64),
                        "values": np.asarray(values[:n], dtype=np.float64),
                    }

            precomputed_line_results = results_data.build_precomputed_line_results()

            cancelled = bool(ctx.cancel_event.is_set())
            return ComputeResult(
                ok=not cancelled,
                h=h,
                hu=hu,
                hv=hv,
                final_sim_time_s=float(loop_result.get("t_accum", t_accum)),
                n_area=n_area,
                area_model=area_model,
                storage_start_model=storage_start_model,
                source_budget_model=runtime_source_manager.source_budget_model,
                source_step_rows_model=runtime_source_manager.source_step_rows_model,
                run_duration_s=run_duration_s,
                boundary_flux_budget_model=runtime_source_manager.boundary_flux_budget_model,
                boundary_flux_step_rows_model=runtime_source_manager.boundary_flux_step_rows_model,
                run_id=run_id,
                mesh_name=mesh_data.get("mesh_name", "") or "",
                output_interval_s=output_interval_s,
                run_perf_start=run_perf_start,
                run_wallclock_start=run_wallclock_start,
                run_log_start_idx=run_log_start_idx,
                thiessen_forcing=ctx.thiessen_forcing,
                rain_stats_acc=rain_stats_acc,
                max_tracking=max_results,
                snapshot_timesteps=snapshot_timesteps,
                coupling_snapshots=coupling_snapshots,
                precomputed_line_results=precomputed_line_results,
                cancelled=cancelled,
                save_line_results=bool(ctx.save_line_results),
                save_coupling_results=bool(ctx.save_coupling_results),
                save_mesh_results=bool(ctx.save_mesh_results),
                save_run_log=bool(ctx.save_run_log),
                save_max_only=bool(ctx.save_max_only),
                h_min=float(ctx.h_min),
            )
        finally:
            if backend is not None:
                try:
                    backend.destroy()
                except Exception:
                    logger.warning("Backend destroy failed", exc_info=True)
