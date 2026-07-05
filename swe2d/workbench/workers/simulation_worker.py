from __future__ import annotations

import datetime
import logging
import os
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from qgis.PyQt.QtCore import QThread, pyqtSignal

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
    execute_run_timestep_loop as _execute_run_timestep_loop_runtime_logic,
)
from swe2d.workbench.workers.run_context import RunContext


logger = logging.getLogger(__name__)


class SnapshotData:
    """Data emitted when a device snapshot is ready for UI sync."""

    def __init__(
        self,
        t_s: float,
        h: np.ndarray,
        hu: np.ndarray,
        hv: np.ndarray,
        line_ts: Any = None,
        line_profiles: Any = None,
        coupling_rows: List[Dict[str, Any]] = None,
    ):
        self.t_s = t_s
        self.h = h
        self.hu = hu
        self.hv = hv
        self.line_ts = line_ts
        self.line_profiles = line_profiles
        self.coupling_rows = coupling_rows or []


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
        line_output_interval_s: float,
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
        self.line_output_interval_s = line_output_interval_s
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


class _WorkerResultsData:
    """Lightweight, Qt-free snapshot accumulator for the worker thread."""

    def __init__(self) -> None:
        self._live_snapshot_timesteps: List[Tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []
        self._live_coupling: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        self._coupling_snap_idx: int = 0

    def clear_live_snapshots(self) -> None:
        self._live_snapshot_timesteps = []
        self._live_coupling = {}
        self._coupling_snap_idx = 0

    def preallocate_output_schedule(
        self,
        n_line_snaps: int,
        coupling_keys: List[Tuple[str, str, str]],
        coupling_object_names: Dict[Tuple[str, str, str], str],
    ) -> None:
        self._coupling_snap_idx = 0
        self._live_coupling = {}
        for key in coupling_keys:
            self._live_coupling[key] = {
                "object_name": coupling_object_names.get(key, key[1]),
                "t_s": np.zeros(n_line_snaps, dtype=np.float64),
                "values": np.zeros(n_line_snaps, dtype=np.float64),
            }

    def append_coupling_snapshot(self, row: Dict[str, Any], snap_idx: int) -> None:
        key = (
            str(row.get("component", "")),
            str(row.get("object_id", "")),
            str(row.get("metric", "")),
        )
        if not all(key):
            return
        d = self._live_coupling.get(key)
        if d is None:
            return
        idx = int(snap_idx)
        if 0 <= idx < d["t_s"].size:
            d["t_s"][idx] = float(row.get("t_s", 0.0))
            d["values"][idx] = float(row.get("value", 0.0))

    def get_live_snapshot_timesteps(self) -> List[Tuple[float, np.ndarray, np.ndarray, np.ndarray]]:
        return self._live_snapshot_timesteps

    def set_live_snapshot_timesteps(
        self,
        timesteps: List[Tuple[float, np.ndarray, np.ndarray, np.ndarray]],
        t_sec: float = 0.0,
    ) -> None:
        self._live_snapshot_timesteps = list(timesteps)

    def populate_live_line_metrics(self, sample_map: Any, sample_callback: Any, cell_solver_z: Any) -> None:
        pass

    def build_precomputed_line_results(self) -> Dict[Any, Any]:
        return {}


class _WorkbenchShim:
    """Adapts the worker/context to the ``wb`` object expected by the runtime loop."""

    def __init__(
        self,
        worker: "SimulationWorker",
        ctx: RunContext,
        results_data: _WorkerResultsData,
        mesh_data: Dict[str, Any],
    ) -> None:
        self._worker = worker
        self._ctx = ctx
        self._results_data = results_data
        self._mesh_data = mesh_data
        self._apply_timeseries_bc_values = ctx.apply_timeseries_bc_values
        self._distribute_total_flow_to_unit_q = ctx.distribute_total_flow_to_unit_q
        self._apply_external_sources = ctx.apply_external_sources
        self._length_unit_name = ctx.length_unit_name
        self._sample_line_metrics = ctx.sample_line_metrics
        self._log = worker.log_message.emit

    @property
    def _cancel_requested(self) -> bool:
        return self._ctx.cancel_event.is_set()


class SimulationWorker(QThread):
    """Background worker that owns the SWE2D backend and runs the timestep loop."""

    log_message = pyqtSignal(str)
    progress_percent = pyqtSignal(int)
    snapshot_ready = pyqtSignal(object)
    compute_finished = pyqtSignal(object)
    compute_failed = pyqtSignal(str)

    def __init__(self, context: RunContext, parent=None):
        super().__init__(parent)
        self._context = context
        self._snapshot_requested = __import__('threading').Event()

    def request_snapshot(self):
        """Request a snapshot readback on the next reporter step."""
        self._snapshot_requested.set()

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
        line_output_interval_s = ctx.line_output_interval_s
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

        backend = None
        try:
            backend_initializer = SWE2DBackendInitializer(
                apply_timeseries_bc_values_callback=ctx.apply_timeseries_bc_values,
                distribute_total_flow_to_unit_q_callback=ctx.distribute_total_flow_to_unit_q,
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
                    extreme_rain_mode=ctx.extreme_rain_mode,
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
            if cp is not None and cp.size > 0:
                apply_cell_permutation(mesh_data, cp)

            coupling_controller = None
            if ctx.pipe_network_cfg is not None or ctx.hydraulic_structures_cfg is not None:
                coupling_controller = build_coupling_controller(
                    pipe_network_cfg=ctx.pipe_network_cfg,
                    hydraulic_structures_cfg=ctx.hydraulic_structures_cfg,
                    cell_area=ctx.mesh_cell_areas(),
                    cell_bed=ctx.mesh_cell_min_bed(),
                    length_scale_si_to_model=float(ctx.length_scale_si_to_model),
                    bridge_cuda_coupling=bool(ctx.bridge_cuda_coupling),
                    bridge_stacked_coupling_mode=str(ctx.bridge_stacked_coupling_mode),
                    culvert_face_flux_mode=str(ctx.culvert_face_flux_mode),
                    culvert_solver_mode=ctx.culvert_solver_mode,
                    drainage_gpu_method_mode=ctx.drainage_gpu_method_mode,
                    use_redistribution=bool(ctx.use_redistribution),
                    cell_centroids=ctx.mesh_cell_centroids(),
                    log_fn=log,
                )

            if coupling_controller is not None:
                _inv_perm = getattr(backend, "_inv_cell_perm", None)
                if _inv_perm is not None and _inv_perm.size > 0:
                    try:
                        coupling_controller._inv_cell_perm = np.asarray(_inv_perm, dtype=np.int32).copy()
                    except Exception as sync_exc:
                        log(f"[WARNING] Failed to sync inv_cell_perm to coupling controller: {sync_exc}")

            sample_map = ctx.build_line_sampling_map()
            cell_solver_z = ctx.mesh_cell_solver_bed() if sample_map else None

            area_model = np.asarray(ctx.mesh_cell_areas(), dtype=np.float64).ravel()
            if area_model.size == 0 and ctx.cell_areas.size > 0:
                area_model = np.asarray(ctx.cell_areas, dtype=np.float64).ravel()
            n_area = int(area_model.size)

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
                rain_rate_si_to_model_callback=lambda _rr=None: ctx.rain_rate_si_to_model,
                internal_flow_source_cms_at_time_callback=ctx.internal_flow_source_cms_at_time,
                flow_si_to_model_callback=lambda _q=None: ctx.flow_si_to_model,
                enable_source_volume_accounting=(not perf_mode),
                enable_boundary_flux_accounting=(not perf_mode),
                record_source_step_rows=(not perf_mode),
                record_boundary_flux_step_rows=(not perf_mode),
            )

            results_data = _WorkerResultsData()
            results_data.clear_live_snapshots()
            if coupling_controller is not None:
                import math as _math
                n_snaps = max(1, _math.ceil(run_duration_s / max(float(line_output_interval_s), 1.0)))
                coupling_keys, coupling_object_names = build_coupling_keys(coupling_controller)
                if coupling_keys:
                    results_data.preallocate_output_schedule(n_snaps, coupling_keys, coupling_object_names)

            runtime_step_executor = SWE2DRuntimeStepExecutor()
            runtime_reporter = SWE2DRuntimeReporter()

            wb = _WorkbenchShim(self, ctx, results_data, mesh_data)

            def _on_snapshot_readback() -> None:
                timesteps = results_data.get_live_snapshot_timesteps()
                if not timesteps:
                    return
                t_s, h, hu, hv = timesteps[-1]
                self.snapshot_ready.emit(
                    SnapshotData(
                        t_s=float(t_s),
                        h=np.asarray(h, dtype=np.float64),
                        hu=np.asarray(hu, dtype=np.float64),
                        hv=np.asarray(hv, dtype=np.float64),
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
            _next_line_snap_t = min(line_output_interval_s, run_span_s)
            _next_coupling_snap_t = min(line_output_interval_s, run_span_s)

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
                cell_solver_z=cell_solver_z,
                timing_totals_ms=timing_totals_ms,
                timing_samples=timing_samples,
                next_snap_t=_next_snap_t,
                next_line_snap_t=_next_line_snap_t,
                next_coupling_snap_t=_next_coupling_snap_t,
                output_interval_s=output_interval_s,
                line_output_interval_s=line_output_interval_s,
                process_events_interval_s=_PROCESS_EVENTS_INTERVAL_S,
                last_process_events_wall=_last_process_events_wall,
                process_events_callback=lambda: None,
                h_min=ctx.h_min,
                uniform_enabled=ctx.uniform_inflow_enabled,
                progress_callback=self.progress_percent.emit,
                perf_mode=perf_mode,
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
                snap_idx = results_data._coupling_snap_idx
                for key, d in results_data._live_coupling.items():
                    t_s = d.get("t_s")
                    values = d.get("values")
                    if t_s is None or values is None:
                        continue
                    n = min(int(t_s.size), int(values.size), snap_idx)
                    if n <= 0:
                        continue
                    coupling_snapshots[key] = {
                        "object_name": d.get("object_name", key[1]),
                        "times": np.asarray(t_s[:n], dtype=np.float64),
                        "values": np.asarray(values[:n], dtype=np.float64),
                    }

            precomputed_line_results = results_data.build_precomputed_line_results()

            return ComputeResult(
                ok=True,
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
                line_output_interval_s=line_output_interval_s,
                run_perf_start=run_perf_start,
                run_wallclock_start=run_wallclock_start,
                run_log_start_idx=run_log_start_idx,
                thiessen_forcing=ctx.thiessen_forcing,
                rain_stats_acc=rain_stats_acc,
                max_tracking=max_results,
                snapshot_timesteps=snapshot_timesteps,
                coupling_snapshots=coupling_snapshots,
                precomputed_line_results=precomputed_line_results,
            )
        finally:
            if backend is not None:
                try:
                    backend.destroy()
                except Exception:
                    logger.warning("Backend destroy failed", exc_info=True)
