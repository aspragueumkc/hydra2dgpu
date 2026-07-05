"""WorkbenchController — mediates between Service Layer and View.

The Controller is the brain of the workbench:
- Receives requests from the View (user actions)
- Calls Service methods
- Pushes results back to the View (updates state, calls view methods)

The Controller does NOT contain business logic itself — it only
orchestrates. Business logic lives in services.
"""
from __future__ import annotations

import datetime
import logging
import os
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import threading

from swe2d.workbench.services.mesh_service import apply_cell_permutation
from swe2d.workbench.workers.simulation_worker import SimulationWorker, SnapshotData, ComputeResult
from swe2d.workbench.workers.persistence_worker import PersistenceWorker
from swe2d.workbench.workers.run_context import RunContext

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from swe2d.workbench.studio_dialog import SWE2DWorkbenchStudioDialog


# Module-level alias used by the snapshot orchestration below. This is kept
# because ``on_snapshot`` still calls into the HEC-RAS HDF5 export.


class RunController:
    """MVP domain controller for the 2D simulation run pipeline."""
    """Mediates between Service Layer and View (SWE2DWorkbenchStudioDialog).

    Holds a reference to the View (dialog). Methods are called either
    directly by the dialog or in response to View signals.
    """

    def __init__(self, view: "SWE2DWorkbenchStudioDialog"):
        self._view = view
        self._simulation_worker = None
        self._persistence_worker = None

    def on_run(self, request: Optional[Any] = None) -> Any:
        """Start a 2D run on a background worker thread.

        Builds a ``RunContext`` from the current view state, creates a
        ``SimulationWorker``, connects its signals to UI slots, and
        starts the thread.  Returns ``None`` when the run is aborted.
        """
        view = self._view
        if view._mesh_data is None:
            view._log("Run aborted: mesh not available after preflight.")
            return None
        if self._simulation_worker is not None and self._simulation_worker.isRunning():
            view._log("Run aborted: another run is already active.")
            return None

        context = self._build_run_context(request=request)
        if context is None:
            return None

        view._cancel_requested = False
        view.set_run_button_enabled(False)
        view.set_cancel_button_enabled(True)
        view.set_run_progress(0)

        # Ensure _results_data exists so snapshot_ready signals have a place to land.
        try:
            view._show_results_panel()
        except Exception:
            pass

        worker = SimulationWorker(context, parent=view)
        worker.log_message.connect(view._log)
        worker.progress_percent.connect(view.set_run_progress)
        worker.snapshot_ready.connect(self._on_worker_snapshot_ready)
        worker.compute_finished.connect(self._on_worker_compute_finished)
        worker.compute_failed.connect(self._on_worker_compute_failed)
        worker.finished.connect(self._on_simulation_worker_finished)
        self._simulation_worker = worker
        worker.start()
        return None

    def _build_run_context(self, request: Optional[Any] = None) -> Optional[RunContext]:
        """Capture all widget values and arrays into a RunContext."""
        view = self._view
        mesh_data = view._mesh_data
        log_fn = view._log

        if mesh_data is None:
            log_fn("Run aborted: mesh not available after preflight.")
            return None

        if not mesh_data.get("mesh_name"):
            _gpkg = view._current_line_results_storage_path() or view._model_gpkg_path or ""
            _stem = os.path.splitext(os.path.basename(_gpkg))[0] if _gpkg else "mesh"
            mesh_data["mesh_name"] = f"{_stem}_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')}"

        run_data_builder = view._run_data_builder
        run_options_builder = view._run_options_builder
        parse_time_hours_fn = view._parse_time_hours
        model_gpkg_path = view._model_gpkg_path
        length_unit_name = view._length_unit_name

        wp = view.collect_run_widget_params()

        last_run_request = view._last_run_request
        if request is None:
            request = last_run_request

        run_input = run_data_builder.build()

        run_options = run_options_builder.build(
            dt=wp["dt_spin"],
            adaptive_cfl_dt=wp["adaptive_cfl_dt_chk"],
            initial_dt=wp["initial_dt_spin"],
            reconstruction_mode=wp["reconstruction_combo"],
            reconstruction_name=wp["reconstruction_combo_text"],
            temporal_order_value=wp["temporal_order_combo"],
            temporal_scheme_name=wp["temporal_order_combo_text"],
            drainage_gpu_method=wp["drainage_gpu_method"],
            culvert_solver_mode=wp["culvert_solver_mode"],
            cuda_graphs_enabled=wp["enable_cuda_graphs_chk"],
            swe2d_perf_mode=wp["swe2d_perf_mode_chk"],
            rain_rate_mmhr=wp["rain_rate_spin"],
            bridge_coupling_mode=wp["bridge_coupling_mode"],
            culvert_face_flux=wp["culvert_face_flux_chk"],
        )

        run_duration_s = run_options.run_duration_s
        if request is not None:
            request_run_duration_text = getattr(request, "run_duration_text", None)
            if request_run_duration_text is not None and str(request_run_duration_text).strip():
                try:
                    run_duration_s = max(0.0, parse_time_hours_fn(str(request_run_duration_text).strip()) * 3600.0)
                except Exception:
                    log_fn("[WARNING] Unexpected error silently caught")

        def _parse_interval_text(text, default_widget_text):
            if text is not None and str(text).strip():
                return parse_time_hours_fn(str(text).strip())
            return parse_time_hours_fn(str(default_widget_text or ""))

        request_output_interval_text = getattr(request, "output_interval_text", None) if request is not None else None
        request_line_output_interval_text = getattr(request, "line_output_interval_text", None) if request is not None else None

        _oi_hr = _parse_interval_text(request_output_interval_text, wp["output_interval_edit"])
        output_interval_s = max(1.0, _oi_hr * 3600.0)
        _line_oi_hr = _parse_interval_text(request_line_output_interval_text, wp["line_output_interval_edit"])
        line_output_interval_s = max(1.0, _line_oi_hr * 3600.0)

        from swe2d.runtime.coupling import pack_coupling_soa
        mesh_cell_areas_fn = view._mesh_cell_areas
        pipe_network_cfg = run_options.pipe_network_cfg
        hydraulic_structures_cfg = run_options.hydraulic_structures_cfg

        model_options = run_options.model_options
        if model_options is not None:
            if pipe_network_cfg is not None:
                model_options.pipe_network = pipe_network_cfg
            if hydraulic_structures_cfg is not None:
                model_options.hydraulic_structures = hydraulic_structures_cfg

        bridge_stacked_plans: List[Any] = []
        try:
            from swe2d.runtime.bridge_stacked_runtime import build_bridge_stacked_plans_for_runtime
            bridge_stacked_plans = build_bridge_stacked_plans_for_runtime(
                mesh_data, hydraulic_structures_cfg, log_fn=log_fn,
            )
        except Exception as exc:
            log_fn(f"Bridge stacked-plan mapping warning: {exc}")

        coupling_soa = None
        if pack_coupling_soa is not None:
            coupling_soa = pack_coupling_soa(
                n_cells=int(mesh_cell_areas_fn().shape[0]),
                pipe_network=pipe_network_cfg,
                hydraulic_structures=hydraulic_structures_cfg,
            )

        results_gpkg_path = str(view._current_line_results_storage_path() or "")
        run_id = datetime.datetime.now().astimezone().strftime("swe2d_%Y%m%dT%H%M%S%z")
        run_wallclock_start = datetime.datetime.now().replace(microsecond=0).isoformat(sep=" ")
        run_log_start_idx = len(view._runtime_log_lines)

        cancel_event = threading.Event()

        return RunContext(
            run_id=run_id,
            run_wallclock_start=run_wallclock_start,
            run_log_start_idx=run_log_start_idx,
            results_gpkg_path=results_gpkg_path,
            model_gpkg_path=str(model_gpkg_path or ""),
            mesh_name=str(mesh_data.get("mesh_name", "") or ""),
            mesh_crs_wkt=str(mesh_data.get("crs_wkt", "") or ""),
            run_duration_s=run_duration_s,
            output_interval_s=output_interval_s,
            line_output_interval_s=line_output_interval_s,
            dt_cfg=run_options.dt_cfg,
            dt_request=run_options.dt_request,
            dt_fixed=run_options.dt_fixed,
            initial_dt=getattr(run_options, "initial_dt", 0.0),
            adaptive_cfl_dt=run_options.adaptive_cfl_dt,
            reconstruction_mode=run_options.reconstruction_mode,
            reconstruction_name=run_options.reconstruction_name,
            temporal_scheme=run_options.temporal_scheme,
            temporal_scheme_name=run_options.temporal_scheme_name,
            solver_backend_mode=str(getattr(run_options, "solver_backend_mode", "gpu")).strip().lower(),
            coupling_loop_mode=run_options.coupling_loop_mode,
            drainage_solver_backend_mode=run_options.drainage_solver_backend_mode,
            drainage_gpu_method_mode=run_options.drainage_gpu_method_mode,
            culvert_solver_mode=getattr(run_options, "culvert_solver_mode", 0),
            cuda_graphs_enabled=run_options.cuda_graphs_enabled,
            bridge_cuda_coupling=bool(getattr(run_options, "bridge_cuda_coupling", False)),
            bridge_stacked_coupling_mode=str(getattr(run_options, "bridge_stacked_coupling_mode", "phase3_spatial")),
            culvert_face_flux_mode=str(getattr(run_options, "culvert_face_flux_mode", "off")),
            gravity=wp["gravity"],
            k_mann=wp["k_mann"],
            n_mann=wp["n_mann_spin"],
            cfl=wp["cfl_spin"],
            h_min=wp["h_min_spin"],
            max_inv_area=wp["max_inv_area_spin"],
            cfl_lambda_cap=wp["cfl_lambda_cap_spin"],
            momentum_cap_min_speed=wp["momentum_cap_min_speed_spin"],
            momentum_cap_celerity_mult=wp["momentum_cap_celerity_mult_spin"],
            depth_cap=wp["depth_cap_spin"],
            max_rel_depth_increase=wp["max_rel_depth_increase_spin"],
            shallow_damping_depth=wp["shallow_damping_depth_spin"],
            extreme_rain_mode=wp["extreme_rain_mode_chk"],
            source_cfl_beta=wp["source_cfl_beta_spin"],
            source_max_substeps=wp["source_max_substeps_spin"],
            source_rate_cap=wp["max_source_rate_spin"],
            source_depth_step_cap=wp["max_source_depth_step_spin"],
            source_true_subcycling=wp["source_true_subcycling_chk"],
            source_imex_split=wp["source_imex_split_chk"],
            gpu_diag_sync_interval_steps=wp["gpu_diag_sync_interval_spin"],
            tiny_mode=wp["tiny_mode_combo"],
            tiny_wet_cell_threshold=wp["tiny_wet_cell_threshold_spin"],
            degen_mode=wp["degen_mode"],
            front_flux_damping=wp["front_flux_damping_spin"],
            active_set_hysteresis=wp["active_set_hysteresis_chk"],
            use_redistribution=wp["use_redistribution_chk"],
            inflow_progressive=wp["inflow_progressive_chk"],
            uniform_inflow_enabled=view.model_tab.is_uniform_inflow(),
            rain_update_interval_s=view.model_tab.get_rain_update_interval_s(),
            node_x=run_input.node_x,
            node_y=run_input.node_y,
            node_z=run_input.node_z,
            cell_nodes=run_input.cell_nodes,
            face_offsets=run_input.face_offsets,
            face_nodes=run_input.face_nodes,
            bc_n0=run_input.bc_n0,
            bc_n1=run_input.bc_n1,
            bc_tp=run_input.bc_tp,
            bc_vl=run_input.bc_vl,
            side_hydrographs=run_input.side_hydrographs,
            edge_hydrographs=run_input.edge_hydrographs,
            edge_group_overrides=run_input.edge_group_overrides,
            h0=run_input.h0,
            hu0=run_input.hu0,
            hv0=run_input.hv0,
            n_mann_cell=run_input.n_mann_cell,
            cell_areas=np.asarray(mesh_cell_areas_fn(), dtype=np.float64).ravel(),
            cell_solver_bed=view._mesh_cell_solver_bed(),
            cell_centroids=view._mesh_cell_centroids(),
            rain_rate_model=run_options.rain_rate_model,
            internal_flow_forcing=run_options.internal_flow_forcing,
            cell_source_model=run_options.cell_source_model,
            thiessen_forcing=run_options.thiessen_forcing,
            pipe_network_cfg=pipe_network_cfg,
            hydraulic_structures_cfg=hydraulic_structures_cfg,
            bridge_stacked_plans=bridge_stacked_plans,
            coupling_soa=coupling_soa,
            save_mesh_results=bool(wp["save_mesh_results_to_gpkg_chk"]),
            save_line_results=bool(wp["save_line_results_to_gpkg_chk"]),
            save_coupling_results=bool(wp["save_coupling_results_to_gpkg_chk"]),
            save_run_log=bool(wp["save_run_log_to_gpkg_chk"]),
            length_unit_name=length_unit_name,
            length_scale_si_to_model=float(view._length_scale_si_to_model()),
            rain_mm_to_model_depth=float(view._rain_mm_to_model_depth()),
            apply_timeseries_bc_values=view._apply_timeseries_bc_values,
            distribute_total_flow_to_unit_q=view._distribute_total_flow_to_unit_q,
            apply_external_sources=view._apply_external_sources,
            sample_line_metrics=view._sample_line_metrics,
            build_line_sampling_map=view._build_line_sampling_map,
            mesh_cell_areas=mesh_cell_areas_fn,
            mesh_cell_min_bed=view._mesh_cell_min_bed,
            mesh_cell_centroids=view._mesh_cell_centroids,
            mesh_cell_solver_bed=view._mesh_cell_solver_bed,
            internal_flow_source_cms_at_time=view._internal_flow_source_cms_at_time,
            cancel_event=cancel_event,
        )

    def _on_worker_snapshot_ready(self, data: SnapshotData):
        view = self._view
        rd = getattr(view, "_results_data", None)
        if rd is None:
            return
        try:
            existing = rd.get_live_snapshot_timesteps()
            rd.set_live_snapshot_timesteps(
                existing + [(
                    float(data.t_s),
                    np.asarray(data.h, dtype=np.float64),
                    np.asarray(data.hu, dtype=np.float64),
                    np.asarray(data.hv, dtype=np.float64),
                )],
                t_sec=float(data.t_s),
            )
        except Exception as exc:
            logger.warning("Snapshot readback: merge failed", exc_info=True)
            view._log(f"[SnapReadback] merge failed: {exc}")
        try:
            temporal = getattr(view, "_temporal_dock", None)
            if temporal is not None:
                temporal.set_data(rd)
        except Exception as exc:
            logger.warning("Snapshot readback: temporal sync failed", exc_info=True)
            view._log(f"[SnapReadback] temporal sync failed: {exc}")
        try:
            view._sync_high_perf_overlay_data()
        except Exception as exc:
            logger.warning("Snapshot readback: overlay sync failed", exc_info=True)
            view._log(f"[SnapReadback] overlay sync failed: {exc}")
        try:
            live_ts = rd.get_live_snapshot_timesteps()
            if live_ts:
                view._update_high_perf_overlay_time(float(live_ts[-1][0]))
        except Exception as exc:
            logger.warning("Snapshot readback: overlay time update failed", exc_info=True)
            view._log(f"[SnapReadback] overlay time update failed: {exc}")
        try:
            view._refresh_plot()
        except Exception as exc:
            logger.warning("Snapshot readback: plot refresh failed", exc_info=True)
            view._log(f"[SnapReadback] plot refresh failed: {exc}")

    def _on_worker_compute_finished(self, result: ComputeResult):
        view = self._view
        if result.cancelled or not result.ok:
            view._log("Run cancelled." if result.cancelled else "Run failed during compute.")
            view.set_run_button_enabled(True)
            view.set_cancel_button_enabled(False)
            self._simulation_worker = None
            return

        view._log("Compute finished; persisting results...")
        view_adapter = self._finalization_adapter(view)
        pworker = PersistenceWorker(view_adapter, result, parent=view)
        pworker.log_message.connect(view._log)
        pworker.persist_finished.connect(self._on_worker_persist_finished)
        pworker.persist_failed.connect(self._on_worker_persist_failed)
        pworker.finished.connect(self._on_persistence_worker_finished)
        self._persistence_worker = pworker
        pworker.start()
        self._simulation_worker = None

    def _on_simulation_worker_finished(self):
        """Called when SimulationWorker's QThread fully exits (main thread)."""
        if self.sender() is self._simulation_worker:
            self._simulation_worker = None

    def _on_persistence_worker_finished(self):
        """Called when PersistenceWorker's QThread fully exits (main thread)."""
        self._persistence_worker = None

    def _on_worker_compute_failed(self, message: str):
        view = self._view
        view.show_critical_message("2D SWE", f"Run failed: {message}")
        view.set_run_button_enabled(True)
        view.set_cancel_button_enabled(False)
        self._simulation_worker = None

    def _on_worker_persist_finished(self, status):
        view = self._view
        view._log("Persistence finished.")
        view.set_run_button_enabled(True)
        view.set_cancel_button_enabled(False)
        self._persistence_worker = None

    def _on_worker_persist_failed(self, message: str):
        view = self._view
        view.show_critical_message("2D SWE", f"Persistence failed: {message}")
        view.set_run_button_enabled(True)
        view.set_cancel_button_enabled(False)
        self._persistence_worker = None

    def _finalization_adapter(self, view):
        from swe2d.workbench.controllers.finalization_adapter import FinalizationAdapter
        return FinalizationAdapter(view)

    # ── Run log viewer ────────────────────────────────────────────────
    def open_run_log_viewer(self) -> None:
        """Open file dialog, select GPKG, pick run, then show the run log viewer."""
        import os as _os

        view = self._view
        db_path = view.get_open_file_name(
            "Select GeoPackage with run logs", "",
            "GeoPackage (*.gpkg);;All Files (*)",
        )
        db_path = str(db_path or "").strip()
        if not db_path:
            return
        if not _os.path.exists(db_path):
            view._log(f"[ERROR] GeoPackage not found: {db_path}")
            return

        # Load full run-log records from the GPKG (not RunRecord list)
        from swe2d.results.run_log_storage import (
            load_run_logs_from_geopackage as _load_logs,
        )
        try:
            records = _load_logs(gpkg_path=db_path)
        except Exception as exc:
            view._log(f"[ERROR] Failed to load run logs: {exc}")
            return
        if not records:
            view.show_information_message(
                "Run Log Viewer",
                "No run logs found in the selected GeoPackage.",
            )
            return

        # If multiple runs, let user pick one via a simple selection dialog
        if len(records) > 1:
            run_ids = [str(r.get("run_id", "") or "") for r in records]
            run_id, ok = view.get_input_item(
                "Select Run", "Choose a run to view logs:",
                run_ids, 0, False,
            )
            if not ok or not run_id:
                return
        else:
            run_id = str(records[0].get("run_id", "") or "")
            if not run_id:
                return

        try:
            from swe2d.workbench.dialogs.run_log_viewer_dialog import (
                SWE2DRunLogViewerDialog,
            )
            dlg_viewer = SWE2DRunLogViewerDialog(
                records=records,
                run_id=run_id,
                db_path=db_path,
                parent=view,
            )
            dlg_viewer.exec()
        except ImportError:
            view._log("[ERROR] Run log viewer dialog not available.")
        except Exception:
            view._log("[ERROR] Run log viewer failed to open.")

    # ── Cancel orchestration ──────────────────────────────────────────
    def on_cancel(self) -> None:
        """Mark the current run as cancelled.

        The view owns the cancel flag; the controller flips it, signals
        the worker thread via the cancel event, and logs the request.
        """
        view = self._view
        view._cancel_requested = True
        if self._simulation_worker is not None:
            self._simulation_worker.request_cancel()
        view._log("Cancellation requested...")

    # ── Batch simulation dialog ──────────────────────────────────────
    def open_batch_simulation_dialog(self) -> None:
        """Open the batch simulation dialog for parameter sweeps."""
        import os as _os
        from swe2d.workbench.dialogs.batch_simulation_dialog import BatchSimulationDialog

        view = self._view

        base_params = {
            "mesh": "",
            "params": {
                "rain_rate_mmhr": 0.0,
                "n_mann": 0.035,
                "duration_s": 3600.0,
            },
        }

        # Auto-populate mesh GPKG path from the current model if available
        gpkg = getattr(view, "_model_gpkg_path", "")
        if not gpkg or not _os.path.isfile(gpkg):
            gpkg = view.get_results_gpkg_path()

        dlg = BatchSimulationDialog(
            parent=view,
            base_params=base_params,
            mesh_gpkg=gpkg,
        )
        dlg.exec()

    # ── Snapshot orchestration ─────────────────────────────────────────
    def on_snapshot(self) -> None:
        """Fetch accumulated device results to host and sync to UI.

        Called when the user clicks "Fetch Device Results" during a live run.
        Triggers a D2H readback of the device snapshot ring buffer on the
        next solver step.  The reporter's post-readback callback computes
        line/coupling metrics from the read-back data and syncs to the
        temporal dock slider, high-perf overlay, and plots.
        """
        view = self._view
        worker = self._simulation_worker
        if worker is not None and worker.isRunning():
            worker.request_snapshot()
            view._log("Device fetch requested.")
            return

        # No active run — refresh UI from existing snapshots.
        results_data = getattr(view, "_results_data", None)
        if results_data is not None:
            view._sync_snapshot_to_ui()

    def on_preview_overrides(self) -> None:
        """Compute and display a summary of BC and Manning overrides.

        Generates the mesh on demand, derives default and overridden
        boundary conditions, and presents a summary via QMessageBox.
        Aborts when no boundary edges are present.
        """
        import numpy as np

        view = self._view
        if view._mesh_data is None:
            view._on_generate_mesh()
        if view._mesh_data is None:
            return

        edge_n0, edge_n1 = view._mesh_boundary_edges()
        if edge_n0.size == 0:
            view._log("No boundary edges detected in mesh.")
            view.show_information_message(
                "Preview Overrides", "No boundary edges detected in mesh."
            )
            return

        _, _, bc_type_preview, bc_val_preview = view._collect_boundary_arrays()
        bc_type_preview = bc_type_preview.copy()
        bc_val_preview = bc_val_preview.copy()
        edge_hydrographs = view._collect_bc_layer_hydrographs(edge_n0, edge_n1)

        # Compute default BC values for comparison
        from swe2d.services.mesh_computation_service import default_bc_for_edges as _compute_default_bc
        bc_type_default_arr, bc_val_default_arr = _compute_default_bc(
            view._mesh_data, edge_n0, edge_n1
        )
        static_mask = (bc_type_preview != bc_type_default_arr) | (
            ~np.isclose(bc_val_preview, bc_val_default_arr)
        )
        static_count = int(np.count_nonzero(static_mask))
        static_type_counts: Dict[str, int] = {}
        if static_count:
            for code in np.unique(bc_type_preview[static_mask]):
                label = view._bc_code_label(int(code))
                static_type_counts[label] = int(
                    np.count_nonzero(bc_type_preview[static_mask] == code)
                )

        mann_arr, mann_applied, mann_total, mann_name = view._preview_spatial_manning()
        if mann_arr is not None and mann_total > 0:
            mann_range = (
                f"{float(np.min(mann_arr)):.5f} to {float(np.max(mann_arr)):.5f}"
            )
        else:
            mann_range = f"{view.get_n_mann_value():.5f}"

        bc_layer_name = "(none)"
        bc_layer = None
        if bc_layer is not None:
            bc_layer_name = bc_layer.name()

        manning_layer_name = mann_name or "(none)"
        summary_lines = [
            f"Boundary edges detected: {edge_n0.size}",
            f"BC layer: {bc_layer_name}",
            f"Static BC overrides applied: {static_count}",
            f"Timeseries BC edges applied: {len(edge_hydrographs)}",
            f"Manning layer: {manning_layer_name}",
            f"Manning cells affected: {mann_applied}/{mann_total}",
            f"Manning n range in solver input: {mann_range}",
        ]
        if static_type_counts:
            details = ", ".join(
                f"{label}={count}"
                for label, count in sorted(static_type_counts.items())
            )
            summary_lines.insert(3, f"Static BC types: {details}")

        summary = "\n".join(summary_lines)
        view._log("Override preview:\n" + summary.replace("\n", " | "))
        view.show_information_message("Preview Overrides", summary)

    # ── Load run settings from results GeoPackage ─────────────────────
    def on_load_simulation_config(self) -> None:
        """Open a GeoPackage file picker, then a config picker, then apply.

        Two-step flow so the user can browse any .gpkg on disk (not just
        the currently-active results GPKG):
          1. ``view.get_open_file_name`` — same picker used by the
             GeoPackage Explorer action so the UX is consistent.
          2. ``SWE2DSimulationConfigDialog`` — pick which config from
             ``swe2d_simulation_configs`` to apply.

        Replaces the old behavior that silently required
        ``_current_line_results_storage_path()`` to already point at a
        valid GPKG.
        """
        view = self._view

        db_path = view.get_open_file_name(
            "Select GeoPackage to load configuration from",
            "",
            "GeoPackage (*.gpkg);;All Files (*)",
        )
        db_path = str(db_path or "").strip()
        if not db_path:
            return  # user cancelled
        if not os.path.exists(db_path):
            view._log(f"Load config skipped: GeoPackage not found: {db_path}")
            return

        from swe2d.services.gpkg_persistence_service import load_simulation_configs
        configs = load_simulation_configs(db_path, log_fn=view._log)
        if not configs:
            view._log(
                "Load config skipped: no saved simulation configs found "
                f"in the selected GeoPackage ({db_path})."
            )
            return

        from swe2d.workbench.dialogs.simulation_config_dialog import SWE2DSimulationConfigDialog
        dlg = SWE2DSimulationConfigDialog(
            configs=configs,
            db_path=db_path,
            parent=view,
            apply_callback=view._apply_run_log_metadata_to_ui,
        )
        result = dlg.exec()
        if not result:
            return
        # After applying widget state, load the associated mesh if available
        selected = getattr(dlg, "_selected_config", None)
        if selected is None:
            return
        mesh_name = str(selected.get("mesh_name", "") or "")
        if not mesh_name:
            return
        try:
            from hydra_swe2d import swe2d_deserialize_mesh
            from swe2d.services.gpkg_persistence_service import load_baked_mesh
            blob = load_baked_mesh(db_path, mesh_name)
            if blob is None:
                view._log(f"Config references mesh '{mesh_name}' but mesh BLOB not found in GPKG.")
                return
            pm = swe2d_deserialize_mesh(blob)
            # Per baked BLOB spec: mesh stays in RCMK order.
            mesh_data = {
                "node_x": np.asarray(pm.node_x, dtype=np.float64),
                "node_y": np.asarray(pm.node_y, dtype=np.float64),
                "node_z": np.asarray(pm.node_z, dtype=np.float64),
                "cell_nodes": np.asarray(pm.cell_face_nodes, dtype=np.int32) if pm.cell_face_nodes is not None else np.empty(0, dtype=np.int32),
            }
            cfo = pm.cell_face_offsets
            if cfo is not None:
                mesh_data["cell_face_offsets"] = np.asarray(cfo, dtype=np.int32)
            cfn = pm.cell_face_nodes
            if cfn is not None:
                mesh_data["cell_face_nodes"] = np.asarray(cfn, dtype=np.int32)
            if mesh_data.get("node_x") is not None:
                view._mesh_data = mesh_data
                view._reset_runtime_snapshot_overlay_cache("mesh loaded from config")
                view._result_data = None
                view.show_mesh_tab()
                try:
                    view._refresh_plot()
                except RuntimeError:
                    pass
                view._log(f"Mesh '{mesh_name}' loaded from config ({mesh_data['node_x'].size} nodes)")
        except Exception as exc:
            view._log(f"[ERROR] Failed to load mesh from config: {exc}")

    def on_save_simulation_config(self) -> None:
        """Save the current widget configuration to a user-chosen GeoPackage.

        Two-step flow mirroring ``on_load_simulation_config``:
          1. ``view.get_save_file_name`` — user picks an existing
             .gpkg or types a new path. Matches the GeoPackage Explorer
             picker so the UX is consistent.
          2. ``view.get_input_text`` — prompt for a descriptive config name
             (timestamp used if blank).

        Replaces the old behavior that silently required
        ``_current_line_results_storage_path()`` to point at a writable
        GPKG.
        """
        from qgis.PyQt import QtWidgets as _QtWidgets

        view = self._view

        # Pre-fill the picker with the current results GPKG if one is set,
        # so the common case is a single click + a config name.
        start_dir = ""
        current_db = str(view._current_line_results_storage_path() or "")
        if current_db and os.path.exists(os.path.dirname(os.path.abspath(current_db))):
            start_dir = current_db

        db_path = view.get_save_file_name(
            "Select GeoPackage to save configuration to",
            start_dir,
            "GeoPackage (*.gpkg);;All Files (*)",
        )
        db_path = str(db_path or "").strip()
        if not db_path:
            return  # user cancelled
        # If the user typed a path without an extension, add .gpkg.
        if not os.path.splitext(db_path)[1]:
            db_path = db_path + ".gpkg"

        # Prompt for a config name
        name, ok = view.get_input_text("Save Config", "Configuration name:", "")
        if not ok:
            return
        config_name = str(name).strip()
        if not config_name:
            config_name = datetime.datetime.now().astimezone().strftime("swe2d_%Y%m%dT%H%M%S%z")

        from swe2d.workbench.bridges.project_settings_bridge import collect_workbench_widget_state
        from swe2d.services.gpkg_persistence_service import persist_simulation_config

        mesh_name = str(getattr(view, "_mesh_data", {}).get("mesh_name", "") or "")
        widget_attrs = list(view.collect_run_widget_params().keys())
        widget_state = collect_workbench_widget_state(
            ui=view,
            widget_attrs=widget_attrs,
            qtwidgets_module=_QtWidgets,
        )

        # Get run duration from the UI
        try:
            run_dur = view.model_tab.get_run_time_hours_parsed() * 3600.0
        except Exception:
            run_dur = 0.0

        persist_simulation_config(
            gpkg_path=db_path,
            config_id=config_name,
            mesh_name=mesh_name,
            run_duration_s=run_dur,
            widget_state=widget_state,
            log_fn=view._log,
        )
        view._log(f"Configuration saved as '{config_name}' to {db_path}.")

    def on_preview_coupling(self) -> None:
        """Compute and display a coupling configuration preview.

        Builds pipe network and hydraulic structure configs from widget state,
        validates them (unknown refs, zero capacity, near-zero head), and
        shows a summary via QMessageBox.
        """
        view = self._view
        if view._mesh_data is None:
            view.show_information_message(
                "Coupling Preview",
                "Generate or load a mesh first so cell-based coupling "
                "indices can be resolved.",
            )
            return

        pipe_cfg = view._build_pipe_network_config()
        struct_cfg = view._build_hydraulic_structure_config()

        if pipe_cfg is None and struct_cfg is None:
            view.show_information_message(
                "Coupling Preview",
                "No valid drainage or structure layers are configured.",
            )
            return

        from swe2d.runtime.coupling import validate_coupling_configs
        lines = validate_coupling_configs(
            pipe_cfg=pipe_cfg, struct_cfg=struct_cfg,
            n_cells=int(view._mesh_cell_areas().shape[0]),
        )

        view.show_information_message("Coupling Preview", "\n".join(lines))
