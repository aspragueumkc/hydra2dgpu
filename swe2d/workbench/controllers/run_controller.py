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

from swe2d.core.executor import ComputeResult, SnapshotData
from swe2d.core.mesh_service import apply_cell_permutation
from swe2d.core.run_context import RunContext
from swe2d.workbench.workers.simulation_worker import SimulationWorker

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from swe2d.workbench.studio_dialog import SWE2DWorkbenchStudioDialog


# Module-level alias used by the snapshot orchestration below. This is kept
# because ``on_snapshot`` still calls into the HEC-RAS HDF5 export.


def _capture_inflow_progressive(view) -> bool:
    """Read the inflow-progressive checkbox from the model tab on the main thread."""
    mtv = getattr(view, "_model_tab_view", None)
    if mtv is None:
        return False
    chk = getattr(mtv, "inflow_progressive_chk", None)
    if chk is None:
        return False
    try:
        return bool(chk.isChecked())
    except Exception:
        return False


def _capture_edge_groups(view, bc_n0, bc_n1) -> Dict[int, str]:
    """Resolve edge-group labels for BC override flows on the main thread."""
    bc_n0 = np.asarray(bc_n0, dtype=np.int32)
    bc_n1 = np.asarray(bc_n1, dtype=np.int32)
    if bc_n0.size == 0 or bc_n1.size == 0:
        return {}
    cached = getattr(view, "_cached_edge_groups", None)
    if cached:
        return dict(cached)
    collector = getattr(view, "_collect_bc_layer_edge_groups", None)
    if collector is None:
        return {}
    try:
        groups = collector(bc_n0, bc_n1)
        return {int(k): str(v) for k, v in dict(groups).items()}
    except Exception as exc:
        logger.debug("_capture_edge_groups failed: %s", exc)
        return {}


class RunController:
    """MVP domain controller for the 2D simulation run pipeline."""
    """Mediates between Service Layer and View (SWE2DWorkbenchStudioDialog).

    Holds a reference to the View (dialog). Methods are called either
    directly by the dialog or in response to View signals.
    """

    def __init__(self, view: "SWE2DWorkbenchStudioDialog"):
        self._view = view
        self._simulation_worker = None
        self._current_run_id = ""
        self._batch_manager = None
        self._batch_dialog = None
        self._gpu_viewer_dialogs = []  # phase 6: keep refs alive

    @property
    def batch_manager(self):
        """Lazy-initialize the BatchManager singleton."""
        if self._batch_manager is None:
            from swe2d.workbench.services.batch_manager import BatchManager
            self._batch_manager = BatchManager()
        return self._batch_manager

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
        self._current_run_id = context.run_id

        view._cancel_requested = False
        view.set_run_button_enabled(False)
        view.set_cancel_button_enabled(True)
        view.set_run_progress(0)

        # Clear stale overlay/snapshot state from any previous run so the
        # new run starts with a clean slate (run records, overlay key, mesh
        # arrays).  Without this, _ensure_live_run_record returns early
        # because old run records are still present, causing numbering
        # scrambles across consecutive runs in the same session.
        try:
            view._reset_runtime_snapshot_overlay_cache("new run starting")
        except Exception:
            pass

        # Ensure _results_data exists so snapshot_ready signals have a place to land.
        try:
            view._show_results_panel()
        except Exception:
            pass

        worker = SimulationWorker(context, parent=view)
        worker.log_message.connect(view._log)
        worker.progress_percent.connect(view.set_run_progress)
        worker.snapshot_ready.connect(self._on_worker_snapshot_ready)
        worker.mesh_permutation_ready.connect(self._on_worker_mesh_permutation_ready)
        worker.compute_finished.connect(self._on_worker_compute_finished)
        worker.compute_failed.connect(self._on_worker_compute_failed)
        worker.finished.connect(self._on_simulation_worker_finished)
        self._simulation_worker = worker
        worker.start()
        return None

    def _build_run_context(self, request: Optional[Any] = None) -> Optional[RunContext]:
        """Capture all widget values and arrays into a RunContext.

        **Phase 1.B true flip.**  This controller method does two things only:

        1. Call live-QGIS-layer dialog callbacks to compute forcing and
           initial-condition data (the inline equivalent of the retired
           ``SWE2DRunDataBuilder`` / ``SWE2DRunOptionsBuilder``).
        2. Pack those objects together with widget scalars, unit-system
           info, and per-run callbacks into a single spec dict, then
           delegate to
           :func:`swe2d.workbench.adapters.run_context_adapter.build_run_context_from_gui`,
           which calls the canonical
           :func:`swe2d.runtime.run_context_builder.build_run_context`
           exactly once.

        The returned ``RunContext`` is the canonical builder's output with
        no ``dataclasses.replace`` post-pass — every GUI-supplied object
        arrives through the spec under its canonical field name, and the
        builder's ``_override`` helper honors it.
        """
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

        parse_time_hours_fn = view._parse_time_hours
        model_gpkg_path = view._model_gpkg_path
        length_unit_name = view._length_unit_name

        wp = view.collect_run_widget_params()

        last_run_request = view._last_run_request
        if request is None:
            request = last_run_request

        # ── 1. Build GUI-only forcing / mesh objects from live layers ────
        # These are the live-QGIS-layer callbacks formerly wrapped by the
        # retired SWE2DRunDataBuilder.build().  They turn live QGIS layer
        # objects into the RunContext-ready initial-condition / BC arrays
        # the canonical builder cannot reproduce from the GPKG path alone.
        bc_n0, bc_n1, bc_tp, bc_vl, bc_relax = view._collect_boundary_arrays()
        side_hydrographs: Dict[Any, Any] = {}  # legacy placeholder — empty for GUI path
        edge_hydrographs = view._collect_bc_layer_hydrographs(bc_n0, bc_n1)
        edge_group_overrides = view._collect_bc_layer_edge_groups(bc_n0, bc_n1)
        h0, hu0, hv0 = view._initial_state(
            bc_n0=bc_n0, bc_n1=bc_n1, bc_tp=bc_tp,
        )
        n_mann_cell = view._build_spatial_manning_array()
        view._update_unit_system_from_crs()

        run_input: Dict[str, Any] = {
            "h0": h0,
            "hu0": hu0,
            "hv0": hv0,
            "bc_tp": bc_tp,
            "bc_vl": bc_vl,
            "bc_relax": bc_relax,
            "bc_n0": bc_n0,
            "bc_n1": bc_n1,
            "side_hydrographs": side_hydrographs,
            "edge_hydrographs": edge_hydrographs,
            "edge_group_overrides": edge_group_overrides,
            "n_mann_cell": n_mann_cell,
        }

        # Keep per-cell scalar arrays in mesh data so the overlay can render
        # mannings_n / curve_number fields without rebuilding them.
        mesh_data["n_mann_cell"] = n_mann_cell
        mesh_data["cn_cell"] = view._build_spatial_cn_array()

        # ── 2. Build GUI-only forcing / coupling objects from live layers ──
        # The retired SWE2DRunOptionsBuilder.build() did this; inlined here.
        # The forcing objects flow through the spec to the canonical builder
        # so the array path is exercised end-to-end in the GUI too.
        if int(wp["reconstruction_combo"]) == 8:
            raise RuntimeError(
                "FV_MP5 (spatial scheme 8) is currently disabled. "
                "It is unstable on unstructured triangular meshes. "
                "Use WENO5 (scheme 7), Barth-Jespersen (scheme 5), or a MUSCL TVD scheme (1–4)."
            )
        from swe2d.runtime.backend import swe2d_gpu_available as _gpu_check
        if not _gpu_check():
            raise RuntimeError("CUDA GPU is required but unavailable or check failed.")

        rain_rate_mmhr = float(wp["rain_rate_spin"])
        rain_rate_model = view._rain_rate_si_to_model(rain_rate_mmhr / 1000.0 / 3600.0)
        internal_flow_forcing = view._build_internal_flow_forcing()
        cell_source_si = view._internal_flow_source_cms_at_time(internal_flow_forcing, 0.0)
        cell_source_model = view._flow_si_to_model(cell_source_si) if cell_source_si is not None else None
        thiessen_forcing = view._build_thiessen_rain_cn_forcing()
        pipe_network_cfg = view._build_pipe_network_config()
        hydraulic_structures_cfg = view._build_hydraulic_structure_config()

        forcing: Dict[str, Any] = {
            "internal_flow_forcing": internal_flow_forcing,
            "cell_source_model": cell_source_model,
            "thiessen_forcing": thiessen_forcing,
            "rain_rate_model": rain_rate_model,
            "pipe_network_cfg": pipe_network_cfg,
            "hydraulic_structures_cfg": hydraulic_structures_cfg,
        }

        # Bridge stacked plans are controller-owned because they bind
        # mesh_data + hydraulic_structures_cfg to runtime layout.  Plumb
        # them through the spec so the canonical builder applies them.
        bridge_stacked_plans: List[Any] = []
        try:
            from swe2d.runtime.bridge_stacked_runtime import build_bridge_stacked_plans_for_runtime
            bridge_stacked_plans = build_bridge_stacked_plans_for_runtime(
                mesh_data, hydraulic_structures_cfg, log_fn=log_fn,
            )
        except Exception as exc:
            log_fn(f"Bridge stacked-plan mapping warning: {exc}")

        # ── 3. Resolve run_duration_s / output_interval_s from request ──
        from swe2d.workbench.services.text_parser_service import parse_duration_seconds as _parse_dur_s
        # Use wp["run_time_edit"] with fail-fast access (Rule 7 — no silent
        # .get() fallback; the widget key must be present, wiring bugs raise).
        _dur_raw = str(wp["run_time_edit"]).strip()
        run_duration_s = _parse_dur_s(_dur_raw) if _dur_raw else 0.0
        if request is not None:
            request_run_duration_text = getattr(request, "run_duration_text", None)
            if request_run_duration_text is not None and str(request_run_duration_text).strip():
                run_duration_s = _parse_dur_s(str(request_run_duration_text).strip())

        # Resolve output_interval_s — prefer request override, else widget.
        _oi_override = None
        if request is not None:
            _oi_override = getattr(request, "output_interval_text", None)
        if _oi_override is not None and str(_oi_override).strip():
            _oi_raw = str(_oi_override).strip()
        else:
            _oi_raw = str(wp["output_interval_edit"]).strip()
        _oi_s = _parse_dur_s(_oi_raw) if _oi_raw else 3600.0  # default 1 hour
        output_interval_s = _oi_s if _oi_s >= 1.0 else 1.0

        results_gpkg_path = str(view._current_line_results_storage_path() or "")
        run_log_start_idx = len(view._runtime_log_lines)
        mesh_cell_areas_fn = view._mesh_cell_areas

        # ── 4. Pack the spec dict and delegate to the adapter ───────────
        # The adapter translates this dict into a swe2d-run/2 spec,
        # forwards GUI-built objects via canonical field names (so the
        # builder's _override helper honors them), and calls the canonical
        # builder exactly once.  No replace() post-pass required.
        widget_state: Dict[str, Any] = dict(wp)
        # Storage-result checkboxes live on _model_tab_view but are NOT in
        # collect_run_widget_params() (they ride the versioned
        # collect_widget_state_for_save() payload).  Mirror them into
        # widget_state here so the GUI RunContext sees the live checkbox
        # value — without this the canonical builder falls back to the
        # _DEFAULTS for save_line_results / save_coupling_results /
        # save_run_log and the GUI path drifts from the CLI replay path.
        mtv = view.model_tab
        for _attr, _key in (
            ("save_mesh_chk", "save_mesh_chk"),
            ("save_line_chk", "save_line_chk"),
            ("save_coupling_chk", "save_coupling_chk"),
            ("save_log_chk", "save_log_chk"),
            ("save_max_only_chk", "save_max_only_chk"),
        ):
            _chk = getattr(mtv, _attr, None)
            if _chk is not None:
                widget_state[_key] = bool(_chk.isChecked())
        widget_state["run_duration_s"] = run_duration_s
        widget_state["output_interval_s"] = output_interval_s
        widget_state["uniform_inflow_enabled"] = view.model_tab.is_uniform_inflow()
        widget_state["rain_update_interval_s"] = view.model_tab.get_rain_update_interval_s()
        widget_state["_length_unit_name"] = length_unit_name
        widget_state["_length_scale_si_to_model"] = float(view._length_scale_si_to_model())
        widget_state["_rain_mm_to_model_depth"] = float(view._rain_mm_to_model_depth())

        # Phase 3.5: stop embedding view-bound callables in RunContext.
        # The executor (core/executor.py) re-binds pure versions internally
        # using RunContext fields (edge_groups, inflow_progressive_enabled,
        # source_rate_cap, etc.) rather than the dialog methods.  The wiring
        # controller still wires `dialog._apply_timeseries_bc_values` and
        # `dialog._distribute_total_flow_to_unit_q` into the backend
        # initializer (these are needed for the legacy initializer path),
        # but the spec/RunContext no longer carries them.
        # No more widget_state["_apply_timeseries_bc"] / _distribute_total_flow
        # / _apply_external_sources / _build_line_sampling_map here.
        widget_state["_mesh_cell_areas_fn"] = mesh_cell_areas_fn
        widget_state["_mesh_cell_min_bed_fn"] = view._mesh_cell_min_bed
        widget_state["_mesh_cell_centroids_fn"] = view._mesh_cell_centroids
        widget_state["_internal_flow_cms_fn"] = view._internal_flow_source_cms_at_time

        # GUI-only runtime values: run_log_start_idx (per-run counter),
        # bridge_stacked_plans (controller-glued mesh + structures),
        # edge_groups (per-run BC layer grouping).
        widget_state["_run_log_start_idx"] = run_log_start_idx
        widget_state["_bridge_stacked_plans"] = bridge_stacked_plans
        widget_state["_edge_groups"] = _capture_edge_groups(view, bc_n0, bc_n1)

        from swe2d.workbench.adapters.run_context_adapter import build_run_context_from_gui
        return build_run_context_from_gui(
            widget_state,
            mesh_data=mesh_data,
            forcing=forcing,
            run_input=run_input,
            sample_map_data=list(view._build_line_sampling_map() or []),
            inflow_progressive_enabled=_capture_inflow_progressive(view),
            edge_groups=widget_state["_edge_groups"],
            results_gpkg_path=results_gpkg_path,
            model_gpkg_path=str(model_gpkg_path or ""),
            mesh_crs_wkt=str(mesh_data.get("crs_wkt", "") or ""),
            parse_time_hours_fn=parse_time_hours_fn,
        )

    def _on_worker_mesh_permutation_ready(self, cell_perm, result_holder):
        """Apply solver cell permutation to the view mesh on the main thread.

        The worker cannot safely touch ``view._mesh_data`` or the sample-lines
        layer from its thread.  We permute the canonical view mesh here, rebuild
        the line-sampling map from the permuted geometry, and signal the worker
        to continue.
        """
        view = self._view
        try:
            if cell_perm is not None and len(cell_perm) > 0:
                cell_perm_arr = np.asarray(cell_perm, dtype=np.int32)
                apply_cell_permutation(view._mesh_data, cell_perm_arr)
                # Keep per-cell scalar arrays in sync with the reordered mesh.
                for key in ("n_mann_cell", "cn_cell"):
                    arr = view._mesh_data.get(key)
                    if arr is not None and arr.size > 0:
                        view._mesh_data[key] = np.asarray(arr, dtype=np.float64).ravel()[cell_perm_arr]
            sample_map = list(view._build_line_sampling_map() or [])
            result_holder.sample_map = sample_map
        except Exception as exc:
            logger.exception("Mesh permutation ready handler failed")
            result_holder.error = str(exc)
        finally:
            result_holder.event.set()

    def _on_worker_snapshot_ready(self, data: SnapshotData):
        view = self._view
        rd = getattr(view, "_results_data", None)
        if rd is None:
            return
        try:
            if getattr(data, "timesteps", None):
                rd.set_live_snapshot_timesteps(
                    list(data.timesteps),
                    t_sec=float(data.timesteps[-1][0]),
                )
        except Exception as exc:
            logger.warning("Snapshot readback: merge failed", exc_info=True)
            view._log(f"[SnapReadback] merge failed: {exc}")
        try:
            self._ensure_live_run_record(rd)
        except Exception as exc:
            logger.warning("Snapshot readback: RunRecord seed failed", exc_info=True)
            view._log(f"[SnapReadback] RunRecord seed failed: {exc}")
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
            if getattr(data, "coupling_data", None):
                rd._live_coupling = data.coupling_data
        except Exception as exc:
            logger.warning("Snapshot readback: coupling sync failed", exc_info=True)
            view._log(f"[SnapReadback] coupling sync failed: {exc}")
        try:
            if getattr(data, "pipe_cell_data", None):
                rd._live_pipe_cell = data.pipe_cell_data
                view._log(f"[SnapReadback] pipe-cell sync: {len(data.pipe_cell_data)} keys")
        except Exception as exc:
            logger.warning("Snapshot readback: pipe-cell sync failed", exc_info=True)
            view._log(f"[SnapReadback] pipe-cell sync failed: {exc}")
        try:
            line_ts = getattr(data, "line_ts", None)
            line_profiles = getattr(data, "line_profiles", None)
            logger.debug("[LINE_DIAG] controller: line_ts=%d keys, line_profiles=%d keys",
                           len(line_ts) if line_ts else 0, len(line_profiles) if line_profiles else 0)
            if line_ts:
                rd._live_line_ts = line_ts
            if line_profiles:
                rd._live_line_profile = line_profiles
            # Sync _live_times from the SnapshotData timesteps so the viewer
            # sees a consistent (t_s, metric) pair.  The worker thread may
            # update rd._live_times after we read it, so we must capture our
            # own copy here alongside the line_ts snapshot.
            ts_list = getattr(data, "timesteps", None)
            if ts_list and len(ts_list) > 0:
                rd._live_times = np.array([float(t[0]) for t in ts_list], dtype=np.float64)
            logger.debug("[LINE_DIAG] controller: rd._live_line_ts=%d keys, rd._live_line_profile=%d keys",
                           len(rd._live_line_ts) if hasattr(rd._live_line_ts, '__len__') else '?',
                           len(rd._live_line_profile) if hasattr(rd._live_line_profile, '__len__') else '?')
        except Exception as exc:
            logger.warning("Snapshot readback: line data sync failed", exc_info=True)
            view._log(f"[SnapReadback] line data sync failed: {exc}")
        try:
            view._refresh_plot()
        except Exception as exc:
            logger.warning("Snapshot readback: plot refresh failed", exc_info=True)
            view._log(f"[SnapReadback] plot refresh failed: {exc}")

    def _ensure_live_run_record(self, rd):
        """Seed a synthetic RunRecord on first snapshot so the runs list,
        plot viewer, and overlay-selected key all show the live run.

        Without this, the plot viewer has nothing to plot and the runs list
        stays empty until persistence completes.
        """
        from swe2d.results.run_service import RunRecord, next_color
        run_id = str(self._current_run_id or "")
        if not run_id:
            return
        # Check if we already have a record for THIS run (not just any run).
        for rec in rd.get_run_records():
            if rec.run_id == run_id:
                return
        rec = RunRecord(
            run_id=run_id,
            gpkg_path="",
            color=next_color(0),
            enabled=True,
            label=f"Live: {run_id}",
        )
        rd._run_records = [rec]
        rd._live_run_id = run_id
        rd._overlay_selected_key = str(rec.key)
        self._view.refresh_results_run_list()

    def _on_worker_compute_finished(self, result: ComputeResult):
        view = self._view
        if not result.ok and not result.cancelled:
            view._log("Run failed during compute.")
            view.set_run_button_enabled(True)
            view.set_cancel_button_enabled(False)
            self._current_run_id = ""
            self._simulation_worker = None
            return

        if result.cancelled:
            view._log("Run cancelled; persisting partial results...")
        else:
            view._log("Compute finished; persisting results...")
        view_adapter = self._finalization_adapter(view)
        try:
            from swe2d.runtime.run_finalizer import SWE2DRunFinalizer
            finalizer = SWE2DRunFinalizer(view_adapter)
            status = finalizer.finalize_and_persist(
                h=result.h,
                hu=result.hu,
                hv=result.hv,
                final_sim_time_s=result.final_sim_time_s,
                n_area=result.n_area,
                area_model=result.area_model,
                storage_start_model=result.storage_start_model,
                source_budget_model=result.source_budget_model,
                source_step_rows_model=result.source_step_rows_model,
                run_duration_s=result.run_duration_s,
                boundary_flux_budget_model=result.boundary_flux_budget_model,
                boundary_flux_step_rows_model=result.boundary_flux_step_rows_model,
                run_id=result.run_id,
                output_interval_s=result.output_interval_s,
                run_perf_start=result.run_perf_start,
                run_wallclock_start=result.run_wallclock_start,
                run_log_start_idx=result.run_log_start_idx,
                thiessen_forcing=result.thiessen_forcing,
                rain_stats_acc=result.rain_stats_acc,
                save_line_results=result.save_line_results,
                save_coupling_results=result.save_coupling_results,
                save_mesh_results=result.save_mesh_results,
                save_run_log=result.save_run_log,
                save_max_only=result.save_max_only,
                h_min=result.h_min,
                mesh_name=result.mesh_name,
                max_tracking=result.max_tracking,
                snapshot_timesteps=result.snapshot_timesteps,
                coupling_snapshots=result.coupling_snapshots,
                precomputed_line_results=result.precomputed_line_results,
                pipe_cell_items=getattr(result, "pipe_cell_items", None),
            )
            for msg in finalizer.drain_log_messages():
                view._log(msg)
            view._log("Persistence finished." if status.get("ok") else f"Persistence completed with issues: {status}")
            # Update the live RunRecord with the real gpkg path so the
            # results viewer can read coupling / baked mesh from it.
            try:
                results_data = getattr(view, "_results_data", None)
                if results_data is not None:
                    gpkg = str(getattr(view, "_current_line_results_storage_path", lambda: "")())
                    if gpkg:
                        for rec in results_data.get_run_records():
                            if rec.run_id == result.run_id:
                                rec.gpkg_path = gpkg
                                break
                        # Auto-save simulation config with run
                        try:
                            from swe2d.services.gpkg_persistence_service import persist_simulation_config
                            from swe2d.core.builder import widget_state_to_flat_params
                            widget_state = self.collect_widget_state_for_save()
                            flat_params = widget_state_to_flat_params(widget_state)
                            if result.run_duration_s:
                                flat_params["run_duration_s"] = result.run_duration_s
                            persist_simulation_config(
                                gpkg_path=gpkg,
                                config_id=result.run_id,
                                mesh_name=result.mesh_name,
                                run_duration_s=result.run_duration_s,
                                widget_state=widget_state,
                                params=flat_params,
                                description=f"Auto-saved run: {result.run_id}",
                                log_fn=view._log,
                            )
                        except Exception:
                            pass
                view._on_results_refresh()
            except Exception:
                pass
        except Exception as exc:
            view.show_critical_message("2D SWE", f"Persistence failed: {exc}")
        finally:
            view.set_run_button_enabled(True)
            view.set_cancel_button_enabled(False)
            self._current_run_id = ""
            self._simulation_worker = None

    def _on_simulation_worker_finished(self):
        """Called when SimulationWorker's QThread fully exits (main thread)."""
        self._simulation_worker = None

    def _on_worker_compute_failed(self, message: str):
        view = self._view
        view.show_critical_message("2D SWE", f"Run failed: {message}")
        view.set_run_button_enabled(True)
        view.set_cancel_button_enabled(False)
        self._current_run_id = ""
        self._simulation_worker = None

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

    # ── GPU Direct Viewer (Phase 6) ───────────────────────────────────
    def open_gpu_direct_viewer(self, mesh_data, parent) -> None:
        """Open the standalone GPU Direct Viewer dialog.

        Owns the keep-alive list for the dialog so Python's GC doesn't
        drop it while the user is interacting.  Multiple opens are
        allowed (each becomes a separate dialog).  Falls back to the
        install dialog if the hydra-swe2d backend isn't available.

        The dialog is GPU-direct only (no CPU rasterizer fallback —
        the high-perf canvas overlay covers that case).  When a
        simulation is currently running, the active ``SimulationWorker``
        owns the underlying ``SWE2DSolver`` whose device pointer is
        registered with CUDA-OpenGL interop.  Outside a run, the
        dialog opens with no live data and shows "waiting for run…".

        Parameters mirror ``GPUViewerDialog.__init__``:
            mesh_data  dict with cell_x / cell_y arrays (may be empty
                        for a dialog opened before a simulation loads)
            parent     Qt widget (typically the workbench dialog)
        """
        view = self._view
        from qgis.PyQt import QtWidgets as _QtWidgets
        try:
            from swe2d.workbench.views.gpu_viewer_dialog import (
                GPUViewerDialog,
            )
        except Exception as exc:
            logging.getLogger(__name__).error(
                "GPU Direct Viewer import failed: %s", exc,
            )
            QtWidgets.QMessageBox.warning(
                parent,
                "HYDRA2DGPU",
                f"GPU Direct Viewer import failed: {exc}",
            )
            return
        # Resolve the active solver / worker, if any.  The GL render
        # path needs a real PySolver to fetch the dev_ptr; without it
        # the widget stays idle (waiting message).  No CPU fallback
        # path — that responsibility belongs to the high-perf canvas
        # overlay.
        active_solver = None
        if self._simulation_worker is not None and self._simulation_worker.isRunning():
            active_solver = self._simulation_worker.get_active_solver()
            if active_solver is not None:
                view._log(
                    "GPU Direct Viewer: using GPU-direct (zero-D2H) path."
                )
            else:
                view._log(
                    "GPU Direct Viewer: run in progress but solver "
                    "not yet exposed — waiting."
                )
        else:
            view._log(
                "GPU Direct Viewer: no active run — dialog will wait."
            )
        try:
            dlg = GPUViewerDialog(
                mesh_data=mesh_data, parent=parent,
                # Callable so the widget picks up the solver when a run
                # starts AFTER the dialog was opened (common workflow).
                get_solver_fn=lambda: (
                    self._simulation_worker.get_active_solver()
                    if self._simulation_worker is not None
                    and self._simulation_worker.isRunning()
                    else None
                ),
            )
        except Exception as exc:
            logging.getLogger(__name__).error(
                "GPU Direct Viewer init failed: %s", exc,
            )
            QtWidgets.QMessageBox.warning(
                parent,
                "HYDRA2DGPU",
                f"GPU Direct Viewer init failed: {exc}",
            )
            return
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()
        # Keep alive — drop on close.
        self._gpu_viewer_dialogs.append(dlg)
        dlg.destroyed.connect(
            lambda _ref=dlg: self._gpu_viewer_dialogs.remove(_ref)
            if _ref in self._gpu_viewer_dialogs else None
        )
        view._log("GPU Direct Viewer opened.")

    # ── Batch simulation dialog ──────────────────────────────────────
    def open_batch_simulation_dialog(self) -> None:
        """Open the batch simulation dialog for parameter sweeps."""
        import os as _os
        from swe2d.workbench.dialogs.batch_simulation_dialog import BatchSimulationDialog

        view = self._view

        # Reuse existing dialog if it exists
        if self._batch_dialog is not None:
            self._batch_dialog.show()
            self._batch_dialog.raise_()
            self._batch_dialog.activateWindow()
            return

        base_params = {
            "mesh": "",
            "params": {
                "rain_rate_mmhr": 0.0,
                "n_mann": 0.035,
                "duration_s": 3600.0,
            },
        }

        gpkg = getattr(view, "_model_gpkg_path", "")
        if not gpkg or not _os.path.isfile(gpkg):
            gpkg = view.get_results_gpkg_path()

        self._batch_dialog = BatchSimulationDialog(
            parent=view,
            base_params=base_params,
            mesh_gpkg=gpkg,
            batch_manager=self.batch_manager,
        )
        self._batch_dialog.show()

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

        _, _, bc_type_preview, bc_val_preview, bc_relax_preview = view._collect_boundary_arrays()
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

    # ── Config save/load helpers ──────────────────────────────────────

    def collect_widget_state_for_save(self) -> dict:
        """Collect widget state + data sources for any config save path.

        Returns a versioned widget_state dict ({"version": 1, "widgets": {...}})
        with _data_sources appended — the format expected by
        ``restore_workbench_widget_state`` and ``build_run_context_from_dict``
        (via the params extractor added to the builder).

        Uses the same widget attr list (excluding gravity/k_mann) as manual save.
        """
        view = self._view
        from qgis.PyQt import QtWidgets as _QtWidgets
        from swe2d.workbench.bridges.project_settings_bridge import collect_workbench_widget_state

        all_attrs = list(view.collect_run_widget_params().keys())
        widget_attrs = [k for k in all_attrs if k not in ("gravity", "k_mann")]
        # Include raw storage checkbox widget names — their derived key names
        # (save_mesh_results_to_gpkg_chk, etc.) are not widget attribute names
        # on _model_tab_view so collect_workbench_widget_state skips them.
        widget_attrs.extend(["save_mesh_chk", "save_line_chk", "save_coupling_chk",
                            "save_log_chk"])
        widget_state = collect_workbench_widget_state(
            ui=view._model_tab_view,
            widget_attrs=widget_attrs,
            qtwidgets_module=_QtWidgets,
        )
        if hasattr(view, "collect_data_source_config"):
            ds = view.collect_data_source_config()
            if ds:
                widget_state["_data_sources"] = ds
        return widget_state

    def _build_replay_payload(
        self,
        widget_state: dict,
        mesh_name: str,
        run_duration_s: float,
        mesh_gpkg_path: str = "",
        run_id: str = "",
    ) -> dict:
        """Build a CLI-replay JSON payload from widget state.

        The payload format matches the ``swe2d-replay/1`` schema understood by
        ``build_run_context_from_dict``.  The ``params`` block is populated
        from the versioned ``widget_state`` so the builder can read actual
        values (not just defaults).  The ``units`` block is intentionally left
        empty so the CLI derives unit conversions from the mesh CRS.
        """
        from swe2d.core.builder import widget_state_to_flat_params
        flat_params = widget_state_to_flat_params(
            widget_state,
            mesh_gpkg=mesh_gpkg_path,
            mesh_name=mesh_name,
        )
        # Discard any computed units block; CLI derives conversions from CRS.
        flat_params.pop("_units_block", None)
        # Always include run_duration_s (not all widgets capture it)
        if run_duration_s:
            flat_params["run_duration_s"] = run_duration_s

        # Capture the CRS from the current mesh data
        crs_wkt = ""
        view = self._view
        mesh_data = getattr(view, "_mesh_data", None) or {}
        crs_wkt = str(mesh_data.get("crs_wkt", "") or "")

        return {
            "schema_version": "swe2d-replay/1",
            "run_id": run_id or datetime.datetime.now().astimezone().strftime("swe2d_%Y%m%dT%H%M%S%z"),
            "mesh": {
                "gpkg_path": mesh_gpkg_path,
                "mesh_name": mesh_name,
                "crs_wkt": crs_wkt,
            },
            "params": flat_params,
            "data_sources": widget_state.get("_data_sources", {}),
            "results": {},
            "units": {},
            "widget_state": widget_state,
            "run_duration_s": run_duration_s or flat_params.get("run_duration_s", 0.0),
        }

    def build_replay_payload(
        self,
        widget_state: dict,
        mesh_name: str,
        run_duration_s: float,
        mesh_gpkg_path: str = "",
        run_id: str = "",
    ) -> dict:
        """Public wrapper for ``_build_replay_payload``.

        Allows child dialogs (e.g. Batch Simulation) to build the same
        ``swe2d-replay/1`` JSON payload used by the main JSON export.
        """
        return self._build_replay_payload(
            widget_state=widget_state,
            mesh_name=mesh_name,
            run_duration_s=run_duration_s,
            mesh_gpkg_path=mesh_gpkg_path,
            run_id=run_id,
        )

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
                    view._update_mesh_canvas_layer()
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
        if not os.path.splitext(db_path)[1]:
            db_path = db_path + ".gpkg"

        from swe2d.services.gpkg_persistence_service import persist_simulation_config, load_simulation_configs

        mesh_name = str((getattr(view, "_mesh_data", None) or {}).get("mesh_name", "") or "")
        widget_state = self.collect_widget_state_for_save()

        try:
            run_dur = view.model_tab.get_run_time_hours_parsed() * 3600.0
        except Exception:
            run_dur = 0.0

        # Query existing tables so the dialog can warn about overwriting a whole table
        try:
            import sqlite3 as _sqlite3
            with _sqlite3.connect(db_path) as _c:
                _cur = _c.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'swe2d_%'"
                )
                existing_tables = [str(r[0]) for r in _cur.fetchall()]
        except Exception:
            existing_tables = []

        default_table = f"swe2d_sim_{mesh_name or 'config'}_{datetime.datetime.now().strftime('%Y%m%dT%H%M%S')}"

        def do_persist(gpkg, tbl, cfg_id, ws, desc, dur):
            from swe2d.core.builder import widget_state_to_flat_params
            flat_params = widget_state_to_flat_params(ws)
            if dur:
                flat_params["run_duration_s"] = dur
            persist_simulation_config(
                gpkg_path=gpkg,
                config_id=cfg_id,
                mesh_name=mesh_name,
                run_duration_s=dur,
                widget_state=ws,
                params=flat_params,
                description=desc,
                log_fn=view._log,
                table_name=tbl,
            )
            view._log(f"Configuration saved as table '{tbl}' in {gpkg}")

        def do_json_export(json_path, ws):
            import json
            payload = self._build_replay_payload(
                widget_state=ws,
                mesh_name=mesh_name,
                run_duration_s=run_dur,
                mesh_gpkg_path=db_path,
                run_id=json_path.rsplit("/", 1)[-1].rsplit(".", 1)[0],
            )
            try:
                with open(json_path, "w") as f:
                    json.dump(payload, f, indent=2, default=str)
                view._log(f"Configuration exported to JSON: {json_path}")
            except Exception as exc:
                view._log(f"[ERROR] JSON export failed: {exc}")

        from swe2d.workbench.dialogs.save_config_dialog import SaveConfigDialog
        dlg = SaveConfigDialog(
            gpkg_path=db_path,
            existing_table_names=existing_tables,
            widget_state=widget_state,
            mesh_name=mesh_name,
            run_duration_s=run_dur,
            default_table_name=default_table,
            save_callback=do_persist,
            json_save_callback=do_json_export,
            parent=view,
        )
        dlg.exec()

    def on_save_simulation_config_as_json(self) -> None:
        """Export the current widget configuration directly to a JSON file.

        No GPKG interaction — pure JSON export for sharing, version control,
        or archival.
        """
        import json
        from qgis.PyQt import QtWidgets as _QtWidgets

        view = self._view
        path, _ = _QtWidgets.QFileDialog.getSaveFileName(
            view,
            "Export Configuration as JSON",
            "",
            "JSON Files (*.json);;All Files (*)",
        )
        if not path:
            return
        if not path.endswith(".json"):
            path += ".json"

        mesh_name = str((getattr(view, "_mesh_data", None) or {}).get("mesh_name", "") or "")
        widget_state = self.collect_widget_state_for_save()

        try:
            run_dur = view.model_tab.get_run_time_hours_parsed() * 3600.0
        except Exception:
            run_dur = 0.0

        payload = self._build_replay_payload(
            widget_state=widget_state,
            mesh_name=mesh_name,
            run_duration_s=run_dur,
            mesh_gpkg_path=str(getattr(view, "_model_gpkg_path", "") or ""),
        )
        try:
            with open(path, "w") as f:
                json.dump(payload, f, indent=2, default=str)
            view._log(f"Configuration exported to JSON: {path}")
        except Exception as exc:
            view.show_critical_message("Export failed", str(exc))

    def on_load_simulation_config_from_json(self) -> None:
        """Load a simulation configuration directly from a JSON file.

        No GPKG interaction — imports a previously-exported JSON file.
        """
        from qgis.PyQt import QtWidgets as _QtWidgets

        view = self._view
        path, _ = _QtWidgets.QFileDialog.getOpenFileName(
            view,
            "Load Configuration from JSON",
            "",
            "JSON Files (*.json);;All Files (*)",
        )
        if not path:
            return
        try:
            import json
            with open(path) as f:
                data = json.load(f)
        except Exception as exc:
            view.show_critical_message("Load failed", f"Could not read JSON:\n{exc}")
            return

        # widget_state is either at top level or nested
        ws = data.get("widget_state", data)
        if not isinstance(ws, dict):
            view.show_critical_message("Load failed", "JSON does not contain valid widget state.")
            return

        metadata = {"workbench_widget_state": ws}
        if hasattr(view, "_apply_run_log_metadata_to_ui"):
            try:
                restored = view._apply_run_log_metadata_to_ui(metadata)
                view._log(f"Loaded config from JSON '{path}': {int(restored)} widgets restored.")
            except Exception as exc:
                view.show_critical_message("Apply failed", str(exc))
        else:
            view._log(f"[WARNING] JSON loaded but apply callback not available.")

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
