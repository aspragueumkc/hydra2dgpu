"""Headless executor: calls SimulationWorker._execute() without QThread.

Mimics the Qt-signal interface of SimulationWorker so the existing
_execute() method runs unmodified outside a QGIS event loop.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class _HeadlessSignal:
    """Drop-in for a pyqtSignal — .emit() is a direct function call."""

    def __init__(self, callback: Any = None) -> None:
        self._callback = callback

    def emit(self, *args: Any) -> None:
        if self._callback is not None:
            self._callback(*args)

    def connect(self, callback: Any) -> None:
        self._callback = callback


class HeadlessWorkerAdapter:
    """Mimics the parts of SimulationWorker that _execute() touches."""

    def __init__(
        self,
        ctx: Any,
        log_cb: Any = None,
        progress_cb: Any = None,
        snapshot_cb: Any = None,
    ) -> None:
        self._context = ctx
        self._runtime_reporter: Any = None

        self.log_message = _HeadlessSignal(log_cb or print)
        self.progress_percent = _HeadlessSignal(progress_cb)
        self.snapshot_ready = _HeadlessSignal(snapshot_cb)
        self.compute_finished = _HeadlessSignal()
        self.compute_failed = _HeadlessSignal()
        self.mesh_permutation_ready = _HeadlessSignal()

    def receivers(self, _signal: Any) -> int:
        return 0

    def request_snapshot(self) -> None:
        pass

    def request_cancel(self) -> None:
        self._context.cancel_event.set()


def execute_swe2d_headless(
    ctx: Any,
    log_cb: Any = None,
    progress_cb: Any = None,
    snapshot_cb: Any = None,
) -> Any:
    """Run SimulationWorker._execute() headless using the adapter.

    Returns the ComputeResult that _execute() produces.  Also calls
    SWE2DRunFinalizer to persist results to GPKG when ctx.results_gpkg_path
    is set (mirroring the GUI on_worker_compute_finished handler).
    """
    from swe2d.workbench.workers.simulation_worker import (
        SimulationWorker,
    )

    adapter = HeadlessWorkerAdapter(
        ctx,
        log_cb=log_cb,
        progress_cb=progress_cb,
        snapshot_cb=snapshot_cb,
    )
    result = SimulationWorker._execute(adapter)

    # ── GPKG persistence (normally handled by GUI signal handler) ────
    results_gpkg = str(getattr(ctx, "results_gpkg_path", "") or "")
    if results_gpkg and result.ok:
        try:
            from swe2d.runtime.run_finalizer import SWE2DRunFinalizer

            # Capture results_data populated during _execute() for pipe-cell / overlay persistence
            results_data = getattr(adapter, "_results_data", None)
            fv = _HeadlessFinalizationView(results_gpkg, log_cb=log_cb, results_data=results_data)
            finalizer = SWE2DRunFinalizer(fv)
            finalizer.finalize_and_persist(
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
            )
            for msg in finalizer.drain_log_messages():
                if log_cb:
                    log_cb(msg)

            # ── Save simulation config to GPKG ────────────────────────
            try:
                from swe2d.services.gpkg_persistence_service import persist_simulation_config
                persist_simulation_config(
                    gpkg_path=results_gpkg,
                    config_id=result.run_id,
                    mesh_name=str(getattr(ctx, "mesh_name", "")),
                    run_duration_s=result.run_duration_s,
                    widget_state=_serialize_ctx_for_config(ctx),
                    description=f"CLI run: {result.run_id}",
                    log_fn=log_cb,
                )
            except Exception as exc:
                logger.warning("Failed to persist sim config: %s", exc)
        except Exception as exc:
            logger.warning("Headless persistence failed: %s", exc)
            if log_cb:
                log_cb(f"[ERROR] Headless persistence failed: {exc}")

    return result


class _HeadlessFinalizationView:
    """Minimal view adapter for SWE2DRunFinalizer headless persistence."""

    def __init__(self, results_gpkg: str, log_cb: Any = None, results_data: Any = None):
        self._results_gpkg = results_gpkg
        self._log_cb = log_cb
        self._log_lines: list = []
        self._results_data = results_data

    def log_message(self, msg: str) -> None:
        if self._log_cb:
            self._log_cb(msg)
        self._log_lines.append(msg)

    def get_line_results_storage_path(self) -> str:
        return self._results_gpkg

    def sync_overlay_data(self) -> None:
        pass

    def refresh_plot(self) -> None:
        pass

    def results_table_name(self, base: str) -> str:
        return base

    def length_unit_name(self) -> str:
        return "m"

    def length_scale_si_to_model(self) -> float:
        return 1.0

    def update_overlay_time(self, t: float) -> None:
        pass

    def runtime_log_lines(self) -> list:
        return self._log_lines

    def collect_run_log_metadata(self) -> dict:
        return {}

    def persist_run_log(self, gpkg_path: str, run_id: str,
                        run_wallclock_start: str, run_wallclock_end: str,
                        run_duration_wallclock_s: float, run_log_text: str,
                        *, metadata: dict) -> None:
        from swe2d.results.run_log_storage import persist_run_log_to_geopackage
        persist_run_log_to_geopackage(
            gpkg_path=gpkg_path, run_id=run_id,
            start_wallclock=run_wallclock_start,
            end_wallclock=run_wallclock_end,
            duration_s=run_duration_wallclock_s,
            log_text=run_log_text, metadata=metadata,
        )

    def is_cancel_requested(self) -> bool:
        return False

    def results_data(self):
        return self._results_data


def _serialize_ctx_for_config(ctx) -> dict:
    """Extract a flat widget_state dict from a RunContext for config persistence.

    Filters out large array fields and callables, keeping only scalar params.
    """
    state: dict = {}
    for key in sorted(dir(ctx)):
        if key.startswith("_"):
            continue
        val = getattr(ctx, key, None)
        # Skip arrays, callables, None, and complex objects
        if val is None:
            continue
        if isinstance(val, (np.ndarray, list)):
            continue
        if callable(val):
            continue
        if not isinstance(val, (str, int, float, bool)):
            continue
        state[key] = val
    return state
