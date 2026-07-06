from __future__ import annotations

import traceback
from typing import Any

from qgis.PyQt.QtCore import QThread, pyqtSignal

from swe2d.runtime.run_finalizer import SWE2DRunFinalizer
from swe2d.workbench.workers.simulation_worker import ComputeResult


class PersistenceWorker(QThread):
    """Background worker that persists simulation results to GeoPackage."""

    log_message = pyqtSignal(str)
    persist_finished = pyqtSignal(object)
    persist_failed = pyqtSignal(str)

    def __init__(self, view, result: ComputeResult, parent=None):
        super().__init__(parent)
        self._view = view
        self._result = result

    def run(self):
        try:
            finalizer = SWE2DRunFinalizer(self._view)
            status = finalizer.finalize_and_persist(
                h=self._result.h,
                hu=self._result.hu,
                hv=self._result.hv,
                final_sim_time_s=self._result.final_sim_time_s,
                n_area=self._result.n_area,
                area_model=self._result.area_model,
                storage_start_model=self._result.storage_start_model,
                source_budget_model=self._result.source_budget_model,
                source_step_rows_model=self._result.source_step_rows_model,
                run_duration_s=self._result.run_duration_s,
                boundary_flux_budget_model=self._result.boundary_flux_budget_model,
                boundary_flux_step_rows_model=self._result.boundary_flux_step_rows_model,
                run_id=self._result.run_id,
                output_interval_s=self._result.output_interval_s,
                run_perf_start=self._result.run_perf_start,
                run_wallclock_start=self._result.run_wallclock_start,
                run_log_start_idx=self._result.run_log_start_idx,
                thiessen_forcing=self._result.thiessen_forcing,
                rain_stats_acc=self._result.rain_stats_acc,
                save_line_results=False,
                save_coupling_results=False,
                save_mesh_results=False,
                save_run_log=False,
                h_min=1.0e-4,
                mesh_name=self._result.mesh_name,
                max_tracking=self._result.max_tracking,
                snapshot_timesteps=self._result.snapshot_timesteps,
                coupling_snapshots=self._result.coupling_snapshots,
                precomputed_line_results=self._result.precomputed_line_results,
            )
            for msg in finalizer.drain_log_messages():
                self.log_message.emit(msg)
            self.persist_finished.emit(status)
        except Exception as exc:
            self.log_message.emit(f"[ERROR] Persistence worker failed: {exc}")
            self.log_message.emit(traceback.format_exc())
            self.persist_failed.emit(str(exc))
