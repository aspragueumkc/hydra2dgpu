from __future__ import annotations

import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from qgis.PyQt.QtCore import QThread, pyqtSignal

from swe2d.workbench.workers.run_context import RunContext


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
        self.log_message.emit("Simulation worker started.")
        # Placeholder for backend init + loop.
        # Real implementation filled in Task 6.
        h = np.asarray(ctx.h0, dtype=np.float64).copy()
        hu = np.asarray(ctx.hu0, dtype=np.float64).copy()
        hv = np.asarray(ctx.hv0, dtype=np.float64).copy()
        self.progress_percent.emit(100)
        return ComputeResult(
            ok=True,
            h=h,
            hu=hu,
            hv=hv,
            final_sim_time_s=ctx.run_duration_s,
            n_area=int(ctx.cell_areas.size),
            area_model=np.asarray(ctx.cell_areas, dtype=np.float64).ravel(),
            storage_start_model=0.0,
            source_budget_model={"rain": 0.0, "cell": 0.0, "coupling": 0.0},
            source_step_rows_model=[],
            run_duration_s=ctx.run_duration_s,
            boundary_flux_budget_model={},
            boundary_flux_step_rows_model=[],
            run_id=ctx.run_id,
            mesh_name=ctx.mesh_name,
            output_interval_s=ctx.output_interval_s,
            line_output_interval_s=ctx.line_output_interval_s,
            run_perf_start=0.0,
            run_wallclock_start=ctx.run_wallclock_start,
            run_log_start_idx=ctx.run_log_start_idx,
            thiessen_forcing=ctx.thiessen_forcing,
            rain_stats_acc={"rain_mm": 0.0, "excess_mm": 0.0, "samples": 0},
            max_tracking=None,
            snapshot_timesteps=[],
            coupling_snapshots={},
            precomputed_line_results=None,
        )
