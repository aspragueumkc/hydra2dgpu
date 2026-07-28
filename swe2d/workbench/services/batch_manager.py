"""BatchManager: service-layer owner of batch execution state and lifecycle.

Follows MVP architecture: owns state, emits signals, no widget imports.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from qgis.PyQt.QtCore import QObject, pyqtSignal

logger = logging.getLogger(__name__)


@dataclass
class SimState:
    """Per-simulation state tracked by the BatchManager."""
    sim_id: str
    status: str = "pending"       # "pending" | "running" | "completed" | "failed"
    progress: float = 0.0         # 0.0-100.0
    status_text: str = ""         # e.g. "Step 450/1000"
    results_path: Optional[str] = None
    error: Optional[str] = None
    log_file: Optional[str] = None


@dataclass
class BatchConfig:
    """Configuration for a batch run."""
    max_workers: int
    results_dir: str
    mesh_path: str
    timeout_per_sim: Optional[int] = None  # seconds, None = no timeout


class BatchManager(QObject):
    """Service-layer owner of batch execution state.

    Creates and manages a BatchWorker thread. Emits signals for progress
    and lifecycle events. Does NOT import Qt widgets.
    """

    batch_started = pyqtSignal(int)                     # total_sims
    batch_finished = pyqtSignal(bool, int, int)         # cancelled, completed, failed
    sim_started = pyqtSignal(str)                        # sim_id
    sim_progress = pyqtSignal(str, float, str)          # sim_id, percent, status_text
    sim_completed = pyqtSignal(str, str)                # sim_id, results_path
    sim_failed = pyqtSignal(str, str)                   # sim_id, error_message
    log_message = pyqtSignal(str, str)                  # sim_id, line

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._worker = None  # Optional[BatchWorker]
        self._sim_states: Dict[str, SimState] = {}
        self._batch_config: Optional[BatchConfig] = None
        self._is_running = False
        self._cancel_event = threading.Event()
        self._completed_count = 0
        self._failed_count = 0

    def start_batch(
        self,
        params_list: List[Dict[str, Any]],
        config: BatchConfig,
    ) -> None:
        """Start a batch run. Config includes max_workers, results_dir, mesh_path."""
        if self._is_running:
            logger.warning("Batch already running, ignoring start_batch()")
            return

        from swe2d.workbench.workers.batch_worker import BatchWorker

        self._batch_config = config
        self._cancel_event.clear()
        self._completed_count = 0
        self._failed_count = 0

        # Initialize sim states
        self._sim_states = {}
        for ps in params_list:
            sim_id = str(ps.get("id", f"sim_{len(self._sim_states)}"))
            log_file = f"{config.results_dir}/batch_runs/{sim_id}.log"
            self._sim_states[sim_id] = SimState(
                sim_id=sim_id, log_file=log_file,
            )

        # Create and connect worker
        self._worker = BatchWorker(
            params_list=params_list,
            config=config,
            cancel_event=self._cancel_event,
        )
        self._worker._sim_started.connect(self._on_sim_started)
        self._worker._sim_progress.connect(self._on_sim_progress)
        self._worker._sim_completed.connect(self._on_sim_completed)
        self._worker._sim_failed.connect(self._on_sim_failed)
        self._worker._log_line.connect(self._on_log_line)
        self._worker._batch_done.connect(self._on_batch_done)
        self._worker.finished.connect(self._on_worker_finished)

        self._is_running = True
        self._worker.start()
        self.batch_started.emit(len(params_list))

    def cancel_batch(self) -> None:
        """Cancel the running batch. In-progress sim completes, no new sims launched."""
        if not self._is_running:
            return
        self._cancel_event.set()
        if self._worker is not None:
            self._worker.request_cancel()

    def get_status(self) -> Dict[str, SimState]:
        """Return current state of all sims. Used when dialog reopens."""
        return dict(self._sim_states)

    def is_running(self) -> bool:
        """Whether a batch is currently executing."""
        return self._is_running

    # ------------------------------------------------------------------
    # Signal handlers (connected to worker signals)
    # ------------------------------------------------------------------

    def _on_sim_started(self, sim_id: str) -> None:
        state = self._sim_states.get(sim_id)
        if state:
            state.status = "running"
            state.progress = 0.0
        self.sim_started.emit(sim_id)

    def _on_sim_progress(self, sim_id: str, percent: float, status_text: str) -> None:
        state = self._sim_states.get(sim_id)
        if state:
            state.progress = percent
            state.status_text = status_text
        self.sim_progress.emit(sim_id, percent, status_text)

    def _on_sim_completed(self, sim_id: str, results_path: str) -> None:
        state = self._sim_states.get(sim_id)
        if state:
            state.status = "completed"
            state.progress = 100.0
            state.results_path = results_path
        self._completed_count += 1
        self.sim_completed.emit(sim_id, results_path)

    def _on_sim_failed(self, sim_id: str, error: str) -> None:
        state = self._sim_states.get(sim_id)
        if state:
            state.status = "failed"
            state.error = error
        self._failed_count += 1
        self.sim_failed.emit(sim_id, error)

    def _on_log_line(self, sim_id: str, line: str) -> None:
        self.log_message.emit(sim_id, line)

    def _on_batch_done(self, cancelled: bool, completed: int, failed: int) -> None:
        self._is_running = False
        self.batch_finished.emit(cancelled, completed, failed)

    def _on_worker_finished(self) -> None:
        """Called when the QThread finishes (regardless of success)."""
        self._is_running = False
        self._worker = None
