"""Background simulation worker that wraps the core executor with Qt signals."""

from __future__ import annotations

import logging
import threading
import traceback
from typing import Any, Optional

import numpy as np
from qgis.PyQt.QtCore import QThread, pyqtSignal

from swe2d.core.executor import ComputeResult, SnapshotData, execute_run
from swe2d.core.run_context import RunContext
from swe2d.core.sink_protocol import PermutationResult

logger = logging.getLogger(__name__)


class _SignalEmittingSink:
    """Sink implementation that forwards executor callbacks to Qt signals."""

    def __init__(self, worker: "SimulationWorker"):
        self._worker = worker
        self.snapshot_request_event = threading.Event()
        self._backend = None  # set by backend_ready, read by GUI main thread

    def log(self, message: str) -> None:
        self._worker.log_message.emit(message)

    def progress(self, percent: float, diagnostics: dict) -> None:
        self._worker.progress_percent.emit(int(percent))

    def snapshot(self, fields: list) -> None:
        # fields = [timesteps, line_ts, line_profiles, coupling_data, pipe_cell_data]
        self._worker.snapshot_ready.emit(
            SnapshotData(
                timesteps=fields[0],
                line_ts=fields[1],
                line_profiles=fields[2],
                coupling_data=fields[3],
                pipe_cell_data=fields[4],
            )
        )

    def finished(self, result: dict) -> None:
        # The worker emits the full ComputeResult after execute_run returns.
        pass

    def failed(self, error: str) -> None:
        self._worker.compute_failed.emit(error)

    def permutation(self, cell_perm: np.ndarray, result: PermutationResult) -> None:
        if self._worker.receivers(self._worker.mesh_permutation_ready) > 0:
            self._worker.mesh_permutation_ready.emit(cell_perm, result)
            if not result.event.wait(timeout=60.0):
                result.error = "Timed out waiting for main-thread mesh permutation."
        else:
            result.event.set()

    def request_snapshot(self) -> None:
        self.snapshot_request_event.set()

    def backend_ready(self, backend) -> None:
        """Sink callback: active SWE2DBackend has been built + initialized.

        Called by ``execute_run`` once the C++ solver handle is ready.
        Stashed on the sink so the GUI main thread can read it via
        ``SimulationWorker.get_active_solver()`` when opening the GPU
        Direct Viewer (avoids round-tripping a heavy backend through Qt
        signals).
        """
        self._backend = backend


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
        self._sink: Optional[_SignalEmittingSink] = None

    def request_snapshot(self):
        """Request a snapshot readback on the next reporter step."""
        if self._sink is not None:
            self._sink.request_snapshot()

    def get_active_solver(self):
        """Return the live ``PySolver`` shared_ptr while a run is in
        progress, else ``None``.

        Used by the GPU Direct Viewer to wire the GL render path
        (zero-D2H) to the live solver so the device pointer can be
        registered with the GL texture.  The sink stashes ``_backend``
        via its ``backend_ready`` callback (called from the worker
        thread); we read it from there.
        """
        if not self.isRunning():
            return None
        sink = getattr(self, "_sink", None)
        if sink is None:
            return None
        backend = getattr(sink, "_backend", None)
        if backend is None:
            return None
        return backend._solver_h

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
        """Thin wrapper around the core executor with a signal-emitting sink."""
        self._sink = _SignalEmittingSink(self)
        return execute_run(self._context, self._sink)

