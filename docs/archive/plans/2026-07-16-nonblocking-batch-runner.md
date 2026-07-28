---
type: plan
status: complete
created: 2026-07-16
completed: 2026-07-25
---

# Non-Blocking Batch Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the batch runner non-blocking so QGIS remains responsive during batch runs, with a modeless dialog, live progress, and per-sim log files on disk.

**Architecture:** A `BatchManager` service owns all batch state and exposes pyqtSignal-based notifications. A `BatchWorker(QThread)` runs the orchestration loop on a background thread, launching subprocesses and emitting progress. The `BatchSimulationDialog` becomes a modeless pure View that connects to `BatchManager` signals.

**Tech Stack:** PyQt5 (QThread, pyqtSignal, QObject), subprocess, threading.Event, concurrent.futures.ProcessPoolExecutor, NVIDIA MPS

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `swe2d/workbench/services/batch_manager.py` | **Create** | Service: state ownership, signal emission, MPS lifecycle, worker management |
| `swe2d/workbench/workers/batch_worker.py` | **Create** | QThread: subprocess orchestration, polling, log writing |
| `swe2d/workbench/dialogs/batch_simulation_dialog.py` | **Modify** | Modeless, pure View, remove orchestration logic, connect to BatchManager |
| `swe2d/workbench/controllers/run_controller.py` | **Modify** | Inject BatchManager, hold dialog instance, change `dlg.exec()` → `dlg.show()` |
| `tests/test_batch_manager.py` | **Create** | Unit tests for BatchManager |
| `tests/test_batch_worker.py` | **Create** | Unit tests for BatchWorker |

---

### Task 1: Create BatchManager service

**Files:**
- Create: `swe2d/workbench/services/batch_manager.py`
- Test: `tests/test_batch_manager.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_batch_manager.py
import threading
from unittest.mock import MagicMock, patch
from swe2d.workbench.services.batch_manager import BatchManager, SimState, BatchConfig


def test_batch_manager_initial_state():
    bm = BatchManager()
    assert not bm.is_running()
    assert bm.get_status() == {}


def test_start_batch_sets_running():
    bm = BatchManager()
    config = BatchConfig(max_workers=1, results_dir="/tmp/test", mesh_path="/tmp/mesh.gpkg")
    with patch("swe2d.workbench.services.batch_manager.BatchWorker") as MockWorker:
        mock_worker = MagicMock()
        MockWorker.return_value = mock_worker
        bm.start_batch([{"id": "sim_001", "params": {}}], config)
        assert bm.is_running()
        mock_worker.start.assert_called_once()


def test_cancel_batch_sets_event():
    bm = BatchManager()
    config = BatchConfig(max_workers=1, results_dir="/tmp/test", mesh_path="/tmp/mesh.gpkg")
    with patch("swe2d.workbench.services.batch_manager.BatchWorker") as MockWorker:
        mock_worker = MagicMock()
        MockWorker.return_value = mock_worker
        bm.start_batch([{"id": "sim_001", "params": {}}], config)
        bm.cancel_batch()
        assert bm._cancel_event.is_set()


def test_get_status_returns_sim_states():
    bm = BatchManager()
    bm._sim_states["sim_001"] = SimState(
        sim_id="sim_001", status="completed", progress=100.0,
        status_text="Done", results_path="/tmp/out.gpkg",
        error=None, log_file="/tmp/sim_001.log",
    )
    status = bm.get_status()
    assert "sim_001" in status
    assert status["sim_001"].status == "completed"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `mamba run -n qgis_stable python -m pytest tests/test_batch_manager.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'swe2d.workbench.services.batch_manager'`

- [ ] **Step 3: Write the BatchManager implementation**

```python
# swe2d/workbench/services/batch_manager.py
"""BatchManager: service-layer owner of batch execution state and lifecycle.

Follows MVP architecture: owns state, emits signals, no widget imports.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from PyQt5.QtCore import QObject, pyqtSignal

logger = logging.getLogger(__name__)


@dataclass
class SimState:
    """Per-simulation state tracked by the BatchManager."""
    sim_id: str
    status: str = "pending"       # "pending" | "running" | "completed" | "failed"
    progress: float = 0.0         # 0.0–100.0
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

    # ── Signal handlers (connected to worker signals) ──────────────────

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `mamba run -n qgis_stable python -m pytest tests/test_batch_manager.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add swe2d/workbench/services/batch_manager.py tests/test_batch_manager.py
git commit -m "feat(batch): add BatchManager service with state ownership and signals"
```

---

### Task 2: Create BatchWorker thread

**Files:**
- Create: `swe2d/workbench/workers/batch_worker.py`
- Test: `tests/test_batch_worker.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_batch_worker.py
import os
import tempfile
import threading
from unittest.mock import patch, MagicMock
from swe2d.workbench.workers.batch_worker import BatchWorker
from swe2d.workbench.services.batch_manager import BatchConfig


def test_batch_worker_sequential_mode():
    """Test that BatchWorker runs sims sequentially when max_workers=1."""
    config = BatchConfig(max_workers=1, results_dir=tempfile.mkdtemp(), mesh_path="/tmp/mesh.gpkg")
    cancel_event = threading.Event()
    params = [{"id": "sim_001", "params": {"n_mann": 0.03}}]

    worker = BatchWorker(params_list=params, config=config, cancel_event=cancel_event)

    with patch.object(worker, "_launch_sim") as mock_launch:
        mock_launch.return_value = None
        worker._run_sequential()
        assert mock_launch.call_count == 1
        mock_launch.assert_called_with(params[0])


def test_batch_worker_cancel_stops_loop():
    """Test that cancelling stops the sequential loop."""
    config = BatchConfig(max_workers=1, results_dir=tempfile.mkdtemp(), mesh_path="/tmp/mesh.gpkg")
    cancel_event = threading.Event()
    params = [
        {"id": "sim_001", "params": {}},
        {"id": "sim_002", "params": {}},
        {"id": "sim_003", "params": {}},
    ]

    worker = BatchWorker(params_list=params, config=config, cancel_event=cancel_event)

    call_count = 0

    def fake_launch(p):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            cancel_event.set()

    with patch.object(worker, "_launch_sim", side_effect=fake_launch):
        worker._run_sequential()
        assert call_count == 2  # sim_001 ran, sim_002 set cancel, sim_003 skipped


def test_batch_worker_creates_log_dir():
    """Test that _launch_sim creates the batch_runs log directory."""
    config = BatchConfig(max_workers=1, results_dir=tempfile.mkdtemp(), mesh_path="/tmp/mesh.gpkg")
    cancel_event = threading.Event()
    worker = BatchWorker(
        params_list=[{"id": "sim_001", "params": {}}],
        config=config, cancel_event=cancel_event,
    )

    log_dir = os.path.join(config.results_dir, "batch_runs")
    assert not os.path.exists(log_dir)

    with patch("subprocess.Popen") as MockPopen:
        mock_proc = MagicMock()
        mock_proc.stdout = iter([])
        mock_proc.returncode = 0
        mock_proc.poll.return_value = 0
        MockPopen.return_value = mock_proc

        worker._launch_sim({"id": "sim_001", "params": {}})
        assert os.path.isdir(log_dir)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `mamba run -n qgis_stable python -m pytest tests/test_batch_worker.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'swe2d.workbench.workers.batch_worker'`

- [ ] **Step 3: Write the BatchWorker implementation**

```python
# swe2d/workbench/workers/batch_worker.py
"""BatchWorker: QThread that orchestrates batch simulation subprocesses.

Runs on a background thread. Launches subprocesses, polls their status,
writes per-sim log files, and emits progress signals via BatchManager.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import time
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from PyQt5.QtCore import QThread, pyqtSignal

logger = logging.getLogger(__name__)


class BatchWorker(QThread):
    """Background worker that orchestrates batch simulation subprocesses.

    Emits signals that BatchManager forwards to the UI layer.
    """

    _sim_started = pyqtSignal(str)                     # sim_id
    _sim_progress = pyqtSignal(str, float, str)        # sim_id, percent, status_text
    _sim_completed = pyqtSignal(str, str)              # sim_id, results_path
    _sim_failed = pyqtSignal(str, str)                 # sim_id, error_message
    _log_line = pyqtSignal(str, str)                   # sim_id, line
    _batch_done = pyqtSignal(bool, int, int)           # cancelled, completed, failed

    def __init__(
        self,
        params_list: List[Dict[str, Any]],
        config: Any,  # BatchConfig
        cancel_event: threading.Event,
        parent: Optional[QThread] = None,
    ):
        super().__init__(parent)
        self._params_list = params_list
        self._config = config
        self._cancel_event = cancel_event
        self._completed = 0
        self._failed = 0

    def run(self):
        """Main orchestration loop. Runs on background thread."""
        try:
            os.makedirs(os.path.join(self._config.results_dir, "batch_runs"), exist_ok=True)

            if self._config.max_workers > 1:
                self._run_parallel()
            else:
                self._run_sequential()
        except Exception as exc:
            logger.exception("BatchWorker failed")
        finally:
            self._batch_done.emit(
                self._cancel_event.is_set(),
                self._completed,
                self._failed,
            )

    def request_cancel(self):
        """Signal the worker to stop launching new sims."""
        self._cancel_event.set()

    def _run_sequential(self):
        """Launch each sim as subprocess.Popen, poll status, emit progress."""
        for params in self._params_list:
            if self._cancel_event.is_set():
                break
            self._launch_sim(params)

    def _run_parallel(self):
        """Use ProcessPoolExecutor with MPS for concurrent GPU execution."""
        from swe2d.cli.batch_runner import _ensure_mps, _stop_mps_if_we_started

        mps_started = _ensure_mps()
        try:
            with ProcessPoolExecutor(max_workers=self._config.max_workers) as pool:
                futures = {}
                for params in self._params_list:
                    if self._cancel_event.is_set():
                        break
                    future = pool.submit(self._launch_sim, params)
                    futures[future] = params.get("id", "unknown")

                for future in as_completed(futures):
                    sim_id = futures[future]
                    try:
                        future.result()
                    except Exception as exc:
                        self._sim_failed.emit(sim_id, str(exc))
                        self._failed += 1
        finally:
            _stop_mps_if_we_started(mps_started)

    def _launch_sim(self, params: Dict[str, Any]) -> None:
        """Launch a single sim subprocess, poll its status file, write log."""
        sim_id = str(params.get("id", "unknown"))
        self._sim_started.emit(sim_id)

        log_dir = os.path.join(self._config.results_dir, "batch_runs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"{sim_id}.log")

        # Build the CLI command
        if params.get("schema_version") == "swe2d-replay/1":
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False,
            ) as tf:
                json.dump(params, tf)
                replay_file = tf.name
            try:
                cmd = [
                    sys.executable, "-m", "swe2d.cli", "replay",
                    "--replay-file", replay_file,
                ]
            except Exception:
                cmd = []
        else:
            params_json = json.dumps(params)
            cmd = [
                sys.executable, "-m", "swe2d.cli", "run",
                self._config.mesh_path, params_json,
                "--results", os.path.join(
                    self._config.results_dir, "batch_results.gpkg"
                ),
            ]

        if not cmd:
            self._sim_failed.emit(sim_id, "Failed to build command")
            self._failed += 1
            return

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            self._sim_failed.emit(sim_id, f"Failed to launch: {exc}")
            self._failed += 1
            return

        # Stream stdout to log file and emit log lines
        try:
            with open(log_path, "w") as log_file:
                for line in proc.stdout:
                    log_file.write(line)
                    log_file.flush()
                    self._log_line.emit(sim_id, line.rstrip("\n"))

                proc.wait(timeout=self._config.timeout_per_sim)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            self._sim_failed.emit(sim_id, "Timed out")
            self._failed += 1
            return
        except Exception as exc:
            proc.kill()
            proc.wait()
            self._sim_failed.emit(sim_id, f"Error: {exc}")
            self._failed += 1
            return

        # Determine results path
        results_path = os.path.join(
            self._config.results_dir, "batch_results.gpkg"
        )

        if proc.returncode == 0:
            self._sim_completed.emit(sim_id, results_path)
            self._completed += 1
        else:
            self._sim_failed.emit(sim_id, f"Exit code {proc.returncode}")
            self._failed += 1
```

- [ ] **Step 4: Run test to verify it passes**

Run: `mamba run -n qgis_stable python -m pytest tests/test_batch_worker.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add swe2d/workbench/workers/batch_worker.py tests/test_batch_worker.py
git commit -m "feat(batch): add BatchWorker QThread for subprocess orchestration"
```

---

### Task 3: Make BatchSimulationDialog modeless and MVP-compliant

**Files:**
- Modify: `swe2d/workbench/dialogs/batch_simulation_dialog.py`

This is the largest task. The dialog must be refactored from a modal blocking dialog to a modeless pure View.

- [ ] **Step 1: Add BatchManager import and constructor parameter**

In `swe2d/workbench/dialogs/batch_simulation_dialog.py`, modify the `BatchSimulationDialog.__init__` to accept a `batch_manager` parameter:

```python
# At the top of the file, add to imports:
from swe2d.workbench.services.batch_manager import BatchManager, SimState

# Modify __init__ signature:
class BatchSimulationDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, base_params=None, mesh_gpkg="",
                 batch_manager=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Simulation")
        self.resize(900, 500)

        self._base_params = dict(base_params or {})
        self._mesh_gpkg = str(mesh_gpkg)
        self._mesh_name = ""
        self._param_sets = []
        self._batch_manager = batch_manager  # NEW: injected service

        # REMOVE: self._processes, self._status_files, self._next_idx,
        # self._active, self._completed, self._failed, self._running,
        # self._orchestrator

        self._build_ui()
        # ... rest of init
```

- [ ] **Step 2: Add closeEvent handler for modeless behavior**

```python
    def closeEvent(self, event):
        """Hide instead of destroy if batch is running."""
        if self._batch_manager and self._batch_manager.is_running():
            event.ignore()
            self.hide()
        else:
            event.accept()
```

- [ ] **Step 3: Remove orchestration logic from _run_batch**

Replace the entire `_run_batch` method. Remove the `BatchOrchestrator` import and all subprocess management. The method should collect param sets, build a `BatchConfig`, and call `self._batch_manager.start_batch()`:

```python
    def _run_batch(self):
        if self._batch_manager and self._batch_manager.is_running():
            return
        param_sets = self._collect_param_sets()
        if not param_sets:
            QtWidgets.QMessageBox.information(self, "Batch Run", "No parameter sets defined.")
            return
        gpkg = self._gpkg_path()
        if not gpkg or not os.path.isfile(gpkg):
            QtWidgets.QMessageBox.warning(self, "Batch Run", "GeoPackage not found.")
            return

        import tempfile
        from swe2d.workbench.services.batch_manager import BatchConfig

        config = BatchConfig(
            max_workers=self._max_workers_spin.value(),
            results_dir=tempfile.mkdtemp(prefix="hydra_batch_"),
            mesh_path=gpkg,
        )

        # Mark all rows as pending
        for i in range(len(param_sets)):
            item = self._table.item(i, _COL_STATUS)
            if item:
                item.setText("pending")

        self._batch_manager.start_batch(param_sets, config)
        self._run_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
```

- [ ] **Step 4: Add signal connection methods**

Add methods to connect/disconnect to BatchManager signals:

```python
    def connect_batch_signals(self):
        """Connect to BatchManager signals for live updates."""
        bm = self._batch_manager
        if not bm:
            return
        bm.sim_started.connect(self._on_sim_started)
        bm.sim_progress.connect(self._on_sim_progress)
        bm.sim_completed.connect(self._on_sim_completed)
        bm.sim_failed.connect(self._on_sim_failed)
        bm.log_message.connect(self._on_log_message)
        bm.batch_finished.connect(self._on_batch_finished)

    def disconnect_batch_signals(self):
        """Disconnect from BatchManager signals."""
        bm = self._batch_manager
        if not bm:
            return
        try:
            bm.sim_started.disconnect(self._on_sim_started)
            bm.sim_progress.disconnect(self._on_sim_progress)
            bm.sim_completed.disconnect(self._on_sim_completed)
            bm.sim_failed.disconnect(self._on_sim_failed)
            bm.log_message.disconnect(self._on_log_message)
            bm.batch_finished.disconnect(self._on_batch_finished)
        except (TypeError, RuntimeError):
            pass
```

- [ ] **Step 5: Add signal handler methods**

```python
    def _find_row_by_sim_id(self, sim_id: str) -> int:
        """Find the table row for a given sim_id."""
        for i in range(self._table.rowCount()):
            item = self._table.item(i, _COL_SIM_ID)
            if item and item.text() == sim_id:
                return i
        return -1

    def _on_sim_started(self, sim_id: str):
        row = self._find_row_by_sim_id(sim_id)
        if row >= 0:
            item = self._table.item(row, _COL_STATUS)
            if item:
                item.setText("running")

    def _on_sim_progress(self, sim_id: str, percent: float, status_text: str):
        row = self._find_row_by_sim_id(sim_id)
        if row >= 0:
            progress_item = self._table.item(row, _COL_PROGRESS)
            if progress_item:
                progress_item.setText(f"{percent:.0f}% {status_text}")

    def _on_sim_completed(self, sim_id: str, results_path: str):
        row = self._find_row_by_sim_id(sim_id)
        if row >= 0:
            item = self._table.item(row, _COL_STATUS)
            if item:
                item.setText("completed")
            progress_item = self._table.item(row, _COL_PROGRESS)
            if progress_item:
                progress_item.setText("100%")

    def _on_sim_failed(self, sim_id: str, error: str):
        row = self._find_row_by_sim_id(sim_id)
        if row >= 0:
            item = self._table.item(row, _COL_STATUS)
            if item:
                item.setText("failed")
            progress_item = self._table.item(row, _COL_PROGRESS)
            if progress_item:
                progress_item.setText(error[:100])

    def _on_log_message(self, sim_id: str, line: str):
        """Append log line to the main workbench log."""
        parent_log = getattr(self.parent(), "_log", None)
        if parent_log:
            parent_log(f"batch> [{sim_id}] {line}")

    def _on_batch_finished(self, cancelled: bool, completed: int, failed: int):
        self._run_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        total = completed + failed
        msg = f"Completed: {completed}/{total}\nFailed: {failed}"
        if cancelled:
            msg = "(Cancelled)\n" + msg
        QtWidgets.QMessageBox.information(self, "Batch Complete", msg)
```

- [ ] **Step 6: Update _cancel_batch to use BatchManager**

```python
    def _cancel_batch(self):
        if self._batch_manager:
            self._batch_manager.cancel_batch()
```

- [ ] **Step 7: Remove _check_batch_status method**

Delete the `_check_batch_status` method entirely (status is now live via signals).

- [ ] **Step 8: Add showEvent to reconnect signals and refresh state**

```python
    def showEvent(self, event):
        """When dialog is shown, reconnect signals and refresh table."""
        super().showEvent(event)
        self.connect_batch_signals()
        self._refresh_table_from_manager()

    def hideEvent(self, event):
        """When dialog is hidden, disconnect signals."""
        super().hideEvent(event)
        self.disconnect_batch_signals()

    def _refresh_table_from_manager(self):
        """Populate table from BatchManager state (for reopening)."""
        if not self._batch_manager:
            return
        status = self._batch_manager.get_status()
        for sim_id, state in status.items():
            row = self._find_row_by_sim_id(sim_id)
            if row >= 0:
                status_item = self._table.item(row, _COL_STATUS)
                if status_item:
                    status_item.setText(state.status)
                progress_item = self._table.item(row, _COL_PROGRESS)
                if progress_item:
                    if state.status == "completed":
                        progress_item.setText("100%")
                    elif state.status == "failed":
                        progress_item.setText((state.error or "")[:100])
                    elif state.status == "running":
                        progress_item.setText(f"{state.progress:.0f}% {state.status_text}")
```

- [ ] **Step 9: Run lint/typecheck**

Run: `mamba run -n qgis_stable python -m py_compile swe2d/workbench/dialogs/batch_simulation_dialog.py`
Expected: No errors

- [ ] **Step 10: Commit**

```bash
git add swe2d/workbench/dialogs/batch_simulation_dialog.py
git commit -m "refactor(batch): make BatchSimulationDialog modeless and MVP-compliant"
```

---

### Task 4: Wire BatchManager into RunController

**Files:**
- Modify: `swe2d/workbench/controllers/run_controller.py`

- [ ] **Step 1: Add BatchManager to RunController.__init__**

Find the `RunController.__init__` method and add BatchManager initialization:

```python
    def __init__(self, view):
        # ... existing init code ...
        self._batch_manager = None  # Lazy init
        self._batch_dialog = None   # Lazy init
```

- [ ] **Step 2: Add lazy BatchManager property**

```python
    @property
    def batch_manager(self):
        """Lazy-initialize the BatchManager singleton."""
        if self._batch_manager is None:
            from swe2d.workbench.services.batch_manager import BatchManager
            self._batch_manager = BatchManager()
        return self._batch_manager
```

- [ ] **Step 3: Rewrite open_batch_simulation_dialog**

Replace the existing method to use modeless dialog and inject BatchManager:

```python
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
```

- [ ] **Step 4: Run lint/typecheck**

Run: `mamba run -n qgis_stable python -m py_compile swe2d/workbench/controllers/run_controller.py`
Expected: No errors

- [ ] **Step 5: Commit**

```bash
git add swe2d/workbench/controllers/run_controller.py
git commit -m "feat(batch): wire BatchManager into RunController, make dialog modeless"
```

---

### Task 5: Verify MVP boundaries

**Files:** None (verification only)

- [ ] **Step 1: Run MVP boundary checks**

```bash
# 1. No Qt widgets in service layer
! grep -q 'QPushButton\|QComboBox\|QDockWidget\|QTableWidget\|setEnabled\|setText\|setValue' \
  swe2d/workbench/services/batch_manager.py && echo "PASS: batch_manager is widget-free"

# 2. No orchestration in dialog (no BatchOrchestrator import)
! grep -q 'BatchOrchestrator' \
  swe2d/workbench/dialogs/batch_simulation_dialog.py && echo "PASS: no orchestrator in dialog"

# 3. No direct subprocess in dialog
! grep -q 'subprocess\.Popen\|subprocess\.run' \
  swe2d/workbench/dialogs/batch_simulation_dialog.py && echo "PASS: no subprocess in dialog"

# 4. No numpy in dialog
! grep -q 'np\.min\|np\.max\|np\.vstack\|np\.argmin\|np\.where' \
  swe2d/workbench/dialogs/batch_simulation_dialog.py 2>/dev/null && echo "PASS: no numpy in dialog"
```

- [ ] **Step 2: Run existing tests to verify no regressions**

Run: `mamba run -n qgis_stable python -m pytest tests/ -v --timeout=60`
Expected: All tests pass

- [ ] **Step 3: Commit any fixups if needed**

```bash
git add -A
git commit -m "fix(batch): address MVP boundary violations"
```

---

### Task 6: Add log file panel to dialog

**Files:**
- Modify: `swe2d/workbench/dialogs/batch_simulation_dialog.py`

- [ ] **Step 1: Add LogDockWidget class**

Add at the bottom of `batch_simulation_dialog.py`, before the end of the file:

```python
class LogDockWidget(QtWidgets.QDockWidget):
    """Dock widget that displays tail of a per-sim log file."""

    def __init__(self, sim_id: str, log_path: str, parent=None):
        super().__init__(f"Log: {sim_id}", parent)
        self._sim_id = sim_id
        self._log_path = log_path
        self._log_view = QtWidgets.QPlainTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setMaximumBlockCount(500)
        font = QtWidgets.QFont("Monospace")
        font.setStyleHint(QtWidgets.QFont.TypeWriter)
        self._log_view.setFont(font)
        self.setWidget(self._log_view)

        self._refresh_timer = QtCore.QTimer(self)
        self._refresh_timer.timeout.connect(self._refresh_log)
        self._last_pos = 0

    def start_auto_refresh(self):
        """Start auto-refreshing the log tail every 1s."""
        self._last_pos = 0
        self._refresh_log()
        self._refresh_timer.start(1000)

    def stop_auto_refresh(self):
        self._refresh_timer.stop()

    def _refresh_log(self):
        """Read new lines from the log file and append."""
        try:
            if not os.path.isfile(self._log_path):
                return
            with open(self._log_path, "r") as f:
                f.seek(self._last_pos)
                new_lines = f.readlines()
                self._last_pos = f.tell()
            for line in new_lines:
                self._log_view.appendPlainText(line.rstrip("\n"))
        except Exception:
            pass

    def load_full_log(self):
        """Load the entire log file (for completed/failed sims)."""
        try:
            if not os.path.isfile(self._log_path):
                self._log_view.setPlainText("(no log file found)")
                return
            with open(self._log_path, "r") as f:
                self._log_view.setPlainText(f.read())
        except Exception as e:
            self._log_view.setPlainText(f"(error reading log: {e})")
```

- [ ] **Step 2: Add _open_log_panel method to BatchSimulationDialog**

```python
    def _open_log_panel(self):
        """Open a log dock widget for the selected row."""
        rows = set(i.row() for i in self._table.selectedIndexes())
        if not rows:
            return
        row = min(rows)
        sim_id_item = self._table.item(row, _COL_SIM_ID)
        if not sim_id_item:
            return
        sim_id = sim_id_item.text()

        bm = self._batch_manager
        if not bm:
            return
        status = bm.get_status()
        state = status.get(sim_id)
        if not state or not state.log_file:
            return

        dock = LogDockWidget(sim_id, state.log_file, parent=self)
        if state.status in ("completed", "failed"):
            dock.load_full_log()
        else:
            dock.start_auto_refresh()
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, dock)
        dock.show()
```

- [ ] **Step 3: Add View button to table or a toolbar button**

In the `_build_ui` method, add a "View Log" button to the toolbar:

```python
        self._view_log_btn = QtWidgets.QPushButton("View Log")
        self._view_log_btn.setToolTip("View log for the selected simulation")
        self._view_log_btn.clicked.connect(self._open_log_panel)
        # Add to toolbar after _edit_sel_btn
        toolbar.addWidget(self._view_log_btn)
```

- [ ] **Step 4: Commit**

```bash
git add swe2d/workbench/dialogs/batch_simulation_dialog.py
git commit -m "feat(batch): add log panel with auto-refresh for running sims"
```

---

### Task 7: Integration smoke test

**Files:** None (manual verification)

- [ ] **Step 1: Clear Python cache**

```bash
find . -type d -name __pycache__ -exec rm -rf {} +
```

- [ ] **Step 2: Run full test suite**

Run: `mamba run -n qgis_stable python -m pytest tests/ -v --timeout=120`
Expected: All tests pass

- [ ] **Step 3: Manual verification checklist**

In QGIS, verify:
1. Click "Batch..." button — dialog opens as modeless (can interact with QGIS behind it)
2. Add param rows, click "Run Batch" — QGIS remains responsive
3. Close dialog while batch running — dialog hides, batch continues
4. Reopen "Batch..." — dialog shows current progress
5. Click "View Log" — log panel shows live output for running sim
6. Click "Cancel" — batch stops, next sim not launched
7. Let batch finish — summary QMessageBox appears

- [ ] **Step 4: Final commit if any fixups needed**

```bash
git add -A
git commit -m "fix(batch): integration test fixups"
```
