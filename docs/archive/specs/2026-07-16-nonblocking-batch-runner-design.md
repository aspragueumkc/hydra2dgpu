---
type: spec
status: complete
created: 2026-07-16
completed: 2026-07-25
---

# Non-Blocking Batch Runner Design

## Problem

The batch runner dialog (`BatchSimulationDialog`) calls `BatchOrchestrator.run()` directly on the main thread, completely blocking the QGIS event loop for the entire duration of the batch. This means QGIS is unusable while batch runs execute — the GUI freezes, no other operations can proceed, and the user must wait for all simulations to finish before doing anything else.

Additionally, the dialog violates MVP architecture: the View layer (`BatchSimulationDialog`) directly imports and manages `BatchOrchestrator` (a service/CLI class), bypassing the Controller layer.

## Goals

1. **Non-blocking execution** — batch runs execute in a background thread; QGIS remains fully responsive
2. **Modeless dialog** — the batch dialog can be closed and reopened freely while runs are in progress
3. **Live progress** — per-simulation progress bars and log tails visible when the dialog is open
4. **Log persistence** — per-simulation logs written to disk, survive QGIS restarts
5. **MVP compliance** — service layer owns state and orchestration; dialog is a pure View
6. **Sequential default, optional parallel** — sequential by default; parallel execution via `ProcessPoolExecutor` with NVIDIA MPS for GPU contention management

## Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────┐
│  View Layer: BatchSimulationDialog (modeless)           │
│  - Pure UI: table, buttons, log panel                   │
│  - Connects to BatchManager signals                     │
│  - Can close/reopen freely                              │
│  - Reads state from BatchManager on reopen              │
└───────────────┬─────────────────────────────────────────┘
                │ signals/slots
┌───────────────▼─────────────────────────────────────────┐
│  Service Layer: BatchManager (QObject)                  │
│  - Owns all batch state (queue, progress, cancel flags) │
│  - Creates/manages BatchWorker thread                   │
│  - Emits signals: batch_started, sim_progress, etc.     │
│  - Manages NVIDIA MPS lifecycle                          │
│  - NO widget imports                                    │
└───────────────┬─────────────────────────────────────────┘
                │ runs on background thread
┌───────────────▼─────────────────────────────────────────┐
│  Worker: BatchWorker(QThread)                           │
│  - Orchestration loop: launches subprocesses            │
│  - Polls status JSON files (0.5s interval)              │
│  - Writes per-sim log files to disk                     │
│  - Emits progress signals via BatchManager              │
│  - Supports sequential or parallel (ProcessPoolExecutor)│
│  - Cancellable via threading.Event                      │
└─────────────────────────────────────────────────────────┘
```

### Files

| File | Action | Purpose |
|------|--------|---------|
| `swe2d/workbench/services/batch_manager.py` | **New** | Service layer: state ownership, signal emission, MPS lifecycle |
| `swe2d/workbench/workers/batch_worker.py` | **New** | QThread: subprocess orchestration, polling, log writing |
| `swe2d/workbench/dialogs/batch_simulation_dialog.py` | **Modify** | Modeless, pure View, connect to BatchManager signals |
| `swe2d/workbench/controllers/run_controller.py` | **Modify** | Inject BatchManager, change `dlg.exec()` to `dlg.show()` |
| `swe2d/cli/batch_runner.py` | **Modify** | Extract reusable logic into BatchManager/BatchWorker (keep CLI path working) |

## Detailed Design

### BatchManager Service

```python
# swe2d/workbench/services/batch_manager.py

class BatchManager(QObject):
    # Signals
    batch_started = pyqtSignal(int)                    # total_sims
    batch_finished = pyqtSignal(bool, int, int)        # cancelled, completed, failed
    sim_started = pyqtSignal(str)                       # sim_id
    sim_progress = pyqtSignal(str, float, str)         # sim_id, percent, status_text
    sim_completed = pyqtSignal(str, str)               # sim_id, results_path
    sim_failed = pyqtSignal(str, str)                  # sim_id, error_message
    log_message = pyqtSignal(str, str)                 # sim_id, line

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: Optional[BatchWorker] = None
        self._sim_states: Dict[str, SimState] = {}     # sim_id → SimState
        self._batch_config: Optional[BatchConfig] = None
        self._is_running = False

    def start_batch(self, params_list: List[dict], config: BatchConfig) -> None:
        """Start a batch run. Config includes max_workers, results_dir, mesh_path."""
        ...

    def cancel_batch(self) -> None:
        """Cancel the running batch. In-progress sim completes, no new sims launched."""
        ...

    def get_status(self) -> Dict[str, SimState]:
        """Return current state of all sims. Used when dialog reopens."""
        ...

    def is_running(self) -> bool:
        """Whether a batch is currently executing."""
        ...

@dataclass
class SimState:
    sim_id: str
    status: str          # "pending" | "running" | "completed" | "failed"
    progress: float      # 0.0–100.0
    status_text: str     # e.g. "Step 450/1000"
    results_path: Optional[str]
    error: Optional[str]
    log_file: Optional[str]  # path to per-sim log file

@dataclass
class BatchConfig:
    max_workers: int
    results_dir: str
    mesh_path: str
    timeout_per_sim: Optional[int] = None  # seconds, None = no timeout
```

### BatchWorker Thread

```python
# swe2d/workbench/workers/batch_worker.py

class BatchWorker(QThread):
    # Internal signals (forwarded by BatchManager)
    _sim_started = pyqtSignal(str)
    _sim_progress = pyqtSignal(str, float, str)
    _sim_completed = pyqtSignal(str, str)
    _sim_failed = pyqtSignal(str, str)
    _log_line = pyqtSignal(str, str)
    _batch_done = pyqtSignal(bool, int, int)  # cancelled, completed, failed

    def __init__(self, params_list, config, cancel_event, parent=None):
        super().__init__(parent)
        self._params_list = params_list
        self._config = config
        self._cancel_event = cancel_event

    def run(self):
        """Main orchestration loop. Runs on background thread."""
        if self._config.max_workers > 1:
            self._run_parallel()
        else:
            self._run_sequential()

    def _run_sequential(self):
        """Launch each sim as subprocess.Popen, poll status JSON, emit progress."""
        for params in self._params_list:
            if self._cancel_event.is_set():
                break
            self._launch_sim(params)

    def _run_parallel(self):
        """Use ProcessPoolExecutor with MPS for concurrent GPU execution.
        
        The executor runs on the BatchWorker thread. Each sim is submitted
        as a subprocess via _launch_sim(). The executor manages parallelism;
        the worker thread handles polling and signal emission.
        """
        with ProcessPoolExecutor(max_workers=self._config.max_workers) as pool:
            futures = {}
            for params in self._params_list:
                if self._cancel_event.is_set():
                    break
                future = pool.submit(self._launch_sim, params)
                futures[future] = params["id"]
            
            for future in as_completed(futures):
                sim_id = futures[future]
                try:
                    future.result()
                except Exception as e:
                    self._sim_failed.emit(sim_id, str(e))

    def _launch_sim(self, params):
        """Launch a single sim subprocess, poll its status file, write log."""
        sim_id = params["id"]
        log_path = os.path.join(self._config.results_dir, "batch_runs", f"{sim_id}.log")

        proc = subprocess.Popen(
            [sys.executable, "-m", "swe2d.cli", "run", ...],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        with open(log_path, "w") as log_file:
            for line in proc.stdout:
                log_file.write(line)
                log_file.flush()
                self._log_line.emit(sim_id, line.strip())

            proc.wait()

        if proc.returncode == 0:
            self._sim_completed.emit(sim_id, results_path)
        else:
            self._sim_failed.emit(sim_id, f"Exit code {proc.returncode}")
```

### Dialog Changes

**Modeless lifecycle:**
- `RunController` holds a single `BatchSimulationDialog` instance (created once, reused)
- `open_batch_simulation_dialog()` calls `self._batch_dialog.show()` + `self._batch_dialog.raise_()` (bring to front)
- Dialog stores `BatchManager` reference (injected via constructor)
- `closeEvent`: if batch is running, `event.ignore()` + `self.hide()` (hide instead of destroy)
- On reopen: `show()` restores visibility; dialog calls `batch_manager.get_status()` to repopulate table from current state

**Table columns:**

| Sim ID | Parameters JSON | Status | Progress | Log |
|--------|----------------|--------|----------|-----|
| sim_001 | `{"n_mann": 0.03}` | Completed | 100% | `[View]` |
| sim_002 | `{"n_mann": 0.04}` | Running | 67% | `[View]` |
| sim_003 | `{"n_mann": 0.05}` | Pending | — | — |

**Log panel:**
- Clicking "View" on a row opens a `QDockWidget` attached to the batch dialog
- Shows last 500 lines from the per-sim log file
- For running sims: `QTimer` re-reads file tail every 1s for auto-refresh
- For completed/failed sims: static display with scroll
- Dock widget title shows the sim ID (e.g., "Log: sim_001")

**Buttons:**
- "Run Batch" → enabled when params loaded, disabled during execution
- "Cancel" → enabled during execution, disabled otherwise
- "Close" → always available (hides dialog if batch running)
- "Check Status" button removed — status is always live via signals

### Log Persistence

- Per-sim logs written to `<results_dir>/batch_runs/<sim_id>.log`
- Created by `BatchWorker` using `subprocess.Popen(..., stdout=PIPE, stderr=STDOUT)`
- Lines are written in real-time (flushed per line) so the dialog can tail them
- Survives QGIS restart — files are on disk

### Error Handling

- Subprocess crash/timeout: worker catches non-zero exit codes, marks sim as failed, emits `sim_failed`
- BatchManager catches worker thread exceptions via `BatchWorker.finished` signal
- Dialog shows a summary `QMessageBox` when batch finishes (N completed, M failed)
- Individual sim failures don't stop the batch — next sim proceeds
- Cancel: sets `threading.Event`, worker checks before launching next sim, in-progress sim is allowed to complete

### MVP Compliance

| Rule | Current | After |
|------|---------|-------|
| View owns no orchestration | `BatchSimulationDialog` imports `BatchOrchestrator` | Dialog only connects signals to `BatchManager` |
| Service owns state | State in dialog attributes | `BatchManager._sim_states` dict |
| Service is Qt-free | N/A | `BatchManager` uses `QObject` only for signals, no widgets |
| Controller orchestrates | `RunController` just calls `dlg.exec()` | `RunController` injects `BatchManager`, calls `dlg.show()` |
| No View computation | Dialog polls subprocesses | `BatchWorker` polls, emits signals |

### Reuse of Existing Code

- `BatchWorker._launch_sim()` reuses the subprocess command construction from `BatchOrchestrator.run()` (lines 260-275 of `batch_runner.py`)
- MPS management reuses `_ensure_mps()` / `_stop_mps_if_we_started()` from `batch_runner.py`
- Sweep expansion reuses `_expand_sweep()` from `batch_runner.py`
- CLI path (`python -m swe2d.cli batch`) continues to use `run_batch()` directly — no change needed

### Testing Strategy

1. **Unit tests for BatchManager:** mock `BatchWorker`, verify signal emission, state transitions
2. **Unit tests for BatchWorker:** mock `subprocess.Popen`, verify progress reporting, cancellation
3. **Integration test:** run a small batch (2-3 sims) in the GUI, verify non-blocking behavior
4. **MVP boundary tests:** grep for violations (no widget imports in service, no orchestration in dialog)
