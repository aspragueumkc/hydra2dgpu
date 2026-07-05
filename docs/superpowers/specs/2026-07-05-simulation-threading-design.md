# Simulation Threading Design — HYDRA2DGPU Workbench

## Status

Approved for implementation.

## Problem

Initiating simulation runs, fetching device snapshots, and finalizing/persisting results to GeoPackage all execute on the QGIS main thread. Long-running native CUDA solver work and GeoPackage writes freeze the QGIS GUI until completion.

## Goal

Move the heavy work (solver loop, device readback, GeoPackage persistence) onto background `QThread` workers while keeping the QGIS UI responsive and all existing behavior intact.

## Scope

This design covers:

1. Running a full 2D SWE simulation.
2. Fetching live device snapshots during a run.
3. Finalizing a run and persisting results to GeoPackage.

Out of scope:

- Mesh generation threading (Gmsh already runs in its own process/thread model).
- Headless/batch CLI execution (`swe2d/cli/`) — it remains single-threaded by design.

## Constraints

- CUDA backend (`SWE2DBackend`) must be created and destroyed in the same thread.
- No Qt imports in `swe2d/services/`, `swe2d/runtime/`, or any service/runtime module.
- Existing `RunFinalizationView` protocol and view adapter must be preserved.
- Cancellation must be safe and leave the backend cleaned up.
- Worker-thread signals must use Qt queued connections to marshal data back to the main thread.

## Architecture

### Thread roles

| Thread | Responsibility |
|--------|---------------|
| Main (QGIS) | Build context, start/stop workers, update widgets, sync map overlay, refresh plots, handle cancel. |
| Worker 1 (`SimulationWorker`) | Own the `SWE2DBackend`, run timestep loop, read snapshots, emit progress/log/snapshot signals. |
| Worker 2 (`PersistenceWorker`) | Compute mass-balance summary, write baked results/line TS/profiles/coupling/run log to GeoPackage. |

A single simulation run uses **Worker 1** for compute and **Worker 2** for persistence. Snapshot fetch during an active run is handled inside **Worker 1**. Snapshot fetch when no run is active may use a small readback worker or remain synchronous if data is already in memory.

### RunContext (immutable, built on main thread)

Captured from the View before the worker starts:

- Parsed run parameters: CFL, dt, run/output/line intervals, schemes, caps, source toggles, tiny-mode config, gravity, etc.
- Mesh arrays: `node_x`, `node_y`, `node_z`, `cell_nodes`, `face_offsets`, `face_nodes`, `cell_areas`, `cell_solver_bed`, `cell_centroids`.
- Boundary data: `bc_n0`, `bc_n1`, `bc_tp`, `bc_vl`, side/edge hydrographs, edge group overrides.
- Initial state: `h0`, `hu0`, `hv0`, `n_mann_cell`.
- Coupling configs: pipe network, hydraulic structures, bridge stacked plans, coupling SoA.
- Results storage: results GeoPackage path, model GeoPackage path, `run_id`, `run_wallclock_start`, `run_log_start_idx`.
- Unit conversion factors and `length_unit_name`.
- Pure-logic callbacks that do not touch Qt widgets.
- `threading.Event` for cancellation.

RunContext is serializable in memory only; it is passed by reference to the worker.

### SimulationWorker

Extends `QThread`.

Signals (all queued):

- `log_message(str)` — append to runtime log.
- `progress_percent(int)` — update progress bar.
- `snapshot_ready(SnapshotData)` — update temporal dock, overlay, plots.
- `compute_finished(ComputeResult)` — hand off to persistence.
- `compute_failed(str)` — report error and clean up.

Execution:

1. Build backend in worker thread using `RunContext`.
2. Apply cell permutation to mesh data copy if required.
3. Configure native BC/rain/source injection.
4. Run timestep loop:
   - Check `RunContext.cancel_event` each iteration.
   - Evaluate BC, sources, coupling, solver step.
   - Emit log/progress as configured.
   - Read device snapshots when due and emit `snapshot_ready`.
5. On completion:
   - Read final snapshots and `get_state()`.
   - Permute state to RCMK order if needed.
   - Emit `compute_finished` with final arrays, snapshot list, coupling data, budgets, and metadata.
6. In `finally`: call `backend.destroy()`.

### PersistenceWorker

Extends `QThread`.

Signals (all queued):

- `log_message(str)` — append to runtime log.
- `persist_finished(PersistStatus)` — re-enable UI, refresh overlay/plot.
- `persist_failed(str)` — report error and clean up.

Execution:

1. Compute mass balance using final state and accumulated budgets.
2. Write baked mesh results to GeoPackage.
3. Write line time-series and profiles.
4. Write coupling metrics.
5. Write run log.
6. Emit `persist_finished` with status.

No QWidget access. All values come from the `ComputeResult` object.

### Controller orchestration

`RunController.on_run()` becomes:

1. Run preflight (`RunController` already has a preflight seam).
2. Build `RunContext` from the View.
3. Disable Run button, enable Cancel button, reset progress.
4. Instantiate `SimulationWorker`, connect signals.
5. Start worker.
6. In slots:
   - `log_message` → `view._log(msg)`
   - `progress_percent` → `view.set_run_progress(pct)`
   - `snapshot_ready` → update `ResultsData`, temporal dock, overlay, plot
   - `compute_finished` → instantiate `PersistenceWorker`, connect signals, start it
   - `compute_failed` → log, show error, re-enable controls
7. On `persist_finished`:
   - Sync overlay and plot.
   - Re-enable Run, disable Cancel.
8. On `persist_failed`:
   - Log and show error.
   - Re-enable controls.

### Snapshot fetch

- During active run: handled by `SimulationWorker` emitting `snapshot_ready` at output intervals. The "Fetch Device Results" button calls `runtime_reporter.request_snapshot_readback()` or a worker-local equivalent; the next due readback emits `snapshot_ready`.
- No active run: existing snapshots are already in `ResultsData`; refresh UI synchronously or via a tiny worker if recomputation is needed.

### Cancellation

- Main thread sets `RunContext.cancel_event.set()`.
- Worker checks `cancel_event.is_set()` at the top of the timestep loop.
- On exit, the worker calls `backend.destroy()` in its own thread and emits `compute_finished` with a cancelled flag.
- `PersistenceWorker` still runs so that partial results are persisted and the run log records the cancellation.

### Error handling

- Worker catches exceptions at the outer boundary, logs them, emits `compute_failed` or `persist_failed`, and ensures resources are released.
- Main thread shows a critical message and re-enables controls.
- No silent fallbacks.

### DRY: shared snapshot UI sync

The existing `_on_snapshot_readback` callback in `run_controller.py` and the new `snapshot_ready` slot both perform the same temporal-dock / overlay / plot updates. Extract a single helper `_sync_snapshot_to_ui(view, snapshot_data)` and call it from both locations to avoid duplication.

### MVP compliance

- Workers live under `swe2d/workbench/workers/` and may import `PyQt5.QtCore`.
- No Qt imports in `swe2d/services/` or `swe2d/runtime/`.
- `SWE2DRunFinalizer` remains a service-layer object; `PersistenceWorker` calls it, but the finalizer itself does not know about threads.
- View methods that currently mix Qt widget access with logic are split:
  - Widget values are read into `RunContext` on the main thread.
  - Pure logic stays in service/runtime modules or becomes worker-local callables.

## Files to touch

- `swe2d/workbench/controllers/run_controller.py` — orchestrate workers, build `RunContext`.
- `swe2d/workbench/workers/simulation_worker.py` — new file.
- `swe2d/workbench/workers/persistence_worker.py` — new file.
- `swe2d/workbench/workers/__init__.py` — new file.
- `swe2d/workbench/controllers/run_component_wiring_controller.py` — wire workers into the View.
- `swe2d/workbench/studio_dialog.py` — split `_apply_external_sources`, `_distribute_total_flow_to_unit_q`, `_sample_line_metrics` into widget-capture + logic; add worker signal slots.
- `swe2d/workbench/services/run_service.py` — optionally move `compute_progress` and parameter collection helpers used by `RunContext`.
- `swe2d/runtime/run_lifecycle.py` — ensure cleanup works when backend is owned by worker.

## Testing strategy

- Unit test `RunContext` builder with mock View.
- Unit test `SimulationWorker` with a mock backend that verifies signals are emitted and cancel event is checked.
- Unit test `PersistenceWorker` with a temporary GeoPackage and small synthetic data.
- GPU validation tests remain the primary integration check:
  - `tests/test_swe2d_gpu_validation_perf.py`
  - `tests/test_swe2d_gpu_unstructured.py`

## Risks and mitigations

| Risk | Mitigation |
|------|-----------|
| CUDA context/thread mismatch | Create and destroy backend inside `SimulationWorker.run()`. |
| QWidget access from worker thread | Capture widget values into `RunContext` on main thread; move logic to services. |
| Stale results during persistence | Keep `ResultsData` live; `PersistenceWorker` only writes, does not mutate live snapshots. |
| Cancel leaves backend alive | Use try/finally in worker run and call `backend.destroy()`. |
| Race on sequential runs | Disable Run button until `persist_finished`; store active worker reference and reject new runs. |

## Open questions

None. Design approved by user.
