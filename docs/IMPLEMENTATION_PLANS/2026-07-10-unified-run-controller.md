# Implementation Plan: Unified Run Controller (Single-Kernel-Entry Refactor)

| | |
|---|---|
| **Plan ID** | `SWE2D-RC-2026-07-10` |
| **Status** | Draft |
| **Owner** | SWE2D runtime + workbench teams |
| **Created** | 2026-07-10 |
| **Target branch** | `feature/unified-run-controller` |
| **Companion reading** | [DEVELOPER_GUIDE.md](../DEVELOPER_GUIDE.md), [CLI_GUIDE.md](../CLI_GUIDE.md), [STUDIO_GUI_API.md](../STUDIO_GUI_API.md) |

---

## 1. Problem statement

There are currently **two parallel paths** into the SWE2D GPU kernel:

### 1.1 GUI path (3 layers, view-coupled)

```
SWE2DWorkbenchStudioDialog
   └─ RunController  (workbench/controllers/run_controller.py)
        └─ SimulationWorker  (workbench/workers/simulation_worker.py, QThread)
             └─ SWE2DBackendInitializer  (runtime/backend_initializer.py)
                  └─ SWE2DBackend  (runtime/backend.py)
                       └─ swe2d_gpu.cu (CUDA kernel)
```

Plus, in parallel, a **preflight** path:

```
SWE2DRunController  (runtime/run_controller.py) ← preflight only, no execution
```

### 1.2 CLI path (flat, headless)

```
headless_runner.execute_run(...)
   ├─ SWE2DBackend()                      ← constructed inline
   ├─ shared_build_mesh(backend, ...)
   ├─ backend.initialize(...)              ← ~40 kwargs spread across two functions
   ├─ coupling assembly inline             ← ~80 lines of inline code
   ├─ while-loop calling backend.step()
   └─ SWE2DRunFinalizer(...)              ← with _HeadlessFinalizationView adapter
```

### 1.3 The cost of the duplication

| What gets duplicated | CLI | GUI | Risk |
|----------------------|:--:|:--:|------|
| Mesh assembly (`build_mesh` + bc arrays + override validation) | ✓ | ✓ | Schema drift — CLI override validation logic absent in GUI path |
| Backend `initialize()` (40+ kwargs) | ✓ | ✓ (via initializer) | Adding a new solver option requires editing both paths |
| Coupling controller assembly (drainage + structures + RCMK perm + centroids) | ✓ | ✓ | Order-of-operations drift (CLI applies `_inv_cell_perm` after construction; GUI before) |
| Sample-line setup | ✓ | ✓ | Different packing conventions in each path |
| Status/progress reporting | ✓ (status file) | ✓ (Qt signals) | Different code paths → different progress semantics |
| Run finalization (`SWE2DRunFinalizer`) | ✓ | ✓ | CLI wraps with `_HeadlessFinalizationView`; GUI wraps with `FinalizationAdapter` |
| Spatial-scheme migration warning | ✓ | ✗ | CLI warns, GUI silently accepts old numbers |
| CFL clamping for MP5 (planned scheme 8) | TODO | TODO | No single place to enforce |

The existing `SWE2DRunController` in [`runtime/run_controller.py`](../../swe2d/runtime/run_controller.py) is **preflight-only** — it checks mesh + backend availability and then bails. It does not execute runs.

The `RunController` in [`workbench/controllers/run_controller.py`](../../swe2d/workbench/controllers/run_controller.py) is a **GUI mediator** — it captures widget state, builds a `RunContext`, and starts a `SimulationWorker`. It is not directly callable headlessly.

### 1.4 What "two files don't get created" means in this plan

We will:

- **Delete** [`runtime/run_controller.py`](../../swe2d/runtime/run_controller.py) (preflight-only stub, subsumed).
- **Delete** the `RunController` class in [`workbench/controllers/run_controller.py`](../../swe2d/workbench/controllers/run_controller.py) (replaced by a thin adapter).
- **Add one new file**: [`swe2d/runtime/run_controller.py`](../../swe2d/runtime/run_controller.py) — the single unified controller.

GUI and CLI both call into the unified controller. **No parallel execution paths.** The GUI's existing `SimulationWorker` becomes a Qt-thread wrapper around the unified controller.

---

## 2. Goals

| # | Goal | Acceptance |
|--:|------|------------|
| G1 | One execution path for CLI and GUI | Both call the same `RunController.execute(ctx) -> RunResult` |
| G2 | One mesh-assembly path | `RunController._build_mesh(ctx)` is the only place `SWE2DBackend.build_mesh` is called |
| G3 | One backend-init path | `RunController._initialize_backend(ctx)` is the only place `backend.initialize()` is called |
| G4 | One coupling-assembly path | `RunController._build_coupling(ctx, backend)` |
| G5 | One sample-line setup path | `RunController._build_sample_lines(ctx, mesh_data, backend)` |
| G6 | One status-reporting seam | `RunController` accepts a `ProgressSink` protocol; CLI provides `FileProgressSink`, GUI provides `QtProgressSink` |
| G7 | One finalization path | `RunController._finalize(ctx, backend, result)` |
| G8 | Scheme migration, CFL clamping, option validation live in one place | All CLI/GUI callers get them automatically |
| G9 | No new public types duplicated between GUI/CLI | `RunContext` is the shared DTO; `RunController` is the shared executor |

---

## 3. Target architecture

### 3.1 Module layout

```
swe2d/runtime/
├── run_controller.py        (NEW — unified controller)
├── run_context.py           (existing — DTO, will be extended)
├── backend.py               (unchanged)
├── backend_initializer.py   (existing — wrapped by RunController)
├── run_finalizer.py         (unchanged)
└── coupling.py              (unchanged)
```

**Deleted**:
- `swe2d/runtime/run_controller.py` (old preflight-only — replaced by new file).
- `class RunController` in `swe2d/workbench/controllers/run_controller.py` (replaced by thin adapter that calls unified controller).
- `class SimulationWorker.run_body()` mesh-assembly + backend-init code (moves to `RunController`).

### 3.2 Data flow

```
                ┌──────────────────────────┐
                │  RunContext (immutable)  │
                │  - mesh_data, bc arrays  │
                │  - solver params         │
                │  - coupling configs      │
                │  - sample_map_data       │
                │  - output paths          │
                └────────────┬─────────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │     RunController    │
                  │                      │
                  │ .execute(ctx, sink): │
                  │   ├─ preflight       │
                  │   ├─ build_mesh      │
                  │   ├─ init_backend    │
                  │   ├─ build_coupling  │
                  │   ├─ build_lines     │
                  │   ├─ step_loop       │
                  │   └─ finalize        │
                  └────────┬─────────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
    ┌─────────────────────┐    ┌─────────────────────┐
    │ ProgressSink (CLI)  │    │ ProgressSink (GUI)  │
    │  FileProgressSink   │    │   QtProgressSink    │
    │  - status file      │    │   - Qt signals      │
    └─────────────────────┘    └─────────────────────┘
```

### 3.3 The unified controller

The new `SWE2DRunController` is the single owner of "how a run happens." It has these public methods:

```python
class SWE2DRunController:
    """Single execution path shared by CLI and GUI.

    GUI callers wrap with a Qt-thread worker.
    CLI callers wrap with a status-file progress sink.
    Both call .execute() identically.
    """

    def __init__(self, ctx: RunContext, sink: ProgressSink): ...

    def execute(self) -> RunResult:
        """Execute the run described by ctx, reporting to sink.

        Returns RunResult with h, hu, hv, max_results, diags, etc.
        Raises PreflightError if preflight fails.
        """

    # ── Internal phases (private, ordered by execute) ──

    def _preflight(self) -> None: ...
    def _build_mesh(self) -> None: ...
    def _initialize_backend(self) -> None: ...
    def _build_coupling(self) -> None: ...
    def _build_sample_lines(self) -> None: ...
    def _apply_cell_permutation(self) -> None: ...
    def _step_loop(self) -> RunResult: ...
    def _finalize(self, result: RunResult) -> None: ...
```

### 3.4 ProgressSink protocol

```python
class ProgressSink(Protocol):
    """Where the controller reports progress, logs, snapshots, and lifecycle events."""

    def on_log(self, msg: str) -> None: ...
    def on_progress(self, step: int, t: float, dt: float, wet_cells: int) -> None: ...
    def on_snapshot(self, snapshot: SnapshotData) -> None: ...
    def on_finished(self, result: RunResult) -> None: ...
    def on_failed(self, error: str) -> None: ...
    def on_cancelled(self) -> None: ...
    def is_cancel_requested(self) -> bool: ...
```

**Concrete implementations**:

```python
class FileProgressSink:
    """Writes status file every N seconds for CLI."""

class QtProgressSink:
    """Emits Qt signals on the controller's owning thread."""

class NullProgressSink:
    """For tests."""
```

### 3.5 Where the GUI plug-point lives

GUI's `SimulationWorker` (QThread) becomes:

```python
class SimulationWorker(QThread):
    def __init__(self, ctx: RunContext, parent=None):
        super().__init__(parent)
        self._ctx = ctx

    def run(self):
        # Hand-off happens on the worker thread, NOT the GUI thread
        sink = QtProgressSink(self)  # emits Qt signals back to GUI thread
        controller = SWE2DRunController(self._ctx, sink)
        try:
            result = controller.execute()
            self.compute_finished.emit(result)
        except PreflightError as e:
            self.compute_failed.emit(str(e))
        except Exception as e:
            self.compute_failed.emit(f"{type(e).__name__}: {e}")
```

The `SimulationWorker` retains:
- QThread plumbing (start, stop, signals).
- The mesh-permutation hand-off (this is GUI-thread specific because the view mesh needs re-permutation; see §6).

But the mesh-assembly + backend-init + coupling + line-setup + step-loop code is **deleted from the worker** and lives in the unified controller.

### 3.6 Where the CLI plug-point lives

CLI's `execute_run()` becomes:

```python
def execute_run(mesh_gpkg, params, results_gpkg=None, ...):
    ctx = build_run_context_from_json_params(mesh_gpkg, params, results_gpkg, ...)
    sink = FileProgressSink(status_file_path, status_interval_s)
    controller = SWE2DRunController(ctx, sink)
    return controller.execute().to_dict()
```

The CLI-specific concerns that **stay in `headless_runner.py`**:
- JSON parsing (`_parse_params`).
- Building `RunContext` from JSON (the GUI builds it from widget state; CLI builds it from JSON).
- `_HeadlessFinalizationView` — but this is now consumed by the controller, not by `execute_run`.

---

## 4. RunContext extensions

The existing [`RunContext`](../../swe2d/workbench/workers/run_context.py) is GUI-flavored (contains widget-derived fields like `mesh_cell_areas_fn`, `mesh_cell_min_bed`, callbacks). The unified controller needs the same fields for CLI but without the callables.

### 4.1 Two factories

```python
# Existing (GUI): captures widget state + callbacks
def from_view(view, request) -> RunContext: ...

# New (CLI): builds from JSON params + GPKG
def from_json_params(mesh_gpkg, params, results_gpkg, ...) -> RunContext: ...
```

Both produce the same `RunContext` shape. The controller is agnostic to which factory built it.

### 4.2 New optional fields (CLI support)

| Field | CLI usage | GUI usage |
|-------|-----------|-----------|
| `cancel_check: Callable[[], bool] | None` | None (use sink) | view._cancel_requested polled |
| `results_gpkg_path: str` | always required | already present |
| `mesh_gpkg_path: str` | required | already present (model_gpkg_path) |
| `log_sink: ProgressSink | None` | injected | injected |

### 4.3 Renames for consistency

- `ctx.run_data_builder` → `ctx.run_options` (already named this in headless; GUI used `run_options_builder`).
- `ctx.run_input` → `ctx.run_input` (unchanged).

No deep restructure; mostly formalizing what already exists.

---

## 5. Phase-by-phase implementation

### Phase 0 — Foundation (1 day)

| Task | Deliverable |
|------|-------------|
| Define `ProgressSink` protocol in `swe2d/runtime/run_controller.py` | New file |
| Add `NullProgressSink`, `FileProgressSink` to same file | Two classes |
| Add `PreflightError`, `CancelledError` exceptions | Two classes |
| Add `RunResult` dataclass (mirrors `ComputeResult` shape) | One dataclass |
| Write `tests/test_run_controller_protocols.py` | Tests |

**No production code touched.** This phase is purely additive infrastructure.

### Phase 1 — Extract mesh-assembly + backend-init from CLI (2 days)

| Task | Deliverable |
|------|-------------|
| Move `_open_cfg`, `_bc_arrays_from_dict`, `_valid_bc_arrays`, `_norm_key`, BC-override validation from `headless_runner.py` to `RunController._build_mesh()` | New private method |
| Move `shared_build_mesh` call into `RunController._build_mesh()` | New private method |
| Move `backend.initialize(...)` call into `RunController._initialize_backend()` | New private method |
| Add `_warn_scheme_migration` call inside `_initialize_backend()` (was already in CLI; was missing in GUI path) | Single line |
| Add CFL clamping by scheme inside `_initialize_backend()` | New behavior (was TODO in MP5 plan) |
| Refactor `headless_runner.execute_run()` to call `SWE2DRunController(ctx, sink).execute()` | Replace inline body |
| Run existing CLI test suite to confirm no regression | `pytest tests/cli_*.py` |

**Risk**: CLI behavior must match byte-for-byte after refactor. Run existing CLI tests as regression check.

### Phase 2 — Extract coupling + sample-line assembly (2 days)

| Task | Deliverable |
|------|-------------|
| Move coupling config assembly (drainage + structures + RCMK perm + centroids + `_build_redistribution_data`) from `headless_runner.py` and from `SimulationWorker.run_body()` into `RunController._build_coupling()` | New private method |
| Move sample-line setup from both paths into `RunController._build_sample_lines()` | New private method |
| Reconcile the `_inv_cell_perm` ordering — both paths must apply it at the same phase | Verified |
| Run CLI + GUI tests | `pytest tests/ -k 'cli or gui or worker or coupling'` |

**Risk**: The RCMK-permutation ordering is sensitive. Document the canonical order in the controller method docstring.

### Phase 3 — Extract step-loop + finalization (1 day)

| Task | Deliverable |
|------|-------------|
| Move `while t < t_end` loop from `headless_runner.py` and from `SimulationWorker.run_body()` into `RunController._step_loop()` | New private method |
| Move `_collect_coupling_snapshot()` into `RunController` as `_collect_coupling_snapshot()` | New private method |
| Move `SWE2DRunFinalizer` call into `RunController._finalize()` | New private method |
| Refactor `headless_runner` to call controller | Replace inline body |
| Refactor `SimulationWorker` to call controller | Replace inline body |
| GUI `_HeadlessFinalizationView` and `FinalizationAdapter` are now **only** constructed by the controller's CLI/GUI factory methods | Two factory methods |

### Phase 4 — Delete duplicated code (1 day)

| Task | Deliverable |
|------|-------------|
| Delete inline mesh assembly from `headless_runner.py` (lines ~210–290) | Removed |
| Delete inline coupling assembly from `headless_runner.py` (lines ~330–430) | Removed |
| Delete inline sample-line setup from `headless_runner.py` (lines ~440–520) | Removed |
| Delete inline step-loop from `headless_runner.py` (lines ~530–680) | Removed |
| Delete `SimulationWorker._build_and_initialize_backend` + coupling + line-setup code | Removed |
| Delete `SWE2DRunController` (preflight-only stub) in `runtime/run_controller.py` | Will be replaced by new file in same path |
| Delete `RunController` class in `workbench/controllers/run_controller.py` | Replaced by `WorkbenchRunAdapter` (see §6) |

### Phase 5 — GUI adapter (1 day)

```python
# swe2d/workbench/controllers/run_controller.py  (renamed to workbench_run_adapter.py? — see §6)
class WorkbenchRunAdapter:
    """Thin shim: GUI's view-layer calls into the unified RunController."""

    def __init__(self, view):
        self._view = view
        # Inherit RunController's interface so the view doesn't change much.

    def on_run(self, request=None) -> None:
        # Build RunContext from view state (uses from_view factory)
        ctx = build_run_context_from_view(self._view, request)
        # Hand off to worker; worker calls RunController
        worker = SimulationWorker(ctx, parent=self._view)
        # ... wire signals ...
        worker.start()
```

**The GUI keeps its `_on_worker_snapshot_ready`, `_on_worker_compute_finished` slots** — those are GUI-side reactions to controller-emitted sink events.

### Phase 6 — Tests + verification (2 days)

| Test | What |
|------|------|
| `test_run_controller_cli.py` | Full CLI run via controller, compare to legacy headless output |
| `test_run_controller_gui_context.py` | Build `RunContext` from synthetic view state, run via controller |
| `test_run_controller_cancel.py` | Cancellation via sink during step-loop |
| `test_run_controller_preflight.py` | Mesh/backend availability checks |
| `test_progress_sink_file.py` | FileProgressSink writes correct JSON status |
| `test_progress_sink_qt.py` | QtProgressSink emits correct signals (QSignalSpy) |
| `test_scheme_migration_unified.py` | Old scheme-6 warning fires from controller regardless of caller |

### Phase 7 — Migration + docs (2 days)

- Mark `runtime/run_controller.py` (old) as deprecated, point to new module.
- Update [DEVELOPER_GUIDE.md](../DEVELOPER_GUIDE.md), [CLI_GUIDE.md](../CLI_GUIDE.md), [STUDIO_GUI_API.md](../STUDIO_GUI_API.md).
- Update CHANGELOG.

---

## 6. GUI-specific concerns: what stays in `workbench/`

The GUI's `SimulationWorker` has two GUI-thread-coupled behaviors that **cannot** move into the headless controller:

### 6.1 Mesh permutation hand-off

The worker calls `view._mesh_data.apply_cell_permutation(...)` from the GUI thread because:
- The mesh lives on the view's QObject.
- Other view tabs (sample lines, drainage overlay) need the new mesh to render.
- The controller thread must wait for the permutation to complete before continuing.

Solution: keep `mesh_permutation_ready` signal in `SimulationWorker`. The controller emits the signal via a Qt-aware progress sink; the GUI's `_on_worker_mesh_permutation_ready` slot applies the permutation and signals back via `result_holder.event.set()`.

### 6.2 View-thread reactivity

The view polls `view._cancel_requested`, updates progress bars, runs the high-perf overlay refresh. These are view-thread concerns.

Solution: the `QtProgressSink` runs on the controller's worker thread but **emits signals that the view consumes via Qt's queued-connection mechanism.** Qt handles the cross-thread marshaling automatically. The view's slots run on the GUI thread.

### 6.3 What goes where

| Concern | Lives in | Thread |
|---------|----------|--------|
| Build mesh from GPKG | `RunController._build_mesh()` | worker |
| Initialize `SWE2DBackend` | `RunController._initialize_backend()` | worker |
| Build coupling | `RunController._build_coupling()` | worker |
| Build sample lines | `RunController._build_sample_lines()` | worker |
| Step loop | `RunController._step_loop()` | worker |
| Finalize run | `RunController._finalize()` | worker |
| Apply mesh permutation to view | `SimulationWorker` slot → view | GUI |
| Update progress bar | `QtProgressSink.on_progress` → view slot | GUI |
| Plot refresh | `QtProgressSink.on_snapshot` → view slot | GUI |
| Cancel handling | `sink.is_cancel_requested` polls `view._cancel_requested` | worker polls GUI thread-safe bool |
| Run-record seeding | `view._on_worker_compute_finished` slot | GUI |

---

## 7. CFL clamping (closes a deferred item from the spatial-schemes plan)

The MP5 plan requires `cfl ≤ 0.4` for scheme 8. Today neither path enforces this.

In the new controller, `_initialize_backend()` does:

```python
def _initialize_backend(self):
    scheme = int(self._ctx.spatial_discretization)
    scheme_max_cfl = {
        0: 0.8, 1: 0.8, 2: 0.8, 3: 0.8, 4: 0.8,
        5: 0.8,
        6: 0.8,
        7: 0.5,
        8: 0.4,
    }[scheme]
    user_cfl = float(self._ctx.cfl)
    effective_cfl = min(user_cfl, scheme_max_cfl)
    if effective_cfl < user_cfl:
        self._sink.on_log(
            f"[SCHEME] Spatial scheme {scheme} requires CFL ≤ {scheme_max_cfl:.2f}; "
            f"user-set {user_cfl:.2f} clamped to {effective_cfl:.2f}."
        )
    self._backend.initialize(..., cfl=effective_cfl, ...)
```

This single change closes both CLI and GUI paths simultaneously.

---

## 8. File-level change list

### 8.1 New files

| Path | Purpose |
|------|---------|
| `swe2d/runtime/run_controller.py` | Unified controller — replaces both existing run controllers |

### 8.2 Modified files

| Path | Change |
|------|--------|
| `swe2d/runtime/run_context.py` | Add `cancel_check`, `log_sink` fields. Add `from_json_params()` factory. |
| `swe2d/cli/headless_runner.py` | Replace inline execution body with `RunController(ctx, sink).execute()`. Keep JSON parsing + `_HeadlessFinalizationView` factory. |
| `swe2d/workbench/workers/simulation_worker.py` | Replace `_build_and_initialize_backend` + inline coupling + line-setup + step-loop with `RunController(ctx, QtProgressSink(self)).execute()`. Keep QThread plumbing + `mesh_permutation_ready` hand-off. |
| `swe2d/workbench/controllers/run_controller.py` | `RunController` class replaced with `WorkbenchRunAdapter` thin shim. |
| `swe2d/runtime/backend.py` | No changes to the backend itself. The `_warn_scheme_migration` call moves up to the controller (already exists; just changes call site). |

### 8.3 Deleted files

None. The path `swe2d/runtime/run_controller.py` is reused (old file overwritten with new content). The `RunController` class inside `swe2d/workbench/controllers/run_controller.py` is removed (replaced by adapter); the file itself remains because it holds `WorkbenchRunAdapter` + many GUI dialog methods (on_load_simulation_config, on_save_simulation_config, etc.).

---

## 9. Migration path

### 9.1 For CLI users

No CLI flag changes. The behavior matches the legacy headless runner byte-for-byte (verified by Phase 6 regression tests).

### 9.2 For GUI users

No user-visible changes. The view dialog buttons continue to work the same way.

### 9.3 For developers

Two touch points to learn:

1. **Where to add a new solver option**: `RunContext` (data) + `RunController._initialize_backend` (consumption). No more "did I update the CLI path?" question.
2. **Where to add a new progress event**: add method to `ProgressSink` protocol; implement in both `FileProgressSink` and `QtProgressSink`.

---

## 10. Risk register

| Risk | Probability | Impact | Mitigation |
|------|:-:|:-:|------|
| CLI/GUI behavior diverges during refactor | Medium | High | Phase 6 byte-for-byte regression tests on both paths |
| RCMK permutation ordering breaks drainage coupling | Medium | High | Document canonical order in `_apply_cell_permutation` docstring; preserve ordering across all callers |
| Qt cross-thread marshaling introduces race conditions | Medium | Medium | Use Qt::QueuedConnection for all sink signals; existing pattern in `SimulationWorker` |
| Cancellation races (worker checks, GUI sets) | Low | Medium | Document atomic bool read pattern; use Qt's signal-slot where possible |
| Phase 0 `ProgressSink` protocol gets over-engineered | Medium | Low | Keep protocol minimal in Phase 0; add methods only as Phase 1–3 needs them |
| GUI's `_on_worker_snapshot_ready` coupling to specific snapshot shape | Low | Low | `SnapshotData` dataclass shared between paths |
| Old `runtime/run_controller.py` import paths still referenced | Medium | Low | Grep for `from swe2d.runtime.run_controller import SWE2DRunController`; the same class name is reused, so most imports continue working |

---

## 11. Acceptance criteria (Definition of Done)

- [ ] `swe2d/runtime/run_controller.py` exists with one `SWE2DRunController` class that owns all execution phases.
- [ ] `headless_runner.execute_run()` delegates to `SWE2DRunController`. Body is < 100 lines.
- [ ] `SimulationWorker.run()` delegates to `SWE2DRunController`. Body is < 50 lines.
- [ ] `grep -r 'backend.initialize' swe2d/cli swe2d/workbench` returns only the unified controller.
- [ ] `grep -r 'SWE2DBackend()' swe2d/cli swe2d/workbench` returns only the unified controller.
- [ ] `grep -r 'shared_build_mesh' swe2d/cli swe2d/workbench` returns only the unified controller.
- [ ] All existing CLI tests pass.
- [ ] All existing GUI tests pass.
- [ ] New test `test_run_controller_cli.py` produces bit-identical output to legacy headless runner for a known test case.
- [ ] New test `test_run_controller_scheme_migration.py` verifies old scheme-6 warning fires from controller for both CLI and GUI paths.
- [ ] CFL clamping works for scheme 8 (manual test).
- [ ] CHANGELOG updated.

---

## 12. Out of scope

- Changes to the C++/CUDA kernel layer.
- Changes to mesh data structures.
- Changes to GPKG schema.
- Refactor of `SWE2DBackend` itself.
- Migration of `RuntimeContext` field semantics (e.g., `coupling_loop_mode` — only renamed, not redesigned).

---

## 13. References

- [DEVELOPER_GUIDE.md](../DEVELOPER_GUIDE.md) — current architecture; will need update after refactor.
- [CLI_GUIDE.md](../CLI_GUIDE.md) — current CLI usage; unchanged after refactor.
- [STUDIO_GUI_API.md](../STUDIO_GUI_API.md) — current GUI contracts; mostly unchanged after refactor.
- [IMPLEMENTATION_PLANS/2026-07-10-advanced-spatial-schemes.md](2026-07-10-advanced-spatial-schemes.md) — companion plan; CFL clamping item closes here.
- [SWE2D_GPU_ARCHITECTURE_REPORT.md](../SWE2D_GPU_ARCHITECTURE_REPORT.md) — kernel-layer reference (unchanged).

---

## 14. Open questions

1. **Should `WorkbenchRunAdapter` live in `workbench/controllers/` or `workbench/adapters/`?** — Recommendation: `workbench/adapters/run_adapter.py` (cleaner separation; the `controllers/` directory implies MVC controllers, but the adapter is more of an anti-corruption layer).
2. **Should the `RunContext` `from_view` factory stay in the view-side `run_data_builder`/`run_options_builder`, or move to `RunContext` itself?** — Recommendation: move to `RunContext` as static methods for symmetry with `from_json_params`.
3. **Should `SimulationWorker` survive, or should the controller live directly on a QThread?** — Recommendation: keep `SimulationWorker` because it owns the Qt signal/slot wiring + thread lifecycle; the controller stays Qt-agnostic.
4. **Should the unified controller be Qt-agnostic enough to be unit-testable without QGIS?** — Recommendation: yes — `RunController` does not import PyQt5. `QtProgressSink` does, but `NullProgressSink` does not. Tests use `NullProgressSink`.
5. **What's the deprecation period for the old `runtime/run_controller.py` content?** — Recommendation: zero — it's being replaced wholesale in the same path; no external imports will break because the class name (`SWE2DRunController`) is preserved.