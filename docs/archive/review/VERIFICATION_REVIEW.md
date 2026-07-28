---
type: reference
status: complete
created: 2026-07-24
completed: 2026-07-25
---

# Verification Review — Post-Fix (refactor/cli-first @ 516e223)

Second-pass review of the 8 fix commits (+1 test commit, `d9d6297`) that
addressed the 7 criticals from `docs/COMPREHENSIVE_REVIEW.md`. Scope: verify
each fix is correct and complete, and that the fixes did not break anything
else.

## Test runs

Canonical command from the review task (run in the worktree):

```
PYTHONPATH=. python3 -m unittest tests.test_import_boundary tests.test_run_context_builder
→ Ran 29 tests — FAILED (errors=5)
```

The 5 errors are all `setUpClass … ModuleNotFoundError: No module named
'hydra_swe2d'` — the native extension is not built inside the worktree.
This is environmental, not caused by the fixes (the C-3 commit message
already notes these 5 errors pre-date it). With the main checkout's build
on the path everything passes:

```
PYTHONPATH=.:../../build python3 -m unittest tests.test_import_boundary tests.test_run_context_builder
→ Ran 48 tests — OK
```

Targeted runs (all with `PYTHONPATH=.:../../build`):

- `TestDrainageGpkgAdapter` + `TestDrainageConfigParity` — 5/5 OK
- `tests.test_import_boundary` — 6/6 OK (delta check, order-independent)
- `tests.test_cli` under `python -m unittest` — **0 tests collected** (the
  file is pytest-style functions; see N-4 below)
- `tests.test_cli` under pytest — collection aborts: pytest imports the
  plugin-root `__init__.py` (QGIS entry point), whose `_import_all()` pulls
  in `swe2d.results.high_perf_viewer`, which crashes headless with
  `TypeError: NoneType takes no arguments` (class inherits a `None`
  `QgsMapCanvasItem`). Pre-existing, unrelated to the fixes (none of the 8
  commits touch `__init__.py`, `swe2d/results/`, or pytest config), and a
  documented limitation — `tests/test_import_boundary.py:22-26` explains
  this is exactly why the boundary tests are unittest-style.
- `tests.test_workbench_gui` — 47 tests, 2 failures + 13 errors. The 2
  failures are source-text assertions on `swe2d/workbench/studio_dialog.py`
  (`test_dialog_has_no_inline_sqlite3`, `load_mesh_snapshot_for_overlay`
  source check); the 13 errors are QGIS-dependent tests in a headless env.
  None of the 8 fix commits touch `studio_dialog.py` or
  `test_workbench_gui.py` — pre-existing/environmental, not regressions.

## Fix Verification Matrix

| # | Status | Evidence | Issues |
|---|--------|----------|--------|
| C-1 | VERIFIED | `swe2d/workbench/adapters/run_context_adapter.py:259-321` builds real `QgsVectorLayer`s (nodes/links/inlets/node-inlets), validates them, delegates to `pipe_network_service.build_pipe_network_config` (`pipe_network_service.py:94`) which returns a `PipeNetworkConfig`. `TestDrainageGpkgAdapter` passes. | QGIS import is lazy with a logged `None` fallback (acceptable — module lives behind the workbench boundary). |
| C-2 | VERIFIED | `swe2d/runtime/run_context_builder.py:468-503` `_drainage_config_dict` emits all 12 keys; the 5 added keys read the `drainage` block first, then fall back to top-level `_v(...)`; defaults match `PipeNetworkConfig` (`swe2d/extensions/extension_models.py:322-330`: friction_method=0, surcharge_method=0, recon_method=0, time_integrator=1, friction_alpha=0.01). 4 `TestDrainageConfigParity` tests pass. | Cosmetic: commits `da2e12e` and `4af6372` share the same message (second is the helper-extraction + tests on top of the first). |
| C-3 | **PARTIAL — regression introduced** | `run_controller.py:147-316` — `dataclasses.replace` post-pass is gone; forcing objects (`internal_flow_forcing`, `thiessen_forcing`, `cell_source_model`, `rain_rate_model`, `pipe_network_cfg`, `hydraulic_structures_cfg`) flow through the spec and are honored by the builder's `_override` (`run_context_builder.py:595-597, 1018-1025`). Docstrings updated. | The fix dropped every *other* field the `replace()` used to supply — see N-1 (critical). |
| C-5 | VERIFIED | `swe2d/workbench/workers/__init__.py:35-60` — PEP 562 `__getattr__` lazy-loads `SimulationWorker`/`ComputeResult`/`SnapshotData`/`PersistenceWorker`; `RunContext` stays eager (Qt-free). 6/6 import-boundary tests pass, including the Qt-delta assertion. | None. |
| H-1 | VERIFIED (code) / WEAK (test) | `swe2d/cli/headless_runner.py:129-172` — `_step_count` increments once per progress callback; payload writes `step`/`step_idx` = step number, `pct` = 0–100 percent. Docstring at lines 58-68 updated. | `test_status_file_step_contract_h1` is tautological — see N-4. Docstring wording misleading — see N-5. |
| H-2 | VERIFIED (code + docs) / WEAK (test) | `headless_runner.py:43` types the callback `Callable[[float, Dict[str, Any]], None]`; line 171 calls `progress_callback(pct, {})`. `docs/CLI_GUIDE.md:202-206` documents `progress_callback(percent, diagnostics)`. | `test_progress_callback_signature_h2` is tautological — see N-4. |
| Test fix | VERIFIED | `tests/test_import_boundary.py:39-57` snapshots Qt modules before each import and asserts on the delta; docstring (lines 13-20) explains the pollution mechanism. 6/6 pass standalone and together with the builder tests (48/48). | None. |

## New Issues Found

### N-1 — CRITICAL (regression introduced by the C-3 fix, commit `673b714`)

Removing the `dataclasses.replace` post-pass also removed the only channel
that carried these GUI-computed fields into the `RunContext`:

- **`h0` / `hu0` / `hv0` (initial conditions).** The view's `_initial_state`
  (`studio_dialog.py:1858-1872`, modes `uniform_depth` / `uniform_wse` /
  `dry` with inflow priming) builds them from the IC widgets; the controller
  still computes them via `run_data_builder.build()` but the adapter
  (`run_context_adapter.py`) forwards **neither** `h0`/`hu0`/`hv0` **nor**
  the scalar widget keys (`initial_wse_spin`, `initial_depth_spin`,
  `initial_condition_combo` are absent from its forwarding list, lines
  110-136). Result: `build_run_context` falls through to
  `h0 = np.zeros(n_cells)` (`run_context_builder.py:907-922`). Every GUI run
  now starts dry regardless of the IC widgets. `simulation_worker.py:343`
  consumes `ctx.h0` directly.
- **BC type/value/relax arrays.** Pre-fix, `bc_tp`/`bc_vl`/`bc_relax` came
  from `_collect_boundary_arrays` (`studio_dialog.py:1799-1818`), which
  applies live BC-layer overrides and the `default_bc_type_combo` widget.
  Post-fix they come from the canonical builder's `default_bc_for_edges`
  with `default_bc_type=1` (`run_context_builder.py:643-650`); the
  GPKG-override path (`bc_lines`) is dead in GUI runs because the adapter
  never sets `spec["bc_lines"]`, and `default_bc_type_combo` is not
  forwarded either.
- **`side_hydrographs`, `edge_hydrographs`, `edge_group_overrides`.** Built
  from live layers by `_collect_bc_layer_hydrographs` etc.; no longer
  forwarded and not reproducible from the spec the adapter emits → all
  empty. `simulation_worker.py:357-359` computes `dynamic_bc` from
  `ctx.bc_tp`/`ctx.edge_hydrographs`, so time-series BCs are disabled
  outright in GUI runs.

The builder already honors all of these via `_override`
(`run_context_builder.py:1002-1012`) and the keys are in
`_VALID_SPEC_KEYS` — the adapter simply never puts them in the spec.
Pre-fix behavior confirmed from the removed `replace(...)` block in
`git show 673b714`.

Side effect: `SWE2DRunDataBuilder.build()` still computes all of these
arrays (including real QgsVectorLayer I/O for hydrographs) and the
controller now consumes only `run_input.n_mann_cell`
(`run_controller.py:204`) — dead work per run. The
`run_data_builder.py:20-22` docstring's claim that "`h0` … flow through
the canonical builder's own GPKG path" is wrong: the builder's `h0` comes
from spec scalars, not the GPKG.

### N-2 — MEDIUM (pre-existing, still live; not caused by the fixes)

Request-level time overrides are dropped. The controller resolves
`request.run_duration_text` / `request.output_interval_text` into
`widget_state["run_duration_s"]` / `["output_interval_s"]`
(`run_controller.py:249-266, 278-279`), but the adapter recomputes both
from the raw widget text (`run_time_edit` / `output_interval_edit`,
`run_context_adapter.py:103-106`) and never reads the controller's values.
Same behavior at `673b714^`, so not a new regression — but the C-3 commit
message's parity claim does not hold for request-driven (replay) GUI runs.

### N-3 — LOW-MEDIUM (pre-existing; the C-3 commit half-touched it)

Key mismatch on two widget_state entries: the controller writes
`widget_state["uniform_inflow_enabled"]` and
`widget_state["rain_update_interval_s"]` (`run_controller.py:280-281`),
but the adapter reads the underscored names `_uniform_inflow_enabled` /
`_rain_update_interval_s` (`run_context_adapter.py:151-152`). Both values
silently fall back to defaults (`False` / `60.0`). Commit `673b714` edited
these exact lines (fixed the spec-side key from `_uniform_inflow_enabled`
to `uniform_inflow_enabled`) but left the read-side mismatch, so the
values still never flow.

### N-4 — LOW (test quality; H-1/H-2 verification is illusory)

Both H-1/H-2 tests are tautological:

- `tests/test_cli.py:205-246` (`_build_h1_helpers`) **re-implements**
  `_write_status`/`_progress_wrapper` inside the test file ("Mirrors the
  structure in headless_runner.execute_run") and only calls
  `headless_runner._atomic_write_json` from the real module. The test
  passes even if the shipped runner is wrong.
- `tests/test_cli.py:249-271` (`test_progress_callback_signature_h2`)
  defines its own local `_progress_wrapper` and asserts against it; it
  exercises no production code.

Additionally the tests effectively never run: `tests/test_cli.py` is
pytest-style, so `python -m unittest tests.test_cli` collects 0 tests,
and pytest collection aborts headless on the plugin-root `__init__.py`
import (see Test runs). The real H-1/H-2 behavior was verified by reading
`headless_runner.py` directly (correct), but the regression tests need to
be rewritten against the real `_progress_wrapper`/`_write_status` (e.g.
extract them to module level or drive `execute_run` with a fake
`execute_swe2d_headless`).

### N-5 — LOW (docstring accuracy)

`headless_runner.py:64-68`: "`pct` is the 0-100 progress percentage
carried in the legacy ``step`` field. Both are written so legacy readers
that interpreted ``step`` as percent keep working" — misleading. `step`
now carries the step number, so legacy readers that treated `step` as
percent do **not** keep working (that was the point of H-1). Reword to
say `pct` is the new home for the percent value.

## Sweep results (post-fix diff `58a52bb..HEAD`)

- **Qt leaks:** none new. Only pre-existing lazy, Windows-only
  `from qgis.PyQt.QtCore import QSettings` inside a `try` in
  `swe2d/runtime/backend.py:154` (platform-gated, function-local).
  Import-boundary tests confirm CLI/runtime paths load no Qt.
- **Silent absorbs:** none new. The one added `except Exception` in the
  diff (`run_controller.py:256-257`) logs a warning; it was moved, not
  added.
- **Dead code:** N-1 side effect — `SWE2DRunDataBuilder`'s BC/hydrograph/
  IC outputs are now computed and discarded (only `n_mann_cell` is used).
- **Docstrings:** updated on all touched functions except the two wording
  issues in N-5 and the `run_data_builder.py:20-22` claim noted in N-1.
- **Git mutations:** 9 commits since `58a52bb` = the 8 fix commits plus
  one test commit (`d9d6297`), matching the progress report. No other
  mutations; working tree has only the two untracked review docs.

## Verdict

**NEEDS_FIXES**

C-1, C-2, C-5, and the test-pollution fix are verified clean. H-1/H-2 are
correct in the shipped code and docs, but their regression tests are
tautological and never execute in the canonical runner (N-4). The C-3 fix
achieves its narrow goal (no `dataclasses.replace`; forcing objects flow
through the spec) but introduced a critical GUI regression (N-1): initial
conditions, BC-layer overrides, and hydrograph BCs no longer reach the
`RunContext`, so GUI runs start dry with default BCs and time-series BCs
disabled. Until the adapter forwards `h0`/`hu0`/`hv0` (or the IC widget
scalars), the BC arrays/hydrographs (or a `bc_lines` data-source block),
and `default_bc_type_combo` through the spec, the GUI run path is
functionally broken.
