---
type: reference
status: complete
created: 2026-07-24
completed: 2026-07-25
---

# Comprehensive Review — CLI-First Refactor (Phase 0–1)

**Branch:** `.worktrees/cli-first` (Phases 0, 1.A, 1.B, 1.C)
**Plan:** `docs/CLI_FIRST_REFACTOR_PLAN.md` (5 phases, 0–4)
**Reviewer:** CLI-first refactor review pass
**Date:** 2026-07-24
**Scope:** spec alignment, code/test quality, QGIS import boundary, cross-phase blockers.

> The plan commits to atomic delivery — there is no migration period and no compat
> shims. "Partially done" therefore means the work is **not landed**, just stacked on
> top of the previous layer. The integration gate (Phase 0 replay-equivalence) is
> GPU-gated and could not be executed in the review environment, so the parity claim
> is treated as unverified below.

---

## Spec Alignment Matrix

| # | Plan item | Status | Evidence | Notes |
|---|---|---|---|---|
| 0.1 | RunContext diff test exists | **DONE** | `tests/test_run_context_parity.py:1-690` | GPU-gated; allowlist is unexercised for drainage, Thiessen, edge_groups, sample_lines, hydrographs (lines 136-174). |
| 0.2 | Replay-equivalence test exists | **DONE** | `tests/test_cli_gui_replay_parity.py:1-153` | Patches known divergent fields from live widgets (lines 68-88), so the test is not asserting unmodified export equivalence. |
| 0.3 | Stale coupling-test helper fixed | **DONE** | `tests/_swe2d_test_helpers.py:529-540` | Inline form still drifts; helper now uses `{nodes_layer,...}` form. |
| 0.4 | `FallbackTracker` on CLI builder | **DONE** | `tests/test_run_context_parity.py:230-239, 416-417` | Watches selected loggers only — does not cover all silent-`return None` paths. |
| 1.1.1 | `swe2d-run/2` spec schema | **PARTIAL** | `docs/RUN_SPEC_SCHEMA.md` exists; `_normalize_spec` rejects unknown top-level keys (`run_context_builder.py:404-443`) | Schema file present in the worktree; nested blocks (`params`, `mesh`, `results`, `units`, `data_sources`) are **explicitly not validated** (see `run_context_builder.py:415-422` skip-list and `test_run_context_builder.py:171-195` test asserting `params.unknown_param_key` is accepted). The plan requires fail-fast nested validation. |
| 1.1.2 | Type-mismatch errors | **PARTIAL** | `run_context_builder.py:348-354` docstring claims `TypeError` for present scalar with wrong type | Only the `mesh` field has explicit type validation (`run_context_builder.py:357-368`). All other scalars are coerced via `bool(...)`, `int(...)`, `float(...)` at `run_context_builder.py:884-929`, so `bool("false") == True`, `int("1.5")` truncates, `float("not-a-number")` raises deep in `RunContext.__init__`. No tests assert type-mismatch behaviour for scalars. |
| 1.1.3 | Single defaults table | **DONE** | `run_context_builder.py:127-202` | `dt_cfg` is 0.05 in the table; `from_replay_json` (line 247) and `from_widget_params` (line 284) both reach the same defaults. The 0.2/0.05 split is closed. |
| 1.2.1 | One canonical `build_run_context` | **DONE** | `run_context_builder.py:450-993` | All four constructors (canonical, `from_replay_json`, `from_widget_params`, `RunContextBuilder.build()`) eventually call `build_run_context` after normalization. |
| 1.2.2 | `from_replay_json` is a thin normalizer | **DONE** | `run_context.py:247-281` | Delegates to `build_run_context` after re-shaping the payload. |
| 1.2.3 | `from_widget_params` is a thin normalizer | **NOT DONE** | `run_context.py:284-364` | Still a hand-rolled RunContext constructor that bypasses `build_run_context`. It uses `_DEFAULTS` for every scalar (line 293 onward) and does **not** load mesh, GPKG data, or any data source. It is reachable through `batch_simulation_dialog:294` (per the audit). The plan explicitly says "RunContext.from_replay_json and from_widget_params re-expressed as thin normalizers into it" — only the first half is done. |
| 1.3.1 | `run_controller._build_run_context` calls the builder | **PARTIAL** | `run_controller.py:147-326` | It does call `build_run_context_from_gui` (line 278), but the **arrays** the canonical builder computes are immediately discarded via `dataclasses.replace(...)` (lines 293-325), which re-derives them from `run_input` and `run_options`. The canonical builder's mesh array path is therefore dead in production for the GUI. |
| 1.3.2 | `SWE2DRunDataBuilder`/`SWE2DRunOptionsBuilder` retired | **NOT DONE** | `run_data_builder.py:1-107` and `run_options_builder.py:1-218` still define the builders, both are still wired in the controller at `run_controller.py:180, 187-202` | The plan (§1.3) calls these out by name: "the dialog-bound `SWE2DRunDataBuilder`/`SWE2DRunOptionsBuilder` callback wiring is retired". They are not retired. |
| 1.3.3 | GUI passes live layers via same API as CLI | **NOT DONE** | `run_controller.py:184-185, 187-202` | GUI uses `view._build_spatial_cn_array`, `view._apply_timeseries_bc_values`, etc. (callback-bound); CLI calls `qgis.core.QgsVectorLayer` inside `_build_drainage_config_from_gpkg_layers`. Different code paths. |
| 1.3.4 | Phase 1 gate: byte-equal specs | **NOT VERIFIED** | The parity test exists but is allow-listed, GPU-gated, and the fixture deselects every layer combo (`test_run_context_parity.py:380-390`). | See "Cross-Phase Dependencies" below. |
| 2.1 | `swe2d/core/` created; RunContext, executor, builders, adapters moved | **NOT DONE** | `swe2d/core/` does not exist (`ls swe2d/core/` → not found). `RunContext` still at `swe2d/workbench/workers/run_context.py:14`. `build_run_context` at `swe2d/runtime/run_context_builder.py:450`. | Plan §2.1 calls for atomic move with zero shims. None of the moves have happened. |
| 2.1 | `batch_simulation_dialog`'s lazy import of RunContext inverted | **NOT DONE** | `run_context.py:294` still imports `swe2d.workbench.dialogs.batch_simulation_dialog` from the workbench class — the same import the plan flagged. | `from_widget_params` is the carrier; it has to stay where the dialog can import it. |
| 2.2 | `SimulationWorker._execute` split into `core/executor.py::execute_run(ctx, sink)` | **NOT DONE** | `swe2d/workbench/workers/simulation_worker.py:322` still owns the body. `headless_executor.py:84` calls `SimulationWorker._execute(adapter)`. | The plan requires extracting into a sink-protocol function. The adapter (`HeadlessWorkerAdapter`) is a stop-gap, not the sink protocol. |
| 2.3 | `runtime/backend.py:154` `QSettings` read replaced | **NOT DONE** | `swe2d/runtime/backend.py:154` still has `from qgis.PyQt.QtCore import QSettings`. | Direct violation of the Phase 2 import-boundary rule. |
| 2.4 | `headless_executor.py` deleted | **NOT DONE** | `swe2d/cli/headless_executor.py` (236 lines) still present; imports `SimulationWorker` from `swe2d.workbench.workers.simulation_worker:74`. | |
| 2.5 | `workbench/workers/__init__.py:2` eager import fixed | **NOT DONE** | `swe2d/workbench/workers/__init__.py:1-7` still eagerly imports `SimulationWorker` (which imports `qgis.PyQt.QtCore` at `simulation_worker.py:13`). | Any code that does `import swe2d.workbench.workers` (and the CLI execution path does — via `headless_executor.py:74`) loads Qt. |
| 2.5 | Import-boundary CI test | **NOT DONE** | No such test under `tests/`. The Phase 0 gate did not deliver it. | |
| 3.1 | Raw-sqlite3 Thiessen deleted | **NOT DONE** | `run_context_builder.py:629` still imports `build_forced_thiessen_from_gpkg`; `gpkg_adapter.py:420-509` is the raw sqlite3 path. The QGIS-based shim `build_thiessen_rain_cn_forcing_from_gpkg` (`gpkg_adapter.py:686-781`) is still dead. | The parity allowlist at `test_run_context_parity.py:136-145` documents this as unexercised — i.e. not yet a passing parity case. |
| 3.2 | Single drainage builder | **NOT DONE** | `run_context_builder.py:670-700` still has the inline block; `run_context_adapter.py:234-300` still hosts the QGIS-based loader (with a fatal bug, see HIGH-1). Inline-JSON drainage form (`{"nodes":[...], "links":[...]}`) is silently dropped (`run_context_builder.py:673` only matches `nodes_layer`). | |
| 3.3 | `edge_groups_dict` / `sample_map_data` wired into RunContext | **NOT DONE** | `run_context_builder.py:790-807` loads `edge_groups_dict`; `:810-826` loads `sample_map_data`; `:991` writes `edge_groups={}`; `:989` writes `sample_map_data=[]`. Loaded then discarded. | Plan §3.3 is explicit: "wire or remove the loads." |
| 3.4 | One batch command builder | **NOT DONE** | `cli/batch_runner.py:170-204` (`_run_one`), `cli/batch_runner.py:235-309` (`BatchOrchestrator`), `workbench/workers/batch_worker.py:113-134` (`_build_command`) — three divergent builders. Plan §1.49-51 / §3.4 calls this out by line. `BatchOrchestrator` is reachable only from `batch_runner`; documented as a likely delete. | |
| 3.5 | Dialog method purge | **NOT DONE** | `studio_dialog.py` still owns `_apply_timeseries_bc_values` (`:2089`), `_distribute_total_flow_to_unit_q` (`:2109`), `_sample_line_metrics` (`:2172`), etc. The plan calls for ~120 methods to be deleted. None of them are. | |
| 3.6 | 8 dead `gpkg_adapter.py` functions removed | **NOT DONE** | 7 of 8 still defined: `query_sample_lines_from_qgis` (`:70`), `query_bc_arrays` (`:124`), `build_thiessen_rain_cn_forcing_from_gpkg` (`:686`), `build_pipe_network_config_from_gpkg` (`:783`), `build_initial_state_from_json` (`:1020`), `read_drainage_config_from_gpkg` (`:1046`), `load_and_configure_hydrographs` (`:1316`), `load_hydrograph_edge_data` (`:1454`). The line count is 1575 (no reduction from the plan's "roughly halved"). | |
| 3.7 | Config round-trip symmetric | **NOT DONE** | CLI writes scalar-attr `widget_state` via `_serialize_ctx_for_config` (`headless_executor.py:216-235`); GUI saves versioned widget state via `run_controller.py:787-817` and `project_settings_bridge.collect_workbench_widget_state`. Two non-overlapping shapes. | |
| 4.1 | `_execute()`-entry validation | **NOT DONE** | `run_context_builder.py:982-985` still assigns no-op lambdas to `apply_timeseries_bc_values`, `distribute_total_flow_to_unit_q`, `apply_external_sources`, `build_line_sampling_map`. `tests/test_run_context.py:40-43` still asserts the no-op behaviour. | |
| 4.2 | Typed errors for silent absorbs | **NOT DONE** | The silent absorbs enumerated in plan §1 are still present: `query_mesh_from_gpkg:66-67` `except Exception: return None`; `run_context_builder.py:313-314, 601-602, 622-623, 666-668, 699-700, 717-718, 725, 736-737, 752-754, 759-760, 773-774, 806-807, 825-826, 840-841` are all `try/except` → log → None. | |
| 4.3 | JSON-schema validation at builder entry with "did you mean" | **PARTIAL** | Top-level unknown keys raise `ValueError` with suggestions (`run_context_builder.py:438-443`). No JSON schema. Nested keys (incl. `params.*`, `mesh.*`, `results.*`, `units.*`) are not validated. The test at `test_run_context_builder.py:171-195` actually asserts nested unknown keys are accepted. | |
| 4.4 | Status-file `step` field receives percent (bug) | **NOT DONE** | `headless_runner.py:149-150` still writes `_status_step[0] = pct` then `_write_status` writes `payload["step"] = _status_step[0]`. The doc at `CLI_GUIDE.md:139` says `"step": 1234`. | |
| 4.4 | `progress_callback` signature drift | **NOT DONE** | `headless_runner.py:151-152` calls `progress_callback(pct, {})` (percent + empty diag). The public doc claims `progress_callback=lambda t, d: ...` where `t` is sim-time (`CLI_GUIDE.md:194`). | |
| 4.5 | `CLI_GUIDE.md` updated to "requires `qgis.core`; no QGIS GUI, iface, or display needed" | **NOT DONE** | `docs/CLI_GUIDE.md:6-9` still claims "no QGIS or Qt dependency". The plan §4.5 explicitly requires rewording. | |

---

## Phase Coverage Summary

### Phase 0 (equivalence gate)
- Parity test scaffolding exists and runs (37 builder tests pass; the parity test itself is GPU-gated and unexercised in the review).
- The allowlist is large and contains entries that are **documented but not actually exercised**, so a "passing" parity test does not establish parity for the highest-risk data sources (drainage, Thiessen, edge_groups, sample_lines, hydrograph BCs).
- The replay-equivalence test patches divergent fields from live widgets rather than asserting on the exported payload; the patches are themselves a known-divergences allowlist.

### Phase 1.A (canonical builder + defaults + normalization)
- **Substantially done.** Single `_DEFAULTS` table, `_normalize_spec` rejects unknown top-level keys, `build_run_context` is the canonical entry point, `RunContextBuilder` fluent API works.
- **Missing:** nested-key schema validation, scalar type validation, `from_widget_params` is not a thin normalizer.

### Phase 1.B (GUI flip)
- **Not done in any meaningful sense.** The controller still calls `SWE2DRunDataBuilder.build()` (`run_controller.py:180`) and `SWE2DRunOptionsBuilder.build()` (`:187-202`) and uses `dataclasses.replace` to overwrite the canonical builder's mesh arrays, forcing configs, and coupling. The canonical builder's array path is therefore dead in production. The plan's "byte-equal specs" gate is not met.

### Phase 1.C (test coverage for Phase 1.A)
- **Partially done.** 37 tests cover `_normalize_spec`, `_DEFAULTS`, `RunContextBuilder.from_defaults` / `merge_context` / `with_params` / `build()`, and the `build_run_context_from_dict` wrapper.
- **Missing:** tests for nested unknown keys (the existing test asserts the opposite of fail-fast), type mismatches, GUI-flip equivalence, no-op-callback behaviour, real-data-source parity, status-file semantics, and the CLI entry point.

### Phase 2 (GUI-free core)
- **Entirely skipped.** No `swe2d/core/` directory, no atomic moves, `runtime/backend.py:154` still uses `QSettings`, `headless_executor.py` still exists, no import-boundary test, no `executor.py` extraction.

### Phase 3 (dedup)
- **Mostly skipped.** The drift allowlist still lists every issue from plan §1 as unfixed. `gpkg_adapter.py` is unchanged. The three batch builders still coexist. Dialog solver methods are still on the dialog.

### Phase 4 (fail-fast + hardening)
- **Entirely skipped.** No `_execute()` validation, no JSON schema, no silent-absorb removal, status-file bug unfixed, progress-callback drift unfixed, `CLI_GUIDE.md` not updated.

---

## QGIS Dependency Audit

### Direct forbidden imports in `swe2d/runtime/` and `swe2d/cli/`

| Source | Line | Forbidden import | Recommendation |
|---|---|---|---|
| `swe2d/runtime/backend.py` | 154 | `from qgis.PyQt.QtCore import QSettings` | Replace per plan §2.3 (config/env lookup). |

No direct `qgis.gui`, `qgis.utils`, `PyQt5`, or `iface.` matches in `swe2d/cli/`, `swe2d/core/`, or `swe2d/runtime/`.

### Transitive forbidden imports (CLI execution path)

| Source | Line | Import | Notes |
|---|---|---|---|
| `swe2d/workbench/workers/simulation_worker.py` | 13 | `from qgis.PyQt.QtCore import QThread, pyqtSignal` | Reached by `headless_executor.py:74` via `swe2d.workbench.workers.simulation_worker`. |
| `swe2d/workbench/workers/persistence_worker.py` | 6 | `from qgis.PyQt.QtCore import QThread, pyqtSignal` | Reached by `swe2d/workbench/workers/__init__.py:7` (eager). |
| `swe2d/workbench/workers/__init__.py` | 1-7 | Eager import of `SimulationWorker` and `PersistenceWorker` | `import swe2d.workbench.workers` loads Qt. |

The review document at `docs/REFACTOR_PHASE1_REVIEW.md:64-68, 151` claims "Can CLI run without QGIS? ✅ Yes" and "no Qt in the CLI path" — **both are false for the actual `python -m swe2d.cli run` execution path.** The CLI imports `execute_swe2d_headless` which imports `HeadlessWorkerAdapter` from `headless_executor.py:74` which imports `SimulationWorker` from `swe2d.workbench.workers.simulation_worker` which imports `qgis.PyQt.QtCore`. Until that chain is broken, the CLI is not Qt-free in practice.

### Cross-package direction violations (core should not depend on workbench)

| Source | Line | Imports | Why it violates §2.5 |
|---|---|---|---|
| `swe2d/cli/gpkg_adapter.py` | 820 | `from swe2d.workbench.services.pipe_network_service import build_pipe_network_config` | Plan §3 says these services belong in `swe2d/core/` (or `swe2d/services/`). |
| `swe2d/cli/gpkg_adapter.py` | 872 | `from swe2d.workbench.services.structure_config_service import ...` | Same. |
| `swe2d/cli/gpkg_adapter.py` | 923 | `from swe2d.workbench.services.constants_service import ...` | Same. |
| `swe2d/cli/headless_executor.py` | 74 | `from swe2d.workbench.workers.simulation_worker import SimulationWorker` | Plan §2.2: executor body moves to `swe2d/core/executor.py`. |
| `swe2d/runtime/run_context_builder.py` | 21 | `from swe2d.workbench.workers.run_context import RunContext` | Plan §2: RunContext moves to `swe2d/core/run_context.py`. |
| `swe2d/runtime/run_context_builder.py` | 676 | `from swe2d.workbench.adapters.run_context_adapter import _build_drainage_config_from_gpkg_layers` | Plan §2: adapters move to `swe2d/core/` (with explicit-layer signatures). |
| `swe2d/runtime/run_context_builder.py` | 609, 656, 708, 794, 815 | `from swe2d.cli.gpkg_adapter import ...` | Dependency direction inverted — core (`runtime`) reaches into `cli`. Per plan §3, these helpers belong in `swe2d/core/` or `swe2d/services/`, not `cli/`. |

---

## Code Quality Findings

### CRITICAL

**C-1 — `_build_drainage_config_from_gpkg_layers` always returns `None` (silent drainage failure)**
- **Location:** `swe2d/workbench/adapters/run_context_adapter.py:254-262`
- **Defect:** The `try` block imports `_QgsVectorLayer` from `qgis.core` — a name that does not exist in `qgis.core` (the real class is `QgsVectorLayer`). The `except ImportError` always fires, so the function returns `None` for every call. The `try` is followed by a `noqa: F401` "guard only" import plus a "placeholder" comment that the author left in.
- **Effect:** Every spec that configures `drainage.nodes_layer` produces `pipe_network_cfg = None` and silently drops the drainage network. This is the exact pattern the plan declared it was removing.
- **Fix:** Replace the malformed import with `from qgis.core import QgsVectorLayer`, use the real symbol, and add the missing drainage config keys (see C-2). Also add a unit test that builds a fixture GPKG with two layers, exercises the builder, and asserts a non-`None` config is produced.

**C-2 — Drainage config dict missing 5 keys vs GUI parity**
- **Location:** `swe2d/runtime/run_context_builder.py:688-696`
- **Defect:** The CLI passes only 7 keys (`solver_mode`, `coupling_substeps`, `gpu_method`, `head_deadband`, `dynamic_relaxation`, `implicit_iters`, `implicit_relax`). The GUI passes 12 — see `studio_dialog.py:2030-2046`. Missing: `friction_method`, `surcharge_method`, `recon_method`, `time_integrator`, `friction_alpha`. These are consumed by `pipe_network_service.py:1136-1140`.
- **Effect:** CLI coupling results are silently incorrect vs GUI for any project that uses non-default solver settings. Defaulting at the consumer is a workaround that masks the parity gap.
- **Fix:** Add the 5 missing keys to the CLI config dict, sourced from `_DEFAULTS` or `_v(...)`. Add a parity test that diffs the GUI's `pipe_network_cfg` against the CLI's for the same drainage layer spec.

**C-3 — Phase 1.B "flip" is structurally broken; canonical builder arrays are dead in GUI**
- **Location:** `swe2d/workbench/controllers/run_controller.py:147-326`
- **Defect:** The controller calls `build_run_context_from_gui` (line 278), then **immediately** overrides every mesh array, forcing object, and coupling object via `dataclasses.replace(...)` (lines 293-325). The canonical builder's mesh/forcing/coupling computation is therefore dead in production.
- **Effect:** The Phase 0 parity gate cannot prove parity because the canonical builder's array path is never exercised in the GUI. If a future change fixes a bug in the canonical builder's mesh loading, the GUI will not pick it up — it will keep using `run_input.node_x`, etc.
- **Fix:** Either (a) feed the GUI's `_build_pipe_network_config` and `_build_hydraulic_structure_config` results back into the spec and let the canonical builder set them, or (b) retire `SWE2DRunDataBuilder` and `SWE2DRunOptionsBuilder` entirely and have the controller pass only scalar widgets + already-built forcing objects into a single spec, leaving arrays to the builder. Plan §1.3 mandates the second.

**C-4 — Drainage inline-JSON form silently dropped**
- **Location:** `swe2d/runtime/run_context_builder.py:672-700`
- **Defect:** The condition only matches `isinstance(drainage_cfg, dict) and "nodes_layer" in drainage_cfg`. The inline form `{"nodes": [...], "links": [...]}` is silently ignored.
- **Effect:** Anyone following `tests/_swe2d_test_helpers.py:478-523`'s example (or the docs' example at `CLI_GUIDE.md:98-103` if rewritten to support the inline form) will get a `RunContext` with no drainage, no error, and the test will appear to pass for the wrong reason.
- **Fix:** Either support the inline form via `swe2d/extensions/drainage_network.py:66` builder, or raise a typed `ValueError` naming the form. The plan §3.2 requires a decision: "decide, no third silent state."

**C-5 — `headless_executor.py` still imports Qt, contradicting the review's "no Qt in CLI" claim**
- **Location:** `swe2d/cli/headless_executor.py:74-76` imports `swe2d.workbench.workers.simulation_worker`, which imports `qgis.PyQt.QtCore`. `swe2d/workbench/workers/__init__.py:2` eagerly imports that worker.
- **Defect:** The chain is `import swe2d.cli` → `__main__` → `headless_runner` → `headless_executor` → `swe2d.workbench.workers` (eager) → `simulation_worker` (Qt).
- **Effect:** `python -c "import swe2d.cli"` fails without Qt; the plan's "GUI-free CLI" claim is false. The review document (`docs/REFACTOR_PHASE1_REVIEW.md:64-68, 151`) over-claims and is unsafe to cite.
- **Fix:** Either (a) follow plan §2.2/§2.4: extract the executor body, delete `headless_executor.py`, and have `headless_runner.py` call `swe2d.core.executor.execute_run(ctx, sink)`, or (b) fix the review's wording and add a CI import-boundary test that fails today.

### HIGH

**H-1 — Status-file `"step"` field receives percent instead of step number**
- **Location:** `swe2d/cli/headless_runner.py:149-150`
- **Defect:** `_progress_wrapper(pct)` sets `_status_step[0] = pct` where `pct ∈ [0, 100]`. `_write_status` then emits `payload["step"] = _status_step[0]`. The doc (`CLI_GUIDE.md:139`) says `"step": 1234` (timestep number). Batch tooling that reads the status file will misinterpret the value.
- **Fix:** Compute the actual step number from the solver context and write that. If unavailable, omit the field during `running` and write a real step on `done`.

**H-2 — `progress_callback` signature drift**
- **Location:** `swe2d/cli/headless_runner.py:151-152`
- **Defect:** The implementation invokes `progress_callback(pct, {})` — a percent plus an empty diagnostic dict. The documented public API (`CLI_GUIDE.md:194`) is `progress_callback=lambda t, d: print(f"t={t:.2f}  dt={d['dt']:.4f}")` where `t` is sim-time and `d` is a diagnostics dict.
- **Fix:** Either build a proper diagnostic dict (`{"dt": current_dt, "wet_cells": n, "elapsed_s": wallclock}`) and pass sim-time, or update the public doc to match the implementation. The plan §4.4 calls for fixing the implementation.

**H-3 — No-op callback lambdas at the builder**
- **Location:** `swe2d/runtime/run_context_builder.py:982-985`
- **Defect:** `apply_timeseries_bc_values`, `distribute_total_flow_to_unit_q`, `apply_external_sources`, `build_line_sampling_map` are assigned `lambda *a, **k: None`. These are exactly the callbacks the plan §4.1 says should be validated at `_execute()` entry.
- **Effect:** A CLI run with hydrograph BCs or sample lines will appear to complete but produce no line results, no BC timeseries, and no error. The parity allowlist (`test_run_context_parity.py:191-213`) documents this as a "behavioural gap, not fixed in Phase 0."
- **Fix:** Add `_execute()` entry validation per plan §4.1; add a test that asserts a non-callable or no-op callback raises.

**H-4 — `edge_groups_dict` and `sample_map_data` loaded then discarded**
- **Location:** `swe2d/runtime/run_context_builder.py:790-807, 810-826, 989, 991`
- **Defect:** The builder spends 30+ lines loading `edge_groups_dict` from the GPKG, then writes `edge_groups={}`. Same for `sample_map_data`.
- **Effect:** The plan §3.3 lists this as a known silent drift. The code is misleading because it looks implemented in logs.
- **Fix:** Either wire the values into `RunContext` (preferred — GUI behaviour is the reference, per the plan) or remove the loads. Pick one.

**H-5 — CLI still uses raw-sqlite3 Thiessen builder (known divergent)**
- **Location:** `swe2d/runtime/run_context_builder.py:629-649` → `gpkg_adapter.py:420-509`
- **Defect:** The plan §1.32-36 and §3.1 documents the cell→gauge mapping mismatch. The QGIS-based shim `build_thiessen_rain_cn_forcing_from_gpkg` is dead.
- **Effect:** Rainfall-driven CLI runs produce different cell-averaged forcing than the GUI for the same project.
- **Fix:** Route the CLI's `rain_cn` source through the QGIS-based shim; delete `build_forced_thiessen_from_gpkg`.

**H-6 — `_serialize_ctx_for_config` does not share with `widget_state_to_flat_params`**
- **Location:** `swe2d/cli/headless_executor.py:216-235`
- **Defect:** The serializer iterates `dir(ctx)`, filters dunders, and keeps only primitive scalars. The GUI's equivalent (`run_context_builder.py:239-316`) walks a versioned `widget_state`. The two produce different shapes for the same context.
- **Effect:** A CLI-saved config cannot be loaded back into the GUI (plan §3.7 explicitly calls this out).
- **Fix:** Define one canonical serialization that both paths use. Either `widget_state_to_flat_params` accepts a `RunContext` and produces the versioned shape, or its inverse accepts the scalar shape and produces the versioned shape. The current "two serializers" path is the third silent state.

**H-7 — Config round-trip asymmetry**
- **Location:** `run_controller.py:787-817` vs `headless_executor.py:216-235`
- **Defect:** GUI saves a versioned `widget_state` with `{version: 1, widgets: {...}}`; CLI writes scalar-attr flat dicts.
- **Effect:** Configs are not portable between the two paths. Plan §3.7 calls for "one direction only" — not done.
- **Fix:** See H-6.

**H-8 — `__main__.py` writes raw traceback to user-specified status file**
- **Location:** `swe2d/cli/__main__.py:16-23` (per audit)
- **Defect:** No policy on what the status file can carry. Path and env data and sensitive exception content could leak to an untrusted consumer of the status file.
- **Fix:** Sanitize exception text, log the full traceback to a private file, and write a small `{error: "type: short"}` to the status file.

**H-9 — `_HeadlessFinalizationView.results_data` always returns `None`**
- **Location:** `swe2d/cli/headless_executor.py:212-213`
- **Defect:** The view's `__init__` stores `self._results_data = results_data`. The value comes from `getattr(adapter, "_results_data", None)`, but `adapter._results_data` is never set anywhere in the file — `SimulationWorker._execute` writes to `worker._results_data`, not to the headless adapter.
- **Effect:** Pipe-cell persistence silently skips in CLI runs.
- **Fix:** Have the headless adapter expose a property that reads the worker's `_results_data` (or the run finalizer) after `_execute()` returns, then pass it to the view.

### MEDIUM

**M-1 — `output_interval_s` default is `run_duration_s` (likely copy-paste bug)**
- **Location:** `swe2d/runtime/run_context_builder.py:876-879`
- **Defect:** `output_interval_s = float(_v("output_interval_s", float(_v("run_duration_s", _DEFAULTS["output_interval_s"]))))` — if the spec has no `output_interval_s`, the default is the run duration in seconds, not the table default `1.0`. A 24-hour run with no output interval would emit a single sample at `t=86400`.
- **Fix:** Replace the chained default with `_v("output_interval_s", _DEFAULTS["output_interval_s"])`. Add a test that asserts the default is `1.0` for a spec with no `output_interval_s` and a non-zero `run_duration_s`.

**M-2 — `swe2d/run/2` spec schema exists as `docs/RUN_SPEC_SCHEMA.md` but is not enforced**
- **Location:** `swe2d/runtime/run_context_builder.py:404-443`
- **Defect:** Only top-level keys are validated. Nested blocks (`params`, `mesh`, `results`, `units`, `data_sources`, `rain_cn`, `hyetograph`, `drainage`, `structures`, `sample_lines`, `bc_lines`, `internal_flow_sources`, `storm_areas`, `infiltration_method`) are explicitly skipped (`:415-422`) and the test at `test_run_context_builder.py:171-195` asserts they are accepted.
- **Effect:** A typo inside `params.cfl` (e.g. `cfl_`) silently disappears; the build proceeds with the table default.
- **Fix:** Validate each nested block against the schema, or document the permissive blocks as legacy and test only those.

**M-3 — `from_widget_params` is not a thin normalizer**
- **Location:** `swe2d/workbench/workers/run_context.py:284-364`
- **Defect:** Hand-rolled `RunContext(...)` constructor that bypasses `build_run_context`. It uses `_DEFAULTS` for every scalar but does not load mesh, GPKG data, or any data source. Reachable through `batch_simulation_dialog:294`.
- **Effect:** Consumers downstream must guard against empty `cell_areas`/`node_x` — undocumented. The Phase 0 0.2-vs-0.05 `dt_cfg` bug is closed, but a similar drift could re-appear in any field the hand-rolled constructor omits.
- **Fix:** Replace with a thin call to `build_run_context({...})` (the canonical builder is already safe for empty-mesh specs — see `test_build_run_context_produces_runcontext`).

**M-4 — `RunContextBuilder.merge_context` uses `val != _DEFAULTS.get(name)` for "is it set?"**
- **Location:** `swe2d/runtime/run_context_builder.py:1072-1077`
- **Defect:** Equality with the table default is used to decide whether a field was user-set. If a user explicitly sets a field to its default value (e.g. `n_mann=0.035` to match the table), the merge silently drops it. Subsequent `with_params` layers then cannot override the field via the dropped layer.
- **Fix:** Track which fields are explicitly set on the source `RunContext` (e.g. via a "user_touched" set) instead of inferring from value equality.

**M-5 — Three parallel RunContext constructors coexist**
- **Location:** `run_context_builder.py:450` (canonical), `:1144-1164` (fluent), `:1172-1206` (legacy), `run_context.py:247` (`from_replay_json`), `run_context.py:284` (`from_widget_params`)
- **Defect:** Plan §1.25-26 says "RunContext.from_replay_json and from_widget_params re-expressed as thin normalizers into it." Only the first half is done. `build_run_context_from_dict` is a documented deprecation wrapper but `from_widget_params` is not.
- **Fix:** Replace `from_widget_params` with a thin `build_run_context` call. Add a deprecation comment.

**M-6 — Drainage config defaults duplicated from `pipe_network_service`**
- **Location:** `swe2d/runtime/run_context_builder.py:692-695`
- **Defect:** `head_deadband=0.001`, `dynamic_relaxation=0.7`, `implicit_iters=3`, `implicit_relax=0.8` are duplicated from `pipe_network_service.py:1130-1133`. Drift is likely.
- **Fix:** Import the defaults from a single source (e.g. `swe2d.workbench.services.pipe_network_service.DRAINAGE_DEFAULTS`).

**M-7 — `gpkg_adapter.py:820, 872, 923` import from `swe2d.workbench.services.*`**
- **Location:** `swe2d/cli/gpkg_adapter.py:820, 872, 923`
- **Defect:** These three imports violate the §2.5 boundary (`swe2d.core`/`swe2d.cli` may not import `swe2d.workbench.*`). They are also dead in the sense that the CLI is not supposed to build drainage configs from the GUI's services — that's the canonical builder's job.
- **Fix:** Move the services to `swe2d/core/` (or keep them in `swe2d/services/`) and update the imports. Per the plan, this is part of Phase 2.1.

**M-8 — `solver_backend_mode`/etc. coerced via `.strip().lower()` even though `_DEFAULTS` is normalized**
- **Location:** `swe2d/runtime/run_context_builder.py:891-899`
- **Defect:** Cosmetic — `_DEFAULTS` already supplies `"gpu"` / `"cuda"`. The chain is defensive but adds nothing.
- **Fix:** Remove the `.strip().lower()` chain. Add a test that asserts the value is `"gpu"` for an empty spec.

**M-9 — `_HeadlessSignal.connect` overwrites without warning**
- **Location:** `swe2d/cli/headless_executor.py:28-29`
- **Defect:** Not currently exploitable, but no defensive copy or warning when an existing callback is replaced.
- **Fix:** Document the overwrite contract, or warn on overwrite.

**M-10 — `studio_dialog.py` still has inline `import sqlite3`**
- **Location:** `studio_dialog.py:2733-2757` (per audit)
- **Defect:** Plan §1.29 said Phase 2 should clean up. Not done. Lazy import — only runs when called.
- **Fix:** Replace with a `gpkg_persistence_service` call.

**M-11 — `REFACTOR_PHASE1_REVIEW.md` over-claims**
- **Location:** `docs/REFACTOR_PHASE1_REVIEW.md:64-68, 151, 149-157`
- **Defect:** "Can CLI run without QGIS? ✅ Yes" and "no Qt in the CLI path" are false (see C-5). The 37/37 test result is for the builder unit tests, not for the parity gate. The "APPROVED" verdict is unsafe to cite.
- **Fix:** Either deliver the import-boundary claim (do Phase 2), or downgrade the verdict to "APPROVED with conditions" and explicitly mark unverified items.

**M-12 — `KNOWN_DIVERGENCES` line refs are stale**
- **Location:** `tests/test_run_context_parity.py:70-175`
- **Defect:** Cited line numbers do not match the post-refactor source. Example: `:76` cites `runtime/run_context_builder.py:668-669` for `dt_fixed`, but the actual `dt_fixed` default is in `_DEFAULTS` at line 141.
- **Effect:** Anyone debugging will go to the wrong place. Allowlist passes anyway because the comparator only checks key existence.
- **Fix:** Either re-cite against the current source or remove the line numbers.

**M-13 — `widget_state` version not validated at restore**
- **Location:** `swe2d/workbench/controllers/run_controller.py:1144-1148`
- **Defect:** `ws = data.get("widget_state", data)` — if the JSON has no `widget_state` key, the entire dict is treated as widget state. A user exporting a CLI replay JSON that does not contain a `widget_state` key will silently restore against the wrong shape.
- **Fix:** Require an explicit `widget_state` key or version marker; raise on missing.

**M-14 — `db_path` not validated before `_load_logs` (path traversal)**
- **Location:** `swe2d/workbench/controllers/run_controller.py:578-586`
- **Defect:** The user-chosen `db_path` is passed to `load_run_logs_from_geopackage` after only an `os.path.exists` check. No canonicalization, no allow-list. A symlink to a sensitive file would be read.
- **Fix:** Resolve and warn for symlinks; restrict to a project tree or allow-list.

### LOW

**L-1 — Magic numbers in drainage config defaults**
- See M-6. The four `0.001 / 0.7 / 3 / 0.8` values should be named constants.

**L-2 — `_uniform_inflow_enabled` dead adapter key**
- **Location:** `swe2d/workbench/adapters/run_context_adapter.py:138` writes `spec["_uniform_inflow_enabled"]`; builder reads `uniform_inflow_enabled` (no underscore). Documented in `REFACTOR_PHASE1_REVIEW.md:131-134` as known dead code.
- **Fix:** Remove from the adapter.

**L-3 — Two `run_controller.py` files**
- `swe2d/runtime/run_controller.py` (50 lines, preflight) vs `swe2d/workbench/controllers/run_controller.py` (1192 lines). Different module names because the GUI uses the latter. Confusing.
- **Fix:** Rename one to make the role clear (e.g. `run_preflight_controller.py`).

**L-4 — `_validate(... , _mode="cli")` parameter is dead**
- **Location:** `swe2d/runtime/run_context_builder.py:337` declares `_mode: str = "cli"`, but the function never reads it.
- **Fix:** Remove the unused parameter or document the intent.

**L-5 — `widget_state_to_flat_params` uses `_parse_duration` but only HH:MM and fraction**
- **Location:** `swe2d/runtime/run_context_builder.py:261-276`
- **Defect:** Only handles `HH:MM` and strings-without-colon (returned unchanged). Does not handle "1.5h", "30m", ISO 8601, or seconds-as-int. The controller's GUI parser `view._parse_time_hours` handles more.
- **Fix:** Either share a single parser or document the supported formats.

**L-6 — `to_replay_json` writes `"data_sources": {}` literally**
- **Location:** `swe2d/workbench/workers/run_context.py:209`
- **Defect:** Round-tripping through `from_replay_json` (line 269) gives an empty `data_sources`, then `build_run_context` skips all GPKG-derived forcing/coupling. The replay is correct for "no forcing" but is **lossy** for any project that used bc_lines/drainage/hyetograph. Plan §3.3 marked this as a Phase 3 item.
- **Fix:** Either write the data sources into the replay payload (preferred), or document the lossy round-trip.

**L-7 — `RunContext._scalar_val` falls through to `str(value)` for unknown types**
- **Location:** `swe2d/workbench/workers/run_context.py:244`
- **Defect:** An object whose `name` returns the empty string produces `""` in the replay JSON. A non-enum object whose `value` is itself a struct produces a confusing nested repr.
- **Fix:** Raise `TypeError` for unrecognised types in a replay context.

**L-8 — `from_replay_json` re-nests `params`/`results`/`units` under spec roots**
- **Location:** `swe2d/workbench/workers/run_context.py:265-269`
- **Defect:** `setdefault("params", payload.get("params", {}))` — if the payload has a top-level `cfl` and a `params.cfl`, the top-level wins. The replay JSON's `params` is canonical, so this is fine in practice, but the precedence rule is undocumented.
- **Fix:** Add a comment.

**L-9 — Documentation drift in `widget_state_to_flat_params` docstring**
- **Location:** `swe2d/runtime/run_context_builder.py:255-257`
- **Defect:** Says "If `mesh_gpkg` and `mesh_name` are provided, the `units` block is also computed". The function does that, but the docstring doesn't mention the bare-`except`-`pass` swallowing all errors at line 313-314.
- **Fix:** Add a note.

**L-10 — Inconsistent docstring style across the new files**
- `run_context_builder.py` uses NumPy-style `Parameters/Returns/Raises`; `run_context_adapter.py` uses Args-style; `headless_executor.py` has no docstrings on `_HeadlessSignal` or `HeadlessWorkerAdapter` methods. Plan §2's "thin flip layer" is fine, but the docstring style should be uniform.

---

## Test Quality Findings

### Coverage gaps

1. **No test for the canonical builder's mesh array path** — `test_build_run_context_produces_runcontext` (line 187) only checks `cell_areas.size > 0`. The build flow that produces `bc_tp`, `bc_vl`, `internal_flow_forcing`, `pipe_network_cfg`, `cell_source_model`, `rain_rate_model`, `h0`, `n_mann_cell`, etc. is **uncovered** because the parity fixture deselects every layer combo (line 380-390).
2. **No test for the GUI flip equivalence** — there is no test that compares a `RunContext` built from the GUI controller to one built from the canonical builder on the same project. The plan's gate is "byte-equal specs" — there is no test for that.
3. **No test for `_build_drainage_config_from_gpkg_layers`** — the broken function is entirely uncovered. C-1 should have been caught by a unit test.
4. **No test for nested-key validation** — the existing test (`test_run_context_builder.py:171-195`) asserts the **opposite** of fail-fast (nested unknown keys are accepted).
5. **No test for type-mismatch validation** — `test_run_context_builder.py:4-6` claims coverage in the docstring, but the only negative test is for the `mesh` field shape.
6. **No test for the no-op callback behaviour** — Plan §4.1 says "required callbacks must not be the no-op default" and `test_run_context.py:40-43` still asserts the no-op default works.
7. **No test for the CLI entry point** — `tests/test_cli.py` only tests sweep expansion, mesh persistence, and the `_atomic_write_json` helper. It does not exercise `__main__._load_params`, `--status-file-path`, `--status-interval`, `--results`, the `replay` command, or the `batch` command.
8. **No test for `progress_callback` signature** — H-2 would be caught by a single `mock.Mock` callback asserting the args.
9. **No test for status-file `step` semantics** — H-1 would be caught by reading the JSON after a 1-second run.
10. **No test for `from_widget_params`** — the constructor is reachable through `batch_simulation_dialog` but unverified.
11. **No test for the config round-trip** — H-7 would be caught by writing a config in one path and reading it in the other.

### Flaky patterns

- No `time.sleep` in the builder tests (good).
- The parity test is GPU-gated (`test_run_context_parity.py:592`) — fine for the integration gate, but the 37-test "all pass" number does not include parity.
- `FallbackTracker` watches selected loggers only; silent `except: pass` paths (e.g. `widget_state_to_flat_params:313-314`) are not detected.

### Mock fidelity

- `tests/test_batch_runner_orchestrator.py:12-15` defines `_FakeCompleted` with only `returncode`. The implementation contains special-case handling at `batch_runner.py:279-285` for the fake. Real `subprocess.Popen` semantics (poll, stdout, blocking, cancellation, timeout) are not tested.
- No mock for `_results_data`; tests don't verify the `H-9` behaviour.

### Negative tests

- Mesh shape (positive + negative for shape).
- Mesh path existence (positive + negative for non-existent file).
- Unknown top-level key (positive).
- **No** negative tests for: unknown nested key, type-mismatched scalar, missing data source, missing GPKG, missing mesh_name, no-op callback, broken config round-trip, drainage layer not present, batch GPKG reference to non-existent file.

---

## Security / Safety

1. **No `shell=True`** in any of the searched CLI code — good, avoids straightforward shell injection.
2. **Arbitrary user-controlled paths** are accepted for mesh, results, replay, status, batch, and per-source GPKGs. No validation for path traversal, output overwriting input mesh, or symlink targets.
3. **Status file path is user-chosen** — full exception text + traceback are written to it. Sensitive data and environment could leak to an untrusted reader. (See H-8.)
4. **Batch subprocess timeout / cancellation** is not handled robustly. `BatchOrchestrator.cancel()` only terminates active processes; no wait, kill escalation, or cleanup test.
5. **`_load_logs(db_path)`** reads a user-chosen GPKG without symlink check. (See M-14.)
6. **`os.replace` cross-filesystem failure** at `headless_runner.py:23-36` is swallowed silently — only a warning. Plan §4.4 doesn't cover this.
7. **`_build_drainage_config_from_gpkg_layers`** constructs URIs from user-controlled layer names: `f"{drainage_gpkg}|layername={nodes_layer}"`. A malicious `nodes_layer` containing a pipe or new-line could break the URI handling. (Untested; rare in practice because the GPKG picker filters it.)

---

## Documentation Audit

| Doc | Current claim | Reality | Required update |
|---|---|---|---|
| `docs/CLI_GUIDE.md:6-9` | "no QGIS or Qt dependency" | False — CLI execution path imports `qgis.PyQt.QtCore` transitively (C-5). | Plan §4.5: "requires `qgis.core`; no QGIS GUI, iface, or display needed." |
| `docs/CLI_GUIDE.md:139` | `"step": 1234` (timestep number) | False — writes percent (H-1). | Either fix the implementation or document the percent semantics. |
| `docs/CLI_GUIDE.md:194` | `progress_callback=lambda t, d: ...` with sim-time | False — implementation passes `pct, {}` (H-2). | Same. |
| `docs/REFACTOR_PHASE1_REVIEW.md:64-68, 151` | "Can CLI run without QGIS? ✅ Yes" / "no Qt in the CLI path" | False (C-5). | Qualify. |
| `docs/REFACTOR_PHASE1_REVIEW.md:149-157` | "APPROVED" + "37 passed" | 37 is builder unit tests; parity gate is GPU-gated and unexercised. | Mark parity gate as unverified. |
| `docs/CLI_GUI_PARITY_DRIFT.md` | "Phase 1 Status (COMPLETED)" | Drift is still open; same items from plan §1. | Distinguish "normalization complete" from "feature parity complete." |

---

## Cross-Phase Dependencies

These blockers must be resolved before downstream phases can be considered complete.

| Blocker | Why it blocks | Location | Phase it belongs to |
|---|---|---|---|
| `_build_drainage_config_from_gpkg_layers` always returns `None` (C-1) | Any spec with `drainage.nodes_layer` silently produces no drainage config. The Phase 3 drainage dedup cannot land safely. | `run_context_adapter.py:254-262` | Phase 3.2 |
| Drainage config missing 5 keys (C-2) | CLI coupling results are silently incorrect vs GUI. Parity gate is unsatisfiable. | `run_context_builder.py:688-696` | Phase 3.2 |
| GUI flip doesn't use canonical builder arrays (C-3) | The Phase 1 "byte-equal specs" gate cannot be proved because the canonical builder's array path is dead. | `run_controller.py:293-325` | Phase 1.3 |
| Inline-JSON drainage silently dropped (C-4) | `_swe2d_test_helpers.py:478-523` and the test suite are wired to a path that doesn't exist. | `run_context_builder.py:672-700` | Phase 3.2 |
| CLI still imports Qt transitively (C-5) | "GUI-free CLI" claim is false; cannot land Phase 2.5 import-boundary test. | `headless_executor.py:74-76` → `simulation_worker.py:13` | Phase 2.2/§2.4 |
| Status-file `step` semantics wrong (H-1) | Plan §4.4 lists this as a known incidental bug. | `headless_runner.py:149-150` | Phase 4.4 |
| `progress_callback` signature drift (H-2) | Public API contract is wrong. | `headless_runner.py:151-152` | Phase 4.4 |
| No-op callback lambdas (H-3) | Plan §4.1 requires validation at `_execute()` entry. | `run_context_builder.py:982-985` | Phase 4.1 |
| `edge_groups_dict` / `sample_map_data` loaded then discarded (H-4) | Plan §3.3 explicit ask. | `run_context_builder.py:790-826, 989, 991` | Phase 3.3 |
| CLI uses raw-sqlite3 Thiessen (H-5) | Plan §3.1 explicit ask; parity allowlist entry stays open. | `run_context_builder.py:629-649` | Phase 3.1 |
| Config round-trip asymmetric (H-6/H-7) | Plan §3.7 explicit ask. | `headless_executor.py:216-235` + `run_controller.py:787-817` | Phase 3.7 |
| `_HeadlessFinalizationView.results_data` always `None` (H-9) | Pipe-cell persistence silently skips in CLI. | `headless_executor.py:212-213` | Phase 2 followup |
| Nested-key validation absent (M-2) | Plan §3.3 + §4.3 require it; test asserts the opposite. | `run_context_builder.py:415-422` + `test_run_context_builder.py:171-195` | Phase 4.3 |
| `from_widget_params` not a thin normalizer (M-3) | Plan §1.25-26. | `run_context.py:284-364` | Phase 1.A |
| `output_interval_s` default is `run_duration_s` (M-1) | Likely copy-paste bug. | `run_context_builder.py:876-879` | Phase 1.A |

Phase 2 cannot begin in any meaningful way until **C-1, C-2, C-3, C-4, C-5** are fixed (or the plan's "atomic delivery" is abandoned in favour of a phased merge with shims — which the plan explicitly forbids).

Phase 3 cannot complete its dedup work until **C-1, C-2, C-4, H-4, H-5, H-6, H-7** are fixed.

Phase 4 cannot ship its "fail-fast" claim until **H-1, H-2, H-3, M-2** are fixed.

---

## Verdict

**NEEDS_FIXES**

Phase 1.A's builder scaffolding is structurally sound and has good test coverage for what it covers. Phase 1.B's "GUI flip" is not a flip — it stacks the legacy callback builders on top of the canonical builder and overrides its array output, leaving the canonical builder's mesh/forcing/coupling path dead in production. The Phase 1 "byte-equal specs" gate is therefore not satisfiable. Phase 2 is entirely skipped (the `swe2d/core/` directory does not exist, no moves have happened, and the Qt import chain is still live in the CLI path). Phase 3's dedup is mostly undone. Phase 4's hardening is entirely skipped. Several items reported as "DONE" in the existing review document are contradicted by the source.

The implementation can land as Phase 1.A (canonical builder + defaults + normalization + builder tests) once the most dangerous regressions are fixed. Everything else belongs to Phases 2, 3, 4, which are not started.

### What is safe to merge today
- `run_context_builder.py` (canonical builder, `_DEFAULTS`, `_normalize_spec`, `RunContextBuilder`).
- `test_run_context_builder.py` (after fixing the nested-unknown-key test that asserts the opposite of the plan).
- The `WIDGET_TO_RC` fix for `bridge_coupling_mode` (`run_context_builder.py:90`).

### What is NOT safe to merge today
- The GUI controller's `_build_run_context` (C-3) — breaks the "byte-equal specs" gate.
- `_build_drainage_config_from_gpkg_layers` (C-1) — silently drops every drainage config.
- The drainag config dict in `run_context_builder.py:688-696` (C-2) — silently incorrect coupling.
- The review document claims of "GUI-free CLI" and "no Qt in the CLI path" (C-5) — false.
- `_serialize_ctx_for_config` as a "round-trippable" config serializer (H-6) — produces a different shape than the GUI's.
- The 8 dead `gpkg_adapter.py` functions must stay until a CI test pins them as dead, otherwise deletion is unsafe.
- `docs/CLI_GUIDE.md` "no QGIS or Qt dependency" claim (Documentation Audit).

---

## Prioritized Fix List

### CRITICAL (must fix before any merge beyond Phase 1.A)

1. **C-1** — Replace `_QgsVectorLayer` placeholder with `QgsVectorLayer` in `run_context_adapter.py:254-262`; add a unit test that exercises the loader with a real fixture GPKG.
2. **C-2** — Add `friction_method`, `surcharge_method`, `recon_method`, `time_integrator`, `friction_alpha` to the CLI drainage config dict at `run_context_builder.py:688-696`; source from `_DEFAULTS` or a single source in `pipe_network_service.py`. Add a parity test.
3. **C-3** — Make the GUI flip a real flip: either (a) feed the GUI's forcing/coupling objects into the spec dict and let `build_run_context` set them, or (b) retire `SWE2DRunDataBuilder` / `SWE2DRunOptionsBuilder` entirely. The current "canonical builder computes → controller overrides" pattern keeps the parity gate unsatisfiable.
4. **C-4** — Either support the inline `{nodes, links}` drainage form via `swe2d.extensions.drainage_network`, or raise a typed `ValueError` naming the form. No third silent state.
5. **C-5** — Either deliver the "GUI-free CLI" claim by extracting `execute_run(ctx, sink)` and deleting `headless_executor.py` (per plan §2.2/§2.4), or downgrade the review's wording and add a CI import-boundary test that fails today.

### HIGH (must fix before claiming Phase 1 complete)

6. **H-1** — Fix status-file `step` semantics in `headless_runner.py:149-150` (write real step number, not percent).
7. **H-2** — Fix `progress_callback` signature in `headless_runner.py:151-152` (write sim-time and a populated diagnostic dict).
8. **H-3** — Add `_execute()`-entry validation per plan §4.1; raise when `apply_timeseries_bc_values` etc. are the no-op default. Update `test_run_context.py:40-43`.
9. **H-4** — Wire `edge_groups_dict` and `sample_map_data` into `RunContext` per plan §3.3, or remove the loads.
10. **H-5** — Route the CLI's `rain_cn` source through the QGIS-based shim and delete `build_forced_thiessen_from_gpkg`.
11. **H-6** — Make `_serialize_ctx_for_config` share a single serializer with `widget_state_to_flat_params`'s inverse.
12. **H-7** — Make CLI config persistence emit the versioned widget-state shape, or teach restore to accept the scalar shape.
13. **H-9** — Have `HeadlessWorkerAdapter` expose `_results_data` so pipe-cell persistence works in CLI.
14. **M-2** — Validate nested spec blocks (`params`, `mesh`, `results`, `units`, `data_sources`); update the test at `test_run_context_builder.py:171-195` to assert the fail-fast behavior.

### MEDIUM (fix during Phase 2/3 work)

15. **M-1** — Replace the chained `output_interval_s` default with `_DEFAULTS["output_interval_s"]`; add a test.
16. **M-3** — Replace `from_widget_params` with a thin `build_run_context` call.
17. **M-4** — Use an explicit "user_touched" set in `RunContextBuilder.merge_context` instead of value equality.
18. **M-5** — Consolidate the three parallel RunContext constructors (canonical + fluent + legacy) per plan §1.25-26.
19. **M-6** — Centralize drainage config defaults in `swe2d.core` / `swe2d.services`.
20. **M-7** — Move `pipe_network_service`, `structure_config_service`, `constants_service` out of `swe2d.workbench.services` per plan §2.1.
21. **M-8** — Remove the cosmetic `.strip().lower()` chain in `run_context_builder.py:891-899`.
22. **M-10** — Replace `studio_dialog.py:2733-2757` inline `sqlite3` with a `gpkg_persistence_service` call.
23. **M-11** — Downgrade the verdict in `REFACTOR_PHASE1_REVIEW.md` to "APPROVED with conditions"; mark parity gate as unverified.
24. **M-12** — Re-cite `KNOWN_DIVERGENCES` line numbers against the current source.
25. **M-13** — Require an explicit `widget_state` key in restored config JSON.

### LOW (cleanup, can be deferred)

26. **L-1, L-2, L-3, L-4, L-5, L-6, L-7, L-8, L-9, L-10** — see individual findings.

### Documentation (must update with the next phase commit)

27. **Documentation Audit** — Update `CLI_GUIDE.md` per plan §4.5; update `REFACTOR_PHASE1_REVIEW.md` to reflect actual state; mark `CLI_GUI_PARITY_DRIFT.md` "Phase 1 complete" carefully.
