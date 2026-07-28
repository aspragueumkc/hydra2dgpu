---
type: plan
status: complete
created: 2026-07-24
completed: 2026-07-25
---

# CLI-First Refactor Plan — HYDRA2DGPU

> Status: plan. Grounded in a codebase audit dated 2026-07-24 (file:line refs throughout).
> Companion doc: `docs/CLI_GUI_PARITY_DRIFT.md` (drift mechanisms), `docs/HYDRA_MCP_SERVER_PLAN.md` (agent-assisted production use and GUI testing).
>
> **Execution context:** implemented in a separate feature repo and landed as one
> coherent change. There is **no production migration period** and therefore **no
> migration/compatibility shims of any kind** — every call site is updated to the new
> paths in the same change (per `.opencode/rules/NO_PREMATURE_BACKWARDS_COMPAT.md`,
> unshipped API shapes get no compatibility code).

## 1. Problem statement

Early development coupled core solver functionality to GUI elements. The CLI was later
added as a **parallel path** that substitutes JSON keywords for widget values and
filepaths for map-layer combos. Today:

- **Two RunContext assemblers are hand-synced.** GUI: `run_controller._build_run_context()`
  (`swe2d/workbench/controllers/run_controller.py:147-357`) via `SWE2DRunDataBuilder`
  (`swe2d/runtime/run_data_builder.py:63`) and `SWE2DRunOptionsBuilder`
  (`swe2d/runtime/run_options_builder.py:97`) built from dialog-bound callbacks
  (`workbench/controllers/run_component_wiring_controller.py:36-62`). CLI:
  `build_run_context_from_dict()` (`swe2d/runtime/run_context_builder.py:195-781`) — a
  ~590-line parallel implementation of the same field list, bridged by the
  hand-maintained `WIDGET_TO_RC` name map (`run_context_builder.py:35-102`, ~60 entries).
- **Silent-absorb fallbacks are everywhere.** `try/except Exception → warning + None` at
  `run_context_builder.py:365,391,438,490,511,518,547,601,618`; bare `except: pass` at
  `:189,529,569,631`; `ImportError → None` shims in `cli/gpkg_adapter.py:533,614,634,720,869,911,962`;
  `query_mesh_from_gpkg` swallows *all* exceptions → "Mesh not found" (`gpkg_adapter.py:66-67`).
  A missing JSON key produces a plausible-but-wrong run (the failure mode
  `CLI_GUI_PARITY_DRIFT.md:12-16` warns about).
- **Real logic divergence exists today, not just drift risk:**
  - CLI Thiessen rainfall uses a raw-sqlite3 reimplementation
    (`build_forced_thiessen_from_gpkg`, `gpkg_adapter.py:420`) whose cell→gauge mapping
    does **not** match the GUI path; the correct QGIS-equivalent shim
    (`build_thiessen_rain_cn_forcing_from_gpkg`, `gpkg_adapter.py:686`) is dead code.
  - CLI drainage block (`run_context_builder.py:442-491`) is missing keys the GUI passes
    (`friction_method`, `surcharge_method`, `recon_method`, `time_integrator`,
    `friction_alpha` — cf. `studio_dialog.py:2040-2044`).
  - Computed-then-discarded data: CLI loads `edge_groups_dict` and `sample_map_data`
    then assembles the RunContext with hardcoded `{}` / `[]`
    (`run_context_builder.py:587-617` vs `:734,777,779`) — sample lines and edge groups
    silently do nothing in CLI runs.
  - Inline-JSON drainage (`{"nodes": [...], "links": [...]}`) is **silently dropped**
    (`run_context_builder.py:445` only handles the `nodes_layer` form), while
    `tests/_swe2d_test_helpers.py:478-523` passes exactly that form — GPU coupling
    tests appear wired to a path that no longer exists.
- **Duplicated infrastructure:**
  - Batch subprocess command construction exists 3×: `cli/batch_runner.py:170`,
    `cli/batch_runner.py:263` (`BatchOrchestrator` — a second, divergent batch engine),
    `workbench/workers/batch_worker.py:113`.
  - Three dict→RunContext constructors with differing defaults:
    `build_run_context_from_dict` (`dt_cfg` default 0.2), `RunContext.from_replay_json`
    (`run_context.py:247`, default 0.05), `RunContext.from_widget_params` (`:324`).
  - ~800 lines of dead code in `cli/gpkg_adapter.py` (8 uncalled functions, incl. a
    270-line `read_drainage_config_from_gpkg`).
  - ~120 solver-adjacent methods still on the dialog (`studio_dialog.py`: e.g.
    `_apply_timeseries_bc_values` `:2089`, `_distribute_total_flow_to_unit_q` `:2109`,
    `_sample_line_metrics` `:2172`) with pure-logic twins in
    `boundary_and_forcing/bc_logic.py` and
    `workbench/services/runtime_source_application_service.py`.
  - Config round-trip asymmetry: CLI-saved configs use scalar-attr widget_state
    (`headless_executor.py:216-236`); GUI expects widget-name-keyed state
    (`run_controller.py:818-848`) — CLI-saved configs don't restore into the GUI.
- **The core is entangled with the GUI bindings.** `RunContext` lives in
  `swe2d/workbench/workers/run_context.py`, and `workbench/workers/__init__.py:2` eagerly
  imports `simulation_worker`, which has a module-level `qgis.PyQt` import
  (`simulation_worker.py:13`) — so `import swe2d.cli` fails without the QGIS **GUI**
  bindings, even though the compute itself never touches a widget. Shared pure builders
  live under `workbench/services/` (`pipe_network_service.py`,
  `structure_config_service.py`, `constants_service.py`, `non_gui_runtime_service.py`).
  `runtime/backend.py:154` reads `QSettings`. The problem is the *iface/widget* coupling,
  **not** the use of `qgis.core` — see the dependency policy in §2.

**What already converges (keep):** `RunContext` (frozen, Qt-free dataclass with
`to_replay_json`/`from_replay_json`), `SimulationWorker._execute()`
(`simulation_worker.py:322`) as the single execution entry, `_WorkbenchShim`,
`SWE2DRuntimeStepExecutor`, `execute_run_timestep_loop`, backend init, coupling,
finalizer, and `persist_simulation_config` (already stores CLI-consumable flat `params`
alongside widget state — `gpkg_persistence_service.py:233,260-262`). The GUI already
emits a CLI-shaped dict: `collect_data_source_config()` (`studio_dialog.py:2711-2834`)
and `_build_replay_payload()` (`run_controller.py:850-898`).

**Why the last attempt broke the GUI:** it changed the GUI path before an equivalence
gate existed. This plan therefore starts with the gate, not the refactor.

## 2. Goal and non-goals

**Goal:** one canonical, GUI-free, JSON-addressable pipeline — *run spec → RunContext →
executor → persisted results* — with the GUI reduced to a view layer that (a) serializes
widget state into the same run spec, and (b) renders results. Current GUI behavior is
preserved bit-for-bit (proven by replay-equivalence tests, not by inspection).

**Non-goals:** no solver/numerics changes; no GUI redesign; no new CLI features beyond
what parity requires; no CPU fallback; mesh arrays stay GPKG-derived at run time (not
embedded in JSON); **no removal of the QGIS dependency** — QGIS is the intended
pre/post-processing platform, not an accident to engineer away; **no backwards
compatibility with intermediate states** — the refactor lands atomically from the
feature repo.

### QGIS dependency policy (binding constraint for all phases)

QGIS is deliberately the GIS backbone of the package: the average hydraulic modeler's
pre/post-processing is GIS work, and keeping it in QGIS is a core feature. Therefore:

- **The CLI/core may freely use `qgis.core` and the PyQGIS processing API** — including
  native QGIS algorithms that don't depend on the GUI. Most "QGIS" geospatial algs are
  GDAL/OGR/GEOS under the hood; where a native QGIS alg exists, importing it via the
  Python API is fine and *preferred* over maintaining a parallel reimplementation.
- **Parity rule:** when the same geospatial computation exists twice — a QGIS-based
  implementation and a raw sqlite3/GDAL/GEOS reimplementation (e.g. the Thiessen case in
  §1) — the **QGIS-based implementation is canonical** and the reimplementation is
  deleted. One implementation, both paths.
- **The CLI/core must not use `qgis.gui`, `qgis.utils.iface`, or `qgis.PyQt`/PyQt5
  widgets, signals, or threads.** That is the whole of the constraint: *no GUI
  dependencies*, not *no QGIS*.
- Where native **processing** algorithms are needed on the CLI path, initialize a
  minimal headless application context: `QgsApplication(argv, GUI=False)` +
  `initQgis()` + `QgsApplication.processingRegistry()` setup in the CLI entry point.
  This is a library context, not a GUI session — no display, no iface.
- `QgsVectorLayer` / `QgsGeometry` / `QgsFeature` use without any `QgsApplication`
  (the current pattern, `run_context_builder.py:343-346`) remains valid for plain
  data access.

## 3. Target architecture

```
                        ┌────────────────────────────┐
                        │  Run Spec (versioned JSON  │
                        │  schema: swe2d-run/2)      │
                        └─────────────┬──────────────┘
            GUI: widgets → spec       │       CLI/batch/replay: file → spec
   studio_dialog.collect_*  (exists)  │       (exists: __main__._load_params)
                                      ▼
                    swe2d/core/builder.py  build_run_context(spec)
                    · schema validation, fail-fast on unknown keys
                    · single set of defaults (one constructor)
                    · data-source resolution via qgis.core adapters
                                      ▼
                    swe2d/core/run_context.py  RunContext (moved, GUI-free)
                                      ▼
                    swe2d/core/executor.py  execute_run(ctx, callbacks)
                    (today's SimulationWorker._execute logic, GUI-free;
                     progress/log/snapshot via injected callbacks)
                                      ▼
              ┌───────────────────────┴────────────────────────┐
    GUI: SimulationWorker (thin QThread wrapper,      CLI: direct call / subprocess
    signals ↔ callbacks, stays in workbench)          (headless_executor deleted)
```

`swe2d/core/` may import `qgis.core` anywhere. What must **never** appear in
`swe2d/core/` or `swe2d/cli/`: `qgis.gui`, `qgis.utils.iface`, `qgis.PyQt`, `PyQt5`.

Package realignment (moves, not rewrites — all call sites updated in the same change,
no re-export shims):

| From | To | Why |
|---|---|---|
| `workbench/workers/run_context.py` | `swe2d/core/run_context.py` (new pkg) | RunContext is the core hand-off object; today it drags in `qgis.PyQt` via `workers/__init__.py:2` |
| `workbench/services/pipe_network_service.py`, `structure_config_service.py`, `constants_service.py`, `non_gui_runtime_service.py`, `mesh_service.py` | `swe2d/core/` (or `swe2d/services/`) | Pure logic imported by runtime+CLI should not live under `workbench/` |
| `runtime/run_context_builder.py` | `swe2d/core/builder.py` | Becomes the single builder |
| `SimulationWorker._execute` body | `swe2d/core/executor.py` | GUI-free executor; `SimulationWorker` becomes a QThread shell emitting signals from callbacks |
| `boundary_and_forcing/*_qgis_adapter.py` | `swe2d/core/` (or stay, minus widget parameters) | These are `qgis.core`-only and headless-safe — legitimate core code per §2 policy. Their *widget-shaped* parameters (combo + `combo_layer_fn`) get replaced by explicit layer arguments, but the modules themselves need no quarantine |
| `workbench/services/batch_manager.py`, `workbench/workers/batch_worker.py` top-level `PyQt5` imports | `qgis.PyQt` equivalents | Correctness under alternate bindings (audit: `batch_manager.py:12`, `batch_worker.py:18`) |
| GUI-only modules (map tools, canvas items, `results/high_perf_viewer.py`, `results/animation.py`, dock/dialog views) | stay in `swe2d/workbench/` / `swe2d/results/` view layer | These legitimately need `qgis.gui`/widgets — they are the view |

## 4. Phases

Phases are sequential commits in the feature repo; each ends green on the GPU
validation gate (see `AGENTS.md`), `tests.test_workbench_gui`, `tests.test_cli`, plus
the new parity tests from Phase 0. The series merges to main as a unit when Phase 4
is green — no intermediate state is ever shipped.

### Phase 0 — Equivalence gate (do first; this is what was missing last time)

1. **RunContext diff test** (`tests/test_run_context_parity.py`, new): build one RunContext
   from the GUI dialog (mocked QGIS via `tests/mocks/qgis_env.py`, real PyQt5 — the
   `test_workbench_gui.py:467-500` pattern) and one from `build_run_context_from_dict`
   fed the same project's exported replay JSON; recursively diff all fields
   (callbacks compared by identity-of-underlying-logic). Initially run with a
   known-differences allowlist capturing today's drift (Thiessen mapping, missing
   drainage keys, dropped edge_groups/sample_map); the allowlist must shrink to zero by
   end of Phase 3.
2. **Replay-equivalence test** (GPU-gated, `tests/test_cli_gui_replay_parity.py`, new):
   run a small project (dambreak + one pipe + one weir) through the GUI path and through
   `python -m swe2d.cli run` from the same replay JSON; assert final `h/hu/hv` match to
   float tolerance. Modeled on `tests/_swe2d_test_helpers.py:427` `_run_cli_coupling`.
3. **Fix the stale coupling-test helper** (`_swe2d_test_helpers.py:478-523`) to use the
   supported drainage form — it currently feeds a silently-dropped path.
4. Extend `tests/test_helpers.py` `FallbackTracker` usage to the CLI builder so any
   `logger.warning` fallback during a parity test fails the test.

Gate: parity tests exist and document current drift in the allowlist.

### Phase 1 — Canonical run spec and single builder

1. Define `swe2d-run/2` (new, additive over `swe2d-replay/1`): JSON-schema-documented in
   `docs/RUN_SPEC_SCHEMA.md`; keys = RunContext field names (no widget names), plus
   `mesh`, `data_sources`, `results`, `units`. Keep `swe2d-replay/1` and widget-shaped
   input as **accepted-but-normalized** inputs.
2. `swe2d/core/builder.py::build_run_context(spec)` — today's
   `build_run_context_from_dict` promoted, with:
   - normalization front-stage: widget names → spec keys (existing `WIDGET_TO_RC` +
     `widget_state_to_flat_params` moved here, clearly marked as GUI-compat input
     normalization, not a second API);
   - **one** defaults table shared by all constructors; `RunContext.from_replay_json`
     and `from_widget_params` re-expressed as thin normalizers into it (kills the
     0.2-vs-0.05 `dt_cfg` default split);
   - unknown-key and type-mismatch errors (fail fast); no `_v()` silent defaults for
     keys present-but-wrong-type.
3. **Flip the GUI onto the same builder (the core move):**
   `run_controller._build_run_context()` becomes: `collect_run_widget_params()` +
   `collect_data_source_config()` (both exist) → spec dict → `build_run_context(spec)`.
   The dialog-bound `SWE2DRunDataBuilder`/`SWE2DRunOptionsBuilder` callback wiring
   (`run_component_wiring_controller.py:36-62`) is retired; layer resolution happens in
   the shared `qgis.core`-based adapters (the `*_qgis_adapter` modules, refactored to
   take explicit `QgsVectorLayer` arguments instead of combo + `combo_layer_fn` pairs).
   The GUI passes its live combo-selected layers directly; the CLI opens the same layers
   from `{table, gpkg}` data-source entries (the existing `collect_data_source_config`
   format). Same function, same `QgsVectorLayer` API, both paths — per the §2 policy.
   Diff-test allowlist enforces identical output.
4. Execution path untouched in this phase (the `headless_executor.py` adapter goes away
   in Phase 2).
5. **Fix the GUI-side serialization gaps found by the Phase 0 gate** (these drop or
   corrupt values before they ever reach the spec, so the single builder can't help
   until they're fixed):
   - `h0` / initial condition: `model_tab_view.collect_params()` omits all `initial_*`
     widgets, so GUI-exported replay JSON silently starts the CLI dry — add the
     `initial_*` widgets to the collectors.
   - `default_bc_type`: widget_state stores combo `currentIndex` (2=Inflow Q) while the
     GUI applies `currentData` (3=Stage) — serialize the data value, not the index.
   - `dt_fixed`/`dt_request`: the GUI derives these from `dt_spin` when adaptive is off;
     make that derivation explicit in the spec instead of per-path defaults.

Risk note: `run_controller.py:307-308` reads widgets inline during context construction
(`uniform_inflow_enabled`, `rain_update_interval_s`); ensure `collect_run_widget_params`
captures these so the spec is complete. Verify `view._parse_time_hours`,
`view._length_unit_name`, `view._model_gpkg_path` are all represented in the spec.

Gate: GUI and CLI produce byte-equal specs for the same project (assert in diff test),
and replay-equivalence passes with a shrunk allowlist.

### Phase 2 — GUI-free core (package realignment)

1. Create `swe2d/core/`; move `RunContext` (fix `run_context.py:329`'s lazy import of
   `batch_simulation_dialog` — invert it), executor body, builders, the `*_qgis_adapter`
   modules (with explicit-layer signatures), and the pure `workbench/services/*` modules
   per the table in §3. **Every import in the repo (GUI, CLI, tests) is updated to the
   new paths in the same commit — no re-export shims, no deprecation period.**
2. Split `SimulationWorker`: `swe2d/core/executor.py::execute_run(ctx, sink)` where
   `sink` is a small callback protocol (log/progress/snapshot/finished/failed —
   formalizing what `HeadlessWorkerAdapter` already fakes at
   `cli/headless_executor.py:32`). `SimulationWorker` keeps its signal API and delegates;
   `_WorkbenchShim` (`simulation_worker.py:148`) moves to core unchanged.
3. Replace `runtime/backend.py:154` `QSettings` read with an explicit config/env lookup
   (GUI writes the setting; core reads env var or an optional settings path).
4. Delete `cli/headless_executor.py` entirely — CLI calls `core.executor` directly with a
   plain sink; keep `SWE2DRunFinalizer` + persistence calls (shared).
5. Fix `workbench/workers/__init__.py:2` eager import. Add a CI import-boundary test:
   `python -c "import swe2d.core, swe2d.cli"` must succeed with `qgis.core` present but
   **without importing `qgis.gui`, `qgis.PyQt`, or `PyQt5`** (assert via
   `sys.modules` inspection after import, in the `qgis_stable` env). If/when the CLI
   needs native processing algs, the CLI entry point creates a headless
   `QgsApplication(argv, GUI=False)` per the §2 policy — still no display, no iface.

Gate: import-boundary test passes; full suite green; GUI behavior unchanged
(replay-equivalence + `test_workbench_gui`).

### Phase 3 — Deduplication and dead-code removal

General rule (from the §2 policy): wherever the same computation exists as both a
QGIS-based implementation and a raw reimplementation, keep the QGIS-based one.

1. **Thiessen:** delete `build_forced_thiessen_from_gpkg` (`gpkg_adapter.py:420`, the
   raw-sqlite3 reimplementation with the mismatched cell→gauge mapping); route the CLI
   spec's `rain_cn` source through the QGIS-based builder
   (`build_thiessen_rain_cn_forcing_qgis`, currently reached via the dead shim at
   `gpkg_adapter.py:686`). Replay-equivalence allowlist entry removed.
2. **Drainage:** single builder — extend
   `workbench/services/pipe_network_service.build_pipe_network_config` (post-move) to
   accept the spec's data-source form; delete the inline block at
   `run_context_builder.py:442-491` and the dead `read_drainage_config_from_gpkg` /
   `build_pipe_network_config_from_gpkg`. Support inline `{nodes, links}` JSON via the
   existing `extensions/drainage_network.py:66` builder or delete that too — decide, no
   third silent state.
3. **Wire the dropped data:** connect `edge_groups_dict` / `sample_map_data` into the
   RunContext (`run_context_builder.py:734,777,779`) or remove the loads; the diff test
   decides which is correct (GUI behavior is the reference).
4. **Batch:** one command builder. `batch_runner._run_one`, `BatchOrchestrator`,
   `batch_worker._build_command` → shared `swe2d/cli/commands.py::build_run_command(spec_path, ...)`.
   Decide `BatchOrchestrator`'s fate (it lacks MPS/status/results — likely delete).
   Fix `batch_worker` never passing `--status-file-path` (`batch_worker.py:113-134`
   vs docstring `:145-146`).
5. **Dialog method purge:** delete the ~120 solver-adjacent dialog methods with pure
   twins (`studio_dialog.py:2089-2245` region) once no caller remains; the GUI keeps
   only view code. Stop embedding view-bound callables in RunContext
   (`run_controller.py:345-355`) — the worker already re-binds pure versions
   (`simulation_worker.py:381-410`).
6. **Dead code:** remove the 8 uncalled `gpkg_adapter.py` functions (~800 lines) and
   extend `tests/test_no_dead_imports.py` to cover them.
7. **Config round-trip symmetry:** make CLI config persistence emit the widget-name-keyed
   widget_state (reuse `widget_state_to_flat_params`'s inverse) or teach restore to
   accept the scalar form — one direction only.
8. **Units/conversion convergence:** `rain_mm_to_model_depth` differs by path (GUI
   `1e-3·model_per_si_m` vs CLI `si_m_per_model` — found by the Phase 0 gate). One
   conversion in `swe2d.units`; both paths call it.
9. **Array-shape and packing conventions:** `cell_centroids` is a `(2,N)` tuple on the
   GUI path vs an `(N,2)` ndarray on the CLI path; `coupling_soa` is always-packed
   (possibly empty) on the GUI path vs only-when-configured on the CLI path (both found
   by the Phase 0 gate). Pick one convention each in the single builder — per
   `.agents/computation-source-truth.md`, prefer whatever the kernels consume natively.
10. **Headless structures zero-flow bug (a real bug, not just drift):**
    `test_cli_run_persists_nonzero_structure_flow` — the structure config builds
    correctly (`enabled=True`, cells resolved, `h0` applied, SoA packed) yet every
    kernel-side coupling metric (`flow`, `available_head_up`, …) is zero on the
    headless path, while the kernel itself passes
    `tests/test_culvert_hds5_validation.py`. Compounding factor: the baked-mesh RCMK
    permutation reverses cell order, so JSON `upstream_cell`/`downstream_cell` indices
    (original order) don't address the same physical cells after the CLI reloads the
    baked mesh — correcting indices still yielded zero flow, so the defect is deeper
    (likely in the run-loop structure-coupling wiring). Root-cause and fix here; until
    fixed, the replay-equivalence gate must not enable structures.

Gate: diff-test allowlist is empty; dead-code test green; line-count of
`cli/gpkg_adapter.py` roughly halved; replay-equivalence passes without allowlist.

### Phase 4 — Fail-fast and hardening

1. `_execute()`-entry RunContext validation (per `CLI_GUI_PARITY_DRIFT.md:28-30`):
   required callbacks must not be the no-op default; required arrays non-empty.
   Update `tests/test_run_context.py:40-43`, which currently asserts the no-op behavior.
2. Remove the silent absorbs enumerated in §1; replace with typed errors naming the
   spec key. `query_mesh_from_gpkg` must distinguish corrupt-BLOB / missing-module /
   missing-table errors.
3. Spec validation against the JSON schema at builder entry; unknown keys = error with
   "did you mean" suggestions (widget-name keys suggest their spec equivalent).
4. Fix incidental bugs found by the audit (each a one-liner with a test):
   status-file `"step"` field receives percent (`headless_runner.py:149-152`);
   `progress_callback` signature drift vs `CLI_GUIDE.md:194`; absolute paths in replay
   JSON → support relative-to-spec-file paths for portability.
5. Update `docs/CLI_GUIDE.md` (it currently claims "no QGIS or Qt dependency" — the
   correct statement per the §2 policy is "requires `qgis.core`; no QGIS GUI, iface, or
   display needed") and `docs/CLI_GUI_PARITY_DRIFT.md` (mechanisms → now enforced, not
   aspirational).

Gate: new negative tests (bad key, missing layer, corrupt mesh) all fail loudly with
actionable messages; docs match behavior.

## 5. Risks and mitigations

- **GUI regression (the known failure mode):** mitigated by Phase 0's replay-equivalence
  gate running *before* any GUI-path change, and by flipping the GUI onto the shared
  builder (Phase 1.3) in a single reviewable commit with the diff test byte-comparing
  specs.
- **`collect_data_source_config` gaps:** some combos may not resolve to `{table, gpkg}`
  (in-memory layers, non-GPKG sources). Mitigation: spec supports an inline-layer form
  (serialized features) as an escape hatch; diff test will surface every unresolved
  case at Phase 1.
- **Cross-thread widget access:** today view-bound callables ride the RunContext into
  the worker thread (`run_controller.py:345-355`). Phase 1 removes them from the
  context; Phase 2's sink protocol marshals all UI updates through Qt signals only.
- **Headless processing-alg initialization:** native QGIS processing algs generally
  require a `QgsApplication` with the processing registry initialized. Confine that
  setup to the CLI entry point (`GUI=False`); never let it creep into library code
  paths the GUI also uses (inside QGIS, the application already exists).
- **Scope creep into numerics:** explicitly out of scope; `_WorkbenchShim`,
  `runtime_step_executor`, kernels untouched.
- **Test-suite blind spot:** the mocked GUI tests can't catch SIP/canvas/rendering
  issues and the real-QGIS CI job is disabled (`test.yml:112` `if: false`). The MCP
  server plan (companion doc) adds live-GUI automation; until then, run
  `tests/run_headless_qgis_tests.sh --conda` locally before merging each phase.

## 6. Suggested sequencing summary

| Phase | Deliverable | Primary risk retired |
|---|---|---|
| 0 | Parity diff + replay-equivalence tests, allowlist of current drift | Blind refactoring |
| 1 | `swe2d-run/2` spec; single builder; GUI delegates to it | Hand-synced dual paths |
| 2 | `swe2d/core` GUI-free (qgis.core allowed); executor split; `headless_executor` deleted; all imports updated, zero shims | Core entangled with GUI bindings |
| 3 | Thiessen/drainage/batch/dialog dedup; dead code gone; allowlist empty | Actual result divergence |
| 4 | Fail-fast validation; docs corrected | Silent wrong answers |

## 7. Drift inventory → phase map

The living inventory is `KNOWN_DIVERGENCES` in `tests/test_run_context_parity.py`:
every entry asserts the drift still exists, so a fix that forgets to delete its entry
turns the gate red. This table assigns each known divergence to the phase that fixes
it. **Rule: when the gate discovers new drift, add the allowlist entry AND a row here
in the same commit.**

| Divergence | Found by | Fixed in |
|---|---|---|
| Thiessen cell→gauge mapping (raw-sqlite3 reimplementation) | 2026-07-24 audit | Phase 3.1 |
| Drainage block missing 5 keys (`friction_method`, `surcharge_method`, `recon_method`, `time_integrator`, `friction_alpha`) | 2026-07-24 audit | Phase 3.2 |
| Dropped `edge_groups` / `sample_map_data` in CLI runs | 2026-07-24 audit | Phase 3.3 |
| Inline-JSON drainage silently dropped | 2026-07-24 audit | Phase 3.2 |
| `dt_cfg` default split (0.2 vs 0.05) across constructors | 2026-07-24 audit | Phase 1.2 |
| `h0` / initial condition lost in replay export | Phase 0 gate | Phase 1.5 |
| `default_bc_type` combo index vs data | Phase 0 gate | Phase 1.5 |
| `dt_fixed`/`dt_request` per-path derivation | Phase 0 gate | Phase 1.5 |
| `rain_mm_to_model_depth` formula mismatch | Phase 0 gate | Phase 3.8 |
| `cell_centroids` `(2,N)` tuple vs `(N,2)` ndarray | Phase 0 gate | Phase 3.9 |
| `coupling_soa` always-packed vs only-when-configured | Phase 0 gate | Phase 3.9 |
| Headless structures zero-flow (incl. RCMK permutation index hazard) | Phase 0 gate | Phase 3.10 |
| CLI↔GUI config round-trip asymmetry | 2026-07-24 audit | Phase 3.7 |
| Status-file `"step"` field receives percent | 2026-07-24 audit | Phase 4.4 |
| `progress_callback` signature drift vs CLI guide | 2026-07-24 audit | Phase 4.4 |
| Absolute paths baked into replay JSON | 2026-07-24 audit | Phase 4.4 |
| Legacy string `mesh` form crashes `build_run_context_from_dict` (`mesh.get` on str, `run_context_builder.py:242`) despite `execute_run` claiming to support it | Phase 0 gate | Phase 1.2 (normalization front-stage must accept or loudly reject the string form) |
