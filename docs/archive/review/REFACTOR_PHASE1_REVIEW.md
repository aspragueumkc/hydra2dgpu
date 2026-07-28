---
type: reference
status: complete
created: 2026-07-24
completed: 2026-07-25
---

# Refactor Phase 1 Review

**Branch**: `.worktrees/cli-first` (Phases 0, 1.A, 1.B, 1.C complete)
**Reviewer**: CLI-first refactor review pass
**Date**: 2026-07-24

---

## Spec Compliance

| Requirement | Status | Notes |
|---|---|---|
| `build_run_context()` accepts canonical `swe2d-run/2` specs | ✅ PASS | Top-level + `params` sub-dict, mesh dict, all nested blocks |
| Accepts legacy widget-name keys | ✅ PASS | `WIDGET_TO_RC` maps all 40+ widget names; combo text keys mapped too |
| Accepts flat CLI JSON | ✅ PASS | `_normalize_spec` handles flat params, `data_sources` flatten |
| `RunContextBuilder` supports CLI (single spec) and GUI (layered) modes | ✅ PASS | `mode="cli"` and `mode="gui"` stacks; `from_spec`/`from_defaults` factories |
| `dt_cfg` default consistent at 0.05 everywhere | ✅ PASS | Single `_DEFAULTS["dt_cfg"] = 0.05` table — eliminated 0.2/0.05 split |
| `WIDGET_TO_RC` mapping complete for all GUI widgets | ⚠️ FIXED | `bridge_coupling_mode` was missing; added mapping to `bridge_stacked_coupling_mode` |
| `_normalize_spec` rejects unknown keys with "did you mean" | ✅ PASS | `difflib.get_close_matches` + widget-name equivalent suggestion |

**Fix applied during review**: `bridge_coupling_mode` (widget name emitted by `model_tab_view.collect_run_widget_params()`) had no entry in `WIDGET_TO_RC`. The builder reads `bridge_stacked_coupling_mode` from specs. Without the mapping, selecting a bridge coupling mode in the GUI would silently use the `phase3_spatial` default instead of the user's choice. Added:

```python
# bridge_coupling_mode: widget name emitted by model_tab_view
"bridge_coupling_mode": "bridge_stacked_coupling_mode",
```

---

## Code Quality

### No Qt/QGIS imports in `run_context_builder.py`
✅ **PASS** — The canonical builder imports only `swe2d.*` modules and stdlib. QGIS is kept entirely in `run_context_adapter.py` via the `_build_drainage_config_from_gpkg_layers` helper.

### No duplicated `_DEFAULTS` logic
✅ **PASS** — `_DEFAULTS` is the single source of truth. Both `build_run_context()` and `RunContextBuilder.build()` use it. `RunContext.from_replay_json()` and `RunContext.from_widget_params()` call `build_run_context()` internally, so they inherit the same defaults.

### No obvious bugs

- `_v()` helper resolves top-level → `params` → default correctly throughout.
- `from_defaults()` starts with `mode="gui"` and pre-populates the stack with `_DEFAULTS`. The GUI stack is the accumulated stack (including defaults); `build_mode_stack()` returns it directly without double-adding defaults. ✅
- `from_spec()` normalizes then calls `build()` → `_build_cli_stack()` which adds `_DEFAULTS` first. ✅
- `_build_cli_stack()` adds `_DEFAULTS` as the base, then extends with user layers. Priority order: `_DEFAULTS → user layers`. ✅
- Array/callback/container fields (`cell_areas`, `mesh_cell_areas`, etc.) are always recomputed from mesh, never taken from the merged spec. ✅
- The `_parse_duration()` helper in `widget_state_to_flat_params` handles `HH:MM` and fractional-hour strings. ✅

### Thin adapter check

`run_context_adapter.py` is 301 lines. It:
1. Translates widget state to a spec dict (mostly raw passthrough; the hard work is in the builder).
2. Calls `build_run_context(spec)`.
3. Attaches GUI-only callbacks via `dataclasses.replace()`.

No duplicated defaults, no GPKG loading, no mesh computation. ✅

### Minor dead-code observation

`build_run_context_from_gui` passes `_uniform_inflow_enabled` (underscore prefix) into the spec. The builder's `_v()` looks for `uniform_inflow_enabled` (no underscore). Since `_uniform_inflow_enabled` starts with `_`, it passes the "skip private keys" guard in `_normalize_spec`, but is never read by the builder. The controller correctly sets `inflow_progressive_enabled` via `replace()` after the adapter returns, so this has no runtime effect — but the adapter key is dead code. Not a functional bug.

---

## Architecture

### Can CLI run without QGIS?
✅ **Yes** — `build_run_context()` and `RunContextBuilder` have zero Qt/QGIS imports. `gpkg_adapter.py` uses `sqlite3` and `ogr` (GDAL), not `QgsVectorLayer`. Drainage layer loading (the only QGIS step) is gated inside `run_context_adapter._build_drainage_config_from_gpkg_layers()` which is only invoked when the spec contains `drainage` block and QGIS is available.

### Can GUI run?
✅ **Yes** — `RunController._build_run_context()` collects widget values, calls `build_run_context_from_gui()`, then applies `dataclasses.replace()` for the array fields and callbacks only the GUI can provide. Signal/slot connections, worker thread orchestration, and finalization are untouched.

### Circular imports
✅ **None detected** — The import graph:
- `run_context_builder.py` → `run_context.py`, `gpkg_adapter`, `mesh_computation_service`, `boundary_qgis_adapter`, `runtime_source_logic`, `bridge_stacked_runtime`, `extensions.structures`
- `run_context_adapter.py` → `run_context_builder`, `run_context`
- `run_controller.py` → `run_context_adapter`, `run_context`, `simulation_worker`
- No cycles.

### Callback handling in the adapter
✅ **Correct** — The adapter attaches GUI callbacks (`apply_timeseries_bc_values`, `_mesh_cell_areas_fn`, etc.) via `dataclasses.replace()` after `build_run_context()` returns. Each callback is individually null-checked before insertion. ✅

---

## Test Coverage

### Builder tests cover `merge_context()` path
✅ `TestRunContextBuilderMergeContext` — 3 tests:
- `test_merge_context_single_run_context` — scalar fields from a RunContext land in the built context
- `test_merge_context_later_layer_wins` — multiple merges obey priority
- `test_merge_context_plus_with_params` — `merge_context` layers come before `with_params` in priority

### Builder tests cover `from_defaults()` → `build()` path
✅ `TestRunContextBuilderFullPipeline` — 3 tests:
- `test_from_defaults_merge_context_build_produces_valid_context`
- `test_from_spec_plus_with_params_equivalent` — `from_spec({}) + with_params` = `from_defaults() + with_params`
- `test_builder_top_level_overrides_params_block`

### Builder tests cover normalization edge cases
✅ `TestNormalizeSpecValidation` (5 tests) + `TestNormalizeSpecNormalization` (5 tests):
- Unknown key → `ValueError` with `did you mean` suggestion
- Widget name suggestion includes both RC name and widget equivalent
- `widget_state` and `version` internal keys allowed
- `mesh` string → dict normalization
- Widget name in top-level and inside `params` block normalized

### All 37 tests are meaningful
✅ All tests have substantive assertions. None are `assert True` stubs. Key assertions:
- Scalar values (`ctx.run_duration_s`, `ctx.dt_cfg`, `ctx.n_mann`)
- Object identity checks (`isinstance(spec, dict)`)
- Mesh arrays populated (`ctx.cell_areas.size > 0`)
- Builder mode string (`b._mode == "gui"`)
- Merge priority (`ctx.n_mann == 0.040`, not 0.030)

---

## Issues Found

### 🔴 Critical (fixed during review)

**`bridge_coupling_mode` not mapped in `WIDGET_TO_RC`**
- **Severity**: Silent wrong default at runtime — bridge coupling mode would always be `phase3_spatial` regardless of GUI selection.
- **Root cause**: `model_tab_view.collect_run_widget_params()` emits `bridge_coupling_mode` as the widget value key. The builder reads `bridge_stacked_coupling_mode` from specs. No mapping existed between them.
- **Fix**: Added `"bridge_coupling_mode": "bridge_stacked_coupling_mode"` to `WIDGET_TO_RC`.
- **File**: `swe2d/runtime/run_context_builder.py:90`

### ⚠️ Minor

**Test runner environment assumption**
- Running `pytest tests/test_run_context_builder.py` from `.worktrees/cli-first/` produces 18 pass + 19 errors (`ModuleNotFoundError: hydra_swe2d`) because the `build/` symlink is not present in the worktree.
- Running from the repo root with `PYTHONPATH="$PWD:$PWD/build"` passes all 37.
- **Recommendation**: Document in `AGENTS.md` or add a `pytest.ini` / conftest note that tests must be run from the repo root.

**Dead `_uniform_inflow_enabled` key in adapter spec**
- The adapter passes `spec["_uniform_inflow_enabled"]` but the builder reads `uniform_inflow_enabled`. The underscore key passes the "skip private" guard but is never read. The controller overrides `inflow_progressive_enabled` via `replace()` anyway, so no runtime bug.
- **Recommendation**: Remove `_uniform_inflow_enabled` from the adapter's spec construction, or rename the builder's `_uniform_inflow_enabled` reading to match.

---

## Recommendations

1. **Add a `test_bridge_coupling_mode_mapping` test** to `TestBuildRunContextMinimalSpec` — verify that when the spec contains `bridge_coupling_mode: "phase2"`, the built `RunContext.bridge_stacked_coupling_mode == "phase2"`.

2. **Add a test for `build_run_context` accepting `swe2d-run/2` canonical format** with a fully-structured spec dict including `mesh`, `params`, `results`, `units`, `data_sources` blocks — to serve as a living schema contract.

3. **Run the existing validation-gated tests** (`tests/test_swe2d_gpu_validation_perf.py`, `tests/test_workbench_gui.py`) against both CLI and GUI paths to confirm end-to-end parity after the Phase 1 changes.

4. **Audit `WIDGET_TO_RC` completeness** whenever a new widget is added to `model_tab_view.collect_run_widget_params()` — a missing entry silently falls through to the default. Consider adding a unit test that asserts `WIDGET_TO_RC.keys()` covers all keys returned by a reference widget-state snapshot.

---

## Verdict: APPROVED (with notes)

The Phase 1 implementation is structurally sound and the architectural goals are met: a single canonical builder, thin GUI adapter, no Qt in the CLI path, and a single source of truth for defaults. The `bridge_coupling_mode` bug was caught during this review and has been fixed.

The 37 tests pass (37/37 from repo root). No regressions in spec compliance, code quality, or architecture. Phase 2 and Phase 3 items (drainage inline JSON, sample lines drift, HEC-RAS HDF5 snapshot via `on_snapshot`, `studio_dialog.py` sqlite3 cleanup) are correctly tracked in `docs/CLI_GUI_PARITY_DRIFT.md`.

**Test result** (from repo root, `PYTHONPATH="$PWD:$PWD/build"`):
```
37 passed in 0.51s
```
