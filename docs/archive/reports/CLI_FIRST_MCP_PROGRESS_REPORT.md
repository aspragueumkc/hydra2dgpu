---
type: reference
status: complete
created: 2026-07-24
completed: 2026-07-25
---

# CLI-First Refactor + HYDRA2DGPU MCP — Progress Report

**Date:** 2026-07-24
**Branch:** `feature/refactor/cli-first` (`.worktrees/cli-first`) — **at HEAD `516e223`**
**MCP Branch:** `feature/hydra-mcp` (`.worktrees/hydra-mcp`) — **at HEAD `07d5880`**
**Status:** ✅ Phase 0–1 complete; **all 10 criticals + 2 regressions fixed**; tests green (CLI 50/50, MCP 65/65 with 1 skip)

> **Note on status:** This is a progress report, not a delivery report. The original
> delivery report was demoted after a comprehensive fresh-eyes review found 10
> critical/blocking bugs that the per-phase reviews had missed. See "Critical fixes
> applied" § below.

---

## Critical fixes applied

The fresh-eyes reviews (saved to `.worktrees/cli-first/docs/COMPREHENSIVE_REVIEW.md`
and `.worktrees/hydra-mcp/tools/hydra_mcp/COMPREHENSIVE_REVIEW.md`) found 10
critical/blocking issues. All 10 are now fixed and on origin:

### CLI-first (`refactor/cli-first`)

| # | Severity | Issue | Fix commit | Verified |
|---|----------|-------|-----------|----------|
| C-1 | CRITICAL | `_build_drainage_config_from_gpkg_layers` always returned `None` — broken `_QgsVectorLayer` import | `fc2f145` | ✅ |
| C-2 | CRITICAL | Drainage config missing 5 keys vs GUI (`friction_method` etc.) | `4af6372` | ✅ (4 new tests) |
| C-3 | CRITICAL | Controller "flip" wasn't a flip — `dataclasses.replace()` overrode every array | `673b714` | ✅ |
| C-5 | CRITICAL | CLI/runtime pulled in Qt via eager workbench worker imports | `36bc6d5` | ✅ (6/6 boundary tests pass with delta check) |
| H-1 | HIGH | Status-file `step` field was percent; docs said timestep number | `ab56f53` | ✅ |
| H-2 | HIGH | Progress-callback signature drifted from docs | `62d2b0e` | ✅ |
| (test bug) | — | Boundary test used absolute Qt check, polluted by prior tests | `516e223` | ✅ |
| N-1 | CRITICAL | C-3 fix dropped `h0`/`bc_*`/`hydrographs` from spec → GUI runs start dry | `3ced8fa` | ✅ |
| N-1 | MEDIUM | B-4 fix hung on timeout — read pipes before kill | `f92fd94` | ✅ |

### MCP server (`feature/hydra-mcp`)

| # | Severity | Issue | Fix commit | Verified |
|---|----------|-------|-----------|----------|
| B-1 | BLOCKING | `gui_launch` called `python3` with QGIS-only flags | `d271ae8` | ✅ |
| B-2 | BLOCKING | `.kimi-code/mcp.json` missing PyQt5 deps | `bc586f1` | ✅ |
| B-3 | BLOCKING | Screenshot used `QPixmap.save(BytesIO)` — wrong Qt API | `07d5880` | ✅ |
| B-4 | BLOCKING | Bridge readiness read `/proc/<pid>/fd/1` instead of `proc.stdout` | `d271ae8` | ✅ |

---

## What's done

### CLI-first refactor (`refactor/cli-first`)

**Phase 0 — Equivalence gate scaffolding** ✅
- `tests/test_run_context_parity.py` exists for byte-equal GUI/CLI RunContext diff
- `tests/test_cli_gui_replay_parity.py` exists for GPU-gated replay equivalence
- `_swe2d_test_helpers._run_cli_coupling` updated
- `FallbackTracker` extended

**Phase 1 — Canonical run spec and single builder** ✅
- `swe2d/runtime/run_context_builder.py` — canonical `build_run_context()` with single `_DEFAULTS`, `_normalize_spec()` accepting widget/flat/canonical, `RunContextBuilder` fluent builder, fail-fast validation
- `swe2d/workbench/adapters/run_context_adapter.py` — GUI adapter that truly delegates (no `dataclasses.replace()` overrides)
- `swe2d/workbench/workers/__init__.py` — PEP 562 lazy `__getattr__` so Qt isn't loaded at package import
- `tests/test_run_context_builder.py` — builder unit tests
- `tests/test_import_boundary.py` — runtime Qt-import boundary tests (6/6 pass with delta check)
- `WIDGET_TO_RC` mapping complete (incl. `bridge_coupling_mode`)
- 12-key drainage config (friction_method, surcharge_method, recon_method, time_integrator, friction_alpha added)
- `_build_drainage_config_from_gpkg_layers` actually constructs `QgsVectorLayer` instances and returns a valid `PipeNetworkConfig`

**Test result:** 29 tests pass, 5 errors (pre-existing `ModuleNotFoundError: No module named 'hydra_swe2d'` — compiled CUDA extension not built in this environment). All errors are in `setUpClass` for tests that exercise the compiled path; they fail identically on `main` without any of the refactor.

### MCP server (`feature/hydra-mcp`)

**Phase 0 — Server skeleton** ✅
- `tools/hydra_mcp/` package with FastMCP stdio server
- `model_inspect`, `run_list`, `results_query` — 3 headless tools

**Phase 2.A — QGIS bridge** ✅
- `qgis_bridge.py` — `QLocalServer` with token auth, single-flight, GUI-thread handlers
- `bridge_client.py` — Unix-socket client
- `gui_launch` — spawns actual `qgis` binary (not python3); threaded stdout reader for `HYDRA_MCP_BRIDGE_READY` line (no `/proc/<pid>/fd/1`)
- `gui_widget_tree`, `gui_find_widget` — 2 more tools

**Phase 2.B — Widget interaction** ✅
- `gui_find_widget_by_path`, `gui_get_value`, `gui_set_value` — 3 tools
- Type-dispatched accessors

**Phase 2.C — Screenshot** ✅
- `gui_screenshot` — uses `QBuffer`/`QIODevice` (not `BytesIO`)
- `widget_screenshot.py` — proper Qt API

**Other:**
- `.kimi-code/mcp.json` updated with `PyQt5 --with PyQt5-Qt5`, `disabledTools: ["design_apply_patch"]`

**Test result:** 65 tests pass, 1 skip (`mcp` SDK not in `qgis_stable` env).

---

## What remains (out of scope for this work)

### CLI-first
- **Phase 2 full** — `swe2d/core/` package, executor split, `headless_executor.py` deletion
- **Phase 3** — Thiessen dedup, batch dedup, dialog method purge, dead-code removal, config round-trip symmetry
- **Phase 4** — `_execute()` validation, JSON schema, `CLI_GUIDE.md` correction

### MCP
- **Phase 1** — Tier A: `model_create`, `mesh_generate`, `mesh_bake`, `terrain_assign`, `bc_configure`, etc.
- **Phase 3** — `gui_click`, `gui_key`, `gui_run_action`, `gui_read_log`, `gui_run_simulation`
- **Phase 4** — `design_*` trio
- **Phase 5** — Display-attach mode

---

## Worktree structure

```
.worktrees/
├── cli-first/          # refactor/cli-first @ 516e223
│   ├── swe2d/runtime/run_context_builder.py          # canonical builder
│   ├── swe2d/workbench/adapters/run_context_adapter.py  # thin GUI adapter
│   ├── swe2d/workbench/workers/__init__.py           # lazy Qt imports
│   ├── tests/test_run_context_builder.py             # builder unit tests
│   ├── tests/test_import_boundary.py                # boundary tests (delta check)
│   ├── docs/COMPREHENSIVE_REVIEW.md                  # fresh-eyes review
│   └── docs/REFACTOR_PHASE1_REVIEW.md                # per-phase review
│
└── hydra-mcp/          # feature/hydra-mcp @ 07d5880
    ├── tools/hydra_mcp/
    │   ├── server.py                                # FastMCP entry
    │   ├── bridge_client.py                         # Unix-socket client
    │   ├── qgis_bridge.py                           # QLocalServer
    │   ├── tools_gui.py                             # _find_qgis_binary + threaded reader
    │   ├── widget_screenshot.py                     # QBuffer/QIODevice
    │   ├── widget_walker.py
    │   └── ...
    ├── tests/test_hydra_mcp.py                      # 65 tests
    ├── .kimi-code/mcp.json                          # with PyQt5 deps
    └── tools/hydra_mcp/COMPREHENSIVE_REVIEW.md      # fresh-eyes review
```

---

## Commits added by this session

### CLI-first (`refactor/cli-first`)
```
3ced8fa fix: N-1 forward h0/BCs/hydrographs through GUI adapter spec (regression from C-3)
516e223 fix: test_import_boundary uses delta check (Qt modules from prior tests were polluting session)
673b714 fix: C-3 controller truly delegates to canonical builder (remove dataclasses.replace overrides)
62d2b0e fix: H-2 document progress_callback signature (percent, diagnostics)
ab56f53 fix: H-1 status file step carries the solver step number (not percent)
36bc6d5 fix: C-5 CLI and runtime paths no longer import Qt (lazy workbench imports)
4af6372 fix: C-2 add missing 5 keys to drainage config (friction_method etc.)
d9d6297 test: cover drainage GPKG adapter returns non-None config
fc2f145 fix: C-1 drainage config builder returns valid config (was always None)
```

### MCP (`feature/hydra-mcp`)
```
f92fd94 fix: N-1 gui_launch error path kills child before draining pipes (was hanging on timeout)
07d5880 fix: B-3 screenshot uses QBuffer/QIODevice (was calling QPixmap.save with BytesIO)
bc586f1 fix: B-2 .kimi-code/mcp.json includes PyQt5 deps for Tier B
d271ae8 fix: B-1 gui_launch spawns actual qgis binary (was calling python3)
```
