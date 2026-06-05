# Implementation Plan: Remove Legacy UI Path & Improve Studio UI

> **Status**: Planning | **Date**: 2026-06-04 | **Scope**: `swe2d_workbench_qt.py`, `shell_dialog_methods.py`, `hydra_*.py`

## TL;DR

Remove `shell_dialog_methods.py` (~650 lines), `SWE2DWorkbenchDesignerDialog`, `SWE2DWorkbenchScenarioDialog`, and `forms/swe2d_workbench_shell.ui`. Move all 18 shell methods directly into `SWE2DWorkbenchStudioDialog` as native methods. Improve error handling, `.ui` file consistency, and widget binding validation. Zero backward compatibility.

### Before / After

```
BEFORE                                    AFTER
────────                                  ─────
SWE2DWorkbenchDialog (base)               SWE2DWorkbenchDialog (base)
├── SWE2DWorkbenchDesignerDialog  ✂️       └── SWE2DWorkbenchStudioDialog
├── SWE2DWorkbenchStudioDialog                ├── _feature_keywords()      (inlined)
│   └── delegates 14 methods to →             ├── _apply_feature_filters() (inlined)
│       shell_dialog_methods.py (~650 lines)  ├── _build_ui()             (inlined)
└── SWE2DWorkbenchScenarioDialog   ✂️          └── ... 14 direct methods

shell_dialog_methods.py           🗑️       shell_dialog_methods.py         🗑️
swe2d_workbench_shell.ui          🗑️       swe2d_workbench_shell.ui        🗑️
```

---

## Phase 1 — Audit & Preparation

*No dependencies. Steps 1–3 can run in parallel.*

| Step | Action | Detail |
|------|--------|--------|
| 1 | **Tag all `shell_dialog_methods` references** | Create a checklist of every import, delegation call, and reference across `swe2d_workbench_qt.py`, `hydra_qt.py`, `hydra_plugin.py`, `monolith_methods.py` |
| 2 | **Audit QGIS plugin UI standards** | Review QGIS docs for `QgsDockWidget`, menu/toolbar conventions, and `.ui` file patterns. Identify compliance gaps in current Studio code. |
| 3 | **Read `docs/STUDIO_UI_ARCHITECTURE.md`** | Understand the current checklist to know exactly what documentation to rewrite in Phase 5. |

---

## Phase 2 — Move Shell Methods into Studio Class

*Sequential — all changes in `swe2d_workbench_qt.py`, one class.*

### Methods to inline (from `shell_dialog_methods.py`)

| Step | Method | Shell Line | Role |
|------|--------|-----------|------|
| 4 | `_studio_feature_keywords()` | 253 | Returns `{feature: (keywords,)}` dict |
| 5 | `_studio_apply_feature_filters()` | 299 | Show/hide widgets by feature flag |
| 6 | `_studio_set_feature_enabled()` | 233 | Toggle flag + re-apply filters |
| 7 | `_studio_widget_text_blob()` | 279 | Extract widget name + text for keyword matching |
| 8 | `_studio_project_scope_key()` | 126 | Layout persistence key |
| 8 | `_studio_layout_settings_keys()` | 145 | Return (main, inspector) settings keys |
| 8 | `_restore_studio_layout_state()` | 152 | Load dock/splitter geometry |
| 8 | `_save_studio_layout_state()` | 193 | Persist dock/splitter geometry |
| 9 | `_studio_build_ui()` | 396 | QMainWindow + docks + toolbar construction |
| 10 | `_studio_mount_widget()` | 207 | Add widget to host |
| 10 | `_studio_select_tab()` | 222 | Show tab by name |
| 10 | `_studio_sync_view_mode()` | 340 | Sync view combo |
| 10 | `_studio_apply_visual_profile()` | 352 | Apply stylesheet theme |
| 10 | `_studio_update_status()` | 375 | Update status bar |

### Integration steps

| Step | Action |
|------|--------|
| 11 | **Update all 14 wrapper methods** — Remove `from shell_dialog_methods import X` and replace each wrapper body with the inlined implementation (or remove the wrapper and call `self._method()` directly). |
| 12 | **Remove the import block** at approximately lines 10416–10502 in `swe2d_workbench_qt.py` — the 25-line `from swe2d.workbench.extracted.shell_dialog_methods import ...` block. |

---

## Phase 3 — Remove Designer & Scenario Dialogs

*Depends on Phase 2 completion.*

| Step | Action | File / Location |
|------|--------|-----------------|
| 13 | **Delete `SWE2DWorkbenchDesignerDialog`** class | `swe2d_workbench_qt.py` ~lines 10246–10375 |
| 14 | **Delete `SWE2DWorkbenchScenarioDialog`** class | `swe2d_workbench_qt.py` ~lines 10511–10575 |
| 15 | **Delete launcher functions** — `launch_swe2d_workbench()`, `launch_swe2d_workbench_designer()`, `launch_swe2d_workbench_scenario()`. Keep only `launch_swe2d_workbench_studio()`. | `swe2d_workbench_qt.py` ~lines 10758–11050 |
| 16 | **Update `hydra_qt.py`** — Remove `open_swe2d_designer_dialog()` and `open_swe2d_scenario_dialog()`. Update any callers. | `hydra_qt.py` ~lines 5858, 5895 |
| 17 | **Update `hydra_plugin.py`** — Remove `HYDRAMenuSWE2DDesignerAction` and `HYDRAMenuSWE2DScenarioAction` menu actions. Update toolbar references. | `hydra_plugin.py` ~lines 457–459 |

---

## Phase 4 — Studio UI Improvements

*Depends on Phase 2. Runs in parallel with Phase 3.*

| Step | Action | Detail |
|------|--------|--------|
| 18 | **Consolidate feature toggles** | Move feature flags dict + keywords + filter logic into a single cohesive section of `SWE2DWorkbenchStudioDialog`. Adopt a clear `_feature_*` method prefix convention. Currently spread across 4 files — reduces to 2 (class + `studio_host_methods.py` for menu/toolbar). |
| 19 | **Add widget binding validation** | On `_build_ui()`, after all tabs are composed, iterate all `.ui` widgets and verify they have Python bindings. Log warnings for unbound widgets. Raise `RuntimeError` for critical missing widgets (e.g., run button). |
| 20 | **Main-window `.ui` file** *(optional, for discussion)* | Extract the QMainWindow/docks/toolbar layout from `_studio_build_ui()` into `forms/swe2d_studio_main.ui` for consistency with the tab-page `.ui` file approach. |
| 21 | **Improve error handling** | Replace bare `findChild()` calls with a helper method that raises `RuntimeError` if the widget is not found, instead of silently returning `None`. |
| 22 | **Add `__init__.py` exports** | Ensure `swe2d/workbench/extracted/__init__.py` has clean, explicit exports. Nothing should accidentally import from the deleted `shell_dialog_methods`. |

---

## Phase 5 — Cleanup & Documentation

*Depends on Phases 2–4 completion.*

| Step | Action |
|------|--------|
| 23 | **Delete** `swe2d/workbench/extracted/shell_dialog_methods.py` |
| 24 | **Delete** `forms/swe2d_workbench_shell.ui` |
| 25 | **Update `AGENTS.md`** — Remove "parallel tab list" warnings. Update Studio UI section to reflect single-path architecture. Remove Designer/Scenario references. |
| 26 | **Rewrite `docs/STUDIO_UI_ARCHITECTURE.md`** — Remove all shell/legacy checklists. Simplify to Studio-only procedures. Add new validation step from Phase 4. |
| 27 | **Update `docs/SWE2D_GPU_ARCHITECTURE_REPORT.md`** — Section 7 (QGIS Studio UI): remove references to shell/legacy dual-path, Designer, Scenario. |
| 28 | **Update `swe2d/workbench/monolith_methods.py`** — Remove any re-exports of deleted modules. |
| 29 | **Purge `__pycache__`** — `find . -type d -name __pycache__ -exec rm -rf {} +` |

---

## Verification

| # | Check | Command / Method |
|---|-------|------------------|
| 1 | No `shell_dialog_methods` references remain | `grep -r "shell_dialog_methods" --include="*.py"` → zero results |
| 2 | No `swe2d_workbench_shell` references remain | `grep -r "swe2d_workbench_shell" --include="*.py" --include="*.ui"` → zero results |
| 3 | No Designer/Scenario function references | `grep -r "designer_populate_left_tabs\|designer_build_ui\|scenario_apply_preset\|scenario_build_ui" --include="*.py"` → zero results |
| 4 | Widget bindings are complete | `python tools/ui_bind_sync.py forms/swe2d_model_tab.ui swe2d_workbench_qt.py --missing` → zero issues |
| 5 | Studio launches correctly | Launch from QGIS: all 7 tabs load, feature toggles work, layout persists across sessions |
| 6 | No test regressions | Run full pytest suite |
| 7 | Manual smoke test | Open Studio → toggle rainfall/drainage/3d-patch → switch tabs → resize docks → close/reopen → layout restores |

---

## Decisions

- **Designer and Scenario dialogs removed entirely** — no backward compat stubs, no redirect launchers.
- **No backward compatibility** for old shell `.ui` file or Designer/Scenario launcher functions.
- **Main-window `.ui` file extraction** (Phase 4, Step 20) is **optional** — flagged for team discussion. If deferred, `_studio_build_ui()` remains as a programmatic method in the Studio class.
- **Studio-only path** going forward — all development targets `SWE2DWorkbenchStudioDialog`.

---

## Relevant Files

| File | Action | Lines |
|------|--------|-------|
| `swe2d_workbench_qt.py` | Main target: remove 2 dialog classes + 3 launchers + 25 shell imports, inline 18 methods | ~300 lines changed |
| `swe2d/workbench/extracted/shell_dialog_methods.py` | **DELETE** | −650 lines |
| `swe2d/workbench/extracted/studio_host_methods.py` | Review for shell references (minimal changes expected) | ~10 lines |
| `forms/swe2d_workbench_shell.ui` | **DELETE** | −1 file |
| `hydra_qt.py` | Remove Designer/Scenario launchers | ~40 lines removed |
| `hydra_plugin.py` | Remove Designer/Scenario menu actions | ~10 lines removed |
| `swe2d/workbench/monolith_methods.py` | Remove shell re-exports | ~5 lines removed |
| `AGENTS.md` | Update Studio UI section | ~10 lines changed |
| `docs/STUDIO_UI_ARCHITECTURE.md` | Rewrite for single-path architecture | Major rewrite |
| `docs/SWE2D_GPU_ARCHITECTURE_REPORT.md` | Update Section 7 | ~20 lines changed |
| `swe2d/workbench/extracted/__init__.py` | Clean exports | ~5 lines |
