# SWE2D Studio UI Architecture

## Philosophy

- **`.ui` files are the source of truth for widget layout and properties.**
  Use Qt Designer to edit them. Do NOT manually create widgets in Python
  that could live in a `.ui` file.

- **Python bind methods wire behaviour** (combo population, signal connections,
  tooltips, validation) to widgets loaded from `.ui` files.

- **`tools/ui_bind_sync.py`** automates the cleanup of stale Python bindings
  when widgets are removed from `.ui` files, and reports new widgets that
  need bindings (`--missing` mode).

- **Widget binding validation**: `SWE2DWorkbenchStudioDialog._build_ui()`
  calls `_validate_widget_bindings()` after all tabs are composed.  Critical
  widgets (e.g. run button) raise `RuntimeError` if missing; optional widgets
  log a warning.  Run `python tools/ui_bind_sync.py forms/swe2d_<name>.ui <py_files> --missing`
  for full `.ui`-level validation.

---

## Structural Change Checklist

When you modify the Studio UI structure (add/remove/move tabs, forms, or
feature toggles), touch EVERY item in the relevant checklist below.

### A. Adding a new tab page to the left pane (e.g. "3D Patch")

| Step | Where | What |
|------|-------|------|
| 1 | `forms/swe2d_*.ui` | Create the new `.ui` file with a QFormLayout |
| 2 | `swe2d_workbench_qt.py` | Add `_build_<name>_tab_page()` that loads the `.ui` |
| 3 | `swe2d_workbench_qt.py` | Add `_bind_<name>_controls()` or reuse an existing bind method |
| 4 | `swe2d_workbench_qt.py` | Add tab to `_compose_left_pane()` via `_left_tabs.addTab()` |
| 5 | `swe2d/workbench/extracted/model_and_run_methods.py` | Add `_find_or_create_*` calls for each interactive widget |
| 6 | Run `tools/ui_bind_sync.py forms/swe2d_<name>.ui <py_files> --missing` | Verify all widgets have bindings |
| 7 | Purge `__pycache__` before testing in QGIS | `find . -type d -name __pycache__ -exec rm -rf {} +` |

### B. Adding a new form to an existing tab page (e.g. a new QToolBox page)

| Step | Where | What |
|------|-------|------|
| 1 | `forms/swe2d_model_tab.ui` | Add QToolBox page with QFormLayout in Qt Designer |
| 2 | `swe2d_workbench_qt.py` | Update `_build_model_tab_page()` to find and return the new form |
| 3 | `swe2d_workbench_qt.py` | Add `_bind_<name>_controls(model_tab_page, new_form)` call in `_compose_left_pane()` |
| 4 | `swe2d/workbench/extracted/model_and_run_methods.py` | Add `_find_or_create_*` calls for each interactive widget in the new form |
| 5 | Run `tools/ui_bind_sync.py forms/swe2d_model_tab.ui <py_files> --missing` | Verify form coverage |
| 6 | Purge `__pycache__` | |

### C. Adding/removing a feature toggle (Rainfall, Drainage/Structures, 3D Patch)

| Step | Where | What |
|------|-------|------|
| 1 | `swe2d_workbench_qt.py` | Add/remove entry in `self._studio_feature_flags` dict |
| 2 | `swe2d_workbench_qt.py` | Add/remove entry in `_studio_feature_keywords()` |
| 3 | `swe2d_workbench_qt.py` | Ensure `_studio_apply_feature_filters()` handles tab visibility |
| 4 | `swe2d/workbench/extracted/studio_host_methods.py` | Add/remove menu action (QGIS menu bar) |
| 5 | `swe2d/workbench/extracted/studio_host_methods.py` | Add/remove toolbar button (QGIS toolbar) |
| 6 | Verify keyword matching: interactive widgets in the target tab MUST have objectNames containing at least one keyword |
| 7 | Set the QScrollArea wrapper's `objectName` if the tab itself needs to be hidden via `setTabVisible()` |

### D. Moving widgets between forms/pages in a `.ui` file

| Step | Where | What |
|------|-------|------|
| 1 | `forms/swe2d_*.ui` | Move widget elements in Qt Designer |
| 2 | `swe2d/workbench/extracted/model_and_run_methods.py` | If the bind method uses `param_form` explicitly, update the target form |
| 3 | `swe2d_workbench_qt.py` | If the bind wrapper passes a specific form, verify it's the correct one |
| 4 | Run `tools/ui_bind_sync.py forms/swe2d_*.ui <py_files> --missing` | Verify no orphaned or unbound widgets |
| 5 | Purge `__pycache__` | |

### E. Renaming widgets in a `.ui` file

| Step | Where | What |
|------|-------|------|
| 1 | `forms/swe2d_*.ui` | Rename widget in Qt Designer |
| 2 | `swe2d/workbench/extracted/model_and_run_methods.py` | Update `_find_or_create_*("old_name", ...)` → `_find_or_create_*("new_name", ...)` |
| 3 | Any `.py` file with `findChild(..., "old_name")` | Update the string literal |
| 4 | Run `tools/ui_bind_sync.py forms/swe2d_*.ui <py_files>` | Removes old-name references |
| 5 | `swe2d_workbench_qt.py` | Update `_studio_feature_keywords()` if the name contained a keyword |
| 6 | Purge `__pycache__` | |

---

## Key Methods Reference

### Tab builders (`swe2d_workbench_qt.py`)

| Method | Loads `.ui` | Returns |
|--------|------------|---------|
| `_build_mesh_tab_page()` | `swe2d_mesh_tab.ui` | QWidget |
| `_build_map_tab_page()` | `swe2d_map_tab.ui` | (page, data_layout, actions_layout, results_layout, tools_layout) |
| `_build_topology_tab_page()` | `swe2d_topology_tab.ui` | (page, topo_layout) |
| `_build_boundary_tab_page()` | `swe2d_boundary_tab.ui` | QWidget |
| `_build_model_tab_page()` | `swe2d_model_tab.ui` | (page, solver_form, rain_form, drain_form) |
| `_build_3d_patch_tab_page()` | `swe2d_3d_patch_tab.ui` | QWidget |
| `_build_run_tab_page()` | `swe2d_run_tab.ui` | QWidget |

### Tab composition (`_compose_left_pane`)

All tabs are added to `self._left_tabs` (QTabWidget) via `addTab()`.
Each tab page is wrapped in `_wrap_left_tab_page()` which returns a QScrollArea.
**New tabs must be added here** to appear in the Studio left pane.

### Feature flag pipeline

1. **Flag store**: `self._studio_feature_flags` dict in `SWE2DWorkbenchStudioDialog.__init__()`
2. **Keyword matching**: `_studio_feature_keywords()` in `SWE2DWorkbenchStudioDialog`
3. **Filter application**: `_studio_apply_feature_filters()` iterates `_left_tabs` children and calls `setVisible()` / `setTabVisible()`
4. **User toggle**: menu actions + toolbar buttons in `_install_studio_host_controls()` (in `studio_host_methods.py`), wired to `_studio_set_feature_enabled()`

### Bind methods → implementation mapping

| Wrapper in `swe2d_workbench_qt.py` | Implementation in `model_and_run_methods.py` |
|-------------------------------------|---------------------------------------------|
| `_bind_model_tab_core_controls(page, solver_form)` | `_bind_model_tab_core_controls` |
| `_bind_model_tab_hydrology_controls(page, rain_form)` | `_bind_model_tab_hydrology_controls` |
| `_bind_model_tab_solver_controls(page, solver_form)` | `_bind_model_tab_solver_controls` |
| `_bind_model_tab_3d_subgrid_drainage_controls(page, drain_form, solver_form)` | `_bind_model_tab_3d_subgrid_drainage_controls` |
| `_bind_model_tab_3d_patch_controls(page, patch_form)` | `_bind_model_tab_3d_patch_controls` |

---

#### Studio (modern): `SWE2DWorkbenchStudioDialog`

- Used when user clicks "2D SWE Workbench" in QGIS toolbar
- `_compose_left_pane()` builds the left tab widget with all tabs
- Each tab is loaded from its own `.ui` file
- Feature flags control tab/page visibility
- Widget binding validation run on `_build_ui()` completion
