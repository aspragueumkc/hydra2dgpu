# MVP Architecture — Model-View-Presenter

This project follows **MVP (Model-View-Presenter)** architecture. Every code change MUST respect these layer boundaries.

## The Three Layers

```
┌──────────────────────────────────────────┐
│  View Layer (Qt UI)                       │
│  - swe2d/workbench/studio_dialog.py       │
│  - swe2d/workbench/views/*.py             │
│  - Owns QWidget references                │
│  - Exposes protocol methods:              │
│      get_xxx(), set_xxx_enabled(bool)     │
│      Read widgets → return plain data     │
│  - Owns all plain data state              │
│      (_snapshot_timesteps, _cancel_req,   │
│       _line_snapshot_rows, etc.)          │
│  - NO numpy math on mesh data             │
│  - NO service-function callbacks          │
├──────────────────────────────────────────┤
│  Controller (orchestration only)          │
│  - swe2d/workbench/controllers/*.py       │
│  - Receives View signals                  │
│  - Calls Service methods                  │
│  - Calls View protocol methods            │
│  - MAY read/write plain data attributes   │
│    on the View (lists, dicts, bools,      │
│    floats, strings, None)                 │
│  - MUST NOT reach through View to access  │
│    Qt widgets (use protocol methods)      │
│  - NO numpy math                          │
├──────────────────────────────────────────┤
│  Service Layer (pure Python, zero Qt)     │
│  ┌─────────────────────────────────────┐ │
│  │ Shared Services (GUI + CLI)          │ │
│  │ - swe2d/services/*.py                │ │
│  │ - swe2d/runtime/*.py                 │ │
│  │ - swe2d/boundary_and_forcing/*.py    │ │
│  │ - swe2d/mesh/*.py                    │ │
│  │ - swe2d/results/*.py                 │ │
│  │ - swe2d/extensions/*.py              │ │
│  │ - NO Qt imports                      │ │
│  │ - NO widget access                   │ │
│  └─────────────────────────────────────┘ │
│  ┌─────────────────────────────────────┐ │
│  │ GUI-only Services                    │ │
│  │ - swe2d/workbench/services/*.py      │ │
│  │ - May import qgis.core               │ │
│  │ - Must NOT import PyQt5/QtWidgets    │ │
│  └─────────────────────────────────────┘ │
│  - Owns all numpy computation            │
│  - Returns plain data structures         │
│  - Does NOT push computation back to View│
│    via callbacks that take mesh data     │
└──────────────────────────────────────────┘
```

## Hard Rules

## Hard Rules

Three rules, all grep-enforceable:

### Rule 1: Controller widget boundary

Controllers MAY read/write plain Python attributes on the View (lists, dicts, bools, floats, strings, None). Controllers MUST NOT reach through the View to access Qt widgets.

```bash
# VIOLATION: controller reaching through View to a widget
grep -n "view\._model_tab_view\.\|view\._map_tab_view\.\|view\._topo_widgets\." \
  swe2d/workbench/controllers/*.py | grep -v "getattr\|is None"

# ALLOWED: controller reading/writing plain data
# view._snapshot_timesteps = []
# view._cancel_requested = True
# view._line_snapshot_rows.clear()
```

### Rule 2: Protocol methods return plain data

View protocol methods MUST return plain data (strings, ints, floats, bools, lists, dicts), NOT widget references. This prevents the controller from obtaining a widget handle through a "合法" protocol call.

```bash
# VIOLATION: protocol method returning a widget reference
grep -n "return.*self\._model_tab_view\.\|return.*self\._map_tab_view\." \
  swe2d/workbench/studio_dialog.py | grep -v "# "

# CORRECT: protocol method returns the widget's value
# def get_run_duration_text(self) -> str:
#     return self._model_tab_view.run_time_edit.text()
```

### Rule 3: Service layer is Qt-free

```bash
# VIOLATION: service touching Qt widgets
! grep -q '\.setEnabled\|\.setText\|\.setValue\|\.currentText\|\.isChecked\|QPushButton\|QComboBox\|QDockWidget' swe2d/runtime/ swe2d/boundary_and_forcing/ swe2d/workbench/*service*.py && echo "PASS"

# VIOLATION: service importing Qt
! grep -q 'from qgis\|from PyQt\|import qgis' swe2d/runtime/ swe2d/boundary_and_forcing/ && echo "PASS"
```

### Rule 4: View does no numpy computation

```bash
# VIOLATION: View doing numpy computation on mesh geometry
! grep -q 'np\.min\|np\.max\|np\.vstack\|np\.argmin\|np\.where\|np\.hypot\|np\.zeros.*shape' swe2d/workbench/studio_dialog.py swe2d/workbench/views/ 2>/dev/null && echo "PASS: no numpy math in View"
```

### Rule 5: No __getattr__ proxy

```bash
# VIOLATION: View proxying attribute access to sub-views
! grep -q '__getattr__' swe2d/workbench/studio_dialog.py && echo "PASS: no __getattr__ proxy"
```

### Rule 6: No widget reparenting

Views MUST NOT reparent widgets from another view's layout. Each view owns its own widgets. Cross-view widget sharing is done through protocol methods (read/write state), not by moving widget objects between parents.

```bash
# VIOLATION: widget reparenting via setParent or layout theft
grep -n "\.setParent\|\.setWidget\|layout().addWidget.*self\._.*_tab_view\." \
  swe2d/workbench/studio_dialog.py | grep -v "# " | grep -v "_build_component\|_compose_left\|_populate_"
```

### Rule 7: No silent fallback on widget reads

Controllers MUST NOT use `.get(key, default)` with a fallback value when reading widget state from `collect_run_widget_params()` or similar protocol dicts. If the widget key is missing, that's a wiring bug — it must raise, not silently substitute.

```bash
# VIOLATION: controller using .get() fallback for widget values
grep -n 'wp\.get.*False)\|wp\.get.*True)\|wp\.get.*None)' \
  swe2d/workbench/controllers/run_controller.py
```

### Rule 8: No backwards-compatibility fallbacks that violate architecture

Never include a fallback path that reads widgets directly from the dialog when the correct path (through a sub-view, toolbox, or protocol method) is available. A "backwards compatibility" fallback that bypasses the architecture is still a violation. If a caller hasn't been updated to use the new path, fix the caller — don't add a silent degradation path.

```bash
# VIOLATION: service falling back to reading dialog widgets directly
#   tb = getattr(view, "_results_toolbox", None)
#   if tb is not None:
#       return getattr(tb, name)
#   return getattr(view, "high_perf_canvas_overlay_" + name, default)  # ← VIOLATION

# VIOLATION: controller falling back to old sub-view when new one exists
#   results = getattr(view, "_results_toolbox", None) or getattr(view, "_old_results_widget")

# CORRECT: fail fast — the new path is the only path
#   return getattr(view._results_toolbox, name)
```

## Widget Migration Checklist

When moving widgets between views, or adding/removing widgets that cross layer boundaries, follow this checklist in order. Each step is mandatory.

### Step 1 — Move widgets in the View layer

**Remove from the source view:**
1. Delete the widget construction code (e.g., `_build_actions_page` in `MapTabView`)
2. Delete any protocol methods on the source view that existed solely for the moved widget (e.g., `is_inflow_progressive`, `get_inflow_progressive_chk`)
3. Remove the widget from `view_protocols.py`'s source view Protocol (e.g., `MapTabViewProtocol`)

**Add to the destination view:**
1. Add the widget construction code using the canonical `_start_param_group` / `_add_param_row` pattern
2. Add protocol accessor methods: `is_xxx()`, `get_xxx_widget()` (if needed for signal wiring), `get_xxx_value()` (for plain-data return)
3. Add to `view_protocols.py`'s destination view Protocol (e.g., `ModelTabViewProtocol`)

**Order matters:** Place the new group in the destination view's `_build_*` method in the correct visual position (e.g., after "Time Stepping").

### Step 2 — Update dialog/controller references

Search the entire `swe2d/workbench/` tree for all references to the moved widget:

```bash
# Find every reference to the old widget location
grep -rn "_old_view_name\.\(widget_name\|method_name\)" swe2d/workbench/ --include="*.py"
```

Update each reference to use the **new view** (e.g., `_model_tab_view` instead of `_map_tab_view`). Apply Rule 8: do NOT add a fallback path — fix every call site.

Typical call sites:
- `studio_dialog.py` — dialog methods that read the widget (e.g., `_collect_boundary_arrays`, `_distribute_total_flow_to_unit_q`, `collect_params`)
- `run_controller.py` — `wp["widget_name"]` entries in `collect_run_widget_params`
- `non_gui_runtime_service.py` — `hasattr(wb, "widget_name")` or `hasattr(wb._model_tab_view, "widget_name")`

### Step 3 — Update service-layer references

If a service accesses the widget via `hasattr(wb, "widget_name")`, update the path to the new view:

```python
# WRONG — stale path to old view:
if not hasattr(wb, "uniform_inflow_velocity_chk"):
    return None

# CORRECT — updated path to new view:
if not hasattr(wb._model_tab_view, "uniform_inflow_velocity_chk"):
    return None
```

If the service needs to check for the widget's existence on the dialog (not sub-view), use `getattr(dialog, "_model_tab_view", None)` and check `is not None` before accessing.

### Step 4 — Update tests

1. Add tests for the new widget presence on the destination view (e.g., `test_view_has_default_bc_type_combo`)
2. Add tests for new protocol methods on the destination view (e.g., `test_boundary_conditions_methods`)
3. Update group-box enumeration tests on the destination view (add the new group title)
4. Remove tests that expect the widget on the old view (e.g., BC widgets no longer on Map tab)

### Step 5 — Verify

Run all relevant tests and the architecture enforcement checks:

```bash
# View tests pass
python -m pytest tests/test_model_tab_view.py tests/test_map_tab_view.py -v

# Architecture checks pass
! grep -q 'from qgis\|from PyQt' swe2d/runtime/ swe2d/boundary_and_forcing/ && echo "PASS: shared service clean"
! grep -q '\.setEnabled\|\.setText\|\.setValue' swe2d/workbench/controllers/ && echo "PASS: controller clean"
! grep -q '__getattr__' swe2d/workbench/studio_dialog.py && echo "PASS: no proxy"

# No stale references remain (should return no output)
grep -rn "_old_view_name\.\(widget_name\|method_name\)" swe2d/workbench/ --include="*.py"
```

### Quick reference: what goes where

| Item | Goes in |
|------|---------|
| Widget construction (`QSpinBox`, `QComboBox`, etc.) | View (`_build_*_form_widgets`) |
| Group box creation (`_start_param_group`) | View |
| Protocol accessor methods (`get_xxx()`, `is_xxx()`) | View |
| Protocol definition (`def get_xxx(...) -> int:`) | `view_protocols.py` |
| Dialog method reading widget value | `studio_dialog.py` |
| Controller reading widget via protocol | `run_controller.py` or other controller |
| Service checking widget existence | `non_gui_runtime_service.py` (update path) |
| Tests for widget presence | `test_model_tab_view.py` or `test_map_tab_view.py` |

## Computational Ownership

**Service Layer owns all numpy computation.** If a View method or Controller method calls `np.min()`, `np.max()`, `np.vstack()`, `np.argmin()`, `np.where()`, `np.hypot()` or any numpy function on mesh geometry arrays, that's a violation. The computation must be moved into a service module.

**Exception**: `np.zeros()`, `np.array()`, `np.float64`, `np.int32` used solely for constructing return arrays from widget values (not computation) are acceptable in View methods.

## Callback Anti-Pattern

Service functions MUST NOT pass raw mesh geometry to View callbacks:

```python
# WRONG — service pushes computation to View:
def collect_boundary_arrays(
    *,
    default_bc_for_edges_fn: Callable[[ndarray, ndarray], ...],
    ...
):
    bc_type, bc_val = default_bc_for_edges_fn(edge_n0, edge_n1)
    # View receives edge_n0, edge_n1 → must do numpy math → violation

# CORRECT — service owns computation, View provides data:
def collect_boundary_arrays(
    *,
    read_boundary_widget_fn: Callable[[ndarray], Tuple[ndarray, ndarray]],
    ...
):
    side_idx = classify_boundary_edges(edge_n0, edge_n1, node_x, node_y)  # service does math
    bc_type, bc_val = read_boundary_widget_fn(side_idx)  # View just reads widgets
```

## Widget Access Pattern

**Service Layer**: Receives data via constructor callbacks or method arguments. NEVER references a widget by name.

```python
# CORRECT (service):
class SWE2DRunLifecycle:
    def __init__(self, ui):  # ui is a protocol, not a widget bag
        self._ui = ui
    def finalize_cleanup(self, backend):
        self._ui.set_run_button_enabled(True)  # calls View protocol method
```

**View Layer**: Exposes protocol methods, not raw widget attributes. Reads widgets → returns plain data. Delegates all computation to services. Protocol methods MUST return plain data, never widget references.

```python
# CORRECT (view):
def set_run_button_enabled(self, enabled: bool) -> None:
    self._model_tab_view.run_btn.setEnabled(enabled)

def get_run_duration_text(self) -> str:
    return self._model_tab_view.run_time_edit.text()

# WRONG (view — returning widget reference):
def get_h_min_spin(self):
    return self._model_tab_view.h_min_spin  # ← VIOLATION: returns widget

# WRONG (view — computation in view):
def _default_bc_for_edges(self, edge_n0, edge_n1):
    xmin = float(np.min(self._mesh_data["node_x"]))  # ← VIOLATION
    ...
```

**Controller**: Calls View protocol methods, never touches widget properties.

```python
# CORRECT (controller):
view.set_run_button_enabled(False)
duration = view.get_run_duration_text()

# WRONG (controller):
view.run_btn.setEnabled(False)
duration = view._model_tab_view.run_time_edit.text()
```

## Enforcement

Before marking any code change complete, run:

```bash
# 1. No Qt in service layer
! grep -q 'from qgis\|from PyQt\|\.setEnabled\|\.setText\|\.setValue' swe2d/runtime/ swe2d/boundary_and_forcing/ && echo "PASS: service layer clean"

# 2. No raw widget access in controller
! grep -q '\.setEnabled\|\.setText\|\.setValue\|\.isChecked' swe2d/workbench/workbench_controller.py && echo "PASS: controller clean"

# 3. No numpy computation in View
! grep -q 'np\.min\|np\.max\|np\.vstack\|np\.argmin\|np\.where\|np\.hypot' swe2d/workbench/studio_dialog.py swe2d/workbench/views/ 2>/dev/null && echo "PASS: view has no computation"

# 4. No service callbacks that force View to compute
! grep -q 'def _default_.*(self.*np\.\|:.*np\.min\|:.*np\.max' swe2d/workbench/studio_dialog.py 2>/dev/null && echo "PASS: no computation callbacks in View"
```

## Import Path Verification

View methods that delegate to service-layer functions via local import MUST point to the correct module. A systematic check:

```bash
# Extract all `from swe2d.xxx import YYY as _logic` lines and verify the module exists
python3 -c "
import ast, os, sys
with open('swe2d/workbench/studio_dialog.py') as f:
    tree = ast.parse(f.read())
errors = []
for node in ast.walk(tree):
    if isinstance(node, ast.ImportFrom):
        if node.module and node.module.startswith('swe2d'):
            for alias in node.names:
                if alias.asname == '_logic':
                    # Verify the imported name exists in the module
                    parts = node.module.split('.')
                    mod_path = os.path.join(*parts) + '.py'
                    if not os.path.exists(mod_path):
                        errors.append(f'  Module not found: {node.module} (line ~{node.lineno})')
if errors:
    print('IMPORT ERRORS:')
    for e in errors:
        print(e)
else:
    print('PASS: all imports resolve')
"
```

## Positional-Argument Trap

Service functions using keyword-only arguments (`*` in signature) MUST be called with keywords, never positional:

```bash
# Check for _logic() calls with positional args to keyword-only functions
python3 -c "
import ast
with open('swe2d/workbench/studio_dialog.py') as f:
    tree = ast.parse(f.read())
violations = 0
for node in ast.walk(tree):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == '_logic':
        if node.args:
            violations += 1
            print(f'  Line {node.lineno}: {len(node.args)} positional args to _logic()')
if violations == 0:
    print('PASS: no positional calls to keyword-only functions')
"
```

If either check fails, the architecture is broken — the computation must move into a service module. Do not add `import numpy` to the View to make it compile.

## Proxy Anti-Pattern

`__getattr__` on the View that falls through to tab views is prohibited. It silently routes widget access from the service layer through to Qt objects, masking architecture violations:

```bash
# VIOLATION: View proxying attribute access to sub-views
! grep -q '__getattr__' swe2d/workbench/studio_dialog.py && echo "PASS: no __getattr__ proxy"
```

Service layer methods MUST receive their data as explicit parameters — not reach through `self._ui` to grab widgets by name. Each `self._ui.xxx` call in a service module means the wrong layer is doing widget I/O.

## Layout Construction Patterns

### Rule 8: Toolbox page builders follow a canonical pattern

Every `QToolBox` page MUST be built by a method that:
1. Creates a `QWidget` page
2. Sets its `objectName`
3. Creates a layout parented to the page at creation time (`QFormLayout(page)`, `QVBoxLayout(page)`, etc.)
4. Populates widgets into that layout
5. Calls `toolbox.addItem(page, "Title")`

**Canonical pattern:**

```python
def _build_xxx_page(self, toolbox: QtWidgets.QToolBox) -> None:
    page = QtWidgets.QWidget()
    page.setObjectName("xxx_page")
    layout = QtWidgets.QFormLayout(page)  # ← parented at creation
    layout.setContentsMargins(0, 0, 0, 0)
    # ... populate widgets via layout.addRow(...) ...
    toolbox.addItem(page, "Page Title")
```

**Anti-pattern — freestanding layout, never parented:**

```python
# WRONG: layout is never parented to any widget — widgets exist in memory
# but are never visible in the UI.
def _build_results_controls(self):
    layout = QtWidgets.QGridLayout()           # ← no parent!
    layout.setObjectName("results_layout")
    self._some_widget = QtWidgets.QCheckBox()  # created, configured, but invisible
    layout.addWidget(self._some_widget)
    self._results_layout = layout              # stored, never used by any parent
```

This anti-pattern silently produces invisible controls. The widgets exist, their signals fire, but no user can see or interact with them. Always use the canonical pattern: create a page widget, parent the layout to it, add the page to the toolbox.

```bash
# VIOLATION: freestanding layout constructor with no parent widget
grep -n "QGridLayout()\|QVBoxLayout()\|QFormLayout()\|QHBoxLayout()" \
  swe2d/workbench/views/*.py | grep -v "(self\|(page\|(toolbox\|(dialog\|(dlg"
```
