# SWE2D Studio UI Component Guide

Developer guide for adding docks, tabs, signal connections, and feature
toggles using the Studio UI component API.

---

## 1. Adding a new left-pane tab

Two approaches:

**Module-level registration** — tab is registered at import time and
picked up automatically by `_compose_left_pane()`:

```python
# In swe2d/workbench/studio_main.py or a separate module:
from swe2d.workbench.studio_component import register_studio_tab

def _build_my_tab_page(dialog):
    page = QWidget()
    layout = QVBoxLayout(page)
    layout.addWidget(QLabel("My tab content"))
    return page

register_studio_tab("My Tab", _build_my_tab_page)
```

**Instance-level registration** — tab is added during `_build_ui()`:

```python
def _build_my_tab_page(self):
    page = QWidget()
    page.setObjectName("my_tab_page")
    # ... populate from .ui or programmatically ...
    return page

# In _build_ui() or _compose_left_pane():
self._register_left_tab("My Tab", self._build_my_tab_page)
```

Tabs are iterated in `_compose_left_pane()` at `studio_main.py:424`:
```python
for name, builder in get_studio_tab_builders().items():
    self._left_tabs.addTab(builder(self), name)
```

---

## 2. Adding a new dockable panel

Define a `populate` callback, then call `_build_component()`:

```python
def _populate_my_panel(self, dock: QDockWidget) -> None:
    inner = QWidget()
    layout = QVBoxLayout(inner)
    layout.addWidget(QLabel("Hello from My Panel"))
    dock.setWidget(inner)

# Single call in _build_ui():
self._build_component(
    name="my_panel",
    title="My Panel",
    area=Qt.RightDockWidgetArea,
    tab_with="inspector",   # optional — tabs inside inspector dock
    populate=self._populate_my_panel,
)
```

`_build_component()` (`studio_main.py:1901`) creates the QDockWidget,
calls `populate()`, builds a `StudioComponent`, and registers it via
`_register_component()` (`studio_main.py:1965`).

---

## 3. Connecting a signal safely

Use `safe_connect` to prevent duplicate connections on rebuild, and
`connect_lambda` for weak-reference lambda safety:

```python
from swe2d.workbench.signal_helpers import safe_connect, connect_lambda

# Idempotent — disconnects first if already connected:
safe_connect(self.run_btn.clicked, self._on_run_clicked)

# Weak-ref lambda — no crash if `self` is GC'd before signal fires:
connect_lambda(action.triggered, self, "_studio_select_tab", "mesh")

# Equivalent manual version (used internally by connect_lambda):
import weakref
_ref = weakref.ref(self)
action.triggered.connect(lambda: (
    _ref() and _ref()._studio_select_tab("mesh")
))
```

---

## 4. Cleaning up on dialog close

The `closeEvent` at `studio_main.py:2484` iterates all registered
components and destroys them:

```python
def closeEvent(self, event):
    self._save_studio_layout_state()
    fut = getattr(self, "_mesh_future", None)
    if fut is not None:
        fut.cancel()
    for name in list(self._studio_components.keys()):
        self._destroy_component(name)
    super().closeEvent(event)
```

`_destroy_component()` (`studio_main.py:1993`) calls `safe_teardown()`,
closes the dock, and schedules deletion. You don't need to write
any additional cleanup for registered components — the framework
handles it.

For ad-hoc signal cleanup, use `safe_disconnect()` and `safe_teardown()`:

```python
from swe2d.workbench.signal_helpers import safe_disconnect, safe_teardown

safe_disconnect(self.run_btn.clicked, self._on_run_clicked)
safe_teardown(widget)
widget.deleteLater()
```

---

## 5. Adding a feature toggle

Three files must be updated together (see `studio_main.py:1815-1819`):

### 5a. Register the flag key in `__init__` (`studio_main.py:191`):

```python
self._studio_feature_flags = {
    "my_feature": True,
}
```

### 5b. Add keyword entries (`studio_main.py:1827`):

```python
def _studio_feature_keywords(self):
    return {
        "my_feature": ("myfeat", "special", "thing"),
        # ...
    }
```

Widgets whose `objectName`, `text`, `title`, or `toolTip` contain any
keyword will be hidden when the flag is disabled.

### 5c. Add menu/toolbar toggle in `_install_studio_host_controls()`
(`studio_main.py:2929`):

```python
my_act = menu.addAction("Enable My Feature")
my_act.setCheckable(True)
my_act.setChecked(True)
my_act.toggled.connect(
    lambda checked: dlg._studio_set_feature_enabled("my_feature", checked)
)
```

### 5d. Toggle the flag at runtime:

```python
self._studio_set_feature_enabled("my_feature", False)
```

This calls `_studio_apply_feature_filters()` (`studio_main.py:1858`)
which iterates all left-pane widgets and tabs, hides any whose text
matches disabled feature keywords, and adjusts tab bar visibility.

---

## Canvas overlay

The high-perf overlay path uses `SWE2DHighPerfCanvasOverlayItem`
(`swe2d_high_perf_viewer.py:1152`), a `QgsMapCanvasItem` subclass:

```python
from swe2d_high_perf_viewer import SWE2DHighPerfCanvasOverlayItem

item = SWE2DHighPerfCanvasOverlayItem(canvas)
item.setImage(image)    # QImage with rendered frame
item.setExtent(xmin, xmax, ymin, ymax)
item.setOpacity(0.65)
item.setVisible(True)
canvas.refresh()
```

Used in the studio dialog at `studio_main.py:1107` for simulation
frame display.
