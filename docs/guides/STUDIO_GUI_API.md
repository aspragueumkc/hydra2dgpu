# Studio GUI API

> **Status:** Public API for the Studio workbench GUI. Stable for the MVP architecture introduced in Phase 1-6 of the alignment plan.

## Overview

The Studio workbench GUI follows a **Model-View-Presenter (MVP)** layered architecture with a thin plugin entry point. This document is the public contract for the API.

```
┌─────────────────────────────────────────────────────────────┐
│  Plugin Entry (thin)                                        │
│  - __init__ just creates dialog and shows it                │
│  - No business logic                                        │
└─────────────────────────────────────────────────────────────┘
         │ creates & shows
         ▼
┌─────────────────────────────────────────────────────────────┐
│  View (Qt UI)                                               │
│  - Tab QWidget subclasses (Mesh, Map, Topo, Boundary, Model)│
│  - Owns widget references                                   │
│  - Implements WorkbenchView protocol                        │
└─────────────────────────────────────────────────────────────┘
         ▲                          ▲
         │ update()                 │ signal
         │                          │
┌────────┴────────────────────────────────────────────────────┐
│  Controller / Presenter                                     │
│  - WorkbenchController (the brain)                          │
│  - Receives View signals, calls services, updates View     │
└─────────────────────────────────────────────────────────────┘
         │                          ▲
         │ call                     │ return
         ▼                          │
┌─────────────────────────────────────────────────────────────┐
│  Service Layer (zero Qt)                                    │
│  - run_service, gpkg_service, mesh_service, etc.            │
│  - Pure Python business logic                               │
└─────────────────────────────────────────────────────────────┘
```

## Public API Module

All public protocols and types are exported from `swe2d.workbench.workbench_api`:

```python
from swe2d.workbench.workbench_api import (
    WorkbenchView,
    OverlayView,
    WorkbenchControllerProtocol,
    MeshSnapshotLoader,
    OverlayParametersCollector,
    OverlayViewInterface,  # backwards-compatible alias for OverlayView
)
```

## View Interfaces

### `WorkbenchView`

The protocol that any View (typically the Studio dialog) must satisfy for the Controller to interact with it.

```python
class WorkbenchView(Protocol):
    _results_panel: Any          # results panel widget
    _high_perf_overlay_cell_x: Any  # numpy array of cell x-coords
    _mesh_data: Any              # current mesh data dict
    def _log(self, msg: str) -> None: ...  # log message to runtime log
```

The Controller reads these attributes and calls `_log()` for logging. The dialog's `_controller._view` is the dialog itself.

### `OverlayView`

The protocol for a View providing overlay parameters. Implemented by tab widgets that contain overlay controls.

```python
class OverlayView(Protocol):
    def get_field_key(self) -> str: ...
    def get_colormap(self) -> str: ...
    def get_opacity(self) -> float: ...
    # ... (19 getter methods total)
    def get_h_min(self) -> float: ...
    def get_gravity(self) -> float: ...
```

The `collect_overlay_parameters()` service accepts any object satisfying this protocol and returns a plain dict.

## Controller Interface

### `WorkbenchControllerProtocol`

The protocol the View uses to call the Controller.

```python
class WorkbenchControllerProtocol(Protocol):
    _view: WorkbenchView
    def load_mesh_snapshot_for_overlay(self, t_s: float) -> bool: ...
```

The Studio dialog's `_controller` attribute conforms to this protocol.

## Service Interfaces

### `MeshSnapshotLoader`

A callable that loads mesh snapshot data from a GPKG file. The signature is:

```python
def load_mesh_snapshot(
    gpkg_path: str,
    run_id: str,
    t_s: float,
) -> Optional[Dict[str, Any]]:
    """Returns dict with keys: h, hu, hv (numpy arrays), t_s (float), cell_count (int).
    Returns None if the data is not available."""
```

The default implementation is `swe2d.workbench.gpkg_service.load_mesh_snapshot`.

### `OverlayParametersCollector`

A callable that collects overlay parameters from a View.

```python
def collect_overlay_parameters(view: OverlayView) -> Dict[str, Any]:
    """Returns dict with all 19 overlay parameters keyed by snake_case name."""
```

The default implementation is `swe2d.workbench.overlay_parameters_service.collect_overlay_parameters`.

## Concrete Implementations

| Component | File | Class |
|-----------|------|-------|
| Dialog | `swe2d/workbench/studio_main.py` | `SWE2DWorkbenchStudioDialog` |
| Builder | `swe2d/workbench/workbench_dialog_builder.py` | `WorkbenchDialogBuilder` |
| Controller | `swe2d/workbench/workbench_controller.py` | `WorkbenchController` |
| Legacy adapter | `swe2d/workbench/legacy_methods_adapter.py` | `LegacyMethodsAdapter` |
| Mesh tab view | `swe2d/workbench/views/mesh_tab_view.py` | `MeshTabView` |
| Map tab view | `swe2d/workbench/views/map_tab_view.py` | `MapTabView` |
| Topology tab view | `swe2d/workbench/views/topology_tab_view.py` | `TopologyTabView` |
| Boundary tab view | `swe2d/workbench/views/boundary_tab_view.py` | `BoundaryTabView` |
| Model tab view | `swe2d/workbench/views/model_tab_view.py` | `ModelTabView` |
| Overlay service | `swe2d/workbench/overlay_parameters_service.py` | `collect_overlay_parameters()` |
| GPKG service | `swe2d/workbench/gpkg_service.py` | `load_mesh_snapshot()` |

## Usage Examples

### Reading overlay parameters from a view (no Qt coupling)

```python
from swe2d.workbench.overlay_parameters_service import collect_overlay_parameters
from swe2d.workbench.workbench_api import OverlayView

def render_overlay(view: OverlayView) -> None:
    params = collect_overlay_parameters(view)
    # params is a plain dict, no Qt dependency
    render_unstructured_snapshot_image(**params)
```

### Loading a mesh snapshot via the controller

```python
from swe2d.workbench.workbench_controller import WorkbenchController

controller = WorkbenchController(view=dialog)
snapshot_loaded = controller.load_mesh_snapshot_for_overlay(t_s=2.5)
if snapshot_loaded:
    # dialog._snapshot_timesteps is now populated
    ...
```

### Creating a custom tab view

A custom tab view is a QWidget subclass that implements the relevant protocol:

```python
from qgis.PyQt.QtWidgets import QWidget, QSpinBox
from swe2d.workbench.workbench_api import OverlayView

class MyCustomView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.opacity_spin = QSpinBox()
    
    def get_opacity(self) -> float:
        return float(self.opacity_spin.value()) / 100.0
    
    # ... other getters
```

## Stability

The protocols in `workbench_api` are part of the public API. Adding new methods to a protocol is allowed; removing or renaming methods requires a deprecation cycle.

Services and controllers can have their implementations swapped without affecting consumers that depend on the protocols.
