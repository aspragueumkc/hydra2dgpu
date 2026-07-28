---
type: spec
status: complete
created: 2026-07-14
completed: 2026-07-25
---

# HYDRA2DGPU QGIS Locator Integration — Design Specification

**Date:** 2026-07-14  
**Status:** Draft  
**Author:** Aaron Sprague

---

## 1. Problem Statement

HYDRA2DGPU plugin currently exposes all functionality through:
- Plugin menu (HYDRA2DGPU ▸ Open Workbench, Settings, etc.)
- Workbench dock UI (toolbar buttons, tab controls, context menus)
- Processing algorithms (registered separately)

Users must navigate menus or open the workbench dock to access features. QGIS provides a global **Locator Bar** (bottom-left search widget) that accepts typed queries and returns actionable results. Integrating HYDRA2DGPU into the locator enables power users to trigger actions instantly via keyboard (e.g., `Ctrl+K`, type `hydra open`, Enter).

---

## 2. Goals

- Register a `QgsLocatorFilter` with prefix `hydra` (5 chars, ≥3 required by QGIS)
- Expose 7 core actions as locator results:
  1. `hydra open` → Open HYDRA2DGPU Workbench
  2. `hydra close` → Close Workbench
  3. `hydra log` → Open Run Log Viewer
  4. `hydra batch` → Open Batch Simulation Dialog
  5. `hydra gpkg` → Open GPKG Explorer
  6. `hydra settings` → Open Settings Dialog
  7. `hydra help` → Open Help/Documentation
- Filter must be stateless, cloneable, and lightweight
- No configuration widget needed (actions are fixed)
- Register in `initGui()`, deregister in `unload()`
- Lazy imports to avoid heavy module loading at plugin startup

---

## 3. Non-Goals

- Full-text search across GPKG results (defer to future enhancement)
- Dynamic/fuzzy matching of action names
- Configuration dialog for the filter itself
- Search-as-you-type with live results (actions are static)

---

## 4. Architecture

### 4.1 QGIS Locator Integration Points

```
┌─────────────────────────────────────────────────────────────────┐
│  QGIS Application                                               │
│  ┌──────────────────┐                                           │
│  │ QgsLocator       │ ← registerFilter(filter)                  │
│  │                  │                                           │
│  │ filters: [       │                                           │
│  │   HydraLocatorFilter  ◄─── plugin registers here             │
│  │   ...other filters                                     │
│  │ ]                │                                           │
│  └──────────────────┘                                           │
└─────────────────────────────────────────────────────────────────┘
         ▲                      │
         │ fetchResults()       │ triggerResult()
         ▼                      ▼
┌─────────────────────────────────────────────────────────────────┐
│  HydraLocatorFilter (QgsLocatorFilter subclass)                 │
│  - prefix() → "hydra"                                           │
│  - displayName() → "HYDRA2DGPU"                                 │
│  - fetchResults(search, context, feedback)                      │
│       Returns QgsLocatorResult[] for matching actions           │
│  - triggerResult(result)                                        │
│       Dispatches to action based on result.userData             │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Filter Implementation

**File:** `swe2d/workbench/locator/hydra_locator_filter.py`

```python
class HydraLocatorFilter(QgsLocatorFilter):
    # QgsLocatorFilter abstract methods:
    def prefix(self) -> str: return "hydra"
    def displayName(self) -> str: return "HYDRA2DGPU"
    def clone(self): return HydraLocatorFilter()
    
    def fetchResults(self, search: str, context, feedback):
        # Parse search after "hydra" prefix
        # Return QgsLocatorResult for each matching action
        # result.userData = action identifier string
    
    def triggerResult(self, result):
        # Switch on result.userData
        # Call appropriate public API function
```

### 4.3 Action Mapping

| Locator Query | Action Key (`userData`) | Trigger Function |
|---------------|------------------------|------------------|
| `hydra open` | `open_workbench` | `launch_swe2d_workbench_studio(iface=self.iface)` |
| `hydra close` | `close_workbench` | `close_workbench_studio(iface=self.iface)` |
| `hydra log` | `open_run_log` | `controller.open_run_log_viewer()` |
| `hydra batch` | `open_batch` | `SWE2DBatchSimulationDialog(parent).exec_()` |
| `hydra gpkg` | `open_gpkg` | `GPKGExplorerDialog(parent, ...).exec_()` |
| `hydra settings` | `open_settings` | `plugin.open_settings()` |
| `hydra help` | `open_help` | `QDesktopServices.openUrl(QUrl("https://github.com/..."))` |

### 4.4 Plugin Registration

**In `hydra_plugin.py`:**

```python
def initGui(self):
    # ... existing code ...
    from swe2d.workbench.locator.hydra_locator_filter import HydraLocatorFilter
    self._locator_filter = HydraLocatorFilter()
    QgsApplication.locator().registerFilter(self._locator_filter)

def unload(self):
    # ... existing code ...
    if hasattr(self, '_locator_filter') and self._locator_filter:
        QgsApplication.locator().deregisterFilter(self._locator_filter)
        self._locator_filter = None
```

---

## 5. Dependencies

### 5.1 Lazy Imports

Filter methods will import heavy modules locally to avoid loading at plugin startup:

```python
def triggerResult(self, result):
    if result.userData == "open_workbench":
        from swe2d.workbench.views.studio_host_methods import launch_swe2d_workbench_studio
        launch_swe2d_workbench_studio(iface=self.iface)
    # ... etc
```

### 5.2 Required QGIS APIs

- `from qgis.core import QgsLocatorFilter, QgsLocatorResult, QgsApplication`
- `from qgis.PyQt.QtCore import QUrl`
- `from qgis.PyQt.QtGui import QDesktopServices`

---

## 6. Testing Strategy

### 6.1 Unit Tests (mock QGIS)

- Test `fetchResults()` returns correct results for each query
- Test `triggerResult()` dispatches to correct action (mock the API calls)
- Test `prefix()`, `displayName()`, `clone()`

### 6.2 Integration Test

- Verify filter registers/deregisters correctly in `initGui()`/`unload()`
- Manual QGIS test: open locator (Ctrl+K), type `hydra`, verify actions appear

---

## 7. File Structure

```
swe2d/workbench/
├── locator/
│   ├── __init__.py
│   └── hydra_locator_filter.py      # NEW: Main filter implementation
└── ...
hydra_plugin.py                       # MODIFIED: Register/deregister filter
tests/
├── test_locator_filter.py            # NEW: Unit tests for filter
└── ...
```

---

## 8. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Locator filter crashes QGIS on unload | Ensure `deregisterFilter()` called before any Qt objects deleted; test unload sequence |
| Heavy imports slow plugin startup | Use lazy imports inside methods, not module level |
| Actions become stale if API changes | Use stable public APIs (`launch_swe2d_workbench_studio`, controller methods) |
| Prefix conflict with other plugins | `hydra` is unique enough; QGIS allows user to disable filters |

---

## 9. Approval

**Reviewed by:** [ ] User  
**Date approved:** ___________  
**Status:** ☐ Approved / ☐ Needs Revision

---

## 10. Next Steps

After approval:
1. Invoke `writing-plans` skill to create detailed implementation plan
2. Implement `hydra_locator_filter.py`
3. Modify `hydra_plugin.py` to register filter
4. Add unit tests
5. Test in QGIS