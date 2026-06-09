# Dead Code & Silent Fallback Audit

Generated: 2026-06-09
Scope: All Python files in the repository (148 files scanned)

---

## Executive Summary

| Metric | Count |
|---|---|
| Python files scanned | 148 |
| Files with dead imports | 14 |
| Dead import statements | 18 |
| Wildcard imports (`from X import *`) | 23 (in 11 files) |
| Dead functions/classes/variables identified | 67 |
| Total `except` blocks | 869 |
| Silent `except` blocks (no logging) | 753 (87%) |
| Bare `except:` (no type specified) | 0 |
| `warnings.filterwarnings('ignore')` | 0 |

**Key finding: 87% of all exception handlers silently swallow errors without any logging, warning, or user notification.** This makes debugging field issues extremely difficult — failures in mesh generation, BC loading, coupling, and file I/O will fail silently and return empty/None values as if nothing went wrong.

---

## Part 1: Dead Code

### 1A. Unused Imports (18 across 14 files)

| File | Line | Dead Import | Notes |
|---|---|---|---|
| `hydra_plugin.py` | 9 | `Qt` from `qgis.PyPy.QtCore` | `QtCore` is used, but `Qt` itself never referenced |
| `swe2d_workbench_qt.py` | 19 | `import math` | No `math.xxx` calls detected in the file |
| `swe2d_workbench_qt.py` | 29 | `QtGui` from `qgis.PyQt` | `QtCore` and `QtWidgets` from the same import are used |
| `swe2d/mesh/meshing.py` | 28 | `Any` from `typing` | Not used in this 10272-line file |
| `swe2d/runtime/native_bc_forcing.py` | 9 | `import math` | No `math.` usage |
| `swe2d/runtime/native_binding_compat.py` | 22 | `Callable` from `typing` | Not referenced |
| `swe2d/runtime/run_options_builder.py` | 12 | `Dict` from `typing` | Not referenced |
| `swe2d/runtime/runtime_reporting.py` | 10 | `import os` | Not used |
| `tests/test_swe2d_backend_tiny_mode_config.py` | 11 | `SolverModelOptions` from `swe2d.extensions.extension_models` | Never referenced in test |
| `tools/benchmark_drainage_coupling.py` | 12 | `import os` | Not used |
| `tools/benchmark_drainage_coupling.py` | 19 | `defaultdict` from `collections` | Not used |
| `tools/create_trapezoid_sampleline_case.py` | 23 | `List` from `typing` | Not used |
| `tools/generate_swe2d_report.py` | 35 | `import os` | Not used |
| `tools/gmsh_topology_mesher.py` | 26 | `Sequence` from `typing` | Not used |
| `tools/restructure_model_tab.py` | 7 | `import re` | Not used |
| `tools/run_trapezoid_sampleline_validation.py` | 11 | `Sequence` from `typing` | Not used |
| `tools/run_trapezoid_sampleline_validation.py` | 307 | `QgsVectorLayer` from `qgis.core` | Lazy import inside function — not used |
| `typings/qgis/PyQt/uic.py` | 1 | `compileUi`, `loadUi` from `PyQt5.uic` | Type stub — benign |

### 1B. Wildcard Imports (23 across 11 files)

| File | Line | Import |
|---|---|---|
| `swe2d/boundary_and_forcing/__init__.py` | 3–13 | `from swe2d.boundary_and_forcing.bc_logic import *` (9 wildcards) |
| `swe2d/core/__init__.py` | 6 | `from swe2d.extensions.extension_models import *` |
| `swe2d/extensions/__init__.py` | 4 | `from swe2d.extensions.patch_qgis_adapter import *` |
| `swe2d/mesh/__init__.py` | 3–6 | `from swe2d.mesh.meshing import *` (4 wildcards) |
| `swe2d/results/__init__.py` | 7 | `from swe2d.results.queries import *` |
| `swe2d/workbench/__init__.py` | 6 | `from swe2d.workbench.non_gui_qgis import *` |
| `swe2d/workbench/__init__.py` | 7 | `from swe2d.workbench.non_gui_runtime import *` |
| `swe2d/workbench/extracted/model_and_run_methods.py` | 4 | `from swe2d_workbench_qt import *` |
| `swe2d/workbench/extracted/results_and_ui_methods.py` | 4 | `from swe2d_workbench_qt import *` |
| `swe2d/workbench/extracted/results_export_methods.py` | 8 | `from swe2d_workbench_qt import *` |
| `swe2d/workbench/extracted/studio_host_methods.py` | 4 | `from swe2d_workbench_qt import *` |
| `swe2d/workbench/extracted/topology_and_io_methods.py` | 6 | `from swe2d_workbench_qt import *` |

Wildcard imports in `__init__.py` files and `extracted/` modules are a design pattern used for re-exporting public APIs. They make it impossible to determine which names are actually part of the public surface vs. internal implementation details leaking through.

### 1C. Dead Functions, Classes, and Variables (67 items)

#### 🔴 High Priority — Entire Classes/Modules Unused

| File | Line | Dead Item | Notes |
|---|---|---|---|
| `swe2d/extensions/patch_observer.py` | 19 | `SWE2DThreeDPatchObserver` (class + 3 methods) | Entire class never instantiated anywhere — unfinished 3D patch feature |
| `swe2d/results/queries.py` | 34 | `ResultsDataset` (dataclass) | Never instantiated |
| `swe2d/extensions/extension_models.py` | 88 | `RainSourceTermState` (dataclass) | Never instantiated |
| `swe2d/mesh/meshing.py` | 10046 | `HybridCppBackend` (class, subclass of `MeshingBackend`) | Never registered or instantiated |

#### 🟠 Medium Priority — Unused Public API / Wiring Dead-ends

| File | Line | Dead Item | Notes |
|---|---|---|---|
| `swe2d/runtime/run_orchestration_bridge.py` | 7 | `prepare_run_orchestration()` | Never imported/called |
| `swe2d/runtime/run_orchestration_bridge.py` | 43 | `initialize_run_timing_and_logging()` | Never imported/called |
| `swe2d/runtime/run_orchestration_bridge.py` | 57 | `finalize_run_ui_state()` | Never imported/called |
| `swe2d/workbench/view.py` | 29 | `on_run_requested()` | Signal-handler method never wired |
| `swe2d/workbench/view.py` | 34 | `on_cancel_requested()` | Never wired |
| `swe2d/workbench/view.py` | 39 | `on_snapshot_requested()` | Never wired |
| `swe2d/workbench/view.py` | 44 | `set_run_state()` | Never wired |
| `swe2d/workbench/view.py` | 52 | `set_progress_value()` | Never wired |
| `swe2d/workbench/view.py` | 57 | `append_runtime_log()` | Never wired |
| `swe2d/results/panel.py` | 1608 | `restore_state()` | Never called |
| `swe2d/results/panel.py` | 1731 | `set_current_time()` | Never called |
| `swe2d/results/panel.py` | 1771 | `set_streamline_overlay_enabled()` | Never called |
| `swe2d/results/panel.py` | 1783 | `run_ids_for_gpkg()` | Never called |
| `swe2d/extensions/structures.py` | 416 | `last_structure_details` property | Never accessed |

#### 🟡 Lower Priority — Orphaned Private Helpers & Duplicates

| File | Line | Dead Item | Notes |
|---|---|---|---|
| `swe2d_workbench_qt.py` | 1223 | `_build_msh_elements()` | Orphaned private helper |
| `swe2d_workbench_qt.py` | 1192 | `_parse_feature_float()` | Orphaned private helper |
| `swe2d_workbench_qt.py` | 1182 | `_resolve_obj_model_path()` | Orphaned private helper |
| `swe2d_workbench_qt.py` | multiple | Additional orphan private methods (13 more `_`-prefixed) | Unreferenced |
| `swe2d/runtime/coupling.py` | 364 | `_structure_source_rate_from_flows()` | Never called |
| `swe2d/runtime/coupling.py` | multiple | `connectivity_map`, `source_element_idx_by_id` etc. (4 diagnostics properties) | Never read |
| `swe2d/mesh/meshing.py` | multiple | `_iter_qgis_polygon_outer_rings()`, `_split_closed_ring_max_segment_length()`, `_geo_polycurve()`, `_hybrid_cpp_available()`, `_as_bool_opt()` (6 helpers) | Never called |
| `swe2d/extensions/structures.py` | 89 | `_m_to_ft()` | Dead unit-conversion helper |
| `swe2d/extensions/structures.py` | 93 | `_cms_to_cfs()` | Dead unit-conversion helper |
| `culvert_routine.py` | 767 | `solve_normal_depth_in_culvert()` | Never called |
| `culvert_routine.py` | 984 | `report_CulvertControl()` | Never called |
| `forms/hydra_form_init.py` | 372 | `hydra_cross_sections_form_open()` | Never called |
| `forms/hydra_form_init.py` | 386 | `hydra_boundary_form_open()` | Never called |
| `qgis_utils.py` | 25 | `extract_xs_from_line()` | Never called |
| `rainfall_hydrology.py` | 286 | `preview_step()` | Never called |
| `rainfall_hydrology.py` | 505 | `composite_curve_number()` | Never called |
| `ui_adapter.py` | 62 | `get_open_filename()` | Never called |
| `ui_adapter.py` | 69 | `get_save_filename()` | Never called |

---

## Part 2: Silent Fallbacks

### 2A. Overall Statistics

| Pattern | Count |
|---|---|
| `except Exception: pass` (silent swallow) | 305 |
| `except Exception: continue` (silent skip iteration) | 65 |
| `except Exception: return None` | 31 |
| `except Exception: return False` | 23 |
| `except Exception: return []` | 5 |
| `except Exception: return {}` | 3 |
| `except Exception: return ""` | 4 |
| `except Exception: return 0.0` | 1 |
| Multi-line except bodies without any logging | ~306 |
| `except Exception: pass` with `logger.warning` | 0 |
| `except Exception: pass` with `logger.error` | 0 |
| **Total silent fallbacks** | **~753** |
| **Total except blocks (all files)** | **869** |
| **% silent** | **87%** |

### 2B. Worst Files by Silent-Swallow Density

| File | Silent excepts | Total excepts | % Silent |
|---|---|---|---|
| `swe2d_workbench_qt.py` | 241 | 269 | **89%** |
| `swe2d/mesh/meshing.py` | 70 | 73 | **96%** |
| `swe2d/workbench/extracted/topology_and_io_methods.py` | 67 | 73 | **92%** |
| `tools/qgis_live_bridge_console.py` | 34 | 37 | **92%** |
| `swe2d/workbench/extracted/results_and_ui_methods.py` | 32 | 34 | **94%** |
| `swe2d/workbench/extracted/studio_host_methods.py` | 28 | 28 | **100%** |
| `swe2d_high_perf_viewer.py` | 19 | 19 | **100%** |
| `hydra_plugin.py` | 18 | 19 | **95%** |
| `forms/hydra_form_init.py` | 17 | 19 | **89%** |
| `swe2d/boundary_and_forcing/boundary_qgis_adapter.py` | 14 | 16 | **87%** |
| `tools/headless_swe2d_gpkg_bc_diag.py` | 13 | 13 | **100%** |
| `swe2d/boundary_and_forcing/spatial_forcing_qgis_adapter.py` | 7 | 8 | **87%** |

### 2C. Representative Examples of Silent Fallbacks by Category

**Silent `except: pass` — no indication of failure:**
```python
# swe2d_workbench_qt.py:266
try:
    self.map_layer_registry.removeMapLayer(layer.id())
except Exception:
    pass
```

**Silent `except: continue` — iteration silently skips elements:**
```python
# swe2d/mesh/meshing.py:442
try:
    point = QgsPointXY.fromWkt(wkt)
except Exception:
    continue
```

**Silent `except: return None` — caller gets None with no indication:**
```python
# swe2d/mesh/meshing.py:426
try:
    return self._load_from_gpkg(...)
except Exception:
    return None
```

**Silent `except: return False` — caller assumes boolean success/failure:**
```python
# swe2d/mesh/meshing.py:10042
try:
    self._gmsh.initialize()
    return True
except Exception:
    return False
```

**Silent `except: return []/{}` — callers get empty container, indistinguishable from "no data":**
```python
# swe2d/results/queries.py:263
try:
    rows = self._fetch(...)
    return [dict(row) for row in rows]
except Exception:
    return []
```

**Multi-line except without logging:**
```python
# swe2d/workbench/extracted/topology_and_io_methods.py:592
try:
    ...
except Exception:
    self.initialized = False
    QMessageBox.critical(...)
    # No logger call, no traceback
```

### 2D. The `studio_host_methods.py` File — 100% Silent

Every single `except` block in `swe2d/workbench/extracted/studio_host_methods.py` (28/28) silently swallows exceptions. This file handles Studio UI actions (mesh generation, run control, topology operations) — exactly the operations where silent failures are most likely to confuse users who see "nothing happening."

### 2E. The `swe2d_high_perf_viewer.py` File — 100% Silent

All 19 `except` blocks in this file silently swallow errors. The high-performance viewer handles OpenGL/rendering operations where failures should be reported.

---

## Part 3: Specific Risk Patterns

### 3A. Silent Fallback Chain (compounding silent failures)

In several QGIS adapter files, a chain of silent failures can occur:
1. A QGIS layer operation fails → silently returns `None`/`False`
2. The caller receives `None` silently → falls back to empty defaults
3. Downstream code operates on empty data → produces incorrect results

Example chain in `boundary_qgis_adapter.py:80`:
```python
def load_layer(self, path):
    try:
        layer = QgsVectorLayer(path, ...)
        ...
        return True
    except Exception:
        return False  # <-- silent
```

### 3B. Mesh Generation Silent Failures

`meshing.py` has 70+ silent except blocks (96% rate). Mesh generation errors silently produce:
- Empty meshes returned as `None`
- Failed topology operations silently skipped
- The user sees no mesh with no error message

### 3C. Result Query Silent Failures

`results/queries.py` silently returns `[]`, `{}`, or `""` on database query failures. This means a corrupted GPKG database produces empty result panels with no indication of the corruption.

---

## Part 4: Recommendations

### Priority 1: Add Logging to Silent Exception Handlers

The single highest-impact change is adding `logger.warning(...)` or `logger.error(...)` with traceback to the ~753 silent except blocks. Target files in order:

1. `swe2d_workbench_qt.py` (241) — UI operations where user feedback matters most
2. `swe2d/mesh/meshing.py` (70) — Mesh generation, critical path
3. `swe2d/workbench/extracted/topology_and_io_methods.py` (67) — I/O operations
4. `swe2d/workbench/extracted/studio_host_methods.py` (28) — Studio UI actions
5. `swe2d/workbench/extracted/results_and_ui_methods.py` (32) — Results display
6. `swe2d/boundary_and_forcing/boundary_qgis_adapter.py` (14) — BC loading
7. `swe2d/boundary_and_forcing/spatial_forcing_qgis_adapter.py` (7) — Forcing loading

### Priority 2: Remove or Reanimate Dead Code

| Action | Items |
|---|---|
| **Remove** | `patch_observer.py` (entire file), `ResultsDataset`, `RainSourceTermState`, `HybridCppBackend`, orphaned private helpers |
| **Reanimate** | `run_orchestration_bridge.py` if intended to be used, `view.py` signal handlers if the view pattern was meant to be wired |
| **Audit** | All wildcard imports in `extracted/` modules — replace with explicit imports |

### Priority 3: Convert Wildcard Imports

The 5 files in `swe2d/workbench/extracted/` that do `from swe2d_workbench_qt import *` should be converted to explicit imports. This makes the public API surface explicit and helps with static analysis.

---

## Methodology

- Dead imports: Manual audit using grep for each imported name against references in the same file
- Dead functions/classes: Cross-referenced definitions with call/inheritance references across all non-test Python files
- Silent fallbacks: Regex pattern search for `except`, classified by body content
- Test files: Included in silent-fallback scan, excluded from most dead-code scans (test functions are auto-discovered)
