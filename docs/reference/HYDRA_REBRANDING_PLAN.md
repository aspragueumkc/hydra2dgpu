---
title: QGIS Backwater → HYDRA Rebranding Plan
version: 1.0
date: 2026-05-20
description: Complete rebranding strategy from "Backwater" to "HYDRA" (QGIS Hydrodynamics & Runoff Application)
status: PLAN (Ready for Implementation)
---

# HYDRA Rebranding Plan

**Plugin New Name:** HYDRA (QGIS Hydrodynamics & Runoff Application)  
**Current Name:** Backwater  
**Total Backwater References:** 478 references across codebase  

## Executive Summary

This plan documents a comprehensive rebranding of the qgis-backwater-plugin to HYDRA with minimal functional changes. The rebranding affects:
- Python module filenames (3 critical files)
- 40+ Python files with import/reference updates
- Plugin metadata and documentation
- Build configuration and native extension names
- Test files and helper scripts

**Implementation Approach:** Non-breaking, systematic regex-based string replacement with validation at each phase.

---

## Phase 1: File & Directory Renaming

### 1.1 Directory Rename (Plugin Root)
```
qgis-backwater-plugin/  →  qgis-hydra-plugin/
```
- This is the top-level plugin directory in QGIS profile
- Path: ~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/qgis-hydra-plugin/

### 1.2 Critical Python Module Renames

| Current Name | New Name | Impact | Notes |
|---|---|---|---|
| `backwater_plugin.py` | `hydra_plugin.py` | **HIGH** | QGIS entry point; main menu registration |
| `backwater_qt.py` | `hydra_qt.py` | **HIGH** | Dock widget and UI orchestration for 1D/2D solvers |
| `backwater_model.py` | `hydra_1d.py` | **HIGH** | 1D unsteady solver and model I/O |
| `forms/backwater_form_init.py` | `forms/hydra_form_init.py` | **MEDIUM** | Form UI initialization |

### 1.3 Validation Post-Rename
- Verify `__init__.py` imports still resolve
- Confirm py_compile passes
- Check pyflakes linting

---

## Phase 2: Plugin Metadata & Configuration

### 2.1 metadata.txt Updates
```ini
[general]
name=HYDRA                                    # OLD: Backwater
description=HYDRA - Hydrodynamics & Runoff Application plugin  # OLD: Backwater and culvert solver GUI plugin
```

### 2.2 CMakeLists.txt Updates
All references to `backwater_*` target names:
- `backwater_native` → `hydra_native` (1D solver C++ extension)
- `backwater_swe2d` → `hydra_swe2d` (2D SWE solver C++ extension)
- `backwater_tqmesh` → `hydra_tqmesh` (mesh generation C++ extension)

**Note:** These are C++ target names; native module binary names must also change:
- `backwater_native.cpython-*.so` → `hydra_native.cpython-*.so`
- `backwater_swe2d.cpython-*.so` → `hydra_swe2d.cpython-*.so`
- `backwater_tqmesh.cpython-*.so` → `hydra_tqmesh.cpython-*.so`

### 2.3 __init__.py Updates
Plugin metadata fields (name, description) if present.

---

## Phase 3: Python Import Rewiring (Critical)

### 3.1 Files Requiring Import Updates (40+ files)

**Category A: Core Module Imports (Direct Module References)**
```python
# BEFORE
from backwater_plugin import ...
from backwater_qt import ...
from backwater_model import ...
import backwater_model as bw

# AFTER
from hydra_plugin import ...
from hydra_qt import ...
from hydra_1d import ...
import hydra_1d as bw
```

**Files in Category A:**
- `__init__.py`
- `backwater_qt.py` (internal imports)
- `backwater_model.py` (internal imports)
- `backwater_plugin.py` (internal imports)
- `unsteady_model.py` (import backwater_model)
- `test_gpu_debug.py`
- All 32 test files in `tests/` directory
- All 4 tools in `tools/` directory

### 3.2 Native Module Imports (C++ Extensions)

**Pattern 1: Direct Module Load**
```python
# BEFORE
import backwater_swe2d as mod
from hydra_1d import CrossSection

# AFTER
import hydra_swe2d as mod
from hydra_1d import CrossSection
```

**Pattern 2: Conditional Imports**
```python
# BEFORE
importlib.util.find_spec("backwater_tqmesh")
import backwater_tqmesh as _tq

# AFTER
importlib.util.find_spec("hydra_tqmesh")
import hydra_tqmesh as _tq
```

**Affected Files:**
- `swe2d/mesh/meshing.py` (backwater_tqmesh references)
- `swe2d/runtime/backend.py` (backwater_swe2d references)
- `swe2d/runtime/coupling.py` (backwater_swe2d references)
- `swe2d/runtime/run_controller.py` (error messages)
- All 8 test files that import native modules

### 3.3 String References (Error Messages, Comments, Docstrings)

**Pattern: Error Messages & Documentation**
```python
# BEFORE
"Build backwater_swe2d first"
"backwater_native C++ module not found"
"Python bridge for the native 2D SWE hybrid GPU/CPU solver (backwater_swe2d)"

# AFTER
"Build hydra_swe2d first"
"hydra_native C++ module not found"
"Python bridge for the native 2D SWE hybrid GPU/CPU solver (hydra_swe2d)"
```

**Affected Files:**
- `swe2d/runtime/backend.py` (docstring, error messages)
- `swe2d/runtime/run_controller.py` (error messages)
- `swe2d/runtime/coupling.py` (error messages)
- All test files with skip messages

---

## Phase 4: Build System & Native Extensions

### 4.1 CMakeLists.txt Changes

**Python Extension Targets:**
```cmake
# BEFORE
add_library(backwater_native SHARED ...)
add_library(backwater_swe2d SHARED ...)
add_library(backwater_tqmesh SHARED ...)

# AFTER
add_library(hydra_native SHARED ...)
add_library(hydra_swe2d SHARED ...)
add_library(hydra_tqmesh SHARED ...)
```

**Python Extension Output Names:**
```cmake
# BEFORE
set_target_properties(backwater_native PROPERTIES PREFIX "" SUFFIX ".cpython-313-x86_64-linux-gnu.so")

# AFTER
set_target_properties(hydra_native PROPERTIES PREFIX "" SUFFIX ".cpython-313-x86_64-linux-gnu.so")
```

### 4.2 C++ Source References
- `cpp/src/backwater_native.cpp` - Contains references in comments, module definitions
- Rename file? **Decision:** Consider for clarity, but optional (internal implementation detail)

### 4.3 Build Artifacts (Auto-Generated)
```
build/CMakeFiles/backwater_native.dir/  →  build/CMakeFiles/hydra_native.dir/
build/CMakeFiles/backwater_swe2d.dir/   →  build/CMakeFiles/hydra_swe2d.dir/
build/CMakeFiles/backwater_tqmesh.dir/  →  build/CMakeFiles/hydra_tqmesh.dir/
```
**Note:** These are regenerated during build; old ones can be deleted (clean build recommended)

---

## Phase 5: Documentation & User-Facing Text

### 5.1 README.md Updates
```markdown
# BEFORE
# QGIS Backwater Plugin

## Features
- Built-in main menu entries under **Backwater** for common actions:
- `backwater_model.py`: hydraulic solver...
- `backwater_qt.py`: dock widget...
- `backwater_plugin.py`: QGIS main menu...

## AFTER
# HYDRA - QGIS Hydrodynamics & Runoff Application

## Features
- Built-in main menu entries under **HYDRA** for common actions:
- `hydra_1d.py`: hydraulic solver...
- `hydra_qt.py`: dock widget...
- `hydra_plugin.py`: QGIS main menu...
```

### 5.2 Code Comments & Docstrings
- Update references to "backwater solver" → "1D hydrodynamic solver"
- Update references to "backwater transients" → "hydrodynamic transients"
- Preserve technical accuracy while updating branding

### 5.3 USER_GUIDE.md
- Menu paths: "Backwater >" → "HYDRA >"
- Dialog titles and window names
- Status messages and log output

---

## Phase 6: Testing & Validation

### 6.1 Validation Checklist
- [ ] **Syntax**: `python3 -m py_compile` passes on all Python files
- [ ] **Imports**: `python3 -m pyflakes` shows no import errors
- [ ] **Module Loading**: All imports resolve correctly at runtime
- [ ] **Build**: CMake configure and build succeed
- [ ] **Tests**: Core test suite runs (at least smoke tests)
- [ ] **QGIS Plugin**: Plugin loads in QGIS without errors

### 6.2 Smoke Test Commands
```bash
cd /path/to/qgis-hydra-plugin

# Syntax check
python3 -m py_compile hydra_plugin.py hydra_qt.py hydra_1d.py

# Lint check
python3 -m pyflakes hydra_plugin.py hydra_qt.py hydra_1d.py

# Import validation
python3 -c "import hydra_1d; import hydra_qt; import hydra_plugin"

# Build
cmake --build build

# Basic test
python3 -m unittest tests.test_river_station_ordering -v
```

### 6.3 Integration Testing
- Load plugin in QGIS
- Verify menu appears as "HYDRA"
- Test basic workflows (create model, run solver, view results)
- Check log messages reference new names

---

## Phase 7: Fallback & Cleanup

### 7.1 Backward Compatibility Considerations
- **Old imports:** No direct compatibility layer needed (this is a one-time rename)
- **Saved projects:** May have hardcoded references to `backwater_*` module names in pickled state
  - **Mitigation:** Version bump to 0.2; users should re-save projects
  - **Alternative:** Implement import hook if needed for legacy projects

### 7.2 Cleanup Tasks
```bash
# Remove old build artifacts (optional but recommended)
rm -rf build/CMakeFiles/backwater_*
rm -f build/*backwater*.so

# Clean pycache (optional)
find . -type d -name __pycache__ -exec rm -rf {} \; 2>/dev/null
```

### 7.3 Git & Version Control
- **Commit Strategy**: Single commit with all changes or grouped by phase
- **Branch**: Feature branch `feature/hydra-rebrand` 
- **Message**: `Rebrand: Backwater → HYDRA (0.2 release preparation)`
- **Version Bump**: 0.1 → 0.2 in metadata.txt

---

## Implementation Guide for Agents

### Quick Start for Next Agent
1. **Read this plan** (you're here!)
2. **Start with Phase 1**: Rename files (manual or scripted)
3. **Run Phase 2**: Update metadata.txt, CMakeLists.txt
4. **Execute Phase 3**: Systematic import rewiring using grep/sed or multi_replace_string_in_file
5. **Validate Phase 6**: Run smoke tests after each phase
6. **Document Phase 7**: Version bump, git commit

### Key Files to Modify (In Order)
1. `metadata.txt` (2 lines)
2. `CMakeLists.txt` (3 target names + properties)
3. `__init__.py` (if imports present)
4. `swe2d/runtime/backend.py` (docstrings, error messages)
5. `swe2d/runtime/coupling.py` (imports, error messages)
6. `swe2d/mesh/meshing.py` (imports, comments)
7. All test files (uniform pattern)
8. README.md, USER_GUIDE.md
9. Rebuild: `cmake --build build`
10. Validate: Smoke tests

### Tools Available
- `multi_replace_string_in_file`: For bulk regex replacements across multiple files
- `grep_search`: For auditing remaining references
- `run_in_terminal`: For batch scripts and build commands
- `replace_string_in_file`: For individual targeted fixes

### Common Pitfalls to Avoid
1. **Partial replacements**: Ensure ALL backwater_* → hydra_* changes are made (478 refs)
2. **Case sensitivity**: Some references may be `Backwater` (capital); ensure proper casing
3. **Native modules**: C++ extension names must match CMakeLists.txt and import statements
4. **Symlink issues**: QGIS plugin scanning may cache old plugin names; restart QGIS after rename
5. **Build artifacts**: Remove stale build products before rebuild

### Success Criteria
- ✅ All 478 backwater references replaced with hydra equivalents
- ✅ py_compile passes on all Python files
- ✅ pyflakes shows no import errors
- ✅ Plugin loads in QGIS with new name
- ✅ Core tests pass (test_river_station_ordering, test_swe2d_*)
- ✅ Menu shows "HYDRA" instead of "Backwater"
- ✅ Documentation updated

---

## Scope Summary

| Category | Count | Effort |
|---|---|---|
| File renames | 4 | 5 min |
| Python files with imports | 40 | 30 min |
| Backwater string references | 478 | 45 min |
| Metadata/config updates | 3-5 | 10 min |
| CMakeLists.txt updates | 6-8 lines | 5 min |
| Validation & testing | — | 20 min |
| **Total Estimated Time** | — | **~2 hours** |

---

## Appendix A: Reference Locations

### High-Priority Imports (Must Update First)
```
```

### Medium-Priority Native Module References
```
- swe2d/runtime/backend.py: backwater_swe2d imports/docstrings
- swe2d/runtime/coupling.py: backwater_swe2d imports
- swe2d/mesh/meshing.py: backwater_tqmesh imports
- swe2d/runtime/run_controller.py: error messages
```

### Test Files (Bulk Updates)
```
tests/test_*.py (32 files)
tools/*.py (4 files)
```

### Build Configuration
```
CMakeLists.txt: 6-8 target name and property changes
```

---

## Appendix B: Native Module Mapping Reference

| Old Name | New Name | Type | Purpose |
|---|---|---|---|
| `backwater_native` | `hydra_native` | C++ Extension | 1D unsteady hydraulic solver |
| `backwater_swe2d` | `hydra_swe2d` | C++ Extension | 2D shallow water equations solver |
| `backwater_tqmesh` | `hydra_tqmesh` | C++ Extension | Triangle mesh generator (TQMesh wrapper) |

---

## Version History

| Version | Date | Author | Status |
|---|---|---|---|
| 1.0 | 2026-05-20 | AI Agent | Plan Ready |

---

**Next Steps:** Execute Phase 1 (file renaming) and proceed systematically through each phase with validation checkpoints.
