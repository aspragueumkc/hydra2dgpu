# Public Branch Sanitization Plan

**Purpose**: Create a clean `public` branch suitable for open-source release, removing internal dev artifacts, large generated data, personal paths, and build outputs.

**Created**: 2026-06-10

---

## 1. Branch Setup

```bash
# Ensure working tree is clean on GPU_ONLY
cd /home/aaron/QGIS_Plugins_dev/qgis-backwater-plugin-GPU_ONLY
git stash  # if needed

# Create the public branch from GPU_ONLY
git checkout -b public GPU_ONLY
```

---

## 2. Files and Directories to Remove

### 2.1 Build Artifacts (61 MB)

| Path | Size | Reason |
|------|------|--------|
| `build/` | 61 MB | CMake build output; users rebuild from source |
| `*.so` (root) | ~30 MB | Compiled `.so` files left in workspace root |

```bash
rm -rf build/
rm -f *.so
```

### 2.2 Large Generated Data (> 440 MB)

| Path | Size | Reason |
|------|------|--------|
| `qgis_testing_project/` | 342 MB | QGIS project + large `.gpkg` with testing results |
| `2d_example/` | 96 MB | Generated solver results (`.nc`, `.hdf`, `.h5`, `.gpkg`, `.npz`, `.msh`) |
| `docs/assets/` | 45 MB | Vendored PDF reference manuals (copyright concerns) |
| `example_project/` | 6.8 MB | Generated example `.gpkg` + CSV |
| `report_output/` | 1.7 MB | Generated PNG figures + investigation notes |
| `unsteady_example/` | — | Generated unsteady example data |

```bash
rm -rf qgis_testing_project/ 2d_example/ docs/assets/ example_project/
rm -rf report_output/ unsteady_example/
```

### 2.3 Internal Developer Documents

| Path | Reason |
|------|--------|
| `AGENTS.md` | AI agent instructions; repo-internal |
| `GPU_AUDIT_REPORT.md` | Internal audit notes |
| `MOMENTUM_CAP_FIX.md` | Internal fix tracking |
| `coupling_diag.log` | Runtime log artifact |

```bash
rm -f AGENTS.md GPU_AUDIT_REPORT.md MOMENTUM_CAP_FIX.md coupling_diag.log
```

### 2.4 Reference / Scaffolding Files

| Path | Size | Reason |
|------|------|--------|
| `reference/` | 820 KB | C++ reference implementations kept for internal comparison |
| `typings/` | 60 KB | Type stubs for IDE use; not needed for distribution |

```bash
rm -rf reference/ typings/
```

### 2.5 Temp / Debug Tools

| Path | Reason |
|------|--------|
| `tools/_tmp_headless_gui_gmsh_full_align.py` | Temp debug script |
| `test_gpu_debug.py` (root) | Ad-hoc GPU debug script |
| `test_workbench_persistence.py` (root) | Ad-hoc persistence test |
| `stacked_bridge_coupling.py` (root) | Standalone dev script (duplicated in `reference/`) |

```bash
rm -f tools/_tmp_headless_gui_gmsh_full_align.py
rm -f test_gpu_debug.py test_workbench_persistence.py
rm -f stacked_bridge_coupling.py
```

### 2.6 Personal Hardcoded Paths

| File | Issue |
|------|-------|
| `tools/qgis_live_bridge_console.py` | Contains `/home/aaron` hardcoded paths |

```bash
rm -f tools/qgis_live_bridge_console.py
# OR: edit the file to replace hardcoded user paths with configurable alternatives
```

---

## 3. Files to Update (Not Remove)

### 3.1 Update `.gitignore`

Append to `.gitignore` to prevent re-accumulation:

```gitignore
# ── Public branch additions ──────────────────────────────────────
# Large generated data (regenerated from examples/ or tests)
2d_example/
example_project/
qgis_testing_project/
report_output/
unsteady_example/

# Compiled shared objects
*.so

# Vendored PDF assets (copyright)
docs/assets/

# Reference implementations
reference/

# Internal dev docs
AGENTS.md
GPU_AUDIT_REPORT.md
MOMENTUM_CAP_FIX.md

# Runtime logs
*.log
coupling_diag*
```

### 3.2 Update `docs/` Structure

After removing `docs/assets/` and internal tracker docs:

```bash
# The following docs/ files are valuable for public readers:
#   docs/README.md              - Overview
#   docs/USER_GUIDE.md          - End-user documentation
#   docs/academic_paper.md      - Technical reference
#   docs/SWE2D_CODEBASE_AUDIT.md - Architecture overview
#   docs/SWE2D_GPU_ARCHITECTURE_REPORT.md - GPU architecture
#   docs/STUDIO_UI_ARCHITECTURE.md - UI architecture

# Remove internal implementation trackers that reference private repo details:
rm -f docs/D2H_H2D_FALLBACK_TRACKER.md
rm -f docs/DEAD_CODE_SILENT_FALLBACKS_AUDIT.md
rm -f docs/STUDIO_UI_LEGACY_REMOVAL_PLAN.md
rm -f docs/UNIT_AGNOSTIC_REFACTOR_PLAN.md
rm -f docs/UNIT_AGNOSTIC_REFACTOR_COMPLETED.md
rm -f docs/WENO5_LSQR_2RING_IMPLEMENTATION_PLAN.md
rm -f docs/FRICTION_IMPROVEMENT_PLAN.md
rm -f docs/FACE_BASED_CULVERT_COUPLING_PLAN.md
rm -f docs/GPU_STEP_GRAPH_ARCHITECTURE.md
rm -f docs/SWE2D_VARIABLE_TIMESTEP_ARCHITECTURE.md
rm -f docs/SWE2D_WORKBENCH_GUI_AUDIT.md
rm -f docs/SWE2D_GEOPACKAGE_SCHEMA_AUDIT.md
rm -f docs/SOLVER_ORDER_AND_STENCIL.md

# Remove internal subdirectories
rm -rf docs/active/ docs/archive/ docs/reference/ docs/validation/
```

### 3.3 Update `metadata.txt`

Fill in plugin metadata:

```ini
[general]
name=HYDRA
description=HYDRA - Hydrodynamics & Runoff Application plugin
version=0.1
qgisMinimumVersion=3.0
author=UMKC ASPIRe Lab
email=asprague@umkc.edu
homepage=
```

---

## 4. Cleaned .gitignore (Full)

```gitignore
# Python byte-compiled / optimised / DLL files
__pycache__/
*.py[cod]
*$py.class
*.pyc

# Compiled C extensions
*.so
*.pyd

# CMake / build outputs
build/
lib/
dist/
*.egg-info/

# Virtual environments
.venv/
venv/
env/

# Editor / IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Local generated artifacts and temporary backups
tests/artifacts/
temp_archive/
forms/*.bak

# ── Large generated data (regenerated from source) ───────────────
2d_example/
example_project/
qgis_testing_project/
report_output/
unsteady_example/

# Vendored PDF assets (copyright concerns)
docs/assets/

# Reference implementations (internal comparison)
reference/

# Internal dev documents
AGENTS.md
GPU_AUDIT_REPORT.md
MOMENTUM_CAP_FIX.md

# Runtime logs
*.log
coupling_diag*

# Temp / debug tools
tools/_tmp_*
```

---

## 5. Complete Git Commands Sequence

```bash
#!/usr/bin/env bash
set -euo pipefail

cd /home/aaron/QGIS_Plugins_dev/qgis-backwater-plugin-GPU_ONLY

# ── Step 1: Create branch ────────────────────────────────────────
git checkout -b public GPU_ONLY

# ── Step 2: Remove build artifacts ───────────────────────────────
rm -rf build/
find . -maxdepth 1 -name "*.so" -delete

# ── Step 3: Remove large generated data ──────────────────────────
rm -rf qgis_testing_project/
rm -rf 2d_example/
rm -rf docs/assets/
rm -rf example_project/
rm -rf report_output/
rm -rf unsteady_example/

# ── Step 4: Remove internal dev docs ─────────────────────────────
rm -f AGENTS.md GPU_AUDIT_REPORT.md MOMENTUM_CAP_FIX.md coupling_diag.log

# ── Step 5: Remove reference/scaffolding ─────────────────────────
rm -rf reference/ typings/

# ── Step 6: Remove temp/debug scripts ────────────────────────────
rm -f tools/_tmp_headless_gui_gmsh_full_align.py
rm -f test_gpu_debug.py test_workbench_persistence.py
rm -f stacked_bridge_coupling.py
rm -f tools/qgis_live_bridge_console.py

# ── Step 7: Remove internal implementation tracker docs ──────────
rm -f docs/D2H_H2D_FALLBACK_TRACKER.md
rm -f docs/DEAD_CODE_SILENT_FALLBACKS_AUDIT.md
rm -f docs/STUDIO_UI_LEGACY_REMOVAL_PLAN.md
rm -f docs/UNIT_AGNOSTIC_REFACTOR_PLAN.md
rm -f docs/UNIT_AGNOSTIC_REFACTOR_COMPLETED.md
rm -f docs/WENO5_LSQR_2RING_IMPLEMENTATION_PLAN.md
rm -f docs/FRICTION_IMPROVEMENT_PLAN.md
rm -f docs/FACE_BASED_CULVERT_COUPLING_PLAN.md
rm -f docs/GPU_STEP_GRAPH_ARCHITECTURE.md
rm -f docs/SWE2D_VARIABLE_TIMESTEP_ARCHITECTURE.md
rm -f docs/SWE2D_WORKBENCH_GUI_AUDIT.md
rm -f docs/SWE2D_GEOPACKAGE_SCHEMA_AUDIT.md
rm -f docs/SOLVER_ORDER_AND_STENCIL.md
rm -rf docs/active/ docs/archive/ docs/reference/ docs/validation/

# ── Step 8: Update .gitignore ────────────────────────────────────
# (See section 4 above for the full .gitignore content)
# Manually update .gitignore or use the template from section 4.

# ── Step 9: Commit ───────────────────────────────────────────────
git add -A
git status
git commit -m "Sanitize repo for public release

- Remove build/ and root-level .so files
- Remove large generated data (qgis_testing_project, 2d_example, etc.)
- Remove vendored PDF assets (copyright)
- Remove internal dev docs (AGENTS.md, audit reports, trackers)
- Remove reference/ and typings/ scaffolding
- Remove temp/debug scripts and personal tool paths
- Update .gitignore to prevent re-accumulation
- Keep core solver, bindings, tests, and user-facing docs"
```

---

## 6. Final File Inventory (Expected)

After sanitization, the public branch should contain:

```
├── CMakeLists.txt           # Build system
├── README.md                # Project README
├── metadata.txt             # QGIS plugin metadata
├── hydra_plugin.py          # QGIS plugin entry point
├── swe2d_workbench_qt.py    # Main workbench dialog
├── swe2d/                   # Core Python package
│   ├── boundary_and_forcing/
│   ├── extensions/
│   ├── mesh/
│   ├── runtime/
│   └── workbench/
├── cpp/                     # C++ / CUDA solver source
│   ├── src/
│   └── third_party/
├── forms/                   # Qt .ui files
├── expressions/             # QGIS expression functions
├── tools/                   # Utility scripts (cleaned)
├── tests/                   # Test suite
├── docs/                    # User-facing documentation
│   ├── README.md
│   ├── USER_GUIDE.md
│   ├── academic_paper.md
│   ├── SWE2D_CODEBASE_AUDIT.md
│   ├── SWE2D_GPU_ARCHITECTURE_REPORT.md
│   └── STUDIO_UI_ARCHITECTURE.md
└── .gitignore               # Updated with data exclusions
```

**Estimated total size**: ~5–8 MB (down from ~550+ MB)

---

## 7. Post-Sanitization Checklist

- [ ] Run `git status` to confirm only intended removals
- [ ] Verify `tests/test_workbench_imports.py` still passes
- [ ] Verify `cmake --build build/` works from clean checkout
- [ ] Verify no hardcoded user paths remain (`grep -rn '/home/' --include='*.py'`)
- [ ] Verify no secrets remain (`grep -rn 'password\|secret\|api_key' --include='*.py'`)
- [ ] Verify `docs/USER_GUIDE.md` renders correctly
- [ ] Push branch: `git push origin public`
