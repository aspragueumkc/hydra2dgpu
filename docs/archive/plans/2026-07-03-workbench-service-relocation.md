---
type: plan
status: complete
created: 2026-07-03
completed: 2026-07-25
---

# Workbench Service Relocation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Move pure-Python services out of `swe2d/workbench/services/` into shared service locations (`swe2d/mesh/`, `swe2d/runtime/`, `swe2d/results/`, `swe2d/extensions/`, `swe2d/services/`, `swe2d/boundary_and_forcing/`) so the CLI no longer imports from the workbench layer and the service layer has no Qt/QGIS widget dependencies.

**Architecture:** Each pure-Python service file is relocated to the shared layer that matches its domain. All workbench/controllers/dialogs and CLI imports are rewritten to the new paths. `widget_persistence_service.py` is moved out of `services/` because it imports `PyQt5.QtWidgets`. A new `tests/test_mvp_imports.py` enforces the CLI-must-not-import-workbench rule.

**Tech Stack:** Python 3.12, numpy, osgeo (gdal/ogr), sqlite3, qgis.core (allowed only in GUI services)

---

## Pre-flight checks

Before any move, verify repository state and locate every consumer.

- [ ] **Step 1: Check git status is clean**

```bash
git status --short
```

Expected: empty output (or only expected local changes).

- [ ] **Step 2: Build a dependency map of `swe2d/workbench/services/` imports**

```bash
for f in swe2d/workbench/services/*.py; do
  echo "=== $f ==="
  rg "^from swe2d\." "$f" | rg -v "^from swe2d\.workbench\.services\.(__init__|schema_definitions)" || true
done
```

Expected: list of cross-service dependencies to update as files move.

---

## Phase 1: Mesh + line sampling (critical — fixes active CLI violation)

### Task 1.1: Move `mesh_service.py` to `swe2d/mesh/`

**Files:**
- Move: `swe2d/workbench/services/mesh_service.py` → `swe2d/mesh/mesh_service.py`
- Modify: `swe2d/cli/headless_runner.py:396,577`
- Modify: every workbench file importing `swe2d.workbench.services.mesh_service`

- [ ] **Step 1: Move the file with git**

```bash
git mv swe2d/workbench/services/mesh_service.py swe2d/mesh/mesh_service.py
```

- [ ] **Step 2: Update CLI imports**

In `swe2d/cli/headless_runner.py`, replace:

```python
from swe2d.workbench.services.mesh_service import (
    build_line_sampling_map,
    sample_line_metrics,
)
```

with:

```python
from swe2d.mesh.mesh_service import (
    build_line_sampling_map,
    sample_line_metrics,
)
```

and the second occurrence at line 577:

```python
from swe2d.mesh.mesh_service import (
    sample_line_aggregate_ts_row,
    sample_line_metrics,
)
```

- [ ] **Step 3: Find and update all workbench imports**

```bash
rg "from swe2d\.workbench\.services\.mesh_service" swe2d/ tests/
```

Replace every hit with `from swe2d.mesh.mesh_service`.

- [ ] **Step 4: Update tests**

In `tests/test_workbench_mesh_service.py`, replace:

```python
from swe2d.workbench.services.mesh_service import ...
```

with:

```python
from swe2d.mesh.mesh_service import ...
```

- [ ] **Step 5: Syntax check**

```bash
mamba run -n qgis_stable python -c "
import py_compile
for f in ['swe2d/mesh/mesh_service.py', 'swe2d/cli/headless_runner.py']:
    py_compile.compile(f, doraise=True)
    print(f'{f}: OK')
"
```

Expected: both files compile.

- [ ] **Step 6: Run mesh-service tests**

```bash
mamba run -n qgis_stable python -m pytest tests/test_workbench_mesh_service.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add swe2d/mesh/mesh_service.py swe2d/cli/headless_runner.py \
  $(git diff --name-only) tests/test_workbench_mesh_service.py
git commit -m "refactor: move mesh_service.py to swe2d/mesh and update CLI imports"
```

---

### Task 1.2: Move `mesh_data_prep_service.py` to `swe2d/results/`

**Files:**
- Move: `swe2d/workbench/services/mesh_data_prep_service.py` → `swe2d/results/mesh_data_prep_service.py`
- Modify: all files importing it

- [ ] **Step 1: Move the file**

```bash
git mv swe2d/workbench/services/mesh_data_prep_service.py swe2d/results/mesh_data_prep_service.py
```

- [ ] **Step 2: Find and rewrite imports**

```bash
rg "from swe2d\.workbench\.services\.mesh_data_prep_service" swe2d/ tests/
```

Replace every hit with `from swe2d.results.mesh_data_prep_service`.

- [ ] **Step 3: Update tests**

In `tests/test_mesh_data_prep_service.py`, update the import path.

- [ ] **Step 4: Syntax check + test**

```bash
mamba run -n qgis_stable python -m pytest tests/test_mesh_data_prep_service.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add swe2d/results/mesh_data_prep_service.py $(git diff --name-only)
git commit -m "refactor: move mesh_data_prep_service.py to swe2d/results"
```

---

### Task 1.3: Add regression test for line-sampling memory safety

**Files:**
- Modify: `tests/test_workbench_mesh_service.py`

- [ ] **Step 1: Append a stress test**

```python
def test_build_line_sampling_map_stress_no_broadcast_oom():
    """Large mesh + sample line must not allocate N_points x N_tri matrices."""
    import time
    n = 320
    xs = np.linspace(0, 10000, n)
    ys = np.linspace(0, 10000, n)
    gx, gy = np.meshgrid(xs, ys)
    node_coords = np.column_stack([gx.ravel(), gy.ravel()])
    cells = []
    for j in range(n - 1):
        for i in range(n - 1):
            a = j * n + i
            cells.append([a, a + 1, a + n])
            cells.append([a + 1, a + n + 1, a + n])
    cell_nodes = np.array(cells, dtype=np.int32)
    line_xy = np.array([[100.0, 5000.0], [9900.0, 5000.0]], dtype=np.float64)
    t0 = time.time()
    sm = build_line_sampling_map(node_coords, cell_nodes, line_xy)
    elapsed = time.time() - t0
    assert sm["cell_idx"].size > 0
    assert elapsed < 30.0  # generous; GEOS intersection should be seconds
    assert np.isclose(sm["weights"].sum(), 1.0)
```

- [ ] **Step 2: Run the test**

```bash
mamba run -n qgis_stable python -m pytest tests/test_workbench_mesh_service.py::test_build_line_sampling_map_stress_no_broadcast_oom -v
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_workbench_mesh_service.py
git commit -m "test: add line-sampling stress test to prevent O(N*M) memory blowup"
```

---

## Phase 2: Runtime services

### Task 2.1: Move `non_gui_runtime_service.py` to `swe2d/runtime/`

**Files:**
- Move: `swe2d/workbench/services/non_gui_runtime_service.py` → `swe2d/runtime/non_gui_runtime_service.py`
- Modify: all importers

- [ ] **Step 1: Move**

```bash
git mv swe2d/workbench/services/non_gui_runtime_service.py swe2d/runtime/non_gui_runtime_service.py
```

- [ ] **Step 2: Rewrite imports**

```bash
rg "from swe2d\.workbench\.services\.non_gui_runtime_service" swe2d/ tests/
```

Replace with `from swe2d.runtime.non_gui_runtime_service`.

- [ ] **Step 3: Syntax check**

```bash
mamba run -n qgis_stable python -c "import py_compile; py_compile.compile('swe2d/runtime/non_gui_runtime_service.py', doraise=True); print('OK')"
```

Expected: OK.

- [ ] **Step 4: Commit**

```bash
git add swe2d/runtime/non_gui_runtime_service.py $(git diff --name-only)
git commit -m "refactor: move non_gui_runtime_service.py to swe2d/runtime"
```

---

### Task 2.2: Move `run_service.py` to `swe2d/runtime/`

**Files:**
- Move: `swe2d/workbench/services/run_service.py` → `swe2d/runtime/run_service.py`
- Modify: `swe2d/workbench/controllers/run_controller.py`, `tests/test_workbench_run_service.py`, and any other importers

- [ ] **Step 1: Move**

```bash
git mv swe2d/workbench/services/run_service.py swe2d/runtime/run_service.py
```

- [ ] **Step 2: Rewrite imports**

```bash
rg "from swe2d\.workbench\.services\.run_service" swe2d/ tests/
```

Replace with `from swe2d.runtime.run_service`.

- [ ] **Step 3: Run tests**

```bash
mamba run -n qgis_stable python -m pytest tests/test_workbench_run_service.py -v
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add swe2d/runtime/run_service.py $(git diff --name-only)
git commit -m "refactor: move run_service.py to swe2d/runtime"
```

---

## Phase 3: Results services

### Task 3.1: Move `hecras_export_service.py` to `swe2d/results/`

**Files:**
- Move: `swe2d/workbench/services/hecras_export_service.py` → `swe2d/results/hecras_export_service.py`
- Modify: importers

- [ ] **Step 1: Move**

```bash
git mv swe2d/workbench/services/hecras_export_service.py swe2d/results/hecras_export_service.py
```

- [ ] **Step 2: Rewrite imports**

```bash
rg "from swe2d\.workbench\.services\.hecras_export_service" swe2d/ tests/
```

Replace with `from swe2d.results.hecras_export_service`.

- [ ] **Step 3: Update tests**

In `tests/test_results_export_service.py`, update imports.

- [ ] **Step 4: Run tests**

```bash
mamba run -n qgis_stable python -m pytest tests/test_results_export_service.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add swe2d/results/hecras_export_service.py $(git diff --name-only)
git commit -m "refactor: move hecras_export_service.py to swe2d/results"
```

---

### Task 3.2: Move `overlay_parameters_service.py` to `swe2d/results/`

**Files:**
- Move: `swe2d/workbench/services/overlay_parameters_service.py` → `swe2d/results/overlay_parameters_service.py`
- Modify: importers (likely `overlay_controller.py`, `high_perf_overlay_bridge.py`/controllers)

- [ ] **Step 1: Move**

```bash
git mv swe2d/workbench/services/overlay_parameters_service.py swe2d/results/overlay_parameters_service.py
```

- [ ] **Step 2: Rewrite imports**

```bash
rg "from swe2d\.workbench\.services\.overlay_parameters_service" swe2d/ tests/
```

Replace with `from swe2d.results.overlay_parameters_service`.

- [ ] **Step 3: Update tests**

In `tests/test_overlay_parameters_service.py`, update imports.

- [ ] **Step 4: Run tests**

```bash
mamba run -n qgis_stable python -m pytest tests/test_overlay_parameters_service.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add swe2d/results/overlay_parameters_service.py $(git diff --name-only)
git commit -m "refactor: move overlay_parameters_service.py to swe2d/results"
```

---

## Phase 4: Extension services

### Task 4.1: Move `pipe_network_service.py` to `swe2d/extensions/`

**Files:**
- Move: `swe2d/workbench/services/pipe_network_service.py` → `swe2d/extensions/pipe_network_service.py`
- Modify: importers

- [ ] **Step 1: Move**

```bash
git mv swe2d/workbench/services/pipe_network_service.py swe2d/extensions/pipe_network_service.py
```

- [ ] **Step 2: Rewrite imports**

```bash
rg "from swe2d\.workbench\.services\.pipe_network_service" swe2d/ tests/
```

Replace with `from swe2d.extensions.pipe_network_service`.

- [ ] **Step 3: Run tests**

```bash
mamba run -n qgis_stable python -m pytest tests/test_swe2d_gpu_drainage_network.py tests/test_swe2d_pipe1d.py -v
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add swe2d/extensions/pipe_network_service.py $(git diff --name-only)
git commit -m "refactor: move pipe_network_service.py to swe2d/extensions"
```

---

### Task 4.2: Move `pipe_network_config_service.py` to `swe2d/extensions/`

**Files:**
- Move: `swe2d/workbench/services/pipe_network_config_service.py` → `swe2d/extensions/pipe_network_config_service.py`
- Modify: importers

- [ ] **Step 1: Move**

```bash
git mv swe2d/workbench/services/pipe_network_config_service.py swe2d/extensions/pipe_network_config_service.py
```

- [ ] **Step 2: Rewrite imports**

```bash
rg "from swe2d\.workbench\.services\.pipe_network_config_service" swe2d/ tests/
```

Replace with `from swe2d.extensions.pipe_network_config_service`.

- [ ] **Step 3: Syntax check**

```bash
mamba run -n qgis_stable python -c "import py_compile; py_compile.compile('swe2d/extensions/pipe_network_config_service.py', doraise=True); print('OK')"
```

Expected: OK.

- [ ] **Step 4: Commit**

```bash
git add swe2d/extensions/pipe_network_config_service.py $(git diff --name-only)
git commit -m "refactor: move pipe_network_config_service.py to swe2d/extensions"
```

---

## Phase 5: Boundary / forcing services

### Task 5.1: Move `text_parser_service.py` to `swe2d/boundary_and_forcing/`

**Files:**
- Move: `swe2d/workbench/services/text_parser_service.py` → `swe2d/boundary_and_forcing/text_parser_service.py`
- Modify: importers

- [ ] **Step 1: Move**

```bash
git mv swe2d/workbench/services/text_parser_service.py swe2d/boundary_and_forcing/text_parser_service.py
```

- [ ] **Step 2: Rewrite imports**

```bash
rg "from swe2d\.workbench\.services\.text_parser_service" swe2d/ tests/
```

Replace with `from swe2d.boundary_and_forcing.text_parser_service`.

- [ ] **Step 3: Update tests**

In `tests/test_text_parser_service.py`, update imports.

- [ ] **Step 4: Run tests**

```bash
mamba run -n qgis_stable python -m pytest tests/test_text_parser_service.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add swe2d/boundary_and_forcing/text_parser_service.py $(git diff --name-only)
git commit -m "refactor: move text_parser_service.py to swe2d/boundary_and_forcing"
```

---

## Phase 6: Generic GPKG / utility services

### Task 6.1: Move `gpkg_operations_service.py` to `swe2d/services/`

**Files:**
- Move: `swe2d/workbench/services/gpkg_operations_service.py` → `swe2d/services/gpkg_operations_service.py`
- Modify: importers

- [ ] **Step 1: Move**

```bash
git mv swe2d/workbench/services/gpkg_operations_service.py swe2d/services/gpkg_operations_service.py
```

- [ ] **Step 2: Rewrite imports**

```bash
rg "from swe2d\.workbench\.services\.gpkg_operations_service" swe2d/ tests/
```

Replace with `from swe2d.services.gpkg_operations_service`.

- [ ] **Step 3: Run tests**

```bash
mamba run -n qgis_stable python -m pytest tests/test_gpkg_operations.py -v
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add swe2d/services/gpkg_operations_service.py $(git diff --name-only)
git commit -m "refactor: move gpkg_operations_service.py to swe2d/services"
```

---

### Task 6.2: Move `gpkg_layer_styles_service.py` to `swe2d/services/`

**Files:**
- Move: `swe2d/workbench/services/gpkg_layer_styles_service.py` → `swe2d/services/gpkg_layer_styles_service.py`
- Modify: importers

- [ ] **Step 1: Move**

```bash
git mv swe2d/workbench/services/gpkg_layer_styles_service.py swe2d/services/gpkg_layer_styles_service.py
```

- [ ] **Step 2: Rewrite imports**

```bash
rg "from swe2d\.workbench\.services\.gpkg_layer_styles_service" swe2d/ tests/
```

Replace with `from swe2d.services.gpkg_layer_styles_service`.

- [ ] **Step 3: Syntax check**

```bash
mamba run -n qgis_stable python -c "import py_compile; py_compile.compile('swe2d/services/gpkg_layer_styles_service.py', doraise=True); print('OK')"
```

Expected: OK.

- [ ] **Step 4: Commit**

```bash
git add swe2d/services/gpkg_layer_styles_service.py $(git diff --name-only)
git commit -m "refactor: move gpkg_layer_styles_service.py to swe2d/services"
```

---

### Task 6.3: Move `topology_template_service.py` to `swe2d/services/`

**Files:**
- Move: `swe2d/workbench/services/topology_template_service.py` → `swe2d/services/topology_template_service.py`
- Modify: importers

- [ ] **Step 1: Move**

```bash
git mv swe2d/workbench/services/topology_template_service.py swe2d/services/topology_template_service.py
```

- [ ] **Step 2: Rewrite imports**

```bash
rg "from swe2d\.workbench\.services\.topology_template_service" swe2d/ tests/
```

Replace with `from swe2d.services.topology_template_service`.

- [ ] **Step 3: Syntax check**

```bash
mamba run -n qgis_stable python -c "import py_compile; py_compile.compile('swe2d/services/topology_template_service.py', doraise=True); print('OK')"
```

Expected: OK.

- [ ] **Step 4: Commit**

```bash
git add swe2d/services/topology_template_service.py $(git diff --name-only)
git commit -m "refactor: move topology_template_service.py to swe2d/services"
```

---

### Task 6.4: Move `non_gui_qgis_service.py` to `swe2d/services/`

**Files:**
- Move: `swe2d/workbench/services/non_gui_qgis_service.py` → `swe2d/services/non_gui_qgis_service.py`
- Modify: importers, including `swe2d/workbench/__init__.py`

- [ ] **Step 1: Move**

```bash
git mv swe2d/workbench/services/non_gui_qgis_service.py swe2d/services/non_gui_qgis_service.py
```

- [ ] **Step 2: Rewrite imports**

```bash
rg "from swe2d\.workbench\.services\.non_gui_qgis_service" swe2d/ tests/
```

Replace with `from swe2d.services.non_gui_qgis_service`.

- [ ] **Step 3: Syntax check**

```bash
mamba run -n qgis_stable python -c "import py_compile; py_compile.compile('swe2d/services/non_gui_qgis_service.py', doraise=True); print('OK')"
```

Expected: OK.

- [ ] **Step 4: Commit**

```bash
git add swe2d/services/non_gui_qgis_service.py $(git diff --name-only)
git commit -m "refactor: move non_gui_qgis_service.py to swe2d/services"
```

---

### Task 6.5: Move `constants_service.py` to `swe2d/services/`

**Files:**
- Move: `swe2d/workbench/services/constants_service.py` → `swe2d/services/constants_service.py`
- Modify: importers (widely used in views/dialogs/controllers)

- [ ] **Step 1: Move**

```bash
git mv swe2d/workbench/services/constants_service.py swe2d/services/constants_service.py
```

- [ ] **Step 2: Rewrite imports**

```bash
rg "from swe2d\.workbench\.services\.constants_service" swe2d/ tests/
```

Replace with `from swe2d.services.constants_service`.

- [ ] **Step 3: Syntax check**

```bash
mamba run -n qgis_stable python -c "import py_compile; py_compile.compile('swe2d/services/constants_service.py', doraise=True); print('OK')"
```

Expected: OK.

- [ ] **Step 4: Commit**

```bash
git add swe2d/services/constants_service.py $(git diff --name-only)
git commit -m "refactor: move constants_service.py to swe2d/services"
```

---

## Phase 7: Fix `widget_persistence_service.py` architecture violation

**Files:**
- Move: `swe2d/workbench/services/widget_persistence_service.py` → `swe2d/workbench/views/widget_persistence_service.py`
- Modify: all importers

- [ ] **Step 1: Move to view layer**

```bash
git mv swe2d/workbench/services/widget_persistence_service.py swe2d/workbench/views/widget_persistence_service.py
```

- [ ] **Step 2: Rewrite imports**

```bash
rg "from swe2d\.workbench\.services\.widget_persistence_service" swe2d/ tests/
```

Replace with `from swe2d.workbench.views.widget_persistence_service`.

- [ ] **Step 3: Run workbench persistence tests**

```bash
mamba run -n qgis_stable python -m pytest tests/test_workbench_persistence.py -v
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add swe2d/workbench/views/widget_persistence_service.py $(git diff --name-only)
git commit -m "refactor: move widget_persistence_service.py out of services into views"
```

---

## Phase 8: Import-guard tests and final verification

### Task 8.1: Create `tests/test_mvp_imports.py`

**Files:**
- Create: `tests/test_mvp_imports.py`

- [ ] **Step 1: Write the test file**

```python
"""MVP architecture import guards.

- CLI must not import from swe2d.workbench.
- Shared service layers must not import PyQt5/QtWidgets.
"""
import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _top_level_imports(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    return imports


def test_cli_does_not_import_workbench():
    cli_root = REPO_ROOT / "swe2d" / "cli"
    for path in cli_root.rglob("*.py"):
        for imp in _top_level_imports(path):
            assert not imp.startswith("swe2d.workbench"), (
                f"{path.relative_to(REPO_ROOT)} imports {imp}"
            )


def test_shared_services_do_not_import_qt_widgets():
    shared_roots = [
        REPO_ROOT / "swe2d" / "services",
        REPO_ROOT / "swe2d" / "mesh",
        REPO_ROOT / "swe2d" / "runtime",
        REPO_ROOT / "swe2d" / "results",
        REPO_ROOT / "swe2d" / "extensions",
        REPO_ROOT / "swe2d" / "boundary_and_forcing",
    ]
    forbidden = {
        "PyQt5.QtWidgets",
        "PySide2.QtWidgets",
        "qgis.PyQt.QtWidgets",
    }
    for root in shared_roots:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            for imp in _top_level_imports(path):
                assert imp not in forbidden, (
                    f"{path.relative_to(REPO_ROOT)} imports {imp}"
                )


def test_workbench_services_do_not_import_qt_widgets():
    services_root = REPO_ROOT / "swe2d" / "workbench" / "services"
    if not services_root.exists():
        return
    forbidden = {
        "PyQt5.QtWidgets",
        "PySide2.QtWidgets",
        "qgis.PyQt.QtWidgets",
    }
    for path in services_root.rglob("*.py"):
        for imp in _top_level_imports(path):
            assert imp not in forbidden, (
                f"{path.relative_to(REPO_ROOT)} imports {imp}"
            )
```

- [ ] **Step 2: Run the new test**

```bash
mamba run -n qgis_stable python -m pytest tests/test_mvp_imports.py -v
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_mvp_imports.py
git commit -m "test: add MVP import guards for CLI and shared services"
```

---

### Task 8.2: Final cross-cutting verification

- [ ] **Step 1: Purge Python cache**

```bash
find . -type d -name __pycache__ -exec rm -rf {} +
```

- [ ] **Step 2: Re-run the audit grep to confirm zero CLI violations**

```bash
rg "from swe2d\.workbench" swe2d/cli/
```

Expected: no matches.

- [ ] **Step 3: Run focused test suites**

```bash
mamba run -n qgis_stable python -m pytest \
  tests/test_cli.py \
  tests/test_workbench_mesh_service.py \
  tests/test_mesh_data_prep_service.py \
  tests/test_workbench_run_service.py \
  tests/test_gpkg_operations.py \
  tests/test_text_parser_service.py \
  tests/test_results_export_service.py \
  tests/test_overlay_parameters_service.py \
  tests/test_mvp_imports.py \
  -v
```

Expected: all PASS.

- [ ] **Step 4: Run workbench smoke tests**

```bash
mamba run -n qgis_stable python -m pytest \
  tests/test_workbench_persistence.py \
  tests/test_workbench_gui.py \
  tests/test_workbench_controller.py \
  -v
```

Expected: all PASS.

- [ ] **Step 5: Final commit**

```bash
git status --short
# verify only intended files changed
git commit -m "refactor: complete workbench service relocation (no CLI->workbench imports)"
```

---

## Superpowers workflow

1. **writing-plans** — this plan.
2. **subagent-driven-development** — dispatch one subagent per phase; each phase is independent and ends with a commit.
3. **verification-before-completion** — run the focused test suite in Task 8.2 before declaring success.
4. **requesting-code-review** — after Phase 8, request review on the overall diff because it touches many files.

---

## Spec coverage self-review

| Audit finding | Phase + Task |
|---|---|
| CLI imports `mesh_service` from workbench | Phase 1, Task 1.1 |
| `mesh_service.py` is pure Python | Phase 1, Task 1.1 → `swe2d/mesh/` |
| `mesh_data_prep_service.py` is pure Python | Phase 1, Task 1.2 → `swe2d/results/` |
| `non_gui_runtime_service.py` is pure Python | Phase 2, Task 2.1 → `swe2d/runtime/` |
| `run_service.py` is pure Python | Phase 2, Task 2.2 → `swe2d/runtime/` |
| `hecras_export_service.py` is pure Python | Phase 3, Task 3.1 → `swe2d/results/` |
| `overlay_parameters_service.py` is pure Python | Phase 3, Task 3.2 → `swe2d/results/` |
| `pipe_network_service.py` is pure Python | Phase 4, Task 4.1 → `swe2d/extensions/` |
| `pipe_network_config_service.py` is pure Python | Phase 4, Task 4.2 → `swe2d/extensions/` |
| `text_parser_service.py` is pure Python | Phase 5, Task 5.1 → `swe2d/boundary_and_forcing/` |
| GPKG/util services are pure Python | Phase 6, Tasks 6.1–6.5 → `swe2d/services/` |
| `widget_persistence_service.py` imports QtWidgets | Phase 7 → `swe2d/workbench/views/` |
| Prevent future CLI→workbench imports | Phase 8, Task 8.1 `tests/test_mvp_imports.py` |

---

## Execution choice

Plan complete and saved to `docs/archive/plans/2026-07-03-workbench-service-relocation.md`.

Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per phase, review between phases, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
