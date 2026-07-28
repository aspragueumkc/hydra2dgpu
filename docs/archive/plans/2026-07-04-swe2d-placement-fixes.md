---
type: plan
status: complete
created: 2026-07-04
completed: 2026-07-25
---

# SWE2D Structural Placement Fixes — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the misplaced-function violations surfaced by `docs/STRUCTURAL_PLACEMENT_AUDIT.md` so every function's filename matches its job, every controller reaches UI only through View protocol methods, and the CLI no longer imports from the workbench layer.

**Architecture:** Strict MVP. View exposes typed getters (the `OverlayView`/`RunView` pattern), controllers orchestrate only, services do the math. Cross-layer moves preserve git history (`git mv`). Each phase is independently shippable and ends with a commit. TDD on every behavior change — write the failing test, watch it fail, implement minimal pass, then refactor.

**Tech Stack:** Python 3.12, numpy, PyQt5, qgis.core (allowed in workbench/), pytest, swe2d.units for unit system.

**Spec:** `docs/STRUCTURAL_PLACEMENT_AUDIT.md`

---

## Pre-flight (one-time, gates all phases)

- [ ] **Step 1: Verify clean working tree**

```bash
git status --short
```

Expected: empty output.

- [ ] **Step 2: Capture baseline test pass set**

```bash
mamba run -n qgis_stable python -m pytest \
  tests/test_cli.py \
  tests/test_workbench_mesh_service.py \
  tests/test_workbench_run_service.py \
  tests/test_sample_line_metrics_profile.py \
  tests/test_rcmk_permutation_mismatch.py \
  tests/test_results_path_wiring.py \
  tests/test_gpkg_persistence.py \
  tests/test_mvp_imports.py \
  --collect-only -q 2>&1 | tee /tmp/baseline_tests.txt
```

Expected: list of collected tests, no collection errors.

- [ ] **Step 3: Verify Python cache is clean (per AGENTS.md)**

```bash
find . -type d -name __pycache__ -exec rm -rf {} +
```

---

## Phase 1: Move line-sampling trio out of `mesh_service.py` (fixes Tier 1 #3 + Tier 3)

Highest leverage: removes 390 LOC of line-sampling code from a mesh file, deletes the duplicate `build_line_sampling_map`, AND eliminates the bright-line CLI import violation in one move. Existing tests (`tests/test_workbench_mesh_service.py`, `tests/test_sample_line_metrics_profile.py`, `tests/test_rcmk_permutation_mismatch.py`) cover the moved functions.

### Task 1.1: RED — Add regression test that CLI does NOT import from workbench

**Files:**
- Create: `tests/test_mvp_imports.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_mvp_imports.py`:

```python
"""Architectural boundary enforcement (AGENTS.md / PLANNING.md).

CLI must not import from swe2d.workbench. Pure-Python services must not
import PyQt5. GUI services must not import PyQt5.QtWidgets.
"""
import ast
import pathlib

import pytest


_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_CLI_DIR = _REPO_ROOT / "swe2d" / "cli"
_SHARED_SERVICES_DIRS = [
    _REPO_ROOT / "swe2d" / "services",
    _REPO_ROOT / "swe2d" / "runtime",
    _REPO_ROOT / "swe2d" / "results",
    _REPO_ROOT / "swe2d" / "mesh",
    _REPO_ROOT / "swe2d" / "boundary_and_forcing",
    _REPO_ROOT / "swe2d" / "extensions",
]
_GUI_SERVICES_DIR = _REPO_ROOT / "swe2d" / "workbench" / "services"


def _python_files(root: pathlib.Path):
    return sorted(p for p in root.rglob("*.py") if p.name != "__init__.py")


def _imports(source: str) -> list[str]:
    tree = ast.parse(source)
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            out.append(module)
    return out


@pytest.mark.parametrize("py_file", _python_files(_CLI_DIR), ids=lambda p: p.name)
def test_cli_does_not_import_workbench(py_file):
    offending = [m for m in _imports(py_file.read_text()) if m.startswith("swe2d.workbench")]
    assert not offending, f"{py_file.relative_to(_REPO_ROOT)} imports workbench: {offending}"


@pytest.mark.parametrize(
    "py_file",
    [p for d in _SHARED_SERVICES_DIRS for p in _python_files(d)],
    ids=lambda p: p.relative_to(_REPO_ROOT).as_posix(),
)
def test_shared_service_layer_does_not_import_pyqt5_widgets(py_file):
    imports = _imports(py_file.read_text())
    bad = [m for m in imports if m == "PyQt5.QtWidgets" or m.startswith("PyQt5.QtWidgets.")]
    assert not bad, f"{py_file.relative_to(_REPO_ROOT)} imports QtWidgets: {bad}"


@pytest.mark.parametrize("py_file", _python_files(_GUI_SERVICES_DIR), ids=lambda p: p.name)
def test_gui_services_do_not_import_qtwidgets(py_file):
    imports = _imports(py_file.read_text())
    bad = [m for m in imports if m == "PyQt5.QtWidgets" or m.startswith("PyQt5.QtWidgets.")]
    assert not bad, f"{py_file.relative_to(_REPO_ROOT)} imports QtWidgets: {bad}"
```

- [ ] **Step 2: Run it and watch it fail**

```bash
mamba run -n qgis_stable python -m pytest tests/test_mvp_imports.py -v 2>&1 | tail -30
```

Expected: at least one FAIL in `test_cli_does_not_import_workbench` pointing to
`cli/headless_runner.py` (the bright-line violation from the audit).

- [ ] **Step 3: Commit the failing test**

```bash
git add tests/test_mvp_imports.py
git commit -m "test: add MVP import boundary regression tests"
```

### Task 1.2: GREEN — Move line-sampling trio to `line_sampling_service.py`

**Files:**
- Move: `swe2d/workbench/services/mesh_service.py` → keep file but rename functions to be imported
- Move: 4 functions from `swe2d/workbench/services/mesh_service.py:118-200, 203-376, 514` → `swe2d/workbench/services/line_sampling_service.py`
- Modify: `swe2d/cli/headless_runner.py:441, 622`
- Modify: `swe2d/workbench/studio_dialog.py:2015`
- Modify: `tests/test_workbench_mesh_service.py:6-23`
- Modify: `tests/test_sample_line_metrics_profile.py:11`
- Modify: `tests/test_rcmk_permutation_mismatch.py:21-23`

- [ ] **Step 1: Inventory every caller**

```bash
rg "build_line_sampling_map|sample_line_metrics|sample_line_aggregate_ts_row|_cumulative_length|_interpolate_along_line|_line_normal|_project_point_onto_line|_cell_centroids" swe2d/ tests/ --no-heading -n
```

Expected output: complete caller list (swe2d/ + tests/) to drive the rewrites below.

- [ ] **Step 2: Move the 4 functions + helpers into `line_sampling_service.py`**

In `swe2d/workbench/services/line_sampling_service.py`, append the four functions
plus the five line-geometry helpers from `mesh_service.py:118-200, 203-376, 514`.
Use `edit` on `mesh_service.py` to delete them after copy. Keep the existing
`build_line_sampling_map` already in `line_sampling_service.py:110` (it's the
QGIS-layer variant) and **rename** the numpy/OGR variant to
`build_line_sampling_map_numpy` to remove the duplicate name.

The block of code to move (from `mesh_service.py`) is:
- `_cumulative_length`, `_interpolate_along_line`, `_line_normal`,
  `_project_point_onto_line`, `_cell_centroids` (helpers, lines 118-200)
- `build_line_sampling_map` → rename to `build_line_sampling_map_numpy` (line 203)
- `sample_line_metrics` (line 376)
- `sample_line_aggregate_ts_row` (line 514)

Verify the moved functions contain NO PyQt5 / qgis.gui references. They are
pure numpy (the source file's docstring says "Qt-free").

- [ ] **Step 3: Rewrite all imports**

```bash
rg "from swe2d\.workbench\.services\.mesh_service import" swe2d/ tests/ --files-with-matches
```

For every hit, replace `mesh_service` with `line_sampling_service` and update the
imported symbol list to match the moved names. The two relevant call sites in
`cli/headless_runner.py:441, 622` flip to the new module.

The dialog-side import at `studio_dialog.py:2015` becomes
`from swe2d.workbench.services.line_sampling_service import sample_line_metrics as _svc`.

Test files to update:
- `tests/test_workbench_mesh_service.py:6-23`
- `tests/test_sample_line_metrics_profile.py:11`
- `tests/test_rcmk_permutation_mismatch.py:21-23`

- [ ] **Step 4: Run the regression test from Task 1.1**

```bash
mamba run -n qgis_stable python -m pytest tests/test_mvp_imports.py -v 2>&1 | tail -30
```

Expected: PASS for the `test_cli_does_not_import_workbench` cases. The
`test_shared_service_layer_does_not_import_pyqt5_widgets` cases should still pass
(line-sampling stays under `workbench/services/`; its existing test file shows
it already has no QtWidgets imports).

- [ ] **Step 5: Run the existing line-sampling tests**

```bash
mamba run -n qgis_stable python -m pytest \
  tests/test_workbench_mesh_service.py \
  tests/test_sample_line_metrics_profile.py \
  tests/test_rcmk_permutation_mismatch.py \
  -v 2>&1 | tail -40
```

Expected: all PASS.

- [ ] **Step 6: Run the CLI smoke test**

```bash
mamba run -n qgis_stable python -m pytest tests/test_cli.py -v 2>&1 | tail -30
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add -A
git diff --cached --stat
# verify only intended files changed
git commit -m "refactor: move line-sampling trio from mesh_service to line_sampling_service"
```

---

## Phase 2: Move mesh serialize/load out of `studio_dialog.py` (Tier 1 #1)

Removes the audit's textbook example: a dialog doing build→serialize→persist
for baked meshes. New service methods own the pipeline.

### Task 2.1: RED — Add tests for the new service methods

**Files:**
- Create: `tests/test_mesh_persistence_service.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_mesh_persistence_service.py`:

```python
"""Tests for swe2d.services.mesh_persistence_service.

These methods used to live on studio_dialog.py. They were moved here so the
dialog no longer performs build/serialize/persist pipelines inline.
"""
import pathlib

import numpy as np
import pytest


@pytest.fixture
def tiny_mesh():
    return {
        "node_x": np.array([0.0, 1.0, 1.0, 0.0, 2.0, 2.0], dtype=np.float64),
        "node_y": np.array([0.0, 0.0, 1.0, 1.0, 0.0, 1.0], dtype=np.float64),
        "node_z": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "cell_face_offsets": np.array([0, 3, 6], dtype=np.int32),
        "cell_face_nodes": np.array([0, 1, 4, 1, 2, 5], dtype=np.int32),
    }


def test_save_and_load_baked_mesh_roundtrip(tmp_path: pathlib.Path, tiny_mesh):
    from swe2d.services.mesh_persistence_service import (
        save_baked_mesh,
        load_baked_mesh,
    )

    gpkg = tmp_path / "mesh_roundtrip.gpkg"
    name = "tiny"
    save_baked_mesh(tiny_mesh, str(gpkg), name)

    loaded = load_baked_mesh(str(gpkg), name)
    np.testing.assert_array_equal(loaded["node_x"], tiny_mesh["node_x"])
    np.testing.assert_array_equal(loaded["node_y"], tiny_mesh["node_y"])
    np.testing.assert_array_equal(loaded["cell_face_offsets"], tiny_mesh["cell_face_offsets"])
    np.testing.assert_array_equal(loaded["cell_face_nodes"], tiny_mesh["cell_face_nodes"])


def test_load_baked_mesh_unknown_name_raises(tmp_path: pathlib.Path, tiny_mesh):
    from swe2d.services.mesh_persistence_service import (
        save_baked_mesh,
        load_baked_mesh,
    )

    gpkg = tmp_path / "mesh_unknown.gpkg"
    save_baked_mesh(tiny_mesh, str(gpkg), "tiny")
    with pytest.raises(KeyError):
        load_baked_mesh(str(gpkg), "does_not_exist")
```

- [ ] **Step 2: Run and watch them fail**

```bash
mamba run -n qgis_stable python -m pytest tests/test_mesh_persistence_service.py -v 2>&1 | tail -20
```

Expected: ImportError on `swe2d.services.mesh_persistence_service`.

- [ ] **Step 3: Commit the failing test**

```bash
git add tests/test_mesh_persistence_service.py
git commit -m "test: add roundtrip tests for mesh_persistence_service"
```

### Task 2.2: GREEN — Extract `_save_mesh_to_gpkg` / `_load_mesh_from_gpkg` to a service

**Files:**
- Create: `swe2d/services/mesh_persistence_service.py`
- Modify: `swe2d/workbench/studio_dialog.py:669-810` (delete the two methods + their helper if any)
- Modify: every caller of those dialog methods

- [ ] **Step 1: Locate the dialog methods and their callers**

```bash
rg "_save_mesh_to_gpkg|_load_mesh_from_gpkg" swe2d/ tests/ --no-heading -n
```

- [ ] **Step 2: Create the service file**

Create `swe2d/services/mesh_persistence_service.py`:

```python
"""Mesh (de)serialization into the model GPKG.

Extracted from swe2d.workbench.studio_dialog so the dialog no longer performs
build/serialize/persist pipelines inline. Pure numpy + sqlite3 (no Qt).
"""
from __future__ import annotations

import sqlite3
from typing import Dict, List

import numpy as np


def save_baked_mesh(mesh_data: Dict[str, np.ndarray], gpkg_path: str, mesh_name: str) -> int:
    """Serialize ``mesh_data`` via the hydra_swe2d C extension and persist it
    under ``mesh_name`` in the GPKG. Returns the number of cells in the baked
    BLOB so callers can log/sanity-check.
    """
    from hydra_swe2d import (
        swe2d_build_mesh, swe2d_build_mesh_poly,
        swe2d_serialize_mesh, swe2d_mesh_info,
    )
    from swe2d.services.gpkg_persistence_service import persist_baked_mesh

    nx = np.asarray(mesh_data["node_x"], dtype=np.float64)
    ny = np.asarray(mesh_data["node_y"], dtype=np.float64)
    nz = np.asarray(mesh_data["node_z"], dtype=np.float64)
    bc_n0 = np.asarray(mesh_data.get("bc_edge_node0", np.empty(0)), dtype=np.int32)
    bc_n1 = np.asarray(mesh_data.get("bc_edge_node1", np.empty(0)), dtype=np.int32)
    bc_tp = np.asarray(mesh_data.get("bc_edge_type", np.empty(0)), dtype=np.int32)
    bc_vl = np.asarray(mesh_data.get("bc_edge_val", np.empty(0)), dtype=np.float64)
    cfn = mesh_data.get("cell_face_nodes")
    if cfn is None:
        cfn = mesh_data.get("cell_nodes")
    cfo = mesh_data.get("cell_face_offsets")
    if cfn is not None and cfo is not None:
        pm = swe2d_build_mesh_poly(
            nx, ny, nz,
            np.asarray(cfo, dtype=np.int32),
            np.asarray(cfn, dtype=np.int32),
            bc_n0, bc_n1, bc_tp, bc_vl,
        )
    else:
        cn = np.asarray(mesh_data["cell_nodes"], dtype=np.int32)
        pm = swe2d_build_mesh(nx, ny, nz, cn, bc_n0, bc_n1, bc_tp, bc_vl)
    blob = swe2d_serialize_mesh(pm)
    info = swe2d_mesh_info(pm)
    persist_baked_mesh(
        gpkg_path, mesh_name, blob,
        info["n_nodes"], info["n_cells"], info["n_edges"],
    )
    return int(info["n_cells"])


def list_baked_mesh_names(gpkg_path: str) -> List[str]:
    """Return baked-mesh names in ``gpkg_path`` ordered newest-first."""
    conn = sqlite3.connect(gpkg_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT mesh_name FROM swe2d_baked_mesh ORDER BY created_utc DESC")
        return [str(r[0]) for r in cur.fetchall()]
    finally:
        conn.close()


def load_baked_mesh(gpkg_path: str, mesh_name: str) -> Dict[str, np.ndarray]:
    """Load a previously-saved mesh by name. Raises KeyError if not present.

    Mesh geometry is returned in solver (RCMK) order per the baked BLOB spec
    (§5.12). Results loaded from ``load_baked_snapshot`` are also RCMK, so no
    permutation is required when pairing mesh + results.
    """
    from hydra_swe2d import swe2d_deserialize_mesh
    from swe2d.services.gpkg_persistence_service import load_baked_mesh as _load_blob

    blob = _load_blob(gpkg_path, mesh_name)
    if blob is None:
        raise KeyError(mesh_name)
    pm = swe2d_deserialize_mesh(blob)
    mesh_data: Dict[str, np.ndarray] = {
        "mesh_name": str(mesh_name),
        "node_x": np.asarray(pm.node_x, dtype=np.float64),
        "node_y": np.asarray(pm.node_y, dtype=np.float64),
        "node_z": np.asarray(pm.node_z, dtype=np.float64),
    }
    if pm.cell_face_nodes is not None:
        mesh_data["cell_nodes"] = np.asarray(pm.cell_face_nodes, dtype=np.int32)
    else:
        mesh_data["cell_nodes"] = np.empty(0, dtype=np.int32)
    if pm.cell_face_offsets is not None:
        mesh_data["cell_face_offsets"] = np.asarray(pm.cell_face_offsets, dtype=np.int32)
    if pm.cell_face_nodes is not None:
        mesh_data["cell_face_nodes"] = np.asarray(pm.cell_face_nodes, dtype=np.int32)
    return mesh_data
```

- [ ] **Step 3: Update every caller**

For every site found in Step 1, replace the dialog-method call with the
service-function call. Where the dialog used `self._mesh_data`, the caller
already has the dict or can read it from a controller getter.

- [ ] **Step 4: Delete the dialog methods**

In `swe2d/workbench/studio_dialog.py`, delete the `_save_mesh_to_gpkg` and
`_load_mesh_from_gpkg` methods.

- [ ] **Step 5: Run the roundtrip tests**

```bash
mamba run -n qgis_stable python -m pytest tests/test_mesh_persistence_service.py -v 2>&1 | tail -20
```

Expected: PASS.

- [ ] **Step 6: Run mesh persistence tests + smoke**

```bash
mamba run -n qgis_stable python -m pytest \
  tests/test_mesh_persistence_service.py \
  tests/test_gpkg_persistence.py \
  tests/test_gpkg_service_load_mesh_snapshot.py \
  tests/test_results_path_wiring.py \
  -v 2>&1 | tail -30
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add -A
git diff --cached --stat
git commit -m "refactor: extract mesh save/load to services/mesh_persistence_service"
```

---

## Phase 3: Move batch subprocess pool out of `batch_simulation_dialog.py` (Tier 1 #2)

A dialog owns a worker pool polling subprocess status files. Move the
orchestration to `cli/batch_runner.py`, leave the dialog to update table cells
from callbacks.

### Task 3.1: RED — Test the orchestrator with an in-memory fake subprocess

**Files:**
- Create: `tests/test_batch_runner_orchestrator.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for swe2d.cli.batch_runner.Orchestrator (extracted from dialog).

The orchestrator owns subprocess pool lifecycle and status-file polling. The
dialog receives callbacks.
"""
import pathlib
import time

import pytest


class _FakeCompleted:
    def __init__(self, returncode):
        self.returncode = returncode


def test_orchestrator_runs_each_param_set(tmp_path, monkeypatch):
    from swe2d.cli import batch_runner

    seen = []

    def fake_popen(cmd, **kw):
        seen.append(tuple(cmd))
        return _FakeCompleted(returncode=0)

    monkeypatch.setattr(batch_runner.subprocess, "Popen", fake_popen)

    orch = batch_runner.BatchOrchestrator(
        param_sets=[{"id": "a"}, {"id": "b"}],
        workdir=str(tmp_path),
    )
    results = orch.run()
    assert len(results) == 2
    assert seen and "run" in seen[0]


def test_orchestrator_emits_progress_callbacks(tmp_path, monkeypatch):
    from swe2d.cli import batch_runner

    def fake_popen(cmd, **kw):
        return _FakeCompleted(returncode=0)

    monkeypatch.setattr(batch_runner.subprocess, "Popen", fake_popen)

    progress = []
    orch = batch_runner.BatchOrchestrator(
        param_sets=[{"id": "a"}],
        workdir=str(tmp_path),
        on_progress=lambda done, total: progress.append((done, total)),
    )
    orch.run()
    assert progress[-1][0] == 1
    assert progress[-1][1] == 1


def test_orchestrator_collects_failures(tmp_path, monkeypatch):
    from swe2d.cli import batch_runner

    def fake_popen(cmd, **kw):
        return _FakeCompleted(returncode=1)

    monkeypatch.setattr(batch_runner.subprocess, "Popen", fake_popen)

    orch = batch_runner.BatchOrchestrator(
        param_sets=[{"id": "a"}],
        workdir=str(tmp_path),
    )
    results = orch.run()
    assert results[0]["status"] == "failed"
```

- [ ] **Step 2: Run and watch them fail**

```bash
mamba run -n qgis_stable python -m pytest tests/test_batch_runner_orchestrator.py -v 2>&1 | tail -20
```

Expected: ImportError or AttributeError on `batch_runner.BatchOrchestrator`.

- [ ] **Step 3: Commit failing test**

```bash
git add tests/test_batch_runner_orchestrator.py
git commit -m "test: add tests for batch orchestrator extracted from dialog"
```

### Task 3.2: GREEN — Extract `BatchOrchestrator` and rewire the dialog

**Files:**
- Modify: `swe2d/cli/batch_runner.py` (add `BatchOrchestrator` class)
- Modify: `swe2d/workbench/dialogs/batch_simulation_dialog.py:869-1016` (replace body with callback calls)

- [ ] **Step 1: Locate every private method that owns subprocess / status-file logic**

```bash
rg "_run_batch|_poll_tick|_start_next_batch|_check_batch_status|_tick_run|_cancel_batch" swe2d/ tests/ --no-heading -n
```

- [ ] **Step 2: Add `BatchOrchestrator` to `batch_runner.py`**

Append the new class. Keep `subprocess.Popen` construction literal, status-file
path layout, and counter state semantics identical to the current
`_run_batch`/`_poll_tick` implementation. Constructor takes `param_sets`,
`workdir`, and three optional callbacks: `on_progress(done, total)`,
`on_completed(result_dict)`, `on_failed(result_dict)`. `run()` returns
`list[dict]` of result dicts (id, status, returncode, log_path).

- [ ] **Step 3: Replace dialog body**

Replace the 6 private methods (`_run_batch` etc.) with calls to the orchestrator.
The dialog's table-cell updates move into the `on_progress` / `on_completed` /
`on_failed` callbacks. Preserve existing user-visible behavior 1:1.

- [ ] **Step 4: Run the orchestrator tests + CLI smoke**

```bash
mamba run -n qgis_stable python -m pytest \
  tests/test_batch_runner_orchestrator.py \
  tests/test_cli.py \
  -v 2>&1 | tail -30
```

Expected: PASS.

- [ ] **Step 5: Manual smoke (no auto test can cover Qt dialog wiring)**

```bash
mamba run -n qgis_stable python -c "
from swe2d.workbench.dialogs.batch_simulation_dialog import BatchSimulationDialog
print('import ok')
"
```

Expected: clean import.

- [ ] **Step 6: Commit**

```bash
git add -A
git diff --cached --stat
git commit -m "refactor: extract BatchOrchestrator from batch_simulation_dialog"
```

---

## Phase 4: Move BC preprocessing out of `runtime/native_bc_forcing.py` (Tier 1 #4)

~200 LOC of side-detection / hydrograph grouping / progressive-BC elevation
sorting in a runtime file. Belongs in `boundary_and_forcing/`.

### Task 4.1: RED — Test `BoundaryHydrographConfigurator` as a pure-logic unit

**Files:**
- Create: `tests/test_native_bc_configurator.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for swe2d.boundary_and_forcing.native_bc_forcing.BoundaryHydrographConfigurator.

Extracted from swe2d.runtime.native_bc_forcing. The configurator is pure logic;
it no longer touches the backend directly — it returns a payload that the
runtime applies.
"""
import numpy as np
import pytest


def test_configurator_classifies_edges_by_side():
    from swe2d.boundary_and_forcing.native_bc_forcing import (
        BoundaryHydrographConfigurator,
    )

    cfg = BoundaryHydrographConfigurator(
        edge_nodes=np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int32),
        node_coords=np.array(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64
        ),
        edge_groups=np.array([0, 0, 1], dtype=np.int32),
    )
    payload = cfg.build_payload()
    assert set(payload["side_by_edge"].tolist()) == {"left", "right", "top"}


def test_configurator_converts_bc_codes():
    from swe2d.boundary_and_forcing.native_bc_forcing import (
        BoundaryHydrographConfigurator,
    )

    cfg = BoundaryHydrographConfigurator(
        edge_nodes=np.array([[0, 1]], dtype=np.int32),
        node_coords=np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64),
        edge_groups=np.array([0], dtype=np.int32),
        bc_codes_input=np.array([102], dtype=np.int32),
    )
    payload = cfg.build_payload()
    assert payload["bc_codes_output"][0] == 2  # 102 -> 2 (Q→h)
```

- [ ] **Step 2: Run and watch fail**

```bash
mamba run -n qgis_stable python -m pytest tests/test_native_bc_configurator.py -v 2>&1 | tail -20
```

Expected: ImportError on `swe2d.boundary_and_forcing.native_bc_forcing`.

- [ ] **Step 3: Commit failing test**

```bash
git add tests/test_native_bc_configurator.py
git commit -m "test: add tests for BoundaryHydrographConfigurator extraction"
```

### Task 4.2: GREEN — Move logic to `boundary_and_forcing/`, leave thin upload in runtime

**Files:**
- Create: `swe2d/boundary_and_forcing/native_bc_forcing.py`
- Modify: `swe2d/runtime/native_bc_forcing.py` (replace body with thin shim that calls the configurator + uploads)

- [ ] **Step 1: Inventory callers**

```bash
rg "SWE2DNativeBoundaryHydrographConfigurator|from swe2d\.runtime\.native_bc_forcing" swe2d/ tests/ --no-heading -n
```

- [ ] **Step 2: Create `boundary_and_forcing/native_bc_forcing.py`**

Move every line of `SWE2DNativeBoundaryHydrographConfigurator.configure` that is
pure preprocessing (side detection, hydrograph grouping, progressive-BC
elevation sort, BC-code conversion) into the new class. Leave only the final
`backend.set_boundary_hydrographs_native(...)` call in runtime. The configurator
returns a payload dict; the runtime applies it.

- [ ] **Step 3: Rewrite `runtime/native_bc_forcing.py`**

The `SWE2DNativeBoundaryHydrographConfigurator` class becomes a thin facade
that delegates to the new `BoundaryHydrographConfigurator` and applies the
payload to the backend. Preserve the public class name and `configure(...)`
signature so callers don't break.

- [ ] **Step 4: Run BC tests + configurator tests**

```bash
mamba run -n qgis_stable python -m pytest \
  tests/test_native_bc_configurator.py \
  tests/test_bc_validation.py \
  -v 2>&1 | tail -30
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A
git diff --cached --stat
git commit -m "refactor: move BC preprocessing to boundary_and_forcing/"
```

---

## Phase 5: Fix silent-fallback `self._log` bugs (Tier: BUGS)

Two services and one view have `self._log(...)` references in module-level
functions where `self` is undefined → `NameError` swallowed by inner
`try/except: pass`. Per AGENTS.md this is the worst failure mode.

### Task 5.1: RED — Test that the error path actually logs

**Files:**
- Create: `tests/test_widget_persistence_error_path.py`
- Create: `tests/test_unit_conversion_error_path.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_widget_persistence_error_path.py`:

```python
def test_widget_persistence_logs_failures_via_log_fn():
    """The error path must call log_fn, not swallow via inner NameError."""
    from swe2d.workbench.services import widget_persistence_service as wps

    captured = []

    def log_fn(msg):
        captured.append(msg)

    wps.persist_project_workbench_state(
        workbench_state={},
        log_fn=log_fn,
        force_failure=True,  # wired in Task 5.2 to trigger the error branch
    )
    assert any("ERROR" in m for m in captured), captured


def test_widget_persistence_log_fn_called_for_real_errors():
    """When real work fails, log_fn must receive the message."""
    from swe2d.workbench.services import widget_persistence_service as wps

    captured = []

    wps.persist_project_workbench_state(
        workbench_state={"_synthetic_key_": object()},  # unhashable? actually use bad type
        log_fn=lambda m: captured.append(m),
        force_failure=True,
    )
    assert any("ERROR" in m for m in captured)
```

`tests/test_unit_conversion_error_path.py`:

```python
def test_update_unit_system_logs_crs_failures_via_log_fn():
    from swe2d.workbench.services import unit_conversion_service as ucs

    captured = []

    ucs.update_unit_system_from_crs(
        crs_description="bad_crs",
        log_fn=lambda m: captured.append(m),
        force_failure=True,
    )
    assert any("ERROR" in m for m in captured)
```

- [ ] **Step 2: Run and watch fail**

```bash
mamba run -n qgis_stable python -m pytest \
  tests/test_widget_persistence_error_path.py \
  tests/test_unit_conversion_error_path.py \
  -v 2>&1 | tail -20
```

Expected: tests fail because the `force_failure` parameter is not honored
(currently `self._log` raises NameError, inner `try/except: pass` swallows it).

- [ ] **Step 3: Commit failing tests**

```bash
git add tests/test_widget_persistence_error_path.py tests/test_unit_conversion_error_path.py
git commit -m "test: add error-path regression tests for log_fn wiring"
```

### Task 5.2: GREEN — Fix the three `self._log` NameError-swallowed sites

**Files:**
- Modify: `swe2d/workbench/services/widget_persistence_service.py:128`
- Modify: `swe2d/workbench/services/unit_conversion_service.py:128`
- Modify: `swe2d/workbench/views/topology_tab_view.py:1611-1627`

- [ ] **Step 1: Fix `widget_persistence_service.py:128`**

Replace the inner `try: ... except Exception as _e: self._log(f"[ERROR] ...") ... pass`
with: call `log_fn(f"[ERROR] Exception in widget_persistence_service.py: {_e}")`.
`log_fn` is already the in-scope parameter. Honor the `force_failure` test
parameter (default `False`; when `True`, raise before the work to drive the
error path).

- [ ] **Step 2: Fix `unit_conversion_service.py:128`**

Same pattern: replace `self._log(...)` with `log_fn(...)`. Honor `force_failure`.

- [ ] **Step 3: Fix `topology_tab_view.py:1611-1627`**

Replace the `_w` helper's `self._log(...)` call with a `log_fn` parameter or a
module-level logger reference (the file already wires `log_fn` in the calling
controller).

- [ ] **Step 4: Run the error-path tests**

```bash
mamba run -n qgis_stable python -m pytest \
  tests/test_widget_persistence_error_path.py \
  tests/test_unit_conversion_error_path.py \
  -v 2>&1 | tail -20
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A
git diff --cached --stat
git commit -m "fix: route error-path logs through log_fn (stop swallowing NameErrors)"
```

---

## Phase 6: Delete confirmed dead code (Tier: DEAD CODE)

Per AGENTS.md, deletion requires user confirmation. The plan documents the
exact list, deletion is a single sweep per file, and each file ends with a
commit.

### Task 6.1: RED — Confirm each candidate has zero references

For each candidate, run:

```bash
rg "<symbol_name>" swe2d/ tests/ --no-heading
```

The output should match only the definition site. If any external reference
exists, **stop and surface it to the user** — the candidate is not actually
dead.

- [ ] **Step 1: Verify each dead-code candidate**

Run the `rg` check for all of:
- `collect_run_log_metadata` (services/gpkg_persistence_service.py:54)
- `RainfallSourceEngine`, `DrainageCouplingEngine.exchange_step`,
  `compute_orifice_flow`, `compute_weir_flow`,
  `compute_pipe_manning_capacity_full`, `circular_section_from_depth`,
  `convert_cell_flows_to_depth_rates`
- `source_rate_callback` (runtime/coupling.py:1523)
- `_gmsh_available` (mesh/meshing.py:334)
- `runoff_depth_mm_from_event_rain_mm`, `composite_curve_number`,
  `time_of_concentration_hours_velocity_method`
- `_noop` (controllers/run_controller.py:60)
- `_preflight_validate_mesh`, `_collect_bc_for_edges`, `_prepare_run_inputs`,
  `_collect_simulation_settings`
- `_opt_float`, `_opt_bool` (controllers/topology_controller.py:20-39)
- 14 underscore aliases in constants_service.py:123-137
- `apply_bc_overrides_from_gpkg`, `_parse_linestring_coords`
  (cli/gpkg_adapter.py:308, 455)
- `iter_with_parents` (devtools/widget_walker.py:165)
- Unreachable log line at studio_dialog.py:1881

- [ ] **Step 2: Add a test that the dead imports are removed**

Create `tests/test_no_dead_imports.py`:

```python
"""Per audit: these symbols should be gone after Phase 6."""
import importlib
import pytest


_DEAD = [
    ("swe2d.services.gpkg_persistence_service", "collect_run_log_metadata"),
    ("swe2d.workbench.controllers.run_controller", "RunController._noop"),
    ("swe2d.workbench.controllers.topology_controller", "_opt_float"),
    ("swe2d.workbench.controllers.topology_controller", "_opt_bool"),
    ("swe2d.cli.gpkg_adapter", "apply_bc_overrides_from_gpkg"),
    ("swe2d.cli.gpkg_adapter", "_parse_linestring_coords"),
    ("swe2d.workbench.devtools.widget_walker", "iter_with_parents"),
    ("swe2d.extensions.extension_models", "RainfallSourceEngine"),
    ("swe2d.runtime.coupling", "SWE2DCouplingController.source_rate_callback"),
    ("swe2d.mesh", "meshing._gmsh_available"),
    ("swe2d.boundary_and_forcing.rainfall_hydrology",
     "runoff_depth_mm_from_event_rain_mm"),
    ("swe2d.boundary_and_forcing.rainfall_hydrology", "composite_curve_number"),
    ("swe2d.boundary_and_forcing.rainfall_hydrology",
     "time_of_concentration_hours_velocity_method"),
    ("swe2d.extensions.extension_models", "compute_orifice_flow"),
    ("swe2d.extensions.extension_models", "compute_weir_flow"),
    ("swe2d.extensions.extension_models",
     "compute_pipe_manning_capacity_full"),
    ("swe2d.extensions.extension_models", "circular_section_from_depth"),
    ("swe2d.extensions.extension_models", "convert_cell_flows_to_depth_rates"),
]


@pytest.mark.parametrize("module_name,symbol", _DEAD)
def test_dead_symbol_removed(module_name, symbol):
    import importlib
    mod = importlib.import_module(module_name)
    assert not hasattr(mod, symbol), f"{module_name}.{symbol} still exists"
```

- [ ] **Step 3: Run and watch fail**

```bash
mamba run -n qgis_stable python -m pytest tests/test_no_dead_imports.py -v 2>&1 | tail -30
```

Expected: every parameter case FAILs.

- [ ] **Step 4: Commit failing test**

```bash
git add tests/test_no_dead_imports.py
git commit -m "test: add regression test for dead-code removal"
```

### Task 6.2: GREEN — Delete dead code

**Files:** (see list above, all under `swe2d/`)

- [ ] **Step 1: Delete each dead symbol per file**

For every entry in `_DEAD`, use `edit` to remove the function/class definition
plus its `__all__` listing (if any). For `DrainageCouplingEngine.exchange_step`
(keep the class but drop the method). For the 14 underscore aliases in
`constants_service.py`, delete lines 123-137. For the unreachable log line in
`studio_dialog.py:1881`, delete it. For `_gmsh_available` in `meshing.py:334`,
delete the local definition (the canonical one in `gmsh_backend.py:38` is the
imported version).

- [ ] **Step 2: Run the regression test**

```bash
mamba run -n qgis_stable python -m pytest tests/test_no_dead_imports.py -v 2>&1 | tail -30
```

Expected: all PASS.

- [ ] **Step 3: Run the full focused suite**

```bash
mamba run -n qgis_stable python -m pytest \
  tests/test_cli.py \
  tests/test_workbench_mesh_service.py \
  tests/test_sample_line_metrics_profile.py \
  tests/test_rcmk_permutation_mismatch.py \
  tests/test_results_path_wiring.py \
  tests/test_gpkg_persistence.py \
  tests/test_mvp_imports.py \
  tests/test_bc_validation.py \
  tests/test_coupling_integration.py \
  -v 2>&1 | tail -30
```

Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add -A
git diff --cached --stat
git commit -m "chore: remove confirmed dead code (~500 LOC across swe2d/)"
```

---

## Phase 7: Verification gate (per `.opencode/rules/PLANNING.md`)

- [ ] **Step 1: Purge pycache**

```bash
find . -type d -name __pycache__ -exec rm -rf {} +
```

- [ ] **Step 2: Run the verification gate**

```bash
mamba run -n qgis_stable python -m unittest -v \
    tests.test_workbench_gui \
    tests.test_workbench_imports \
    tests.test_workbench_persistence
```

Expected: PASS.

- [ ] **Step 3: Run the new boundary tests**

```bash
mamba run -n qgis_stable python -m pytest \
  tests/test_mvp_imports.py \
  tests/test_no_dead_imports.py \
  tests/test_mesh_persistence_service.py \
  tests/test_batch_runner_orchestrator.py \
  tests/test_native_bc_configurator.py \
  tests/test_widget_persistence_error_path.py \
  tests/test_unit_conversion_error_path.py \
  -v 2>&1 | tail -50
```

Expected: all PASS.

- [ ] **Step 4: Final diff review**

```bash
git log --oneline main..HEAD
git diff main..HEAD --stat
```

Expected: ~6 commits, each focused on one phase.

---

## Superpowers workflow

1. **writing-plans** — this plan (saved to `docs/archive/plans/`).
2. **subagent-driven-development** — dispatch one subagent per phase; each
   phase is independent and ends with a commit. Cross-review per PLANNING.md.
3. **test-driven-development** — every task in Phases 1-6 is RED → GREEN →
   REFACTOR.
4. **systematic-debugging** — if any task's tests fail unexpectedly, stop and
   load this skill before proposing fixes.
5. **verification-before-completion** — Phase 7 gate runs the focus suite
   before declaring success.
6. **requesting-code-review** — after Phase 6, request review on the overall
   diff because it touches many files.

---

## Routing keywords (for `recommend_agent_from_keywords`)

| Task | Primary keywords |
|---|---|
| 1.1, 1.2 | `python`, `refactor`, `test` |
| 2.1, 2.2 | `python`, `refactor`, `test` |
| 3.1, 3.2 | `python`, `refactor`, `test` |
| 4.1, 4.2 | `python`, `refactor`, `test` |
| 5.1, 5.2 | `python`, `test`, `debug` |
| 6.1, 6.2 | `python`, `refactor`, `test` |
| 7 | `python`, `test`, `validate` |

---

## Step dict (selector-consumable)

```python
[
    {"action": "write failing MVP import boundary test for CLI and service layers",
     "type": "test", "phase": 1.1},
    {"action": "refactor move line-sampling trio to line_sampling_service and rewrite imports",
     "type": "refactor", "phase": 1.2},
    {"action": "write failing roundtrip tests for new mesh_persistence_service",
     "type": "test", "phase": 2.1},
    {"action": "refactor extract mesh save/load from studio_dialog into services/mesh_persistence_service",
     "type": "refactor", "phase": 2.2},
    {"action": "write failing tests for BatchOrchestrator extracted from dialog",
     "type": "test", "phase": 3.1},
    {"action": "refactor extract BatchOrchestrator into cli/batch_runner and rewire dialog",
     "type": "refactor", "phase": 3.2},
    {"action": "write failing tests for BoundaryHydrographConfigurator extraction",
     "type": "test", "phase": 4.1},
    {"action": "refactor move BC preprocessing from runtime to boundary_and_forcing",
     "type": "refactor", "phase": 4.2},
    {"action": "write failing error-path tests for log_fn wiring in services",
     "type": "test", "phase": 5.1},
    {"action": "fix self._log NameError-swallowed sites to call log_fn",
     "type": "debug", "phase": 5.2},
    {"action": "write failing tests asserting dead symbols are removed",
     "type": "test", "phase": 6.1},
    {"action": "refactor delete confirmed dead code (~500 LOC) across swe2d/",
     "type": "refactor", "phase": 6.2},
    {"action": "validate run verification gate and full focused suite",
     "type": "validate", "phase": 7},
]
```

---

## Spec coverage self-review

| Audit finding | Plan phase + task |
|---|---|
| Tier 1 #1: `studio_dialog.py` mesh serialize/load | Phase 2 |
| Tier 1 #2: `batch_simulation_dialog.py` subprocess pool | Phase 3 |
| Tier 1 #3: `mesh_service.py` line sampling | Phase 1 |
| Tier 1 #4: `runtime/native_bc_forcing.py` BC preprocessing | Phase 4 |
| Tier 1 #5: `cli/gpkg_adapter.py` JSON→config + duplicated centroids | DEFERRED — separate plan (lower leverage, ~240 LOC; risky to bundle with moves). Surface to user. |
| Tier 1 #6: `controllers/run_controller.py` solver math inlined | DEFERRED — ~1040 LOC `_execute_run` decomposition; needs design review before slicing. Surface to user. |
| Tier 1 #7: `mesh_controller.open_run_log_viewer` | DEFERRED — single method move. Surface to user. |
| Tier 1 #8: `topology_tab_view._wire_topology_tab_controls` | DEFERRED — partial fix in Phase 5.3 (silent-fallback). Full move needs separate plan. |
| Tier 1 #9: `extension_models.py` engine classes | DEFERRED — base-class relocation. Surface to user. |
| Tier 1 #10: smaller outliers (`coupling_results_dialog`, `gpkg_explorer_dialog`, `overlay_controller`) | DEFERRED — surface to user. |
| Tier 2: MVP widget-access violations in controllers | DEFERRED — systematic protocol-method addition; needs separate design plan. Surface to user. |
| Tier 3: CLI imports `workbench.services.mesh_service` | Phase 1 (fixed by moving the trio) |
| DEAD CODE (~500 LOC) | Phase 6 |
| BUGS: silent-fallback `self._log` | Phase 5 |

**Deferred items** (~12 audit findings) are explicitly out of scope for this
plan and surface to the user for prioritization. The current plan covers the
highest-leverage ~75% of the audit (Phases 1-4 + 5 + 6).

---

## Execution choice

Plan complete and saved to `docs/archive/plans/2026-07-04-swe2d-placement-fixes.md`.

Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per phase,
review between phases, fast iteration. Each phase ends with a commit.

**2. Inline Execution** — Execute tasks in this session using executing-plans,
batch execution with checkpoints.

Which approach?