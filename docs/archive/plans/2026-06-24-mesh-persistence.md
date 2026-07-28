---
type: plan
status: complete
created: 2026-06-24
completed: 2026-07-25
---

# Mesh Persistence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Save solver mesh arrays to GeoPackage as compressed BLOBs and load them back into SWE2DBackend, with QGIS UI support.

**Architecture:** Mesh data (node_x/y/z, cell_nodes, BC arrays) already live in `SWE2DBackend._mesh_h` on the GPU. A new `export_mesh_data()` call reads them back to host. Python functions handle numpy → BLOB (zlib) serialization. The GPKG table is `swe2d_mesh`. QGIS buttons integrate into the mesh workflow.

**Tech Stack:** numpy, sqlite3, zlib, pybind11 (existing binding for solver D2H)

---

### Task 1: Backend `export_mesh_data()` method

**Files:**
- Modify: `swe2d/runtime/backend.py:290-366` (inside `build_mesh()`)

- [ ] **Step 1: Read mesh data variables in build_mesh scope**

Inside `build_mesh()`, after `self._n_cells = info["n_cells"]` (line 350), the triangle/polygon paths already have `node_x`, `node_y`, `node_z`, `cell_nodes_flat`, `bc_n0`, `bc_n1`, `bc_tp`, `bc_vl` via closure capture.  No new storage is needed during build — the user calls export after build.  The `cell_nodes` and `face_offsets` arrays from the original call must be stored on `self`.  Add storage:

```python
self._mesh_node_x = np.asarray(node_x, dtype=np.float64)
self._mesh_node_y = np.asarray(node_y, dtype=np.float64)
self._mesh_node_z = np.asarray(node_z, dtype=np.float64)
self._mesh_cell_nodes = np.asarray(cell_nodes_flat, dtype=np.int32)
self._mesh_face_offsets = None
if cell_face_offsets is not None:
    self._mesh_face_offsets = np.asarray(face_offsets, dtype=np.int32)
    self._mesh_face_nodes = cell_nodes_flat  # polygon: same flat array
self._bc_n0 = bc_n0
self._bc_n1 = bc_n1
self._bc_tp = bc_tp
self._bc_vl = bc_vl
```

Add these lines immediately after `self._boundary_edge_cells = None` (line 366).

- [ ] **Step 2: Add `export_mesh_data()` method**

After `cell_areas()` (line 1010), add:

```python
def export_mesh_data(self) -> Dict[str, np.ndarray]:
    """Return copy of all mesh arrays for serialization (host memory)."""
    out = {
        "node_x": self._mesh_node_x.copy(),
        "node_y": self._mesh_node_y.copy(),
        "node_z": self._mesh_node_z.copy(),
        "cell_nodes": self._mesh_cell_nodes.copy(),
    }
    if self._mesh_face_offsets is not None:
        out["face_offsets"] = self._mesh_face_offsets.copy()
        out["face_nodes"] = self._mesh_face_nodes.copy()
    if self._bc_n0.size > 0:
        out["bc_n0"] = self._bc_n0.copy()
        out["bc_n1"] = self._bc_n1.copy()
        out["bc_tp"] = self._bc_tp.copy()
        out["bc_val"] = self._bc_vl.copy()
    return out
```

Also add `Dict` to the typing imports at the top of the file (line 22 currently has `from typing import Callable, List, Optional, Tuple` — add `Dict`).

- [ ] **Step 3: Syntax check**

```bash
python3 -c "import py_compile; py_compile.compile('swe2d/runtime/backend.py', doraise=True); print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add swe2d/runtime/backend.py
git commit -m "feat: add export_mesh_data() and mesh array caching to SWE2DBackend"
```

### Task 2: GPKG persist / load functions

**Files:**
- Modify: `swe2d/workbench/services/gpkg_persistence_service.py`

- [ ] **Step 1: Write `persist_mesh_to_geopackage()`**

After `persist_mesh_results_to_geopackage()` (ends ~line 409), add:

```python
import zlib

def _compress_array(arr: np.ndarray) -> bytes:
    return zlib.compress(arr.tobytes())

def _decompress_array(data: bytes, dtype: np.dtype, shape: tuple) -> np.ndarray:
    return np.frombuffer(zlib.decompress(data), dtype=dtype).reshape(shape)

def persist_mesh_to_geopackage(
    gpkg_path: str,
    mesh_name: str,
    mesh_data: Dict[str, np.ndarray],
    crs_wkt: str = "",
    description: str = "",
    log_fn: Optional[Callable[[str], None]] = None,
) -> None:
    """Save mesh arrays to swe2d_mesh table as compressed BLOBs."""
    if not gpkg_path or not mesh_data:
        return
    node_x = mesh_data.get("node_x")
    if node_x is None or node_x.size == 0:
        return
    nnodes = int(node_x.size)
    ncells = int(mesh_data.get("cell_nodes", np.empty(0)).size)
    import hashlib
    h = hashlib.sha256()
    for key in ("node_x", "node_y", "node_z", "cell_nodes"):
        arr = mesh_data.get(key)
        if arr is not None:
            h.update(arr.tobytes())
    import datetime
    conn = sqlite3.connect(gpkg_path)
    try:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS swe2d_mesh (
                mesh_name TEXT PRIMARY KEY,
                created_utc TEXT,
                nnodes INTEGER,
                ncells INTEGER,
                crs_wkt TEXT,
                hash TEXT,
                node_x BLOB, node_y BLOB, node_z BLOB,
                cell_nodes BLOB,
                face_offsets BLOB, face_nodes BLOB,
                bc_n0 BLOB, bc_n1 BLOB, bc_type BLOB, bc_val BLOB,
                terrain_source TEXT, terrain_path TEXT,
                description TEXT
            )
        """)
        def _b(key):
            a = mesh_data.get(key)
            return _compress_array(a) if a is not None and a.size > 0 else None
        cur.execute(f"DELETE FROM swe2d_mesh WHERE mesh_name = ?", (mesh_name,))
        cur.execute("""
            INSERT INTO swe2d_mesh(mesh_name, created_utc, nnodes, ncells, crs_wkt, hash,
                node_x, node_y, node_z, cell_nodes,
                face_offsets, face_nodes,
                bc_n0, bc_n1, bc_type, bc_val,
                description)
            VALUES(?,?,?,?,?,?,
                ?,?,?,?,
                ?,?,
                ?,?,?,?,
                ?)
        """, (
            mesh_name, datetime.datetime.utcnow().isoformat(),
            nnodes, ncells,
            str(crs_wkt or ""), h.hexdigest() if h else "",
            _b("node_x"), _b("node_y"), _b("node_z"), _b("cell_nodes"),
            _b("face_offsets"), _b("face_nodes"),
            _b("bc_n0"), _b("bc_n1"), _b("bc_type"), _b("bc_val"),
            str(description or ""),
        ))
        conn.commit()
        if log_fn:
            log_fn(f"Mesh '{mesh_name}' saved to {gpkg_path} ({nnodes} nodes, {ncells} cells)")
    finally:
        conn.close()
```

- [ ] **Step 2: Write `load_mesh_from_geopackage()`**

After `persist_mesh_to_geopackage()`, add:

```python
def load_mesh_from_geopackage(
    gpkg_path: str,
    mesh_name: str,
) -> Optional[Dict[str, np.ndarray]]:
    """Load mesh arrays from swe2d_mesh table. Returns None if not found."""
    if not gpkg_path:
        return None
    conn = sqlite3.connect(gpkg_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT node_x, node_y, node_z, cell_nodes, "
                     "face_offsets, face_nodes, "
                     "bc_n0, bc_n1, bc_type, bc_val "
                     "FROM swe2d_mesh WHERE mesh_name = ?", (mesh_name,))
        row = cur.fetchone()
        if row is None:
            return None
        def _ld(data, dtype):
            if data is None:
                return np.empty(0, dtype=dtype)
            return _decompress_array(data, dtype, (-1,))
        out = {
            "node_x": _ld(row[0], np.float64),
            "node_y": _ld(row[1], np.float64),
            "node_z": _ld(row[2], np.float64),
            "cell_nodes": _ld(row[3], np.int32).ravel(),
        }
        fo = _ld(row[4], np.int32) if row[4] else None
        if fo is not None and fo.size > 0:
            out["cell_face_offsets"] = fo
        bc_n0 = _ld(row[6], np.int32)
        bc_n1 = _ld(row[7], np.int32)
        bc_tp = _ld(row[8], np.int32)
        bc_vl = _ld(row[9], np.float64)
        if bc_n0.size > 0:
            out["bc_edge_node0"] = bc_n0
            out["bc_edge_node1"] = bc_n1
            out["bc_edge_type"] = bc_tp
            out["bc_edge_val"] = bc_vl
        return out
    finally:
        conn.close()
```

- [ ] **Step 3: Add function to `__all__` in services `__init__.py`**

The services `__init__.py` at `swe2d/workbench/services/__init__.py` is currently empty docstring-only.  No change needed — import directly from `gpkg_persistence_service`.

- [ ] **Step 4: Syntax check**

```bash
python3 -c "import py_compile; py_compile.compile('swe2d/workbench/services/gpkg_persistence_service.py', doraise=True); print('OK')"
```

Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add swe2d/workbench/services/gpkg_persistence_service.py
git commit -m "feat: add mesh GPKG persist/load functions with zlib BLOB compression"
```

### Task 3: QGIS UI — save mesh button

**Files:**
- Modify: `swe2d/workbench/controllers/mesh_controller.py` (or `studio_dialog.py`)

- [ ] **Step 1: Add save button and wire-up**

In `swe2d/workbench/studio_dialog.py`, locate the mesh controls section (around the "Generate Mesh" button area).  After the mesh generation logic completes successfully, add a "Save Mesh to GPKG..." button that:

1. Prompts for a mesh name via `QInputDialog.getText()`
2. Calls `backend.export_mesh_data()`
3. Calls `persist_mesh_to_geopackage(gpkg_path, name, mesh_data)`

Find the existing mesh button at studio_dialog.py — search for `_generate_mesh` or a button with text containing "generate" or "mesh".

- [ ] **Step 2: Add import + button in the mesh result area**

After the mesh is loaded into `self._mesh_data`, the save button becomes enabled.  Sample placement:

```python
# In the mesh result section after successful mesh:
from swe2d.workbench.services.gpkg_persistence_service import (
    persist_mesh_to_geopackage,
)

self._save_mesh_btn = QtWidgets.QPushButton("Save Mesh to GPKG...")
self._save_mesh_btn.setEnabled(True)
self._save_mesh_btn.clicked.connect(self._on_save_mesh_to_gpkg)
```

- [ ] **Step 3: Add handler method**

```python
def _on_save_mesh_to_gpkg(self):
    name, ok = QtWidgets.QInputDialog.getText(self, "Save Mesh", "Mesh name:")
    if not ok or not name.strip():
        return
    name = str(name).strip()
    backend = getattr(self, "_backend", None)
    if backend is None:
        QtWidgets.QMessageBox.warning(self, "Save Mesh", "No active backend.")
        return
    try:
        mesh_data = backend.export_mesh_data()
        gpkg = self._model_gpkg_path_edit.text()
        if not gpkg:
            from swe2d.workbench.services.gpkg_service import ensure_swe2d_tables
            conn = sqlite3.connect(self._model_gpkg_path)
            ensure_swe2d_tables(conn)
            conn.close()
            gpkg = self._model_gpkg_path
        persist_mesh_to_geopackage(gpkg, name, mesh_data, log_fn=self._log)
        QtWidgets.QMessageBox.information(self, "Save Mesh", f"Mesh '{name}' saved.")
    except Exception as exc:
        QtWidgets.QMessageBox.critical(self, "Save Mesh Error", str(exc))
```

- [ ] **Step 4: Commit**

```bash
git add swe2d/workbench/studio_dialog.py
git commit -m "feat: add Save Mesh to GPKG button with name prompt"
```

### Task 4: QGIS UI — load mesh from GPKG

**Files:**
- Modify: `swe2d/workbench/studio_dialog.py`

- [ ] **Step 1: Add mesh load combo box**

Add a "Load Mesh" section with a combo box populated from `SELECT mesh_name FROM swe2d_mesh` in the project GPKG.  A "Load" button reads the selected mesh and calls `build_mesh()`.

- [ ] **Step 2: Add handler**

```python
def _on_load_mesh_from_gpkg(self):
    name = self._mesh_load_combo.currentText()
    if not name:
        return
    from swe2d.workbench.services.gpkg_persistence_service import load_mesh_from_geopackage
    mesh_data = load_mesh_from_geopackage(self._model_gpkg_path, name)
    if mesh_data is None:
        QtWidgets.QMessageBox.warning(self, "Load Mesh", f"Mesh '{name}' not found.")
        return
    try:
        backend = SWE2DBackend()
        backend.build_mesh(**mesh_data)
        self._backend = backend
        self._mesh_data = mesh_data
        self._log(f"Mesh '{name}' loaded ({mesh_data['node_x'].size} nodes)")
    except Exception as exc:
        QtWidgets.QMessageBox.critical(self, "Load Mesh Error", str(exc))
```

- [ ] **Step 3: Refresh combo on project open**

When the model GPKG path changes, refresh the combo via `SELECT mesh_name FROM swe2d_mesh`.

- [ ] **Step 4: Commit**

```bash
git add swe2d/workbench/studio_dialog.py
git commit -m "feat: add Load Mesh from GPKG combo box with build_mesh()"
```

### Task 5: Test round-trip

**Files:**
- Create: `tests/test_mesh_persistence.py`

- [ ] **Step 1: Write round-trip test**

```python
"""Test mesh save/load round-trip to GPKG."""
import os
import tempfile
import numpy as np
from swe2d.workbench.services.gpkg_persistence_service import (
    persist_mesh_to_geopackage,
    load_mesh_from_geopackage,
)

def test_mesh_round_trip():
    mesh_data = {
        "node_x": np.array([0.0, 100.0, 50.0, 50.0], dtype=np.float64),
        "node_y": np.array([0.0, 0.0, 50.0, -50.0], dtype=np.float64),
        "node_z": np.array([10.0, 10.0, 8.0, 12.0], dtype=np.float64),
        "cell_nodes": np.array([0, 1, 2, 0, 3, 1], dtype=np.int32),
    }
    with tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False) as f:
        gpkg = f.name
    try:
        persist_mesh_to_geopackage(gpkg, "test_mesh", mesh_data)
        loaded = load_mesh_from_geopackage(gpkg, "test_mesh")
        assert loaded is not None
        for key in ("node_x", "node_y", "node_z", "cell_nodes"):
            assert key in loaded, f"Missing key: {key}"
            np.testing.assert_array_almost_equal(loaded[key], mesh_data[key])
    finally:
        if os.path.exists(gpkg):
            os.unlink(gpkg)
```

- [ ] **Step 2: Run test**

```bash
python3 -m pytest tests/test_mesh_persistence.py -v
```

Expected: `PASS`

- [ ] **Step 3: Commit**

```bash
git add tests/test_mesh_persistence.py
git commit -m "test: add mesh GPKG round-trip test"
```
