---
type: plan
status: complete
created: 2026-07-10
completed: 2026-07-25
---

# RCMK Ordering Elimination Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate all pre-RCMK vs post-RCMK ordering mismatches by ensuring `view._mesh_data` is always populated from the post-RCMK C++ mesh handle, making `apply_cell_permutation` dead code.

**Architecture:** Currently `backend.build_mesh()` stores pre-RCMK input arrays in `self._mesh_*` while the C++ handle (`self._mesh_h`) has RCMK-applied arrays. The fix: always read mesh data from the C++ handle after build, never from the input. This makes `apply_cell_permutation` unnecessary and eliminates the quad-permutation bug permanently. Loaded/GPKG meshes already work this way via `build_mesh_from_baked`.

**Tech Stack:** Python (swe2d/runtime/backend.py, workbench/services/mesh_service.py, controllers), C++ pybind11 property accessors

---

### Task 1: Store post-RCMK arrays in `backend.build_mesh()`

**Files:**
- Modify: `swe2d/runtime/backend.py:340-405`

After the C++ mesh handle is built (lines 342-352/363-365), overwrite `self._mesh_*` with arrays read from the C++ handle's property accessors, matching what `build_mesh_from_baked()` already does at lines 441-452.

Add extraction of `cell_cx`, `cell_cy` from the handle — these are exposed as PyMesh properties but never stored by `build_mesh()`.

Change lines 399-405 from:
```python
self._mesh_node_x = np.asarray(node_x, dtype=np.float64)
self._mesh_node_y = np.asarray(node_y, dtype=np.float64)
self._mesh_node_z = np.asarray(node_z, dtype=np.float64)
self._mesh_cell_nodes = np.asarray(cell_nodes_flat, dtype=np.int32)
self._mesh_face_offsets = None
if cell_face_offsets is not None:
    self._mesh_face_offsets = np.asarray(face_offsets, dtype=np.int32)
```

To:
```python
pm = self._mesh_h
self._mesh_node_x = np.asarray(pm.node_x, dtype=np.float64)
self._mesh_node_y = np.asarray(pm.node_y, dtype=np.float64)
self._mesh_node_z = np.asarray(pm.node_z, dtype=np.float64)
self._mesh_cell_cx = np.asarray(pm.cell_cx, dtype=np.float64)
self._mesh_cell_cy = np.asarray(pm.cell_cy, dtype=np.float64)
self._cell_zb = np.asarray(pm.cell_zb, dtype=np.float64)
self._cell_area = np.asarray(pm.cell_area, dtype=np.float64)
cfn = pm.cell_face_nodes
self._mesh_cell_nodes = np.asarray(cfn, dtype=np.int32) if cfn is not None else np.empty(0, dtype=np.int32)
cfo = pm.cell_face_offsets
self._mesh_face_offsets = np.asarray(cfo, dtype=np.int32) if cfo is not None else None
```

Remove the pre-C++-call area/zb Python computation at lines 330-341 and 354-362 since the C++ handle now provides authoritative values.

### Task 2: Expose `_mesh_cell_cx` and `_mesh_cell_cy` in backend

**Files:**
- Modify: `swe2d/runtime/backend.py`

Add getters similar to `cell_areas()`:

```python
@property
def cell_cx(self) -> np.ndarray:
    return self._mesh_cell_cx.copy() if hasattr(self, '_mesh_cell_cx') and self._mesh_cell_cx.size > 0 else self._cell_centroids()[0]

@property
def cell_cy(self) -> np.ndarray:
    return self._mesh_cell_cy.copy() if hasattr(self, '_mesh_cell_cy') and self._mesh_cell_cy.size > 0 else self._cell_centroids()[1]
```

Update `build_mesh_from_baked()` (lines 442-452) to also store `_mesh_cell_cx` and `_mesh_cell_cy` from `pm.cell_cx` / `pm.cell_cy`.

### Task 3: Update `export_mesh_data()` to include all post-RCMK arrays

**Files:**
- Modify: `swe2d/runtime/backend.py:1318-1333`

Add `cell_cx`, `cell_cy`, `cell_area`, `cell_zb` to the exported dict. These are now all in RCMK order (from the C++ handle).

```python
def export_mesh_data(self) -> Dict[str, np.ndarray]:
    out = {
        "node_x": self._mesh_node_x.copy(),
        "node_y": self._mesh_node_y.copy(),
        "node_z": self._mesh_node_z.copy(),
        "cell_cx": self._mesh_cell_cx.copy(),
        "cell_cy": self._mesh_cell_cy.copy(),
        "cell_area": self._cell_area.copy(),
        "cell_zb": self._cell_zb.copy(),
        "cell_face_offsets": self._mesh_face_offsets.copy() if self._mesh_face_offsets is not None else np.empty(0, dtype=np.int32),
        "cell_face_nodes": self._mesh_cell_nodes.copy(),
    }
    ...
```

### Task 4: Populate `view._mesh_data` from backend after build

**Files:**
- Modify: `swe2d/workbench/workers/simulation_worker.py:487-506`
- Modify: `swe2d/workbench/controllers/run_controller.py:344-355`

Currently the worker emits `mesh_permutation_ready` signal, and the controller calls `apply_cell_permutation(view._mesh_data, perm)`. Instead:

- **In the worker** (`simulation_worker.py`): After backend build completes, call `backend.export_mesh_data()` and emit a signal carrying the post-RCMK mesh data dict (instead of just the perm array).
- **In the controller** (`run_controller.py`): On receiving the signal, overwrite `view._mesh_data` with the post-RCMK dict. No permutation needed.

Signal/function rename:
- Old: `mesh_permutation_ready.emit(cell_perm, result_holder)`
- New: `mesh_data_ready.emit(mesh_data, result_holder)` where `mesh_data` is the dict from `export_mesh_data()`

### Task 5: Delete `apply_cell_permutation`

**Files:**
- Remove: `swe2d/workbench/services/mesh_service.py:110-153` (the `apply_cell_permutation` function)
- Remove: All imports and callers of `apply_cell_permutation`

Search for all references to `apply_cell_permutation`:
- `mesh_service.py` (definition)
- `simulation_worker.py` (direct call at line 506)
- `run_controller.py` (signal handler at line 344-355)

### Task 6: Fix overlay live path to use post-RCMK centroids

**Files:**
- Modify: `swe2d/workbench/controllers/overlay_controller.py:165-211`

The live path currently calls `view._mesh_cell_centroids()` which computes centroids from `_mesh_data` using Python mesh_runtime_logic functions. Since `_mesh_data` is now post-RCMK, these centroids will be in the correct order.

No code change needed if `_mesh_data["cell_cx"]` and `_mesh_data["cell_cy"]` are now populated from the C++ handle. But verify that `view._mesh_cell_centroids()` reads from the dict, not from the original MeshResult.

### Task 7: Verify and clean up

**Files:**
- Test: `tests/` (all GPU tests + any mesh permutation tests)

- [ ] Run the full GPU test suite to confirm no regressions

```bash
cd /path/to/repo
python -m pytest tests/test_swe2d_gpu_unstructured.py tests/test_swe2d_gpu_lake_at_rest_immersed_bump.py tests/test_swe2d_gpu_lake_at_rest_steep_island.py tests/test_swe2d_gpu_rain_volume_conservation.py tests/test_swe2d_imex_subcycling.py tests/test_swe2d_gpu_river_at_rest_varying_topo_width.py -v -q
```

- [ ] Search for any remaining references to `apply_cell_permutation` and remove

```bash
git grep -n "apply_cell_permutation"
```
