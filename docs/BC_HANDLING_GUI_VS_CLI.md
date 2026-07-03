# Boundary Condition Handling: GUI vs CLI Paths

**Date:** 2026-07-03  
**Scope:** How SWE2D collects, stores, and applies boundary conditions (BCs) in the QGIS workbench versus the headless CLI / batch runner.  
**Motivation:** Batch runs currently log `Boundary edge (665, 20) not found in mesh` and the results viewer logs `name 'sqlite3' is not defined`. This document explains why, and what the intended data flow should be.

---

## 1. Where BC arrays live

SWE2D uses four parallel arrays to describe a boundary condition on an edge:

| Array | Meaning |
|---|---|
| `bc_edge_node0` | First node index of the boundary edge |
| `bc_edge_node1` | Second node index of the boundary edge |
| `bc_edge_type`  | BC type enum (0=wall, 1=flow, 7=stage, etc.) |
| `bc_edge_val`   | BC value (e.g. unit discharge for flow BC) |

These arrays are passed to the C++ kernel through `SWE2DBackend.build_mesh(...)` and are **stored inside the native mesh handle**.  When the mesh handle is serialized with `swe2d_serialize_mesh()`, the BC arrays are part of the blob.  When the blob is deserialized, they come back as `pm.edge_n0`, `pm.edge_n1`, `pm.edge_bc`, `pm.edge_bc_val`.

The arrays are also exposed by the backend:

- `backend.get_mesh_data()` returns them if `_bc_n0.size > 0` (`backend.py:1224-1228`).
- `backend._boundary_edge_index_by_nodes` is a Python dict built from the native boundary-edge query after mesh build (`backend.py:381-395`).  It maps `(min(node0), max(node1))` → native edge index.  This is what `set_boundary_conditions()` validates against.

---

## 2. GUI run path

### 2.1 Collect mesh boundary edges

`RunController._execute_run()` → `run_data_builder.build()` → `collect_boundary_arrays_callback()` → `StudioDialog._collect_boundary_arrays()` → `boundary_runtime_logic.collect_boundary_arrays()`.

`collect_boundary_arrays()`:

1. Calls `mesh_boundary_edges_fn()` → `mesh_boundary_edges(mesh_data)` from `swe2d.boundary_and_forcing.boundary_runtime_logic`.
2. This walks the cell connectivity and returns every edge that belongs to exactly one cell (`edge_count[key] == 1`).
3. Computes default BCs for all boundary edges via `default_bc_for_edges()`.
4. Applies BC-layer overrides via `apply_bc_layer_overrides_fn()` → `boundary_qgis_adapter.apply_bc_layer_overrides_qgis()`.

### 2.2 Apply BC layer overrides

`apply_bc_layer_overrides_qgis()` (in `swe2d/boundary_and_forcing/boundary_qgis_adapter.py`):

- Reads the `swe2d_bc_lines` QgsVectorLayer selected in the Map tab.
- For **each mesh boundary edge**, tests whether that edge matches or intersects a BC line feature.
- If it matches, sets `bc_type[i]` and `bc_val[i]` from the feature attributes.
- Returns arrays whose length equals the number of mesh boundary edges.

Important: the GUI starts from the mesh boundary and assigns BCs to it. It does **not** snap arbitrary BC-line vertices to mesh nodes.

### 2.3 Build and bake the mesh

`RunController` calls `backend_initializer.build_and_initialize()` with the `bc_n0/bc_n1/bc_tp/bc_vl` arrays.

Inside `backend_initializer.py:108-117`:

```python
shared_build_mesh(
    b,
    node_x=node_x, node_y=node_y, node_z=node_z,
    cell_nodes=cell_nodes,
    cell_face_offsets=face_offsets,
    cell_face_nodes=face_nodes,
    bc_edge_node0=bc_n0, bc_edge_node1=bc_n1,
    bc_edge_type=bc_tp_init, bc_edge_val=bc_vl_init,
)
```

`shared_build_mesh` forwards them to `SWE2DBackend.build_mesh()` → native `swe2d_build_mesh_poly()`.

Then, if a `gpkg_path` was provided (`backend_initializer.py:120-134`):

```python
baked_blob = b._mod.swe2d_serialize_mesh(b._mesh_h)
persist_baked_mesh(gpkg_path, mesh_name, baked_blob, ...)
```

**Result:** the `swe2d_baked_mesh` entry in the results GPKG contains the mesh **plus** the BC arrays that were active for that run.

---

## 3. GUI "export mesh" paths

These do **not** create a `swe2d_baked_mesh` entry and therefore do **not** bake BCs into a GPKG blob.

| Button | File / service | Includes BCs? |
|---|---|---|
| Export Mesh To Map Layers | `mesh_controller.export_mesh_to_layers()` → `mesh_export_service.py` | No — creates `SWE2D_Mesh_Nodes` / `SWE2D_Mesh_Cells` vector layers only. |
| Export Mesh To UGRID | `mesh_controller.export_mesh_to_ugrid()` → `ugrid_export_service.write_ugrid_nc()` | No — writes NetCDF mesh geometry only. |
| Save Mesh to GPKG | `studio_dialog._save_mesh_to_gpkg()` | **Maybe** — see below. |

### 3.1 "Save Mesh to GPKG" details

Button object name: `save_mesh_gpkg_btn`. Wired in `swe2d/workbench/views/studio_tab_builder.py:105` to `dialog._save_mesh_to_gpkg()` (`swe2d/workbench/studio_dialog.py:634`).

What it does:

1. Reads `mesh_data` from the dialog.
2. Extracts `bc_edge_node0`, `bc_edge_node1`, `bc_edge_type`, `bc_edge_val` from `mesh_data` if present; otherwise uses empty arrays.
3. Calls native `swe2d_build_mesh` or `swe2d_build_mesh_poly` with those BC arrays.
4. Serializes the mesh handle with `swe2d_serialize_mesh()`.
5. Stores the blob via `persist_baked_mesh()`.

**Whether the saved blob contains BCs depends on whether `mesh_data` itself contains BC arrays.**

- If "Save Mesh to GPKG" is clicked **after a simulation run**, `mesh_data` has been populated with `bc_edge_*` from `backend.get_mesh_data()`, so the saved blob includes BCs.
- If it is clicked **before any run**, `mesh_data` typically has no `bc_edge_*` keys, so the saved blob has **no BCs**.

This is the source of ambiguity in the CLI: a `swe2d_baked_mesh` entry may or may not carry BCs, and the CLI cannot tell from the blob alone whether the absence of BCs was intentional.

So a GPKG `swe2d_baked_mesh` blob has BCs only when it was produced by a simulation run **or** by "Save Mesh to GPKG" after a run.

---

## 4. CLI / batch run path

### 4.1 Load mesh

`headless_runner.execute_run()` → `query_mesh_from_gpkg(mesh_gpkg, mesh_name)` in `swe2d/cli/gpkg_adapter.py`.

`query_mesh_from_gpkg` deserializes the baked blob and returns:

```python
{
    "node_x": ...,
    "node_y": ...,
    "node_z": ...,
    "cell_nodes": ...,
    "cell_face_offsets": ...,   # if polygon mesh
    "cell_face_nodes": ...,
    "bc_edge_node0": ...,       # from pm.edge_n0
    "bc_edge_node1": ...,       # from pm.edge_n1
    "bc_edge_type": ...,        # from pm.edge_bc
    "bc_edge_val": ...,         # from pm.edge_bc_val
}
```

These BC arrays match the baked mesh because they came from the same C++ mesh object.

### 4.2 Build backend

`headless_runner.py:185-194` passes the baked BC arrays to `backend.build_mesh(...)`:

```python
backend.build_mesh(
    node_x=mesh_data["node_x"],
    ...,
    bc_edge_node0=mesh_data.get("bc_edge_node0"),
    bc_edge_node1=mesh_data.get("bc_edge_node1"),
    bc_edge_type=mesh_data.get("bc_edge_type"),
    bc_edge_val=mesh_data.get("bc_edge_val"),
)
```

### 4.3 Apply `bc_lines` override (the problematic step)

After building the backend, the CLI opens the GPKG named in `p["bc_lines"]` and calls `query_bc_arrays()`:

- If the table has `node0`/`node1` columns, those node IDs are used directly.
- If the table is geometry-only, each BC line feature is split into vertex pairs, and `_find_nearest_node()` snaps the vertices to the nearest mesh node within a hard-coded `0.1` tolerance.

The resulting `bc_n0/bc_n1/bc_tp/bc_vl` arrays are then passed to `backend.set_boundary_conditions()`.

`set_boundary_conditions()` validates every `(node0, node1)` pair against `backend._boundary_edge_index_by_nodes`. If an edge is not a boundary edge of the **built mesh**, it raises:

```python
raise ValueError(f"Boundary edge ({a}, {b}) not found in mesh")
```

This exception is caught in `headless_runner.py:297-302` and logged as a warning, leaving the run with **no BCs applied**.

### 4.4 Why the mismatch happens

In the failing batch JSON:

```json
"mesh": "baked_test_20260630_135822",
"mesh_gpkg": "/home/aaron/QGIS_Plugins_dev/baked_test.gpkg",
"bc_lines": {
    "table": "swe2d_bc_lines",
    "gpkg": "/home/aaron/Desktop/realease1-1_test.gpkg"
}
```

- `baked_test.gpkg` was produced by a GUI run on (or exported from) one mesh.
- `realease1-1_test.gpkg` contains BC lines drawn for a different mesh version.
- The geometry coordinates do not snap to the same node IDs, so the override produces edges like `(665, 20)` that are not boundary edges of the baked mesh.

Diagnostic evidence:

- The baked mesh has `113,614` nodes and `226,083` cells.
- Its boundary edge count is `1,143`.
- Its serialized blob already contains `1,143` non-zero `edge_bc` entries (the BCs from the original GUI run).
- The `swe2d_bc_lines` table in `realease1-1_test.gpkg` has two features; one cannot snap any vertex within `0.1` units, and the other snaps to edge `(25, 24)` which is not in the baked-mesh boundary set.

---

## 5. Correct CLI behavior

The CLI should treat the baked mesh's BC arrays as authoritative.  The `bc_lines` entry in the batch JSON should only be used when it is guaranteed to describe the **same mesh** (e.g. the `bc_lines` GPKG is the same GPKG that produced the baked mesh, and the BC table was built against that exact node numbering).

Recommended rules:

1. **Primary source:** use `bc_edge_node0/1/type/val` from the baked mesh blob.
2. **Override source:** `bc_lines` table may override, but only after validating that every edge it produces exists in `backend._boundary_edge_index_by_nodes`.
3. **On mismatch:** fall back to the baked BC arrays and emit a clear warning naming the mismatched edges.
4. **Future improvement:** support re-computing BCs from a BC layer using the same mesh-boundary-first logic as the GUI (`apply_bc_layer_overrides_qgis`), rather than vertex-to-node snapping.

---

## 6. Overlay viewer issue (separate)

When opening batch results in the results viewer, `OverlayController` reads `swe2d_baked_mesh` from the results GPKG with `sqlite3.connect(...)` but did not import `sqlite3`.  This was fixed by adding `import sqlite3` to `swe2d/workbench/controllers/overlay_controller.py`.

---

## 7. Files involved

| File | Role |
|---|---|
| `swe2d/boundary_and_forcing/boundary_runtime_logic.py` | Computes mesh boundary edges and default BC arrays (GUI path). |
| `swe2d/boundary_and_forcing/boundary_qgis_adapter.py` | Applies BC layer overrides to mesh boundary edges (GUI path). |
| `swe2d/runtime/backend.py` | `build_mesh()`, `set_boundary_conditions()`, `get_mesh_data()`. |
| `swe2d/runtime/backend_initializer.py` | Builds backend and persists baked mesh blob during a GUI run. |
| `swe2d/services/gpkg_persistence_service.py` | `persist_baked_mesh()`, `load_baked_mesh()`. |
| `swe2d/cli/gpkg_adapter.py` | `query_mesh_from_gpkg()`, `query_bc_arrays()` (CLI path). |
| `swe2d/cli/headless_runner.py` | Loads mesh, optionally applies `bc_lines`, runs simulation. |
| `swe2d/workbench/studio_dialog.py` | `_save_mesh_to_gpkg()` — "Save Mesh to GPKG" button handler. |
| `swe2d/workbench/controllers/overlay_controller.py` | Reads baked mesh from results GPKG for rendering. |

---

## 8. Open questions

1. Should the batch snapshot JSON store a flag indicating whether `bc_lines` should override baked BCs, or should it always prefer the baked BCs?
2. If a user wants to change BCs for a batch run, should the CLI re-run the GUI's mesh-boundary-first override logic (`apply_bc_layer_overrides_qgis`) instead of vertex snapping?
3. Should `query_mesh_from_gpkg` continue to expose `bc_edge_*` arrays, or should the CLI always re-derive BCs from a source table to avoid surprising behavior?
4. Does the planned "Export Mesh to GeoPackage" widget (if different from Export to Layers) intend to write a `swe2d_baked_mesh` entry, and if so, should it include BCs?
