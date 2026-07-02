# CLI/GPKG Adapter Audit — 2026-07-02

## Headless Runner (`swe2d/cli/headless_runner.py`)

### 1. CRASH — `line_output_interval` used before assignment
**Severity: CRITICAL**
**Lines: ~314, ~342**

The entire sample-line setup block is unreachable because `line_output_interval` is referenced on line ~314 but not assigned until line ~342 (after the block). Every execution raises `NameError`.

```python
# Line ~314 — UNREACHABLE
if line_output_interval > 0:
    sl_cfg = p.get("sample_lines")
    ...

# Line ~342 — actual first assignment
line_output_interval = float(rp.get("line_output_interval_s", t_end))
```

### 2. Architecture violation — imports from `swe2d.workbench` in headless code
**Severity: HIGH**
**Lines: ~321-324, ~506-509**

```python
from swe2d.workbench.services.mesh_service import (
    build_line_sampling_map,
    sample_line_metrics,
    sample_line_aggregate_ts_row,
)
```

Per the repo's MVP architecture rules, CLI must not import from `swe2d/workbench/`. While `mesh_service` itself does not import PyQt5, it pulls the full workbench service graph into the headless environment. These functions should be moved to a shared location (e.g., `swe2d/services/mesh_service.py` or a new `swe2d/mesh/mesh_sampling.py`).

### 3. Silent corruption — `_u.configure()` mutates global unit state
**Severity: MEDIUM**
**Lines: ~112-118**

```python
_u.configure(si_m_per_model)
```

`swe2d.units` uses module-level globals (`_gravity`, `_manning`, etc.). If multiple simulations run in the same Python process (batch), the unit system is permanently mutated by the first run. A second run with a different CRS inherits the first run's config. No teardown or context manager exists.

**Fix**: Add `swe2d.units.reset_defaults()` or wrap in a context manager, or accept that headless runs are single-shot per process.

### 4. Crash risk — `_collect_coupling_snapshot` accesses private attrs without `hasattr`
**Severity: HIGH**
**Lines: ~382-411**

```python
def _collect_coupling_snapshot(cc, t_s: float) -> None:
    if cc is None:
        return
    if hasattr(cc, "_gpu_node_depth") and cc._gpu_node_depth is not None:
        ...
        for i, node in enumerate(getattr(cfg, "nodes", [])):
            if i >= len(cc._gpu_node_depth):  # ← accesses private attr directly
                continue
```

After the `hasattr` check, the code still accesses `_gpu_node_depth` and `_gpu_link_flow` via `hasattr` checks — this part is guarded. However, `_last_structure_flows` and `_structures_cfg` are accessed directly without any `hasattr` guard:

```python
nb_flows = getattr(cc, "_last_structure_flows", None)
structures_cfg_local = getattr(cc, "_structures_cfg", ())
```

If these attrs are absent, `getattr` returns the default (correct), so this is fine.

### 5. Minor — `wet_frac` naming confusion
**Severity: INFO**

`wet_frac` (fraction 0.0-1.0) is used for scalar TS, `wet` (int32) for profile arrays. Not a bug but the names are inconsistent with `persist_baked_line_ts` (`wet_frac`) vs `persist_baked_line_profile` (`wet`).

---

## GPKG Adapter (`swe2d/cli/gpkg_adapter.py`)

### 6. Silent no-op — WKT prefix check on binary GPKG WKB geometry
**Severity: HIGH**
**Lines: ~449-452** (in `apply_bc_overrides_from_gpkg`)

```python
raw = str(row[0] or "")
if not raw.startswith("LINESTRING") and not raw.startswith("LineString"):
    continue
```

GPKG stores geometry as binary WKB in BLOB columns. `str(wkb_bytes)` produces `"b'\\x01\\x02...'"` — it never starts with "LINESTRING". The geometry-based fallback silently does nothing for all GPKG sources. Only the pre-split edge table path (node0/node1 columns) works.

**Contrast**: `query_bc_arrays` correctly uses `_parse_wkb_linestring()` for binary geometry.

### 7. Dead code — `n_cells` parameter unused in adapter functions
**Severity: LOW**
**Lines: ~589, ~726** (`build_drainage_config_from_json`, `build_structures_config_from_json`)

Both functions accept `n_cells: int` but never use it. Creates maintenance hazard and false implication of cell-count dependency.

### 8. Dead code — unused column sets from PRAGMA queries
**Severity: LOW**
**Lines: ~933-936, ~1001-1002, ~1017-1018**

```python
nodes_cur = conn.execute(f'PRAGMA table_info("{nodes_table}")').fetchall()
node_cols = {r[1] for r in nodes_cur}  # never read
links_cur = conn.execute(f'PRAGMA table_info("{links_table}")').fetchall()
link_cols = {r[1] for r in links_cur}  # never read
```

### 9. Style — redundant local `import sqlite3`
**Severity: TRIVIAL**
**Line: ~75**

`sqlite3` is already imported at the top of the file (line ~11). Local re-import is unnecessary.

### 10. Logic bug — triangle-only centroid reshape breaks mixed/quad meshes
**Severity: HIGH**
**Line: ~911** (`_compute_cell_centroids`)

```python
tris = cell_nodes.reshape((-1, 3))
cx = np.mean(node_x[tris], axis=1)
```

Assumes all cells are triangles. For quads or mixed meshes, `reshape((-1, 3))` raises `ValueError`. Used in `build_forced_thiessen_from_gpkg` (line ~879) and `read_drainage_config_from_gpkg` (line ~974).

**Fix**: Use `cell_face_offsets` to handle variable node counts per cell, or fall back to the pure-triangle path when `cell_face_offsets` is available (which the caller always has).

### 11. Minor — redundant `node_x_map`/`node_y_map` dictionaries
**Severity: TRIVIAL**
**Lines: ~938-946**

Maps are populated but the same data is available in `nodes_raw`. Not harmful, just redundant.

---

## Batch Simulation Dialog (`swe2d/workbench/dialogs/batch_simulation_dialog.py`)

### 12. Crash risk — `parent._build_hydraulic_structure_config()` unverified
**Severity: HIGH**
**Line: ~608**

```python
struct_cfg = parent._build_hydraulic_structure_config()
if struct_cfg is not None:
    structures_cfg = struct_cfg.to_dict()
```

No existence check on `_build_hydraulic_structure_config`. If the parent dialog lacks this method, `AttributeError` propagates into the Qt event loop. The code does have a `try/except` around `collect_fn()` (line ~499), but not around this call.

### 13. Logging bug — f-string in `logger.warning()` instead of format string
**Severity: LOW**
**Lines: ~531, headless_runner.py ~72**

```python
logger.warning(f"[ERROR] Exception in batch_simulation_dialog.py: {_e}")
```

Should be `logger.warning("[ERROR] Exception in batch_simulation_dialog.py: %s", _e)`. The f-string works but bypasses lazy evaluation.

### 14. CRASH — `None.currentData()` if combo attr is None
**Severity: CRITICAL**
**Line: ~618**

```python
im = str(getattr(mtab, "infiltration_method_combo", None).currentData() or "none")
```

`getattr(mtab, "infiltration_method_combo", None)` returns the default `None` if the attribute doesn't exist. Then `.currentData()` on `None` raises `AttributeError`.

**Fix**:
```python
combo = getattr(mtab, "infiltration_method_combo", None)
im = str(combo.currentData() or "none") if combo else "none"
```

### 15. Data loss risk — `params.pop` without `deepcopy` in `__main__.py` batch path
**Severity: MEDIUM**

In `batch_runner.py` line ~72, `sweep = params.pop("sweep", None)` mutates the caller's dict. If the same dict is reused for re-runs, the sweep key is permanently missing. The `batch_runner.py` path correctly uses `deepcopy`, but the `__main__.py` batch subcommand path passes the raw loaded dict.

---

## Summary Table

| # | Severity | Status | File | Issue |
|---|----------|--------|------|-------|
| 1 | **CRITICAL** | **FIXED** | headless_runner.py | `line_output_interval` used before assignment — unreachable block |
| 4 | **CRITICAL** | **FIXED** | batch_simulation_dialog.py | `None.currentData()` if combo attr is None |
| 2 | **HIGH** | OPEN | headless_runner.py | Imports from `swe2d.workbench` in headless code |
| 4 | **HIGH** | OPEN | gpkg_adapter.py | Triangle-only centroid reshape fails on quad/mixed meshes |
| 6 | **HIGH** | OPEN | gpkg_adapter.py | WKT prefix check on binary GPKG WKB — always silent no-op |
| 12 | **HIGH** | OPEN | batch_simulation_dialog.py | `parent._build_hydraulic_structure_config()` unverified call |
| 3 | **MEDIUM** | OPEN | headless_runner.py | `_u.configure()` global state mutation, no reset |
| 15 | **MEDIUM** | OPEN | batch_runner.py | `params.pop` without `deepcopy` in __main__ batch path |
| 5 | INFO | — | headless_runner.py | `wet_frac` naming confusion (not a bug) |
| 7 | LOW | OPEN | gpkg_adapter.py | `n_cells` parameter unused |
| 8 | LOW | OPEN | gpkg_adapter.py | Unused `node_cols`, `link_cols`, `it_cols`, `ni_cols` |
| 9 | TRIVIAL | OPEN | gpkg_adapter.py | Redundant local `import sqlite3` |
| 11 | TRIVIAL | OPEN | gpkg_adapter.py | Redundant `node_x_map`/`node_y_map` dicts |
| 13 | LOW | **FIXED** | batch_sim_dialog.py, headless_runner.py | f-string in `logger.warning()` |
| 4b | **HIGH** | **FIXED** | headless_runner.py | USC gravity not propagated: coupling controller defaulted to SI, drainage config hardcoded 9.81 |

## No Issues Found

- No `from qgis import` patterns (all correctly use `from qgis.core import`)
- No `shapely` imports in CLI code
- No `from qgis.PyQt` imports in CLI code
- All `QgsVectorLayer` usage correctly imports from `qgis.core`

## Fixes Applied 2026-07-02

1. **`headless_runner.py`**: Moved `t_end`, `output_interval`, `line_output_interval` to before the sample-line setup block. The block at line ~312 was unreachable because `line_output_interval` was referenced before assignment.

2. **`batch_simulation_dialog.py`**: Changed `getattr(..., None).currentData()` to `combo.currentData() or "none" if combo else "none"` with separate `getattr` call and null check.

3. **`headless_runner.py`**: Changed `logger.warning(f"[ERROR] ...{_e}")` to `logger.warning("[ERROR] ...", _e)` format style.
