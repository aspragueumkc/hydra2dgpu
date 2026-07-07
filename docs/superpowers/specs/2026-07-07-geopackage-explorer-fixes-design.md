# Geopackage Explorer Fixes — Design Spec

## 1. Blob fields rendered as human-readable text

### Problem
`SWE2DSQLiteTablePreviewDialog` (sqlite_preview_dialog.py:71) renders every cell as `str(val)`. BLOB fields show raw bytes like `b'\x00\x01...'` or `<memory at 0x...>`.

### Solution
Add a `_format_blob()` helper in `sqlite_preview_dialog.py` called from the cell-rendering loop. It inspects the table name and sibling column values to produce:

| Table | Column(s) | Display |
|-------|-----------|---------|
| `swe2d_baked_mesh` | `baked_blob` | `"Binary mesh (N nodes, M cells, E edges)"` |
| `swe2d_baked_results` | `h_blob`, `hu_blob`, `hv_blob` | `"float64[T×M]"` (T=n_timesteps, M=n_cells) |
| `swe2d_baked_results` | `times_blob` | `"float64[T]"` |
| any | any other blob | `"binary N bytes"` |

The table info (`get_table_info`) returns ordered column names. Sibling metadata columns (`n_nodes`, `n_cells`, `n_timesteps`, `n_edges`) are found by name and their integer values read from the row tuple by index.

### Files changed
- `swe2d/workbench/dialogs/sqlite_preview_dialog.py` — single `_format_blob()` helper + call site change.

---

## 2. Empty widget config fix

### Problem
`collect_run_widget_params()` returns keys that are attribute names on sub-views (e.g. `n_mann_spin` → `ModelTabView.n_mann_spin`). The caller passes these keys to `collect_workbench_widget_state(ui=view, ...)` where `view` is the dialog, not the sub-view. `getattr(dialog, "n_mann_spin", None)` returns `None` → silently skipped → `{"widgets": {}}`.

### Root cause locations
- `run_controller.py:891-896` — `on_save_simulation_config`
- `finalization_adapter.py:63-66` — `collect_run_log_metadata`

### Fix

In both sites:
1. Pass `view._model_tab_view` as the `ui` argument (all widget attrs live on ModelTabView).
2. Exclude non-widget keys `"gravity"` and `"k_mann"` from the attrs list.

```python
all_attrs = list(view.collect_run_widget_params().keys())
widget_attrs = [k for k in all_attrs if k not in ("gravity", "k_mann")]
widget_state = collect_workbench_widget_state(
    ui=view._model_tab_view,
    widget_attrs=widget_attrs,
    qtwidgets_module=QtWidgets,
)
```

### Files changed
- `swe2d/workbench/controllers/run_controller.py` — fix call site.
- `swe2d/workbench/controllers/finalization_adapter.py` — fix call site.

---

## 3. Dedicated model-config viewer

### Problem
`swe2d_simulation_configs` shows `widget_state` as raw JSON string in the generic preview dialog. No way to see parsed widget values.

### Solution

**New dialog**: `SWE2DSimulationConfigViewerDialog` in `swe2d/workbench/dialogs/simulation_config_viewer_dialog.py`.

Layout:
- Top: `QComboBox` listing all config rows (by `config_id` + date).
- Middle: metadata labels (mesh_name, duration, created).
- Bottom: `QTableWidget` with columns `[Widget Name, Value, Type]` — populated by parsing the selected config's `widget_state` JSON.

**Explorer routing** (`gpkg_explorer_dialog.py`):
- `_table_kind()`: add `"config"` category for `swe2d_simulation_configs`.
- `open_selected()`: dispatch `"config"` kind to the new viewer dialog.

### Files changed
- `swe2d/workbench/dialogs/simulation_config_viewer_dialog.py` — new file.
- `swe2d/workbench/dialogs/gpkg_explorer_dialog.py` — add import, `_table_kind` update, dispatch.

---

## Verification

```bash
# 1. Blob display — launch explorer, preview baked_mesh/baked_results, verify blobs show metadata
# 2. Widget config — save a config, inspect swe2d_simulation_configs, widget_state has widget entries
# 3. Config viewer — open swe2d_simulation_configs from explorer, see key-value table
```
