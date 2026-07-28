---
type: spec
status: complete
created: 2026-07-18
completed: 2026-07-25
---

# GeoPackage Explorer Enhanced Viewer — Design Spec

## Overview

Transform the GeoPackage Explorer from a table-listing tool into a full-featured
results viewer with inline blob deserialization, structured filtering, custom
XY plotting, and CSV export.

## Architecture

```
SWE2DModelGeoPackageExplorerDialog (QDialog)
├── QTabWidget
│   ├── "Tables" tab — existing table list (unchanged)
│   └── "Plot" tab   — XY plotting from current table's data
├── [Open Viewer] button → opens SWE2DEnhancedTablePreviewDialog
├── [Plot] button     → switches to Plot tab with logged data
└── CSV export on Plot tab

SWE2DEnhancedTablePreviewDialog (QDialog) — replaces SWE2DSQLiteTablePreviewDialog
├── Filter bar: [column ▼] [operator ▼] [value] [Apply] [Clear]
├── Metadata table (top panel, filterable)
│   └── Blob cells show clickable "View [n×m] float64"
├── Array viewer (bottom panel, updates on blob cell click)
│   ├── Tab "Data"     — QTableWidget of deserialized values
│   └── Tab "Plot"     — inline matplotlib figure of array vs index
├── [Export CSV]      — saves filtered metadata rows
└── [Send to Plot]    — pushes selected columns to explorer's Plot tab

numpy_blob_service.py (pure Python, zero Qt)
├── deserialize_blob_to_array() — reads blob, casts to np.float64
├── discover_plottable_columns() — returns shape metadata per column
├── build_where_clause()         — safe SQL WHERE from structured filter
└── export_table_to_csv()        — writes filtered rows to .csv file
```

### Layer boundaries

| Layer | Files | Responsibility |
|---|---|---|
| View | `gpkg_explorer_dialog.py`, new plot widget, enhanced preview dialog | Qt widgets, signals, layout |
| Service | `numpy_blob_service.py`, `gpkg_operations_service.py` (amend) | Blob deserialization, CSV export, WHERE clause building — no Qt imports |
| Controller | None new — explorer is standalone QDialog with direct service calls (existing pattern) |

## Data Flow

### Table preview flow
1. User selects table in explorer → clicks "Open Viewer"
2. Explorer instantiates `SWE2DEnhancedTablePreviewDialog(gpkg_path, table_name)`
3. Dialog queries metadata columns via `PRAGMA table_info`
4. Dialog calls `numpy_blob_service.deserialize_blob_to_array()` on-demand when user clicks a blob cell
5. Bottom panel shows array viewer (QTableWidget for numeric data + optional quick-plot)

### Filter flow
1. User picks column from dropdown, operator from dropdown, enters value
2. Dialog calls `numpy_blob_service.build_where_clause(column, op, value)` → safe SQL fragment
3. Re-runs: `SELECT * FROM table WHERE {clause} LIMIT {limit}`
4. Repopulates metadata table, clears array viewer

### Plot flow
1. User selects table in explorer → navigates to "Plot" tab
2. Plot tab calls `numpy_blob_service.discover_plottable_columns()` → populates X/Y dropdowns
3. User selects X column, Y column, optional slice index for 2D arrays
4. Plot tab extracts arrays, renders matplotlib scatter/line plot
5. User can toggle log scale, export PNG, export CSV of plotted data

## SWE2DEnhancedTablePreviewDialog — State Machine

```
┌──────────┐  init   ┌──────────────┐  click blob cell  ┌──────────────┐
│  Closed  │ ──────→ │  Data Loaded │ ────────────────→ │ Array Shown  │
└──────────┘         │  (no filter) │ ←──────────────── └──────────────┘
                     │  top table   │   click another    │ bottom panel │
                     │  populated   │   blob cell        │ has data+plot│
                     └──────┬───────┘                    └──────────────┘
                            │ Apply filter                  │
                            ▼                               │
                     ┌──────────────┐                       │
                     │ Filter Active│ ──────────────────────┘
                     │ rows filtered│  Clear filter
                     │ top table    │
                     └──────────────┘
```

## Blob Deserialization Rules

The service layer knows how to interpret known SWE2D blob schemas:

### swe2d_baked_results
```
times_blob → np.float64[n_timesteps]  — 1D time array
h_blob     → np.float64[n_timesteps × n_cells] — 2D water depth
hu_blob    → np.float64[n_timesteps × n_cells] — 2D x-momentum
hv_blob    → np.float64[n_timesteps × n_cells] — 2D y-momentum
max_h_blob → np.float64[n_cells] — 1D max depth
max_hu_blob, max_hv_blob similarly
```

### swe2d_baked_line_ts
```
times_blob  → np.float64[n_timesteps]
depth_blob  → np.float64[n_timesteps]
vel_blob    → np.float64[n_timesteps]
wse_blob    → np.float64[n_timesteps]
bed_blob    → np.float64[n_timesteps]
flow_blob   → np.float64[n_timesteps]
wet_frac_blob → np.float64[n_timesteps]
fr_blob     → np.float64[n_timesteps]
```

### swe2d_baked_line_profiles
```
station_blob → np.float64[n_stations]
times_blob   → np.float64[n_timesteps]
depth_blob   → np.float64[n_stations × n_timesteps]
vel_blob     → np.float64[n_stations × n_timesteps]
wse_blob     → np.float64[n_stations × n_timesteps]
etc. (2D: time × station)
```

### swe2d_baked_coupling / swe2d_baked_pipe_cell_ts / swe2d_baked_overlay_fields
```
times_blob  → np.float64[n_timesteps]
values_blob → np.float64[n_timesteps]
```

For unknown BLOB columns: show raw byte count.

For 2D arrays: the array viewer provides spin boxes to select which slice
(timestep index or cell index) to view/plot.

## Structured Filter Definition

The filter bar uses three UI elements:

1. **Column dropdown** — populated from non-blob columns + deserializable blob columns
2. **Operator dropdown**: `=`, `!=`, `>`, `<`, `>=`, `<=`, `LIKE`, `IN`, `BETWEEN`, `IS NULL`, `IS NOT NULL`
3. **Value input** — `QLineEdit` (text mode) or `QDoubleSpinBox` (numeric mode), changes depending on column type

When the operator is `IS NULL` or `IS NOT NULL`, the value input is hidden.

`build_where_clause()` produces a parameterized SQL fragment using `?` placeholders
to prevent injection. Example:

```python
# Input: column="n_cells", op=">", value="5000"
# Output: ('"n_cells" > ?', [5000])
```

## CSV Export

### Preview dialog CSV export
- Exports the currently visible (filtered) metadata rows
- Non-blob columns written directly
- Blob columns: write metadata summary string (e.g., `"float64[500×200]"`)
- File save dialog: `QFileDialog.getSaveFileName()` with `.csv` filter

### Plot tab CSV export
- Exports the plotted X and Y data arrays as two columns
- Header row with column names
- File save dialog: `QFileDialog.getSaveFileName()` with `.csv` filter

### Implementation
- Service layer: `export_table_to_csv(gpkg_path, table, filepath, where_clause=None)`
- Uses Python `csv` module, no pandas dependency
- View layer handles `QFileDialog`, calls service for actual write

## Files Changed

| File | Type | Change |
|---|---|---|
| `swe2d/workbench/dialogs/gpkg_explorer_dialog.py` | Modify | Wrap content in QTabWidget; add "Plot" tab; add [Plot] and [Export CSV] buttons |
| `swe2d/workbench/dialogs/sqlite_preview_dialog.py` | Rewrite | Replace with `SWE2DEnhancedTablePreviewDialog` (dual-panel, filter, array viewer) |
| `swe2d/workbench/services/numpy_blob_service.py` | **New** | Blob deserialization, column discovery, WHERE builder, CSV export |
| `swe2d/workbench/dialogs/gpkg_plot_tab.py` | **New** | Plot tab widget (matplotlib canvas, column selectors, slice controls) |
| `swe2d/workbench/dialogs/gpkg_array_viewer_widget.py` | **New** | Reusable array data viewer widget (table + mini-plot) |
| `swe2d/workbench/services/gpkg_operations_service.py` | Amend | Add `export_table_to_csv()` |
| `tests/test_numpy_blob_service.py` | **New** | Tests for blob deserialization, WHERE builder, CSV export |
| `tests/test_gpkg_explorer_dialog.py` | Amend | Tests for new tabs, plot tab, enhanced preview dialog |

## Test Plan

| Test | What it verifies |
|---|---|
| `test_blob_deserialize_1d` | times_blob → correct shape and dtype |
| `test_blob_deserialize_2d` | h_blob → correct 2D shape |
| `test_blob_deserialize_unknown` | Unknown blob → None |
| `test_where_clause_numeric` | `build_where_clause("n_cells", ">", 5000)` → safe SQL |
| `test_where_clause_injection` | Value with `' OR 1=1` → parameterized, not injected |
| `test_where_clause_is_null` | No value parameter emitted |
| `test_discover_plottable` | Returns correct column names for swe2d_baked_results |
| `test_csv_export` | Exported CSV matches filtered rows |
| `test_enhanced_preview_dialog_import` | Dialog imports without QApplication |
| `test_plot_tab_import` | Plot widget imports without crash |

Existing tests (`test_gpkg_operations.py`, `test_workbench_gpkg_service.py`,
`test_results_path_audit_fixes.py`) must pass unchanged.

## Open Questions / Future

- Real-time array slicing for 2D data (slider for timestep)
- Overlay multiple run results on the same plot
- Statistics panel (min, max, mean, std of selected column)
- Histogram tab
