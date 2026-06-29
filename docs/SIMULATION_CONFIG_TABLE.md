# Simulation Configuration Table

**Date:** 2026-06-29
**Status:** Implemented
**Motivation:** Avoid re-configuring all UI widgets (BC layers, hyetographs, solver knobs) every time a mesh is loaded. Persist the full run configuration in the GPKG so a mesh + all its inputs can be restored with one click.

## Table: `swe2d_simulation_configs`

```sql
CREATE TABLE IF NOT EXISTS swe2d_simulation_configs (
    config_id       TEXT PRIMARY KEY,
    mesh_name       TEXT,
    created_utc     TEXT NOT NULL,
    run_duration_s  REAL DEFAULT 0.0,
    description     TEXT DEFAULT '',
    widget_state    TEXT NOT NULL)   -- JSON blob from collect_workbench_widget_state()
```

The `widget_state` column stores the full output of `collect_workbench_widget_state()` as JSON. This dict has the structure:
```json
{
  "version": 1,
  "widgets": {
    "cfl_spin": {"type": "QDoubleSpinBox", "value": 0.45},
    "h_min_spin": {"type": "QDoubleSpinBox", "value": 0.0001},
    ...
  }
}
```

This format is produced by `collect_workbench_widget_state()` in `project_settings_bridge.py` and consumed by `restore_workbench_widget_state()`. The individual-column approach was abandoned in favor of the JSON blob because:
- Widget parameters change frequently (adds/removes)
- The typed-widget restore infrastructure already existed
- No schema migration needed when new widgets appear

## How It Works

**Save:** When the user clicks "Run", the controller calls `collect_workbench_widget_state(ui, widget_attrs, ...)` with all attribute names from `collect_run_widget_params()`, then calls `persist_simulation_config()` to write to the GPKG.

**Restore:** The "Load Model Config from GPKG..." button opens `SWE2DSimulationConfigDialog`, which lists saved configs. Selecting one and clicking "Apply" calls `_apply_run_log_metadata_to_ui()` which delegates to `restore_workbench_widget_state()`.

## Files Changed

| File | Change |
|------|--------|
| `swe2d/services/gpkg_persistence_service.py` | Added `persist_simulation_config()`, `load_simulation_configs()`, table DDL in `_ensure_ogc_gpkg_tables()` |
| `swe2d/workbench/views/model_tab_view.py` | Button text changed to "Load Model Config from GPKG..." |
| `swe2d/workbench/views/studio_tab_builder.py` | Wiring changed to `on_load_simulation_config` |
| `swe2d/workbench/controllers/run_controller.py` | Added `on_load_simulation_config()` handler; config saved at run start |
| `swe2d/workbench/dialogs/simulation_config_dialog.py` | New config picker dialog |

## Relationship to existing data

- `swe2d_baked_mesh` stores the mesh BLOB — referenced by `mesh_name`
- BC feature tables (`swe2d_bc_lines`, etc.) are referenced by GPKG path (stored in combo widgets)
- Hyetograph tables (`swe2d_hyetographs`, `swe2d_rain_gages`) are referenced by GPKG path + table name in combo widgets
- Drainage/structures feature tables are referenced similarly in combo widgets

The widget state captures the *current selection* of all combo boxes, spin boxes, and checkboxes. When restored, it sets those widgets — if the referenced GPKG/layer is available, the combo will find it; if not, the combo will show an empty/default state.
- Drainage/structures feature tables already exist
- The config table just points to all of them + stores scalar params
