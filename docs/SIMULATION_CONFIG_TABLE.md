# Simulation Configuration Table — Idea Sketch

**Date:** 2026-06-29
**Status:** Idea, not implemented
**Motivation:** Avoid re-configuring all UI widgets (BC layers, hyetographs, solver knobs) every time a mesh is loaded. Persist the full run configuration in the GPKG so a mesh + all its inputs can be restored with one click.

## Proposed Table: `swe2d_simulation_configs`

```sql
CREATE TABLE IF NOT EXISTS swe2d_simulation_configs (
    config_id       TEXT PRIMARY KEY,
    mesh_name       TEXT REFERENCES swe2d_baked_mesh(mesh_name),
    created_utc     TEXT NOT NULL,
    description     TEXT DEFAULT '',

    -- Solver parameters
    h_min           REAL DEFAULT 1e-4,
    cfl             REAL DEFAULT 0.45,
    dt_max          REAL DEFAULT 0.2,
    dt_initial      REAL DEFAULT 0.0,
    gravity         REAL DEFAULT 32.174,
    spatial_scheme  INTEGER DEFAULT 1,
    temporal_scheme INTEGER DEFAULT 2,
    reconstruction  INTEGER DEFAULT 1,
    depth_cap       REAL DEFAULT 0.0,
    momentum_cap_min_speed    REAL DEFAULT 1.0,
    momentum_cap_celerity_mult REAL DEFAULT 10.0,
    max_rel_depth_increase    REAL DEFAULT 2.0,
    shallow_damping_depth     REAL DEFAULT 0.0,
    source_rate_cap           REAL DEFAULT 0.0,
    source_depth_step_cap     REAL DEFAULT 0.0,
    extreme_rain_mode         INTEGER DEFAULT 0,
    source_imex_split         INTEGER DEFAULT 0,
    front_flux_damping        REAL DEFAULT 0.0,
    active_set_hysteresis     INTEGER DEFAULT 0,
    n_mann         REAL DEFAULT 0.035,
    k_mann         REAL DEFAULT 1.0,
    gpu_diag_sync_interval    INTEGER DEFAULT 100,

    -- Boundary conditions
    default_bc_type INTEGER DEFAULT 0,

    -- Rainfall
    hyetograph_gpkg   TEXT,
    hyetograph_table  TEXT,
    rain_gauge_layer  TEXT,
    cn_table          TEXT,
    cn_field          TEXT DEFAULT 'cn',
    infiltration_method TEXT DEFAULT 'scs_cn',

    -- BC layers (JSON: list of {table, gpkg, type_field, val_field})
    bc_layers_json TEXT,

    -- Drainage / structures (GPKG paths containing the feature tables)
    drainage_gpkg     TEXT,
    structures_gpkg   TEXT,

    -- Results output
    output_interval_s       REAL DEFAULT 120.0,
    line_output_interval_s  REAL DEFAULT 120.0,
    run_duration_s          REAL DEFAULT 10800.0,

    -- Flags
    save_mesh_results       INTEGER DEFAULT 1,
    save_line_results       INTEGER DEFAULT 0,
    save_coupling_results   INTEGER DEFAULT 0,
    save_run_log            INTEGER DEFAULT 1
);
```

## How It Would Work

**Save:** When the user clicks "Run", snapshot the current widget state into a config row. Associate it with the mesh name and any GPKG paths used for inputs.

**Restore:** A "Load Config" button reads the config row, loads the mesh from `swe2d_baked_mesh`, then programmatically sets every widget (combo boxes, spin boxes, checkboxes) from the stored values. BC layers/hyetographs are referenced by GPKG path + table name so the QGIS layer combo can be re-populated.

**Relationship to existing data:**
- `swe2d_baked_mesh` already stores the mesh BLOB
- BC feature tables (`swe2d_bc_lines`, etc.) already exist in the GPKG
- Hyetograph tables (`swe2d_hyetographs`, `swe2d_rain_gages`) already exist
- Drainage/structures feature tables already exist
- The config table just points to all of them + stores scalar params
