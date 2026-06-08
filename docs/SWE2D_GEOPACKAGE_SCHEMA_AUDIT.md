# SWE2D Model GeoPackage — Schema & Results Storage Audit

## Table of Contents

1. [Overview](#overview)
2. [GeoPackage as Storage Platform](#geopackage-as-storage-platform)
3. [Spatial Schema — Model Input Layers](#spatial-schema--model-input-layers)
   - [Layer List & Field Definitions](#layer-list--field-definitions)
   - [Layer Binding System](#layer-binding-system)
   - [Model Metadata Table](#model-metadata-table)
4. [Results / Run-Time Tables](#results--run-time-tables)
   - [Sample-Line (1D Cross-Section) Results](#sample-line-1d-cross-section-results)
   - [Mesh (Cell-Centered) Results](#mesh-cell-centered-results)
   - [Drainage/Structure Coupling Results](#drainagestructure-coupling-results)
   - [Conservation Forensics](#conservation-forensics)
   - [Run Logs](#run-logs)
   - [Face-Flux / Velocity Tables](#face-flux--velocity-tables)
5. [Unsteady (1D Backwater) Subsystem Tables](#unsteady-1d-backwater-subsystem-tables)
6. [1D Hydra (Steady Backwater) Schema](#1d-hydra-steady-backwater-schema)
7. [GeoPackage Creation & Loading Flow](#geopackage-creation--loading-flow)
   - [Creating a New 2D Model GPKG](#creating-a-new-2d-model-gpkg)
   - [Loading a 2D Model GPKG](#loading-a-2d-model-gpkg)
   - [Migrating an Existing GPKG](#migrating-an-existing-gpkg)
8. [Results Persistence Flow (Runtime → GPKG)](#results-persistence-flow-runtime--gpkg)
9. [Table-Prefix Namespacing](#table-prefix-namespacing)
10. [Key Files & Roles](#key-files--roles)
11. [Summary Diagram](#summary-diagram)

---

## Overview

The SWE2D hydraulic model plugin uses **GeoPackage** (SQLite-backed OGC standard) as its
universal storage container. A single `.gpkg` file holds:

- **Spatial input layers** for mesh generation (topology nodes/arcs/regions, Manning's n,
  BC lines, sample lines, drainage networks, hydraulic structures).
- **Attribute-only input tables** for timeseries data (hyetographs, hydrographs,
  drainage inlet catalogs).
- **Run-time result tables** for line samples, mesh snapshots, coupling diagnostics,
  conservation forensics, and run logs.
- **Metadata tables** for schema versioning and layer-role bindings.

Two parallel subsystems each have their own GeoPackage conventions:

| Subsystem | GPKG Scope | Primary File |
|-----------|-----------|--------------|
| **SWE2D (2D)** | Single GPKG holds both input layers *and* result tables | `swe2d_workbench_qt.py` + `swe2d/workbench/extracted/topology_and_io_methods.py` |
| **1D Unsteady** | Separate GPKG (e.g. `unsteady_example.gpkg`) for binary result blobs | `unsteady_model.py` |
| **1D Hydra** | Separate GPKG (e.g. `example.gpkg`) with `cross_sections`/`centerline`/`boundary_conditions` layers | `hydra_1d.py` |

---

## GeoPackage as Storage Platform

GeoPackage is built on **SQLite** with a defined metadata schema (`gpkg_contents`,
`gpkg_geometry_columns`, etc.). All tables are standard SQLite tables; spatial layers
register themselves in the OGC metadata tables when written via QGIS's OGR provider.

The codebase uses two parallel write paths:

1. **QGIS vector-layers** (`QgsVectorLayer` + `QgsVectorFileWriter.writeAsVectorFormatV2`)
   for spatial input layers — this automatically handles the OGC metadata.
2. **Raw `sqlite3`** connections for non-spatial result tables and metadata —
   these bypass the OGC machinery but still use standard SQLite.

> **Important**: Because result tables are created via `sqlite3.connect()` (not through
> QGIS's OGR), they are **not** registered in `gpkg_contents` or `gpkg_geometry_columns`.
> This is intentional — they are pure attribute tables and are meant to be consumed
> programmatically, not as QGIS map layers.

---

## Spatial Schema — Model Input Layers

### Layer List & Field Definitions

A canonical SWE2D model GeoPackage is created by `_create_2d_model_geopackage()` in
`topology_and_io_methods.py`. It materializes **18 layers** (16 spatial, 2 attribute-only)
into a single `.gpkg` file:

| # | Layer Name | Geometry | Fields |
|---|------------|----------|--------|
| 1 | `swe2d_topo_nodes` | Point | `node_id: Integer` |
| 2 | `swe2d_topo_arcs` | LineString | `arc_id: Integer`, `node0: Integer`, `node1: Integer`, `use_global_arc_ctrl: Integer`, `arc_mode_override: String(24)`, `arc_soft_size_override: Double`, `arc_soft_dist_override: Double` |
| 3 | `swe2d_topo_regions` | Polygon | `region_id: Integer`, `target_size: Double`, `cell_type: String(32)`, `edge_len[1-4]: Double` |
| 4 | `swe2d_topo_constraints` | Polygon | `constraint_id: Integer`, `target_size: Double`, `cell_type: String(32)`, `edge_len[1-4]: Double` |
| 5 | `swe2d_topo_quad_edges` | LineString | `region_id: Integer`, `edge_id: Integer`, `target_size: Double`, `n_layers: Integer`, `first_height: Double`, `growth_rate: Double` |
| 6 | `swe2d_manning_zones` | Polygon | `zone_id: Integer`, `n_mann: Double`, `priority: Integer` |
| 7 | `swe2d_bc_lines` | LineString | `bc_type: Integer`, `bc_value: Double`, `priority: Integer`, `hydrograph: String(1024)`, `hydrograph_id: String(64)`, `hydrograph_layer: String(128)` |
| 8 | `swe2d_sample_lines` | LineString | `line_id: Integer`, `name: String(128)`, `enabled: Integer`, `priority: Integer` |
| 9 | `swe2d_rain_gages` | Point | `gage_id: String(64)`, `name: String(128)`, `hyetograph_id: String(64)`, `units: String(32)`, `priority: Integer` |
| 10 | `swe2d_storm_areas` | Polygon | `storm_id: Integer`, `name: String(128)`, `priority: Integer` |
| 11 | `swe2d_cn_zones` | Polygon | `zone_id: Integer`, `cn: Double`, `priority: Integer` |
| 12 | **`swe2d_hyetographs`** | *(None — attribute table)* | `hyetograph_id: String(64)`, `Time: String(32)`, `Value: Double`, `value_type: String(24)`, `units: String(24)`, `description: String(256)` |
| 13 | **`swe2d_hydrographs`** | *(None — attribute table)* | `hydrograph_id: String(64)`, `bc_type: Integer`, `Time: String(32)`, `Value: Double`, `description: String(256)` |
| 14 | `swe2d_drainage_nodes` | Point | `node_id: String(64)`, `invert_elev: Double`, `max_depth: Double`, `rim_elev: Double`, `crest_elev: Double`, `node_type: String(32)`, `surface_area: Double`, `outfall_area: Double`, `zero_storage: Integer` |
| 15 | `swe2d_drainage_links` | LineString | `link_id: String(64)`, `from_node: String(64)`, `to_node: String(64)`, `link_type: String(32)`, `link_shape: String(32)`, `length: Double`, `roughness_n: Double`, `diameter: Double`, `span: Double`, `rise: Double`, `area_m2: Double`, `equiv_diameter_m: Double`, `max_flow: Double`, `cd: Double` |
| 16 | **`swe2d_drainage_inlets`** | *(None — attribute table)* | `inlet_type_id: String(64)`, `name: String(128)`, `weir_length: Double`, `orifice_area: Double`, `coeff_weir: Double`, `coeff_orifice: Double`, `max_capture: Double`, `description: String(256)` |
| 17 | **`swe2d_drainage_node_inlets`** | *(None — attribute table)* | `node_id: String(64)`, `inlet_type_id: String(64)`, `inlet_count: Double`, `crest_offset: Double`, `description: String(256)` |
| 18 | `swe2d_structures` | LineString | `structure_id: String(64)`, `structure_type: Integer`, `crest_elev: Double`, `enabled: Integer`, `width: Double`, `height: Double`, `diameter: Double`, `culvert_shape: String(32)`, `culvert_code: Integer`, `culvert_rise: Double`, `culvert_span: Double`, `culvert_area_m2: Double`, `culvert_barrels: Integer`, `culvert_slope: Double`, `inlet_invert_elev: Double`, `outlet_invert_elev: Double`, `entrance_loss_k: Double`, `exit_loss_k: Double`, `embankment_enabled: Integer`, `embankment_crest_elev: Double`, `embankment_overflow_width: Double`, `embankment_weir_coeff: Double`, `length: Double`, `roughness_n: Double`, `coeff: Double`, `cd: Double`, `opening: Double`, `q_pump: Double`, `max_flow: Double`, `inlet_loss_k: Double`, `outlet_loss_k: Double`, `stacked_enabled: Integer`, `influence_width_m: Double`, `upstream_buffer_m: Double`, `downstream_buffer_m: Double`, `deck_soffit_elev: Double`, `deck_top_elev: Double`, `model_top_elev: Double`, `under_layers: Integer`, `over_layers: Integer`, `pier_count: Integer`, `pier_width: Double` |

**Field-type mapping** during migration (`_migrate_2d_model_geopackage`):
- `integer`/`int` → `INTEGER`
- `double`/`real` → `REAL`
- all others (including `string(N)`) → `TEXT`

### Layer Binding System

To decouple the UI combo-box selectors from hardcoded layer names, the workbench uses a
**layer-role binding** system defined in `_MODEL_LAYER_BINDINGS` (a dict in
`swe2d_workbench_qt.py`):

```python
_MODEL_LAYER_BINDINGS = {
    "rain_gages":        {"layer_name": "swe2d_rain_gages",        "combo_attr": "rain_gage_layer_combo",        "geometry": "point",  "required_fields": ("gage_id", "hyetograph_id")},
    "hyetographs":       {"layer_name": "swe2d_hyetographs",       "combo_attr": "hyetograph_layer_combo",       "geometry": "table",  "required_fields": ("hyetograph_id", "Time", "Value")},
    "storm_areas":       {"layer_name": "swe2d_storm_areas",       "combo_attr": "storm_area_layer_combo",       "geometry": "polygon","required_fields": ("storm_id",)},
    "drainage_nodes":    {"layer_name": "swe2d_drainage_nodes",    "combo_attr": "drain_nodes_layer_combo",      "geometry": "point",  "required_fields": ("node_id", "invert_elev")},
    "drainage_links":    {"layer_name": "swe2d_drainage_links",    "combo_attr": "drain_links_layer_combo",      "geometry": "line",   "required_fields": ("link_id", "from_node", "to_node")},
    "drainage_inlets":   {"layer_name": "swe2d_drainage_inlets",   "combo_attr": "drain_inlets_layer_combo",     "geometry": "table",  "required_fields": ("inlet_type_id", "weir_length", "coeff_weir", "coeff_orifice")},
    "drainage_node_inlets":{"layer_name": "swe2d_drainage_node_inlets","combo_attr": "drain_node_inlets_layer_combo","geometry": "table",  "required_fields": ("node_id", "inlet_type_id")},
    "hydraulic_structures":{"layer_name": "swe2d_structures",      "combo_attr": "structures_layer_combo",       "geometry": "line",   "required_fields": ("structure_id", "structure_type", "crest_elev", "enabled")},
}
_MODEL_LAYER_BINDINGS_VERSION = 5
```

Bindings are persisted to `swe2d_layer_bindings` and loaded on startup so the UI can
automatically populate the correct combo-box selections after loading a GPKG.

### Model Metadata Table

**Table name**: `swe2d_model_metadata`

Created by `_persist_model_layer_bindings()` in `swe2d_workbench_qt.py`:

```sql
CREATE TABLE IF NOT EXISTS swe2d_model_metadata (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,
    updated_utc TEXT NOT NULL
);
```

**Well-known keys**:
| Key | Value |
|-----|-------|
| `swe2d_coupling_schema_version` | `_MODEL_LAYER_BINDINGS_VERSION` (currently `5`) |
| `swe2d_coupling_layer_roles` | JSON array of all role names from `_MODEL_LAYER_BINDINGS` |
| `swe2d_explicit_layer_roles` | JSON array of roles that have explicit combo selections |

**Layer bindings table**: `swe2d_layer_bindings`

```sql
CREATE TABLE IF NOT EXISTS swe2d_layer_bindings (
    role            TEXT PRIMARY KEY,
    layer_name      TEXT NOT NULL,
    geometry_type   TEXT,
    required_fields TEXT,
    updated_utc     TEXT NOT NULL
);
```

---

## Results / Run-Time Tables

All result tables are created on-demand via `sqlite3` connections (not QGIS OGR). They
share a common architecture: a **runs** metadata table paired with one or more **data**
tables, joined by `run_id`.

### Sample-Line (1D Cross-Section) Results

Persisted by `_persist_line_results_to_geopackage()` in `topology_and_io_methods.py`.

**Runs metadata** — `swe2d_line_results_runs`:
```sql
CREATE TABLE IF NOT EXISTS swe2d_line_results_runs (
    run_id           TEXT PRIMARY KEY,
    created_utc      TEXT,
    mesh_interval_s  REAL,
    line_interval_s  REAL,
    row_count        INTEGER
);
```

**Time-series table** — `swe2d_line_results_ts`:
```sql
CREATE TABLE IF NOT EXISTS swe2d_line_results_ts (
    run_id      TEXT,
    t_s         REAL,
    line_id     INTEGER,
    line_name   TEXT,
    depth_m     REAL,
    velocity_ms REAL,
    wse_m       REAL,
    bed_m       REAL,
    flow_cms    REAL,
    wet_frac    REAL,
    fr          REAL,
    PRIMARY KEY (run_id, t_s, line_id)
);
```

**Profile table** — `swe2d_line_results_profile`:
```sql
CREATE TABLE IF NOT EXISTS swe2d_line_results_profile (
    run_id      TEXT,
    t_s         REAL,
    line_id     INTEGER,
    line_name   TEXT,
    station_m   REAL,
    depth_m     REAL,
    velocity_ms REAL,
    wse_m       REAL,
    bed_m       REAL,
    flow_qn     REAL,
    wet         INTEGER,
    fr          REAL,
    PRIMARY KEY (run_id, t_s, line_id, station_m)
);
```

Indexes:
```sql
CREATE INDEX IF NOT EXISTS idx_swe2d_line_results_ts_run_line_t
    ON swe2d_line_results_ts(run_id, line_id, t_s);
CREATE INDEX IF NOT EXISTS idx_swe2d_line_results_profile_run_line_t_s
    ON swe2d_line_results_profile(run_id, line_id, t_s, station_m);
```

### Mesh (Cell-Centered) Results

Persisted by `persist_mesh_results_to_geopackage()` in `results_export_methods.py`.

**Runs metadata** — `swe2d_mesh_results_runs`:
```sql
CREATE TABLE IF NOT EXISTS swe2d_mesh_results_runs (
    run_id       TEXT PRIMARY KEY,
    created_utc  TEXT,
    interval_s   REAL,
    row_count    INTEGER
);
```

**Data table** — `swe2d_mesh_results`:
```sql
CREATE TABLE IF NOT EXISTS swe2d_mesh_results (
    run_id  TEXT,
    t_s     REAL,
    cell_id INTEGER,
    h       REAL,
    hu      REAL,
    hv      REAL,
    PRIMARY KEY (run_id, t_s, cell_id)
);
```

Column meanings: `h` = water depth, `hu` = x-momentum, `hv` = y-momentum (all in model units).

The table name can be overridden via the `_selected_mesh_results_table_name()` / `_results_table_name()` methods with a user-specified prefix.

### Drainage/Structure Coupling Results

Persisted by `_persist_coupling_results_to_geopackage()` in `swe2d_workbench_qt.py`.

**Runs metadata** — `swe2d_coupling_results_runs`:
```sql
CREATE TABLE IF NOT EXISTS swe2d_coupling_results_runs (
    run_id      TEXT PRIMARY KEY,
    created_utc TEXT,
    interval_s  REAL,
    row_count   INTEGER
);
```

**Data table** — `swe2d_coupling_results`:
```sql
CREATE TABLE IF NOT EXISTS swe2d_coupling_results (
    run_id      TEXT,
    t_s         REAL,
    component   TEXT,     -- 'structure', 'drainage_node', 'drainage_link'
    object_id   TEXT,
    object_name TEXT,
    metric      TEXT,     -- 'flow', 'depth', 'head', etc.
    value       REAL,
    PRIMARY KEY (run_id, t_s, component, object_id, metric)
);
```

Index:
```sql
CREATE INDEX IF NOT EXISTS idx_swe2d_coupling_results_run_component_metric_obj_t
    ON swe2d_coupling_results(run_id, component, metric, object_id, t_s);
```

### Conservation Forensics

Persisted by `persist_conservation_forensics_to_geopackage()` in `results_export_methods.py`.

Four tables, all with column-addition migration via `_ensure_columns()`:

**Runs summary** — `swe2d_conservation_runs`:
```sql
CREATE TABLE IF NOT EXISTS swe2d_conservation_runs (
    run_id                         TEXT PRIMARY KEY,
    created_utc                    TEXT,
    run_duration_s                 REAL,
    source_rain_model              REAL,
    source_cell_model              REAL,
    source_coupling_model          REAL,
    source_total_model             REAL,
    storage_start_model            REAL,
    storage_end_model              REAL,
    storage_delta_model            REAL,
    implied_net_boundary_out_model REAL,
    avg_implied_boundary_q_model   REAL,
    boundary_group_volume_sum_model REAL,
    source_total_m3                REAL,
    storage_start_m3               REAL,
    storage_end_m3                 REAL,
    storage_delta_m3               REAL,
    implied_net_boundary_out_m3    REAL,
    avg_implied_boundary_q_cms     REAL,
    boundary_group_volume_sum_m3   REAL,
    -- Late-added columns (via ALTER TABLE):
    boundary_face_flux_table       TEXT,
    boundary_face_flux_status      TEXT,
    boundary_face_flux_rows        INTEGER,
    boundary_face_flux_total_model REAL,
    boundary_face_flux_total_cms   REAL,
    effective_net_boundary_method  TEXT,
    effective_net_boundary_out_model REAL,
    effective_net_boundary_out_m3  REAL,
    effective_avg_q_model          REAL,
    effective_avg_q_cms            REAL,
    closure_residual_model         REAL,
    closure_residual_m3            REAL
);
```

**Storage time-series** — `swe2d_conservation_storage_ts`:
```sql
CREATE TABLE IF NOT EXISTS swe2d_conservation_storage_ts (
    run_id            TEXT,
    t_s               REAL,
    storage_model     REAL,
    storage_delta_model REAL,
    storage_m3        REAL,
    storage_delta_m3  REAL,
    PRIMARY KEY (run_id, t_s)
);
```

**Boundary flux forensics** — `swe2d_boundary_flux_forensics_ts`:
```sql
CREATE TABLE IF NOT EXISTS swe2d_boundary_flux_forensics_ts (
    run_id             TEXT,
    t_s                REAL,
    group_name         TEXT,
    q_requested_model  REAL,
    q_effective_model  REAL,
    vol_requested_model REAL,
    vol_effective_model REAL,
    q_requested_cms    REAL,
    q_effective_cms    REAL,
    vol_requested_m3   REAL,
    vol_effective_m3   REAL,
    source_note        TEXT,
    PRIMARY KEY (run_id, t_s, group_name)
);
```

**Source budget forensics** — `swe2d_source_budget_forensics_ts`:
```sql
CREATE TABLE IF NOT EXISTS swe2d_source_budget_forensics_ts (
    run_id               TEXT,
    t_s                  REAL,
    rain_vol_model       REAL,
    cell_vol_model       REAL,
    coupling_vol_model   REAL,
    source_total_vol_model REAL,
    rain_vol_m3          REAL,
    cell_vol_m3          REAL,
    coupling_vol_m3      REAL,
    source_total_vol_m3  REAL,
    PRIMARY KEY (run_id, t_s)
);
```

All forensics tables store *both* model-unit and SI-unit columns, computed at write time
via the CRS-derived `length_scale_si_to_model()` factor.

### Run Logs

Persisted by `persist_run_log_to_geopackage()` in `swe2d_run_log_storage.py`.

**Table name**: `swe2d_run_logs` (or `{prefix}_swe2d_run_logs` when table prefixing is active).

```sql
CREATE TABLE IF NOT EXISTS swe2d_run_logs (
    run_id         TEXT PRIMARY KEY,
    created_utc    TEXT,
    start_wallclock TEXT,
    end_wallclock  TEXT,
    duration_s     REAL,
    log_text       TEXT,
    metadata_json  TEXT
);
```

The `metadata_json` column was added in a schema migration (late addition via `ALTER TABLE`).
It stores a JSON blob of workbench widget state (used for "Load Run Settings from Results"
restoration).

### Face-Flux / Velocity Tables

These are **not explicitly created** by Python code but are written by the C++ CUDA kernel
during GPU runs. The Python query layer discovers them at read time:

- `swe2d_face_flux_results`
- `swe2d_face_results`
- `swe2d_flux_faces`

The velocity reconstruction code in `swe2d/results/velocity_layer.py` dynamically discovers
whichever table exists and queries columns by name, tolerating many naming variants:

| Logical Column | Accepted Names |
|---------------|----------------|
| `run_id` | `run_id`, `run`, `result_id` |
| `t_s` | `t_s`, `time_s`, `time`, `t` |
| `cell_id` | `cell_id`, `cell`, `cell_idx`, `owner_cell` |
| Normal X | `nx`, `normal_x`, `face_nx` |
| Normal Y | `ny`, `normal_y`, `face_ny` |
| Normal flux | `flux_n`, `qn`, `normal_flux`, `q_normal`, `flux` |
| Edge length/weight | `face_length`, `edge_length`, `ds`, `weight` |

The layer performs a **weighted least-squares reconstruction** of cell-centered `(hu, hv)`
from face-normal fluxes, using the CRS-derived normal direction and optional edge-length
weighting.

---

## Unsteady (1D Backwater) Subsystem Tables

**File**: `unsteady_model.py`

The 1D unsteady solver stores results in its own GeoPackage using raw `sqlite3` DDL.
Four tables are defined as module-level `_TABLE_DDL` constants:

### `unsteady_results`

```sql
CREATE TABLE IF NOT EXISTS unsteady_results (
    run_id         TEXT    PRIMARY KEY,
    run_time       TEXT    NOT NULL,
    n_sections     INTEGER NOT NULL,
    n_output_times INTEGER NOT NULL,
    dt_s           REAL    NOT NULL,
    t_end_s        REAL    NOT NULL,
    section_ids    TEXT    NOT NULL,
    times_blob     BLOB    NOT NULL,
    wse_blob       BLOB    NOT NULL,
    q_blob         BLOB    NOT NULL,
    max_wse_blob   BLOB    NOT NULL,
    metadata       TEXT
);
```

Arrays (`times`, `wse`, `q`, `max_wse`) are stored as raw `float64` binary blobs for
compactness and fast I/O. Shapes: `times` = `(n_output_times,)`, `wse`/`q` =
`(n_output_times, n_sections)`.

### `unsteady_hydrographs`

```sql
CREATE TABLE IF NOT EXISTS unsteady_hydrographs (
    hydrograph_id TEXT PRIMARY KEY,
    bc_type       TEXT NOT NULL,
    label         TEXT,
    data_json     TEXT NOT NULL
);
```

### `unsteady_plans`

```sql
CREATE TABLE IF NOT EXISTS unsteady_plans (
    plan_id      TEXT PRIMARY KEY,
    plan_name    TEXT NOT NULL,
    created_utc  TEXT NOT NULL,
    updated_utc  TEXT NOT NULL,
    data_json    TEXT NOT NULL
);
```

### `unsteady_debug_steps`

```sql
CREATE TABLE IF NOT EXISTS unsteady_debug_steps (
    run_id       TEXT    NOT NULL,
    step_idx     INTEGER NOT NULL,
    time_s       REAL    NOT NULL,
    record_kind  TEXT    NOT NULL,
    payload_blob BLOB    NOT NULL,
    PRIMARY KEY (run_id, step_idx, record_kind)
);
```

Debug records are `pickle`-serialized before storage.

---

## 1D Hydra (Steady Backwater) Schema

**File**: `hydra_1d.py`

### Input Tables

Three OGR-registered spatial/attribute layers:

| Layer | Geometry | Fields |
|-------|----------|--------|
| `cross_sections` | LineStringZ | `centerline_id`, `river_station`, `left_bank_station`, `right_bank_station`, `n_lob`, `n_ch`, `n_rob`, `contraction_coeff`, `expansion_coeff`, `L_lob_to_next`, `L_ch_to_next`, `L_rob_to_next`, `culvert_code`, `culvert_shape`, `culvert_diameter`, `culvert_width`, `culvert_height`, `culvert_upstream_invert`, `culvert_downstream_invert`, `culvert_length`, `culvert_weir_coeff`, `culvert_weir_sta_left`, `culvert_weir_sta_right`, `culvert_slope` |
| `centerline` | LineString | `centerline_id` |
| `boundary_conditions` | *(None)* | `boundary_type` (str), `boundary_value` (float), `flow_cfs` (float) |

### Results Table

```python
# Written by _save_results_to_geopackage_qgis()
layer_name = 'model_results'
# Fields:
# result_index (Int), river_station (String), solver (String),
# run_time_utc (String),
# wse, depth_at_min, alpha,
# A_lob, A_ch, A_rob,
# K_lob, K_ch, K_rob,
# Q_lob, Q_ch, Q_rob,
# V_t, K_t, A_t,
# Sf_total, Froude (all Double)
```

---

## GeoPackage Creation & Loading Flow

### Creating a New 2D Model GPKG

1. User invokes **File → Create New Model GeoPackage** or clicks the create button.
2. `_create_2d_model_geopackage()` in `topology_and_io_methods.py` is called.
3. A **Save File** dialog captures the output path.
4. The project CRS is read (`EPSG:4326` fallback).
5. **18 memory layers** are constructed with URI-encoded field schemas.
6. Each layer has editor widgets configured via `_configure_swe2d_layer_editors()`:
   - Value maps for `cell_type`, `bc_type`, `arc_mode_override`
   - Expression constraints (`target_size > 0`, `bc_type IN (1,2,3,...)`, etc.)
   - Field aliases
7. Layers are written sequentially to the GPKG using `_write_memory_layer_to_gpkg()`:
   ```python
   # QGIS OGR bridge with CreateOrOverwriteFile / CreateOrOverwriteLayer
   QgsVectorFileWriter.writeAsVectorFormatV2(layer, path, transformContext, opts)
   ```
8. Layer bindings are persisted via `_persist_model_layer_bindings()`:
   - Creates `swe2d_model_metadata` and `swe2d_layer_bindings` tables
   - Records `swe2d_coupling_schema_version`, `swe2d_coupling_layer_roles`, `swe2d_explicit_layer_roles`
   - Saves each combo-box selection to `swe2d_layer_bindings`
9. The new GPKG is loaded into QGIS via `_load_2d_model_geopackage(path_override=...)`.

### Loading a 2D Model GPKG

1. User invokes **File → Load Model GeoPackage** or the GPKG is auto-loaded after creation.
2. `_load_2d_model_geopackage()` in `swe2d_workbench_qt.py`:
   - Opens a file dialog (unless `path_override` is given)
   - Resets runtime snapshot caches (`_reset_runtime_snapshot_overlay_cache()`)
   - Loads each of the 18 layers via `QgsVectorLayer(f"{gpkg_path}|layername={name}", name, "ogr")`
   - Configures editor widgets for each layer
   - Refreshes all layer combo-boxes
   - Stores path in `self._model_gpkg_path`
   - Restores layer bindings from `swe2d_layer_bindings` / `swe2d_model_metadata`
   - Logs schema warnings (missing layers, missing fields)

### Migrating an Existing GPKG

`_migrate_2d_model_geopackage()` in `topology_and_io_methods.py`:

1. Opens file dialog for GPKG selection.
2. Iterates over the canonical `layer_specs` list.
3. For each layer:
   - If the table does not exist: writes an empty QGIS memory layer to the GPKG.
   - If the table exists: checks `PRAGMA table_info`, adds missing columns via `ALTER TABLE`.
4. Reports summary of added layers and columns.

---

## Results Persistence Flow (Runtime → GPKG)

The runtime finalization is orchestrated by `SWE2DRunFinalizer.finalize_and_persist()` in
`swe2d/runtime/run_finalizer.py`:

```
Step               Data                    Target Table(s)
----               ----                    ----------------
1. Mass balance    storage_start_model,    swe2d_conservation_runs (row)
   computation     storage_end_model,      swe2d_conservation_storage_ts (rows)
                   source budgets,         swe2d_boundary_flux_forensics_ts (rows)
                   boundary flux groups    swe2d_source_budget_forensics_ts (rows)

2. Line results    _line_snapshot_rows     swe2d_line_results_runs (row)
   (if enabled)    _line_snapshot_profile  swe2d_line_results_ts (batch)
                                       swe2d_line_results_profile (batch)

3. Coupling        _coupling_snapshot_rows swe2d_coupling_results_runs (row)
   results (if                             swe2d_coupling_results (batch)
   enabled)

4. Mesh results    _snapshot_timesteps     swe2d_mesh_results_runs (row)
   (if enabled)    → _build_mesh_       swe2d_mesh_results (batch)
                     snapshot_rows()

5. Run log         run_id, timestamps,     swe2d_run_logs (row)
                   log_text, metadata_json

6. High-perf       3D patch snapshots     (In-memory; optionally exported to HDF5)
   overlay update
```

Each persistence step is gated by a checkbox in the UI:
- `save_line_results_to_gpkg_chk`
- `save_coupling_results_to_gpkg_chk`
- `save_mesh_results_to_gpkg_chk`

The target GPKG path is determined by `_current_line_results_storage_path()`:
1. If `results_gpkg_path_edit` has a custom path, use it.
2. Otherwise, use `_model_gpkg_path` (the loaded model GPKG).
3. Fallback to the GPKG source of the sample-lines layer.
4. Last resort: temp file `swe2d_line_results.gpkg`.

---

## Table-Prefix Namespacing

The workbench supports **table-name prefixing** to allow multiple independent result sets
in the same GeoPackage, controlled by `results_table_name_edit` in the UI.

```python
# _selected_results_table_prefix() sanitizes the user input:
# - removes non-alphanumeric characters
# - ensures it starts with a letter or underscore
# e.g. "project-2024" → "project_2024"

# _results_table_name("swe2d_mesh_results") with prefix "project_a":
# → "project_a_swe2d_mesh_results"
```

The prefix is applied to **all result tables** (mesh, line, coupling, conservation,
run logs, boundary forensics, source forensics). The `swe2d_run_log_storage.py` module
also accepts its own `table_prefix` parameter independently.

This makes it possible to store results from multiple model runs or scenarios in a
single GPKG without name collisions.

---

## Key Files & Roles

| File | Role |
|------|------|
| `swe2d_workbench_qt.py` | Main workbench dialog: model creation/loading, runtime state, result persistence bridges. Defines `_MODEL_LAYER_BINDINGS`, `_persist_model_layer_bindings()`, `_restore_model_layer_bindings()`, `_persist_coupling_results_to_geopackage()`, `_current_line_results_storage_path()`, `_results_table_name()`, `_selected_results_table_prefix()`. |
| `swe2d/workbench/extracted/topology_and_io_methods.py` | GeoPackage creation (`_create_2d_model_geopackage`, `_migrate_2d_model_geopackage`), layer editor config, line-results persistence (`_persist_line_results_to_geopackage`), migration logic. |
| `swe2d/workbench/extracted/results_export_methods.py` | Mesh results (`persist_mesh_results_to_geopackage`), conservation forensics (`persist_conservation_forensics_to_geopackage`), coupling loading (`load_coupling_results_from_geopackage`), HDF5/NetCDF export, run log viewer. |
| `swe2d/runtime/run_finalizer.py` | `SWE2DRunFinalizer` — orchestrates end-of-run persistence, mass-balance computation, snapshot capture. |
| `swe2d_run_log_storage.py` | Dedicated module for `swe2d_run_logs` table: `persist_run_log_to_geopackage()`, `load_run_logs_from_geopackage()`. |
| `swe2d/results/queries.py` | Qt-free read-layer for multi-run results panel: `discover_line_result_runs()`, `load_timeseries()`, `load_profile()`, `load_structure_flows_at_time()`. |
| `swe2d/results/velocity_layer.py` | Face-flux velocity reconstruction from C++-written face tables. |
| `swe2d/results/db_utils.py` | Shared helpers: `open_ro()`, `table_exists()`, `table_columns()`. |
| `swe2d/workbench/startup_state.py` | Initializes result-tracking state variables (`_line_results_latest_run_id`, `_coupling_results_latest_db_path`, etc.) to empty values. |
| `unsteady_model.py` | 1D unsteady solver's binary-blob GeoPackage I/O: `save_unsteady_results_to_geopackage()`, `save_hydrograph_to_geopackage()`, `save_unsteady_plan_to_geopackage()`, debug-step pickle storage. |
| `hydra_1d.py` | 1D steady backwater: `save_to_geopackage()` (input tables), `save_results_to_geopackage()` (result attributes). |

---

## Summary Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SWE2D Model GeoPackage (.gpkg)                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  OGC Spatial Layers (via QGIS OGR)                                  │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ swe2d_topo_nodes, swe2d_topo_arcs, swe2d_topo_regions,      │   │
│  │ swe2d_topo_constraints, swe2d_topo_quad_edges,              │   │
│  │ swe2d_manning_zones, swe2d_bc_lines, swe2d_sample_lines,    │   │
│  │ swe2d_rain_gages, swe2d_storm_areas, swe2d_cn_zones,        │   │
│  │ swe2d_drainage_nodes, swe2d_drainage_links,                  │   │
│  │ swe2d_drainage_inlets, swe2d_drainage_node_inlets,           │   │
│  │ swe2d_structures                                            │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Attribute-Only Input Tables                                        │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ swe2d_hyetographs, swe2d_hydrographs                        │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Metadata & Bindings                                                │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ swe2d_model_metadata     (key/value schema versioning)      │   │
│  │ swe2d_layer_bindings     (combo → layer role mapping)      │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Result Tables (created via sqlite3, NOT OGC-registered)            │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ swe2d_line_results_runs / _ts / _profile  (sample lines)   │   │
│  │ swe2d_mesh_results_runs / swe2d_mesh_results (cells)       │   │
│  │ swe2d_coupling_results_runs / _results      (drainage/str) │   │
│  │ swe2d_conservation_runs / _storage_ts / _boundary_flux_    │   │
│  │   forensics_ts / _source_budget_forensics_ts               │   │
│  │ swe2d_run_logs                                 (run logs)  │   │
│  │ swe2d_face_flux_results / _face_results / _flux_faces      │   │
│  │   (C++ CUDA kernel output)                                 │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│       1D Unsteady Model GeoPackage (separate .gpkg, e.g.            │
│                   unsteady_example.gpkg)                             │
├─────────────────────────────────────────────────────────────────────┤
│  unsteady_results        (binary blob: times/wse/q/max_wse)          │
│  unsteady_hydrographs    (JSON hydrograph storage)                  │
│  unsteady_plans          (JSON plan configuration)                  │
│  unsteady_debug_steps    (pickle-blob debug records)                │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│       1D Steady Backwater GeoPackage (separate .gpkg)               │
├─────────────────────────────────────────────────────────────────────┤
│  cross_sections     (LineStringZ, ~30 attribute fields)              │
│  centerline         (LineString)                                     │
│  boundary_conditions (attribute table)                               │
│  model_results      (attribute table with wse/Q/hydraulic props)     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Key Observations

1. **Result tables are not OGC-registered.** They are pure SQLite tables created via
   `sqlite3.Cursor.execute()`. This means they do not appear in `gpkg_contents` and are
   invisible to standard GeoPackage readers. The `SWE2DModelGeoPackageExplorerDialog`
   provides the primary UI for browsing them.

2. **Dual-unit storage.** Conservation forensics tables store values in *both* model units
   and SI units (m³, CMS). This avoids runtime conversion on read but doubles column count
   and risks inconsistency if the conversion factors change.

3. **Schema migrations are manual and ad-hoc.** The `_ensure_columns()` pattern in
   `persist_conservation_forensics_to_geopackage()` and the side-table approach in
   `swe2d_run_log_storage.py` (`metadata_json` column addition) are the only migration
   mechanisms. There is no formal migration framework.

4. **Table prefixing creates name variants.** The `_results_table_name()` method (line
   10186) and `_find_prefixed_or_default_table()` in `queries.py` handle the mapping,
   but the C++ kernel's face-flux tables (`swe2d_face_flux_results` etc.) do **not**
   participate in prefixing — they use hardcoded names.

5. **Binary blob storage** (1D unsteady) vs. **row-per-timestep** (2D SWE2D). The unsteady
   solver stores full arrays as numpy bytes (compact, fast I/O), while the 2D solver
   stores one row per cell per timestep (relational, queryable). This is a fundamental
   architectural difference driven by data volume: 2D cell counts can be large enough to
   make fully relational storage expensive.

6. **Face-flux tables are schema-flexible.** The velocity reconstruction code in
   `velocity_layer.py` performs column-name discovery with multiple accepted aliases,
   allowing the C++ kernel to evolve its column naming without breaking the Python reader.
