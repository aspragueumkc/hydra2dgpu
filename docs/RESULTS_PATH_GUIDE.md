# Results Path Guide

How SWE2D results are captured, persisted, and reloaded.

## Overview

After a simulation, all results live in a single GeoPackage (`.gpkg`). The
same file holds the input model, mesh, and output results — there are no
separate result files to manage.

Each simulation run is identified by a unique `run_id` string. A single
GeoPackage can hold many runs, enabling side-by-side comparison.

## Data Flow During a Run

```
GPU solver (h, hu, hv arrays)
    │
    ├─► Live snapshot arrays (in-memory numpy)
    │       └─► Canvas overlay (real-time display on map)
    │
    ├─► Live line timeseries (per sample line)
    │
    ├─► Live line profiles (cross-section snapshots)
    │
    └─► Live structure coupling timeseries
            │
            ▼
    Run finalizer (end of run)
            │
            ▼
    GeoPackage persistence (baked BLOBs)
```

### During the Run

1. The GPU solver produces `(h, hu, hv)` arrays each output interval.
2. These are appended to in-memory numpy arrays via
   `SWE2DResultsData.append_live_snapshot()`.
3. The `OverlayController` picks up the live arrays on each timer tick,
   rasterizes them via the C++ `hydra_overlay` extension, and pushes the
   resulting QImage to the map canvas for real-time display.
4. Line timeseries, profiles, and structure coupling data are appended
   simultaneously.

### After the Run Completes

The `SWE2DRunFinalizer` persists everything to the GeoPackage:

| Data | GPKG Table | Format |
|------|-----------|--------|
| Mesh snapshots (h/hu/hv at each output time) | `swe2d_baked_results` | Flat BLOBs (one per variable per run) |
| Line timeseries (depth, velocity, WSE, flow per sample line) | `swe2d_baked_line_ts` | Flat BLOBs per line |
| Line profiles (cross-section snapshots) | `swe2d_baked_line_profiles` | 2D BLOBs (time × station) |
| Structure/drainage coupling timeseries | `swe2d_baked_coupling` | Flat BLOBs per metric |
| Run log (full text + metadata) | `swe2d_run_logs` | TEXT + JSON |
| Conservation water-budget accounting | `swe2d_conservation_*` | Per-run tables |

All BLOBs are numpy arrays serialized via `tobytes()`. They are loaded back
via `np.frombuffer()` — no lossy conversion, no rounding.

## Reloading Results

When you open a results GeoPackage (or click "Discover Runs" in the results
panel):

1. `collect_runs_from_gpkg()` scans `swe2d_baked_results` for available runs.
2. Each run's timesteps are loaded from its BLOB.
3. The union of all enabled run timesteps is computed for the timeline slider.
4. Selecting a timestep loads the snapshot from the baked BLOBs and renders it
   through the same `render_unstructured_snapshot_image()` pipeline used during
   live runs.

## Results Panel

The results panel (right dock) provides:

- **Timeline slider** — scrub through simulation time. Shows all timesteps
  across all enabled runs.
- **Variable selector** — choose what to display: depth, speed, WSE, Froude
  number, Courant number, or shear stress.
- **Run list** — toggle runs on/off for overlay comparison. Each run gets a
  distinct color.
- **Sample line selector** — pick a sample line to view timeseries and
  cross-section profiles.
- **Structure results** — view coupling timeseries for drainage nodes, links,
  and hydraulic structures.
- **Export** — export timeseries or profiles to CSV.

## Canvas Overlay

The high-performance canvas overlay renders results directly on the QGIS map
canvas:

- Rasterization uses the C++ `hydra_overlay` extension for GPU-accelerated
  triangle fill.
- Supports face-segment highlighting (for line-flux sampling) and station
  indicators.
- A scalar legend (color bar) is drawn automatically.
- Velocity arrows or streamlines can be overlaid.
- Opacity is adjustable via the overlay controls.

## Exporting Results

### CSV Export

Use the export buttons in the results panel to save timeseries or profile data
as CSV files. The CSV format includes a header row with column names.

### QGIS Map Layers

After a run, you can export the in-memory mesh (nodes + cells) as QGIS map
layers for inspection using the "Export Mesh To Map Layers" button in the
Layers tab.

## GeoPackage Table Reference

### Results Tables (BLOB format)

| Table | Primary Key | Contents |
|-------|------------|----------|
| `swe2d_baked_results` | `run_id` | Mesh snapshot BLOBs: times, h, hu, hv (and optional max-tracking) |
| `swe2d_baked_line_ts` | `run_id, line_id` | Per-line timeseries BLOBs: times, depth, velocity, WSE, bed, flow, wet fraction, Froude |
| `swe2d_baked_line_profiles` | `run_id, line_id` | Per-line 2D profile BLOBs: stations, times, depth, velocity, WSE, bed, normal flow, Froude, wet flag |
| `swe2d_baked_coupling` | `run_id, component, object_id, metric` | Structure/drainage coupling timeseries |
| `swe2d_baked_mesh` | `mesh_name` | Serialized mesh BLOB for reload |

### Run Metadata Tables

| Table | Primary Key | Contents |
|-------|------------|----------|
| `swe2d_run_logs` | `run_id` | Full run log text, wallclock times, JSON metadata |
| `swe2d_conservation_runs` | `run_id` | Conservation summary per run |
| `swe2d_conservation_storage_ts` | `run_id, time_s` | Storage volume timeseries |
| `swe2d_boundary_flux_forensics_ts` | `run_id, time_s, group_name` | Boundary flux volumes by BC group |
| `swe2d_source_budget_forensics_ts` | `run_id, time_s, component` | Source budget by component (rain, coupling, cell) |

### Input Tables (also in the same GPKG)

| Table | Purpose |
|-------|---------|
| `swe2d_topo_nodes` | Topology nodes |
| `swe2d_topo_arcs` | Topology arcs |
| `swe2d_topo_regions` | Topology regions |
| `swe2d_topo_constraints` | Mesh refinement constraints |
| `swe2d_topo_quad_edges` | Quad edge controls |
| `swe2d_manning_zones` | Manning's n zones |
| `swe2d_bc_lines` | Boundary condition lines |
| `swe2d_sample_lines` | Sample/monitoring lines |
| `swe2d_rain_gages` | Rain gages |
| `swe2d_storm_areas` | Storm areas |
| `swe2d_cn_zones` | Curve Number zones |
| `swe2d_hyetographs` | Hyetograph time series |
| `swe2d_hydrographs` | Hydrograph time series |
| `swe2d_drainage_nodes` | Drainage network nodes |
| `swe2d_drainage_links` | Drainage network links |
| `swe2d_drainage_inlets` | Surface-to-network inlets |
| `swe2d_drainage_node_inlets` | Node-inlet assignments |
| `swe2d_structures` | Hydraulic structures |
| `layer_styles` | Embedded QML styles for all layers |

## Troubleshooting

### "No runs found" after loading a GPKG

Ensure the GPKG was created by SWE2D and contains `swe2d_baked_results`.
Older GPKG files from pre-baked versions will not have results tables.

### Overlay shows nothing

Check that:
1. The results panel has a run selected (checkbox enabled).
2. A timestep is selected on the timeline slider.
3. The overlay checkbox is enabled in the overlay controls.

### Stale overlay after re-run

If you re-run a simulation and the overlay doesn't update, click "Discover
Runs" in the results panel to refresh the run list from the GPKG.

---

## Related Documentation

- **[Documentation Index](INDEX.md)** — All guides by audience
- **[User Guide](USER_GUIDE.md)** — Results panel, timeline, overlays
- **[Results GeoPackage Schema](RESULTS_GEOPACKAGE_SCHEMA.md)** — All output table formats
- **[Model GeoPackage Schema](MODEL_GEOPACKAGE_SCHEMA.md)** — Input table definitions
- **[Repository Knowledge Graph](../graphify-out/wiki/index.md)** — Results module connections
