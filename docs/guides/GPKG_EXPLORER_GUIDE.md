# GeoPackage Explorer Guide

The GeoPackage Explorer lets you inspect, manage, and clean up tables inside a
SWE2D model GeoPackage (`.gpkg`).

## Opening the Explorer

Click **Open Model GeoPackage Explorer** in the Utilities page of the Layers
tab. Browse to your `.gpkg` file.

## Table Listing

The explorer shows all tables in the GeoPackage with:

| Column | Description |
|--------|-------------|
| **Table** | Table name |
| **Rows** | Number of rows (or `?` if unavailable) |
| **Type** | Auto-classified category |
| **Actions** | Available operations |

### Table Types

| Type | Description |
|------|-------------|
| `run_log` | Run log entries (`swe2d_run_logs`) |
| `line_results` | Sample line timeseries/profiles |
| `coupling_results` | Structure/drainage coupling data |
| `mesh_results` | Mesh snapshot results |
| `conservation` | Water-budget forensics tables |
| `system` | OGC metadata tables (`spatial_ref_sys`, `gpkg_contents`, etc.) |
| `table` | Other tables (input layers, configs) |

## Actions

### Open Viewer

Opens a read-only preview of the table contents. For run logs, this opens
the Run Log Viewer. For line results, it opens a timeseries viewer. For
other tables, a generic SQLite table preview is shown.

**Double-click** any row to open its viewer directly.

### Preview Table

Opens a read-only table preview for any selected table. Useful for
inspecting input layer data (`swe2d_topo_*`, `swe2d_manning_zones`, etc.).

### Rename Table

Renames a `swe2d_*` table. Non-model tables (OGC metadata, etc.) cannot be
renamed. Enter a new name when prompted.

### Delete Table

Permanently deletes a `swe2d_*` table with a confirmation dialog. Non-model
tables cannot be deleted.

### Delete by Run ID

Opens a sub-dialog that:
1. Enumerates all run IDs found in the GeoPackage
2. Lets you select one or more runs to delete
3. Shows which tables contain data for the selected runs
4. Bulk-deletes all matching rows

This is the recommended way to remove old or unwanted simulation runs.

### Refresh

Reloads the table listing from the GeoPackage.

## Tips

- **Cleaning up after test runs**: Use "Delete by Run ID" to remove
  individual test runs without affecting other data.
- **Inspecting input data**: Select a `swe2d_topo_*` table and click
  "Preview Table" to check your topology before meshing.
- **Checking results**: Select a `swe2d_baked_results` table to verify that
  snapshots were persisted correctly.
