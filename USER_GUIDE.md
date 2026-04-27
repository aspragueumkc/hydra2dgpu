# QGIS Backwater Plugin User Guide

This guide covers day-to-day use of the Backwater plugin inside QGIS.

## 1. What the Plugin Does

The plugin performs steady-flow 1D backwater calculations using a GeoPackage model and shows:

- Tabular result output
- Longitudinal and section plotting views
- QGIS layer-based editing workflow for model data

## 2. Prerequisites

- QGIS 3.x with Python plugin support
- Backwater plugin installed and enabled
- A writable location for GeoPackage model files (`*.gpkg`)

Optional:

- SciPy for the `scipy` solver option
- matplotlib for in-plugin plots

## 3. Open the Plugin

From the QGIS menu bar:

- Open **Backwater > Open Backwater Panel**

The dock widget opens on the right side of QGIS.

## 4. Create a New Model

1. Select **Backwater > Create New Model GeoPackage...**
2. Choose output path and file name
3. Choose CRS when prompted
4. The plugin creates a starter model and loads it into the panel

A model GeoPackage includes these key layers:

- `cross_sections`
- `centerline`
- `boundary_conditions`
- `model_results` (written after a successful run)

## 5. Load an Existing Model

1. Select **Backwater > Load Model GeoPackage...**
2. Pick a `*.gpkg` model
3. The panel loads boundary settings, sections, and persisted results (if present)

## 6. Edit Model Data

The plugin is designed for layer/form editing in QGIS.

1. Select **Backwater > Enable/Disable Layer Editing** (to enable)
2. Edit features in `cross_sections`, `centerline`, and `boundary_conditions`
3. Use configured layer forms and actions
4. Select **Backwater > Save Layer Edits** to commit

Important:

- If editing is enabled and unsaved changes exist, running is blocked.
- Save layer edits before running.

### 6.1 Cross-Section Form Actions

The cross-section form includes actions such as:

- Select terrain raster (stores selected raster id in project variable)
- Update feature Z values from terrain

Culvert fields are shown/hidden based on `culvert_code`.

## 7. Set Boundary and Solver Options

In the panel Boundary tab, you can set:

- DS BC: `known_wse` or `normal_depth`
- DS value (WSE or slope input)
- Flow (cfs)

From the QGIS menu:

- **Backwater > Options > Solver > Python (py)**
- **Backwater > Options > Solver > SciPy (scipy)**
- **Backwater > Options > Alpha Method > Conveyance**
- **Backwater > Options > Alpha Method > Area**

These menu choices mirror the panel selector state.

## 8. Run the Model

1. Ensure edits are saved
2. Select **Backwater > Run Model**
3. Review status and outputs in the panel

The run persists results to `model_results` in the same GeoPackage.

## 9. View Results

Use menu actions:

- **Backwater > Open Results Plot**
- **Backwater > Open Results Table**

Results can also reload automatically when opening a model with existing `model_results` rows.

## 10. Save a Copy of the Model

Use:

- **Backwater > Save Model GeoPackage As...**

This writes the current model to a selected GeoPackage path.

## 11. Troubleshooting

### Run blocked by unsaved edits

- Save with **Backwater > Save Layer Edits**
- Re-run the model

### "GeoPackage required" warnings

- The plugin workflow supports `*.gpkg` models
- Load or create a GeoPackage model first

### Missing plots / matplotlib warnings

- Install matplotlib in the QGIS Python environment
- Restart QGIS and rerun

### SciPy option selected but unavailable

- Install SciPy in the QGIS Python environment
- If unavailable, use solver option `py`

### Centerline-related load errors

- Confirm `centerline` exists and has at least one valid feature

## 12. Recommended Workflow

1. Create or load a GeoPackage model
2. Enable layer editing
3. Update cross sections, centerline, and boundary conditions
4. Save layer edits
5. Choose Solver and Alpha Method
6. Run model
7. Review plot and table
8. Save model copy if needed
