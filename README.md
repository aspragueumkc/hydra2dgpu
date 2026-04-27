# QGIS Backwater Plugin

QGIS plugin workspace for steady-flow backwater profile modeling, culvert/weir handling, and comparison tooling.

## Contents

- Core solver: `backwater2.py`
- Qt UI: `backwater_qt.py`
- Plugin entry: `backwater_plugin.py`
- Test scripts and diagnostics in project root and `tests/`
- Example HEC-RAS files in `hec_ras_project/`

## Model Editing Workflow

- Use `Create Model GeoPackage...` in the plugin UI to create a new model file.
- The plugin now runs in GeoPackage-only mode: model load/run inputs must be `*.gpkg`.
- The create workflow prompts for:
	- save location/name
	- model projection (CRS)
- New model GeoPackages always include the required layers:
	- `cross_sections`
	- `centerline`
	- `boundary_conditions`
	- `model_results` (written after each successful run)
- For GeoPackage-backed models, edits are made through native QGIS layer attribute forms.
- In-widget section/property edit controls are read-only in this mode; use layer forms/actions for model edits.
- `cross_sections` forms now include layer actions:
	- `Backwater: Select Terrain Raster` stores the selected raster in a project variable (`backwater_terrain_raster_id`).
	- `Backwater: Update Z From Terrain` updates feature vertex Z values using `expressions/vertices_z_from_raster.py` and refreshes `river_station` from centerline chainage.
- `boundary_conditions` forms include `Backwater: Run Model`, which triggers the plugin's run command from the form context.
- Custom Qt Designer form files are applied for:
	- `cross_sections` -> `forms/cross_sections_form.ui`
	- `boundary_conditions` -> `forms/boundary_conditions_form.ui`
	- form init/button styling and action handlers -> `forms/backwater_form_init.py`
- Drop-down editors are enforced for:
	- `cross_sections.culvert_shape` (`'', circular, rect`)
	- `boundary_conditions.boundary_type` (`known_wse, normal_depth`)
- No plugin-added field constraints are enforced; form validation is left to user workflow and provider-level constraints.
- In `cross_sections` forms, culvert detail fields are shown only when `culvert_code > 0`.
- Default values are set for `contraction_coeff` (`0.1`) and `expansion_coeff` (`0.3`).
- Successful model runs are persisted to the `model_results` GeoPackage layer, and reloaded into the UI results table/plots when the model is opened.
- GeoPackage geospatial I/O now prefers native PyQGIS APIs, with geopandas/shapely used only as a fallback when PyQGIS is unavailable.
- `centerline` is required when loading/saving a model. The plugin no longer treats centerline as optional.
- Default reach lengths (`L_ch_to_next`, and when unset also `L_lob_to_next`/`L_rob_to_next`) are derived from centerline spacing between neighboring cross sections.
