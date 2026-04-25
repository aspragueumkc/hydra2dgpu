# HEC-RAS Project (generated)
This folder contains per-cross-section CSV files and a metadata CSV derived from `test3.json`.

## Files
- `XS_<river_station>.csv`: offset (station) and elevation.
- `sections_metadata.csv`: left/right bank stations, Manning n values, and L_* reach lengths.
- `Project.txt`: simple project summary and suggested settings.

## How to import into HEC-RAS
1) In HEC-RAS, create a new project.
2) Open Geometry -> Cross Sections -> Import -> from Files -> select the CSV files.
   - Map `Offset` to Station and `Elevation` to Elevation.
   - Use leftmost offset (0) as left bank; adjust Left/Right Bank stations in the cross section editor.
3) In Geometry, set Manning n for each cross section using `sections_metadata.csv`.
4) Create a Steady Flow plan with `flow_cfs` and assign Normal Depth boundary condition per `Project.txt`.
