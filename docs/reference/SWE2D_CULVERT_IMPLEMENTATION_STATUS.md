# SWE2D Culvert Implementation Status

## Implemented in this slice

- Replaced the old culvert branch in `swe2d/extensions/structures.py` with a richer flow law that now:
  - uses FHWA inlet-control capacity from `culvert_routine.py`
  - applies a tailwater-sensitive outlet-control surrogate using the direct-step culvert profile helper
  - retains area/orifice and full-pipe Manning caps as conservative limiters
  - supports optional embankment overflow via a weir equation on the same structure
  - supports multiple barrels via `culvert_barrels`
- Added per-structure detail reporting in `swe2d/extensions/structures.py` so culverts expose runtime metrics such as inlet-control flow, outlet-control flow, Manning/orifice caps, tailwater depth, and embankment overflow.
- Added a compiled native structure-flow helper in `cpp/src/swe2d_bindings.cpp` and wired it into the CUDA coupling branch in `swe2d/runtime/coupling.py`.
- The compiled helper now evaluates culverts with a native C++ port of the inlet-control and direct-step outlet-control routines (including the same caps/embankment logic), and the temporary Python-side culvert override in the controller has been removed.
- Extended structure ingestion in `swe2d_workbench_qt.py` so culvert-specific metadata is read from the structures layer.
- Extended default structures-layer templates so new model GeoPackages include culvert-specific fields.
- Added a dedicated QGIS structures form at `forms/swe2d_structures_culvert_form.ui` and hooked it into the structures-layer editor configuration.
- Migrated `qgis_testing_project/swe3d_model.gpkg` so the existing repository model can store the new culvert inputs.
- Added regression tests covering:
  - reduced culvert flow under higher tailwater
  - added discharge when embankment overtopping is enabled
- extended CUDA coupling controller routing through the native structure helper when present
- extended coupling snapshot persistence/viewing to include culvert-specific metrics in the existing drainage/structure results viewer

## New structures-layer fields

The `swe2d_structures` layer now carries these culvert-focused fields:

- `culvert_shape`
- `culvert_code`
- `culvert_rise`
- `culvert_span`
- `culvert_area_m2`
- `culvert_barrels`
- `culvert_slope`
- `inlet_invert_elev`
- `outlet_invert_elev`
- `entrance_loss_k`
- `exit_loss_k`
- `embankment_enabled`
- `embankment_crest_elev`
- `embankment_overflow_width`
- `embankment_weir_coeff`

## Not implemented yet

This is still not a full HEC-RAS/FHWA culvert system and it is not yet a dedicated device-kernel culvert kernel. The main remaining work is:

- richer culvert geometry families and inlet-control code coverage validation
- richer custom form behavior such as FHWA code lookup tables, conditional widget visibility, and validation messages
- a dedicated culvert-focused results viewer if the generic coupling viewer becomes too dense
- explicit culvert diagnostics/output tables for regime, inlet-control limit, outlet-control limit, embankment overflow, and barrel utilization
- validation against FHWA/HEC-RAS benchmark cases

## Recommended next implementation phase

1. Tighten parity against FHWA/HEC-RAS reference datasets for outlet-control and regime transitions, now that the compiled helper no longer depends on Python routine calls.
2. Split culvert forensics into a dedicated GeoPackage table if we want cleaner reporting than the generic `swe2d_coupling_results` stream.
3. Add benchmark validation against FHWA/HEC-RAS culvert cases before relying on the native helper for production calibration.
