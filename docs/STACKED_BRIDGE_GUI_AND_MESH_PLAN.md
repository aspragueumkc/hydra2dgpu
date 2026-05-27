# Stacked Bridge GUI and Mesh Generation Plan

## Goal
Define a practical GUI schema for bridge geometry and a deterministic method to build a stacked bridge representation from the base SWE2D mesh.

This is for coupling and calibration in the current staged rollout. It does not replace the core 2D mesh.

## GUI geometry definition

Bridge geometry is defined in the existing hydraulic structures layer. Each bridge feature is still a line feature, but now includes additional bridge-specific fields.

Required fields for bridge rows:

- `structure_id`
- `structure_type` = `bridge`
- `crest_elev`

Core flow metadata already used by bridge coupling:

- `width`
- `height`
- `opening`
- `inlet_loss_k`
- `outlet_loss_k`

New stacked-geometry fields (all numeric):

- `stacked_enabled`: 0/1 switch for stacked bridge planning.
- `influence_width_m`: transverse corridor width centered on the bridge line.
- `upstream_buffer_m`: corridor extension upstream of the bridge line start.
- `downstream_buffer_m`: corridor extension downstream of the bridge line end.
- `deck_soffit_elev`: deck underside elevation.
- `deck_top_elev`: deck top elevation.
- `model_top_elev`: top of vertical modeling column for stacked planning.
- `under_layers`: number of vertical fluid layers below the deck soffit.
- `over_layers`: number of vertical fluid layers above the deck top.
- `pier_count`: number of equally spaced piers across the transverse width.
- `pier_width`: pier width.

Auto-populated from the feature geometry when config is built:

- `axis_x0`, `axis_y0`, `axis_x1`, `axis_y1`

These preserve the original bridge line endpoints in structure metadata so stacked plans can be regenerated consistently from saved runs/projects.

Notes:

- The line endpoints define the streamwise axis.
- `influence_width_m` defines the transverse footprint around that axis.
- Piers are represented as blocked transverse bands within that footprint.

## Base-mesh to stacked-mesh mapping algorithm

Implementation module: `swe2d/mesh/bridge_stacked_mesh.py`

Inputs:

- Base SWE2D mesh (`node_x`, `node_y`, and polygon/triangle topology).
- One or more bridge geometry specs from the structure layer.

Bridge specs can be assembled directly from `HydraulicStructureConfig` using `bridge_specs_from_structure_config(...)`.

Runtime helper: `swe2d/runtime/bridge_stacked_runtime.py`

- `build_bridge_stacked_plans_for_runtime(mesh_data, hydraulic_structures_cfg, log_fn)`
- used in run assembly to build and attach bridge stacked plans before stepping.
- `bridge_stacked_source_scale(plan)` derives a conservative source multiplier from
   the mean opening fraction and layer-role mix.
- `apply_bridge_stacked_source_weight(source_rate, plan)` applies that multiplier
   to the bridge source-term array before it is accumulated into the coupled source field.

Per bridge feature:

1. Compute cell centroids from the base mesh.
2. Build local bridge coordinates:
   - streamwise coordinate `s` along the bridge line,
   - transverse coordinate `n` orthogonal to the bridge line.
3. Select bridge corridor cells where:
   - `s in [-upstream_buffer_m, L + downstream_buffer_m]`,
   - `|n| <= influence_width_m / 2`.
4. Build vertical layer interfaces from elevations:
   - under-deck layers between `[0, deck_soffit_elev]`,
   - over-deck layers between `[deck_top_elev, model_top_elev]`.
5. Build pier bands across the transverse width and assign per-cell opening fraction:
   - `1.0` in open bands,
   - `0.0` in pier bands.
6. Return a `BridgeStackedPlan` with:
   - selected base cells,
   - local coordinates (`s`, `n`),
   - layer interfaces and role tags,
   - opening fraction mask,
   - effective opening width.

## Runtime use

Current runtime coupling uses hydraulic-structure source terms. The stacked plan is intended to be consumed by the next coupling stage so bridge losses and pressure-split logic can be applied over selected corridor cells with layer-aware weighting.

## Validation targets

Short-term tests:

- Corridor selection consistency when rotating the bridge line.
- Layer count and bounds from deck/model elevations.
- Pier band blocking and effective opening width.

Implemented unit tests:

- `tests/test_bridge_stacked_mesh.py`
- `tests/test_bridge_stacked_runtime.py`

## GUI validation behavior

When a bridge row has `stacked_enabled > 0`, the config builder validates required stacked fields and value consistency:

- required fields present and numeric,
- `influence_width_m > 0`,
- `deck_top_elev > deck_soffit_elev`,
- `model_top_elev > deck_top_elev`.

If validation fails, stacked mode is disabled for that bridge row and a run log warning is emitted.

## Source weighting rule

The current runtime bridge coupling uses a uniform scaling factor per bridge plan:

- `opening_fraction` contributes the corridor openness term,
- `layer_role` contributes the under-deck vs over-deck band balance,
- the result is used as a conservative attenuation factor on the bridge source array.

This keeps the native bridge helper interface unchanged while still making the stacked geometry visible to the coupled source assembly.
