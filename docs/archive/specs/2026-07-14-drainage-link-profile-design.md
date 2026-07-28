---
type: spec
status: complete
created: 2026-07-14
completed: 2026-07-25
---

# Drainage Link Profile Viewer Enhancement

**Date:** 2026-07-14
**Status:** Design approved

## 1. Problem Statement

The drainage-link longitudinal profile in the Studio viewer currently:
- Renders a flat invert and crown because the per-cell pipe geometry from the C++ readback is not persisted.
- Uses a uniform blue fill for the water area between invert and water surface.
- Does not leverage the existing Fill/Colormap controls that the WSE+Bed line profile viewer already supports.

In addition, users want the profile to show a sloped pipe with velocity-based color shading, and eventually to superimpose 2D-surface bed/WSE sampling and an EGL line.

## 2. Goals

1. Make the drainage-link profile show correct sloped invert and crown geometry for both live and baked results.
2. Shade the water-fill area between invert and current water surface by a user-selectable metric (velocity by default), using the existing Fill + Colormap controls.
3. (Future) Superimpose 2D mesh bed elevation and surface water depth sampled along the link, plus an EGL line.

## 3. Constraints & Context

- The viewer is implemented in `swe2d/workbench/views/studio_viewer_profile_pg.py`.
- Per-cell pipe data is stored in `swe2d/results/data.py::_live_pipe_cell` and persisted to the GeoPackage table `swe2d_baked_pipe_cell_ts`.
- C++ readback already returns `cell_invert`, `cell_width`, `cell_height`, `cell_shape_type`, `cell_owner_link`, and `cell_sub_idx`.
- The live snapshot path only stores `cell_invert` and `cell_width` today; `cell_height` and `cell_shape_type` are dropped.
- The persistence layer only stores `times_blob` and `values_blob` per `(run_id, link_id, cell_sub_idx, metric)`; no geometry is persisted.
- MVP architecture: the View may read plain-data state from the data layer. Heavy computation and GPKG access belong in the service/data layer.

## 4. Design

### 4.1 Geometry persistence

Extend the pipe-cell data model so every per-cell record carries the geometry fields needed for drawing the pipe cross-section:

- `cell_invert` — elevation of the pipe invert at the cell center
- `cell_width` — pipe width (or diameter for circular conduits)
- `cell_height` — pipe height (rise for rectangular/elliptical conduits)
- `cell_shape_type` — shape discriminator (0 = circular, 1 = rectangular, 2 = elliptical, etc.)

**Live path:**
- `append_pipe_cell_snapshot()` in `swe2d/results/data.py` will store all four geometry fields on the first write for each cell.

**Persistence path:**
- Add the four columns to `swe2d_baked_pipe_cell_ts`.
- `persist_baked_pipe_cell_ts()` will serialize the geometry fields alongside the time series.
- `load_baked_pipe_cell_ts()` will return them.
- Backwards compatibility: if an existing GPKG lacks the geometry columns, fall back to zeros/defaults so the viewer still renders something rather than crashing.

### 4.2 Viewer rendering (Step 1)

In `swe2d/workbench/views/studio_viewer_profile_pg.py`, for element type `drainage_link`:

- Read geometry from `_live_pipe_cell` for each sub-cell.
- Compute crown elevation per sub-cell:
  - circular (`shape_type == 0`): `crown = invert + width`
  - rectangular/elliptical: `crown = invert + height`
- Draw invert and crown as **solid lines** (remove the dashed crown style).
- Compute the water surface per sub-cell as `invert + depth` at the current time.
- Read the fill metric from the Fill combo (`velocity_ms` by default; the combo will be enabled for `drainage_link` mode).
- Segment the water-fill polygon between adjacent sub-cells and color each segment by the normalized fill-metric value using the existing colormap lookup (`_cmap_color`).
- Keep the existing WSE+Bed line profile behavior unchanged.

### 4.3 Future additions (Step 2)

After Step 1 is verified, add:

- **EGL line**: from node depth/head records, compute `EGL = node_invert + depth + v²/(2g)` and draw a dashed line along the link.
- **2D mesh bed sample line**: sample the 2D mesh bed elevation along the link geometry and draw it.
- **2D surface WSE sample line**: sample the 2D surface water depth/WSE along the link geometry and draw it.

These samples will be computed in the service layer and delivered to the viewer as plain station/value arrays.

## 5. Data Flow

```
C++ readback
    ├── cell_invert, cell_width, cell_height, cell_shape_type
    │       └── append_pipe_cell_snapshot()  ──►  _live_pipe_cell[(link_id, sub_idx, metric)]
    │                                                   └── geometry fields
    │
    └── velocity, depth, flow, head
            └── _live_pipe_cell[(link_id, sub_idx, metric)]
                    └── times, values

Run finalization
    └── build_pipe_cell_items()  ──►  persist_baked_pipe_cell_ts()
                                          └── geometry columns + times/values BLOBs

Viewer refresh (drainage_link)
    └── load_baked_pipe_cell_ts()  ──►  _live_pipe_cell
                                          └── geometry + times/values
    └── profile render: invert, crown, water surface, velocity-shaded fill
```

## 6. UI/UX

- The existing Fill and Colormap combos become active in `drainage_link` mode.
- Fill defaults to `velocity_ms` for drainage links.
- The color legend is implicit via the combo labels; no new legend widget is added in Step 1.

## 7. Error Handling

- If a baked GPKG lacks geometry columns, the viewer falls back to default geometry (zeros) and shows a log warning.
- If a sub-cell metric is missing for a given timestep, the corresponding fill segment is skipped.
- If `max_cell_length` is still zero (separately fixed), the viewer renders a single-cell profile but still works.

## 8. Testing

- Unit test: `build_pipe_network_config()` reads `max_cell_length` from the layer (already verified).
- Unit test: `append_pipe_cell_snapshot()` stores geometry fields.
- Unit test: `persist_baked_pipe_cell_ts()` / `load_baked_pipe_cell_ts()` round-trip geometry.
- Manual verification: run a drainage simulation and confirm the profile shows sloped invert/crown and velocity-colored fill.

## 9. Out of Scope

- Step 2 (EGL, 2D bed/WSE superposition) is not part of this implementation plan; it will be planned separately after Step 1 is complete.
- Changing the pipe cell subdivision logic (that was fixed separately by reading `max_cell_length`).
- Adding new colormaps beyond the existing five.

## 10. Approval

Design approved by user via visual companion review on 2026-07-14.
