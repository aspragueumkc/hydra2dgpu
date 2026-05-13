# Sprint 3 Task Card: Velocity Vectors + Structures + EGL

## Objective
Add hydraulic interpretation overlays by rendering animated velocity vectors on the map and structure/EGL context in profile views.

## Scope In
- Velocity vector layer generation at current timestep.
- Vector density controls and caching policy.
- Profile structure overlays (culverts/inlets/outfalls/related assets where available).
- Optional Energy Grade Line (EGL) computation and display.

## Scope Out
- PDF/HTML reporting and export wizard.
- 3D visualization workflows.
- Non-results-panel UI refactors.

## Implementation Tasks
- [x] Implement velocity layer builder for run+timestep snapshots.
- [x] Add vector style mapping (direction, magnitude, color ramp).
- [x] Add vector-density control and visibility toggle in UI.
- [x] Add timestep-synced velocity cache (LRU or equivalent).
- [x] Implement structure query path for profile overlays.
- [x] Render structure glyphs/annotations at profile station/elevation.
- [x] Implement EGL toggle and computation (`EGL = WSE + V^2/(2g)`).
- [x] Add defensive fallbacks when structure metadata is incomplete.

## Current Sprint 3 Evidence
- Added `swe2d_velocity_layer.py` with mesh snapshot query, LRU cache, vector build, and speed-to-style mapping.
- Added `load_structure_flows_at_time(...)` in `swe2d_results_queries.py` against `swe2d_coupling_results`.
- Added profile EGL mode and structure overlay toggle in `swe2d_results_panel.py`.
- Added panel controls for velocity visibility/density/min-speed and connected them to workbench map refresh.
- Wired a live `SWE2D_Velocity_Vectors` memory layer update path in `swe2d_workbench_qt.py` using mesh snapshots.
- Refined structure overlays to use GeoPackage structure geometry intersections with the active sample line and crest elevation for station-aware profile placement.

## Dependencies
- Sprint 2 animation controller and synchronized timestep events.
- Existing coupling/structure result persistence tables where present.

## Acceptance Criteria
1. Velocity vectors update correctly with animation timestep.
2. Vector density control changes layer clutter/performance as expected.
3. Structure overlays appear at consistent, interpretable profile positions.
4. EGL toggle renders/removes dashed EGL line without side effects.
5. Overlay refresh remains responsive with caching enabled.

## Validation Commands / Manual Checks
1. `python3 -m py_compile swe2d_velocity_layer.py swe2d_results_queries.py swe2d_results_panel.py swe2d_workbench_qt.py`
2. Toggle vectors on/off while animating and verify synchronization.
3. Adjust density control and confirm expected redraw behavior.
4. Enable structures and verify profile glyph placement.
5. Toggle EGL and verify correct curve behavior above WSE where velocity is non-zero.

## Risks and Mitigations
- Risk: Per-frame layer rebuild cost too high.
  Mitigation: Cache by `(run_id, timestep, density)` with bounded memory.
- Risk: Missing/inconsistent structure metadata.
  Mitigation: Graceful no-data rendering and explicit warning logs.
- Risk: Unit mismatch for EGL calculation.
  Mitigation: Reuse existing unit conversion helpers and test with known cases.

## Handoff Notes
- Sprint 4 export/report should consume already-rendered datasets (not recompute heavy overlays).
- Keep vector/structure query code modular so it can be reused by report generation.
