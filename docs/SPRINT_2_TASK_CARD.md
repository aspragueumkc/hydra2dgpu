# Sprint 2 Task Card: Map Canvas + Animation Core

## Objective
Enable map-first interaction and timeline playback by adding sample-line drawing tools and an animation controller synchronized with the results panel.

## Scope In
- `QgsMapTool` for drawing sample lines on map canvas.
- Animation controller with play/pause, step, and speed controls.
- Timeline synchronization between animation state and profile/time-series views.
- UI controls for frame/time display and manual timestep selection.

## Scope Out
- Velocity vectors and structure overlays.
- Advanced report/export generation.
- Full workflow redesign of unrelated workbench tabs.

## Implementation Tasks
- [x] Implement `SWE2DLineDrawTool` with rubber-band visual feedback.
- [x] Wire draw tool activation/deactivation from workbench controls.
- [x] On line completion, create/update sample-line features and trigger resampling.
- [x] Implement `ResultsAnimationController` with timer-driven playback.
- [x] Add play/pause, step forward/back, and speed controls to results panel.
- [x] Synchronize animation index with panel timestep state.
- [x] Add adaptive playback guardrails for slow frame renders.
- [x] Persist animation state (current index/speed/playback mode as needed).

## Dependencies
- Sprint 1 multi-run synchronized timestep model.
- Existing sample-line persistence workflow in workbench.

## Acceptance Criteria
1. User can draw a line on map and obtain refreshed line results.
2. Play/pause/step controls function without desynchronization.
3. Slider/time label reflect current animation frame correctly.
4. Animation updates selected runs consistently.
5. Playback remains usable (target >= 20 fps on representative datasets).

## Validation Commands / Manual Checks
1. `python3 -m py_compile swe2d_results_panel.py swe2d_results_animation.py swe2d_map_tools.py swe2d_workbench_qt.py`
2. Activate draw tool, draw a sample line, confirm generated/updated sampling.
3. Start animation and verify profile/time marker progression.
4. Change speed setting and verify playback rate changes.
5. Step forward/back and verify exact timestep movement.

## Current Validation Evidence
- [x] `python3 -m py_compile swe2d_results_panel.py swe2d_results_animation.py swe2d_map_tools.py swe2d_workbench_qt.py swe2d_results_queries.py` (pass; no output)
- [x] `python3 -m unittest tests.test_swe2d_drainage_structures -v` (pass; 33 tests)
- [~] Manual QGIS UI checks for draw-tool interaction and playback ergonomics remain pending in a live QGIS session.

## Risks and Mitigations
- Risk: Signal loops between slider and animation timer.
  Mitigation: Guarded update flags and single source-of-truth index.
- Risk: Excessive redraw cost per frame.
  Mitigation: Cache/static artists and partial redraw where practical.
- Risk: Tool state conflicts with other map tools.
  Mitigation: Explicit activation/deactivation lifecycle and restoration.

## Handoff Notes
- Sprint 3 should consume the same animation timestep signal for velocity/structure overlays.
- Keep controller logic isolated from plotting widgets to simplify unit testing.
