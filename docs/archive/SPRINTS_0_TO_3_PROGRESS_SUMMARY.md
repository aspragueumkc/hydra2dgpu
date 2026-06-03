# Sprints 0-3 Progress Summary

## Completed So Far
- Sprint 0 baseline was completed:
  - `swe2d_results_queries.py` created.
  - `swe2d_results_panel.py` created.
  - Panel wired into workbench.
  - Compile/smoke checks validated.
- Sprint 1 implementation scope is now largely complete in code:
  - Multi-run discovery + run checklist selection.
  - Shared timestep synchronization and slider/controller wiring.
  - Overlaid Time-Series and Profile rendering.
  - Project-state persist/restore hooks.
  - Fallback messaging for missing/no-data cases.
- Runtime diagnostics stabilization work (Cmax/WSEres continuity) was prepared and validated in session flow; final persistence extension remains part of Sprint 3 execution scope.
- Sprint 2 core implementation is now in place:
  - Map-canvas sample line draw tool with workbench activation wiring.
  - Draw completion persistence plus best-effort latest-run resampling refresh.
  - Animation controller with play/pause/step/speed and timestep synchronization.
  - Animation state persistence for frame/speed/play mode.
  - Adaptive playback interval guardrails for slow render loops.

## In Progress
- Sprint 1 manual acceptance validation in interactive QGIS session (save/reopen state verification).
- Sprint 2 representative-project playback performance validation (usability target checks).
- Sprint 3 implementation kickoff:
  - EGL profile toggle/computation integrated in results panel.
  - Structure-flow query path plus station-aware profile placement from structure/sample-line geometry added.
  - Velocity overlay helper module and LRU snapshot cache scaffolded.
  - Velocity overlay visibility/density/min-speed controls wired to a live map vectors layer.

## Not Started
- Sprint 4 export/report workflows.

## Evidence Table
| Claim | Status | Evidence Source | Confidence |
|---|---|---|---|
| Sprint 0 query module exists | Completed | workspace file `swe2d_results_queries.py` | Confirmed |
| Sprint 0 panel module exists | Completed | workspace file `swe2d_results_panel.py` | Confirmed |
| Workbench wiring was completed in sprint 0 scope | Completed | session todo list marked complete | Confirmed |
| Sprint 1 multi-run panel features are implemented | Completed | `swe2d_results_panel.py`, `swe2d_results_queries.py`, `swe2d_workbench_qt.py` | Confirmed |
| Sprint 1 panel-state restore is production-ready | In Progress | restore path implemented; interactive reopen validation pending | Partially confirmed |
| Sprint 1-3 scoped and ready for handoff | Completed | docs task cards + board in this docs package | Confirmed |
| Runtime metrics GPKG persistence already landed | Pending | prior session notes indicated draft/undo cycle | Assumed |

## Next 5 Executable Actions
1. Execute interactive QGIS acceptance checks for Sprint 1/Sprint 2 pending manual items.
2. Wire velocity vectors into an actual map overlay layer synchronized to current frame.
3. Add vector visibility + density controls to the panel UI and persist their state.
4. Improve structure glyph placement to true station/elevation coordinates where metadata allows.
5. Add targeted tests for new Sprint 3 query and EGL profile behavior.

## Notes on Confidence Labels
- Confirmed: directly verified in workspace files or explicit completed checklist.
- Assumed: inferred from prior session context but requires direct file-level verification before marking done in a release note.
