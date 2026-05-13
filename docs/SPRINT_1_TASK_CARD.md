# Sprint 1 Task Card: Dockable Multi-Run Results Panel

## Objective
Deliver a dockable results panel that supports simultaneous comparison of multiple runs, synchronized time navigation, and project-state persistence.

## Scope In
- Dockable panel lifecycle and workspace integration.
- Multi-run discovery and selection from GeoPackage results tables.
- Time-Series and Profile tabs with synchronized time cursor/slider.
- Per-run color assignment and overlaid plotting.
- Persist/restore selected runs, active tab, and current timestep.

## Scope Out
- Map draw tool implementation.
- Animation playback controls beyond basic synchronized time navigation.
- Velocity vector overlays and structure glyph rendering.
- Report/PDF export workflows.

## Implementation Tasks
- [x] Create/extend dockable results panel class for multi-run operation.
- [x] Implement run discovery query path for `swe2d_line_results_ts_*` tables.
- [x] Implement run checklist UI (multi-select with stable color mapping).
- [x] Implement synchronized timestep controls across all selected runs.
- [x] Render overlaid time-series plots for selected runs.
- [x] Render profile tab for selected runs at shared timestep.
- [x] Persist panel state via project settings.
- [x] Add user-facing fallback messaging when runs/tables are missing.

## Current Validation Status
- [x] `python3 -m py_compile swe2d_results_queries.py swe2d_results_panel.py swe2d_workbench_qt.py`
- [x] Results panel integration path compiles and workbench hook points resolve.
- [~] Full manual UI acceptance walk-through (open panel, compare two runs, move timestep, save/reopen project) pending interactive confirmation in QGIS session.

## Dependencies
- Sprint 0 completed baseline panel integration and query utilities.
- Existing `swe2d_results_queries.py` and `swe2d_results_panel.py` are available.

## Acceptance Criteria
1. Two or more runs can be checked simultaneously and render together.
2. Shared timestep control updates all selected runs in both tabs.
3. Per-run colors are stable across refresh/reopen.
4. Panel state restores correctly after project reload.
5. No crash on empty/missing results tables.

## Validation Commands / Manual Checks
1. `python3 -m py_compile swe2d_results_queries.py swe2d_results_panel.py swe2d_workbench_qt.py`
2. Launch plugin and open results panel from workbench.
3. Select two runs and verify overlaid time-series lines.
4. Move timestep control and verify profile updates for all selected runs.
5. Save/reopen project and verify state restoration.

## Risks and Mitigations
- Risk: Large result tables cause sluggish redraws.
  Mitigation: Cache per-run line/timestep arrays and throttle refresh.
- Risk: Run table naming inconsistencies.
  Mitigation: Defensive discovery logic with explicit log warnings.
- Risk: State restore applies stale run ids.
  Mitigation: Validate restored run ids against current discovery list.

## Handoff Notes
- Next sprint consumes synchronized timestep state as the animation source.
- Keep plotting/data access split (UI vs query layer) for testability.
- Preserve backward compatibility with existing modal viewer until full migration is approved.
- Sprint 1 implementation is code-complete; remaining work is manual acceptance validation and any resulting polish.
