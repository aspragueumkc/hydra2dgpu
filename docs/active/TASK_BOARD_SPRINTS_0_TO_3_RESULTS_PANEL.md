# Task Board: Sprint 0-3 Results Panel Modernization

Status legend:
- [ ] not started
- [~] in progress
- [x] done
- [!] blocked

## Sprint 0 Baseline (Completed)
- [x] Create `swe2d_results_queries.py`.
- [x] Create `swe2d_results_panel.py`.
- [x] Wire panel into workbench.
- [x] Validate compile/smoke test.

Definition of Done:
1. Panel opens from workbench.
2. Query utilities import and run.
3. Smoke tests pass without regressions.

## Sprint 1: Dockable Multi-Run Panel
Blocked By: Sprint 0 baseline completion.
- [x] Multi-run discovery and checklist selection.
- [x] Shared timestep synchronization across selected runs.
- [x] Time-series overlay plotting.
- [x] Profile overlay plotting at shared timestep.
- [~] Panel state persistence/restore (implemented; pending manual reopen validation).

Definition of Done:
1. [x] Two or more runs can be compared simultaneously.
2. [x] Shared timestep updates all selected runs.
3. [~] State restore is stable after reopen.

## Sprint 2: Map Tool + Animation Core
Blocked By: Sprint 1 synchronized timestep model.
- [x] Implement sample line draw map tool.
- [x] Persist and resample line geometry on draw completion.
- [x] Add animation controller (play/pause/step/speed).
- [x] Synchronize animation state with panel timestep.
- [~] Ensure responsive frame updates on representative projects.

Definition of Done:
1. [x] Drawn lines feed the results workflow.
2. [x] Animation controls are functional and synchronized.
3. [~] Playback quality meets minimum usability targets.

## Sprint 3: Velocity + Structures + EGL
Blocked By: Sprint 2 animation events and timestep sync.
- [x] Add velocity vector overlay layer generation.
- [x] Add vector visibility and density controls.
- [x] Add vector caching for interactive performance.
- [x] Add structure overlays in profile view.
- [x] Add EGL toggle and rendering.

Definition of Done:
1. Velocity overlays animate correctly.
2. Structure overlays are rendered at valid profile locations.
3. EGL rendering is physically consistent and toggleable.

## Agent Handoff Protocol
Before handoff, each agent updates:
1. Tasks touched and status changes.
2. Validation commands executed and outcomes.
3. Files changed and rationale.
4. Open risks/blockers and next immediate task.

## References
- `docs/SPRINT_1_TASK_CARD.md`
- `docs/SPRINT_2_TASK_CARD.md`
- `docs/SPRINT_3_TASK_CARD.md`
- `docs/SPRINTS_0_TO_3_PROGRESS_SUMMARY.md`
