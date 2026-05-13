# TASK BOARD: GUI Refactor + Hydrology Expansion

## Scope

- Add 2D spatial-temporal rainfall with Thiessen interpolation.
- Add NRCS CN infiltration/excess workflow for SWE2D (event mode).
- Add lumped hydrology template and reusable NRCS helper calculations.
- Refactor plugin UX so key workflows are menu-accessible from Backwater dropdown.
- Introduce independent documentation/help panels for the 4 major plugin components.

## Related Sprint 0-3 Results Panel Docs

- `docs/SPRINT_1_TASK_CARD.md`
- `docs/SPRINT_2_TASK_CARD.md`
- `docs/SPRINT_3_TASK_CARD.md`
- `docs/TASK_BOARD_SPRINTS_0_TO_3_RESULTS_PANEL.md`
- `docs/SPRINTS_0_TO_3_PROGRESS_SUMMARY.md`

## External Guidance Snapshot

Sources reviewed:
- QGIS PyQGIS Plugin Cookbook: plugin menu/action organization and help integration.
- HEC-HMS Technical Reference: SCS Curve Number equations and event-simulation notes.

Actionable guidance applied:
- Keep plugin actions grouped consistently under plugin menu entries.
- Expose help/documentation actions as first-class menu actions.
- Treat CN infiltration as event-based cumulative loss/excess method:
  - $S = \frac{25400}{CN} - 254$ (mm)
  - $I_a = 0.2S$
  - $P_e = \frac{(P-I_a)^2}{P+0.8S}$ for $P > I_a$, else $0$
  - incremental excess from cumulative difference.

## Delivery Phases

## Phase 1 (implemented in this cycle)

- [x] New rainfall/hydrology utility module:
  - Thiessen nearest-gage assignment.
  - Hyetograph parsing (intensity, incremental depth, cumulative depth modes).
  - Stateful NRCS CN incremental excess calculator.
  - Lumped helper functions (event runoff depth, composite CN, velocity-method Tc from segment velocities).
- [x] SWE2D run-loop integration:
  - Optional Thiessen + CN forcing from model layers.
  - Fallback to existing uniform rain rate when forcing layers are absent.
- [x] 2D model GeoPackage schema expanded:
  - swe2d_rain_gages (point)
  - swe2d_hyetographs (table)
  - swe2d_cn_zones (polygon)
- [x] Lumped hydrology template generator added:
  - lumped_subbasins
  - lumped_flow_paths
  - lumped_rain_events

## Phase 2 (next)

- [ ] Add dedicated 2D rainfall panel in SWE2D workbench:
  - Layer validation badges (gage count, hyetograph linkage completeness).
  - Unit and value-type quick validators.
  - Preview of Thiessen assignment on mesh.
- [ ] Add a Lumped Hydrology dialog:
  - Compute Tc from flow path table.
  - Compute event runoff and excess hyetograph.
  - Export computed hydrograph rows back to GeoPackage.

## Phase 3 (GUI architecture refactor)

- [ ] Split workflow UI into component-focused controllers:
  - steady 1D
  - unsteady 1D
  - SWE2D
  - lumped hydrology
- [ ] Keep a single menu router in Backwater dropdown that can launch each component and its docs panel directly.
- [ ] Convert heavy tabs into lazy-loaded panes to reduce startup cost and simplify state management.

## Backwater Dropdown Requirement

- [ ] Every major action must be directly reachable from Backwater dropdown with no hidden-only tab dependency.
- [ ] Minimum action set:
  - Open each component workspace.
  - Create/load each model template type.
  - Run/preview/validate for each component.
  - Open component docs/help panel.

## Independent Doc Panels Requirement

Target four doc panels:
- [ ] 1D Steady Docs
- [ ] 1D Unsteady Docs
- [ ] 2D SWE Docs
- [ ] Lumped Hydrology Docs

Each panel should include:
- Inputs schema checklist
- Units conventions
- Common failure modes + fixes
- Formula reference section
- Quick links to example datasets

## Risks and Mitigations

- Risk: layer-schema drift breaks forcing discovery.
  - Mitigation: tolerant field-name matching + explicit missing-field logs.
- Risk: forcing source applied with bad units.
  - Mitigation: enforce units fields and add parser warnings for unknown unit tokens.
- Risk: CN method misused for continuous simulations.
  - Mitigation: label as event-based in UI/docs and require explicit user acknowledgement for long runs.

## Validation Checklist

- [ ] New 2D GPKG includes all rainfall/CN layers.
- [ ] Thiessen mapping assigns every cell when at least one gage exists.
- [ ] CN forcing produces zero excess until $P > I_a$.
- [ ] Dry/wet masking behavior remains consistent with prior fixes.
- [ ] Existing mesh and export tests still pass.
