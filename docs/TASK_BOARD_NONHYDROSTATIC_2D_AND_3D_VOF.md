# Task Board: Non-Hydrostatic 2D + Coupled 3D VoF

Status legend:
- TODO: not started
- IN_PROGRESS: actively being worked
- BLOCKED: cannot proceed without prerequisite
- DONE: completed

## Epic A: Boundary Conditions and Inflow Controls

- DONE: A1. Change flow BC input semantics to total discharge Q in workbench UI.
- DONE: A2. Convert total Q to unit q before native solver calls.
- DONE: A3. Add progressive low-elevation edge activation option for inflow boundaries.
- DONE: A4. Add regression tests for total-Q conversion and progressive activation.
- TODO: A5. Add BC diagnostics logging (Q_total, active length, active edges, q_unit).
- TODO: A6. Add BC grouping field support in bc_line layer (`flow_group`) for independent inflow sources on one side.
- TODO: A7. Add unit labels for SI/US customary on BC controls (Q in m3/s or ft3/s).

## Epic B: Non-Hydrostatic 2D Solver Option

- TODO: B1. Add solver mode enum and UI selector (`Hydrostatic`, `Non-hydrostatic`).
- TODO: B2. Implement predictor step reuse from existing hydrostatic solver path.
- TODO: B3. Implement non-hydrostatic pressure correction equation (CPU).
- TODO: B4. Implement velocity/momentum correction and dry-cell safeguards.
- TODO: B5. Add non-hydrostatic benchmark tests (dispersive wave, hydraulic jump).
- TODO: B6. Add matrix-free PCG pressure solver with baseline preconditioner.
- TODO: B7. Add GPU kernels for NH pressure iterations and residual reductions.
- TODO: B8. Add performance/accuracy comparison report hydrostatic vs NH.

## Epic C: Structures in 2D (Bridge/Culvert/Weir)

- TODO: C1. Define structure schema (geometry, losses, control type, roughness, limits).
- TODO: C2. Implement internal-boundary discharge operator interface (`Q=f(headwater, tailwater, state)`).
- TODO: C3. Add weir/orifice formula module with calibration coefficients.
- TODO: C4. Add culvert operator (inlet/outlet control modes, transition logic).
- TODO: C5. Add bridge opening/contraction module with headloss parameterization.
- TODO: C6. Add structure unit tests against analytical and reference tables.

## Epic D: 3D VoF Solver (Hydraulic Structures Focus)

- TODO: D1. Finalize 3D discretization and data layout design note.
- TODO: D2. Implement Cartesian grid generation + geometry mask ingestion.
- TODO: D3. Implement VoF advection core.
- TODO: D4. Implement momentum predictor + pressure projection.
- TODO: D5. Add boundary condition suite for inflow/outflow/walls/free-surface handling.
- TODO: D6. Add GPU acceleration path for advection/projection kernels.
- TODO: D7. Validate against spillway and culvert canonical cases.

## Epic E: 2D-3D Coupling

- TODO: E1. Define interface data contract (fluxes, stages, momenta, timestamps).
- TODO: E2. Implement one-way 2D->3D forcing.
- TODO: E3. Implement one-way 3D->2D feedback (headloss/flux correction).
- TODO: E4. Implement two-way sub-iteration with relaxation controls.
- TODO: E5. Add conservation auditing (mass/momentum across interface).
- TODO: E6. Add coupled benchmark scenario and stability envelope tests.

## Epic F: Literature and Design Reviews

- IN_PROGRESS: F1. Curate HEC-RAS 2D BC behaviors relevant to inflow hydrographs and practical setup.
- IN_PROGRESS: F2. Curate ANUGA structure operator patterns and identify reusable abstractions.
- TODO: F3. Curate SRH-2D/SRH-3D boundary and structure modeling references.
- TODO: F4. Create decision memo on minimal-physics assumptions for riverine structure fidelity.
- TODO: F5. Create GPU scalability memo (target mesh sizes, memory limits, solver strategy).

## Epic G: Release/UX Integration

- TODO: G1. Add migration note in UI/docs: existing projects using unit q must be converted to total Q.
- TODO: G2. Add project-version flag and compatibility shim for legacy BC meaning.
- TODO: G3. Add user guide section with worked examples (1->1000 CFS hydrograph over sloped boundary).
- TODO: G4. Add troubleshooting section for over-concentrated inflow and activation tuning.

## Suggested Execution Order (Next 4 Weeks)

1. Week 1: A5, A6, G1, G2
2. Week 2: B1, B2, B3 (CPU NH prototype)
3. Week 3: C1, C2, C3 (2D structure operator framework)
4. Week 4: D1, D2, E1 (3D/coupling foundation)

## Open Decisions Needing Discussion

1. Should progressive activation be default-on for all flow BCs or only hydrographs?
2. Should active-length growth be linear with Q/Qpeak (current behavior) or use a rating-curve-based schedule?
3. Which pressure solver path for NH/3D should be first GPU target: PCG+Jacobi or geometric multigrid?
4. For 3D near structures, should first geometry method be immersed boundary or cut-cell?
