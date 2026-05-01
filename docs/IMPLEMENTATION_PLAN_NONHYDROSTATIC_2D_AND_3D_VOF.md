# Implementation Plan: Non-Hydrostatic 2D SWE Option + Coupled 3D VoF Solver

## 1. Goals

1. Extend the existing 2D SWE solver with a selectable non-hydrostatic equation set (same mesh and BC workflow).
2. Add a 3D single-fluid VoF solver focused on hydraulic structures (spillways, culverts, bridges, weirs).
3. Couple 2D and 3D so river-scale routing remains cheap (2D) while local structure physics are resolved (3D).
4. Preserve GPU scalability and practical runtime for engineering workflows.

## 2. What We Started Now (Boundary Condition Foundation)

Completed in this iteration:
- Flow BC inputs now use total discharge Q (not unit q) in the 2D workbench UI and hydrographs.
- Runtime converts each flow BC source from total Q to solver unit discharge q by dividing over active boundary-edge length.
- Added progressive low-elevation activation option:
  - as Q increases, inflow is applied across more boundary edges,
  - edges are activated in ascending bed-elevation order.
- Regression tests added for total-Q conversion and progressive activation.

Rationale:
- This behavior is aligned with practical 2D BC usage where users often provide total inflow hydrographs and expect effective conveyance width to grow with flow.

## 3. External Guidance Used

HEC-RAS 2D (documentation reviewed):
- Uses Flow Hydrograph as external 2D boundary input with positive flow into domain.
- Supports boundary-driven inflow/outflow behavior tied to hydraulic state.

ANUGA (source reviewed):
- Structure operators (boyd box/pipe, weir-orifice, internal boundary) use discharge routines, energy/stage driving head logic, smoothing, and limits.
- Internal boundary operator provides a reusable pattern for coupling via discharge functions and sign reversal.

SRH-2D/SRH-3D:
- SRH source/doc extraction should be completed in a follow-up research pass; current plan keeps abstractions compatible with SRH-style inflow/discharge and structure controls.

## 4. Architecture Strategy

### 4.1 Solver Modes

Add a mode selector in native backend:
- `hydrostatic_2d` (current path)
- `nonhydrostatic_2d` (new pressure-correction path)

Maintain shared components:
- mesh topology and metric terms
- wetting/drying controls
- friction and source terms
- BC parser and runtime BC update hooks

### 4.2 3D Solver Scope (MVP)

MVP 3D scope:
- incompressible single-fluid Navier-Stokes with VoF free surface
- block-structured Cartesian grid (for GPU simplicity)
- local 3D domains attached to selected structure zones only
- RANS turbulence closure deferred (start laminar/eddy-viscosity option)

Out of scope for MVP:
- fully unstructured 3D mesh
- two-way sediment transport
- full air-phase dynamics (single-fluid VoF only)

### 4.3 Coupling Concept

Use partitioned two-way coupling at interface boundaries:
- 2D -> 3D provides stage/discharge/momentum targets at interface strips.
- 3D -> 2D returns integrated flux and head loss corrections.
- Sub-iterations per macro time step (1..N) with relaxation.

Conservation requirements:
- exact mass flux matching across interfaces each macro step
- bounded momentum exchange with damping for stability

## 5. Non-Hydrostatic 2D Equation Plan

## 5.1 Equations

Start from depth-averaged SWE and add non-hydrostatic pressure correction:
- predictor step with hydrostatic fluxes and sources
- Poisson-like solve for non-hydrostatic pressure (or pressure increment)
- velocity/momentum correction

Discrete approach:
- colocated depth/momentum storage (reuse current)
- face-normal pressure gradients via compact stencil
- implicit or semi-implicit pressure solve (CG/PCG)

## 5.2 Numerics and Stability

- keep HLLC/hydrostatic flux path for convective terms
- non-hydrostatic correction can be activated by local slope/Froude/depth criteria later
- initial implementation: global on/off for clarity

## 5.3 GPU Considerations

- pressure solve dominates; use matrix-free PCG with Jacobi or block-Jacobi preconditioner first
- avoid sparse assembly where possible (stencil operator kernel)
- overlap reductions and vector ops with existing CUDA streams

## 6. 3D VoF Solver Plan

## 6.1 Core Numerics

- finite-volume on Cartesian cells
- projection method for incompressibility
- geometric or compressive VoF advection
- pressure Poisson solve each time step

## 6.2 Hydraulic-Structure Focused Simplifications

- fixed geometry (no FSI)
- rigid bed/structure
- isothermal, incompressible
- optional rough-wall law
- optional porous-headloss region for coarse culvert racks/piers

These assumptions maximize speed while preserving key structure hydraulics.

## 6.3 Structure Modeling Path

Stage A:
- immersed-boundary or cut-cell solids for spillways and bridge decks

Stage B:
- 1D/2D structure operators coupled to 3D near-field patches
- tabulated local loss relations where full 3D is not needed

## 7. Bridges/Culverts/Weirs Implementation Roadmap

Near-term (within current 2D framework):
- Introduce internal-boundary discharge operators inspired by ANUGA patterns:
  - direction-aware Q function
  - optional smoothing timescale
  - energy/stage based control variants
- Add bridge/culvert parameter schemas (geometry + losses + controls)

Mid-term:
- Hybrid approach:
  - 2D internal-boundary operator for routine runs
  - optional embedded 3D patch for high-fidelity runs

## 8. Milestones

M0 (done):
- Total-Q boundary input + progressive low-edge activation.

M1:
- Non-hydrostatic 2D predictor-corrector on CPU, unit tests on dispersive benchmarks.

M2:
- Non-hydrostatic 2D GPU path + performance parity tests vs hydrostatic mode.

M3:
- 3D VoF prototype on Cartesian mesh with static spillway benchmark.

M4:
- One-way 2D->3D coupling (boundary forcing only).

M5:
- Two-way conservative 2D<->3D coupling with relaxation controls.

M6:
- Structure library v1 (culvert/bridge/weir templates + calibration workflows).

## 9. Validation and Benchmark Plan

2D non-hydrostatic:
- solitary wave / dispersive propagation
- hydraulic jump location and sequent depth checks
- comparison against hydrostatic mode where appropriate

3D VoF:
- broad-crested weir nappe profile
- culvert barrel pressurization transitions
- bridge deck overtopping case

Coupled 2D/3D:
- floodplain + bridge opening test with mass closure and stage continuity checks

Acceptance metrics:
- mass error < 0.5% over scenario duration
- stable coupling without unphysical oscillations
- GPU speedup over CPU for target cell counts

## 10. Risks and Mitigations

1. Pressure solve cost dominates:
- Mitigation: matrix-free PCG, multigrid evaluation, adaptive local NH regions.

2. Coupling instability:
- Mitigation: under-relaxation, sub-iterations, interface flux limiting.

3. Geometry complexity in 3D:
- Mitigation: start with Cartesian + immersed/cut-cell only.

4. User workflow complexity:
- Mitigation: staged UI exposure (basic/advanced), defaults tuned for stability.

## 11. Immediate Next Development Steps

1. Add BC diagnostics panel output:
- total Q in,
- active boundary length,
- computed unit q,
- number of active edges.

2. Add BC-layer grouping identifier support (e.g., `flow_group`) so multiple inflow lines on same side can remain independent total-Q sources.

3. Start non-hydrostatic branch skeleton in native backend:
- new mode enum,
- pressure workspace buffers,
- predictor/corrector function hooks,
- baseline CPU test case.

4. Start 3D solver design doc v0 with data layout options (AoS vs SoA, tile/block sizing, pressure solver choice).
