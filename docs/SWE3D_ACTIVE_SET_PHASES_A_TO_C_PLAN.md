# SWE3D Active-Set and Interface Hardening Plan (Phases A-C)

## Objective
Implement a GPU-first SWE3D upgrade that:

1. Solves only water/interface-relevant cells (plus a minimal halo),
2. Reduces interface-driven spurious momentum in the water phase,
3. Sharpens VOF transport near the interface,
4. Preserves and de-risks future two-way 2D-3D coupling.

## Coupling Compatibility Constraints (Non-Negotiable)

- Keep all 2D-3D contract data structures and kernels unchanged in API/shape.
- Do not remove or alter one-way and two-way coupling call order in the step loop.
- Ensure boundary cells participating in coupling remain active when they are wet/interface or in a one-cell interface halo.
- Keep feedback integrals (`phi * vof` weighted wet area, face-normal fluxes, pressure means) physically meaningful; avoid introducing inactive-cell artifacts at boundary faces.
- Add changes as opt-in runtime behavior where practical, with conservative defaults.

## Phase A: 3D Wet-Active Set (Water + Interface + Halo)

### Scope
- Add per-cell active mask in 3D patch state.
- Build active mask each 3D substep from `vof`, `phi`, and local neighbor context.
- Restrict predictor/projection/velocity correction to active cells.

### Active Criteria
- Water core: `vof >= alpha_wet`.
- Interface band: `alpha_gas < vof < alpha_wet`.
- Halo: any cell adjacent to interface band (6-neighbor stencil).
- Solid/void cells (`phi ~ 0`) remain inactive.

### Kernels touched
- `swe3d_single_phase_predictor_kernel`
- `swe3d_compute_pressure_rhs_kernel`
- `swe3d_pressure_jacobi_kernel`
- `swe3d_velocity_correction_kernel`
- `swe3d_vof_transport_upwind_kernel` (renamed/extended in Phase C)
- `swe2d_gpu_step_3d_single_phase_free_surface` orchestration

### Coupling notes
- Boundary forcing/feedback kernels remain unchanged.
- Active mask must not zero-out valid boundary wet/interface states used by coupling reductions.

## Phase B: Interface-Consistent Pressure/Velocity Closure

### Scope
- Introduce interface-aware face weighting in pressure RHS, Jacobi stencil, and velocity correction.
- Suppress gas-side contamination of water momentum updates.

### Method
- Compute face activity/weight from local `vof` and active mask.
- Use weighted divergence and weighted pressure Laplacian in projection.
- Use weighted pressure gradients in velocity correction.

### Expected effect
- Reduced parasitic interfacial velocities.
- Better water-phase momentum robustness near moving free surface.

### Coupling notes
- Preserve absolute pressure field availability for two-way feedback kernel.
- Do not force pressure to zero on active coupling boundary cells.

## Phase C: Interface-Focused Higher-Order VOF Transport

### Scope
- Upgrade from purely first-order upwind to bounded MUSCL-like face reconstruction for interface-relevant fluxes.
- Keep robust low-order behavior in bulk regions as fallback.

### Method
- Minmod-limited reconstruction on upwind side at faces.
- Apply higher-order reconstruction only where either side is in interface/near-interface active region.
- Preserve `0 <= vof <= phi` bounds.

### Coupling notes
- Keep boundary VOF BC semantics unchanged (`INFLOW`, `INFLOW_FLOW_RATE` use prescribed `bc_vof`).
- Preserve total mass trend needed for stable two-way exchange.

## Validation and Risk Controls

### Build/Runtime checks
- Rebuild native module after each phase.
- Run targeted 3D uncoupled validation tests.
- Run focused two-way coupling regression(s) already present in `tests/test_swe3d_uncoupled_validation.py`.

### Acceptance checks
- No API break in coupling contract upload/use paths.
- No regression in one-way/two-way coupling tests.
- Reduced interface velocity noise in quiescent/interface-focused cases.
- Stable runtime with finite diagnostics and bounded VOF.

## Implementation Order
1. Add runtime controls + patch active-mask storage.
2. Implement Phase A active-mask build + kernel gating.
3. Implement Phase B weighted projection/velocity correction.
4. Implement Phase C bounded MUSCL-like VOF transport near interface.
5. Build + run focused validations.
6. Adjust thresholds only if tests indicate instability.
