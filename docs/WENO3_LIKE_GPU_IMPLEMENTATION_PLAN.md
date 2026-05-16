# WENO3-like GPU Spatial Scheme Plan

## Objective

Add one new GPU-first spatial discretization mode (WENO3-like) to pair with higher-order temporal schemes (RK4/RK5 graph-safe), then validate accuracy/stability/performance on existing GPU-priority suites.

## Scope

In scope:

- Add selectable spatial scheme id 5 end-to-end (GUI -> Python -> native/CUDA).
- Implement a safe WENO3-like reconstruction branch in CUDA flux kernel.
- Extend GPU unstructured validation loops to include scheme 5.
- Keep existing well-balancing and wet/dry safeguards active.

Out of scope for first slice:

- CPU parity implementation for new scheme.
- Full face-flux persistence pipeline.
- New benchmark framework beyond existing tests/logging.

## Design Choice: WENO3-like (Safety-First)

Use a two-candidate nonlinear blend per side at each edge for reconstructed variables:

- Candidate A: Green-Gauss gradient extrapolation to edge midpoint.
- Candidate B: pair-centered midpoint state along the cell pair jump.

Compute WENO-style weights from local smoothness indicators:

- Smoothness for A: projection magnitude relative to parent cell value.
- Smoothness for B: pair jump magnitude.
- Nonlinear weights favor smooth candidate, damp oscillatory candidate near fronts.

Then enforce pair-bounds clamp to preserve monotonicity.

Variables reconstructed with this branch:

- eta = h + zb (using eta gradients currently stored in grad_hx/grad_hy)
- hu
- hv

Depth recovery remains:

- h = max(0, eta_rec - zb)

Existing safeguards kept:

- shallow-front fallback to first-order path
- hydrostatic reconstruction
- bed-slope correction
- front momentum damping
- momentum magnitude caps

## Concrete File Edits

1. Enum + GUI wiring

- `cpp/src/swe2d_solver.hpp`
  - Add `FV_WENO3_LIKE = 5` to `SWE2DSpatialScheme`.
- `swe2d_extensions.py`
  - Add `FV_WENO3_LIKE = 5` to `SpatialDiscretization`.
- `swe2d_workbench_qt.py`
  - Add dropdown option label/value for scheme 5 in `_RECONSTRUCTION_OPTIONS`.

2. CUDA reconstruction branch

- `cpp/src/swe2d_gpu.cu`
  - In `swe2d_flux_kernel`, add scheme id constant for WENO3-like.
  - Add `weno3_like_reconstruct(...)` lambda alongside existing TVD helper.
  - Branch: if `spatial_scheme == scheme_weno3`, use WENO3-like helper for eta/hu/hv; otherwise use existing TVD helper.

3. Tests

- `tests/test_swe2d_gpu_unstructured.py`
  - Extend `range(5)` loops to `range(6)` in:
    - `test_stability_all_schemes`
    - `test_well_balanced_all_schemes`

## Validation Plan

Primary GPU-first validation:

1. `tests/test_swe2d_gpu_unstructured.py`
   - Stability across schemes including 5.
   - Lake-at-rest well-balancing including 5.

2. `tests/test_swe2d_gpu_validation_perf.py`
   - Ensure no major regression in runtime sanity checks after integration.

Acceptance criteria for first slice:

- Scheme 5 is selectable from GUI and reaches CUDA path.
- No NaN/divergence in unstructured dam-break stability test.
- Lake-at-rest drift remains within existing tolerance envelope.
- No regressions for schemes 0..4.

## Performance Checkpoints

Use existing runtime logs and timing lines to compare:

- step wall ms
- step kernel ms
- GPU fraction

Compare scheme 5 against scheme 3/4 on representative runs.

## Risk Notes

- WENO-like blend may still be too diffusive or too sharp on some meshes.
- If oscillations appear, tighten clamp and/or increase candidate-B bias near strong jumps.
- If excessive diffusion appears, increase Candidate A linear weight in smooth regions.

## Follow-up Iterations

- Tune linear weights and epsilon/power in nonlinear weights.
- Evaluate adding edge-direction smoothness terms.
- Optionally persist face flux diagnostics for deeper reconstruction analysis.
