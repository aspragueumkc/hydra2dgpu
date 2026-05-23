# SWE3D Boundary-Condition Hardening Plan (Uncoupled First, Coupled-Compatible)

## Goal
Deliver stable, physically consistent boundary conditions for the single-fluid free-surface SWE3D path in both:
- uncoupled mode (GUI/env-driven face BCs), and
- coupled mode (2D-3D interface exchange).

This plan replaces heuristic outlet/free-surface behavior with explicit boundary operators that are numerically robust under strong inflow/outflow forcing.

## Current Behavior Snapshot (As Implemented)

### Where BC state comes from
- Face BC definitions are stored in `SWE3DCartesianPatchDesc` (`bc_mode`, `bc_u/v/w`, `bc_q`, `bc_vof`, `bc_p`).
- Sources:
  - env load at solver construction: `cpp/src/swe2d_solver.cpp` (`swe3d_load_face_bc_from_env`),
  - runtime uploads from GUI/backend: `swe2d/runtime/backend.py` -> `swe2d_set_3d_patch_face_bc` -> `cpp/src/swe2d_gpu.cu`.

### What gets enforced each step
- Velocity enforcement kernels:
  - wall no-slip: `swe3d_apply_wall_noslip_kernel`,
  - inflow/inflow-Q velocity imposition: `swe3d_apply_face_inflow_bc_kernel`.
- Transport uses boundary face flux helper `swe3d_boundary_face_velocity_component` and `swe3d_boundary_face_vof`.
- Pressure projection has a special zmax free-surface pressure band only.

### Known weakness points
- `OUTFLOW` is passive (`return inside`) rather than non-reflecting.
- `FREE_SURFACE` zmax uses an explicit vent bias heuristic (`+1.0`) in boundary velocity helper.
- Side-face open boundary pressure behavior is not explicitly constrained in projection.
- `INFLOW_FLOW_RATE` uses total face area normalization, not dynamic wet/open area.

## Target End-State
A single BC framework used by both uncoupled and coupled runtimes:
1. Inflow (velocity and volumetric forms) with wet-area-aware normalization.
2. Open/outflow characteristic treatment (non-reflecting by default).
3. Free-surface pressure boundary as explicit projection constraint policy.
4. One BC target provider interface with two backends:
   - uncoupled provider: GUI/env,
   - coupled provider: interface-contract/exchange outputs.

## Phase U1: Uncoupled BC Hardening (Primary)

### U1.1 Add explicit BC policies and runtime controls

#### File changes
- `cpp/src/swe2d_gpu.cuh`
- `cpp/src/swe2d_gpu.cu`
- `swe2d/runtime/backend.py`

#### Additions
- New runtime controls in `SWE3DRuntimeControls` and env parsing:
  - `BACKWATER_SWE3D_OUTFLOW_POLICY` (0=legacy_passive, 1=characteristic_nonreflecting),
  - `BACKWATER_SWE3D_FREE_SURFACE_VENT_BIAS` (default `0.0` in hardened mode, compatibility option to legacy behavior),
  - `BACKWATER_SWE3D_Q_INFLOW_AREA_POLICY` (0=total_face_area, 1=dynamic_wet_open_area),
  - `BACKWATER_SWE3D_OPEN_BC_DAMPING` (optional mild damping near open boundaries for stabilization),
  - `BACKWATER_SWE3D_PROJECTION_BOUNDARY_POLICY` (0=legacy_zmax_only, 1=face-aware open/free-surface constraints).

- Backend helper expansion in `configure_swe3d_runtime(...)` for the above controls.

### U1.2 Replace volumetric-Q normalization with dynamic active face area

#### Core change
For `INFLOW_FLOW_RATE`, replace `q / patch_face_total_area` with `q / active_face_area` where:
- `active_face_area` uses local open-area tensors (`ax/ay/az`) and wet activity (`vof`, active mask),
- floor with epsilon to avoid division by zero,
- fallback policy keeps legacy total-area option for compatibility.

#### File changes
- `cpp/src/swe2d_gpu.cu`

#### New/updated kernels/helpers
- Add per-face area reduction helper (device reduction or two-pass accumulation).
- Update `swe3d_boundary_face_velocity_component(...)` to consume per-face effective area context.

### U1.3 Non-reflecting outflow/open boundary operator

#### Core change
Implement open boundary normal-velocity treatment that avoids hard reflection:
- characteristic-like copy/extrapolation for outgoing information,
- suppress incoming spurious waves,
- keep tangential behavior minimally constrained unless configured.

#### File changes
- `cpp/src/swe2d_gpu.cu`

#### New kernel
- `swe3d_apply_open_boundary_bc_kernel(...)`

#### Step-order integration
In `swe2d_gpu_step_3d_single_phase_free_surface(...)` substep loop:
1. predictor
2. wall BC
3. inflow BC
4. open/outflow BC
5. pressure rhs + projection
6. velocity correction
7. wall BC
8. inflow BC
9. open/outflow BC
10. transport

### U1.4 Free-surface pressure boundary policy cleanup

#### Core change
- Remove hard-coded venting behavior from velocity helper path.
- Keep free-surface pressure target handling in projection path as the authoritative mechanism.
- Extend pressure boundary handling beyond zmax-only in hardened mode where applicable.

#### File changes
- `cpp/src/swe2d_gpu.cu`

#### Notes
- Retain legacy policy toggle for backward compatibility in short synthetic windows.

### U1.5 Open-boundary diagnostics and guardrails

#### Add diagnostics to step diag
- open boundary flux summary (per face),
- effective inflow area and equivalent imposed `vn`,
- boundary reflection indicator proxy (in/out flux sign oscillation metric),
- per-face pressure clamp hit counts (if enabled).

#### File changes
- `cpp/src/swe2d_gpu.cu`
- `cpp/src/swe2d_solver.hpp` / bindings if needed for surfaced metrics.

## Phase U2: Uncoupled Validation and Regression Gates

### Add focused tests
- Extend `tests/test_swe3d_uncoupled_validation.py` with:
  - `test_open_boundary_nonreflecting_stability_under_q_inflow`,
  - `test_q_inflow_dynamic_wet_area_normalization`,
  - `test_free_surface_pressure_policy_no_legacy_vent_bias`,
  - `test_outflow_policy_regression_legacy_vs_hardened`.

### Real-world gate
- Add scripted replay harness for the runlog-style scenario class (fixed-dt, inflow/outflow, uncoupled).
- Acceptance checks:
  - bounded `u_rms`, bounded `p_abs_max`,
  - no monotone CFL explosion,
  - no sustained projection retry exhaustion streak.

## Phase C1: Coupled-Compatible BC Unification

### Objective
Use the same BC operator kernels in coupled mode by swapping data provider, not algorithm.

### BC target provider abstraction
Introduce a lightweight provider struct per face:
- target normal velocity / flux,
- target pressure band,
- target vof,
- mode flags (inflow/open/free-surface/wall),
- provenance (`uncoupled_gui_env` vs `coupled_exchange`).

### Integration points
- Uncoupled provider fills targets from `desc.bc_*`.
- Coupled provider fills targets from interface/exchange outputs.
- `swe2d_gpu_apply_2d3d_exchange_skeleton(...)` evolves to produce BC targets and conservative exchange terms; BC kernels consume the same target format.

## Phase C2: Coupled Stability and Conservation

### Requirements
- preserve interface conservation,
- avoid contradictory BC forcing between coupling exchange and face BC operator,
- enforce ordering contract: exchange update -> BC target assembly -> shared BC kernels.

### Tests
- add coupled cases with open boundary forcing and verify:
  - mass balance closure,
  - absence of boundary-driven high-frequency instability,
  - consistent behavior across one-way and two-way modes.

## Implementation Order (Execution-Ready)
1. U1.1 runtime controls + backend plumbing.
2. U1.2 dynamic inflow-Q area normalization.
3. U1.3 open boundary kernel + step ordering integration.
4. U1.4 free-surface vent heuristic retirement behind compatibility flag.
5. U1.5 diagnostics.
6. U2 test and real-world gate.
7. C1 provider abstraction and coupled reuse.
8. C2 coupled stability/conservation validation.

## Migration / Compatibility
- Keep legacy modes behind explicit env flags for reproducibility while hardening rolls out.
- Default workbench behavior should switch to hardened policies after U2 validation passes.
- Document policy defaults and tuning knobs in user/developer docs before making hardened defaults mandatory.

## Immediate Next Coding Task
Implement U1.1 + U1.2 in code first:
- add controls and backend plumbing,
- replace `INFLOW_FLOW_RATE` area normalization path with dynamic active area policy,
- add tests for normalization behavior.
