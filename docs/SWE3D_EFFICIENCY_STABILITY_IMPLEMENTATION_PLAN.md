# SWE3D GPU Efficiency and Stability Implementation Plan

## Scope and intent
This plan targets high-impact efficiency and numerical-stability improvements in the active CUDA SWE3D path (`cpp/src/swe2d_gpu.cu`).

Design constraints:
- GPU-first implementation and validation.
- Preserve public APIs and existing runtime controls where possible.
- Deliver changes in small, testable increments with explicit rollback points.

## Success criteria
- Reduce avoidable host-device synchronization and allocator churn in the 3D step loop.
- Remove hardcoded free-surface vent forcing and route behavior through runtime control.
- Improve interface robustness by avoiding non-physical VOF zeroing on inactive cells.
- Complete manual validation checklist with reproducible commands and observed outputs.

## Phase plan

### Phase 1: Hot-path stability and performance hardening (implement now)
1. Replace hardcoded ZMAX free-surface vent `+1.0` with runtime-configurable `free_surface_vent_bias`.
2. Eliminate per-call `cudaMalloc/cudaFree` in `compute_vof_sum` by adding a persistent patch-level reduction buffer.
3. Preserve clamped local VOF for inactive cells in `swe3d_vof_transport_upwind_kernel` (instead of forcing `0.0`).

Acceptance checks:
- Build compiles with no SWE3D signature mismatches.
- Existing SWE3D validation tests pass.
- No regressions in BC validation baseline.

### Phase 2: Projection loop efficiency (implemented)
1. Reduce per-iteration host synchronization in Jacobi residual checking.
2. Add configurable residual sampling cadence (for example every N iterations).
3. Keep fail-fast and retry semantics unchanged.

Acceptance checks:
- Projection iterations complete with fewer host sync points.
- Retry diagnostics remain consistent.

### Phase 3: Numerical closure refinement (in progress)
1. Revisit fixed projection correction scale and make it adaptive/bounded.
2. Add divergence-oriented projection quality metric to complement pressure-delta residual.

Acceptance checks:
- Improved stability in stressed interface cases.
- No degradation in invariant gates.

### Phase 4: Hardware interaction and throughput (implemented)
1. Replace heavily contended atomics in reduction-style kernels with hierarchical reductions.
2. Cache runtime controls once per solver step (or per solver init) instead of repeated env parsing.

Acceptance checks:
- Reduced kernel and step wall time on representative SWE3D cases.
- No change to externally visible behavior defaults.

### Phase 5: Control-surface and rollout hardening (in progress)
1. Expose new projection divergence gate controls through Python runtime configuration API.
2. Wire/expand diagnostics payload for gate observability in Python-facing step outputs.

Acceptance checks:
- New controls available without direct env editing.
- Step diagnostics include divergence gate telemetry fields used by retry logic.

## Implementation log
- [x] Plan authored and committed to docs.
- [x] Phase 1.1 runtime vent-bias parameterization.
- [x] Phase 1.2 persistent VOF sum reduction buffer.
- [x] Phase 1.3 inactive-cell VOF preservation.
- [x] Build and initial unit/invariant validation.
- [x] Volumetric inlet dry-start mass injection fix (INFLOW_FLOW_RATE boundary handling).
- [x] SWE3D validation damping gate robustness fix (non-increasing with tolerance, zero-initial-safe).
- [x] Manual SWE3D validation sweep (completed and passing).
- [x] Phase 2.1/2.2 projection residual sampling cadence control and reduced Jacobi host sync.
- [x] Phase 2 acceptance validation (build + SWE3D invariant/unit checks).
- [x] GUI input wired for projection residual sampling cadence (`BACKWATER_SWE3D_PROJECTION_RESIDUAL_SAMPLE_ITERS`).
- [x] Phase 3 started: adaptive/bounded projection correction scaling implemented (runtime-controlled).
- [x] Phase 3.2 divergence-oriented projection quality metric implemented (`projection_divergence_ratio`, `projection_divergence_ratio_max`).
- [x] Phase 4.1 hierarchical block-reduction updates for SWE3D reduction kernels (`projection_residual_max`, `velocity_absmax`, `vof_min`, `vof_sum`, `sum_sq`).
- [x] Phase 4.2 runtime-control cache path (`swe3d_load_runtime_controls_cached`) used in 3D dt and step paths.
- [x] Phase 5 started: Python runtime API exposure for divergence gate controls.
- [x] Phase 5 GUI wiring: 3D patch projection divergence gate/target controls added to workbench UI + env override mapping.
- [x] Phase 5 observability: run log now emits active 3D projection control values for each Experimental 3D run.

Phase 2 runtime control added:
- `BACKWATER_SWE3D_PROJECTION_RESIDUAL_SAMPLE_ITERS` (default `1`, range `1..1024`).
- Behavior: samples Jacobi residual every `N` iterations and on the final Jacobi iteration, reducing D2H sync frequency while preserving retry/fail-fast flow.

Phase 3 runtime controls added (initial step):
- `BACKWATER_SWE3D_PROJECTION_CORRECTION_SCALE_MIN` (default `1.5`, range `0.1..4.0`).
- `BACKWATER_SWE3D_PROJECTION_CORRECTION_SCALE_MAX` (default `1.5`, range `0.1..4.0`, clamped to `>= min`).
- Behavior: uses residual-ratio health to adapt projection correction scale within bounds and exposes `projection_correction_scale_used` in step diagnostics.

Phase 3 diagnostics added:
- `projection_divergence_ratio`: last-attempt ratio of post-correction divergence RMS to pre-projection divergence RMS.
- `projection_divergence_ratio_max`: worst divergence RMS ratio observed across retry attempts.

Phase 4 implementation notes:
- Reduction kernels now perform per-block shared-memory reductions with one global atomic update per block.
- Runtime controls are loaded through a cached parser keyed by env-signature changes to avoid repeated full parsing across 3D dt/step paths.

Phase 5 runtime API additions:
- `configure_swe3d_runtime(..., projection_divergence_gate_enable=..., projection_divergence_ratio_target=...)`.
- Step diagnostics now include `projection_divergence_gate_enabled` and `projection_divergence_ratio_target`.

Phase 5 GUI/control-surface additions:
- Workbench controls: `3D projection divergence gate` and `3D projection divergence ratio target`.
- Env mapping: `BACKWATER_SWE3D_PROJECTION_DIVERGENCE_GATE_ENABLE`, `BACKWATER_SWE3D_PROJECTION_DIVERGENCE_RATIO_TARGET`.
- Run observability: Experimental 3D run logs include a `3D projection controls` line with residual stride, gate state, and ratio target.

## Execution results (current pass)
1. Build:
- `cmake --build build -j` -> `ninja: no work to do`.

2. SWE3D uncoupled validation (`tests/test_swe3d_uncoupled_validation.py`):
- Result: `30 passed, 6 skipped`.

3. SWE3D validation runner (`tools/run_swe3d_validation.py`):
- Invariant summary: `8 passed, 0 failed`.

4. BC baseline:
- `tests/test_bc_validation.py`: `27 passed, 8 warnings` in `889.67s`.

## Manual validation checklist (final gate)
1. Build native module.
2. Run SWE3D uncoupled validation tests.
3. Run SWE3D validation script with invariant checks.
4. Run BC regression baseline.
5. Inspect logs for projection retry, state-guard, and VOF bound warnings.

## Planned command set
```bash
# from repo root
cmake -S . -B build
cmake --build build -j

conda run -n qgis_stable python -m pytest tests/test_swe3d_uncoupled_validation.py -q
conda run -n qgis_stable python tools/run_swe3d_validation.py
conda run -n qgis_stable python -m pytest tests/test_bc_validation.py -q
```

## Risks and mitigations
- Risk: signature drift while threading vent-bias through device helpers and kernels.
  - Mitigation: update all call sites in one patch and run immediate compile/test checks.
- Risk: persistent buffer lifecycle leaks.
  - Mitigation: pair allocation in patch alloc with release in patch teardown.
- Risk: inactive-cell VOF preservation alters edge-case mass behavior.
  - Mitigation: run invariant gates and compare warning telemetry.
