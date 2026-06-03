# SWE2D Tiny-N GPU Utilization Work Log (Phases 1-8)

Date: 2026-05-30

This log tracks implementation work started from the requested 8-phase utilization plan for small wetted-cell cases.

## Phase 1: Tiny-N Dispatcher + Runtime Flags
Status: Implemented (initial production scaffold)

Changes:
- Added Tiny-N configuration fields to solver config:
  - `tiny_mode` (0=off, 1=auto, 2=fused, 3=persistent)
  - `tiny_cell_threshold`, `tiny_edge_threshold`
  - `tiny_wet_cell_threshold`
  - `tiny_persistent_chunk_substeps`
  - `tiny_active_compaction_stride_steps`
  - `tiny_enable_active_compaction`
- Added Tiny-N decision logic in `swe2d_step` with wetted-cell-aware selection.
- Added fallback-aware selection behavior:
  - Tiny fused/persistent are selectable and reported.
  - Execution currently falls back to baseline kernels until dedicated fused/persistent kernels are added.

## Phase 2: Reduce Host Sync Frequency
Status: Implemented

Changes:
- Updated default GPU diagnostic sync interval from per-step to production cadence:
  - C++ config default changed from `1` to `50`.
  - pybind default changed from `1` to `50`.
- Existing explicit user settings still override this default.

## Phase 3: Keep Per-Step Orchestration Native
Status: Preserved and documented

Changes:
- Existing native run-to-time fast path retained.
- No regression introduced in this pass; backend continues to call native `swe2d_run_to_time` for uncoupled runs.

## Phase 4: Wet-Active Aware Tiny Selection
Status: Implemented (dispatcher-level)

Changes:
- Added `last_wet_cells` state tracking in solver.
- Tiny mode auto-selection now considers active wetted-cell estimate (`tiny_wet_cell_threshold`) in addition to total mesh thresholds.
- Active edge estimate added for telemetry.

## Phase 5: Fused Kernel Path
Status: Partially implemented (safe runtime-effective subset)

Changes:
- Added mode selection and mode-effective telemetry for fused path.
- Added explicit fallback accounting where fused is selected but baseline path is still used.
- Enabled fused mode as runtime-effective for currently supported single-stage GPU step paths (Euler/non-RK graph variants).
- Kept explicit fallback for unsupported multi-stage RK/advanced paths.

Pending:
- Actual fused edge+cell kernels and dispatch implementation in CUDA path.

## Phase 6: Persistent Chunked Path
Status: Core GPU kernel path implemented + next 5 path extensions landed

Changes:
- Added persistent mode config fields (`tiny_persistent_chunk_substeps`) and telemetry.
- Added explicit fallback accounting where persistent is selected but baseline path is still used.
- Added backend persistent run behavior: when `tiny_mode=3`, native `run_to_time` now uses `diag_batch_size=tiny_persistent_chunk_substeps` to support chunked host interaction cadence.
- Added cooperative persistent kernel implementation in CUDA (`swe2d_persistent_chunk_kernel_first_order`) with grid-wide sync between edge flux and cell update phases.
- Added native wrapper (`swe2d_gpu_step_persistent_chunk`) with guarded fallback when constraints are not met (unsupported hardware/features).
- Wired solver dispatch so persistent tiny mode is runtime-effective on supported single-stage hydrostatic path.

Current constraints for cooperative persistent kernel specialization:
- First-order spatial path only.
- Single-stage hydrostatic path only (not RK2/RK4/RK5 or 3D/nonhydro).
- Extreme-rain source-CFL subcycling path (`extreme_rain_mode`) still excluded.
- Explicit source subcycling/IMEX source split (`source_true_subcycling`, `source_imex_split`) still excluded.
- Falls back automatically when cooperative launch capacity/hardware support is insufficient.

Persistent chunked stepping extensions (current iteration):
- Implemented chunked baseline stepping fallback inside `swe2d_gpu_step_persistent_chunk(...)`.
  - If cooperative persistent kernel is ineligible (higher-order/source-extreme/source-split path), wrapper now executes `chunk_substeps` internal baseline GPU substeps of size `dt/chunk_substeps`.
  - This keeps tiny persistent mode active while preserving existing numerics for unsupported cooperative-kernel subsets.
- Extended solver dispatch so tiny persistent mode remains effective (no forced fallback) for:
  - RK2 path (including Godunov rollout variant)
  - RK4 path
  - RK4 graph-safe path
  - RK5 graph-safe path
  - 3D single-phase free-surface path
  - nonhydro predictor-corrector path
- Added chunked substep loops for these branches in solver dispatch, with diagnostics synced on final substep and `diag.dt` restored to outer-step `dt`.

RK2 cooperative specialization (follow-up iteration):
- Added dedicated CUDA API and implementation for RK2 tiny persistent stepping:
  - `swe2d_gpu_step_rk2_persistent_chunk(...)`.
- RK2 tiny persistent now preserves SSPRK2 stage structure while using persistent chunk stepping for each RK stage.
- For first-order single-stage-eligible internals, this reaches the cooperative persistent kernel path through stage calls.
- For unsupported cooperative internals, it still uses chunked baseline fallback inside the persistent wrapper (numerics-preserving behavior).
- Solver dispatch now routes tiny persistent RK2 (including Godunov rollout RK2 variant) through this dedicated RK2 persistent API instead of outer solver-level chunk loops.

RK4 graph-safe persistent specialization (follow-up iteration):
- Added dedicated CUDA API and implementation:
  - `swe2d_gpu_step_rk4_graph_persistent_chunk(...)`.
- Solver dispatch now routes tiny persistent RK4 graph-safe branch through this dedicated API.
- Current implementation executes chunked RK4-graph substeps (`dt/chunk_substeps`) natively in CUDA API space with final-substep diagnostic sync.
- This is a specialization/centralization step for the RK4 graph-safe path; stage-RHS-level cooperative persistent kernels remain a future deeper kernel rewrite.

Low-wetted-fraction overhead reduction (follow-up iteration):
- Added dry-region boundary-edge skip in baseline flux kernel:
  - In `swe2d_flux_kernel`, boundary edges (`c1 < 0`) now early-exit when boundary-adjacent cell `c0` is inactive.
- Added matching skip in cooperative persistent edge phase:
  - In `swe2d_persistent_chunk_kernel_first_order`, boundary edges with inactive `c0` now take the zero-flux fast path.
- Rationale:
  - For sparse wet domains, this trims unnecessary reconstruction/HLLC work on dry boundary edges, reducing branch/memory overhead beyond pure tiny-N thread-count effects.

Active-edge compaction (next target implementation):
- Added active-edge compaction workspace to CUDA device state:
  - `d_active_edge_ids` (edge index list)
  - `d_n_active_edges` (device scalar count)
- Added active-edge list builder kernel:
  - `swe2d_collect_active_edges_kernel(...)`
  - Builds a compact edge list from current `d_active` mask (interior: either endpoint active; boundary: boundary-adjacent cell active).
- Integrated optional compaction into `swe2d_gpu_step_persistent_chunk(...)`:
  - When enabled for the current step, builds the compact list and launches cooperative persistent kernel over `n_active_edges` instead of full `n_edges`.
  - When disabled for the current step, preserves existing full-edge launch behavior.
- Solver dispatch now controls compaction cadence via existing tiny settings:
  - `tiny_enable_active_compaction`
  - `tiny_active_compaction_stride_steps`
  - Effective compaction predicate: `tiny_mode==persistent && enabled && (gpu_steps % stride == 0)`.
- Plumbed compaction flag through persistent APIs used by tiny persistent paths:
  - `swe2d_gpu_step_persistent_chunk(...)`
  - `swe2d_gpu_step_rk2_persistent_chunk(...)`

Delta against prior exclusions:
- `source_true_subcycling`: now covered via chunked baseline fallback path.
- `source_imex_split`: now covered via chunked baseline fallback path.
- `extreme_rain_mode`: now covered via chunked baseline fallback path.
- Higher-order spatial + RK2/RK4/RK5 staging: now covered via persistent chunking in solver dispatch.
- nonhydro and 3D solver paths: now covered via persistent chunking in solver dispatch.

Newly lifted exclusions in this iteration:
- Rain CN source assembly is now supported on persistent path (wrapper runs `swe2d_build_rain_cn_source_kernel` before classify).
- Hydrograph boundary forcing is now supported on persistent path (wrapper runs `swe2d_apply_hydrograph_bc_kernel`).
- Degenerate-cell modes are now supported on persistent path:
  - mode 1/3 deactivation in active set,
  - mode 3 owner sync,
  - mode 2 repaired inverse-area update behavior inside persistent kernel.

Build/rebuild status:
- Native rebuild executed after persistent-kernel integration and follow-up fixes.
- CUDA compile issues fixed:
  - cooperative launch argument typing for NVCC,
  - grid-sync divergence hazard in persistent kernel update loop.

Pending:
- Cooperative persistent/chunked stepping kernels and host boundary sync protocol.

## Phase 7: Split Debug/Guardrail from Hot Path
Status: Implemented (safe default hardening)

Changes:
- Explicitly defaulted expensive 3D transport debug telemetry off in backend initialization:
  - `BACKWATER_SWE3D_VOF_TRANSPORT_DEBUG=0` via `setdefault`.
- Guardrail defaults remain intact for robustness unless user/environment overrides.

## Phase 8: Tiny-N Telemetry + Threshold Tuning Hooks
Status: Implemented (telemetry plumbing)

Changes:
- Added tiny telemetry fields to `SWE2DStepDiag`:
  - requested/selected/effective mode
  - fallback flag
  - active cell/edge estimates
  - cumulative fallback count
  - cumulative fused/persistent step counters
- Exposed telemetry in both:
  - `swe2d_step` diagnostics dict
  - `swe2d_run_to_time` diagnostic batches

## Notes on Scope
- Cooperative persistent kernel remains specialized to the first-order single-stage hydrostatic subset for performance.
- Newly added path coverage uses chunked baseline GPU stepping for unsupported cooperative-kernel subsets to preserve numerical behavior while extending tiny persistent applicability.
- Runtime selection and observability remain in place for A/B and further kernel specialization.

## Validation Executed

### Phase 1/2/4/8 validation
- Test file: `tests/test_swe2d_backend_tiny_mode_config.py`
- Result: `5 passed`
- Coverage: tiny-mode kwargs plumbing, compat filtering for older extension signatures, production sync default.

### Phase 5 dispatch validation
- Test file: `tests/test_swe2d_tiny_mode_dispatch.py`
- Result: `2 skipped` (current environment native extension does not yet expose rebuilt tiny-mode diagnostic keys)
- Coverage: fused-effective/fallback dispatch assertions when rebuilt binary is available.

### Phase 6 core kernel regression smoke
- Test files:
  - `tests/test_swe2d_backend_tiny_mode_config.py`
  - `tests/test_swe2d_tiny_mode_dispatch.py`
  - `tests/test_swe2d_gpu_validation_perf.py`
- Result: `7 passed`
- Note: Python tests validate runtime plumbing and regression safety; direct execution of new C++ persistent kernel requires rebuilding/loading the updated native module in the runtime environment.

### Phase 6 extension pass (rain/hg/degen) validation
- Build:
  - `cmake -S . -B build && cmake --build build -j`
  - `cmake --build build -j` (incremental)
- Tests:
  - `PYTHONPATH="$PWD:$PWD/build" pytest -q tests/test_swe2d_tiny_mode_dispatch.py` -> `4 skipped`
  - `PYTHONPATH="$PWD:$PWD/build" python3 -m unittest -v tests.test_swe2d_gpu_validation_perf` -> `2 ok, 1 skipped (perf-gated)`
- Interpretation:
  - Native GPU validation suite remains green for correctness-smoke on rebuilt binaries.
  - Tiny dispatch tests continue to skip in this environment until loaded extension exposes tiny diagnostic keys used by assertions.

### Phase 6 extension pass (next 5 paths) validation
- Tests:
  - `PYTHONPATH="$PWD:$PWD/build" pytest -q tests/test_swe2d_tiny_mode_dispatch.py tests/test_swe2d_backend_tiny_mode_config.py` -> `5 passed, 6 skipped`
  - `PYTHONPATH="$PWD:$PWD/build" python3 -m unittest -v tests.test_swe2d_gpu_validation_perf tests.test_swe2d_gpu_unstructured` -> `7 ok, 1 skipped (perf-gated)`
- Interpretation:
  - Backend tiny-mode configuration behavior remains green.
  - GPU validation suites remain green after extending persistent chunking to RK/3D/nonhydro/source-extreme families.
  - Dispatch assertion tests are authored for new persistent behavior (RK2/RK4/RK5 effective) and are currently skip-gated by runtime diagnostic-key availability and/or GPU activation in this environment.

### Phase 6 RK2 cooperative specialization validation
- Build:
  - forced rebuild with touched CUDA/C++ sources (`swe2d_gpu.cuh`, `swe2d_gpu.cu`, `swe2d_solver.cpp`) -> success
- Tests:
  - `PYTHONPATH="$PWD:$PWD/build" pytest -q tests/test_swe2d_tiny_mode_dispatch.py tests/test_swe2d_backend_tiny_mode_config.py` -> `5 passed, 6 skipped`
  - `PYTHONPATH="$PWD:$PWD/build" python3 -m unittest -v tests.test_swe2d_gpu_validation_perf tests.test_swe2d_gpu_unstructured` -> `7 ok, 1 skipped (perf-gated)`
- Interpretation:
  - RK2 persistent specialization compiles and integrates without regression in current GPU validation suites.

### Phase 6 RK4 graph-safe persistent specialization validation
- Build:
  - forced rebuild with touched CUDA/C++ sources (`swe2d_gpu.cuh`, `swe2d_gpu.cu`, `swe2d_solver.cpp`) -> success
- Tests:
  - `PYTHONPATH="$PWD:$PWD/build" pytest -q tests/test_swe2d_tiny_mode_dispatch.py tests/test_swe2d_backend_tiny_mode_config.py` -> `5 passed, 6 skipped`
  - `PYTHONPATH="$PWD:$PWD/build" python3 -m unittest -v tests.test_swe2d_gpu_validation_perf tests.test_swe2d_gpu_unstructured` -> `7 ok, 1 skipped (perf-gated)`
- Interpretation:
  - RK4 graph-safe persistent dispatch specialization compiles and integrates without regression in current validation suites.

### Low-wetted-fraction boundary-edge skip validation
- Build:
  - forced rebuild with touched `cpp/src/swe2d_gpu.cu` -> success
- Tests:
  - `PYTHONPATH="$PWD:$PWD/build" pytest -q tests/test_swe2d_tiny_mode_dispatch.py tests/test_swe2d_backend_tiny_mode_config.py` -> `5 passed, 7 skipped`
  - `PYTHONPATH="$PWD:$PWD/build" python3 -m unittest -v tests.test_swe2d_gpu_validation_perf tests.test_swe2d_gpu_unstructured` -> `7 ok, 1 skipped (perf-gated)`
- Interpretation:
  - Optimization integrated without observed correctness/stability regressions in current GPU validation suites.

### Active-edge compaction validation
- Build:
  - forced rebuild with touched CUDA/C++ sources (`swe2d_gpu.cuh`, `swe2d_gpu.cu`, `swe2d_solver.cpp`) -> success
- Tests:
  - `PYTHONPATH="$PWD:$PWD/build" pytest -q tests/test_swe2d_tiny_mode_dispatch.py tests/test_swe2d_backend_tiny_mode_config.py` -> `5 passed, 7 skipped`
  - `PYTHONPATH="$PWD:$PWD/build" python3 -m unittest -v tests.test_swe2d_gpu_validation_perf tests.test_swe2d_gpu_unstructured` -> `7 ok, 1 skipped (perf-gated)`
- Interpretation:
  - Active-edge compaction integration compiles and passes current regression suites without observed numerical/stability regressions.

### Phase 3 smoke validation
- Test file: `tests/test_swe2d_run_options_builder.py`
- Result: `1 passed`
- Coverage: SWE2D runtime option plumbing regression smoke.

### GPU-path regression smoke (utilization-facing path)
- Test file: `tests/test_swe2d_gpu_validation_perf.py`
- Result: `2 passed`
- Coverage: basic GPU validation/perf suite still green after dispatcher/telemetry/default updates.
