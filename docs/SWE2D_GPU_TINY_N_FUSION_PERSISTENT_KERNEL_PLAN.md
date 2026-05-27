# SWE2D GPU Tiny-N Kernel Fusion and Persistent-Kernel Plan

Date: 2026-05-27

## Objective
Improve end-to-end performance for tiny cell-count SWE2D runs by reducing launch overhead and increasing useful work per launch on the existing GPU-only solver path.

This plan explicitly prioritizes CUDA behavior and does not target CPU/GPU parity.

## Scope
- Target regime: small unstructured meshes where kernel launch overhead is a dominant share of step time.
- Solver path: CUDA SWE2D in [cpp/src/swe2d_gpu.cu](../cpp/src/swe2d_gpu.cu).
- Integrators in scope: baseline integrator and graph-enabled RK variants already supported by `KernelGraphCache`.

## Current Baseline (from code)
- Time-step path still involves multiple kernels (classify/neighbor/gradient/flux/update/CFL/pack).
- CUDA Graph replay is already present and should be retained as the default fast path.
- A persistent CUDA stream is used, but not a persistent looping kernel.

## Design Goals
1. Keep existing correctness and safety behavior (wet/dry logic, degenerate-cell handling, source controls).
2. Preserve graph replay for normal and medium-to-large meshes.
3. Add a tiny-N optimized execution mode with low launch count.
4. Keep changes incremental and reversible behind runtime flags.

## Proposed Architecture

### 1) Runtime Tiny-N Dispatcher
Add a tiny-N dispatcher in the GPU step entry path that selects one of three modes:
- Mode A: Existing graph replay (default when not tiny).
- Mode B: Fused micro-step kernels (tiny-N default path).
- Mode C: Persistent-kernel stepping loop (experimental tiny-N path).

Selection inputs:
- `n_cells`, `n_edges`
- integrator order
- enabled physics features (rain/source/coupling)
- runtime feature flags

Initial heuristic:
- Tiny regime candidate: `n_cells <= 8k` and `n_edges <= 24k`.
- Final thresholds are profiling-derived and per-device configurable.

### 2) Aggressive Kernel Fusion (Primary Deliverable)
Implement fused kernels that merge hot step stages while preserving numerical ordering.

#### 2.1 Fused edge kernel
Merge edge-centric work into one launch:
- optional hydrograph boundary sampling (if needed per-step)
- front/active checks
- reconstruction + Riemann flux
- write compact edge flux buffers

#### 2.2 Fused cell kernel
Merge cell-centric work into one launch:
- active/degen gating
- incident flux reduction via CSR
- update and source application
- positivity/momentum/depth controls
- per-cell CFL contribution and block reduction into global max
- compact diagnostics packing

#### 2.3 Optional gradient fusion tier
For schemes requiring gradients, evaluate two options:
- Tier 1: keep separate gradient kernel (lower risk)
- Tier 2: fuse gradient computation with edge reconstruction reads using shared staging for tiny-N

Deliver Tier 1 first; Tier 2 only if profiling shows launch overhead still dominates.

### 3) Persistent-Kernel Mode (Secondary Deliverable)
Add an opt-in persistent kernel that performs an internal substep loop for tiny-N runs:
- one cooperative launch per chunk of `k` substeps
- device-side loop handles classify/flux/update/CFL progression
- host only synchronizes at chunk boundaries for diagnostics, stop checks, and BC/source updates

Guardrails:
- limit to fixed feature subset in v1 (no complex coupling updates inside loop)
- strict timeout/iteration guard to avoid runaway loops
- fallback to fused-kernel mode on any capture/cooperative launch failure

## Implementation Phases

## Phase 0: Measurement and Guardrail Baseline (1-2 days)
- Add fine-grained timers and counters:
  - per-kernel launch count per step
  - launch-overhead estimate
  - tiny-N path hit rate
- Add benchmark fixture set for tiny meshes (for example 500, 1k, 2k, 4k, 8k cells).

Acceptance:
- Baseline report generated with current graph path.
- Threshold candidates documented per target GPU.

## Phase 1: Tiny-N Dispatcher + Flags (1 day)
- Add runtime options (env and API flags):
  - `BACKWATER_SWE2D_TINY_MODE=auto|off|fused|persistent`
  - `BACKWATER_SWE2D_TINY_CELL_THRESHOLD`
  - `BACKWATER_SWE2D_TINY_EDGE_THRESHOLD`
- Integrate dispatcher in GPU step path.

Acceptance:
- Default behavior unchanged when mode is `auto` and thresholds not met.
- Telemetry reports selected mode per run.

## Phase 2: Fused-Kernel Path v1 (3-5 days)
- Implement fused edge and fused cell kernels.
- Keep gradient kernel separate (Tier 1).
- Reuse existing data layouts and limiter logic; avoid topology format changes.
- Wire fused path into dispatcher.

Acceptance:
- Correctness parity within existing tolerances on GPU validation suites.
- Launch count reduced significantly vs baseline tiny-N graph path.
- Measured speedup at tiny sizes (target: 1.2x to 1.8x depending on device).

## Phase 3: Persistent-Kernel Prototype (3-5 days)
- Implement opt-in persistent substep kernel for constrained feature subset.
- Add chunked synchronization interface (`k` substeps per host interaction).
- Add robust fallback path.

Acceptance:
- Stable execution on target GPUs.
- Additional tiny-N speedup over fused path for supported subset.
- No regression when disabled.

## Phase 4: Threshold Tuning and Hardening (2-3 days)
- Auto-tune or table-tune thresholds by GPU architecture.
- Expand support matrix for persistent mode features.
- Finalize docs and runtime recommendations.

Acceptance:
- Documented threshold recommendations.
- Reproducible benchmark plots and selected defaults.

## File-Level Change Plan
- [cpp/src/swe2d_gpu.cuh](../cpp/src/swe2d_gpu.cuh)
  - add tiny-mode config structs, counters, and state for fused/persistent kernels
- [cpp/src/swe2d_gpu.cu](../cpp/src/swe2d_gpu.cu)
  - add dispatcher and fused kernels
  - add persistent-kernel implementation and fallback
  - integrate telemetry/timers
- [cpp/src/swe2d_bindings.cpp](../cpp/src/swe2d_bindings.cpp)
  - expose tiny-mode controls and telemetry to Python
- [cpp/src/swe2d_solver.cpp](../cpp/src/swe2d_solver.cpp)
  - plumb configuration and expose summary metrics in diagnostics
- [tests/test_swe2d_gpu_validation_perf.py](../tests/test_swe2d_gpu_validation_perf.py)
  - add tiny-N performance A/B cases and correctness checks
- [tests/test_swe2d_gpu_unstructured.py](../tests/test_swe2d_gpu_unstructured.py)
  - add tiny mesh cases across schemes 0..4 under fused mode

## Validation Strategy
1. Correctness first:
- GPU validation and unstructured suites with tiny mode off, fused, and persistent.
- Compare mass conservation, wet/dry behavior, and CFL stability metrics.

2. Performance second:
- Measure end-to-end step throughput and per-step kernel launches.
- Use Nsight Systems and Nsight Compute for representative tiny meshes.

3. Reliability third:
- Stress with feature toggles (rain, BC hydrographs, degenerate handling modes).
- Confirm deterministic fallback on any kernel-mode error.

## Telemetry to Add
- `tiny_mode_selected`
- `tiny_mode_fallback_count`
- `fused_path_steps`
- `persistent_path_steps`
- `avg_launches_per_step`
- `graph_replay_count` (already present; keep)

## Risk Register
- Numerical ordering drift from fusion.
  - Mitigation: preserve operation order where possible; strict A/B tolerances.
- Register pressure reducing occupancy in fused kernels.
  - Mitigation: split fused kernels into two stages and tune block sizes.
- Persistent kernel complexity with dynamic BC/source updates.
  - Mitigation: start with constrained subset and chunked synchronization.
- Feature matrix explosion.
  - Mitigation: dispatcher capability checks and explicit unsupported-mode fallback.

## Recommended First Milestone
Deliver Phase 1 + Phase 2 (dispatcher + fused kernels) before persistent mode. This provides most of the tiny-N gain with lower integration risk and leverages your current CUDA graph-enabled architecture.

## Exit Criteria
This plan is complete when:
- Tiny-N fused mode is production-usable and defaulted by threshold in `auto` mode.
- Persistent mode is available as experimental opt-in with documented support limits.
- Benchmarks show clear tiny-N wins without regressions in existing GPU validation suites.
