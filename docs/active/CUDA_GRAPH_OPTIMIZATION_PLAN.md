# CUDA Graph Optimization Plan (Suggestion 9)

Date: 2026-05-13

## Objective
Reduce kernel launch overhead and improve GPU occupancy in the SWE2D solver by implementing CUDA graph capture for repeated kernel sequences.

## Current State
- Graph infrastructure and bindings are implemented (`KernelGraphCache`, enable/destroy APIs)
- Flux→Update→CFL→Pack graph capture/replay is implemented in `swe2d_gpu_step`
- Replay is gated by strict config signature matching and falls back safely on mismatch
- RK2/RK4 now tag graph integrator context so cache entries are partitioned by temporal order

## Implementation Status
- Phase 1: Implemented
- Phase 2: Implemented
- Phase 3: Implemented (runtime env toggle + replay telemetry)

## Target Kernels for Graph Capture

### Primary Sequence (Flux → Update)
1. `swe2d_flux_kernel` — Compute Riemann fluxes across edges
2. `swe2d_update_kernel` — Apply fluxes to cell states (RK2 or RK4 substep)
3. `swe2d_cfl_kernel` — Compute max CFL for time-step stability check

### Expected Frequency
- Called once per RK2/RK4 substep
- In tight coupling loops with many implicit iterations, called tens-to-hundreds of times per timestep
- Same topology (thread counts, block counts) on every call for same mesh size

## Implementation Strategy

### Phase 1: Single-Sequence Graph Capture (Highest Impact)
- Create CUDA graph template for: Flux → Update → CFL sequence
- Capture on first call with given mesh size and config
- Replay on subsequent calls with same configuration
- Store graph handle per mesh configuration

### Phase 2: Support for Different RK Orders
- Separate graphs for RK2 vs RK4 combining kernels
- Runtime selection based on `time_integrator` setting
- Cache both variants if mixed usage detected

### Phase 3: Conditional Graph Activation
- Add environment variable `BACKWATER_ENABLE_CUDA_GRAPHS=1` for opt-in
- Diagnostic counter: `cuda_graph_replays` to track replay frequency
- Fallback to individual launches if graph construction fails

## Code Changes Required

### cpp/src/swe2d_gpu.cuh
- Add graph handle storage in GPU solver struct:
  ```cpp
  struct KernelGraphCache {
      cudaGraph_t graph = nullptr;
      cudaGraphExec_t exec = nullptr;
      int n_cells = 0;
      int time_integrator = 0;
      bool is_valid = false;
  };
  ```

### cpp/src/swe2d_gpu.cu
- Implement `create_rk_step_graph()` function to capture Flux→Update→CFL
- Implement `destroy_kernel_graphs()` for cleanup
- Modify step functions to check cache and replay vs launch

### cpp/src/swe2d_bindings.cpp
- Add `enable_cuda_graphs` parameter to solver initialization
- Expose diagnostic counters: `cuda_graph_replays`, `cuda_graph_launches`
- Add `reset_kernel_graphs()` for cleanup between runs

## Validation Plan
1. Build with graph support and enable via environment variable
2. Run test suite with graphs enabled → verify correctness
3. Profile with/without graphs on target model:
   - Compare kernel launch overhead (via CUDA profiler)
   - Measure total coupling time reduction
   - Verify GPU occupancy improvement
4. Run A/B comparison of step vs iterative with graphs enabled

## Expected Outcomes
- Kernel launch overhead reduced by ~40–50% in coupling loops
- Total coupling time reduced by ~15–25% (when launch overhead is significant)
- GPU occupancy improvement in dense scheduling scenarios
- Diagnostic counters show graph replay frequency

## Risks & Mitigations
- **Risk**: Graph capture overhead on first call
  - Mitigation: Capture asynchronously during first real run; accept one slow first timestep
- **Risk**: Graph invalidation if kernel parameters change
  - Mitigation: Validate kernel params match cached graph before replay; fall back to individual launches if mismatch
- **Risk**: Memory overhead from graph handles
  - Mitigation: Limit to ~2–3 graph variants per solver instance (RK2, RK4, variants)

## Next Steps (after implementation)
1. Run correctness tests with graphs enabled (GPU validation + unstructured suites)
2. Profile target model with/without graphs and feed results into Suggestion 10 A/B report
3. Integrate graph counters into A/B benchmark script output summaries
