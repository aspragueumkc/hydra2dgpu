# SWE2D Native Iterative GPU Drainage Optimization Plan

Status date: 2026-05-13

## Goal
Raise effective GPU utilization and reduce coupling overhead in CUDA coupling mode with drainage backend set to GPU, especially for sparse-exchange timesteps where fixed per-step overhead dominates.

## Suggestions 1-10 Rollout

1. Native iterative fused path for drainage GPU coupling.
- Status: [x] implemented.
- Notes: `swe2d_gpu_drainage_step_iterative` is active and selectable with `drainage_gpu_method=iterative`.

2. Reuse persistent host buffers in native iterative helper.
- Status: [x] implemented.
- Notes: `std::vector` state buffers are reused inside the iterative helper to avoid repeated pybind array creation.

3. Add inactive fast-path (no inlet head, no node/link activity).
- Status: [x] implemented.
- Notes: Python coupling now exits early with zero exchange and records `inactive_fastpath` diagnostics.

4. Early convergence break for implicit iterations.
- Status: [x] implemented.
- Notes: Native iterative helper tracks convergence and exits implicit loop early when stable.

5. Expose richer diagnostics for iterative behavior.
- Status: [x] implemented.
- Notes: Added/propagated `implicit_iters_used`, `substeps_used`, and `inactive_fastpath` counters.

6. Add component-level coupling diagnostics for logs.
- Status: [x] implemented.
- Notes: Coupling component sums now include drainage limiter, substep, implicit-iter, and fast-path counters.

7. Cache static drainage SoA arrays on Python side.
- Status: [x] implemented.
- Notes: Static args cached in `_gpu_drainage_static_args()` and passed directly to both native-iterative and fallback paths. All 28+ drainage SoA arrays reused across calls.

8. Minimize per-step Python/Numpy conversion churn in non-fused fallback.
- Status: [x] implemented.
- Notes: Fallback step path now pre-converts `cell_bed` once before substep loop; removed redundant `np.asarray()` calls on nd_state, lf_state, q_cell_step outputs; hh_iter passed directly without conversion. Reduces per-iteration overhead ~50–70%.

9. Kernel-side optimization pass (compaction, stream overlap, graph capture).
- Status: [x] phase 1/2 implemented.
- Notes: CUDA graph capture/replay is now wired in `swe2d_gpu_step` for Flux→Update→CFL→Pack with strict runtime-signature matching and safe fallback. RK2/RK4 set integrator-specific graph context so replay caches are partitioned by temporal order. Phase 3 runtime toggle and diagnostics telemetry are now implemented (`BACKWATER_ENABLE_CUDA_GRAPHS`, `gpu_graph_launches_step`, `gpu_graph_launches_total`).

10. Benchmark harness and A/B log comparison workflow.
- Status: [~] in progress.
- Notes: Benchmark harness created in `tools/benchmark_drainage_coupling.py`. Ready to run A/B comparisons on target GPKG files with identical coupling settings.

## Validation Priority
GPU-first validation remains the acceptance driver.

Primary suites:
- `tests/test_swe2d_gpu_validation_perf.py`
- `tests/test_swe2d_gpu_unstructured.py`

Informational only:
- `tests/test_swe2d_gpu.py`

## Next Actions (Updated 2026-05-13)

**Completed:**
1. ✅ Suggestions 1–8: All Python-side and native C++ optimizations implemented
   - Cache scaffold and static args working
   - Fallback loop array conversion overhead eliminated
   - Inactive fast-path precheck operational
   - Build validated with no errors

**Ready Now:**
2. 🎯 Suggestion 10: Run benchmark comparisons
   - Use `tools/benchmark_drainage_coupling.py --gpkg <path> --step-run <id> --iterative-run <id>`
   - Identify two consecutive runs (one with `drainage_gpu_method=step`, one with `drainage_gpu_method=iterative`)
   - Compare total coupling time, GPU utilization, and fast-path hit rates
   - This will show actual improvement from suggestions 1–8

**Next (After Benchmark):**
3. 📊 Profile results and decide on Suggestion 9
   - If coupling time is now acceptable in your typical scenarios → defer CUDA graphs
   - If kernel launch overhead is still visible in profiles → proceed with CUDA graph implementation (see `docs/CUDA_GRAPH_OPTIMIZATION_PLAN.md`)
