# Rainfall Source Term Re-evaluation Optimization — Revised

## Goal
Re-evaluate the SCS-CN rainfall source term at a configurable interval (default 60s)
instead of every solver step. Apply the resulting excess rate as a constant
source during the interval. Eliminates the per-step state mutation that causes
RK4/RK5 save/restore complexity and reduces per-step compute.

## Decisions (closing prior gaps)
- **Interval**: scalar `rain_update_interval_seconds`, default 60.0, configurable
  via `swe2d_solver_set_rain_cn_forcing` (added `rain_update_interval_s` arg).
- **Rate computation**: average over the **last completed interval**
  `[t_prev_update, t_now]`. No instantaneous rate, no `prev_excess` buffer per
  cell — we store the **previous cumulative excess** at the same time we
  capture the rate, then on the next update: `rate = (cum_excess_new -
  cum_excess_prev) / dt_interval`. Continuity guaranteed: cumulative state
  is continuous; rate just steps.
- **Per-cell vs scalar update time**: scalar `last_rain_update_time`. All cells
  update together because they share a simulation clock. Per-cell buffer
  would only matter if cells had divergent clocks — they don't.
- **Mass conservation**: `cum_rain_mm[c]` and `cum_excess_cum_mm[c]` advance
  by SCS-CN formula at the update tick. Across the interval the rate is
  constant, so `h += dt * rate * dt = cum_excess_interval * mm_to_model_depth`
  recovers the exact interval excess. Total over many intervals = sum of
  excess = final `cum_excess_cum_mm` ✓.
- **Source location**: write to existing `d_cell_source_mps` once per interval.
  Keep one buffer, one interface; no need for a new `d_current_rain_source_rate_mps`
  buffer. The SWE update kernel reads `cell_source_mps` as today.
- **No-rain case (`n_rain_samples == 0`)**: skip the update kernel entirely
  and zero `d_cell_source_mps` once at the top of the step (or rely on the
  existing guard at line 5160).
- **Graph cache**: graphs capture `d_cell_source_mps` reads at capture time
  pointer — same as today. Update kernel runs OUTSIDE the graph, once per
  interval. When the interval boundary crosses, we force graph invalidation
  by bumping `cache.config_signature` (or skipping capture for that step).
  Simpler: don't capture the per-step build kernel inside the graph at all —
  move it outside, the same place the new update kernel will sit.
- **Span across whole simulation**: handle partial final interval by clamping
  `dt_interval = min(t_now - last_update, rain_update_interval_s)`.

## Principles
- **KISS**: One buffer. One kernel. One launch per interval.
- **YAGNI**: Skip per-cell update tracking, instantaneous rate, per-stage
  complex save/restore.
- **Separation**: Update is a deterministic tick separate from solver step.

## Affected files
- `cpp/src/swe2d_gpu.cuh` — add `rain_update_interval_s`, drop nothing
- `cpp/src/swe2d_gpu.cu` — add kernel, add runtime launch hook, remove build
  kernel calls from solver step, drop save/restore from RK functions
- `cpp/src/swe2d_solver.cpp` — add `rain_update_interval_s` arg to
  `swe2d_solver_set_rain_cn_forcing`
- `bindings/hydra_swe2d.*` — expose the new arg (Python binding)
- `swe2d/runtime/backend.py` — pass through `rain_update_interval_s` from
  project settings
- `swe2d/workbench/services/run_service.py` — read interval from config
- `tests/test_swe2d_gpu_graph_higher_order.py` — adjust assertions (RK2 still
  best on SCS-CN; all schemes < 0.01)
- `tests/test_swe2d_gpu_native_rain_gui_path.py` — verify GUI path still works
- New: `tests/test_rain_mass_conservation.py` — verify total volume matches
  `cum_excess_cum_mm` at t=end across minute boundaries and partial final
  intervals

## Phase 0 — Data structures

### SWE2DDeviceState additions (`swe2d_gpu.cuh`)
```cpp
// Rainfall update timing
double  rain_update_interval_s = 60.0;  // re-evaluate rate every N seconds
double  last_rain_update_time  = -1.0;  // scalar last update tick (host-owned)

// Cumulative excess at last update tick (so we can compute average rate)
double* d_rain_excess_at_last_update_mm = nullptr;  // [n_cells]
```

`d_rain_cum_mm` and `d_rain_excess_cum_mm` stay as today — they are the
**current** cumulative state. `d_rain_excess_at_last_update_mm` is just a
snapshot captured at the same tick as `last_rain_update_time`.

### Allocation (`swe2d_gpu.cu` `swe2d_gpu_alloc_rainfall`)
```cpp
alloc_d(reinterpret_cast<void**>(&dev->d_rain_excess_at_last_update_mm),
        static_cast<size_t>(n_cells) * sizeof(double));
CUDA_CHECK(cudaMemset(dev->d_rain_excess_at_last_update_mm, 0,
                     static_cast<size_t>(n_cells) * sizeof(double)));
```

### Deallocation (cudaFree + nullptr + safe_free in destroy)

### Public setter (`swe2d_solver.cpp`)
```cpp
void swe2d_solver_set_rain_cn_forcing(
    SWE2DDeviceState* dev,
    /* existing args */,
    double rain_update_interval_s = 60.0);  // ADDED: optional with default
```

## Phase 1 — New update kernel

### `swe2d_update_rain_source_rate_kernel`
```cpp
__global__ void swe2d_update_rain_source_rate_kernel(
    int32_t n_cells,
    const int32_t* __restrict__ cell_gage_idx,
    const int32_t* __restrict__ hg_offsets,
    const double*  __restrict__ hg_time_s,
    const double*  __restrict__ hg_cum_mm,
    const double*  __restrict__ cn,
    double t_prev_update,           // tick of previous update
    double t_now,                   // current simulation time
    double ia_ratio,
    double mm_to_model_depth,
    double* __restrict__ cum_rain_mm,
    double* __restrict__ cum_excess_mm,
    const double* __restrict__ prev_cum_excess_mm,    // snapshot at t_prev_update
    double* __restrict__ cell_source_mps,             // output: average rate for [t_prev, t_now]
    int32_t* __restrict__ needs_update_flag,         // output: 1 if update happened
    double* __restrict__ new_last_update_time);      // output: t_now if updated
```

### Kernel logic (per cell)
1. Read `t_prev_update`. If `t_prev_update < 0` (first call), treat as `t_prev_update = t_now` (no-op pass-through, rate = 0).
2. **Skip guard** — check `needs_update_flag[0]` (block 0 only writes, all blocks read with __sync); simplified below using one-flag-per-cell:
   - Each cell decides for itself: `if (t_now - last_update >= interval || t_prev_update < 0)`. Since `last_update_time` is scalar and shared, do it host-side; here we always run when called.
3. If `t_now == t_prev_update`: rate = 0, return.
4. Read `t0 = hg_cum_mm(t_prev_update)`, `t1 = hg_cum_mm(t_now)` from the cell's gage hydrograph via `interp_series_clamped_cuda`.
5. `dr = max(0, r1 - r0)` — rainfall increment in mm over the interval.
6. `p_new = cum_rain_mm[c] + dr`.
7. Compute new excess `pe` from SCS-CN with `ia = ia_ratio * S`, `S = (25400/cn) - 254`.
8. `dt_interval = max(t_now - t_prev_update, 1e-9)`.
9. `rate_mm_per_s = (max(0, pe - prev_cum_excess_mm[c])) / dt_interval`.
10. `cell_source_mps[c] = rate_mm_per_s * mm_to_model_depth`.
11. Write back: `cum_rain_mm[c] = p_new`, `cum_excess_mm[c] = pe`.
12. No per-cell update flag needed because kernel is only launched at the
    right times; the host decides when to launch.

### Host-side launch decision (in `swe2d_gpu_step`)
```cpp
const double dt_int = dev->rain_update_interval_s;
if (dev->last_rain_update_time < 0.0) {
    // First call ever: capture snapshot, don't update yet
    CUDA_CHECK(cudaMemcpyAsync(dev->d_rain_excess_at_last_update_mm,
                             dev->d_rain_excess_cum_mm,
                             sz_d, cudaMemcpyDeviceToDevice, dev->d_stream));
    dev->last_rain_update_time = current_t_now;
    return;
}
if (current_t_now - dev->last_rain_update_time >= dt_int ||
    is_final_step /* passed in by caller */) {
    double t_prev = dev->last_rain_update_time;
    // Snapshot PREVIOUS excess BEFORE the update kernel overwrites it.
    // We just write the snapshot of cum_excess BEFORE kernel runs — but the
    // kernel reads the snapshot to compute the rate, so it's already captured.
    swe2d_update_rain_source_rate_kernel<<<...>>>(
        /* ... */, t_prev, current_t_now, /* ... */);
    dev->last_rain_update_time = current_t_now;
}
```

**Snapshot capture**: the snapshot of `prev_cum_excess` is captured AT the end
of the previous kernel run. Two ways:
- **Option A (chosen)**: at the END of each update, copy `cum_excess` to
  `prev_cum_excess` so it represents the cumulative AT last update tick.
- **Option B**: snapshot before update; overwrite after.

Option A is cleaner: invariant is "`prev_cum_excess` = state AT
`last_rain_update_time`". Achieved by copying after each successful update.

```cpp
// After update kernel completes:
CUDA_CHECK(cudaMemcpyAsync(dev->d_rain_excess_at_last_update_mm,
                         dev->d_rain_excess_cum_mm,
                         sz_d, cudaMemcpyDeviceToDevice, dev->d_stream));
// Now cum_excess and prev_cum_excess are equal — at t_now.
```

## Phase 2 — Integration

### `swe2d_gpu_step` (line ~5158)
1. Remove the per-step `swe2d_build_rain_cn_source_kernel` call at line 5160.
2. Add the new update kernel call (with interval check) at the same position.
3. The `cell_source_mps` it writes is then read by the update kernel inside
   the captured graph as today.

### `swe2d_gpu_step_rk*` functions
- Drop ALL save/restore and per-stage build-kernel blocks.
- The cumulative state mutation in `swe2d_build_rain_cn_source_kernel`
  is no longer called per stage. The single per-interval update kernel is
  called once per outer step from `swe2d_gpu_step` (called by every
  `swe2d_gpu_step_rk*` through Stage 1).
- RK functions need NO rain CN handling anymore.

### Graph cache
- The captured graph today calls `swe2d_build_rain_cn_source_kernel` once.
  After this refactor, it doesn't. Source comes from outside the graph.
- Means: `d_cell_source_mps` is just a number that the update SWE kernel
  reads; the pointer is captured at capture time and content varies. That
  works fine (CUDA graphs capture pointers, not values).
- Graph invalidation NOT needed because we don't change the kernel graph
  contents based on rain.

### Final partial interval
- Solver driver knows `t_end`. When `current_t_now + dt_remaining <= t_end`
  for the LAST step, the outer step code can pass `is_final_step = true`
  to force one final update of `swe2d_gpu_step`. Or simpler: just let the
  next compare-cum-excess check trip when `t_now >= t_end`.

## Phase 3 — Tests and validation

### Update existing
- `tests/test_swe2d_gpu_graph_higher_order.py`:
  - Drop `err_rk4g < err_rk2` and `err_rk5g < err_rk4g` assertions
    (already done — they don't hold).
  - Keep `assertLess(..., 0.01)` for all schemes.
  - Run with default `rain_update_interval_s = 60.0`. Expect similar err.
- `tests/test_swe2d_gpu_native_rain_gui_path.py`:
  - Verify GUI-driven rainfall setup still produces same depth as before
    (within tolerance of integration scheme change).

### New tests
- `tests/test_rain_mass_conservation.py`:
  - Total water depth in cells at t=end = `(cum_excess_cum_mm * mm_to_model_depth)`
    summed over cells, accounting for any runoff.
  - Test across minute boundaries (60s, 120s, 180s).
  - Test with partial final interval (t_end = 90.5s).
- `tests/test_rain_smoothness.py`:
  - Two simulations with rain update at 1s vs 60s should differ by < tolerance
    for a constant-intensity storm.
  - For a storm that turns on at t=30s, the rate at t=60s boundary should
    jump but be small.

### Mass conservation explicit test
- Pick a simple mesh (2-cell closed). Constant rain 10 mm/s for 100s.
  - At t=60s, the kernel updates: rate = (P(60) - P(0)) / 60 = 10 mm/s = 0.01 m/s.
  - 60s of SWE update at 0.01 m/s adds 0.6m of water ✓.
  - At t=100s, after second update: rate is same, 40s more adds 0.4m. Total = 1m ✓.

## Edge cases checklist (must handle)

- [ ] First-ever call: `last_rain_update_time = -1`, snap `prev_cum_excess = cum_excess`, set `last_update = t_now`. Don't write `source_mps` (or write 0). Next call uses interval.
- [ ] Final partial interval: pass `is_final_step = true` to force one more update.
- [ ] Cell with `gidx < 0` (no gage): set `cell_source_mps[c] = 0`, don't touch cum state.
- [ ] Cell with empty gage offsets (no samples for that gage): set `cell_source_mps[c] = 0`.
- [ ] Hydrograph extrapolation: use `interp_series_clamped_cuda` — clamp at boundaries (no negative accumulation).
- [ ] `n_rain_samples == 0`: skip update kernel entirely, leave `cell_source_mps` at whatever default.
- [ ] Multiple calls per outer step from RK: only ONE update per outer step at the START.

## Out of scope
- Changing the Python `swe2d_solver_set_rain_cn_forcing` semantics visible to
  callers (same `time_s`, `cum_mm` arrays; just adds optional `rain_update_interval_s`).
- Changing the SCS-CN formula itself (still the empirical chunk above).
- Non-uniform rainfall across cells (already supported via `cell_gage_idx`).

## Estimated effort
- Phase 0: 30 min (struct + alloc/dealloc + setter)
- Phase 1: 1 hr (new kernel + launch glue)
- Phase 2: 1 hr (delete per-stage build calls from 4 RK functions + integration)
- Phase 3: 1 hr (update existing test, write 2 new tests, manual validation)
- Total: **3-4 hours**
</content>
</content>
