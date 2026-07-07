# Per-Step GPU↔CPU Transfer Elimination Plan

**Goal:** Eliminate ALL device-to-host (D2H) and host-to-device (H2D) transfers
from the per-step hot path. After these changes, the only CPU↔GPU communication
during the solve loop will be a single kernel launch + graph replay per step.
All state changes (rain, BC updates, coupling, redistribution) happen
on-device via GPU kernels.

**Guiding principle:** No backwards compatibility. If a code path exists only
to support an older approach that has been superseded, delete it entirely.
Every "legacy" or "fallback" path is a maintenance burden and a performance
liability.

---

## C1: CFL `lambda_max` D2H Readback + Stream Sync

**File:** `cpp/src/swe2d_gpu.cu` — `swe2d_gpu_compute_dt()`

### What it does
Every step (when `dt_fixed <= 0.0` — the default), the function:
1. Runs `swe2d_cfl_kernel` + `swe2d_cfl_reduce_blocks_kernel` on-device
2. `cudaStreamSynchronize(dev->d_stream)` — flattens pipeline (line 6324)
3. `cudaMemcpy(&lambda_max, d_lambda_max, 8B, D2H)` — sync readback (line 6331)

### Plan: Accept one-step-lag `lambda_max`

Replace the synchronous readback with a stale-value approach: use the previous
step's `lambda_max` to compute dt, while the next reduction runs concurrently
in the stream.

**Changes:**

1. **`swe2d_gpu.cu` — Rewrite `swe2d_gpu_compute_dt()`:**
   - Remove `cudaStreamSynchronize(dev->d_stream)` (line 6324)
   - Replace `cudaMemcpy(&lambda_max, d_lambda_max, ..., D2H)` with
     `cudaMemcpyAsync(&lambda_max, d_lambda_max, ..., D2H, dev->d_stream)`
     followed immediately by `cudaEventRecord(dev->d_dt_event, dev->d_stream)`
   - Store the event + a pinned host copy. On the next call, do
     `cudaEventSynchronize(dev->d_dt_event)` (non-blocking if already
     completed) then read the pinned buffer.
   - First call: seed with `cfl_factor / 1.0` (safe conservative default)
     or accept the initial `dt_initial` from the solver config.

2. **`swe2d_solver.cpp` — Update call site (line 235):**
   - Call is unchanged — signature stays `double swe2d_gpu_compute_dt(...)`.
   - The one-step lag is trivially safe for CFL stability: wave speeds
     change slowly relative to the timestep.

3. **`swe2d_gpu.cuh` — Add to `SWE2DDeviceState`:**
   ```cpp
   cudaEvent_t d_dt_event;   // records when lambda_max is available
   double h_lambda_max;      // pinned host copy
   bool h_lambda_max_valid;  // false until first compute
   ```

**Files affected:** `cpp/src/swe2d_gpu.cu`, `cpp/src/swe2d_gpu.cuh`,
`cpp/src/swe2d_solver.cpp`

**Removed:** The per-step `cudaStreamSynchronize` stall. Stream continues
executing subsequent work while the 8-byte D2H resolves asynchronously.

---

## C2: Per-Step H2D Source Uploads (Rain + Cell Sources)

**Files:** `cpp/src/swe2d_gpu.cu` — `swe2d_gpu_accumulate_external_source()`
and `swe2d_gpu_set_external_sources()`

### What it does
There are **two** H2D uploads on the per-step hot path:

1. **`accumulate_external_source` (line 2823–2825):**
   When `native_rain_cn_forcing=False`: Python computes rain rates and calls
   `backend.accumulate_external_sources_native(rain_src)`, which does a
   **synchronous** `cudaMemcpy` (not async!) of `n_cells × 8 bytes` to
   `d_drainage_q`, then launches an accumulation kernel.

2. **`set_external_sources` (line 7936):**
   When native coupling is NOT active (`_native_device_applied=False`):
   `apply_external_sources_callback` calls `backend.set_external_sources_native()`
   which does `cudaMemcpyAsync` of `n_cells × 8 bytes` to
   `d_external_source_mps` every step. This path is triggered by the
   `apply_external_sources_callback` branch in `runtime_step_executor.py:116`.

### Root cause
The non-native rain CN path still exists. The native path
(`swe2d_solver_set_rain_cn_forcing`) already uploads rain hyetographs + CN data
once at configure time and computes rain rates entirely on-device. But there's
a flag `native_rain_cn_forcing` that defaults to False in some code paths,
causing the Python rain-rate computation + H2D upload path to still run.

### Plan: Delete non-native rain CN path completely

1. **`swe2d/runtime/runtime_sources.py`:**
   - Remove the `native_rain_cn_forcing` flag and branching
   - `rain_source_for_window()` becomes a no-op that always returns `0.0`
   - Rain is always handled on-device via `swe2d_solver_set_rain_cn_forcing`
   - Delete the non-native `rain_rate_model` / `rain_rate_si_to_model` path

2. **`swe2d/runtime/backend.py`:**
   - Remove `set_external_sources_native()` (the overwrite path)
   - Keep `accumulate_external_sources_native()` as it's used to fold
     structure/drainage source rates on-device when coupling is active
   - But change the `cudaMemcpy` at line 2825 from synchronous to
     `cudaMemcpyAsync` on `dev->d_stream` — the stream ordering ensures
     the data arrives before the accumulation kernel runs

3. **`swe2d/runtime/runtime_step_executor.py`:**
   - Remove the `if not _native_device_applied:` branch (lines 109–115)
     that calls `accumulate_source_volume_model_callback` +
     `apply_external_sources_callback`
   - The `_native_device_applied` flag becomes purely diagnostic — when it's
     False, the solver step runs with zero external sources (which should
     not happen in normal operation since coupling is always enabled when
     structures/drainage exist, and rain is always natively handled)
   - Remove `apply_external_sources_callback` parameter

4. **`swe2d/workbench/workers/simulation_worker.py`:**
   - Remove `apply_external_sources_callback` wiring from `_WorkbenchShim`
   - Remove `cell_source_model_at_time_callback` wiring (cell sources should
     be handled via native device path or not exist)

5. **`swe2d/boundary_and_forcing/` + `swe2d/workbench/services/non_gui_runtime_service.py`:**
   - Clean up all callers that configure the non-native rain path
   - Remove `native_rain_cn_forcing` parameter everywhere

**Also** — fix the synchronous `cudaMemcpy` at line 2823-2825:
```cpp
// BEFORE: synchronous
CUDA_CHECK(cudaMemcpy(cpl_ws.d_drainage_q, host_src,
                      n_cells * sizeof(double), cudaMemcpyHostToDevice));

// AFTER: async on stream
CUDA_CHECK(cudaMemcpyAsync(cpl_ws.d_drainage_q, host_src,
                           n_cells * sizeof(double), cudaMemcpyHostToDevice,
                           dev->d_stream));
```

**Files affected:**
- `cpp/src/swe2d_gpu.cu` (lines 2823–2825: sync→async)
- `swe2d/runtime/runtime_sources.py` (remove non-native rain path)
- `swe2d/runtime/backend.py` (remove `set_external_sources_native`)
- `swe2d/runtime/runtime_step_executor.py` (remove apply_external_sources branch)
- `swe2d/runtime/run_options_builder.py` (clean up)
- `swe2d/workbench/workers/simulation_worker.py` (remove callbacks)
- `swe2d/workbench/services/non_gui_runtime_service.py` (remove parameter)
- `swe2d/boundary_and_forcing/` (clean up native_rain_cn_forcing flag)

---

## C3: Persistent Chunk Active Edge Count D2H Readback

**File:** `cpp/src/swe2d_gpu.cu` — around line 6131

### What it does
When persistent chunk mode (`tiny_mode=3`) is active WITH active edge
compaction enabled, the code reads back `n_flux_edges` (4 bytes D2H) every
`tiny_compaction_stride` steps (default 8). This scalar tells the persistent
chunk kernel how many edges to iterate over.

### Status: Already eliminated

Wait — let me verify. Looking at the code path:

```cpp
if (enable_active_edge_compaction && dev->d_active_edge_ids && dev->d_n_active_edges) {
    CUDA_CHECK(cudaMemsetAsync(dev->d_n_active_edges, 0, sizeof(int32_t), dev->d_stream));
    swe2d_collect_active_edges_kernel<<<...>>>(...);
    CUDA_CHECK(cudaMemcpy(&n_flux_edges, dev->d_n_active_edges, sizeof(int32_t), cudaMemcpyDeviceToHost));
    ...
}
```

The `n_flux_edges` is only used to set `d_flux_edge_ids` and control the
persistent chunk loop. If the kernel knows the maximum possible edges and uses
a conditional check per edge instead of a dynamic loop bound, the readback is
unnecessary.

### Plan: Replace scalar readback with max-edges bound

Since `n_edges` is known at compile time (it's static for the mesh geometry),
we can always pass `n_edges` as the loop bound and let the kernel's
per-edge activation check filter inactive edges. The kernel already has
`dev->d_active` — checking it costs one extra load per edge but eliminates
the D2H round-trip entirely.

**Changes in `swe2d_gpu.cu` (around line 6131):**

```cpp
// BEFORE: D2H readback of n_flux_edges
cudaMemcpy(&n_flux_edges, dev->d_n_active_edges, sizeof(int32_t), cudaMemcpyDeviceToHost);
d_flux_edge_ids = dev->d_active_edge_ids;
// ... use n_flux_edges as loop bound in persistent chunk kernel

// AFTER: use n_edges as loop bound, kernel filters by d_active
// Remove the cudaMemcpy entirely.
d_flux_edge_ids = dev->d_active_edge_ids;
```

The persistent chunk kernel should check `d_active[edge_id]` before processing.
If compaction is disabled, it already processes all `n_edges`.

**Also check:** Is `tiny_mode=3` (persistent chunk) even the default, or is
`tiny_mode=1` (auto) the default? Default is `kTinyModeAuto=1` which selects
the fused path — the persistent chunk compaction path may not even be active.
If it's not the default, this is lower priority but still correct to fix.

**Default is `kTinyModeAuto=1`** — the persistent chunk compaction readback
only fires when a user explicitly sets `tiny_mode=3` AND
`tiny_enable_active_compaction=true`. However, fix it anyway for correctness
when someone enables that experimental mode.

**Files affected:** `cpp/src/swe2d_gpu.cu` (remove D2H at line ~6131)

---

## C4: Structure Flows D2H Round-Trip + Legacy Redistribution

**Files:**
- `swe2d/runtime/coupling.py` (lines 1354–1383: round-trip)
- `cpp/src/swe2d_gpu_redistribute.cu` (lines 188–320: legacy path)
- `cpp/src/swe2d_gpu.cu` (line 8494: readback)

### What it does
When the persistent on-device redistribution function IS available (line 1359:
`hasattr(native_mod, "swe2d_gpu_redistribute_structure_sources_persistent")`),
the code still does a **D2H readback** of structure flows:

```python
nb_flows = np.asarray(
    native_mod.swe2d_gpu_readback_structure_flows(nb_n),  # D2H!
    dtype=np.float64,
)
```

These flows are then immediately re-uploaded H2D as arguments to
`swe2d_gpu_redistribute_structure_sources_persistent()`.

The structure flows are a tiny array (n_structures × 8 bytes, typically
10s–100s of elements). The round-trip is architecturally wrong but
performance impact is small.

More critically, the **legacy** function
`swe2d_gpu_redistribute_structure_sources()` (lines 188–320 in
`swe2d_gpu_redistribute.cu`) does a full `n_cells × 8 bytes` D2H download —
this is the real performance killer. It's wrapped in the `_apply_redistribution`
method at coupling.py:957.

### Plan: Nuke legacy redistribution, fix persistent path readback

**Step 1: Delete `swe2d_gpu_redistribute_structure_sources` entirely**

In `cpp/src/swe2d_gpu_redistribute.cu`, remove the legacy function (lines
188–320). The only caller in `coupling.py` (`_apply_redistribution` at line
957) checks for the persistent function first — if the persistent function
exists, the legacy path is dead code.

Also remove the pybind11 binding in `swe2d_bindings.cpp` (line 1542).

**Step 2: Make structure flows available on-device**

The persistent redistribution function at line 1379 needs `nb_flows` as an
argument. Currently the flows are read back from GPU just to be sent right
back. Instead, have `compute_coupling_full_on_device` leave the flows in a
known device buffer that the redistribution kernel can read directly.

Two approaches:
- **Option A (preferred):** Modify `swe2d_gpu_redistribute_structure_sources_persistent`
  to read flows directly from `dev->sf_ws.d_structure_flow` instead of
  taking a host pointer. This eliminates the Python middleman entirely.
- **Option B (simpler):** Keep the host pointer argument but change the
  Python side to source flows from the last cached copy
  (`self._last_structure_flows`) instead of doing a fresh D2H readback.
  The flows are already cached at line 1373: `self._last_structure_flows =
  nb_flows.copy()`. On the second step, use the cached copy.

**Option A is strongly preferred.** It requires a C++ change but removes
the entire round-trip at the architectural level.

**Step 3: Remove `swe2d_gpu_readback_structure_flows` from the Python path**

Once Option A is implemented, remove `swe2d_gpu_readback_structure_flows`
binding and the `_last_structure_flows` caching in `coupling.py`.

**Files affected:**
- `cpp/src/swe2d_gpu_redistribute.cu` (delete legacy function lines 188–320)
- `cpp/src/swe2d_bindings.cpp` (remove legacy binding line ~1542)
- `swe2d/runtime/coupling.py` (remove readback, fix persistent call)
- `cpp/src/swe2d_gpu.cu` (optionally remove readback function line 8494)

---

## W1: `native_bc_forcing` — Delete Non-Native BC Path

**Files:** `swe2d/runtime/runtime_step_executor.py` (lines 61–86),
`non_gui_runtime_service.py`, `simulation_worker.py`

### What it does
Every step, when `dynamic_bc` is True AND `native_bc_forcing` is False:
- Python evaluates timeseries hydrographs for each BC edge
- Calls `distribute_total_flow_to_unit_q_callback` (Python)
- Uploads BC values H2D via `backend.set_boundary_conditions()`
- This is a Python→CUDA round-trip every step

The native BC forcing path (`SWE2DNativeBoundaryHydrographConfigurator`)
uploads all hydrograph data once at configure time and GPU kernel
interpolates per-step. It's strictly superior.

### Plan: Make native BC forcing mandatory

1. **`swe2d/runtime/runtime_step_executor.py`:**
   - Remove `native_bc_forcing` parameter
   - Remove the entire `if dynamic_bc and not native_bc_forcing:` block
     (lines 61–86)
   - Remove `apply_timeseries_bc_values_callback`,
     `distribute_total_flow_to_unit_q_callback`,
     `uniform_inflow_velocity_normalize_callback` parameters

2. **`swe2d/workbench/services/non_gui_runtime_service.py`:**
   - Remove `native_bc_forcing` parameter from `execute_run_timestep_loop()`
   - Remove the `SWE2DNativeBoundaryHydrographConfigurator` skip path —
     always configure it

3. **`swe2d/workbench/workers/simulation_worker.py`:**
   - In `_WorkbenchShim`, remove the closure-based
     `_apply_timeseries_bc_values` wrapper — it was only needed for the
     non-native path
   - Remove `distribute_total_flow_to_unit_q_callback` wiring
   - Always create and upload native hydrograph data via
     `BoundaryHydrographConfigurator`

4. **`swe2d/boundary_and_forcing/native_bc_forcing.py`:**
   - This file becomes the ONLY BC path — clean up any feature flags or
     conditional imports

**Files affected:**
- `swe2d/runtime/runtime_step_executor.py`
- `swe2d/workbench/services/non_gui_runtime_service.py`
- `swe2d/workbench/workers/simulation_worker.py`
- `swe2d/boundary_and_forcing/native_bc_forcing.py`
- `swe2d/workbench/controllers/run_controller.py` (where `dynamic_bc`
  detection happens)

---

## W2: Coupling Graph Cache Invalidation

**File:** `swe2d/runtime/coupling.py` (lines 1304–1305),
`cpp/src/swe2d_gpu.cu` (line 8592: `use_culvert_face_flux = false/true`)

### What it does
After `compute_coupling_full_on_device` runs, the code invalidates the CUDA
graph because `dev->use_culvert_face_flux` changed from `false` (pre-coupling)
to `true` (post-coupling, because the culvert solver is about to be used).
The graph signature includes this flag, so the old graph can't be replayed.

### Analysis: This is necessary
The `use_culvert_face_flux` flag is fundamental to the kernel launch
configuration — it determines whether `d_ext_struct_flux_h/hu/hv` are
passed as kernel arguments. It can't be set at configure time because the
culvert face-flux geometry is only known after `coupling_controller` is
built, which happens during the run setup, not at backend initialization.

The graph invalidation only fires on the **first coupling step** of a run
that has culverts with face-flux mode enabled. After re-capture, all
subsequent steps replay the cached graph.

**Verdict:** Keep as-is. The one-time re-capture cost is negligible. Add a
comment explaining why it's necessary and that it only fires once.

No changes needed.

---

## W3: Legacy Redistribution Path

**File:** `cpp/src/swe2d_gpu_redistribute.cu` (lines 188–320)

### What it does
The legacy `swe2d_gpu_redistribute_structure_sources()` function:
- Allocates GPU memory every step (`cudaMalloc` inside a graph region — BAD)
- Uploads redistribution geometry every step (no content-hash tracking)
- Runs redistribution kernel
- Synchronizes stream
- Downloads full `n_cells × 8 bytes` D2H
- Frees GPU memory every step (`cudaFree` inside a graph region — BAD)

The persistent path (`swe2d_gpu_redistribute_structure_sources_persistent`):
- Uses device-resident buffers (one-time alloc)
- Content-hash tracking to skip geometry re-upload (lines 164–184)
- Operates directly on `d_external_source_mps` — no D2H readback
- No per-step alloc/free

### Plan: Delete entirely with C4

The legacy redistribution path is completely subsumed by the persistent path
in C4's scope. The `_apply_redistribution` method in `coupling.py` at line
957 already checks for the persistent function first and only falls back to
legacy if it's missing. Since there is no backward compatibility, the legacy
path has no justification.

Delete the following:
- `cpp/src/swe2d_gpu_redistribute.cu` lines 188–320
- The pybind11 binding at `swe2d_bindings.cpp` line 1542
- The `_apply_redistribution` method's legacy branch in `coupling.py:957`
  (though this may already be dead if `swe2d_gpu_redistribute_structure_sources_persistent`
  is always available — verify and remove the dead code)

---

## Implementation Order

1. **W1: Delete non-native BC path** — Biggest structural change, touches the
   most files. Do this first as it removes the `apply_timeseries_bc_values_callback`
   and related parameters that other work might otherwise need to touch.

2. **C2: Delete non-native rain CN path + fix sync memcpy** — Remove
   `set_external_sources_native`, make native rain mandatory, change sync
   `cudaMemcpy` to async.

3. **C4 + W3: Nuke legacy redistribution + fix persistent readback** —
   Remove `swe2d_gpu_redistribute_structure_sources`, fix the structure
   flows readback in the persistent path.

4. **C1: CFL lambda_max one-step lag** — Self-contained C++ change, no
   Python involvement. Lower risk than the Python restructuring.

5. **C3: Active edge compaction readback** — Only matters for experimental
   persistent chunk mode. Lowest priority. Can be done at any time.

## Summary of Removals

| Item | What's deleted | Lines removed |
|------|---------------|---------------|
| W1 | Non-native BC path: callbacks, dynamic_bc branch, runtime_step_executor BC block | ~50 Python |
| C2 | `set_external_sources_native()`, `native_rain_cn_forcing` flag, non-native rain path | ~80 Python, ~30 C++ |
| C4+W3 | Legacy `swe2d_gpu_redistribute_structure_sources()`, binding, `_apply_redistribution` fallback | ~130 C++, ~20 Python |
| C1 | `cudaStreamSynchronize` in dt compute | 1 line C++ |
| C3 | D2H readback of `n_active_edges` | 3 lines C++ |

**Total: ~260 lines of code deleted, ~45 Python + ~165 C++ removed.**
