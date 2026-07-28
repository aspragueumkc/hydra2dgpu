---
type: plan
status: complete
created: 2026-07-22
completed: 2026-07-25
---

# Coupling Flux Buffer — Rename + Zero + Fix Structure Path

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task.

**Goal:** Fix the coupling flux buffer lifecycle (stale accumulation, dead structure path, misleading name) so both drainage and structure coupling actually work.

**Architecture:** Rename `d_ext_struct_flux_*` → `d_ext_cpl_flux_*` to reflect shared ownership by drainage + structures. Zero the buffer before every pipe1d step (not just inside the retired culvert path). Re-enable structure flow computation by resurrecting the relevant section of `compute_coupling_full_on_device` inside `apply_native_device_sources`. Keep a single shared buffer — both paths `atomicAdd` into it, the 2D update kernel reads it. No kernel signature changes on the 2D side.

**Tech Stack:** C++ CUDA (`swe2d_gpu.cu`, `swe2d_gpu.cuh`, `pipe1d.cu`), Python (`coupling.py`)

---

### Task 1: Rename `d_ext_struct_flux_*` → `d_ext_cpl_flux_*` across C++

**Files:**
- Modify: `cpp/src/swe2d_gpu.cuh` — struct member names + comments
- Modify: `cpp/src/swe2d_gpu.cu` — every reference (zero, allocate, read, write, debug print)
- Modify: `cpp/src/pipe1d.cu` — kernel parameters + atomicAdd sites + host wrapper

- [ ] **Step 1: Rename in header**

`swe2d_gpu.cuh:603-605`:
```cpp
double*  d_ext_cpl_flux_h  = nullptr;   // [n_cells] net mass flux from coupled drainage + structures (L³/T)
double*  d_ext_cpl_flux_hu = nullptr;   // [n_cells] net x-momentum flux
double*  d_ext_cpl_flux_hv = nullptr;   // [n_cells] net y-momentum flux
```

Also rename the alloc/readback/upload helper declarations at lines 1381-1391:
- `swe2d_gpu_alloc_ext_struct_flux` → `swe2d_gpu_alloc_cpl_flux`
- `swe2d_gpu_readback_ext_struct_flux` → `swe2d_gpu_readback_cpl_flux`
- `swe2d_gpu_upload_ext_struct_flux_h` → `swe2d_gpu_upload_cpl_flux_h`

Add a comment at the declaration:
```cpp
// This single buffer holds contributions from BOTH drainage (pipe1d SURFACE_2D faces)
// and structure (culvert face-flux kernel). Both paths atomicAdd into it.
// The 2D update kernel reads it and applies to h/hu/hv each step.
```

- [ ] **Step 2: Rename in `swe2d_gpu.cu`**

Find-and-replace across the entire file: `d_ext_struct_flux_` → `d_ext_cpl_flux_` and `ext_struct_flux` → `cpl_flux` in function names and local variables. This touches ~40 sites (zeroing, kernel calls, debug prints, the fold kernel, the update kernel call site at line 5771-5774, the alloc function at line 7904, the upload function).

The update kernel call at line 5743 remains unchanged in *signature* (same parameter position) — only the variable name changes.

- [ ] **Step 3: Rename in `pipe1d.cu`**

Find-and-replace: `d_ext_struct_flux_` → `d_ext_cpl_flux_` and `ext_struct_flux` → `cpl_flux` in function parameter names. This touches:
- The `swe2d_unified_face_flux_kernel` kernel parameter declarations (lines 1946-1948)
- The atomicAdd sites (lines 2451-2453, 2584-2586, 2635, 2665-2666)
- The `swe2d_gpu_apply_unified_face_flux` host wrapper (lines 3177-3179, 3468-3471)
- The godunov step caller (lines 3248-3250, 3312-3314)
- The fold kernel and fold kernel call sites
- Comments referencing `ext_struct_flux`

- [ ] **Step 4: Rename in `swe2d_bindings.cpp`**

Rename any binding that exposes `swe2d_gpu_readback_ext_struct_flux` or `swe2d_gpu_upload_ext_struct_flux_h`.

- [ ] **Step 5: Rename in Python**

`swe2d/runtime/coupling.py` — this file currently never reads or writes `d_ext_struct_flux_*` directly from Python (the GPU manages it internally). No Python rename needed unless there's a diagnostic readback call. Check:
```bash
grep -rn 'ext_struct_flux\|ext_cpl_flux' swe2d/
```

- [ ] **Step 6: Build + check compile**

```bash
cd build && mamba run -n qgis_stable /usr/bin/cmake --build . -j$(nproc) 2>&1 | tail -20
```
Expected: no undefined symbols, no mismatched declarations.

---

### Task 2: Zero `d_ext_cpl_flux_*` before every pipe1d step

**Files:**
- Modify: `cpp/src/pipe1d.cu` — zero in `swe2d_gpu_apply_unified_face_flux` host wrapper

- [ ] **Step 1: Add zeroing before kernel launch**

In `swe2d_gpu_apply_unified_face_flux` (`pipe1d.cu:3367`), add `cudaMemsetAsync` for the three buffers right after the null-guard returns and before launching the kernel. The function already takes `d_ext_cpl_flux_h/hu/hv` as parameters and already has a `stream` argument:

```cpp
// Zero the external coupling flux accumulators before the kernel
// atomicAdds into them.  Both drainage (class 3/4/5) and structure
// (class 6) faces contribute to these buffers.  Without this zero,
// stale fluxes from the previous step accumulate.
if (d_ext_cpl_flux_h && n_cells_2d > 0) {
    const size_t sz = static_cast<size_t>(n_cells_2d) * sizeof(double);
    CUDA_CHECK(cudaMemsetAsync(d_ext_cpl_flux_h,  0, sz, stream));
    CUDA_CHECK(cudaMemsetAsync(d_ext_cpl_flux_hu, 0, sz, stream));
    CUDA_CHECK(cudaMemsetAsync(d_ext_cpl_flux_hv, 0, sz, stream));
}
```

Insert this at `pipe1d.cu` around line 3405, just before the `if (n_faces <= 0) return;` block.

- [ ] **Step 2: Remove redundant zeroing elsewhere**

In `swe2d_gpu.cu`, the zeroing inside `swe2d_gpu_compute_coupling_full_on_device` (lines 7690-7692) and `swe2d_gpu_apply_culvert_face_flux` (lines 8033-8038) is now redundant but harmless. Add a comment noting the zero is now handled by `swe2d_gpu_apply_unified_face_flux`. Do NOT remove them yet — the retired `compute_coupling_full_on_device` may be resurrected (see Task 4).

- [ ] **Step 3: Build + run pipe1d tests**

```bash
cd build && mamba run -n qgis_stable /usr/bin/cmake --build . -j$(nproc)
cd .. && find . -type d -name __pycache__ -exec rm -rf {} +
mamba run -n qgis_stable env PYTHONPATH="$PWD:$PWD/build" \
    python3 -m unittest -v tests.test_pipe1d_face_indexed_mesh 2>&1 | tail -10
```
Expected: 11/11 PASS (no functional change, just lifecycle fix).

---

### Task 3: Fix `swe2d_gpu_alloc_cpl_flux` to zero on allocation

**Files:**
- Modify: `cpp/src/swe2d_gpu.cu` — alloc function

- [ ] **Step 1: Switch from `cudaMalloc` to `cudaMalloc + cudaMemset`**

In the renamed `swe2d_gpu_alloc_cpl_flux` (currently `swe2d_gpu_alloc_ext_struct_flux`, ~line 7904), add zeroing after each `cudaMalloc`:

```cpp
cudaMemset(d_ext_cpl_flux_h,  0, sz);
cudaMemset(d_ext_cpl_flux_hu, 0, sz);
cudaMemset(d_ext_cpl_flux_hv, 0, sz);
```

This ensures the first step reads zero even if the unified face kernel hasn't run yet.

- [ ] **Step 2: Build**

```bash
mamba run -n qgis_stable /usr/bin/cmake --build build -j$(nproc) 2>&1 | tail -5
```

---

### Task 4: Resurrect structure flow computation in the hot path

**Files:**
- Modify: `swe2d/runtime/coupling.py` — `apply_native_device_sources`
- Modify: `cpp/src/swe2d_gpu.cu` — expose structure flow computation as a separate entry point

- [ ] **Step 1: Decide entry point**

Two options:
- **A:** Call the full `swe2d_gpu_compute_coupling_full_on_device` from Python, but with the `host_structure_flows` path only (skip the face-flux kernel part since pipe1d already ran)
- **B:** Extract just the structure flow computation + culvert face-flux kernel call into a new function `swe2d_gpu_apply_structure_coupling`, call it from Python after `swe2d_pipe1d_step`

**Recommendation: B.** The full function zeroes `d_external_source_mps` which would wipe drainage contributions. Extracting a targeted function is safer.

Define in `swe2d_gpu.cu`:
```cpp
void swe2d_gpu_apply_structure_coupling(
    SWE2DDeviceState* dev,
    int32_t n_cells,
    int32_t n_structures,
    bool use_culvert_face_flux,
    double dt,
    double gravity,
    double h_min)
```

This function:
1. Computes structure flows via `swe2d_compute_structure_flows_kernel` (reads `sf_ws.d_cell_wse`)
2. If `use_culvert_face_flux`: runs `swe2d_culvert_face_flux_kernel` which atomicAdds to `d_ext_cpl_flux_*`
3. If NOT `use_culvert_face_flux`: runs `swe2d_coupling_structure_source_kernel` which writes to `d_external_source_mps` (non-culvert only)
4. Does NOT zero any buffers (already zeroed by Task 2)

- [ ] **Step 2: Bind in `swe2d_bindings.cpp`**

```cpp
m.def("swe2d_gpu_apply_structure_coupling",
    [](int32_t n_cells, int32_t n_structures,
       bool use_culvert_face_flux, double dt,
       double gravity, double h_min) {
        extern SWE2DDeviceState* s_coupling_dev;
        swe2d_gpu_apply_structure_coupling(
            s_coupling_dev, n_cells, n_structures,
            use_culvert_face_flux, dt, gravity, h_min);
    },
    "...");
```

- [ ] **Step 3: Wire into `apply_native_device_sources`**

In `coupling.py`, after the `swe2d_pipe1d_step` call (line 1791) and after the culvert face-flux preload (line 1813), add:

```python
if self.structures is not None and self._structure_count > 0:
    native_mod.swe2d_gpu_apply_structure_coupling(
        int(self.n_cells),
        int(self._structure_count),
        bool(self.culvert_face_flux_mode == "face_flux"),
        float(dt_s),
        float(self._gravity),
        float(self._h_min),
    )
```

Also remove the comment at line 1821 claiming `compute_coupling_full_on_device` is retired — it IS now, replaced by the targeted call.

- [ ] **Step 4: Remove the stale `swe2d_gpu_fold_culvert_mass_to_source` call at line 1850**

The fold-at-source path at coupling.py:1848-1855 is now superseded by the face-flux path. Delete that block (lines 1848-1855).

- [ ] **Step 5: Build + test**

```bash
mamba run -n qgis_stable /usr/bin/cmake --build build -j$(nproc)
# Run whatever tests currently pass to verify no regression:
mamba run -n qgis_stable env PYTHONPATH="$PWD:$PWD/build" \
    python3 -m unittest -v tests.test_pipe1d_face_indexed_mesh 2>&1 | tail -10
```

---

### Task 5: Port `TestPipeEndExchange` to cover the fixed path

**Files:**
- Modify: `tests/test_swe2d_gpu_drainage_network.py` — `TestPipeEndExchange` class

- [ ] **Step 1: Remove the dead `hasattr` gate**

Line 908: change `@unittest.skipUnless(hasattr(_MOD, "swe2d_gpu_apply_pipe_end_bc")` to remove this skip condition. The test should run whenever the unified mesh build + pipe1d_step are available (which is always for GPU builds).

- [ ] **Step 2: Port setUp to use `swe2d_pipe1d_step` with solver_dev**

Instead of calling the old `swe2d_gpu_apply_pipe_end_bc`, the test should:
1. Build the unified mesh via `swe2d_build_unified_mesh`
2. Upload pipe-end surface faces via `swe2d_pipe1d_upload_pipe_end_surface_faces`
3. Initialize cell area via `swe2d_pipe1d_init_cell_area`
4. Run `swe2d_pipe1d_step(dev_ptr, dt, ...)` with the solver_dev pointer
5. Read back either `d_ext_cpl_flux_h` via `swe2d_gpu_readback_cpl_flux` or cell depths via `swe2d_gpu_readback_snapshots`

- [ ] **Step 3: Add mass conservation assertion**

The test `test_wet_pipe_drains_into_dry_surface_cells` should measure:
```
pipe_volume_before = sum(A_pipe * cell_length)
surface_volume_before = sum(h_surface * cell_area)
run step
pipe_volume_after = sum(A_pipe * cell_length)
surface_volume_after = sum(h_surface * cell_area)
assert |(pipe_vol + surf_vol)_after - (pipe_vol + surf_vol)_before| < tolerance
```

- [ ] **Step 4: Run the test**

```bash
mamba run -n qgis_stable env PYTHONPATH="$PWD:$PWD/build" \
    python3 -m unittest -v \
    tests.test_swe2d_gpu_drainage_network.TestPipeEndExchange 2>&1 | tail -20
```
Expected: both tests PASS (no more skipped).

---

### Task 6: Cleanup — remove dead bindings

**Files:**
- Modify: `cpp/src/swe2d_bindings.cpp`

- [ ] **Step 1: Remove old binding exports**

Delete the bindings for:
- `swe2d_gpu_compute_coupling_full_on_device` (line 1153)
- `swe2d_gpu_upload_outfall_free_bc_nodes` — actually, keep this as a no-op if Python still calls it, but add a deprecation warning. Or remove the Python call site in Task 4 step 4.

- [ ] **Step 2: Build**

```bash
mamba run -n qgis_stable /usr/bin/cmake --build build -j$(nproc) 2>&1 | tail -10
```
Expected: clean compile, no test regressions.
