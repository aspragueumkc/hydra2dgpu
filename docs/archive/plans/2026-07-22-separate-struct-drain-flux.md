---
type: plan
status: complete
created: 2026-07-22
completed: 2026-07-25
---

# Separate Structure and Drainage Coupling Flux Buffers

**Goal:** Untangle the structure coupling path from the drainage coupling path by giving each its own GPU flux buffer, and re-enable the dead structure flow computation.

**Architecture:** Add `d_ext_drain_flux_h/hu/hv` for drainage (pipe1d SURFACE_2D faces class 3/4/5). Keep `d_ext_struct_flux_h/hu/hv` for structures only (class-6 CULVERT + standalone culvert face-flux kernel). The 2D update kernel reads both buffers and sums them per cell. Zero each buffer independently before its owning path writes. Structure coupling (culvert lookup tables, empirical weir/orifice equations) is fully independent of the pipe1d solver — no shared state beyond the final sum in the update kernel.

---

### Task 1: Add `d_ext_drain_flux_*` buffer to device state

**Files:**
- Modify: `cpp/src/swe2d_gpu.cuh` — add new members
- Modify: `cpp/src/swe2d_gpu.cu` — alloc helper + zero-on-alloc + ensure function

- [ ] **Step 1: Declare new buffer in header**

In `SWE2DDeviceState` (swe2d_gpu.cuh), right after the existing `d_ext_struct_flux_*` (lines 603-605):

```cpp
double*  d_ext_struct_flux_h  = nullptr;   // [n_cells] structure coupling mass flux (L³/T)
double*  d_ext_struct_flux_hu = nullptr;   // [n_cells] structure coupling x-momentum flux
double*  d_ext_struct_flux_hv = nullptr;   // [n_cells] structure coupling y-momentum flux

double*  d_ext_drain_flux_h   = nullptr;   // [n_cells] drainage coupling mass flux (L³/T)
double*  d_ext_drain_flux_hu  = nullptr;   // [n_cells] drainage coupling x-momentum flux
double*  d_ext_drain_flux_hv  = nullptr;   // [n_cells] drainage coupling y-momentum flux
```

Also declare `swe2d_gpu_alloc_drain_flux` near the existing `swe2d_gpu_alloc_ext_struct_flux` declaration at line 1381.

- [ ] **Step 2: Add alloc function**

In `swe2d_gpu.cu`, after `swe2d_gpu_alloc_ext_struct_flux` (~line 7904):

```cpp
void swe2d_gpu_alloc_drain_flux(SWE2DDeviceState* dev, int32_t n_cells) {
    if (!dev || n_cells <= 0) return;
    if (dev->d_ext_drain_flux_h) return;  // already allocated
    const size_t sz = static_cast<size_t>(n_cells) * sizeof(double);
    CUDA_CHECK(cudaMalloc(&dev->d_ext_drain_flux_h,  sz));
    CUDA_CHECK(cudaMalloc(&dev->d_ext_drain_flux_hu, sz));
    CUDA_CHECK(cudaMalloc(&dev->d_ext_drain_flux_hv, sz));
    CUDA_CHECK(cudaMemset(dev->d_ext_drain_flux_h,  0, sz));
    CUDA_CHECK(cudaMemset(dev->d_ext_drain_flux_hu, 0, sz));
    CUDA_CHECK(cudaMemset(dev->d_ext_drain_flux_hv, 0, sz));
}
```

- [ ] **Step 3: Add ensure-and-zero call in `swe2d_gpu_step`**

In `swe2d_gpu.cu`, in the section at ~line 5733 that currently ensures `ext_struct_flux` is allocated when pipe1d has faces:

```cpp
// Drainage coupling flux buffer: allocated + zeroed before the update
// kernel reads it.  The pipe1d step writes to this buffer (via atomicAdd
// in swe2d_unified_face_flux_kernel for class 3/4/5 faces).  We zero
// here so the update kernel always sees a clean accumulator, regardless
// of whether the pipe1d step ran.
if (dev->pipe1d.n_faces > 0
    && dev->pipe1d.d_flux_Q_scratch
    && dev->d_h && dev->d_cell_zb) {
    swe2d_gpu_alloc_drain_flux(dev, n_cells);
    const size_t sz = static_cast<size_t>(n_cells) * sizeof(double);
    CUDA_CHECK(cudaMemsetAsync(dev->d_ext_drain_flux_h,  0, sz, dev->d_stream));
    CUDA_CHECK(cudaMemsetAsync(dev->d_ext_drain_flux_hu, 0, sz, dev->d_stream));
    CUDA_CHECK(cudaMemsetAsync(dev->d_ext_drain_flux_hv, 0, sz, dev->d_stream));
}
```

- [ ] **Step 4: Build check**

```bash
cd build && mamba run -n qgis_stable /usr/bin/cmake --build . -j$(nproc) 2>&1 | tail -10
```
Expected: clean compile, no undefined symbols.

---

### Task 2: Route drainage SURFACE_2D faces (class 3/4/5) to `d_ext_drain_flux_*`

**Files:**
- Modify: `cpp/src/pipe1d.cu` — unified face flux kernel + host wrapper

- [ ] **Step 1: Add drain flux parameters to `swe2d_unified_face_flux_kernel`**

Add three new kernel parameters after the existing `d_ext_struct_flux_*` params (lines 1946-1948):

```cpp
double*                     d_ext_struct_flux_h,
double*                     d_ext_struct_flux_hu,
double*                     d_ext_struct_flux_hv,
double*                     d_ext_drain_flux_h,     // ← new
double*                     d_ext_drain_flux_hu,    // ← new
double*                     d_ext_drain_flux_hv,    // ← new
```

- [ ] **Step 2: Change atomicAdd targets in class-3/4/5 branches**

In the SURFACE_2D_PIPE_END branch (line 2451-2453):
```cpp
// was: atomicAdd(&d_ext_struct_flux_h[R], fh);
atomicAdd(&d_ext_drain_flux_h[R], fh);
atomicAdd(&d_ext_drain_flux_hu[R], fhu);
atomicAdd(&d_ext_drain_flux_hv[R], fhv);
```

In the SURFACE_2D_INLET branch (line 2584-2586):
```cpp
atomicAdd(&d_ext_drain_flux_h[R], -Q);
atomicAdd(&d_ext_drain_flux_hu[R], -u_2d * dh_2d);
atomicAdd(&d_ext_drain_flux_hv[R], -v_2d * dh_2d);
```

In the SURFACE_2D_JUNCTION_OVERFLOW branch (line 2635):
```cpp
atomicAdd(&d_ext_drain_flux_h[R], Q);
```

Class-6 CULVERT (line 2665-2666) stays on `d_ext_struct_flux_*`:
```cpp
atomicAdd(&d_ext_struct_flux_h[L], -Q_struct);
atomicAdd(&d_ext_struct_flux_h[R], +Q_struct);
```

- [ ] **Step 3: Update host wrapper declaration + definition**

`swe2d_gpu_apply_unified_face_flux` in pipe1d.cu (line 3166 declaration, line 3367 definition): add the three new parameters after the existing `d_ext_struct_flux_*` params.

- [ ] **Step 4: Update call sites in `swe2d_pipe1d_godunov_step_internal`**

At lines 3246 and 3310 (the two calls to `swe2d_gpu_apply_unified_face_flux`), pass `solver_dev->d_ext_drain_flux_h/hu/hv` as the new args. For the no-solver-dev path (lines 3254, 3318), pass `nullptr`.

- [ ] **Step 5: Add nullptr guard in kernel for new params**

At the class-3 guard block (line 2298), also check the drain flux pointers:
```cpp
if (!d_ext_drain_flux_h || !d_ext_drain_flux_hu || !d_ext_drain_flux_hv) return;
```

Same for class-4 (line 2470) and class-5 (line 2600).

- [ ] **Step 6: Build**

```bash
mamba run -n qgis_stable /usr/bin/cmake --build build -j$(nproc) 2>&1 | tail -10
```

---

### Task 3: Wire drain flux into the 2D update kernel

**Files:**
- Modify: `cpp/src/swe2d_gpu.cu` — `swe2d_update_kernel` call site + any helper kernel signature
- The `swe2d_update_kernel` itself reads `d_ext_struct_flux_*` as separate params. Add `d_ext_drain_flux_*` alongside them.

- [ ] **Step 1: Add drain flux params to `swe2d_update_kernel`**

Find the kernel declaration (around `swe2d_gpu.cu` line 2470s area, inside the `__global__` kernel). Add after the existing struct_flux params:

```cpp
const double* __restrict__ d_ext_drain_flux_h,   // [n_cells] drainage coupling mass flux
const State*  __restrict__ d_ext_drain_flux_hu,  // [n_cells] drainage coupling x-momentum
const State*  __restrict__ d_ext_drain_flux_hv,  // [n_cells] drainage coupling y-momentum
```

Inside the kernel body, sum both contributions wherever `ext_struct_flux_h[c]` is currently read:

```cpp
// Before (single term):
const double ext_h = ext_struct_flux_h[c];
// After (sum):
const double ext_h = ext_struct_flux_h[c] + d_ext_drain_flux_h[c];
```

Do the same for hu/hv.

- [ ] **Step 2: Update the kernel launch site**

At the `swe2d_update_kernel<<<...>>>` call (swe2d_gpu.cu ~line 5743), add the three new args after the existing struct_flux args.

- [ ] **Step 3: Update the host wrapper** (if `swe2d_update_kernel` has a host wrapper that forwards args)

- [ ] **Step 4: Update graph signature**

In `swe2d_kernel_graph_signature` (line 155), `d_ext_drain_flux_*` pointers are device pointers that remain valid — they don't change the hash unless the function takes them as template-like compile-time constants. If the kernel takes them as plain pointer params (not `__constant__`), the graph captures them by value and replays correctly. **No signature change needed** — the pointers are the same allocation across steps.

Verify this by checking if `d_ext_struct_flux_h` already appears in the graph signature. If not, `d_ext_drain_flux_h` won't either.

- [ ] **Step 5: Build**

```bash
mamba run -n qgis_stable /usr/bin/cmake --build build -j$(nproc) 2>&1 | tail -10
```

---

### Task 4: Fix structure coupling (resurrect dead path)

**Files:**
- Modify: `cpp/src/swe2d_gpu.cu` — new `swe2d_gpu_apply_structure_coupling` function
- Modify: `cpp/src/swe2d_bindings.cpp` — bind the new function
- Modify: `swe2d/runtime/coupling.py` — call it from `apply_native_device_sources`

- [ ] **Step 1: Extract target function**

In `swe2d_gpu.cu`, define a new function that extracts just the structure-relevant parts from `swe2d_gpu_compute_coupling_full_on_device`:

```cpp
void swe2d_gpu_apply_structure_coupling(
    SWE2DDeviceState* dev,
    int32_t n_cells,
    int32_t n_structures,
    bool use_culvert_face_flux,
    double dt,
    double gravity,
    double h_min)
{
    if (!dev) throw std::runtime_error("apply_structure_coupling: no GPU device state");
    if (n_cells <= 0) n_cells = dev->n_cells;
    if (n_structures <= 0 || !dev->sf_ws.params_preloaded) return;
    auto& sf_ws = dev->sf_ws;
    cudaStream_t stream = dev->d_stream;
    constexpr int BLOCK = 256;

    // Zero structure flux buffer (structures write here, drainage writes to its own)
    swe2d_gpu_alloc_ext_struct_flux(dev, n_cells);
    CUDA_CHECK(cudaMemsetAsync(dev->d_ext_struct_flux_h,  0, static_cast<size_t>(n_cells) * sizeof(double), stream));
    CUDA_CHECK(cudaMemsetAsync(dev->d_ext_struct_flux_hu, 0, static_cast<size_t>(n_cells) * sizeof(double), stream));
    CUDA_CHECK(cudaMemsetAsync(dev->d_ext_struct_flux_hv, 0, static_cast<size_t>(n_cells) * sizeof(double), stream));

    // Compute WSE from current device state
    if (sf_ws.cell_capacity >= n_cells && sf_ws.d_cell_wse) {
        int grid_wse = (n_cells + BLOCK - 1) / BLOCK;
        swe2d_coupling_wse_from_state_kernel<<<grid_wse, BLOCK, 0, stream>>>(
            n_cells, dev->d_h, dev->d_cell_zb, sf_ws.d_cell_wse);
    }

    // Save previous flows, zero current flows
    if (sf_ws.d_prev_structure_flow) {
        CUDA_CHECK(cudaMemcpyAsync(sf_ws.d_prev_structure_flow, sf_ws.d_structure_flow,
            static_cast<size_t>(n_structures) * sizeof(double), cudaMemcpyDeviceToDevice, stream));
    }
    CUDA_CHECK(cudaMemsetAsync(sf_ws.d_structure_flow, 0,
        static_cast<size_t>(n_structures) * sizeof(double), stream));

    // Compute structure flows
    int grid = (n_structures + BLOCK - 1) / BLOCK;
    swe2d_compute_structure_flows_kernel<<<grid, BLOCK, 0, stream>>>(
        n_cells, n_structures, sf_ws.d_cell_wse,
        sf_ws.d_structure_type, sf_ws.d_upstream_cell, sf_ws.d_downstream_cell,
        sf_ws.d_crest_elev, sf_ws.d_width, sf_ws.d_height,
        sf_ws.d_diameter, sf_ws.d_length, sf_ws.d_roughness_n,
        sf_ws.d_coeff, sf_ws.d_cd, sf_ws.d_opening,
        sf_ws.d_q_pump, sf_ws.d_max_flow,
        sf_ws.d_culvert_code, sf_ws.d_culvert_shape,
        sf_ws.d_culvert_rise, sf_ws.d_culvert_span, sf_ws.d_culvert_area,
        sf_ws.d_culvert_barrels, sf_ws.d_culvert_slope,
        sf_ws.d_inlet_invert_elev, sf_ws.d_outlet_invert_elev,
        sf_ws.d_entrance_loss_k, sf_ws.d_exit_loss_k,
        sf_ws.d_embankment_enabled, sf_ws.d_embankment_crest_elev,
        sf_ws.d_embankment_overflow_width, sf_ws.d_embankment_weir_coeff,
        sf_ws.gravity, sf_ws.model_to_ft, sf_ws.d_structure_flow,
        sf_ws.d_prev_structure_flow,
        s_culvert_solver_mode, s_culvert_table_header, s_culvert_table_data,
        s_culvert_table_n_hw, s_culvert_table_n_tw,
        sf_ws.d_culvert_diagnostics);

    // Face-flux kernel for culverts (writes to d_ext_struct_flux_*)
    if (use_culvert_face_flux
        && dev->culvert_ff_ws.params_preloaded
        && dev->culvert_ff_ws.n_culvert_faces > 0)
    {
        auto& ff = dev->culvert_ff_ws;
        int grid_ff = (ff.n_culvert_faces + BLOCK - 1) / BLOCK;
        swe2d_culvert_face_flux_kernel<<<grid_ff, BLOCK, 0, stream>>>(
            ff.n_culvert_faces,
            sf_ws.d_structure_flow,
            ff.d_culvert_struct_idx,
            ff.d_face_nx, ff.d_face_ny, ff.d_face_width,
            ff.d_donor_cell, ff.d_receiver_cell,
            ff.d_invert_elev, ff.d_depth_safety,
            ff.d_donor_cell_area,
            dev->d_h, dev->d_hu, dev->d_hv, dev->d_cell_zb,
            sf_ws.gravity, dt, h_min,
            n_cells,
            dev->d_ext_struct_flux_h,
            dev->d_ext_struct_flux_hu,
            dev->d_ext_struct_flux_hv);
    }

    // Non-culvert structures: fold into d_external_source_mps
    // (weirs, orifices, pumps use the source-term path since they
    // don't participate in face-flux coupling)
    int grid_src = (n_structures + BLOCK - 1) / BLOCK;
    swe2d_coupling_structure_source_kernel<<<grid_src, BLOCK, 0, stream>>>(
        n_structures, sf_ws.d_structure_type, sf_ws.d_upstream_cell,
        sf_ws.d_downstream_cell, sf_ws.d_structure_flow,
        dev->coupling_ws.d_cell_area, n_cells,
        dev->d_external_source_mps);
}
```

- [ ] **Step 2: Bind the new function**

In `swe2d_bindings.cpp`, add after the `swe2d_gpu_preload_coupling_cell_area` binding (~line 1150):

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
    py::arg("n_cells"), py::arg("n_structures"),
    py::arg("use_culvert_face_flux"),
    py::arg("dt"), py::arg("gravity"), py::arg("h_min"),
    "Compute structure flows and apply coupling flux (culverts → d_ext_struct_flux_*, "
    "weirs/orifices → d_external_source_mps). Drainage uses a separate buffer.");
```

- [ ] **Step 3: Wire into coupling.py**

In `apply_native_device_sources` (coupling.py ~line 1800), after the pipe1d step finishes and after culvert face-flux preload, add:

```python
# ── Structures: compute flows and write to d_ext_struct_flux_* ──
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

Remove the old retried-comment block at lines 1821-1829 and the stale fold call at lines 1848-1855.

- [ ] **Step 4: Build + test**

```bash
mamba run -n qgis_stable /usr/bin/cmake --build build -j$(nproc)
mamba run -n qgis_stable env PYTHONPATH="$PWD:$PWD/build" \
    python3 -m unittest -v tests.test_pipe1d_face_indexed_mesh 2>&1 | tail -10
```

---

### Task 5: Remove the retired `compute_coupling_full_on_device` binding

**Files:**
- Modify: `cpp/src/swe2d_bindings.cpp`

- [ ] **Step 1: Delete the binding**

Remove lines 1153-1176 (the `m.def("swe2d_gpu_compute_coupling_full_on_device", ...)` block).

Keep the C++ function body in `swe2d_gpu.cu` for now (it may be useful for tests). Just remove the Python-facing binding.

- [ ] **Step 2: Clean up `swe2d_gpu_upload_outfall_free_bc_nodes`**

Either delete the Python call in coupling.py line 1751-1754, or add a deprecation warning to the C++ no-op body. The function does nothing since the refactor.

- [ ] **Step 3: Build**

```bash
mamba run -n qgis_stable /usr/bin/cmake --build build -j$(nproc) 2>&1 | tail -5
```

---

### Task 6: Port `TestPipeEndExchange` to cover the drain-flux path

**Files:**
- Modify: `tests/test_swe2d_gpu_drainage_network.py`

- [ ] **Step 1: Remove the dead `hasattr` gate**

Line 908: change from `@unittest.skipUnless(hasattr(_MOD, "swe2d_gpu_apply_pipe_end_bc")` to no gate (or gate on `swe2d_build_unified_mesh`).

- [ ] **Step 2: Port test to use `swe2d_pipe1d_step` with solver_dev**

The test should:
1. Build unified mesh
2. Upload pipe-end surface faces
3. Init cell area
4. Run `swe2d_pipe1d_step` with solver_dev pointer
5. Read back `d_ext_drain_flux_*` via `swe2d_gpu_readback_cpl_flux` (or read cell snapshots)

- [ ] **Step 3: Add mass conservation assertion**

```python
pipe_vol_before = np.sum(pipe_A_before * cell_length)
surf_vol_before = np.sum(surf_h_before * cell_area)
# run step
pipe_vol_after = np.sum(pipe_A_after * cell_length)
surf_vol_after = np.sum(surf_h_after * cell_area)
self.assertAlmostEqual(pipe_vol_before + surf_vol_before,
                       pipe_vol_after + surf_vol_after, delta=1e-6)
```

- [ ] **Step 4: Run**

```bash
mamba run -n qgis_stable env PYTHONPATH="$PWD:$PWD/build" \
    python3 -m unittest -v \
    tests.test_swe2d_gpu_drainage_network.TestPipeEndExchange 2>&1 | tail -20
```
Expected: PASS (no longer skipped).

---

### Task 7: Clean up the old `compute_coupling_full_on_device` C++ body

**Files:**
- Modify: `cpp/src/swe2d_gpu.cu`

- [ ] **Step 1: Mark as deprecated**

At the top of `swe2d_gpu_compute_coupling_full_on_device` function body (line 7602), add:

```cpp
// DEPRECATED: structure coupling now goes through
// swe2d_gpu_apply_structure_coupling (called from Python).
// Drainage coupling uses its own d_ext_drain_flux_* buffer.
// This function is retained for test compatibility only.
```

No functional change. The function can be deleted entirely once all tests are verified to use the new path.

- [ ] **Step 2: Build + run all drainage/structure tests**

```bash
find . -type d -name __pycache__ -exec rm -rf {} +
mamba run -n qgis_stable env PYTHONPATH="$PWD:$PWD/build" \
    python3 -m unittest -v \
    tests.test_swe2d_gpu_drainage_network \
    tests.test_swe2d_gpu_coupling_kernel \
    tests.test_swe2d_gpu_full_solver_structures \
    tests.test_pipe1d_face_indexed_mesh \
    tests.test_coupling_setup \
    2>&1 | tail -30
```
