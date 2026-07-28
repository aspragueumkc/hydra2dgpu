---
type: plan
status: complete
created: 2026-07-15
completed: 2026-07-25
---

# Apply SWMM-style minor loss to diffusion_wave pipe1d kernel

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the `diffusion_wave` pipe1d solver kernel so that minor (entrance/exit) losses are treated in the same SWMM-style momentum-sink form already used by the `fully_dynamic` kernel.

**Architecture:** The fully-dynamic kernel already computes end areas from node depths and applies `k_in/(2*A1*L_link) + k_out/(2*A2*L_link)` as an implicit loss term in the momentum equation. The diffusion-wave kernel previously relied on a node-level HEC-22 loss reduction. This refactor moves the loss into the diffusion-wave momentum equation, matching SWMM and removing the double-count. The diffusion-wave head gradient is kept per sub-cell (`L`) to preserve the existing convergence behavior and avoid regressions in open-channel tests; only the loss term uses the total link length (`L_link`).

**Tech Stack:** CUDA C++, SWMM reference source (`reference/Stormwater-Management-Model-develop/src/solver/dwflow.c`), project test suite (`pytest`).

---

### Files touched
- `cpp/src/pipe1d.cuh` — update `swe2d_pipe1d_diffusion_wave_kernel_host` signature.
- `cpp/src/pipe1d.cu` — update `swe2d_pipe1d_diffusion_wave_kernel`, its host wrapper, and the call site in `swe2d_pipe1d_advance`.
- `docs/AGENT_SESSION_RECOVERY_LOG.md` — record the change.

### Task 1: Update device kernel signature and implementation

**Files:**
- Modify: `cpp/src/pipe1d.cu:663-733` (kernel definition)

- [ ] **Step 1.1: Add SWMM-style inputs to the kernel signature**

Add after existing `cell_k_loss`:
```c
    const double*  __restrict__ cell_link_length,
    const double*  __restrict__ cell_k_loss_in,
    const double*  __restrict__ cell_k_loss_out,
    const int32_t* __restrict__ cell_shape_type,
    const double*  __restrict__ cell_width,
    const double*  __restrict__ cell_height,
```

- [ ] **Step 1.2: Keep the existing per-cell head gradient**

The diffusion-wave head gradient is intentionally left per sub-cell (`L`) to preserve existing convergence.  Read `L_link` only for the loss denominator:

```c
    const double L_link = cell_link_length[i];
```

Leave:
```c
    const double dHdx = (H_to - H_from) / fmax(1e-6, L);
```
unaltered.

- [ ] **Step 1.3: Compute end areas from node depths**

After the `dHdx` line, add:
```c
    const double depth_from = (fn >= 0) ? node_depth[fn] : 0.0;
    const double depth_to   = (tn >= 0) ? node_depth[tn] : 0.0;
    const double A1 = pipe1d_area_from_depth(
        cell_shape_type[i], cell_width[i], cell_height[i], A_full, depth_from);
    const double A2 = pipe1d_area_from_depth(
        cell_shape_type[i], cell_width[i], cell_height[i], A_full, depth_to);
```

- [ ] **Step 1.4: Replace old minor-loss denominator with SWMM local loss**

Change the comment and denominator block from:
```c
    // Implicit Manning friction (minor/expansion losses handled explicitly in
    // swe2d_pipe1d_accumulate_node_flux_kernel; do NOT double-apply here).
    const double cf = g * n * n / (A * R43 + 1e-12);
    const double denom = 1.0 + dt * cf * absQ;
```
to:
```c
    // SWMM-style local (minor) loss term in the momentum equation. Entrance
    // loss uses the upstream end area, exit loss uses the downstream end area,
    // both divided by the total link length.
    const double cf = g * n * n / (A * R43 + 1e-12);
    const double cm = 0.0
        + (k_in  > 0.0 && A1 > 0.0 ? k_in  / (2.0 * A1 * L_link) : 0.0)
        + (k_out > 0.0 && A2 > 0.0 ? k_out / (2.0 * A2 * L_link) : 0.0);
    const double denom = 1.0 + dt * cf * absQ + dt * cm * absQ;
```

where `k_in` and `k_out` are read from `cell_k_loss_in[i]` and `cell_k_loss_out[i]`.

---

### Task 2: Update the host wrapper signature and call

**Files:**
- Modify: `cpp/src/pipe1d.cu:945-973` (host wrapper)

- [ ] **Step 2.1: Add the same parameters to `swe2d_pipe1d_diffusion_wave_kernel_host`**

```c
void swe2d_pipe1d_diffusion_wave_kernel_host(
    int32_t               n_cells,
    const double*         cell_length,
    const double*         cell_link_length,
    const double*         cell_area_full,
    const double*         cell_perim,
    const double*         cell_n,
    const double*         cell_k_loss,
    const double*         cell_k_loss_in,
    const double*         cell_k_loss_out,
    const int32_t*        cell_shape_type,
    const double*         cell_width,
    const double*         cell_height,
    const int32_t*        cell_from_node,
    const int32_t*        cell_to_node,
    const double*         node_invert,
    const double*         node_depth,
    const double*         cell_A,
    const double*         cell_Q,
    const double*         flux_Q,
    double                dt,
    double                g,
    double*               cell_A_new,
    double*               cell_Q_new,
    const double*         cell_tables,
    int32_t               table_N)
```

- [ ] **Step 2.2: Pass the new arguments in the kernel launch**

Update the `<<< >>>` call to match the new kernel signature.

---

### Task 3: Update the header declaration

**Files:**
- Modify: `cpp/src/pipe1d.cuh:199-218`

- [ ] **Step 3.1: Mirror the new host wrapper signature in the header**

Add the same new parameters and update the doxygen comment to note that losses are now applied in the momentum equation.

---

### Task 4: Update the call site in `swe2d_pipe1d_advance`

**Files:**
- Modify: `cpp/src/pipe1d.cu:1339-1349`

- [ ] **Step 4.1: Pass the device arrays that already exist in `Pipe1DDeviceState`**

The `p` struct already has `d_cell_link_length`, `d_cell_link_k_in`, `d_cell_link_k_out`, `d_cell_shape_type`, `d_cell_width`, `d_cell_height`. Pass them to the wrapper.

---

### Task 5: Build and verify

- [ ] **Step 5.1: Rebuild the native extension**

```bash
find . -type d -name __pycache__ -exec rm -rf {} +
cd build
cmake --build . -j$(nproc)
```

- [ ] **Step 5.2: Run the targeted test suites**

```bash
mamba run -n qgis_stable python3 -m unittest -v \
    tests.test_swe2d_pipe1d \
    tests.test_swe2d_pipe1d_surcharge \
    tests.test_swe2d_gpu_drainage_network \
    tests.test_pipe1d_accumulation \
    tests.test_pipe_cell_coupling_output
```

Expected: all pass.

- [ ] **Step 5.3: Check the diffusion-wave vs-SWMM tests**

```bash
mamba run -n qgis_stable python3 -m unittest -v \
    tests.test_pipe1d_vs_swmm
```

Expected: the 3 pre-existing `diffusion_wave` failures may still be present (Q ≈ 0 for pressurized/sloped cases). `TestOpenChannel.test_half_pipe_reasonable` should now pass. Report the new counts and do not leave unreported changes.

---

### Task 6: Update session recovery log

- [ ] **Step 6.1: Append a short note to `docs/AGENT_SESSION_RECOVERY_LOG.md`**

Document that the SWMM-style local-loss treatment was extended to the `diffusion_wave` kernel, including the parameter additions and the removal of the old node-level loss double-count.
