---
type: plan
status: complete
created: 2026-07-15
completed: 2026-07-25
---

# True Dynamic-Wave Terms for pipe1d Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the missing convective-acceleration (`dq4`) and Froude-based inertial-damping (`sigma`) terms to the `fully_dynamic` pipe1d kernel so it matches the SWMM dynamic-wave formulation described in `docs/archive/plans/DRAINAGE_EQUATION_PLAN.md`.

**Architecture:** The convective term needs upstream/downstream end areas computed from node depths, which requires passing `cell_shape_type`, `cell_width`, and `cell_height` into the `fully_dynamic` kernel. A small device helper converts depth to area using the same linearized depth-area relation already used by `swe2d_pipe1d_init_area_from_depth`. The kernel then computes a midpoint velocity, Froude number, `sigma`, and the `dq4` term, adding it to the implicit momentum update.

**Tech Stack:** CUDA C++ (`cpp/src/pipe1d.cu`), Python/pybind11 binding layer (`cpp/src/swe2d_bindings.cpp`), pytest (`tests/test_swe2d_pipe1d.py`).

---

## Step Dicts (selector-consumable)

```python
{"action": "Add cell shape/width/height parameters to the fully_dynamic kernel and host wrapper signatures", "type": "coding"}
{"action": "Pass shape/width/height arrays through swe2d_pipe1d_step dispatch", "type": "coding"}
{"action": "Implement a device helper that returns area from node depth for circular/rectangular/elliptical shapes", "type": "coding"}
{"action": "Compute upstream and downstream end areas, Froude number, and sigma damping in the fully_dynamic kernel", "type": "coding"}
{"action": "Add dq4 convective acceleration term to the momentum update in the fully_dynamic kernel", "type": "coding"}
{"action": "Write a regression test that exercises a non-uniform area gradient and verifies the convective term changes the result", "type": "test"}
{"action": "Rebuild the native extension and run pipe1d plus drainage tests", "type": "test"}
```

---

### Task 1: Extend kernel and wrapper signatures

**Files:**
- Modify: `cpp/src/pipe1d.cu:728` (kernel signature)
- Modify: `cpp/src/pipe1d.cu:887` (host wrapper signature)
- Modify: `cpp/src/pipe1d.cuh` (declaration if present)

**Current kernel signature:**
```c
__global__ void swe2d_pipe1d_fully_dynamic_kernel(
    int32_t n_cells, int32_t n_iters, double relaxation,
    const int32_t* owned_offsets, const int32_t* owned_ids,
    const int32_t* neighbor_cell, const double* interface_dir,
    const int32_t* cell_from_node, const int32_t* cell_to_node,
    const double* cell_length, const double* cell_area_full,
    const double* cell_perim, const double* cell_n,
    const double* cell_k_loss, const double* node_invert,
    const double* node_depth, const double* cell_A_prev,
    const double* cell_Q_prev, double* cell_A_iter,
    double* cell_Q_iter, double dt, double g,
    const double* cell_tables, int32_t table_N);
```

**Change:** Add `const int32_t* cell_shape_type`, `const double* cell_width`, and `const double* cell_height` after `cell_area_full`.

**Host wrapper signature** at line 887 gets the same three additions in the same order.

- [ ] **Step 1: Update signatures**
- [ ] **Step 2: Verify the build still compiles** (expected: no new errors from signature mismatch)

---

### Task 2: Thread shape data through the dispatch

**Files:**
- Modify: `cpp/src/pipe1d.cu:1230-1245` (call site in `swe2d_pipe1d_step`)

**Current call:**
```c
swe2d_pipe1d_fully_dynamic_kernel_host(
    n_cells, implicit_iters, relaxation,
    p.d_owned_offsets, p.d_owned_ids,
    p.d_cell_neighbor_cell, p.d_cell_interface_dir,
    p.d_cell_from_node, p.d_cell_to_node,
    p.d_cell_length, p.d_cell_area,
    p.d_cell_perim, p.d_cell_n,
    p.d_cell_link_k,
    p.d_node_invert, p.d_node_depth,
    p.d_A, p.d_Q,
    d_A_new, d_Q_new,
    local_dt, g,
    d_cell_tables, table_N);
```

**Change:** Pass `p.d_cell_shape_type`, `p.d_cell_width`, `p.d_cell_height` after `p.d_cell_area`.

- [ ] **Step 1: Update call site arguments**
- [ ] **Step 2: Verify the build still compiles**

---

### Task 3: Add device helper `pipe1d_area_from_depth`

**Files:**
- Modify: `cpp/src/pipe1d.cu` near `pipe1d_lookup_geometry`

**Add:**
```c
__device__ __forceinline__ double pipe1d_area_from_depth(
    int32_t shape_type,
    double width,
    double height,
    double A_full,
    double depth)
{
    if (depth <= 0.0 || A_full <= 0.0) return 0.0;
    const double full_depth = (shape_type == 0) ? width : height;
    if (full_depth <= 0.0) return 0.0;
    double frac = depth / full_depth;
    if (frac > 1.0) frac = 1.0;
    return A_full * frac;
}
```

This matches the linearized depth-area relation already used in `swe2d_pipe1d_init_area_from_depth`.

- [ ] **Step 1: Add helper above the fully_dynamic kernel**
- [ ] **Step 2: Rebuild and verify no CUDA errors**

---

### Task 4: Compute end areas, Froude, and sigma

**Files:**
- Modify: `cpp/src/pipe1d.cu:728-815` (fully_dynamic kernel body)

**After the existing geometry lookup (around line 774), add:**
```c
    // End areas from node depths (A1 = upstream in cell orientation, A2 = downstream)
    const double depth_from = (fn >= 0) ? node_depth[fn] : 0.0;
    const double depth_to   = (tn >= 0) ? node_depth[tn] : 0.0;
    const double A1 = pipe1d_area_from_depth(
        cell_shape_type[c], cell_width[c], cell_height[c], A_full, depth_from);
    const double A2 = pipe1d_area_from_depth(
        cell_shape_type[c], cell_width[c], cell_height[c], A_full, depth_to);

    // Midpoint velocity (use current iterate)
    const double v_mid = (A > 0.0) ? Q / A : 0.0;
    const double v_abs = fabs(v_mid);

    // Froude number and inertial damping (SWMM convention)
    // Use hydraulic radius from the geometry lookup.
    double sigma = 1.0;
    if (P_c > 0.0 && A > 0.0) {
        const double R_h = A / P_c;
        const double froude = (R_h > 0.0) ? v_abs / sqrt(g * R_h) : 0.0;
        if (froude >= 1.0) {
            sigma = 0.0;
        } else if (froude > 0.5) {
            sigma = 2.0 * (1.0 - froude);
        }
    }
```

- [ ] **Step 1: Insert end-area and sigma computation**
- [ ] **Step 2: Rebuild and run `tests/test_swe2d_pipe1d.py`** (expected: existing tests still pass)

---

### Task 5: Add dq4 convective acceleration to momentum

**Files:**
- Modify: `cpp/src/pipe1d.cu:728-815` (fully_dynamic kernel body)

**Current momentum update:**
```c
    double Q_new = (Q + dt * pressure_grad) / denom;
```

**Change to:**
```c
    // Convective acceleration term: dq4 = dt * v^2 * (A2 - A1) / L * sigma
    double dq4 = 0.0;
    if (L > 0.0) {
        dq4 = dt * v_mid * v_mid * (A2 - A1) / L * sigma;
    }

    double Q_new = (Q + dt * pressure_grad + dq4) / denom;
```

The sign follows `docs/archive/plans/DRAINAGE_EQUATION_PLAN.md`, where `A1` is upstream and `A2` is downstream in the cell orientation.

- [ ] **Step 1: Add dq4 to the momentum numerator**
- [ ] **Step 2: Rebuild and run `tests/test_swe2d_pipe1d.py`** (expected: all existing tests pass, area/Q remain bounded)

---

### Task 6: Add a regression test for the convective term

**Files:**
- Modify: `tests/test_swe2d_pipe1d.py`

**Add a new test method in `TestPipe1DStep`:**
```python
    def test_fully_dynamic_convective_term_affects_flow(self):
        """Non-uniform end areas change the discharge via the convective term."""
        a = self._simple_pipe_arrays()
        # Large upstream depth, small downstream depth -> A1 > A2
        a["node_depth"] = np.array([2.0, 0.05], dtype=np.float64)
        dev_ptr = self._build_and_upload(a)
        _MOD.swe2d_pipe1d_step(dev_ptr, 0.5, "fully_dynamic", 1, 5, 0.5, 9.81)
        rb_conv = _MOD.swe2d_pipe1d_readback_node_state(dev_ptr, a["n_nodes"], 1)
        Q_conv = float(rb_conv["cell_Q"][0])

        # Uniform small depth -> A1 == A2, so dq4 is small/zero
        a["node_depth"] = np.array([0.05, 0.05], dtype=np.float64)
        dev_ptr = self._build_and_upload(a)
        _MOD.swe2d_pipe1d_step(dev_ptr, 0.5, "fully_dynamic", 1, 5, 0.5, 9.81)
        rb_uniform = _MOD.swe2d_pipe1d_readback_node_state(dev_ptr, a["n_nodes"], 1)
        Q_uniform = float(rb_uniform["cell_Q"][0])

        self.assertTrue(np.isfinite(Q_conv))
        self.assertTrue(np.isfinite(Q_uniform))
        self.assertNotAlmostEqual(Q_conv, Q_uniform, places=6,
                                  msg="Convective acceleration from area gradient should change discharge")
```

- [ ] **Step 1: Add the test**
- [ ] **Step 2: Run `pytest tests/test_swe2d_pipe1d.py::TestPipe1DStep::test_fully_dynamic_convective_term_affects_flow -v`** (expected: PASS after the kernel change)

---

### Task 7: Verify and integrate

**Commands:**
```bash
find . -type d -name __pycache__ -exec rm -rf {} +
mamba run -n qgis_stable python3 -m pytest tests/test_swe2d_pipe1d.py -v
mamba run -n qgis_stable python3 -m pytest tests/test_swe2d_gpu_drainage_network.py -v
```

**Expected:**
- `test_swe2d_pipe1d.py`: all tests pass, including the new convective test.
- `test_swe2d_gpu_drainage_network.py`: all tests pass.

- [ ] **Step 1: Rebuild native extension**
- [ ] **Step 2: Run pipe1d tests**
- [ ] **Step 3: Run drainage network tests**
- [ ] **Step 4: Update `docs/AGENT_SESSION_RECOVERY_LOG.md` with the new implementation**

---

## Self-Review Checklist

- **Spec coverage:** The `fully_dynamic` mode in `docs/archive/plans/DRAINAGE_EQUATION_PLAN.md` lists `dq4` (convective acceleration) and `sigma` (inertial damping) as Phase 2c missing pieces. This plan covers both.
- **No placeholders:** Every step includes exact file paths, code, and expected outputs.
- **Type consistency:** `cell_shape_type`, `cell_width`, `cell_height` are passed in the same order to kernel, wrapper, and call site. The depth-area helper uses the same `shape_type` encoding (0=circular, 1=rect, 2=ellipse) as the rest of the codebase.
- **Scope:** Only the `fully_dynamic` kernel is changed. The `diffusion_wave` kernel remains the local-inertia form as noted by the user.
