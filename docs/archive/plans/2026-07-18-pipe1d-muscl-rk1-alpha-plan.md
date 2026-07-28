---
type: plan
status: complete
created: 2026-07-18
completed: 2026-07-25
---

# Pipe1d MUSCL-minmod + RK1 + Runtime Alpha Boost — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans.

**Goal:** Add MUSCL-minmod slope-limited reconstruction (second-order), RK1 (Forward Euler) time integrator, and runtime-selectable friction alpha boost to the pipe1d Godunov solver, with full widget plumbing.

**Architecture:** Three new C++ kernel params (`recon_method`, `time_integrator`, `friction_alpha`) plumbed via `PipeNetworkConfig` → `SWE2DDrainageSoA` → pybind11 → C++, following the existing `friction_method`/`surcharge_method` pattern. A new `d_slope_H` GPU array stores minmod-limited gradients computed by a dedicated kernel before each flux evaluation.

**Tech Stack:** CUDA C++ (new slope kernel, modified flux/godunov kernels), PyQt5 (widgets), Python dataclasses, pybind11.

---

## Task Outline

| # | Task | Layer | Files |
|---|------|-------|-------|
| 1 | `d_slope_H` device array + lifecycle | C++ | pipe1d.cuh, pipe1d.cu |
| 2 | Slopes kernel + MUSCL in flux kernel | C++ | pipe1d.cu |
| 3 | RK1 path + runtime friction_alpha in godunov kernel | C++ | pipe1d.cu |
| 4 | Update host wrappers + step function | C++ | pipe1d.cu |
| 5 | Update function signatures | C++ | pipe1d.cuh |
| 6 | Remove constexpr FRICTION_STABILITY_ALPHA | C++ | swe2d_xsect_constants.h |
| 7 | Update pybind11 binding | C++ | swe2d_bindings.cpp |
| 8 | PipeNetworkConfig dataclass fields | Python | extension_models.py |
| 9 | SWE2DDrainageSoA fields + coupling call | Python | coupling.py |
| 10 | Pipe config services | Python | pipe_network_config_service.py, pipe_network_service.py |
| 11 | Widgets + getters | Python | model_tab_view.py |
| 12 | Controller wiring | Python | studio_dialog.py |
| 13 | Build + cache cleanup | Build | |

---

### Task 1: Add `d_slope_H` to `Pipe1DDeviceState` + lifecycle

**Files:**
- Modify: `cpp/src/pipe1d.cuh:55-57` (add pointer), `:199` (free in destroy)
- Modify: `cpp/src/pipe1d.cu:3045-3054` (allocate in step), `:3251-3254` (free in step)

- [ ] **Step 1: Add `d_slope_H` pointer to struct + destroy free**

In `pipe1d.cuh`, after `d_vnode_Q` (line 54), add:
```cpp
double*   d_slope_H;         // [n_pipe_cells] minmod-limited WSE gradient (∂H/∂x) for MUSCL reconstruction
```

In the `destroy()` function, line 195, add `_P_FREE(d_slope_H);` alongside `d_vnode_H`/`d_vnode_Q`.

- [ ] **Step 2: Allocate + free `d_slope_H` in `swe2d_pipe1d_step`**

In `pipe1d.cu`, in `swe2d_pipe1d_step`, after the existing cudaMalloc for d_flux_Q (line 3047), add:
```cpp
if (p.d_slope_H) { cudaFree(p.d_slope_H); p.d_slope_H = nullptr; }
CUDA_CHECK(cudaMalloc(&p.d_slope_H, static_cast<size_t>(n_cells) * sizeof(double)));
```

After the existing `cudaFree(d_flux_Q)` (line 3251), add:
```cpp
if (p.d_slope_H) { cudaFree(p.d_slope_H); p.d_slope_H = nullptr; }
```

---

### Task 2: Write slopes kernel + modify flux kernel for MUSCL

**Files:**
- Modify: `cpp/src/pipe1d.cu` — add new kernel after the existing flux kernel (after line 1800), modify flux kernel at `~1348-1356`

- [ ] **Step 1: Write `swe2d_pipe1d_compute_slopes_kernel`**

Add immediately before the godunov kernel (before line 1802):

```cpp
__global__ __launch_bounds__(256, 1) void swe2d_pipe1d_compute_slopes_kernel(
    int32_t                     n_cells,
    const double*  __restrict__ cell_H,
    const double*  __restrict__ cell_length,
    const int32_t* __restrict__ cell_owner_link,
    const int32_t* __restrict__ cell_sub_idx,
    double*                     d_slope_H)
{
    int32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_cells) { d_slope_H[c] = 0.0; return; }

    const int32_t link = cell_owner_link[c];
    const int32_t sub  = cell_sub_idx[c];
    const double  L_c  = fmax(cell_length[c], 1.0e-12);

    // Boundary cells: zero gradient (first or last sub-cell in link)
    double slope = 0.0;

    // Interior cell: has both left and right neighbors within same link
    // We search for adjacent cells — they share the same link_id and sub_idx differs by 1.
    // Since cells are stored contiguously per link and sub_idx is contiguous,
    // c-1 and c+1 are the neighbors IF they share the same link.
    // However, to be robust we check cell_owner_link.
    if (c > 0 && c < n_cells - 1) {
        if (cell_owner_link[c-1] == link && cell_owner_link[c+1] == link) {
            const double H_c   = cell_H[c];
            const double H_prev = cell_H[c-1];
            const double H_next = cell_H[c+1];
            const double L_left  = 0.5 * (cell_length[c-1] + L_c);
            const double L_right = 0.5 * (L_c + cell_length[c+1]);
            const double dH_left  = (H_c - H_prev) / fmax(L_left, 1.0e-12);
            const double dH_right = (H_next - H_c) / fmax(L_right, 1.0e-12);
            // minmod: zero if slopes have opposite sign, else choose min magnitude
            if (dH_left * dH_right > 0.0) {
                slope = (fabs(dH_left) < fabs(dH_right)) ? dH_left : dH_right;
            } // else slope stays 0.0
        }
    }
    d_slope_H[c] = slope;
}

// Host wrapper:
void swe2d_pipe1d_compute_slopes_kernel_host(
    int32_t               n_cells,
    const double*         cell_H,
    const double*         cell_length,
    const int32_t*        cell_owner_link,
    const int32_t*        cell_sub_idx,
    double*               d_slope_H)
{
    const int32_t n_blocks = (n_cells + 255) / 256;
    swe2d_pipe1d_compute_slopes_kernel<<<n_blocks, 256>>>(
        n_cells, cell_H, cell_length, cell_owner_link, cell_sub_idx, d_slope_H);
    CUDA_CHECK(cudaGetLastError());
}
```

- [ ] **Step 2: Modify flux kernel to accept + use slopes for MUSCL reconstruction**

Add two new trailing args to the flux kernel signature:
```cpp
    const double*  __restrict__ d_slope_H,
    int32_t                     recon_method)
```

At the upwind H write (line 1348-1356), replace:
```cpp
if (dir > 0.0 && vnode_idx && vnode_idx[k] >= 0) {
    const int32_t v = vnode_idx[k];
    const double H_v = (cell_Q[c] > 0.0) ? H_c : H_n;
    if (vnode_H) vnode_H[v] = H_v;
    if (vnode_Q) vnode_Q[v] = F;
}
```

With:
```cpp
if (dir > 0.0 && vnode_idx && vnode_idx[k] >= 0) {
    const int32_t v = vnode_idx[k];
    double H_v;
    if (recon_method == 1 && d_slope_H != nullptr) {
        // MUSCL-minmod reconstruction at the face
        const double L_c = fmax(cell_length[c], 1.0e-12);
        const double L_n = fmax(cell_length[nbr], 1.0e-12);
        const double H_L = H_c + 0.5 * L_c * d_slope_H[c];
        const double H_R = H_n - 0.5 * L_n * d_slope_H[nbr];
        // Upwind-biased choice: if Q_c > 0, use H_L; else use H_R
        H_v = (cell_Q[c] > 0.0) ? H_L : H_R;
    } else {
        // First-order upwind (original behavior)
        H_v = (cell_Q[c] > 0.0) ? H_c : H_n;
    }
    if (vnode_H) vnode_H[v] = H_v;
    if (vnode_Q) vnode_Q[v] = F;
}
```

- [ ] **Step 3: Update flux kernel host wrapper signature**

Add `d_slope_H` and `recon_method` params to `swe2d_pipe1d_flux_kernel_host`, pass through to the kernel launch.

---

### Task 3: Modify godunov update kernel for RK1 + runtime friction_alpha

**Files:**
- Modify: `cpp/src/pipe1d.cu` — godunov kernel at `~1943` (FRICTION_STABILITY_ALPHA) and `~2003-2010` (stage logic)

- [ ] **Step 1: Replace FRICTION_STABILITY_ALPHA with runtime friction_alpha**

Add `double friction_alpha` as a new trailing kernel arg. Replace line 1943:
```cpp
const double gamma_stable = friction_alpha * absQ / fmax(A_full, 1.0e-12);
```

- [ ] **Step 2: Add RK1 single-stage path**

Replace the stage logic at lines 2003-2010:
```cpp
double A_next;
if (time_integrator == 0) {
    // RK1 (Forward Euler): single stage, directly write result
    A_next = A_curr - dt * flux_A / L;
} else if (stage == 0) {
    // RK2 stage 0: save start-of-step state, compute intermediate update
    A_start_save[c] = A_curr;
    Q_start_save[c] = Q_curr;
    A_next = A_curr - dt * flux_A / L;
} else {
    // RK2 stage 1: average with saved start-of-step state
    const double A_mid = A_curr;
    A_next = 0.5 * (A_start_save[c] + A_mid - dt * flux_A / L);
}
```

And similarly for the Q update — when `time_integrator == 0`, skip the RK2 averaging. The Q update already uses the `(Q_curr + dt*explicit_force) / (1+gamma*dt)` pattern which is correct for RK1.

Add `int32_t time_integrator` and `double friction_alpha` as new trailing kernel args. 

- [ ] **Step 3: Launch slopes kernel before flux kernel when recon_method==1**

Modified: When `recon_method == 1`, launch `swe2d_pipe1d_compute_slopes_kernel_host` with `cell_y` (or H reconstructed from A/Q) before each flux kernel call.

---

### Task 4: Update host wrappers + step function

**Files:**
- Modify: `cpp/src/pipe1d.cu` — `swe2d_pipe1d_godunov_step_internal` (lines 2066-2222) and `swe2d_pipe1d_step` (lines 3017-)

- [ ] **Step 1: Update `swe2d_pipe1d_godunov_step_internal` signature + body**

Add `recon_method`, `time_integrator`, `friction_alpha` params. Before each flux kernel call, when `recon_method == 1`, launch the slopes kernel. Pass new params to both flux and godunov kernel calls.

For RK1 (`time_integrator == 0`):
- Run stage 0 flux kernel (with p.d_A/p.d_Q)
- Run stage 0 godunov update kernel
- Copy d_A_new→p.d_A, d_Q_new→p.d_Q
- Skip stage 1 entirely
- The node_net_q accumulator has already been written at full weight by the single flux call — the caller's scale-by-0.5 for RK2 averaging needs to be conditional.

- [ ] **Step 2: Update `swe2d_pipe1d_step` for RK1 node_net_q scaling**

When `time_integrator == 0` (RK1), the node_net_q scaling should be 1.0 (not 0.5), because there's only one flux evaluation per substep. Replace the hardcoded 0.5 at line 3096 with:
```cpp
const double scale = (time_integrator == 0) ? 1.0 : 0.5;
pipe1d_scale_double_kernel<<<n_grid, 256, 0, dev->d_stream>>>(
    p.n_nodes + p.n_vnodes, p.d_node_net_q, scale);
```

Pass new params through to `swe2d_pipe1d_godunov_step_internal`.

---

### Task 5: Update function signatures in pipe1d.cuh

**Files:**
- Modify: `cpp/src/pipe1d.cuh:570-583`

Replace `swe2d_pipe1d_step` declaration with:
```cpp
void swe2d_pipe1d_step(
    SWE2DDeviceState* dev,
    double            dt,
    const char*       solver_mode,
    int32_t           coupling_substeps,
    int32_t           implicit_iters,
    double            relaxation,
    double            g,
    double            k_mann,
    double            h_min,
    int32_t           surcharge_method,
    double            theta       = 1.0,
    double            omega_min   = 1e-6,
    int32_t           friction_method = 0,
    int32_t           recon_method    = 0,
    int32_t           time_integrator = 1,
    double            friction_alpha  = 0.01);
```

Add declaration for the slopes kernel host wrapper.

---

### Task 6: Remove FRICTION_STABILITY_ALPHA constexpr

**Files:**
- Modify: `cpp/src/swe2d_xsect_constants.h:29`

Remove the line:
```cpp
constexpr double FRICTION_STABILITY_ALPHA = 0.01;
```

---

### Task 7: Update pybind11 binding

**Files:**
- Modify: `cpp/src/swe2d_bindings.cpp:1905-1949`

Add three new defaulted params to the lambda:
```cpp
int32_t recon_method = 0,
int32_t time_integrator = 1,
double friction_alpha = 0.01
```

Pass to `swe2d_pipe1d_step`. Add `py::arg(...)` entries for each:
```cpp
py::arg("recon_method") = 0,
py::arg("time_integrator") = 1,
py::arg("friction_alpha") = 0.01,
```

Update docstring.

---

### Task 8: PipeNetworkConfig dataclass fields

**Files:**
- Modify: `swe2d/extensions/extension_models.py:284-326`

Add after `friction_method`:
```python
recon_method: int = 0        # 0=FIRST_ORDER, 1=MUSCL_MINMOD
time_integrator: int = 1     # 0=RK1, 1=RK2
friction_alpha: float = 0.01  # alpha boost (additive linear-Q damping)
```

---

### Task 9: SWE2DDrainageSoA fields + coupling call

**Files:**
- Modify: `swe2d/runtime/coupling.py` — SWE2DDrainageSoA fields (~line 250), `_apply_coupling` call (~line 1742)

Add fields to `SWE2DDrainageSoA`:
```python
recon_method: int = 0
time_integrator: int = 1
friction_alpha: float = 0.01
```

In the `_apply_coupling` call to `swe2d_pipe1d_step` (line 1742), add after `friction_method`:
```python
recon_method=int(dsoa.recon_method),
time_integrator=int(dsoa.time_integrator),
friction_alpha=float(dsoa.friction_alpha),
```

---

### Task 10: Pipe config services

**Files:**
- Modify: `swe2d/workbench/services/pipe_network_config_service.py` (~line 70)
- Modify: `swe2d/workbench/services/pipe_network_service.py` (~line 1133)

In `pipe_network_config_service.py`, add to config dict:
```python
"recon_method": recon_method,
"time_integrator": time_integrator,
"friction_alpha": friction_alpha,
```

In `pipe_network_service.py` `build_pipe_network_config`, read from config:
```python
recon_method = int(config.get("recon_method", 0))
time_integrator = int(config.get("time_integrator", 1))
friction_alpha = float(config.get("friction_alpha", 0.01))
```

Add to the returned `PipeNetworkConfig`.

---

### Task 11: Widgets + getters in model_tab_view.py

**Files:**
- Modify: `swe2d/workbench/views/model_tab_view.py`

Add three widgets after the existing pipe surcharge method combo (after line ~1357):

```python
# ── Pipe reconstruction method ──────────────────────────────────────────
pipe_recon_layout = QHBoxLayout()
pipe_recon_label = QLabel("Pipe reconstruction")
self.pipe_recon_method_combo = QComboBox()
self.pipe_recon_method_combo.addItem("First-order upwind", 0)
self.pipe_recon_method_combo.addItem("MUSCL-minmod", 1)
self.pipe_recon_method_combo.setCurrentIndex(0)
pipe_recon_layout.addWidget(pipe_recon_label)
pipe_recon_layout.addWidget(self.pipe_recon_method_combo, 1)
pipe_solver_form_layout.addRow(pipe_recon_label, self.pipe_recon_method_combo)

# ── Pipe time integrator ───────────────────────────────────────────────
self.pipe_time_integrator_combo = QComboBox()
self.pipe_time_integrator_combo.addItem("Forward Euler (RK1)", 0)
self.pipe_time_integrator_combo.addItem("RK2", 1)
self.pipe_time_integrator_combo.setCurrentIndex(1)
pipe_solver_form_layout.addRow("Pipe time integrator", self.pipe_time_integrator_combo)

# ── Friction alpha boost ───────────────────────────────────────────────
self.pipe_friction_alpha_spin = QDoubleSpinBox()
self.pipe_friction_alpha_spin.setDecimals(4)
self.pipe_friction_alpha_spin.setRange(0.0, 1.0)
self.pipe_friction_alpha_spin.setSingleStep(0.001)
self.pipe_friction_alpha_spin.setValue(0.01)
self.pipe_friction_alpha_spin.setTooltip(
    "Adds linear-Q damping to Manning friction at large discharges.\n"
    "gamma += alpha * |Q| / A_full\n\n"
    "Range: 0.0 (Manning's only) to 1.0 (very strong damping).\n"
    "Typical values: 0.001-0.1 provide mild damping for stability.\n"
    "Default 0.01. Higher values stabilize high-flow conditions\n"
    "but can slow startup transients."
)
pipe_solver_form_layout.addRow("Friction alpha boost", self.pipe_friction_alpha_spin)
```

Add getter methods:
```python
def get_pipe_recon_method(self) -> int:
    return int(self.pipe_recon_method_combo.currentData())

def get_pipe_time_integrator(self) -> int:
    return int(self.pipe_time_integrator_combo.currentData())

def get_pipe_friction_alpha(self) -> float:
    return float(self.pipe_friction_alpha_spin.value())
```

---

### Task 12: Controller wiring in studio_dialog.py

**Files:**
- Modify: `swe2d/workbench/studio_dialog.py`

In `_collect_run_widget_params` (around line 1995-1996), add:
```python
recon_method=self._model_tab_view.get_pipe_recon_method(),
time_integrator=self._model_tab_view.get_pipe_time_integrator(),
friction_alpha=self._model_tab_view.get_pipe_friction_alpha(),
```

---

### Task 13: Build + verify

- [ ] **Step 1: Build**

```bash
cd build && cmake .. -DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_BUILD_TYPE=Release && cmake --build . -j$(nproc) 2>&1 | tail -30
```

Expected: clean build with no errors.

- [ ] **Step 2: Verify Python import**

```bash
mamba run -n qgis_stable python3 -c "import swe2d.coupling; print('Import OK')"
```

- [ ] **Step 3: Purge caches**

```bash
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; echo "Done"
```

- [ ] **Step 4: Test from QGIS (manual)**

Load MESH1 with 500 cfs hydrograph, run at dt=0.05 with default settings (first-order, RK2, alpha=0.01). Verify stability. Then toggle MUSCL-minmod, verify profile smoothness improves. Then toggle RK1, verify single-stage behavior (same CFL). Then vary alpha slider.

---

## Self-Review Checklist

- **Spec coverage:** All three features (MUSCL, RK1, alpha) have corresponding C++ kernels, Python plumbing, and widgets. Pipe cell persistence fix from earlier session is separate.
- **Placeholder scan:** All code blocks are concrete. No TBD/TODO.
- **Type consistency:** recon_method=0/1, time_integrator=0/1, friction_alpha=double. Consistent across all layers.
