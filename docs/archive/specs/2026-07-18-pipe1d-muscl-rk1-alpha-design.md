---
type: spec
status: complete
created: 2026-07-18
completed: 2026-07-25
---

# Pipe1d MUSCL-minmod + RK1 + Runtime Alpha Boost

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement.

**Goal:** Add second-order MUSCL-minmod reconstruction, RK1 (Forward Euler) time integrator, and runtime-selectable friction alpha boost to the 1D pipe Godunov solver, all with GUI widgets.

**Architecture:** Three new C++ kernel parameters (`recon_method`, `time_integrator`, `friction_alpha`) plumbed from Qt combo/spin widgets through `PipeNetworkConfig` → `SWE2DDrainageSoA` → pybind11 binding, following the existing `friction_method`/`surcharge_method` pattern.

**Tech Stack:** CUDA C++ (new slope kernel, modified flux/godunov kernels), PyQt5 (widgets), Python (dataclass plumbing), pybind11.

---

## 1. Slope-limited MUSCL reconstruction

### Kernel: `swe2d_pipe1d_compute_slopes_kernel`

New GPU kernel that writes `d_slope_H[n_cells]` (double). Running for each cell:

```
H_left  = (cell is first in link) ? H[cell] : H[cell-1]
H_right = (cell is last in link)  ? H[cell] : H[cell+1]

delta_left  = H[cell] - H_left
delta_right = H_right - H[cell]

slope = minmod(delta_left / L_left, delta_right / L_right)
       where L_left  = 0.5*(L[cell-1] + L[cell])
             L_right = 0.5*(L[cell] + L[cell+1])
       boundary: zero gradient
```

`minmod(a,b) = 0.5*(sign(a)+sign(b)) * min(|a|,|b|)` — zero if signs differ.

### Modified: `swe2d_pipe1d_flux_kernel`

When `recon_method == 1`:
- Read pre-computed `d_slope_H[cell]` and `d_slope_H[neighbor]`
- Reconstruct face WSEs: `H_L = H_c + 0.5*L_c*slope_c` and `H_R = H_n - 0.5*L_n*slope_n`
- Compute HLLE flux using `(H_L, A_from_H(H_L))` and `(H_R, A_from_H(H_R))` instead of first-order upwind value
- When `recon_method == 0` (default), existing first-order upwind behavior is preserved

### Parameter: `recon_method`

| Value | Meaning |
|-------|---------|
| 0 | First-order upwind (default, current behavior) |
| 1 | MUSCL-minmod |

---

## 2. RK1 (Forward Euler) time integrator

### Modified: `swe2d_pipe1d_godunov_step_internal`

Current RK2 flow:
1. Stage 0: compute flux, write `d_A_new`, `d_Q_new` (partial update = Δt·R(Q⁰))
2. Stage 1: compute flux from stage-0 state, average with stage 0: `Q = 0.5*(Q⁰ + Q¹)`

When `time_integrator == 0` (RK1):
- Run stage 0 only (flux from current state, write to `d_A_new`/`d_Q_new`)
- Copy `d_A_new → d_A`, `d_Q_new → d_Q` directly (no averaging)
- Skip stage 1 entirely
- Same CFL condition (`dt ≤ L / max(c+u)`)
- This matches the 2D model's single-stage Forward Euler pattern

### Parameter: `time_integrator`

| Value | Meaning |
|-------|---------|
| 0 | Forward Euler (RK1) |
| 1 | RK2 (default, current behavior) |

---

## 3. Runtime-selectable friction alpha boost

Currently `FRICTION_STABILITY_ALPHA = 0.01` is a compile-time constexpr. It will become a runtime parameter `friction_alpha` passed through the kernel argument list.

In the godunov update kernel:

```c
const double gamma_stable = friction_alpha * absQ / fmax(A_full, 1.0e-12);
const double gamma = gamma_nat + gamma_stable;
```

### Parameter: `friction_alpha`

| Attribute | Value |
|-----------|-------|
| C++ type | `double` |
| Default | 0.01 |
| Range | 0.0 (Manning's only) to 1.0 (very strong damping) |
| Python widget | `QDoubleSpinBox` |
| Tooltip | "Additive linear-Q damping term added to Manning friction: gamma += alpha·|Q|/A_full. Alpha=0 uses Manning's only. Values 0.001-0.1 provide mild damping; >0.1 may over-damp during startup transients. Default 0.01." |

---

## 4. Widget layout

All three in **Advanced → Drainage → Equation Set**, below the existing pipe friction method and surcharge method combos:

| Label | Widget | Items / Range | Data |
|-------|--------|---------------|------|
| Pipe reconstruction | QComboBox | "First-order upwind", "MUSCL-minmod" | recon_method (0,1) |
| Pipe time integrator | QComboBox | "Forward Euler (RK1)", "RK2" | time_integrator (0,1) |
| Friction alpha boost | QDoubleSpinBox | 0.000–1.000, step 0.001, default 0.010 | friction_alpha |

Tooltip on the friction alpha spin box:
> "Adds linear-Q damping to Manning friction at large discharges. Higher values stabilize high-flow conditions but can slow startup. Range 0.0-1.0; typical values 0.001-0.1. Default 0.01."

---

## 5. Parameter plumbing

### `PipeNetworkConfig` (extension_models.py)
```
recon_method: int = 0
time_integrator: int = 1
friction_alpha: float = 0.01
```

### `SWE2DDrainageSoA` (coupling.py)
```
recon_method: int = 0
time_integrator: int = 1
friction_alpha: float = 0.01
```

### `_apply_coupling` (coupling.py)
```
recon_method=int(dsoa.recon_method),
time_integrator=int(dsoa.time_integrator),
friction_alpha=float(dsoa.friction_alpha),
```

### `swe2d_pipe1d_step` pybind11 binding (swe2d_bindings.cpp)
Add defaulted params after `friction_method`:
```
int32_t recon_method = 0,
int32_t time_integrator = 1,
double friction_alpha = 0.01
```

### `swe2d_pipe1d_step` C++ signature (pipe1d.cuh)
```
int32_t recon_method = 0,
int32_t time_integrator = 1,
double friction_alpha = 0.01
```

### Widget getters (model_tab_view.py)
```python
def get_pipe_recon_method(self) -> int
def get_pipe_time_integrator(self) -> int
def get_pipe_friction_alpha(self) -> float
```

---

## 6. Persistence (no changes needed)

Pipe cell data is already handled by the `pipe_cell_items` fix in `ComputeResult`. The new solver parameters affect only runtime behavior, not result storage.

---

## 7. GPU kernel interface changes

### `swe2d_pipe1d_godunov_update_kernel`
Add trailing params:
```
const double friction_alpha,
const int32_t recon_method
```

### `swe2d_pipe1d_flux_kernel`
Add trailing params:
```
const double* d_slope_H,
const int32_t recon_method
```

### New deviceside arrays
- `d_slope_H`: `n_cells × sizeof(double)`, allocated in device state, written by `swe2d_pipe1d_compute_slopes_kernel`, read by `swe2d_pipe1d_flux_kernel`.

---

## 8. Files Modified

| File | Change |
|------|--------|
| `cpp/src/swe2d_xsect_constants.h` | Remove FRICTION_STABILITY_ALPHA constexpr (now runtime) |
| `cpp/src/pipe1d.cuh` | Update swe2d_pipe1d_step signature, add slopes kernel declaration |
| `cpp/src/pipe1d.cu` | New slopes kernel, modified flux kernel, modified godunov kernel, modified step wrapper, add d_slope_H management to device state allocation |
| `cpp/src/swe2d_bindings.cpp` | Add recon_method/time_integrator/friction_alpha to pybind11 binding |
| `swe2d/extensions/extension_models.py` | Add fields to PipeNetworkConfig |
| `swe2d/runtime/coupling.py` | Add fields to SWE2DDrainageSoA, pass to step call |
| `swe2d/workbench/views/model_tab_view.py` | Add 3 widgets + getters |
| `swe2d/workbench/services/pipe_network_config_service.py` | Forward new params |
| `swe2d/workbench/services/pipe_network_service.py` | Forward new params |
