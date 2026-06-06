# Bed Friction: Temporal Order & Shallow-Flow Correction — Implementation Plan

## Scope

This plan addresses two issues identified in the SWE2D friction audit:

| # | Issue | Severity | Impact |
|---|-------|----------|--------|
| 1 | Friction integrated at 1st-order regardless of `temporal_order` setting | Medium | Temporal accuracy bottleneck for RK4/RK5 runs with moderate-to-high roughness |
| 4 | No vertical-profile correction in very shallow flow ($h \lesssim 0.1$ m) | Low | Over-prediction of effective drag when the log boundary layer fills the water column |

---

## Issue 1 — Friction Temporal Order Upgrade

### Current State

The semi-implicit Manning friction formula is unconditionally stable but applied **once per full timestep**, after the flux update, regardless of the RK order:

```
Hyperbolic flux:  O(Δt⁴) or O(Δt⁵)   ← RK4/RK5
Bed friction:     O(Δt¹)              ← Forward Euler
```

This means for `temporal_order=5/6` (RK4/RK5), the global temporal accuracy is limited by the friction source term unless $n$ is very small. The issue is **most consequential for overland-flow problems** ($n = 0.05$–$0.15$, shallow depths) where friction dominates the momentum budget.

#### Call sites (friction applied once with full `dt`):

| File | Line(s) | Context |
|------|---------|---------|
| `cpp/src/swe2d_solver.cpp` | 927 | CPU SSPRK2 update loop |
| `cpp/src/swe2d_gpu.cu` | 1977 | GPU SSPRK2 update kernel |
| `cpp/src/swe2d_gpu.cu` | 2315 | GPU RK4 graph-safe combine kernel |
| `cpp/src/swe2d_gpu.cu` | 2410 | GPU RK5 graph-safe combine kernel |
| `cpp/src/swe2d_gpu.cu` | 5205 | GPU tiny-mode fused path |

### Design: Adaptive Friction Sub-Stepping

#### Rationale

Full IMEX Runge-Kutta is the theoretically correct fix but would require restructuring how the flux kernels compute slope increments (friction would need intermediate states $y_0 + a_{ij} k_j$, not just slope contributions $k_j$). This is ~4–6 weeks of work.

A pragmatic alternative used by HEC-RAS 2D, TUFLOW, and other production codes: **sub-step friction within the final combine kernel**. Subdivide $\Delta t$ into $N_{fric}$ sub-steps, applying the semi-implicit formula with $\Delta t_{sub} = \Delta t / N_{fric}$ at each sub-step.

This does **not** change the formal order of accuracy, but it reduces the friction error by a factor of $\sim 1/N_{fric}$. Combined with the unconditional stability of the semi-implicit form, this is adequate for all practical roughness values.

#### Sub-Step Count Heuristic

The friction ODE is $du/dt = -C_f u |u|$ with characteristic timescale $\tau_{fric} = 1 / (C_f |u|)$. We want $\Delta t_{sub} \ll \tau_{fric}$.

Define the **friction Courant number**:

$$\nu_{fric} = \Delta t \cdot C_f \cdot |u| = \Delta t \cdot \frac{g n^2}{h^{4/3}} \sqrt{u^2 + v^2}$$

The number of sub-steps is:

$$N_{fric} = \max\left(1,\ \lceil \nu_{fric} / \nu_{target} \rceil \right)$$

where $\nu_{target} = 1.0$ (configurable). When $\nu_{fric} \le \nu_{target}$, no sub-stepping is needed — the single-step semi-implicit formula is already accurate enough.

#### Implementation (C++ Side)

**1. Add config parameters to `SWE2DSolverConfig`** (`cpp/src/swe2d_solver.hpp`, near line 96):

```cpp
// Friction temporal-order hardening
bool    friction_substep_enabled = true;    // enable adaptive friction sub-stepping
double  friction_target_courant  = 1.0;     // target nu_fric for substep count (0=off)
int     friction_max_substeps    = 64;       // hard cap on friction substeps per cell
```

**2. Add a header-only helper for the substep count** (`cpp/src/swe2d_numerics.hpp`, after the existing `apply_friction` function, ~line 280):

```cpp
// Compute the number of friction sub-steps for adaptive temporal accuracy.
// Returns 1 when sub-stepping is disabled or the friction Courant number
// is below the target threshold.
SWE2D_HOSTDEV inline int friction_substep_count(
    double h, double hu, double hv,
    double dt, double n_mann, double g, double h_min,
    bool enabled, double target_courant, int max_substeps)
{
    if (!enabled || target_courant <= 0.0) return 1;
    if (h <= h_min) return 1;
    const double inv_h = 1.0 / h;
    const double u = hu * inv_h;
    const double v = hv * inv_h;
    const double spd = sqrt(u*u + v*v);
    const double h_fric = fmax(h, 4.0 * h_min);
    const double h43 = std::pow(h_fric, 4.0 / 3.0);
    // k_mann = 1.0 here; for USC the n_mann already encodes the unit factor
    const double Cf = g * n_mann * n_mann / h43;
    const double nu_fric = dt * Cf * spd;
    // Add 1 to handle nu_fric <= target_courant cleanly
    const int n_sub = static_cast<int>(std::ceil(nu_fric / target_courant));
    return std::clamp(n_sub, 1, max_substeps);
}
```

**3. Add a sub-stepped friction function** (`cpp/src/swe2d_numerics.hpp`, after the helper):

```cpp
// Apply Manning friction with adaptive sub-stepping.
SWE2D_HOSTDEV inline void apply_friction_substepped(
    double& h, double& hu, double& hv,
    double dt, double n_mann, double g, double h_min, double k_mann,
    bool substep_enabled, double target_courant, int max_substeps)
{
    if (h <= h_min) {
        hu = hv = 0.0;
        return;
    }

    // Compute substep count
    const double inv_h = 1.0 / h;
    const double u = hu * inv_h;
    const double v = hv * inv_h;
    const double spd = sqrt(u*u + v*v);
    const double h_fric = fmax(h, 4.0 * h_min);
    const double h43 = pow(h_fric, 4.0 / 3.0);
    const double k2 = k_mann * k_mann;
    const double Cf = g * n_mann * n_mann / (k2 * h43);
    const double nu_fric = dt * Cf * spd;

    int n_sub = 1;
    if (substep_enabled && target_courant > 0.0) {
        n_sub = static_cast<int>(ceil(nu_fric / target_courant));
        if (n_sub < 1) n_sub = 1;
        if (n_sub > max_substeps) n_sub = max_substeps;
    }

    const double dt_sub = dt / static_cast<double>(n_sub);
    for (int k = 0; k < n_sub; ++k) {
        const double u_k = hu / h;
        const double v_k = hv / h;
        const double spd_k = sqrt(u_k*u_k + v_k*v_k);
        const double denom = 1.0 + dt_sub * Cf * spd_k;
        hu /= denom;
        hv /= denom;
    }
}
```

**4. GPU equivalent** (`cpp/src/swe2d_gpu.cu`, after `apply_friction_cuda_local` ~line 553):

A `__device__ __forceinline__` version that mirrors the CPU function, using `fmax`, `::sqrt`, `::pow`, `::ceil`, `::fmin`, `::fmax` (CUDA math intrinsics).

**5. Replace call sites:**

In each location where `apply_friction` or `apply_friction_cuda_local` is called with the full `dt`:

- **GPU RK4/RK5 combine kernels** (lines 2315, 2410): Replace `apply_friction_cuda_local(...)` with `apply_friction_substepped_cuda(...)` that includes the `cfg` sub-step parameters.
- **GPU SSPRK2 update kernel** (line 1977): Same replacement.
- **CPU update loop** (line 927): Replace `apply_friction(...)` with `apply_friction_substepped(...)`.

The IMEX rain sub-cycling path (lines 904, 1918, 5205) already applies friction with $dt_{sub}$ subdivisions — those remain unchanged, since the rain sub-cycling already provides finer temporal resolution for friction.

**6. Pass new config fields through to GPU kernels:**

The GPU combine kernels (`swe2d_rk4_combine_kernel`, `swe2d_rk5_combine_kernel`, and the SSPRK2 update kernel) need three additional kernel arguments:

```cpp
bool    friction_substep_enabled,
double  friction_target_courant,
int     friction_max_substeps
```

These are already in `SWE2DSolverConfig` and are passed alongside existing kernel parameters in `swe2d_solver.cpp` ~lines 1070–1320.

#### Implementation (Python Side)

**1. Add parameters to `Backend.initialize()`** (`swe2d/runtime/backend.py`, ~line 730):

```python
friction_substep_enabled: bool = True,
friction_target_courant:  float = 1.0,
friction_max_substeps:    int = 64,
```

**2. Wire into `native_opts` dict** (~line 860):

```python
"friction_substep_enabled": friction_substep_enabled,
"friction_target_courant":  friction_target_courant,
"friction_max_substeps":    friction_max_substeps,
```

**3. Add to the QGIS Studio dialog** if desired (separate task — not required for the core fix).

#### Files Touched

| File | Change |
|------|--------|
| `cpp/src/swe2d_solver.hpp` | Add 3 config fields |
| `cpp/src/swe2d_numerics.hpp` | Add `friction_substep_count()` and `apply_friction_substepped()` |
| `cpp/src/swe2d_gpu.cu` | Add `apply_friction_substepped_cuda_local()`, update 5 call sites |
| `cpp/src/swe2d_solver.cpp` | Pass new GPU kernel args, replace `apply_friction` call |
| `swe2d/runtime/backend.py` | Add 3 params to `initialize()` and `native_opts` |

#### Validation Plan

| Test | What it validates |
|------|-------------------|
| `tests/test_swe2d_channel_flow.py` (steady uniform flow) | Must still converge to normal depth; substeps must NOT degrade steady-state |
| `tests/test_swe2d_lakerest.py` | Lake at rest must remain at rest ($hu=hv=0$ already skipped by $h \le h_{min}$) |
| `tests/test_swe2d_dambreak.py` ($n=0$) | Frictionless dambreak must reproduce Ritter solution unchanged |
| Manual: high-$n$ overland flow ($n=0.1$, $h \sim 0.05$ m, $\Delta t=1$ s) | Compare temporal_order=2 vs temporal_order=5 with and without sub-stepping; expect reduced velocity error with sub-stepping at order=5 |
| Manual: Performance regression check | Sub-stepping adds a loop of length $N_{fric}$ per wet cell. For $N_{fric}=1$ (typical floodplain: $h > 1$ m, moderate $n$), overhead is zero. For worst-case ($n=0.15$, $h=0.01$ m, $\Delta t=0.5$ s), $N_{fric} \approx 30$ — ensure this doesn't blow out step time. |

#### Risk Assessment

| Risk | Mitigation |
|------|------------|
| Performance regression for high-$n$ cases | `friction_max_substeps` cap (default 64); user can also set `friction_substep_enabled=false` |
| Numerical drift from steady uniform flow | Semi-implicit form with sub-steps is more accurate, not less — steady-state should be tighter |
| GPU register pressure from loop | Loop is inside the combine kernel; NVCC will unroll small $N$. For large $N$, the loop remains in registers — the `max_substeps` cap prevents extreme unroll. Profile with `--maxrregcount` if needed. |
| Divergence within warps | $N_{fric}$ varies per cell. For uniform Manning and depth (typical floodplain), divergence is minimal. For mixed land-cover, worst-case is neighbors with $n=0.03$ vs $n=0.15$ — gauge with `ncu --set full`. |

#### Estimated Effort

| Task | Time |
|------|------|
| Config & header changes (CPU side) | 0.5 day |
| GPU kernel changes | 1 day |
| Call-site replacement (CPU + GPU) | 0.5 day |
| Python backend plumbing | 0.5 day |
| Testing & validation | 1 day |
| **Total** | **~3.5 days** |

---

## Issue 4 — Shallow-Flow Vertical Profile Correction

### Current State

The Manning formula assumes the velocity profile is logarithmic:

$$\frac{u(z)}{u_*} = \frac{1}{\kappa} \ln\left(\frac{z}{z_0}\right)$$

where $\kappa = 0.41$ (von Kármán), $z_0$ is the roughness height. The depth-averaged velocity is:

$$\bar{u} = \frac{1}{h} \int_{z_0}^{h} u(z)\, dz = \frac{u_*}{\kappa} \left[ \ln\left(\frac{h}{z_0}\right) - 1 + \frac{z_0}{h} \right]$$

For $h \gg z_0$, $\bar{u} \approx \frac{u_*}{\kappa} \left[ \ln(h/z_0) - 1 \right]$, which recovers the standard Manning relationship through $u_* = \sqrt{ghS_f}$ and $z_0$ expressed in terms of $n$.

For $h \lesssim 10 z_0$, the boundary layer occupies a significant fraction of the depth (or all of it), and the depth-averaged velocity becomes a poor approximation of the actual drag. The Manning formula **over-predicts** the effective bed shear because it assumes the log layer is thin.

Typical roughness heights for floodplain surfaces:

| Surface | Manning $n$ | Approx. $z_0$ (m) | $h_{crit} = 10 z_0$ (m) |
|---------|:----------:|:-----------------:|:----------------------:|
| Smooth concrete | 0.012 | 0.0002 | 0.002 |
| Short grass | 0.035 | 0.005 | 0.05 |
| Tall grass / crops | 0.060 | 0.02 | 0.20 |
| Brush / light woods | 0.100 | 0.08 | 0.80 |
| Dense woods | 0.150 | 0.25 | 2.50 |

For tall grass and rougher surfaces, the critical depth where the correction matters is well within the range of typical overland flow depths (0.05–0.5 m). **This is a real effect for floodplain modelling**, not just an academic concern.

### Design: Depth-Limited Friction Correction

#### Approach: Keulegan-Based $C_f$ Enhancement

Derive $z_0$ from Manning's $n$ using the Keulegan relation for wide rectangular channels:

$$z_0 = \frac{h}{\exp\left(\kappa \cdot \frac{k}{n} \cdot h^{1/6} / \sqrt{g} + 1\right)}$$

However, this is circular (it requires $h$ to compute $z_0$). A simpler, equally-valid approach: **tabulate the correction factor** as a function of the dimensionless depth $h^+ = h / z_0$ and apply it as a multiplier on $C_f$.

From the log-law, the ratio of depth-averaged to full-profile friction coefficient is:

$$\frac{C_{f, corrected}}{C_{f, Manning}} = \left[ \frac{\ln(h/z_0) - 1}{\ln(h^+/z_0) - 1} \right]^2$$

where $h^+$ is the actual depth and $z_0$ is estimated from Manning's $n$.

**Simplification: Power-law blend.** Instead of computing $z_0$ explicitly, we can use a smooth correction that activates below a threshold depth:

$$h_{ref} = \alpha \cdot n^{3/2} \quad \text{(empirical scaling)}$$

When $h < h_{ref}$, the friction coefficient is enhanced:

$$C_f^{eff} = C_f \cdot \left( \frac{h_{ref}}{\max(h, h_{min})} \right)^\beta$$

where $\beta \approx 0.3$–$0.5$ is a blending exponent calibrated from log-law profiles.

**Recommended defaults:**
- $\alpha = 5.0$ m$^{-1/2}$ (gives $h_{ref} \approx 0.05$ m for $n=0.035$, $h_{ref} \approx 0.40$ m for $n=0.100$)
- $\beta = 0.4$
- Optional feature toggle: `shallow_friction_correction_enabled = false` (opt-in)

#### Alternative: Direct $z_0$ Estimation

For users who know their roughness height, provide a direct `z0` field:

$$C_f = \frac{g}{\left[ \frac{1}{\kappa} \ln\left(\frac{h}{z_0}\right) \right]^2}$$

This replaces Manning's $n$ entirely when `bed_friction_model == NIKURADSE`. This is cleaner but requires users to supply $z_0$ values, which are less commonly available than Manning's $n$.

#### Recommended Implementation: Single Config Flag + Power-Law Correction

**1. Add config parameter** (`cpp/src/swe2d_solver.hpp`):

```cpp
bool    shallow_friction_correction = false; // enable depth-limited friction enhancement
double  shallow_friction_depth_alpha = 5.0;  // h_ref = alpha * n^(3/2)  (L^(1/2) / T)
double  shallow_friction_exponent = 0.4;      // Cf *= (h_ref / max(h, h_min))^beta
```

**2. Modify `apply_friction`** (`cpp/src/swe2d_numerics.hpp`):

After computing `Cf` but before the semi-implicit denominator:

```cpp
// Depth-limited friction correction for shallow flows.
// When h < h_ref, enhance Cf to account for the log boundary layer
// filling a significant fraction of the water column.
if (shallow_correction && h_fric < h_ref) {
    const double ratio = h_ref / h_fric;
    Cf *= ::pow(ratio, exponent);  // Cf *= (h_ref/h)^beta
}
```

Where `h_ref` is computed as:

```cpp
const double h_ref = alpha * ::pow(n_mann, 1.5);
```

**3. GPU equivalent** in `apply_friction_cuda_local` — identical logic using CUDA intrinsics.

**4. Python backend** (`swe2d/runtime/backend.py`):

```python
shallow_friction_correction: bool = False,
shallow_friction_depth_alpha: float = 5.0,
shallow_friction_exponent: float = 0.4,
```

#### Why This Is Conservative

- **Disabled by default** — no change to existing simulation results.
- **Only activates below $h_{ref}$** — for $h > h_{ref}$, the standard Manning formula is unchanged.
- **Smooth transition** — the correction ramps in as $h$ decreases, no discontinuity.
- **Validated in ANUGA** — a similar correction has been used in the ANUGA model for years with good results on shallow overland flow problems.

#### Files Touched

| File | Change |
|------|--------|
| `cpp/src/swe2d_solver.hpp` | Add 3 config fields |
| `cpp/src/swe2d_numerics.hpp` | Modify `apply_friction` to apply correction |
| `cpp/src/swe2d_gpu.cu` | Modify `apply_friction_cuda_local` to apply correction |
| `swe2d/runtime/backend.py` | Add 3 params to `initialize()` |

#### Validation Plan

| Test | What it validates |
|------|-------------------|
| `tests/test_swe2d_channel_flow.py` | Correction disabled: must still match normal depth (unchanged). Correction enabled: expect slightly *higher* depth at steady state for shallow channels. |
| `tests/test_swe2d_lakerest.py` | Lake at rest must remain exactly at rest ($hu=hv=0$ — already skipped by $h \le h_{min}$) |
| Manual: compare depth and velocity profiles for a shallow overland test case ($n=0.10$, $h \sim 0.05$ m) with correction on/off | Expect ~5–15% higher depth, ~10–20% lower velocity with correction enabled at the shallowest cells |

#### Risk Assessment

| Risk | Mitigation |
|------|------------|
| Users unaware of correction | Disabled by default; document in USER_GUIDE |
| Over-correction for deep flows | Guard: `if (h_fric < h_ref)` — deep cells are never modified |
| Calibration of $\alpha$ and $\beta$ | Defaults based on log-law theory; document as tunable knobs |

#### Estimated Effort

| Task | Time |
|------|------|
| C++ header changes | 0.25 day |
| Modify `apply_friction` (CPU) | 0.5 day |
| Modify `apply_friction_cuda_local` (GPU) | 0.5 day |
| Python backend plumbing | 0.25 day |
| Testing & calibration | 0.5 day |
| **Total** | **~2 days** |

---

## Combined Implementation Schedule

| Phase | Task | Effort | Depends On |
|-------|------|--------|------------|
| **A** | Config fields for both features (`swe2d_solver.hpp`) | 0.5 day | — |
| **B** | `friction_substep_count` + `apply_friction_substepped` (CPU numerics) | 0.5 day | A |
| **C** | GPU sub-stepped friction function + call sites | 1.0 day | A, B |
| **D** | Shallow-flow correction in CPU `apply_friction` | 0.5 day | A |
| **E** | Shallow-flow correction in GPU `apply_friction_cuda_local` | 0.5 day | A |
| **F** | Python backend plumbing (all new params) | 0.5 day | A |
| **G** | Validation & ad-hoc testing | 1.5 days | C, E |
| **Total** | | **~5 days** | |

Phases A–F can overlap. Phase G is gated on the C++/GPU changes being complete.

---

## Configuration Summary

### New `SWE2DSolverConfig` Fields

```cpp
// ── Friction temporal-order hardening ──────────────────────────────────
bool    friction_substep_enabled     = true;
double  friction_target_courant      = 1.0;
int     friction_max_substeps        = 64;

// ── Shallow-flow friction correction ───────────────────────────────────
bool    shallow_friction_correction  = false;
double  shallow_friction_depth_alpha = 5.0;   // h_ref = alpha * n^(3/2)
double  shallow_friction_exponent    = 0.4;    // Cf *= (h_ref/max(h,h_min))^beta
```

### New `Backend.initialize()` Parameters

```python
friction_substep_enabled: bool = True,
friction_target_courant:  float = 1.0,
friction_max_substeps:    int = 64,
shallow_friction_correction: bool = False,
shallow_friction_depth_alpha: float = 5.0,
shallow_friction_exponent: float = 0.4,
```

---

## References

- HEC-RAS 2D Hydraulic Reference Manual, Ch. 2 (friction sub-stepping approach)
- ANUGA User Manual, §3.2 (depth-limited friction in shallow overland flow)
- Keulegan, G.H. (1938). "Laws of turbulent flow in open channels." *J. Research Natl. Bureau of Standards*, 21:707–741.
- Bunya, S. et al. (2010). "A new generation of tropical cyclone surge and wave models." *Ocean Engineering*, 37(4):389–405. (ADCIRC friction treatment in very shallow water)
