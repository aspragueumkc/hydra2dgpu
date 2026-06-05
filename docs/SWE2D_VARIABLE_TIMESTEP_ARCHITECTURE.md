# SWE2D Variable Time Stepping Architecture

> **Author:** AI-assisted documentation  
> **Date:** 2026-06-05  
> **Status:** Current  
> **Related:** `cpp/src/swe2d_solver.hpp`, `cpp/src/swe2d_gpu.cu`, `swe2d/runtime/backend.py`

---

## 1. Algorithm Overview

SWE2D supports two timestep modes:

| Mode | UI Setting | Behavior |
|------|-----------|----------|
| **Fixed dt** | `adaptive_cfl_dt_chk` unchecked | `dt = dt_fixed` every step (CFL ignored) |
| **CFL-adaptive dt** | `adaptive_cfl_dt_chk` checked | `dt = min(cfl / λ_max, dt_max)` per step |

### 1.1 CFL-Based Adaptive dt

The CFL condition for explicit shallow-water solvers on triangular meshes requires:

$$
\Delta t \leq \frac{\text{CFL}}{\lambda_{\max}}
$$

where $\lambda_{\max}$ is the global maximum wave speed across all edges:

$$
\lambda = \frac{|u| + \sqrt{gh}}{1} + \frac{|v| + \sqrt{gh}}{1}
$$

and CFL is a safety factor (default 0.45, appropriate for triangular meshes).

### 1.2 GPU CFL Reduction (Two-Stage)

1. **Stage 1 — Per-edge kernel** (`swe2d_cfl_kernel`): One thread per edge computes local $\lambda$. Block-level `atomicMax` produces per-block maxima.
2. **Stage 2 — Block reduce** (`swe2d_cfl_reduce_blocks_kernel`): Single-block reduction of block maxima to global $\lambda_{\max}$.
3. **Final dt**: `dt = min(cfl / λ_max, dt_max)`. If $\lambda_{\max} \leq 0$ (dry domain), returns `dt_max`.

### 1.3 Fixed dt Mode

When `dt_fixed > 0` in `SWE2DSolverConfig`, the solver bypasses CFL computation entirely and uses `dt_fixed` every step. CFL is still available as a secondary cap via `dt_request`.

---

## 2. dt Computation Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ UI (swe2d_model_tab.ui)                                        │
│   cfl_spin → CFL safety factor (0.45)                         │
│   dt_spin → dt_fixed or dt_max (0.05)                          │
│   adaptive_cfl_dt_chk → mode selector                         │
│   initial_dt_spin → first-step override (0.0 = auto)          │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│ RunOptionsBuilder.build()                                      │
│   dt_fixed = -1.0 if adaptive else dt_cfg                     │
│   dt_request = -1.0 if adaptive else dt_cfg                   │
│   initial_dt = float(ui.initial_dt_spin.value())               │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│ SWE2DBackend.initialize(cfl, dt_max, dt_fixed, initial_dt)    │
│   Stores config → forwards to C++ SWE2DSolverConfig            │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│ swe2d_step(solver, dt_request)                                 │
│   if cfg.dt_initial > 0 && !first_step_done:                   │
│       dt = cfg.dt_initial       ← first-step override         │
│       first_step_done = true                                   │
│   elif cfg.dt_fixed > 0:                                       │
│       dt = cfg.dt_fixed         ← fixed mode                  │
│   else:                                                        │
│       dt_cfl = compute_cfl_dt()  ← GPU CFL reduction          │
│       dt = min(dt_request, dt_cfl) if dt_request > 0           │
│          else dt_cfl                                           │
└─────────────────────────────────────────────────────────────────┘
```

### 2.1 The `dt_request` Parameter

`dt_request` is a secondary cap on top of CFL. It allows the Python run loop to limit dt without changing the solver config. When `dt_request > 0`, the solver uses `min(dt_request, dt_cfl)`. When `dt_request <= 0` (default), CFL governs alone.

---

## 3. Temporal Integration Schemes

| `temporal_order` | Method | Stages | SSP | Graph-Safe | Notes |
|:-:|--------|:-:|:-:|:-:|-------|
| 1 | Forward Euler | 1 | Yes | No | Simplest; unconditional SSP |
| 2 | SSP-RK2 (Heun) | 2 | Yes | No | Default; good robustness/accuracy tradeoff |
| 4 | Classic RK4 | 4 | No | No | Composed method; higher accuracy |
| 5 | Graph-safe RK4 | 6 | No | Yes | Cash-Karp variant; CUDA graph compatible |
| 6 | Graph-safe RK5 | 6 | No | Yes | Cash-Karp with 5th-order error; highest accuracy |

### 3.1 How dt Is Used Across Stages

**All stages within a single step use the same dt.** There is no sub-cycling or adaptive dt within a step. The dt is computed once at step start and used for all internal stages.

For RK5, the stage times are:
```
stage_c = {0, 0.2, 0.3, 0.6, 1.0, 0.875}
t_stage[i] = t_now + stage_c[i] * dt
```

### 3.2 SSP Property

**SSP (Strong Stability Preserving)** schemes guarantee that the solution remains within physical bounds (e.g. positivity of depth, TVD property) if the CFL condition is satisfied. SSP-RK2 has this property; RK5 does **not**.

This means:
- **SSP-RK2** is robust for wet/dry fronts, shocks, and discontinuous BCs
- **RK5** can overshoot on non-smooth solutions even when CFL is satisfied
- **Recommendation:** Use RK2 for robustness; RK5 only for smooth problems with well-resolved initial conditions

### 3.3 Graph-Safe Schemes

Graph-safe schemes (`temporal_order >= 5`) precompute all stage-forcing (BCs, rainfall, sources) upfront in a batch before executing the 6 RHS evaluations. This is required for CUDA graph replay because graph capture cannot contain host-side decisions or memory allocations.

Non-graph schemes (Euler, RK2) apply BCs inline at each stage time, which is simpler but incompatible with graph capture.

---

## 4. Boundary Condition Interaction with dt

### 4.1 Constant BCs (Q-inflow, WSE)

Constant boundary conditions are **independent of dt magnitude**. The ghost cell state is set to the same value regardless of timestep:
- **Constant Q:** `hu_ghost = -Q * nx`, `hv_ghost = -Q * ny`
- **Constant WSE:** `h_ghost = WSE - bed_elev`

### 4.2 Time-Series Hydrograph BCs

Hydrograph BCs are interpolated at the RK stage times:
```
t_eval = t_now + stage_c[i] * dt
```

**Critical insight:** If dt is large, stage times extrapolate far into the future. For example, with `dt = 0.05s` and stage 5 (`stage_c = 1.0`), the BC is evaluated at `t + 0.05s`. With `dt = 10s` (possible on a dry domain), stage 5 evaluates at `t + 10s` — which may be well beyond the hydrograph data range.

### 4.3 The Dry-Start Problem

On a dry domain:
1. $\lambda_{\max} = 0$ (no water → no wave speed)
2. `compute_cfl_dt()` returns `dt_max` (safe fallback)
3. First step uses full `dt_max` as the timestep
4. **RK5 evaluates BCs at `t + dt_max`** — far into the future
5. With constant Q BC, the ghost cell momentum is injected at the wrong time scale
6. Culvert flows are computed algebraically but the volume injection per step is `Q * dt`, which can be excessive

**This is the root cause of RK5 producing wildly different results from RK2 on cold-start domains with large `dt_max`.**

**Solution:** The `initial_dt` parameter lets the user start with a small timestep (e.g. 0.001s) to build up the flow field before CFL takes over.

---

## 5. Culvert/Structure Coupling

### 5.1 Decoupled from 2D dt

The 2D CFL computation does **not** account for culvert/structure flows. The 2D solver computes dt purely from wave speeds in the 2D domain.

### 5.2 1D Network Substeps

The drainage/structure 1D solver can subdivide the 2D dt:
```
dt_1d = dt_2d / coupling_substeps
```
This allows the 1D solver to converge stiff network dynamics without affecting the 2D CFL.

### 5.3 Algebraic Fluxes

Structure fluxes (orifice/weir formulas) are **algebraic**, not time-integrated:
```
Q_structure = f(headwater, tailwater, geometry)
volume_injected = Q_structure * dt
```

**Potential issue:** Very large dt can cause excessive volume injection per step. The volume is bounded by `Q_max * dt`, which grows linearly with dt.

---

## 6. Stability Controls

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `cfl` | 0.45 | CFL safety factor for explicit dt |
| `dt_max` | 10.0 s | Upper bound on adaptive dt |
| `dt_fixed` | -1.0 | If > 0, overrides CFL with fixed dt |
| `dt_initial` | -1.0 | If > 0, overrides dt for first step only |
| `cfl_lambda_cap` | 1e6 | Prevents division by huge wave speeds in degenerate cells |
| `max_rel_depth_increase` | 2.0 | Per-step limit: `h ≤ h_old + rel * max(h_old, h_min)` |
| `momentum_cap_min_speed` | 50.0 | Absolute min speed cap for momentum limiting |
| `momentum_cap_celerity_mult` | 20.0 | Speed cap = `max(min_speed, mult * √(gh))` |
| `depth_cap` | 1e6 | Hard upper bound on depth |
| `shallow_damping_depth` | 1e-4 | Momentum damping toward zero as h → h_min |
| `front_flux_damping` | 0.5 | Momentum-flux scale on wet/dry front edges |
| `active_set_hysteresis` | true | Keep cells active 1 extra step after drying |

---

## 7. Known Issues and Improvement Opportunities

### 7.1 No Mid-Step dt Adaptation

All RK stages use the same dt computed at step start. If the CFL condition changes during intermediate stages (e.g. due to BC injection or rapid wetting), the dt may be stale.

**Improvement:** Re-check CFL between RK stages for very long dt steps. Complexity: high; benefit: marginal for most use cases.

### 7.2 RK5 BC Extrapolation with Large dt

RK5 stage times (`stage_c = {0, 0.2, 0.3, 0.6, 1.0, 0.875}`) can evaluate hydrograph BCs far beyond the next data point when dt is large.

**Improvement:** Clamp dt to a fraction of the BC timeseries spacing (e.g. `dt ≤ 0.5 * min_spacing`). Complexity: moderate; benefit: significant for hydrograph-driven problems.

### 7.3 No dt Feedback from Structures

Structure/culvert flows don't feedback to reduce 2D dt. Very large dt can cause excessive volume injection per step.

**Improvement:** Volume-limited dt capping when structures are active: `dt ≤ V_max / Q_max`. Complexity: moderate; benefit: prevents unphysical volume injection.

### 7.4 Non-SSP RK5 Coefficients

The Cash-Karp coefficients used in the graph-safe RK5 combine kernel are **not SSP-stable**. This means the scheme can overshoot on non-smooth solutions (wet/dry fronts, shock-like BCs) even when CFL is satisfied.

**Improvement:** Document this clearly; recommend RK2 for robustness. Consider SSP-RK variants for higher-order if needed in the future.

### 7.5 Global CFL, Not Per-Cell

The most restrictive cell in the entire domain governs dt for all cells. This is conservative but can be very restrictive if one cell has high wave speed (e.g. a deep cell with high velocity).

**Improvement:** Local time stepping (different dt per cell). Complexity: very high; benefit: significant for domains with large variation in wave speed. Not recommended for near-term.

### 7.6 Dry Domain Returns dt_max

When $\lambda_{\max} \leq 0$, `compute_cfl_dt()` returns `dt_max`. This is correct for dry domains (no stability constraint) but problematic for cold starts.

**Improvement:** The `initial_dt` parameter addresses this directly (implemented in this changeset).

---

## 8. File Reference

| File | Role |
|------|------|
| `cpp/src/swe2d_solver.hpp` | `SWE2DSolverConfig` struct, `SWE2DSolver` struct |
| `cpp/src/swe2d_solver.cpp` | `compute_cfl_dt()`, `swe2d_step()` dispatch |
| `cpp/src/swe2d_gpu.cu` | GPU CFL kernels, RK2/RK5 step functions |
| `swe2d/runtime/backend.py` | `SWE2DBackend.initialize()`, `step()`, `run()` |
| `swe2d/runtime/run_options_builder.py` | `SWE2DRunOptionsData`, `SWE2DRunOptionsBuilder.build()` |
| `swe2d/workbench/extracted/model_and_run_methods.py` | UI binding, `_on_run()` orchestration |
| `swe2d/extensions/extension_models.py` | `TemporalScheme` enum, `SolverModelOptions` |
| `swe2d/extensions/drainage_network.py` | `solve_network_step()` — 1D coupling substeps |
| `forms/swe2d_model_tab.ui` | Solver parameter UI widgets |
