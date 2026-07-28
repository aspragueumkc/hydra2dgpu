---
type: plan
status: superseded
created: 2026-07-17
completed: 2026-07-25
superseded_by: docs/pipe1d_face_indexed_refactor_plan.md
---

# Pipe1D Unified Semi-Implicit Solver — Implementation Plan

**Date:** 2026-07-17
**Status:** Proposed (pre-implementation discussion)
**Goal:** Replace the current RK2 Godunov + Sjoberg Preissmann slot solver in `cpp/src/pipe1d.cu` with a unified semi-implicit θ-method solver along the lines of Casulli & Stelling (2013) [1] and Hu, Li, Yao & Jin (2019) [2], with two key extensions:

  (a) **No iterative solver** — Hu et al.'s linearisation of $V(\eta)$ by $B^n \cdot \Delta\eta$ gives a tridiagonal linear system, solved directly (no nested Newton).
  (b) **Semi-implicit friction in the (1+γ·Δt) denominator** (Casulli's form) — matching our 2D SWE solver's existing treatment at `cpp/src/swe2d_gpu.cu:444`.
  (c) **No Preissmann slot** for drainage-scale flows — the implicit pressure gradient removes the slot's reason for existing; for our use case (urban drainage, slow regime transitions), the linearised system remains well-defined when $B^n \to 0$ in pressurised cells.

**Architecture:** Hybrid of Hu et al. (linearised, no Newton) and Casulli (semi-implicit pressure + friction), structured as a single tridiagonal solve per timestep on a staggered grid ($\eta$ at pressure points, $u$ at side centres).

**Tech Stack:** CUDA C++ (`cpp/src/pipe1d.{cu,cuh}`, `cpp/src/swe2d_bindings.cpp`), Python bindings, the existing test suite under `tests/`, mamba env `qgis_stable`.

**References:**
- [1] V. Casulli & G. S. Stelling, "A semi-implicit numerical model for urban drainage systems," *Int. J. Numer. Meth. Fluids* 73:600–614, 2013, doi:10.1002/fld.3817.
- [2] D. Hu, S. Li, S. Yao, Z. Jin, "A Simple and Unified Linear Solver for Free-Surface and Pressurized Mixed Flows in Hydraulic Systems," *Water* 11(10):1979, 2019, doi:10.3390/w11101979.

---

## 1. Background — synthesis of the two papers

Both papers solve the same governing equations in 1D:

$$\partial_t u + u \partial_x u = -g \partial_x \eta - \gamma(u, A)\, u \tag{1}$$
$$\partial_t A + \partial_x (A u) = 0 \tag{2}$$

where $\eta$ is piezometric head (water level for open-channel, pressure head for pressurised — same symbol, switching meaning at the regime boundary); $A(\eta)$ is wetted cross-sectional area; $\gamma(u, A) = g\, n_m^2 |u| / R^{4/3}$ is Manning's friction coefficient.

| Aspect | Casulli 2013 | Hu et al. 2019 | **Our target** |
|---|---|---|---|
| Continuity | $V_i(\eta^{n+1})$ form (volume) | $B^n \partial_t \eta + \partial_x(Au) = 0$ (linearised) | **Hu's linearisation** |
| Pressure gradient | Semi-implicit θ | Semi-implicit θ | **Semi-implicit θ** |
| Friction | Implicit in $(1+\gamma\Delta t)$ denominator | Same | **Implicit** (already in 2D) |
| Per-timestep solve | Mildly nonlinear → nested Newton | Tridiagonal linear | **Tridiagonal linear** |
| Iterations | 1–14 outer × 1–32 inner | 0 | **0** |
| Slot policy | None | A-slot @ ε ∈ [0.05, 0.1] for TYPE-III | **None** (no slot, no A-slot) |
| Cross-section $a(\eta) = dA/d\eta$ | Jordan decomposition (non-monotone) | Linearised $B^n$ at $t^n$ | **Linearised $B^n$** |
| Advection | ELM | ELM | **HLLE-style flux** in $F[u^n]$ (kept from current code) |

The key insights that converge the two papers are documented at §2.3.2 of Hu et al.: the linearisation of $V_i(\eta)$ by $B^n \cdot \Delta\eta$ reduces the mildly nonlinear system to a tridiagonal linear one, eliminating Newton iteration. The slot is then a *consequence* of needing $B > 0$ for the linearised matrix to remain well-conditioned in pressurised cells — but for drainage-scale flows where regime transitions are slow, $B^n = 0$ in pressurised cells produces a system that's driven purely by upstream/downstream boundary heads, which is consistent with the incompressible closed-pipe limit.

---

## 2. Target architecture

### 2.1 Staggered variable layout

- $\eta_i$ at pressure points: $i = 1, \ldots, N_p$ where $N_p = $ `n_nodes + n_vnodes` (all node indices, real + virtual).
  - Storage: **new** `d_eta` array of length $N_p$; populated from `d_node_invert + d_node_depth` at the top of each timestep.
- $u_j$ at side centres: $j = 1, \ldots, N_s$ where $N_s = $ `n_owned_faces + n_boundary_faces` (interior + boundary faces).
  - Storage: **new** `d_u` array of length $N_s$ per timestep; constructed from $Q/A$ at faces.
- $A_j$ at side centres: $j = 1, \ldots, N_s$.
  - Storage: **new** `d_side_A`; constructed from interpolating $\eta$ at side endpoints.
- $a_i = (\partial A/\partial \eta)\big|_{\eta_i^n}$ at pressure points.
  - Storage: **new** `d_a_at_node`; precomputed from cross-section tables.

The existing `d_A`, `d_Q` arrays become diagnostic mirrors (written from side values for backwards-compatible diagnostics). Or are dropped entirely — see §6.3 for discussion.

### 2.2 Discrete equations (per Hu et al. 2019 eq 7)

For interior pressure point $i$ with neighbouring segments $S_i$:

$$B_i^n \Delta x_i (\eta_i^{n+1} - \eta_i^n) + g \theta^2 \Delta t^2 \sum_{j \in S_i} \frac{A_j^n}{(1 + \gamma_j^n \Delta t)\, \Delta x_j}\, (\eta_{\omega(i,j)}^{n+1} - \eta_i^{n+1}) = \Delta t \sum_{j \in S_i} \sigma_{ij} A_j^n \cdot \text{(advection at } j\text{)} \tag{3}$$

where:
- $B_i^n = (\partial A/\partial \eta)\big|_{\eta_i^n}$ — wet surface width at pressure point, evaluated at $t^n$ (Hu's linearisation trick).
- $\gamma_j^n = g\, n_m^2 |u_j^n| / R_j^{4/3}$ — Manning's friction coefficient at side $j$.
- $A_j^n = $ cross-sectional area at side $j$, evaluated at $t^n$.
- $\omega(i,j)$ — the *other* endpoint of segment $j$ from pressure point $i$.
- $\sigma_{ij} \in \{-1, +1\}$ — segment orientation sign.
- $\theta \in [0.5, 1.0]$ — semi-implicit weighting factor (typically 1.0 for first-order accuracy).

Rearranged into a tridiagonal system $T \boldsymbol{\eta} = \mathbf{b}$:

$$C_i^a \eta_{i-1}^{n+1} + C_i^b \eta_i^{n+1} + C_i^c \eta_{i+1}^{n+1} = B_i^n \Delta x_i \eta_i^n + r_i^n$$

with coefficients:

$$C_i^a = -\frac{g \theta^2 \Delta t^2 A_{i-1/2}^n}{(1 + \gamma_{i-1/2}^n \Delta t)\, \Delta x_{i-1/2}}$$
$$C_i^c = -\frac{g \theta^2 \Delta t^2 A_{i+1/2}^n}{(1 + \gamma_{i+1/2}^n \Delta t)\, \Delta x_{i+1/2}}$$
$$C_i^b = B_i^n \Delta x_i - C_i^a - C_i^c$$
$$r_i^n = \Delta t \sum_{j \in S_i} \sigma_{ij} A_j^n \cdot F[u^n]_j$$

where $F[u^n]_j$ is the explicit advection flux at side $j$ (HLLE in our case).

### 2.3 Post-solve back-substitution (Casulli eq 5, our variables)

Once $\eta^{n+1}$ is known, $u_j^{n+1}$ is recovered from:

$$u_j^{n+1} = \frac{F[u_j^n] - g \theta \Delta t \cdot (\eta_{\omega(i,j)}^{n+1} - \eta_i^{n+1}) / \Delta x_j}{1 + \gamma_j^n \Delta t} \tag{4}$$

This is the Casulli `q/(1 + γ·Δt)` form — friction in the LHS, no iteration needed.

### 2.4 Mass consistency

With $\eta^{n+1}$ known for every node, $A(\eta^{n+1})$ is recovered per side via interpolation between adjacent node heads. Continuity is then exact at the discrete level (Hu's volume-linearised form is conservative by construction).

### 2.5 Boundary conditions

- **Pipe-end (network node $n$, coupled to 2D cell)**: $\eta_n^{n+1}$ is set by the 2D cell's WSE at this timestep (or computed via weir/orifice face-flux coupling — see §6.4).
- **Outfall with fixed WSE**: $\eta_n^{n+1}$ = fixed prescribed WSE (Dirichlet BC, modifies tridiagonal RHS or coefficient).
- **Inlet with rating curve / prescribed Q**: $A_n^{n+1} u_n^{n+1} = Q_{\text{prescribed}}$ (Robin BC, modified momentum equation at endpoint).

Each branch's tridiagonal system is *independent* except at junctions, where one pressure point is shared by 3+ segments. Junctions are resolved by either (a) iterative block-Gauss-Seidel over junction coupling (1 outer iteration typically suffices for urban drainage), or (b) constructing a block-tridiagonal system per junction.

### 2.6 Why no slot

For our drainage-scale flows:
- Regime transitions (open-channel ↔ pressurised) are spread over tens of seconds across 100s of metres of conduit — much longer than typical $\Delta t \sim 1$ s.
- The Casulli / Hu linearisation remains accurate when $B^n$ changes slowly.
- Pressurised cells ($B^n = 0$) drive $\eta$ from upstream/downstream heads — physically consistent with incompressible closed-pipe behaviour (the "surge" propagates as a head gradient, not as an area growth).
- $C_i^b > 0$ for pressurised cells when $B^n = 0$: matrix reduces to the head-coupling terms alone, still positive-definite and tridiagonal.

The A-slot (Hu et al.'s ε·W) is reserved as a **fallback mode** (rare-encounter flag) for cases with rapid regime transitions (TYPE-III Hu et al., Test Case 3). For drainage-scale work, slot width $T_{\text{slot}} = 0$ is the production setting.

---

## 3. Storage layout changes

### 3.1 New fields in `Pipe1DDeviceState`

| Field | Type | Size | Purpose |
|---|---|---|---|
| `d_eta` | `double*` | `n_nodes + n_vnodes` | Piezometric head at every pressure point |
| `d_u` | `double*` | `n_sides` (owned + boundary) | Velocity $u_j$ at every side centre (per-timestep scratch) |
| `d_side_A` | `double*` | `n_sides` | Cross-section area $A_j$ at side centre |
| `d_a_at_node` | `double*` | `n_nodes + n_vnodes` | Wet surface width $a_i = \partial A/\partial \eta$ at pressure point |
| `d_dA_deta_table` | `double*` | `(n_nodes) * PIPE1D_TABLE_N` | Per-cell $dA/d\eta$ lookup table (replaces Sjoberg slot machinery) |

### 3.2 Obsoleted fields (kept for diagnostic, deprecated for runtime use)

- `d_A`, `d_Q` — cell-centered $(A, Q)$ diagnostic mirrors. Computed post-step from `d_side_A` and `d_u`.
- `d_A_start_save`, `d_Q_start_save` — RK2 machinery. **Removed.**
- `d_A_prev`, `d_Q_iter` — Picard-iteration scratch. **Removed.**
- `d_cell_slot_width` — Preissmann slot width. **Removed.**
- `d_vnode_H`, `d_vnode_Q` — virtual-node WSE/flux. **Removed** (replaced by staggered fields).
- `d_cell_y`, `d_cell_q`, `d_cell_fr`, `d_cell_h` — diagnostic outputs. **Kept but written from staggered state.**
- `d_node_depth` — repurposed as the open-channel portion of $\eta$. For pressurised, $\eta - z > y_{Full}$ is preserved (no clipping).

### 3.3 Constant storage

Two new constants in `cpp/src/swe2d_xsect_constants.h`:
```cpp
constexpr int SEMIIMPLICIT_THETA_DEFAULT = 1;       // 1.0 = first-order; 0.5 = Crank-Nicolson
constexpr double OMEGA_MIN = 0.01;                 // floor for γ·Δt in the LHS denominator (avoids 1/0 near dry cells)
```

The `surcharge_method` enum (`SURCHARGE_NONE`, `SURCHARGE_SLOT`, `SURCHARGE_EXTRAN`) is **decommissioned**: all three values continue to be accepted for API compatibility, but the implementation is the unified η-formulation regardless. A future commit may drop the enum; until then, `SURCHARGE_SLOT` becomes a no-op alias.

---

## 4. Kernel plan

### 4.1 New kernels (in `cpp/src/pipe1d.cu`)

| Kernel | Purpose | Replaces |
|---|---|---|
| `pipe1d_init_eta_from_state` | Convert existing `d_node_depth` to `d_eta` at step start; choose conservative target (open-channel: $\eta = z + h$; pressurised: $\eta$ tracks current `d_node_depth + z` if it exceeds $z + y_F$) | `swe2d_pipe1d_init_area_from_depth` |
| `pipe1d_compute_dA_deta_tables` | Build per-cell $dA/d\eta$ lookup tables from existing cross-section tables (already in `cell_tables`) | — |
| `pipe1d_assemble_tridiagonal_kernel` | Per-branch/per-junction: compute $C_i^a$, $C_i^c$, $C_i^b$, RHS $r_i^n$ | Replaces the existing flux kernel in `swe2d_pipe1d_step` |
| `pipe1d_tridiagonal_solve_kernel` | Per-branch independent solve via parallel cyclic reduction (PCR). Each branch gets a sub-batch of the kernel. Junction coupling handled by a single block-Gauss-Seidel sweep. | Replaces both stages of RK2 Godunov |
| `pipe1d_back_substitute_kernel` | Per side centre: compute $u_j^{n+1}$ from $\eta^{n+1}$ via Casulli's eq 5 form (eq 4 here) — implicit friction in `1 + γ·Δt` | Replaces Godunov update |
| `pipe1d_compute_diagnostics_kernel` | Write `d_A`, `d_Q`, `d_cell_y`, `d_cell_q`, `d_cell_fr` from staggered state for backwards compatibility with downstream kernels (BCs, weir/orifice, face-flux coupling) | Existing diagnostics paths |

### 4.2 Host wrappers

- `swe2d_pipe1d_step_v2(dev, dt, theta, omega_min, g, k_mann, h_min)` — new entry point with η-based solver. Calls:
  1. `init_eta_from_state`
  2. `compute_dA_deta_tables`
  3. `assemble_tridiagonal_kernel`
  4. `tridiagonal_solve_kernel` (junction Gauss-Seidel sweep if needed)
  5. `back_substitute_kernel`
  6. `compute_diagnostics_kernel`
  7. Existing `swe2d_pipe1d_node_mass_balance_host` for manhole node-depth updates (use η instead of cell-centered A)
  8. Existing `swe2d_drainage_pipe_end_bc_kernel_host` (read η instead of h)
  9. Existing pipe-end/weir-orifice/exchange kernel (read η instead of h, but the underlying physics is identical because $\eta - \eta_{\text{2D}} = h_{\text{2D-cell-surcharge}}$ for unpressurised ends and tracks piezometric head for pressurised ends)

The existing `swe2d_pipe1d_step` host wrapper is preserved as a thin shim that calls `swe2d_pipe1d_step_v2` with default $\theta = 1.0$. API-compatible.

### 4.3 Python bindings (`cpp/src/swe2d_bindings.cpp:1905`)

The existing `swe2d_pipe1d_step` Python binding is **kept** for backward compatibility. A new `swe2d_pipe1d_step_v2` binding exposes the θ-parameter and the unified solver.

`swe2d/runtime/coupling.py:1738` is updated to call `swe2d_pipe1d_step_v2` with $\theta = 1.0$ default. The `pipe_solver_mode` field on the configuration is deprecated and rounded to a single value ("semi_implicit"). The `surcharge_method` parameter passed at this call site is removed.

---

## 5. Validation plan

### 5.1 Hu et al. 2019 test cases (Test 1, 2, 3 of their paper)

Hu et al. is the closest reference; we should reproduce their published figures.

**Test 1 — Frictionless reservoir-to-reservoir, valve opens** (Hu et al. §3.1):
- Square pipe, 1 m² × L=400 m, $\eta_0 - \eta_L = 1$ m, valve opens at $t=0$
- Analytical: $u(x, t) = u_0 \tanh(t/t_0)$, $\eta(x, t) = \eta_L + (\eta_0 - \eta_L)(L-x)/L \cdot \cosh^{-2}(t/t_M)$
- Hu's grid: $\Delta x = 40, 20, 16, 10, 5$ m; $\Delta t = 1$ s; $\theta = 1$
- Run our solver in single-link / single-node / single-pressure-point configuration
- Assert: $|u^{\text{sim}} - u^{\text{analytical}}| / u_0 \le 5\%$ for $t \ge t_0$
- Assert: $|\eta^{\text{sim}} - \eta^{\text{analytical}}| / (\eta_0 - \eta_L) \le 5\%$ for $t \ge t_0$

**Test 2 — Steady pressurised flow through abrupt expansion / contraction** (Hu et al. §3.2):
- Two-part pipe L₁ = L₂ = 200 m, D₁ ≠ D₂. Compare Borda-Carnot analytical.
- Assert: $\eta(x, \infty)$ matches analytical to 5%.

**Test 3 — U-tube oscillation (TYPE-III mixed flow)** (Hu et al. §3.3):
- Square pipe, 1 m² × L=32 m. Three scenarios at $\theta = 0.5, \Delta t = 0.01$ s.
- Scenario 1 (pressurised): $\omega = \sqrt{2g/L}$ — assert 5% match.
- Scenario 2 (free-surface): $\omega = \pi\sqrt{gH}/L$ — assert 5% match.
- Scenario 3 (mixed): compare against Casulli & Stelling 2013 reference (fig 9) at $\epsilon = 0$ (no A-slot — our default). Note: Casulli-Stelling reference is their numerical experiment, no analytical solution exists.
- *Slot-off flag:* if Hu's Test 3 fails without the A-slot, we expose a `use_a_slot = false` (default) and `use_a_slot = true` configuration to verify Hu's $\epsilon = 0.05$ recommendation holds.

### 5.2 Casulli & Stelling 2013 test cases

Casulli's tests 1–4 cover additional validation (steady pressurised frictionless, hydraulic jump). Same validation strategy: replicate the published figures.

### 5.3 Our existing regression tests

| Test | Status | Required behaviour |
|---|---|---|
| `tests/test_swe2d_pipe1d_surcharge.py` | Full rewrite | Pressurised cell Q bounds (≤ 1500 cfs), no mass drift |
| `tests/test_pipe1d_mass_conservation.py` | Update tolerance | Mass conservation at the discrete level (≤ 1e-12 absolute drift; no F vs Q mismatch) |
| `tests/test_swe2d_pipe1d.py` | Should pass unchanged | Godunov-replacement behaviour |
| `tests/test_pipe1d_accumulation.py` | Should pass unchanged | Manhole accumulation tests |
| `tests/test_drainage_inlet_outfall_vs_swmm.py` | **Loosen tolerance** | SWMM uses Preissmann slot; our solver is slot-free. Compare trajectory envelopes within 15% (SWMM paper documents the difference between slot and no-slot simulations) |
| `tests/test_ns_manning_validation.py` | Should pass | Standalone Python friction validation — confirms μ=1.486 convention |
| `tests/test_pipe1d_slot_analytical_q.py` (new in slot plan) | Discard / fold into Hu Test 3 above | Slot-floor plan tested the slot formula; this test is now obsolete |

### 5.4 CLI replay of `test_drainage_coupling1.json`

Run for 60 s at $\Delta t = 0.5$ s, compare against current run. Expected:
- Node 2 depth trajectory within ±15% of run with current slot scheme (acceptable given known slot/no-slot differences)
- Coupled 2D cell 28157 receives water via face-flux within first 10 s
- Link 1 $Q \le 1500$ cfs
- Interior 2D cell depths bounded

---

## 6. Open decisions

These four items must be confirmed before execution. Recommended defaults are given in **bold**.

### 6.1 **θ value**

Recommended default: **θ = 1.0** (first-order, matches Casulli's test 1 default). Crank-Nicolson ($\theta = 0.5$) gives second-order accuracy but can exhibit overshoot in pressurised flows. — Decision: start with θ = 1.0, expose as configuration parameter.

### 6.2 **Staggered variable layout — full vs hybrid**

Two options:
- **A) Full staggered** (η at nodes, u at sides). Matches Casulli / Hu precisely. Requires migrating diagnostic state away from cell-centered `d_A` / `d_Q`.
- **B) Hybrid** — keep cell-centered `d_A` / `d_Q`, compute η at cell-ends as `invert + h`. Cheaper migration, but loses true Casulli-style staggering at junctions.
- Recommended: **A** (full staggered). Migration cost is incremental, and it's the only way to get the tridiagonal assembly right at junctions.

### 6.3 **Backward-compat for `d_A` / `d_Q` diagnostics**

Two options:
- **A) Keep** as diagnostic mirrors. Written from staggered state at end of each step. Required by existing test_swe2d_pipe1d.py and BC kernels that read `d_A`/`d_Q`.
- **B) Drop**. Update all consumers to read `d_side_A` / `d_u` instead.
- Recommended: **A**. The diagnostic-mirror cost is negligible and preserves the existing test suite.

### 6.4 **Slot-fallback mode for TYPE-III**

If Hu Test 3 (U-tube oscillation) fails without the A-slot, we need a fallback. The slot-floor fix in `docs/pipe1d_slot_fix_plan.md` becomes a configurable on/off mode:
- `pipe1d_use_a_slot` flag in the device state (default `false`).
- If `true`, $a(\eta) = $ `max(epsilon · W, ∂A/∂η)` where epsilon ∈ [0.05, 0.10] (Hu's recommendation).
- Implementation cost: ~30 lines, isolated module.
- Recommended: **slot-fallback** exposed, default `false`.

### 6.5 **Implicit-friction substepping**

Our 2D solver has adaptive friction substepping (`cpp/src/swe2d_gpu.cu:430-447`). Should our 1D solver get the same treatment?
- Pro: matches 2D treatment, numerical consistency, same robustness on harsh dry cells.
- Con: more complex; substepping creates mass-balance concerns.
- Recommended: **Start without** substepping (simple `(1 + γ·Δt)` denominator); add substepping only if dry-cell test cases show spurious velocity.

---

## 7. Phased implementation

The work is large. Phases let us ship each piece testably and stop at any phase.

### Phase A — Semi-implicit pressure gradient + semi-implicit friction (1-2 weeks)

**Goal:** Modifications to the existing flux kernel and Godunov update kernel to add θ-implicit pressure and (1+γ·Δt) friction denominator. Slot machinery kept for now.

**Files:**
- `cpp/src/pipe1d.cu`: modify `swe2d_pipe1d_godunov_update_kernel:1858-1870` to use:
  - Implicit friction: `Q_new = (Q_explicit + g·A·dη·dt/L) / (1 + γ·dt)`
  - Implicit pressure gradient: combine bed-slope + implicit head-difference
- `cpp/src/pipe1d.cu`: modify flux kernel `swe2d_pipe1d_flux_kernel` boundary-face pressure term to use implicit `η^{n+1}`.
- `cpp/src/swe2d_xsect_constants.h`: add `OMEGA_MIN` constant.

**Validation:** existing test suite + new test `tests/test_swe2d_pipe1d_implicit_friction.py` verifying that stable simulations at $\Delta t = 5$ s produce results matching $\Delta t = 1$ s to 5%.

### Phase B — Tridiagonal linear solver (1-2 weeks)

**Goal:** Replace RK2 Godunov with tridiagonal assembly + solve + back-substitution. Slot machinery kept but inactive.

**Files:**
- `cpp/src/pipe1d.cu`: add `pipe1d_assemble_tridiagonal_kernel`, `pipe1d_tridiagonal_solve_kernel`, `pipe1d_back_substitute_kernel`.
- `cpp/src/pipe1d.cu`: add `swe2d_pipe1d_step_v2` host wrapper.
- `cpp/src/swe2d_bindings.cpp`: add Python binding for `swe2d_pipe1d_step_v2`.
- `swe2d/runtime/coupling.py`: call `swe2d_pipe1d_step_v2` from runtime.
- `cpp/src/swe2d_xsect_constants.h`: add `SEMIIMPLICIT_THETA_DEFAULT`.

**Validation:** Hu et al. 2019 Test 1 (frictionless reservoir-to-reservoir valve) reproduced; existing tests pass.

### Phase C — Drop slot + drop A-slot, expose fallback flag (1 week)

**Goal:** Decommission `d_cell_slot_width`, `SURCHARGE_SLOT`, Sjoberg formula. Replace slot-pressure treatment with the unified η formulation.

**Files:**
- `cpp/src/pipe1d.cu`: remove `slot_width` (lines 239-245), `xsect_getAofY_pressurised_inv` slot branch (line 1889), `d_cell_slot_width` writes (line 1902).
- `cpp/src/swe2d_xsect_constants.h`: `SURCHARGE_SLOT` becomes alias for `SURCHARGE_NONE`.
- `cpp/src/pipe1d.cuh`: `d_cell_slot_width` field removed; `pipe1d_use_a_slot` flag added.

**Validation:** Hu et al. 2019 Tests 1-3 reproduced; existing tests pass with relaxed tolerances.

### Phase D — Cross-section V(η) tables + junction coupling (1-2 weeks, optional)

**Goal:** Implement full Casulli 2013 with non-linear V(η) tables for arbitrary cross-section (circular, elliptical with non-monotone $a(\eta)$). For our drainage-scale flow, only required if Phase C tests reveal deficiency.

**Files:**
- `cpp/src/pipe1d.cu`: add per-cell V(η) tables.
- `cpp/src/pipe1d.cu`: junction coupling via block-Gauss-Seidel sweep.

**Validation:** Casulli 2013 Tests 1-4 reproduced.

### Effort estimates

| Phase | Duration | Risk |
|---|---|---|
| A | 1-2 weeks | Low — additive change to existing kernels. |
| B | 1-2 weeks | Medium — tridiagonal assembly is the new critical path. |
| C | 1 week | Low — pure deletion + replacement. |
| D | 1-2 weeks | Medium-High — full non-linear V(η) integration. |

**Recommended rollout:** ship Phase A (already addresses the big friction-stability issue and the Q runaway), then evaluate against Casulli/Hu Test 1. If validated, proceed to Phase B/C. Phase D only if needed.

---

## 8. Selector-consumable step dicts

```python
# Each step is one unit of work with routing keywords for agent assignment.
steps = [
    {"action": "Add OMEGA_MIN constant + SEMIIMPLICIT_THETA_DEFAULT to cpp/src/swe2d_xsect_constants.h",
     "type": "coding", "phase": "A"},

    {"action": "Modify swe2d_pipe1d_godunov_update_kernel (cpp/src/pipe1d.cu:1858-1870) to add "
                "implicit pressure gradient and (1+γ·Δt) friction denominator",
     "type": "refactor", "phase": "A"},

    {"action": "Modify swe2d_pipe1d_flux_kernel boundary-face pressure term to be θ-implicit "
                "in cpp/src/pipe1d.cu:1375-1427",
     "type": "refactor", "phase": "A"},

    {"action": "Write tests/test_swe2d_pipe1d_implicit_friction.py validating "
                "Δt=5s simulation matches Δt=1s simulation",
     "type": "test", "phase": "A"},

    {"action": "Add pipe1d_assemble_tridiagonal_kernel + pipe1d_tridiagonal_solve_kernel + "
                "pipe1d_back_substitute_kernel + swe2d_pipe1d_step_v2 host wrapper in "
                "cpp/src/pipe1d.cu",
     "type": "coding", "phase": "B"},

    {"action": "Add Python binding for swe2d_pipe1d_step_v2 in cpp/src/swe2d_bindings.cpp:1905",
     "type": "coding", "phase": "B"},

    {"action": "Update swe2d/runtime/coupling.py:1738 to call swe2d_pipe1d_step_v2",
     "type": "refactor", "phase": "B"},

    {"action": "Implement Hu et al. 2019 Test 1 (valve-open analytical solution) regression "
                "in tests/test_pipe1d_hu_test1.py",
     "type": "test", "phase": "B"},

    {"action": "Decommission slot machinery: remove slot_width() (cpp/src/pipe1d.cu:239-245), "
                "d_cell_slot_width writes, SURCHARGE_SLOT branch "
                "(cpp/src/pipe1d.cu:1880)",
     "type": "refactor", "phase": "C"},

    {"action": "Add pipe1d_use_a_slot flag (default false) for TYPE-III fallback",
     "type": "coding", "phase": "C"},

    {"action": "Implement Hu et al. 2019 Tests 2-3 (steady pressurised, U-tube oscillation) "
                "regressions",
     "type": "test", "phase": "C"},

    {"action": "Re-run all existing pipe1d tests (test_swe2d_pipe1d, "
                "test_pipe1d_mass_conservation, test_pipe1d_accumulation, "
                "test_drainage_inlet_outfall_vs_swmm) with relaxed tolerance for SWMM regression",
     "type": "test", "phase": "C"},

    {"action": "Add per-cell V(η) cross-section tables + junction coupling via "
                "block-Gauss-Seidel",
     "type": "coding", "phase": "D"},

    {"action": "Implement Casulli 2013 Tests 1-4 regressions",
     "type": "test", "phase": "D"},

    {"action": "Update docs/PIPE1D_AUDIT_2026-07-17.md and add new doc explaining the "
                "η-based formulation",
     "type": "docs", "phase": "D"},
]
```

### Pre-computed routing

| Step | Routing keywords | Agent | Model |
|---|---|---|---|
| Phase A: add OMEGA_MIN constant | cpp, code | cpp-pro | kimi-for-coding/k3 |
| Phase A: modify godunov kernel implicit pressure+friction | cpp, refactor | cpp-pro | kimi-for-coding/k3 |
| Phase A: modify flux kernel implicit pressure | cpp, refactor | cpp-pro | kimi-for-coding/k3 |
| Phase A: write implicit friction test | python, test | test-automator | kimi-for-coding/kimi-for-coding-highspeed |
| Phase B: add tridiagonal solver kernels + host wrapper | cpp, code | cpp-pro | kimi-for-coding/k3 |
| Phase B: add Python binding | cpp, pybind11 | cpp-pro | kimi-for-coding/k3 |
| Phase B: update coupling.py | python, refactor | python-pro | kimi-for-coding/k3 |
| Phase B: Hu Test 1 regression | python, test, validate | test-automator | kimi-for-coding/kimi-for-coding-highspeed |
| Phase C: decommission slot | cpp, refactor | cpp-pro | kimi-for-coding/k3 |
| Phase C: add a_slot flag | cpp, code | cpp-pro | kimi-for-coding/k3 |
| Phase C: Hu Tests 2-3 + existing tests | python, test, validate | test-automator | kimi-for-coding/kimi-for-coding-highspeed |
| Phase D: V(η) tables + junction coupling | cpp, code, math | cpp-pro | kimi-for-coding/k3 |
| Phase D: Casulli 1-4 regressions | python, test, validate | test-automator | kimi-for-coding/kimi-for-coding-highspeed |
| Phase D: docs update | docs | general | commandcode/mimo-v2.5 |

---

## 9. Superpowers workflow

**Skills to load:**
- `superpowers:test-driven-development` — write Hu Test 1 (Phase B) and friction-implicit test (Phase A) before the corresponding kernel changes.
- `superpowers:systematic-debugging` — when tridiagonal assembly reveals wrong coefficients (symptom: $\eta$ doesn't propagate correctly), trace back through to the matrix builder.
- `superpowers:verification-before-completion` — every "Phase complete" claim runs the full test suite, not just the new tests.

**Cross-review rule:**
- Each phase must be reviewed by a different subagent than the one that implemented it. Phase A reviewer is independent of Phase A implementer.
- Phase B (tridiagonal kernel) is high-risk; mandatory second review.

---

## 10. Verification gate (post-build, post-tests)

```bash
# Cache discipline
find . -type d -name __pycache__ -exec rm -rf {} +

# All-Python validation
mamba run -n qgis_stable python3 -m unittest -v \
    tests.test_swe2d_pipe1d \
    tests.test_swe2d_pipe1d_surcharge \
    tests.test_pipe1d_mass_conservation \
    tests.test_pipe1d_accumulation \
    tests.test_swe2d_pipe1d_implicit_friction \
    tests.test_pipe1d_hu_test1 \
    tests.test_drainage_inlet_outfall_vs_swmm \
    tests.test_ns_manning_validation \
    2>&1 | tail -40
# Expected: all tests pass. SWMM regression within 15% (loosened tolerance).

# Build & run integration smoke test
cd build
cmake --build . -j$(nproc) 2>&1 | tail -5

mamba run -n qgis_stable python3 reference/example_test_project/cli_replay.py \
    reference/example_test_project/test_drainage_coupling1.json \
    --t-end 60 --dt 0.5 2>&1 | tail -30
# Expected: no NaN; node 2 ≤ 2× steady-state; link Q ≤ 1500 cfs; cell 28157 receives water.
```

---

## 11. Out of scope for this plan

- Wholesale adoption of full Casulli 2013 with nested Newton (rejected: user wants no iterative solvers).
- Quasi-steady friction relaxation toward Manning's law (rejected: friction handled correctly by the implicit `(1 + γ·Δt)` denominator).
- 2D SWE solver changes (already has semi-implicit friction and θ-implicit pressure; see `cpp/src/swe2d_gpu.cu:444`).
- Face-flux coupling modifications (`swe2d_pipe_face_flux_kernel`).
- UI / config / dropdown changes.
- Drop-in replacement of all existing test fixtures — existing tests will be updated incrementally.

---

## 12. Acknowledgements

This plan synthesises Casulli & Stelling 2013 and Hu et al. 2019 into a target architecture that fits our drainage-scale use case without requiring nested Newton iteration. The slot is dropped in favour of the implicit pressure gradient, which removes the slot's reason for existing. Friction is handled with the same `(1+γ·Δt)` form as our 2D solver, giving 1D/2D numerical consistency.

The work is sizeable (4-6 weeks total for Phases A-D). Phase A alone (1-2 weeks) addresses the immediate Q-runaway and friction-stability issues. Phases B-D are progressive refactors toward the Casulli/Hu target. We can stop at any phase and ship a working improved model.
