---
type: plan
status: complete
created: 2026-07-17
completed: 2026-07-25
---

# Pipe1D Phase A — Implicit Pressure + Semi-Implicit Friction

**Date:** 2026-07-17
**Status:** Proposed (Phase A only)
**Parent plan:** `docs/pipe1d_casulli_hu_plan.md` (Phases B/C/D deferred)
**Goal:** Add **semi-implicit θ-method pressure gradient** (Casulli 2013 / Hu et al. 2019 contribution) and **semi-implicit friction in the `(1 + γ·Δt)` denominator** (matching our 2D solver at `cpp/src/swe2d_gpu.cu:444`) to the existing RK2 Godunov solver. **No slot changes**, **no architectural rewrite** — just two targeted modifications to two existing kernels.

This addresses two issues simultaneously:
1. **Slot-induced Q runaway**: the slot's pressure term is constrained by making the pressure gradient **implicit**, removing the slot's reason for existing in this kernel.
2. **Friction-stability mismatch with 2D**: 1D currently uses explicit friction; matched to 2D's treatment.

After Phase A lands, we evaluate against validation §5 before deciding whether to proceed to Phase B (tridiagonal linear solve).

**Tech Stack:** CUDA C++ (`cpp/src/pipe1d.cu`), Python pytest, mamba env `qgis_stable`.

---

## 1. What changes

### 1.1 Semi-implicit pressure gradient in the Godunov update

**File:** `cpp/src/pipe1d.cu`
**Kernel:** `swe2d_pipe1d_godunov_update_kernel:1765-1913`
**Lines to modify:** 1856-1870 (the `source_Q` term)

Currently the momentum equation in the kernel has the form:

```
source_Q = flux_mom_div + g·A_eff·S0_cell − g·A_eff·Sf
```

where:
- `flux_mom_div` is the explicit HLLE momentum flux divergence (from the flux kernel).
- `g·A_eff·S0_cell` is the bed-slope source (explicit, evaluated at time n).
- `g·A_eff·Sf` is the explicit friction (evaluated at time n).

**Casulli-style semi-implicit form:**

```
# Pressure gradient contribution (now implicit in η^{n+1})
# Casulli's eq 5: u^{n+1} = (F[u^n] − g·dη^{n+θ}/dx) / (1 + γ·Δt)
# For our cell-centered momentum, the implicit part goes into the LHS.

γ = g · n² · |Q_curr| / (k_mann² · A_eff² · R^(4/3))   # friction coefficient
P_implicit = g · A_eff · (η_{r(j)}^{n+θ} − η_{ℓ(j)}^{n+θ}) / Δx
              = g · A_eff · θ · (η_r^{n+1} − η_ℓ^{n+1}) / Δx
              + g · A_eff · (1 − θ) · (η_r^n − η_ℓ^n) / Δx

P_explicit_bed = g · A_eff · S0_cell   # bed-slope is always explicit (geometry)

Q_new = (Q_curr + Δt · (flux_mom_div + P_explicit_explicit + P_implicit_explicit_part))
        / (1 + γ · Δt)
```

Where:
- $\eta_{r(j)}^{n+1}, \eta_{\ell(j)}^{n+1}$ are piezometric heads at the cell's two endpoints (right/left nodes). For open-channel these are `node_invert + node_depth`; for pressurised they exceed `invert + yFull` (no slot, no clipping).
- $\theta \in [0.5, 1.0]$ is the implicit factor (default 1.0 for first-order accuracy).
- The implicit part `$g·A·θ·(η_r^{n+1} − η_ℓ^{n+1})/Δx$` would normally require solving a system across cells, but for our existing RK2 cell-centered scheme we make the simplifying assumption that the implicit part uses the cell's own `d_eta_curr` (a local approximation). The full tridiagonal solve is deferred to **Phase B**.
- The `source_Q` term becomes two separate pieces: an explicit body force + an implicit pressure gradient that goes into the `(1 + γ·Δt)` denominator.

### 1.2 Get η at cell endpoints

We need $\eta$ at the two endpoints of each cell. The existing storage has:
- `d_node_invert[from_node], d_node_invert[to_node]` (line 59)
- `d_node_depth[from_node], d_node_depth[to_node]` (line 60)

So $\eta = z + h$ per endpoint is read directly. No new storage needed for Phase A.

For pressurised cells, `d_node_depth` is currently clipped at crown elevation per `swe2d_pipe1d_update_node_depth_kernel`. For Phase A we **don't change this clipping** (slot machinery preserved); the implicit pressure gradient fix alone should make the Q runaway bounded.

### 1.3 Semi-implicit friction in the (1+γ·Δt) denominator

The friction coefficient $\gamma = g \cdot n^2 \cdot |u| / R^{4/3}$ appears in the LHS denominator for the implicit solve, exactly as in Casulli's eq 5 and our 2D solver's `apply_friction_cuda_local` (`cpp/src/swe2d_gpu.cu:404-448`).

For our cell-centered variables:
```
γ_cell = g · n_val² · |Q_curr| / (k_mann² · A_eff² · R_(4/3))
```

where `Sf = γ_cell / g` matches our existing `Sf = n²·Q²/(k²·A²·R^(4/3))` computation (line 1843-1846).

So:
```
explicit_force = flux_mom_div + g·A_eff·S0_cell − g·A_eff·Sf·sign(Q_curr) − g·A_eff·(1−θ)·dη_n/dx
implicit_factor = 1 + γ_cell · Δt + g·A_eff·θ·dη_cell/dx_estimate  (≈ 1 + γ·Δt for Phase A; θ=1 simplifies)
Q_new = (Q_curr + Δt · explicit_force) / implicit_factor
```

For θ=1.0 (default), the implicit pressure term contributes to the denominator; for θ<1, only the friction denominator implicitly treats the pressure explicitly.

### 1.4 OMEGA_MIN floor

When $A_{eff}$ is tiny or $|Q_{curr}|$ is near zero, $\gamma \cdot \Delta t$ could become degenerate. Mirror the 2D solver's guard at `cpp/src/swe2d_gpu.cu:416` (`h_fric = max(h, 4*h_min)`):

```cpp
const double gamma_floor = OMEGA_MIN;  // avoids 1/(1 + 0) when Q=0
const double gamma = (Q_curr != 0.0 && A_eff > A_floor)
    ? g * n_val*n_val * fabs(Q_curr) / (k_mann²·A_eff²·R^(4/3))
    : gamma_floor;
const double denom = 1.0 + gamma * dt;
```

The `OMEGA_MIN` constant (e.g., `1e-6`) lives in `cpp/src/swe2d_xsect_constants.h` (next to existing `SURCHARGE_*` constants).

### 1.5 What does NOT change

- **Slot machinery** (`SURCHARGE_SLOT` branch, `slot_width()`, `d_cell_slot_width`, `xsect_getAofY_pressurised_inv`): unchanged. Phase A leaves the slot in place; the implicit pressure + friction are independent improvements.
- **RK2 / Godunov time-integration**: unchanged. The implicit friction is layered on top of the existing RK2 stages.
- **`swe2d_pipe1d_step` signature**: unchanged. New parameters (theta) are backend constants, not user-facing yet.
- **Python bindings**: unchanged.
- **All existing tests** (`test_swe2d_pipe1d`, `test_pipe1d_mass_conservation`, `test_pipe1d_accumulation`, `test_drainage_inlet_outfall_vs_swmm`, `test_ns_manning_validation`): should pass unchanged.

### 1.6 Why no slot change

The user-confirmed rationale (parent plan §2.6):
- Drainage-scale flows have slow regime transitions; slot pathologies are rare in our test scenarios.
- Implicit pressure gradient bounds the slot's instability regardless of slot width.
- Removing the slot entirely is Phase B (tridiagonal reformulation), not Phase A.

---

## 2. File-by-file changes

| File | Lines | Change |
|------|-------|--------|
| `cpp/src/swe2d_xsect_constants.h` | after line 13 | Add `constexpr double OMEGA_MIN = 1e-6;` |
| `cpp/src/pipe1d.cu` | 1765 (kernel signature) | Add `theta` and `omega_min` parameters to kernel signature; default to 1.0 and `OMEGA_MIN` respectively |
| `cpp/src/pipe1d.cu` | 1856-1858 | Replace `source_Q` line with explicit + implicit decomposition |
| `cpp/src/pipe1d.cu` | 1860-1870 | Replace `Q_next = Q_curr + dt * source_Q` with `Q_new = (Q_curr + dt * explicit_force) / (1 + γ·dt + θ·dη_implicit_estimate)` |
| `cpp/src/pipe1d.cu` | 1896-1899 (cell-centered A initialization for Q_new=clamp test) | Adjust clamp bounds to account for new denominator |
| `cpp/src/pipe1d.cu` | 1916 (godunov_step_internal host wrapper) | Forward `theta` and `omega_min` parameters |
| `cpp/src/pipe1d.cu` | 2855 (swe2d_pipe1d_step host wrapper) | Add `theta` and `omega_min` parameters with defaults |
| `cpp/src/pipe1d_bindings.cpp:1905` | — | (no change in Phase A; parameter additions deferred to Phase B) |

Total: ~30 lines of kernel logic + 1 line of constants + parameter forwarding through 2 host wrappers. Roughly 50-80 lines of code change.

---

## 3. The new momentum form (code-level)

```cpp
// In swe2d_pipe1d_godunov_update_kernel, REPLACE lines 1843-1870:

const double absQ = fabs(Q_curr);
const double Sf = (R43 > 0.0 && A_eff > 0.0)
    ? (n_val * n_val) * absQ * Q_curr / (k_mann * k_mann * A_eff * A_eff * R43 + 1.0e-12)
    : 0.0;

// IMPLEMENTATION 1: explicit bed slope (always)
const double S0_cell = (cell_S0 != nullptr) ? cell_S0[c] : 0.0;
const double src_gravity = g * A_eff * S0_cell;

// IMPLEMENTATION 2: HLLE momentum flux divergence (explicit from flux kernel)
const double flux_mom_div = -flux_mom[c] / L;

// IMPLEMENTATION 3: piezometric head gradient at the cell
//   Cell endpoints are the from_node and to_node.
//   η_end = z_end + h_end (interpreted as absolute WSE; for pressurized, may exceed crown).
double eta_left = 0.0, eta_right = 0.0;
if (from_node >= 0) {
    eta_left = (from_node >= n_nodes)
        ? vnode_H[from_node - n_nodes]
        : node_invert[from_node] + node_depth[from_node];
}
if (to_node >= 0) {
    eta_right = (to_node >= n_nodes)
        ? vnode_H[to_node - n_nodes]
        : node_invert[to_node] + node_depth[to_node];
}
const double d_eta_n = (eta_right - eta_left) / L;       // explicit part (time level n)
const double d_eta_implicit = theta * d_eta_n;             // implicit part (time level n+1)

// IMPLEMENTATION 4: friction coefficient γ (Casulli form, ms⁻¹)
const double gamma = (R43 > 0.0 && A_eff > 0.0 && absQ > 1e-9)
    ? g * n_val * n_val * absQ / (k_mann * k_mann * A_eff * A_eff * R * R43)
    : omega_min;

// IMPLEMENTATION 5: assemble explicit force and implicit denominator
const double explicit_force = flux_mom_div
                            + src_gravity
                            - g * A_eff * Sf
                            - g * A_eff * (1.0 - theta) * d_eta_n;
const double denom = 1.0 + gamma * dt + g * A_eff * theta * d_eta_implicit / max(gamma * dt, 1e-12);

// IMPLEMENTATION 6: integrate Q
double Q_next = (Q_curr + dt * explicit_force) / denom;

// CFL limiter (unchanged)
const double Q_cfl = A_eff * L / max(dt, 1.0e-12);
if (Q_next >  Q_cfl) Q_next =  Q_cfl;
if (Q_next < -Q_cfl) Q_next = -Q_cfl;

// (Continuity, slot clamp, h_new, y_new, q_new, slot_w diagnostics unchanged)
```

**Note on Phase A approximation:** `d_eta_implicit` uses η at time n as a placeholder for η at n+1. This is *exactly* what makes it not-truly-implicit in Phase A — we treat the cell's own endpoints as if their piezometric head doesn't change in this timestep. Phase B replaces this approximation with the tridiagonal solve.

For θ=1.0 default, this approximation has bounded error: the local η gradient adjusts in O(1) steps as upstream/downstream head changes propagate. Comparison against the full tridiagonal solve (Phase B) quantifies the error.

---

## 4. Validation

### 4.1 New test (writes BEFORE the kernel change, per TDD): `tests/test_swe2d_pipe1d_implicit_friction.py`

```python
"""Validate implicit friction in 1D pipe solver matches 2D solver's treatment
and produces stable results at large timesteps (Δt ≥ 5 s) where the explicit
friction diverges."""
```

Test cases:
1. **Friction stability test**: build a 3-cell link with S0 = 2%, run for 100 simulated seconds at Δt = 0.5 s (explicit) and Δt = 5 s (implicit). Assert Q values agree within 5% and neither produces NaN.
2. **θ-parameter sensitivity test**: run at θ=1.0 and θ=0.5, compare trajectories. (θ=0.5 should give a different but bounded trajectory; no NaN.)
3. **Bounded Q under slot surcharge test**: the existing Q runaway symptom was 71 819 cfs in pipe-end cells. After the implicit pressure fix, assert Q ≤ 1500 cfs (the analytical Darcy-Weisbach upper bound for box 10×5 at 2%).
4. **Mass conservation test**: existing test_pipe1d_mass_conservation passes without modification.

### 4.2 Existing tests must all pass

```bash
find . -type d -name __pycache__ -exec rm -rf {} +
mamba run -n qgis_stable python3 -m unittest -v \
    tests.test_swe2d_pipe1d \
    tests.test_swe2d_pipe1d_surcharge \
    tests.test_pipe1d_mass_conservation \
    tests.test_pipe1d_accumulation \
    tests.test_drainage_inlet_outfall_vs_swmm \
    tests.test_ns_manning_validation \
    2>&1 | tail -40
```

Expected: all 6 pass. The Q values in `test_swe2d_pipe1d_surcharge` may shift from buggy 70 000+ cfs down to plausible ≤ 1500 cfs.

### 4.3 CLI replay smoke test

```bash
mamba run -n qgis_stable python3 reference/example_test_project/cli_replay.py \
    reference/example_test_project/test_drainage_coupling1.json \
    --t-end 60 --dt 0.5 \
    2>&1 | tail -30
```

Expected:
- No NaN values in pipe-end Q
- Node 2 depth ≤ 2× its current value (the slot-floor fix in `docs/pipe1d_slot_fix_plan.md` would also bound this; the implicit pressure fix is independent)
- Link 1 Q ≤ 1500 cfs
- 2D cell 28157 receives water via face-flux within first 10 s

### 4.4 Diagnostic: comparison of full vs approximate implicit pressure

After Phase A lands, add a comparison metric: run a small synthetic problem (1 link, 6 cells) with θ=1.0 and compare against an existing run. The Phase A approximation error should be ≤ 5% in steady-state Q.

---

## 5. Step-by-step implementation

### Step 1: Add the constant

In `cpp/src/swe2d_xsect_constants.h` after line 13:

```cpp
constexpr double OMEGA_MIN = 1e-6;  // Floor for γ in 1+γ·Δt denominator (1D friction)
```

Build:
```bash
cd build && cmake --build . -j$(nproc) 2>&1 | tail -5
```

Verify: clean compile.

### Step 2: Write the failing test

Create `tests/test_swe2d_pipe1d_implicit_friction.py` with the four test cases in §4.1. Run them and verify they fail (or skip with a feature flag).

### Step 3: Modify the kernel

Apply the changes in §3 to `cpp/src/pipe1d.cu:1843-1870`. Replace the 3-line `Sf` computation with the 4-piece decomposition (`src_gravity`, `flux_mom_div`, `d_eta_n`, `gamma`).

### Step 4: Modify host wrappers

In `cpp/src/pipe1d.cu:1916` (`swe2d_pipe1d_godunov_step_internal`) and `cpp/src/pipe1d.cu:2855` (`swe2d_pipe1d_step`):
- Add `double theta = 1.0` parameter (or use a constexpr default at top of file)
- Forward to the kernel call

### Step 5: Build and run failing test

```bash
cd build && cmake --build . -j$(nproc) 2>&1 | tail -5
mamba run -n qgis_stable python3 -m unittest -v tests.test_swe2d_pipe1d_implicit_friction 2>&1 | tail -30
```

Expected: the friction stability test passes; the surcharge Q-bound test passes (Q bounded by analysis); other tests still pass.

### Step 6: Run full regression

(See §4.2.)

### Step 7: CLI replay smoke test

(See §4.3.)

### Step 8: Commit

Single commit:
```
git add cpp/src/pipe1d.cu cpp/src/swe2d_xsect_constants.h tests/test_swe2d_pipe1d_implicit_friction.py
git commit -m "feat(pipe1d): implicit pressure gradient + semi-implicit friction (Phase A)"
```

---

## 6. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Implicit pressure approximation (treating η^{n+1} = η^n) produces steady-state error > 10% | Low-Medium | Medium | Compare against full Phase B tridiagonal once available; tune θ if needed |
| Q cap (cell CFL) interferes with new denominator form | Low | Low | Tested in §4.1 test case 1 |
| Dry-cell guard (ω_min) produces visible drift in dry-bed test | Low | Low | Comparable to 2D solver's behaviour; verified in §4.2 |
| Existing mass conservation test fails due to asymmetric treatment of θ | Medium | High | If violated, raise θ (use 1.0 exclusively) or adjust half-step weighting |

---

## 7. Out of scope

- **Phase B (tridiagonal linear solver)** — separately planned in `pipe1d_casulli_hu_plan.md` §Phase B. Deferred until Phase A is validated.
- **Phase C (drop slot)** — same.
- **Phase D (full V(η) tables + junction coupling)** — same.
- **Slot-floor fix** in `docs/pipe1d_slot_fix_plan.md` — kept as documentation; superseded by Phase A.
- **Python binding changes** — deferred to Phase B.
- **UI / config / dropdown changes** — none needed for backend-only Phase A.

---

## 8. Estimated effort

| Step | Duration | Notes |
|---|---|---|
| 1. Add constant | 5 min | Trivial |
| 2. Failing test | 1-2 h | Standard pytest scaffolding |
| 3. Kernel modification | 2-4 h | Core logic; requires understanding existing kernel structure |
| 4. Host wrapper changes | 30 min | Trivial parameter forwarding |
| 5. Build and iterate | 1-2 h | Compile-fail iterations + GPU debug |
| 6. Full regression | 30 min | Mostly automatic |
| 7. CLI replay | 1-2 h | May require script plumbing |
| 8. Commit | 5 min | |

**Total:** ~1-2 days, possibly 3-4 days if Iteration 5 uncovers surprises.

---

## 9. Stop condition for Phase A

Phase A is "complete" when ALL of:
1. All 6 existing pipe1d tests pass with the new kernel.
2. The 4 new tests in `test_swe2d_pipe1d_implicit_friction.py` pass.
3. The CLI replay produces sensible node 2 depth and link Q values.
4. Comparison with the current run shows the Q runaway is bounded.

If any of these fail, iterate on Phase A before committing. Do NOT proceed to Phase B without Phase A in good shape.

---

## 10. References

- Casulli, V. & Stelling, G. S. (2013). "A semi-implicit numerical model for urban drainage systems." *Int. J. Numer. Meth. Fluids* 73:600–614. doi:10.1002/fld.3817.
- Hu, D., Li, S., Yao, S., Jin, Z. (2019). "A Simple and Unified Linear Solver for Free-Surface and Pressurized Mixed Flows in Hydraulic Systems." *Water* 11(10):1979. doi:10.3390/w11101979.
- Existing 2D friction impl: `cpp/src/swe2d_gpu.cu:404-448` (`apply_friction_cuda_local`).
- Parent plan: `docs/pipe1d_casulli_hu_plan.md`.
- Predecessor plan (slot-floor): `docs/pipe1d_slot_fix_plan.md` (superseded for runtime use; kept for documentation).
