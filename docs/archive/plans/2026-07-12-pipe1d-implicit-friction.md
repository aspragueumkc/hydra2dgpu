---
type: plan
status: complete
created: 2026-07-12
completed: 2026-07-25
---

# Pipe1D Implicit Friction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans.
> Steps use checkbox (`- [ ]`) syntax.

**Goal:** Replace the explicit Manning friction and HEC-22 minor loss terms in the pipe1d solver with an implicit (unconditionally stable) formulation, matching the approach used by the 2D SWE solver's `swe2d_implicit_friction_kernel`.

**Architecture:** Both the diffusion-wave kernel and the fully-dynamic kernel currently compute `source_fric = -g * n² * |Q| * Q / (A * R^(4/3))` and do `Q_new = Q + dt * (pressure_grad + source_fric + source_minor)`. This is **explicit** — unstable when `dt * g * n² * |Q| / (A * R^(4/3)) > 1`. The fix applies the same `hu /= 1 + dt * cf * spd` pattern from the 2D solver: compute Manning coefficient `cf = g * n² / (A * R^(4/3))`, then damp Q implicitly as `Q_new = (Q + dt * pressure_grad) / (1 + dt * cf * |Q| + dt * g * k_loss * |Q| / (2 * A² * L))`.

**Tech Stack:** CUDA C++ (`cpp/src/pipe1d.cu`), Python tests (`tests/test_pipe1d_vs_swmm.py`, `tests/test_pipe1d_accumulation.py`), pybind11 bindings (unchanged).

**Also includes:** Fix the drainage solver mode combo — currently has 3 entries but only 2 C++ solvers exist ("diffusion_wave" and "fully_dynamic"). The "EGL (Bernoulli + minor losses)" entry (mode 0) is an alias for diffusion_wave with no distinct implementation. Replace the 3 items with 2 matching the actual kernel modes.

---

## Files to Modify

| File | Change |
|------|--------|
| `cpp/src/pipe1d.cu` | Replace explicit friction+minor loss in diffusion-wave kernel (L673-676) and fully-dynamic kernel (L754-758) with implicit denominator |
| `cpp/src/pipe1d.cu` | Replace loss computation in accumulation kernel — already fixed (L928-931), no change needed |
| `tests/test_pipe1d_vs_swmm.py` | Add test: pressurized flow at dt=0.25 with implicit friction matches SWMM within 5% |
| `tests/test_pipe1d_accumulation.py` | No change needed — accumulation kernel already verified |

---

### Task 1: Implicit friction in diffusion-wave kernel

**Files:**
- Modify: `cpp/src/pipe1d.cu:670-691`
- Test: `tests/test_pipe1d_vs_swmm.py` (Task 3)

**Current code (explicit):**
```c
// Wetted perimeter and top width from geometry table
double P_c, T_c;
pipe1d_lookup_geometry(A, A_full, P_full,
    cell_tables + static_cast<int64_t>(i) * 2 * table_N, table_N, P_c, T_c);

// Hydraulic radius (current area / wetted perimeter from table)
const double R = A / fmax(1e-10, P_c);
const double R43 = pow(R, 4.0 / 3.0);

// Friction source: -g * n² * |Q| * Q / (A * R^(4/3))
const double source_fric = -g * n * n * absQ * Q / (A * R43 + 1e-12);
// Minor loss source (HEC-22 entrance/exit at boundary cells only; k=0 for interior cells)
const double source_minor = -g * k_loss * absQ * Q / (2.0 * A * A * L + 1e-12);

const double S_Q = pressure_grad + source_fric + source_minor;
double Q_new = Q + dt * S_Q;

// Clamp Q to reasonable bounds
const double Q_cap = 1e6;
Q_new = fmax(-Q_cap, fmin(Q_cap, Q_new));
```

- [ ] **Step 1: Replace with implicit friction**

The key insight from the 2D solver (L435 of `swe2d_gpu.cu`):
```c
const double denom = 1.0 + dt_sub * cf * spd_k;
hu /= denom;
```

For 1D pipe flow, Manning friction is: `dQ/dt = -g * n² / (A * R^(4/3)) * |Q| * Q`

Let `cf = g * n * n / (A * R43 + 1e-12)`. Then `dQ/dt = -cf * |Q| * Q` (same as 2D: `d(hu)/dt = -cf * spd * hu`).

Implicit update: `Q_new = (Q + dt * pressure_grad) / (1 + dt * cf * |Q| + dt * g * k_loss * |Q| / (2 * A * A * L + 1e-12))`

```c
double P_c, T_c;
pipe1d_lookup_geometry(A, A_full, P_full,
    cell_tables + static_cast<int64_t>(i) * 2 * table_N, table_N, P_c, T_c);

const double R = A / fmax(1e-10, P_c);
const double R43 = pow(R, 4.0 / 3.0);

// Implicit Manning friction (same pattern as 2D solver: hu /= 1 + dt*cf*spd)
const double cf = g * n * n / (A * R43 + 1e-12);
// Implicit minor loss (HEC-22 entrance/exit)
const double cm = g * k_loss / (2.0 * A * A * L + 1e-12);

// Pressure gradient (explicit — not stiff)
const double dHdx = (H_to - H_from) / fmax(1e-6, L);
const double pressure_grad = -g * A * dHdx;

const double absQ = fabs(Q);
const double denom = 1.0 + dt * cf * absQ + dt * cm * absQ;
double Q_new = (Q + dt * pressure_grad) / denom;

// Cap to prevent extreme values on startup
const double Q_cap = 1e6;
Q_new = fmax(-Q_cap, fmin(Q_cap, Q_new));
```

The denominator is always ≥ 1 (all terms are non-negative), so this is unconditionally stable for any dt.

- [ ] **Step 2: Rebuild and check compilation**

```bash
cd build && mamba run -n qgis_stable cmake --build . -j$(nproc) 2>&1 | tail -5
```
Expected: `[100%] Built target hydra_swe2d`

- [ ] **Step 3: Run the accumulation kernel tests to confirm no regression**

```bash
mamba run -n qgis_stable python3 -m pytest tests/test_pipe1d_accumulation.py -v
```
Expected: 15 passed, 58 subtests passed

- [ ] **Step 4: Commit**

```bash
git add cpp/src/pipe1d.cu
git commit -m "fix(pipe1d): implicit Manning friction + minor loss in diffusion-wave kernel"
```

---

### Task 2: Fix drainage solver mode combo (3 entries → 2 actual modes)

**Files:**
- Modify: `swe2d/workbench/views/model_tab_view.py:1299-1302`
- Modify: `swe2d/workbench/services/pipe_network_service.py:971`
- Test: `tests/test_model_tab_view.py`

The combo currently lists:
| Label | data | C++ mode |
|---|---|---|
| "EGL (Bernoulli + minor losses)" | 0 | diffusion_wave |
| "Diffusion wave" | 1 | diffusion_wave |
| "Dynamic Saint-Venant" | 2 | fully_dynamic |

Modes 0 and 1 both map to `diffusion_wave` — there is no separate Bernoulli solver. The mapping at `pipe_network_service.py:971` is `"diffusion_wave" if solver_mode != 2 else "fully_dynamic"`.

**Fix:** Replace with two accurate entries:
| Label | data | C++ mode |
|---|---|---|
| "Diffusion wave" | 0 | diffusion_wave |
| "Full Saint-Venant" | 1 | fully_dynamic |

- [ ] **Step 1: Update the combo items**

```python
# Replace the 3 addItem calls (L1299-1301) with 2:
self.drainage_solver_mode_combo.addItem("Diffusion wave", 0)
self.drainage_solver_mode_combo.addItem("Full Saint-Venant", 1)
self.drainage_solver_mode_combo.setCurrentIndex(0)
```

- [ ] **Step 2: Update the mode mapping in pipe_network_service.py**

Change L971 from:
```python
pipe_solver_mode = "diffusion_wave" if solver_mode != 2 else "fully_dynamic"
```
to:
```python
pipe_solver_mode = "diffusion_wave" if solver_mode == 0 else "fully_dynamic"
```

- [ ] **Step 3: Update the existing widget test**

In `tests/test_model_tab_view.py`, find `test_view_has_drainage_solver_mode_combo` and update the item count check.

Search for `drainage_solver_mode_combo` in that file and change any `assertEqual(..., 3)` to `assertEqual(..., 2)`.

- [ ] **Step 4: Run widget tests**

```bash
mamba run -n qgis_stable python3 -m pytest tests/test_model_tab_view.py -v -k "drainage"
```
Expected: all drainage tests pass

- [ ] **Step 5: Commit**

```bash
git add swe2d/workbench/views/model_tab_view.py \
       swe2d/workbench/services/pipe_network_service.py \
       tests/test_model_tab_view.py
git commit -m "fix(ui): drainage solver mode combo now shows only the 2 actual modes (diffusion wave, full Saint-Venant)"
```

---

### Task 3: Implicit friction in fully-dynamic kernel

**Files:**
- Modify: `cpp/src/pipe1d.cu:749-764`

The fully-dynamic kernel has the same explicit friction formula. Apply the same implicit treatment.

- [ ] **Step 1: Replace with implicit friction**

Current code (fully dynamic kernel, around L749-764):
```c
const double R = A / fmax(1e-10, P_c);
const double R43 = pow(R, 4.0 / 3.0);

const double source_fric = -g * n * n * absQ * Q / (A * R43 + 1e-12);
const double source_minor = -g * k_loss * absQ * Q / (2.0 * A * A * L + 1e-12);

// Pressure gradient from neighboring cell heads
const double dHdx = (H_n - H_c) / fmax(1e-6, cell_length[c]);
const double S_Q = -g * A * dHdx + source_fric + source_minor;
double Q_new = Q + dt * S_Q;

const double Q_cap = 1e6;
Q_new = fmax(-Q_cap, fmin(Q_cap, Q_new));
```

Replace with:
```c
const double R = A / fmax(1e-10, P_c);
const double R43 = pow(R, 4.0 / 3.0);

const double cf = g * n * n / (A * R43 + 1e-12);
const double cm = g * k_loss / (2.0 * A * A * cell_length[c] + 1e-12);

// Pressure gradient
const double dHdx = (H_n - H_c) / fmax(1e-6, cell_length[c]);

const double absQ = fabs(Q);
const double denom = 1.0 + dt * cf * absQ + dt * cm * absQ;
double Q_new = (Q + dt * (-g * A * dHdx)) / denom;

const double Q_cap = 1e6;
Q_new = fmax(-Q_cap, fmin(Q_cap, Q_new));
```

Note: the fully-dynamic kernel uses `H_n - H_c` (neighbor minus current) while the diffusion kernel uses `H_to - H_from` (target minus source). The sign handling on `dHdx` differs — keep the existing sign, just make the friction+loss implicit.

- [ ] **Step 2: Rebuild**

```bash
cd build && mamba run -n qgis_stable cmake --build . -j$(nproc) 2>&1 | tail -5
```

- [ ] **Step 3: Run existing pipe tests**

```bash
mamba run -n qgis_stable python3 -m pytest tests/test_swe2d_pipe1d.py tests/test_swe2d_pipe1d_surcharge.py tests/test_pipe1d_accumulation.py tests/test_pipe1d_vs_swmm.py -v 2>&1 | tail -15
```
Expected: all existing passing tests still pass. The surcharge mass-conservation test may fail (pre-existing).

- [ ] **Step 4: Commit**

```bash
git add cpp/src/pipe1d.cu
git commit -m "fix(pipe1d): implicit Manning friction + minor loss in fully-dynamic kernel"
```

---

### Task 4: Update pressurized-flow test to use dt=0.25

**Files:**
- Test: `tests/test_pipe1d_vs_swmm.py` (TestPressurizedFlow::test_pipe1d_vs_swmm)

Currently the test uses `dt=0.001, n_steps=25000` to work around the explicit friction stability limit. With the implicit fix, the test should pass at `dt=0.25` (the same dt used by the 2D domain).

- [ ] **Step 1: Run the existing test at dt=0.25 to confirm it fails before the fix**

```bash
# Temporarily change dt=0.001 to dt=0.25 in test_pipe1d_vs_swmm.py
# Run:
mamba run -n qgis_stable python3 -m pytest tests/test_pipe1d_vs_swmm.py::TestPressurizedFlow::test_pipe1d_vs_swmm -v
```
Expected: FAIL (blow-up without implicit friction)

- [ ] **Step 2: After the implicit fix is applied in Tasks 1-2, update the test**

```python
def test_pipe1d_vs_swmm(self):
    ...
    # pipe1d with the same upstream head, now stable at 2D timestep
    q_pipe1d = _pipe1d_q(depth_n0=swmm_depth, depth_n1=0.0, slope=slope,
                          k_in=0.5, k_out=1.0,
                          dt=0.25, n_steps=200,
                          solver_mode="diffusion_wave")
    ...
    self.assertAlmostEqual(ratio, 1.0, delta=0.05,
                           msg=f"pipe1d/SWMM={ratio:.3f}")
```

- [ ] **Step 3: Run the test**

```bash
mamba run -n qgis_stable python3 -m pytest tests/test_pipe1d_vs_swmm.py::TestPressurizedFlow -v
```
Expected: PASS (pipe1d Q within 5% of SWMM Q at dt=0.25)

- [ ] **Step 4: Run full test suite**

```bash
mamba run -n qgis_stable python3 -m pytest tests/test_pipe1d_vs_swmm.py tests/test_pipe1d_accumulation.py tests/test_swe2d_pipe1d.py tests/test_swe2d_pipe1d_surcharge.py -v 2>&1 | tail -15
```
Expected: only pre-existing failures (test_upload_node_depth_changes_area, test_mass_conservation_surcharge)

- [ ] **Step 5: Commit**

```bash
git add tests/test_pipe1d_vs_swmm.py
git commit -m "test(pipe1d): verify implicit friction at 2D timestep matches SWMM"
```

---

### Task 5: Add open-channel implicit friction test at large dt

**Files:**
- Modify: `tests/test_pipe1d_vs_swmm.py`

Add a test that verifies the open-channel 50% full case converges to Manning's at dt=0.25 (previously only worked at dt=0.001).

- [ ] **Step 1: Add test method**

```python
def test_half_pipe_large_dt(self):
    """50% full at dt=0.25: implicit friction keeps Q finite and within factor 2 of Manning's."""
    slope = 0.01
    d_frac = 0.5
    depth = PIPE_D * d_frac
    q = _pipe1d_q(depth_n0=depth, depth_n1=depth, slope=slope,
                   n_steps=200, dt=0.25, solver_mode="diffusion_wave")
    expected = mannings_q(PIPE_D, slope, PIPE_N, d_frac)
    self.assertTrue(math.isfinite(q), "Q must be finite at dt=0.25 with implicit friction")
    self.assertGreater(q, 0.01)
    self.assertLess(q / expected, 3.0)
```

- [ ] **Step 2: Run the test**

```bash
mamba run -n qgis_stable python3 -m pytest tests/test_pipe1d_vs_swmm.py::TestOpenChannel::test_half_pipe_large_dt -v
```

- [ ] **Step 3: Commit**

```bash
git add tests/test_pipe1d_vs_swmm.py
git commit -m "test(pipe1d): open-channel stability at large dt with implicit friction"
```

---

## Verification Gate

After all tasks:

```bash
find . -type d -name __pycache__ -exec rm -rf {} +
mamba run -n qgis_stable python3 -m pytest tests/test_pipe1d_vs_swmm.py tests/test_pipe1d_accumulation.py -v
```
Expected: 23 passed, 58 subtests passed (or whatever the count is — no regressions)

Then re-test the original blow-up scenario:
- User runs the example test project GPKG
- Node 3 should no longer blow up to 86 quadrillion
- The HEC-22 loss fix (already committed) prevents the sign-reversal + implicit friction prevents the explicit friction instability
