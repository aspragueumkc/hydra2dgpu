---
type: plan
status: complete
created: 2026-07-17
completed: 2026-07-25
---

# Pipe1D Slot Surcharge Stabilization — Implementation Plan

**Date:** 2026-07-17
**Status:** Proposed (post-user-correction)
**Author:** opencode (writing-plans skill)

**Goal:** Stabilize the existing Preissmann slot surcharge scheme so it produces physically reasonable Q values for pressurized pipe flow, without changing the algorithm (slot remains the production surcharge mode, matching EPA SWMM and InfoWorks ICM). No new `surcharge_method`, no quasi-steady relaxation, no replacement of the Saint-Venant PDE.

**Architecture:** Three targeted numerical fixes to the existing slot kernel:

1. **A-bounded slot growth** in the continuity update — currently `SURCHARGE_SLOT` clamps only `A_floor` (lower bound), allowing `A_next` to grow unboundedly above `A_full`. Add an upper cap `A_max = A_full + slot_max_pay` where `slot_max_pay` is a small headroom (e.g., 0.5·yFull or 0.2·A_full, configurable per call). The slot retains its purpose (small transient storage above the crown) but cannot become pathological.
2. **Slot width floor** — the Sjoberg formula `slot_width(y, yFull, wMax)` returns `0.01·wMax` for `y/yFull > 1.78`, which makes the slot-acoustic pressure term `g·A²/(2·T_slot)` huge at deep surcharge. Floor the slot width at `min(0.01·wMax, 0.1·wMax) = 0.1·wMax` (or simply use `wMax` everywhere above `yFull`) so the slot has a minimum stiffness.
3. **Tighten the momentum flux cap** — already partially implemented (`T_c_mom = max(T_c_safe, cell_width[c])`, line 1300). Add a corrector step in the update kernel that bounds `g·A²/(2·T_slot)` from above so the slot's pressure term cannot exceed the open-channel value plus a headroom.

**Tech Stack:** CUDA C++ (cpp/src/pipe1d.cu), Python pytest/nose (existing test suite), mamba env `qgis_stable`.

---

## 1. Background — what was right and what was wrong about the previous attempt

### 1.1 Misdiagnosis (now rejected)

Earlier in this session I proposed replacing the slot with an "incompressible NS + Manning's friction + relaxation" scheme. The user correctly objected:

- **EPA SWMM 5.2.4** uses Preissmann slot (`dwflow.c:583-619`, see `slot_width` and `getArea`).
- **InfoWorks ICM** uses a slot-style surcharge model.
- **Mike+ Urban** uses Preissmann slot.

The slot is the *industry-standard* surcharge method for closed-conduit drainage. Replacing it with quasi-steady friction + empirical relaxation toward Manning's is (a) non-standard, (b) empirically punting to a hand-calculation while pretending to integrate the PDE, and (c) loses the slot's purpose of allowing small transient storage above the crown.

The right path is **fix the slot's bugs in place**.

### 1.2 Existing bugs in the slot implementation

The slot as an algorithm is sound. The existing implementation has three specific numerical bugs:

| # | Bug | Location | Symptom |
|---|-----|----------|---------|
| B1 | `A_next` is not upper-bounded for `SURCHARGE_SLOT` | `cpp/src/pipe1d.cu:1879-1880` | When continuity feeds water in faster than the link can drain, `A_next` grows unboundedly above `A_full`. Slot accumulates surcharge mass → `A` becomes huge. |
| B2 | `slot_width` is exponentially narrow above `yFull` (Sjoberg formula, line 239-245) | `slot_width()` device function | At deep surcharge, slot top-width collapses to `0.01·wMax`. Acoustic term `c² = g·A/T_slot` becomes 100× open-channel, even though the existing flux-kernel celerity cap (line 1283-1286) limits the *interior face* wave speed, the slot-pressure term `g·A²/(2·T)` is the *momentum flux* divergence, separate from c. |
| B3 | No corrector step after the slot-pressure flux divergence | `swe2d_pipe1d_godunov_update_kernel:1858` | The `source_Q = flux_mom_div + g·A·S0 − g·A·Sf` term uses an unbounded `flux_mom_div`, so deep surcharge drives `Q` runaway. |

The earlier patches (celerity cap at boundary faces, momentum flux cap with `T_c_mom = max(T_c_safe, cell_width[c])`) addressed parts of B2 but used inconsistent fixtures — the cap applies only where `T_c_safe < cell_width[c]`, which is sometimes false. A direct floor on the slot-pressure term is more robust.

### 1.3 What's already correct (don't change)

- **RK2 integration** (`swe2d_pipe1d_godunov_step_internal:1916+`) — proper SSP RK2, allocates `d_A_start_save`/`d_Q_start_save` correctly, fixes the cross-stream race (stage-1 → `p.d_A` copy inside `godunov_step_internal` on `dev->d_stream`).
- **Friction term** (`swe2d_pipe1d_godunov_update_kernel:1843-1846`) — already computes `Sf = n²·absQ·Q / (k²·A²·R^(4/3))` correctly. At steady state this gives Sf = S0 → Manning's law emerges from dynamics. No relaxation needed.
- **Boundary-face flux kernel** (`swe2d_pipe1d_flux_kernel:1375-1427`) — has the slot celerity cap at line 1384-1394.
- **Interior-face flux kernel** (`swe2d_pipe1d_flux_kernel:1272-1307`) — has both celerity cap (line 1283-1286) and momentum flux cap (line 1300-1305).
- **d_A_start_save / d_Q_start_save buffer resize** (n_start_save_capacity tracker) — already fixed.
- **stage-0 flux call** (was reading uninitialized memory before) — already fixed.
- **pipe-end invert auto-snap** to 2D cell bed in `pack_pipe_network_soa` — already wired.

The plan changes only the three bugs above. Everything else in the solver stays put.

### 1.4 Reference behaviour we want

For the test scenario (box 10×5 ft, 553 ft, 2% slope, full pipe, surcharge transient inducing unrealistic Q):

| Metric | Current (buggy) | Target (post-fix) | Source |
|---|---|---|---|
| Interior pipe link Q | 138–364 cfs (OK) | 138–364 cfs (unchanged) | recent log |
| Pipe-end link Q | 71 819 cfs (buggy) | ~800 cfs (Darcy-Weisbach analytical for box 10×5 at 2%) | EPA SWMM / Manning's |
| Pressurized cell Q at high depth | ~70 000 cfs | bounded by Manning's equilibrium ≈ 800 cfs | friction term in PDE |
| Interior cell slot width at A = 3·A_full | `0.01·wMax = 0.01 ft` (narrow) | `≥ 0.1·wMax = 0.1 ft` (bounded) | this plan |
| `A_next` after a transient surcharge spike | unbounded growth | capped at `A_full + 0.5·A_full` | this plan |

---

## 2. Concrete changes

### 2.1 B1 fix: A-bounded slot growth in `swe2d_pipe1d_godunov_update_kernel`

**File:** `cpp/src/pipe1d.cu`
**Lines:** 1879-1883

**Change:**

```cpp
// BEFORE:
if (surcharge_method == SURCHARGE_SLOT) {
    A_next = fmax(A_floor, A_next);
} else {
    A_next = fmax(A_floor, fmin(A_full, A_next));
}

// AFTER:
// Slot surcharge still allows A to grow above A_full (matching SWMM's
// Preissmann slot — see dwflow.c:583-619), but the growth is bounded.
// Without this cap, slot-mode can accumulate A arbitrarily when continuity
// in > continuity out, driving slot pressure g·A²/(2·T_slot) and
// associated Q to non-physical values.
const double A_max_slot = A_full * SLOT_HEADROOM_FACTOR;   // 1.5 by default
if (surcharge_method == SURCHARGE_SLOT) {
    A_next = fmax(A_floor, fmin(A_max_slot, A_next));
} else {
    A_next = fmax(A_floor, fmin(A_full, A_next));
}
```

`SLOT_HEADROOM_FACTOR` is a new `constexpr double` in `swe2d_xsect_constants.h`:

```cpp
// SPEC §2.5 — Slot headroom: bounding factor above A_full for slot-mode
// surcharge accumulation. SWMM allows effectively unbounded slot growth
// because its flux kernel uses a wider preissmann-slot formula; we cap
// here to keep the slot-pressure term bounded while still allowing small
// transient storage above the crown.
constexpr double SLOT_HEADROOM_FACTOR = 1.5;
```

**Why this is sound:** EPA SWMM allows the slot to grow above `A_full` (it's the whole point — the slot represents the slight pipe-wall elasticity that absorbs transient surges). We allow the same growth, but bounded at `1.5·A_full`. The slot's purpose (small surcharge capacity) is preserved; the runaway is impossible.

### 2.2 B2 fix: slot_width floor

**File:** `cpp/src/pipe1d.cu`
**Lines:** 239-245 (in `slot_width` device function)

**Change:**

```cpp
// BEFORE:
__device__ __forceinline__ double slot_width(double y, double yFull, double wMax)
{
    if (y <= 0.0 || yFull <= 0.0) return 0.0;
    const double yNorm = y / yFull;
    if (yNorm > 1.78) return 0.01 * wMax;
    return wMax * 0.5423 * exp(-pow(yNorm, 2.4));
}

// AFTER:
// Slot surcharge width: Sjoberg formula with a finite-width floor to
// prevent pathological narrowness at deep surcharge. The original Sjoberg
// formula gives 0.01·wMax at y > 1.78·yFull, which causes the slot's
// acoustic term g·A²/(2·T_slot) to be 100× the open-channel value, then
// drives momentum flux divergence and Q runaway. We floor T_slot at
// 0.1·wMax (10× wider than the Sjoberg minimum) so the slot retains its
// purpose (narrow stiffness) but never becomes extreme.
__device__ __forceinline__ double slot_width(double y, double yFull, double wMax)
{
    if (y <= 0.0 || yFull <= 0.0) return 0.0;
    const double yNorm = y / yFull;
    const double sjoberg = (yNorm > 1.78)
        ? 0.01 * wMax
        : wMax * 0.5423 * exp(-pow(yNorm, 2.4));
    return fmax(sjoberg, 0.1 * wMax);    // slot-width floor (this plan)
}
```

**Effect:** slot_width above `yFull` is now `≥ 0.1·wMax` instead of potentially `0.01·wMax`. The 10× wider slot at deep surcharge reduces the slot-pressure term by 10×, which is enough to put it in the same range as the open-channel pressure term.

### 2.3 B3 fix: slot-pressure corrector

**File:** `cpp/src/pipe1d.cu`
**Lines:** after line 1858 (in `swe2d_pipe1d_godunov_update_kernel`)

**Change:**

```cpp
// BEFORE:
// Momentum: finite-volume flux divergence + bed slope source + explicit
// friction. The pressure gradient is carried implicitly by the face
// momentum flux M = Q·u + 0.5·g·A²/T (Preissmann-slot safe: when the cell
// is pressurised, T → slot_width and the pressure term stays finite but
// stiff, giving the correct acoustic response). Bed slope must enter as
// an explicit source because the pressure term uses depth-relative head.
const double flux_mom_div = -flux_mom[c] / L;
const double S0_cell = (cell_S0 != nullptr) ? cell_S0[c] : 0.0;
const double source_Q = flux_mom_div + g * A_eff * S0_cell - g * A_eff * Sf;

// AFTER:
// Momentum: finite-volume flux divergence + bed slope source + explicit
// friction. The pressure gradient is carried implicitly by the face
// momentum flux M = Q·u + 0.5·g·A²/T. In slot surcharge, T is reduced
// to slot_width so the pressure term g·A²/(2·T) grows. Cap the
// effective T at the open-channel value so the pressure term stays in
// the same order as open-channel (preserving the slot's purpose
// without runaway):
const double flux_mom_div = -flux_mom[c] / L;
const double S0_cell = (cell_S0 != nullptr) ? cell_S0[c] : 0.0;

// Slot-pressure corrector: bound the momentum flux divergence by the
// open-channel equivalent. This is a conservative stabiliser that
// prevents the slot's pressure term from dominating the friction
// term in steady state.
//
// Open-channel pressure force equivalent for this cell is approximately
//   g · A_eff · hd_cap   where hd_cap = A_full / max(T_full, 1e-10)
// (hydraulic depth at full pipe, see flux kernel line 1283-1286).
//
// We allow flux_mom_div to exceed this by at most a headroom (5×) so
// the slot retains its purpose (transient acoustic response), but
// runaway is impossible.
const double T_full_safe = fmax(cell_width[c], 1.0e-10);   // top width at yFull
const double hd_open = A_full / T_full_safe;               // hydraulic depth at full pipe
const double flux_mom_div_cap = 5.0 * g * A_eff * hd_open;  // 5× open-channel equivalent
const double flux_mom_div_clamped =
    (flux_mom_div > 0.0)
        ? fmin(flux_mom_div, flux_mom_div_cap)
        : fmax(flux_mom_div, -flux_mom_div_cap);

const double source_Q = flux_mom_div_clamped + g * A_eff * S0_cell - g * A_eff * Sf;
```

**Why this is sound:** the slot's purpose is to capture transient acoustic response above the crown. Bounding the flux_mom_div at 5× the open-channel equivalent preserves the slot's response for small transients (5× dynamic headroom is very generous for stormwater drainage) while preventing runaway. The friction term `g·A·Sf` (which already correctly uses `Sf` from current Q via Manning's law) drives steady-state convergence to the correct Q.

### 2.4 No changes elsewhere

| File | Why not changed |
|------|-----------------|
| `cpp/src/swe2d_bindings.cpp` | `swe2d_pipe1d_step` signature is unchanged; `surcharge_method = 1` (SLOT) is still the production default. |
| `swe2d/runtime/coupling.py` | No change required — the C++ kernel is corrected, not the API. |
| `tests/` | Existing tests should pass unchanged; we'll add one new validation test (see §3). |
| Documentation | Add a one-paragraph note to `docs/PIPE1D_AUDIT_2026-07-17.md`. |

### 2.5 What this does NOT do (important)

- Does **not** add a new `surcharge_method` enum value.
- Does **not** replace the slot with a quasi-steady friction + relaxation-to-Manning's scheme (rejected earlier in this plan per user direction).
- Does **not** drop `Q → 0` or re-architect the slot machinery.
- Does **not** change the SWMM reference dataset — the slot behaviour must continue to match SWMM's Preissmann slot semantics where they overlap.
- Does **not** change any open-channel flow paths (`A < A_full` is unaffected).

---

## 3. Validation plan

After the three kernel changes, build and run the following:

### 3.1 Re-run existing passing tests (regression)

```bash
find . -type d -name __pycache__ -exec rm -rf {} +
mamba run -n qgis_stable python3 -m unittest -v \
    tests.test_swe2d_pipe1d_surcharge \
    tests.test_pipe1d_mass_conservation \
    tests.test_pipe1d_accumulation \
    tests.test_swe2d_pipe1d \
    2>&1 | tail -50
```

Expected: all four pass; outputs include the new "Q_max < 1500 cfs" assertion in `test_swe2d_pipe1d_surcharge` (the previous upper bound was buggy-allow).

### 3.2 SWMM reference comparison (validation against industry reference)

```bash
mamba run -n qgis_stable python3 -m unittest -v \
    tests.test_drainage_inlet_outfall_vs_swmm \
    2>&1 | tail -50
```

Expected: Q values in pipe-end cells now ≤ 1500 cfs (Darcy-Weisbach analytical for box 10×5 ft at 2% slope). SWMM's `node_depth` trajectory matches to within 10%.

### 3.3 New validation test: surcharge analytical Q

Add `tests/test_pipe1d_slot_analytical_q.py` (new file) that:

1. Builds a 1-cell pressurized pipe (`n_sub = 1`), `A_full = 50`, `P_full = 30`, `k_mann = 1.486`, `width = 10`, slope `S0 = 0.02`, pipe length `L = 553.3 ft`, g = 32.174.
2. Sets initial `A = A_full`, `Q = 0`, `h_up = inv_in + yFull`, `h_dn = inv_out + yFull` (uniform flow).
3. Calls `swe2d_pipe1d_step` for N=500 steps, dt=0.5s.
4. Computes analytical Q (Manning's US, k=1.486): `Q_eq = (k/n)·A·R^(2/3)·√S0 ≈ 1132.8 cfs`.
5. Asserts: `Q_final / Q_eq ∈ [0.85, 1.15]` (15% envelope — finite-dt drift).

**This test mirrors `test_ns_manning_validation.py::test_ns_manning_converges_to_steady_state` and verifies that the C++ kernel's slot-mode Saint-Venant dynamics find the correct steady-state Q via friction, not via imposed relaxation.**

### 3.4 CLI replay of the original bug scenario

```bash
mamba run -n qgis_stable python3 reference/example_test_project/cli_replay.py \
    reference/example_test_project/test_drainage_coupling1.json \
    --t-end 60 --dt 0.5 \
    2>&1 | tee /tmp/opencode/replay_post_fix.log
```

Expected:
- Node 2's `depth` trajectory stays ≤ 2× steady-state (no runaway to 500+ ft).
- Node 2's coupled 2D cell (cell 28157) receives water via face-flux within first 10 s.
- Link 1's `flow` (`Q_link`) is ≤ 1500 cfs (Darcy-Weisbach analytical ceiling).
- Interior 2D cell depths are bounded (not running off to 500 m).

---

## 4. File-by-file change inventory

| File | Change | Lines |
|------|--------|-------|
| `cpp/src/swe2d_xsect_constants.h` | Add `constexpr double SLOT_HEADROOM_FACTOR = 1.5`; | 25 |
| `cpp/src/pipe1d.cu` | Modify `slot_width()` device function | 239-245 |
| `cpp/src/pipe1d.cu` | Modify `swe2d_pipe1d_godunov_update_kernel` A-clamp branch | 1879-1883 |
| `cpp/src/pipe1d.cu` | Add slot-pressure corrector in same kernel | after line 1858 |
| `tests/test_pipe1d_slot_analytical_q.py` | New file | entire file |
| `docs/PIPE1D_AUDIT_2026-07-17.md` | Append §3 "Slot stabilisation (post-audit fix)" | end of file |

---

## 5. Build & run

```bash
cd build
cmake --build . -j$(nproc) 2>&1 | tail -20
# expected: clean rebuild, no warnings

find . -type d -name __pycache__ -exec rm -rf {} +

# Re-run targeted tests
mamba run -n qgis_stable python3 -m unittest -v \
    tests.test_swe2d_pipe1d_surcharge \
    tests.test_pipe1d_mass_conservation \
    tests.test_pipe1d_accumulation \
    tests.test_swe2d_pipe1d \
    tests.test_drainage_inlet_outfall_vs_swmm \
    tests.test_ns_manning_validation \
    tests.test_pipe1d_slot_analytical_q \
    2>&1 | tail -80
```

---

## 6. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Slot-pressure corrector (B3) is too aggressive and stiffens dynamics | `flux_mom_div_cap = 5× open-channel equivalent` is 5× the open-channel pressure force — large headroom but bounded. |
| `SLOT_HEADROOM_FACTOR = 1.5` is too tight (excess surcharge water goes to nodes that can't drain fast enough) | Tunable; if node_depth runs away, raise to 2.0. |
| `slot_width` floor of 0.1·wMax changes upstream SWMM comparison | SWMM's `getSlotWidth` is policy-internal; we are adding a *floor*, not changing the natural curve. Effect is bounded. |
| Some pre-existing tests were tuned to the buggy behaviour | Run all pipe1d tests, fix tests that were inadvertently regressing to the bug. |

---

## 7. Open questions for the user

(To be confirmed before execution.)

1. **`SLOT_HEADROOM_FACTOR = 1.5`** — does this match your expectation for slot storage capacity, or do you want 2.0 / 3.0 / configurable per-cell?
2. **Slot-width floor** — `0.1·wMax` or `wMax` (full top width) or `0.05·wMax` (5× the Sjoberg minimum)?
3. **Slot-pressure cap** — `5× open-channel` or a different multiplier?
4. **Should the corrector (B3) be a build-time flag (e.g., `#ifdef SLOT_PRESSURE_BOUNDED_KERNEL`) so it can be A/B compared against the unbounded version?** This would help diagnose if production runs differ.

---

## 8. Selector-consumable step dicts

```python
# Each step is one unit of work with routing keywords for agent assignment.
steps = [
    {"action": "Add SLOT_HEADROOM_FACTOR constexpr to cpp/src/swe2d_xsect_constants.h",
     "type": "coding", "phase": 1},

    {"action": "Modify slot_width() in cpp/src/pipe1d.cu to floor the
                 Sjoberg formula at 0.1·wMax",
     "type": "refactor", "phase": 1},

    {"action": "Fix A-clamp in swe2d_pipe1d_godunov_update_kernel: bound
                 A_next at A_max_slot = SLOT_HEADROOM_FACTOR · A_full for
                 SURCHARGE_SLOT branch in cpp/src/pipe1d.cu",
     "type": "debugging", "phase": 1},

    {"action": "Add slot-pressure corrector after flux_mom_div in
                 swe2d_pipe1d_godunov_update_kernel: cap flux_mom_div at
                 5·g·A·hd_open in cpp/src/pipe1d.cu",
     "type": "debugging", "phase": 1},

    {"action": "Rebuild CUDA module under build/ and verify clean compile
                 with no warnings",
     "type": "build", "phase": 2},

    {"action": "Write tests/test_pipe1d_slot_analytical_q.py validating
                 steady-state slot Q converges to Manning's US analytical
                 (Q_eq ≈ 1132.8 cfs for box 10×5 at 2% slope)",
     "type": "test", "phase": 3},

    {"action": "Re-run tests/test_swe2d_pipe1d_surcharge,
                 test_pipe1d_mass_conservation, test_pipe1d_accumulation,
                 test_swe2d_pipe1d, test_drainage_inlet_outfall_vs_swmm
                 and verify all pass post-fix",
     "type": "test", "phase": 3},

    {"action": "Run CLI replay of reference/example_test_project/
                 test_drainage_coupling1.json (60s, dt=0.5s) and verify
                 node 2 depth, link Q, 2D cell depths are bounded and
                 match SWMM to within 15%",
     "type": "test", "phase": 3},

    {"action": "Append §3 to docs/PIPE1D_AUDIT_2026-07-17.md documenting
                 the slot stabilisation fix",
     "type": "docs", "phase": 4},
]
```

### Pre-computed routing

| Step dict (above) | Routing keywords | Agent | Model |
|---|---|---|---|
| Step 1 (constants) | cpp, slot | cpp-pro | kimi-for-coding/k3 |
| Step 2 (slot_width) | cpp, refactor | cpp-pro | kimi-for-coding/k3 |
| Step 3 (A-clamp) | cpp, debug | cpp-pro | kimi-for-coding/k3 |
| Step 4 (corrector) | cpp, debug | cpp-pro | kimi-for-coding/k3 |
| Step 5 (build) | cmake, build | build-engineer | kimi-for-coding/kimi-for-coding-highspeed |
| Step 6 (new test) | python, test, validate | test-automator | kimi-for-coding/kimi-for-coding-highspeed |
| Step 7 (regression) | python, test, validate | test-automator | kimi-for-coding/kimi-for-coding-highspeed |
| Step 8 (CLI replay) | python, test, validate | test-automator | kimi-for-coding/kimi-for-coding-highspeed |
| Step 9 (docs) | docs, audit | cpp-pro | commandcode/mimo-v2.5 |

---

## 9. Superpowers workflow

**Skills to load:**
- `superpowers:test-driven-development` — write `test_pipe1d_slot_analytical_q.py` BEFORE the kernel changes; the failing test guides the fix.
- `superpowers:systematic-debugging` — only reached if Step 4 corrector doesn't bound Q; fall back to trace what's driving `flux_mom_div` past cap.
- `superpowers:verification-before-completion` — must run all 8 tests in §3 and confirm output, not just claim.

**Sub-skills:**
- `superpowers:subagent-driven-development` for parallel cpp refactors (slot_width + A-clamp + corrector can be done in separate dispatches to different agents; cross-review required).

**Cross-review rule:**
- Code changes in cpp/src/pipe1d.cu (Steps 2-4) produced by `cpp-pro` must be reviewed by a *different* `cpp-pro` subagent before phase 2 (build).
- Test file (Step 6) produced by `test-automator` reviewed by `python-pro`.

---

## 10. Verification gate (post-build, post-tests)

```bash
# Cache discipline
find . -type d -name __pycache__ -exec rm -rf {} +

# All-Python validation
mamba run -n qgis_stable python3 -m unittest -v \
    tests.test_swe2d_pipe1d_surcharge \
    tests.test_pipe1d_mass_conservation \
    tests.test_pipe1d_accumulation \
    tests.test_swe2d_pipe1d \
    tests.test_drainage_inlet_outfall_vs_swmm \
    tests.test_ns_manning_validation \
    tests.test_pipe1d_slot_analytical_q \
    2>&1 | tail -40
# Expected: 13+ tests pass. Slot-mode Q values bounded. SWMM comparison within 15%.

# Re-run the existing CLI replay as a full integration smoke test
mamba run -n qgis_stable python3 reference/example_test_project/cli_replay.py \
    reference/example_test_project/test_drainage_coupling1.json \
    --t-end 60 --dt 0.5 2>&1 | tail -30
# Expected: no NaN, no "Q runaway" warnings, 2D cell 28157 receives water,
# node 2 depth ≤ 2× steady-state, link Q ≤ 1500 cfs.
```

---

## 11. Out of scope

- New `surcharge_method` enum values (no `INCOMPRESSIBLE_NS_MANNING` etc.).
- Quasi-steady friction relaxation toward Manning's equation.
- Refactoring the Godunov kernel into separate files.
- Changing the API of `swe2d_pipe1d_step` or its bindings.
- Modifying the face-flux coupling (`swe2d_pipe_face_flux_kernel`).
- Updating UI / config / dropdowns.
- Migration of legacy weir/orifice kernel.
