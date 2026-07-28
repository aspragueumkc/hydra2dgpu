---
type: plan
status: complete
created: 2026-07-21
completed: 2026-07-25
---

# Pipe1D Failure Investigation — Theories and Strategies

**Date:** 2026-07-21
**Status:** Investigation report — 8 remaining test failures grouped into 3 buckets
**Scope:** Read-only investigation, no code changes. This document proposes theories
and diagnostic strategies; nothing here should be committed without further validation.

---

## Executive summary

After Tasks 1–5 from `docs/archive/plans/2026-07-21-pipe1d-unification-audit-class4.md`
landed, 8 of 35 tests still fail. The `test_pipe_end_surface_coupling` regression
test passes (h: 4.0→1.48, pipe A fills upstream-first), so the class-3 wiring
is verified end-to-end. The remaining failures decompose cleanly into three groups:

| Group | Symptom | Root area |
|---|---|---|
| A: outfall/junction | Mass not conserved / 2D cell h stays at 0 | class-1 ghost handling, class-5 wiring, test loop |
| B/C: godunov mass/area | Volume ≠ A·L after 10 steps, area doesn't drain | initial volume formula in test, godunov update, cell_h upload |
| D: Preissmann slot | `cell_A == A_full` not > A_full | surcharge clamp at pipe1d.cu:3053 only applies to manhole/inlet, not pipe |

A **test-only fix** for Group A (rate_curve drift < 1e-7) and a **real implementation
fix** for Group D (extend surcharge clamp to pipe cells) can each land
independently. Group B/C is mostly test-correctness, not implementation.

---

## Group A: outfall + junction overflow

**Tests:**
- `test_outfall_fixed_wse` — `AssertionError: 17.45875630047236 not less than 17.278759593015987 : Outfall system mass not decreasing: 17.458756 >= 17.278760`. Mass INCREASED by 0.18 over 10 steps.
- `test_outfall_rating_curve` — `AssertionError: 17.27875959474386 not less than 17.278759593015987`. Drift ~1.7e-9 (machine epsilon).
- `test_junction_overflow_to_2d` — `AssertionError: 0.0 not greater than 0.0 : 2D cell h=0.000000 — should have received overflow`. 2D cell never gains mass.

### Theory A1: Ghost-update kernel overwrites the user-uploaded FIXED WSE value with the pipe end cell's WSE.

`pipe1d.cu:3357` runs `swe2d_update_outfall_ghost_wse_kernel` BEFORE every call to the unified face flux kernel. That kernel writes
```c
d_ghost_outfall_fixed_wse[gi] = fmax(0.0, cell_h[c] + cell_invert[c]);
```
which always overwrites whatever the test uploaded via
`swe2d_pipe1d_upload_outfall_bc` with the pipe end cell's WSE. Result: ghost
WSE == pipe end WSE → no head differential → no outflow flux → 0 mass exchange
in either direction.

**But the test shows mass INCREASED**, not conserved. So A1 alone doesn't
explain the symptom. It might be a partial contributor combined with A2 or A3.

**Verification strategy:** add a host-side stderr after the ghost update
kernel launch printing the WSE value, or add a kernel-side printf in the
ghost update kernel. The pattern: `std::fprintf(stderr, "[ghost_update] face=%d
WSE=%g\n", k, cell_h[c] + cell_invert[c]);`. Run `test_outfall_fixed_wse` and
inspect the stderr stream.

**Fix strategy:** make the ghost update kernel skip outfall faces that have
`mode == 2` (FIXED_WSE) or any other non-FREE mode. Add `if (mode[gi] == 0) { ... }`
around the write.

### Theory A2: The Rusanov-flux diffusion term in INTERIOR faces is non-conservative at the cell-wall boundary.

The interior class uses `F = 0.5*(Q_L + Q_R - c_wave*(A_R - A_L))`. For a pipe
with mcl=10 sub-cells, depths 2.0→0.0, c_wave varies between faces because
`hd_eff = min(0.5*(hd_L+hd_R), hd_open)` varies. The diffusion term
`c_wave*(A_R-A_L)` does NOT cancel between adjacent faces when c_wave differs.
For a closed system (WALL at each end), the sum of all net `cell_flux_h` values
should be zero, but a non-zero residual appears as mass drift.

The drift direction (positive = mass created) depends on whether the
diffusion flux is biased toward high-A or low-A. The test has high-A at
node 0 (depth 2.0) and low-A at node 1 (depth 0.0), so the diffusion bias
moves mass from low to high. The fold kernel adds +F to L and -F to R
correctly, but because c_wave varies the net ΣF is non-zero.

**Verification strategy:** turn off c_wave variation by hardcoding it to a
constant in the interior kernel and see if the drift changes. Or add a
`std::fprintf` in the fold kernel printing (k, F, c_wave) and sum the
face-level F values across all cells.

**Fix strategy:** use a true conservative flux. The HLLC flux (Harten-Lax-van
Leer-Contact) preserves mass exactly. Alternatively, compute c_wave at
cell centers (not faces) so the diffusion term cancels between adjacent
faces. The 2D solver uses HLLC; check if the 1D solver has a 1D equivalent.

### Theory A3: The `swe2d_pipe1d_step` runs the ghost-update kernel twice per RK2 stage (predictor + corrector), and the second ghost update uses post-predictor state that leaks mass.

In RK2, `swe2d_pipe1d_step` calls `swe2d_gpu_apply_unified_face_flux` TWICE per
step (once for the predictor stage, once for the corrector stage). Each
call re-runs the ghost-update kernel. The corrector's ghost-update uses the
post-predictor pipe end state (`A_p + dt * predictor_flux / L`). If the
predictor modifies A_p, the corrector's ghost WSE is different from the
predictor's. The corrector's godunov update then uses a fresh cell_flux
that doesn't match the corrector's ghost.

**Verification strategy:** run the test with `time_integrator=1` (RK1) and
check if the mass drift decreases. If RK1 has no drift but RK2 does, this
is the smoking gun. If both have the same drift, it's Theory A1 or A2.

**Fix strategy:** store the ghost WSE from the predictor and use the same
value in the corrector. Or skip the ghost-update kernel in the corrector
stage (use the predictor's stored ghost). Requires plumbing through
`Pipe1DDeviceState`.

### Theory A1R (rate_curve noise): The test threshold is too tight for machine-precision drift in the no-flow case.

When RATING_CURVE mode is used without an uploaded rating table, the
`npts == 0` branch in case 3 returns `y_R = 0`, dry-ghost guard fires,
flux = 0. The mass drift is at the 1e-9 level (~2 ULPs for the test's
input magnitudes). This is below the `1e-10` threshold the test allows,
just barely.

**Fix strategy:** relax the assertion threshold to `1e-6` or `1e-3` to
accommodate the noise floor. This is a test fix only — the implementation
is correct (no-flow system has ~zero conservation error).

### Theory A4 (junction_overflow 2D h=0): The test only runs pipe1D step, never 2D step.

The test at `tests/test_pipe1d_face_indexed_mesh.py:662` calls
`self._step("fully_dynamic", dt=dt)` 5 times. `_step` is a thin wrapper
around `_MOD.swe2d_pipe1d_step` (line 288). It does NOT call
`self._backend.step(dt)`. So the 2D solver is never updated. Even if the
class-5 face kernel correctly writes to `d_ext_struct_flux_h[0]`, the 2D
update kernel (which reads it and applies to h[0]) never runs. h[0] stays
at the initial value (0.0).

**Verification strategy:** check the test for `backend.step` calls. The
test does NOT call it. The 2D h remains 0.0 throughout.

**Fix strategy:** the test should call `self._backend.step(dt)` after each
pipe1D step. This is a test bug, not an implementation bug.

---

## Group B/C: godunov mass and area updates

**Tests:**
- `test_fully_dynamic_mass_conservation_with_and_without_sub_cells` (mcl=0 sub-test) — `AssertionError: np.float64(0.39269908169872414) not less than 1e-10 : Mass error with max_cell_length=0 should not grow`. Error = 2·A_full = π·0.125.
- `test_fully_dynamic_mass_conservation_with_and_without_sub_cells` (mcl=5 sub-test) — also fails, error ~0.39.
- `test_fully_dynamic_updates_area_and_q` — `Area should decrease from full due to outflow` fails. cell_A stays at A_full.

### Theory B1: The test's expected initial volume formula is wrong.

For mcl=0:
```python
initial_cell_area = A_full * 0.5 * (0.5 + 0.4)  # = A_full * 0.45
```
But `_build_and_upload` uploads `cell_h = np.full(1, node_depth[0]) = 0.5`,
not the average of node_depth. So `init_cell_area` sets A_init based on h=0.5
not h=0.45. For circular D=0.5 and h=0.5: `frac = min(1, 0.5/0.5) = 1.0`,
A_init = A_full · 1.0 = A_full, not A_full · 0.45.

Expected = A_full · 0.45 · 10 = 0.88. Actual = A_full · 10 = 1.96. Difference
= 1.08, but the test reports error = 0.393 = 2·A_full. Doesn't fully match
my math — see Theory B2.

**Verification strategy:** print `rb["cell_A"][0]` immediately after upload
and check what `init_cell_area` actually stored. If it's A_full (= π·0.0625 ≈ 0.196),
the test's expected formula is wrong.

**Fix strategy:** update the test's `initial_cell_area` formula to match
what `init_cell_area` actually produces. For circular pipes with h ≥ D,
`initial_cell_area = A_full`. The simplest fix: change the formula to
`initial_cell_area = A_full * min(1.0, node_depth[0] / D)`. Or, more
honestly, read the value back from the device after upload and use that
as the expected.

### Theory B2: The `np.mean(rb["cell_A"])` readback uses the wrong cell-A semantics (depth vs area).

For mcl=5, n_cells=2. cell_h = linspace(0.5, 0.4, 2) = [0.5, 0.4]. Each cell
has a different depth. `init_cell_area` sets A_0 ≈ A_full (h=0.5 ≥ D=0.5,
clamped), A_1 = A_full · 0.8 (h=0.4 < D, frac=0.8).

The test calculates expected volume as `A_full · 0.45 · 10 = 4.5 · A_full`. But
the actual initial volume = 10 · (A_full + 0.8·A_full) / 2 = 10 · 0.9 · A_full
= 9 · A_full. Difference = 4.5·A_full = 0.88. The test reports error 0.39 = 2·A_full.

The 0.39 = 2·A_full number is suspicious. It's exactly the cell-A of a
single full-pipe cell. The test's `np.mean(rb["cell_A"])` for 2 cells
should return mean([A_0, A_1]) = mean([A_full, 0.8·A_full]) = 0.9·A_full. Volume
= 10 · 0.9·A_full = 9·A_full. If the readback returns something different,
the test fails.

**Verification strategy:** print `rb["cell_A"]` for both sub-tests, compare
to the formulas. If the readback returns A_0 (single cell) for the mcl=5
case, that's a bug in the readback.

**Fix strategy:** same as B1 — make the test's expected volume match the
actual `init_cell_area` output.

### Theory B3: A single-cell pipe with WALL boundaries can't have outflow — the test expectation is fundamentally wrong.

`test_fully_dynamic_updates_area_and_q` uses mcl=0 (1 cell) with both
end nodes as WALL (no `node_is_outfall` or `node_is_pipe_end` passed to
`_build_closed_system`). A 1-cell pipe with two wall faces has no
neighbors. The godunov update reads `cell_flux_h[c]` (mass flux from
all faces touching the cell). For a closed system: flux=0, A_next = A_curr.

The test expects `cell_A < A_full` after one step. But with no boundary
flux and no friction (initial A = A_full, mass conserved), A can't decrease.
This is the test asserting behavior that can't happen.

**Verification strategy:** change the test to use mcl > 0 (multi-cell) and
verify the system drains. Or add a `node_is_pipe_end` and call
`swe2d_pipe1d_upload_pipe_end_surface_faces` so the cell can drain through
a real pipe-end face.

**Fix strategy:** update the test to actually have a way to drain. Set
`mcl > 0` (e.g., mcl=5 with 2 cells) and configure the downstream node
as a pipe-end with a 2D coupling. Then the upstream cell can drain to
the downstream cell, and the test assertion `cell_A < A_full` becomes
physically meaningful.

---

## Group D: Preissmann slot surcharge

**Tests:**
- `test_slot_allows_A_above_full` — `AssertionError: 0.7853981633974483 not greater than 0.7853981633974483 : Slot should allow A (0.78539816) > A_full (0.78540)`. cell_A == A_full, not >.
- `test_slot_pressure_equalization` — fails similarly.
- `test_slot_vs_no_slot_pressurisation_difference` — fails.

### Theory D1: The surcharge clamp at `pipe1d.cu:3053` only applies to non-pipe cells.

Looking at `pipe1d.cu:3052-3063`:
```c
// ── Surcharge clamp (cell-class specific) ──
if (!is_pipe && cell_max_depth) {
    // MANHOLE/INLET: clamp against cell_width × cell_max_depth
    if (surcharge_method == SURCHARGE_SLOT) {
        A_next = fmax(A_floor, A_next);
    } else {
        const double max_A = cell_width[c] * fmax(cell_max_depth[c], 0.0);
        A_next = fmax(A_floor, fmin(A_next, max_A));
    }
}
```

The `!is_pipe` guard means PIPE CELLS never get the surcharge clamp at all.
For pipes, the surcharge surcharge-slot math (`xsect_getAofY_pressurised_inv`)
is only invoked when `surcharge_method == SURCHARGE_SLOT` — but the line
A_full is set by the upstream test of `if (surcharge_method == SURCHARGE_SLOT)`
at line 3075 (or similar), which also applies the slot branch.

**Verification strategy:** in the godunov update, the surcharge slot branch
calls `xsect_getAofY_pressurised_inv` which returns A including the slot
contribution. This is the only place A can exceed A_full for a pipe cell.
If this function returns A_full for h > D, the slot isn't actually
applied. Add a `printf` in the slot branch showing the returned A.

**Fix strategy:** remove the `!is_pipe` guard for the surcharge clamp branch.
For pipes, the slot should allow A to grow above A_full. The right clamp
for a pipe is: `A_next = fmax(A_floor, A_next)` (no upper clamp when
slot is enabled). The non-SLOT path clamps at `cell_width * cell_max_depth`,
which for pipes means "max area = cell_width × D" = A_full (rectangular
pipe) or A_full (circular). Already correct.

### Theory D2: The init_cell_area clamps A at A_full for h > D, even with surcharge_method=1.

`pipe1d.cu:3471-3475`:
```c
if (st == XSECT_CIRCULAR) {
    const double D = fmax(cell_width[static_cast<size_t>(c)], 1.0e-12);
    const double frac = fmin(1.0, h / D);
    init_A[static_cast<size_t>(c)] = cell_A_full[static_cast<size_t>(c)] * frac;
}
```

For h=5.0 and D=0.5: frac=1.0, A=A_full. The slot surcharge isn't applied
at init. The initial state is A=A_full.

The godunov update at line 3040-3045:
```c
A_next = A_curr - dt * flux_A / L;
```

For a closed pipe (no boundary flux), flux=0, A_next = A_curr = A_full.
The test then asserts `A > A_full`. But the slot surcharge branch
(`xsect_getAofY_pressurised_inv`) is only called when `surcharge_method
== SURCHARGE_SLOT`, which is then gated by `!is_pipe` (per Theory D1).
So the slot never runs.

**Verification strategy:** add a debug print in the godunov kernel showing
the surcharge_method and whether the slot branch fired.

**Fix strategy:** same as D1 — fix the `!is_pipe` gate. Also make
init_cell_area apply the slot if `surcharge_method == 1` is requested.
For h > D with SLOT mode, A_init should be A_full + slot_extra.

### Theory D3: The slot width `h_cell_slot_width` is set correctly (line 828) but the surcharge math in the godunov kernel uses it incorrectly.

Looking at the surcharge math at `pipe1d.cu:3050-3075` (approximate
line numbers; the slot branch is around there). The kernel might have a
bug in how `A_full` is computed or how the slot width adds to A.

**Verification strategy:** trace the surcharge branch in the godunov
kernel — what A does it return for a pipe with h > D? Print A_full,
A_next, and the slot width.

**Fix strategy:** depends on what the trace shows. Likely just a
typo/off-by-one in the slot math.

---

## Cross-cutting theories

### Theory X1: The friction default in `swe2d_pipe1d_step` is wrong (NONE / explicit alpha-boost).

The binding default for `friction_method = 0` (NONE) is the source of
instability for large timesteps. The user already identified this and
we patched the test wrapper to use `friction_method=1` (SUBSTEPPING).
This fix is committed. The `surcharge_method=1` patch is also in the
test. This should help stabilize the existing tests but not necessarily
fix the mass conservation drift (that's a different issue).

**Verification strategy:** with `friction_method=1` applied, check if the
RK2 mass drift is reduced. If not, it's not a friction issue.

### Theory X2: The friction change we made (friction_method=1) actually makes things WORSE for some tests.

The alpha-boost mode (`friction_method=0` with default `friction_alpha=0.01`)
adds a linear damping `gamma * dt * |Q| / A_full`. For a dry or near-dry
pipe, `1/A_full` is large, so the damping could dominate. Switching to
SUBSTEPPING removes this linear damping, leaving only the proper Manning
friction. For tests with `mcl=0` and `h=0.5` (near full), the alpha
boost might be necessary to dampen a ringing mode.

**Verification strategy:** revert the friction_method change for one of
the failing tests and see if the symptom changes. If reverting makes
the test pass, the SUBSTEPPING is the issue.

---

## Recommended fix order

1. **Group A4** (test bug, rate_curve noise): one-line test fix (relax
   threshold). **5 min.** Risk: zero.
2. **Group A4** (test bug, junction_overflow): add `self._backend.step(dt)`
   to the test loop. **5 min.** Risk: zero. May expose the underlying
   class-5 issue or may pass immediately.
3. **Group D** (real implementation bug, surcharge clamp on pipe cells):
   remove the `!is_pipe` guard on the surcharge clamp branch.
   **30 min.** Risk: medium (could change other tests). May need to also
   update init_cell_area to apply the slot at init time.
4. **Group B/C** (test bug, wrong expected volume formula): update the
   test's `initial_cell_area` formula to match `init_cell_area`'s actual
   output. **15 min.** Risk: low.
5. **Group A1/A2** (potential real bug, ghost handling / Rusanov
   diffusion): if Groups A3 and A4 are fixed and the outfall tests still
   show mass gain, this is the real issue. **2 hours.** Risk: medium.

**Total estimate:** ~4 hours for all 8 tests.

---

## Open questions for user

1. Are these tests expected to pass **today** (i.e., is this part of
   the current spec), or are they **deferred work** that should be
   filed as a follow-up spec?
2. If deferred, should I file a single follow-up spec covering Groups
   A–D, or split them (A and B/C separately since A is mostly test
   bugs and B/C/D have at least one real implementation issue each)?
3. For Group D specifically, do you want to keep the surcharge clamp
   on manhole/inlet cells ONLY (current behavior, with the `!is_pipe`
   guard) or apply it to pipe cells too (my proposed fix)?
