---
type: spec
status: complete
created: 2026-07-21
completed: 2026-07-25
---

# Pipe1D Unification — Close Audit Gates + Wire Class-4 Inlet

**Date:** 2026-07-21
**Status:** Proposed. Closing the 9-failure pipe1D audit gap and wiring the
HEC-22 inlet face end-to-end so the user's CLI replay shows non-noise pipe
flow.

---

## 1. Goal

All 9 currently-failing pipe1D tests pass, **and** the user's
`pipe_1d_test.json` CLI replay shows physically meaningful pipe flow
(> 1 CFS through the 8'×4' box culvert at WSE=935).

**Both gates must pass for completion.**

Out of scope for this spec, deferred for the next session:

- Keep `swe2d_gpu_apply_unified_face_flux` permanently non-`static`
  (architectural decision; not blocking).
- Verify RK-stage `d_ext_struct_flux_*` zero-accumulate order across
  predictor + corrector (only matters if Gate 3 surfaces a regression).
- Test-directory cleanup
  (`tests/test_pipe_end_surface_coupling.py` vs
  `tests/test_pipe1d_face_indexed_mesh.py`; duplicate test paths).

## 2. Architecture (unchanged from current session)

This work uses the patterns proven this session. Three new bindings +
one direct-write patch follow the exact class-3 shape:

| Pattern | Reference impl | Reuse target |
|---|---|---|
| Real upload-binding (read face_class, find class-N, patch face_owner_R from host array) | `swe2d_pipe1d_upload_pipe_end_surface_faces` (binding + impl already in place, ~50 lines) | outfall (class-1), junction overflow (class-5), 2D-culvert (class-6) |
| Direct `atomicAdd(&d_A[L], -Q * dt / cell_length[L])` in the face kernel (bypasses fold+godunov when caller is `swe2d_gpu_step`) | class-3 kernel `cpp/src/pipe1d.cu:2413` (the patch landed this session) | class-1, class-4 (HEC-22 inlet), class-5, class-6 |

Class-0 (INTERIOR) and class-2 (INLET_BC) do **not** need the direct-write —
their flux comes from the existing fold+godunov path inside
`swe2d_pipe1d_step`. Adding the atomicAdd to class-0/class-2 would double-count.

## 3. Components

### 3.1 Real upload binding for class-1 (outfall)

**Where:** new impl `swe2d_pipe1d_upload_outfall_surface_faces` in
`cpp/src/swe2d_bindings.cpp` (~50 lines, mirroring the class-3 binding's
read-face-class → find class-1 → patch-face_owner_R pattern).

**Backed by:** new mesh-build parameter `*outfall_2d_cells[]` of length
`n_pipe_ends` or `n_outfalls`, indexed by face-creation order.

**Order invariant:** face-creation order is (link_i, from_node → to_node)
per `pipe1d.cu:1411`. Outfall faces are created via the same path as
class-3 (at line 1450), so class-1 entries in `*outfall_2d_cells` align
with that order.

**Where R=-1 stays legitimate:** if a pipe-end is marked outfall but no
2D cell is supplied (host array smaller than the face count), the face
is left at `-1` and continues to exit early at the kernel guard.

### 3.2 Real upload binding for class-5 (junction overflow)

**Where:** new impl `swe2d_pipe1d_upload_junction_overflow_2d_cells` in
`cpp/src/swe2d_bindings.cpp` (~50 lines, same pattern).

**Backed by:** new mesh-build parameter `*junction_2d_cells[]` of
length `n_junction_overflow_faces`.

**Order invariant:** face-creation order for class-5 is the manhole-cell
iteration order at `pipe1d.cu:1486`. The junction overflow entry array
follows manhole-cell index, which is the order the host uploads.

### 3.3 Real upload binding for class-6 (2D-culvert face-flux)

**Where:** new impl `swe2d_pipe1d_upload_culvert_2d_cells` in
`cpp/src/swe2d_bindings.cpp` (~50 lines, same pattern).

**Backed by:** new mesh-build parameter `*culvert_2d_cells[]` of length
`n_culvert_faces`.

**Order invariant:** face-creation order for class-6 follows structure
index ordering at `pipe1d.cu:1651`.

### 3.4 Class-4 (HEC-22 inlet) direct pipe-side write

**Where:** `cpp/src/pipe1d.cu:~2551` (class-4 branch). Add:

```cpp
if (d_A && cell_length && cell_length[L] > 0.0 && Q > 0.0) {
    atomicAdd(&d_A[L], Q * dt / fmax(cell_length[L], 1.0e-3));
}
```

**Why +Q not -Q:** class-4 captures **from 2D into pipe**. `face_F_h[k]`
is already 0 (the patch we land this session goes through `d_A` only,
not the fold path). Sign convention: capture flow into the sump is
positive, mass in the sump grows → add to `d_A`.

### 3.5 Direct-write for class-1, class-5, class-6 (mirrors class-3 sign)

**Where:** in each face kernel branch (class-1, class-5, class-6).
For these, mass direction is determined by the face. Each kernel needs:

```cpp
atomicAdd(&d_A[L], -fh * dt / fmax(cell_length[L], 1.0e-3));
```

Sign per class:

| Class | `fh` direction (HLLC) | Pipe-side sign |
|---|---|---|
| class-3 | positive = L→R = pipe→2D | `-fh` (pipe loses mass) |
| class-1 (outfall) | positive = L→R = pipe→2D (outfall is the 2D exit) | `-fh` |
| class-5 (overflow) | positive = L→R = manhole→2D | `-fh` |
| class-6 (2D-culvert) | positive = L→R = donor→receiver (both 2D) | n/a — both ends are 2D, pipe-side doesn't apply |

Class-6 special case: the existing `swe2d_culvert_face_flux_kernel`
already handles 2D-to-2D coupling via `d_ext_struct_flux_h` (no
pipe-side update needed). The class-6 path inside the unified face
kernel handles pipe1D-internal culverts; this design does NOT add
direct-write for class-6 since the 2D-coupling path is already covered.

## 4. Data flow (unchanged)

1. `swe2d_gpu_step` → lazy-allocate `d_ext_struct_flux_*` → call
   `swe2d_gpu_apply_unified_face_flux` with real 2D arrays
   (already in place from fix #3 + fix #4 this session).
2. Class-N kernel: dispersion → atomicAdd to 2D removal
   (`d_ext_struct_flux_h[R]`) → **direct atomicAdd to pipe
   cell A[L]** (NEW for class-1, 4, 5).
3. `swe2d_update_kernel` consumes `d_ext_struct_flux_h` → updates
   `h[]` for the 2D cell (already in place).
4. Pipe1D godunov step (next call, `apply_native_device_sources`)
   computes fold + godunov with **fresh** `face_F_h` (zeroed by the
   wrapper just before kernel launch). No double-counting.

## 5. Error handling

- **Wiring failures (Group A):** if a host array is shorter than the
  device face count, the binding copies the prefix and leaves the
  remainder at `-1` (already at `-1` placeholder; kernel exits early
  on `if (R < 0)`). Existing behavior, no regression.
- **Lazy allocation:** `swe2d_gpu_alloc_ext_struct_flux(dev, n_cells)`
  is null-safe (`if (!dev || n_cells <= 0) return;`) and idempotent
  (short-circuits at right size). No new failure modes.
- **Direct-write `d_A[L]`:** guarded by
  `if (d_A && cell_length[L] > 0.0)` — same guard as the class-3 fix.
- **No precision decay:** `Q * dt / cell_length` is bounded by CFL;
  Preissmann slot branches keep separate scope and are independent of
  this change.

## 6. Triage of existing failures

Group A | Tests | Wiring gap (R=-1) | Same fix as class-3 |
|---|---|---|---|
| A1 | `test_junction_overflow_to_2d` | class-5 (junction overflow) | new binding 3.2 |
| A2 | `test_outfall_fixed_wse` | class-1 (outfall) | new binding 3.1 |
| A3 | `test_outfall_rating_curve` | class-1 (outfall) | new binding 3.1 |
| A4 | `test_pipe_end_face_coupling_conserves_total_mass` (ERROR) | R=-1 + Python import error in test | new binding 3.1 and diagnostics |

Group B+C | Test | Likely cause |
|---|---|---|
| B  | `test_fully_dynamic_mass_conservation_with_and_without_sub_cells (max_cell_length=0)` | `mcl=0` leads to `n_sub = max(1, ceil(L/0))` → integer-divide-by-zero or pathological A_normalisation. Read the source first (`swe2d_build_pipe1d_mesh`, `swe2d_pipe1d_compute_sub_cell_counts`) to confirm. |
| C  | `test_fully_dynamic_updates_area_and_q` | `cell_A == A_full` exactly; if the godunov update never advances A past A_full under `mcl=10` with non-zero initial depth, the patch this session killed a sub-step somewhere. |

Group D | Test | Likely cause |
|---|---|---|
| D1 | `test_slot_allows_A_above_full` | G2 fix only sets `d_cell_slot_width` to non-zero; the surcharge-slot branch inside `swe2d_pipe1d_godunov_update_kernel` (line ~3050) may still gate on something else (`surcharge_method == SURCHARGE_SLOT` flag, `wMax > 0.0`, etc.). |
| D2 | `test_slot_pressure_equalization` | Same as D1; also requires two pressurised cells for the equalisation to occur. |
| D3 | `test_slot_vs_no_slot_pressurisation_difference` | Same as D1. |

**Triage order** (lowest blast radius first):
1. Read the Godunov kernel + `swe2d_build_pipe1d_mesh` code before
   touching anything.
2. Add class-1/4/5/6 wiring first (Group A and class-4) — these are the
   mechanical fixes proven by the class-3 patch.
3. Re-run the audit. Group D2 tests will probably light up next because
   the surcharge logic is independent of the face-class wiring.
4. Group B and C triage separately; may be RK2 noise gate, RK2 combine
   kernel bug, or `mcl=0` integer-division.

## 7. Component ordering (per wave)

| Wave | What | Files | Gates |
|---|---|---|---|
| W1 | Group A1 (class-5 upload), A2/A3/A4 (class-1 upload), 3.4 (class-4 direct-write), 3.5 (class-5 direct-write) | `swe2d_bindings.cpp`, `pipe1d.cu` mesh-build loop, `pipe1d.cu` class-4 kernel, new regression test in `test_pipe1d_face_indexed_mesh.py` | re-run `test_pipe1d_face_indexed_mesh` |
| W2 | Group B + C — read godunov/RK2 source first, fix root cause | `pipe1d.cu` godunov kernel, possibly `pipe1d.cu` `swe2d_pipe1d_compute_sub_cell_counts` | re-run `test_swe2d_pipe1d` |
| W3 | Group D — read slot-surcharge branch, fix what's still gated | `pipe1d.cu` godunov kernel | re-run `test_swe2d_pipe1d_surcharge` |
| W4 | CLI replay verification (Gate 3 + Gate 4) | nothing new | `pipe_1d_test.json` replay shows non-noise flow |

W1 and W2 can run in parallel only if the godunov doesn't share code
with the face kernels. They share `d_cell_*` arrays and `cell_A`
arrays; the surcharges branch reads `d_cell_slot_width` only. **Recommend
sequential** — W1 lands first, W2/W3 diagnose + fix.

## 8. Testing

Four gates, all required for completion:

**Gate 1 — audit-gate suite** (9 currently-failing tests):

```bash
find . -type d -name __pycache__ -exec rm -rf {} +
mamba run -n qgis_stable env CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13 \
  CMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 \
  /usr/bin/cmake --build build -j$(nproc)
mamba run -n qgis_stable env PYTHONPATH="$PWD:$PWD/build" \
  python3 -m unittest -v \
  tests.test_pipe1d_face_indexed_mesh \
  tests.test_swe2d_pipe1d \
  tests.test_swe2d_pipe1d_surcharge
```

Expected: 0 failures, 0 errors.

**Gate 2 — minimal pipe-end regression test stays green:**

```bash
mamba run -n qgis_stable env PYTHONPATH="$PWD:$PWD/build" \
  python3 -m unittest -v tests.test_pipe_end_surface_coupling
```

Expected: 1 ok.

**Gate 3 — CLI replay with non-noise pipe flow:**

```bash
mamba run -n qgis_stable env PYTHONPATH="$PWD:$PWD/build" \
  python3 -m swe2d.cli replay --replay-file pipe_1d_test.json > /tmp/replay.json

# Parse coupling_results to extract drainage_link_1_flow.
# Verify the value at t≈300 s is ≥ 1 CFS.
```

Verification: `drainage_link_1_flow` (max absolute value over the
6 snapshots) ≥ 1 CFS — vs the 0.0057 CFS pre-fix ceiling. HEC-22
capture at the inlet (cell 1, grate 10×5, crest 930.8) with WSE=935
should produce ~774 CFS weir flow at peak; the 8'×4' box culvert
should carry most of it (300–700 CFS). Pass = max ≥ 1 CFS.

**Gate 4 — class-4 inlet capture regression test:**

```bash
mamba run -n qgis_stable env PYTHONPATH="$PWD:$PWD/build" \
  python3 -m unittest -v tests.test_pipe1d_face_indexed_mesh
```

New test method `test_inlet_capture_from_2d` — 2D cell with h=4, pipe1D
inlet at crest 2 below WSE, 5 steps, expect inlet cell h > 0 after.

## 9. Reference

- Predecessor plan: `docs/pipe1d_face_indexed_refactor_followup_plan.md`
- Original face-indexed refactor: `docs/pipe1d_face_indexed_refactor_plan.md`
- Audit outcome (this session): `docs/AGENT_SESSION_RECOVERY_LOG.md`
- Class-3 fix pattern (proof of concept for Group A): `cpp/src/pipe1d.cu:2413`,
  `cpp/src/swe2d_bindings.cpp:2271-2316`

## 10. Risk register

1. **Class-1 / class-5 face-creation order changes** — the upload
   bindings index faces in the same order the mesh build creates them.
   If a future mesh-build edit reorders face creation, the bindings
   become silently wrong (data lands on the wrong face). Mitigation:
   add a debug-only `assert(host_array_size == n_classN_faces)` at the
   start of each binding for early detection in dev.
2. **Duplicate kernel launch** — class-6 has both the unified-kernel
   path and the legacy `swe2d_culvert_face_flux_kernel`. Today's
   `use_culvert_face_flux` flag is the only thing preventing
   double-counting. We may widen it. Mitigation: leave the flag in
   place; the unified kernel only fires its class-6 branch when
   called with real 2D arrays (which the legacy kernel doesn't).
3. **RK2 zero-accumulate** — the `d_ext_struct_flux_*` is zeroed once
   per `swe2d_gpu_step` call. RK2 calls `swe2d_gpu_step` twice
   (predictor + corrector), so the value is correctly zeroed at the
   start of each stage. But the corrector operates on the predictor's
   output; if the face flux doesn't match the corrector's expectation
   of the 2D state, mass may not balance. Mitigation: Gate 3 (CLI
   replay) catches this end-to-end. If trajectory diverges, diagnose
   in W2/W3.
4. **Slot surcharge happens on `mcl=0`?** — Group B with `mcl=0` may
   crash in `swe2d_pipe1d_compute_sub_cell_counts` (`ceil(0/0)` →
   undefined). Mitigation: gate `surcharge_method=1` SKIP when
   `max_cell_length <= 0`. Small kernel patch; only if the root cause
   is confirmed.

## 11. Out-of-scope (deferred)

- `swe2d_gpu_apply_unified_face_flux` API: keep as `extern` declared
  in `swe2d_gpu.cu`, defined (non-`static`) in `pipe1d.cu`. Long-term
  move: wrap the call in a `pipe1d_apply_face_coupling_to_2d`
  helper inside `pipe1d.cu` and re-`static`-ify; defer until this work
  is stable.
- Verify RK2 stage ordering: only relevant if Gate 3 surfaces
  between-stage mass drift; add a separate spec if needed.
- Test-directory cleanup: pick one of the two pipe-end tests to own
  the space. Defer.
