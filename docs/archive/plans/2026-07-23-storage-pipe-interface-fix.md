---
type: plan
status: complete
created: 2026-07-23
completed: 2026-07-25
---

# Storage→Pipe Interface Fix Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement.

**Goal:** Replace the current HLLC Riemann solver at storage→pipe interfaces (manhole/inlet cells connected to pipe cells via INTERIOR class-0 faces) with a dedicated hydraulic control that correctly passes flow between a storage node and a conduit.

**Architecture:** The mesh builder currently creates INTERIOR (class-0) HLLC faces between storage cells and adjacent pipe cells. The HLLC solver assumes matching cross-sectional geometry and uses a CFL limiter (`A_min * L_avg / dt`) that clamps to zero when one side is dry — permanently blocking flow between series-connected links through an intermediate node. Two approaches are proposed; choose one before implementation.

**Root cause:** The HLLC Riemann solver at INTERIOR class-0 faces operates on `A` (cross-sectional area) and `Q` as primitive variables. When a rectangular storage cell (A = w×h, T = w = 6ft) abuts a circular pipe cell (A = f(h), T = f(h) = D·sin(θ/2)), the vastly different geometry means:
- Wave speeds differ by 10× (c_storage = sqrt(g·A/w) vs c_pipe = sqrt(g·A/T))
- The F_max CFL limiter `fmin(A_L, A_R)·L_avg/dt` goes to zero when EITHER side is dry
- The pressure integral term `P = g·I1(A)` uses different reference frames

**Tech Stack:** C++20 / CUDA, Python 3.12, pybind11

---

## Option A: Dedicated STORAGE_PIPE Face Class

Create a new face class (8 = STORAGE_PIPE) with standard weir/orifice hydraulics.

### Face kernel (class 8, pass 2)

Replace the HLLC branch for storage→pipe faces with:

```
For each STORAGE_PIPE face (owner_L = storage_cell, owner_R = pipe_cell):
  h_storage = cell_y[L] - cell_invert[L]
  h_pipe    = cell_y[R] - cell_invert[R]
  crest     = cell_invert[R]  (pipe invert at the face interface)
  width     = pipe_diameter    (or pipe_top_width at current fill)

  If h_storage > crest AND h_pipe > crest:
    # Both sides submerged — orifice
    head = |cell_y[L] - cell_y[R]|
    A_open = min(storage_surface_area, pipe_cross_section_area)
    Q = Cd * A_open * sqrt(2*g*head)
    direction = sign(cell_y[L] - cell_y[R])
  Elif h_storage > crest:
    # Storage side only — weir (inflow to pipe)
    head = h_storage - crest
    Q = Cw * width * head^1.5
    direction = +1  (storage→pipe)
  Elif h_pipe > crest:
    # Pipe side only — weir (backflow to storage)
    head = h_pipe - crest
    Q = Cw * width * head^1.5
    direction = -1  (pipe→storage)
  Else:
    Q = 0  (both dry)

  face_F_h[k] = direction * Q
  face_F_Q[k] = 0.0  (momentum flux carried by Godunov update)
```

**Face parameters needed on device:**
- Weir coefficient (Cw ≈ 3.33 USC, 1.84 SI) — stored in `face_k_in[face_idx]`
- Orifice coefficient (Cd ≈ 0.65) — stored in `face_k_out[face_idx]`
- Crest elevation = pipe invert at the interface face
- Flow width = pipe diameter (or top width at full)

**Changes:**

| File | Change |
|---|---|
| `cpp/src/pipe1d.cu` — mesh builder | Change storage→pipe faces from class 0 to class 8. Set `face_k_in`, `face_k_out` to weir/orifice coefficients. |
| `cpp/src/pipe1d.cu` — face kernel | Add class-8 branch with weir/orifice equations. |
| `cpp/src/pipe1d.cu` — fold kernel | Class 8 faces need both L and R accumulation (same as class 0). |
| `cpp/src/pipe1d.cu` — godunov update | Storage cells (class 1/2) get Q from the face flux. Ensure momentum update accounts for the face flow. |
| `tests/` | Add test for series links through storage node — verify flow passes at expected rate. |

### Advantages
- Physically correct hydraulics (matches SWMM, HEC-RAS approach)
- No CFL limiter issue (Q is bounded by weir/orifice physics)
- Works for both free-surface and submerged conditions

### Disadvantages
- New code path to maintain
- Need to tune weir/orifice coefficients
- Doesn't account for pipe friction in the interface flux

---

## Option B: Transition Cell

Insert a short auxiliary cell between the storage cell and the first pipe cell that smoothly varies geometry from storage (rectangular) to pipe (circular).

**Layout change:**
```
[storage cell: rect, w×h] —INTERIOR→ [transition: rect→circ] —INTERIOR→ [pipe: circular, D]

transition cell:
  cell_length = 0.5 ft (very short)
  A_full(transition) = average of storage A_full and pipe A_full
  HLLC between transition and pipe uses matching circular geometry
```

**Changes:**

| File | Change |
|---|---|
| `cpp/src/pipe1d.cu` — mesh builder | After creating storage→pipe INTERIOR faces, insert an extra cell between them. Set its geometry to a smooth blend. |
| `cpp/src/pipe1d.cu` — geometry tables | Compute P/T/I1 tables for the transition shape |
| `cpp/src/pipe1d.cu` | No change to face kernel (uses existing HLLC class 0) |
| `tests/` | Verify flow passes at expected rate |

### Advantages
- No new face class needed — reuses existing HLLC solver
- The HLLC solver operates between cells with similar geometry
- The F_max limiter uses the transition cell's cross-section, not the storage cell's

### Disadvantages
- Additional cell per storage→pipe interface increases mesh size
- Transition cell geometry is artificial (doesn't represent real hydraulics)
- May need careful tuning of the transition length and blending function

---

## Recommendation

**Option A** (dedicated weir/orifice face class) is the correct engineering approach. Storage→pipe connections are hydraulically weir/orifice controls in every standard stormwater model (SWMM, HEC-RAS, InfoWorks). The HLLC solver is designed for conduit-to-conduit connections, not node-to-conduit. The transition cell (Option B) would mask the problem without fixing the physics.

---

## Implementation Plan

### Step 1: Revert the F_max floor change

The `A_floor = 1e-6 * max(A_L, A_R)` band-aid in the HLLC CFL limiter should be reverted — it masks the symptom without fixing the physics, and the correct fix is a dedicated face class.

**Files:** `cpp/src/pipe1d.cu` — revert lines around F_max computation in class-0 HLLC branch.

### Step 2: Add STORAGE_PIPE face class (8)

In `swe2d_unified_face_flux_kernel`, add:
```c++
// ── STORAGE_PIPE face (class 8) — weir/orifice between storage cell and pipe cell ──
if (pass == 2 && cls == 8) {
    // Weir/orifice equations
}
```

**Coefficients:**
- Weir: Cw = 1.84 (SI) / 3.33 (USC), stored in `face_k_in`
- Orifice: Cd = 0.65, stored in `face_k_out`
- Crest elevation = pipe invert at the interface

**Files:** `cpp/src/pipe1d.cu` — add class-8 branch.

### Step 3: Update mesh builder

In the storage→pipe face construction at lines 1657-1708:
- Change `face_class_v[face_idx]` from `0` (INTERIOR) to `8` (STORAGE_PIPE)
- Set `face_k_in_v[face_idx]` = weir coefficient (1.84)
- Set `face_k_out_v[face_idx]` = orifice coefficient (0.65)

**Files:** `cpp/src/pipe1d.cu` — lines 1664-1704.

### Step 4: Update fold kernel

Ensure class-8 faces accumulate into both L and R (same as class 0):
```c++
if ((cls == 0 || cls == 8) && R >= 0 && R < n_cells_all) {
    atomicAdd(&cell_flux_h[R], -face_F_h[k]);
}
```

**Files:** `cpp/src/pipe1d.cu` — `swe2d_fold_face_flux_to_cells` kernel.

### Step 5: Update Godunov update kernel

The storage cell (class 1/2) receives flow through the STORAGE_PIPE face. The continuity equation already handles this correctly (A_next = A - dt*flux_A/L). No change needed.

The pipe cell receives flow through the STORAGE_PIPE face. The continuity equation also handles this (A_next = A - dt*(-F)/L = A + dt*F/L). No change needed.

But the momentum flux (F_Q) for the weir/orifice face should be set to 0 — the face only carries mass, not momentum. The momentum in the pipe is handled by the interior HLLC faces.

**Files:** No change needed for Godunov update.

### Step 6: Add tests

Add to `tests/test_pipe1d_solver.py`:

1. **Series links through manhole**: Create 2 links connected through a manhole. Fill upstream link with water. Verify downstream link receives flow at a physically reasonable rate (≥50% of upstream within 100 steps).

2. **Series links through inlet**: Same but with an inlet cell (storage node) at the junction.

3. **Single pipe→storage→pipe conservation**: Verify mass is conserved across the storage→pipe interfaces (total volume = pipe + storage volume is constant in a closed system).

### Step 7: Build, test, commit

```bash
find . -type d -name __pycache__ -exec rm -rf {} +
mamba run -n qgis_stable env CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13 /usr/bin/cmake --build build -j$(nproc)
mamba run -n qgis_stable env PYTHONPATH="$PWD:$PWD/build" python3 -m unittest -v tests.test_pipe1d_solver
git add -A && git commit -m "feat: add STORAGE_PIPE face class with weir/orifice hydraulics"
```

---

## Self-Review

- [ ] Step 1 reverts the band-aid before the real fix
- [ ] Step 2 adds the face kernel branch
- [ ] Step 3 changes the mesh builder to use class 8
- [ ] Step 4 ensures fold works for class 8
- [ ] Step 5 confirms Godunov update needs no change
- [ ] Step 6 adds tests that fail before and pass after
- [ ] All 20+ existing tests still pass
