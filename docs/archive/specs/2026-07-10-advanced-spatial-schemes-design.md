---
type: spec
status: complete
created: 2026-07-10
completed: 2026-07-25
---

# Advanced Spatial Reconstruction Schemes — Design Spec

| | |
|---|---|
| **Spec ID** | `SWE2D-SPEC-2026-07-10` |
| **Status** | Draft |
| **Owner** | SWE2D solver team |
| **Created** | 2026-07-10 |
| **Companion** | [Technical Guide: Advanced Spatial Schemes](../ADVANCED_SPATIAL_SCHEMES.md) |

## 1. Goal

Add three new spatial reconstruction schemes to the SWE2D GPU solver, filling gaps in robustness (Barth-Jespersen limiter), order-efficiency (true 3-sub-stencil WENO3), and high-order-at-low-cost (mapped MP5). This expands the solver from 7 schemes to 9.

## 2. Current State

### 2.1 Existing scheme lineup

From `swe2d/extensions/extension_models.py:17-34` and `cpp/src/swe2d_solver.hpp`:

| Slot | Name | Order | Stencil |
|-----:|------|:-:|:-:|
| 0 | `FV_FIRST_ORDER` | 1st | 1-cell |
| 1 | `FV_MUSCL_FAST` (Superbee) | 2nd | 1-ring |
| 2 | `FV_MUSCL_MINMOD` | 2nd | 1-ring |
| 3 | `FV_MUSCL_MC` | 2nd | 1-ring |
| 4 | `FV_MUSCL_VAN_LEER` | 2nd | 1-ring |
| 6 | `FV_WENO5` | ~3rd | 2-ring |

Slots 5, 7, 8 are absent. Slot 5 was described as "WENO3-like (experimental)" in docs but never implemented.

### 2.2 Reconstruction dispatch (current)

Spatial reconstruction occurs entirely inside the CUDA kernel `swe2d_flux_kernel` in `cpp/src/swe2d_gpu.cu:1989-2231`. The dispatch is a single if/else:

```cuda
if (spatial_scheme == scheme_weno5 && !near_boundary) {
    weno5_reconstruct(lambda captures...)  // scheme 6
} else {
    tvd_reconstruct(lambda captures...)    // schemes 1-4 (and 0 by identity)
}
```

There is **no** Python-level reconstruction dispatch. The `spatial_scheme` integer passes from CLI → `backend.initialize()` → C++ `swe2d_create()` → the kernel.

There is **no** `swe2d/runtime/solver.py` file; CFL/timestep logic lives in `backend.py` and C++ `swe2d_step()`.

### 2.3 Build system

All C++ headers live alongside `.cpp` files in `cpp/src/`. The top-level `CMakeLists.txt` lists GPU sources. No `cpp/include/` directory exists.

### 2.4 Mesh data structures

`MeshResult` (`swe2d/mesh/mesh_models.py`) contains node coordinates, cell connectivity, and 1-ring neighbor CSR (`cell_face_offsets` + `cell_face_nodes`). The C++ `SWE2DMesh` has the same plus `cell_ring2_*` arrays for the current WENO5 2-ring stencil.

Neither Python nor C++ has face-level sub-stencil tables for WENO3 or 5-cell walks for MP5.

## 3. Proposed Schemes

### 3.1 Scheme 5 — `FV_BARTH_JESPERSEN`

**What:** Green-Gauss gradient reconstruction with Barth-Jespersen gradient limiter applied at 1-ring neighbor faces. The limiter scales the gradient uniformly (`chi = min_j chi_ij`) to keep extrapolated face values within the local min/max envelope.

**Order:** 2nd on smooth regions; degrades to 1st isotropically on poor cells (vs. directional clipping of TVD MUSCL).

**Stencil:** 1-ring only — no new mesh data needed.

**Kernel:** One thread per cell. Reads `q`, `grad_x/y`, 1-ring neighbor values. Produces limited `grad_x_lim/grad_y_lim`. These feed into the existing MUSCL face-value extrapolation path.

**CFL:** ≤ 0.8 (same as MUSCL).

### 3.2 Scheme 6 — `FV_WENO3` (replaces current slot 6)

**Breaking change:** Current `FV_WENO5` (slot 6) re-numbers to slot 7.

**What:** True 3-sub-stencil WENO reconstruction per face. Three candidate values come from LSQ fits on sub-stencil lobes (upwind, central, downwind). Nonlinear convex combination via Hu-Shu 1999 triangular-grid WENO weights.

**Order:** 3rd on smooth; degrades to 2nd (central sub-stencil) near discontinuities.

**Stencil:** 1-ring — drops the 2-ring memory cost of current WENO5. Requires new face-level sub-stencil CSR tables built at mesh assembly.

**Kernel:** One thread per interior face. Reads `q`, node coordinates, precomputed sub-stencil indices. Produces `q_face_recon[f]`.

**CFL:** ≤ 0.8.

**Migration:** Old persisted configs with `spatial_scheme=6` must warn and/or auto-migrate to 7.

### 3.3 Scheme 8 — `FV_MP5`

**What:** Suresh-Huynh 1997 Mapped Monotonicity-Preserving scheme adapted to unstructured meshes. Per face: walk 5 cells along the face-normal direction, fit a 4th-degree polynomial for the high-order candidate, compute a TVD fallback, and apply the MP5 mapped limiter (4 cases) to produce the final face value.

**Order:** 4th–5th in smooth regions; TVD fallback near shocks.

**Stencil:** 5-cell 1D walk along face normal — requires precomputed walk table built at mesh assembly. Boundary faces fall back to TVD MUSCL.

**Kernel:** One thread per face. Reads `q` values for 5 cells, MP5 case indicator. Produces `q_face_recon[f]`.

**CFL:** ≤ 0.4 (Suresh-Huynh constraint). Solver must clamp timestep when scheme=8.

## 4. Architecture

### 4.1 Enum layer

```
Python: swe2d/extensions/extension_models.py (SpatialDiscretization IntEnum)
C++:    cpp/src/swe2d_solver.hpp (SWE2DSpatialScheme enum class)
```

Both enums gain three new members. They must stay in sync (same integer values). The C++ enum is the source of truth for the kernel dispatch; the Python enum is used for validation, UI, and CLI.

### 4.2 Reconstruction dispatch (kernel)

Current if/else in `swe2d_flux_kernel` is replaced by a dispatch table:

```cuda
// Per-face reconstruction dispatch
if (spatial_scheme == scheme_weno3) {
    // read precomputed from weno3_face_recon[face]
} else if (spatial_scheme == scheme_mp5) {
    // read precomputed from mp5_face_recon[face]
} else if (spatial_scheme == scheme_weno5 && !near_boundary) {
    weno5_reconstruct(...)  // unchanged
} else if (spatial_scheme != scheme_first_order) {
    // barth_jespersen: limited gradient already computed in separate pass
    // then TVD-style face extrapolation using the limited gradient
    tvd_reconstruct(...)
}
```

New kernels (`barth_jespersen_kernel`, `weno3_kernel`, `mp5_kernel`) each compute face (or cell) values into output arrays. The flux kernel then reads from these arrays rather than doing inline reconstruction.

This separation means:
- New schemes are added without ballooning the flux kernel
- Each kernel can be profiled independently
- The dispatch remains a simple read from precomputed arrays

### 4.3 Kernel launch orchestration

In `cpp/src/swe2d_gpu.cu`, before `swe2d_flux_kernel`:

```
if scheme == 5:  launch barth_jespersen_kernel
if scheme == 6:  launch weno3_kernel
if scheme == 8:  launch mp5_kernel
```

The existing `swe2d_stabilize` (well-balancing) kernel runs after these and before the flux kernel, unchanged.

### 4.4 Mesh assembly extensions

Three new data arrays added to `MeshResult` (Python) and `SWE2DMesh` (C++):

| Array | Shape | Purpose |
|-------|-------|---------|
| `face_stencil_S0_offsets` | `[n_faces + 1]` | Prefix-sum into S0 cells |
| `face_stencil_S0_cells` | variable-length | Upwind-lobe cell indices |
| `face_stencil_S1` | `[2 * n_faces]` | `{owner, neighbor}` pairs |
| `face_stencil_S2_offsets` | `[n_faces + 1]` | Prefix-sum into S2 cells |
| `face_stencil_S2_cells` | variable-length | Downwind-lobe cell indices |
| `face_stencil_5` | `[5 * n_faces]` | `{u2, u1, u, v, v1}` 5-cell walk |
| `face_mp5_case` | `[n_faces]` | MP5 case (1-4), computed at runtime |

Built during mesh assembly in `cpp/src/swe2d_mesh.cpp` after face construction. Boundary faces get empty S0/S2 and fall back to S1-only linear interpolation.

### 4.5 CFL enforcement

In `swe2d/runtime/backend.py`, before launching a step:

```python
_SCHEME_MAX_CFL = {0: 0.8, 1: 0.8, 2: 0.8, 3: 0.8, 4: 0.8,
                   5: 0.8, 6: 0.8, 7: 0.5, 8: 0.4}

def _clamp_cfl(self) -> None:
    scheme = self._spatial_scheme
    max_cfl = _SCHEME_MAX_CFL[scheme]
    if self._cfl > max_cfl:
        logger.warning(f"CFL={self._cfl} > scheme max {max_cfl}, clamping")
        self._cfl = max_cfl
```

### 4.6 File structure

```
swe2d/extensions/extension_models.py      — add enum members 5, 7, 8
swe2d/runtime/backend.py                  — CFL clamping, scheme validation
swe2d/cli/headless_runner.py              — --spatial-scheme 0..8, migration warning
swe2d/cli/batch_runner.py                 — validate_scheme()
swe2d/mesh/mesh_models.py                 — MeshResult: new stencil fields
QML/form_init.py                          — reconstruction scheme combo box

cpp/src/swe2d_solver.hpp                  — SWE2DSpatialScheme: add 5, 7, 8
cpp/src/swe2d_mesh.hpp                    — SWE2DMesh: new stencil arrays
cpp/src/swe2d_mesh.cpp                    — build stencil tables
cpp/src/swe2d_reconstruct.cu (NEW)        — barth_jespersen, weno3, mp5 kernels
cpp/src/swe2d_gpu.cu                      — dispatch table, orchestration
cpp/src/swe2d_gpu.cuh                     — kernel declarations, device helpers
cpp/src/swe2d_bindings.cpp                — expose new MeshResult fields to Python
CMakeLists.txt                            — add swe2d_reconstruct.cu

docs/USER_GUIDE.md                        — document new schemes
docs/SOLVER_ORDER_AND_STENCIL.md          — extend table, add scheme sections
docs/INDEX.md                             — cross-link
CHANGELOG.md                              — add entry

tests/test_swe2d_barth_jespersen_convergence.py   (NEW)
tests/test_swe2d_weno3_convergence.py              (NEW)
tests/test_swe2d_mp5_convergence.py                (NEW)
tests/test_swe2d_poor_mesh_robustness.py           (NEW)
tests/test_face_value_monotonicity.py              (NEW)
tests/test_spatial_scheme_perf.py                  (NEW)
```

## 5. Data Flow

```
User (GUI/CLI)
  → spatial_scheme: int
    → backend.initialize(spatial_discretization=scheme)
      → native_opts["spatial_scheme"] = scheme
        → sweep2d_create(config)  [C++]
          → solver_config.spatial_scheme = scheme
            → mesh_assembly: build stencil tables (schemes 6, 8)
            → swe2d_step:
                1. if scheme ∈ {5}: launch barth_jespersen_kernel → grad_lim
                2. if scheme ∈ {6}: launch weno3_kernel → q_face_recon
                3. if scheme ∈ {8}: launch mp5_kernel → q_face_recon
                4. launch swe2d_stabilize (well-balancing)
                5. launch swe2d_flux_kernel → reads precomputed values
                6. launch swe2d_update (time integration)
```

## 6. Error Handling

| Scenario | Response |
|----------|----------|
| Invalid scheme number (negative or >8) | Raise `ValueError` at layer boundary |
| MP5 with CFL > 0.4 | Warn + clamp to 0.4 |
| Old scheme-6 (WENO5) in CLI | Emit deprecation warning, suggest --spatial-scheme=7 |
| Boundary face in WENO3 kernel | S0/S2 empty → fall back to S1-only linear interpolation |
| Boundary face in MP5 kernel | 5-cell walk fails → fall back to TVD MUSCL |
| Barth-Jespersen chi becomes NaN | Clamp chi ∈ [0, 1], fallback to first-order (chi=0) |
| CUDA out of memory for new stencil arrays | ~30% mesh size increase; document memory budget |

## 7. Testing Strategy

### 7.1 Convergence tests

Manufactured solution on gmsh triangular meshes at increasing refinement levels. Fit `log(error) vs log(1/h)` to extract empirical convergence order.

| Scheme | Target order |
|--------|:-:|
| Barth-Jespersen (5) | ≥ 1.8 |
| WENO3 (6) | ≥ 2.5 |
| MP5 (8) | ≥ 3.5 |

### 7.2 Robustness tests

- Barth-Jespersen on stretched-quad + sliver-triangle mesh: no NaN, no oscillation
- WENO3 smooth-extrema recovery: peak height preserved within 1%
- MP5 shock capturing on 1D dam-break over triangular mesh: L1 error within 10% of TVD-MUSCL
- Mixed-element (quad-tri interface) mesh: no artifacts at element boundaries

### 7.3 Monotonicity envelope

For each scheme, on a manufactured solution with extrema, every face value must satisfy min(q_i, q_j) ≤ q_hat_ij ≤ max(q_i, q_j).

### 7.4 Performance benchmarks

On 100k-triangle mesh, target speed relative to first-order:

| Scheme | Target ratio |
|--------|:-:|
| 5 (Barth-Jespersen) | ≥ 0.85× |
| 6 (WENO3) | ≥ 0.65×; must exceed current scheme 6 |
| 8 (MP5) | ≥ 0.75× |

### 7.5 Regression

Full existing `pytest` suite must pass with no new failures after all changes.

## 8. Rollout

### Phase order (dependency-driven)

1. **Foundation** — Enum entries, build system, mesh assembly extensions, kernel stubs
2. **Scheme 5 (Barth-Jespersen)** — Simplest (no new mesh data), validates dispatch pattern
3. **Scheme 6 (WENO3)** — Uses new sub-stencil data, breaking scheme renumber
4. **Scheme 8 (MP5)** — Uses 5-cell walk data, CFL enforcement
5. **Tests & Verification** — All convergence/robustness/perf tests
6. **GUI/CLI/Docs** — User-facing polish

## 9. Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| MPI instability near contact discontinuities | Use MP5-CF (cap-fitted) variant as default |
| Scheme-6 renumber breaks persisted configs | Auto-migration on load + warning log |
| Barth-Jespersen divergence on cyclic meshes | Bound chi ∈ [0,1], NaN → chi=0 |
| New CSR arrays bloat mesh ~30% | Document memory budget; flag to skip for non-6/8 runs |
| WENO3 LSQ per sub-stencil slow on GPU | Precompute QR; benchmark before merge |

## 10. References

- Barth & Jespersen (1989). *Upwind schemes on unstructured meshes.* AIAA 89-0366.
- Suresh & Huynh (1997). *Accurate monotonicity-preserving schemes.* JCP 136, 83–99.
- Hu & Shu (1999). *WENO on triangular meshes.* JCP 150, 97–127.
- [ADVANCED_SPATIAL_SCHEMES.md](../ADVANCED_SPATIAL_SCHEMES.md) — Full mathematical development.

## 11. Open Questions

1. Should the current WENO5 be dropped or kept as scheme 7? **Decision: keep as 7** — highest accuracy option, supports existing production runs.
2. Should MP5 use MP5-CF variant by default? **Decision: yes** — avoids contact-discontinuity instability; document as deliberate choice.
3. Build stencil tables always or only when selected scheme needs them? **Decision: build always** — cheap on host, negligible memory overhead vs. complexity of conditional build path.
