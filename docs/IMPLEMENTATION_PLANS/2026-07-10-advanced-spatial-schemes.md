# Implementation Plan: Advanced Spatial Reconstruction Schemes

| | |
|---|---|
| **Plan ID** | `SWE2D-SP-2026-07-10` |
| **Status** | Draft |
| **Owner** | SWE2D solver team |
| **Created** | 2026-07-10 |
| **Target branch** | `feature/advanced-spatial-schemes` |
| **Estimated scope** | 3 schemes × ~1 kernel each + tests + GUI/CLI plumbing |

> Companion reading: [Technical guide: Advanced Spatial Schemes](../ADVANCED_SPATIAL_SCHEMES.md) — the math, references, and per-scheme behavior. This plan is the **how**; the tech guide is the **why**.

---

## 1. Context and Motivation

The current solver ships seven spatial schemes (`SpatialDiscretization` enum, [`extension_models.py:17-34`](../SWE2D_GPU_ARCHITECTURE_REPORT.md)):

| # | Name | Order | Stencil | Cost |
|--:|------|:-:|:-:|:-:|
| 0 | `FV_FIRST_ORDER` | 1st | 1-cell | 1.0× |
| 1 | `FV_MUSCL_FAST` (Superbee) | 2nd | 1-ring | 1.1× |
| 2 | `FV_MUSCL_MINMOD` | 2nd | 1-ring | 1.1× |
| 3 | `FV_MUSCL_MC` | 2nd | 1-ring | 1.1× |
| 4 | `FV_MUSCL_VAN_LEER` | 2nd | 1-ring | 1.1× |
| 6 | `FV_WENO5` (2-ring LSQ) | ~3rd | 2-ring | 2.5× |

Three gaps identified during review (see [ALS_NOTES2.md](../../reference/ALS_NOTES2.md)):

1. **Enum slot 5 is vacant.** Documentation describes a "WENO3-like (experimental)" scheme but no implementation exists.
2. **No scheme is robust on poor-quality meshes.** TVD MUSCL limiters clip directionally and produce artifacts at sliver cells, mixed-element boundaries, and wet–dry interfaces.
3. **WENO5 (scheme 6) costs 2.5× but achieves only ~3rd-order** on triangles because the underlying reconstruction is 2nd-order linear LSQ plus WENO weighting. There is no scheme that achieves >3rd-order without paying the 2-ring memory cost.

This plan adds **three new schemes** that fill these gaps:

| New # | Name | Order | Stencil | Cost | Primary win |
|--:|------|:-:|:-:|:-:|---|
| 5 | `FV_BARTH_JESPERSEN` | 2nd | 1-ring | 1.2× | Robustness on poor meshes, mixed-element, urban drainage |
| 6 (replace stub) | `FV_WENO3` (true 3-sub-stencil) | ~3rd | 1-ring | 1.5× | Drops 2-ring cost vs current scheme 6 |
| 8 (new slot) | `FV_MP5` (mapped monotonicity-preserving) | ~4th | 5-cell 1D walk | 1.3× | Higher-order at *lower* cost than WENO5 |

---

## 2. Scope

### In scope

- Three new CUDA reconstruction kernels and their Python enum entries.
- Test suite extensions: convergence, robustness on poor meshes, monotonicity envelopes.
- GUI/QML dropdown update to expose the new schemes.
- CLI flag passthrough for headless runner.
- Update to [SOLVER_ORDER_AND_STENCIL.md](../SOLVER_ORDER_AND_STENCIL.md) and [USER_GUIDE.md](../USER_GUIDE.md).

### Out of scope (deferred)

- True WENO5 (the textbook 5-cell-stencil WENO on 1D grid) — kept as scheme 7, no change.
- DG / spectral element methods — different solver family, future RFC.
- AMR / hp-refinement — orthogonal workstream.
- Coupled drainage-solver scheme changes — orthogonal.

---

## 3. Scheme Specifications

### 3.1 Scheme 5 — `FV_BARTH_JESPERSEN` (Barth-Jespersen gradient limiter)

**Math** — see [tech guide §3](../ADVANCED_SPATIAL_SCHEMES.md#3-fv_barth_jespersen-barth-jespersen-gradient-limiter).

**Stencil:** 1-ring of cell-centers around each face neighbor (already in `cell_face_offsets` / `cell_face_nodes`).

**Algorithm:**
1. Compute Green-Gauss gradient $\nabla q_i$ over the 1-ring (already implemented in current MUSCL path).
2. For each face neighbor $j$:
   - Compute extrapolated value $q^*_j = q_i + \nabla q_i \cdot (\vec{x}_j - \vec{x}_i)$.
   - Compute $\chi_{ij}$ such that $q_i + \chi_{ij} \nabla q_i \cdot (\vec{x}_j - \vec{x}_i) \in [\min(q_i, q_j), \max(q_i, q_j)]$.
3. $\chi_i = \min_j \chi_{ij}$, $\nabla \tilde{q}_i = \chi_i \nabla q_i$.
4. Use $\nabla \tilde{q}_i$ for face-value extrapolation as in current MUSCL.

**Properties:**
- Order: 2nd on smooth, 1st on poor cells — but **isotropic** degradation.
- Wet–dry: clipping envelope auto-tightens at dry cells (no special-case).
- Mixed-element: works identically on triangles, quads, polygons.
- CFL: same as MUSCL (≤ 0.8).

**Enum entry** (in [extension_models.py:17-34](../../swe2d/extensions/extension_models.py)):
```python
FV_BARTH_JESPERSEN = 5   # LSQ gradient + Barth-Jespersen face-value limiter
```

### 3.2 Scheme 6 (replaces current) — `FV_WENO3` (true 3-sub-stencil WENO)

**Math** — see [tech guide §4](../ADVANCED_SPATIAL_SCHEMES.md#4-fv_weno3-true-3-sub-stencil-weno).

**Stencil:** For face $e_{ij}$:
- $S_0$: cells $k$ adjacent to $i$ but not $j$ (upwind lobe)
- $S_1$: $\{i, j\}$ (central pair)
- $S_2$: cells $k$ adjacent to $j$ but not $i$ (downwind lobe)

Each sub-stencil does a small LSQ fit and evaluates at the face midpoint → 3 candidate values $q^{(0)}_{ij}, q^{(1)}_{ij}, q^{(2)}_{ij}$.

Smoothness indicators $\beta_k$ = residual of LSQ fit within sub-stencil $S_k$.

Weights (Hu-Shu adapted): $d_k = (0.1, 0.6, 0.3) \cdot N_k / \sum_m N_m d_m$.

Nonlinear weights: $w_k = \alpha_k / \sum_j \alpha_j$, $\alpha_k = d_k / (\varepsilon + \beta_k)^2$.

Reconstruction: $\hat{q}_{ij} = \sum_k w_k q^{(k)}_{ij}$.

**Properties:**
- Order: 3rd on smooth, 2nd on discontinuities.
- Stencil cost: 1-ring only — drops the 2-ring memory of current scheme 6.
- Smooth-extrema recovery: yes.
- Wet–dry: weights $\to 0$ on sub-stencils crossing dry cells naturally.

**Enum entry:**
```python
FV_WENO3 = 6   # True 3-sub-stencil WENO (1-ring, 3rd-order)
FV_WENO5 = 7   # 5-sub-stencil WENO via 2-ring LSQ (was 6)
```

> **Breaking change:** current scheme 6 becomes 7. Update CLI flag mapping and any persisted run configs. Migration: `FV_WENO5 (old=6) → FV_WENO5 (new=7)`.

### 3.3 Scheme 8 (new slot) — `FV_MP5` (Suresh-Huynh Mapped Monotonicity-Preserving)

**Math** — see [tech guide §5](../ADVANCED_SPATIAL_SCHEMES.md#5-fv_mp5-suresh-huynh-mapped-monotonicity-preserving).

**Stencil:** 5-cell 1D walk along the face normal. For face $e_{ij}$, walk the cell-to-cell graph upwind from $u$ for two hops and downwind from $v$ for two hops. Result: 5 cell-center values $\{f_{u-2}, f_{u-1}, f_u, f_{u+1}, f_{u+2}\}$.

**Algorithm** (per face):
1. Build 5-cell stencil.
2. Compute high-order value $f^{HO}$ from 4th-degree polynomial fit through 5 cells.
3. Compute TVD bound $f^{TVD}$ from neighboring cell differences (3rd-order TVD fallback).
4. Compute clip envelope: $f^{min}, f^{max}$ from $\min$/$\max$ of adjacent pairs.
5. Apply MP5 mapping (4 cases from Suresh-Huynh 1997, §3.2):
   - `fcase = 1`: $f^{HO} \in [f^{min}, f^{max}]$ → use $f^{HO}$ (high-order).
   - `fcase = 2, 3, 4`: apply mapped compression toward $f^{TVD}$.
6. CFL ≤ 0.4 required (vs ≤ 0.8 for TVD).

**Properties:**
- Order: 4th–5th on smooth flow (5th in 1D, 4th on polygons).
- Per-face cost: cheaper than WENO5 (no β computation, no weight sums).
- Shock handling: explicit clip to TVD envelope — same robustness as TVD MUSCL.
- Stencil: only needs 1D-style cell-graph walk; works on arbitrary polygons.

**Enum entry:**
```python
FV_MP5 = 8   # Suresh-Huynh Mapped Monotonicity-Preserving (5-cell walk)
```

**CFL handling:** MP5 requires CFL ≤ 0.4. Add `cfl_max_mp5 = 0.4` to solver config and clamp timestep in `SWE2DBackend.step()` when scheme == 8.

---

## 4. File-Level Changes

### 4.1 Python layer

| File | Change |
|------|--------|
| `swe2d/extensions/extension_models.py` | Add `FV_BARTH_JESPERSEN = 5`, renumber `FV_WENO5 = 7`, add `FV_MP5 = 8`. Update docstring. |
| `swe2d/extensions/__init__.py` | Re-export new enum members if needed. |
| `swe2d/runtime/backend.py` | Add scheme-dispatch branches in `SWE2DBackend._reconstruct_face_value()`. |
| `swe2d/runtime/solver.py` | Pass `scheme` to backend; enforce CFL ≤ 0.4 for `FV_MP5`. |
| `swe2d/cli/headless_runner.py` | CLI flag `--spatial-scheme` accept 5, 6, 7, 8. |
| `swe2d/cli/batch_runner.py` | Update scheme validation in `validate_scheme()`. |
| `swe2d/workbench/studio.py` | Update `reconstruction_combo` (QGIS dropdown) entries. |

### 4.2 CUDA layer

| File | Change |
|------|--------|
| `cpp/include/swe2d_spatial_scheme.h` (new) | Enum mirror of Python: `FV_BARTH_JESPERSEN`, `FV_WENO3`, `FV_WENO5`, `FV_MP5`. |
| `cpp/include/swe2d_mesh.h` | Add `n_cells_in_1_ring[i]` accessor (precomputed count) for kernel launch bounds. |
| `cpp/src/swe2d_reconstruct.cu` (new) | Three kernels: `barth_jespersen_kernel`, `weno3_kernel`, `mp5_kernel`. |
| `cpp/src/swe2d_gpu.cu` | Replace `case 6:` spatial reconstruction with dispatch table. Current WENO5 path becomes `case 7:`. |
| `cpp/CMakeLists.txt` | Add `swe2d_reconstruct.cu` to GPU sources. |

### 4.3 GUI layer

| File | Change |
|------|--------|
| `QML/form_init.py` | Update reconstruction scheme combo entries. |
| `docs/USER_GUIDE.md` | Document scheme 5, 6 (true WENO3), 8 in the Spatial Discretization section. |
| `docs/SOLVER_ORDER_AND_STENCIL.md` | Major rewrite — extend table, add sections for each new scheme. |
| `docs/INDEX.md` | Link the new plan + tech guide. |

---

## 5. Kernel Implementation Detail

### 5.1 `barth_jespersen_kernel` — scheme 5

**Launch:** `<<<(n_cells + 255) / 256, 256>>>` — one thread per cell.

**Memory reads per thread:**
- $q_i$ (1 read)
- $\nabla q_i$ (2 reads, precomputed by existing Green-Gauss pass)
- 1-ring neighbor values $q_j$, coords $\vec{x}_j$ via `cell_face_offsets` / `cell_face_nodes` (variable-length loop, up to ~12 reads)

**Pseudocode:**
```cuda
__global__ void barth_jespersen_kernel(
    const real* __restrict__ q,           // [n_cells]
    const real* __restrict__ grad_x,      // [n_cells]
    const real* __restrict__ grad_y,      // [n_cells]
    const real* __restrict__ node_x,      // [n_nodes]
    const real* __restrict__ node_y,      // [n_nodes]
    const int*  __restrict__ cell_face_offsets,
    const int*  __restrict__ cell_face_nodes,
    const int    n_cells,
    real*        __restrict__ grad_x_lim, // [n_cells] output
    real*        __restrict__ grad_y_lim)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_cells) return;

    const real qi  = q[i];
    const real gx  = grad_x[i];
    const real gy  = grad_y[i];

    // Cell-center coords (precomputed centroid table)
    const real xi = cell_centroid_x[i];
    const real yi = cell_centroid_y[i];

    real chi = 1.0;
    const int start = cell_face_offsets[i];
    const int end   = cell_face_offsets[i + 1];

    for (int k = start; k < end; ++k) {
        const int j = cell_face_nodes[k];
        const real qj = q[j];
        const real xj = cell_centroid_x[j];
        const real yj = cell_centroid_y[j];

        const real dx = xj - xi;
        const real dy = yj - yi;
        const real q_face = qi + gx * dx + gy * dy;

        const real q_min = fminf(qi, qj);
        const real q_max = fmaxf(qi, qj);

        real chi_k = 1.0;
        if (q_face > q_max && q_face != qi) {
            chi_k = (q_max - qi) / (q_face - qi);
        } else if (q_face < q_min && q_face != qi) {
            chi_k = (qi - q_min) / (qi - q_face);
        }
        chi = fminf(chi, chi_k);
    }

    grad_x_lim[i] = chi * gx;
    grad_y_lim[i] = chi * gy;
}
```

**Register budget:** ~12 registers/thread (fits easily).
**Shared memory:** none.
**Warp divergence:** mild — the `if` branches are typically aligned across a warp (most cells see similar solution values).

### 5.2 `weno3_kernel` — scheme 6

**Launch:** `<<<(n_faces + 255) / 256, 256>>>` — one thread per interior face.

**Sub-stencil construction:** precompute three neighbor lists per face during mesh assembly:
- `face_stencil_S0[f]`, `face_stencil_S1[f]` (= edge owners), `face_stencil_S2[f]`

**Pseudocode:**
```cuda
__global__ void weno3_kernel(
    const real* __restrict__ q,            // [n_cells]
    const real* __restrict__ node_x,       // [n_nodes]
    const real* __restrict__ node_y,
    const real* __restrict__ face_mid_x,   // [n_faces]
    const real* __restrict__ face_mid_y,
    const int*  __restrict__ face_stencil_S0_offsets,
    const int*  __restrict__ face_stencil_S0_cells,
    const int*  __restrict__ face_stencil_S1,   // {owner, neighbor}
    const int*  __restrict__ face_stencil_S2_offsets,
    const int*  __restrict__ face_stencil_S2_cells,
    const real   d_weights[3],             // (0.1, 0.6, 0.3)
    const real   epsilon,                  // 1e-6
    real*        __restrict__ q_face_recon) // [n_faces] output
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= n_faces) return;

    const real xf = face_mid_x[f];
    const real yf = face_mid_y[f];

    // S1: central pair {owner, neighbor}
    const int i = face_stencil_S1[2*f + 0];
    const int j = face_stencil_S1[2*f + 1];

    // Three candidate reconstructions via small LSQ fits per sub-stencil
    real q_cand[3];

    // S0: upwind lobe
    {
        const int s = face_stencil_S0_offsets[f];
        const int e = face_stencil_S0_offsets[f + 1];
        const int n = e - s;
        // Solve 2x2 LSQ for (a, bx, by) s.t. q[k] ≈ a + bx*x[k] + by*y[k]
        // then q_cand[0] = a + bx*xf + by*yf
        q_cand[0] = lsq2d_evaluate(q, node_x, node_y, xf, yf,
                                   face_stencil_S0_cells + s, n);
    }
    // S1: central pair (linear interp is exact for 2 points)
    {
        const real qi = q[i], qj = q[j];
        const real xi = node_x[i], yi = node_y[i];
        const real xj = node_x[j], yj = node_y[j];
        // 1D linear along the i→j line evaluated at face midpoint
        const real t  = hypotf(xf - xi, yf - yi) /
                        hypotf(xj - xi, yj - yi);
        q_cand[1] = qi + t * (qj - qi);
    }
    // S2: downwind lobe (mirror of S0)
    {
        const int s = face_stencil_S2_offsets[f];
        const int e = face_stencil_S2_offsets[f + 1];
        const int n = e - s;
        q_cand[2] = lsq2d_evaluate(q, node_x, node_y, xf, yf,
                                   face_stencil_S2_cells + s, n);
    }

    // Smoothness indicators: LSQ residual within each sub-stencil
    real beta[3];
    beta[0] = lsq2d_residual(q, node_x, node_y,
                             face_stencil_S0_cells + face_stencil_S0_offsets[f],
                             face_stencil_S0_offsets[f+1] - face_stencil_S0_offsets[f]);
    beta[1] = (q[i] - q[j]) * (q[i] - q[j]);
    beta[2] = lsq2d_residual(q, node_x, node_y,
                             face_stencil_S2_cells + face_stencil_S2_offsets[f],
                             face_stencil_S2_offsets[f+1] - face_stencil_S2_offsets[f]);

    // Nonlinear weights
    real alpha[3];
    real alpha_sum = 0.0;
    for (int k = 0; k < 3; ++k) {
        alpha[k] = d_weights[k] / ((epsilon + beta[k]) * (epsilon + beta[k]));
        alpha_sum += alpha[k];
    }

    real q_recon = 0.0;
    for (int k = 0; k < 3; ++k) {
        q_recon += (alpha[k] / alpha_sum) * q_cand[k];
    }
    q_face_recon[f] = q_recon;
}
```

**Helper kernels (called per-sub-stencil):**
- `lsq2d_evaluate`: solves 2×2 normal equations $\mathbf{A}^T\mathbf{A}\mathbf{x} = \mathbf{A}^T\mathbf{b}$ and evaluates at target point.
- `lsq2d_residual`: returns sum-of-squares residual of the same fit.

Both can be inlined or extracted to `__device__` helpers in the same file.

**Register budget:** ~24 registers/thread (3 candidates + 3 betas + LSQ scratch).
**Shared memory:** none.
**Precomputation:** `face_stencil_S0`, `face_stencil_S2` must be built once during mesh assembly — see §6.2.

### 5.3 `mp5_kernel` — scheme 8

**Launch:** `<<<(n_faces + 255) / 256, 256>>>`.

**Stencil:** 5-cell 1D walk along face normal. Build during mesh assembly:

```cuda
// face_stencil_5[f] = {u2, u1, u, v, v1} where u=upwind cell, v=downwind cell
__device__ void mp5_stencil_walk(
    int face_id,
    const CSRGraph& g,
    int* out_5_cells)
{
    // u2 = neighbor-of-neighbor upwind (2 hops)
    // u1 = immediate upwind neighbor (1 hop)
    // u  = upwind cell (= edge owner where F·n < 0)
    // v  = downwind cell
    // v1 = immediate downwind neighbor
    // v2 = neighbor-of-neighbor downwind (2 hops)
    // ...
}
```

For **interior faces**, walk succeeds. For **boundary faces**, fall back to one-sided 5-cell walk or to TVD MUSCL.

**Pseudocode (per face, after stencil walk):**
```cuda
__global__ void mp5_kernel(
    const real* __restrict__ q,           // [n_cells]
    const real*  q5[5],                   // {f_{u-2}, f_{u-1}, f_u, f_{u+1}, f_{u+2}}
    const int*   fcase_in,                // [n_faces] precomputed case (1–4) from mesh walk
    real*        __restrict__ q_face_recon)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= n_faces) return;

    const real fm2 = q5[0][f], fm1 = q5[1][f], f0 = q5[2][f];
    const real fp1 = q5[3][f], fp2 = q5[4][f];

    // High-order 4th-degree polynomial fit at the face
    const real f4 = (1.0/12.0) * ( 3.0*fm2 - 25.0*fm1 + 150.0*f0
                                  - 75.0*fp1 + 15.0*fp2 + ... );
    // (Coefficients from Suresh-Huynh 1997, §3.2 — see tech guide)

    // TVD bound: 3rd-order TVD value at the face
    const real f_tvd = f0 + 0.5 * minmod(fp1 - f0, f0 - fm1);

    // Clip envelope from adjacent pairs
    const real f_min = min(min(fm1, f0), fp1);
    const real f_max = max(max(fm1, f0), fp1);

    // MP5 mapped compression
    real f_mp5;
    switch (fcase_in[f]) {
        case 1:
            f_mp5 = f4;
            if (f4 < f_min) f_mp5 = f_min;
            if (f4 > f_max) f_mp5 = f_max;
            break;
        case 2:
            // mapped toward f_tvd, see tech guide §5
            f_mp5 = f_tvd + map2(f4 - f_tvd, f_min - f_tvd, f_max - f_tvd);
            break;
        case 3:
            f_mp5 = f_tvd + map3(f4 - f_tvd, f_min - f_tvd, f_max - f_tvd);
            break;
        case 4:
            f_mp5 = f_tvd;
            break;
        default:
            f_mp5 = f_tvd;
    }
    q_face_recon[f] = f_mp5;
}
```

**Register budget:** ~16 registers/thread.
**Shared memory:** none.
**Precomputed:** `face_stencil_5` (5 cells per face) — built once during mesh assembly, see §6.3.

---

## 6. Mesh Assembly Extensions

### 6.1 Current CSR topology (verified)

From [`mesh_models.py:82-84`](../../swe2d/mesh/mesh_models.py):
- `cell_face_offsets`: prefix-sum, length `n_cells + 1`
- `cell_face_nodes`: flattened neighbor-cell ids per cell

This gives 1-ring neighbors per cell directly. **No change needed** for scheme 5 (Barth-Jespersen).

### 6.2 New: face sub-stencil CSR for WENO3

Add to `MeshResult`:
```python
face_stencil_S0_offsets: np.ndarray  # [n_faces + 1]
face_stencil_S0_cells:   np.ndarray  # variable-length upwind-lobe cells
face_stencil_S1:         np.ndarray  # [2 * n_faces] = {owner, neighbor}
face_stencil_S2_offsets: np.ndarray  # [n_faces + 1]
face_stencil_S2_cells:   np.ndarray  # variable-length downwind-lobe cells
```

**Build algorithm** (in `swe2d_build_mesh_poly` after current face construction):
```
for each face f with owner i, neighbor j:
    S0 = []  # cells adjacent to i, excluding j
    for k in cell_face_nodes[cell_face_offsets[i]:cell_face_offsets[i+1]]:
        if k != j: S0.append(k)
    S2 = []  # cells adjacent to j, excluding i
    for k in cell_face_nodes[cell_face_offsets[j]:cell_face_offsets[j+1]]:
        if k != i: S2.append(k)
    emit S0, S2 for face f
```

Boundary faces (one-sided): emit empty S0 or S2; kernel falls back to S1-only linear interpolation.

### 6.3 New: face 5-cell stencil for MP5

Add to `MeshResult`:
```python
face_stencil_5: np.ndarray  # [5 * n_faces] = {u2, u1, u, v, v1}
face_mp5_case:  np.ndarray  # [n_faces] ∈ {1,2,3,4} — which MP5 case applies
```

**Build algorithm** (host-side, after WENO3 sub-stencils exist):
```
for each face f with owner i, neighbor j:
    # Determine upwind (u) and downwind (v) from current flow F·n
    # For initial construction, use mesh-edge convention: u=i, v=j
    u1 = first cell in S0 if any, else i (1-hop fallback)
    u2 = first cell in neighbor-of-u1 if any, else u1 (2-hop fallback)
    v1 = first cell in S2 if any, else j
    # (v2 not stored; MP5 needs only 5 cells)
    emit {u2, u1, u=i, v=j, v1}
    case = 1  # default; mesh-assembly doesn't know solution yet
```

The `case` value is re-evaluated at runtime in the kernel based on $f^{HO}$ relative to the clip envelope — but the **stencil structure** is fixed at assembly time.

---

## 7. CFL Handling

| Scheme | CFL limit | Source |
|--------|:-:|--------|
| 0–5 (current + Barth-Jespersen) | ≤ 0.8 | existing |
| 6 (WENO3) | ≤ 0.8 | same as TVD (WENO weights $\to 0$ on crossing) |
| 7 (WENO5, current scheme 6) | ≤ 0.5 | existing |
| 8 (MP5) | ≤ 0.4 | Suresh-Huynh 1997 |

**Implementation:** in `SWE2DBackend.step()`:
```python
def max_cfl_for_scheme(scheme: int) -> float:
    return {
        0: 0.8, 1: 0.8, 2: 0.8, 3: 0.8, 4: 0.8,
        5: 0.8,
        6: 0.8,
        7: 0.5,
        8: 0.4,
    }[scheme]
```

CLI flag `--max-cfl` clamps to `min(user_value, scheme_max)`.

---

## 8. Test Plan

### 8.1 Convergence tests (`tests/`)

| Test | What | Acceptance |
|------|------|------------|
| `test_swe2d_barth_jespersen_convergence.py` | 2nd-order convergence on smooth manufactured solution | $L_2$ order ≥ 1.8 on refined meshes |
| `test_swe2d_weno3_convergence.py` | 3rd-order convergence | $L_2$ order ≥ 2.5 |
| `test_swe2d_mp5_convergence.py` | 4th-order convergence | $L_2$ order ≥ 3.5 |

Mirror structure of existing [`test_swe2d_weno5_convergence.py`](../../tests/test_swe2d_weno5_convergence.py).

### 8.2 Robustness tests

| Test | What | Acceptance |
|------|------|------------|
| `test_barth_jespersen_poor_mesh.py` | Run on stretched-quad + sliver-triangle mesh | No NaN, no oscillation near boundaries, max \|h\| < 5× analytical |
| `test_weno3_extrema_recovery.py` | Manufactured smooth peak | Peak height preserved to within 1% |
| `test_mp5_shock_capturing.py` | 1D dam-break on triangular mesh | $L_1$ error comparable to TVD-MUSCL within 10% |
| `test_mixed_element_schemes.py` | Quad-tri interface mesh | No artifacts at element boundary for any scheme |

### 8.3 Monotonicity envelope

Add `test_face_value_monotonicity.py`:
- For each scheme, on a manufactured solution with extrema:
  - face value $\hat{q}_{ij}$ must satisfy $\min(q_i, q_j) \le \hat{q}_{ij} \le \max(q_i, q_j)$.
- Asserts the **local monotonicity preservation** property.

### 8.4 Performance benchmarks

| Scheme | Mesh | Target speedup vs Scheme 0 |
|--------|------|:-:|
| 5 (Barth-Jespersen) | 100k tri | ≥ 0.85× |
| 6 (true WENO3) | 100k tri | ≥ 0.65× (WENO3 cost) |
| 6 (true WENO3) | 100k tri | **>** current scheme 6 (WENO5) |
| 8 (MP5) | 100k tri | ≥ 0.75× |

Add to `tests/_perf_benchmarks.py` (or new `test_spatial_scheme_perf.py`).

---

## 9. GUI / QML Changes

### 9.1 `QML/form_init.py` reconstruction combo

Current entries (5): 1st-order, MUSCL Fast, MUSCL MinMod, MUSCL MC, MUSCL Van Leer, WENO3-like, WENO5.

New entries (8):
1. First-order (0)
2. MUSCL Fast (1)
3. MUSCL MinMod (2)
4. MUSCL MC (3)
5. MUSCL Van Leer (4)
6. **Barth-Jespersen (5)** — NEW
7. **WENO3 — 3-sub-stencil (6)** — NEW (was "WENO3-like experimental")
8. WENO5 — 2-ring LSQ (7) — was (6)
9. **MP5 — Mapped MP (8)** — NEW

### 9.2 Studio help text

For each new scheme, add a tooltip explaining:
- Order
- Best use case (riverine / floodplain / urban drainage)
- CFL constraint
- Mesh-quality sensitivity

---

## 10. CLI / Headless Changes

### 10.1 `headless_runner.py`

Current `--spatial-scheme` accepts 0–6. Update to accept 0–8 with validations:
```
--spatial-scheme {0..8}
```

### 10.2 `batch_runner.py`

Update `validate_scheme()` to accept 5, 6, 7, 8.

### 10.3 Backwards compatibility

- `--spatial-scheme 6` → maps to new `FV_WENO3` (was `FV_WENO5`).
- `--spatial-scheme 7` → maps to new `FV_WENO5` (was invalid).
- Add `--spatial-scheme-old` flag for explicit old-numbering compatibility during deprecation period.

### 10.4 Migration warning

On encountering old-numbered scheme 6, emit:
```
WARNING: spatial-scheme=6 was FV_WENO5; now it is FV_WENO3 (true 3-sub-stencil).
To keep WENO5, use --spatial-scheme=7.
```

---

## 11. Documentation Updates

| File | Change |
|------|--------|
| `docs/SOLVER_ORDER_AND_STENCIL.md` | Major rewrite — extend table, add per-scheme sections, add "Accuracy Ceiling" updates. Reference tech guide for math. |
| `docs/USER_GUIDE.md` | Update "Spatial Discretization" section with new schemes. |
| `docs/SWE2D_GPU_ARCHITECTURE_REPORT.md` | Add new schemes to architecture summary. |
| `docs/ADVANCED_SPATIAL_SCHEMES.md` | NEW — full technical reference (companion to this plan). |
| `docs/IMPLEMENTATION_PLANS/2026-07-10-advanced-spatial-schemes.md` | This file. |
| `docs/INDEX.md` | Add links to new docs. |
| `CHANGELOG.md` | Add entry under unreleased. |

---

## 12. Rollout Plan

### Phase 1: Scheme 5 — Barth-Jespersen (1 week)

| Day | Task |
|----:|------|
| 1 | Python enum + backend dispatch |
| 2 | CUDA kernel `barth_jespersen_kernel` |
| 3 | Convergence test (`test_swe2d_barth_jespersen_convergence.py`) |
| 4 | Poor-mesh robustness test |
| 5 | GUI/QML plumbing |
| 6 | CLI flag + docs |
| 7 | Code review + merge |

### Phase 2: Scheme 6 — True WENO3 (2 weeks)

> **Breaking change:** existing scheme 6 (`FV_WENO5`) becomes scheme 7. Coordinate with API/CLI consumers before merge.

| Day | Task |
|----:|------|
| 1–2 | Mesh-assembly extension: build `face_stencil_S0` / `S2` CSR |
| 3–4 | CUDA kernel `weno3_kernel` + `__device__` LSQ helpers |
| 5   | Convergence test (`test_swe2d_weno3_convergence.py`) |
| 6   | Smooth-extrema test |
| 7   | GUI/QML plumbing + renumber scheme 6→7 |
| 8   | CLI flag + migration warning |
| 9   | Backwards-compat shim |
| 10  | Code review + merge |

### Phase 3: Scheme 8 — MP5 (2 weeks)

| Day | Task |
|----:|------|
| 1–2 | Mesh-assembly extension: build `face_stencil_5` |
| 3–4 | CUDA kernel `mp5_kernel` with mapped limiter (4 cases) |
| 5   | Convergence test (`test_swe2d_mp5_convergence.py`) |
| 6   | Shock-capturing test (dam-break on triangles) |
| 7   | CFL ≤ 0.4 enforcement in `SWE2DBackend.step()` |
| 8   | GUI/QML plumbing |
| 9   | CLI flag + docs |
| 10  | Code review + merge |

### Phase 4: Documentation & Rollout (1 week)

| Day | Task |
|----:|------|
| 1–2 | Rewrite [SOLVER_ORDER_AND_STENCIL.md](../SOLVER_ORDER_AND_STENCIL.md) |
| 3   | Update [USER_GUIDE.md](../USER_GUIDE.md) |
| 4   | Update [SWE2D_GPU_ARCHITECTURE_REPORT.md](../SWE2D_GPU_ARCHITECTURE_REPORT.md) |
| 5   | Update [INDEX.md](../INDEX.md) + cross-links |
| 6   | CHANGELOG entry + release notes |
| 7   | Final review |

---

## 13. Risk Register

| Risk | Probability | Impact | Mitigation |
|------|:-:|:-:|------|
| WENO3 2×2 LSQ per sub-stencil slow on GPU | Medium | Medium | Precompute QR or use explicit inverse; benchmark before final merge |
| MP5 CFL ≤ 0.4 doubles wall-time in production | High | High | Document prominently; add `--max-cfl` advisory when scheme=8 selected |
| Scheme-6 renumber breaks persisted run configs | High | Low | Auto-migration on load + warning log |
| Barth-Jespersen divergence on cyclic meshes | Low | High | Bound $\chi \in [0, 1]$, fallback to first-order if $\chi$ becomes NaN |
| MP5 instability near strong contact discontinuities | Medium | Medium | Use MP5-CF variant (cap-fitted) as default; add to kernel |
| New CSR extensions bloat mesh by ~30% | Medium | Low | Document; offer `--mesh-no-stencil-ext` flag for production runs that don't use schemes 6 or 8 |

---

## 14. Acceptance Criteria (Definition of Done)

- [ ] All three CUDA kernels pass unit tests on the dev workstation.
- [ ] All convergence tests pass at the order targets in §8.1.
- [ ] All robustness tests pass with no NaN / oscillation violations.
- [ ] Performance benchmarks within target ratios (§8.4).
- [ ] GUI dropdown correctly exposes all 9 entries (0–8).
- [ ] CLI `--spatial-scheme` accepts 0–8 with valid validation.
- [ ] Migration warning fires when old-numbered scheme 6 is supplied.
- [ ] All docs updated and cross-linked from [INDEX.md](../INDEX.md).
- [ ] CHANGELOG entry under unreleased.
- [ ] No regression in existing test suite (full `pytest` run green).

---

## 15. Open Questions

1. Should we drop the current scheme-6 (WENO5) entirely or keep as scheme 7? **Recommendation: keep as 7** — it's still the highest-accuracy option and supports existing production runs.
2. Should MP5 use the "MP5-CF" (cap-fitted) variant by default to avoid the contact-discontinuity instability? **Recommendation: yes**, document as a deliberate choice.
3. For Barth-Jespersen, should we also offer a **vertex-based** variant (limiter applied at shared vertices rather than faces)? Defer to a follow-up plan.
4. Is `face_stencil_5` worth its memory cost for non-MP5 runs? **Recommendation: build always** (cheap on host), only read in `mp5_kernel`.

---

## 16. References

- Suresh, A., & Huynh, H. T. (1997). *Accurate monotonicity-preserving schemes with Runge–Kutta time stepping.* J. Comput. Phys. 136, 83–99.
- Jiang, G.-S., & Shu, C.-W. (1996). *Efficient implementation of weighted ENO schemes.* J. Comput. Phys. 126, 202–228.
- Hu, C., & Shu, C.-W. (1999). *Weighted essentially non-oscillatory schemes on triangular meshes.* J. Comput. Phys. 150, 97–127.
- Zhang, X., & Shu, C.-W. (2012). *Positivity-preserving high order finite volume WENO schemes on unstructured meshes.* J. Comput. Phys. 231, 2165–2185.
- Barth, T. J., & Jespersen, D. C. (1989). *The design and application of upwind schemes on unstructured meshes.* AIAA Paper 89-0366.
- Sweby, P. K. (1984). *High resolution schemes using flux limiters for hyperbolic conservation laws.* SIAM J. Numer. Anal. 21, 995–1011.

For full mathematical development of each scheme, see [Advanced Spatial Schemes — Technical Guide](../ADVANCED_SPATIAL_SCHEMES.md).