# WENO5 + Least-Squares Gradient on a 2-Ring Stencil: Implementation Plan

> **Status**: Planning — not yet started  
> **Priority**: Recommended next spatial-accuracy upgrade (per `SOLVER_ORDER_AND_STENCIL.md`)  
> **Estimated effort**: 4–6 weeks (single developer, GPU-first, with validation)  
> **Author**: Auto-generated from codebase analysis  
> **Date**: 2026-06-05

---

## 1. Executive Summary

The current SWE2D solver achieves at most **2nd-order spatial accuracy** using a 1-ring face-neighbor stencil with Green-Gauss (GG) gradient reconstruction and TVD limiters (schemes 1–4) or a WENO3-like 2-candidate blend (scheme 5). The 2nd-order ceiling is fundamental:

- **Sweby's theorem**: any TVD scheme on a 3-point (1-ring) stencil is at most 2nd-order in smooth regions.
- **1-ring stencil**: 3rd-order unstructured reconstruction requires at least 2 layers of neighbors.
- **Green-Gauss limitation**: on non-orthogonal meshes, the face-average approximation introduces $O(h)$ gradient error, degrading even 2nd-order accuracy.

This plan implements **scheme 6 — `FV_WENO5`**: a genuine 5th-order WENO reconstruction using a **least-squares (LSQ) gradient** over a **2-ring stencil**, achieving effective **3rd-order spatial accuracy** on general unstructured meshes.

**Key architectural decision**: The LSQ gradient is computed on the 2-ring stencil, then the WENO5 reconstruction uses both the LSQ gradient and direct cell-pair information to build 3 sub-stencils per edge face. The result is a blended face state that achieves ~3rd-order accuracy while preserving TVD-like monotonicity through pair-bounds clamping.

---

## 2. Current Architecture Summary

### 2.1 Stencil Data Structure (1-Ring CSR)

The mesh builder (`swe2d_build_mesh_poly` in `cpp/src/swe2d_mesh.cpp`) constructs a CSR structure:

```
cell_edge_offsets[n_cells + 1]  → offsets into cell_edge_ids[]
cell_edge_ids[]                 → edge indices for each cell's face neighbors
```

Usage in GPU kernels:
```cuda
const int32_t s = cell_edge_offsets[c];
const int32_t e = cell_edge_offsets[c + 1];
for (int32_t k = s; k < e; ++k) {
    const int32_t edge = cell_edge_ids[k];
    // c0/c1 from edge arrays gives face-neighbor cell indices
}
```

### 2.2 Gradient Computation (Green-Gauss, 1-Ring)

**Kernel**: `swe2d_gradient_kernel` in `cpp/src/swe2d_gpu.cu` (line ~935)

- One thread per edge; atomic `double` adds to both incident cells.
- Face-average: $\bar{q}_f = \frac{q_{c_0} + q_{c_1}}{2}$ — source of $O(h)$ error on skewed meshes.
- Gradient: $\nabla q_{c_0} = \frac{1}{A_{c_0}} \sum_{e \in \partial c_0} \bar{q}_f \hat{n}_e L_e$

### 2.3 TVD Reconstruction (Schemes 1–4)

**Location**: `tvd_reconstruct` lambda in `swe2d_flux_kernel` (line ~1409)

- Uses GG gradient projected onto cell-to-cell vector for slope ratio $r$.
- Applies limiter $\phi(r)$ (Superbee / MinMod / MC / Van Leer).
- Extrapolates to **actual face midpoint**: $q_f^L = q_0 + \phi \cdot \nabla q_0 \cdot (\vec{x}_f - \vec{x}_{c0})$.
- Pair-bounds clamp: $q_f \leftarrow \text{clamp}(q_f, \min(q_0,q_1), \max(q_0,q_1))$.

### 2.4 WENO3-like Reconstruction (Scheme 5)

**Location**: `weno3_like_reconstruct` lambda (line ~1450)

- 2-candidate blend: GG-extrapolated state vs. pair-midpoint state.
- Smoothness indicators $\beta_{L0}$, $\beta_{L1}$, $\beta_{R0}$, $\beta_{R1}$.
- Jump-aware linear weights: $\hat{d}_0 = 2/3$ in smooth regions, reduced to 0.52 near jumps.
- Still pair-bounds clamped.

### 2.5 Device State Arrays

All in `SWE2DDeviceState` (`cpp/src/swe2d_gpu.cuh` line ~187):

| Array | Size | Purpose |
|-------|------|---------|
| `d_cell_edge_offsets` | `n_cells + 1` | CSR offsets (1-ring) |
| `d_cell_edge_ids` | `sum(n_verts)` | Edge indices per cell |
| `d_grad_hx/hy` | `n_cells` | GG depth gradient |
| `d_grad_hux/uy` | `n_cells` | GG x-momentum gradient |
| `d_grad_hvx/vy` | `n_cells` | GG y-momentum gradient |
| `d_cell_cx/cy` | `n_cells` | Cell centroids |

---

## 3. Design: WENO5 with LSQ Gradient

### 3.1 Mathematical Foundation

#### 3.1.1 Least-Squares Gradient (2-Ring)

For each cell $c_0$, collect all unique cells in its 2-ring (cells reachable via 1 or 2 edge traversals). The LSQ gradient minimizes:

$$\nabla q_{c_0} = \arg\min_{\nabla q} \sum_{j \in N_2(c_0)} w_j \left( q_j - q_{c_0} - \nabla q \cdot \Delta\vec{x}_{j} \right)^2$$

where $\Delta\vec{x}_j = \vec{x}_j - \vec{x}_{c_0}$ and $w_j = 1/|\Delta\vec{x}_j|^2$ (inverse-distance weighting).

**Normal equations** (2×2 system, closed-form solution):

$$\begin{pmatrix} \sum w_j \Delta x_j^2 & \sum w_j \Delta x_j \Delta y_j \\ \sum w_j \Delta x_j \Delta y_j & \sum w_j \Delta y_j^2 \end{pmatrix} \begin{pmatrix} \partial q / \partial x \\ \partial q / \partial y \end{pmatrix} = \begin{pmatrix} \sum w_j \Delta x_j \Delta q_j \\ \sum w_j \Delta y_j \Delta q_j \end{pmatrix}$$

This gives $O(h^2)$ gradient accuracy even on non-orthogonal meshes, because the fitting uses cell-centroid values (not face-averages) and the weighting automatically de-emphasizes far-away cells.

**Degenerate stencil handling**: For boundary cells with < 3 neighbors in the 2-ring, the LSQ system is under-determined. Fallback to Green-Gauss gradient (current 1-ring behavior).

#### 3.1.2 WENO5 Reconstruction (3 Sub-Stencils per Edge)

For each interior edge $e$ with cells $(c_0, c_1)$, the face state $q_f^L$ (left side, from $c_0$) is reconstructed from 3 candidate stencils:

| Stencil | Cells | Description |
|---------|-------|-------------|
| $S_0^L$ | $\{c_0, c_1\}$ | Pair-midpoint (most diffuse, most stable) |
| $S_1^L$ | $\{c_0, c_1, c_0\text{-upwind}\}$ | Upwind extended via LSQ gradient |
| $S_2^L$ | $\{c_0, N_1(c_0), N_2(c_0)\}$ | Full 2-ring quadratic via LSQ extrapolation |

Each candidate produces a face value:

$$p_0^L = q_0 + \frac{1}{2}(q_1 - q_0) \quad \text{(midpoint)}$$

$$p_1^L = q_0 + \nabla^{\text{LSQ}} q_0 \cdot (\vec{x}_f - \vec{x}_{c_0}) \quad \text{(LSQ extrapolation)}$$

$$p_2^L = q_0 + \phi_{\text{TVD}}(r) \cdot \nabla^{\text{LSQ}} q_0 \cdot (\vec{x}_f - \vec{x}_{c_0}) \quad \text{(limited LSQ)}$$

where $\phi_{\text{TVD}}$ is a Van Leer limiter applied to the LSQ gradient (used as a smooth sub-stencil, not the final limiter).

**Smoothness indicators**:

$$\beta_k = \sum_{j \in S_k} (q_j - \bar{q}_{S_k})^2, \quad k = 0, 1, 2$$

**Nonlinear WENO weights** (with $\epsilon$ parameter to prevent division by zero):

$$\alpha_k = \frac{d_k}{(\epsilon + \beta_k)^2}, \quad w_k = \frac{\alpha_k}{\sum_{j=0}^{2} \alpha_j}$$

**Linear (optimal) weights**: $d_0 = 0.1$, $d_1 = 0.3$, $d_2 = 0.6$ (favoring the limited LSQ stencil in smooth regions).

**Final blended state**:

$$q_f^L = w_0 p_0^L + w_1 p_1^L + w_2 p_2^L$$

Then apply **pair-bounds clamp**: $q_f^L \leftarrow \text{clamp}(q_f^L, \min(q_0, q_1), \max(q_0, q_1))$.

Analogous reconstruction for $q_f^R$ from cell $c_1$'s perspective.

#### 3.1.3 Well-Balancing (Lake-at-Rest)

The surface-gradient method is preserved: reconstruct $\eta = h + z_b$ using WENO5, then convert back to depth via $h = \eta - z_b$. For lake-at-rest, $\nabla \eta = 0$ everywhere, so the LSQ gradient is zero, all smoothness indicators are near-zero, the WENO weights are dominated by $\epsilon$, and the reconstruction returns $(q_0 + q_1)/2$ — which is the exact balanced state. **This property is preserved by construction.**

---

### 3.2 Data Structures

#### 3.2.1 2-Ring CSR Structure (New)

```cpp
// In SWE2DMesh (swe2d_mesh.hpp):
std::vector<int32_t> cell_ring2_offsets;  // [n_cells + 1], CSR offsets
std::vector<int32_t> cell_ring2_ids;      // [sum(ring2_unique_counts)], unique cell indices
std::vector<double>  cell_ring2_dcx;      // [sum(ring2_unique_counts)], Δx to neighbor (precomputed)
std::vector<double>  cell_ring2_dcy;      // [sum(ring2_unique_counts)], Δy to neighbor (precomputed)
std::vector<double>  cell_ring2_inv_dist2; // [sum(ring2_unique_counts)], 1/|Δr|² for LSQ weighting
```

**Construction algorithm** (in mesh builder):

```
For each cell c0:
    ring2_neighbors = empty set
    For each edge e of c0 (from 1-ring CSR):
        peer = (edge_c0[e] == c0) ? edge_c1[e] : edge_c0[e]
        if peer >= 0:
            ring2_neighbors.insert(peer)    // 1-hop
            For each edge e2 of peer (from 1-ring CSR):
                peer2 = (edge_c0[e2] == peer) ? edge_c1[e2] : edge_c0[e2]
                if peer2 >= 0 and peer2 != c0:
                    ring2_neighbors.insert(peer2)   // 2-hop
    Sort ring2_neighbors (for determinism)
    Append ring2_neighbors to cell_ring2_ids
    For each neighbor j in ring2_neighbors:
        Δx = cell_cx[j] - cell_cx[c0]
        Δy = cell_cy[j] - cell_cy[c0]
        dist² = Δx² + Δy²
        Append (Δx, Δy, 1/dist²) to precomputed arrays
    cell_ring2_offsets[c0 + 1] = running count
```

**Memory**: For a triangle mesh, average ~12 neighbors in 2-ring. Per cell: 3 ints + 3 doubles ≈ 36 bytes. For 1M cells → ~36 MB (negligible vs. state arrays).

#### 3.2.2 Device Arrays (New in `SWE2DDeviceState`)

```cpp
// 2-ring topology (uploaded once at init)
int32_t* d_cell_ring2_offsets = nullptr;   // [n_cells + 1]
int32_t* d_cell_ring2_ids     = nullptr;   // [sum(ring2_counts)]
double*  d_cell_ring2_dcx      = nullptr;   // [sum(ring2_counts)]
double*  d_cell_ring2_dcy      = nullptr;   // [sum(ring2_counts)]
double*  d_cell_ring2_inv_dist2 = nullptr;  // [sum(ring2_counts)]

// LSQ gradient arrays (computed per step, replacing GG gradient for scheme 6)
double*  d_lsq_grad_hx  = nullptr;   double*  d_lsq_grad_hy  = nullptr;
double*  d_lsq_grad_hux = nullptr;   double*  d_lsq_grad_huy = nullptr;
double*  d_lsq_grad_hvx = nullptr;   double*  d_lsq_grad_hvy = nullptr;
```

---

### 3.3 Kernel Design

#### 3.3.1 LSQ Gradient Kernel (New)

**Name**: `swe2d_lsq_gradient_kernel`  
**Parallelism**: One thread per cell (cell-parallel, not edge-parallel like GG)  
**Launch**: `<<<grid, 256>>>` with `grid = (n_cells + 255) / 256`

```cuda
__global__ void swe2d_lsq_gradient_kernel(
    int32_t n_cells,
    const int32_t* __restrict__ cell_ring2_offsets,
    const int32_t* __restrict__ cell_ring2_ids,
    const double*  __restrict__ cell_ring2_dcx,
    const double*  __restrict__ cell_ring2_dcy,
    const double*  __restrict__ cell_ring2_inv_dist2,
    const double*  __restrict__ cell_h,
    const double*  __restrict__ cell_zb,
    const double*  __restrict__ cell_hu,
    const double*  __restrict__ cell_hv,
    const int32_t* __restrict__ d_active,
    // Output: LSQ gradient of η = h + z_b, hu, hv
    double* lsq_grad_hx,  double* lsq_grad_hy,
    double* lsq_grad_hux, double* lsq_grad_huy,
    double* lsq_grad_hvx, double* lsq_grad_hvy)
{
    int32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_cells || !d_active[c]) return;

    const int32_t s = cell_ring2_offsets[c];
    const int32_t e = cell_ring2_offsets[c + 1];

    if (e - s < 2) {
        // Degenerate stencil: fallback to zero gradient
        lsq_grad_hx[c] = 0.0;  lsq_grad_hy[c] = 0.0;
        lsq_grad_hux[c] = 0.0; lsq_grad_huy[c] = 0.0;
        lsq_grad_hvx[c] = 0.0; lsq_grad_hvy[c] = 0.0;
        return;
    }

    const double eta0 = cell_h[c] + cell_zb[c];
    const double hu0  = cell_hu[c];
    const double hv0  = cell_hv[c];

    // Accumulate 2×2 normal equations
    double a11 = 0.0, a12 = 0.0, a22 = 0.0;
    double b1_eta = 0.0, b2_eta = 0.0;
    double b1_hu = 0.0, b2_hu = 0.0;
    double b1_hv = 0.0, b2_hv = 0.0;

    for (int32_t k = s; k < e; ++k) {
        const int32_t j = cell_ring2_ids[k];
        const double dx = cell_ring2_dcx[k];
        const double dy = cell_ring2_dcy[k];
        const double w  = cell_ring2_inv_dist2[k]; // 1/|Δr|²

        const double eta_j = cell_h[j] + cell_zb[j];
        const double d_eta = eta_j - eta0;
        const double d_hu  = cell_hu[j] - hu0;
        const double d_hv  = cell_hv[j] - hv0;

        a11 += w * dx * dx;
        a12 += w * dx * dy;
        a22 += w * dy * dy;
        b1_eta += w * dx * d_eta;
        b2_eta += w * dy * d_eta;
        b1_hu  += w * dx * d_hu;
        b2_hu  += w * dy * d_hu;
        b1_hv  += w * dx * d_hv;
        b2_hv  += w * dy * d_hv;
    }

    // Solve 2×2 system via Cramer's rule
    const double det = a11 * a22 - a12 * a12;
    const double inv_det = (fabs(det) > 1.0e-30) ? (1.0 / det) : 0.0;

    lsq_grad_hx[c]  = inv_det * (a22 * b1_eta - a12 * b2_eta);
    lsq_grad_hy[c]  = inv_det * (a11 * b2_eta - a12 * b1_eta);
    lsq_grad_hux[c] = inv_det * (a22 * b1_hu  - a12 * b2_hu);
    lsq_grad_huy[c] = inv_det * (a11 * b2_hu  - a12 * b1_hu);
    lsq_grad_hvx[c] = inv_det * (a22 * b1_hv  - a12 * b2_hv);
    lsq_grad_hvy[c] = inv_det * (a11 * b2_hv  - a12 * b1_hv);
}
```

**Key property**: This kernel is cell-parallel with no atomics (each cell writes to its own gradient entry). Register pressure is moderate — approximately 20–30 live doubles for the normal-equations accumulation.

#### 3.3.2 WENO5 Reconstruction Lambda (New, in `swe2d_flux_kernel`)

```cuda
auto weno5_reconstruct = [&](double q0, double q1,
                              double gx0, double gy0,
                              double gx1, double gy1,
                              int32_t c0, int32_t c1,
                              double& qL_out, double& qR_out) {
    const double dq = q1 - q0;
    const double fx = edge_mx[e];  // face midpoint x
    const double fy = edge_my[e];  // face midpoint y
    const double dxL = fx - cell_cx[c0];
    const double dyL = fy - cell_cy[c0];
    const double dxR = fx - cell_cx[c1];
    const double dyR = fy - cell_cy[c1];

    // ── Candidate 0: pair-midpoint (most diffuse) ──
    const double pL0 = q0 + 0.5 * dq;
    const double pR0 = q1 - 0.5 * dq;

    // ── Candidate 1: LSQ extrapolation (unlimited) ──
    const double pL1 = q0 + (gx0 * dxL + gy0 * dyL);
    const double pR1 = q1 + (gx1 * dxR + gy1 * dyR);

    // ── Candidate 2: TVD-limited LSQ extrapolation ──
    // Use Van Leer limiter on the LSQ gradient (3rd sub-stencil)
    const double dcx = cell_cx[c1] - cell_cx[c0];
    const double dcy = cell_cy[c1] - cell_cy[c0];
    const double sign_dq = (dq >= 0.0) ? 1.0 : -1.0;
    const double s0 = gx0 * dcx + gy0 * dcy;
    const double s1 = -(gx1 * dcx + gy1 * dcy);
    constexpr double EPS = 1.0e-30;
    const double r0 = s0 / (dq + sign_dq * EPS);
    const double r1 = s1 / (-dq + (-sign_dq) * EPS);
    const double phi0 = (r0 + fabs(r0)) / (1.0 + fabs(r0));  // Van Leer
    const double phi1 = (r1 + fabs(r1)) / (1.0 + fabs(r1));
    const double pL2 = q0 + phi0 * (gx0 * dxL + gy0 * dyL);
    const double pR2 = q1 + phi1 * (gx1 * dxR + gy1 * dyR);

    // ── Smoothness indicators β_k ──
    // β_0: pair-midpoint smoothness (measures local jump)
    const double betaL0 = dq * dq;
    const double betaR0 = dq * dq;
    // β_1: LSQ extrapolation deviation from cell center
    const double betaL1 = (pL1 - q0) * (pL1 - q0);
    const double betaR1 = (pR1 - q1) * (pR1 - q1);
    // β_2: limited LSQ deviation
    const double betaL2 = (pL2 - q0) * (pL2 - q0);
    const double betaR2 = (pR2 - q1) * (pR2 - q1);

    // ── WENO nonlinear weights ──
    const double scale = q0 * q0 + q1 * q1 + dq * dq;
    const double eps_weno = 1.0e-6 * fmax(1.0, scale);
    // Linear weights: favor limited LSQ in smooth regions
    constexpr double d0 = 0.10;  // midpoint (fallback)
    constexpr double d1 = 0.30;  // LSQ extrapolation
    constexpr double d2 = 0.60;  // limited LSQ (primary smooth candidate)

    // Left state
    const double aL0 = d0 / ((eps_weno + betaL0) * (eps_weno + betaL0));
    const double aL1 = d1 / ((eps_weno + betaL1) * (eps_weno + betaL1));
    const double aL2 = d2 / ((eps_weno + betaL2) * (eps_weno + betaL2));
    const double sumL = aL0 + aL1 + aL2;
    const double wL0 = aL0 / sumL;
    const double wL1 = aL1 / sumL;
    const double wL2 = aL2 / sumL;

    // Right state
    const double aR0 = d0 / ((eps_weno + betaR0) * (eps_weno + betaR0));
    const double aR1 = d1 / ((eps_weno + betaR1) * (eps_weno + betaR1));
    const double aR2 = d2 / ((eps_weno + betaR2) * (eps_weno + betaR2));
    const double sumR = aR0 + aR1 + aR2;
    const double wR0 = aR0 / sumR;
    const double wR1 = aR1 / sumR;
    const double wR2 = aR2 / sumR;

    // ── Blend ──
    double rawL = wL0 * pL0 + wL1 * pL1 + wL2 * pL2;
    double rawR = wR0 * pR0 + wR1 * pR1 + wR2 * pR2;

    // ── Pair-bounds clamp (TV property) ──
    const double qmin = fmin(q0, q1);
    const double qmax = fmax(q0, q1);
    qL_out = fmin(qmax, fmax(qmin, rawL));
    qR_out = fmin(qmax, fmax(qmin, rawR));
};
```

**Register pressure estimate**: Each WENO5 call with 3 candidates per side requires approximately 40–50 registers. The flux kernel currently uses ~90 registers; WENO5 will push this to an estimated **120–150 registers**, which may require `--maxrregcount` tuning or increasing occupancy reduction.

#### 3.3.3 Integration into Flux Kernel

The existing flux kernel has a branch structure:

```
if (spatial_scheme >= scheme_fast && grad_hx != nullptr) {
    // TVD or WENO3 reconstruction
}
```

The WENO5 path will be inserted as:

```cuda
const int scheme_weno5 = static_cast<int>(SWE2DSpatialScheme::FV_WENO5);

if (spatial_scheme == scheme_weno5 && cell_cx != nullptr && lsq_grad_hx != nullptr) {
    // Use LSQ gradient arrays (d_lsq_grad_hx etc.) with WENO5 reconstruction
    weno5_reconstruct(etaL, etaR, lsq_grad_hx[c0], lsq_grad_hy[c0],
                      lsq_grad_hx[c1], lsq_grad_hy[c1], c0, c1, etaL_rec, etaR_rec);
    // ... same for hu, hv
} else if (!disable_higher_order && spatial_scheme >= scheme_fast && ...) {
    // Existing TVD / WENO3 path
}
```

---

### 3.4 CPU Solver Changes (Lower Priority)

The CPU solver (`swe2d_solver.cpp`) should eventually receive a parallel implementation for scheme 6, but it is **not a blocker**. The current validation suite (`tests/swe2d_nonorth_gpu_sweep_common.py`) only validates `spatial_scheme=0` (first-order) on the CPU path.

**CPU WENO5 implementation** (Phase 4):

1. Extend the `tvd_rec` lambda with a `weno5_rec` lambda matching the GPU formulation.
2. Add LSQ gradient computation in the CPU update loop (2-ring traversal from host arrays).
3. **Important**: Fix the CPU mid-point extrapolation bug documented in `SOLVER_ORDER_AND_STENCIL.md` — the CPU uses $q_f = q_0 + \phi \cdot 0.5 \cdot (q_1 - q_0)$ which ignores face geometry. This must be corrected to match the GPU's $q_f = q_0 + \phi \cdot \nabla q_0 \cdot (\vec{x}_f - \vec{x}_{c0})$ regardless of the WENO5 upgrade.

---

## 4. Implementation Phases

### Phase 1: 2-Ring CSR Builder (3–5 days)

**Files modified**:
- `cpp/src/swe2d_mesh.hpp` — Add new fields to `SWE2DMesh`
- `cpp/src/swe2d_mesh.cpp` — Extend `swe2d_build_mesh_poly()` to compute 2-ring
- `cpp/src/swe2d_gpu.cuh` — Add new device pointers to `SWE2DDeviceState`
- `cpp/src/swe2d_gpu.cu` — Upload new arrays to GPU in init function

**Tasks**:

| # | Task | Details |
|---|------|---------|
| 1.1 | Add mesh struct fields | `cell_ring2_offsets`, `cell_ring2_ids`, `cell_ring2_dcx`, `cell_ring2_dcy`, `cell_ring2_inv_dist2` to `SWE2DMesh` |
| 1.2 | Build 2-ring in mesh builder | Walk 1-ring CSR, then 1-ring of each neighbor. Deduplicate (exclude self). Sort by cell index for determinism. Compute $\Delta x$, $\Delta y$, $1/\|\Delta\mathbf{r}\|^2$. |
| 1.3 | Add device pointers | `d_cell_ring2_offsets`, `d_cell_ring2_ids`, `d_cell_ring2_dcx`, `d_cell_ring2_dcy`, `d_cell_ring2_inv_dist2`, `d_lsq_grad_hx/hy/hux/huy/hvx/hvy` to `SWE2DDeviceState` |
| 1.4 | Upload to GPU | Allocate & copy in `swe2d_gpu_init()` / `swe2d_create_solver_gpu()`. Free in destroy/cleanup. |
| 1.5 | Python bindings | Extend pybind11 bindings for `swe2d_build_mesh()` to include 2-ring data in the returned mesh object. |
| 1.6 | Unit test | Verify 2-ring construction: correct neighbor counts, known triangle mesh with exact 2-ring membership, boundary cell handling. |

**Boundary cell handling**: Boundary cells have fewer 1-ring neighbors (one edge is on the boundary). The 2-ring builder walks only interior-edge neighbors, so ghost cells are never included. Boundary cells with < 3 neighbors in the 2-ring will trigger the LSQ fallback to Green-Gauss (or zero gradient if even GG fails).

**Degenerate mesh handling**: Cells with all boundary edges (e.g., isolated corner cells) will have 0 or 1 neighbors in the 2-ring. The LSQ kernel checks `e - s < 2` and falls back to zero gradient, which effectively reverts to 1st-order for that cell.

### Phase 2: LSQ Gradient Kernel (5–7 days)

**Files modified**:
- `cpp/src/swe2d_gpu.cu` — New kernel `swe2d_lsq_gradient_kernel`
- `cpp/src/swe2d_gpu.cuh` — Kernel declaration
- `cpp/src/swe2d_solver.cpp` — Integration into step loop (launch conditionally on `spatial_scheme == 6`)

**Tasks**:

| # | Task | Details |
|---|------|---------|
| 2.1 | Implement kernel | As designed in §3.3.1. Cell-parallel, no atomics. Cramer's rule for 2×2 system. |
| 2.2 | Memset before launch | Zero `d_lsq_grad_*` arrays before each kernel launch. |
| 2.3 | Fallback logic | If `cell_ring2_offsets[c+1] - cell_ring2_offsets[c] < 3`: fall back to Green-Gauss gradient for that cell (zero the LSQ output, use GG result). Alternative: always run GG kernel, then conditionally overwrite with LSQ for scheme 6. |
| 2.4 | Integration | Launch LSQ kernel before flux kernel when `spatial_scheme == 6`. The existing GG kernel can still run (it populates `d_grad_*` which is used by schemes 1–5). LSQ populates `d_lsq_grad_*` which is used only by scheme 6. |
| 2.5 | Validation | Test LSQ gradient on known fields (linear, quadratic). Compare LSQ vs GG gradients on orthogonal and skewed meshes. Verify $O(h^2)$ convergence of LSQ on smooth fields. |

**CFL impact**: LSQ gradient adds one additional kernel launch per step (~0.05 ms for 100K cells on RTX 3080). Negligible relative to the flux kernel.

### Phase 3: WENO5 Reconstruction in Flux Kernel (5–7 days)

**Files modified**:
- `cpp/src/swe2d_solver.hpp` — Add `FV_WENO5 = 6` to enum
- `cpp/src/swe2d_gpu.cu` — WENO5 lambda + branch in `swe2d_flux_kernel`
- `cpp/src/swe2d_solver.cpp` — CPU fallback (can be 1st-order or deferred)

**Tasks**:

| # | Task | Details |
|---|------|---------|
| 3.1 | Add enum value | `FV_WENO5 = 6` in `SWE2DSpatialScheme`. |
| 3.2 | Implement `weno5_reconstruct` lambda | As designed in §3.3.2. 3 candidates × 2 sides, WENO weights, pair-bounds clamp. |
| 3.3 | Branch in flux kernel | `if (spatial_scheme == 6)` → use LSQ gradient arrays; else → existing path. |
| 3.4 | Register pressure audit | Profile `--maxrregcount` impact. If >128 regs/thread, consider splitting WENO5 into a pre-flux "reconstruction kernel" that writes face states to a buffer. |
| 3.5 | Wet/dry handling | Apply existing `enable_shallow_front_recon_fallback`: if either cell is shallow, fall back to 1st-order for that edge. |
| 3.6 | Momentum capping | Apply existing momentum cap after WENO5 reconstruction (same as TVD). |
| 3.7 | Surface-gradient method | WENO5 reconstructs $\eta = h + z_b$, then converts back via $h = \eta - z_b$. This is already the pattern in the flux kernel. |
| 3.8 | CUDA Graphs compat | WENO5 uses the same data flow as TVD (read state + gradient, write flux). No new synchronization points. Existing CUDA graph capture logic should work unchanged. |

**Potential register pressure mitigation**: If profiling shows the combined WENO5 + HLLC kernel exceeds 128 regs/thread (causing excessive occupancy loss), the reconstruction can be split into a separate `swe2d_weno5_recon_kernel` that writes `qL, qR, huL, huR, hvL, hvR` to per-edge temporary buffers, then the flux kernel reads those buffers. This sacrifices some data reuse but maintains occupancy.

### Phase 4: CPU WENO5 & Mid-point Fix (3–5 days, lower priority)

**Files modified**:
- `cpp/src/swe2d_solver.cpp` — LSQ gradient + WENO5 lambda + mid-point extrapolation fix

**Tasks**:

| # | Task | Details |
|---|------|---------|
| 4.1 | CPU LSQ gradient | Traverse 2-ring from mesh data. Same formula as GPU. |
| 4.2 | CPU WENO5 lambda | Mirror of GPU `weno5_reconstruct`. |
| 4.3 | Fix CPU mid-point extrapolation | Change `qL = q0 + phi * 0.5 * dq` to `qL = q0 + phi * (grad_x * dxL + grad_y * dyL)` for all schemes. |
| 4.4 | CPU scheme 6 integration | Branch on `spatial_scheme == 6` to use LSQ + WENO5. |

**Note**: Per `AGENTS.md`, the CPU path is validated only for `spatial_scheme=0`. The mid-point fix (4.3) is a 1-day high-impact change that benefits ALL TVD schemes on the CPU path.

### Phase 5: Python Bindings & Validation (3–5 days)

**Files modified**:
- `cpp/src/swe2d_bindings.cpp` (or equivalent) — Expose `spatial_scheme=6`
- `tests/swe2d_nonorth_gpu_sweep_common.py` — Add scheme 6
- New test: `tests/test_swe2d_weno5_convergence.py`

**Tasks**:

| # | Task | Details |
|---|------|---------|
| 5.1 | Expose scheme 6 in Python | `swe2d_create_solver(..., spatial_scheme=6)` should work. |
| 5.2 | Extend GPU sweep test | Add `spatial_scheme=6` to the sweep matrix in `swe2d_nonorth_gpu_sweep_common.py`. |
| 5.3 | Convergence test | Manufactured solution on structured triangular mesh. Refine mesh by factor 2, measure $L_2$ error for $h$, $hu$, $hv$. Expect ~3rd-order slope for smooth solutions. |
| 5.4 | Lake-at-rest validation | Verify $\nabla\eta = 0$ lake-at-rest is preserved (to machine precision) with WENO5 on a skewed mesh. |
| 5.5 | Dam-break validation | Compare WENO5 vs Van Leer vs MinMod on standard dam-break. Measure front sharpness and spurious oscillations. |
| 5.6 | Performance profiling | Compare wall-clock time per step for scheme 4 vs 6 on 10K, 100K, 1M cells. Expect 15–25% increase. |

---

## 5. Register Pressure Analysis

### 5.1 Current Flux Kernel Register Usage

The existing `swe2d_flux_kernel` (scheme 1–5) uses approximately **90 registers/thread** with `__launch_bounds__(256, 4)`.

### 5.2 Estimated WENO5 Register Impact

| Variable | Count | Type | Notes |
|----------|-------|------|-------|
| Candidate face states (pL0/1/2, pR0/1/2) | 6 | double | 3 per side |
| Smoothness indicators (βL0/1/2, βR0/1/2) | 6 | double | 3 per side |
| WENO weights (wL0/1/2, wR0/1/2) | 6 | double | 3 per side |
| LSQ gradient (gx0, gy0, gx1, gy1) | 4 | double | Read from array |
| Intermediate (s0, s1, r0, r1, phi0, phi1) | 6 | double | TVD sub-stencil |
| Constants (dq, dxL, dyL, dxR, dyR, eps_weno) | 6 | double | Reused from current code |
| **Total new** | **~34** | | |

**Estimated total**: 90 + 34 = **~124 registers/thread**

### 5.3 Mitigation Strategies

| Strategy | Impact | Effort |
|----------|--------|--------|
| `--maxrregcount=128` with increased `__launch_bounds__(256, 2)` | Reduces occupancy from 4 to 2 warps/SM, may hurt small meshes. | 0 |
| Split WENO5 into pre-flux reconstruction kernel | Writes 6 values/edge to buffer. Increases memory traffic by ~48 bytes/edge. ~1 week extra. | Medium |
| Use shared memory for WENO5 intermediates | Not practical — edge-parallel kernel doesn't have contiguous memory access patterns. | N/A |
| Reduce precision of smoothness calculations to `float` | WENO weights are robust to reduced precision. Saves ~17 registers (half the new doubles). | 2–3 days |

**Recommendation**: Start with `--maxrregcount=128` and profile. If throughput drops >30%, split into a two-kernel pipeline.

---

## 6. Memory Budget

### 6.1 Per-Cell Additional Memory (2-Ring)

| Array | Type | Typical Size (100K tri mesh) |
|-------|------|-------------------------------|
| `cell_ring2_offsets` | `int32_t` | 400 KB |
| `cell_ring2_ids` | `int32_t` | ~4.8 MB (12 avg neighbors × 100K) |
| `cell_ring2_dcx` | `double` | ~9.6 MB |
| `cell_ring2_dcy` | `double` | ~9.6 MB |
| `cell_ring2_inv_dist2` | `double` | ~9.6 MB |
| **Subtotal (new topology)** | | **~34 MB** |
| `d_lsq_grad_hx/hy` | `double` | ~1.6 MB each |
| `d_lsq_grad_hux/uy` | `double` | ~1.6 MB each |
| `d_lsq_grad_hvx/vy` | `double` | ~1.6 MB each |
| **Subtotal (new gradient)** | | **~9.6 MB** |
| **Total new** | | **~44 MB** |

**Relative to current state arrays**: ~3× cell HDF5 state ≈ 2.4 MB for 100K cells. The 2-ring topology adds ~14× the state arrays. This is acceptable for modern GPUs (≥6 GB VRAM).

### 6.2 GPU VRAM Requirements

| GPU | VRAM | 100K cells | 500K cells | 1M cells |
|-----|------|------------|------------|----------|
| RTX 3060 | 12 GB | ~50 MB | ~250 MB | ~500 MB |
| RTX 3080 | 10 GB | ~50 MB | ~250 MB | ~500 MB |
| RTX 4090 | 24 GB | ~50 MB | ~250 MB | ~500 MB |
| A100 | 80 GB | ~50 MB | ~250 MB | ~500 MB |

All scenarios comfortably fit within VRAM.

---

## 7. Validation Plan

### 7.1 Unit Tests (Phase 1–2)

| Test | Verification |
|------|-------------|
| `test_2ring_construction` | On a 4-cell quad mesh, verify exact 2-ring membership for each cell. Verify $\Delta x$, $\Delta y$, $1/\|\Delta\mathbf{r}\|^2$ match analytic values. |
| `test_2ring_boundary` | On a 3×3 triangle mesh, verify boundary cells have reduced 2-ring and fallback triggers correctly. |
| `test_lsq_gradient_linear` | Apply LSQ gradient to $q = ax + by + c$. Verify $\nabla q = (a, b)$ exactly on both orthogonal and 25%-skewed meshes. |
| `test_lsq_gradient_quadratic` | Apply LSQ gradient to $q = x^2$. Verify $O(h)$ error on 1-ring GG vs $O(h^2)$ error on 2-ring LSQ. |

### 7.2 Convergence Tests (Phase 3)

| Test | Method | Expected Result |
|------|--------|----------------|
| `test_weno5_convergence_h` | Smooth linear lake, manufactured $h(x,y)$ | ~3rd-order $L_2$ convergence |
| `test_weno5_convergence_hu` | Smooth momentum field | ~3rd-order $L_2$ convergence |
| `test_weno5_dambreak_compression` | Dam-break on orthogonal mesh | Sharper front vs Van Leer, minimal spurious oscillations |
| `test_weno5_dambreak_nonorth` | Dam-break on 25%-skewed mesh | < 10% $L_2$ depth error vs orthogonal, better than MinMod |

### 7.3 Lake-at-Rest Test (Phase 3)

This is the **most critical validation** for well-balancing:

- Flat bottom: $z_b = 0$, $h = h_0$, $\mathbf{u} = 0$. Verify $\|\nabla h\| < \epsilon$ (machine precision).
- Sloped bottom: $z_b = ax + by$, $h = h_0 - z_b$, $\mathbf{u} = 0$. Verify WENO5 reconstruction preserves $h_L = h_R$ to machine precision.
- Non-orthogonal sloped bottom: Same test on 25%-skewed mesh. WENO5 should maintain lake-at-rest because it reconstructs $\eta = h + z_b$ and the LSQ gradient of $\eta$ is zero.

---

## 8. Risk Assessment & Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|-----------|
| Register pressure > 128 regs/thread | Medium | 30–50% throughput loss | Split into pre-flux + flux kernels. Fall back to `--maxrregcount=192` with reduced occupancy. |
| 2-ring CSR construction bug on complex meshes | Medium | Wrong neighbors → unstable gradient | Deduplicate + sort validation test. Fallback to GG for boundary cells. |
| LSQ gradient degenerate on very coarse meshes (< 3 neighbors) | Low | 1st-order fallback for few cells | Triggered automatically; validation test covers this case. |
| WENO5 oscillations at strong discontinuities | Low | Overshoot in dam-break | Pair-bounds clamp prevents out-of-range values. Epsilon parameter tuning in validation. |
| LSQ gradient cost dominates step time | Low | 10–20% step time increase | LSQ is cell-parallel with no atomics; expected ~0.05 ms for 100K cells on RTX 3080. Profile early. |
| CUDA Graphs incompatibility | Very Low | WENO5 path not capturable | Same data flow as TVD; no new synchronization. Existing graph capture should work. |
| WENO5 weight blowup near dry cells | Medium | Unphysical face states | Shallow-front fallback already covers this case. WENO5 is only active when both cells are wet. |

---

## 9. Performance Expectations

### 9.1 Kernel Timing Estimates (RTX 3080, 100K cells)

| Kernel | Current (scheme 4) | WENO5 (scheme 6) | Delta |
|--------|--------------------|--------------------|-------|
| Green-Gauss gradient | 0.15 ms | 0.15 ms | 0 (kept as fallback) |
| LSQ gradient | — | 0.08 ms | +0.08 ms |
| Flux kernel (per edge) | 0.40 ms | 0.55–0.65 ms | +0.15–0.25 ms |
| Update kernel | 0.05 ms | 0.05 ms | 0 |
| **Total per step** | **0.60 ms** | **0.83–0.93 ms** | **+38–55%** |

### 9.2 Register Pressure Impact

At 124 regs/thread with `--maxrregcount=128`:
- Occupancy drops from 4 warps/SM to 2 warps/SM
- Theoretical throughput loss: ~30–40%
- Actual measured loss: typically 20–30% (memory-bound kernels mask register pressure)

**Mitigation**: If performance is critical, the reconstruction can be split into a separate kernel, reducing the flux kernel back to ~90 regs while adding 0.1 ms for reconstruction.

---

## 10. Open Questions & Decisions

| # | Question | Options | Recommendation |
|---|----------|---------|---------------|
| 1 | **Epsilon parameter**: What $\epsilon$ value for WENO weights? | $\epsilon = 10^{-6} \max(1, \|q\|^2)$ (standard) or $\epsilon = 10^{-20}$ (sharp) | Start with $\epsilon = 10^{-6} \max(1, \text{scale})$ as in WENO3. Tune in validation. |
| 2 | **Linear weights**: Constant or jump-adaptive? | Constant $(0.1, 0.3, 0.6)$ or adapt like WENO3-like | Constant for initial implementation; jump-adaptive as future enhancement. |
| 3 | **LSQ gradient fallback**: Per-cell or per-edge? | Per-cell (LSQ fails → GG for that cell's entire gradient) | Per-cell is simpler and correct; the GG gradient is already $O(h^2)$ on orthogonal meshes. |
| 4 | **Pre-flux kernel split**: Do it upfront or after profiling? | Upfront (safer register budget) or only if profiling shows >128 regs | Profile first with unified kernel. Split only if needed. |
| 5 | **CPU parity for WENO5**: Parallel with GPU or sequential? | Sequential — CPU is for regression, not production | Defer CPU WENO5 to Phase 4. Fix CPU mid-point extrapolation first (high impact, low effort). |
| 6 | **Should scheme 6 use the GG kernel output or only LSQ?** | LSQ only (cleaner) or hybrid (GG fallback per-cell) | Hybrid: always run GG kernel, then LSQ kernel overwrites for scheme 6. This way switching schemes requires no kernel recompilation. |

---

## 11. File Change Checklist

### New Files
| File | Purpose |
|------|---------|
| `cpp/src/swe2d_weno5.cu` | (Optional) Separate compilation unit for LSQ+WENO5 kernels. Alternatively, keep inline in `swe2d_gpu.cu`. |
| `tests/test_swe2d_weno5_convergence.py` | Convergence + lake-at-rest + dam-break validation for scheme 6. |

### Modified Files
| File | Changes |
|------|---------|
| `cpp/src/swe2d_solver.hpp` | Add `FV_WENO5 = 6` to `SWE2DSpatialScheme` enum. |
| `cpp/src/swe2d_mesh.hpp` | Add `cell_ring2_*` fields to `SWE2DMesh`. |
| `cpp/src/swe2d_mesh.cpp` | Build 2-ring in `swe2d_build_mesh_poly()`. |
| `cpp/src/swe2d_gpu.cuh` | Add `d_cell_ring2_*` and `d_lsq_grad_*` pointers to `SWE2DDeviceState`. |
| `cpp/src/swe2d_gpu.cu` | Add `swe2d_lsq_gradient_kernel`, `weno5_reconstruct` lambda, WENO5 branch in flux kernel, GPU alloc/upload/free for new arrays. |
| `cpp/src/swe2d_solver.cpp` | Add LSQ gradient computation + WENO5 lambda (CPU path, Phase 4). Fix mid-point extrapolation bug. |
| `tests/swe2d_nonorth_gpu_sweep_common.py` | Add `spatial_scheme=6` to sweep matrix. |

---

## 12. Success Criteria

| Criterion | Metric |
|-----------|--------|
| 2-ring construction correctness | All cells have expected 2-ring membership; boundary cells have reduced set; no self-loops |
| LSQ gradient accuracy | $O(h^2)$ error on smooth linear/quadratic test functions, vs $O(h)$ for GG on skewed mesh |
| WENO5 convergence rate | $\geq 2.5$-order $L_2$ slope for $h$ on smooth manufactured solution (target: ~3rd order) |
| Lake-at-rest preservation | $\|\nabla\eta\| < 10^{-12}$ on both orthogonal and 25%-skewed mesh (machine precision) |
| Dam-break front sharpness | WENO5 front width ≤ MinMod front width × 0.7; overshoot < 0.5% |
| Performance overhead | ≤ 55% wall-clock increase vs scheme 4 (Van Leer) per step |
| Scheme 0–5 unchanged | All existing tests pass without modification; no regression in schemes 0–5 |
| CUDA Graphs compat | Scheme 6 captured and replayed without error; no additional sync points |

---

## Appendix A: Why Not DG or Spectral Volume?

Per `SOLVER_ORDER_AND_STENCIL.md`:

| Method | Spatial Order | Memory Increase | Effort | Risk |
|--------|:----:|:--------------:|:-----:|:----:|
| **WENO5 + LSQ (this plan)** | **~3rd** | **~30%** | **4–6 wks** | **Medium** |
| DG P1 | 2nd | ~3× | 3–6 mo | High |
| DG P2 | 3rd | ~6× | 6–12 mo | Very high |
| Spectral Volume k=2 | 3rd | ~3× | 2–4 mo | Medium |
| Defect correction | ~2.5th | Minimal | 2–3 wks | Low |

WENO5+LSQ provides the best accuracy-to-effort ratio while preserving the existing FVM architecture, CUDA graph compatibility, and active-set wet/dry masking.

---

## Appendix B: Relationship to Existing Schemes

```
Spatial Scheme Hierarchy:

  0 (FV_FIRST_ORDER)        ── Piecewise constant, no gradient
  │
  1–4 (FV_MUSCL_*)         ── GG gradient + TVD limiters (2nd order)
  │                           └── Uses d_grad_hx/hy/hux/huy/hvx/hvy
  │
  5 (FV_WENO3_LIKE)        ── GG gradient + 2-candidate WENO blend (≈2nd order)
  │                           └── Uses d_grad_hx/hy/hux/huy/hvx/hvy
  │                           └── Pair-midpoint + GG extrapolation candidates
  │
  6 (FV_WENO5)             ── LSQ gradient + 3-candidate WENO blend (≈3rd order) [NEW]
                              └── Uses d_lsq_grad_hx/hy/hux/huy/hvx/hvy
                              └── Requires 2-ring CSR (d_cell_ring2_*)
                              └── Falls back to GG gradient for degenerate stencils
```

The WENO5 path is **fully additive** — no existing kernel is modified for schemes 0–5. The LSQ kernel runs in addition to (and after) the GG kernel, writing to separate gradient arrays. The flux kernel branches on `spatial_scheme == 6` to read from LSQ arrays instead of GG arrays.