# SWE2D Solver: Spatial/Temporal Order & Stencil Architecture

## Spatial Discretization

### Stencil Definition

The FVM stencil is built from the **cell-edge CSR** (`cell_edge_offsets[]` / `cell_edge_ids[]`) constructed during mesh assembly in `swe2d_build_mesh_poly`. Each cell's stencil consists of all face-adjacent (edge-sharing) neighbors:

- **Unstructured triangles** → **3 neighbors** (interior)
- **Quadrilaterals** → **4 neighbors**
- **General polygons** → **N neighbors** (equals vertex count)
- **Boundary cells** → N−1 active face neighbors (one edge is boundary)

The stencil width is always the **1-ring** — only immediate face neighbors. No scheme expands the stencil beyond this; higher order changes the *reconstruction accuracy*, not which cells participate.

### Spatial Schemes (0–5)

| Scheme | Value | Reconstruction | Limiter | Max. Order | Notes |
|--------|:-----:|---------------|---------|:----------:|-------|
| `FV_FIRST_ORDER` | 0 | None (cell-center → face) | None | **1st** | Piecewise constant — most diffusive, most robust |
| `FV_MUSCL_FAST` | 1 | Green-Gauss + TVD | **Superbee** | **2nd** | Most aggressive TVD, sharpest fronts, may overshoot on skewed meshes |
| `FV_MUSCL_MINMOD` | 2 | Green-Gauss + TVD | **MinMod** | **2nd** | Most conservative TVD, most stable, safest for mixed/hybrid meshes |
| `FV_MUSCL_MC` | 3 | Green-Gauss + TVD | **MC (monotonized-central)** | **2nd** | Balanced between accuracy and stability |
| `FV_MUSCL_VAN_LEER` | 4 | Green-Gauss + TVD | **Van Leer** (smooth) | **2nd** | Smooth limiter, graceful degradation on skewed cells — best all-rounder |
| `FV_WENO3_LIKE` | 5 | WENO3 nonlinear blend | Weighted GG + midpoint | **2nd** | "WENO-like": 2-candidate blend, not true 3-stencil WENO3 |

All schemes 1–5 use the **surface-gradient method** (Zhou et al. 2001): reconstruct η = h + zb via ∇η, then convert back to depth. This preserves lake-at-rest to machine precision independent of mesh irregularity.

### Accuracy Limit

**2nd-order is the ceiling** for the current architecture, bounded by:

1. **Sweby's theorem** — any TVD scheme with a 3-point stencil is at most 2nd-order accurate in smooth regions.
2. **1-ring stencil** — 3rd-order unstructured reconstruction requires at least 2 layers of neighbors (neighbors-of-neighbors).
3. **WENO3-like is not true WENO3** — genuine unstructured WENO3 needs 3 sub-stencils (cell triplets) per edge, not 2 candidates.

### Non-Orthogonal Mesh Accuracy

#### Why the Green-Gauss Gradient Loses Accuracy

The Green-Gauss gradient at cell $c_0$ is:

$$ \nabla q_{c_0} = \frac{1}{A_{c_0}} \sum_{e \in \partial c_0} \frac{q_{c_0} + q_{c_1}}{2} \, \hat{n}_e \, L_e $$

The face-average approximation $\frac{q_{c_0} + q_{c_1}}{2}$ is the source of the non-orthogonal error. Let $\vec{d}_{c0 \to f}$ be the vector from cell centroid $c_0$ to the face midpoint $f$, and $\vec{d}_{c1 \to f}$ be the vector from $c_1$ to $f$. A Taylor expansion about the face midpoint gives:

$$ q_{c_0} = q_f + \nabla q \cdot \vec{d}_{c0 \to f} + O(h^2) $$
$$ q_{c_1} = q_f + \nabla q \cdot \vec{d}_{c1 \to f} + O(h^2) $$

The face-average error is therefore:

$$ \frac{q_{c_0} + q_{c_1}}{2} - q_f = \frac{1}{2} \nabla q \cdot (\vec{d}_{c0 \to f} + \vec{d}_{c1 \to f}) + O(h^2) $$

On an **orthogonal mesh** (centroidal-dual / Delaunay), $\vec{d}_{c0 \to f} = -\vec{d}_{c1 \to f}$, so the error term vanishes and the gradient is **2nd-order accurate**.

On a **non-orthogonal mesh**, the two vectors do not sum to zero. The leading error is $O(h)$, so the gradient formally drops to **1st-order accurate**. The magnitude of the error is proportional to:

$$ \epsilon_{\text{grad}} \propto \|\vec{d}_{c0 \to f} + \vec{d}_{c1 \to f}\| \cdot \|\nabla q\| $$

This depends on the **mesh skew** — how far the face midpoint deviates from the midpoint of the line connecting cell centroids. A common measure is the **non-orthogonality angle** $\theta_e$ between the face normal $\hat{n}_e$ and the cell-to-cell vector $\vec{d}_{c0 \to c1}$.

#### How the Error Propagates Through the Solver

```
Green-Gauss gradient (O(h) error on skewed meshes)
    │
    ▼
Slope ratio r = (∇q · Δx_cc) / (q₁ − q₀)      ← contaminated gradient
    │
    ▼
TVD limiter φ(r)                                  ← may over-limit or under-limit
    │
    ▼
Face state: q_f = q₀ + φ · ∇q₀ · (x_f − x_c₀)   ← extrapolation to actual face (GPU)
         or: q_f = q₀ + φ · 0.5 · (q₁ − q₀)      ← midpoint step (CPU)
    │
    ▼
Pair-bounds clamp: q_f = clamp(raw, min(q₀,q₁), max(q₀,q₁))
    │                                              ← clamps more often on skewed cells
    ▼
HLLC Riemann solver                                ← flux computed from clamped states
```

**Three compounding effects:**

1. **Gradient direction is wrong** — the erroneous $\nabla q$ points partially in the wrong direction, so the face-state extrapolation accuracy degrades.

2. **Limiter misclassification** — the slope ratio $r$ is contaminated by the tangential component of the gradient. A physically smooth profile can appear non-monotonic (or vice versa), causing the limiter to activate unnecessarily or fail to activate when needed.

3. **Pair-bounds clamping fires more often** — the erroneous extrapolation overshoots the cell-pair range $[\min(q_0,q_1), \max(q_0,q_1)]$, triggering the safety clamp and reverting that edge to 1st-order. This is the dominant mechanism by which non-orthogonality reduces the effective order.

#### GPU vs CPU Reconstruction: A Critical Difference

| Path | Files | Extrapolation Formula | How $\phi$ Is Used |
|------|-------|---------------------|--------------------|
| **GPU** | `swe2d_flux_kernel` in `swe2d_gpu.cu` | $q_f^L = q_0 + \phi \cdot \nabla q_0 \cdot (\vec{x}_f - \vec{x}_{c0})$ | $\phi$ scales the *gradient-projected* extrapolation to the **actual face midpoint** |
| **CPU** | `swe2d_solver.cpp` `tvd_rec` lambda | $q_f^L = q_0 + \phi \cdot \frac{1}{2}(q_1 - q_0)$ | $\phi$ scales a **pure midpoint step**, ignoring face geometry |

The **GPU path is geometrically correct** — it uses the full gradient vector $\nabla q_0$ projected onto the exact vector from the cell centroid to the face midpoint $(\vec{x}_f - \vec{x}_{c0})$. If the gradient were exact, this would give the exact face value.

The **CPU path uses a simplified midpoint step** — it computes $\phi$ from the gradient but then applies it as $\phi \times 0.5 \times (q_1 - q_0)$, which is a pair-midpoint interpolation independent of face position. This approach was explicitly removed from the GPU kernel with the comment:

> *"The pair-only midpoint approach (coefficient 0.5) was removed because it produces identical face states on both sides of every edge, which cancels the HLLC solver's upwind dissipation and causes neutral-to-unstable behaviour on non-trivial meshes regardless of unit system."*

On a non-orthogonal mesh the CPU path has an additional error: even with a perfect gradient, the $0.5 \times dq$ step does not land on the face midpoint unless the face is equidistant from both centroids.

#### How the Safety Mechanisms Interact with Non-Orthogonality

| Mechanism | File Location | Effect on Non-Orthogonal Meshes |
|-----------|---------------|--------------------------------|
| **Pair-bounds clamping** | GPU: `tvd_reconstruct` lambda, CPU: `tvd_rec` lambda | Fires more often — the erroneous gradient overshoots the cell-pair bounds. Each clamp forces 1st-order on that edge. |
| **Shallow-front fallback** | GPU/CPU: `enable_shallow_front_recon_fallback` | Depth-based check, independent of mesh quality. Still active. |
| **Surface-gradient method** (reconstruct $\eta = h + z_b$, convert back) | GPU/CPU: applied to depth only | **Lake-at-rest is exact regardless of mesh skew** — $\nabla\eta = 0$ for constant $\eta$, so $r=0$, $\phi=0$, and the reconstruction returns $q_0/q_1$ unchanged. This is a provable property of the Zhou et al. method. |
| **Momentum cap** | GPU: after reconstruction, CPU: after reconstruction | Geometry-independent, always active. Caps prevent unbounded velocity from erroneous gradients on skewed cells. |
| **WENO3 nonlinear blend** (scheme 5 only) | GPU: `weno3_like_reconstruct` lambda | Partially mitigates — on skewed cells the gradient candidate has larger smoothness indicator $\beta$, so the WENO weight shifts toward the midpoint candidate, effectively reducing to 1st-order on the most skewed cells. |
| **Active-set dry skipping** | GPU: `d_active` check in flux kernel | Unaffected by mesh quality. |

#### Empirical Validation

The sweep at `tests/swe2d_nonorth_gpu_sweep_common.py` tests all 6 spatial schemes × 2 temporal orders × 2 Godunov modes at `skew_fraction_dx=0.25$ (25% of cell width interior node perturbation) on GPU.

The orthogonal vs non-orthogonal solution comparison uses these tolerances:
- `rel_l2_h < 10%` — depth field differs by up to 10%
- `rel_l2_hu < 15%` — momentum field differs by up to 15%
- `|q_nonorth - q_orth| / q_orth < 8%` — mean discharge differs by up to 8%

These tolerances are relatively loose, confirming that non-orthogonality introduces measurable (but stable) error. The test only validates first-order (`spatial_scheme=0`) on the CPU path. A formal order-of-accuracy study across mesh refinement levels on orthogonal vs skewed grids has not yet been performed.

#### Practical Guidance for Mesh Generation

| Skew Level ($\theta_e$) | Expected Accuracy | Recommended Scheme |
|:-----------------------:|:-----------------:|:------------------:|
| $< 10^\circ$ (good quad, well-shaped tri) | Full 2nd-order | Van Leer (4) or MC (3) |
| $10^\circ$–$30^\circ$ (reasonable unstructured) | ~1.5th–2nd order | MinMod (2) or Van Leer (4) |
| $30^\circ$–$60^\circ$ (poor quality) | 1st-order dominant | MinMod (2) — safest |
| $> 60^\circ$ (nearly degenerate) | 1st-order + risk of instability | First-order (0) or fix mesh |

For production floodplain work, typical meshes generated by GMSH or TQMesh with reasonable element quality fall in the $10^\circ$–$25^\circ$ range, where the higher-order schemes still provide meaningful benefit over first-order.

### What Higher Spatial Order (>2nd) Would Enable

| Application | Current 2nd-order limitation | Benefit of 3rd–4th order |
|------------|-----------------------------|--------------------------|
| Wet/dry front tracking | Front artificially lengthened by ~2–3 cells | Sharper front, ~1 cell |
| Bridge contraction/expansion | Velocity jet diffused, afflux underestimated | Sharper velocity field, better afflux prediction |
| **Scour / sediment transport** | $V^3$ nonlinearity amplifies velocity errors → ~33% transport error per 10% $V$ error | Reduced truncation error → tighter scour depth bounds |
| Scour hole resolution | ~8–10 cells needed for <5% amplitude error | ~3–4 cells with 3rd-order |
| Bedform migration | Numerical diffusion damps migration rate | Better preservation of bed features |

For most floodplain engineering tolerances (±20% water level, factor-of-2 scour), 2nd-order is adequate. For **detailed bridge scour analysis** (±10% velocity, ±20% scour depth), the jump to 3rd-order spatial would be beneficial.

---

### What It Would Take to Go Higher (>2nd Order Spatial)

Achieving 3rd-order or higher spatial accuracy on an unstructured mesh requires fundamental changes to the solver architecture. Below is a systematic breakdown of the options, ordered by implementation complexity.

#### Option 1: WENO5 on a 2-Ring Stencil (Most Practical Upgrade)

**Current limitation:** The 1-ring stencil only provides 3 cells (c0, c1, and c0's own neighbors for GG gradient). WENO5 requires 5 points per reconstruction direction.

**What would need to change:**

| Component | Current Implementation | Required for WENO5 |
|-----------|----------------------|-------------------|
| **Stencil data structure** | `cell_edge_offsets[]` / `cell_edge_ids[]` (1-ring CSR) | **2-ring CSR** — for each cell, store its neighbors-of-neighbors. New arrays: `cell_ring2_offsets[]`, `cell_ring2_ids[]` |
| **Gradient method** | Green-Gauss (face-average, $O(h)$ on skewed meshes) | **Least-squares** gradient over 2-ring stencil — solves $\min \sum_{j \in N_2(i)} (q_j - q_i - \nabla q_i \cdot \Delta x_{ij})^2$. This gives $O(h^2)$ gradients even on non-orthogonal meshes. |
| **Face-state reconstruction** | 1-sided TVD extrapolation from c0/c1 to face midpoint | **WENO5** with 5 candidate stencils: 3 forward + 3 backward (Big-Stencil WENO) or 5 directional sub-stencils on unstructured grids |
| **Memory per edge** | ~5 doubles per edge (flux arrays) | ~7–9 doubles per edge (additional candidate fluxes) |
| **Kernel complexity** | Single `swe2d_flux_kernel` with inline TVD | New kernel or heavily refactored kernel with WENO5 weight computation |

**Implementation cost estimate:** ~4–6 weeks

1. **2-ring CSR builder** (mesh construction, CPU-side): 3–5 days
   - Extend `swe2d_build_mesh_poly` to walk `cell_edge_ids` and build neighbor-of-neighbor lists
   - Handle boundary cells (mirror/ghost stencil contraction)
   - Upload to GPU (new device arrays in `SWE2DDeviceState`)

2. **Least-squares gradient kernel** (GPU): 5–7 days
   - For each cell, assemble the $N \times 2$ least-squares system from the 2-ring
   - Solve via normal equations $(A^T A)^{-1} A^T b$ (2×2 matrix, closed-form)
   - Handle degenerate stencils (boundary, insufficient neighbors)
   - Add new device arrays for the LSQ gradients (replaces or supplements GG arrays)

3. **WENO5 reconstruction** (GPU): 5–7 days
   - Compute 5 candidate states per edge (3 forward sub-stencils, 3 backward)
   - Compute smoothness indicators $\beta_k$ for each candidate
   - Nonlinear weight computation with $\epsilon$ protection
   - Clamp to pair bounds (TVD-like safeguard)
   - Integrate into `swe2d_flux_kernel` or as a separate pre-flux kernel

4. **Validation**: 3–5 days
   - Manufactured-solution convergence test (expect $\approx 3$rd order)
   - Lake-at-rest well-balancing test (must still pass)
   - Non-orthogonal mesh comparison vs 2nd-order

**Register pressure impact:** WENO5 would increase register usage from the current 90 regs/thread to an estimated **120–150 regs/thread**, making `--maxrregcount` even more critical.

#### Option 2: Discontinuous Galerkin (DG) P1 / P2

**Fundamental architecture change.** Instead of reconstructing face states from cell-averaged values, DG stores a polynomial representation **within each cell** and communicates via numerical fluxes at faces.

| Property | Current FV | DG P1 | DG P2 |
|----------|-----------|-------|-------|
| DOFs per cell | 3 (h, hu, hv) | 9 (3 polynomials × 3 coefficients) | 18 (3 × 6 coefficients) |
| Memory | $3N$ doubles | $9N$ doubles | $18N$ doubles |
| Time integration | RK2/RK4/RK5 | RK4/RK5 (higher-order needed) | RK5+ |
| CFL constraint | 0.45 | ~0.2–0.3 | ~0.1–0.15 |
| Steady-state convergence | Fast (upwind FV) | Slower (DG requires explicit RK) | Slower still |
| Wet/dry handling | Active-set mask | More complex — polynomial can go negative in dry cells |
| GPU friendliness | Edge-parallel, CSR-based | Element-parallel + face flux exchange |
| CUDA Graphs | Currently used | Possible but more complex graph structure |

**Key challenge for DG:** The active-set wet/dry framework would need significant rework. In FV, a dry cell just has $h=0$. In DG, the polynomial representation can produce negative $h$ values even when the cell-average is positive, requiring slope limiting that couples adjacent elements — complicating the GPU parallel decomposition.

**Implementation cost:** ~3–6 months (P1), ~6–12 months (P2)

#### Option 3: Spectral Volume (SV) Method

SV partitions each cell into internal sub-cells (control volumes) and reconstructs a high-order polynomial across the whole cell using the sub-cell averages. This is a middle ground between FV and DG.

| Property | FV (1-ring) | SV (k=2) | SV (k=3) |
|----------|:----------:|:--------:|:--------:|
| Internal CVs per cell (triangle) | 1 | 3 | 6 |
| Effective spatial order | 2nd | 3rd | 4th |
| Stencil | 1-ring (face neighbors) | 1-ring + internal CVs | 1-ring + internal CVs |
| Flux evaluations per face | 1 | 3 | 6 |
| CFL | 0.45 | ~0.3 | ~0.2 |

**Advantage over DG:** Wet/dry handling is more natural — each sub-cell stores a depth, and the active-set mask can be applied at the sub-cell level. The reconstruction is done per-cell (closed form via Vandermonde), not per-element-pair.

**Disadvantage:** Sub-cell mesh generation is non-trivial for arbitrary polygons, especially quads. The internal CV partitioning (e.g., Wang 2002) must be pre-computed per cell type and uploaded alongside the mesh.

**Implementation cost:** ~2–4 months (k=2 tri only), ~6+ months (mixed polygons)

#### Option 4: $p$-Multigrid with High-Order Correction

Keep the current 2nd-order FV as the base solver and add a high-order defect-correction step:

1. Solve with 2nd-order FV (current code, unchanged)
2. Compute the high-order residual using WENO5/DG reconstruction
3. Correct the solution with the residual (one extra step per N base steps)

**Advantage:** Incremental — the base solver is untouched, and the correction can be turned on/off. The correction kernel is a single additional launch that reads the base state and writes a correction increment.

**Disadvantage:** Limited accuracy gain — defect correction only converges to the high-order solution if the base solver is dissipative enough to be stable. For transient flows, the correction lags by one step.

**Implementation cost:** ~2–3 weeks (prototype), ~6–8 weeks (production)

#### Summary: Upgrade Paths

| Path | Order | GPU Kernels Changed | Memory Increase | Effort | Risk |
|------|:----:|:------------------:|:--------------:|:-----:|:----:|
| **WENO5 + 2-ring + LSQ** | 3rd | 3 (gradient, flux, CSR) | ~30% | **4–6 wks** | Medium |
| DG P1 | 2nd | All | ~3× | 3–6 mo | High |
| DG P2 | 3rd | All | ~6× | 6–12 mo | Very high |
| SV k=2 (tri only) | 3rd | Most | ~3× | 2–4 mo | Medium |
| SV k=3 | 4th | Most | ~6× | 6+ mo | High |
| Defect correction | ~2.5th | 1 extra kernel | Minimal | 2–3 wks | Low |

**Recommended path:** WENO5 with least-squares gradient on a 2-ring stencil provides the best accuracy-to-effort ratio. The mesh builder already supports general polygons, so the 2-ring extension is purely topological. The existing GPU architecture (edge-parallel, stream-ordered, CUDA-graph-compatible) is well-suited to the additional kernels.

The most impactful single change, however, would be **fixing the CPU reconstruction to use the actual face-midpoint extrapolation** (matching the GPU path). This is a ~1-day fix that would eliminate a known non-orthogonal accuracy gap — the code comment already identifies it as producing unstable behaviour on non-trivial meshes.

---

## Temporal Integration

### Available Methods

| `temporal_order` | Method | Stages | Formal Accuracy | SSP | Graph-Capturable | Practical CFL |
|:---------------:|--------|:-----:|:---------------:|:---:|:----------------:|:------------:|
| 1 | Forward Euler | 1 | 1st | ✅ | ✅ | 0.3 |
| **2** | **SSPRK2 (Heun)** | **2** | **2nd** | **✅** | **✅** | **0.45–0.5** |
| 4 | Classic RK4 (composed) | 4 | 4th | ❌ | ❌ (D→D copies) | 0.7 |
| 5 | RK4 graph-safe | 4 | 4th | ❌ | ✅ (single graph) | 0.7 |
| 6 | RK5 graph-safe | 6 | 5th | ❌ | ✅ (single graph) | 0.8 |

The composed RK4 (order 4) uses separate D→D memcpy for stage management and cannot be captured in a CUDA graph. The graph-safe variants (order 5/6) write slopes directly to dedicated buffers and combine in a single final kernel — fully capturable.

### Practical Guidance

| Use Case | Recommended Temporal Order | Rationale |
|----------|:------------------------:|-----------|
| Steady flood routing (days) | 1 or 2 | Temporal error doesn't accumulate |
| Dam-break / flash flood | 4 or 5 | Unsteady wave propagation — phase error smears the wave front |
| Bridge scour (unsteady flow) | 4 or 5 | Ebb/flood asymmetry requires accurate temporal phasing |
| Sediment transport (bedload) | 2 or 4 | Exner equation is stiff — higher order allows larger dt |
| Sediment transport (suspended) | 4 or 5 | Advection-diffusion of concentration needs low phase error |
| Coupled 2D–3D | 4 or 5 | 3D projection sub-stepping — temporal accuracy prevents drift |

---

## Key Architectural Facts

- **Mesh type**: Unstructured polygons (triangles, quads, mixed)
- **Stencil**: 1-ring face neighbors via CSR (`cell_edge_offsets` / `cell_edge_ids`)
- **Gradient method**: Green-Gauss divergence theorem (edge-parallel, CAS atomics on GPU)
- **Reconstruction**: Surface-gradient method (η = h + zb) for well-balancing
- **Flux**: HLLC approximate Riemann solver (positivity-preserving)
- **Active set**: Wet/dry classification + 1-hop neighbor marking + hysteretic hold
- **Degenerate cell handling**: Skip (mode 1), repair inv_area (mode 2), merge to neighbor (mode 3)
