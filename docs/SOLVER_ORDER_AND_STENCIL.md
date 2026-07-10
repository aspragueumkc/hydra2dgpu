# SWE2D Solver: Spatial/Temporal Order & Stencil Architecture

> **Audience**: solver developers, applied mathematicians, reviewers of accuracy
> claims, anyone choosing a spatial scheme for a production simulation.

This document explains how the SWE2D solver achieves its advertised accuracy
on unstructured meshes — and where the practical accuracy ceiling sits given
the 1-ring cell-edge stencil and TVD-limiter constraints. It is the
reference for choosing between `FV_FIRST_ORDER`, `FV_MUSCL_*`, `FV_BARTH_JESPERSEN`,
`FV_WENO3`, `FV_WENO5`, and `FV_MP5` schemes.

> **See also**: [Advanced Spatial Schemes — Technical Guide](ADVANCED_SPATIAL_SCHEMES.md)
> for the math, references, and properties of `FV_BARTH_JESPERSEN` (5),
> `FV_WENO3` true-3-sub-stencil (6), and `FV_MP5` (8). Those three schemes are
> introduced in [Implementation Plan: Advanced Spatial Reconstruction Schemes](IMPLEMENTATION_PLANS/2026-07-10-advanced-spatial-schemes.md).

## Spatial Discretization

### Stencil Definition

The FVM stencil is built from the **cell-edge CSR** (`cell_edge_offsets[]` / `cell_edge_ids[]`) constructed during mesh assembly in `swe2d_build_mesh_poly`. Each cell's stencil consists of all face-adjacent (edge-sharing) neighbors:

- **Unstructured triangles** → **3 neighbors** (interior)
- **Quadrilaterals** → **4 neighbors**
- **General polygons** → **N neighbors** (equals vertex count)
- **Boundary cells** → N−1 active face neighbors (one edge is boundary)

The stencil width is always the **1-ring** — only immediate face neighbors. No scheme expands the stencil beyond this; higher order changes the *reconstruction accuracy*, not which cells participate.

### Spatial Schemes (0–8)

| Scheme | Value | Reconstruction | Limiter | Max. Order | Notes |
|--------|:-----:|---------------|---------|:----------:|-------|
| `FV_FIRST_ORDER` | 0 | None (cell-center → face) | None | **1st** | Piecewise constant — most diffusive, most robust |
| `FV_MUSCL_FAST` | 1 | Green-Gauss + TVD | **Superbee** | **2nd** | Most aggressive TVD, sharpest fronts, may overshoot on skewed meshes |
| `FV_MUSCL_MINMOD` | 2 | Green-Gauss + TVD | **MinMod** | **2nd** | Most conservative TVD, most stable, safest for mixed/hybrid meshes |
| `FV_MUSCL_MC` | 3 | Green-Gauss + TVD | **MC (monotonized-central)** | **2nd** | Balanced between accuracy and stability |
| `FV_MUSCL_VAN_LEER` | 4 | Green-Gauss + TVD | **Van Leer** (smooth) | **2nd** | Smooth limiter, graceful degradation on skewed cells — best all-rounder |
| `FV_BARTH_JESPERSEN` | **5** | Green-Gauss + Barth-Jespersen | **Barth-Jespersen** (isotropic) | **2nd** | Graceful 1st-order degradation on poor meshes; same CFL as MUSCL; best for urban drainage, mixed-element meshes |
| `FV_WENO3` | **6** | 3-sub-stencil WENO (Hu-Shu) | **Nonlinear weights** (no TVD) | **3rd** | True 3-sub-stencil WENO on 1-ring; smoother than TVD on smooth flows; replaces old WENO3-like stub |
| `FV_WENO5` | **7** | WENO5 + LSQ 2-ring gradient | Nonlinear weights + 2-ring stencil | **~3rd** | True 5th-order WENO on unstructured via 2-ring least-squares gradient; GPU-first *(moved from scheme 6)* |
| `FV_MP5` | **8** | 5-cell directional walk + MP5 mapped limiter | **MP5** (Suresh-Huynh, 4-case) | **4th** | Highest-order option; CFL ≤ 0.4; best for smooth flows where accuracy matters more than speed |

All schemes 1–5 (MUSCL variants and Barth-Jespersen) use the **surface-gradient method** (Zhou et al. 2001): reconstruct η = h + zb via ∇η, then convert back to depth. This preserves lake-at-rest to machine precision independent of mesh irregularity. WENO3 and MP5 reconstruct primitive variables directly.

### Accuracy Limit

**Order depends on scheme selection.** TVD schemes are bounded by:

1. **Sweby's theorem** — any TVD scheme with a 3-point stencil is at most 2nd-order accurate in smooth regions.
2. **1-ring stencil** — standard Green-Gauss reconstruction is at most 2nd-order on a 1-ring. True WENO3 (scheme 6) achieves 3rd-order on the 1-ring via 3 sub-stencils per edge.
3. **WENO3-like stub replaced** — the planned-but-never-implemented 2-candidate blend (old scheme 5) is superseded by true WENO3 (scheme 6), which uses 3 sub-stencils. See [ADVANCED_SPATIAL_SCHEMES.md](ADVANCED_SPATIAL_SCHEMES.md) for details.
4. **MP5 (scheme 8)** circumvents the 1-ring ceiling via a directional 5-cell walk along the face normal, achieving 4th-order at CFL ≤ 0.4.

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
Face state: q_f = q₀ + φ · ∇q₀ · (x_f − x_c₀)   ← extrapolation to actual face
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

#### How the Safety Mechanisms Interact with Non-Orthogonality

| Mechanism | File Location | Effect on Non-Orthogonal Meshes |
|-----------|---------------|--------------------------------|
| **Pair-bounds clamping** | `tvd_reconstruct` lambda | Fires more often — the erroneous gradient overshoots the cell-pair bounds. Each clamp forces 1st-order on that edge. |
| **Shallow-front fallback** | `enable_shallow_front_recon_fallback` | Depth-based check, independent of mesh quality. Still active. |
| **Surface-gradient method** (reconstruct $\eta = h + z_b$, convert back) | Applied to depth only | **Lake-at-rest is exact regardless of mesh skew** — $\nabla\eta = 0$ for constant $\eta$, so $r=0$, $\phi=0$, and the reconstruction returns $q_0/q_1$ unchanged. This is a provable property of the Zhou et al. method. |
| **Momentum cap** | Applied after reconstruction | Geometry-independent, always active. Caps prevent unbounded velocity from erroneous gradients on skewed cells. |
| **Barth-Jespersen limiter** (scheme 5) | GPU: `barth_jespersen_limiter` kernel | Applies per-face isotropic limiting — on skewed cells the gradient is uniformly reduced by the minimum neighbor ratio $\psi = \min_j \phi_{ij}$, degrading to 1st-order isotropically. Unlike TVD limiters, does **not** depend on the Green-Gauss gradient direction for monotonicity detection. |
| **WENO3 nonlinear blend** (scheme 6) | GPU: `weno3_reconstruct` kernel | 3-sub-stencil smoothness indicators $\beta_k$ weighted by Jiang-Shu formula. On skewed cells the stencil with largest $\beta$ is down-weighted, shifting weight to smoother stencils — effectively reducing to 2nd or 1st order where mesh quality is poor. |
| **MP5 mapped limiter** (scheme 8) | GPU: `mp5_reconstruct` kernel | 4-case Suresh-Huynh mapped function. On skewed cells the high-order candidate $f^{HO}$ falls outside the TVD interval, triggering case 2 (midpoint) or case 4 (full TVD collapse). Mesh-independent monotonicity enforcement. |
| **Active-set dry skipping** | GPU: `d_active` check in flux kernel | Unaffected by mesh quality. |

#### Empirical Validation

The sweep at `tests/swe2d_nonorth_gpu_sweep_common.py` tests all 9 spatial schemes × 2 temporal orders × 2 Godunov modes at `skew_fraction_dx=0.25$ (25% of cell width interior node perturbation) on GPU.

The orthogonal vs non-orthogonal solution comparison uses these tolerances:
- `rel_l2_h < 10%` — depth field differs by up to 10%
- `rel_l2_hu < 15%` — momentum field differs by up to 15%
- `|q_nonorth - q_orth| / q_orth < 8%` — mean discharge differs by up to 8%

These tolerances are relatively loose, confirming that non-orthogonality introduces measurable (but stable) error. A formal order-of-accuracy study across mesh refinement levels on orthogonal vs skewed grids has not yet been performed.

#### Practical Guidance for Mesh Generation

| Skew Level ($\theta_e$) | Expected Accuracy | Recommended Scheme |
|:-----------------------:|:-----------------:|:------------------:|
| $< 10^\circ$ (good quad, well-shaped tri) | Full 2nd–4th order | MP5 (8), WENO3 (6), Van Leer (4), MC (3) |
| $10^\circ$–$30^\circ$ (reasonable unstructured) | ~1.5th–2nd order | Van Leer (4), Barth-Jespersen (5), or MinMod (2) |
| $30^\circ$–$60^\circ$ (poor quality) | 1st-order dominant | Barth-Jespersen (5) — isotropic degradation, or MinMod (2) |
| $> 60^\circ$ (nearly degenerate) | 1st-order + risk of instability | First-order (0) or fix mesh |

For production floodplain work, typical meshes generated by GMSH or TQMesh with reasonable element quality fall in the $10^\circ$–$25^\circ$ range, where the higher-order schemes still provide meaningful benefit over first-order.

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

---

## Related Documentation

- **[Documentation Index](INDEX.md)** — All guides by audience
- **[GPU Architecture Report](SWE2D_GPU_ARCHITECTURE_REPORT.md)** — Device-side kernel details
- **[Developer Guide](DEVELOPER_GUIDE.md)** — Spatial scheme enum, runtime module
- **[cpp/GPU_KERNEL_STRATEGY.md](cpp/GPU_KERNEL_STRATEGY.md)** — Kernel launch hierarchy
- **[cpp/ARCHITECTURE.md](cpp/ARCHITECTURE.md)** — Mesh structure, solver config
