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

On non-orthogonal meshes the Green-Gauss gradient formally drops to **1st-order** because the face-average approximation `(q₀+q₁)/2` no longer cancels the centroid offset error. The TVD limiter compensates by limiting more aggressively — effectively pushing the scheme back toward 1st-order on the most skewed cells.

**GPU vs CPU difference in the reconstruction:**

| Path | Extrapolation | Non-orthogonal accuracy |
|------|--------------|------------------------|
| **GPU** (`swe2d_flux_kernel`) | `q₀ + φ ⋅ ∇q₀ · (x_f − x_c₀)` | Geometrically exact direction |
| **CPU** (`swe2d_solver.cpp`) | `q₀ + φ ⋅ 0.5 ⋅ (q₁−q₀)` | Midpoint-only, ignores face geometry |

The GPU path is more accurate on non-orthogonal meshes — it extrapolates to the actual face midpoint using the full gradient vector. The CPU path simplifies to a midpoint step and was explicitly flagged in the code as producing neutral-to-unstable behaviour on non-trivial meshes.

### What Higher Spatial Order (>2nd) Would Enable

| Application | Current 2nd-order limitation | Benefit of 3rd–4th order |
|------------|-----------------------------|--------------------------|
| Wet/dry front tracking | Front artificially lengthened by ~2–3 cells | Sharper front, ~1 cell |
| Bridge contraction/expansion | Velocity jet diffused, afflux underestimated | Sharper velocity field, better afflux prediction |
| **Scour / sediment transport** | V³ nonlinearity amplifies velocity errors → ~33% transport error per 10% V error | Reduced truncation error → tighter scour depth bounds |
| Scour hole resolution | ~8–10 cells needed for <5% amplitude error | ~3–4 cells with 3rd-order |
| Bedform migration | Numerical diffusion damps migration rate | Better preservation of bed features |

For most floodplain engineering tolerances (±20% water level, factor-of-2 scour), 2nd-order is adequate. For **detailed bridge scour analysis** (±10% velocity, ±20% scour depth), the jump to 3rd-order spatial would be beneficial.

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
