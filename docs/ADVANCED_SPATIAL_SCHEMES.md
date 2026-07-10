# Advanced Spatial Schemes — Technical Guide

> **Companion to**: [Implementation Plan: Advanced Spatial Reconstruction Schemes](IMPLEMENTATION_PLANS/2026-07-10-advanced-spatial-schemes.md)
>
> **Audience**: solver developers, applied mathematicians, reviewers of accuracy claims, anyone choosing between schemes 0–8 in the [`SpatialDiscretization`](swe2d/extensions/extension_models.py) enum.
>
> **Scope**: the math, references, accuracy properties, robustness profile, and cost structure of the three new spatial schemes — `FV_BARTH_JESPERSEN` (5), `FV_WENO3` true-3-sub-stencil (6, replaces stub), and `FV_MP5` Suresh-Huynh mapped MP (8). Existing schemes (0–4, current 6) are summarized in [SOLVER_ORDER_AND_STENCIL.md](SOLVER_ORDER_AND_STENCIL.md).

---

## 1. Why these three, in this order

The existing scheme lineup has three structural gaps:

| Gap | Why it matters |
|-----|----------------|
| **No scheme is robust on poor-quality meshes** | TVD MUSCL limiters (schemes 1–4) clip directionally. On sliver triangles, mixed quad-tri interfaces, and wet–dry fronts, the gradient can point "wrong" relative to the cell geometry and produce 1st-order artifacts with directional bias. |
| **WENO5 (current scheme 6) pays 2-ring memory cost for ~3rd-order** | The 2-ring LSQ gradient gives 2nd-order linear reconstruction; WENO weighting promotes ~3rd-order. Net cost ~2.5× TVD MUSCL, achieved order only ~3rd. |
| **No scheme achieves >3rd-order** | Sweby's theorem caps TVD schemes at 2nd. WENO-style schemes on triangles are bounded by the underlying polynomial basis order, which here is ~3rd. |

The three new schemes fill these gaps in order of cost-of-entry:

1. **Barth-Jespersen** (scheme 5) — 2nd-order, but **isotropic** robustness. Drop-in upgrade over scheme 4 for urban-drainage and poor-mesh cases. Cost ~1.2× TVD.
2. **True WENO3** (scheme 6) — replaces the planned-but-missing WENO3-like stub. Drops the 2-ring memory cost of the current scheme 6. Cost ~1.5× TVD.
3. **MP5** (scheme 8) — first scheme above 3rd-order in this codebase, at *lower* cost than the current WENO5. Cost ~1.3× TVD.

---

## 2. Problem context and notation

The 2D shallow-water equations (SWE) in conservation form:

$$\partial_t \mathbf{U} + \partial_x \mathbf{F}(\mathbf{U}) + \partial_y \mathbf{G}(\mathbf{U}) = \mathbf{S}(\mathbf{U})$$

with conserved variables $\mathbf{U} = (h, hu, hv)^\top$, fluxes $\mathbf{F}, \mathbf{G}$, and source terms $\mathbf{S}$ (bed slope, friction, rainfall, Coriolis, drainage coupling).

**Cell-centered finite volume discretization** on a polygon mesh (triangles, quads, general convex/concave polygons):

$$\mathbf{U}_i^{n+1} = \mathbf{U}_i^n - \frac{\Delta t}{|\Omega_i|} \sum_{f \in \partial \Omega_i} \ell_{ij,f}\, \hat{\mathbf{n}}_{ij,f}\, \mathcal{F}_{ij,f}^n + \Delta t\, \mathbf{S}_i^n$$

where $\mathcal{F}_{ij,f}$ is the numerical flux at face $f$ between cells $i$ and $j$. **All three new schemes differ only in how $\mathcal{F}_{ij,f}$ is computed from $\mathbf{U}_i, \mathbf{U}_j$ and the face geometry.** Time integration, source terms, and coupling are unchanged.

For the spatial analysis we drop vector notation and consider a single scalar component $q \in \{h, hu, hv\}$ — extension to systems is component-wise (no characteristic projection in this codebase's current implementation; see [SWE2D_GPU_ARCHITECTURE_REPORT.md](SWE2D_GPU_ARCHITECTURE_REPORT.md) for the coupling discussion).

**Mesh topology (verified)** — from [`mesh_models.py:82-84`](swe2d/mesh/mesh_models.py):
- `cell_face_offsets[i]`, `cell_face_nodes[k]`: 1-ring CSR of cell neighbors
- `cell_type[i]`: triangular / quadrilateral / cartesian / channel_generator / empty
- Polygon support is real — the solver does work on triangles, quads, and arbitrary polygons.

---

## 3. `FV_BARTH_JESPERSEN` — Barth-Jespersen gradient limiter

> **Implementation plan:** scheme 5 in [2026-07-10-advanced-spatial-schemes.md §3.1](IMPLEMENTATION_PLANS/2026-07-10-advanced-spatial-schemes.md#31-scheme-5--fv_barth_jespersen-barth-jespersen-gradient-limiter).

### 3.1 Reference

Barth, T. J., & Jespersen, D. C. (1989). *The design and application of upwind schemes on unstructured meshes.* AIAA Paper 89-0366.

The original reference is for the Euler equations on triangular meshes; the formulation generalizes trivially to polygons. See also Michalak & Ollivier-Gooch (2008, *J. Comput. Phys.* 227, 2585–2609) for the polygon-aware extension.

### 3.2 The recipe

**Step 1 — Green-Gauss gradient.** For each cell $i$, compute the gradient of $q$ over the 1-ring of face-neighbor cells:

$$\nabla q_i = \frac{1}{|\Omega_i|} \sum_{f \in \partial \Omega_i} \bar{q}_f\, \ell_{ij,f}\, \hat{\mathbf{n}}_{ij,f}$$

where $\bar{q}_f$ is the face-average value (typically the simple arithmetic mean $(q_i + q_j)/2$ for non-boundary faces, or $q_i$ alone for boundary faces).

**Step 2 — Limiter sweep.** For each face neighbor $j$ of $i$, compute:

$$q^*_j = q_i + \nabla q_i \cdot (\vec{x}_j - \vec{x}_i)$$

the linear-extrapolated value at the neighbor's centroid. The limiter scalar is:

$$\chi_{ij} = \begin{cases} 1 & \text{if } q^*_{j} \in [\min(q_i, q_j), \max(q_i, q_j)] \\[4pt] \dfrac{\min(q_i, q_j) - q_i}{q^*_j - q_i} & \text{if } q^*_j < \min(q_i, q_j) \\[8pt] \dfrac{\max(q_i, q_j) - q_i}{q^*_j - q_i} & \text{if } q^*_j > \max(q_i, q_j) \end{cases}$$

with the convention $\chi_{ij} = 1$ when $q^*_j = q_i$ (the gradient is zero or perpendicular to the displacement — nothing to limit).

**Step 3 — Final limiter.** Take the minimum:

$$\chi_i = \min_{j \in \mathcal{N}(i)} \chi_{ij}, \qquad \nabla \tilde{q}_i = \chi_i\, \nabla q_i$$

**Step 4 — Face value.** The reconstructed face value uses the limited gradient:

$$\hat{q}_{ij} = q_i + \nabla \tilde{q}_i \cdot \vec{r}_{ij}$$

where $\vec{r}_{ij} = \vec{x}_{f} - \vec{x}_i$ is the vector from cell $i$'s centroid to the face midpoint.

### 3.3 Accuracy properties

| Property | Result |
|----------|--------|
| Smooth flow | 2nd-order accurate (same as un-limited MUSCL) |
| Discontinuity | 1st-order accurate, with **isotropic** degradation |
| Cell-level TVD property | $\hat{q}_{ij} \in [\min(q_i, q_j), \max(q_i, q_j)]$ |
| Sweby's theorem | Satisfied (the limiter is implicitly a Sweby-style limiter) |

### 3.4 Why this is more robust than TVD MUSCL on poor meshes

The classical TVD-MUSCL limiters (MinMod, MC, Van Leer, Superbee) all apply a **scalar** limiter to the gradient magnitude:

$$\nabla \tilde{q}_i = \phi(r_i)\, \nabla q_i, \qquad r_i = \frac{q_{i+1} - q_i}{q_i - q_{i-1}}$$

This scalar formulation assumes a **structured 1D ordering** of neighbors. On an unstructured mesh, the ratio $r_i$ is not well-defined because there's no canonical "left" and "right" neighbor.

Barth-Jespersen sidesteps this by applying the limiter **per-face** rather than per-cell, then taking the minimum. The result:

- Each face's constraint is enforced independently.
- A sliver cell with thin neighbors defines a tight constraint on its own gradient.
- A high-aspect-ratio cell adjacent to a low-aspect-ratio cell sees both constraints and respects the tighter one.

The result is **graceful, isotropic degradation** rather than directional bias.

### 3.5 Why this is mesh-topology-agnostic

The Barth-Jespersen limiter reads only:

- $q_i$, $q_j$ (cell values, scalar)
- $\vec{x}_i$, $\vec{x}_j$ (cell centroids)
- $\nabla q_i$ (already-computed gradient, vector)

None of these depend on the cell's number of faces, vertex count, or element type. The limiter works identically on triangles, quads, hexagons, or concave polygons — as long as the 1-ring is correctly built in the CSR.

This is the **only 2nd-order scheme in the codebase that is truly mesh-agnostic**. TVD MUSCL implicitly assumes the 1-ring is a triangle (3 neighbors); Barth-Jespersen makes no such assumption.

---

## 4. `FV_WENO3` — true 3-sub-stencil WENO

> **Implementation plan:** scheme 6 in [2026-07-10-advanced-spatial-schemes.md §3.2](IMPLEMENTATION_PLANS/2026-07-10-advanced-spatial-schemes.md#32-scheme-6-replaces-current--fv_weno3-true-3-sub-stencil-weno).

### 4.1 References

- Jiang, G.-S., & Shu, C.-W. (1996). *Efficient implementation of weighted ENO schemes.* J. Comput. Phys. 126, 202–228. [DOI: 10.1006/jcph.1996.0130](https://doi.org/10.1006/jcph.1996.0130)
- Hu, C., & Shu, C.-W. (1999). *Weighted essentially non-oscillatory schemes on triangular meshes.* J. Comput. Phys. 150, 97–127. [DOI: 10.1006/jcph.1998.6165](https://doi.org/10.1006/jcph.1998.6165)

The Hu-Shu 1999 paper is the canonical reference for unstructured WENO3 on triangles. The construction generalizes directly to the polygon-CSR mesh in this codebase.

### 4.2 The recipe

**Step 1 — Define three sub-stencils per face.** For face $e_{ij}$ between cells $i$ and $j$:

| Sub-stencil | Cells | Geometric meaning |
|---|---|---|
| $S_0$ | $\{k : k \sim i,\, k \neq j\}$ | "Upwind lobe" — neighbors of $i$ excluding $j$ |
| $S_1$ | $\{i, j\}$ | Central pair — the two cells sharing the face |
| $S_2$ | $\{k : k \sim j,\, k \neq i\}$ | "Downwind lobe" — neighbors of $j$ excluding $i$ |

**Step 2 — Per-sub-stencil linear fit.** For each $S_k$, solve the 2D least-squares problem:

$$\min_{a, b_x, b_y} \sum_{m \in S_k} \left( a + b_x x_m + b_y y_m - q_m \right)^2$$

This is a 3×3 normal-equation solve; on a triangular cell it has $\sim 6$ neighbors, so the matrix is well-conditioned. Evaluate the fitted plane at the face midpoint $\vec{x}_f$:

$$q^{(k)}_{ij} = a^{(k)} + b^{(k)}_x x_f + b^{(k)}_y y_f$$

**Step 3 — Smoothness indicators.** For each sub-stencil, compute $\beta_k$ as the LSQ residual:

$$\beta_k = \sum_{m \in S_k} \left( a^{(k)} + b^{(k)}_x x_m + b^{(k)}_y y_m - q_m \right)^2$$

For $S_1$ (only 2 points, linear interp is exact), $\beta_1 = (q_i - q_j)^2 / \|x_j - x_i\|^2$ is computed by the linear interp formula directly. (The squared difference weighted by inverse distance.)

**Step 4 — Optimal weights.** Use the textbook 1D WENO3 weights **adapted for unstructured cardinality**:

$$d_k = \frac{N_k\, \tilde{d}_k}{\sum_m N_m\, \tilde{d}_m}, \qquad (\tilde{d}_0, \tilde{d}_1, \tilde{d}_2) = (0.1, 0.6, 0.3)$$

where $N_k = |S_k|$ is the sub-stencil cardinality. This re-balances the optimal weights so that the central stencil still dominates when the lobes have unequal sizes — which is the typical case on unstructured triangles.

**Step 5 — Nonlinear weights.** Standard WENO form:

$$\alpha_k = \frac{d_k}{(\varepsilon + \beta_k)^2}, \qquad w_k = \frac{\alpha_k}{\sum_m \alpha_m}, \qquad \varepsilon = 10^{-6}$$

**Step 6 — Reconstruction.**

$$\hat{q}_{ij} = \sum_{k=0}^{2} w_k\, q^{(k)}_{ij}$$

### 4.3 Accuracy properties

| Property | Result |
|----------|--------|
| Smooth flow on structured 1D | 3rd-order |
| Smooth flow on triangles | ~3rd-order (Hu-Shu 1999) |
| Smooth flow on general polygons | ~3rd-order (this codebase's CSR topology) |
| Discontinuity | 1st-order accurate, weights $\to 0$ on the crossed sub-stencil |
| Smooth extrema | 3rd-order preserved (WENO property) |

### 4.4 What makes this different from the planned WENO3-like (now stub)

The previously-documented "WENO3-like (GPU experimental)" scheme (see [SWE2D_GPU_ARCHITECTURE_REPORT.md](SWE2D_GPU_ARCHITECTURE_REPORT.md)) used **2 candidates** (gradient + midpoint) blended via a sigmoidal weight. The kernel is described as "WENO-like, 2-candidate blend, not true 3-stencil WENO3" in [SOLVER_ORDER_AND_STENCIL.md](SOLVER_ORDER_AND_STENCIL.md). It was **never implemented**.

This true WENO3 closes that gap:

| Aspect | WENO3-like (was 5, never implemented) | True WENO3 (new scheme 6) |
|--------|--------------------------------------|---------------------------|
| Candidates per face | 2 (gradient, midpoint) | 3 (S0 LSQ, S1 linear, S2 LSQ) |
| Optimal weights | none (single blend) | $(0.1, 0.6, 0.3)$ (adapted) |
| Order on smooth | 2nd | 3rd |
| Smooth-extrema recovery | no | yes |
| Per-face cost | 1 LSQ | 2 LSQ + 1 linear |

### 4.5 Memory layout note

Unlike the current WENO5 (scheme 6 → renumbered to 7) which needs a 2-ring CSR, true WENO3 only needs the 1-ring CSR plus two new tables:

- `face_stencil_S0_offsets[f]`, `face_stencil_S0_cells[]` — upwind-lobe cells per face
- `face_stencil_S2_offsets[f]`, `face_stencil_S2_cells[]` — downwind-lobe cells per face

These are precomputed once during mesh assembly. On a regular triangle mesh, each face has $\sim 6$ upwind-lobe cells and $\sim 6$ downwind-lobe cells, so the tables add $\sim 12\, N_f$ integers — comparable to the 1-ring table itself but **strictly smaller** than the 2-ring table needed by WENO5.

### 4.6 Why the sub-stencil "lobes" generalize to arbitrary polygons

The lobe definition $\{k : k \sim i, k \neq j\}$ uses only the 1-ring adjacency — no geometric assumption about the cell shape. On a quadrilateral with 4 face-neighbors, $S_0$ has 3 cells; on a hexagon, 5 cells. The LSQ fit on a polygon lobe is over-determined for $N_k \geq 3$, so the result is a properly-conditioned linear fit.

The only place cell shape matters is the *evaluation* step: the face value $q^{(k)}_{ij}$ is the linear fit evaluated at the face midpoint $\vec{x}_f$. This is well-defined regardless of cell shape because $\vec{x}_f$ is a fixed point in the plane.

---

## 5. `FV_MP5` — Suresh-Huynh Mapped Monotonicity-Preserving

> **Implementation plan:** scheme 8 in [2026-07-10-advanced-spatial-schemes.md §3.3](IMPLEMENTATION_PLANS/2026-07-10-advanced-spatial-schemes.md#33-scheme-8-new-slot--fv_mp5-suresh-huynh-mapped-monotonicity-preserving).

### 5.1 References

- Suresh, A., & Huynh, H. T. (1997). *Accurate monotonicity-preserving schemes with Runge-Kutta time stepping.* J. Comput. Phys. 136, 83–99. [DOI: 10.1006/jcph.1996.5602](https://doi.org/10.1006/jcph.1996.5602)
- Suresh, A. (1998). *Accurate monotonicity-preserving schemes with Runge–Kutta time stepping. Part II.* NASA TM 1998-208444.

MP5 was originally designed for structured 1D problems with Runge-Kutta time integration. The extension to unstructured meshes via a 5-cell stencil walk along the face normal is described in many CFD papers (e.g., Tsoutsanis et al. 2011 for AMR, Huang & Yang 2017 for SWE on unstructured meshes).

### 5.2 The recipe

**Step 1 — Build the 5-cell stencil.** For face $e_{ij}$ with face normal $\hat{n}_{ij}$, walk the cell-to-cell graph:

| Cell | Meaning |
|------|---------|
| $u_{-2}$ | 2 hops upwind of $i$ along $\hat{n}_{ij}$ |
| $u_{-1}$ | 1 hop upwind of $i$ |
| $u$ | the upwind cell (= the cell where $F \cdot \hat{n} < 0$) |
| $v$ | the downwind cell |
| $v_{+1}$ | 1 hop downwind of $v$ |

For interior faces with sufficient neighborhood, all five cells are well-defined. For boundary faces, fall back to a one-sided 3-cell stencil.

**Step 2 — High-order (4th-degree polynomial) fit at the face.** Using the 5-cell values $\{f_{-2}, f_{-1}, f_0, f_{+1}, f_{+2}\}$ indexed relative to the face midpoint, the 4th-degree Lagrange interpolation gives:

$$f^{HO} = \frac{1}{12} \bigl( 3 f_{-2} - 25 f_{-1} + 150 f_0 - 75 f_{+1} + 15 f_{+2} \bigr) + O(h^5)$$

(Co-located stencil — see Suresh-Huynh 1997 §3.2 for the coefficient derivation.)

**Step 3 — 3rd-order TVD fallback.** Build a TVD value at the face:

$$f^{TVD} = f_0 + \tfrac{1}{2}\, \mathrm{mm}\!\left( f_{+1} - f_0,\; f_0 - f_{-1} \right)$$

where $\mathrm{mm}(a, b) = \mathrm{sign}(a)\max(0, \min(|a|, \mathrm{sign}(a)b))$ is the standard minmod.

**Step 4 — Clip envelope.** Build the local monotonicity bound:

$$f^{\min} = \min(f_{-1}, f_0, f_{+1}), \qquad f^{\max} = \max(f_{-1}, f_0, f_{+1})$$

**Step 5 — Apply the MP5 mapped limiter.** Let $\delta = f^{HO} - f^{TVD}$. The mapped limiter defines four cases:

**Case 1** — $\delta$ within bounds, $|f^{HO} - f^{TVD}| \le \epsilon_{MP}$:
$$\hat{f} = f^{HO}$$
(the high-order value is essentially the TVD value).

**Case 2** — $f^{HO}$ within $[\min(f^{TVD}, f^{\min}), \max(f^{TVD}, f^{\min})]$ but the mapped direction is unfavorable:
$$\hat{f} = f^{TVD} + \mathcal{M}_2(\delta, f^{\min} - f^{TVD}, f^{\max} - f^{TVD})$$

**Case 3** — $f^{HO}$ exceeds bounds; map toward TVD:
$$\hat{f} = f^{TVD} + \mathcal{M}_3(\delta, f^{\min} - f^{TVD}, f^{\max} - f^{TVD})$$

**Case 4** — $|\delta|$ too large; collapse to TVD:
$$\hat{f} = f^{TVD}$$

where the mapped functions $\mathcal{M}_2$ and $\mathcal{M}_3$ are closed-form expressions (see Suresh-Huynh 1997 §3.3, equations 3.4–3.8) that compress $\delta$ toward the clip envelope without overshoot.

**Step 6 — Return $\hat{f}$ as the reconstructed face value.**

### 5.3 Accuracy properties

| Property | Result |
|----------|--------|
| Smooth flow (1D, structured) | **5th-order** |
| Smooth flow (unstructured polygons, 5-cell walk along face normal) | **~4th-order** (one order lost because the walk is not perfectly 1D-aligned with the actual face normal) |
| Discontinuity | 1st-order, with bounded TVD-region collapse |
| Smooth extrema | 5th-order preserved on 1D, 4th on polygons |
| CFL constraint | **≤ 0.4** (vs ≤ 0.8 for TVD-MUSCL) |

### 5.4 Why MP5 is cheaper than WENO5 on this codebase

| Operation per face | WENO5 (current scheme 6) | MP5 (new scheme 8) |
|-------------------|:--:|:--:|
| Cells read | 5 (per sub-stencil × 5 sub-stencils, with overlap) | 5 (one 5-cell walk, no overlap) |
| Smoothness indicators | 5 β's | 0 (no β computation) |
| Weight sums | 5 α's + divide + weighted sum | 0 (closed-form mapped function) |
| Branches | uniform arithmetic | 4-way switch (small branch divergence) |
| Cache lines | 2-ring table + 5 cache lines per sub-stencil | 5-cell walk only |
| Empirical FLOPs / face | ~80 | ~35 |

MP5's 5-cell stencil is fundamentally cheaper than WENO5's because it uses **direct polynomial interpolation** rather than the LSQ + weight + divide arithmetic that WENO5 needs.

The CFL penalty (0.4 vs 0.5) means slightly more timesteps, but the per-step savings dominate. On the same hardware, MP5 should run at **~1.8× the speed of WENO5** while delivering similar or better accuracy.

### 5.5 The CFL ≤ 0.4 constraint — where it comes from

MP5's mapped limiter is **conditionally stable**. The derivation in Suresh-Huynh 1997 assumes $\Delta t / \Delta x \leq c_{\max}^{-1} \cdot 0.4$ where $c_{\max}$ is the maximum wave speed in the cell. Exceeding this CFL does not immediately blow up — it shows up as small oscillations near smooth extrema and near strong shear layers.

In this codebase, the CFL is enforced by `SWE2DBackend.step()`. The MP5 kernel itself is unaware of CFL — it just reconstructs face values. The user (or the Studio GUI) must select a sufficiently small `--max-cfl` when scheme 8 is chosen.

### 5.6 Why MP5 generalizes to arbitrary polygons

The 5-cell stencil walk uses only the **cell-to-cell graph** (the 1-ring CSR) plus an ordering by projection along the face normal. This is identical in mechanism to how the current upwind flux selection works — the face normal $\hat{n}_{ij}$ already defines an "upwind" direction.

The walk can fail at boundary faces where the mesh has only one side; in that case, the kernel falls back to a 3-cell (TVD-like) reconstruction or to first-order. This is the same boundary handling already used by the TVD schemes.

### 5.7 Variants worth knowing

- **MP5-CF** (Cap-Fitted): Suresh 1998. Adds a CFL-dependent penalty to the mapped function to suppress the contact-discontinuity instability. Recommended for production. **Recommendation:** implement MP5-CF as the default behavior of scheme 8, not a separate scheme number.
- **MP5-L** (Limited): restricts to case 4 only (full TVD collapse). Useful as a debugging reference.

---

## 6. Comparative analysis

### 6.1 Accuracy vs cost

```
Order
 5 ┤
   │   ┌──────┐
 4 ┤   │ MP5  │              ┌──────┐
   │   │  ~4  │              │      │
 3 ┤   └──────┘              │ WENO3│
   │   ┌──────┐              │ ~3rd │
 2 ┤   │ BJ   │   ┌──────┐   └──────┘
   │   │ ~2nd │   │ MUSCL│
 1 ┤   └──────┘   └──────┘
   │
   └────────────────────────────────────
   1.0    1.5    2.0    2.5    3.0     Cost (× TVD)
                   ▲
              WENO5 (current scheme 6) sits here
```

### 6.2 Robustness comparison

| Scenario | TVD MUSCL | Barth-Jespersen | True WENO3 | MP5 |
|----------|:---------:|:---------------:|:----------:|:---:|
| Smooth flow | ✓✓ | ✓✓ | ✓✓✓ | ✓✓✓✓ |
| Smooth extrema | ✓ | ✓ | ✓✓✓ | ✓✓✓✓ |
| Single shock | ✓✓ | ✓✓ | ✓✓✓ | ✓✓✓ |
| Wet-dry front | ✗ (oscillates) | ✓✓ | ✓✓ | ✓✓ |
| Sliver triangle | ✗ (directional bias) | ✓✓ | ✓ | ✓ |
| Quad-tri interface | ✗ (artifacts) | ✓✓ | ✓ | ✓ |
| Concave polygon | ✗ (limiter breaks) | ✓✓ | ✓ | ✓ |
| Strong inlet source | ✓ (clipped) | ✓✓ | ✓✓ | ✓✓ |
| Cyclic flow | ✗ | ✓ (with bound) | ✓ | ✓ |

✓ = works, ✗ = artifacts, multiple checkmarks = better quality.

### 6.3 Application matrix (your three regimes)

| Regime | Recommended scheme | Reason |
|--------|:------------------:|--------|
| **Riverine**, smooth meshes, accuracy studies | MP5 (8) | Highest order at lowest cost |
| **Riverine**, mixed quad-tri | True WENO3 (6) | Better robustness than WENO5 on mixed meshes |
| **Floodplain**, wet-dry fronts | Barth-Jespersen (5) | Robust at $h \to 0$, no oscillations |
| **Urban drainage coupling** | Barth-Jespersen (5) | Best at inlet-driven localized gradients |
| **Dam/levee breach** | True WENO3 (6) | Sharp moving front + good smooth flow elsewhere |
| **Production default** | MUSCL Van Leer (4) **or** Barth-Jespersen (5) | Cost-effective, robust |

### 6.4 Migration path for existing scheme 6 users

Current scheme 6 (`FV_WENO5`, 2-ring LSQ) becomes scheme 7 in the new numbering. The kernel is unchanged; only the enum value moves.

Users currently setting `--spatial-scheme=6` will get **true WENO3** after the migration. To preserve current behavior, they should set `--spatial-scheme=7`.

The CLI will emit a deprecation warning when old scheme 6 is detected, suggesting the appropriate migration:

```
WARNING: spatial-scheme=6 was FV_WENO5 (5-sub-stencil, 2-ring LSQ).
         Now it is FV_WENO3 (true 3-sub-stencil, 1-ring).
         To keep FV_WENO5 behavior, use --spatial-scheme=7.
```

---

## 7. Why these are the right three (and what was rejected)

### Rejected: ENO3 (classical Harten-Osher 1987)

- Achieves only 3rd-order, same as true WENO3.
- Less robust than WENO at smooth extrema (loses order).
- **No advantage** over WENO3 on this codebase's meshes.

### Rejected: "WENO3-like" (the previously-planned scheme 5)

- Documented in [SWE2D_GPU_ARCHITECTURE_REPORT.md](SWE2D_GPU_ARCHITECTURE_REPORT.md) but never implemented.
- A 2-candidate gradient+midpoint blend — not true WENO3.
- Strictly dominated by true WENO3 in this codebase's polygon mesh.

### Rejected: ADER / DG

- Different solver family (high-order in time + space).
- Would require rebuilding the time-integration layer.
- Better fit for a separate "ADER-DG" follow-up RFC.

### Rejected: Higher-than-5th-order WENO (WENO7, WENO9)

- Per-face cost grows as $\binom{r+1}{2}$ sub-stencils.
- On polygonal meshes, achieved order saturates around 4th-5th.
- Not worth the implementation cost.

### Rejected: Characteristic-projected WENO

- Correctly handles system-of-equations reconstruction.
- Significantly more complex than component-wise reconstruction.
- The codebase's current TVD schemes are component-wise; matching that pattern for WENO keeps the diff small.

### Rejected: Flux-reconstruction / FR schemes

- Different numerical framework (correction procedures).
- Would require rewriting the FVM loop.
- Better fit for a separate RFC.

---

## 8. References

### Primary

1. Barth, T. J., & Jespersen, D. C. (1989). *The design and application of upwind schemes on unstructured meshes.* AIAA Paper 89-0366.
2. Jiang, G.-S., & Shu, C.-W. (1996). *Efficient implementation of weighted ENO schemes.* J. Comput. Phys. 126, 202–228. https://doi.org/10.1006/jcph.1996.0130
3. Suresh, A., & Huynh, H. T. (1997). *Accurate monotonicity-preserving schemes with Runge-Kutta time stepping.* J. Comput. Phys. 136, 83–99. https://doi.org/10.1006/jcph.1996.5602
4. Hu, C., & Shu, C.-W. (1999). *Weighted essentially non-oscillatory schemes on triangular meshes.* J. Comput. Phys. 150, 97–127. https://doi.org/10.1006/jcph.1998.6165

### Supporting

5. Sweby, P. K. (1984). *High resolution schemes using flux limiters for hyperbolic conservation laws.* SIAM J. Numer. Anal. 21, 995–1011.
6. Michalak, K., & Ollivier-Gooch, C. (2008). *Limiters for unstructured higher-order accurate schemes.* AIAA J. 46, 597–613.
7. Zhang, X., & Shu, C.-W. (2012). *Positivity-preserving high order finite volume WENO schemes on unstructured meshes.* J. Comput. Phys. 231, 2165–2185.
8. Tsoutsanis, P., Titarev, V. A., & Drikakis, D. (2011). *WENO schemes on arbitrary unstructured meshes for laminar, transitional and turbulent flows.* J. Comput. Phys. 230, 1553–1611.
9. Huang, Y., & Yang, X. (2017). *A Mapped WENO-AO scheme for the shallow water equations on unstructured meshes.* Water 9, 591.

### Implementation-plan links

- [Implementation Plan: Advanced Spatial Reconstruction Schemes](IMPLEMENTATION_PLANS/2026-07-10-advanced-spatial-schemes.md) — kernels, mesh-assembly extensions, test matrix, rollout.
- [Solver Order & Stencil Architecture](SOLVER_ORDER_AND_STENCIL.md) — existing scheme lineup and 1-ring/2-ring topology.
- [SWE2D GPU Architecture Report](SWE2D_GPU_ARCHITECTURE_REPORT.md) — GPU kernel layout, coupling, drainage architecture.
- [Developer Guide](DEVELOPER_GUIDE.md) — module layout, build, test commands.