# Phase 8: PCG Pressure Solver for Nonhydrostatic 2D SWE

**Goal**: Implement functional Preconditioned Conjugate Gradient (PCG) pressure correction solver as core of nonhydrostatic predictor-corrector scheme.

## Overview

Phase 8 spans three stages:
- **Stage 8A**: Pressure RHS + matrix-free Laplacian stencil
- **Stage 8B**: PCG iteration loop (convergence, reductions, termination)
- **Stage 8C**: Momentum correction (apply pressure gradient to velocities)

Each stage is independently testable before integration.

---

## Stage 8A: Pressure RHS & Laplacian Stencil

### Kernel: `swe2d_gpu_compute_pressure_rhs()`

**Purpose**: Compute pressure RHS from velocity divergence at previous timestep.

**Input**:
- `d_hu_pred, d_hv_pred`: predicted depth-averaged velocities (m²/s)
- `d_h_pred`: predicted water depth (m)
- Mesh geometry: `cell_area`, `edge connectivity`
- Time step: `dt`

**Output**:
- `d_p_rhs`: right-hand side vector for pressure system
- Semantics: `RHS_i = -div(hu, hv)_i * cell_area_i / dt`

**Formula**:
```
RHS_i = -(1/dt) * Σ_edges [ flux_out_normal_k ]
where flux_out_normal_k = (hu*nx + hv*ny) * face_area_k
```

### Kernel: `swe2d_gpu_laplacian_matrix_free_matvec()`

**Purpose**: Matrix-free Laplacian evaluation on triangular mesh. Builds diagonal for Jacobi preconditioner.

**Input**:
- `d_p`: pressure field (current iterate)
- Mesh: `face_nx, face_ny, face_area` (edge geometry)
- `cell_area`: cell areas for normalization

**Output**:
- `d_Ap`: result of `A * p` (Laplacian action)
- `d_stencil_diag`: diagonal of Laplacian (for preconditioner)

**5-Point Stencil** (on triangular mesh):
```
(A*p)_i = Σ_neighbors [ coeff_{ij} * (p_j - p_i) ]
where coeff_{ij} = (face_area_{ij} / distance_{ij}) / cell_area_i
Diagonal: D_ii = Σ_j |coeff_{ij}|
```

**Boundary Condition**: Neumann BC `∂p/∂n = 0` at domain edges (no ghost cells needed; sum includes only interior neighbors).

---

## Stage 8B: PCG Iteration Loop

### Algorithm Outline

```
Initialize:
  p ← 0 (initial guess)
  r ← RHS - A*p = RHS (since p=0)
  diag ← compute_stencil_diagonal()
  z ← M⁻¹*r = r / diag  (Jacobi preconditioner)
  rz_old ← r·z

Main Loop (while k < max_iter AND ||r||/||RHS|| > tol):
  (store k-th reduction results: rr, rz, pAp)
  
  if (k == 0):
    d ← z
    rz_old ← r·z
  else:
    beta ← rz / rz_old
    d ← z + beta * d  (search direction update)
    rz_old ← rz
  
  Ad ← A*d  (single matvec call)
  pAd ← d·Ad  (dot product)
  alpha ← rz / pAd  (step length)
  
  p ← p + alpha*d  (update pressure)
  r ← r - alpha*Ad  (update residual)
  z ← M⁻¹*r = r / diag
  rz ← r·z
  
  rr ← r·r  (check convergence)
  k ← k+1

Output:
  p (final pressure)
  n_iter (iterations used)
  residual_norm (final ||r||)
  converged (boolean)
```

### Kernel: `swe2d_gpu_pcg_dot_product_reduction()`

**Purpose**: Compute multiple dot products efficiently in a single kernel pass.

**Input**:
- Vectors to dot: `v1, v2, ...` (up to 3 concurrent dot-products)
- `n_cells`

**Output**:
- `d_reductions[]`: array of results

**Method**: Block reduction using shared memory; one block = one result.

### Host Function: `swe2d_gpu_solve_pressure_pcg()`

**Purpose**: CPU-side PCG loop driver. Orchestrates kernel launches and convergence checks.

**Parameters**:
- `max_iter`: max iterations (default 50)
- `tol_rel`: relative residual tolerance (default 1e-3)
- `tol_abs`: absolute residual tolerance (default 1e-6)
- `config.preconditioner`: Jacobi (0), Block-Jacobi (1), ILU0 (2)

**Returns**: `SWE2DNonhydroPcDiag` with iteration count, final residual, convergence flag.

---

## Stage 8C: Momentum Correction

### Kernel: `swe2d_gpu_apply_momentum_correction()`

**Purpose**: Update momentum using pressure gradient.

**Input**:
- `d_p`: pressure field (from PCG solve)
- `d_hu_pred, d_hv_pred`: predicted momentum
- `d_h_pred`: predicted depth (for scaling)
- Time step: `dt`
- Config: velocity correction method (divergence-free vs energy-stable)

**Output**:
- `d_hu, d_hv`: corrected momentum

**Formula** (divergence-free correction):
```
grad_p_x_i ← (1/cell_area_i) * Σ_edges [ p_right * face_area * nx ]
grad_p_y_i ← (1/cell_area_i) * Σ_edges [ p_right * face_area * ny ]

u_corr_i ← -grad_p_x_i / (rho * dt)
v_corr_i ← -grad_p_y_i / (rho * dt)

hu_new ← hu_pred + dt * rho * u_corr = hu_pred - grad_p_x
hv_new ← hv_pred + dt * rho * v_corr = hv_pred - grad_p_y
```

**Safety Checks**:
- Cap |u_corr| ≤ min(CFL_velocity, max_correction_speed)
- Check energy: if |hu_new|² > |hu_pred|² + tolerance → scale correction

### Integration into Nonhydro Step

In `swe2d_gpu_step_nonhydro_predictor_corrector()`:
```
1. Run hydrostatic predictor (existing)
2. Compute pressure RHS (8A kernel)
3. Solve for pressure via PCG (8B host loop + kernels)
4. Apply momentum correction (8C kernel)
5. Update state: (h, hu, hv) ← (h_pred, hu_corrected, hv_corrected)
6. Call exchange kernel if 2D-3D coupling enabled (Phase 9)
7. Return diagnostics
```

---

## Memory Layout

```
NonhydroPressureWorkspace (in SWE2DDeviceState):
├─ Pressure system:
│  ├─ d_p: [n_cells] double     (pressure iterate)
│  ├─ d_p_rhs: [n_cells] double (RHS vector)
│  └─ d_stencil_diag: [n_cells] double (Laplacian diagonal for precond)
├─ PCG storage (13 buffers):
│  ├─ d_r: [n_cells] double (residual)
│  ├─ d_p: [n_cells] double (search direction) [CONFLICT: rename to d_sd]
│  ├─ d_Ap: [n_cells] double (matvec result)
│  ├─ d_z: [n_cells] double (preconditioned residual)
│  ├─ d_r_dot_r: [1] double (global reduction)
│  ├─ d_r_dot_z: [1] double
│  ├─ d_p_dot_Ap: [1] double
│  ├─ Workspace for block reductions
│  └─ ...more as needed
├─ Corrections:
│  ├─ d_u_corr: [n_cells] double (velocity corrections)
│  └─ d_v_corr: [n_cells] double
└─ Diagnostic buffers (GPU diagnostics sync):
   └─ Device diagnostics shadow copy

Total: ~16 KB per 100k cells (negligible for most meshes)
```

---

## Performance Targets

- **RHS compute**: ~0.5 ms (100k cells)
- **Laplacian stencil**: ~1 ms (100k cells, 10 neighbors avg)
- **PCG iteration** (30 iters): 
  - Matvec: ~1 ms/iter → 30 ms total
  - Jacobi precond: ~0.2 ms/iter → 6 ms total
  - Reductions: ~0.1 ms/iter → 3 ms total
  - **PCG total**: ~40 ms (30 iterations)
- **Momentum correction**: ~2 ms (gradient reconstruction)
- **Nonhydro step total**: ~50 ms (100k cells, 30 PCG iters)
- **Hydrostatic step (for comparison)**: ~5 ms

---

## Convergence & Stability

### Typical PCG Behavior

- **Lake-at-rest**: 1-3 iterations (pressure ≈ 0, RHS ≈ 0)
- **Dam-break (initial)**: 20-40 iterations (sharp fronts)
- **Steady flow**: 15-25 iterations (balanced flow field)

### Adaptive Timestep Tuning

If PCG iterations exceed 50, consider:
1. Reduce `dt` (smaller pressure perturbations → easier solve)
2. Increase `tol_rel` (relax convergence target)
3. Switch to Block-Jacobi or ILU0 preconditioner (Phase 9)

### Safeguards

- Hard cap on iterations: prevents solver stall
- Absolute residual floor: prevents oscillatory convergence
- Momentum correction magnitude check: prevents unphysical velocity spikes

---

## Test Cases for Validation

### Test 1: Lake-at-Rest
- Setup: Flat bed, quiescent water
- Expected: RHS ≈ 0, pressure ≈ 0, 1-2 PCG iters
- Pass: mass/momentum conserved, no spurious motion

### Test 2: Dam-Break (1D)
- Setup: Initial discontinuity in depth
- Expected: Symmetric expansion, PCG 30-40 iters
- Pass: Wave speed matches analytical (~√(g*h)), no overshoot

### Test 3: Manufactured Solution
- Setup: Prescribed u(x,t), compute required pressure field
- Expected: Errors decrease with grid refinement
- Pass: L2 error slope matches theory (2nd order)

### Test 4: Energy Stability
- Setup: Initial kinetic energy in velocities
- Expected: |velocity| decreases or stays constant (pressure dissipates energy)
- Pass: No energy blow-up; corrections are physically meaningful

---

## Deferred (Phase 9+)

- **Multigrid preconditioner**: 2-3x faster convergence; requires prolongation/restriction ops
- **Flexible GMRES** (if PCG diverges): more robust for unsymmetric systems
- **Fully-coupled 2D-3D solver** (6D system): currently 2D pressure only
- **Momentum correction feedback** to 3D via exchange kernel

---

## References

- Salgado, A. J. et al. "Stable Schemes for Nonhydrostatic Shallow Water Waves." *J. Comp. Phys.* (2018)
- CG algorithm: Golub & van Loan, *Matrix Computations*, 4th ed., Ch. 11
- CUDA best practices: Nvidia Developer Blog, "Optimized Parallel Reductions"

