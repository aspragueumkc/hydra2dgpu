# Drainage Solver Equation Parity Plan — SWMM Alignment

## Goal
Align the three drainage `solver_mode` options with their SWMM equivalents by fixing
equation fidelity. Mode 1 becomes the simplified diffusion wave (inertia-free) that
SWMM's kinematic wave approximates in the limit. Modes 0 and 2 get the full
SWMM equation sets.

## Scope
`swe2d_drainage_link_kernel` — the pipe/link flow solver in `swe2d_gpu.cu`.
All changes are within the kernel; no changes to the kernel signature or the
calling interface.

---

## Mode 0 — SWMM Steady Flow Routing (NO_ROUTING / `steadyflow_execute`)

### SWMM Reference
`flowrout.c:steadyflow_execute` (lines 748–802):

```c
// 1. Subtract losses from inflow
q -= link_getLossRate(j, SF, q, tStep);

// 2. Cap at full Manning's capacity
if ( q > Link[j].qFull ) {
    q = Link[j].qFull;
    a1 = Link[j].xsect.aFull;
} else {
    // 3. Infer area from Manning's section factor
    s = q / Conduit[k].beta;
    a1 = xsect_getAofS(&Link[j].xsect, s);
}
```

SWMM has **no Bernoulli energy solver** — it's a Manning-capacity cap with
area inference. The `beta` factor is `beta = (1/n) · sqrt(slope)` and
`Link[j].qFull = xsect.sFull * beta`.

### Required Changes

Replace the entire mode 0 block (`swe2d_drainage_link_kernel` lines 4220–4224)
with:

```c
} else if (solver_mode == 0) {
    // Mode 0: SWMM Steady Flow routing.
    // Q = min(Q_in, Q_Manning_full)
    // where Q_Manning_full = xsect.sFull * beta
    // and beta = (1/n) * sqrt(slope) with slope from link geometry.
    // Area is inferred from Manning's section factor if not at full flow.

    const double q_cap = link_max_flow[i];
    const double s_link = fabs(dh_raw) / L;  // bed slope from head diff (approx.)
    const double q_full = xsect.sFull *
        (c_k_mann / n_mann) * pow(s_link, 0.5);  // full Manning's capacity

    if (q_full <= 0.0) {
        q = 0.0;
    } else {
        // Use the smaller of upstream inflow (from link flow prev) or capacity.
        // For the first node in a link chain, use dh to determine direction.
        // q_inbound = sign(dh) * min(capacity, |q_leave|)
        // For a standalone pipe with no incoming link: use dh directly.
        const double q_inbound = (dh_raw >= 0.0) ? fabs(q) : -fabs(q);
        const double q_mann = (dh_raw >= 0.0) ?
            fmin(q_full, fabs(q_inbound)) :
            -fmin(q_full, fabs(q_inbound));

        if (q_cap > 0.0) {
            q = fmax(-q_cap, fmin(q_cap, q_mann));
        } else {
            q = q_mann;
        }
    }
    // area computed from section factor if needed by caller
    // (for surface area exchange — uses midpoint depth, not this Q)
```

**Simplifications accepted:**
- SWMM uses the inlet flow `q_in` as the upstream discharge. The kernel
  architecture passes `link_flow_prev` as the previous-step outflow. Using
  `q_leave` from the link's to-node as the "inflow" to the next link is
  consistent with SWMM's steady-flow model.
- Bed slope approximated as `|dh| / L` (hydraulic grade line slope) since
  true geometric slope is not stored. This is consistent with how Mode 1
  currently uses `dh` and is reasonable for the steady-flow assumption.
- SWMM's `link_getLossRate` (evap/seepage) is not modeled — out of scope.

---

## Mode 1 — Diffusion Wave (g·∂h/∂x + g(S_f − S) = 0)

### SWMM Reference
Not a separate SWMM routing mode. The diffusion wave is the **limit case** of
SWMM's kinematic wave (`kinwave.c`) when the inertial terms are negligible, and
is also the form implied by SWMM's dynamic wave (`dwflow.c`) when `sigma = 0`
(full inertial damping) and `dq3 = dq4 = 0`.

The user's specified form:
```
g·∂h/∂x + g(S_f − S) = 0
```

Rearranged:
```
S_f = S   (friction slope equals bed slope)
```

This means: the pressure gradient term `g·∂h/∂x` balances the bed slope `g·S`,
so the net driving force is zero and flow proceeds at the normal depth
Manning's velocity. The flow is **pure Manning's at normal depth** driven by
the prescribed bed slope, not by the actual head gradient.

The simplification is: **discard the head-difference driving term entirely** and
use Manning's normal flow equation scaled by the bed slope. The link flow is:

```
Q = β · A · R^(2/3) · sqrt(S)   (Manning's normal flow)
  = (k/n) · A · R^(2/3) · sqrt(|S|) · sign(S)
```

Where `S = slope` from the link geometry (stored or approximated as `|dh|/L`).

### Required Changes

Replace the mode 1 block (`swe2d_drainage_link_kernel` lines 4225–4227) with:

```c
} else if (solver_mode == 1) {
    // Mode 1: Diffusion wave — g*dh/dx + g*(Sf-S) = 0
    // Simplification: Sf = S (friction slope = bed slope), so flow is
    // Manning's normal flow. No head-driven term.
    // Q = (k/n) * A * R^(2/3) * sqrt(|S|) * sign(dh_raw)
    // where S = |dh_raw| / L (bed slope approximation from HGL).
    const double s_slope = fabs(dh_raw) / fmax(L, 1.0e-9);
    const double beta_mann = c_k_mann / n_mann;
    // Use the midpoint area (computed above) for the normal flow.
    q = beta_mann * area * pow(r_h, 2.0/3.0) * sqrt(s_slope);
    if (dh_raw < 0.0) q = -q;  // reverse flow direction

    const double q_cap = link_max_flow[i];
    if (q_cap > 0.0) {
        q = fmax(-q_cap, fmin(q_cap, q));
    }
}
```

**Key changes from current code:**
- **Removed** the `sqrt(s_w)` with `s_w = |dh|/L` — this was using the head
  gradient as the driving slope. The diffusion wave simplification discards
  the ∂h/∂x term entirely, leaving only Manning's normal flow.
- Sign is from `dh_raw` (head difference direction), not from the current
  sign convention.
- `area` and `r_h` are still from the midpoint depth computation — these
  are used correctly for Manning's normal flow.

**What this achieves:**
- Consistent with the user's specified equation `g·∂h/∂x + g(S_f − S) = 0`.
- No artificial mass creation from incorrect coupling of head gradient.
- Physically: flow depends only on geometry and Manning's n, not on the
  transient head field.

---

## Mode 2 — Full Dynamic Wave (SWMM DW / `dwflow_findConduitFlow`)

### SWMM Reference
`dwflow.c:dwflow_findConduitFlow` (lines 57–293). The full finite-difference
Saint-Venant equations solved by Picard iteration.

**Momentum equation** (lines 210–240):
```
dq1 = dt * S_f                      // friction slope term
     = dt * (n² * |v| * v) / R^(4/3)          (Manning's)
   = dt * (n² * |q| * q) / (A² * R^(4/3))

dq2 = dt * g * A * (h2 - h1) / L              // energy slope term
   = dt * g * A * dh / L                       (dh = h2-h1)

dq3 = 2 * v * (A_mid - A_old) * sigma         // local acceleration
dq4 = dt * v * v * (A2 - A1) / L * sigma       // convective acceleration
dq5 = local losses term (from findLocalLosses)
dq6 = evaporation/seepage losses

denom = 1.0 + dq1 + dq5
q_new = (q_old - dq2 + dq3 + dq4 + dq6) / denom
```

Key features:
- **dq3**: local acceleration `∂Q/∂t` with inertial damping factor `sigma`
  (sigma = 0 at Froude ≥ 1, sigma = 1 at Froude ≤ 0.5)
- **dq4**: convective acceleration `∂(Q²/A)/∂x`
- **dq1**: true Manning's friction with `|v|*v` in numerator
- **Inertial damping**: `sigma = 1.0` for low Froude, `sigma = 0.0` for high Froude,
  interpolated between
- **dqdh**: sensitivity of flow to head change for node coupling

### Required Changes

The mode 2 block (`swe2d_drainage_link_kernel` lines 4228–4239) needs a
complete replacement. The kernel needs access to:
- `a_old` (area from previous time step) — currently `link_flow_prev` is `Q`,
  not `A`. Need to either compute `a_old` from `q_old` via Manning's inverse,
  or store `a_old` separately.
- `q_old` from previous iteration (not previous time step — needs per-iteration
  storage within the Picard loop)

**Architecture change required**: The kernel currently has no per-link state
between iterations. The Picard loop in `dynwave_execute` (SWMM) calls
`dwflow_findConduitFlow` repeatedly, and `Conduit[k].q1` holds the flow from
the previous iteration while `Conduit[k].a1` / `Conduit[k].a2` hold area state.

For the GPU kernel, we need persistent per-link state:
```cpp
// New device buffers needed in SWE2DDeviceState or coupling_ws:
double* d_link_q_old;       // Q from previous Picard iteration
double* d_link_area_mid;    // A_mid from previous Picard iteration
double* d_link_area_old;    // A from previous time step
```

**Simplified approach for the kernel**: Since the Picard iteration is already
done on the host side in the iterative drainage wrapper
(`swe2d_gpu_drainage_step_iterative` in `swe2d_bindings.cpp`), we can
implement the dynamic wave **on the host** (in Python or C++ wrapper) using
the same iterative approach as SWMM, and only call the GPU kernel for the
final Q computation. This avoids duplicating the Picard iteration on-GPU.

**Plan**: Refactor mode 2 to call a new host-side function
`swe2d_drainage_dynwave_step` that mirrors `dwflow_findConduitFlow`, using
the existing iterative loop structure already in `swe2d_bindings.cpp` lines
2132–2236.

### New kernel launch for mode 2 — compute Q and A_mid from current state:

```c
} else if (solver_mode == 2) {
    // Mode 2: Full Dynamic Wave (SWMM DW)
    //
    // Solve finite-difference Saint-Venant:
    //
    // denom = 1.0 + dq1 + dq5
    // q_new = (q_old - dq2 + dq3 + dq4 + dq6) / denom
    //
    // where:
    //   dq1 = dt * n² * |v| * v / R^(4/3)   [friction slope term]
    //   dq2 = dt * g * A_mid * dh / L        [energy slope term]
    //   dq3 = 2*v*(A_mid - A_old)*sigma      [local acceleration]
    //   dq4 = dt * v*v * (A2-A1) / L * sigma [convective acceleration]
    //   dq5 = dt * local_losses              [minor losses]
    //   dq6 = dt * evap/seep losses
    //   sigma = inertial damping (0..1 based on Froude)
    //
    // For this kernel call: compute dq1..dq6 from current state and
    // produce the updated q_out. The Picard iteration / node coupling is
    // handled by the calling wrapper.

    // Get previous-step area (stored in persistent buffer or re-computed)
    const double A_old = (link_area_prev != nullptr) ?
        link_area_prev[i] : area;  // fallback to midpoint if no prev

    const double v_mid = (area > 0.0) ? q_old / area : 0.0;
    const double v_old = (A_old > 0.0) ? q_old / A_old : 0.0;

    // Froude number for sigma
    const double v_abs = fabs(v_mid);
    const double froude = (area > 0.0 && r_h > 0.0) ?
        v_abs / sqrt(fmax(GRAVITY * r_h, 1e-12)) : 0.0;

    double sigma = 1.0;
    if      (froude >= 1.0) sigma = 0.0;
    else if (froude >  0.5) sigma = 2.0 * (1.0 - froude);

    // dq1: friction slope (Manning's, true form with |v|*v)
    const double dq1 = dt_s * n_mann * n_mann * v_abs * v_mid /
        pow(fmax(r_h, 1e-9), 4.0/3.0);

    // dq2: energy slope term
    const double dq2 = dt_s * GRAVITY * area * dh_raw / fmax(L, 1e-9);

    // dq3: local acceleration
    const double dq3 = (sigma > 0.0) ?
        2.0 * v_mid * (area - A_old) * sigma : 0.0;

    // dq4: convective acceleration
    const double dq4 = (sigma > 0.0 && L > 0.0) ?
        dt_s * v_mid * v_mid * (area - area) * sigma / L : 0.0;
    // Note: A2-A1 is not easily available without full cross-section data.
    // SWMM uses upstream/downstream area difference here.
    // For a circular pipe at midpoint: approximate dq4 ≈ 0 (A2 ≈ A1 ≈ A_mid).

    // dq5: minor losses (if hasLosses flag is available)
    // For now: dq5 = 0 (add later when cLossInlet/cLossOutlet are passed)

    // dq6: evap/seep = 0 for this implementation

    const double denom = 1.0 + dq1;  // + dq5 when added
    q = (q_old - dq2 + dq3 + dq4) / denom;

    // Sign convention: dh_raw already encodes direction
    if (dh_raw < 0.0 && q > 0.0) q = -q;
    if (dh_raw >= 0.0 && q < 0.0) q = -q;

    // Flow capping
    const double q_cap = link_max_flow[i];
    if (q_cap > 0.0) {
        q = fmax(-q_cap, fmin(q_cap, q));
    }

    // Under-relaxation toward candidate (same as SWMM lines 263-267)
    // Note: this requires qLast from previous iteration, not available here.
    // The relaxation should be applied in the calling Picard loop.
}
```

### Critical missing pieces for full DW parity

1. **dq4 (convective acceleration)**: Needs `A2` (downstream area) and `A1`
   (upstream area). Currently the kernel only computes `area` (midpoint). Need
   to compute `a1` and `a2` from `depth0` and `depth1` at the call site, or
   store them in the kernel.

2. **A_old from previous time step**: The kernel receives `link_flow_prev`
   but needs `A_old`. Can be computed from `link_flow_prev` by inverting
   Manning's: `A_old = (Q_old / beta)^(3/5)` or stored in a new buffer.

3. **Minor losses (dq5)**: Requires `cLossInlet` and `cLossOutlet`
   coefficients to be passed to the kernel. Not currently in the kernel
   signature.

4. **dqdh for node coupling**: SWMM computes `dqdh = g·dt·A_weighted·barrels/L`
   to couple into the node continuity equation system. The current code has
   no such coupling.

5. **Flow classification**: SWMM classifies flow as DRY, UP_DRY, DN_DRY,
   SUBCRITICAL, SUPCRITICAL and adjusts area contributions accordingly.
   Not currently implemented.

**Recommended priority for initial implementation**:
- Phase 2a: Fix dq1 (true Manning's friction) + dq2 (energy slope) — these
  are the dominant terms for most drainage conditions.
- Phase 2b: Add dq3 (local acceleration) + A_old from previous time step.
- Phase 2c: Add dq4 (convective acceleration) + upstream/downstream areas.
- Phase 2d: Add minor losses + dqdh coupling.

---

## Testing

### `tests/test_swe2d_drainage_structures.py`
Add three new test cases, one per mode, mirroring the SWMM verification cases:

1. **Mode 0 (Steady Flow)**: Simple pipe with constant head difference.
   Expected: Q = min(Q_in, Q_Manning_full). No accumulation.
   Geometry: D = 0.3m, L = 50m, n = 0.013, slope = 0.01, dh = 0.5m.
   Expected Q ≈ 0.47 m³/s (Manning's full-pipe capacity).

2. **Mode 1 (Diffusion Wave)**: Pipe with varying head difference.
   Expected: Q = Manning's normal flow independent of dh (friction = bed slope).
   Verify Q is constant regardless of head difference magnitude.

3. **Mode 2 (Dynamic Wave)**: Surcharge scenario where downstream node
   is higher than upstream, creating backwater.
   Expected: dq2 (energy slope) resists reverse flow; Q reduced compared
   to Mode 1 when downstream is elevated.

### Mass conservation test
Add a closed-loop test: two cells with a single drainage pipe connecting them.
Apply constant inflow at one node and verify total volume is conserved
(sinks to outfall = inflow volume minus storage change).

---

## Files to Change

| File | Change |
|------|--------|
| `cpp/src/swe2d_gpu.cu` | Replace mode 0, 1, 2 blocks in `swe2d_drainage_link_kernel`. Add GRAVITY constant if not already defined. |
| `cpp/src/swe2d_gpu.cuh` | Add `d_link_area_prev`, `d_link_q_iter` buffers to `SWE2DDeviceState` or `DrainageWorkspace`. |
| `cpp/src/swe2d_bindings.cpp` | If mode 2 moves to host-side Picard loop, add `swe2d_drainage_dynwave_step` wrapper function. |
| `swe2d/runtime/coupling.py` | Pass `slope` (or compute from `dh/L`) to kernel; pass `cLossInlet/cLossOutlet` if available. |
| `tests/test_swe2d_drainage_structures.py` | Add three mode-specific tests. |
| `docs/DRAINAGE_EQUATION_PLAN.md` | This document. |

---

## Effort Estimate

- Mode 0 (Steady Flow): 1–2 hours — delete and replace with Manning cap.
- Mode 1 (Diffusion Wave): 30 min — delete the head-gradient term.
- Mode 2 (Dynamic Wave): 4–6 hours — Phase 2a first, full parity Phase 2b–2d later.
- Testing: 2–3 hours.
- **Total initial implementation: 8–12 hours (Mode 2 full parity = separate follow-on task).**
