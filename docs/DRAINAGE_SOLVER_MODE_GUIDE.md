# Drainage Solver Mode Guide

The 1D drainage network solver supports three equation sets. Each has
different accuracy, stability, and performance characteristics.

## Mode Selection

Select the solver mode in the **Parameters tab → Drainage Network** section
under **Drainage solver mode**.

| Mode | Best For |
|------|----------|
| **EGL** (default) | Storm drain systems, pressurized pipe flow |
| **DIFFUSION** | Partially-full gravity sewers, open-channel reaches |
| **DYNAMIC** | Surge, bore propagation, backwater transients |

## EGL — Energy Grade Line

The default mode. Uses the Bernoulli equation with Manning friction and
minor losses. Analogous to FHWA HEC-22 outlet-control equations.

**When to use:**
- Storm drain systems with full or nearly-full pipes
- Pressurized flow conditions
- Design-mode simulations where HEC-22 consistency matters

**HEC-22 boundary losses:** Entrance loss (`k_in`) at the first sub-cell and exit loss
(`k_out`) at the last sub-cell are applied via `cell_k_loss` in the flux accumulation kernel.
This matches HEC-22 practice: energy is lost as flow enters and exits the pipe.

**Stability:** Very stable. No CFL constraint on the 1D solve.

**Performance:** Fastest. Single-pass solve per coupling step.

## DIFFUSION — Diffusion Wave

Slope-driven Manning flow using partial-flow circular-section hydraulic
geometry.

**When to use:**
- Partially-full gravity sewers
- Open-channel reaches with free-surface flow
- Systems where pressure-flow assumptions break down

**Stability:** Stable for most conditions. May oscillate with very steep
slopes or rapidly-varying flows.

**Performance:** Comparable to EGL.

## DYNAMIC — Full Saint-Venant

Full 1D Saint-Venant equations with semi-implicit per-link momentum update.

**When to use:**
- Surge and bore propagation
- Backwater transients
- Systems with rapidly changing flow regimes
- When EGL or DIFFUSION produce unrealistic results

**Stability:** Subject to CFL constraint. The adaptive substepping controller
automatically adjusts the 1D timestep to maintain stability. If you see
many substeps in the log, the solver is working hard to stay stable.

**Performance:** Slowest due to substepping. May require 10–100× more 1D
timesteps than EGL for the same simulation.

## Adaptive Substepping

All modes support adaptive substepping via these parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| **Coupling substeps** | 1 | Number of 1D sub-steps per 2D coupling call |
| **Max coupling substeps** | 64 | Upper limit for adaptive controller |
| **Adaptive depth fraction** | 0.2 | Allowable fractional node-depth change per substep |
| **Adaptive wave Courant** | 0.5 | Courant target for dynamic-wave links |

For DYNAMIC mode, the adaptive controller tightens the 1D timestep when
large depth changes are detected. The `max_coupling_substeps` parameter
acts as a safety valve — if the controller requests more substeps than
this limit, the simulation logs a warning.

## Advanced Coupling Parameters

The drainage module supports advanced coupling control through these parameters
(available in Parameters → Drainage Network):

| Parameter | Default | Range | Purpose |
|-----------|---------|-------|---------|
| **Coupling substeps** | 1 | 1–256 | Number of 1D sub-steps per 2D coupling call. Use 2–4 for stiff networks with oscillatory 2D-1D interface behavior. |
| **Max adaptive substeps** | 64 | 1–1024 | Upper limit for adaptive controller. Increase if simulation logs "max substeps exceeded" warnings. |
| **Implicit coupling iterations** | 2 | 1–8 | Predictor-corrector iterations for implicit coupling. Higher values improve convergence but increase runtime. |
| **Implicit coupling relaxation** | 0.5 | 0.1–1.0 | Under-relaxation factor for implicit coupling. Reduce to 0.3–0.4 for very stiff networks to improve stability. |
| **Recon method** | 1 | 0, 1, 2, 3, 6 | Spatial reconstruction scheme: 0=First-order, 1=MUSCL-MinMod (default), 2=MUSCL-MC, 3=MUSCL-Van Leer, 6=WENO3. Higher-order reduces numerical diffusion but may cause oscillations. |

### Tuning Guidelines

**When to increase coupling_substeps:**
- Oscillatory behavior at inlet/outfall cells
- Large depth differences between adjacent cells
- Rapidly changing boundary conditions

**When to increase implicit iterations:**
- Poor convergence in DYNAMIC mode
- Residual warnings in solver logs
- Mass balance errors increase over time

**When to change recon method:**
- First-order (0): Most stable, most numerical diffusion. Use for debugging.
- MUSCL-MinMod (1): Default balance of accuracy and stability.
- MUSCL-MC (2): More accurate for smooth flows.
- WENO3 (6): Best accuracy for smooth transients, but expensive and may oscillate.

### Stability Checklist

If you see instability:

1. **Increase coupling_substeps** to 2–4
2. **Reduce implicit coupling relaxation** to 0.3–0.4
3. **Check CFL condition**: `dt * sqrt(g * max_depth) / min_cell_length < 0.5`
4. **Switch recon method** to First-order (0) temporarily
5. **Verify loss coefficients** aren't excessive (>5.0)
6. **Check for dry cells**: Very shallow cells can cause numerical issues

## Performance Optimization

### Mode Selection Impact

| Mode | Relative Speed | Memory | Use Case |
|------|----------------|--------|----------|
| **EGL** | 1× (baseline) | Low | Storm drains, pressurized flow |
| **DIFFUSION** | ~1.2× | Low | Gravity sewers, open channels |
| **DYNAMIC** | 10–100× (network-dependent) | Medium | Surge, transients, backwater |

### GPU Memory Usage

Per-cell memory (~336 bytes with current implementation):
- 10,000 cells → ~3.4 MB
- 100,000 cells → ~34 MB
- 1,000,000 cells → ~340 MB

### Runtime Estimation

Approximate runtime (modern RTX 3060):
```
runtime ≈ (n_cells * coupling_substeps * n_timesteps) / 1e6  [seconds]
```

Example: 50,000 cells, 2 substeps, 1000 timesteps → ~100 seconds

---

## Tuning Tips

1. **Start with EGL** — it's the fastest and most stable. Only switch to
   DYNAMIC if EGL produces unrealistic results.

2. **Increase coupling substeps** if you see oscillatory behavior at the
   2D-1D interface. Values of 2–4 are usually sufficient.

3. **Reduce adaptive_depth_fraction** (e.g. 0.1) if node depths oscillate.
   This forces smaller 1D timesteps but improves stability.

4. **Check the log** for substep counts. If `substeps_used` consistently
   hits `max_coupling_substeps`, your network may be too stiff for the
   current mode — consider switching to EGL or reducing the 2D timestep.

5. **Dynamic flow relaxation** (default 1.0) can be reduced (e.g. 0.7) to
   damp oscillatory link flow updates in DYNAMIC mode.

6. **For complex networks**: Start with EGL + First-order recon, then gradually
   increase coupling_substeps and switch to MUSCL-MinMod once stable.

---

## Related Documentation

- **[Documentation Index](INDEX.md)** — All guides by audience
- **[User Guide](USER_GUIDE.md)** — End-to-end simulation workflow
- **[GPU Architecture Report](SWE2D_GPU_ARCHITECTURE_REPORT.md)** — Coupling section
- **[Developer Guide](DEVELOPER_GUIDE.md)** — `SWE2DUrbanDrainageModule`, `DrainageSolverMode` enum
- **[Repository Knowledge Graph](../graphify-out/wiki/index.md)** — Drainage & Pipes community
