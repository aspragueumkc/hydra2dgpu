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

---

## Related Documentation

- **[Documentation Index](INDEX.md)** — All guides by audience
- **[User Guide](USER_GUIDE.md)** — End-to-end simulation workflow
- **[GPU Architecture Report](SWE2D_GPU_ARCHITECTURE_REPORT.md)** — Coupling section
- **[Developer Guide](DEVELOPER_GUIDE.md)** — `SWE2DUrbanDrainageModule`, `DrainageSolverMode` enum
- **[Repository Knowledge Graph](../graphify-out/wiki/index.md)** — Drainage & Pipes community
