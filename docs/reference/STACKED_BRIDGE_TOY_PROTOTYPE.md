# Stacked-Region Bridge Toy Prototype

## Purpose
This toy model is a fast Python prototype to evaluate whether a pressure-correction solver plus specific face treatment can represent:

- pressurized flow below a deck block,
- free-surface flow above the deck,
- local acceleration losses at the upstream and downstream deck interfaces.

It is intentionally simple and aimed at calibration/stability experiments before integrating with the production SWE2D path.

## Geometry
The script uses a fixed Cartesian toy mesh inspired by the attached diagram:

- cell size: 2 ft x 2 ft,
- gravity: 32.17 ft/s^2,
- flow direction: +x,
- bed row: solid,
- deck block: interior solid rectangle,
- waterline: fixed row creating free surface in upper open cells.

File: `stacked_bridge_toy.py`

## Inflow forcing
Inlet velocity now ramps in time:

- `inlet_velocity_start_ft_s` (default `1.0`)
- `inlet_velocity_end_ft_s` (default `4.0`)
- `inlet_ramp_duration_s` (default `4.0`)

Outflow can be moderated with `outlet_velocity_factor` to produce visible transient storage/free-surface response in this toy setup.

## Extruded bridge setup (z direction)
The prototype now supports an extruded configuration in z:

- total width: `z_units = 24`
- two piers, each `pier_width_units = 3`
- pier bands are placed to create three equal openings

With defaults this produces:

- openings: 6 + 6 + 6 units
- piers: 3 + 3 units
- pier bands: `[6:9]` and `[15:18]`

Piers run the full bridge length in x and occupy the under-deck region.

## Numerics (prototype level)
The model uses a projection-style step:

1. Apply inlet velocity and outflow copy boundary.
2. Enforce wall/no-penetration on faces touching solids.
3. Apply local loss damping on under-deck entry/exit faces.
4. Solve a cell-centered pressure Poisson equation on fluid cells.
5. Correct face velocities with pressure gradients.
6. Update per-column water volume from face fluxes and rebuild a moving wet/air mask.

The moving-surface reconstruction now uses fractional fill in the top wet cell of each column, and the rendered free surface is a continuous polyline derived from that fraction.

Free-surface cells are handled as atmospheric pressure (Dirichlet `p = 0`).
Cells under the deck do not receive this atmospheric condition, enabling a pressurized response.

## Physics model (what is being approximated)
The prototype is best interpreted as an incompressible, depth-integrated-in-z toy flow model in the x-y profile plane, with pressure correction and explicit geometry masking.

Core physical assumptions:

1. Water is incompressible with constant density.
2. Gravity acts in the y direction.
3. Air pressure at the free surface is atmospheric (gauge pressure zero).
4. Deck and pier faces are impermeable (no-penetration wall behavior).
5. Local bridge losses are represented through empirical coefficients at under-deck entry and exit faces.

Conservation statements in toy form:

- Mass conservation is enforced approximately through:
  - face-flux divergence control via pressure projection,
  - column-volume updates that move the free surface in time.
- Momentum is evolved in a simplified way:
  - pressure-gradient correction step,
  - local empirical damping at deck interfaces.

This setup captures first-order behavior relevant to bridge hydraulics in a screening model:

- under-deck pressurization tendency,
- backwater/free-surface rise,
- discharge sensitivity to local loss coefficients.

## Numerical formulation details
The implementation follows a staggered-face velocity / cell-centered pressure pattern.

Per time step, conceptually:

1. Boundary forcing:
	- Inlet velocity ramp from `inlet_velocity_start_ft_s` to `inlet_velocity_end_ft_s`.
	- Outflow velocity scaled by `outlet_velocity_factor`.
2. Geometry constraints:
	- Zero normal flux across blocked faces (bed, deck, piers, dry interfaces).
3. Local loss operator:
	- Under-deck entry/exit face velocities are damped with coefficients
	  `loss_k_upstream`, `loss_k_downstream`.
4. Pressure Poisson solve:
	- Solve a sparse cell-centered equation with SOR iterations on wet cells.
	- Free-surface cells impose `p = 0`.
5. Velocity projection:
	- Correct face velocities with discrete pressure gradients.
6. Moving free-surface reconstruction:
	- Integrate net x-flux per column to update column water volume.
	- Rebuild wet/air masks with fractional top-cell filling.
	- Extract a continuous free-surface polyline for diagnostics and plotting.

Numerical notes:

- The method is low-order and intentionally robust/simple for rapid iteration.
- The wet/dry transition is mask-based and not based on a full Riemann SWE treatment.
- Stability is managed by moderate time step, projection, and bounded volume updates.

## Comparison with hydrostatic SWE and 3D Navier-Stokes VOF
This section positions the current prototype against two common modeling levels.

### 1) Hydrostatic SWE (2D depth-averaged)
Hydrostatic SWE assumes vertical acceleration is negligible and pressure is hydrostatic through depth.

Typical strengths:

- Very efficient for floodplain-scale routing.
- Mature shock-capturing finite-volume methods.
- Good for broad backwater and inundation footprints.

Typical limits near bridges:

- Pressurized under-deck flow is not naturally represented unless augmented with internal boundary/source logic.
- Vertical structure of flow (separation, contraction profile) is collapsed into depth-averaged closure.

Relative to this prototype:

- The toy projection model is still simplified, but it can represent under-deck pressurization behavior more explicitly than plain hydrostatic SWE because pressure is solved with non-hydrostatic response in the profile plane and constrained by deck masks.
- Hydrostatic SWE is still more rigorous for large-scale conservative routing when implemented with full SWE fluxes and source balancing.

### 2) 3D Navier-Stokes + VOF
3D NS-VOF resolves full 3D momentum with a tracked/captured free surface volume fraction.

Typical strengths:

- Highest fidelity for local bridge hydraulics.
- Can resolve complex 3D flow features:
  - contraction/expansion jets,
  - recirculation and separation,
  - vertical structure and detailed pressure loads.

Typical limits:

- High computational cost and setup complexity.
- Calibration/mesh/turbulence model sensitivity can be substantial.

Relative to this prototype:

- The toy model is orders of magnitude cheaper and faster for design-space exploration.
- It is not a replacement for NS-VOF where detailed local loads or strongly 3D/turbulent behavior must be resolved.
- It is best used as an intermediate screening/calibration stage before high-fidelity CFD.

### Practical hierarchy for bridge studies
A practical staged workflow is:

1. Hydrostatic SWE for catchment/floodplain-wide context.
2. This stacked-region toy non-hydrostatic prototype for structure-focused screening and coefficient sensitivity.
3. 3D NS-VOF for final local-detail confirmation in critical scenarios.

## Loss calibration hooks
Two explicit coefficients are exposed:

- `loss_k_upstream`
- `loss_k_downstream`

They damp face-normal velocity at the under-deck entry/exit interfaces and are the primary calibration knobs.

## Suggested toy validation checks
Use this prototype to check:

1. Trend correctness: increasing `loss_k_*` should reduce under-deck discharge.
2. Pressure behavior: persistent non-trivial pressure split between under-deck and over-deck regions.
3. Stability behavior: bounded divergence residual with moving free-surface updates.
4. Forcing behavior: inlet velocity ramp is achieved and free-surface elevation changes over time.

## Empirical calibration and stability sweep notes
The following sweep was run on the toy prototype with symmetric bridge loss coefficients
(`loss_k_upstream = loss_k_downstream = k`) and the default inlet ramp from 1 ft/s to 4 ft/s.

Observed trend from the sweep:

- as `k` increased from `0.0` to `2.0`, under-deck discharge decreased smoothly,
- mean under-deck pressure stayed positive and changed moderately,
- the moving free surface remained active across the tested range,
- the divergence residual stayed bounded rather than blowing up.

Practical takeaway for the alpha-phase compiled solver:

- recommended first calibration window: `0.5 <= k <= 1.5`,
- strong starting guess: `k ≈ 1.0`,
- if you need a narrower initial fit window: `0.75 <= k <= 1.25`.

This is not a final physical calibration range, only a reasonable starting envelope for early solver integration.

The timestep stability sweep was also insensitive over the tested interval:

- `dt = 0.025` to `0.125` s produced nearly identical tail metrics,
- the toy divergence residual remained around `0.50` in all cases,
- the surface motion and mean Froude response stayed qualitatively consistent.

So for alpha prototyping, the model appears numerically comfortable within that timestep band, subject to the same caveat that this is a reduced toy model rather than the final compiled 2DSWE solver.

## Tests
Pytest coverage:

- `tests/test_stacked_bridge_toy.py::test_local_loss_coefficients_reduce_underdeck_discharge`
- `tests/test_stacked_bridge_toy.py::test_underdeck_pressure_exceeds_overdeck_pressure_when_constricted`
- `tests/test_stacked_bridge_toy.py::test_projection_reduces_divergence_residual`
- `tests/test_stacked_bridge_toy.py::test_inlet_ramp_and_surface_motion_present`

## Run
Quick run:

```bash
python stacked_bridge_toy.py
```

This run now also writes a video (default path):

- `docs/stacked_bridge_toy_extruded_profile_cross.mp4` (or a `.gif` fallback if MP4 encoding is unavailable)

Per frame, cells are shaded by pressure and the overlay text shows:

- mean velocity,
- mean Froude,
- mean pressure,
- step/time.

Additionally, frames include:

- a cyan moving free-surface line,
- white velocity vectors (quiver) sampled by `velocity_quiver_stride`.

The extruded video is two-panel:

1. Left panel: profile view for the unit-width opening slice.
2. Right panel: y-z cross section at x centered on the bridge.

Tests:

```bash
pytest tests/test_stacked_bridge_toy.py -q
```

## Limitations
This is a toy prototype, not a production solver:

- moving free surface is reconstructed from 1D column volume updates, not a full multiphase interface transport equation,
- no turbulence closure,
- no high-order advection,
- no full 2D SWE flux model.

It is only for early feasibility, coefficient sensitivity, and stability screening.