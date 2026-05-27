# Stacked Bridge Coupling Implementation Plan

## Objective
Prototype bridge coupling in the cheapest environment first, then migrate the validated interface laws into the compiled solver and finally into SWE2D coupling.

## Phase 1: Toy coupling baseline
Status: starting now.

Scope:

- keep the toy bridge-loss law as a first-class reusable object,
- sweep symmetric and asymmetric loss coefficients,
- verify monotonic discharge response and bounded stability,
- use the toy model as the behavioral reference for the later compiled implementation.

Acceptance criteria:

- loss coefficients can be parameterized independently upstream/downstream,
- the loss law is reusable outside the toy solver,
- calibration sweeps produce a narrow recommended initial fit range.

## Phase 2: Initial CUDA solver prototype

Status: started; workbench runtime wiring in progress.

Scope:

- add the bridge-loss coupling seam to the initial CUDA solver path,
- keep the bridge loss logic localized so it can be toggled on/off,
- verify that the CUDA solver reproduces the toy model's qualitative trends.
- auto-enable the bridge helper from hydraulic-structure bridge metadata when CUDA is available.

Acceptance criteria:

- bridge coupling enters through a small and testable interface,
- the solver remains stable for the toy calibration envelope,
- the implementation does not require redesigning the main SWE2D kernels.

## Phase 3: SWE2D coupling integration

Scope:

- couple the validated bridge law into the full SWE2D solver,
- map the same upstream/downstream loss logic to the 2D mesh representation,
- reuse the toy and CUDA calibrations as initial values.

Acceptance criteria:

- SWE2D integration matches the toy/CUDA qualitative bridge response,
- calibration parameters transfer without major retuning,
- bridge-specific coupling remains isolated from the core hydrodynamic kernels.

## Recommended initial parameter envelope

From the current toy sweeps:

- start at `k = 1.0`,
- first window: `0.5 <= k <= 1.5`,
- narrower search window: `0.75 <= k <= 1.25`.

## Next implementation slice

1. Extend the bridge helper to accept geometry-specific opening width and loss metadata from SWE2D coupling.
2. Map the same bridge-coupling flag into the SWE2D mesh/structure ingestion path once the 2D bridge geometry is ready.
3. Add a small override control only if manual bridge-helper toggling becomes necessary in practice.