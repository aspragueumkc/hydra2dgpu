# SWE2D/3D GPU Solver Design v0

Date: 2026-05-15
Status: Initial implementation scaffold

## Scope

This document defines the first implementation slice for:

1. Nonhydrostatic 2D SWE mode (GPU-first).
2. 3D single-phase free-surface VoF mode (GPU-first).
3. Partitioned 2D/3D coupling interface (GPU-first).

CPU paths are retained only for prototyping/debug support where practical.
Production use targets CUDA execution.

## Decisions Locked In

1. Equation-set selector is explicit and native:
   - 0: hydrostatic 2D (existing path)
   - 1: nonhydrostatic 2D (new path scaffolded)
2. Coupling selector is explicit and native:
   - 0: off
   - 1: one-way 2D to 3D
   - 2: two-way 2D and 3D
3. 3D solver model selector is explicit and native:
   - 0: disabled
   - 1: single-phase free-surface VoF
4. Advanced modes (nonhydrostatic and/or coupled) enforce GPU-only when enabled.

## First-Step Implementation Delivered

1. Python config/plumbing:
   - New enums in swe2d_extensions.py for equation set, coupling mode, and 3D model.
   - SolverModelOptions now carries advanced-mode fields and GPU-only policy.
   - swe2d_backend.py validates advanced-mode GPU requirements before solver creation.
2. Native config/plumbing:
   - SWE2DSolverConfig now includes equation_set, coupling_mode, three_d_solver_model,
     enforce_gpu_only_advanced_modes, and three_d_single_phase_free_surface.
   - Pybind solver creation accepts and forwards these fields.
3. Native runtime guardrails:
   - Solver creation rejects advanced modes when GPU-only policy is on and CUDA is unavailable.
   - Step dispatch raises explicit scaffold-not-implemented error for advanced modes
     until NH/3D kernels are introduced.

## Next Coding Steps

1. Nonhydrostatic 2D (GPU-first)
   - Add predictor/corrector CUDA entry points.
   - Introduce nonhydrostatic pressure workspace buffers on device.
   - Implement matrix-free PCG pressure increment solve and residual diagnostics.
2. 3D single-phase free-surface VoF (GPU-first)
   - Add Cartesian patch data model and device allocation API.
   - Add VoF advection kernel set (compressive/bounded formulation).
   - Add projection pressure solve and free-surface boundary treatment.
3. Coupling (GPU-first)
   - Define conservative interface buffers (mass flux, momentum, head loss terms).
   - Implement one-way 2D to 3D forcing first.
   - Implement 3D to 2D feedback and relaxed sub-iterations second.

## Validation Gates for This Scaffold

1. Hydrostatic mode remains default and behavior-preserving.
2. Advanced modes fail fast with clear messages if selected before kernels are implemented.
3. Advanced modes fail fast when GPU policy is violated.

## Notes

- This v0 intentionally avoids placeholder CPU solvers for NH/3D production paths.
- CPU prototypes can be added for algorithm exploration only, then promoted to CUDA.
