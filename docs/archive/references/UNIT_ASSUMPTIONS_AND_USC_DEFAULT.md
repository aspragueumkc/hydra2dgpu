---
type: reference
status: complete
created: 2026-07-15
completed: 2026-07-25
---

# Unit-System Assumptions for this Repository

## Context

The SWE2D solver is **unit-agnostic in its data model**: all lengths, areas, times, and derived quantities are expected to be in the **model's native length unit** (usually feet for US Customary / USC projects, or meters for SI projects). The unit system is derived from the project CRS/map unit, not from hardcoded defaults.

## Default Assumptions for Agents

- **Unless explicitly told otherwise, assume models are run in US Customary (USC) units.** The project owner runs models in feet.
- **Never assume SI/metric constants are correct** when touching the 1D pipe solver, coupling kernels, or drainage network code unless the unit system is explicitly parameterized.
- The 2D SWE solver already propagates unit-aware parameters (`gravity`, `k_mann`, `h_min`) from the CRS through `swe2d.units`.
- The 1D pipe solver currently does **not** receive `k_mann` or `h_min` and contains a hardcoded `PIPE1D_MIN_DEPTH` in meters. This is a known bug to be fixed by the unit-awareness plan.

## What Is Unit-Aware

- 2D SWE friction: uses `gravity` and `k_mann` (1.0 for SI, 1.486 for USC) from `swe2d.units`.
- Structure/culvert coupling kernels: receive `gravity` and `model_to_ft` through `swe2d_gpu_preload_structure_params`.

## What Is NOT Unit-Aware (Known)

- `cpp/src/pipe1d.cu`: Manning friction term `g * n^2 / (A * R^(4/3))` is missing the `k_mann` factor; USC models are off by `1.486^2`.
- `cpp/src/pipe1d.cu`: `PIPE1D_MIN_DEPTH` is hardcoded as `1.0e-4` meters instead of being derived from the configured `h_min` or unit system.
- `swe2d_gpu_apply_pipe_end_bc` hardcodes `h_min = 1.0e-6` instead of using the solver's `h_min`.

## Culvert Solver Exception

The culvert/structure solver is already unit-aware via preloaded `gravity` and `model_to_ft`; do not change its conventions.

---

*Last updated: 2026-07-15*
