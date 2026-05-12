# Godunov 2D GPU Implementation Guide

This guide is the working handoff for the selectable Godunov FVM rollout in SWE2D.

## Goal

Deliver a second-order GPU-first shallow-water solver mode that remains selectable from the 2D Workbench GUI during rollout and can coexist with the current GPU path while validation matures.

## Implementation Order

1. Native config and dispatch plumbing.
2. GPU kernel implementation for the Godunov path.
3. Workbench GUI selection and persisted rollout controls.
4. GPU validation coverage and regression baselines.
5. Performance hardening and rollout cleanup.

## Required Numerical Targets

- Second-order accuracy as the minimum rollout bar.
- Hydrostatic reconstruction for bed/surface balance.
- HLLC-style Godunov fluxing on edges.
- Wet/dry robustness with conservative fallback near shallow fronts.
- Point-implicit or otherwise stable source/friction handling for stiff terms.

## Validation Priority

- `tests/test_swe2d_gpu_validation_perf.py`
- `tests/test_swe2d_gpu_unstructured.py`
- `tests/test_swe2d_gpu_unstructured_rain.py`
- `tests/test_swe2d_gpu_coupling_kernel.py`

## Notes for Future Edits

- Keep the current GPU path selectable while the Godunov mode is rolled out.
- Prefer CUDA changes over CPU fallback parity work.
- Make each rollout stage independently testable before broadening the scope.