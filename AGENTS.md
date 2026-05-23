# AGENTS

## SWE2D Direction

- Treat SWE2D as a GPU-first and GPU-only-in-practice solver effort.
- Prioritize CUDA numerics, CUDA validation, and CUDA performance work.
- Do not treat CPU/GPU parity as an acceptance criterion for SWE2D changes unless a task explicitly asks for a CPU fallback fix.
- The CPU SWE2D path in `cpp/src/swe2d_solver.cpp` is maintenance/debug fallback code only.
- If there is a tradeoff between improving CUDA behavior and preserving CPU similarity, prefer the CUDA path.

## Validation Priority

- Prefer GPU-focused validation suites first:
  - `tests/test_swe2d_gpu_validation_perf.py`
  - `tests/test_swe2d_gpu_unstructured.py`
- The legacy parity suite `tests/test_swe2d_gpu.py` is informational only and should not drive SWE2D design decisions.

## Current Known GPU Status

- CUDA passes the gmsh-based unstructured dam-break checks for spatial schemes 0..4.
- CUDA passes the gmsh-based unstructured lake-at-rest checks for spatial schemes 0..4 after eta-based reconstruction in the higher-order GPU path.
- Current SWE2D engineering priority is CUDA optimization and robustness hardening, not CPU parity.

## Godunov Rollout Handoff

- Use [docs/GODUNOV_2D_GPU_IMPLEMENTATION_GUIDE.md](docs/GODUNOV_2D_GPU_IMPLEMENTATION_GUIDE.md) as the main implementation handoff for the selectable Godunov FVM rollout.

## Repository Session Documentation

- Store implementation handoff and recovery notes in repository-tracked docs under `docs/` so they can be pushed to origin.
- Current rolling session log: [docs/AGENT_SESSION_RECOVERY_LOG.md](docs/AGENT_SESSION_RECOVERY_LOG.md).
