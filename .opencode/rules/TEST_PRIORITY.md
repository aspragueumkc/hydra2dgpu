# Test Priority

## Validation Priority

- Prefer GPU-focused validation suites first:
  - `tests/test_swe2d_gpu_validation_perf.py`
  - `tests/test_swe2d_gpu_unstructured.py`
- The legacy parity suite `tests/test_swe2d_gpu.py` is informational only and should not drive SWE2D design decisions.

## Running Tests

All tests use `unittest` (not pytest). Run from repo root:

```bash
# GPU validation (primary gate)
PYTHONPATH="$PWD:$PWD/build" python3 -m unittest -v \
    tests.test_swe2d_gpu_validation_perf \
    tests.test_swe2d_gpu_unstructured

# Single test file
PYTHONPATH="$PWD:$PWD/build" python3 -m unittest -v tests.test_swe2d_dambreak

# Discover all available tests
PYTHONPATH="$PWD:$PWD/build" python3 -m unittest discover -s tests -p "test_*.py" -v
```

## Current Known GPU Status

- CUDA passes the gmsh-based unstructured dam-break checks for spatial schemes 0..4.
- CUDA passes the gmsh-based unstructured lake-at-rest checks for spatial schemes 0..4 after eta-based reconstruction in the higher-order GPU path.
- Current SWE2D engineering priority is CUDA optimization and robustness hardening, not CPU parity.

## Godunov Rollout Handoff

- Use `docs/GODUNOV_2D_GPU_IMPLEMENTATION_GUIDE.md` as the main implementation handoff for the selectable Godunov FVM rollout.
