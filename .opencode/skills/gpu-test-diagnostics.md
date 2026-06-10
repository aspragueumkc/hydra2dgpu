---
name: gpu-test-diagnostics
description: Diagnose GPU test failures (CUDA, hydra_swe2d)
---

# GPU Test Diagnostics

When a GPU test fails, follow these steps:

## 1. Check GPU availability

```python
from swe2d.runtime.backend import swe2d_gpu_available
print(f"GPU available: {swe2d_gpu_available()}")
print(f"swe2d available: {swe2d_available()}")
```

If `swe2d_gpu_available()` is False:
- Native module not built — run `cd build && cmake .. && make -j$(nproc)`
- CUDA toolkit not found — check `nvidia-smi` and `which nvcc`
- GPU not detected — check `python -c "import torch; print(torch.cuda.is_available())"` or `nvidia-smi`

## 2. Check the test output

GPU test failures typically show:
- `NaN` in state arrays → numerical instability (lower CFL, use first-order)
- `AssertionError: ...` → expected tolerance not met
- `segfault` → native module crash (rebuild, check CUDA version)

## 3. Run with diagnostics

```bash
export BACKWATER_SWE2D_DIAG_MODE=1
PYTHONPATH="$PWD:$PWD/build" python3 -m unittest \
    tests.test_swe2d_gpu_unstructured -v
```

## 4. Check tiny-mode

If the mesh is small (< 200 cells), tiny-mode dispatch may be active:
- Check `tiny_mode_effective` in step diagnostics
- Disable with `tiny_mode=0` in solver config
- Fused (mode 2) and persistent (mode 3) have different numerical paths

## 5. Check graph caching

If CUDA graph replay fails:
- Disable with `gpu_enable_kernel_graphs=False`
- Graph cache invalidates on mesh size, scheme, or RK order change

## 6. Common GPU test failures

| Symptom | Likely cause | Fix |
|---|---|---|
| `swe2d_gpu_available() == False` | hydra_swe2d not built | `cd build && cmake .. && make` |
| NaN in h/hu/hv | CFL too high | Lower CFL to 0.3, use first-order |
| `max_courant > 1.0` | Stability violation | Reduce dt, enable damping |
| Graph replay fails | Changed config mid-run | Disable CUDA graphs |
| Tiny mode mismatch | Small mesh | Disable tiny mode: `tiny_mode=0` |
| WENO5 gradient error | 2-ring stencil issue | Fall back to MUSCL scheme |
