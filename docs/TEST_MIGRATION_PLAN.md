# Test Migration & Rename Plan: Drainage Pipe1D Tests

## Context

The old `swe2d_gpu_drainage_step` C++ kernel was deleted and replaced with
`swe2d_build_pipe1d_mesh` + `swe2d_pipe1d_step`. Two test files exist:

1. `tests/test_swe2d_gpu_drainage_network.py` — Tests the deleted C API directly.
   All 4 drainage tests skip. The culvert face-flux class still works.
2. `tests/test_swe2d_drainage_structures.py` — Tests coupling/controller integration.
   Name is misleading (tests coupling orchestration, not structure formulas).
   No direct tests of the pipe1d C kernels.

**Gap**: Zero isolated tests of the pipe1d GPU kernels.

## Changes

### 1. Rename `test_swe2d_drainage_structures.py` → `test_coupling_integration.py`

Pure rename. No test logic changes.

### 2. Add `swe2d_pipe1d_readback_node_state` binding

Needed to verify pipe1d kernel results in tests. Follows existing readback pattern
(`swe2d_gpu_readback_coupling_sources` etc.) — uses `s_coupling_dev` global.

**Files**:
- `cpp/src/swe2d_gpu.cu`: Add `swe2d_pipe1d_readback_node_state(double* node_depth_out, double* cell_A_out, double* cell_Q_out, int32_t n_nodes, int32_t n_cells)`
- `cpp/src/swe2d_gpu.cuh`: Add declaration
- `cpp/src/swe2d_bindings.cpp`: Add pybind11 binding

Returns node_depth, cell_A, cell_Q as numpy arrays via dict.

### 3. Rewrite `test_swe2d_gpu_drainage_network.py` → `test_swe2d_pipe1d.py`

Keep `TestGPUCulvertFaceFluxComputeSanitizer` class untouched.
Replace skipped `TestGPUDrainageStepComputeSanitizer` with new pipe1d tests.

#### `TestPipe1DMeshBuild`
| Test | Validates |
|------|-----------|
| `test_build_mesh_single_link` | 1 link, 2 nodes. `n_pipe_cells >= 1`, no crash. |
| `test_build_mesh_subdivision` | `max_cell_length` triggers sub-cells. |
| `test_build_mesh_idempotent` | Double-build doesn't leak or crash. |

#### `TestPipe1DStep`
| Test | Validates |
|------|-----------|
| `test_diffusion_wave_converges` | Sloped pipe (invert 1→0). Flow > 0, node depth changes. |
| `test_fully_dynamic_converges` | Same setup, fully_dynamic mode. Flow > 0. |
| `test_dry_pipe_no_change` | Zero depths → A/Q unchanged after step. |
| `test_substeps_vs_single` | coupling_substeps=4 produces different result than 1. |

Each test: skip if no GPU → create solver → get device capsule → build mesh → step → readback → assert.

## File Changes

| File | Action |
|------|--------|
| `tests/test_swe2d_drainage_structures.py` | Rename → `test_coupling_integration.py` |
| `tests/test_swe2d_gpu_drainage_network.py` | Rewrite → `test_swe2d_pipe1d.py` (keep culvert class) |
| `cpp/src/swe2d_gpu.cu` | Add `swe2d_pipe1d_readback_node_state` |
| `cpp/src/swe2d_gpu.cuh` | Add declaration |
| `cpp/src/swe2d_bindings.cpp` | Add pybind11 binding |

## Verification

```bash
cd build && make -j$(nproc)
mamba run -n qgis_stable python -m pytest tests/test_swe2d_pipe1d.py -v
mamba run -n qgis_stable python -m pytest tests/test_coupling_integration.py -v
```
