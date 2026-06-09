# D2H/H2D Fallback Path Elimination Tracker

**Goal**: Remove all per-step device-to-host (D2H) and host-to-device (H2D) fallback paths from the runtime step executor. Every code path must execute entirely on the GPU without Python-side state readback or upload.

**Created**: 2026-06-09
**Last Updated**: 2026-06-09

---

## Status Legend

| Status | Meaning |
|--------|---------|
| 🔴 TODO | Not started |
| 🟡 IN PROGRESS | Being implemented |
| 🟢 DONE | On-device path complete, fallback removed |
| ⚪ N/A | Acceptable (e.g., one-time setup transfer) |

---

## 1. Coupling Loop Always CUDA

**Status**: 🟢 DONE

**Location**: `swe2d/runtime/coupling.py`
**Fallback removed**: The CPU coupling loop parameter and entire Python fallback branch in `compute_source_rates()` have been removed. `coupling_loop` is now hardcoded to `"cuda"`. ~50 lines of Python drainage/structure evaluation code deleted.

---

## 2. Native BC Forcing Disabled for Progressive Inflow

**Status**: 🟢 DONE

**Location**: `swe2d/runtime/native_bc_forcing.py` + `cpp/src/swe2d_gpu.cu` + `swe2d/runtime/backend.py`

**Fallback removed**: Progressive inflow no longer disables native BC forcing. The Python side pre-computes group-structured arrays (edge sorting by elevation, cumulative lengths, peak Qs) and uploads them to the GPU. A new CUDA kernel `swe2d_apply_progressive_bc_kernel` handles the Q→q redistribution on-device.

**Implementation**:
- **Python**: `native_bc_forcing.py` groups edges by (side/hydrograph), sorts by bed elevation, builds group-structured arrays, uploads via `set_progressive_bc_data()`
- **C++ kernel**: One block per group, serial scan of cumulative lengths, writes q_unit to active edges and 0 to inactive
- **Graph integration**: Progressive kernel runs AFTER graph replay (outside captured graph) to avoid variant-key explosion. Non-graph and RK5 paths also call it.
- **Files**: `native_bc_forcing.py`, `backend.py`, `model_and_run_methods.py`, `swe2d_gpu.cuh`, `swe2d_gpu.cu`, `swe2d_solver.cpp`, `swe2d_solver.hpp`, `swe2d_bindings.cpp`

---

## 3. Bridge Structures Trigger Python Coupling Path

**Status**: 🔴 TODO

**Location**: `swe2d/runtime/coupling.py`
**Fallback**: `apply_native_device_sources()` returns `False` when `self._has_enabled_bridge_structures` is True, causing D2H state readback and Python source computation.

**Fix**: Implement GPU bridge structure flow computation (stacked deck flow, embankment overflow) in `swe2d_gpu_compute_structure_flows_kernel`.

---

## 4. No CUDA Module Available

**Status**: ⚪ N/A (GPU-only build always has CUDA)

Since GPU-only, `native_mod` is always available. Could add an assertion/guard at coupling controller init.

---

## 5. Drainage Active → Skip Persistent Coupling Path

**Status**: 🟢 DONE

**Location**: `swe2d/runtime/coupling.py`
**Fallback removed**: The `skip_persistent = (self.drainage is not None)` flag has been removed. The persistent GPU coupling path is now always called regardless of drainage. The fused GPU kernel handles both structures and drainage sources on-device.

---

## 6. Python compute_source_rates() Fallback (All Cases)

**Status**: 🟡 PARTIALLY DONE (#1 and #5 resolved; #3 still pending)

**Location**: `swe2d/runtime/coupling.py`
**Current state**: The Python fallback path is now only reachable when `apply_native_device_sources()` returns `False` — which currently happens only for bridge structures (#3). Resolving #3 makes the Python fallback completely unreachable.

---

## 7. State Readback for Predictor-Corrector (Stage-Coupled IMEX)

**Status**: 🟢 N/A (acceptable)

**Location**: `runtime_step_executor.py` lines 81, 119
Stage-coupled IMEX requires two state reads per step for the predictor-corrector average. This is inherent to the IMEX scheme and cannot be eliminated without restructuring the algorithm. Acceptable if IMEX is rarely used.

---

## 8. GPU Coupling Fallback to Python Source Rates

**Status**: 🟢 DONE (already implemented)

**Location**: `coupling.py` lines 980–1100
The `apply_native_device_sources()` method already implements the GPU path. The remaining fallback conditions (#3 bridge structures) are the specific blockers.

---

## 9. External Sources H2D Upload

**Status**: 🟢 DONE (already implemented)

**Location**: `backend.py` `set_external_sources_native()`
Already uses `set_external_sources_native()` for direct GPU upload. No action needed.

---

## Summary

| # | Path | Status | Priority |
|---|------|--------|----------|
| 1 | Coupling loop always CUDA | 🟢 DONE | P0 |
| 2 | Progressive inflow → GPU kernel | 🟢 DONE | P1 |
| 3 | Bridge structures GPU kernel | 🔴 TODO | P1 |
| 4 | CUDA module always available | ⚪ N/A | — |
| 5 | Fused drainage+structure GPU | 🟢 DONE | P0 |
| 6 | Python coupling fallback (all) | 🟡 PARTIAL | P0 (only #3 remaining) |
| 7 | Predictor-corrector state readback | 🟢 N/A | — |
| 8 | GPU coupling implementation | 🟢 DONE | — |
| 9 | External sources native | 🟢 DONE | — |

**Overall progress: 5/9 DONE, 1 N/A, 1 PARTIAL, 1 TODO**

---

## Implementation Log

### 2026-06-09: #1 Coupling Loop Always CUDA
- Removed `coupling_loop` parameter from `SWE2DCouplingController.__init__()`
- Kept `self.coupling_loop = "cuda"` as hardcoded attribute (for logging)
- Removed CPU fallback branch in `compute_source_rates()` (~50 lines)
- Removed dead `if self.coupling_loop != "cuda"` check in `apply_native_device_sources()`

### 2026-06-09: #2 Progressive Inflow → GPU Kernel
- Removed progressive guard that returned `native_bc_forcing=False`
- Added group-structured array pre-computation in `native_bc_forcing.py`
- Added `set_progressive_bc_data()` to `backend.py`
- Added `swe2d_apply_progressive_bc_kernel` in `swe2d_gpu.cu`
- Added `swe2d_gpu_set_progressive_bc_data()` upload function
- Added pybind11 binding in `swe2d_bindings.cpp`
- Wired progressive kernel into all 3 step paths (swe2d_gpu_step, persistent_chunk, rk5_graph)
- Graph replay enabled: progressive kernel runs after graph replay outside captured graph

### 2026-06-09: #5 Fused Drainage+Structure GPU
- Removed `skip_persistent = (self.drainage is not None)` from `coupling.py`
- Persistent GPU coupling path now always used, regardless of drainage

---

## Test Verification

After each fix, run:
```bash
conda run -n qgis_stable python -m pytest tests/test_workbench_imports.py -v
```
Expected: All 17 workbench import tests pass.
