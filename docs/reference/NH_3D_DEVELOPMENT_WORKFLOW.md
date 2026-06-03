# Nonhydrostatic 2D & 3D Development Workflow

Date: 2026-05-15
Status: Active development

## Quick Start for Future Work

**Always use the insertion plan subagent before touching code files.** This practice ensures safe, compile-safe scaffolding.

### Subagent Invocation Pattern

```
You are assisting with C++/CUDA architecture work in qgis-backwater-plugin. 
Do read-only analysis only (no file edits). 
Goal: identify minimal, compile-safe scaffolding changes for [TASK_NAME].
Constraints: GPU-first; advanced modes are scaffold-only for now, not full implementation. 
Return: 
1) exact files and symbols to touch
2) proposed struct/function signatures
3) ordering of edits to minimize breakage
4) pitfalls with existing code patterns
```

### Current Scaffolding Phases

#### Phase 1: Structure & Config (COMPLETED)
- Enum selectors: equation_set, coupling_mode, 3d_solver_model
- GPU-only enforcement policy
- Config pass-through from Python → native

**Files touched:** `swe2d_extensions.py`, `swe2d_backend.py`, `swe2d_solver.hpp`, `swe2d_bindings.cpp`, `swe2d_solver.cpp`

#### Phase 2: B2 Nonhydro Hooks (IN PROGRESS)
- Predictor/corrector entry point
- Dispatch routing in `swe2d_step`
- Fail-fast scaffold implementation

**Files to touch:** `swe2d_gpu.cuh`, `swe2d_gpu.cu`, `swe2d_solver.cpp`

#### Phase 3: D2 3D Patch Allocation (IN PROGRESS)
- Cartesian patch descriptor
- Device allocation/release API
- Cleanup integration

**Files to touch:** `swe2d_gpu.cuh`, `swe2d_gpu.cu`

#### Phase 4: E1 2D-3D Interface Contract (IN PROGRESS)
- Host and device contract types
- Upload/clear/apply scaffold APIs
- Pybind exposure (optional first pass)

**Files to touch:** `swe2d_solver.hpp`, `swe2d_gpu.cuh`, `swe2d_gpu.cu`, potentially `swe2d_bindings.cpp`

#### Phase 5: NH Pressure Workspace (NEXT)
- Pressure increment buffer on device
- Residual tracking workspace
- Pressure coefficient matrix skeleton

**Files to touch:** `swe2d_gpu.cuh`, `swe2d_gpu.cu`

#### Phase 6: 2D-3D Exchange Kernel (NEXT)
- Mass flux sign convention (m³/s, positive 2D→3D)
- Momentum flux components (momx, momy)
- Head-loss correction terms

**Files to touch:** `swe2d_gpu.cu` (new kernel family)

#### Phase 7: Pybind Contract API (NEXT AFTER KERNELS)
- Optional Python exposure for setting/clearing contracts
- Conservative defaults; backward-compatible with hydrostatic callers

**Files to touch:** `swe2d_bindings.cpp`

#### Phase 8: STL Geometry Ingestion + Structured Patch Build (NEW)
- Import STL solids and validate manifold/units assumptions
- Define patch ROI + `(nx, ny, nz)` controls from workbench
- Build Cartesian patch occupancy/porosity/open-area tensors (`phi`, `ax`, `ay`, `az`)

**Files to touch:** `swe2d_workbench_qt.py`, `swe2d_backend.py`, `cpp/src/swe2d_solver.cpp`, `cpp/src/swe2d_gpu.cuh/.cu`

#### Phase 9: QGIS 3D Viewer Outputs (NEW)
- Snapshot export of 3D patch slices/surfaces for QGIS 3D viewer
- Minimal velocity/scalar review products (`vof`, `p`, `|u|`)
- Keep full-volume rendering out of MVP scope

**Files to touch:** `swe2d_workbench_qt.py`, export/query helpers, native snapshot hooks as needed

#### Phase 10: 3D Numerics Core Replacement (IMMEDIATE PRIORITY)
- Replace scaffold damping kernel with real 3D operator split:
   - advection/diffusion predictor
   - pressure Poisson/projection
   - velocity correction
   - bounded VoF transport
- Preserve uncoupled-mode validation gates as blocking acceptance criteria

**Files to touch:** `cpp/src/swe2d_gpu.cuh/.cu`, related 3D test harnesses

## Safe Editing Strategy

1. **Always call subagent first** with the specific task name (B2, D2, E1, etc.)
   - Returns insertion locations, signature proposals, pitfall notes
   - Avoids guessing about linkage, memory ownership, dispatch order

2. **Edit in the proposed order** to minimize linker breaks
   - Header declarations first (`.cuh`, `.hpp`)
   - No-op implementations next (`.cu`)
   - Dispatch/integration last (`.cpp`)

3. **Preserve fail-fast semantics**
   - Unimplemented paths throw explicit scaffold-not-ready errors
   - Do not silently fall back to hydrostatic behavior for advanced modes

4. **Avoid scope creep**
   - Each slice: one config field, one API entry point, or one kernel family
   - Do not implement pressure-solving algorithms yet; just allocate buffers

## Testing Checkpoints

After each slice:
1. Build clean: `cmake --build build/`
2. No new link errors
3. Hydrostatic mode runs unchanged
4. Advanced modes fail fast with clear errors (not silent success)
5. For 3D slices, uncoupled validation tests remain green before moving to coupling work

## Commit Hygiene

Group slices by phase before committing:
- Commit after Phase 2 (B2 predictor/corrector hooks complete)
- Commit after Phase 3 (D2 3D patch API complete)
- Commit after Phase 4 (E1 interface contract complete)
- Then: Phase 5–7 constitute v0-alpha (internal testing before public push)

## Notes for Implementation

- **Sign convention for 2D-3D fluxes**: positive = 2D source cell → 3D patch cell (removal from surface)
- **Pressure matrix format**: scaffold only stores pointers; actual sparse format TBD per preconditioner choice
- **Graph cache interaction**: ensure nonhydro/coupled paths do not replay hydrostatic CUDA graphs
- **Device memory hygiene**: every new pointer in device state must be null-initialized, allocated on demand, and freed in destroy()
- **STL preprocessing discipline**: perform topology/unit checks before voxelization; never silently auto-fix invalid geometry without logging what changed
