# C++ / CUDA Architecture

## Module Layout

```
cpp/src/
  swe2d_mesh.hpp / .cpp         — Mesh structure (SWE2DMesh), builder, validation
  swe2d_solver.hpp / .cpp       — Solver config (SWE2DSolverConfig), GPU state management
  swe2d_numerics.hpp / .cpp     — Numerical kernels (HLLC Riemann, bed slope, friction, CFL)
  swe2d_gpu.cuh                 — CUDA device state (SWE2DDeviceState), kernel API declarations
  swe2d_gpu.cu                  — CUDA kernel implementations (flux, reconstruction, update)
  swe2d_gpu_redistribute.cu     — CUDA redistribution kernel for structure coupling
  swe2d_units.cuh               — USC/SI unit conversion constants for device code
  swe2d_bindings.cpp            — pybind11 module: exposes all functions to Python
  hybrid_mesh_bindings.cpp      — pybind11 module: hybrid mesh builder
  meshing_native_bindings.cpp   — pybind11 module: native Gmsh mesh helpers
  overlay_backend.cpp           — pybind11 module: high-performance canvas overlay
```

All modules are built by CMake from `CMakeLists.txt` at the repo root.

## Build System (`CMakeLists.txt`)

| Target | Module Name | Source | Purpose |
|---|---|---|---|
| `hydra_swe2d` | `hydra_swe2d` | `swe2d_*.cpp`, `swe2d_gpu.cu` | Main solver (mesh + GPU/CPU solver + coupling) |
| `hydra_hybridmesh` | `hydra_hybridmesh` | `hybrid_mesh_bindings.cpp` | Channel-guided hybrid meshing |
| `hydra_meshing_native` | `hydra_meshing_native` | `meshing_native_bindings.cpp` | Native polyline math for Gmsh |
| `hydra_overlay` | `hydra_overlay` | `overlay_backend.cpp` | High-performance canvas overlay |

### Key Build Options

| Option | Default | Description |
|---|---|---|
| `BACKWATER_USE_CUDA` | ON | Enable CUDA GPU path. Falls back to CMake fatal error if CUDA toolkit not found. |
| `CMAKE_BUILD_TYPE` | RelWithDebInfo | Optimised with debug info (`-O2 -g`). |

### CUDA Architecture Targets

| CUDA Version | Architectures |
|---|---|
| ≥ 13.0 | sm_75, sm_80, sm_86, sm_89, sm_90 |
| < 13.0 | sm_70, sm_75, sm_80, sm_86 |

### Compiler Flags

- **Host**: `-O3 -mtune=native -march=native -ffast-math` (GCC/Clang)
- **Device**: `--expt-relaxed-constexpr --expt-extended-lambda -use_fast_math -O3`
- **CUDA host compiler fallback**: GCC ≥ 14 auto-detected and replaced with `gcc-13` for CUDA 13+ compatibility

## Unit Convention in C++ Kernels

**The kernel accepts geometry in model units** (feet or meters). Key rules:

- `gravity` parameter is provided by the caller in model units (9.81 m/s² or 32.17 ft/s²)
- Weir, orifice, bridge, and pump formulas are unit-agnostic — correct for any unit system as long as `gravity` matches
- **HDS-5 culvert is the only path that converts to feet internally**: geometry → feet, compute in USC, result back to model units
- Culvert output: CFS → model units via `÷ model_to_ft³`
- Unit constants for device code are in `swe2d_units.cuh`

## Data Flow

```
Python (SWE2DBackend)
    ↓  numpy arrays
swe2d_bindings.cpp (pybind11)
    ↓
swe2d_mesh.hpp/cpp — SWE2DMesh construction, validation, edge reordering
    ↓
swe2d_solver.hpp/cpp — SWE2DSolverConfig, SWE2DPySolver lifecycle
    ↓
swe2d_gpu.cuh/.cu — GPU upload, kernel launch, state management
    ↓
swe2d_numerics.hpp — __host__ __device__ inline numerical kernels
```

## Mesh Structure (`SWE2DMesh`)

Defined in `swe2d_mesh.hpp`. Structure-of-Arrays (SoA) layout for GPU coalescing.

| Array | Shape | Description |
|---|---|---|
| `node_x/y/z` | `[n_nodes]` | Node coordinates |
| `cell_face_offsets` | `[n_cells+1]` | CSR offsets into `cell_face_nodes` |
| `cell_face_nodes` | `[sum(n_verts)]` | Node indices for each cell ring (CCW) |
| `cell_edge_offsets` | `[n_cells+1]` | CSR offsets into `cell_edge_ids` |
| `cell_edge_ids` | `[sum(n_verts)]` | Edge indices for each cell |
| `edge_c0/c1` | `[n_edges]` | Left/right cell indices (-1 = boundary) |
| `edge_n0/n1` | `[n_edges]` | Endpoint node indices |
| `edge_nx/ny` | `[n_edges]` | Unit outward normal (from c0) |
| `edge_len` | `[n_edges]` | Edge length |
| `edge_bc` | `[n_edges]` | Boundary condition type (`BCType` enum) |
| `cell_cx/cy/area/zb` | `[n_cells]` | Derived geometry |
| `cell_ring2_*` | `[sum(ring2)]` | 2-ring stencil for WENO5 gradient |

## Solver Configuration (`SWE2DSolverConfig`)

Defined in `swe2d_solver.hpp`. Controls all solver parameters including spatial scheme, temporal order, friction model, stability hardening, tiny-mode dispatch, and degenerate cell handling.

Key enums:
- `SWE2DSpatialScheme`: FV_FIRST_ORDER (0), FV_MUSCL_FAST (1), FV_MUSCL_MINMOD (2), FV_MUSCL_MC (3), FV_MUSCL_VAN_LEER (4), FV_WENO5 (6)
- `SWE2DTurbulenceModel`: NONE, SMAGORINSKY, K_EPSILON, K_OMEGA_SST
- `SWE2DBedFrictionModel`: MANNING, CHEZY, DARCY_WEISBACH, NIKURADSE
- `BCType`: INTERIOR, WALL, INFLOW_Q, STAGE, OPEN, REFLECT, NORMAL_DEPTH, NORMAL_DEPTH_SLOPE
