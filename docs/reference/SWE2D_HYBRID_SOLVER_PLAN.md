# 2D SWE Hybrid GPU/CPU Solver — Detailed Plan and Skeleton

## Status Update (Current Direction)

The active implementation strategy has moved to a GPU-primary SWE2D roadmap.
CUDA kernels are now the main target for higher-order numerics and validation
work, while the CPU path is retained as a compatibility and debugging fallback.

## 1. Overview

This document details the design, data contracts, and implementation skeleton for the
2D Shallow Water Equations (SWE) solver component of the backwater plugin native backend.

The solver targets real-world riverine geometries and uses:
- **Unstructured triangular mesh** for boundary-conforming spatial discretization.
- **HLLC Riemann solver** with positivity-preserving wave speeds for robust, conservative flux computation.
- **CUDA GPU path** (NVIDIA) for parallel flux/update kernels; auto-detected at CMake time.
- **OpenMP CPU path** as the always-available fallback; same numerical kernels.
- **Well-balanced bed-slope source term** to preserve lake-at-rest to machine precision.
- **Python/QGIS orchestration layer** unchanged — path selection is internal to the native backend.

---

## 2. File Layout

```
cpp/src/
  swe2d_mesh.hpp          # SoA mesh structs, edge layout, BC type enum
  swe2d_mesh.cpp          # Mesh builder (from node/element arrays), edge connectivity
  swe2d_numerics.hpp      # HLLC kernel, reconstruction, well-balanced bed slope, CFL
  swe2d_numerics.cpp      # CPU implementations (called by solver and unit tests)
  swe2d_solver.hpp        # Solver lifecycle API (opaque handle)
  swe2d_solver.cpp        # CPU solver: OpenMP flux + update + CFL loops
  swe2d_gpu.cuh           # CUDA declarations (only compiled when CUDA found)
  swe2d_gpu.cu            # CUDA kernels: flux, update, CFL reduction
  swe2d_bindings.cpp      # pybind11 module: backwater_swe2d

swe2d_backend.py          # Python bridge: path selector, GPU query, numpy wrappers
tests/
  test_swe2d_mesh.py      # Mesh construction and edge classification tests
  test_swe2d_dambreak.py  # Dam-break analytical comparison
  test_swe2d_lakerest.py  # Still-water / lake-at-rest balance test
  test_swe2d_uniform.py   # Uniform flow test (Manning friction + slope)
  test_swe2d_gpu.py       # GPU path smoke test (skipped if CUDA unavailable)
```

---

## 3. Data Contracts

All arrays are **double precision** (`float64`) unless noted. All arrays are C-contiguous
(row-major) numpy arrays when passed over the pybind11 boundary.

### 3.1 Mesh Input (Python → native)

| Array          | Shape   | Units | Description                          |
|----------------|---------|-------|--------------------------------------|
| `node_x`       | `(N,)`  | m     | Node X coordinates                   |
| `node_y`       | `(N,)`  | m     | Node Y coordinates                   |
| `node_z`       | `(N,)`  | m     | Bed elevation at nodes               |
| `cell_nodes`   | `(M,3)` | —     | Node index triplets per cell (CCW)   |
| `bc_edge_type` | `(E,)`  | —     | int32 BC type per boundary edge      |
| `bc_edge_value`| `(E,)`  | m/s²  | BC prescribed value (h or Q)         |

N = node count, M = cell count, E = boundary edge count.

### 3.2 BC Type Enum

| Value | Name        | Description                              |
|-------|-------------|------------------------------------------|
| 0     | INTERIOR    | Not a boundary edge                      |
| 1     | WALL        | Zero normal flux                         |
| 2     | INFLOW_Q    | Prescribed inflow discharge (m³/s/m)     |
| 3     | STAGE       | Prescribed water-surface elevation (m)   |
| 4     | OPEN        | Riemann outflow (zero-gradient)          |
| 5     | REFLECT     | Reflecting (velocity sign flip)          |

### 3.3 Solver State (Python ↔ native)

| Array    | Shape  | Units | Description                  |
|----------|--------|-------|------------------------------|
| `h`      | `(M,)` | m     | Water depth per cell         |
| `hu`     | `(M,)` | m²/s  | x-momentum per cell          |
| `hv`     | `(M,)` | m²/s  | y-momentum per cell          |

### 3.4 Diagnostics Returned Per Step

| Field          | Type    | Description                                |
|----------------|---------|--------------------------------------------|
| `dt`           | float   | Timestep actually used (s)                 |
| `wet_cells`    | int     | Number of cells with h > h_min             |
| `max_depth`    | float   | Maximum h over all cells (m)               |
| `min_depth`    | float   | Minimum non-zero h (m)                     |
| `mass_total`   | float   | Total water volume (m³)                    |
| `gpu_active`   | bool    | True if this step ran on GPU               |

---

## 4. Internal Mesh Representation (SoA, native side)

```cpp
// swe2d_mesh.hpp

enum class BCType : int32_t {
    INTERIOR = 0,
    WALL     = 1,
    INFLOW_Q = 2,
    STAGE    = 3,
    OPEN     = 4,
    REFLECT  = 5
};

struct SWE2DMesh {
    // --- Nodes ---
    int32_t  n_nodes;
    double*  node_x;      // [n_nodes]
    double*  node_y;      // [n_nodes]
    double*  node_z;      // [n_nodes] bed elevation

    // --- Cells ---
    int32_t  n_cells;
    int32_t* cell_n0;     // [n_cells] node indices
    int32_t* cell_n1;
    int32_t* cell_n2;
    double*  cell_cx;     // [n_cells] centroid x
    double*  cell_cy;     // [n_cells] centroid y
    double*  cell_area;   // [n_cells]
    double*  cell_zb;     // [n_cells] bed elevation at centroid

    // --- Edges ---
    int32_t  n_edges;
    int32_t* edge_c0;     // [n_edges] left cell (-1 = boundary)
    int32_t* edge_c1;     // [n_edges] right cell (-1 = boundary)
    int32_t* edge_n0;     // [n_edges] endpoint node 0
    int32_t* edge_n1;     // [n_edges] endpoint node 1
    double*  edge_nx;     // [n_edges] outward unit normal x (c0→c1)
    double*  edge_ny;     // [n_edges] outward unit normal y
    double*  edge_len;    // [n_edges] edge length
    BCType*  edge_bc;     // [n_edges] BC type
    double*  edge_bc_val; // [n_edges] prescribed BC value
};
```

### Edge Connectivity Build Algorithm
```
For each cell (c0, c1, c2) ordered CCW:
  For each local edge (n_a, n_b) in {(c0,c1),(c1,c2),(c2,c0)}:
    Canonical key = (min(n_a,n_b), max(n_a,n_b))
    If key exists in edge_map:
      Set edge_c1 = current cell (edge_c0 was set by first cell that saw this edge)
    Else:
      Insert new edge: edge_c0 = current cell, edge_c1 = -1 (boundary)
      Compute normal from node positions (outward from c0)
After build:
  All edges with edge_c1 == -1 are boundary edges → classify BC type
```

---

## 5. Numerical Kernel Contracts

### 5.1 HLLC Riemann Solver

```cpp
// swe2d_numerics.hpp
// Input: left/right states (hL,uL,vL,zbL), (hR,uR,vR,zbR), edge normal (nx,ny)
// Input: g = 9.81, h_min = 1e-6
// Output: flux_h, flux_hu, flux_hv (normal fluxes in the edge direction)
void hllc_flux(
    double hL, double uL, double vL, double zbL,
    double hR, double uR, double vR, double zbR,
    double nx, double ny,
    double g, double h_min,
    double& flux_h, double& flux_hu, double& flux_hv
);
```

HLLC wave speed estimates (Einfeldt / Roe-averaged):
```
c_L = sqrt(g * max(hL, 0))
c_R = sqrt(g * max(hR, 0))
u_star = (uL + uR) / 2 + c_L - c_R          # Roe-averaged
c_star = (c_L + c_R) / 2 + (uL - uR) / 4
S_L = min(u_L_n - c_L, u_star - c_star)
S_R = max(u_R_n + c_R, u_star + c_star)
```
where `u_L_n = uL*nx + vL*ny`, `u_R_n = uR*nx + vR*ny`.

### 5.2 Well-Balanced Bed Slope (Hydrostatic Reconstruction)
For each edge, reconstruct interface depths that preserve h+zb:
```
eta_L = hL + zbL    # water surface
eta_R = hR + zbR
zb_face = max(zbL, zbR)
hL_star = max(0, eta_L - zb_face)
hR_star = max(0, eta_R - zb_face)
```
Use `hL_star`, `hR_star` in the Riemann solve. Add hydrostatic bed-slope correction:
```
dz = zbR - zbL    # (signed, along normal)
bed_slope_correction = -0.5 * g * (hL_star^2 - hL^2)   # per edge, distributed to c0
```

### 5.3 Manning Friction Source
Per cell, after flux update:
```
U = hu / max(h, h_min)
V = hv / max(h, h_min)
spd = sqrt(U^2 + V^2)
R_h = h                           # hydraulic radius ≈ h for shallow flow
Cf = g * n^2 / R_h^(4/3)
# Semi-implicit limiter: clamp so friction can't reverse velocity
denom = 1 + dt * Cf * spd / max(h, h_min)
hu_new = hu / denom
hv_new = hv / denom
```

### 5.4 CFL Timestep
```
For each edge:
  h_face = (hL + hR) / 2
  c_face = sqrt(g * max(h_face, 0))
  u_face = abs((huL + huR) / (2 * max(h_face, h_min)))
  lambda = (u_face + c_face) / edge_len  (characteristic speed / cell size proxy)
dt = CFL_factor / max_over_all_edges(lambda)
```
CFL_factor default: 0.45 (explicit; stable for triangular meshes).

---

## 6. CPU Solver Loop (OpenMP)

```cpp
// swe2d_solver.cpp
void swe2d_step_cpu(SWE2DMesh& mesh, double* h, double* hu, double* hv,
                    double dt, double g, double n_mann, double h_min,
                    double* flux_h_acc, double* flux_hu_acc, double* flux_hv_acc)
{
    // Zero flux accumulators
    #pragma omp parallel for schedule(static)
    for (int c = 0; c < mesh.n_cells; c++) {
        flux_h_acc[c] = flux_hu_acc[c] = flux_hv_acc[c] = 0.0;
    }

    // Flux loop over all edges (interior + boundary)
    #pragma omp parallel for schedule(static)
    for (int e = 0; e < mesh.n_edges; e++) {
        int c0 = mesh.edge_c0[e];
        int c1 = mesh.edge_c1[e];
        // Extract left/right states (ghost cell for boundary edges)
        // ... compute ghost state based on BCType ...
        double fh, fhu, fhv;
        hllc_flux(hL, uL, vL, zbL, hR, uR, vR, zbR,
                  mesh.edge_nx[e], mesh.edge_ny[e], g, h_min,
                  fh, fhu, fhv);
        double len = mesh.edge_len[e];
        // Accumulate into c0 (subtract) and c1 (add) — note: race condition → use atomic or separate pass
        #pragma omp atomic
        flux_h_acc[c0]  -= fh  * len;
        #pragma omp atomic
        flux_hu_acc[c0] -= fhu * len;
        #pragma omp atomic
        flux_hv_acc[c0] -= fhv * len;
        if (c1 >= 0) {
            #pragma omp atomic
            flux_h_acc[c1]  += fh  * len;
            #pragma omp atomic
            flux_hu_acc[c1] += fhu * len;
            #pragma omp atomic
            flux_hv_acc[c1] += fhv * len;
        }
    }

    // Update loop over all cells
    #pragma omp parallel for schedule(static)
    for (int c = 0; c < mesh.n_cells; c++) {
        double inv_area = 1.0 / mesh.cell_area[c];
        h[c]  += dt * flux_h_acc[c]  * inv_area;
        hu[c] += dt * flux_hu_acc[c] * inv_area;
        hv[c] += dt * flux_hv_acc[c] * inv_area;
        // Positivity enforcement
        h[c] = max(h[c], 0.0);
        if (h[c] < h_min) { hu[c] = hv[c] = 0.0; }
        // Manning friction
        // ... apply semi-implicit limiter ...
    }
}
```

**Note on atomics**: For performance on large meshes, the flux accumulation can be rewritten using a pre-sorted edge-to-cell CSR layout that eliminates atomic ops (one pass per cell over its contributing edges). This is a follow-up optimization — atomics are correct and sufficient for MVP.

---

## 7. GPU Kernel Contracts (CUDA)

```cuda
// swe2d_gpu.cu

// Kernel 1: Flux computation — one thread per edge
__global__ void swe2d_flux_kernel(
    int n_edges,
    const int32_t* edge_c0, const int32_t* edge_c1,
    const double* edge_nx, const double* edge_ny, const double* edge_len,
    const int32_t* edge_bc, const double* edge_bc_val,
    const double* cell_h, const double* cell_hu, const double* cell_hv,
    const double* cell_zb,
    double* flux_h_acc, double* flux_hu_acc, double* flux_hv_acc,
    double g, double h_min
);

// Kernel 2: State update — one thread per cell
__global__ void swe2d_update_kernel(
    int n_cells,
    double* h, double* hu, double* hv,
    const double* flux_h_acc, const double* flux_hu_acc, const double* flux_hv_acc,
    const double* cell_area,
    double dt, double g, double n_mann, double h_min
);

// Kernel 3: CFL reduction — one thread per cell, block reduce to shared mem, then global atomicMin
__global__ void swe2d_cfl_kernel(
    int n_cells,
    const double* h, const double* hu, const double* hv,
    const double* cell_area,  // used as proxy for dx
    double g, double h_min,
    double* d_lambda_max       // device scalar, output
);
```

### Device Memory Lifecycle

```cpp
struct SWE2DDeviceState {
    // Mesh topology (static after init)
    int32_t *d_edge_c0, *d_edge_c1, *d_edge_n0, *d_edge_n1;
    double  *d_edge_nx, *d_edge_ny, *d_edge_len;
    int32_t *d_edge_bc;
    double  *d_edge_bc_val;
    double  *d_cell_zb, *d_cell_area;

    // State (updated each step)
    double *d_h, *d_hu, *d_hv;

    // Flux accumulators (zeroed each step)
    double *d_flux_h, *d_flux_hu, *d_flux_hv;

    // CFL workspace
    double *d_lambda_max;

    int n_cells, n_edges;
};

// API:
SWE2DDeviceState* swe2d_gpu_init(const SWE2DMesh& mesh, const double* h0, const double* hu0, const double* hv0);
void swe2d_gpu_step(SWE2DDeviceState* dev, double dt, double g, double n_mann, double h_min, double cfl_factor, SWE2DStepDiag* diag);
void swe2d_gpu_get_state(SWE2DDeviceState* dev, double* h_out, double* hu_out, double* hv_out);
void swe2d_gpu_destroy(SWE2DDeviceState* dev);
bool swe2d_gpu_available();
```

---

## 8. Solver Handle API (CPU + GPU unified)

```cpp
// swe2d_solver.hpp

struct SWE2DSolverConfig {
    double g        = 9.81;
    double n_mann   = 0.035;    // Manning's n (global default; per-cell override planned)
    double h_min    = 1e-6;     // Thin-film threshold (m)
    double cfl      = 0.45;     // CFL safety factor
    double dt_max   = 10.0;     // Maximum allowed timestep (s)
    double dt_fixed = -1.0;     // If > 0, override CFL timestep with fixed value
    bool   use_gpu  = true;     // Try GPU; fall back to CPU if unavailable
    int    n_threads = 0;       // CPU thread count (0 = auto)
};

struct SWE2DStepDiag {
    double  dt;
    int32_t wet_cells;
    double  max_depth;
    double  min_depth;
    double  mass_total;
    bool    gpu_active;
};

// Opaque handle
struct SWE2DSolver;

SWE2DSolver* swe2d_create(const SWE2DMesh& mesh, const double* h0, const double* hu0,
                           const double* hv0, const SWE2DSolverConfig& cfg);
SWE2DStepDiag swe2d_step(SWE2DSolver* s, double dt_request);
void swe2d_get_state(SWE2DSolver* s, double* h, double* hu, double* hv);
void swe2d_destroy(SWE2DSolver* s);
```

---

## 9. pybind11 Bindings (swe2d_bindings.cpp)

```python
# Exposed Python API (backwater_swe2d module)

backwater_swe2d.swe2d_gpu_available() -> bool
backwater_swe2d.swe2d_build_mesh(
    node_x, node_y, node_z,       # np.ndarray float64 (N,)
    cell_nodes,                   # np.ndarray int32   (M,3)
    bc_edge_type,                 # np.ndarray int32   (E,)
    bc_edge_value,                # np.ndarray float64 (E,)
) -> SWE2DMeshHandle

backwater_swe2d.swe2d_create_solver(
    mesh_handle,                  # SWE2DMeshHandle
    h0, hu0, hv0,                 # np.ndarray float64 (M,)
    g, n_mann, h_min,
    cfl, dt_max, dt_fixed,
    use_gpu, n_threads
) -> SWE2DSolverHandle

backwater_swe2d.swe2d_step(
    solver_handle,
    dt_request                    # float, seconds
) -> dict   # {dt, wet_cells, max_depth, min_depth, mass_total, gpu_active}

backwater_swe2d.swe2d_get_state(
    solver_handle
) -> (h, hu, hv)                  # np.ndarray float64 (M,) each

backwater_swe2d.swe2d_destroy(solver_handle) -> None
```

---

## 10. Python Bridge (swe2d_backend.py)

```python
class SWE2DBackend:
    """
    High-level Python interface to the native 2D SWE solver.
    Handles path selection (GPU/CPU), state tracking, and diagnostics.
    """

    def __init__(self, use_gpu: bool = True):
        ...

    def build_mesh(self, node_x, node_y, node_z, cell_nodes,
                   bc_edge_type, bc_edge_value) -> None:
        """Build mesh from numpy arrays. Must be called before run()."""
        ...

    def initialize(self, h0, hu0=None, hv0=None,
                   n_mann=0.035, g=9.81, cfl=0.45, dt_max=10.0) -> None:
        """Create solver handle with initial conditions."""
        ...

    def step(self, dt_request: float) -> dict:
        """Advance one timestep. Returns diagnostics dict."""
        ...

    def run(self, t_end: float, dt_request: float,
            progress_callback=None, cancel_check=None) -> list:
        """Run to t_end. Returns list of per-step diagnostic dicts."""
        ...

    def get_state(self) -> tuple:
        """Return (h, hu, hv) numpy arrays for current state."""
        ...

    def gpu_active(self) -> bool:
        """True if the last step ran on GPU."""
        ...

    def destroy(self) -> None:
        """Free native resources."""
        ...
```

---

## 11. Validation Test Suite

### Test 1: Lake at Rest (Still-Water Balance)
- Domain: Flat rectangle with sinusoidal bed (non-trivial zb).
- Initial condition: h + zb = const (flat water surface), hu = hv = 0.
- Expected: h, hu, hv unchanged to machine precision after N steps.
- Pass criterion: max(|dh|) < 1e-12 after 100 steps.

### Test 2: 1D Dam Break (Analytical Stoker)
- Domain: Long rectangle (1km × 50m) triangulated.
- IC: hL = 2.0m, hR = 0.5m, u = v = 0, flat bed.
- Run to t = 10s.
- Compare depth profile at centerline to Stoker analytical solution.
- Pass criterion: L∞ error < 2% of hL - hR.

### Test 3: Uniform Flow (Friction + Slope)
- Domain: Straight channel, 100m × 10m, slope S0 = 0.001, n = 0.030.
- IC: Normal depth from Manning's equation.
- Run to steady state.
- Expected: h ≈ y_n, u ≈ Q/A, mass conserved.
- Pass criterion: max velocity deviation < 1% of target, no mass drift.

### Test 4: GPU Parity (if CUDA available)
- Run dam-break test on both CPU and GPU paths.
- Compare final state arrays.
- Pass criterion: max(|h_gpu - h_cpu|) / max(h_cpu) < 1e-8.

---

## 12. CMake Integration

```cmake
# In CMakeLists.txt, after OpenMP block:

option(BACKWATER_USE_CUDA "Enable CUDA GPU path for 2D SWE solver" ON)

# 2D SWE sources (always compiled)
set(SWE2D_CPU_SOURCES
    cpp/src/swe2d_mesh.cpp
    cpp/src/swe2d_numerics.cpp
    cpp/src/swe2d_solver.cpp
    cpp/src/swe2d_bindings.cpp
)

if(BACKWATER_USE_CUDA)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(SWE2D_GPU_SOURCES cpp/src/swe2d_gpu.cu)
        add_compile_definitions(BACKWATER_HAS_CUDA=1)
        message(STATUS "CUDA found: ${CUDAToolkit_VERSION} — GPU path enabled")
    else()
        message(STATUS "CUDA not found — 2D solver will use CPU path only")
    endif()
endif()

pybind11_add_module(backwater_swe2d
    ${SWE2D_CPU_SOURCES}
    ${SWE2D_GPU_SOURCES}
)

if(CUDAToolkit_FOUND)
    target_link_libraries(backwater_swe2d PRIVATE CUDA::cudart)
    set_property(TARGET backwater_swe2d PROPERTY CUDA_SEPARABLE_COMPILATION ON)
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(backwater_swe2d PRIVATE OpenMP::OpenMP_CXX)
endif()
```

---

## 13. Implementation Sequence

### Slice 1: Mesh infrastructure (C0)
1. Write `swe2d_mesh.hpp` — SoA structs and BCType enum.
2. Write `swe2d_mesh.cpp` — edge connectivity builder.
3. Write `test_swe2d_mesh.py` — verify edge count, normal directions, BC classification.
4. Update CMakeLists.txt to compile new sources.

### Slice 2: CPU numerics and solver (C1)
1. Write `swe2d_numerics.hpp/cpp` — HLLC, bed slope, CFL.
2. Write `swe2d_solver.hpp/cpp` — CPU OpenMP loop.
3. Write `swe2d_bindings.cpp` — minimal pybind11 module.
4. Write `swe2d_backend.py` — Python wrapper.
5. Run lake-at-rest and dam-break tests.

### Slice 3: GPU path (C2)
1. Write `swe2d_gpu.cuh/cu` — flux, update, CFL kernels.
2. Wire CUDA path behind `BACKWATER_HAS_CUDA` guard in solver.
3. Write `test_swe2d_gpu.py` — parity vs CPU.
4. Rebuild with CUDA; run GPU parity test.

### Slice 4: Source terms + BCs (C3)
1. Add inflow, stage, open, reflecting BC ghost-cell logic.
2. Add Manning friction with semi-implicit limiter.
3. Run uniform-flow test.

### Slice 5: Plugin integration (Epic D)
1. Wire `SWE2DBackend` into plugin orchestration.
2. Add 2D run mode controls to UI.
3. Add GeoPackage persistence for 2D outputs.
