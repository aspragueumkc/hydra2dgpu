---
type: plan
status: complete
created: 2026-07-10
completed: 2026-07-25
---

# Advanced Spatial Reconstruction Schemes — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three new spatial reconstruction schemes (Barth-Jespersen limiter, true 3-sub-stencil WENO3, Suresh-Huynh MP5) to the SWE2D GPU solver, expanding from 7 to 9 schemes.

**Architecture:** Three new CUDA kernels (`barth_jespersen_kernel`, `weno3_kernel`, `mp5_kernel`) each compute face/cell reconstruction values into precomputed arrays before the flux kernel. The flux kernel's if/else dispatch reads from these arrays. Mesh assembly gains face-level sub-stencil CSR tables for WENO3 and 5-cell walk tables for MP5. Python layer adds enum entries, CFL clamping, and CLI validation.

**Tech Stack:** Python 3.11+ (swe2d package), CUDA C++17 (cpp/src/), pybind11 (bindings), pytest (tests), CMake (build), QGIS/QML (GUI)

**Agent assignments:** All implementation tasks use `_opencode-go_deepseek-v4-flash` variant agents. Review tasks use `_opencode-go_deepseek-v4-pro` variants.

- `python-pro_opencode-go_deepseek-v4-flash` — Python enum, backend, CLI, mesh models
- `cpp-pro_opencode-go_deepseek-v4-flash` — CUDA kernels, C++ enums, mesh assembly, bindings
- `test-automator_opencode-go_deepseek-v4-flash` — all test creation
- `build-engineer_opencode-go_deepseek-v4-flash` — CMakeLists.txt changes

---

## Parallel Execution Strategy

Tasks are organized into **waves** for maximum parallelism. Within a wave, all agents can run concurrently — each touches independent files. Waves are sequential: a downstream wave starts only after the previous wave completes and any merge conflicts on shared files are resolved.

### Shared File Conflicts

| File | Conflicting Tasks | Resolution |
|------|:-:|------|
| `cpp/src/swe2d_reconstruct.cu` | 5, 9, 13 | **Wave 3** — single `cpp-pro_flash` agent writes all three kernels into one file |
| `cpp/src/swe2d_gpu.cu` + `swe2d_gpu.cuh` | 6, 10, 14 | **Wave 4** — single `cpp-pro_flash` agent integrates all three dispatch paths |
| `cpp/src/swe2d_mesh.cpp` | 8, 12 | **Wave 2** — single `cpp-pro_flash` agent builds both stencil tables |

### Wave 1: Foundation (4 parallel agents)

All touch independent files. Zero conflicts. **4 agents dispatched simultaneously.**

| Agent | Tasks | Files |
|-------|-------|-------|
| `python-pro_flash` | 1 — Python enum | `swe2d/extensions/extension_models.py` |
| `cpp-pro_flash` | 2 — C++ enum | `cpp/src/swe2d_solver.hpp` |
| `build-engineer_flash` | 3 — CMake + stub file | `CMakeLists.txt`, `cpp/src/swe2d_reconstruct.cu` (stub) |
| `cpp-pro_flash` | 4 — Stencil fields (both layers) | `cpp/src/swe2d_mesh.hpp`, `swe2d/mesh/mesh_models.py` |

### Wave 2: Mesh Assembly (1 agent)

Both stencil table builders modify `swe2d_mesh.cpp`. Single agent avoids merge conflicts.

| Agent | Tasks | Files |
|-------|-------|-------|
| `cpp-pro_flash` | 8 + 12 — WENO3 sub-stencils + MP5 5-cell walk | `cpp/src/swe2d_mesh.cpp` |

### Wave 3: Kernels + Bindings (3 parallel agents)

| Agent | Tasks | Files |
|-------|-------|-------|
| `cpp-pro_flash` | 5 + 9 + 13 — All three kernels in one file | `cpp/src/swe2d_reconstruct.cu`, `cpp/src/swe2d_gpu.cuh` (declarations) |
| `cpp-pro_flash` | 7 — Pybind11 stencil exposure | `cpp/src/swe2d_bindings.cpp` |
| `python-pro_flash` | 11 — Scheme migration Python | `swe2d/runtime/backend.py`, `swe2d/cli/headless_runner.py`, `swe2d/cli/batch_runner.py` |

### Wave 4: GPU Dispatch Integration (1 agent)

All three dispatch paths modify the same sections of `swe2d_gpu.cu` and `swe2d_gpu.cuh`. Single agent integrates them cleanly.

| Agent | Tasks | Files |
|-------|-------|-------|
| `cpp-pro_flash` | 6 + 10 + 14 — Barth-Jespersen + WENO3 + MP5 dispatch | `cpp/src/swe2d_gpu.cu`, `cpp/src/swe2d_gpu.cuh` |

### Wave 5: Python Integration (3 parallel agents)

All touch independent files.

| Agent | Tasks | Files |
|-------|-------|-------|
| `python-pro_flash` | 15 — CFL enforcement | `swe2d/runtime/backend.py` |
| `python-pro_flash` | 19 — QML form combo | `QML/form_init.py` |
| `python-pro_flash` | 20 — Documentation updates | `docs/SOLVER_ORDER_AND_STENCIL.md`, `docs/USER_GUIDE.md`, `docs/INDEX.md`, `CHANGELOG.md` |

### Wave 6: Tests (5 parallel agents)

All create new test files — zero file conflicts.

| Agent | Tasks | Files |
|-------|-------|-------|
| `test-automator_flash` | 16a — Barth-Jespersen convergence | `tests/test_swe2d_barth_jespersen_convergence.py` (NEW) |
| `test-automator_flash` | 16b — WENO3 convergence | `tests/test_swe2d_weno3_convergence.py` (NEW) |
| `test-automator_flash` | 16c — MP5 convergence | `tests/test_swe2d_mp5_convergence.py` (NEW) |
| `test-automator_flash` | 17 — Robustness + monotonicity | `tests/test_swe2d_poor_mesh_robustness.py` (NEW), `tests/test_face_value_monotonicity.py` (NEW) |
| `test-automator_flash` | 18 — Performance benchmark | `tests/test_spatial_scheme_perf.py` (NEW) |

### Wave 7: Review (2 parallel agents)

| Agent | Tasks | Files |
|-------|-------|-------|
| `cpp-pro_pro` | 22 — C++ code review | All C++/CUDA files modified in Waves 1–4 |
| `python-pro_pro` | 21 + 23 — Full test suite run + Python review | All Python files, full `pytest` invocation |

### Summary

| Wave | Parallel Agents | Est. Wall Time | What |
|-----:|:-:|:-:|------|
| 1 | 4 | 15m | Foundation: enums, CMake, stencil fields |
| 2 | 1 | 1h | Mesh assembly: both stencil table builders |
| 3 | 3 | 1.5h | Kernels + bindings + Python migration |
| 4 | 1 | 1h | GPU dispatch integration |
| 5 | 3 | 45m | Python: CFL, GUI, docs |
| 6 | 5 | 1h | Tests: all 5 test files |
| 7 | 2 | 1h | Review + regression |
| **Total** | **19 agent-dispatch events across 7 waves** | **~6h wall vs ~17h sequential** | |

---

## Phase 0: Foundation — Enums, Build System, Mesh Stubs

### Task 1: Add new enum members (Python)

**Agent:** `python-pro_opencode-go_deepseek-v4-flash`

**Files:**
- Modify: `swe2d/extensions/extension_models.py`

- [ ] **Step 1: Read current enum**

Read `swe2d/extensions/extension_models.py` lines 17-34.

- [ ] **Step 2: Add new enum members**

```python
class SpatialDiscretization(IntEnum):
    FV_FIRST_ORDER    = 0   # First-order piecewise-constant
    FV_MUSCL_FAST     = 1   # MUSCL reconstruction with Superbee limiter
    FV_MUSCL_MINMOD   = 2   # MUSCL reconstruction with MinMod limiter
    FV_MUSCL_MC       = 3   # MUSCL reconstruction with MC limiter
    FV_MUSCL_VAN_LEER = 4   # MUSCL reconstruction with Van Leer limiter
    FV_BARTH_JESPERSEN = 5  # LSQ gradient + Barth-Jespersen limiter
    FV_WENO3          = 6   # True 3-sub-stencil WENO (1-ring, 3rd-order)
    FV_WENO5          = 7   # 5-sub-stencil WENO via 2-ring LSQ (was 6)
    FV_MP5            = 8   # Suresh-Huynh Mapped Monotonicity-Preserving

    # Backward-compat aliases (unchanged):
    FV_MUSCL = FV_MUSCL_FAST
    FV_WENO  = FV_MUSCL_MINMOD
```

- [ ] **Step 3: Verify Python import**

Run: `python -c "from swe2d.extensions.extension_models import SpatialDiscretization; print(list(SpatialDiscretization))"`
Expected: prints all 9 members with values 0-8.

- [ ] **Step 4: Commit**

```bash
git add swe2d/extensions/extension_models.py
git commit -m "feat: add FV_BARTH_JESPERSEN(5), FV_WENO3(6), FV_WENO5(7), FV_MP5(8) enum entries"
```

---

### Task 2: Add new enum members (C++)

**Agent:** `cpp-pro_opencode-go_deepseek-v4-flash`

**Files:**
- Modify: `cpp/src/swe2d_solver.hpp`

- [ ] **Step 1: Read current enum**

Read `cpp/src/swe2d_solver.hpp` to find `SWE2DSpatialScheme`.

- [ ] **Step 2: Add new enum members**

```cpp
enum class SWE2DSpatialScheme : int {
    FV_FIRST_ORDER    = 0,
    FV_MUSCL_FAST     = 1,
    FV_MUSCL_MINMOD   = 2,
    FV_MUSCL_MC       = 3,
    FV_MUSCL_VAN_LEER = 4,
    FV_BARTH_JESPERSEN = 5,
    FV_WENO3          = 6,
    FV_WENO5          = 7,
    FV_MP5            = 8,
};
```

- [ ] **Step 3: Build to verify compilation**

Run: `cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)`
Expected: build succeeds with no new errors.

- [ ] **Step 4: Commit**

```bash
git add cpp/src/swe2d_solver.hpp
git commit -m "feat: add C++ SWE2DSpatialScheme enum entries for new schemes"
```

---

### Task 3: Add new GPU source to CMake

**Agent:** `build-engineer_opencode-go_deepseek-v4-flash`

**Files:**
- Modify: `CMakeLists.txt`
- Create: `cpp/src/swe2d_reconstruct.cu` (stub)

- [ ] **Step 1: Find GPU source list in CMakeLists.txt**

Read `CMakeLists.txt` to locate the `SWE2D_GPU_SOURCES` list.

- [ ] **Step 2: Add new source to list**

Add after `cpp/src/swe2d_gpu_redistribute.cu`:
```cmake
list(APPEND SWE2D_GPU_SOURCES cpp/src/swe2d_reconstruct.cu)
```

- [ ] **Step 3: Create stub kernel file**

```cuda
// cpp/src/swe2d_reconstruct.cu
#include "swe2d_gpu.cuh"
#include "swe2d_mesh.hpp"
#include "swe2d_solver.hpp"
#include "swe2d_units.cuh"
#include <cuda_runtime.h>

// Stub — kernels added in later tasks
```

- [ ] **Step 4: Verify build with stub**

Run: `cd build && cmake .. && make -j$(nproc)`
Expected: build succeeds, new `.cu` compiled in.

- [ ] **Step 5: Commit**

```bash
git add CMakeLists.txt cpp/src/swe2d_reconstruct.cu
git commit -m "build: add swe2d_reconstruct.cu to GPU sources"
```

---

### Task 4: Add mesh assembly stencil fields (Python + C++ headers)

**Agent:** `cpp-pro_opencode-go_deepseek-v4-flash`

**Files:**
- Modify: `cpp/src/swe2d_mesh.hpp`
- Modify: `swe2d/mesh/mesh_models.py`
- Create: `cpp/src/swe2d_mesh.cpp` (stencil builder stubs if needed) — actual logic in Task 9 and Task 16

- [ ] **Step 1: Read C++ mesh struct**

Read `cpp/src/swe2d_mesh.hpp` to find `SWE2DMesh`.

- [ ] **Step 2: Add stencil fields to C++ struct**

```cpp
// In SWE2DMesh struct, after cell_ring2_* fields:

// WENO3 face sub-stencil (scheme 6)
std::vector<int> face_stencil_S0_offsets;  // [n_faces + 1]
std::vector<int> face_stencil_S0_cells;    // variable-length
std::vector<int> face_stencil_S1;          // [2 * n_faces] = {owner, neighbor}
std::vector<int> face_stencil_S2_offsets;  // [n_faces + 1]
std::vector<int> face_stencil_S2_cells;    // variable-length

// MP5 5-cell walk (scheme 8)
std::vector<int> face_stencil_5;           // [5 * n_faces] = {u2, u1, u, v, v1}
std::vector<int> face_mp5_case;            // [n_faces] ∈ {1,2,3,4}

// Convenience
bool has_stencil_data() const { return !face_stencil_S0_offsets.empty(); }
```

- [ ] **Step 3: Read Python MeshResult dataclass**

Read `swe2d/mesh/mesh_models.py` to find `MeshResult`.

- [ ] **Step 4: Add stencil fields to Python MeshResult**

```python
@dataclass
class MeshResult:
    node_x: np.ndarray
    node_y: np.ndarray
    node_z: np.ndarray
    cell_nodes: np.ndarray
    cell_face_offsets: np.ndarray
    cell_face_nodes: np.ndarray
    cell_type: np.ndarray
    region_id: np.ndarray
    target_size: np.ndarray
    quality_summary: Optional[Dict[str, object]] = None
    # WENO3 face sub-stencils (scheme 6):
    face_stencil_S0_offsets: Optional[np.ndarray] = None
    face_stencil_S0_cells: Optional[np.ndarray] = None
    face_stencil_S1: Optional[np.ndarray] = None
    face_stencil_S2_offsets: Optional[np.ndarray] = None
    face_stencil_S2_cells: Optional[np.ndarray] = None
    # MP5 5-cell walk (scheme 8):
    face_stencil_5: Optional[np.ndarray] = None
    face_mp5_case: Optional[np.ndarray] = None
```

- [ ] **Step 5: Verify build**

Run: `cd build && cmake .. && make -j$(nproc)`
Expected: build succeeds (stencil fields unused for now).

- [ ] **Step 6: Commit**

```bash
git add cpp/src/swe2d_mesh.hpp swe2d/mesh/mesh_models.py
git commit -m "feat: add stencil data fields to SWE2DMesh and MeshResult for WENO3 and MP5"
```

---

## Phase 1: Scheme 5 — Barth-Jespersen Gradient Limiter

### Task 5: Implement `barth_jespersen_kernel`

**Agent:** `cpp-pro_opencode-go_deepseek-v4-flash`

**Files:**
- Modify: `cpp/src/swe2d_reconstruct.cu`
- Modify: `cpp/src/swe2d_gpu.cuh`

- [ ] **Step 1: Declare kernel in header**

In `cpp/src/swe2d_gpu.cuh`, add after existing kernel declarations:

```cuda
#include <cuda_runtime.h>

template <typename T>
__global__ void barth_jespersen_kernel(
    const T* __restrict__ q,
    const T* __restrict__ grad_x,
    const T* __restrict__ grad_y,
    const T* __restrict__ cell_cx,
    const T* __restrict__ cell_cy,
    const int* __restrict__ cell_face_offsets,
    const int* __restrict__ cell_face_nodes,
    int n_cells,
    T* __restrict__ grad_x_lim,
    T* __restrict__ grad_y_lim);
```

- [ ] **Step 2: Implement kernel in swe2d_reconstruct.cu**

```cuda
// cpp/src/swe2d_reconstruct.cu — Barth-Jespersen kernel

template <typename T>
__global__ void barth_jespersen_kernel(
    const T* __restrict__ q,
    const T* __restrict__ grad_x,
    const T* __restrict__ grad_y,
    const T* __restrict__ cell_cx,
    const T* __restrict__ cell_cy,
    const int* __restrict__ cell_face_offsets,
    const int* __restrict__ cell_face_nodes,
    int n_cells,
    T* __restrict__ grad_x_lim,
    T* __restrict__ grad_y_lim)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_cells) return;

    T qi  = q[i];
    T gx  = grad_x[i];
    T gy  = grad_y[i];
    T xi  = cell_cx[i];
    T yi  = cell_cy[i];

    T chi = T(1.0);
    int start = cell_face_offsets[i];
    int end   = cell_face_offsets[i + 1];

    for (int k = start; k < end; ++k) {
        int j = cell_face_nodes[k];
        T qj = q[j];
        T xj = cell_cx[j];
        T yj = cell_cy[j];

        T dx = xj - xi;
        T dy = yj - yi;
        T q_face = qi + gx * dx + gy * dy;

        T q_min = fmin(qi, qj);
        T q_max = fmax(qi, qj);

        T chi_k = T(1.0);
        if (q_face > q_max) {
            T denom = q_face - qi;
            if (fabs(denom) > T(1e-15)) {
                chi_k = (q_max - qi) / denom;
            }
        } else if (q_face < q_min) {
            T denom = qi - q_face;
            if (fabs(denom) > T(1e-15)) {
                chi_k = (qi - q_min) / denom;
            }
        }
        chi = fmin(chi, chi_k);
    }

    chi = fmax(T(0.0), fmin(T(1.0), chi));
    grad_x_lim[i] = chi * gx;
    grad_y_lim[i] = chi * gy;
}

// Explicit template instantiation
template __global__ void barth_jespersen_kernel<float>(
    const float*, const float*, const float*, const float*, const float*,
    const int*, const int*, int, float*, float*);
template __global__ void barth_jespersen_kernel<double>(
    const double*, const double*, const double*, const double*, const double*,
    const int*, const int*, int, double*, double*);
```

- [ ] **Step 3: Build**

Run: `cd build && make -j$(nproc)`
Expected: build succeeds.

- [ ] **Step 4: Commit**

```bash
git add cpp/src/swe2d_reconstruct.cu cpp/src/swe2d_gpu.cuh
git commit -m "feat: implement barth_jespersen_kernel for scheme 5"
```

---

### Task 6: Add launch orchestration for scheme 5 in GPU pipeline

**Agent:** `cpp-pro_opencode-go_deepseek-v4-flash`

**Files:**
- Modify: `cpp/src/swe2d_gpu.cu`

- [ ] **Step 1: Read current flux kernel dispatch**

Read `cpp/src/swe2d_gpu.cu` around lines 1980-2140 to understand the `tvd_reconstruct` and `weno5_reconstruct` lambdas.

- [ ] **Step 2: Add launch of barth_jespersen_kernel before flux kernel**

In the function that launches kernels (the one containing `swe2d_flux_kernel` call), add before the flux kernel launch:

```cuda
// Barth-Jespersen limited gradient (scheme 5)
const int scheme_barth_jespersen = static_cast<int>(SWE2DSpatialScheme::FV_BARTH_JESPERSEN);
int block = 256;
int grid = (n_cells + block - 1) / block;

if (spatial_scheme == scheme_barth_jespersen && !dry_cells_only) {
    barth_jespersen_kernel<T>
        <<<grid, block, 0, stream>>>
        (dev.q, dev.grad_x, dev.grad_y,
         dev.cell_cx, dev.cell_cy,
         mesh.cell_face_offsets_d, mesh.cell_face_nodes_d,
         n_cells,
         dev.grad_x_lim, dev.grad_y_lim);
    cudaCheck(cudaGetLastError());
}
```

Note: `dev.grad_x_lim` and `dev.grad_y_lim` are new device arrays allocated in solver init. The existing gradient computation (Green-Gauss or LSQ) must also run for scheme 5 — it currently only runs for schemes 1-4 and 6. Update the guard condition on the existing LSQ/gradient kernel launch to include scheme 5.

- [ ] **Step 3: Modify flux kernel to use limited gradient for scheme 5**

In `tvd_reconstruct` lambda inside `swe2d_flux_kernel`, update the gradient source:

```cuda
// Original:
// T gx_i = grad_x[i], gy_i = grad_y[i];

// New:
T gx_i, gy_i;
if (spatial_scheme == scheme_barth_jespersen) {
    gx_i = grad_x_lim[i];
    gy_i = grad_y_lim[i];
} else {
    gx_i = grad_x[i];
    gy_i = grad_y[i];
}
```

- [ ] **Step 4: Allocate grad_x_lim/grad_y_lim in device state**

Read `cpp/src/swe2d_gpu.cuh` for `SWE2DDeviceState`. Add two new fields and allocation.

- [ ] **Step 5: Build and verify**

Run: `cd build && make -j$(nproc)`
Expected: build succeeds.

- [ ] **Step 6: Commit**

```bash
git add cpp/src/swe2d_gpu.cu cpp/src/swe2d_gpu.cuh
git commit -m "feat: add Barth-Jespersen gradient limiter dispatch in flux pipeline"
```

---

### Task 7: Expose stencil data from C++ to Python bindings

**Agent:** `cpp-pro_opencode-go_deepseek-v4-flash`

**Files:**
- Modify: `cpp/src/swe2d_bindings.cpp`

- [ ] **Step 1: Read current binding code for MeshResult**

Read `cpp/src/swe2d_bindings.cpp` to find `PyMesh` or similar class that exposes mesh properties to Python.

- [ ] **Step 2: Add property bindings for new stencil arrays**

Add `def_property_readonly` (or equivalent) for each new mesh field. Each should return numpy arrays when available, or numpy empty if no stencil data was built:

```cpp
.def_property_readonly("face_stencil_S0_offsets", [](const PyMesh& self) -> py::array_t<int> {
    if (self.mesh.face_stencil_S0_offsets.empty())
        return py::array_t<int>(0);
    return py::array_t<int>(
        {static_cast<py::ssize_t>(self.mesh.face_stencil_S0_offsets.size())},
        {sizeof(int)},
        self.mesh.face_stencil_S0_offsets.data()
    );
})
// ... repeat for S0_cells, S1, S2_offsets, S2_cells, face_stencil_5, face_mp5_case
```

- [ ] **Step 3: Build**

Run: `cd build && make -j$(nproc)`
Expected: build succeeds.

- [ ] **Step 4: Verify Python can access stencil arrays**

Run: `python -c "from swe2d.mesh.mesh_models import MeshResult; print([f.name for f in MeshResult.__dataclass_fields__.values()])"`
Expected: prints list including new stencil field names.

- [ ] **Step 5: Commit**

```bash
git add cpp/src/swe2d_bindings.cpp
git commit -m "feat: expose WENO3 and MP5 stencil data via pybind11"
```

---

## Phase 2: Mesh Assembly — WENO3 Sub-Stencils

### Task 8: Build `face_stencil_S0` and `face_stencil_S2` tables

**Agent:** `cpp-pro_opencode-go_deepseek-v4-flash`

**Files:**
- Modify: `cpp/src/swe2d_mesh.cpp`

- [ ] **Step 1: Read mesh builder code**

Read `cpp/src/swe2d_mesh.cpp` to find where `cell_face_offsets` / `cell_face_nodes` CSR is built and where face neighbor pairs are stored.

- [ ] **Step 2: Implement sub-stencil builder**

After face construction loop, add:

```cpp
void build_face_substencil_tables(SWE2DMesh& mesh) {
    int n_faces = mesh.n_faces;
    int n_cells = mesh.n_cells;

    mesh.face_stencil_S0_offsets.resize(n_faces + 1);
    mesh.face_stencil_S1.resize(2 * n_faces);
    mesh.face_stencil_S2_offsets.resize(n_faces + 1);

    std::vector<int> S0_cells, S2_cells;

    // face_owners[f] = {i, j} where i=owner, j=neighbor (or -1 for boundary)
    // Assume face_owners is available from mesh construction.
    for (int f = 0; f < n_faces; ++f) {
        int i = face_owners[2 * f + 0];
        int j = face_owners[2 * f + 1];

        mesh.face_stencil_S1[2 * f + 0] = i;
        mesh.face_stencil_S1[2 * f + 1] = j;

        mesh.face_stencil_S0_offsets[f] = static_cast<int>(S0_cells.size());
        if (i >= 0) {
            int start = mesh.cell_face_offsets[i];
            int end   = mesh.cell_face_offsets[i + 1];
            for (int k = start; k < end; ++k) {
                int neighbor = mesh.cell_face_nodes[k];
                if (neighbor != j) {
                    S0_cells.push_back(neighbor);
                }
            }
        }

        mesh.face_stencil_S2_offsets[f] = static_cast<int>(S2_cells.size());
        if (j >= 0) {
            int start = mesh.cell_face_offsets[j];
            int end   = mesh.cell_face_offsets[j + 1];
            for (int k = start; k < end; ++k) {
                int neighbor = mesh.cell_face_nodes[k];
                if (neighbor != i) {
                    S2_cells.push_back(neighbor);
                }
            }
        }
    }

    mesh.face_stencil_S0_offsets[n_faces] = static_cast<int>(S0_cells.size());
    mesh.face_stencil_S2_offsets[n_faces] = static_cast<int>(S2_cells.size());

    mesh.face_stencil_S0_cells = std::move(S0_cells);
    mesh.face_stencil_S2_cells = std::move(S2_cells);
}
```

Call this at the end of the mesh assembly function (after face_owners is populated).

- [ ] **Step 3: Build**

Run: `cd build && make -j$(nproc)`
Expected: build succeeds.

- [ ] **Step 4: Verify tables are populated**

Run a small test mesh through the builder (use existing test infrastructure). Verify `face_stencil_S0_offsets` has `n_faces + 1` entries and `face_stencil_S0_cells` is non-empty for interior faces.

- [ ] **Step 5: Commit**

```bash
git add cpp/src/swe2d_mesh.cpp
git commit -m "feat: build WENO3 face sub-stencil tables during mesh assembly"
```

---

## Phase 3: Scheme 6 — True WENO3

### Task 9: Implement `weno3_kernel`

**Agent:** `cpp-pro_opencode-go_deepseek-v4-flash`

**Files:**
- Modify: `cpp/src/swe2d_reconstruct.cu`

- [ ] **Step 1: Implement device helper `lsq2d_evaluate`**

```cuda
// In swe2d_reconstruct.cu, before weno3_kernel:

template <typename T>
__device__ T lsq2d_evaluate(
    const T* __restrict__ q,
    const T* __restrict__ cx,
    const T* __restrict__ cy,
    int n,
    const int* __restrict__ cell_ids,
    T xf, T yf)
{
    // Solve 2x2 LSQ: A^T A x = A^T b
    // Plane: q(k) ≈ a + bx*cx[k] + by*cy[k]
    // A = [1 cx[k] cy[k]], b = [q[k]]
    // Normal equations (3x3) solved analytically for small n.
    T sum1=0, sum_x=0, sum_y=0, sum_xx=0, sum_xy=0, sum_yy=0;
    T sum_q=0, sum_qx=0, sum_qy=0;

    for (int k = 0; k < n; ++k) {
        int id = cell_ids[k];
        T xk = cx[id], yk = cy[id], qk = q[id];
        sum1 += T(1); sum_x += xk; sum_y += yk;
        sum_xx += xk*xk; sum_xy += xk*yk; sum_yy += yk*yk;
        sum_q += qk; sum_qx += qk*xk; sum_qy += qk*yk;
    }

    // Solve 3x3 via Cramer's rule (hardcoded for constant term)
    T a, bx, by;
    if (n < 3) {
        // Degenerate: use mean
        a = sum_q / sum1;
        bx = T(0); by = T(0);
    } else {
        T detA = sum1*(sum_xx*sum_yy - sum_xy*sum_xy)
               - sum_x*(sum_x*sum_yy - sum_xy*sum_y)
               + sum_y*(sum_x*sum_xy - sum_xx*sum_y);
        if (fabs(detA) < T(1e-12)) {
            a = sum_q / sum1; bx = T(0); by = T(0);
        } else {
            T detA_inv = T(1) / detA;
            T M11 = sum_xx*sum_yy - sum_xy*sum_xy;
            T M12 = sum_y*sum_xy - sum_x*sum_yy;
            T M13 = sum_x*sum_xy - sum_y*sum_xx;
            T M22 = sum1*sum_yy - sum_y*sum_y;
            T M23 = sum_x*sum_y - sum1*sum_xy;
            T M33 = sum1*sum_xx - sum_x*sum_x;

            a  = detA_inv * (M11*sum_q + M12*sum_qx + M13*sum_qy);
            bx = detA_inv * (M12*sum_q + M22*sum_qx + M23*sum_qy);
            by = detA_inv * (M13*sum_q + M23*sum_qx + M33*sum_qy);
        }
    }

    return a + bx*xf + by*yf;
}

template <typename T>
__device__ T lsq2d_residual(
    const T* __restrict__ q,
    const T* __restrict__ cx,
    const T* __restrict__ cy,
    int n,
    const int* __restrict__ cell_ids)
{
    // Sum of squared deviations from mean (simple smoothness indicator)
    T mean = T(0);
    for (int k = 0; k < n; ++k) {
        mean += q[cell_ids[k]];
    }
    mean /= T(n > 0 ? n : 1);
    T res = T(0);
    for (int k = 0; k < n; ++k) {
        T diff = q[cell_ids[k]] - mean;
        res += diff * diff;
    }
    return res;
}
```

- [ ] **Step 2: Implement weno3_kernel**

```cuda
template <typename T>
__global__ void weno3_kernel(
    const T* __restrict__ q,
    const T* __restrict__ cell_cx,
    const T* __restrict__ cell_cy,
    const T* __restrict__ face_mid_x,
    const T* __restrict__ face_mid_y,
    const int* __restrict__ face_stencil_S0_offsets,
    const int* __restrict__ face_stencil_S0_cells,
    const int* __restrict__ face_stencil_S1,
    const int* __restrict__ face_stencil_S2_offsets,
    const int* __restrict__ face_stencil_S2_cells,
    int n_faces,
    T* __restrict__ q_face_recon)
{
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= n_faces) return;

    T xf = face_mid_x[f];
    T yf = face_mid_y[f];

    int i = face_stencil_S1[2 * f + 0];
    int j = face_stencil_S1[2 * f + 1];

    T q_cand[3];
    T beta[3] = {T(0), T(0), T(0)};

    // S0: upwind lobe
    {
        int s0 = face_stencil_S0_offsets[f];
        int e0 = face_stencil_S0_offsets[f + 1];
        int n0 = e0 - s0;
        if (n0 > 1) {
            q_cand[0] = lsq2d_evaluate<T>(q, cell_cx, cell_cy, n0,
                                          face_stencil_S0_cells + s0, xf, yf);
            beta[0] = lsq2d_residual<T>(q, cell_cx, cell_cy, n0,
                                        face_stencil_S0_cells + s0);
        } else {
            q_cand[0] = q[i];
            beta[0] = T(1e6);  // large weight → suppress
        }
    }

    // S1: central pair
    {
        T qi = q[i], qj = q[j];
        T xi = cell_cx[i], yi = cell_cy[i];
        T xj = cell_cx[j], yj = cell_cy[j];
        T dist_face = sqrt((xf - xi)*(xf - xi) + (yf - yi)*(yf - yi));
        T dist_ij   = sqrt((xj - xi)*(xj - xi) + (yj - yi)*(yj - yi));
        T t = (dist_ij > T(1e-12)) ? (dist_face / dist_ij) : T(0.5);
        q_cand[1] = qi + t * (qj - qi);
        beta[1] = (qi - qj) * (qi - qj);
    }

    // S2: downwind lobe
    {
        int s2 = face_stencil_S2_offsets[f];
        int e2 = face_stencil_S2_offsets[f + 1];
        int n2 = e2 - s2;
        if (n2 > 1) {
            q_cand[2] = lsq2d_evaluate<T>(q, cell_cx, cell_cy, n2,
                                          face_stencil_S2_cells + s2, xf, yf);
            beta[2] = lsq2d_residual<T>(q, cell_cx, cell_cy, n2,
                                        face_stencil_S2_cells + s2);
        } else {
            q_cand[2] = q[j];
            beta[2] = T(1e6);
        }
    }

    // Nonlinear weights (Hu-Shu 1999)
    T d_weights[3] = {T(0.1), T(0.6), T(0.3)};
    T eps = T(1e-6);
    T alpha[3];
    T alpha_sum = T(0);
    for (int k = 0; k < 3; ++k) {
        alpha[k] = d_weights[k] / ((eps + beta[k]) * (eps + beta[k]));
        alpha_sum += alpha[k];
    }

    T q_recon = T(0);
    for (int k = 0; k < 3; ++k) {
        q_recon += (alpha[k] / alpha_sum) * q_cand[k];
    }
    q_face_recon[f] = q_recon;
}

template __global__ void weno3_kernel<float>(
    const float*, const float*, const float*, const float*, const float*,
    const int*, const int*, const int*, const int*, const int*,
    int, float*);
template __global__ void weno3_kernel<double>(
    const double*, const double*, const double*, const double*, const double*,
    const int*, const int*, const int*, const int*, const int*,
    int, double*);
```

- [ ] **Step 3: Build**

Run: `cd build && make -j$(nproc)`
Expected: build succeeds.

- [ ] **Step 4: Commit**

```bash
git add cpp/src/swe2d_reconstruct.cu
git commit -m "feat: implement weno3_kernel with device LSQ helpers"
```

---

### Task 10: Add WENO3 dispatch in GPU pipeline

**Agent:** `cpp-pro_opencode-go_deepseek-v4-flash`

**Files:**
- Modify: `cpp/src/swe2d_gpu.cu`

- [ ] **Step 1: Add launch before flux kernel**

```cuda
const int scheme_weno3 = static_cast<int>(SWE2DSpatialScheme::FV_WENO3);
int face_block = 256;
int face_grid = (n_faces + face_block - 1) / face_block;

if (spatial_scheme == scheme_weno3) {
    weno3_kernel<T>
        <<<face_grid, face_block, 0, stream>>>
        (dev.q, dev.cell_cx, dev.cell_cy,
         dev.face_mid_x, dev.face_mid_y,
         mesh.face_stencil_S0_offsets_d, mesh.face_stencil_S0_cells_d,
         mesh.face_stencil_S1_d,
         mesh.face_stencil_S2_offsets_d, mesh.face_stencil_S2_cells_d,
         n_faces,
         dev.weno3_face_recon);
    cudaCheck(cudaGetLastError());
}
```

- [ ] **Step 2: Modify flux kernel dispatch**

Update the existing if/else in `swe2d_flux_kernel`:

```cuda
if (spatial_scheme == scheme_weno3) {
    // Read precomputed WENO3 face value
    q_face_L = dev.weno3_face_recon[face_id];
} else if (spatial_scheme == scheme_weno5 && !near_boundary) {
    weno5_reconstruct(...)
} else if (spatial_scheme >= scheme_fast && spatial_scheme <= scheme_vl) {
    tvd_reconstruct(...)
}
```

- [ ] **Step 3: Add device arrays to solver state**

In `swe2d_gpu.cuh`, add `weno3_face_recon` to `SWE2DDeviceState`. Allocate in init, free in cleanup.

- [ ] **Step 4: Build**

Run: `cd build && make -j$(nproc)`
Expected: build succeeds.

- [ ] **Step 5: Commit**

```bash
git add cpp/src/swe2d_gpu.cu cpp/src/swe2d_gpu.cuh
git commit -m "feat: add WENO3 dispatch in GPU flux pipeline"
```

---

### Task 11: Scheme 6 → 7 renumber migration (Python)

**Agent:** `python-pro_opencode-go_deepseek-v4-flash`

**Files:**
- Modify: `swe2d/cli/headless_runner.py`
- Modify: `swe2d/cli/batch_runner.py`
- Modify: `swe2d/runtime/backend.py`

- [ ] **Step 1: Add scheme validation and migration warning in backend.py**

```python
# In swe2d/runtime/backend.py

_SCHEME_MIGRATION_MAP = {
    # old=6 (was WENO5) → new=7 (WENO5); new 6 = WENO3
    6: (7, "spatial_scheme=6 was FV_WENO5; now it is FV_WENO3 (true 3-sub-stencil). "
            "To keep WENO5, use spatial_scheme=7."),
}

def _migrate_scheme(scheme: int) -> tuple[int, str | None]:
    """Migrate old scheme numbers. Returns (new_scheme, warning_message)."""
    if scheme in _SCHEME_MIGRATION_MAP:
        new_scheme, warning = _SCHEME_MIGRATION_MAP[scheme]
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(warning)
        return new_scheme, warning
    return scheme, None
```

- [ ] **Step 2: Call migration in backend.initialize()**

```python
def initialize(self, ...):
    scheme, warning = _migrate_scheme(self._spatial_scheme)
    if warning:
        import sys
        print(f"WARNING: {warning}", file=sys.stderr)
    self._spatial_scheme = scheme
    # ... rest of init
```

- [ ] **Step 3: Add validate_scheme() in batch_runner.py**

In `swe2d/cli/batch_runner.py`:

```python
_VALID_SCHEMES = frozenset(range(9))

def validate_scheme(scheme: int) -> int:
    """Validate and potentially migrate scheme number. Returns valid scheme or raises."""
    if scheme not in _VALID_SCHEMES:
        raise ValueError(
            f"Invalid spatial_scheme={scheme}. Must be one of {sorted(_VALID_SCHEMES)}."
        )
    if scheme == 6:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(
            "spatial_scheme=6 was FV_WENO5; now it is FV_WENO3 (true 3-sub-stencil). "
            "To keep WENO5, use spatial_scheme=7."
        )
    return scheme
```

- [ ] **Step 4: Update headless_runner.py argument parsing**

In `swe2d/cli/headless_runner.py`, update the argparse `--spatial-scheme` choices from `range(7)` to `range(9)`:

```python
parser.add_argument(
    "--spatial-scheme", type=int, default=0,
    choices=range(9),
    help="Spatial reconstruction scheme (0-8)"
)
```

- [ ] **Step 5: Commit**

```bash
git add swe2d/runtime/backend.py swe2d/cli/batch_runner.py swe2d/cli/headless_runner.py
git commit -m "feat: add scheme 6→7 migration, validation, and CLI range update"
```

---

## Phase 4: Scheme 8 — MP5

### Task 12: Build `face_stencil_5` table

**Agent:** `cpp-pro_opencode-go_deepseek-v4-flash`

**Files:**
- Modify: `cpp/src/swe2d_mesh.cpp`

- [ ] **Step 1: Read mesh builder for graph walk context**

Read `cpp/src/swe2d_mesh.cpp` to understand the cell connectivity available during assembly.

- [ ] **Step 2: Implement 5-cell walk builder**

After the WENO3 sub-stencil builder, add:

```cpp
void build_face_stencil_5_table(SWE2DMesh& mesh) {
    int n_faces = mesh.n_faces;
    int n_cells = mesh.n_cells;

    mesh.face_stencil_5.resize(5 * n_faces);
    mesh.face_mp5_case.resize(n_faces, 1);  // default case 1

    for (int f = 0; f < n_faces; ++f) {
        int i = face_owners[2 * f + 0];  // owner = upwind
        int j = face_owners[2 * f + 1];  // neighbor = downwind
        int* st = &mesh.face_stencil_5[5 * f];

        // u2: neighbor-of-neighbor of i (2 hops upwind)
        int u2 = i;
        if (i >= 0) {
            int s = mesh.cell_face_offsets[i];
            int e = mesh.cell_face_offsets[i + 1];
            for (int k = s; k < e; ++k) {
                int nbr = mesh.cell_face_nodes[k];
                if (nbr != j && nbr >= 0) {
                    // Walk one more hop upwind from this neighbor back past i
                    int s2 = mesh.cell_face_offsets[nbr];
                    int e2 = mesh.cell_face_offsets[nbr + 1];
                    for (int k2 = s2; k2 < e2; ++k2) {
                        int nbr2 = mesh.cell_face_nodes[k2];
                        if (nbr2 != i && nbr2 >= 0) {
                            u2 = nbr2;
                            goto found_u2;
                        }
                    }
                }
            }
            found_u2:;
        }

        // u1: first upwind neighbor (any neighbor of i, not j)
        int u1 = i;
        if (i >= 0) {
            int s = mesh.cell_face_offsets[i];
            int e = mesh.cell_face_offsets[i + 1];
            for (int k = s; k < e; ++k) {
                int nbr = mesh.cell_face_nodes[k];
                if (nbr != j && nbr >= 0) {
                    u1 = nbr;
                    break;
                }
            }
        }

        // v1: first downwind neighbor (any neighbor of j, not i)
        int v1 = j;
        if (j >= 0) {
            int s = mesh.cell_face_offsets[j];
            int e = mesh.cell_face_offsets[j + 1];
            for (int k = s; k < e; ++k) {
                int nbr = mesh.cell_face_nodes[k];
                if (nbr != i && nbr >= 0) {
                    v1 = nbr;
                    break;
                }
            }
        }

        st[0] = u2; st[1] = u1; st[2] = i; st[3] = j; st[4] = v1;
    }
}
```

Call this after the WENO3 sub-stencil builder.

- [ ] **Step 3: Build**

Run: `cd build && make -j$(nproc)`
Expected: build succeeds.

- [ ] **Step 4: Commit**

```bash
git add cpp/src/swe2d_mesh.cpp
git commit -m "feat: build MP5 5-cell walk table during mesh assembly"
```

---

### Task 13: Implement `mp5_kernel`

**Agent:** `cpp-pro_opencode-go_deepseek-v4-flash`

**Files:**
- Modify: `cpp/src/swe2d_reconstruct.cu`

- [ ] **Step 1: Implement minmod device helper**

```cuda
template <typename T>
__device__ T minmod(T a, T b) {
    if (a * b <= T(0)) return T(0);
    return (fabs(a) < fabs(b)) ? a : b;
}
```

- [ ] **Step 2: Implement mp5_kernel**

```cuda
template <typename T>
__global__ void mp5_kernel(
    const T* __restrict__ q,
    const T* __restrict__ face_mid_x,
    const T* __restrict__ face_mid_y,
    const T* __restrict__ cell_cx,
    const T* __restrict__ cell_cy,
    const int* __restrict__ face_stencil_5,
    const int* __restrict__ face_mp5_case,
    int n_faces,
    T* __restrict__ q_face_recon)
{
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= n_faces) return;

    const int* st = &face_stencil_5[5 * f];
    int u2 = st[0], u1 = st[1], u = st[2], v = st[3], v1 = st[4];

    T fm2 = q[u2], fm1 = q[u1], f0 = q[u], fp1 = q[v], fp2 = q[v1];

    T xf = face_mid_x[f], yf = face_mid_y[f];
    T xu = cell_cx[u], yu = cell_cy[u];
    T xv = cell_cx[v], yv = cell_cy[v];
    T dist_uv = sqrt((xv - xu)*(xv - xu) + (yv - yu)*(yv - yu));
    T dist_uf = sqrt((xf - xu)*(xf - xu) + (yf - yu)*(yf - yu));
    T t = (dist_uv > T(1e-12)) ? (dist_uf / dist_uv) : T(0.5);

    T f4 = (T(1.0)/T(60.0)) * (
        T(2.0)*fm2 - T(13.0)*fm1 + T(47.0)*f0 + T(27.0)*fp1 - T(3.0)*fp2
    );

    T f_linear = f0 + t * (fp1 - f0);

    T f_tvd = f0 + T(0.5) * minmod(fp1 - f0, f0 - fm1);

    T f_min = fmin(fmin(fm1, f0), fp1);
    T f_max = fmax(fmax(fm1, f0), fp1);

    int fcase = face_mp5_case[f];
    T f_mp5;

    switch (fcase) {
        case 1: {
            f_mp5 = f4;
            f_mp5 = fmax(f_min, fmin(f_max, f_mp5));
            break;
        }
        case 2: {
            T d4_scaled = f4 - f_linear;
            T d_min = f_min - f_linear;
            T d_max = f_max - f_linear;
            T denom = fmax(T(1e-12), d_max - d_min);
            T mapped = d_min + T(2.0)*d4_scaled/T(3.0);
            f_mp5 = f_linear + fmax(d_min, fmin(d_max, mapped));
            break;
        }
        case 3: {
            f_mp5 = f_linear + T(0.5) * (f_tvd - f_linear);
            break;
        }
        default: {
            f_mp5 = f_linear;
            break;
        }
    }

    q_face_recon[f] = f_mp5;
}
```

- [ ] **Step 3: Build**

Run: `cd build && make -j$(nproc)`
Expected: build succeeds.

- [ ] **Step 4: Commit**

```bash
git add cpp/src/swe2d_reconstruct.cu
git commit -m "feat: implement mp5_kernel with mapped monotonicity-preserving limiter"
```

---

### Task 14: Add MP5 dispatch and CFL enforcement

**Agent:** `cpp-pro_opencode-go_deepseek-v4-flash`

**Files:**
- Modify: `cpp/src/swe2d_gpu.cu`

- [ ] **Step 1: Add MP5 launch before flux kernel**

```cuda
const int scheme_mp5 = static_cast<int>(SWE2DSpatialScheme::FV_MP5);

if (spatial_scheme == scheme_mp5) {
    mp5_kernel<T>
        <<<face_grid, face_block, 0, stream>>>
        (dev.q, dev.face_mid_x, dev.face_mid_y,
         dev.cell_cx, dev.cell_cy,
         mesh.face_stencil_5_d, mesh.face_mp5_case_d,
         n_faces,
         dev.mp5_face_recon);
    cudaCheck(cudaGetLastError());
}
```

- [ ] **Step 2: Update flux kernel dispatch for MP5**

```cuda
if (spatial_scheme == scheme_mp5) {
    q_face_L = dev.mp5_face_recon[face_id];
} else if (spatial_scheme == scheme_weno3) {
    q_face_L = dev.weno3_face_recon[face_id];
} else if (spatial_scheme == scheme_weno5 && !near_boundary) {
    // ...
```

- [ ] **Step 3: Add device array**

In `swe2d_gpu.cuh`, add `mp5_face_recon` to `SWE2DDeviceState`.

- [ ] **Step 4: Build**

Run: `cd build && make -j$(nproc)`
Expected: build succeeds.

- [ ] **Step 5: Commit**

```bash
git add cpp/src/swe2d_gpu.cu cpp/src/swe2d_gpu.cuh
git commit -m "feat: add MP5 dispatch in GPU flux pipeline"
```

---

### Task 15: Add CFL enforcement in backend

**Agent:** `python-pro_opencode-go_deepseek-v4-flash`

**Files:**
- Modify: `swe2d/runtime/backend.py`

- [ ] **Step 1: Add CFL constant and clamping**

```python
# At module level in swe2d/runtime/backend.py
_SCHEME_MAX_CFL: dict[int, float] = {
    0: 0.8, 1: 0.8, 2: 0.8, 3: 0.8, 4: 0.8,
    5: 0.8,   # Barth-Jespersen
    6: 0.8,   # WENO3
    7: 0.5,   # WENO5
    8: 0.4,   # MP5
}
```

- [ ] **Step 2: Add CFL clamping method**

```python
def _clamp_cfl_for_scheme(self) -> None:
    scheme = getattr(self, '_spatial_scheme', 0)
    max_cfl = _SCHEME_MAX_CFL.get(scheme, 0.8)
    if self._cfl > max_cfl:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(
            f"CFL={self._cfl} exceeds scheme max {max_cfl}, clamping to {max_cfl}"
        )
        self._cfl = max_cfl
```

- [ ] **Step 3: Call in initialize()**

Add `self._clamp_cfl_for_scheme()` at the end of `initialize()` after `self._spatial_scheme` is set.

- [ ] **Step 4: Commit**

```bash
git add swe2d/runtime/backend.py
git commit -m "feat: enforce per-scheme CFL limits with MP5 CFL≤0.4"
```

---

## Phase 5: Tests

### Task 16: Convergence tests for all new schemes

**Agent:** `test-automator_opencode-go_deepseek-v4-flash`

**Files:**
- Create: `tests/test_swe2d_barth_jespersen_convergence.py`
- Create: `tests/test_swe2d_weno3_convergence.py`
- Create: `tests/test_swe2d_mp5_convergence.py`

- [ ] **Step 1: Read existing convergence test for pattern**

Read `tests/test_swe2d_weno5_convergence.py` to understand the helper structure and convergence rate computation.

- [ ] **Step 2: Create Barth-Jespersen convergence test**

```python
# tests/test_swe2d_barth_jespersen_convergence.py
"""Convergence order test for FV_BARTH_JESPERSEN (scheme 5)."""

import numpy as np
import pytest
from swe2d.extensions.extension_models import SpatialDiscretization

SCHEME = int(SpatialDiscretization.FV_BARTH_JESPERSEN)

def _run_scheme_on_mesh(mesh_path, end_time=0.1, cfl=0.5):
    """Run scheme on a gmsh mesh and return L2 error."""
    from swe2d.runtime.backend import SWE2DBackend
    backend = SWE2DBackend()
    backend.initialize(
        mesh_path=mesh_path,
        spatial_discretization=SCHEME,
        cfl=cfl,
        end_time=end_time,
    )
    backend.run()
    h = backend.get_stage("h")
    h_exact = backend.get_exact_manufactured()
    error = np.sqrt(np.sum(backend.cell_areas * (h - h_exact)**2))
    backend.cleanup()
    return error


def _make_gmsh_triangle_mesh(nx, ny, tmp_path):
    """Create gmsh .msh2 file for convergence study."""
    import gmsh
    gmsh.initialize()
    gmsh.model.add("convergence")
    h = 1.0 / max(nx, ny)
    gmsh.model.occ.add_rectangle(0, 0, 0, 1.0, 1.0)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.set_size(gmsh.model.get_entities(0), h)
    gmsh.model.mesh.generate(2)
    mesh_path = str(tmp_path / "mesh.msh2")
    gmsh.write(mesh_path)
    gmsh.finalize()
    return mesh_path


@pytest.mark.parametrize("resolution", [(8, 8), (16, 16), (32, 32)])
def test_barth_jespersen_convergence(resolution, tmp_path):
    nx, ny = resolution
    mesh_path = _make_gmsh_triangle_mesh(nx, ny, tmp_path)
    error = _run_scheme_on_mesh(mesh_path)
    assert not np.isnan(error), "L2 error must not be NaN"
    assert error > 0, "L2 error must be positive for finite meshes"


def test_barth_jespersen_convergence_order(tmp_path):
    """Fit convergence order from three refinement levels."""
    errors = []
    h_values = []
    for nx in [16, 32, 64]:
        mesh_path = _make_gmsh_triangle_mesh(nx, nx, tmp_path)
        error = _run_scheme_on_mesh(mesh_path)
        errors.append(error)
        h_values.append(1.0 / nx)

    log_h = np.log(np.array(h_values))
    log_e = np.log(np.array(errors))
    slope, _ = np.polyfit(log_h, log_e, 1)
    order = -slope
    assert order >= 1.8, f"Expected order ≥ 1.8, got {order:.2f}"
```

- [ ] **Step 3: Create WENO3 convergence test**

```python
# tests/test_swe2d_weno3_convergence.py
"""Convergence order test for FV_WENO3 (scheme 6)."""

import numpy as np
import pytest
from swe2d.extensions.extension_models import SpatialDiscretization

SCHEME = int(SpatialDiscretization.FV_WENO3)

def _run_scheme_on_mesh(mesh_path, end_time=0.1, cfl=0.5):
    from swe2d.runtime.backend import SWE2DBackend
    backend = SWE2DBackend()
    backend.initialize(
        mesh_path=mesh_path,
        spatial_discretization=SCHEME,
        cfl=cfl,
        end_time=end_time,
    )
    backend.run()
    h = backend.get_stage("h")
    h_exact = backend.get_exact_manufactured()
    error = np.sqrt(np.sum(backend.cell_areas * (h - h_exact)**2))
    backend.cleanup()
    return error


def _make_gmsh_triangle_mesh(nx, ny, tmp_path):
    import gmsh
    gmsh.initialize()
    gmsh.model.add("convergence")
    h = 1.0 / max(nx, ny)
    gmsh.model.occ.add_rectangle(0, 0, 0, 1.0, 1.0)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.set_size(gmsh.model.get_entities(0), h)
    gmsh.model.mesh.generate(2)
    path = str(tmp_path / "mesh.msh2")
    gmsh.write(path)
    gmsh.finalize()
    return path


def test_weno3_convergence_order(tmp_path):
    errors, h_vals = [], []
    for nx in [16, 32, 64]:
        mp = _make_gmsh_triangle_mesh(nx, nx, tmp_path)
        err = _run_scheme_on_mesh(mp)
        errors.append(err)
        h_vals.append(1.0 / nx)
    log_h = np.log(np.array(h_vals))
    log_e = np.log(np.array(errors))
    slope, _ = np.polyfit(log_h, log_e, 1)
    order = -slope
    assert order >= 2.5, f"Expected order ≥ 2.5, got {order:.2f}"
```

- [ ] **Step 4: Create MP5 convergence test**

```python
# tests/test_swe2d_mp5_convergence.py
"""Convergence order test for FV_MP5 (scheme 8)."""

import numpy as np
import pytest
from swe2d.extensions.extension_models import SpatialDiscretization

SCHEME = int(SpatialDiscretization.FV_MP5)

def _run_scheme_on_mesh(mesh_path, end_time=0.1, cfl=0.3):
    from swe2d.runtime.backend import SWE2DBackend
    backend = SWE2DBackend()
    backend.initialize(
        mesh_path=mesh_path,
        spatial_discretization=SCHEME,
        cfl=cfl,
        end_time=end_time,
    )
    backend.run()
    h = backend.get_stage("h")
    h_exact = backend.get_exact_manufactured()
    error = np.sqrt(np.sum(backend.cell_areas * (h - h_exact)**2))
    backend.cleanup()
    return error


def _make_gmsh_triangle_mesh(nx, ny, tmp_path):
    import gmsh
    gmsh.initialize()
    gmsh.model.add("convergence")
    h = 1.0 / max(nx, ny)
    gmsh.model.occ.add_rectangle(0, 0, 0, 1.0, 1.0)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.set_size(gmsh.model.get_entities(0), h)
    gmsh.model.mesh.generate(2)
    path = str(tmp_path / "mesh.msh2")
    gmsh.write(path)
    gmsh.finalize()
    return path


def test_mp5_convergence_order(tmp_path):
    errors, h_vals = [], []
    for nx in [16, 32, 64]:
        mp = _make_gmsh_triangle_mesh(nx, nx, tmp_path)
        err = _run_scheme_on_mesh(mp, cfl=0.3)
        errors.append(err)
        h_vals.append(1.0 / nx)
    log_h = np.log(np.array(h_vals))
    log_e = np.log(np.array(errors))
    slope, _ = np.polyfit(log_h, log_e, 1)
    order = -slope
    assert order >= 3.5, f"Expected order ≥ 3.5, got {order:.2f}"
```

- [ ] **Step 5: Run convergence tests**

Run: `pytest tests/test_swe2d_barth_jespersen_convergence.py tests/test_swe2d_weno3_convergence.py tests/test_swe2d_mp5_convergence.py -v`
Expected: tests compile and run (may fail initially if kernels have bugs; failures should be documented).

- [ ] **Step 6: Commit**

```bash
git add tests/test_swe2d_barth_jespersen_convergence.py tests/test_swe2d_weno3_convergence.py tests/test_swe2d_mp5_convergence.py
git commit -m "test: add convergence tests for Barth-Jespersen, WENO3, MP5"
```

---

### Task 17: Robustness and monotonicity tests

**Agent:** `test-automator_opencode-go_deepseek-v4-flash`

**Files:**
- Create: `tests/test_swe2d_poor_mesh_robustness.py`
- Create: `tests/test_face_value_monotonicity.py`

- [ ] **Step 1: Create poor-mesh robustness test**

```python
# tests/test_swe2d_poor_mesh_robustness.py
"""Robustness tests for spatial schemes on poor-quality meshes."""

import numpy as np
import pytest
from swe2d.extensions.extension_models import SpatialDiscretization


SCHEMES = [5, 6, 8]
SCHEME_NAMES = {
    5: "Barth-Jespersen",
    6: "WENO3",
    8: "MP5",
}


def _make_poor_mesh(tmp_path, n=20):
    """Create mesh with stretched quads and sliver triangles."""
    import gmsh
    gmsh.initialize()
    gmsh.model.add("poor_mesh")
    gmsh.model.occ.add_rectangle(0, 0, 0, 10.0, 1.0)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.set_size(gmsh.model.get_entities(0), 0.5)
    gmsh.model.mesh.generate(2)
    path = str(tmp_path / "poor.msh2")
    gmsh.write(path)
    gmsh.finalize()
    return path


@pytest.mark.parametrize("scheme", SCHEMES)
def test_no_nan_on_poor_mesh(scheme, tmp_path):
    mesh_path = _make_poor_mesh(tmp_path)
    from swe2d.runtime.backend import SWE2DBackend
    backend = SWE2DBackend()
    cfl = 0.3 if scheme == 8 else 0.5
    backend.initialize(
        mesh_path=mesh_path,
        spatial_discretization=scheme,
        cfl=cfl,
        end_time=0.5,
    )
    backend.run()
    h = backend.get_stage("h")
    assert not np.any(np.isnan(h)), f"{SCHEME_NAMES[scheme]} produced NaN"
    h_max = np.max(np.abs(h))
    analytical_max = 5.0
    assert h_max < 5 * analytical_max, (
        f"{SCHEME_NAMES[scheme]}: max |h|={h_max:.2f} exceeds 5× analytical"
    )
    backend.cleanup()


@pytest.mark.parametrize("scheme", SCHEMES)
def test_no_oscillation_near_boundaries(scheme, tmp_path):
    mesh_path = _make_poor_mesh(tmp_path)
    from swe2d.runtime.backend import SWE2DBackend
    backend = SWE2DBackend()
    cfl = 0.3 if scheme == 8 else 0.5
    backend.initialize(
        mesh_path=mesh_path,
        spatial_discretization=scheme,
        cfl=cfl,
        end_time=0.5,
    )
    backend.run()
    h = backend.get_stage("h")
    gradient = np.abs(np.diff(h.flatten()))
    max_gradient = np.max(gradient)
    assert max_gradient < 1.0, (
        f"{SCHEME_NAMES[scheme]}: max gradient {max_gradient:.3f} indicates oscillation"
    )
    backend.cleanup()
```

- [ ] **Step 2: Create monotonicity test**

```python
# tests/test_face_value_monotonicity.py
"""Verify local monotonicity preservation for all schemes."""

import numpy as np
import pytest
from swe2d.extensions.extension_models import SpatialDiscretization


SCHEMES = list(range(9))


def _run_scheme(scheme, tmp_path):
    import gmsh
    gmsh.initialize()
    gmsh.model.add("mono")
    gmsh.model.occ.add_rectangle(0, 0, 0, 1.0, 1.0)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.set_size(gmsh.model.get_entities(0), 0.1)
    gmsh.model.mesh.generate(2)
    mesh_path = str(tmp_path / "mono.msh2")
    gmsh.write(mesh_path)
    gmsh.finalize()

    from swe2d.runtime.backend import SWE2DBackend
    backend = SWE2DBackend()
    cfl = 0.3 if scheme == 8 else 0.5
    backend.initialize(
        mesh_path=mesh_path,
        spatial_discretization=scheme,
        cfl=cfl,
        end_time=0.1,
    )
    backend.run()
    h = backend.get_stage("h")
    backend.cleanup()
    return h


@pytest.mark.parametrize("scheme", SCHEMES)
def test_solution_non_negative(scheme, tmp_path):
    """Water depth should not go negative for any scheme."""
    h = _run_scheme(scheme, tmp_path)
    assert np.all(h >= -1e-10), f"Scheme {scheme} produced negative water depth"
    assert not np.any(np.isnan(h)), f"Scheme {scheme} produced NaN"
```

- [ ] **Step 3: Run robustness tests**

Run: `pytest tests/test_swe2d_poor_mesh_robustness.py tests/test_face_value_monotonicity.py -v`
Expected: tests run.

- [ ] **Step 4: Commit**

```bash
git add tests/test_swe2d_poor_mesh_robustness.py tests/test_face_value_monotonicity.py
git commit -m "test: add poor-mesh robustness and monotonicity envelope tests"
```

---

### Task 18: Performance regression test

**Agent:** `test-automator_opencode-go_deepseek-v4-flash`

**Files:**
- Create: `tests/test_spatial_scheme_perf.py`

- [ ] **Step 1: Create performance test**

```python
# tests/test_spatial_scheme_perf.py
"""Performance benchmarks for spatial schemes."""

import time
import numpy as np
import pytest
from swe2d.extensions.extension_models import SpatialDiscretization


BENCHMARK_SCHEMES = [0, 5, 6, 8]
SCHEME_NAMES = {0: "First-order", 5: "Barth-Jespersen", 6: "WENO3", 8: "MP5"}


def _make_mesh(n, tmp_path):
    import gmsh
    gmsh.initialize()
    gmsh.model.add("perf")
    gmsh.model.occ.add_rectangle(0, 0, 0, 1.0, 1.0)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.set_size(gmsh.model.get_entities(0), 1.0 / n)
    gmsh.model.mesh.generate(2)
    path = str(tmp_path / "perf.msh2")
    gmsh.write(path)
    gmsh.finalize()
    return path


@pytest.mark.parametrize("scheme", BENCHMARK_SCHEMES)
def test_scheme_performance_ratio(scheme, tmp_path):
    """Measure per-step wall time relative to first-order."""
    mesh_path = _make_mesh(32, tmp_path)  # ~2k triangles for quick test
    from swe2d.runtime.backend import SWE2DBackend

    backend = SWE2DBackend()
    cfl = 0.3 if scheme == 8 else 0.5
    backend.initialize(
        mesh_path=mesh_path,
        spatial_discretization=scheme,
        cfl=cfl,
        end_time=0.2,
    )

    t0 = time.perf_counter()
    backend.run()
    elapsed = time.perf_counter() - t0

    ncells = backend.n_cells
    backend.cleanup()

    print(f"Scheme {scheme} ({SCHEME_NAMES[scheme]}): "
          f"{elapsed:.3f}s for {ncells} cells")
    assert elapsed < 60.0, f"Scheme {scheme} took {elapsed:.1f}s — too slow"
```

- [ ] **Step 2: Run performance test**

Run: `pytest tests/test_spatial_scheme_perf.py -v -s`
Expected: prints timing for each scheme.

- [ ] **Step 3: Commit**

```bash
git add tests/test_spatial_scheme_perf.py
git commit -m "test: add spatial scheme performance benchmark"
```

---

## Phase 6: GUI / CLI / Docs

### Task 19: Add reconstruction scheme combo to QML form

**Agent:** `python-pro_opencode-go_deepseek-v4-flash`

**Files:**
- Modify: `QML/form_init.py`

- [ ] **Step 1: Read QML/form_init.py**

Read `QML/form_init.py` to find form initialization and combo box population.

- [ ] **Step 2: Add reconstruction scheme combo**

```python
# In the form initialization function, add after existing combos:

_spatial_schemes = [
    ("First-order (0)", 0),
    ("MUSCL Fast / Superbee (1)", 1),
    ("MUSCL MinMod (2)", 2),
    ("MUSCL MC (3)", 3),
    ("MUSCL Van Leer (4)", 4),
    ("Barth-Jespersen (5)", 5),
    ("WENO3 — 3-sub-stencil (6)", 6),
    ("WENO5 — 2-ring LSQ (7)", 7),
    ("MP5 — Mapped Monotonicity-Preserving (8)", 8),
]

for label, value in _spatial_schemes:
    reconstruction_combo.addItem(label, value)

# Set default
reconstruction_combo.setCurrentIndex(3)  # MUSCL MC default
```

Note: The exact combo widget name and form structure must be verified against the actual QML file. Read `QML/form_init.py` first to confirm widget names.

- [ ] **Step 3: Test GUI loads**

Open QGIS and verify the reconstruction scheme combo appears with all 9 entries.

- [ ] **Step 4: Commit**

```bash
git add QML/form_init.py
git commit -m "feat: add reconstruction scheme combo with all 9 spatial schemes"
```

---

### Task 20: Update documentation

**Agent:** `python-pro_opencode-go_deepseek-v4-flash`

**Files:**
- Modify: `docs/SOLVER_ORDER_AND_STENCIL.md`
- Modify: `docs/USER_GUIDE.md`
- Modify: `docs/INDEX.md`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Update SOLVER_ORDER_AND_STENCIL.md**

Read the existing file. Extend the scheme table to include all 9 schemes:

| Scheme | Order | Stencil | CFL | Best Use |
|--------|:-:|:-:|:-:|----------|
| 0 — First-order | 1st | 1-cell | 0.8 | Debug, very coarse |
| 1 — MUSCL Fast | 2nd | 1-ring | 0.8 | High-res riverine |
| 2 — MUSCL MinMod | 2nd | 1-ring | 0.8 | Robust general purpose |
| 3 — MUSCL MC | 2nd | 1-ring | 0.8 | Balanced accuracy/robustness |
| 4 — MUSCL Van Leer | 2nd | 1-ring | 0.8 | Smooth flows |
| 5 — Barth-Jespersen | 2nd | 1-ring | 0.8 | Poor meshes, urban drainage |
| 6 — WENO3 | 3rd | 1-ring | 0.8 | Higher accuracy, 1-ring budget |
| 7 — WENO5 | ~3rd | 2-ring | 0.5 | Max accuracy, fine meshes |
| 8 — MP5 | 4th | 5-cell | 0.4 | Highest order, smooth flow |

- [ ] **Step 2: Update USER_GUIDE.md**

Add a section for each new scheme describing:
- What it does (1 sentence)
- When to use it (use case guidance)
- Mesh quality sensitivity
- CFL constraint (for MP5)

- [ ] **Step 3: Update INDEX.md**

Ensure links to:
- `ADVANCED_SPATIAL_SCHEMES.md` (tech guide)
- `docs/IMPLEMENTATION_PLANS/2026-07-10-advanced-spatial-schemes.md` (or old plan)
- New spec: `docs/archive/specs/2026-07-10-advanced-spatial-schemes-design.md`

- [ ] **Step 4: Update CHANGELOG.md**

```markdown
## [Unreleased]

### Added
- New spatial reconstruction schemes:
  - `FV_BARTH_JESPERSEN` (scheme 5): Barth-Jespersen gradient limiter for robust 2nd-order on poor meshes
  - `FV_WENO3` (scheme 6): True 3-sub-stencil WENO (3rd-order, 1-ring stencil)
  - `FV_MP5` (scheme 8): Suresh-Huynh Mapped Monotonicity-Preserving (4th-order)

### Changed
- `FV_WENO5` renumbered from scheme 6 to scheme 7 (breaking: persisted configs with scheme=6 will warn)
```

- [ ] **Step 5: Commit**

```bash
git add docs/SOLVER_ORDER_AND_STENCIL.md docs/USER_GUIDE.md docs/INDEX.md CHANGELOG.md
git commit -m "docs: update user guide, scheme table, index, and changelog for new schemes"
```

---

## Phase 7: Review & Verification

### Task 21: Run full test suite regression

**Agent:** `test-automator_opencode-go_deepseek-v4-flash`

- [ ] **Step 1: Run full test suite**

```bash
pytest tests/ -v --timeout=300 2>&1 | tee test_results.log
```

- [ ] **Step 2: Verify zero regressions**

Check that no existing tests fail:
```bash
grep -c "FAILED" test_results.log
```
Expected: 0 (or only known-flaky tests unrelated to spatial schemes).

- [ ] **Step 3: Run specific new tests**

```bash
pytest tests/test_swe2d_barth_jespersen_convergence.py tests/test_swe2d_weno3_convergence.py tests/test_swe2d_mp5_convergence.py tests/test_swe2d_poor_mesh_robustness.py tests/test_face_value_monotonicity.py -v
```

- [ ] **Step 4: Document any test failures with root cause analysis**

### Task 22: C++ code review

**Agent:** `cpp-pro_opencode-go_deepseek-v4-pro` (review phase — use pro variant)

- [ ] **Step 1: Review kernel implementations for correctness**
  - Verify Barth-Jespersen limiter math matches Barth & Jespersen 1989
  - Verify WENO3 sub-stencil construction and Hu-Shu weights
  - Verify MP5 polynomial coefficients and mapped limiter cases match Suresh-Huynh 1997

- [ ] **Step 2: Review memory safety**
  - Check that all CUDA allocations have matching frees
  - Verify array bounds in stencil table accesses
  - Check for potential race conditions (no shared memory writes with conflicts)

- [ ] **Step 3: Review performance**
  - Profile kernel launch overhead
  - Check occupancy and register pressure
  - Verify no unnecessary data copies

- [ ] **Step 4: Verify no warnings with `-Wall -Wextra`**

```bash
cd build && cmake .. -DCMAKE_CXX_FLAGS="-Wall -Wextra" && make -j$(nproc) 2>&1 | grep -i warning
```

### Task 23: Python code review

**Agent:** `python-pro_opencode-go_deepseek-v4-pro` (review phase — use pro variant)

- [ ] **Step 1: Review enum changes**
  - Verify all enum values are contiguous 0-8
  - Check backward-compat aliases are preserved

- [ ] **Step 2: Review CFL enforcement**
  - Verify clamping is applied before first step
  - Check that user-set CFL is not silently overridden without warning

- [ ] **Step 3: Review scheme migration**
  - Verify old scheme-6 configs get auto-migrated to 7
  - Check warning message is informative and actionable
  - Ensure headless_runner, batch_runner, and backend all have consistent migration

- [ ] **Step 4: Review code quality**
  ```bash
  ruff check swe2d/
  mypy swe2d/extensions/extension_models.py swe2d/runtime/backend.py
  ```

---

## Task Summary

| Phase | Task | Agent | Est. effort |
|-------|------|-------|:-:|
| 0 — Foundation | 1: Python enum | `python-pro_flash` | 15m |
| 0 — Foundation | 2: C++ enum | `cpp-pro_flash` | 15m |
| 0 — Foundation | 3: CMake + stub file | `build-engineer_flash` | 15m |
| 0 — Foundation | 4: Stencil fields (both layers) | `cpp-pro_flash` | 30m |
| 1 — Barth-Jespersen | 5: `barth_jespersen_kernel` | `cpp-pro_flash` | 45m |
| 1 — Barth-Jespersen | 6: GPU dispatch for scheme 5 | `cpp-pro_flash` | 1h |
| 1 — Barth-Jespersen | 7: Pybind11 stencil exposure | `cpp-pro_flash` | 30m |
| 2 — Mesh Assembly | 8: WENO3 sub-stencil builder | `cpp-pro_flash` | 1h |
| 3 — WENO3 | 9: `weno3_kernel` | `cpp-pro_flash` | 1.5h |
| 3 — WENO3 | 10: WENO3 GPU dispatch | `cpp-pro_flash` | 45m |
| 3 — WENO3 | 11: Scheme migration Python | `python-pro_flash` | 30m |
| 4 — MP5 | 12: 5-cell walk builder | `cpp-pro_flash` | 1h |
| 4 — MP5 | 13: `mp5_kernel` | `cpp-pro_flash` | 1h |
| 4 — MP5 | 14: MP5 GPU dispatch | `cpp-pro_flash` | 30m |
| 4 — MP5 | 15: CFL enforcement | `python-pro_flash` | 20m |
| 5 — Tests | 16: Convergence tests (3 files) | `test-automator_flash` | 1h |
| 5 — Tests | 17: Robustness + monotonicity tests | `test-automator_flash` | 45m |
| 5 — Tests | 18: Performance benchmark test | `test-automator_flash` | 30m |
| 6 — GUI/Docs | 19: QML form combo | `python-pro_flash` | 30m |
| 6 — GUI/Docs | 20: Documentation updates | `python-pro_flash` | 45m |
| 7 — Review | 21: Full test suite regression | `test-automator_flash` | 30m |
| 7 — Review | 22: C++ code review | `cpp-pro_pro` | 1h |
| 7 — Review | 23: Python code review | `python-pro_pro` | 30m |

**Estimated total:** ~15 hours of implementation + 2 hours review.
