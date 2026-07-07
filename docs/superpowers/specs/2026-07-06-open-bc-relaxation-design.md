# Open-BC Relaxation / Reflection Damping — Design Spec

> **Status:** Design spec ready for implementation plan.  
> **Replaces:** `docs/OPEN_BC_RELAXATION_PLAN.md`

## Problem Statement

Higher-order spatial schemes (MUSCL, WENO5) can become unstable near **outflow-style boundaries** when a hydraulic jump or bore approaches the boundary. The affected BC types are `OPEN` (4), `NORMAL_DEPTH` (6), `NORMAL_DEPTH_SLOPE` (7), and `REFLECT` (5).

### Root Cause

Outflow-style BCs are handled inside the GPU flux kernel by `make_ghost_cuda_local()` at `cpp/src/swe2d_gpu.cu:269`. The interior edge uses MUSCL/WENO5 reconstruction (5-cell stencil), while the boundary edge gets a first-order ghost state constructed from the current interior state. When a jump hits the boundary, the reconstructed interior state can overshoot the local bounds and feed a mismatch into the Riemann solver against the frozen ghost state. The reflected wave can create a positive feedback loop, producing runaway inflow / super-cell growth at the boundary.

### Scope

This feature adds a configurable **reflection-damping coefficient** `r` that blends the constructed ghost state toward the interior state for outflow-style BCs only. WALL (1), INFLOW_Q (2), and STAGE (3) are deliberately unchanged because they impose specific constraints that must be honored.

---

## Boundary-Condition Model (Corrected)

The codebase does **not** use per-side boundary conditions. BCs are assigned as follows:

1. **Default value:** every boundary edge starts with the default BC type (configurable in the UI).
2. **Override lines:** a GeoPackage polyline layer overrides the BC type and value on individual boundary edges whose geometry matches the line. The matching uses edge midpoint proximity and a `priority` field to resolve conflicts.

This feature follows the same model:

- A **global default** `open_bc_relaxation` is set in the Numerics / Stability UI group.
- The same BC override line layer may optionally carry a `bc_relax` (or `open_bc_relax`, `relax`) field that overrides the relaxation coefficient for matched boundary edges.
- Edges not matched by any override line use the global default.

---

## Solution Overview

Add a scalar `open_bc_relaxation` in `[0.0, 1.0]` and a per-edge override array that is uploaded to the GPU. For `bc_type ∈ {4, 5, 6, 7}` the ghost state is blended with the interior state after the normal BC construction:

```
g.h'  = (1 - r) * g.h  + r * hI
g.hu' = (1 - r) * g.hu + r * huI
g.hv' = (1 - r) * g.hv + r * hvI
```

| `r` | Meaning |
|-----|---------|
| `0.0` | Default — current behavior, no relaxation |
| `0.10 – 0.25` | Mild damping: keep most of the prescribed/zero-gradient state, but smooth reflections |
| `0.50` | Mid: ghost is halfway between prescribed and interior |
| `1.00` | Fully transmissive: ghost equals interior |

The final relaxed depth is clamped to `h_min` only (`depth_cap` is enforced in the update kernel, not in ghost construction).

---

## Key Design Decisions

1. **Global default + per-edge override from the BC line layer.** The same polyline layer that supplies `bc_type` / `bc_val` may also supply a `bc_relax` override. This mirrors the existing BC override model and keeps the data in one place.

2. **Pass `open_bc_relaxation` through `SWE2DSolverConfig` at solver create time.** This matches how `front_flux_damping` is handled. The device array is initialized to the global default during solver creation; per-edge overrides are uploaded afterwards via a binding that takes `(edge_index, relax)` arrays.

3. **Remove the unused CPU `make_ghost()` implementation.** `cpp/src/swe2d_numerics.hpp:478` is dead code and is removed rather than kept in lock-step.

4. **WALL and REFLECT are split in `make_ghost_cuda_local`.** Currently they share a case. WALL is excluded from relaxation; REFLECT is included so that a “soft reflecting” outflow can be damped.

5. **No graph-signature change for the scalar.** The kernel reads the per-edge coefficient from `d_edge_bc_relax`, so the scalar itself does not need to be passed into step wrappers or mixed into the graph signature. Graph replay will see updated values because the device pointer is stable and only the memory contents change. The only initialization work is done in `swe2d_gpu_init`.

6. **Default value `0.0` preserves exact current behavior.** When `r = 0.0` the relaxation branch is skipped, so bit-identical evolution is expected for the existing test suite.

---

## Files to Modify

### C++ Core

| File | Change |
|------|--------|
| `cpp/src/swe2d_solver.hpp` | Add `double open_bc_relaxation = 0.0;` to `SWE2DSolverConfig` |
| `cpp/src/swe2d_gpu.cuh` | Add `double* d_edge_bc_relax` to `SWE2DDeviceState`; extend `swe2d_gpu_init` to accept `open_bc_relaxation` and allocate/fill the array there; free it in `swe2d_gpu_destroy`. |
| `cpp/src/swe2d_gpu.cu` | (a) Add `d_edge_bc_relax` pointer to the flux-kernel signatures and pass `dev->d_edge_bc_relax` at every launch site<br>(b) Modify `make_ghost_cuda_local` to accept the per-edge relaxation value and apply it for `bc_type ∈ {4,5,6,7}`<br>(c) Split WALL and REFLECT cases<br>(d) Update the two `make_ghost_cuda_local` call sites (lines 2089 and 5780) |
| `cpp/src/swe2d_numerics.hpp` | **Delete** the unused `make_ghost()` function and the `GhostState` struct if no longer referenced |
| `cpp/src/swe2d_solver.cpp` | Pass `s->cfg.open_bc_relaxation` to `swe2d_gpu_init`; allocate/fill `d_edge_bc_relax` during solver init |
| `cpp/src/swe2d_bindings.cpp` | Add `open_bc_relaxation` to `swe2d_solver_create` signature and `cfg` assignment; add `swe2d_solver_set_edge_bc_relax(solver, edge_index, relax)` |

### Python Plumbing

| File | Change |
|------|--------|
| `swe2d/runtime/backend.py` | Add `open_bc_relaxation: float = 0.0` to `initialize()`; add `set_boundary_relaxation(bc_edge_node0, bc_edge_node1, bc_edge_relax)` method that computes edge indices and calls the new binding; add feature-availability check |
| `swe2d/runtime/backend_initializer.py` | Accept `open_bc_relaxation` and `bc_relax` from `RunContext`; pass to `backend.initialize()` and `backend.set_boundary_relaxation()` |
| `swe2d/workbench/workers/run_context.py` | Add `open_bc_relaxation: float = 0.0` and `bc_relax: np.ndarray` |
| `swe2d/workbench/views/model_tab_view.py` | Add `open_bc_relax_spin` in Numerics / Stability group; collect value in `collect_params()` |
| `swe2d/workbench/dialogs/batch_simulation_dialog.py` | Add `open_bc_relax_spin` to `_WIDGET_TO_CLI_MAP` |
| `swe2d/workbench/controllers/run_controller.py` | Read `wp.get("open_bc_relax_spin", 0.0)` into `RunContext`; store `bc_relax` returned from boundary collection |
| `swe2d/runtime/run_data_builder.py` | Carry `bc_relax` in `SWE2DRunInputData` |
| `swe2d/workbench/studio_dialog.py` | Pass the global default to `_apply_bc_layer_overrides`; unpack `bc_relax` from `collect_boundary_arrays()` |
| `swe2d/boundary_and_forcing/boundary_runtime_logic.py` | Extend `collect_boundary_arrays` to return `bc_relax`; pass `default_relax` to the override callback |
| `swe2d/boundary_and_forcing/boundary_qgis_adapter.py` | Read `bc_relax` / `open_bc_relax` / `relax` field from the BC layer and return a `bc_relax` array alongside `bc_type`/`bc_val` |
| `swe2d/workbench/workers/simulation_worker.py` | Pass `open_bc_relaxation` and `bc_relax` to backend initializer; call `set_boundary_relaxation` after `initialize()` |
| `swe2d/cli/headless_runner.py` | Read `rp.get("open_bc_relaxation", 0.0)`; read optional `bc_relax` from BC table; pass to `backend.initialize()` and `backend.set_boundary_relaxation()` |
| `swe2d/cli/gpkg_adapter.py` | Detect optional `bc_relax` column in BC table and return it |
| `swe2d/runtime/native_binding_compat.py` | No change required (existing helpers used) |

### UI / Tooltip

The `open_bc_relax_spin` tooltip:

> Reflection damping at open / normal-depth / reflect boundaries. Blends the constructed ghost state toward the interior state. `0.0` = current behavior. `0.1–0.5` if instability (runaway inflow, NaN h, oscillating jumps) appears near the boundary with higher-order schemes. `1.0` = fully transmissive boundary. WALL / INFLOW_Q / STAGE BCs are NOT affected. Per-edge override can be set with a `bc_relax` field on the BC line layer.

---

## Detailed Implementation Outline

### Step 1 — `SWE2DSolverConfig`

Add to `cpp/src/swe2d_solver.hpp`:

```cpp
struct SWE2DSolverConfig {
    // ... existing fields ...

    /// Reflection damping applied at OUTLET-style BCs (OPEN, REFLECT,
    /// NORMAL_DEPTH, NORMAL_DEPTH_SLOPE).  0.0 = disabled (current behavior).
    /// 0.1–0.5 = typical damping range.  1.0 = fully transmissive boundary.
    double open_bc_relaxation = 0.0;
};
```

### Step 2 — `SWE2DDeviceState`

Add to `cpp/src/swe2d_gpu.cuh`:

```cpp
struct SWE2DDeviceState {
    // ... existing fields ...

    /// Per-edge relaxation coefficient.  Always allocated and initialized to
    /// cfg.open_bc_relaxation; per-edge overrides are uploaded afterwards.
    double* d_edge_bc_relax = nullptr;
};
```

Allocate and fill with the global default during solver creation. Free in the destructor.

### Step 3 — `make_ghost_cuda_local`

New signature and relaxation block:

```cpp
__device__ __forceinline__ GhostStateLocal make_ghost_cuda_local(
    double hI, double huI, double hvI, double zbI,
    double nx, double ny,
    int bc_type,
    double bc_val,
    double h_min, double n_mann,
    double edge_bc_relax)
{
    GhostStateLocal g{};
    g.zb = zbI;

    switch (bc_type) {
        case 1: { // WALL: reflect normal velocity
            g.h = hI;
            const double un = huI * nx + hvI * ny;
            g.hu = huI - 2.0 * un * nx;
            g.hv = hvI - 2.0 * un * ny;
            break;
        }
        case 2: // INFLOW_Q
            g.h = hI;
            g.hu = -bc_val * nx;
            g.hv = -bc_val * ny;
            break;
        case 3: { // STAGE
            const double h_ghost = bc_val - zbI;
            g.h = (h_ghost > h_min) ? h_ghost : h_min;
            g.hu = huI;
            g.hv = hvI;
            break;
        }
        case 4: // OPEN
            g.h = hI;
            g.hu = huI;
            g.hv = hvI;
            break;
        case 5: { // REFLECT: same construction as WALL pre-relaxation
            g.h = hI;
            const double un = huI * nx + hvI * ny;
            g.hu = huI - 2.0 * un * nx;
            g.hv = hvI - 2.0 * un * ny;
            break;
        }
        case 6: // NORMAL_DEPTH
            g.h = (bc_val > h_min) ? bc_val : h_min;
            g.hu = huI;
            g.hv = hvI;
            break;
        case 7: { // NORMAL_DEPTH_SLOPE
            const double sf = fmax(fabs(bc_val), 1.0e-8);
            const double qn = huI * nx + hvI * ny;
            const double qmag = fabs(qn);
            if (qmag <= 1.0e-12) {
                g.h = (hI > h_min) ? hI : h_min;
            } else {
                const double n_eff = fmax(fabs(n_mann), 1.0e-6);
                const double h_nd = pow((qmag * n_eff) / sqrt(sf), 3.0 / 5.0);
                g.h = (h_nd > h_min) ? h_nd : h_min;
            }
            g.hu = huI;
            g.hv = hvI;
            break;
        }
        default:
            g.h = hI;
            g.hu = huI;
            g.hv = hvI;
            break;
    }

    // Apply reflection damping only to OUTLET-style BCs.
    if (bc_type == 4 || bc_type == 5 || bc_type == 6 || bc_type == 7) {
        if (edge_bc_relax > 0.0) {
            const double r = fmin(edge_bc_relax, 1.0);
            g.h  = (1.0 - r) * g.h  + r * hI;
            g.hu = (1.0 - r) * g.hu + r * huI;
            g.hv = (1.0 - r) * g.hv + r * hvI;
            g.h  = fmax(g.h, h_min);
        }
    }
    return g;
}
```

### Step 4 — Kernel Signatures

Add a `const double* __restrict__ edge_bc_relax` pointer to the flux-kernel signatures in `cpp/src/swe2d_gpu.cu` and pass `dev->d_edge_bc_relax` at every launch site. The scalar `open_bc_relaxation` does **not** need to be added to the step wrappers or to the graph signature because the kernel reads the per-edge value from device memory.

### Step 5 — `make_ghost_cuda_local` Call Sites

Both call sites (lines ~2089 and ~5780) read the per-edge value and pass it to `make_ghost_cuda_local`:

```cpp
const double edge_relax = d_edge_bc_relax[e];
GhostStateLocal gs = make_ghost_cuda_local(
    hL, huL, hvL, zbL, nx, ny,
    edge_bc[e], edge_bc_val[e], h_min, n_local, edge_relax);
```

### Step 6 — `swe2d_solver.cpp`

Pass `s->cfg.open_bc_relaxation` to every `swe2d_gpu_step*` call and to `swe2d_gpu_init`. Add `d_edge_bc_relax` to `swe2d_gpu_destroy`.

### Step 7 — Bindings

Extend `swe2d_solver_create` in `cpp/src/swe2d_bindings.cpp` with:

```cpp
py::arg("open_bc_relaxation") = 0.0,
```

and assign `cfg.open_bc_relaxation = open_bc_relaxation;`.

Add the per-edge override binding:

```cpp
__global__ void swe2d_apply_edge_relax_kernel(
    int32_t n_updates,
    const int32_t* edge_index,
    const double* relax,
    double* d_edge_bc_relax)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_updates) {
        d_edge_bc_relax[edge_index[i]] = relax[i];
    }
}

m.def("swe2d_solver_set_edge_bc_relax",
    [](std::shared_ptr<PySolver>& ps,
       py::array_t<int32_t, py::array::c_style | py::array::forcecast> edge_index,
       py::array_t<double, py::array::c_style | py::array::forcecast> relax) {
        if (!ps || !ps->solver || !ps->solver->dev)
            throw std::runtime_error("solver not initialized");
        if (edge_index.size() != relax.size())
            throw std::runtime_error("edge_index and relax must have the same length");
        if (!ps->solver->dev->d_edge_bc_relax)
            throw std::runtime_error("device relaxation array not allocated");
        const int32_t n = static_cast<int32_t>(edge_index.size());
        constexpr int BLOCK = 256;
        const int grid = (n + BLOCK - 1) / BLOCK;
        swe2d_apply_edge_relax_kernel<<<grid, BLOCK>>>(
            n, edge_index.data(), relax.data(), ps->solver->dev->d_edge_bc_relax);
        CUDA_CHECK(cudaGetLastError());
    },
    py::arg("solver"), py::arg("edge_index"), py::arg("relax"),
    "Upload per-edge relaxation overrides for boundary edges.");
```

A small scatter kernel is preferred over a per-element `cudaMemcpy` loop.

### Step 8 — Python Backend

Add to `swe2d/runtime/backend.py`:

```python
def set_boundary_relaxation(
    self,
    bc_edge_node0: np.ndarray,
    bc_edge_node1: np.ndarray,
    bc_edge_relax: np.ndarray,
) -> None:
    """Upload per-edge relaxation overrides."""
    if not self._boundary_edge_index_by_nodes:
        return
    n0 = np.ascontiguousarray(bc_edge_node0, dtype=np.int32).ravel()
    n1 = np.ascontiguousarray(bc_edge_node1, dtype=np.int32).ravel()
    r = np.ascontiguousarray(bc_edge_relax, dtype=np.float64).ravel()
    if not (n0.size == n1.size == r.size):
        raise ValueError("bc edge relax arrays must have the same length")
    edge_index = np.empty(n0.size, dtype=np.int32)
    for i in range(n0.size):
        key = (int(n0[i]), int(n1[i]))
        key = key if key[0] < key[1] else (key[1], key[0])
        edge_index[i] = self._boundary_edge_index_by_nodes[key]
    if self._solver_h is not None:
        self._mod.swe2d_solver_set_edge_bc_relax(self._solver_h, edge_index, r)
```

Add `open_bc_relaxation: float = 0.0` to `initialize()` and pass it into the native create call.

### Step 9 — BC Override Layer

In `swe2d/boundary_and_forcing/boundary_qgis_adapter.py`, detect a relaxation field alongside `bc_type` and `bc_val`:

```python
relax_field = None
for cand in ("bc_relax", "open_bc_relax", "relax"):
    if cand in fields:
        relax_field = cand
        break
```

When a feature overrides an edge, also read the relaxation value if present; otherwise use the supplied `default_relax`. Return `(bc_type, bc_val, bc_relax)`.

Update `boundary_runtime_logic.collect_boundary_arrays` to accept `default_relax` and return `bc_relax`.

### Step 10 — UI / Workbench / CLI

- Add `open_bc_relax_spin` to `model_tab_view.py` in the Numerics / Stability group, near `front_flux_damping_spin`.
- Collect it in `model_tab_view.collect_params()` as `open_bc_relax_spin`.
- In `run_controller.py`, add `open_bc_relaxation=wp.get("open_bc_relax_spin", 0.0)` to `_build_run_context`.
- In `studio_dialog.py`, read the spin value and pass it as `default_relax` to `collect_boundary_arrays`; store the returned `bc_relax` in `RunContext`.
- In `simulation_worker.py`, pass `open_bc_relaxation` and `bc_relax` to `backend_initializer.initialize`, and call `backend.set_boundary_relaxation(...)` after `initialize()`.
- In `headless_runner.py`, read `rp.get("open_bc_relaxation", 0.0)`; read `bc_relax` from `query_bc_arrays`; pass to `backend.initialize()` and `backend.set_boundary_relaxation()`.

### Step 11 — Remove CPU Twin

Delete `make_ghost()` and `GhostState` from `cpp/src/swe2d_numerics.hpp`. Verify the file still compiles; if `GhostState` is referenced elsewhere, replace those references with `GhostStateLocal` from `swe2d_gpu.cu` or inline the same logic.

---

## Validation Strategy

### Test 1 — Backwards Compatibility at `r = 0.0`

Run the full GPU validation suite with `open_bc_relaxation = 0.0`. Existing results should be unchanged.

### Test 2 — 1D Dam Break with Open Downstream

- Channel 5 km × 100 m, flat bed, 1 m depth upstream of dam at x = 1 km.
- Dam break; downstream BC = OPEN.
- Spatial scheme = WENO5, CFL = 0.45.
- Compare `r = 0.0, 0.1, 0.25, 0.5, 1.0`.
- Expected: boundary depth stays bounded at `r ≥ 0.1`; bulk solution matches analytical within 5% after the bore passes.

### Test 3 — 2D Radial Dam Break + Normal Depth

- Square domain, dam in center, `NORMAL_DEPTH_SLOPE` on all four sides.
- WENO5.
- Verify no runaway inflow at any boundary edge.

### Test 4 — Unaffected BC Types

Unit test that constructs ghost states for WALL, INFLOW_Q, and STAGE with `r > 0` and confirms the ghost state is identical to the `r = 0` case.

### Test 5 — Per-Edge Override

A small mesh with two open boundaries: one with global default `r = 0.0`, one with a `bc_relax = 0.5` override line. Verify the correct edges are damped and the rest are not.

### Test 6 — Mass Conservation

Confirm total mass is conserved at `r > 0` to floating-point tolerance. The blending is linear in conserved quantities, so mass flux is unchanged.

---

## Risks and Trade-offs

| Risk | Mitigation |
|------|------------|
| **WALL / INFLOW_Q / STAGE affected by mistake** | Explicit `bc_type` check; unit test asserts those three are unchanged. |
| **Graph cache not updated after per-edge override** | The kernel reads from a stable device pointer; data uploads are visible to graph replay without invalidation. |
| **Per-edge array memory overhead** | One `double` per edge, allocated once at solver creation. Negligible compared to state arrays. |
| **Floating-point drift in mass balance** | Linear blending of conserved quantities; error is bounded by `r × \|hI - g.h\|` per edge and is sub-μL for `r ≤ 0.5`. |
| **User applies large `r` to a STAGE BC by accident** | STAGE is explicitly excluded. |
| **Default `0.0` keeps current behavior exactly** | The `r > 0.0` guard skips relaxation math when disabled. |

---

## Verification Checklist Before Merging

- [ ] All existing tests pass with `open_bc_relaxation = 0.0`.
- [ ] New `test_open_bc_relaxation.py` covers global scalar, affected BC types, unaffected BC types, and per-edge override.
- [ ] `make_ghost` CPU twin is removed and the project still builds.
- [ ] CHANGELOG entry under `[Unreleased]` describes the new option.
- [ ] `docs/USER_GUIDE.md` BC section mentions the new knob and the `bc_relax` layer field.
- [ ] CI build artifacts build on Linux and Windows.
- [ ] User-visible tooltip reads cleanly with no jargon leaks.
- [ ] `metadata.txt`, `pyproject.toml`, `CHANGELOG.md` versions are NOT bumped.
