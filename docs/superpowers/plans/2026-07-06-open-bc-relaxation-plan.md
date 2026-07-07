# Open-BC Relaxation / Reflection Damping — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a global `open_bc_relaxation` knob plus per-edge overrides from the BC line layer that damp reflected waves at OPEN / REFLECT / NORMAL_DEPTH / NORMAL_DEPTH_SLOPE boundaries.

**Architecture:** The global scalar is passed through `SWE2DSolverConfig` at solver create time and used to initialize a per-edge device array. The flux kernels read from that array and blend the constructed ghost state toward the interior state for outflow-style BCs. Per-edge overrides are uploaded via a binding that takes `(edge_index, relax)` arrays. The unused CPU `make_ghost()` twin in `swe2d_numerics.hpp` is removed.

**Tech Stack:** C++17 / CUDA, pybind11, Python 3.12, PyQt5 (QGIS workbench UI), NumPy.

---

## Task 1: Add `open_bc_relaxation` to `SWE2DSolverConfig`

**Files:**
- Modify: `cpp/src/swe2d_solver.hpp:77-80`

- [ ] **Step 1: Add the field**

```cpp
    // Wet/dry front stability controls
    double  front_flux_damping = 0.5;     // momentum-flux scale factor on wet/dry front edges (0=full damp, 1=none)
    double  open_bc_relaxation = 0.0;     // reflection damping at outflow-style BCs (0=disabled, 1=fully transmissive)
    bool    active_set_hysteresis = true; // keep cells active 1 extra step after drying to suppress oscillatory front switching
```

- [ ] **Step 2: Verify no build break**

Run: `mamba run -n qgis_stable python -c "print('ok')"`
Expected: ok

---

## Task 2: Extend `SWE2DDeviceState` and `swe2d_gpu_init`

**Files:**
- Modify: `cpp/src/swe2d_gpu.cuh:68`, `cpp/src/swe2d_gpu.cuh:617-624`
- Modify: `cpp/src/swe2d_gpu.cu:4643`, `cpp/src/swe2d_gpu.cu:10926-10937`
- Modify: `cpp/src/swe2d_solver.cpp:87-91`

- [ ] **Step 1: Add the device pointer**

In `cpp/src/swe2d_gpu.cuh` after `d_edge_bc_val`:

```cpp
    double*  d_edge_bc_val = nullptr;
    double*  d_edge_bc_relax = nullptr;  // per-edge reflection-damping coefficient
```

- [ ] **Step 2: Update the `swe2d_gpu_init` declaration and definition**

In `cpp/src/swe2d_gpu.cuh:617-624`:

```cpp
SWE2DDeviceState* swe2d_gpu_init(
    const SWE2DMesh& mesh,
    const double*    h0,
    const double*    hu0,
    const double*    hv0,
    const double*    n_mann_cell,
    int              degen_mode   = 0,
    double           max_inv_area = 1.0e6,
    double           open_bc_relaxation = 0.0);
```

In `cpp/src/swe2d_gpu.cu:4605-4610`, add the parameter to the definition:

```cpp
SWE2DDeviceState* swe2d_gpu_init(
    const SWE2DMesh& mesh,
    const double*    h0,
    const double*    hu0,
    const double*    hv0,
    const double*    n_mann_cell,
    int              degen_mode,
    double           max_inv_area,
    double           open_bc_relaxation)
{
```

- [ ] **Step 3: Allocate and fill the array in `swe2d_gpu_init`**

In `cpp/src/swe2d_gpu.cu` after `d_edge_bc_val` allocation (around line 4643):

```cpp
    alloc_d(reinterpret_cast<void**>(&dev->d_edge_bc_relax), sz_edges * sizeof(double));
    {
        std::vector<double> edge_bc_relax(sz_edges, open_bc_relaxation);
        copy_h2d_d(dev->d_edge_bc_relax, edge_bc_relax.data(), sz_edges);
    }
```

- [ ] **Step 4: Free the array in `swe2d_gpu_destroy`**

In `cpp/src/swe2d_gpu.cu:10937` after `safe_free(dev->d_edge_bc_val);`:

```cpp
    safe_free(dev->d_edge_bc_val);
    safe_free(dev->d_edge_bc_relax);
```

- [ ] **Step 5: Pass the scalar from `swe2d_create`**

In `cpp/src/swe2d_solver.cpp`:

```cpp
        s->dev = swe2d_gpu_init(mesh,
                                s->h.data(), s->hu.data(), s->hv.data(),
                                s->n_mann_cell.data(),
                                cfg.degen_mode,
                                cfg.max_inv_area,
                                cfg.open_bc_relaxation);
```

- [ ] **Step 6: Compile-check the C++ changes**

Run: `mamba run -n qgis_stable python -c "import sys; print(sys.version)"`
(Real CUDA compile happens in Task 13.)

---

## Task 3: Modify `make_ghost_cuda_local`

**Files:**
- Modify: `cpp/src/swe2d_gpu.cu:269-333`

- [ ] **Step 1: Split WALL/REFLECT and add the relaxation parameter**

Replace the existing function with:

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
        case 1: { // WALL: reflect normal velocity (NOT relaxed)
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
        case 5: { // REFLECT: same construction as WALL, then relaxed
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

    // Apply reflection damping only to outflow-style BCs.
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

- [ ] **Step 2: Verify the function compiles mentally**

Check: `fmin` is available from `<cmath>`/CUDA math. `GhostStateLocal` is defined earlier in the file.

---

## Task 4: Add `edge_bc_relax` to the Flux Kernel Signature

**Files:**
- Modify: `cpp/src/swe2d_gpu.cu:1815-1849`

- [ ] **Step 1: Add the parameter**

After `edge_bc_val` in `swe2d_flux_kernel`:

```cpp
    const int32_t* __restrict__ edge_bc,
    const double*  __restrict__ edge_bc_val,
    const double*  __restrict__ edge_bc_relax,
    const State*   __restrict__ cell_h,
```

- [ ] **Step 2: Update the call to `make_ghost_cuda_local` in the flux kernel**

Around line 2089:

```cpp
        const double n_local = cell_n_mann ? cell_n_mann[c0] : 0.03;
        const double edge_relax = edge_bc_relax[e];
        GhostStateLocal gs = make_ghost_cuda_local(
            hL, huL, hvL, zbL, nx, ny,
            edge_bc[e], edge_bc_val[e], h_min, n_local, edge_relax);
```

- [ ] **Step 3: Update the persistent-chunk kernel**

This kernel (`swe2d_persistent_chunk_kernel_first_order`) starts around line 5660. Add `edge_bc_relax` to its signature after `edge_bc_val` and update the `make_ghost_cuda_local` call around line 5780:

```cpp
                    const double n_local = cell_n_mann ? cell_n_mann[c0] : 0.03;
                    const double edge_relax = edge_bc_relax[e];
                    GhostStateLocal gs = make_ghost_cuda_local(
                        hL, huL, hvL, zbL, nx, ny,
                        edge_bc[e], edge_bc_val[e], h_min, n_local, edge_relax);
```

Then add the device pointer to the cooperative-launch `args[]` array around line 6218:

```cpp
        &dev->d_edge_bc,
        &dev->d_edge_bc_val,
        &dev->d_edge_bc_relax,
        &dev->d_cell_owned_offsets,
```

- [ ] **Step 4: Find and update any remaining `make_ghost_cuda_local` call sites**

Run: `rg "make_ghost_cuda_local" cpp/src/swe2d_gpu.cu -n`
Expected: only the definition and the two updated call sites above.
If more appear, apply the same pattern.

---

## Task 5: Pass `d_edge_bc_relax` at Kernel Launch Sites

**Files:**
- Modify: `cpp/src/swe2d_gpu.cu:5288-5310`, `cpp/src/swe2d_gpu.cu:5491-5510`

- [ ] **Step 1: Update the first flux kernel launch**

Around line 5288:

```cpp
                swe2d_flux_kernel<<<grid_flux, BLOCK, 0, dev->d_stream>>>(
                    n_edges,
                    dev->d_edge_c0, dev->d_edge_c1,
                    dev->d_edge_nx, dev->d_edge_ny, dev->d_edge_len,
                    dev->d_edge_mx, dev->d_edge_my,
                    dev->d_edge_bc, dev->d_edge_bc_val, dev->d_edge_bc_relax,
                    dev->d_h, dev->d_hu, dev->d_hv,
                    dev->d_n_mann_cell,
                    dev->d_cell_zb, dev->d_cell_inv_area,
                    dev->d_cell_cx, dev->d_cell_cy,
                    dev->d_grad,
                    dev->d_flux_h, dev->d_flux_hu, dev->d_flux_hv,
                    dev->d_flux_hu_r, dev->d_flux_hv_r,
                    d_dbg_fh, d_dbg_fhu, d_dbg_fhv,
                    spatial_scheme,
                    g, h_min, max_inv_area,
                    momentum_cap_min_speed, momentum_cap_celerity_mult,
                    dev->d_degen_mask, dev->d_merge_owner, degen_mode,
                    dev->d_active, front_flux_damping, shallow_damping_depth,
                    enable_shallow_front_recon_fallback);
```

- [ ] **Step 2: Update the second flux kernel launch**

Around line 5491:

```cpp
        swe2d_flux_kernel<<<grid, BLOCK, 0, dev->d_stream>>>(
            n_edges,
            dev->d_edge_c0, dev->d_edge_c1,
            dev->d_edge_nx, dev->d_edge_ny, dev->d_edge_len,
            dev->d_edge_mx, dev->d_edge_my,
            dev->d_edge_bc, dev->d_edge_bc_val, dev->d_edge_bc_relax,
            dev->d_h, dev->d_hu, dev->d_hv,
            dev->d_n_mann_cell,
            dev->d_cell_zb, dev->d_cell_inv_area,
            dev->d_cell_cx, dev->d_cell_cy,
            dev->d_grad,
            dev->d_flux_h, dev->d_flux_hu, dev->d_flux_hv,
            dev->d_flux_hu_r, dev->d_flux_hv_r,
            d_dbg_fh, d_dbg_fhu, d_dbg_fhv,
            spatial_scheme,
            g, h_min, max_inv_area,
            momentum_cap_min_speed, momentum_cap_celerity_mult,
            dev->d_degen_mask, dev->d_merge_owner, degen_mode,
            dev->d_active, front_flux_damping, shallow_damping_depth,
            enable_shallow_front_recon_fallback);
```

- [ ] **Step 3: Update the persistent-chunk kernel launch**

Find the launch of `swe2d_persistent_chunk_kernel_first_order` and add `dev->d_edge_bc_relax` after `dev->d_edge_bc_val` in the argument list.

Run: `rg "swe2d_persistent_chunk_kernel_first_order" cpp/src/swe2d_gpu.cu -n -A20`
Locate the launch and edit it.

---

## Task 6: Remove the Unused CPU `make_ghost` and `GhostState`

**Files:**
- Modify: `cpp/src/swe2d_numerics.hpp:460-555`

- [ ] **Step 1: Verify `make_ghost` is not called anywhere**

Run: `rg "make_ghost\(" cpp/src --include="*.{cu,cuh,hpp,cpp}" -n`
Expected: only the definition in `swe2d_numerics.hpp` and the CUDA-local definition in `swe2d_gpu.cu`.

- [ ] **Step 2: Delete the CPU ghost code**

Remove from `cpp/src/swe2d_numerics.hpp`:

```cpp
// ─────────────────────────────────────────────────────────────────────────────
// Ghost cell states for boundary edges
...
// ─────────────────────────────────────────────────────────────────────────────

struct GhostState {
    double h, hu, hv, zb;
};

// BCType values passed as int to stay CUDA-compatible without enum class overhead
/** Construct a ghost cell state ...
...
SWE2D_HOSTDEV inline GhostState make_ghost(
...
}

} // namespace swe2d
```

Delete the entire block from `struct GhostState` through `make_ghost()`.

- [ ] **Step 3: Verify `GhostState` is not referenced elsewhere**

Run: `rg "GhostState" cpp/src --include="*.{cu,cuh,hpp,cpp}" -n`
Expected: only `GhostStateLocal` in `swe2d_gpu.cu`.
If `GhostState` appears elsewhere, replace with `GhostStateLocal` or inline the struct.

- [ ] **Step 4: Commit the C++ core changes**

```bash
git add cpp/src/swe2d_solver.hpp cpp/src/swe2d_gpu.cuh cpp/src/swe2d_gpu.cu cpp/src/swe2d_numerics.hpp cpp/src/swe2d_solver.cpp
git commit -m "feat(cpp): add open_bc_relaxation device array and remove CPU ghost twin"
```

---

## Task 7: Expose `open_bc_relaxation` and `set_edge_bc_relax` in Bindings

**Files:**
- Modify: `cpp/src/swe2d_bindings.cpp:2209-2393`

- [ ] **Step 1: Add the scalar to the create-solver signature**

After `double front_flux_damping,` in the lambda parameter list and after `bool active_set_hysteresis,` in `py::arg` list, add:

Parameter list:
```cpp
               double front_flux_damping,
               double open_bc_relaxation,
               bool   active_set_hysteresis,
```

`cfg` assignment after `cfg.front_flux_damping = front_flux_damping;`:
```cpp
            cfg.front_flux_damping = front_flux_damping;
            cfg.open_bc_relaxation = open_bc_relaxation;
            cfg.active_set_hysteresis = active_set_hysteresis;
```

`py::arg` list:
```cpp
        py::arg("front_flux_damping") = 0.5,
        py::arg("open_bc_relaxation") = 0.0,
        py::arg("active_set_hysteresis") = true,
```

- [ ] **Step 2: Add the per-edge override binding**

After the `swe2d_solver_set_boundary_values` binding (around line 2098) add:

```cpp
    // ── Per-edge BC relaxation overrides ─────────────────────────────────────
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

Add the helper kernel near the other small kernels in `swe2d_gpu.cu` (e.g., after `swe2d_apply_boundary_updates_kernel`):

```cpp
__global__ void swe2d_apply_edge_relax_kernel(
    int32_t n_updates,
    const int32_t* __restrict__ edge_index,
    const double*  __restrict__ relax,
    double*        __restrict__ d_edge_bc_relax)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_updates) {
        d_edge_bc_relax[edge_index[i]] = relax[i];
    }
}
```

- [ ] **Step 3: Commit binding changes**

```bash
git add cpp/src/swe2d_bindings.cpp cpp/src/swe2d_gpu.cu
git commit -m "feat(bindings): expose open_bc_relaxation and per-edge override"
```

---

## Task 8: Update `backend.py` — `initialize()` and `set_boundary_relaxation()`

**Files:**
- Modify: `swe2d/runtime/backend.py:235-246`, `swe2d/runtime/backend.py:880-1042`, `swe2d/runtime/backend.py:475-508`

- [ ] **Step 1: Add feature-availability check in `__init__`**

After `self._supports_solver_external_sources = ...`:

```python
        self._supports_solver_edge_bc_relax = log_feature_unavailable(
            self._mod, "swe2d_solver_set_edge_bc_relax", logger,
        )
```

- [ ] **Step 2: Add `open_bc_relaxation` to `initialize()` signature**

After `front_flux_damping: float = 0.5,` add:

```python
        front_flux_damping: float = 0.5,
        open_bc_relaxation: float = 0.0,
        active_set_hysteresis: bool = True,
```

Also add it to the docstring.

- [ ] **Step 3: Pass it to the native create call**

In the `native_opts` dict or directly in the `call_solver_create_compat` call (after `front_flux_damping=float(front_flux_damping),`):

```python
            front_flux_damping=float(front_flux_damping),
            open_bc_relaxation=float(open_bc_relaxation),
            active_set_hysteresis=bool(active_set_hysteresis),
```

- [ ] **Step 4: Add `set_boundary_relaxation()` method**

After `set_boundary_conditions()` add:

```python
    def set_boundary_relaxation(
        self,
        bc_edge_node0: np.ndarray,
        bc_edge_node1: np.ndarray,
        bc_edge_relax: np.ndarray,
    ) -> None:
        """Upload per-edge open-bc relaxation overrides.

        Parameters
        ----------
        bc_edge_node0, bc_edge_node1 : array_like int32, shape (E,)
            Endpoint node indices for boundary edges to update.
        bc_edge_relax : array_like float64, shape (E,)
            Relaxation coefficient per edge in [0.0, 1.0].
        """
        if not self._boundary_edge_index_by_nodes:
            return
        if not self._supports_solver_edge_bc_relax:
            logger.warning("swe2d_solver_set_edge_bc_relax not available; ignoring per-edge relaxation overrides")
            return
        if self._solver_h is None:
            raise RuntimeError("initialize() must be called before set_boundary_relaxation().")
        n0 = np.ascontiguousarray(bc_edge_node0, dtype=np.int32).ravel()
        n1 = np.ascontiguousarray(bc_edge_node1, dtype=np.int32).ravel()
        r = np.ascontiguousarray(bc_edge_relax, dtype=np.float64).ravel()
        if not (n0.size == n1.size == r.size):
            raise ValueError("bc edge relax arrays must have the same length")
        if n0.size == 0:
            return
        edge_index = np.empty(n0.size, dtype=np.int32)
        for i in range(n0.size):
            a = int(n0[i])
            b = int(n1[i])
            key = (a, b) if a < b else (b, a)
            if key not in self._boundary_edge_index_by_nodes:
                raise ValueError(f"Boundary edge ({a}, {b}) not found in mesh")
            edge_index[i] = self._boundary_edge_index_by_nodes[key]
        self._mod.swe2d_solver_set_edge_bc_relax(self._solver_h, edge_index, r)
```

- [ ] **Step 5: Commit**

```bash
git add swe2d/runtime/backend.py
git commit -m "feat(backend): add open_bc_relaxation parameter and set_boundary_relaxation"
```

---

## Task 9: Update `RunContext`

**Files:**
- Modify: `swe2d/workbench/workers/run_context.py:73-94`

- [ ] **Step 1: Add the fields**

After `front_flux_damping: float = 0.0`:

```python
    front_flux_damping: float = 0.0
    open_bc_relaxation: float = 0.0
    active_set_hysteresis: bool = False
```

After `bc_vl: np.ndarray`:

```python
    bc_vl: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    bc_relax: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
```

- [ ] **Step 2: Commit**

```bash
git add swe2d/workbench/workers/run_context.py
git commit -m "feat(run_context): add open_bc_relaxation and bc_relax"
```

---

## Task 10: Update `model_tab_view.py` — UI Spinbox

**Files:**
- Modify: `swe2d/workbench/views/model_tab_view.py:1005-1016`, `swe2d/workbench/views/model_tab_view.py:1518-1520`

- [ ] **Step 1: Add the spinbox after `front_flux_damping_spin`**

```python
        self.front_flux_damping_spin = QtWidgets.QDoubleSpinBox()
        self.front_flux_damping_spin.setObjectName("front_flux_damping_spin")
        self.front_flux_damping_spin.setToolTip(
            "Damping factor applied to fluxes at wet/dry fronts (0–1). "
            "Higher values = more damping = more stability at front. "
            "Default: 0.5. Increase if front oscillations occur."
        )
        self._add_param_row(form, "Front flux damping:", self.front_flux_damping_spin)
        self.front_flux_damping_spin.setRange(0.0, 1.0)
        self.front_flux_damping_spin.setDecimals(2)
        self.front_flux_damping_spin.setSingleStep(0.05)
        self.front_flux_damping_spin.setValue(0.5)

        self.open_bc_relax_spin = QtWidgets.QDoubleSpinBox()
        self.open_bc_relax_spin.setObjectName("open_bc_relax_spin")
        self.open_bc_relax_spin.setToolTip(
            "Reflection damping at open / normal-depth / reflect boundaries.\n"
            "Blends the constructed ghost state toward the interior state.\n"
            "0.0 = disabled (current behavior).\n"
            "0.1–0.5 if instability (runaway inflow, NaN h, oscillating\n"
            "hydraulic jumps) appears near the boundary with higher-order\n"
            "schemes (MUSCL, WENO5). 1.0 = fully transmissive boundary.\n"
            "WALL / INFLOW_Q / STAGE BCs are NOT affected.\n"
            "Per-edge override can be set with a 'bc_relax' field on the BC line layer."
        )
        self._add_param_row(form, "Open BC relax:", self.open_bc_relax_spin)
        self.open_bc_relax_spin.setRange(0.0, 1.0)
        self.open_bc_relax_spin.setDecimals(3)
        self.open_bc_relax_spin.setSingleStep(0.05)
        self.open_bc_relax_spin.setValue(0.0)
```

- [ ] **Step 2: Add it to the batch dialog map**

In `swe2d/workbench/dialogs/batch_simulation_dialog.py`, add to `_WIDGET_TO_CLI_MAP`:

```python
    "front_flux_damping_spin": "front_flux_damping",
    "open_bc_relax_spin": "open_bc_relaxation",
    "k_mann_spin": "k_mann",
```

- [ ] **Step 3: Collect the value in `collect_params()`**

In the dict returned by `collect_params()`, after `"front_flux_damping_spin"`:

```python
            "front_flux_damping_spin": float(self.front_flux_damping_spin.value()),
            "open_bc_relax_spin": float(self.open_bc_relax_spin.value()),
            "active_set_hysteresis_chk": bool(self.active_set_hysteresis_chk.isChecked()),
```

- [ ] **Step 4: Commit**

```bash
git add swe2d/workbench/views/model_tab_view.py swe2d/workbench/dialogs/batch_simulation_dialog.py
git commit -m "feat(ui): add open_bc_relax_spin control"
```

---

## Task 11: Update `run_controller.py`

**Files:**
- Modify: `swe2d/workbench/controllers/run_controller.py:288-289`

- [ ] **Step 1: Pass the scalar to `RunContext`**

After `front_flux_damping=wp["front_flux_damping_spin"],`:

```python
            front_flux_damping=wp["front_flux_damping_spin"],
            open_bc_relaxation=wp.get("open_bc_relax_spin", 0.0),
            active_set_hysteresis=wp["active_set_hysteresis_chk"],
```

- [ ] **Step 2: Store `bc_relax` in `RunContext` from boundary collection**

Find where `_collect_boundary_arrays()` is called and its results are unpacked. Update the unpacking to include `bc_relax` and store it in `RunContext`.

Look for lines around `bc_n0, bc_n1, bc_type_preview, bc_val_preview = view._collect_boundary_arrays()` (line 696). Update to:

```python
        bc_n0, bc_n1, bc_type_preview, bc_val_preview, bc_relax_preview = view._collect_boundary_arrays()
```

Then find where `RunContext` is constructed and add:

```python
            bc_n0=run_input.bc_n0,
            bc_n1=run_input.bc_n1,
            bc_tp=run_input.bc_tp,
            bc_vl=run_input.bc_vl,
            bc_relax=run_input.bc_relax,
```

- [ ] **Step 3: Update `run_data_builder.py` to carry `bc_relax`**

In `swe2d/runtime/run_data_builder.py`:

Add `bc_relax` to `SWE2DRunInputData`:

```python
    bc_vl: np.ndarray
    bc_relax: np.ndarray
    side_hydrographs: Dict[str, object]
```

Update the callback type annotation:

```python
        collect_boundary_arrays_callback: Callable[[], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
```

Update the unpacking and return:

```python
        bc_n0, bc_n1, bc_tp, bc_vl, bc_relax = self._collect_boundary_arrays_callback()
        ...
        return SWE2DRunInputData(
            ...
            bc_vl=bc_vl,
            bc_relax=bc_relax,
            ...
        )
```

- [ ] **Step 4: Commit**

```bash
git add swe2d/workbench/controllers/run_controller.py swe2d/runtime/run_data_builder.py
git commit -m "feat(run_controller): wire open_bc_relaxation and bc_relax into RunContext"
```

---

## Task 12: Update `boundary_qgis_adapter.py` — Read `bc_relax` from Layer

**Files:**
- Modify: `swe2d/boundary_and_forcing/boundary_qgis_adapter.py:94-299`

- [ ] **Step 1: Update the function signature**

Change `apply_bc_layer_overrides_qgis` to accept `default_relax` and return `bc_relax`:

```python
def apply_bc_layer_overrides_qgis(
    *,
    mesh_data: Optional[Dict[str, np.ndarray]],
    have_qgis_core: bool,
    bc_lines_layer_combo: Any,
    combo_layer_fn: Callable[[Any, str], Any],
    edge_n0: np.ndarray,
    edge_n1: np.ndarray,
    bc_type: np.ndarray,
    bc_val: np.ndarray,
    default_relax: float = 0.0,
    qgs_geometry_cls: Any,
    qgs_pointxy_cls: Any,
    log_fn: Callable[[str], None],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
```

Update the docstring return type to `Tuple[np.ndarray, np.ndarray, np.ndarray]` for `(bc_type, bc_val, bc_relax)`.

- [ ] **Step 2: Detect the relaxation field**

After detecting `val_field`:

```python
    relax_field = None
    for cand in ("bc_relax", "open_bc_relax", "relax"):
        if cand in fields:
            relax_field = cand
            break
```

- [ ] **Step 3: Initialize and populate `bc_relax`**

After `bc_type = bc_type.copy(); bc_val = bc_val.copy()` (or equivalent), add:

```python
    bc_relax = np.full(edge_n0.size, float(default_relax), dtype=np.float64)
```

When reading a feature, also read its relaxation value:

```python
        v = 0.0
        if val_field is not None:
            ...
        r = default_relax
        if relax_field is not None:
            try:
                raw_r = ft[relax_field]
                if raw_r is None or (isinstance(raw_r, str) and raw_r.strip().upper() == 'NULL'):
                    r = default_relax
                else:
                    r = float(str(raw_r))
                    r = max(0.0, min(1.0, r))
            except Exception as e:
                log_fn(f"[ERROR] BC relax field parse failed: {e}")
                r = default_relax
        features.append((pr, geom, t, v, r))
```

- [ ] **Step 4: Apply the relaxation override alongside BC overrides**

Update the unpacking and assignment in both the strict and fallback loops:

```python
        if best is not None:
            t, v, r = best
            changed = (int(bc_type[i]) != int(t)) or (not np.isclose(float(bc_val[i]), float(v)))
            bc_type[i] = int(t)
            bc_val[i] = float(v)
            bc_relax[i] = float(r)
            if changed:
                applied += 1
```

Return:

```python
    return bc_type, bc_val, bc_relax
```

- [ ] **Step 5: Commit**

```bash
git add swe2d/boundary_and_forcing/boundary_qgis_adapter.py
git commit -m "feat(bc_adapter): read bc_relax field from BC override layer"
```

---

## Task 13: Update `boundary_runtime_logic.py` — Return `bc_relax`

**Files:**
- Modify: `swe2d/boundary_and_forcing/boundary_runtime_logic.py:57-107`

- [ ] **Step 1: Update the function signature and callback type**

```python
def collect_boundary_arrays(
    *,
    mesh_data: Optional[Dict[str, np.ndarray]],
    mesh_boundary_edges_fn: Callable[[], Tuple[np.ndarray, np.ndarray]],
    default_bc_type: int = 0,
    default_relax: float = 0.0,
    apply_bc_layer_overrides_fn: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float], Tuple[np.ndarray, np.ndarray, np.ndarray]],
    log_fn: Callable[[str], None],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
```

- [ ] **Step 2: Update the docstring and return**

Update the docstring return type to `Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]` and the description to include `bc_relax`.

- [ ] **Step 3: Pass `default_relax` and unpack `bc_relax`**

```python
    bc_type, bc_val = _compute_default_bc(mesh_data, edge_n0, edge_n1, default_bc_type=default_bc_type)
    bc_type, bc_val, bc_relax = apply_bc_layer_overrides_fn(edge_n0, edge_n1, bc_type, bc_val, default_relax)
    return edge_n0, edge_n1, bc_type, bc_val, bc_relax
```

- [ ] **Step 4: Commit**

```bash
git add swe2d/boundary_and_forcing/boundary_runtime_logic.py
git commit -m "feat(bc_runtime): collect and return bc_relax from BC layer"
```

---

## Task 14: Update `studio_dialog.py` — Pass Default and Unpack `bc_relax`

**Files:**
- Modify: `swe2d/workbench/studio_dialog.py:1711-1724`, `swe2d/workbench/studio_dialog.py:2110-2119`

- [ ] **Step 1: Update `_apply_bc_layer_overrides` to pass `default_relax` and return `bc_relax`**

```python
    def _apply_bc_layer_overrides(self, edge_n0, edge_n1, bc_type, bc_val, default_relax=0.0):
        """Apply boundary condition overrides from BC vector layer."""
        from swe2d.boundary_and_forcing.boundary_qgis_adapter import apply_bc_layer_overrides_qgis as _logic
        return _logic(
            mesh_data=self._mesh_data, have_qgis_core=_HAVE_QGIS_CORE,
            bc_lines_layer_combo=getattr(self._model_tab_view, "bc_lines_layer_combo", None),
            combo_layer_fn=self._combo_layer,
            edge_n0=edge_n0, edge_n1=edge_n1, bc_type=bc_type, bc_val=bc_val,
            default_relax=float(default_relax),
            qgs_geometry_cls=QgsGeometry, qgs_pointxy_cls=QgsPointXY, log_fn=self._log,
        )
```

- [ ] **Step 2: Update `_collect_boundary_arrays` to read the spin and return `bc_relax`**

```python
    def _collect_boundary_arrays(self):
        """Collect boundary condition arrays from the model tab view."""
        from swe2d.boundary_and_forcing.boundary_runtime_logic import collect_boundary_arrays as _logic
        default_bc_type = 0
        default_bc_combo = getattr(self._model_tab_view, "default_bc_type_combo", None)
        if default_bc_combo is not None:
            default_bc_type = int(default_bc_combo.currentData())
        default_relax = 0.0
        relax_spin = getattr(self._model_tab_view, "open_bc_relax_spin", None)
        if relax_spin is not None:
            default_relax = float(relax_spin.value())
        return _logic(
            mesh_data=self._mesh_data,
            mesh_boundary_edges_fn=self._mesh_boundary_edges,
            default_bc_type=default_bc_type,
            default_relax=default_relax,
            apply_bc_layer_overrides_fn=self._apply_bc_layer_overrides,
            log_fn=self._log,
        )
```

- [ ] **Step 3: Commit**

```bash
git add swe2d/workbench/studio_dialog.py
git commit -m "feat(studio_dialog): pass default_relax and collect bc_relax"
```

---

## Task 15: Update `backend_initializer.py` — Pass `open_bc_relaxation` and `bc_relax`

**Files:**
- Modify: `swe2d/runtime/backend_initializer.py:32-175`

- [ ] **Step 1: Add parameters to `build_and_initialize`**

After `front_flux_damping: float,`:

```python
        front_flux_damping: float,
        open_bc_relaxation: float,
        bc_relax: np.ndarray,
```

- [ ] **Step 2: Pass `open_bc_relaxation` to `b.initialize()`**

After `front_flux_damping=float(front_flux_damping),`:

```python
            front_flux_damping=float(front_flux_damping),
            open_bc_relaxation=float(open_bc_relaxation),
            active_set_hysteresis=bool(active_set_hysteresis),
```

- [ ] **Step 3: Call `set_boundary_relaxation` after `initialize()`**

Before `return b`:

```python
        if bc_relax is not None and bc_relax.size > 0:
            b.set_boundary_relaxation(bc_n0, bc_n1, bc_relax)
        return b
```

- [ ] **Step 4: Commit**

```bash
git add swe2d/runtime/backend_initializer.py
git commit -m "feat(backend_initializer): apply open_bc_relaxation and bc_relax"
```

---

## Task 16: Update `simulation_worker.py` — Pass Values to Initializer

**Files:**
- Modify: `swe2d/workbench/workers/simulation_worker.py:458-463`

- [ ] **Step 1: Add the arguments to `_build_and_initialize_backend`**

After `front_flux_damping=ctx.front_flux_damping,`:

```python
                    front_flux_damping=ctx.front_flux_damping,
                    open_bc_relaxation=ctx.open_bc_relaxation,
                    bc_relax=ctx.bc_relax,
                    active_set_hysteresis=ctx.active_set_hysteresis,
```

- [ ] **Step 2: Commit**

```bash
git add swe2d/workbench/workers/simulation_worker.py
git commit -m "feat(simulation_worker): pass open_bc_relaxation and bc_relax to initializer"
```

---

## Task 17: Update `headless_runner.py` — CLI Path

**Files:**
- Modify: `swe2d/cli/headless_runner.py:235-272`, `swe2d/cli/headless_runner.py:346-357`

- [ ] **Step 1: Extract `bc_relax` from BC arrays**

Update `_bc_arrays_from_dict` to return `bc_relax`:

```python
    def _bc_arrays_from_dict(d: Dict[str, np.ndarray]):
        return (
            d.get("bc_edge_node0", np.empty(0, dtype=np.int32)),
            d.get("bc_edge_node1", np.empty(0, dtype=np.int32)),
            d.get("bc_edge_type", np.empty(0, dtype=np.int32)),
            d.get("bc_edge_val", np.empty(0, dtype=np.float64)),
            d.get("bc_relax", np.empty(0, dtype=np.float64)),
        )
```

Update `_valid_bc_arrays` to:

```python
    def _valid_bc_arrays(n0, n1, tp, vl, rl):
        return n0.size > 0 and n0.size == n1.size == tp.size == vl.size == rl.size
```

Update the unpacking to include `bc_relax`:

```python
    bc_n0, bc_n1, bc_tp, bc_vl, bc_relax = _bc_arrays_from_dict(bc)
    if _valid_bc_arrays(bc_n0, bc_n1, bc_tp, bc_vl, bc_relax):
        ...
        bc_n0, bc_n1, bc_tp, bc_vl, bc_relax = _bc_arrays_from_dict(mesh_data)
    else:
        bc_n0, bc_n1, bc_tp, bc_vl, bc_relax = _bc_arrays_from_dict(mesh_data)
```

- [ ] **Step 2: Pass `open_bc_relaxation` to `backend.initialize()`**

After `front_flux_damping=float(rp.get("front_flux_damping", 0.5)),`:

```python
        front_flux_damping=float(rp.get("front_flux_damping", 0.5)),
        open_bc_relaxation=float(rp.get("open_bc_relaxation", 0.0)),
        active_set_hysteresis=bool(rp.get("active_set_hysteresis", True)),
```

- [ ] **Step 3: Call `set_boundary_relaxation()` after `set_boundary_conditions()`**

After `backend.set_boundary_conditions(bc_n0, bc_n1, bc_tp, bc_vl)`:

```python
    if bc_relax.size > 0:
        try:
            backend.set_boundary_relaxation(bc_n0, bc_n1, bc_relax)
        except Exception as _e:
            logger.warning("Failed to set boundary relaxation: %s", _e)
```

- [ ] **Step 4: Commit**

```bash
git add swe2d/cli/headless_runner.py
git commit -m "feat(headless): support open_bc_relaxation and bc_relax in CLI"
```

---

## Task 18: Update `gpkg_adapter.py` — Read `bc_relax` Column

**Files:**
- Modify: `swe2d/cli/gpkg_adapter.py:157-235`

- [ ] **Step 1: Detect and read `bc_relax` in pre-split edge table**

```python
    if {"node0", "node1", "bc_type", "bc_val"}.issubset(col_names_lower):
        relax_col = None
        for cand in ("bc_relax", "open_bc_relax", "relax"):
            if cand in col_names_lower:
                relax_col = cand
                break
        cols = ["node0", "node1", "bc_type", "bc_val"]
        if relax_col:
            cols.append(relax_col)
        cur.execute(
            f'SELECT {", ".join(cols)} FROM "{bc_table}" ORDER BY rowid'
        )
        rows = cur.fetchall()
        if rows:
            out = {
                "bc_edge_node0": np.array([r[0] for r in rows], dtype=np.int32),
                "bc_edge_node1": np.array([r[1] for r in rows], dtype=np.int32),
                "bc_edge_type": np.array([r[2] for r in rows], dtype=np.int32),
                "bc_edge_val": np.array([r[3] for r in rows], dtype=np.float64),
            }
            if relax_col:
                out["bc_relax"] = np.array(
                    [0.0 if r[4] is None else float(r[4]) for r in rows],
                    dtype=np.float64,
                )
            return out
```

- [ ] **Step 2: Detect and read `bc_relax` in geometry-based table**

After detecting `bc_col` and `val_col`:

```python
    relax_col = next((c for c, _ in col_info if c.lower() in ("bc_relax", "open_bc_relax", "relax")), None)
```

Update the query and output to include `bc_relax`:

```python
    q_cols = f"\"{geom_col}\""
    if bc_col:
        q_cols += f", \"{bc_col}\""
    if val_col:
        q_cols += f", \"{val_col}\""
    if relax_col:
        q_cols += f", \"{relax_col}\""

    cur.execute(f"SELECT {q_cols} FROM \"{bc_table}\" ORDER BY rowid")

    all_n0, all_n1, all_type, all_val, all_relax = [], [], [], [], []
    for row in cur.fetchall():
        geom_raw = row[0]
        bt = int(row[1]) if bc_col and len(row) > 1 and row[1] is not None else 1
        bv = float(row[2]) if val_col and len(row) > 2 and row[2] is not None else 0.0
        br = 0.0
        if relax_col and len(row) > 3 and row[3] is not None:
            try:
                br = float(row[3])
            except Exception:
                br = 0.0
        ...
        for i in range(len(node_ids) - 1):
            all_n0.append(node_ids[i])
            all_n1.append(node_ids[i + 1])
            all_type.append(bt)
            all_val.append(bv)
            all_relax.append(br)
        ...

    return {
        "bc_edge_node0": np.array(all_n0, dtype=np.int32),
        "bc_edge_node1": np.array(all_n1, dtype=np.int32),
        "bc_edge_type": np.array(all_type, dtype=np.int32),
        "bc_edge_val": np.array(all_val, dtype=np.float64),
        "bc_relax": np.array(all_relax, dtype=np.float64),
    }
```

- [ ] **Step 3: Commit**

```bash
git add swe2d/cli/gpkg_adapter.py
git commit -m "feat(gpkg_adapter): read optional bc_relax column from BC table"
```

---

## Task 19: Compile the C++ Extension

**Files:**
- All C++ files modified above

- [ ] **Step 1: Run the build**

```bash
mamba run -n qgis_stable python setup.py build_ext --inplace
```

Expected: build succeeds with no errors.

- [ ] **Step 2: Fix any compile errors**

If errors occur, read the error message, fix the offending file, and re-run until the build succeeds.

- [ ] **Step 3: Commit once clean**

```bash
git commit -m "build: compile open_bc_relaxation C++ changes"
```

---

## Task 20: Add Tests

**Files:**
- Create: `tests/test_open_bc_relaxation.py`

- [ ] **Step 1: Create the test file**

```python
import numpy as np
import pytest

from swe2d.runtime.backend import SWE2DBackend, BCType


def _build_simple_channel(backend):
    """Build a 1x5 rectangular channel. Return boundary edge arrays."""
    node_x = np.array([0.0, 100.0, 200.0, 300.0, 400.0, 500.0], dtype=np.float64)
    node_y = np.array([0.0, 0.0, 0.0, 100.0, 100.0, 100.0], dtype=np.float64)
    node_z = np.zeros_like(node_x)
    cell_nodes = np.array([
        [0, 1, 3], [1, 4, 3], [1, 2, 4], [2, 5, 4],
    ], dtype=np.int32)
    # Boundary edges: left, right, bottom, top
    bc_n0 = np.array([0, 2, 0, 3], dtype=np.int32)
    bc_n1 = np.array([3, 5, 1, 4], dtype=np.int32)
    bc_tp = np.full(bc_n0.size, BCType.OPEN, dtype=np.int32)
    bc_vl = np.zeros(bc_n0.size, dtype=np.float64)
    backend.build_mesh(
        node_x, node_y, node_z, cell_nodes,
        bc_edge_node0=bc_n0, bc_edge_node1=bc_n1,
        bc_edge_type=bc_tp, bc_edge_val=bc_vl,
    )
    return bc_n0, bc_n1, bc_tp, bc_vl


def test_open_bc_relaxation_zero_no_change():
    """At r=0 the solution must match the existing no-relaxation path."""
    h0 = np.zeros(4, dtype=np.float64)
    h0[0] = 1.0

    backend = SWE2DBackend()
    _build_simple_channel(backend)
    backend.initialize(h0=h0, n_mann=0.03, h_min=1e-4, cfl=0.45, dt_max=1.0,
                       open_bc_relaxation=0.0)
    d1 = backend.step()
    h1, hu1, hv1 = backend.get_state()

    backend2 = SWE2DBackend()
    _build_simple_channel(backend2)
    backend2.initialize(h0=h0, n_mann=0.03, h_min=1e-4, cfl=0.45, dt_max=1.0)
    d2 = backend2.step()
    h2, hu2, hv2 = backend2.get_state()

    assert d1["dt"] == pytest.approx(d2["dt"])
    np.testing.assert_array_almost_equal(h1, h2)
    np.testing.assert_array_almost_equal(hu1, hu2)
    np.testing.assert_array_almost_equal(hv1, hv2)


def test_open_bc_relaxation_affects_open_only():
    """WALL/INFLOW_Q/STAGE ghosts are unchanged even with r>0."""
    h0 = np.zeros(4, dtype=np.float64)
    h0[0] = 1.0

    bc_n0, bc_n1, bc_tp, bc_vl = _build_simple_channel(SWE2DBackend())
    # left=WALL, right=OPEN, bottom/top unchanged
    bc_tp = bc_tp.copy()
    bc_tp[0] = BCType.WALL
    bc_tp[1] = BCType.OPEN

    backend = SWE2DBackend()
    backend.build_mesh(
        np.array([0.0, 100.0, 200.0, 300.0, 400.0, 500.0]),
        np.array([0.0, 0.0, 0.0, 100.0, 100.0, 100.0]),
        np.zeros(6),
        np.array([[0, 1, 3], [1, 4, 3], [1, 2, 4], [2, 5, 4]], dtype=np.int32),
        bc_edge_node0=bc_n0, bc_edge_node1=bc_n1,
        bc_edge_type=bc_tp, bc_edge_val=bc_vl,
    )
    backend.initialize(h0=h0, n_mann=0.03, h_min=1e-4, cfl=0.45, dt_max=1.0,
                       open_bc_relaxation=0.5)
    backend.step()
    h_relax, _, _ = backend.get_state()

    backend2 = SWE2DBackend()
    backend2.build_mesh(
        np.array([0.0, 100.0, 200.0, 300.0, 400.0, 500.0]),
        np.array([0.0, 0.0, 0.0, 100.0, 100.0, 100.0]),
        np.zeros(6),
        np.array([[0, 1, 3], [1, 4, 3], [1, 2, 4], [2, 5, 4]], dtype=np.int32),
        bc_edge_node0=bc_n0, bc_edge_node1=bc_n1,
        bc_edge_type=bc_tp, bc_edge_val=bc_vl,
    )
    backend2.initialize(h0=h0, n_mann=0.03, h_min=1e-4, cfl=0.45, dt_max=1.0,
                        open_bc_relaxation=0.0)
    backend2.step()
    h_no_relax, _, _ = backend2.get_state()

    wall_cell = 0
    assert h_relax[wall_cell] == pytest.approx(h_no_relax[wall_cell])


def test_open_bc_relaxation_per_edge_override():
    """Per-edge overrides via set_boundary_relaxation are applied."""
    h0 = np.zeros(4, dtype=np.float64)
    h0[0] = 1.0

    bc_n0, bc_n1, bc_tp, bc_vl = _build_simple_channel(SWE2DBackend())

    backend = SWE2DBackend()
    backend.build_mesh(
        np.array([0.0, 100.0, 200.0, 300.0, 400.0, 500.0]),
        np.array([0.0, 0.0, 0.0, 100.0, 100.0, 100.0]),
        np.zeros(6),
        np.array([[0, 1, 3], [1, 4, 3], [1, 2, 4], [2, 5, 4]], dtype=np.int32),
        bc_edge_node0=bc_n0, bc_edge_node1=bc_n1,
        bc_edge_type=bc_tp, bc_edge_val=bc_vl,
    )
    backend.initialize(h0=h0, n_mann=0.03, h_min=1e-4, cfl=0.45, dt_max=1.0,
                       open_bc_relaxation=0.0)
    relax = np.full(bc_n0.size, 0.5, dtype=np.float64)
    backend.set_boundary_relaxation(bc_n0, bc_n1, relax)
    backend.step()
    h1, _, _ = backend.get_state()

    backend2 = SWE2DBackend()
    backend2.build_mesh(
        np.array([0.0, 100.0, 200.0, 300.0, 400.0, 500.0]),
        np.array([0.0, 0.0, 0.0, 100.0, 100.0, 100.0]),
        np.zeros(6),
        np.array([[0, 1, 3], [1, 4, 3], [1, 2, 4], [2, 5, 4]], dtype=np.int32),
        bc_edge_node0=bc_n0, bc_edge_node1=bc_n1,
        bc_edge_type=bc_tp, bc_edge_val=bc_vl,
    )
    backend2.initialize(h0=h0, n_mann=0.03, h_min=1e-4, cfl=0.45, dt_max=1.0,
                        open_bc_relaxation=0.5)
    backend2.step()
    h2, _, _ = backend2.get_state()

    np.testing.assert_array_almost_equal(h1, h2)
```

- [ ] **Step 2: Run the new tests**

```bash
mamba run -n qgis_stable pytest tests/test_open_bc_relaxation.py -v
```

Expected: tests pass. If not, fix the implementation.

- [ ] **Step 3: Commit**

```bash
git add tests/test_open_bc_relaxation.py
git commit -m "test: add open_bc_relaxation coverage"
```

---

## Task 21: Run the Validation Suite

**Files:**
- All modified files

- [ ] **Step 1: Run GPU validation tests**

```bash
mamba run -n qgis_stable pytest tests/test_swe2d_gpu_validation_perf.py tests/test_swe2d_gpu_unstructured.py -v
```

Expected: all tests pass.

- [ ] **Step 2: Run the full test suite**

```bash
mamba run -n qgis_stable pytest tests/ -x
```

Expected: all tests pass. If failures are related to the new code, fix them.

- [ ] **Step 3: Commit**

```bash
git commit -m "test: confirm open_bc_relaxation passes validation suite"
```

---

## Task 22: Update Documentation

**Files:**
- Modify: `CHANGELOG.md`, `docs/USER_GUIDE.md`

- [ ] **Step 1: Add CHANGELOG entry**

Under `[Unreleased]`:

```markdown
### Added
- New `open_bc_relaxation` stability knob (Numerics / Stability) that damps reflections at OPEN, REFLECT, NORMAL_DEPTH, and NORMAL_DEPTH_SLOPE boundaries. Per-edge overrides can be supplied via a `bc_relax` field on the BC line layer.
```

- [ ] **Step 2: Update `docs/USER_GUIDE.md` BC section**

Add a short paragraph explaining:
- The global `Open BC relax` spinbox.
- The `bc_relax` field on BC override lines (values 0.0–1.0).
- That WALL/INFLOW_Q/STAGE are unaffected.
- When to use it (higher-order schemes with boundary instabilities).

- [ ] **Step 3: Commit**

```bash
git add CHANGELOG.md docs/USER_GUIDE.md
git commit -m "docs: document open_bc_relaxation knob and bc_relax layer field"
```

---

## Task 23: Final Verification and Cache Cleanup

**Files:**
- All modified files

- [ ] **Step 1: Purge Python caches**

```bash
find . -type d -name __pycache__ -exec rm -rf {} +
```

- [ ] **Step 2: Run lint and typecheck if available**

Check for available commands in `pyproject.toml` or existing scripts. Common ones:

```bash
mamba run -n qgis_stable python -m ruff check swe2d tests
mamba run -n qgis_stable python -m pyright swe2d/runtime/backend.py swe2d/workbench/workers/run_context.py
```

If `ruff` / `pyright` are not installed, run the Python test imports instead:

```bash
mamba run -n qgis_stable python -c "import swe2d.runtime.backend; import swe2d.workbench.workers.run_context; import swe2d.runtime.run_data_builder"
```

Expected: imports succeed with no errors.

- [ ] **Step 3: Final status check**

```bash
git status --short
```

Expected: all intended files modified; no stray untracked files.

- [ ] **Step 4: Commit**

```bash
git commit -m "chore: final cleanup after open_bc_relaxation implementation"
```

---

## Self-Review Checklist

Before claiming the plan is complete, run this checklist mentally:

- [ ] **Spec coverage:** Every section of the design spec has at least one task here.
- [ ] **Placeholder scan:** No "TBD", "TODO", "implement later", or vague "add error handling" steps.
- [ ] **Type consistency:** `open_bc_relaxation` is consistently named across C++, Python, and UI. `bc_relax` is the per-edge array name everywhere.
- [ ] **No missing call sites:** Both `make_ghost_cuda_local` call sites and both flux-kernel launches are covered.
- [ ] **Graph cache:** No graph-signature changes are required because the kernel reads from device memory.

Gaps found during self-review: `run_data_builder.py` and `batch_simulation_dialog.py` were added to carry `bc_relax` and expose the new spinbox to batch runs.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-06-open-bc-relaxation-plan.md`.

Two execution options:

1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — execute tasks in this session using `executing-plans`, batch execution with checkpoints.

Which approach would you like?
