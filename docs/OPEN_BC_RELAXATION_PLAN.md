# Open-BC Relaxation / Reflection Damping — Implementation Plan

## Problem Statement

For higher-order spatial schemes (MUSCL, WENO5), instability can occur near
**open/outflow boundaries** (BC types `OPEN=4`, `NORMAL_DEPTH=6`,
`NORMAL_DEPTH_SLOPE=7`, `REFLECT=5`) at user-setting CFLs when a hydraulic
jump or bore approaches the boundary.

### Root Cause

Today, the BC types above are handled inside the flux kernel via the
`make_ghost_cuda_local(...)` helper at `cpp/src/swe2d_gpu.cu:269-325` (and
its CPU twin in `cpp/src/swe2d_numerics.hpp:468-547`):

```cpp
case 4: // OPEN — zero-gradient outflow
    g.h  = hI;
    g.hu = huI;
    g.hv = hvI;
    break;
case 6: // NORMAL_DEPTH — prescribed depth
    g.h = (bc_val > h_min) ? bc_val : h_min;
    g.hu = huI;
    g.hv = hvI;
    break;
```

The interior edge uses MUSCL/WENO5 reconstruction (5-cell stencil at
`cpp/src/swe2d_gpu.cu:1987–2059`); the boundary edge skips the limiter and
gets the first-order ghost state above.

When a hydraulic jump propagates into the boundary edge:

1. The interior reconstructed WENO state can overshoot the local bounds
2. That overshoot is fed straight into the Riemann solver (HLLC) against a
   "frozen" ghost state (zero-gradient or prescribed depth)
3. The flux mismatch reflects back into the cell at the boundary
4. The reflected wave reaches the second cell, which reconstructs even harder,
   positive feedback loop

The result is runaway inflow / super-cell `h` growth at the boundary.

---

## Solution: Optional Boundary Ghost-State Relaxation

Add a per-run (optionally per-edge) `open_bc_relaxation` coefficient `r` in
`[0.0, 1.0]` that smoothly blends the constructed ghost state toward the
interior state for outflow-style BCs only.

For `bc_type ∈ {4, 5, 6, 7}`:

```
g.h'  = (1 - r) * g.h  + r * hI
g.hu' = (1 - r) * g.hu + r * huI
g.hv' = (1 - r) * g.hv + r * hvI
```

| `r` | Meaning |
|-----|---------|
| `0.0` | Default — current behavior, no relaxation |
| `0.10 – 0.25` | Mild damping: keep most of prescribed depth / zero-gradient, but smooth the reflection |
| `0.50` | Mid: ghost is halfway between prescribed and interior — strong damping |
| `1.00` | Fully transmissive: ghost equals interior — boundary imposes no condition |

WALL (1), INFLOW_Q (2), and STAGE (3) are deliberately **skipped** —
they impose specific conditions that must be honored. Reflection damping
only applies where the BC is allowing the flow to leave the domain.

---

## Key Design Decisions

1. **Single global scalar with optional per-edge override.** Default value
   stored on `SWE2DSolverConfig`; an optional `d_edge_bc_relax` device
   array allows per-edge override. If the pointer is null, every edge uses
   the global value.

2. **Modified kernel signature changes the graph cache key.** The flux
   kernel already mixes `front_flux_damping` into `get_signature()`. Adding
   `open_bc_relaxation` follows the same pattern. The signature change is
   a one-time cost — after the first re-capture the cached graph replays.

3. **`make_ghost` (CPU twin) is updated identically.** The CPU reference
   kernel in `cpp/src/swe2d_numerics.hpp` is kept in lock-step so the
   reference run agrees with the GPU run.

4. **`h_min` clamp applies to the relaxed ghost depth.** Final relaxed
   depth is clamped to `[h_min, depth_cap]` to match the un-relaxed
   behavior — no separate clamping logic.

5. **Default value `0.0` preserves exact current behavior.** No behavior
   change for users who don't touch the knob.

---

## Files to Modify

### C++ Core

| File | Change |
|------|--------|
| `cpp/src/swe2d_solver.hpp` | Add `double open_bc_relaxation = 0.0;` to `SWE2DSolverConfig` |
| `cpp/src/swe2d_gpu.cuh` | Add `d_edge_bc_relax` device pointer to `SWE2DDeviceState` |
| `cpp/src/swe2d_gpu.cu` | (a) Change `make_ghost_cuda_local` to take `relax` + per-edge `relax_override_ptr`<br>(b) Apply relaxation at the end of switch<br>(c) Add `open_bc_relaxation` to `get_signature()`<br>(d) Pass through to flux kernel call sites (~lines 2089 and 5780) |
| `cpp/src/swe2d_numerics.hpp` | Update CPU `make_ghost()` to match |
| `cpp/src/swe2d_solver.cpp` | Pass relaxation to solver/config; initialize `d_edge_bc_relax` allocation in `swe2d_create_solver` / `swe2d_solver_init` if `open_bc_relaxation > 0` (else leave null) |
| `cpp/src/swe2d_bindings.cpp` | Expose `swe2d_solver_set_open_bc_relaxation(solver, double)` setter for Python + add `swe2d_solver_set_edge_bc_relax(solver, np.ndarray)` for per-edge overrides |

### Python Plumbing

| File | Change |
|------|--------|
| `swe2d/runtime/backend.py` | `set_open_bc_relaxation(r: float)` wrapper around the binding; optional `set_edge_bc_relax(per_edge: np.ndarray)` |
| `swe2d/workbench/workers/run_context.py` | Add `open_bc_relaxation: float = 0.0` to `RunContext` |
| `swe2d/workbench/views/model_tab_view.py` | Add `self.open_bc_relax_spin = QDoubleSpinBox()` in the Numerics / Stability group near `cfl_spin`<br>Object name `open_bc_relax_spin` |
| `swe2d/workbench/controllers/run_controller.py` | Read `wp.get("open_bc_relax_spin", 0.0)` → pass through `_build_run_context` |
| `swe2d/workbench/services/non_gui_runtime_service.py` | Forward `open_bc_relaxation` into the tuple fed to `execute_run_timestep_loop` (or `backend.set_open_bc_relaxation(r)` at solver init) |
| `swe2d/workbench/workers/simulation_worker.py` | Call `backend.set_open_bc_relaxation(ctx.open_bc_relaxation)` once during solver init |
| `swe2d/cli/headless_runner.py` | Read `rp.get("open_bc_relaxation", 0.0)`; pass to solver init |

### UI / Tooltip

The QDoubleSpinBox should have tooltip:

> *Reflection damping at open / normal-depth / reflect boundaries.* Blends the
> constructed ghost state toward the interior state. `0` = current behavior.
> `0.1–0.5` if instability (runaway inflow, NaN h, oscillating jumps) appears
> near the boundary with higher-order schemes. `1` makes the boundary fully
> transmissive (no imposed condition).

---

## Detailed Implementation Outline

### Step 1 — `swe2d_solver.hpp`

```cpp
struct SWE2DSolverConfig {
    // ... existing fields ...

    /// Reflection damping applied at OUTLET-style BCs (OPEN, REFLECT,
    /// NORMAL_DEPTH, NORMAL_DEPTH_SLOPE).  0.0 = disabled (current behavior).
    /// 0.1–0.5 = typical damping range.  1.0 = fully transmissive boundary.
    double  open_bc_relaxation = 0.0;
};
```

### Step 2 — `swe2d_gpu.cuh`

```cpp
struct SWE2DDeviceState {
    // ... existing fields ...

    /// Per-edge relaxation override.  Nullptr → use cfg.open_bc_relaxation.
    /// Allocated by swe2d_solver_set_edge_bc_relax (length = n_edges).
    double* d_edge_bc_relax = nullptr;
};
```

### Step 3 — `swe2d_gpu.cu` — `make_ghost_cuda_local` signature change

**Current:**

```cpp
__device__ __forceinline__ GhostStateLocal make_ghost_cuda_local(
    double hI, double huI, double hvI, double zbI,
    double nx, double ny,
    int bc_type, double bc_val,
    double h_min, double n_mann)
```

**New:**

```cpp
__device__ __forceinline__ GhostStateLocal make_ghost_cuda_local(
    double hI, double huI, double hvI, double zbI,
    double nx, double ny,
    int bc_type, double bc_val,
    double h_min, double n_mann,
    double open_bc_relaxation)         // NEW
{
    GhostStateLocal g{};
    g.zb = zbI;

    switch (bc_type) {
        // ... existing cases 1, 2, 3, 4, 5, 6, 7 unchanged ...
    }

    // Apply reflection damping to OUTLET-style BCs only.
    // Skip WALL / INFLOW_Q / STAGE because they impose specific conditions.
    if (bc_type == 4 || bc_type == 5 || bc_type == 6 || bc_type == 7) {
        if (open_bc_relaxation > 0.0) {
            const double r = (open_bc_relaxation > 1.0) ? 1.0 : open_bc_relaxation;
            g.h  = (1.0 - r) * g.h  + r * hI;
            g.hu = (1.0 - r) * g.hu + r * huI;
            g.hv = (1.0 - r) * g.hv + r * hvI;
        }
    }
    return g;
}
```

**Call site updates** — there are 2 sites:

| Line | Caller |
|------|--------|
| `cpp/src/swe2d_gpu.cu:2089` | Inside the per-RK2-stage reconstruction kernel (the interior/BC-aware fused kernel) |
| `cpp/src/swe2d_gpu.cu:5780` | Inside the non-graph / Euler / SSPRK2 fallback path |

Both pass `open_bc_relaxation` as an additional kernel argument sourced from
`dev->cfg.open_bc_relaxation` (assuming the cfg is mirrored on the device,
which it already is via the `s_cfg_open_bc_relaxation` static global).

### Step 4 — `swe2d_gpu.cu` — `get_signature()`

Add the relaxation term to the signature hash. This forces a one-time
graph re-capture on first use:

```cpp
h = swe2d_mix_u64(h, swe2d_u64_from_double(open_bc_relaxation));
```

(Add to the parameter list and the kernel signature.)

### Step 5 — `swe2d_numerics.hpp` — CPU twin

Mirror the change in `make_ghost()`. Keep behavior identical for `r=0.0`.

### Step 6 — `swe2d_solver.cpp` + `swe2d_bindings.cpp`

Add bindings:

```cpp
m.def("swe2d_solver_set_open_bc_relaxation",
      [](SWE2DSolver* s, double r) {
          if (!s || !s->solver || !s->solver->dev)
              throw std::runtime_error("solver not initialized");
          s->solver->dev->cfg.open_bc_relaxation = r;
          // Invalidate cached graph because signature changed
          swe2d_gpu_invalidate_graph_cache(s->solver->dev);
      },
      py::arg("solver"), py::arg("relaxation"));

m.def("swe2d_solver_set_edge_bc_relax",
      [](SWE2DSolver* s, py::array_t<double, py::array::c_style | py::array::forcecast> relax) {
          if (!s || !s->solver || !s->solver->dev)
              throw std::runtime_error("solver not initialized");
          // Validate length and upload to device buffer
          const int32_t n = (int32_t) relax.size();
          if (n != s->solver->dev->n_edges)
              throw std::runtime_error("relax array must have n_edges elements");
          // allocate d_edge_bc_relax if needed (persistent buffer)
          if (!s->solver->dev->d_edge_bc_relax) {
              CUDA_CHECK(cudaMalloc(&s->solver->dev->d_edge_bc_relax,
                                    static_cast<size_t>(n) * sizeof(double)));
          }
          CUDA_CHECK(cudaMemcpy(s->solver->dev->d_edge_bc_relax, relax.data(),
                                static_cast<size_t>(n) * sizeof(double),
                                cudaMemcpyHostToDevice));
          swe2d_gpu_invalidate_graph_cache(s->solver->dev);
      },
      py::arg("solver"), py::arg("per_edge_relaxation"));
```

### Step 7 — UI

In `model_tab_view.py` after the existing `cfl_spin` row (around line 446):

```python
self.open_bc_relax_spin = QtWidgets.QDoubleSpinBox()
self.open_bc_relax_spin.setObjectName("open_bc_relax_spin")
self.open_bc_relax_spin.setToolTip(
    "Reflection damping at open / normal-depth / reflect boundaries.\n"
    "Blends the constructed ghost state toward the interior state.\n"
    "0.0 = disabled (current behavior).\n"
    "0.1–0.5 if instability (runaway inflow, NaN h, oscillating\n"
    "hydraulic jumps) appears near the boundary with higher-order\n"
    "schemes (MUSCL, WENO5). 1.0 = fully transmissive boundary.\n"
    "WALL / INFLOW_Q / STAGE BCs are NOT affected."
)
self.open_bc_relax_spin.setRange(0.0, 1.0)
self.open_bc_relax_spin.setDecimals(3)
self.open_bc_relax_spin.setSingleStep(0.05)
self.open_bc_relax_spin.setValue(0.0)
self._add_param_row(form, "Open BC relax:", self.open_bc_relax_spin)
```

Surface into `collect_params()`:

```python
"open_bc_relax_spin": float(self.open_bc_relax_spin.value()),
```

### Step 8 — `run_controller.py`

In `_build_run_context(...)`:

```python
open_bc_relaxation=float(wp.get("open_bc_relax_spin", 0.0)),
```

### Step 9 — `simulation_worker.py` + `headless_runner.py`

In `_WorkbenchShim` initialization for the solver, after the existing
`backend.initialize(...)`:

```python
backend.set_open_bc_relaxation(float(ctx.open_bc_relaxation))
```

Equivalent in `headless_runner.execute_run()`.

---

## Validation Strategy

### Test 1 — 1D Dam Break with Open Downstream
- Channel 5 km × 100 m, bed slope 0, 1 m deep upstream of dam at x=1 km
- Dam break; downstream BC = OPEN
- Spatial scheme = WENO5, CFL = 0.45 (the previously-problematic setting)
- Run for 30 min sim time
- Compare `r = 0.0, 0.1, 0.25, 0.5, 1.0`
- Expected: depth at boundary cell stays bounded at `r ≥ 0.1`; reference
  analytical approach should be matched within 5% in the bulk after the bore
  has passed

### Test 2 — 2D Radial Dam Break + Normal Depth
- Square domain, dam in center, `NORMAL_DEPTH_SLOPE` on all four sides
- WENO5
- Verify no runaway inflow at any boundary edge

### Test 3 — Backwards Compatibility
- Existing test suite must pass at `r = 0.0` with **no behavior change**
  (bit-equal `h/hu/hv` evolution)
- Run the full test suite at `r = 0.0` to confirm

### Test 4 — Mass Conservation
- Verify that mass is still conserved at `r > 0` (the relaxation is linear
  blending, mass balance should be preserved to floating-point tolerance)

---

## Roll-out Strategy

Phase 1 — internal-only at `r = 0.0`:
1. Add the config knob (default 0.0), wire through bindings, Python, UI.
2. Run full test suite — confirm zero behavior change.
3. Build CI artifacts.

Phase 2 — enable in dev testing:
1. Open the channel flow test case at `r = 0.1`, compare against an
   analytical solution (Ritter solution or equivalent)
2. If accuracy is comparable (within 5%), advance `r = 0.25` and re-test
3. Document sweet spot in UI tooltip

Phase 3 — promote to general availability:
1. CHANGELOG entry under `[1.2.x]` with the new knob
2. Update `docs/USER_GUIDE.md` BC section with recommendation
3. Add reference images / plots if helpful

---

## Risks and Trade-offs

| Risk | Mitigation |
|------|------------|
| **WALL / INFLOW_Q / STAGE affected by mistake** | Explicit check `bc_type ∈ {4,5,6,7}` inside the relaxation block; unit test asserts those three are unchanged |
| **Graph cache invalidation hurt perf** | Only invalidates once when the setter is called, not every step |
| **Floating-point drift in mass balance** | Linear blending of conserved quantities; error bounded by r × |hI - g.h|, summed across edges and timesteps; worst-case error is sub-μL for r ≤ 0.5 |
| **User applies large r to a STAGE BC by accident** | The check explicitly excludes STAGE — relax is only applied to outflow-side BCs |
| **Default of 0.0 keeps current behavior exactly** | Floating-point operations short-circuit when r=0 (the compiler emits `0 * g.h + 0 * hI` which folds to `g.h` if compiler is sensible; if not, the `r > 0.0` branch is skipped entirely) |

---

## Why Per-Edge Override?

The global knob covers 95% of cases. Per-edge override is included for:

- A model with an open downstream BC and an internal culvert that must keep
  imposing Stage; the user can globally enable reflection damping at 0.2
  and override zero for the culvert's cells.

The cost of supporting per-edge is essentially zero (one extra pointer +
a `d_edge_bc_relax ? per_edge[e] : cfg.open_bc_relaxation` ternary in the
relaxation block) — keep it as the design above.

---

## Verification Checklist Before Merging

- [ ] All existing tests pass with `open_bc_relaxation = 0.0`
- [ ] New `test_open_bc_relaxation.py` covers the four cases (open,
      reflect, normal_depth, normal_depth_slope) plus the three
      unaffected types (wall, inflow_q, stage)
- [ ] CHANGELOG entry under `[Unreleased]` describes the new option
- [ ] `docs/USER_GUIDE.md` mentions the new knob in the BC section
- [ ] CI build artifacts build on both linux-2022 and windows-2022
- [ ] User-visible string (tooltip) reads cleanly with no jargon leaks
- [ ] `metadata.txt`, `pyproject.toml`, `CHANGELOG.md` versions NOT bumped
      (this is an enhancement, not a release)
