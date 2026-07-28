---
type: plan
status: complete
created: 2026-07-22
completed: 2026-07-25
---

# Pipe1D–SWE2D Temporal Coupling Fix

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix temporal integration asymmetry where pipe1d advances as a closed system (no 2D exchange), causing pipe-end cells to overfill and the 2D surface to receive only a fraction of the routed flow.

**Architecture:** Two changes: (1) pass real 2D solver state arrays into the unified face flux during pipe1d's internal Godunov advance so exchange runs in both domains simultaneously; (2) during 2D RK stages, skip the pipe-side atomicAdd to prevent double-counting the exchange on the pipe side. The 2D side re-evaluates the exchange per RK stage for temporal accuracy; the pipe side is corrected once during its own advance.

**Tech Stack:** C++20, CUDA, pybind11, Python 3.12

---

## Status (2026-07-22): Tasks 1–5 implemented; **race-fix Task 7 added after runtime discovery**

Tasks 1–5 below are implemented on `public-sanitize` (commits `0b05fda`, `78400b5`, `243a6fd`).
Tasks 1–5 land the architecture and were verified by mass-balance bookkeeping on CLI replay.

**However**, runtime testing revealed that the unified face flux kernel still had a
**pre-existing data race** that the architecture changes did NOT address. Without this
race fix, the symptom is: 2D cells at the inlet/outlet show correct exchange, but the
pipe **interior** has near-zero flow — because the interior face flux is computed
against a stale `d_A` value (see Task 7 below for full diagnosis). Task 7 fixes this.

The race was **not in the original spec**. It is a pre-existing bug in
`swe2d_unified_face_flux_kernel` that existed before the temporal-coupling work
and was masked when `solver_dev_ptr` was null (the closed-system path skipped
class 3/4/5 atomicAdds entirely). The temporal-coupling changes exposed it by
making the open-system path (with `solver_dev` valid) the default execution path.

### Discovery

Memcheck + racecheck on the QGIS GUI session. `compute-sanitizer --tool=memcheck`
on the user's `sanitizer_memcheck_log.txt` shows only host-side leaks (benign QGIS
shutdown); racecheck was too slow on the GUI mesh to complete in reasonable time.
Static analysis of `swe2d_unified_face_flux_kernel` identified the race directly.

---

### Task 1: Add solver device state ptrs to C++ binding

**Files:**
- Modify: `cpp/src/swe2d_bindings.cpp`

Add a binding function that extracts the solver's device state pointers (`d_h`, `d_hu`, `d_hv`, `d_cell_zb`, `n_cells`, `d_ext_struct_flux_h/d_hu/d_hv`) from a `PySolver` handle and exposes them to Python. Needed so the coupling controller can pass 2D state arrays into the pipe1d step.

- [ ] **Step 1: Add binding `swe2d_get_solver_dev_ptr`**

Insert after the existing `swe2d_get_coupling_dev_ptr` binding (around line 1785):

```cpp
m.def("swe2d_get_solver_dev_ptr",
    [](const std::shared_ptr<PySolver>& ps) -> int64_t {
        if (!ps || !ps->solver || !ps->solver->dev)
            return 0;
        return reinterpret_cast<int64_t>(ps->solver->dev);
    },
    "Return the solver's SWE2DDeviceState* as an int64, or 0 if not initialized. "
    "Used by coupling to pass 2D state arrays into the pipe1d advance.");
```

- [ ] **Step 2: Confirm it builds**

Run: `mamba run -n qgis_stable env CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13 CMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 /usr/bin/cmake --build build -j$(nproc) 2>&1 | tail -20`

Expected: link succeeds, no errors.

---

### Task 2: Add `solver_dev_ptr` param to `swe2d_pipe1d_step` binding

**Files:**
- Modify: `cpp/src/swe2d_bindings.cpp`
- Modify: `cpp/src/pipe1d.cu`
- Modify: `cpp/src/swe2d_gpu.cuh` (if adding stored ptr to state)

Pass the solver's `SWE2DDeviceState*` through the pipe1d step chain so the Godunov internal step can access 2D arrays.

- [ ] **Step 3: Change Python binding signature**

In `swe2d_bindings.cpp`, add `uintptr_t solver_dev_ptr = 0` parameter to the lambda:

```cpp
m.def("swe2d_pipe1d_step",
    [](uintptr_t dev_ptr,
       double dt,
       std::string solver_mode,
       int32_t coupling_substeps,
       int32_t implicit_iters,
       double relaxation,
       double gravity,
       double k_mann,
       double h_min,
       int32_t surcharge_method,
       double theta = 1.0,
       double omega_min = 1.0e-6,
       int32_t friction_method = 0,
       int32_t recon_method = 0,
       int32_t time_integrator = 1,
       double friction_alpha = 0.01,
       uintptr_t solver_dev_ptr = 0) -> void  // NEW
    {
        auto* dev = reinterpret_cast<SWE2DDeviceState*>(dev_ptr);
        auto* solver_dev = solver_dev_ptr
            ? reinterpret_cast<SWE2DDeviceState*>(solver_dev_ptr)
            : nullptr;
        swe2d_pipe1d_step(dev, dt, solver_mode.c_str(), coupling_substeps,
                          implicit_iters, relaxation, gravity, k_mann, h_min,
                          surcharge_method, theta, omega_min, friction_method,
                          recon_method, time_integrator, friction_alpha,
                          solver_dev);
    },
    py::arg("dev_ptr"),
    py::arg("dt"),
    py::arg("solver_mode"),
    py::arg("coupling_substeps"),
    // ... existing args ...
    py::arg("friction_alpha") = 0.01,
    py::arg("solver_dev_ptr") = 0);  // NEW
```

- [ ] **Step 4: Update `swe2d_pipe1d_step` declaration in pipe1d.cu**

Change the signature to accept the optional solver dev:

```cpp
void swe2d_pipe1d_step(
    SWE2DDeviceState* dev,
    double            dt,
    const char*       solver_mode,
    int32_t           coupling_substeps,
    int32_t           implicit_iters,
    double            relaxation,
    double            g,
    double            k_mann,
    double            h_min,
    int32_t           surcharge_method,
    double            theta,
    double            omega_min,
    int32_t           friction_method,
    int32_t           recon_method     /* = 0 */,
    int32_t           time_integrator  /* = 1 */,
    double            friction_alpha   /* = 0.01 */,
    SWE2DDeviceState* solver_dev       /* = nullptr */);
```

- [ ] **Step 5: Forward solver_dev inside `swe2d_pipe1d_step`**

At `pipe1d.cu:3583-3589`, pass solver_dev through to the Godunov step:

```cpp
for (int32_t sub = 0; sub < coupling_substeps; ++sub) {
    swe2d_pipe1d_godunov_step_internal(
        dev, local_dt, g, k_mann, h_min, surcharge_method,
        p.d_flux_Q_scratch, p.d_flux_mom_scratch,
        p.d_A_new_scratch, p.d_Q_new_scratch,
        theta, omega_min, friction_method,
        recon_method, time_integrator, friction_alpha,
        solver_dev);  // NEW
}
```

---

### Task 3: Update `swe2d_pipe1d_godunov_step_internal` to accept solver_dev

**Files:**
- Modify: `cpp/src/pipe1d.cu`

Change the Godunov internal step to accept the solver's device state and pass real 2D arrays + ext_struct_flux buffers to the unified face flux.

- [ ] **Step 6: Change Godunov step signature**

```cpp
static void swe2d_pipe1d_godunov_step_internal(
    SWE2DDeviceState* dev,
    double dt, double g, double k_mann, double h_min,
    int32_t surcharge_method,
    double* d_flux_Q, double* d_flux_mom, double* d_A_new, double* d_Q_new,
    double theta,
    double omega_min,
    int32_t friction_method,
    int32_t recon_method,
    int32_t time_integrator,
    double  friction_alpha,
    SWE2DDeviceState* solver_dev = nullptr)  // NEW
```

- [ ] **Step 7: Replace the two unified face flux calls with real 2D arrays**

At the stage-0 flux call (line 3204), change from:

```cpp
swe2d_gpu_apply_unified_face_flux(dev, dt, g, d_flux_Q, d_flux_mom, stream,
    nullptr, nullptr, nullptr, nullptr,
    nullptr, nullptr, nullptr, nullptr,
    0, 0.0, h_min);
```

To:

```cpp
if (solver_dev) {
    swe2d_gpu_alloc_ext_struct_flux(solver_dev, solver_dev->n_cells);
    swe2d_gpu_apply_unified_face_flux(dev, dt, g, d_flux_Q, d_flux_mom, stream,
        solver_dev->d_h, solver_dev->d_hu, solver_dev->d_hv, solver_dev->d_cell_zb,
        solver_dev->d_ext_struct_flux_h,
        solver_dev->d_ext_struct_flux_hu,
        solver_dev->d_ext_struct_flux_hv,
        dev->pipe1d.d_A,  // ← explicitly pass pipe d_A for pipe-side update
        solver_dev->n_cells, 0.0, h_min);
} else {
    swe2d_gpu_apply_unified_face_flux(dev, dt, g, d_flux_Q, d_flux_mom, stream,
        nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr,
        0, 0.0, h_min);
}
```

Identical change at the stage-1 flux call (line 3256).

---

### Task 4: Remove `d_A_ptr` fallback in host wrapper

**Files:**
- Modify: `cpp/src/pipe1d.cu`

Currently the host wrapper falls back from `d_A_ptr` to `p.d_A` when `d_A_ptr` is nullptr. This means the pipe-side atomicAdd always fires. Change it so `nullptr` means "skip pipe side" — the kernel already checks `if (d_A && ...)` before doing the atomicAdd.

- [ ] **Step 8: Change the fallback logic**

At `pipe1d.cu:3359`, change:

```cpp
double* d_A_inject = d_A_ptr ? d_A_ptr : p.d_A;
```

To:

```cpp
double* d_A_inject = d_A_ptr;  // nullptr → kernel skips pipe-side atomicAdd
```

- [ ] **Step 9: Update pipe1d call sites to pass `p.d_A` explicitly**

Both stage-0 and stage-1 calls in `swe2d_pipe1d_godunov_step_internal` already pass `dev->pipe1d.d_A` from step 7 above (via the `solver_dev` branch). For the no-solver-dev fallback branch (nullptr 2D arrays), keep passing nullptr for d_A_ptr — the old behavior used `p.d_A` as fallback, but without a solver dev, there's no 2D exchange to evaluate, so the call is a no-op for SURFACE_2D faces anyway (they return early at the guard `if (!cell_h_2d ...)`). The fallback-branch d_A_ptr nullptr just means no pipe-side update happens, which is correct since no exchange was computed.

No change needed for the 2D RK stages call site in `swe2d_gpu.cu:5738` — it already passes `nullptr` for d_A_ptr. After step 8, this correctly skips the pipe-side atomicAdd during 2D RK stages.

- [ ] **Step 10: Verify build**

Run: `mamba run -n qgis_stable env CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13 CMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 /usr/bin/cmake --build build -j$(nproc) 2>&1 | tail -30`

Expected: clean build.

---

### Task 5: Python side — pass solver dev ptr from coupling controller

**Files:**
- Modify: `swe2d/runtime/coupling.py`
- Modify: `swe2d/runtime/backend.py`

The coupling controller already stores `self._backend`. Add a method to the backend to expose the solver dev ptr, then pass it to `swe2d_pipe1d_step`.

- [ ] **Step 11: Add `get_solver_dev_ptr` to SWE2DBackend**

In `swe2d/runtime/backend.py`, add:

```python
def get_solver_dev_ptr(self) -> int:
    if self._solver_h is None:
        return 0
    return int(self._mod.swe2d_get_solver_dev_ptr(self._solver_h))
```

- [ ] **Step 12: Pass solver_dev_ptr in coupling.py**

In `swe2d/runtime/coupling.py`, at the `swe2d_pipe1d_step` call site (line 1774), add the solver dev ptr:

```python
# Before the call, get solver dev ptr
_solver_dev_ptr = int(self._backend.get_solver_dev_ptr()) if self._backend is not None else 0

native_mod.swe2d_pipe1d_step(
    dev_ptr,
    float(dt_s),
    str(dsoa.pipe_solver_mode),
    int(getattr(cfg, "coupling_substeps", 1)),
    int(getattr(cfg, "implicit_coupling_iterations", 2)),
    float(getattr(cfg, "implicit_coupling_relaxation", 0.5)),
    float(g),
    float(k_mann),
    float(h_min),
    surcharge_method=int(dsoa.surcharge_method),
    friction_method=int(dsoa.friction_method),
    recon_method=int(dsoa.recon_method),
    time_integrator=int(dsoa.time_integrator),
    friction_alpha=float(dsoa.friction_alpha),
    solver_dev_ptr=_solver_dev_ptr,  # NEW
)
```

No changes needed on the caller side (`runtime_step_executor.py` or `simulation_worker.py`) — they already call `coupling_controller.apply_native_device_sources()` which internally calls `swe2d_pipe1d_step`.

---

### Task 6: Build, test, verify regression

- [ ] **Step 13: Full rebuild**

```bash
find . -type d -name __pycache__ -exec rm -rf {} +
mamba run -n qgis_stable env CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13 CMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 /usr/bin/cmake --build build -j$(nproc)
```

Expected: clean build, no warnings.

- [ ] **Step 14: Run regression tests**

```bash
mamba run -n qgis_stable python3 -m unittest -v \
    tests.test_workbench_gui \
    tests.test_workbench_imports \
    tests.test_workbench_persistence
```

Expected: all pass.

- [ ] **Step 15: Run pipe1d-specific tests**

```bash
mamba run -n qgis_stable python3 -m unittest discover -s tests -p '*pipe1d*' -v
```

Expected: all pass.

- [ ] **Step 16: CLI replay test to verify mass conservation**

Create or use an existing pipe1d+2D replay JSON. Run:

```bash
mamba run -n qgis_stable python3 -m swe2d.cli replay --replay-file test_replay.json 2>&1 | tail -50
```

Check the mass balance log lines. Expected: no mass loss at pipe-end to 2D exchanges (pipe in/out flows should match more closely than before).

- [ ] **Step 17: Commit**

```bash
git add cpp/src/swe2d_bindings.cpp cpp/src/pipe1d.cu swe2d/runtime/coupling.py swe2d/runtime/backend.py
git commit -m "fix(coupling): evaluate pipe1d-2D exchange during pipe1d advance, skip pipe-side during 2D RK stages

Pipe1d Godunov step now receives the 2D solver's device state pointers
so the unified face flux evaluates the SURFACE_2D exchange during the
pipe's own RK stages — not just as a post-hoc correction during 2D
RK stages.  This prevents pipe-end cells from overfilling during the
closed-system advance and then only partially draining.

2D RK stages skip the pipe-side atomicAdd (d_A is not modified) to
avoid double-counting the exchange on the pipe side.  The 2D's own
d_ext_struct_flux_h continues to be re-evaluated per stage for
temporal accuracy.

Also remove the d_A_ptr → p.d_A fallback in the host wrapper so
that nullptr means 'skip pipe-side update', which the kernel
already guards against."
```

---

### Task 7: Two-pass split of `swe2d_unified_face_flux_kernel` (race fix)

**Files:**
- Modify: `cpp/src/pipe1d.cu`

**Why this task exists:** Tasks 1–5 implement the temporal-coupling architecture
correctly, but exposed a pre-existing race in the unified face flux kernel that
was masked when `solver_dev` was null. The race is between `class-3/4/5` threads
that `atomicAdd` to `d_A[L_end]` and `class-0 INTERIOR` threads that read
`d_A[L_end]` (the same cell) non-atomically. The interior face flux
`F = 0.5*(Q_L + Q_R - c_wave*(A_R - A_L))` is computed against a stale
`A_L`/`A_R` depending on GPU scheduling, so the gradient-driven interior flux
that propagates end-cell mass into the pipe is wrong. End-cell mass exchange
still works (atomicAdd survives), but interior `Q` stays near zero — exactly
the user-observed symptom.

**Bug location:** `swe2d_unified_face_flux_kernel` at `cpp/src/pipe1d.cu:1898+`.
The kernel dispatches all 7 face classes (0–6) in a single launch, one thread per face.
- Class 3 (SURFACE_2D_PIPE_END) at line 2431: `atomicAdd(&d_A[L], -fh * dt / L)`
- Class 4 (SURFACE_2D_INLET) at line 2564: `atomicAdd(&d_A[L], Q * dt / L)`
- Class 5 (SURFACE_2D_JUNCTION_OVERFLOW) at line 2615: `atomicAdd(&d_A[L], -Q * dt / L)`
- Class 0 (INTERIOR) at lines 1997–1998: `A_L = cell_A[L]; A_R = cell_A[R];` (non-atomic read)

When an end pipe cell (cell 0 or cell N-1) is `L` for BOTH a class-3 face AND
an adjacent class-0 interior face, the read in the class-0 thread races with the
write in the class-3 thread. This is a single-kernel race on shared memory.

**Was this bug in the original spec?** No. Tasks 1–5 specify the temporal-coupling
architecture (passing `solver_dev_ptr` through to the unified face flux). They do
not mention splitting the unified kernel into two passes. The race was a pre-existing
issue in the unified kernel that was harmless before the temporal-coupling work
because the closed-system path (null `solver_dev`) skipped the class-3/4/5
atomicAdds entirely. Once `solver_dev` is valid in the open-system path, the
race fires.

**Fix:** Two-pass split — pass 1 writes (atomicAdds to `d_A`, writes
`face_F_h`/`face_F_Q` for classes 3/4/5/6); pass 2 reads `d_A` (now consistent)
and writes `face_F_h`/`face_F_Q` for classes 0/1/2. Same kernel, dispatched
twice from the host wrapper with `pass=1` then `pass=2`.

- [ ] **Step 18: Add `pass` parameter to the unified kernel signature**

```cpp
__global__ void swe2d_unified_face_flux_kernel(
    int32_t                     pass,    // 1 = SURFACE_2D (class 3/4/5/6), 2 = pipe (class 0/1/2)
    int32_t                     n_faces,
    // ... (rest unchanged)
);
```

- [ ] **Step 19: Make the prologue conditional on which pass handles the face**

Replace the unconditional zero at the start of the kernel:

```cpp
    const int32_t cls = face_class[k];
    const int32_t solve_mode = (face_solve_mode) ? face_solve_mode[k] : 0;
    const int32_t L = face_owner_L[k];
    const int32_t R = face_owner_R[k];
    const int32_t gi = (face_ghost_idx) ? face_ghost_idx[k] : -1;

    // Default-zero face_F_h / face_F_Q for faces THIS pass handles, so the
    // class branch can write the actual values.  We must not zero faces the
    // OTHER pass handles — pass-1 writes face_F_Q = fh*uL_p for class 3 in
    // direct_inject mode, and pass-2 must not clobber that.
    const bool this_pass_handles =
        (pass == 1 && (cls == 3 || cls == 4 || cls == 5 || cls == 6))
     || (pass == 2 && (cls == 0 || cls == 1 || cls == 2));
    if (this_pass_handles) {
        face_F_h[k] = 0.0;
        face_F_Q[k] = 0.0;
    }
```

Without this guard, pass-2 would zero `face_F_Q` for class-3 faces that pass-1
already wrote, losing the momentum contribution to the fold kernel.

- [ ] **Step 20: Guard each class branch with `pass == N &&`**

```cpp
    if (pass == 2 && cls == 0) { /* INTERIOR */ }
    if (pass == 2 && cls == 1) { /* OUTFALL_BC */ }
    if (pass == 2 && cls == 2) { /* INLET_BC */ }
    if (pass == 1 && cls == 3) { /* SURFACE_2D_PIPE_END */ }
    if (pass == 1 && cls == 4) { /* SURFACE_2D_INLET */ }
    if (pass == 1 && cls == 5) { /* SURFACE_2D_JUNCTION_OVERFLOW */ }
    if (pass == 1 && cls == 6) { /* CULVERT */ }
```

All 7 `if (cls == N)` blocks updated. Threads of the wrong pass return early
without touching `d_A` or `face_F_h`/`face_F_Q`.

- [ ] **Step 21: Launch kernel twice from the host wrapper**

In `swe2d_gpu_apply_unified_face_flux` at `cpp/src/pipe1d.cu:~3413`, replace the
single launch with two sequential launches on the same stream:

```cpp
    // Pass 1: classes 3/4/5/6 — atomicAdd to d_A + d_ext_struct_flux_*.
    swe2d_unified_face_flux_kernel<<<grid, BLOCK, 0, stream>>>(
        1,  // pass=1
        n_faces,
        // ... (full parameter list unchanged)
    );
    // Stream sync ensures pass-1 atomicAdds to d_A are visible to pass-2 reads.
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Pass 2: classes 0/1/2 — reads d_A (now consistent post-pass-1).
    swe2d_unified_face_flux_kernel<<<grid, BLOCK, 0, stream>>>(
        2,  // pass=2
        n_faces,
        // ... (full parameter list, with nullptr for d_structure_flows in pass 2)
    );
    CUDA_CHECK(cudaStreamSynchronize(stream));
```

The two launches share the same `<<<grid, BLOCK, 0, stream>>>` config. Stream
ordering guarantees pass-2 reads post-pass-1 `d_A`. The host-side
`cudaStreamSynchronize` after pass-1 is a defensive belt-and-braces (same as
the original code had after the unified launch).

- [ ] **Step 22: Build, regression test, commit**

```bash
mamba run -n qgis_stable env CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13 \
  CMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 \
  /usr/bin/cmake --build build -j$(nproc) 2>&1 | tail -20
```

Expected: clean build, no new warnings beyond the pre-existing `-Wunused-parameter`
ones.

```bash
mamba run -n qgis_stable python3 -m unittest discover -s tests -p '*pipe1d*' -v
mamba run -n qgis_stable python3 -m swe2d.cli replay \
    --replay-file reference/example_test_project/test_drainage_coupling1.json \
    2>&1 | tail -50
```

Expected after fix: pipe interior cells now show non-zero flow during the
drainage replay; 2D-side exchange unchanged.

```bash
git add cpp/src/pipe1d.cu
git commit -m "fix(pipe1d): two-pass split of unified face flux to eliminate race between SURFACE_2D atomicAdd and INTERIOR read of d_A

The unified face flux kernel dispatched all 7 face classes in one launch.
Class-3/4/5 threads did atomicAdd(&d_A[L_end], ...) for the SURFACE_2D
exchange, while class-0 INTERIOR threads did a non-atomic read of
cell_A[L_end] (== d_A[L_end]) to compute the interior face flux. When an
end pipe cell was L for both a class-3 face and an adjacent class-0 face,
the read raced with the write — interior face flux was computed against
a stale or fresh d_A depending on GPU scheduling, so the gradient-driven
flux that propagates end-cell mass into the pipe interior was wrong.

End-cell mass exchange still worked (atomicAdd survives), but interior
Q stayed near zero — the user-visible symptom of '2D inflow/outflow
correct, pipe interior near-zero flow'.

Fix: split the kernel into pass=1 (classes 3/4/5/6, atomicAdd to d_A)
and pass=2 (classes 0/1/2, reads d_A). Same kernel definition, dispatched
twice from the host wrapper on the same stream. The prologue zero-write
to face_F_h/face_F_Q is now guarded so pass-2 doesn't clobber pass-1's
momentum write for class-3 in direct_inject mode.

This race was a pre-existing bug in the unified kernel that was masked
when solver_dev was null (closed-system path skipped class-3/4/5
atomicAdds). The temporal-coupling work in Tasks 1-5 made the
open-system path the default, exposing the race."
```
