# Implementation Plan: Operator-Split IMEX Source Split, Remove Extreme-Rain Mode and Persistent-Chunk Kernel

## Header

- **Goal:** Remove the broken `extreme_rain_mode` flag and the incomplete persistent-chunk kernel, then implement a clean operator-split IMEX source split where explicit flux/source steps are followed by an implicit friction solve.
- **Architecture:** C++ solver drops dead config and code paths; Python UI/CLI remove associated widgets and parameters; tests are updated and new regression tests added. All changes respect the MVP layer boundaries in `.opencode/rules/MVP_ARCHITECTURE.md`.
- **Tech Stack:** C++17/CUDA 13, pybind11, Python 3.12, PyQt5, QGIS, unittest.

## Requirements

1. Remove `extreme_rain_mode` from C++ config, bindings, and Python layers.
2. Remove persistent-chunk kernel and its C++ dispatch, Python config, and tests.
3. Implement operator-split IMEX: when `source_imex_split` is true, `swe2d_update_kernel` does NOT apply friction; a new `swe2d_implicit_friction_kernel` is called after each explicit stage.
4. Keep `source_cfl_beta`, `source_max_substeps`, `source_rate_cap`, `source_depth_step_cap`, `source_true_subcycling` as the source subcycling mechanism.
5. Update existing tests to remove references to deleted fields.
6. Add regression tests for IMEX and source subcycling.
7. Build and run GPU validation tests.

## Task Structure

### Phase 1: C++ cleanup and IMEX implementation

- [ ] **Task 1.1: Remove `extreme_rain_mode` from the C++ solver**
  - **Files:** `cpp/src/swe2d_solver.hpp`, `cpp/src/swe2d_gpu.cu`, `cpp/src/swe2d_gpu.cuh`, `cpp/src/swe2d_bindings.cpp`
  - Delete `bool extreme_rain_mode = false;` from `SWE2DSolverConfig` (`swe2d_solver.hpp` line 84).
  - Remove the `extreme_rain_mode` bit from `swe2d_kernel_graph_signature` (`swe2d_gpu.cu` lines 129-151).
  - Remove the `extreme_rain_mode` parameter from all GPU step functions and from `swe2d_update_kernel`.
  - In `swe2d_update_kernel`, remove the `extreme_rain_mode &&` guard around the `source_cfl_beta` substep calculation; keep the `source_cfl_beta > 0.0` condition.
  - In `swe2d_update_kernel`, remove the `extreme_rain_mode &&` guard in the non-true-subcycling branch so sources are applied directly.
  - Remove `extreme_rain_mode` from `swe2d_bindings.cpp` lambda and `py::arg` defaults.

- [ ] **Task 1.2: Remove the persistent-chunk kernel and its config**
  - **Files:** `cpp/src/swe2d_solver.hpp`, `cpp/src/swe2d_solver.cpp`, `cpp/src/swe2d_gpu.cu`, `cpp/src/swe2d_gpu.cuh`, `cpp/src/swe2d_bindings.cpp`
  - Delete the `swe2d_persistent_chunk_kernel_first_order`, `swe2d_gpu_step_persistent_chunk`, and `swe2d_gpu_step_rk2_persistent_chunk` functions from `swe2d_gpu.cu` and `swe2d_gpu.cuh`.
  - Delete `tiny_persistent_chunk_substeps`, `tiny_active_compaction_stride_steps`, and `tiny_enable_active_compaction` from `SWE2DSolverConfig` (`swe2d_solver.hpp` lines 103-109).
  - In `swe2d_solver.cpp`, remove `use_tiny_persistent_chunking`, `tiny_chunk_substeps`, and `use_tiny_active_edge_compaction` logic (lines 268-274, 325-354).
  - Map `tiny_mode == 3` (`kTinyModePersistent`) to `kTinyModeOff` (0) in the tiny-mode selection block.
  - Remove `tiny_persistent_chunk_substeps`, `tiny_active_compaction_stride_steps`, and `tiny_enable_active_compaction` from `swe2d_bindings.cpp` lambda and `py::arg` defaults.

- [ ] **Task 1.3: Implement operator-split IMEX source split**
  - **Files:** `cpp/src/swe2d_gpu.cu`, `cpp/src/swe2d_gpu.cuh`
  - Add `swe2d_implicit_friction_kernel` after `swe2d_update_kernel` in `swe2d_gpu.cu`.
  - In `swe2d_update_kernel`, skip the final friction call when `source_imex_split` is true.
  - In `swe2d_update_kernel`, do **not** apply friction inside the `source_true_subcycling` subcycling loop when `source_imex_split` is true.
  - In `swe2d_gpu_step`, after each explicit update stage, call `swe2d_implicit_friction_kernel` if `source_imex_split` is true.
  - Apply the same post-stage friction call in `swe2d_gpu_step_rk2`, `swe2d_gpu_step_rk3`, `swe2d_gpu_step_rk4`, and `swe2d_gpu_step_rk5`.
  - Update `swe2d_gpu.cuh` declarations for the new kernel and changed signatures.

### Phase 2: Python cleanup

- [ ] **Task 2.1: Remove `extreme_rain_mode` from Python**
  - **Files:** `swe2d/runtime/backend.py`, `swe2d/runtime/backend_initializer.py`, `swe2d/runtime/native_binding_compat.py`, `swe2d/workbench/services/run_service.py`, `swe2d/workbench/workers/run_context.py`, `swe2d/workbench/workers/simulation_worker.py`, `swe2d/workbench/controllers/run_controller.py`, `swe2d/workbench/views/model_tab_view.py`, `swe2d/cli/headless_runner.py`, `swe2d/workbench/dialogs/batch_simulation_dialog.py`
  - Remove `extreme_rain_mode` parameter/field references from each file.
  - Remove `extreme_rain_mode_chk` widget construction from `model_tab_view.py` (lines 924-932) and remove it from `collect_params()` (line 1516).
  - Remove `extreme_rain_mode` from `batch_simulation_dialog.py` mapping.
  - Remove `extreme_rain_mode` argument from `backend.initialize()` and `backend_initializer.initialize()` calls.
  - Remove `extreme_rain_mode` from `run_service.collect_run_parameters()`.
  - Remove `extreme_rain_mode` from `native_binding_compat.py` filtered kwargs list.

- [ ] **Task 2.2: Remove persistent-chunk config from Python**
  - **Files:** `swe2d/runtime/backend.py`, `swe2d/runtime/native_binding_compat.py`
  - Remove `tiny_persistent_chunk_substeps`, `tiny_active_compaction_stride_steps`, and `tiny_enable_active_compaction` from `SWE2DBackend.initialize()` signature and from the `native_opts` dict.
  - Remove the `_tiny_persistent_chunk_substeps` attribute and the `tiny_mode == 3` diag-batch-size logic in `SWE2DBackend.run()` (lines 1162-1163).
  - Remove the three persistent-chunk keys from `native_binding_compat.py` filtered kwargs list.

- [ ] **Task 2.3: Update CLI headless runner**
  - **File:** `swe2d/cli/headless_runner.py`
  - Remove `extreme_rain_mode` from the `backend.initialize()` call.
  - Remove any persistent-chunk keys from the `backend.initialize()` call.

### Phase 3: Tests

- [ ] **Task 3.1: Update existing tests**
  - **Files:** `tests/test_model_tab_view.py`, `tests/test_workbench_run_service.py`, `tests/test_swe2d_tiny_mode_dispatch.py`, `tests/test_swe2d_backend_tiny_mode_config.py`
  - Remove `test_view_has_extreme_rain_mode_chk` from `test_model_tab_view.py`.
  - Remove `extreme_rain_mode` argument from `test_workbench_run_service.py`.
  - Remove all persistent-chunk tests from `test_swe2d_tiny_mode_dispatch.py`; keep only the fused-path tests.
  - Remove persistent-chunk assertions from `test_swe2d_backend_tiny_mode_config.py` and update the `diag_batch_size` test to verify zero for all `tiny_mode` values.

- [ ] **Task 3.2: Add IMEX/subcycling regression tests**
  - **File:** `tests/test_swe2d_imex_subcycling.py`
  - Add a closed-box mass-conservation test with rain for:
    - `source_true_subcycling=False`, `source_imex_split=False`
    - `source_true_subcycling=True`, `source_imex_split=False`
    - `source_true_subcycling=True`, `source_imex_split=True`
  - Assert that total mass is conserved within 1% relative to the no-subcycling case.
  - Add a finite-result sanity test for `source_imex_split=True` with friction on a sloped channel.
  - Keep tests small (1–3 s runtime) and GPU-gated with `skipTest` if `gpu_active` is false.

### Phase 4: Build and verification

- [ ] **Task 4.1: Build the C++ extension**
  ```bash
  mamba run -n qgis_stable cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBACKWATER_USE_CUDA=ON
  mamba run -n qgis_stable cmake --build build --target hydra_swe2d -j$(nproc)
  ```
- [ ] **Task 4.2: Purge Python cache and run unit tests**
  ```bash
  find . -type d -name __pycache__ -exec rm -rf {} +
  PYTHONPATH="$PWD:$PWD/build" python3 -m unittest -v \
      tests.test_model_tab_view \
      tests.test_workbench_run_service \
      tests.test_swe2d_backend_tiny_mode_config \
      tests.test_swe2d_imex_subcycling
  ```
- [ ] **Task 4.3: Run GPU validation tests**
  ```bash
  PYTHONPATH="$PWD:$PWD/build" python3 -m unittest -v \
      tests.test_swe2d_gpu_validation_perf \
      tests.test_swe2d_gpu_unstructured
  ```

## Code Snippets

### New implicit friction kernel

```cpp
__global__ __launch_bounds__(256, 4) void swe2d_implicit_friction_kernel(
    int32_t n_cells,
    State* cell_h,
    State* cell_hu,
    State* cell_hv,
    const double* __restrict__ cell_n_mann,
    double dt, double g, double h_min)
{
    int32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_cells) return;

    double h = static_cast<double>(cell_h[c]);
    double hu = static_cast<double>(cell_hu[c]);
    double hv = static_cast<double>(cell_hv[c]);
    apply_friction_cuda_local(h, hu, hv, dt, cell_n_mann[c], g, h_min);
    cell_hu[c] = static_cast<State>(hu);
    cell_hv[c] = static_cast<State>(hv);
}
```

### Update-kernel friction branch

```cpp
if (!source_imex_split) {
    double n_mann = cell_n_mann[c];
    apply_friction_cuda_local(h_new, hu_new, hv_new, dt, n_mann, g, h_min);
}
```

### Source subcycling branch (no friction inside subcycling)

```cpp
if (source_true_subcycling && nsub > 1 && src > 0.0) {
    const double dt_sub = dt / static_cast<double>(nsub);
    for (int k = 0; k < nsub; ++k) {
        h_trial += dt_sub * src;
        if (ext_struct_flux_h) {
            h_trial += dt_sub * ext_struct_flux_h[c] * inv_a;
        }
        if (h_trial < 0.0) h_trial = 0.0;
        // Friction is intentionally NOT applied here in IMEX mode.
    }
}
```

### Python backend `initialize` signature change

Remove these parameters from `SWE2DBackend.initialize()` and from the `native_opts`/`call_solver_create_compat` call:

```python
extreme_rain_mode: bool = False,
tiny_persistent_chunk_substeps: int = 8,
tiny_active_compaction_stride_steps: int = 8,
tiny_enable_active_compaction: bool = True,
```

### Python `run()` diag batching simplification

```python
# Before:
if int(self._tiny_mode) == 3:
    diag_batch_size = max(1, int(self._tiny_persistent_chunk_substeps))

# After: no persistent-mode batching override
```

## Self-Review

- **Requirement coverage:** Every requirement (1–7) has a corresponding task above.
- **No silent fallbacks:** Removing `extreme_rain_mode` means source subcycling is controlled solely by `source_cfl_beta` and `source_true_subcycling`; there is no hidden compatibility shim that re-enables the old broken path.
- **Architecture boundaries:** Python cleanup stays in UI/service layers; C++ cleanup stays in the solver/runtime. No Qt imports move into shared services, and no numpy computation moves into the View.
- **Testability:** The new IMEX kernel is isolated and tested independently through the regression suite.

## Execution Handoff

This plan is written for parallel subagent-driven execution. Dispatch as follows:

```json
{
  "dispatch_table": {
    "cpp-pro": {
      "model": "opencode-go/deepseek-v4-pro",
      "skills": ["fvm-cfd-solver-patterns"]
    },
    "python-pro": {
      "model": "opencode-go/deepseek-v4-flash",
      "skills": ["pyqt5-desktop-patterns", "qgis-plugin-conventions"]
    },
    "test-automator": {
      "model": "opencode-go/deepseek-v4-flash",
      "skills": ["test-driven-development"]
    }
  },
  "steps": [
    { "action": "Remove extreme_rain_mode and persistent-chunk kernel from C++ solver, bindings, and config", "type": "coding", "phase": "1" },
    { "action": "Implement operator-split IMEX source split with a separate implicit friction kernel", "type": "coding", "phase": "1" },
    { "action": "Remove extreme_rain_mode and persistent-chunk config from Python UI, CLI, and services", "type": "coding", "phase": "2" },
    { "action": "Update existing tests and add IMEX/subcycling regression tests", "type": "test", "phase": "3" },
    { "action": "Build C++ extension and run all validation suites", "type": "test", "phase": "4" }
  ]
}
```

**Parallel tracks:**

1. **Track A — C++ (`cpp-pro` / `opencode-go/deepseek-v4-pro`):** Execute Phase 1 (Tasks 1.1–1.3). This is the critical path because it defines the Python/C++ API contract.
2. **Track B — Python cleanup (`python-pro` / `opencode-go/deepseek-v4-flash`):** Execute Phase 2 (Tasks 2.1–2.3). Track B can run in parallel with Track A once the API contract is confirmed above.
3. **Track C — Tests (`test-automator` / `opencode-go/deepseek-v4-flash`):** Execute Phase 3 (Task 3.1). Track C should start after Track A and Track B finish, or can be done in parallel with stubs if the exact API is guaranteed by this plan.
4. **Final verification:** Run Phase 4 after Tracks A, B, and C are complete.
