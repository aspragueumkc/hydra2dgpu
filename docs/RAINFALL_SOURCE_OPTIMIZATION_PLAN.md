# Rainfall Source Term Re-evaluation Optimization

## Goal
Optimize the rainfall excess calculation by performing it only once per minute. The calculated average excess rate for that minute will then be applied as a constant source term across all timesteps within that minute, simplifying Runge-Kutta (RK) integration and avoiding complex per-substep state management for cumulative rainfall.

## Principles
- **KISS (Keep It Simple, Stupid):** Simplify the integration of rainfall into the RK scheme.
- **YAGNI (You Aren't Gonna Need It):** Avoid unnecessary higher-order integration for a term that has inherent empirical limitations and negligible higher-order physical effects in a depth-averaged model.
- **Separation of Concerns:** Clearly separate the infrequent update of the rainfall excess rate from its application in the high-frequency RK substeps.

## Plan

### 1. Steps

```json
[
  {
    "action": "Introduce new device buffers for minute-based rainfall state tracking.",
    "type": "refactor",
    "phase": "0_init",
    "agent": "cpp-pro",
    "model": "claude-3-5-sonnet"
  },
  {
    "action": "Modify SWE2DDeviceState struct in `swe2d_gpu.cuh` to include new buffers.",
    "type": "refactor",
    "phase": "0_init",
    "agent": "cpp-pro",
    "model": "claude-3-5-sonnet"
  },
  {
    "action": "Update `swe2d_gpu_alloc_rainfall` and deallocation to manage the new buffers.",
    "type": "refactor",
    "phase": "0_init",
    "agent": "cpp-pro",
    "model": "claude-3-5-sonnet"
  },
  {
    "action": "Update `swe2d_create` and `swe2d_destroy` in `swe2d_solver.cpp` to initialize/free the new buffers.",
    "type": "refactor",
    "phase": "0_init",
    "agent": "cpp-pro",
    "model": "claude-3-5-sonnet"
  },
  {
    "action": "Create a new GPU kernel to update the minute-based rainfall source rate.",
    "type": "cuda",
    "phase": "1_kernel_dev",
    "agent": "cpp-pro",
    "model": "claude-3-5-sonnet"
  },
  {
    "action": "Implement the logic within `swe2d_update_rain_source_rate_kernel`.",
    "type": "cuda",
    "phase": "1_kernel_dev",
    "agent": "cpp-pro",
    "model": "claude-3-5-sonnet"
  },
  {
    "action": "Modify `swe2d_gpu_step` and all `swe2d_gpu_step_rkX` functions to call the new update kernel.",
    "type": "cuda",
    "phase": "2_integration",
    "agent": "cpp-pro",
    "model": "claude-3-5-sonnet"
  },
  {
    "action": "Replace per-substep `swe2d_build_rain_cn_source_kernel` calls with the new constant rate.",
    "type": "cuda",
    "phase": "2_integration",
    "agent": "cpp-pro",
    "model": "claude-3-5-sonnet"
  },
  {
    "action": "Remove the previous 'save/restore' logic for cumulative rain state from RK step functions.",
    "type": "refactor",
    "phase": "2_integration",
    "agent": "cpp-pro",
    "model": "claude-3-5-sonnet"
  },
  {
    "action": "Update relevant unit and integration tests to reflect new rainfall behavior.",
    "type": "test",
    "phase": "3_validation",
    "agent": "test-automator",
    "model": "claude-3-5-sonnet"
  },
  {
    "action": "Perform full regression tests to ensure no unintended side effects.",
    "type": "test",
    "phase": "3_validation",
    "agent": "test-automator",
    "model": "claude-3-5-sonnet"
  }
]
```

### 2. Pre-computed Agent and Model Assignments (from `auto_agent_selector`)

(Assignments are included in the `steps` JSON above.)

### 3. Machine-Readable JSON Block for Plugin Hook

```json
{
  "plugin_hook": "post_plan_generated",
  "plan_details": {
    "title": "Rainfall Source Term Re-evaluation Optimization",
    "phases": [
      "0_init: Setup new state buffers",
      "1_kernel_dev: Develop minute-based update kernel",
      "2_integration: Integrate into RK steps",
      "3_validation: Test and validate changes"
    ],
    "affected_files": [
      "cpp/src/swe2d_gpu.cuh",
      "cpp/src/swe2d_gpu.cu",
      "cpp/src/swe2d_solver.cpp",
      "tests/*.py"
    ]
  }
}
```

### 4. Superpowers Workflow

-   **`FVM / CFD Solver Patterns`**: Essential for understanding the existing time integration schemes, GPU kernel structures, and how source terms are applied within the solver. This skill will guide the modification of RK step functions and the overall approach to integrating the new rainfall logic.
-   **`GPU Test Diagnostics`**: Crucial for diagnosing any numerical instabilities, correctness issues, or performance regressions that may arise during the integration and validation phases. This includes interpreting CUDA error messages, analyzing kernel output, and debugging parallel execution.
-   **`Python-Pro`**: For coordinating the overall development process, handling file operations (like saving this plan), and potentially assisting with test harness modifications if needed.
-   **`Cpp-Pro`**: Directly responsible for all C++ and CUDA kernel development, including modifying existing structs, writing new kernels, and updating solver dispatch logic.
-   **`Test-Automator`**: Dedicated to updating, creating, and running the necessary tests to ensure the changes are correct and do not introduce regressions.

## Detailed Implementation Notes

### Phase 0: Initial Setup and State Management

-   **`SWE2DDeviceState` (`swe2d_gpu.cuh`)**:
    Add members for:
    -   `d_last_rain_update_time`: `double*` (stores last simulation time when rain rate was updated, per cell).
    -   `d_current_rain_source_rate_mps`: `double*` (stores the constant excess rainfall rate in m/s, per cell, for the current minute interval).

-   **`swe2d_gpu_alloc_rainfall` (`swe2d_gpu.cu`)**:
    -   Allocate device memory for `d_last_rain_update_time` and `d_current_rain_source_rate_mps`.
    -   Initialize `d_last_rain_update_time` to `-1.0` (or `t_start`) to trigger an immediate update.
    -   Initialize `d_current_rain_source_rate_mps` to `0.0`.

-   **Deallocation and `swe2d_destroy` (`swe2d_gpu.cu`, `swe2d_solver.cpp`)**:
    -   Free the newly allocated device buffers.

### Phase 1: Minute-Based Rainfall Re-evaluation Kernel

-   **New Kernel Definition (`swe2d_gpu.cu`)**:
    ```cpp
    __global__ void swe2d_update_rain_source_rate_kernel(
        double t_now,
        double dt, // The current dt for the full step
        double* d_rain_cum_mm,
        double* d_rain_excess_cum_mm,
        const double* d_rain_input_mms, // Raw rainfall input rate (mm/s)
        const double* d_cn_params_s,    // S parameter for SCS-CN
        double* d_last_rain_update_time,
        double* d_current_rain_source_rate_mps,
        int n_cells,
        double minute_interval_seconds // 60.0
    );
    ```

-   **Kernel Logic**:
    For each cell:
    1.  Read `t_now`, `d_last_rain_update_time[c]`.
    2.  Check `if (t_now >= d_last_rain_update_time[c] + minute_interval_seconds)`:
        a.  If `d_last_rain_update_time[c] < 0.0` (initial run), set `d_last_rain_update_time[c] = t_now`.
        b.  Calculate `time_since_last_update = t_now - d_last_rain_update_time[c]`.
        c.  Calculate total rainfall `P_increment_mm = d_rain_input_mms[c] * time_since_last_update`.
        d.  Update `d_rain_cum_mm[c] += P_increment_mm`.
        e.  Compute new `d_rain_excess_cum_mm[c]` based on updated `d_rain_cum_mm[c]` and `d_cn_params_s[c]` using the SCS-CN formula. This is where the core SCS-CN calculation will now reside *only* in this kernel.
        f.  Calculate the *average excess rate* over `time_since_last_update`:
            `excess_rate_mms = (d_rain_excess_cum_mm[c] - previous_d_rain_excess_cum_mm_before_update) / time_since_last_update` (Need to store previous excess for this. Alternatively, calculate the *total excess for the minute* and divide by 60s). A simpler way might be to compute the *instantaneous* excess rainfall rate based on the *new `d_rain_cum_mm`* and `d_cn_params_s[c]`, then convert this to `m/s`. Let's assume for now `d_current_rain_source_rate_mps` stores the *instantaneous* excess rate calculated at `t_now` after updating cumulative state.
            `d_current_rain_source_rate_mps[c] = calculated_excess_rate_mms * 1e-3;`
        g.  Update `d_last_rain_update_time[c] = t_now`.
    3.  If no update needed, `d_current_rain_source_rate_mps[c]` retains its previous value.

### Phase 2: Integration into RK Steps

-   **`swe2d_gpu_step_rkX` functions (`swe2d_gpu.cu`)**:
    1.  At the very beginning of the overall `dt` loop (before any RK substep begins), launch `swe2d_update_rain_source_rate_kernel`.
    2.  Within each RK substep (e.g., `swe2d_rkX_stageY_kernel` or similar combined kernels), instead of calling `swe2d_build_rain_cn_source_kernel` or performing dynamic SCS-CN calculations:
        -   Simply add `d_current_rain_source_rate_mps[c]` to the `dh/dt` term.
        -   Ensure this addition happens consistently across all substeps.
    3.  **Crucially, remove** the "save/restore" logic for `d_rain_cum_mm` and `d_rain_excess_cum_mm` from all RK step functions, as these buffers are now only managed by the minute-based update kernel.

### Phase 3: Testing and Validation

-   **Update Existing Tests**:
    -   `tests/test_swe2d_gpu_graph_higher_order.py::test_rain_exact_depth_error_improves_with_higher_order` will need its assertions adjusted. The error profile will change as rainfall is now integrated as a first-order source. The goal is stability and correct volume, not necessarily lower error than RK2 (which it previously struggled to achieve accurately).
    -   Ensure other tests involving rainfall (e.g., `test_swe2d_gpu_native_rain_gui_path`) still pass and produce reasonable results.
-   **Focus on Mass Conservation**: Verify that total water volume from rainfall is correctly conserved over longer simulation periods with the new update frequency.
-   **Stability Checks**: Run simulations with various rainfall patterns to ensure the new approach does not introduce instabilities or NaNs.

This plan aims to meet your requirements for simplifying the rainfall integration while still allowing for reasonable temporal variation in rainfall excess.
