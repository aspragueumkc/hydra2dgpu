# Native 1D Backend Contract

This document freezes the current 1D native backend contract used by the Python unsteady solver bridge.

## Scope

Current native entrypoints cover bounded arithmetic slices and a full single-timestep Newton loop:

1. `solve_table_state(...)`
2. `assemble_system_core(...)`
3. `adaptive_damping_scale(...)`
4. `solve_banded_full(...)`
5. `run_one_timestep_unsteady_1d_cpp(...)`
6. `build_section_hydraulic_table_cpp(...)` (HP1 slice-1)
7. `build_section_hydraulic_table_from_geometry_cpp(...)` (HP1 slice-2)
8. `configure_table_threads_cpp(int)` (HP1 slice-3)
9. `get_table_threads_cpp()` (HP1 slice-3)

The Python layer remains responsible for:

1. GeoPackage loading and persistence.
2. QGIS/UI orchestration and time-series handling.
3. GeoPackage and model object decoding to section polyline arrays.
4. Hydraulic-table orchestration (native call + fallback) and activation-elevation metadata.
5. Output formatting, progress callbacks, and debug diagnostics.

## Units and Conventions

1. Length: feet.
2. Area: square feet.
3. Discharge: cfs.
4. Time: seconds.
5. Gravity: `32.174 ft/s^2`.
6. Node ordering: upstream to downstream.
7. Unknown ordering in banded system: `x[2i] = Δz_i`, `x[2i+1] = ΔQ_i`.
8. Banded storage layout matches SciPy `solve_banded((2, 2), ab, rhs)` with shape `(5, n)` and `ab[2 + i - j, j] = A[i, j]`.
9. Downstream BC mode is passed as `ds_is_stage` boolean. `False` means normal-depth slope input.
10. Native acceleration is optional and must preserve Python fallback semantics.

## Entry Point Contracts

### `solve_table_state(...)`

Inputs:
1. Scalar `z`, scalar `q_total`.
2. 1D equal-length stage/property arrays for `z_values`, `A_*`, `T_*`, `K_*`.
3. Scalar left/right overbank activation elevations and ramp depth.

Outputs:
1. Tuple of subsection and total hydraulic state values matching `UnsteadySectionState` field order used in Python.

### `assemble_system_core(...)`

Inputs:
1. `reach_lengths`: shape `(N - 1,)`.
2. Per-node arrays of shape `(N,)`: `z_values`, `q_values`, `area_values`, `conveyance_values`, `top_width_values`, `velocity_values`, `alpha_values`, `dkdz_values`.
3. Scalar `dt`, `theta`, `q_upstream_next`, `ds_is_stage`, `ds_bc_value`, `ds_bc_ramp_factor`.

Outputs:
1. `ab`: banded coefficient matrix, shape `(5, 2N)`.
2. `rhs`: right-hand-side vector, shape `(2N,)`.

### `adaptive_damping_scale(...)`

Inputs:
1. Per-node arrays of shape `(N,)`: `bed_elevations`, `z_iter`, `q_iter`, `dz_raw`, `dq_raw`.
2. Scalar `wetting_depth`.

Output:
1. Scalar damping factor clamped to `[0.05, 1.0]`.

### `solve_banded_full(...)`

Inputs:
1. `ab`: shape `(5, n)`.
2. `rhs`: shape `(n,)`.

Output:
1. Solution vector of shape `(n,)`.

### `run_one_timestep_unsteady_1d_cpp(...)`

Inputs:
1. Per-node arrays `z_n_input`, `q_n_input`: current state, shape `(N,)`.
2. `reach_lengths`: shape `(N - 1,)`.
3. `bed_elevations`: shape `(N,)`.
4. Per-node property arrays `area_values`, `conveyance_values`, `top_width_values`, `velocity_values`, `alpha_values`, `dkdz_values`: shape `(N,)`.
5. Scalar control parameters: `dt`, `theta`, `q_upstream_next`, `ds_is_stage`, `ds_bc_value`, `ds_bc_ramp_factor`, `max_iter`, `tol`, `wetting_depth`.

Outputs:
1. `z_out`, `q_out`: updated state arrays, shape `(N,)`.
2. `inner_iter_count`: number of Newton iterations executed (int).
3. `max_update_error`: largest update magnitude in final iteration (double).
4. `converged`: boolean flag (True if tolerance met before max_iter).

**Contract notes:**
- Orchestrates a full Newton iteration loop for one timestep in native code.
- Calls `assemble_system_core`, `solve_banded_full`, and `adaptive_damping_scale` internally.
- Enforces minimum wetting depth on all nodes after state updates.
- Returns both the updated state and diagnostic information for logging/debugging.

### `build_section_hydraulic_table_cpp(...)`

Inputs:
1. Subsection geometry arrays for one cross section: `lob_x`, `lob_z`, `ch_x`, `ch_z`, `rob_x`, `rob_z` (all 1D).
2. Stage grid `z_values` (1D, monotonic increasing expected).
3. Manning roughness values `n_lob`, `n_ch`, `n_rob`.

Outputs:
1. `A_lob_raw`, `T_lob_raw`, `K_lob_raw` (1D arrays).
2. `A_ch`, `T_ch`, `K_ch` (1D arrays).
3. `A_rob_raw`, `T_rob_raw`, `K_rob_raw` (1D arrays).
4. `K_total_raw` and `dK_dz_raw` (1D arrays).

Contract notes:
- Native kernel computes subsection area/perimeter/top-width using trapezoidal submerged geometry integration.
- `dK_dz_raw` is computed with a second-order edge stencil compatible with Python `np.gradient(..., edge_order=2)` behavior.
- Activation elevations are still computed in Python and attached to `SectionHydraulicTable`.

### `build_section_hydraulic_table_from_geometry_cpp(...)`

Inputs:
1. Full section polyline arrays: `geom_x`, `geom_z` (1D).
2. Bank stations: `left_bank_station`, `right_bank_station`.
3. Stage grid `z_values` (1D, monotonic increasing expected).
4. Manning roughness values `n_lob`, `n_ch`, `n_rob`.

Outputs:
1. Same table arrays as `build_section_hydraulic_table_cpp(...)`:
	`A_lob_raw`, `T_lob_raw`, `K_lob_raw`, `A_ch`, `T_ch`, `K_ch`,
	`A_rob_raw`, `T_rob_raw`, `K_rob_raw`, `K_total_raw`, `dK_dz_raw`.

Contract notes:
- Native kernel sorts raw section polyline points by station.
- Native kernel clips LOB/CH/ROB subsection geometry internally from bank stations.
- Intended as the primary HP1 path to remove Python subsection clipping from startup hot path.
- Python keeps a compatibility fallback to subsection-array entrypoint and pure Python builder.

### `configure_table_threads_cpp(int)`

Inputs:
1. `thread_count` (int): Number of OpenMP threads to use for table-build kernels. Clamped to `[1, hardware_concurrency]` internally.

Outputs:
1. None.

Contract notes:
- Sets process-global `g_table_threads` used by `omp_set_num_threads(...)` before parallel table-build regions.
- Has no effect if the extension was compiled without OpenMP (`BACKWATER_HAS_OPENMP` not defined).
- Python bridge reads `BACKWATER_NATIVE_TABLE_THREADS` env var at `_build_hydraulic_tables` entry; if unset, auto-computes `min(cpu_count, n_sections)` and calls this function.

### `get_table_threads_cpp()`

Inputs:
1. None.

Outputs:
1. Current `g_table_threads` value (int). Returns 1 if OpenMP is not compiled in.

## Error Semantics

1. Native functions raise Python-visible exceptions on shape mismatch or invalid dimensions.
2. Python caller records fallback counts and last fallback error message.
3. Solver execution must continue through Python fallback when native acceleration is unavailable or raises.

## Current Gaps

1. No frozen golden-reference 1D fixtures for regression testing yet.
2. No 2D contract yet.
3. Full-run `run_unsteady_1d_cpp(...)` binding (for all timesteps) not yet implemented; Python orchestration layer still needed.

## Hybrid Acceleration Design (Single Simulation)

This section defines the next native contract extensions for startup-heavy work while keeping Python orchestration intact.

### Design Goal

Reduce total wall-clock by accelerating both:
1. One-time startup work (geometry preprocessing and hydraulic-table construction).
2. Per-timestep Newton solve path (already accelerated by `run_one_timestep_unsteady_1d_cpp(...)`).

### New Native Entry Points (Planned)

1. `build_section_geometry_cpp(...)`
	- Purpose: Cross-section geometry preprocessing from section definitions.
	- Input: Per-section station/elevation arrays and overbank metadata.
	- Output: Compact, contiguous geometry arrays used by table generation and runtime property interpolation.

2. `build_hydraulic_tables_cpp(...)`
	- Purpose: Precompute stage-dependent tables (`A`, `T`, `K`, `alpha`, optional derivatives) for each section.
	- Input: Preprocessed geometry arrays, Manning parameters, table controls (`dz`, padding, ramp controls).
	- Output: Table bundle arrays suitable for direct use by `solve_table_state(...)` and timestep solver.

3. `compute_node_properties_cpp(...)` (optional intermediate)
	- Purpose: Batch property evaluation for all nodes from tables for current `z`, `q`.
	- Input: Table bundle, per-node states.
	- Output: Node arrays (`area`, `conveyance`, `top_width`, `velocity`, `alpha`, `dkdz`).

### Required Single-Simulation Acceleration Methods

All native geometry/table work MUST implement the following four methods:

1. Keep hot loops in C++
	- No Python loops over sections or stage rows in preprocessing/table build paths.
	- Python should pass contiguous arrays and receive contiguous arrays.

2. Vectorization/SIMD and cache-friendly layouts
	- Use structure-of-arrays (SoA) for table outputs where practical.
	- Prefer contiguous `double` buffers and alignment-friendly traversal order.
	- Keep stride-1 loops for stage-major operations to help autovectorization.

3. OpenMP threading for table/property kernels
	- Parallelize across sections and/or stage rows (`#pragma omp parallel for`).
	- Threading policy: deterministic reduction behavior and reproducible output order.
	- Provide environment/flag control to cap threads in plugin contexts.

4. Batched linear algebra where possible
	- For repeated small solves or repeated coefficient operations, use batched operations rather than scalar-call loops.
	- Keep the current banded timestep solve API unchanged, but allow internal batching in startup and property phases.

### MPI Policy

MPI is NOT part of the single-simulation in-process plugin contract.

1. In-simulation domain-decomposed MPI is excluded for current 1D implicit solve because global Newton coupling would add high communication overhead.
2. MPI is allowed only for outer-level parallelism (many independent runs), handled outside this in-process contract.
3. Any MPI orchestration should consume the same deterministic single-run C++ kernels as workers.

### Performance Expectations (Planning Envelope)

Based on current benchmark decomposition:
1. Startup/preprocess is roughly `0.36-0.37 s` and is mostly one-time cost.
2. If startup work is accelerated by about `3x-6x`, startup can drop to roughly `0.06-0.12 s`.
3. Expected wall-clock impact from this port alone:
	- Short/medium runs: typically meaningful additional gain.
	- Long runs: moderate additional gain because per-step solve dominates.

These are planning estimates; benchmark evidence remains the source of truth.