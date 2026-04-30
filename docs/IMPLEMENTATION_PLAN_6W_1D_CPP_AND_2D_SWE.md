# 6-Week Implementation Plan: 1D C++ Port + 2D SWE Solver

## Objective
Deliver a production-capable native backend strategy for this plugin by:
1. Porting the existing 1D unsteady solver compute core to C++.
2. Building a new 2D Shallow Water Equations (SWE) compute core in C++.
3. Keeping Python/QGIS as orchestration, UI, persistence, and diagnostics layers.

## Scope Boundaries
In scope:
1. C++ compute kernels and pybind11 bindings.
2. 1D parity and performance migration path.
3. 2D MVP (robust explicit finite-volume) integrated into plugin runtime.
4. Benchmarking, diagnostics, and GeoPackage persistence updates.

Out of scope for this 6-week window:
1. High-order 2D reconstruction and AMR.
2. Full GPU path.
3. Multi-domain 1D/2D coupling beyond simple coexistence in the UI.

## Architecture Decision
1. C++ first for both 1D and 2D compute kernels.
2. Python remains system-of-record for plugin workflows.
3. One native backend layout reused by both solvers.

## Hybrid Design: Single-Run + Multi-Run Parallelism

This plan adopts a hybrid model:
1. Single simulation acceleration is in-process C++ (pybind11) with OpenMP.
2. Multi-run acceleration (ensembles/calibration/sweeps) may use MPI outside the plugin process.

### Single Simulation Acceleration Blueprint (Required)

The single-run path explicitly includes four acceleration items:

1. Keep hot loops in C++
	- Port cross-section geometry preprocessing and hydraulic table construction into native kernels.
	- Keep Python in orchestration mode only (I/O, UI, callbacks, persistence).

2. Vectorization/SIMD + cache-friendly layouts
	- Convert startup and table/property arrays to SoA-friendly contiguous buffers.
	- Preserve stride-1 loops for stage-major operations and compiler autovectorization.

3. OpenMP threads for table/property kernels
	- Parallelize by section and stage rows for preprocessing/table build.
	- Add thread cap control for plugin safety (`OMP_NUM_THREADS` or explicit setter).

4. Batched linear algebra where possible
	- Replace repeated scalar coefficient operations with batched kernels in startup/property phases.
	- Maintain current banded timestep solve contract; only internal batching strategy changes.

### MPI Position

1. Do not use MPI domain decomposition for one implicit 1D run in plugin context (communication-heavy global coupling).
2. Use MPI only for outer-level independent-run parallelism (scenario sets, parameter sweeps, Monte Carlo).
3. MPI workers should each run the same validated single-simulation C++ path for deterministic comparability.

## Weekly Plan

### Week 1: Foundations and Contracts
Goals:
1. Create native module structure under `cpp/` and Python bridge package.
2. Add CMake + pybind11 build path for local development.
3. Freeze data contracts for 1D and 2D engine I/O.
4. Create golden baseline outputs for current 1D Python solver.

Deliverables:
1. Importable extension module with a smoke-test function.
2. Data contract document for array layouts, units, BC conventions, and error semantics.
3. Golden reference test fixtures for 1D.

Acceptance criteria:
1. Extension imports from plugin Python environment.
2. Rebuild is reproducible with a single documented command.
3. Golden references committed and validated.

### Week 2: 1D C++ Parity Port
Goals:
1. Port 1D assembly and solve loop to C++ (matching existing behavior).
2. Preserve DS/US BC behavior, ramping logic, and iteration controls.
3. Add runtime switch between Python and C++ backends.

Deliverables:
1. `run_unsteady_1d_cpp(...)` binding callable from existing plugin logic.
2. Parity test harness against Python outputs.
3. Feature flag to select engine.

Acceptance criteria:
1. WSE and Q parity within agreed tolerances on baseline cases.
2. Existing save/load output paths remain unchanged.
3. UI flow unchanged for current users.

### Week 3: 1D Optimization and Production Readiness
Goals:
1. Optimize C++ 1D hotspots after parity lock.
2. Add deterministic diagnostics mode and robust error guards.
3. Run decomposition benchmarks for startup vs per-step cost.
4. Implement native geometry preprocessing and hydraulic table build kernels with OpenMP.

Deliverables:
1. Optimized C++ 1D backend.
2. Benchmark report with direct comparison to Python+Numba path.
3. C++ backend default enabled for supported configurations.
4. Hybrid startup acceleration path (`build_section_geometry_cpp`, `build_hydraulic_tables_cpp`) behind a runtime flag.

Acceptance criteria:
1. Measured speedup target achieved on representative scenarios.
2. Stability diagnostics show no regression in solver robustness.
3. Fallback path to Python remains available.
4. Startup benchmark shows measurable reduction and no parity drift in table-derived properties.

### Week 4: 2D SWE MVP Compute Core
Goals:
1. Implement 2D explicit finite-volume SWE on structured grid.
2. Add robust wet/dry handling and positivity controls.
3. Support core boundaries: inflow, stage/open, wall.

Deliverables:
1. C++ 2D solver core with pybind entrypoint.
2. Canonical tests (dam-break, still-water balance, uniform flow sanity).
3. Basic output arrays and timestep diagnostics.

Acceptance criteria:
1. Stable runs on canonical tests.
2. Mass balance trend within configured tolerance.
3. No NaN/negative-depth instability in tested envelope.

### Week 5: 2D Plugin Integration and Persistence
Goals:
1. Integrate 2D execution path into plugin orchestration.
2. Add runtime monitor metrics for 2D.
3. Add GeoPackage tables for 2D outputs and debug summaries.

Deliverables:
1. 2D run mode in plugin with parameter controls.
2. Runtime monitor fields: CFL, wet cell count, min/max depth, mass error trend.
3. Save/load support for 2D runs.

Acceptance criteria:
1. User can run 2D from GUI and reload results.
2. Runtime log export includes 2D metrics.
3. Existing 1D functionality unaffected.

### Week 6: Validation, Tuning, and Release Hardening
Goals:
1. Full regression matrix for 1D parity and 2D stability.
2. Performance tuning pass and documentation finalization.
3. Packaging and release candidate preparation.

Deliverables:
1. Validation report (accuracy, stability, performance).
2. Release notes and known-limitations section.
3. Installation and troubleshooting guide for native backend.

Acceptance criteria:
1. 1D C++ marked production default.
2. 2D marked MVP/Beta with documented operational envelope.
3. Build, test, and packaging are repeatable from clean checkout.

## Cross-Cutting Quality Gates
1. No silent solver failures; explicit error reporting with step/time context.
2. Runtime diagnostics available in both success and failure scenarios.
3. Numerical acceptance thresholds defined before optimization changes.
4. Every performance claim backed by benchmark command and output.
5. Single-run acceleration changes must include separate startup and per-step decomposition metrics.
6. Parallel paths (OpenMP or MPI orchestration) must preserve deterministic tolerances and fallback behavior.

## Suggested Milestone Tags
1. `M1-foundation` (Week 1)
2. `M2-1d-parity` (Week 2)
3. `M3-1d-performance` (Week 3)
4. `M4-2d-core` (Week 4)
5. `M5-2d-integration` (Week 5)
6. `M6-release-candidate` (Week 6)

## Solo Execution Notes
1. Keep daily checkpoint notes in commit messages and PR descriptions.
2. Prefer small vertical slices: contract -> implementation -> test -> benchmark.
3. Defer optional features if they threaten parity or stability gates.

## Progress Snapshot (2026-04-29)

Completed or materially advanced:
1. Native module scaffold, CMake build path, and local Linux build doc are in place.
2. Runtime backend flag is implemented with Python fallback semantics.
3. Native 1D arithmetic slices now cover table-state interpolation, banded matrix assembly, adaptive damping scale, and banded solve.
4. Focused parity tests exist for `solve_table_state`, `assemble_system_core`, and `adaptive_damping_scale`.
5. Benchmark tooling now supports backend-specific timing with `--backend python|native|compare` and decomposition mode.
6. A current 1D native contract doc has been added in `docs/NATIVE_1D_BACKEND_CONTRACT.md`.

Still pending against the original weekly plan:
1. A single `run_unsteady_1d_cpp(...)` binding for the full 1D timestep loop is not implemented yet.
2. Golden 1D reference fixtures are not frozen yet.
3. 2D input/output contract and 2D solver work have not started.
4. Native backend is not ready to be the default path yet.

Immediate next steps being followed from the plan:
1. Continue Week 2 parity work by moving remaining per-step scalar preparation out of Python.
2. Keep the vertical-slice workflow: contract update, implementation, parity test, benchmark evidence.
3. Update the task board after each native milestone with concrete test and benchmark commands.

### HP1 Progress Update (2026-04-30)
Completed slice:
1. Added native entrypoint `build_section_hydraulic_table_cpp(...)` to construct subsection hydraulic tables from subsection geometry arrays.
2. Wired Python bridge fallback path so `_build_section_hydraulic_table(...)` uses native when enabled and falls back safely on errors.
3. Added parity test `tests/test_native_table_build.py` and validated against existing Python implementation.
4. Added native entrypoint `build_section_hydraulic_table_from_geometry_cpp(...)` to perform subsection clipping/preprocessing and table construction from raw section geometry arrays.
5. Switched runtime preference to geometry-preprocessing native path (with fallback chain preserved).

HP1 slice 3 completed (2026-04-30):
1. Added OpenMP `#pragma omp parallel for if(n_points >= 64)` in both table-build kernels.
2. Added `configure_table_threads_cpp` / `get_table_threads_cpp` entrypoints.
3. Conditionally enabled via `find_package(OpenMP QUIET)` — degrades gracefully if not present.
4. Auto thread config in `_build_hydraulic_tables`: `min(cpu_count, n_sections)` or `BACKWATER_NATIVE_TABLE_THREADS` env override.
5. Process-pool oversubscription guard: Python pool disabled when native is active.

Startup benchmark deltas (5 sections, dz=0.01, pad=5.0):
- Python: 0.366 s → Native (4 threads): 0.017 s → **~21.5× speedup**

HP1 is fully complete. All planned entrypoints implemented and parity-tested.

## Added Execution Track: Hybrid Acceleration Work Packages

### HP1: Native startup path (geometry + tables)
1. Add `build_section_geometry_cpp(...)` and `build_hydraulic_tables_cpp(...)` bindings.
2. Wire optional Python bridge path with transparent fallback.
3. Add parity tests versus current Python-generated tables.

Definition of done:
1. Section/table arrays match within tolerance across representative cross-sections.
2. Startup benchmark improvement is documented.

### HP2: Memory layout and SIMD pass
1. Move startup and property buffers to contiguous SoA-friendly layout.
2. Validate compiler vectorization on hot loops.

**HP2 completed (2026-04-30):**
1. Added `compute_node_properties_cpp` C++ kernel: batch evaluation of area, conveyance, top-width, velocity, alpha, dK/dz and discharge-weighted reach lengths from 2D SoA-packed table arrays (N × max_len, row-major).
2. Added `native_backend.py` wrappers: `compute_node_properties` (bridge) and `pack_node_property_bundle` (one-shot packer).
3. Wired into `_compute_node_properties` in `unsteady_model.py` with `bed_elevations` and pre-packed `node_property_bundle` parameters; `node_property_bundle` is built once before the time loop and reused each timestep.
4. Added parity test `tests/test_native_node_properties.py` (3 tests: parity, shapes, physical constraints).
5. All 8 parity tests pass (3 new + 5 HP1 carried forward).

Per-timestep property-evaluation benchmark (5 sections, 2000 reps):
- Python: ~207 µs/call → Native: ~77 µs/call → **~2.7× speedup**

Definition of done:
1. Benchmarks show startup and/or property-evaluation improvement. ✓
2. No parity regressions. ✓

### HP3: OpenMP threading pass
1. Parallelize section/table/property loops.
2. Add thread-count controls and safe defaults for desktop/plugin use.

Definition of done:
1. Multi-core speedup measured at fixed tolerances.
2. No nondeterministic instability beyond accepted tolerance envelope.

### HP4: Batched algebra pass
1. Batch repeated coefficient operations where possible in startup/property phases.
2. Keep current solver API unchanged to minimize integration risk.

Definition of done:
1. Reduced CPU time in targeted kernels.
2. Existing timestep solver parity tests stay green.

### HP5: MPI outer-run orchestrator (optional)
1. Provide a separate CLI/tooling path for independent-run distribution.
2. Keep MPI outside in-process QGIS plugin execution model.

Definition of done:
1. Throughput scaling shown on independent scenario batches.
2. Single-run plugin behavior remains unchanged.
