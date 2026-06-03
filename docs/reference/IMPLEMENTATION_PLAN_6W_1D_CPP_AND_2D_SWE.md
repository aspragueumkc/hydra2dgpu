# 6-Week Implementation Plan: 1D C++ Port + 2D SWE Solver

## Status Update (Current Direction)

For ongoing SWE2D work, this repository now follows a GPU-primary strategy:
new numerical accuracy and performance features are implemented first in CUDA,
with CPU SWE2D treated as a fallback/debug path rather than a parity target.

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
1. High-order 2D reconstruction and adaptive mesh refinement (AMR).
2. Full multi-domain 1D/2D dynamic coupling beyond simple coexistence in the UI.
3. Multi-GPU distribution and MPI domain decomposition for the 2D solver.

Revised in scope (supersedes prior exclusion):
1. Hybrid GPU/CPU 2D SWE solver on an unstructured triangular mesh.
   - GPU path via CUDA (NVIDIA); graceful runtime fallback to OpenMP CPU path.
   - Unstructured mesh required for realistic riverine geometry (meanders, confluences, floodplains).
   - CUDA detection at CMake configure time; CPU-only builds remain valid.

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

## Hybrid GPU/CPU 2D SWE Solver Architecture

### Rationale for Unstructured Grid
Realistic riverine domains — meandering channels, braided reaches, confluences, irregular floodplains — cannot be accurately represented on structured Cartesian grids without extreme over-refinement or staircase boundary artifacts. An unstructured triangular mesh:
1. Conforms to arbitrary bank geometry and islands.
2. Supports local refinement in narrow channels without penalizing the full domain.
3. Maps directly to GIS input layers (polygon boundaries → mesh generation) that QGIS already holds.

### Rationale for Hybrid GPU/CPU
1. The per-timestep work in an explicit FV scheme is embarrassingly parallel over cells and edges — the ideal GPU workload.
2. Mesh topology (connectivity, BC classification) is static after setup and lives on CPU; no branchy logic goes to GPU.
3. CUDA is the most mature path for NVIDIA workstations where this plugin is likely deployed.
4. A clean CPU fallback (OpenMP, same kernels) keeps the solver portable and testable on any machine.

### Grid Representation: Structure of Arrays (SoA)
All mesh arrays use contiguous SoA layout for both SIMD auto-vectorization on CPU and coalesced memory access on GPU.

Node arrays (N nodes):
- `node_x[N]`, `node_y[N]`, `node_z[N]`  — coordinates and bed elevation

Cell arrays (M triangular cells):
- `cell_n0[M]`, `cell_n1[M]`, `cell_n2[M]`  — node indices
- `cell_cx[M]`, `cell_cy[M]`               — centroid coordinates
- `cell_area[M]`                           — cell area
- `cell_zb[M]`                             — bed elevation at centroid
- `cell_h[M]`, `cell_hu[M]`, `cell_hv[M]`  — conserved state

Edge arrays (E interior + boundary edges):
- `edge_c0[E]`, `edge_c1[E]`               — left/right cell indices (-1 = boundary)
- `edge_n0[E]`, `edge_n1[E]`               — endpoint node indices
- `edge_nx[E]`, `edge_ny[E]`               — outward unit normal (from c0 to c1)
- `edge_len[E]`                            — edge length
- `edge_bc_type[E]`                        — boundary type enum

### Numerical Scheme
1. **Riemann solver**: HLLC (positivity-preserving wave-speed estimates from Einfeldt).
2. **Reconstruction**: Piecewise-constant (1st order) MVP; MUSCL with MinMod limiter for 2nd order (deferred).
3. **Wet/dry**: Thin-film regularization (`h_min = 1e-6 m`); deactivate cells below threshold; enforce h ≥ 0 after update.
4. **Bed slope source**: Well-balanced hydrostatic reconstruction on each edge (maintains lake-at-rest exactly).
5. **Friction source**: Manning, explicit Euler (semi-implicit limiter to prevent velocity blowup in shallow cells).
6. **Timestep**: Global CFL reduction over all edges; `dt = CFL_factor × min(dx/|u+c|)`.

### GPU Kernel Layout (CUDA)
Each timestep executes three kernel launches:
1. `swe2d_flux_kernel<<<E/256, 256>>>` — parallel over edges; writes per-edge flux contributions.
2. `swe2d_update_kernel<<<M/256, 256>>>` — parallel over cells; accumulates fluxes, applies sources, enforces positivity.
3. `swe2d_cfl_kernel<<<M/256, 256>>>` + device reduction — computes minimum dt for next step.

### CPU Path (OpenMP)
Identical numerical operations, vectorized with `#pragma omp parallel for simd` over edge and cell loops. Thread count controlled by `BACKWATER_SWE2D_THREADS` env var or `configure_swe2d_threads_cpp()`.

### Path Selection
At runtime, `swe2d_backend.py` queries `swe2d_gpu_available()` and selects the CUDA path if available and not explicitly disabled via `BACKWATER_SWE2D_GPU=0`. Either path is invoked through the same pybind11 API surface — Python orchestration is path-agnostic.

### File Layout
```
cpp/src/
  swe2d_mesh.hpp         # Mesh SoA structs and edge connectivity builder
  swe2d_mesh.cpp         # Mesh construction and validation
  swe2d_numerics.hpp     # HLLC solver, reconstruction, well-balanced bed slope
  swe2d_numerics.cpp     # CPU implementations of numerical kernels
  swe2d_solver.hpp       # Solver API (init, step, query, destroy)
  swe2d_solver.cpp       # CPU solver: OpenMP flux + update + CFL loops
  swe2d_gpu.cuh          # CUDA declarations (included only when CUDA found)
  swe2d_gpu.cu           # CUDA kernels and device memory management
  swe2d_bindings.cpp     # pybind11 module backwater_swe2d
```

### Weekly Plan

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

### Week 4: 2D SWE MVP Compute Core — Hybrid GPU/CPU Unstructured Mesh
Goals:
1. Implement unstructured triangular mesh infrastructure (SoA layout for GPU-readiness).
2. Implement explicit finite-volume SWE on unstructured mesh with HLLC Riemann solver.
3. Add wet/dry positivity preservation and CFL timestep control.
4. Add CUDA GPU acceleration path with OpenMP CPU fallback.
5. Support core boundaries: inflow, stage/open, wall, reflecting.

Deliverables:
1. C++ 2D solver core (`swe2d_mesh`, `swe2d_solver`) with pybind11 entrypoint.
2. CUDA kernel path (`swe2d_gpu.cu`) wired behind CMake `BACKWATER_USE_CUDA` flag.
3. CPU OpenMP path that runs identically when CUDA is absent or disabled at runtime.
4. Python bridge `swe2d_backend.py` with GPU-availability query and path selector.
5. Canonical validation tests (dam-break, still-water balance, uniform flow).
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
