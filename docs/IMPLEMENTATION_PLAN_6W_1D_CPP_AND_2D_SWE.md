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

Deliverables:
1. Optimized C++ 1D backend.
2. Benchmark report with direct comparison to Python+Numba path.
3. C++ backend default enabled for supported configurations.

Acceptance criteria:
1. Measured speedup target achieved on representative scenarios.
2. Stability diagnostics show no regression in solver robustness.
3. Fallback path to Python remains available.

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
