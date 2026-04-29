# Task Board: 6-Week 1D C++ Port + 2D SWE

This board is designed for solo execution with optional LLM-agent delegation.

## Usage
1. Keep statuses updated inline.
2. Each task should reference one PR or commit range.
3. Do not start a dependent task until its prerequisites are marked done.

Status legend:
- [ ] not started
- [~] in progress
- [x] done
- [!] blocked

## Epic A: Native Backend Foundation (Week 1)

### A1 Build and Toolchain
- [ ] Add `cpp/` layout and `CMakeLists.txt`.
- [ ] Add pybind11 module scaffold.
- [ ] Add local build instructions for Linux.

Definition of done:
1. Clean checkout builds extension successfully.
2. Python imports extension and passes smoke test.

### A2 API Contracts
- [ ] Define 1D input/output arrays and metadata contract.
- [ ] Define 2D input/output arrays and metadata contract.
- [ ] Freeze units, sign conventions, and BC naming.

Definition of done:
1. Contract doc committed.
2. Contract used by at least one test call path.

### A3 Baseline References
- [ ] Export golden 1D reference outputs from current Python solver.
- [ ] Add parity test fixtures under `tests/`.

Definition of done:
1. Golden fixtures versioned and test-readable.

## Epic B: 1D C++ Port and Hardening (Weeks 2-3)

### B1 Parity Port (Week 2)
- [ ] Port core 1D assembly path to C++.
- [ ] Port solve path and iteration loop.
- [ ] Bind `run_unsteady_1d_cpp(...)` into Python.
- [ ] Add backend selector flag (Python vs C++).

Definition of done:
1. Parity tests pass within tolerance.
2. Existing UI flow can run either backend.

### B2 Runtime Behavior Parity
- [ ] Match DS/US BC behavior.
- [ ] Match `max_iter`, `tol`, and ramp handling.
- [ ] Match debug and error semantics at major failure points.

Definition of done:
1. Regression tests pass for representative BC/ramp variants.

### B3 Optimization and Production Default (Week 3)
- [ ] Optimize hotspots after parity lock.
- [ ] Add deterministic diagnostics mode.
- [ ] Benchmark C++ vs Python+Numba using existing benchmark tooling.
- [ ] Enable C++ as default for supported configurations.

Definition of done:
1. Speedup target met on representative cases.
2. No stability regression in test suite.

## Epic C: 2D SWE Core (Week 4)

### C1 Numerics MVP
- [ ] Implement explicit finite-volume update for `h, hu, hv`.
- [ ] Add CFL timestep control.
- [ ] Add wet/dry positivity protection.

Definition of done:
1. Canonical tests run without instability blowups.

### C2 Source Terms and Boundaries
- [ ] Add bed slope source term.
- [ ] Add Manning friction source term.
- [ ] Add inflow, stage/open, and wall boundaries.

Definition of done:
1. Test cases demonstrate expected behavior for each BC type.

## Epic D: 2D Plugin Integration (Week 5)

### D1 Run Orchestration
- [ ] Add Python orchestration wrapper for 2D native core.
- [ ] Add 2D run controls to plugin UI (minimal set).
- [ ] Add progress callbacks and cancellation-safe checks.

Definition of done:
1. 2D run starts and completes from GUI.

### D2 Persistence and Monitoring
- [ ] Add GeoPackage tables for 2D outputs.
- [ ] Add runtime monitor metrics: CFL, wet-cell count, min/max depth, mass trend.
- [ ] Add exportable runtime log for 2D runs.

Definition of done:
1. 2D run can be saved, reloaded, and inspected.

## Epic E: Validation and Release Candidate (Week 6)

### E1 Validation Matrix
- [ ] Run full 1D parity matrix (Python vs C++).
- [ ] Run 2D canonical stability/accuracy matrix.
- [ ] Record performance baselines and deltas.

Definition of done:
1. Validation report committed under `docs/`.

### E2 Packaging and Documentation
- [ ] Finalize build/install docs for native module.
- [ ] Document known limitations and fallback behavior.
- [ ] Prepare release notes.

Definition of done:
1. Another agent/human can follow docs from clean checkout.

## Blockers and Risks
- [ ] Packaging issues in plugin Python environment.
- [ ] Numerical parity drift in 1D edge cases.
- [ ] 2D wet/dry instability under extreme gradients.

Mitigation checklist:
- [ ] Keep Python 1D fallback enabled through first native release.
- [ ] Add stress tests before performance tuning changes.
- [ ] Gate 2D as MVP/Beta behind explicit mode label.

## Agent Handoff Protocol
Before handoff, each agent should update:
1. Tasks touched and status changes.
2. Tests executed and results.
3. Benchmarks executed and command lines.
4. Files changed and rationale.
5. Open risks and next immediate task.
