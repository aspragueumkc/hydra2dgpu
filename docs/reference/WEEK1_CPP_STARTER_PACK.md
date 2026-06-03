# Week 1 C++ Starter Pack (1D Solver)

Execution order for solo + agent implementation. Keep each item as a PR-sized change.

1. Build scaffolding and CI smoke check
- Add `CMakeLists.txt` and compile `backwater_native` on Linux runner.
- Publish artifact or upload build logs.

2. Native linear solve parity
- Keep `solve_banded_full` as first native kernel.
- Add parity test against SciPy solve on random diagonally dominant pentadiagonal systems.

3. Python runtime switch hardening
- Keep env-gated toggle `BACKWATER_USE_CPP_SOLVER`.
- Add one-time runtime log line indicating native on/off/fallback.

4. Hydraulic table interpolation kernel
- Port table interpolation hot path to C++.
- Expose vectorized interpolation API for node batches.

5. Jacobian/Residual assembly kernel
- Port `_assemble_system` arithmetic core to C++ using contiguous arrays.
- Keep BC logic and persistence in Python until parity verified.

6. End-to-end Newton iteration function
- Add native function to run one full Newton iteration from supplied state arrays.
- Return `delta` and diagnostics (norm, convergence flag).

7. Benchmark harness integration
- Extend `tools/unsteady_benchmark.py` to report native-vs-python speedup and breakdown.
- Emit JSON summary for trend tracking.

8. Numerical regression suite
- Add fixtures and tolerances for stage/discharge trajectories.
- Require max relative error thresholds per test case.

9. Failure-mode resilience
- Force fallback on native exceptions and continue run.
- Add tests for missing module, malformed arrays, singular matrix path.

10. Week 1 exit criteria
- Native path builds locally.
- Native linear solve validated with parity tests.
- Runtime switch + fallback confirmed.
- Benchmark report generated and checked into docs artifacts folder.
