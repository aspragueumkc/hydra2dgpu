# SWE3D Stability Sprint Plan

## Goal
Improve 3D runtime stability on real-world cases through GPU-first hardening, better guardrails, and validation gates that reflect production geometry and forcing.

## Prioritized Backlog

1. Sprint 1: Geometry and runtime guardrails (in progress)
- Add strict geometry quality gates for 3D sub-grid uploads.
- Promote high-risk geometry diagnostics from warning-only to optional fail-fast behavior.
- Expose guardrail controls through Python runtime configuration helpers.
- Acceptance criteria:
  - Gate violations are logged with explicit metric values.
  - Strict mode aborts run before unstable timestepping starts.
  - Controls are configurable without recompiling.

2. Sprint 2: Coupling physics completion (highest physics impact)
- Replace 2D-3D exchange skeleton with conservative one-way 2D->3D exchange implementation.
- Add sanity checks for flux balance and interface stability metrics.
- Acceptance criteria:
  - Coupling path no longer zeros exchange buffers.
  - One-way coupling regression suite passes.

3. Sprint 3: Projection and adaptive dt robustness
- Add projection failure diagnostics and adaptive dt telemetry per retry.
- Add hardened bounds and health metrics for projection rejection loop.
- Acceptance criteria:
  - Retry path emits reproducible diagnostics.
  - No silent degradation to unstable dt behavior.

4. Sprint 4: Real-world validation harness
- Add real-world unstructured test set with terrain + OBJ fixtures.
- Add pass/fail thresholds for mass drift, VoF bounds, and stability runtime health.
- Acceptance criteria:
  - Real-world suite runs in CI/manual gate mode.
  - Stability regressions are detected before release.

## Sprint 1 Initial Implementation (completed in this pass)

- Added strict geometry gate controls in non-GUI runtime upload path:
  - BACKWATER_SWE3D_GEOM_STRICT
  - BACKWATER_SWE3D_GEOM_MAX_SOLID_FRACTION
  - BACKWATER_SWE3D_GEOM_MAX_SEED_LEAK_FALLBACKS
- Added violation logging and strict-mode fail-fast in 3D geometry upload.
- Exposed these controls through `configure_swe3d_runtime(...)` in `swe2d_backend.py`.

## Sprint 1 Next Tasks

1. Add unit tests for geometry gate parsing and strict/non-strict behavior.
2. Add run log metadata fields so postmortems capture gate thresholds and violations.
3. Add optional UI controls for strict gate mode and thresholds (advanced section).
