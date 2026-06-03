# Agent Session Recovery Log

Purpose: keep implementation decisions and handoff context inside the repository so they can be pushed to origin and recovered after local environment or disk failures.

## 2026-05-22

### SWE3D Boundary-Condition Hardening
- Added execution guide: `docs/SWE3D_BC_HARDENING_IMPLEMENTATION_PLAN.md`.
- Linked active task board D5 to the hardening guide.
- Implemented U1.1 runtime policy plumbing:
  - CUDA runtime controls/env parsing for BC policy knobs.
  - Python `configure_swe3d_runtime(...)` support for BC policy knobs.

### U1.2 In Progress
- Implemented CUDA path for dynamic wet/open-area normalization for `INFLOW_FLOW_RATE`:
  - Added per-face effective area reduction kernel.
  - Routed effective area into inflow BC and transport boundary flux helper.
  - Enabled by `BACKWATER_SWE3D_Q_INFLOW_AREA_POLICY=1` (legacy behavior remains policy `0`).
- Added regression test comparing legacy vs dynamic area policy under limited wet inlet area in `tests/test_swe3d_uncoupled_validation.py`.

### Notes
- Keep all major design decisions and migration notes in `docs/` files tracked by git.
- Avoid relying on transient Copilot memory/session folders for critical handoff information.
