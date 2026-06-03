# AGENTS

## Validation Priority

- Prefer GPU-focused validation suites first:
  - `tests/test_swe2d_gpu_validation_perf.py`
  - `tests/test_swe2d_gpu_unstructured.py`
- The legacy parity suite `tests/test_swe2d_gpu.py` is informational only and should not drive SWE2D design decisions.

## Current Known GPU Status

- CUDA passes the gmsh-based unstructured dam-break checks for spatial schemes 0..4.
- CUDA passes the gmsh-based unstructured lake-at-rest checks for spatial schemes 0..4 after eta-based reconstruction in the higher-order GPU path.
- Current SWE2D engineering priority is CUDA optimization and robustness hardening, not CPU parity.

## Godunov Rollout Handoff

- Use [docs/GODUNOV_2D_GPU_IMPLEMENTATION_GUIDE.md](docs/GODUNOV_2D_GPU_IMPLEMENTATION_GUIDE.md) as the main implementation handoff for the selectable Godunov FVM rollout.

## Repository Session Documentation

- Store implementation handoff and recovery notes in repository-tracked docs under `docs/` so they can be pushed to origin.
- Current rolling session log: [docs/AGENT_SESSION_RECOVERY_LOG.md](docs/AGENT_SESSION_RECOVERY_LOG.md).

## Python Cache Discipline

- After any structural change to a Python module (signature changes, new return values, new classes, changed imports), **always purge `__pycache__`** before the user restarts QGIS:
  ```bash
  find . -type d -name __pycache__ -exec rm -rf {} +
  ```
- QGIS holds modules in memory for the session, and stale `.pyc` files cause invisible failures (wrong arity, missing attributes, silent fallback paths).
- When in doubt, purge before asking the user to restart.

## Studio UI Architecture & Structural Changes

- **`.ui` files are the source of truth** for widget layout and properties.
  Use Qt Designer to edit them.  Only create widgets programmatically when
  they cannot live in a `.ui` file (dynamically populated combos, etc.).
- When making structural changes (new tabs, new forms, new feature toggles,
  widget moves/renames), follow the checklists in
  [docs/STUDIO_UI_ARCHITECTURE.md](docs/STUDIO_UI_ARCHITECTURE.md).
- After any `.ui` change, run:
  ```bash
  python tools/ui_bind_sync.py forms/swe2d_<name>.ui <py_files> --missing
  ```
  to verify all widgets have bindings and no orphans remain.
- The Studio and legacy shell dialog have **parallel tab lists** — changes
  to `_compose_left_pane()` must be mirrored in `studio_build_ui()` within
  `swe2d/workbench/extracted/shell_dialog_methods.py`.
- Feature toggles touch 4 files: feature flags dict, keyword function,
  menu actions, and toolbar buttons.  All four must be updated together.
