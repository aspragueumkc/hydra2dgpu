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

## Git Safety — Destructive File Operations

- **NEVER** run `git checkout -- <file>` (or any command that overwrites working-tree files with committed versions) without first checking for uncommitted changes across the **entire repo**:
  ```bash
  git status --short
  ```
- If ANY file shows `M` (modified), `A` (added), `D` (deleted), or `??` (untracked) that could be relevant, do NOT use destructive git commands. Use manual `replace_string_in_file` edits to revert only specific changes instead.
- `git checkout -- <file>` silently discards ALL uncommitted changes in `<file>` — including changes the agent didn't make. There is no undo.
- QGIS holds modules in memory for the session, and stale `.pyc` files cause invisible failures (wrong arity, missing attributes, silent fallback paths).
- When in doubt, purge before asking the user to restart.

## Unit System Conventions

- **Never assume a specific unit system** (SI or USC). All conversions must be based on the CRS-derived map units via `swe2d.units`.
- **C++ kernel accepts model units** for all geometry. Weir, orifice, bridge, and pump formulas are unit-agnostic — they produce correct results in whatever units the inputs are in, as long as the `gravity` parameter matches. Only the HDS-5 culvert path converts geometry to feet internally, computes in USC, then converts the result back to model units using the caller-supplied `model_to_ft` factor.
- **C++ kernel culvert output** is converted from CFS back to model units (÷ `model_to_ft³`) before returning. Non-culvert types return values directly in model units.
- **Python coupling controller** (`coupling.py`) converts kernel CFS output to model units via `SI_M3_PER_USC_FT3 / si_m3_per_model_volume()`.
- **Python structure module** (`swe2d/extensions/structures.py`) always returns **CMS** because culvert routines adopted from SWMM compute in USC and explicitly convert CFS→CMS.
- **Diagnostics stored in `SWE2DCouplingDiagnostics`** are in **model units** (not SI). The coupling controller converts from kernel/Python output units to model units before storing.
- **Runtime reporter** (`runtime_reporting.py`) displays diagnostics using `length_unit_name` and assumes values are already in model units.
- **Heap gravity bug fix**: Orifice/bridge formulas now use CRS-derived `gravity()` (9.81 m/s² for SI, 32.17 ft/s² for USC) instead of the old hardcoded 9.81 — this was a ~45% underestimation for USC projects.
- **`model_to_ft`**: Passed from Python to the C++ kernel as `units.model_to_ft()`. Needed so culvert code can convert model-unit geometry to feet internally.

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
- Feature toggles touch 3 files: feature flags dict + keyword function in
  `SWE2DWorkbenchStudioDialog`, and menu/toolbar actions in `studio_host_methods.py`.
  All three must be updated together.
