---
type: plan
status: active
created: 2026-07-27
progress:
  total: 10
  done: 8
  current: "8. Remove duplicate implementations after call-site verification"
  blockers: []
  last_updated: 2026-07-28
---

# Canonical Sample-Line Sampling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `subagent-driven-development` to execute this plan task-by-task with spec and code-quality review after each task. The worktree is already isolated at `.worktrees/sample-line-canonical`; do not create additional implementation worktrees.

**Goal:** Replace the split QGIS/OGR sample-line implementations with one plain-data canonical sampling pipeline, then validate its reported flow against an independent kernel-state reference.

**Architecture:** Source adapters extract sample-line records and mesh-cell coordinate records as plain data. A single Qt-free geometry service computes line normals, candidate cells, intersection lengths, station ordering, and the GPU-facing sampling map. The executor and GPU backend retain their existing public result layout while gaining a deterministic flow-reference validation path.

**Tech Stack:** Python 3.12, NumPy, installed `osgeo.ogr`/GEOS, PyQt5/QGIS adapters, CUDA C++, pybind11, `unittest`, repository fast-fail and GPU validation suites.

---

## Worktree and execution policy

All implementation occurs in:

```text
.worktrees/sample-line-canonical/
```

Use one implementation branch/worktree because the canonical input/output contract is shared by the geometry service, GPKG adapter, GUI adapter, executor, and tests. Parallel work is limited to read-only research and independent review; implementation tasks are sequential to avoid merge conflicts and intermediate-contract drift.

Do not commit unless explicitly requested by the user. Keep the active plan and spec in the worktree; update the plan checkbox and frontmatter progress together after each verified task.

## Files and responsibilities

Expected files to create or modify; confirm exact call sites before editing:

- Modify `swe2d/services/line_sampling_service.py` — canonical plain-data geometry/intersection service; remove QGIS geometry from the canonical path.
- Modify `swe2d/core/gpkg_io.py` — extract GeoPackage sample features into plain records and call the canonical service.
- Modify `swe2d/workbench/studio_dialog.py` — replace View-side mesh `QgsGeometry` construction with a plain-data adapter call.
- Modify `swe2d/mesh/mesh_runtime_logic.py` — make mesh-cell coordinate extraction the shared topology adapter contract.
- Modify `swe2d/core/executor.py` and, only if required, `swe2d/runtime/backend.py` — preserve the GPU-facing flattened map contract and add validation hooks without changing result schemas.
- Modify `cpp/src/swe2d_gpu.cu` and `cpp/src/swe2d_bindings.cpp` only if a direct GPU-state/reference hook is required; no kernel formula changes are planned.
- Add or modify focused tests under `tests/` for canonical geometry, adapters, executor contract, and numerical flow validation.
- Do not modify persisted result schema files unless an existing test proves an unrelated schema defect.

## Selector-consumable steps

```python
{"action": "Write failing Python tests for the canonical sample-line contract and independent flow reference", "type": "test", "phase": "red"}
{"action": "Implement the Python plain-data canonical sample-line geometry service", "type": "coding", "phase": "geometry"}
{"action": "Refactor Python GPKG and mesh adapters to feed the canonical sample-line service", "type": "refactor", "phase": "adapters"}
{"action": "Refactor PyQt5 GUI sample-line integration to remove View-side mesh geometry construction", "type": "ui", "phase": "gui"}
{"action": "Validate Python executor flattening and sample-line result persistence contract", "type": "validate", "phase": "runtime"}
{"action": "Implement CUDA kernel-state flow-reference validation for sample-line metrics", "type": "cuda", "phase": "numerics"}
{"action": "Run sample-line convergence, GPU flow validation, and sanitizer tests", "type": "test", "phase": "validation"}
{"action": "Remove duplicate sample-line builders and dead callers after graph and test verification", "type": "refactor", "phase": "cleanup"}
{"action": "Run architecture, lint, type, import, and full fast-fail verification", "type": "validate", "phase": "acceptance"}
{"action": "Request final code review for the canonical sample-line rewrite", "type": "validate", "phase": "review"}
```

| Step | Precomputed agent | Model | Why |
|---|---|---|---|
| 1 | `test-automator` | `minimax-coding-plan/MiniMax-M3` | Failing regression and numerical reference tests |
| 2 | `python-pro` | `minimax-coding-plan/MiniMax-M3` | Qt-free Python geometry service |
| 3 | `python-pro` | `minimax-coding-plan/MiniMax-M3` | GPKG/mesh adapter refactor |
| 4 | `python-pro` | `minimax-coding-plan/MiniMax-M3` | PyQt5/MVP GUI adapter migration |
| 5 | `test-automator` | `minimax-coding-plan/MiniMax-M3` | Executor/result contract validation |
| 6 | `cpp-pro` | `minimax-coding-plan/MiniMax-M3` | CUDA/pybind11 state readback or reference hook |
| 7 | `test-automator` | `minimax-coding-plan/MiniMax-M3` | GPU, convergence, and sanitizer gates |
| 8 | `python-pro` | `minimax-coding-plan/MiniMax-M3` | Safe duplicate/dead-path removal |
| 9 | `test-automator` | `minimax-coding-plan/MiniMax-M3` | Acceptance and architecture checks |
| 10 | `general` | `minimax-coding-plan/MiniMax-M3` | Independent final review |

## Machine-readable dispatch block

```json
{
  "plan": "docs/plans/2026-07-27-canonical-sample-line-sampling.md",
  "worktree": ".worktrees/sample-line-canonical",
  "strategy": "sequential_shared_worktree_with_parallel_read_only_research_and_reviews",
  "steps": [
    {"id": 1, "action": "Write failing Python tests for the canonical sample-line contract and independent flow reference", "type": "test", "phase": "red", "agent": "test-automator", "model": "minimax-coding-plan/MiniMax-M3"},
    {"id": 2, "action": "Implement the Python plain-data canonical sample-line geometry service", "type": "coding", "phase": "geometry", "agent": "python-pro", "model": "minimax-coding-plan/MiniMax-M3"},
    {"id": 3, "action": "Refactor Python GPKG and mesh adapters to feed the canonical sample-line service", "type": "refactor", "phase": "adapters", "agent": "python-pro", "model": "minimax-coding-plan/MiniMax-M3"},
    {"id": 4, "action": "Refactor PyQt5 GUI sample-line integration to remove View-side mesh geometry construction", "type": "ui", "phase": "gui", "agent": "python-pro", "model": "minimax-coding-plan/MiniMax-M3"},
    {"id": 5, "action": "Validate Python executor flattening and sample-line result persistence contract", "type": "validate", "phase": "runtime", "agent": "test-automator", "model": "minimax-coding-plan/MiniMax-M3"},
    {"id": 6, "action": "Implement CUDA kernel-state flow-reference validation for sample-line metrics", "type": "cuda", "phase": "numerics", "agent": "cpp-pro", "model": "minimax-coding-plan/MiniMax-M3"},
    {"id": 7, "action": "Run sample-line convergence, GPU flow validation, and sanitizer tests", "type": "test", "phase": "validation", "agent": "test-automator", "model": "minimax-coding-plan/MiniMax-M3"},
    {"id": 8, "action": "Remove duplicate sample-line builders and dead callers after graph and test verification", "type": "refactor", "phase": "cleanup", "agent": "python-pro", "model": "minimax-coding-plan/MiniMax-M3"},
    {"id": 9, "action": "Run architecture, lint, type, import, and full fast-fail verification", "type": "validate", "phase": "acceptance", "agent": "test-automator", "model": "minimax-coding-plan/MiniMax-M3"},
    {"id": 10, "action": "Request final code review for the canonical sample-line rewrite", "type": "validate", "phase": "review", "agent": "general", "model": "minimax-coding-plan/MiniMax-M3"}
  ]
}
```

## Task 1: Establish failing tests and numerical oracle

**Files:**

- Add or modify the focused sample-line test module under `tests/` after inspecting existing test naming and QGIS setup helpers.
- Do not modify production code in this task.

- [x] Write a failing test proving a deterministic rectangular mesh and a sample `LineString` can be represented as plain records and produce the expected crossed-cell IDs, station ordering, normal, and intersection lengths.
- [x] Write a failing regression test for the reported GPKG path: a sample-line source must not pass tuples into a QGIS-geometry-only callback or raise the reported `AttributeError`.
- [x] Write a failing flow-reference test with known `h`, `hu`, `hv`, normal, cell indices, and weights. The independent Python oracle must compute:

```python
wet = h[cell_idx] > h_min
qn = hu[cell_idx] * normal_x + hv[cell_idx] * normal_y
expected_flow = float(np.sum(weights * np.where(wet, qn, 0.0)))
```

This algebraically matches the kernel's wet-cell `h * normal_velocity` without
performing an unsafe division by zero. Avoid calling any line-metrics
implementation from the reference.
- [x] Add cases for reversed line orientation, uniform flow, nonuniform manufactured state, dry cells, exact edge overlap, corner-only touch, and a line crossing multiple cells.
- [x] Run the focused tests and verify they fail for the expected missing canonical API or current tuple contract, not because of test setup errors.

Run inside the activated environment:

```bash
bash -c 'eval "$(/home/aaron/miniforge3/bin/mamba shell hook --shell bash)" && mamba activate qgis_stable && python -m unittest -v tests.test_sample_line_canonical'
```

Expected: targeted failures demonstrating the missing canonical path and regression.

## Task 2: Implement the canonical plain-data geometry service

**Files:**

- Modify `swe2d/services/line_sampling_service.py`.
- Modify or add a small focused plain-data type module only if the existing project conventions require it; do not add an abstraction with one consumer.

- [ ] Define the service-facing input contract for sample lines and mesh cells using existing project typing conventions.
- [ ] Implement validation for array shape, finite coordinates, minimum distinct points, valid cell index, and nonzero polygon area. Raise an actionable typed error for invalid required mesh data; report and skip only malformed optional source features under the documented policy.
- [ ] Implement one geometry backend using an already-installed OGR/GEOS capability. Do not add a dependency and do not import Qt or QGIS in the canonical service.
- [ ] Implement bounding-box candidate rejection, line orientation/normal calculation, line-cell intersection, positive intersection-length filtering, station projection, deterministic station sorting, and duplicate/overlap handling.
- [ ] Initialize all accumulators before intersection processing, including the current `exact_face_lens` defect area if face metadata remains temporarily.
- [ ] Return the canonical fields `line_id`, `line_name`, `normal_x`, `normal_y`, `cell_idx`, `weights`, and `station_m` with documented dtypes and equal lengths.
- [ ] Make the focused unit tests from Task 1 pass without adding a tuple/QGIS compatibility branch.

Run:

```bash
bash -c 'eval "$(/home/aaron/miniforge3/bin/mamba shell hook --shell bash)" && mamba activate qgis_stable && python -m unittest -v tests.test_sample_line_canonical'
```

Expected: canonical geometry and edge-case tests pass.

## Task 3: Migrate mesh and GeoPackage adapters

**Files:**

- Modify `swe2d/mesh/mesh_runtime_logic.py`.
- Modify `swe2d/core/gpkg_io.py`.
- Modify focused builder/GPKG tests under `tests/`.

- [x] Keep mesh topology conversion in `mesh_runtime_logic.py`, but make its plain coordinate-array output the explicit shared adapter contract for CSR polygons and triangle cells.
- [x] Add a GPKG adapter that extracts each enabled sample feature's line ID, name, enabled state, and coordinate array from the layer without constructing mesh-cell `QgsGeometry` objects for the canonical service.
- [x] Preserve source line orientation and fail clearly when required geometry or fields are invalid.
- [x] Replace `_mesh_cell_polygons_fn` callback wiring in `build_line_sampling_map_from_gpkg` with direct plain-data calls to the canonical service.
- [x] Keep builder behavior that rejects a configured sample-line source producing no valid map, but ensure malformed geometry errors identify the table and feature ID.
- [x] Make the GPKG regression test and existing run-context builder tests pass.

Run:

```bash
bash -c 'eval "$(/home/aaron/miniforge3/bin/mamba shell hook --shell bash)" && mamba activate qgis_stable && python -m unittest -v tests.test_sample_line_canonical tests.test_run_context_builder'
```

Expected: the original tuple `AttributeError` is absent and configured GPKG sample lines build a nonempty canonical map.

## Task 4: Migrate the PyQt5 GUI adapter under MVP boundaries

**Files:**

- Modify `swe2d/workbench/studio_dialog.py`.
- Modify related Studio adapter/controller files only where existing ownership requires it.
- Modify GUI parity tests.

- [x] Replace `_mesh_cell_polygons` QGIS polygon construction in the View with a protocol/adaptor call that returns plain sample-line records or invokes a service-owned mesh-data builder.
- [x] Keep widget reads in View methods and keep controllers free of raw widget access.
- [x] Ensure the GUI adapter passes the same mesh data and line-record shape as the GPKG adapter.
- [x] Preserve line IDs, names, enabled filtering, orientation, and user-visible result naming.
- [x] Add a parity test that feeds identical line and mesh data through GUI and GPKG adapters and compares canonical output arrays within deterministic geometry tolerances.
- [x] Verify no canonical service import or call path uses `QgsGeometry`, `QComboBox`, or widget references.

Run:

```bash
bash -c 'eval "$(/home/aaron/miniforge3/bin/mamba shell hook --shell bash)" && mamba activate qgis_stable && python -m unittest -v tests.test_sample_line_canonical tests.test_workbench_imports tests.test_workbench_gui'
```

Expected: GUI parity and MVP architecture checks pass.

## Task 5: Preserve executor and results contracts

**Files:**

- Modify `swe2d/core/executor.py` only if canonical map field access needs a direct update.
- Modify `swe2d/runtime/backend.py` only if validation metadata must be exposed.
- Modify focused executor/results tests.

- [ ] Verify the executor consumes only the canonical fields required to create `station_offsets`, flat `cell_idx`, flat `weights`, `normal_x`, `normal_y`, and `station_m`.
- [ ] Reject inconsistent array lengths and invalid indices before backend configuration; do not silently clip or substitute values.
- [ ] Preserve line ordering and line IDs used by `SWE2DResultsData.populate_live_line_metrics_from_gpu`.
- [ ] Prove `swe2d_baked_line_ts` and `swe2d_baked_line_profiles` persistence/readback remains byte/schema compatible.
- [ ] Add an executor contract test that asserts the exact flattened arrays sent to a test backend from a canonical map.

Run:

```bash
bash -c 'eval "$(/home/aaron/miniforge3/bin/mamba shell hook --shell bash)" && mamba activate qgis_stable && python -m unittest -v tests.test_sample_line_canonical tests.test_run_context_parity tests.test_workbench_persistence'
```

Expected: flattened GPU inputs and result persistence tests pass without schema changes.

## Task 6: Add kernel-state flow-reference validation

**Files:**

- Modify `cpp/src/swe2d_gpu.cu` and `cpp/src/swe2d_bindings.cpp` only if the existing snapshot/readback API cannot expose the exact state needed by the test.
- Modify `swe2d/runtime/backend.py` only for a narrow explicit readback/diagnostic API.
- Add focused GPU validation tests.

- [ ] First attempt validation using existing exact `h`, `hu`, and `hv` snapshot/readback arrays at the line-metric timestep; do not add a native hook if that is sufficient.
- [ ] If required, add an explicit read-only diagnostic readback for the state or per-station `qn` used by the line metric. It must not recompute the value in Python when the kernel already computes it; the kernel remains the source of truth.
- [ ] Keep the production line-flow formula unchanged unless the independent reference proves a real defect.
- [ ] Compare GPU line flow against the independent reference with both absolute and relative tolerances, recording line ID, timestep, expected, actual, absolute error, relative error, and map shape/checksum.
- [ ] Add a direct face-flux comparison only if the kernel exposes a compatible oriented face-flux quantity. Document sign conventions, boundary faces, wet/dry rules, timestep stage, and why the quantities are mathematically comparable.
- [ ] Do not claim a universal analytic maximum error. Implement a manufactured-state refinement/convergence test and optionally calculate a conservative per-cell bound only when a valid flux variation bound is available.

Required GPU checks when native code changes:

```bash
bash -c 'eval "$(/home/aaron/miniforge3/bin/mamba shell hook --shell bash)" && mamba activate qgis_stable && python3 tools/run_compute_sanitizer.py --tool memcheck --test dambreak'
PYTHONPATH="$PWD:$PWD/build" bash -c 'eval "$(/home/aaron/miniforge3/bin/mamba shell hook --shell bash)" && mamba activate qgis_stable && python3 -m unittest -v tests.test_swe2d_gpu_validation_perf tests.test_swe2d_gpu_unstructured tests.test_swe2d_gpu_dambreak'
```

Expected: sanitizer is clean and flow-reference tests pass within documented tolerances.

## Task 7: Run convergence and integrated validation

**Files:**

- Add or modify numerical validation tests and artifacts under `tests/` only.
- Do not alter production behavior to make a numerical test pass without confirming the reference state and units.

- [x] Validate uniform flow analytically: constant `h`, `u`, `v`, and line normal must produce the expected discharge from total intersection length.
- [x] Validate a manufactured spatially varying state against the independent weighted-cell reference.
- [x] Refine the mesh or station/cell representation and report error trend. If no convergence or valid bound exists, fail the test with diagnostic evidence rather than accepting an arbitrary threshold.
- [x] Validate reversed line orientation produces opposite signed flow while preserving magnitude for the same physical cut.
- [x] Validate dry-cell handling against the kernel's `h_min` rule.
- [x] Run the focused test module, workbench import/persistence tests, and the repository fast-fail set.

Run:

```bash
bash -c 'eval "$(/home/aaron/miniforge3/bin/mamba shell hook --shell bash)" && mamba activate qgis_stable && python -m unittest -v tests.test_sample_line_canonical tests.test_workbench_imports tests.test_workbench_persistence tests.test_run_context_builder'
bash -c 'eval "$(/home/aaron/miniforge3/bin/mamba shell hook --shell bash)" && mamba activate qgis_stable && bash tools/fast_fail.sh'
```

Expected: focused tests and fast-fail pass; any GPU-only gate is reported with its exact result.

## Task 8: Remove duplicate implementations after call-site verification

**Files:**

- Modify `swe2d/services/line_sampling_service.py`.
- Modify `swe2d/workbench/services/` or legacy callers only where graph/call-site analysis proves they are dead.
- Modify tests that directly target deleted APIs only when they are obsolete and replaced by canonical tests.

- [x] Use graph/call-site analysis and repository search to enumerate all callers of `build_line_sampling_map`, `build_line_sampling_map_numpy`, legacy CPU metric helpers, `_mesh_cell_polygons`, and related callbacks.
- [x] Remove duplicate builders and dead helpers only after every production caller uses the canonical service.
- [x] Do not leave a compatibility shim that accepts both tuples and QGIS geometries.
- [x] Update `__all__`, imports, tests, and documentation references in the same change.
- [x] Purge Python caches after structural Python changes:

```bash
find . -type d -name __pycache__ -exec rm -rf {} +
```

Expected: no dead production caller remains and no old QGIS-geometry callback contract is reachable.

## Task 9: Acceptance verification

**Files:**

- No production changes unless verification exposes a defect; fix the smallest root cause and rerun the relevant review.

- [ ] Run the project’s architecture checks: no Qt/QGIS imports in canonical service, no raw widget access in controllers, no View-side NumPy geometry computation, and no silent fallback reads.
- [ ] Run import-path and keyword-only-call checks required by `MVP_ARCHITECTURE.md`.
- [ ] Run focused sample-line, builder, workbench, persistence, and parity tests.
- [ ] Run lint/typecheck commands if configured by the repository; otherwise record that no dedicated command exists and use the required Python test gates.
- [ ] Run GPU sanitizer and GPU validation if Task 6 changed native code or GPU-facing behavior.
- [ ] Inspect `git diff`, `git status --short`, and all changed files for scope, secrets, stale caches, and accidental schema changes.

Expected: all applicable gates pass with evidence recorded before any completion claim.

## Task 10: Independent final review

- [ ] Dispatch a fresh reviewer with the spec, plan, changed-file list, test evidence, and base/head diff.
- [ ] Require review of numerical correctness, source-of-truth compliance, MVP boundaries, error handling, schema compatibility, and removal of silent fallbacks.
- [ ] Resolve all critical and important findings, rerun affected tests, and request re-review.
- [ ] Do not mark the plan complete or archive the spec/plan without explicit user approval.

## Superpowers workflow

- `brainstorming`: completed before this plan; the user approved the design and the added numerical validation requirements.
- `using-git-worktrees`: completed; continue in `.worktrees/sample-line-canonical`.
- `test-driven-development`: use for every production task; each behavior begins with a failing test and is verified red before implementation.
- `subagent-driven-development`: dispatch one fresh implementation subagent per sequential task, followed by spec-compliance review and code-quality review before advancing.
- `dispatching-parallel-agents`: use only for independent read-only research, call-site inventory, or review; do not parallel-edit shared implementation files.
- `fvm-cfd-solver-patterns`: use during Task 6 and Task 7 for GPU source-of-truth, units, wet/dry, face orientation, and sanitizer requirements.
- `verification-before-completion`: use before every task completion and before claiming the rewrite is complete.
- `requesting-code-review`: use after major phases and for Task 10 final review.

## Review checkpoints

After each implementation task:

1. Run the task’s focused tests.
2. Dispatch a spec-compliance reviewer; fix any missing or extra behavior.
3. Dispatch a different code-quality reviewer; fix correctness, architecture, or maintainability findings.
4. Update that task checkbox and the frontmatter `progress:` block in the same edit.
5. Purge `__pycache__` after structural Python changes.

The final integration review must inspect the complete diff from the worktree base commit, not only the most recent task.
