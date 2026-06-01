# Native Batched Coupled Stepping Implementation Plan

Date: 2026-05-31

## Objective

Increase SWE2D computational efficiency and GPU utilization by moving coupled timestep execution out of the Python per-step loop and into a native batched run path.

The highest-impact target is coupled runs with drainage, hydraulic structures, bridges, rain/source forcing, or dynamic boundary conditions. These currently fall back to Python-driven stepping, which can force full state copies from GPU to host every timestep before uploading source fields back to the device.

## Current Bottleneck

Uncoupled runs can use the native `swe2d_run_to_time` path, which keeps stepping inside C++/CUDA and avoids Python orchestration overhead.

Coupled runs currently use `SWE2DRuntimeStepExecutor`, which commonly does:

1. `backend.step(...)`
2. `backend.get_state()` device-to-host copy of `h`, `hu`, `hv`
3. Python coupling/source assembly
4. `set_external_sources_native(...)` host-to-device source upload
5. repeat every timestep

This host/device ping-pong limits GPU occupancy and makes per-step Python overhead a dominant cost on many real workbench runs.

## Target Design

Add a native coupled run API, tentatively:

```text
swe2d_run_coupled_to_time(...)
```

The native path should keep these resident across many timesteps:

- solver state arrays: `h`, `hu`, `hv`
- external/coupled source buffers
- packed hydraulic-structure and bridge metadata
- packed drainage exchange metadata, where supported
- boundary hydrograph tables
- rain/CN source state
- compact diagnostics buffers

Python/QGIS should only re-enter for coarse progress, cancellation, snapshots, line exports, and UI updates.

## Phase 1: Native Coupled Structures And Bridges

Scope:

- Extend native packed structure data to include enough bridge metadata for source generation.
- Port the existing bridge CUDA helper path and Phase 3 stacked bridge redistribution to native/CUDA-side source assembly.
- Add a native source buffer accumulation step for structures/bridges before each SWE update.

Primary files:

- `cpp/src/swe2d_solver.cpp`
- `cpp/src/swe2d_gpu.cu`
- `cpp/src/swe2d_bindings.cpp`
- `swe2d/runtime/coupling.py`
- `swe2d/runtime/bridge_stacked_runtime.py`

Acceptance:

- Bridge/structure coupled runs can step without per-step `get_state()`.
- Legacy scalar bridge mode and Phase 3 spatial bridge mode both remain selectable.
- Net source conservation matches the current Python path within tolerance.

## Phase 2: Native Batched Run Entry Point

Scope:

- Add `swe2d_run_coupled_to_time` beside `swe2d_run_to_time`.
- Start with fixed timestep and native source assembly only.
- Return batched diagnostics at configurable intervals.
- Preserve Python fallback when unsupported coupling features are active.

Primary files:

- `cpp/src/swe2d_solver.cpp`
- `cpp/src/swe2d_bindings.cpp`
- `swe2d/runtime/backend.py`
- `swe2d/workbench/extracted/model_and_run_methods.py`
- `swe2d/runtime/runtime_step_executor.py`

Acceptance:

- Workbench can select the native coupled run path automatically when supported.
- Existing Python loop remains the fallback path.
- Cancellation/progress still work at a coarse interval.

## Phase 3: Dynamic Forcing Coverage

Scope:

- Ensure native dynamic BC hydrographs, rain/CN forcing, and internal cell sources can be evaluated inside the native loop.
- Keep source and boundary forensic accounting optional and batched rather than per-step host-driven.
- Add adaptive timestep support after fixed-dt validation is stable.

Acceptance:

- Native coupled path supports common real project combinations:
  - dynamic flow/stage BCs
  - rain/CN forcing
  - bridge/structure coupling
  - device-resident external sources
- Unsupported combinations are logged clearly and fall back safely.

## Phase 4: Drainage Coupling Expansion

Scope:

- Reuse or extend existing GPU drainage kernels for native-loop execution.
- Keep CPU drainage as fallback.
- Avoid host state copies for surface exchange where GPU drainage is selected.

Acceptance:

- GPU drainage coupled runs remain device-resident across a batch.
- CPU drainage runs still use the current Python path unless/until a native CPU batch path is worthwhile.

## Validation Plan

Correctness:

- Compare Python-loop and native-loop results for fixed-dt bridge/structure cases.
- Check mass/source conservation per batch and at final time.
- Preserve existing GPU validation priority:
  - `tests/test_swe2d_gpu_validation_perf.py`
  - `tests/test_swe2d_gpu_unstructured.py`

New tests:

- Native coupled bridge conservation test.
- Native vs Python coupled source equivalence test.
- Unsupported-feature fallback test.
- Batched diagnostics cadence test.

Performance:

- Use existing timing diagnostics to compare average `wall`, `step`, `state`, `source`, and `coupling` times.
- Success metric: coupled native path should move average `gpu_frac` close to uncoupled/native-run behavior, with `state`, `source`, and Python `coupling` costs near zero during supported batches.

## Rollout Strategy

1. Add the native path behind an opt-in environment flag or GUI advanced toggle.
2. Enable automatically only for supported fixed-dt bridge/structure cases.
3. Expand support feature-by-feature as validation lands.
4. Keep the current Python loop as the compatibility path until native coverage is broad.

Suggested flag:

```text
BACKWATER_SWE2D_NATIVE_COUPLED_RUN=1
```

## Main Risks

- Divergence between Python and native coupling semantics.
- Harder debugging when source generation is fully native.
- Batched diagnostics hiding short-lived instability.
- Feature matrix complexity as drainage, rain, dynamic BCs, and bridges combine.

Mitigations:

- Keep small deterministic equivalence tests for each supported feature.
- Preserve verbose per-step Python mode as a debug fallback.
- Return enough batched diagnostics to identify the first failing interval.
- Roll out one coupling family at a time.
