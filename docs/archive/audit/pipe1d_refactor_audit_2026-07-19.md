---
type: audit
status: complete
created: 2026-07-19
completed: 2026-07-25
---

# Pipe1D Refactor — Implementation Audit against Plan

**Date:** 2026-07-19  
**Status:** Refactor at functional but inconsistent state. Refactoring completed in increments; each phase shipped working but added compat layers between phases rather than refactoring fully.

---

## TL;DR — What the refactor actually achieved

| Goal from plan | Status | Notes |
|---|---|---|
| Eliminate `d_node_*` / `d_vnode_*` / `d_cell_from_node` / `d_cell_to_node` | ✅ Achieved in struct | Struct no longer references them; remaining read sites are in fallback code paths only |
| Manhole/inlet become full FV cells | ✅ Achieved in C++ | `d_cell_class` enum, `d_cell_crown`, `d_cell_rim`, volume-equivalent geometry |
| Unified face-flux kernel with 7 face classes | ✅ Achieved | `swe2d_unified_face_flux_kernel` dispatches on `face_class` |
| Two solve modes: Riemann + source-sink | ✅ Implemented | Source-sink used for `SURFACE_2D_INLET` and `SURFACE_2D_JUNCTION_OVERFLOW` |
| 11 Phase-1 TDD tests pass | ✅ All pass | `588f7b4` |
| `coupling.py` fully ported to new API | ❌ **Not done — uses compat shims everywhere** |
| Legacy bindings removed from `swe2d_bindings.cpp` | ❌ **Not done — old bindings still exported alongside new** |
| DRAINAGE_SOA semantics consistent | ⚠️ Partial — DSOA has fields for vnode-style indexing that no longer apply |

---

## Critical finding: `coupling.py` was NOT ported

Despite the commit message saying "removes all hasattr fallbacks", the actual file contains **at least 16 `hasattr` guards around the new bindings** and **8 active compat paths** that call the OLD bindings that should have been removed.

The pattern is uniformly:

```python
# Comment claims compat layer was removed.
if hasattr(native_mod, "swe2d_gpu_apply_pipe_end_bc"):
    native_mod.swe2d_gpu_apply_pipe_end_bc(int(self.n_cells), h_min)  # ← dead/legacy code
if hasattr(native_mod, "swe2d_pipe1d_outfall_bc_kernel_host"):
    native_mod.swe2d_pipe1d_outfall_bc_kernel_host(...)              # ← dead/legacy code
if hasattr(native_mod, "swe2d_gpu_apply_coupling_drainage"):
    native_mod.swe2d_gpu_apply_coupling_drainage(...)                # ← dead/legacy code
# Phase 3 — Face-flux coupling is now handled internally by swe2d_pipe1d_step
# ... retained as stubs for backward compat ...
```

This pattern **hides bugs**. The Phase-1 tests failed for genuine C++ reasons (sign error in SURFACE_2D_PIPE_END, slot surcharge not activating, null d_junction_node). The compat layers in coupling.py masked the test errors by either running old code or returning values from the wrong API surface.

---

## Detailed findings

### F1: coupling.py `_read_pipe1d_state` (lines 1465-1536) is a hybrid

- **Input**: Uses `swe2d_readback_cell_state` (NEW API), gated by `hasattr` fallback to `swe2d_pipe1d_readback_node_state` (OLD API).
- **Output**: Writes to keys `"node_depth"`, `"link_flow"`, `"struct_flow"` — these are OLD names. The new API returns `cell_h`, `cell_A`, etc.

**Required port:** Remove the `hasattr` fallback; use the new API only. Update the OUTPUT dict to use new keys (`cell_h`, `cell_surface_integral`, etc.) and update all downstream consumers.

File: `swe2d/runtime/coupling.py` lines 1465-1571.

### F2: coupling.py `apply_native_device_sources` (lines 1666-1789) keeps old BC kernels

Lines 1751-1759 still call the OLD BC kernels via `hasattr` guards. These are explicitly retained "as stubs for backward compat with older native modules" — but no such older native module exists. The new module is the only module that will be used after this commit.

**Required port:** DELETE lines 1751-1759 (entire `if hasattr(...)` chain).

Lines 1778-1786 still call `_build_and_apply_pipe_face_flux` "as a no-op fallback". The Python function itself should be deleted since `swe2d_pipe1d_step` now handles face flux internally.

**Required port:** Delete `_build_and_apply_pipe_face_flux` Python function (lines 2272-2380 approximately); delete the call at lines 1782-1786.

File: `swe2d/runtime/coupling.py` lines 1747-1786, 2272-2380.

### F3: `_build_pipe1d_mesh_on_device` (lines 2052-2228) still uses old names

Lines 2052 says "if native_mod is None or not hasattr(native_mod, 'swe2d_build_pipe1d_mesh')" — the OLD binding name. Should be `swe2d_build_unified_mesh`.

Lines 2111-2116: upload pipe-end rim via OLD `swe2d_pipe1d_upload_node_rim` — this is fine if kept (the binding still exists), but the new mesh build should handle rim internally.

Lines 2119-2166: `swe2d_pipe1d_upload_pipe_ends_and_junctions` — OLD API. Should be ported to the new per-face upload.

Lines 2176-2188: `swe2d_pipe1d_upload_outfall_state` — OLD API. Should be `swe2d_pipe1d_upload_outfall_bc`.

Lines 2204-2215: `swe2d_pipe1d_upload_junction_overflow_state` — keep but check signature alignment.

Lines 2225-2226: `swe2d_pipe1d_init_area_from_depth` — OLD API. Should be `swe2d_pipe1d_init_cell_area`.

**Required port:** Replace old binding names with new ones. Update argument shapes if the new bindings have different signatures.

File: `swe2d/runtime/coupling.py` lines 2052-2228.

### F4: Many small `hasattr` checks remain throughout coupling.py

Lines 1291, 1375, 1409, 1453, 1455, 1470, 1538, 1552, 1577, 1582, 1658, 1669, 1679, 1692, 1736, 1813, 1840, 1857, 1883, 1901, 1906, 1944, 1990, 2060.

These `hasattr` checks are defensive guards against missing bindings. After this refactor, ALL the bindings they check should exist. Remove the guards and let Python raise AttributeError if a binding is missing (fail-loud, fail-fast).

**Required port:** Remove every `hasattr(mod, "swe2d_...")` where the binding should exist after the refactor. Retain ONLY the guards for genuinely optional features (none currently).

File: `swe2d/runtime/coupling.py` (multiple sites).

### F5: `swe2d_bindings.cpp` exports BOTH old and new bindings

Lines 2052-2300 export the OLD bindings (`swe2d_build_pipe1d_mesh`, `swe2d_pipe1d_upload_node_depth`, etc.).
Lines 2328-2700 export the NEW bindings (`swe2d_build_unified_mesh`, `swe2d_pipe1d_upload_cell_h`, etc.).

Both sets are present. The old ones are dead unless external code uses them. Per the user's directive ("the refactor is supposed to touch the entire codebase"), the old bindings should be removed.

**Required port:** Delete the old binding exports from `cpp/src/swe2d_bindings.cpp`. The C++ functions behind them (e.g., `swe2d_pipe1d_upload_node_depth`, `swe2d_pipe1d_readback_node_state`, `swe2d_pipe1d_upload_pipe_ends_and_junctions`, `swe2d_pipe1d_upload_outfall_state`, `swe2d_pipe1d_upload_junction_overflow_state`) can be removed too — verify no internal callers remain.

Special cases to check:
- `swe2d_pipe1d_upload_node_rim` (line 2111) — might still be needed; verify
- `swe2d_pipe1d_upload_junction_overflow_state` (line 2204) — may or may not be obsolete; check if the data flow it supports is still needed
- `swe2d_pipe1d_step` (line 1988 area) — keep, this is the main step entry
- `swe2d_pipe1d_outfall_bc_kernel_host` (line 2024 area) — DELETE, absorbed into unified kernel

### F6: Tests still use OLD binding names

The test files all use `import hydra_swe2d as _MOD` or similar and call OLD binding names:
- `tests/test_swe2d_pipe1d.py` uses `swe2d_pipe1d_step`, `swe2d_pipe1d_upload_node_depth`, `swe2d_pipe1d_init_area_from_depth`
- `tests/test_swe2d_pipe1d_surcharge.py` uses similar old names
- `tests/test_swe2d_gpu_drainage_network.py` uses old SWE2D-side names (these are fine since they were not the refactor target)
- `tests/swmm_validation/compare.py` uses test runner

**Two options:**
1. **Port the tests to use new binding names** (most thorough)
2. **Keep the old binding names as aliases / wrappers** until tests can be ported separately

Per the user's directive ("the refactor is supposed to touch the entire codebase"), the tests should be ported. But this is a larger scope. Recommend doing this as a separate phase after the C++/coupling.py port lands.

### F7: `_depth_from_area` Python helper may need work

Line 1527 calls `_depth_from_area(cell_A, cell_shape_type, cell_width, cell_height)`. The new cell readback provides `cell_h` directly (depth above invert). The depth computation may be redundant or may compute differently than the C++ side. Verify and consider using the C++-computed `cell_h` directly.

### F8: Surface coupling not exercised in Phase 4 regression

Phase 4 found:
- `test_swe2d_gpu_drainage_network.TestPipeEndExchange.test_pipe_end_moves_water_downhill` FAIL — 14.5 m³ mass imbalance (real C++ bug in SURFACE_2D_PIPE_END kernel sign convention)
- `test_swe2d_gpu_drainage_network.TestPipeEndExchange.test_wet_pipe_drains_into_dry_surface_cells` FAIL — pipe storage INCREASES (real C++ sign error)

These are GENUINE bugs the compat layers are hiding. Once coupling.py is ported to the new API, these failures will be loud and the C++ kernels can be fixed.

### F9: Slot surcharge not activating

`tests/test_swe2d_pipe1d_surcharge.TestPreissmannSlot` — 3 tests fail because `cell_A == A_full` exactly. The runtime Godunov kernel is not applying the Preissmann slot extension.

This is a real C++ bug in the surcharge clamp logic — needs investigation of `swe2d_pipe1d_godunov_update_kernel`'s surcharge handling path. The init_cell_area simplification (Phase 3) removed slot from initialization (correct), but the runtime kernel path needs to actually apply slot.

### F10: CUDA null pointer in `upload_junction_overflow_state`

Lines 5498-5549 of `cpp/src/pipe1d.cu` has a patch loop that requires `p.d_junction_node != nullptr`. When there are no junctions in the mesh, `p.d_junction_node` is null and `cudaMemcpy` crashes.

**Required fix:** Add `p.d_junction_node` to the guard at line 5504.

---

## Recommended next steps (in order)

### Phase 5a — `coupling.py` clean port (no shims)

1. Delete every `hasattr` guard around old `swe2d_*` binding names except where the binding may genuinely be absent
2. Replace every OLD binding name with NEW binding name (use the table in F1-F3)
3. Update the OUTPUT dict schema in `_read_pipe1d_state` (F1)
4. Delete `_build_and_apply_pipe_face_flux` Python function
5. Delete the 1751-1759 compat calls
6. Delete the 1782-1786 `_build_and_apply_pipe_face_flux` call
7. Verify with: `tests.test_pipe1d_face_indexed_mesh`, `tests.test_swe2d_pipe1d`

### Phase 5b — `swe2d_bindings.cpp` legacy removal

1. Delete OLD binding exports: `swe2d_build_pipe1d_mesh`, `swe2d_pipe1d_upload_node_depth`, `swe2d_pipe1d_init_area_from_depth`, `swe2d_pipe1d_readback_node_state`, `swe2d_pipe1d_upload_pipe_ends_and_junctions`, `swe2d_pipe1d_upload_outfall_state`
2. Verify C++ function bodies behind them are also unused (or convert them to internal helpers if still needed by new code)
3. Build and re-run tests

### Phase 5c — Real C++ bug fixes (no compat layers)

These are real C++ bugs the compat layers were hiding:

1. **Sign error in SURFACE_2D_PIPE_END** — `test_pipe_end_moves_water_downhill`, `test_wet_pipe_drains_into_dry_surface_cells`. Investigate the unified face flux kernel for SURFACE_2D_PIPE_END class.
2. **Slot surcharge not activating** — `tests/test_swe2d_pipe1d_surcharge.TestPreissmannSlot`. Investigate Godunov update kernel surcharge clamp.
3. **Null d_junction_node in upload patch loop** — add guard.

These were masked by coupling.py still calling the old pipe_face_flux_kernel and pipe_end_bc kernels.

### Phase 5d — Test port

1. Port remaining test files to use new binding names
2. Remove SKIP-after-Phase-2.5 expectations
3. Update all `import _MOD = hydra_swe2d` / `swe2d_pipe1d_*` calls

---

## Why the refactor agents kept adding compat layers

Three reasons identified from inspecting the commits:

1. **Self-defensive behavior under uncertainty.** When an agent doesn't know if some other code depends on the old API, it adds an `if hasattr` guard to be safe. Refactoring requires the opposite: prove nothing depends on the old API, then delete it.

2. **Mismatch between comment-aspirations and reality.** Comments like "Phase 3 — Replaced by unified face-flux kernel inside swe2d_pipe1d_step (no separate BC/exchange calls needed)" describe what the code SHOULD do, then leave the OLD call in place "as a stub for backward compat". The comment is right about the architecture; the code contradicts it.

3. **No end-to-end test of coupling.py.** Phase 4 ran `tests.test_pipe1d_face_indexed_mesh` (which uses `import hydra_swe2d as _MOD` and direct calls) and `tests.test_swe2d_pipe1d` (which uses the same approach). Neither goes through `coupling.py`. So incompatibilities between coupling.py and the new APIs were never tested.

The test gap is structural: coupling.py is never exercised by the Phase-1 tests. To validate the refactor end-to-end we need either:
1. Add `tests/test_coupling_unified_mesh.py` that exercises `SWE2DCouplingController` against the new mesh, OR
2. Port `tests/test_swe2d_pipe1d.py` etc. to use `SWE2DCouplingController` instead of direct `_MOD` calls

---

## Files to modify (next pass)

| File | Type | Scope |
|---|---|---|
| `swe2d/runtime/coupling.py` | Port | F1, F2, F3, F4 — no compat shims |
| `cpp/src/swe2d_bindings.cpp` | Remove | F5 — delete old binding exports |
| `cpp/src/pipe1d.cu` | Fix | F9 (slot), F10 (null guard) |
| `cpp/src/swe2d_gpu.cu` | Fix | F8 (SURFACE_2D_PIPE_END sign) |
| `tests/test_swe2d_pipe1d.py` etc. | Port | F6 — rename old binding calls |
| New: `tests/test_coupling_unified_mesh.py` | Add | E2E coverage of coupling.py |
