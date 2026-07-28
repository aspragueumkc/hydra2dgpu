---
type: plan
status: complete
created: 2026-07-20
completed: 2026-07-25
---

# Plan — Delete compat shims in coupling.py (Phase 5a)

**Date:** 2026-07-20
**Status:** Implementation in progress.

## Goal
Delete every Python-side compat shim in `swe2d/runtime/coupling.py` so the file
calls the new `swe2d_*` binding names directly. Do NOT add any new compat shim.
Do NOT add any new `hasattr` guard around a binding that should exist.

## Step-by-step

### Step 1 — Rename OLD bindings to NEW bindings in coupling.py
OLD → NEW mapping (only `coupling.py` is touched):
| OLD binding name | NEW binding name |
|---|---|
| `swe2d_build_pipe1d_mesh` | `swe2d_build_unified_mesh` |
| `swe2d_pipe1d_readback_node_state` | `swe2d_readback_cell_state` |
| `swe2d_pipe1d_upload_node_depth` | `swe2d_pipe1d_upload_cell_h` |
| `swe2d_pipe1d_init_area_from_depth` | `swe2d_pipe1d_init_cell_area` |
| `swe2d_pipe1d_upload_pipe_ends_and_junctions` | `swe2d_pipe1d_upload_pipe_end_surface_faces` |
| `swe2d_pipe1d_upload_outfall_state` | `swe2d_pipe1d_upload_outfall_bc` |

### Step 2 — DELETE calls to absorbed BC kernels (replaced inside `swe2d_pipe1d_step`)
- `swe2d_gpu_apply_pipe_end_bc`
- `swe2d_pipe1d_outfall_bc_kernel_host`
- `swe2d_gpu_apply_coupling_drainage`
- `swe2d_gpu_apply_pipe_face_flux`

### Step 3 — DELETE the Python function `_build_and_apply_pipe_face_flux`
~lines 2277–2386. Remove both the def and the call site.

### Step 4 — Remove every `hasattr(native_mod, "swe2d_…")` guard
Audit doc F4 lists: lines 1291, 1375, 1409, 1453, 1455, 1470, 1538, 1552,
1577, 1582, 1658, 1669, 1679, 1692, 1736, 1813, 1840, 1857, 1883, 1901,
1906, 1944, 1990, 2060. None of these bindings is documented as optional
in the codebase. Remove each.

### Step 5 — Update `readback_coupling_state` output dict to NEW keys
Reads (input) use `swe2d_readback_cell_state` returning `cell_h`, `cell_A`,
`cell_Q`, `cell_invert`, `cell_width`, `cell_height`, `cell_q`, etc.
Writes (output) should expose the SAME keys the new API returns, plus the
3 derived aggregate keys (`link_q`, `struct_q`, `coupling_node_depth`,
`coupling_rain_cum_mm`, …) that consumers actually use. Downstream consumers
will be updated to read new keys.

### Step 6 — Build, test, commit

```bash
cd build
mamba run -n qgis_stable cmake --build . -j$(nproc)
cd ..
find . -type d -name __pycache__ -exec rm -rf {} +
mamba run -n qgis_stable env PYTHONPATH="$PWD:$PWD/build" \
    python3 -m unittest -v tests.test_pipe1d_face_indexed_mesh 2>&1 | tail -25
```

Expected: 11/11 PASS (unchanged — these tests already use new bindings).

```bash
mamba run -n qgis_stable env PYTHONPATH="$PWD:$PWD/build" \
    python3 -m unittest -v tests.test_swe2d_pipe1d 2>&1 | tail -25
```

Expected: many failures (or not) — DOCUMENTED in the commit body.

```bash
git add -A
git commit -m "..."
git push
```

## OUT OF SCOPE (NOT in this change)

- C++ bindings: do NOT delete `swe2d_build_pipe1d_mesh`, `swe2d_pipe1d_upload_node_depth`, etc. from `cpp/src/swe2d_bindings.cpp`. Those remain so old test files keep working until Phase 5b.
- Tests: do NOT port `tests/test_swe2d_pipe1d.py` etc. Failure is acceptable because the tests directly call OLD bindings; compat layers in `coupling.py` do not affect those tests. Phase 5d ports the tests.

## Files to modify
- `swe2d/runtime/coupling.py` (rename + delete)

## Files NOT to modify
- `cpp/src/swe2d_bindings.cpp` (old bindings stay)
- `tests/test_swe2d_pipe1d.py` (port in Phase 5d)
- `tests/test_swe2d_pipe1d_surcharge.py` (port in Phase 5d)
- `tests/test_pipe1d_mass_conservation.py` (port in Phase 5d)
- `tests/test_swe2d_gpu_drainage_network.py` (port in Phase 5d)
- `tests/test_hllc_standalone.py` (port in Phase 5d)
- `tests/test_drainage_inlet_outfall_vs_swmm.py` (port in Phase 5d)
- `tests/swmm_validation/compare.py` (port in Phase 5d)
- `tests/test_pipe1d_face_indexed_mesh.py` (already uses new API)
