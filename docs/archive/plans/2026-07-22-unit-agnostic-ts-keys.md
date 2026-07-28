---
type: plan
status: complete
created: 2026-07-22
completed: 2026-07-25
---

# Unit-Agnostic TS/Profile Key Names — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename internal dict keys `depth_m`→`depth`, `velocity_ms`→`velocity`, `wse_m`→`wse`, `bed_m`→`bed`, `flow_cms`→`flow` across the codebase with backward-compat fallback in the one live-data read path.

**Architecture:** `_ts_keys` / `_prof_keys` tuples in `data.py` are the single source of truth — every loop iterates them. Change the tuples and all consumers follow. GPU kernel uses numeric indices (0–6), unchanged. GPKG column names (`flow_blob` etc.) are already unit-agnostic, unchanged.

**Tech Stack:** Python, C++/CUDA (no kernel changes), numpy, pybind11

---

### Task 1: Update central key tuples in data.py

**Files:**
- Modify: `swe2d/results/data.py`

- [ ] **Step 1: Update `append_line_snapshot` key tuple**

```python
        for key in ("depth", "velocity", "wse", "bed", "flow", "wet_frac", "fr"):
```

- [ ] **Step 2: Update `build_precomputed_line_results` TS keys**

```python
            for k in ("depth", "velocity", "wse", "bed", "flow", "wet_frac", "fr"):
```

- [ ] **Step 3: Update profile keys in same method**

```python
            for k in ("depth", "velocity", "wse", "bed", "flow_qn", "fr"):
```

- [ ] **Step 4: Update `get_live_line_snapshot_rows` keys**

```python
                for key in ("depth", "velocity", "wse", "bed", "flow", "wet_frac", "fr"):
```

- [ ] **Step 5: Update `populate_live_line_metrics_from_gpu` `_ts_keys`**

```python
        _ts_keys = ("depth", "velocity", "wse", "bed", "flow", "wet_frac", "fr")
```

- [ ] **Step 6: Update `_prof_keys` in same function**

```python
        _prof_keys = ("depth", "velocity", "wse", "bed", "flow_qn", "fr")
```

- [ ] **Step 7: Update `ts_var_key` default**

```python
        self.ts_var_key: str = "flow"
```

---

### Task 2: Update display label maps (results_render_service.py)

**Files:**
- Modify: `swe2d/services/results_render_service.py`

- [ ] **Step 1: Update `_label_for_var` table keys**

```python
    table = {
        "flow":            f"Flow ({u['flow']})",
        "depth":           f"Depth ({u['len']})",
        "wse":             f"WSE ({u['len']})",
        "velocity":        f"Velocity ({u['vel']})",
        "station":         f"Station ({u['len']})",
        "bed":             f"Bed ({u['len']})",
        "egl":             f"EGL Error ({u['len']})",
        "fr":              "Froude number",
        "flow_qn":         f"Normal flow ({u['flow']})",
    }
```

Note: `station_m` and `egl_m` are also used as keys in profile data — those refer to profile-specific fields (`station_m` is the station distance array, `egl_m` is energy grade line error). These map to `station` and `egl`.

- [ ] **Step 2: Update `_ts_var_labels` return**

```python
    return [
        (f"Flow ({u['flow']})",          "flow"),
        (f"Depth ({u['len']})",          "depth"),
        (f"WSE ({u['len']})",            "wse"),
        (f"Velocity ({u['vel']})",       "velocity"),
    ]
```

- [ ] **Step 3: Update `_profile_fill_labels` return**

```python
        (f"Flow ({u['flow']})",          "flow"),
```

- [ ] **Step 4: Update `_profile_var_labels` return keys** (depth, velocity)

```python
        (f"Depth ({u['len']})",          "depth"),
        (f"Velocity ({u['vel']})",       "velocity"),
        (f"EGLError ({u['len']})",       "egl"),
```

---

### Task 3: Update display label maps (studio_viewer_pg.py)

**Files:**
- Modify: `swe2d/workbench/views/studio_viewer_pg.py`

- [ ] **Step 1: Update `_label_for_var` table**

```python
    table = {
        "flow":            f"Flow ({u['flow']})",
        "depth":           f"Depth ({u['len']})",
        "wse":             f"WSE ({u['len']})",
        "velocity":        f"Velocity ({u['vel']})",
    }
```

- [ ] **Step 2: Update `_var_from_label` reverse map**

```python
    rev = {
        "flow": "Flow",
        "depth": "Depth",
        "wse": "WSE",
        "velocity": "Velocity",
    }
    for key, frag in rev.items():
        if frag in label:
            return key
    return "flow"
```

- [ ] **Step 3: Update default metric return values** (lines 67, 107, 303, 358, 677, 884)

All `return "flow_cms"` → `return "flow"` and `"flow_cms"` default → `"flow"`.

- [ ] **Step 4: Update metric combo keys** (lines 338, 858)

```python
        for key in ("flow", "depth", "wse", "velocity"):
```

---

### Task 4: Update display label maps (studio_viewer_profile_pg.py)

**Files:**
- Modify: `swe2d/workbench/views/studio_viewer_profile_pg.py`

- [ ] **Step 1: Update `_label_for_var` table**

```python
    table = {
        "flow":            f"Flow ({u['flow']})",
        "depth":           f"Depth ({u['len']})",
        "wse":             f"WSE ({u['len']})",
        "velocity":        f"Velocity ({u['vel']})",
        "station":         f"Station ({u['len']})",
        "bed":             f"Bed ({u['len']})",
        "egl":             f"EGL Error ({u['len']})",
    }
```

---

### Task 5: Update GPKG persistence service

**Files:**
- Modify: `swe2d/services/gpkg_persistence_service.py`

- [ ] **Step 1: Update `persist_baked_line_ts` parameter names and usages**

Function signature parameters `depth_m`, `velocity_ms`, `wse_m`, `bed_m`, `flow_cms` → `depth`, `velocity`, `wse`, `bed`, `flow`. Also the `VALUES` clause uses positional args, so no change needed there — just the parameter names and the docstring.

- [ ] **Step 2: Update `persist_baked_line_ts_batch` docstring and item key reads**

Line 1396 docstring: `depth_m` → `depth`, etc.
Line 1433 item key reads: `.get("depth_m")` → `.get("depth")` etc.

- [ ] **Step 3: Update `load_baked_line_timeseries` return dict keys**

```python
        return {
            "t_s": _f64(1),
            "depth": _f64(2),
            "velocity": _f64(3),
            "wse": _f64(4),
            "bed": _f64(5),
            "flow": _f64(6),
            "wet_frac": _f64(7),
            "fr": _f64(8),
        }
```

- [ ] **Step 4: Update `load_baked_line_timeseries` live-data path (lines 1644-1647)**

```python
            for k in ("depth", "velocity", "wse", "bed", "flow", "wet_frac", "fr"):
                v = raw.get(k, raw.get({  # backward compat w/ old names
                    "depth": "depth_m", "velocity": "velocity_ms",
                    "wse": "wse_m", "bed": "bed_m", "flow": "flow_cms",
                }.get(k)))
                result[k] = np.asarray(v, dtype=np.float64) if v is not None else np.empty(0, dtype=np.float64)
```

Wait, the fallback logic is a bit tricky inline. Let me think of a cleaner approach:

```python
_OLD_TS_KEY = {
    "depth": "depth_m", "velocity": "velocity_ms",
    "wse": "wse_m", "bed": "bed_m", "flow": "flow_cms",
}
for k in ("depth", "velocity", "wse", "bed", "flow", "wet_frac", "fr"):
    v = raw.get(k) or raw.get(_OLD_TS_KEY.get(k, ""))
    result[k] = np.asarray(v, dtype=np.float64) if v is not None else np.empty(0, dtype=np.float64)
```

Better to define the compat map at module level or as a closure.

---

### Task 6: Update run_finalizer.py

**Files:**
- Modify: `swe2d/runtime/run_finalizer.py`

- [ ] **Step 1: Update TS key remapping in the finalizer loop**

Lines 260-265:
```python
                        for k, pk in (("depth", "ts_depth"), ("velocity", "ts_velocity"),
                                      ("wse", "ts_wse"), ("bed", "ts_bed"),
                                      ("flow", "ts_flow"), ("wet_frac", "ts_wet_frac"),
                                      ("fr", "ts_fr")):
```

- [ ] **Step 2: Update profile key remapping**

Lines 270-275:
```python
                        for k, pk in (("depth", "prof_depth"), ("velocity", "prof_velocity"),
                                      ("wse", "prof_wse"), ("bed", "prof_bed"),
                                      ("flow_qn", "prof_flow_qn"), ("fr", "prof_fr"),
                                      ("wet", "prof_wet")):
```

- [ ] **Step 3: Update line_ts_items dict keys**

Lines 284-290:
```python
                        "depth": np.array(ld.get("depth", []), dtype=np.float64),
                        "velocity": np.array(ld.get("velocity", []), dtype=np.float64),
                        "wse": np.array(ld.get("wse", []), dtype=np.float64),
                        "bed": np.array(ld.get("bed", []), dtype=np.float64),
                        "flow": np.array(ld.get("flow", []), dtype=np.float64),
```

---

### Task 7: Update line_sampling_service.py

**Files:**
- Modify: `swe2d/services/line_sampling_service.py`
- Modify: `swe2d/workbench/services/line_sampling_service.py`

- [ ] **Step 1: Update return dict keys in `sample_line_aggregate_ts_row`**

`swe2d/services/line_sampling_service.py` lines 1113-1115:
```python
        "flow": flow_cms,
        "flow_cell": flow_cell_cms,
        "flow_fv": flow_fv_cms,
```

Wait — `flow_cell_cms` and `flow_fv_cms` are debug/component keys, not user-facing TS keys. These two are diagnostics (cell-based vs face-value flow). Let me keep them for diagnostics but rename the output key.

Actually, looking at the code more carefully, `flow_cell_cms` and `flow_fv_cms` are separate diagnostic fields returned by `sample_line_aggregate_ts_row`. They're read by `workbench/services/line_sampling_service.py`. Let me check what keys that file uses.

Let me look at the workbench version of this file.

Actually, I know from the explore agent's report that `workbench/services/line_sampling_service.py` reads `agg["flow_cms"]`, `agg["flow_cell_cms"]`, `agg["flow_fv_cms"]`. So I need to update all three files.

- [ ] **Step 2: Update workbench line_sampling_service reads**

`swe2d/workbench/services/line_sampling_service.py` lines 105-107:
```python
            "flow": agg["flow"],
            "flow_cell": agg["flow_cell"],
            "flow_fv": agg["flow_fv"],
```

---

### Task 8: Update remaining service files

**Files:**
- Modify: `swe2d/results/profile_service.py`
- Modify: `swe2d/results/structure_service.py`

- [ ] **Step 1: Update profile_service.py key references**

Lines 36-41 (base dict), lines 50-53 (array vars), lines 58-60 (rec.get), line 63 (k skip set):
```python
    base = {
        "station": np.empty(0, dtype=np.float64),
        "wse": np.empty(0, dtype=np.float64),
        "bed": np.empty(0, dtype=np.float64),
        "depth": np.empty(0, dtype=np.float64),
        "wet": np.empty(0, dtype=np.float64),
    }
```
And the variable names (wse, bed, depth arrays) — these are local variables, keep the short names but update the dict key names.

Actually, `profile_service.py` is used for profile data which comes from a different data source (coupling records / structure service). The profile dict keys (`wse_m`, `bed_m`, `depth_m`) are populated from coupling record data. Let me check how those records are structured...

Looking at the code, `extract_profile_arrays` reads from coupling records which have their own key schema. These profile keys may be used downstream in viewers that display profile data. Let me trace this more carefully.

Actually, looking at `structure_service.py` which builds profile records for structures, it uses `flow_cms` as a key. The profile service then reads those keys. Since structure profiles use coupling records (which use `flow_cms`, `wse_m` etc as metric names), the profile_service reads those keys from the coupling data.

But the user only asked about the results path TS/profile keys. The coupling/structure records are a different data domain with their own key naming. Let me check if structure_service keys overlap with the TS keys.

Looking at `structure_service.py`:
- Line 164: `"flow_cms": float` — this is in the profile data dict for structures
- Line 258: `"flow_cms": float(flow_by_id[sid])` — setting the key
- Line 274: `"flow_cms": float(r.get("value", 0.0))` — reading from coupling

These are structure/coupling profile records — they use a different schema than TS. The `flow_cms` key in structure profiles represents a different thing (per-structure flow). The `_cms` suffix here is part of the coupling metric system.

Actually, structure_service is not part of the line TS results path. The user's scope was "every variable with a unit suffix in the results path." Let me be conservative and only change what's clearly in the TS/profile results path. The coupling_results_dialog already handles `_cms` suffix heuristically.

Hmm, but `structure_service.py` is in `swe2d/results/` which is the results path. Let me re-examine...

Actually, I think the structure_service is about coupling results (structure flows), not line TS results. The `flow_cms` there refers to a structure coupling metric. Let me not touch that unless the keys overlap with the TS tuples.

Similarly, `profile_service.py` reads from coupling records. The profile keys like `wse_m`, `bed_m`, `depth_m` in profile_service are used to build profile dicts for coupling structures. These are NOT the same as the line TS keys (which I'm renaming).

So I should NOT touch profile_service.py or structure_service.py — those deal with coupling data, not line TS/profile data. The line profile keys (`prof_depth_m`, `prof_velocity_ms`, etc.) are already handled by the `_prof_keys` tuple rename, and those profiles flow through `data.py` → `run_finalizer.py` → `gpkg_persistence_service.py`, which I'm already updating.

Let me remove profile_service.py and structure_service.py from the plan — they're out of scope.

---

### Task 8 (revised): Update CLI/adapter field fallback lists

**Files:**
- Modify: `swe2d/cli/gpkg_adapter.py`
- Modify: `swe2d/boundary_and_forcing/internal_flow_logic.py`
- Modify: `swe2d/boundary_and_forcing/internal_flow_qgis_adapter.py`
- Modify: `swe2d/workbench/services/schema_definitions.py`

These files use `q_cms` / `flow_cms` as QGIS field name candidates (user data), not internal TS keys. The field name lookup is already a fallback chain. No change needed unless the user's GPKG layer actually uses `flow_cms` as a field name — but that's user data, not our code.

Actually, let me check `gpkg_adapter.py` line 570:
```python
    for cand in (requested_field_name, "q_cms", "flow_cms", "q", "flow"):
```

This is for user-data field names in QGIS layers. If users have fields named `q_cms` in their data, we should keep searching for those. But we don't need to add "flow" here — the `requested_field_name` already covers the user's configured name, and the fallback is for old naming conventions in user data. No change.

**Decision: skip Task 8 — these are user-data field names, not our internal keys.**

---

### Task 8: Update tests

**Files:**
- Modify: All test files that reference old key names

- [ ] **Step 1: Find all test files with old key references**

```bash
rg '"depth_m"|"velocity_ms"|"wse_m"|"bed_m"|"flow_cms"' tests/ -l
```

Expected files: `test_gpkg_line_results_roundtrip.py`, `test_gpkg_persistence.py`, `test_in_memory_results_render.py`, `test_line_results_plot.py`, `test_line_results_refactored.py`, `test_live_run_id.py`, `test_network_profile_plot_widget.py`, `test_profile_fallback_diagnostic.py`, `test_rcmk_permutation_mismatch.py`, `test_results_path_audit_fixes.py`, `test_results_render_service.py`, `test_results_structure_service.py`, `test_run_finalizer_profile_aggregation.py`, `test_sample_line_metrics_logic.py`, `test_sample_line_metrics_profile.py`, `test_workbench_gpkg_service.py`, `test_workbench_mesh_service.py`

- [ ] **Step 2: Replace old keys with new keys in all test files**

For each file, replace in assertions and test data:
- `"depth_m"` → `"depth"`
- `"velocity_ms"` → `"velocity"`
- `"wse_m"` → `"wse"`
- `"bed_m"` → `"bed"`
- `"flow_cms"` → `"flow"`

Exclude false positives — some tests reference `q_cms` (internal flow field name), `ts_depth_m` (GPKG table prefix), `_ms` in unit strings. Only replace dict key names and expected result keys.

---

### Task 9: Cleanup and verify

- [ ] **Step 1: Search for any missed references**

```bash
rg '"depth_m"|"velocity_ms"|"wse_m"|"bed_m"|"flow_cms"' swe2d/ --type py
```

If anything appears outside the backward-compat fallback, fix it.

- [ ] **Step 2: Purge __pycache__**

```bash
find . -type d -name __pycache__ -exec rm -rf {} +
```

- [ ] **Step 3: Run test suite**

```bash
mamba run -n qgis_stable python3 -m unittest discover -s tests -p 'test_*.py' -v 2>&1 | tail -30
```

- [ ] **Step 4: Fix any test failures from renamed keys**

If tests fail, check that assertions use the new key names and fix any missed references.

---

### Backward Compat Design (reference for Task 5 Step 4)

The compat fallback in `load_baked_line_timeseries` live-data path:

```python
# Mapping from new key → old key for backward compat
_TS_OLD_KEYS = {
    "depth": "depth_m",
    "velocity": "velocity_ms",
    "wse": "wse_m",
    "bed": "bed_m",
    "flow": "flow_cms",
}
for k in ("depth", "velocity", "wse", "bed", "flow", "wet_frac", "fr"):
    v = raw.get(k)
    if v is None:
        old_k = _TS_OLD_KEYS.get(k)
        v = raw.get(old_k) if old_k else None
    result[k] = np.asarray(v, dtype=np.float64) if v is not None else np.empty(0, dtype=np.float64)
```

No GPKG schema migration needed — column names `flow_blob` etc. are already unit-agnostic.

---

### Files NOT modified (verified out of scope)

| File | Reason |
|------|--------|
| `cpp/src/swe2d_gpu.cu` | GPU kernel uses numeric indices, not string keys |
| `cpp/src/swe2d_bindings.cpp` | No string key names in bindings |
| `swe2d/results/structure_service.py` | Coupling results, not line TS |
| `swe2d/results/profile_service.py` | Profile records from coupling data, not line TS |
| `swe2d/workbench/dialogs/coupling_results_dialog.py` | `_cms`/`_m` suffix heuristics for external coupling metrics |
| `swe2d/units.py` | `flow_si_to_model()` is a function name, not a key |
| `swe2d/cli/gpkg_adapter.py` | `q_cms` is user-data field name fallback |
| `swe2d/boundary_and_forcing/internal_flow_logic.py` | Same |
| `swe2d/workbench/studio_dialog.py` | Metric tuples use index-based refs through `results_render_service` |
