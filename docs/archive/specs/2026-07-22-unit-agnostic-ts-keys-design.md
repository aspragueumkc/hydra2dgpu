---
type: spec
status: complete
created: 2026-07-22
completed: 2026-07-25
---

# Unit-Agnostic Line TS / Profile Key Names

## Problem

Line timeseries and profile dict keys encode units in their names
(`depth_m`, `velocity_ms`, `wse_m`, `bed_m`, `flow_cms`). The model
may run in feet/seconds (US Customary), making the `_m`/`_cms`/`_ms`
suffixes factually wrong.  Display labels are already unit-aware via
`_unit_labels()` / `_label_for_var()` — the key names themselves
drive no conversion.  This is purely a naming clean-up.

## Scope

| Old | New | GPU index | Used where |
|-----|-----|-----------|------------|
| `depth_m` | `depth` | 0 | TS + profile |
| `velocity_ms` | `velocity` | 1 | TS + profile |
| `wse_m` | `wse` | 2 | TS + profile |
| `bed_m` | `bed` | 3 | TS only |
| `flow_cms` | `flow` | 4 | TS only |
| `wet_frac` | `wet_frac` | — | TS only (keep) |
| `fr` | `fr` | — | TS + profile (keep) |
| `flow_qn` | `flow_qn` | — | Profile only (keep — `_qn` is quantity, not unit) |

No change to GPU kernel numeric indices (0–6).
No change to GPKG column names (already unit-agnostic: `flow_blob` etc.).
No change to `q_cms` field-name resolution in internal flow logic (user data, not ours).

## Touchpoints

### Category A — Internal dict keys (`_ts_keys` / `_prof_keys` tuples)
- `swe2d/results/data.py` — `_ts_keys`, `_prof_keys`, `ts_var_key` default
- `swe2d/runtime/run_finalizer.py` — key remapping dicts
- `swe2d/services/line_sampling_service.py` — return dict keys
- `swe2d/workbench/services/line_sampling_service.py` — agg read
- `swe2d/results/profile_service.py` — known key set
- `swe2d/results/structure_service.py` — `flow_cms` usage

### Category B — Display label maps
- `swe2d/services/results_render_service.py` — `_label_for_var` table, label tuple builders
- `swe2d/workbench/views/studio_viewer_pg.py` — `_label_for_var`, `_var_from_label`, defaults
- `swe2d/workbench/views/studio_viewer_profile_pg.py` — `_label_for_var` table

### Category C — GPKG serialization
- `swe2d/services/gpkg_persistence_service.py` — `persist_baked_line_ts` param names, `load_baked_line_timeseries` return dict keys

### Category D — Unit function names
- `swe2d/units.py` — `flow_si_to_model()` (keep name, it's a function not a key)
- `swe2d/workbench/services/unit_conversion_service.py` — same

### Category E — Coupling dialog suffix check
- `swe2d/workbench/dialogs/coupling_results_dialog.py` — `endswith("_cms")` / `endswith("_m")` logic (no change — applies to external coupling metric names, not our TS keys)

### Category F — Tests (~20 files)
Straightforward find-and-replace in assertions.

## Backward Compatibility

All changes happen within a single restart boundary (`_ts_keys` is the
single source of truth, every reader iterates it).  No old-key-in-new-code
path exists because the rename is atomic across the entire codebase.

The **only** compat concern is `load_baked_line_timeseries()` in the
live-data path (`data.py:1637–1648` and `gpkg_persistence_service.py:1644–1647`),
where it reads `raw.get("flow_cms")` from `_live_line_ts`.  Add a fallback:
```python
result["flow"] = np.asarray(
    raw.get("flow", raw.get("flow_cms")), dtype=np.float64
) if raw.get("flow") is not None or raw.get("flow_cms") is not None else np.empty(0, dtype=np.float64)
```
Same pattern for `depth`, `velocity`, `wse`, `bed`.

No GPKG schema migration needed — column names `flow_blob` etc. are
already unit-agnostic.

## Order of Implementation

1. Update `_ts_keys` / `_prof_keys` tuples in `data.py` (domino effect — all loops iterate these)
2. Update display label maps in render service and viewer files
3. Update GPKG persistence parameter names and return dict keys
4. Add compat fallback in `load_baked_line_timeseries` live-data path
5. Update remaining service files (structure_service, line_sampling_service, run_finalizer, profile_service)
6. Update tests
7. Rebuild `.so` (no C++ changes needed — GPU kernel uses numeric indices)
8. Purge `__pycache__`
