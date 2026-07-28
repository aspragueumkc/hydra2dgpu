---
type: audit
status: complete
created: 2026-07-13
completed: 2026-07-25
---

# Unit Handling Consistency Audit & Remediation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan phase-by-phase. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate hardcoded unit suffixes from variable names, standardize unit-naming conventions, and fix the `_m` suffix lie where variables named `depth_m` etc. are actually in model units (which may be feet).

**Architecture:** The codebase has a centralized unit system (`swe2d/units.py`, `cpp/src/swe2d_units.cuh`) but uses unit suffixes embedded in variable names as Hungarian-like documentation. This creates a data-contract problem (dictionary keys and SQL columns embed unit suffixes) and a correctness problem (the `_m` suffix is wrong in USC/feet mode). The fix must be phased: internal renames first, then data-contract migration with backward compatibility, then C++/CUDA cleanup.

**Tech Stack:** Python 3.12+, C++20, CUDA, PyQt5, QGIS, pybind11, GeoPackage persistence

**Audit date:** 2026-07-13

---

## Executive Summary

The codebase has **~1000+ variable names with embedded unit suffixes** across Python and C++/CUDA. There is a central `swe2d/units.py` that handles SI↔USC conversion, and a service layer wrapper at `swe2d/workbench/services/unit_conversion_service.py`.

**The single most dangerous inconsistency:** Variables named `depth_m`, `wse_m`, `bed_m`, `station_m`, `velocity_ms`, `flow_cms` are actually in **model units** (either meters or feet depending on CRS). When running in USC/feet mode, these `_m` and `_ms` suffixes are **lies** — the values are in feet, not meters. No runtime bug currently results because all downstream code treats them as opaque model-unit values, but rename `_m` → `_model` or remove the suffix entirely so the code doesn't mislead.

**Secondary inconsistencies:**
- Time: `t_sec` / `t_s` / `time_s` / `_t_s` / `duration_s` / `run_duration_s` / `output_interval_s` — 6 patterns for the same thing
- Velocity: `velocity_ms` / `_mps` / `_m_per_s` / `rate_mps` / `rate_mm_s` — mixed naming
- Flow: `flow_cms` / `_cms` / `q_cms` / `_cfs` / `_m3_s` / `_m3s` — mixed naming
- Dictionary keys in results data model (`depth_m`, `velocity_ms`, etc.) act as a **data contract** shared across processes and persisted to GeoPackage

---

## Audit Findings

### Finding 1: The `_m` suffix lie (HIGH risk — misleads about actual units)

**500+ occurrences** of variables named with `_m` suffix that are actually in model units (either m or ft):

| Pattern | Example | Files |
|---------|---------|-------|
| `depth_m` | `swe2d/results/data.py:259` | results, persistence, rendering, CLI, runtime |
| `wse_m` | `swe2d/results/data.py:259` | same spread |
| `bed_m` | `swe2d/results/data.py:259` | same spread |
| `station_m` | `swe2d/results/data.py:348` | line_sampling, persistence, C++ bindings |
| `velocity_ms` | `swe2d/results/data.py:259` | same spread — `_ms` means "meters/second" → wrong in USC |
| `flow_cms` | `swe2d/results/data.py:97` | same spread — `_cms` means "cubic meters/second" → wrong in USC |
| `area_m2` | `swe2d/extensions/extension_models.py:242` | extensions, GPU kernel parameters |
| `equiv_diameter_m` | `swe2d/runtime/coupling.py:1884` | coupling, drainage |
| `head_deadband_m` | `swe2d/extensions/extension_models.py:306` | drainage network |
| `influence_width_m` | `swe2d/mesh/bridge_stacked_mesh.py:20` | bridge mesh |

**These are found in:**
- In-memory dict keys (`data.py:259`: `for key in ("depth_m", "velocity_ms", ...)`)
- Persistent storage (`gpkg_persistence_service.py:523-527`: SQL column names via dict keys)
- C++ bindings (`swe2d_bindings.cpp:961`: `station_m` parameter)
- GPU kernel outputs (`swe2d_gpu.cu:9794-9798`: comments documenting output arrays)
- CLI headless runner (`headless_runner.py:745-756`: `ts_depth_m`, `prof_depth_m`)

### Finding 2: Mixed time-unit naming (MEDIUM risk — consistency)

| Pattern | Count | Examples |
|---------|-------|----------|
| `t_s` | ~80 | `runtime/backend.py`, `runtime/coupling.py`, `runtime/runtime_sources.py` |
| `t_sec` | ~40 | `results/data.py:106`, `boundary_and_forcing/bc_logic.py`, `results/queries.py` |
| `time_s` | ~20 | `runtime/backend.py:578`, `runtime/coupling.py:40` |
| `_t_s` | ~10 | `results/data.py:739` |
| `duration_s` | ~20 | `runtime/run_finalizer.py`, `run_log_storage.py` |
| `run_duration_s` | ~30 | `runtime/run_finalizer.py:118`, `run_service.py:46` |
| `output_interval_s` | ~30 | `swe2d/workbench/` across many files |
| `dt_s` | ~20 | `runtime/coupling.py:41`, C++ `pipe1d.cuh` |
| `t_hr` | ~15 | `results_render_service.py:130`, `studio_viewer_pg.py:735`, `hecras_export_service.py:216` |

### Finding 3: Mixed velocity naming (MEDIUM risk — confusion)

| Pattern | Examples |
|---------|----------|
| `velocity_ms` | `results/data.py:259` — dict key, most common output name |
| `_mps` | `swe2d_gpu.cuh:303-304` — `d_cell_source_mps`, `d_external_source_mps` |
| `_m_per_s` | `extension_models.py:102` — `cell_rain_rate_m_per_s` |
| `rate_mps` | `runtime/backend.py:673` — `source_rate_mps` |
| `rate_mm_s` | `runtime_setup_configurator.py:76` — mm/s, different unit entirely |
| `_ft_s` | (implied by `USC_GRAVITY` in ft/s²) |

### Finding 4: Mixed flow naming (MEDIUM risk — confusion)

| Pattern | Examples |
|---------|----------|
| `flow_cms` | `results/data.py:97` — most common output key |
| `q_cms` | `internal_flow_logic.py:16` |
| `_cms` suffix | `coupling_results_dialog.py:143` — detected via `.endswith("_cms")` |
| `_cfs` suffix | `swe2d_gpu.cu:3732` — used inside culvert hydraulics (truly in CFS) |
| `q_cfs` | `swe2d_gpu.cu:3732-4007` — local culvert variables truly in CFS |

### Finding 5: Rainfall `_mm` is CORRECT (LOW risk)

Rainfall is always tracked in mm (the raw data and hyetographs), then converted to model depth via `mm_to_model_depth` factor. This is the one area where the unit suffix is accurate.

| Pattern | Examples |
|---------|----------|
| `hg_cum_mm` | `rainfall_hydrology.py` — hyetograph cumulative mm |
| `cumulative_mm` | `rainfall_hydrology.py:96` |
| `rain_mm` | `rainfall_hydrology.py:357` — parameter |
| `total_depth_mm` | `rainfall_hydrology.py:224` |
| `scs_retention_mm` | `rainfall_hydrology.py:305` |
| `mm_to_model_depth` | `unit_conversion_service.py:65` — conversion factor |
| `rain_mm_to_model_depth` | `studio_dialog.py:1706` — conversion call |

### Finding 6: Culvert `_ft` naming is CORRECT (LOW risk)

Culvert hydraulics use HDS-5 tables which are inherently in feet. The `model_to_ft` conversion factor is well-documented.

### Finding 7: Dictionary keys as data contract (HIGH risk for migration)

The data model dictionary keys (`depth_m`, `velocity_ms`, `flow_cms`, `wet_frac`, `fr`, `flow_qn`, `station_m`, `wse_m`, `bed_m`) are:
1. Used as dictionary keys in `SWE2DResultData` (`results/data.py`)
2. Persisted as GeoPackage column names (`gpkg_persistence_service.py`)
3. Read back from GeoPackage with the same names
4. Transferred to GPU via pybind11 with the same naming
5. Exposed in CLI output files

Changing these keys requires backward-compatible read logic.

---

## Remediation Phases

### Phase 0: Naming Convention Standard (DOCUMENTATION ONLY — before any code change)

Define the target naming convention:

| Quantity | Current (bad) | Target convention |
|----------|---------------|-------------------|
| Depth in model units | `depth_m` | `depth` (no suffix — model units are abstract) |
| WSE in model units | `wse_m` | `wse` |
| Bed elevation in model units | `bed_m` | `bed` |
| Station in model units | `station_m` | `station` |
| Velocity in model units | `velocity_ms` | `velocity` |
| Flow in model units | `flow_cms` | `flow` |
| Flow in model units (boundary) | `q_cms` | `boundary_flow` |
| Time in seconds | `t_sec` / `time_s` / `t_s` | `time_s` (standardize on `time_s` or `t_s`) |
| Run duration | `run_duration_s` | `duration_s` |
| Output interval | `output_interval_s` | `output_interval_s` (keep — unambiguous) |
| Time step | `dt_s` | `dt` (context implies seconds) |
| Cumulative rainfall | `cumulative_mm` / `hg_cum_mm` | `rainfall_cumulative` (always in mm, documented) |
| Rain rate | `rate_mps` / `rain_rate_mps` | `rainfall_rate` |
| Cell area | `area_m2` / `cell_area_m2` | `cell_area` |
| Manning's n | `n_mann` | `mannings_n` |
| Manning factor | `k_mann` | `mannings_k` or `mannings_factor` |
| Froude number | `fr` | `froude` |
| Wet fraction | `wet_frac` | `wetted_fraction` |
| Flow QN | `flow_qn` | `flow_norm` (dimensionless) |
| Gravity | (in units.py) | `SI_GRAVITY` → keep (module-level constant) |

- [ ] Write `docs/UNIT_NAMING_CONVENTION.md` with the convention table and rationale

---

### Phase 1: Internal Python Variables (NO data contract changes)

Rename local variables and function parameters only — no dict keys, no SQL columns, no public API changes.

**Scope:** ~300 variables across `swe2d/runtime/`, `swe2d/mesh/`, `swe2d/extensions/`, `swe2d/workbench/controllers/`

**Risk:** Low — internal renames only, no persistence impact

- [ ] **Task 1.1:** Rename `_m` suffix internal variables in `swe2d/runtime/coupling.py`
  - `influence_width_m` → `influence_width`
  - `width_m` → `width`
  - `equiv_diameter_m` → `equiv_diameter`
  - `head_deadband_m` → `head_deadband`

- [ ] **Task 1.2:** Rename `_m` suffix internal variables in `swe2d/mesh/bridge_stacked_mesh.py`
  - Fields: `influence_width_m`, `upstream_buffer_m`, `downstream_buffer_m`, etc. → drop `_m` suffix

- [ ] **Task 1.3:** Rename `_m2` / `_m3` suffix variables in `swe2d/extensions/extension_models.py`
  - `area_m2` → `area`
  - `equiv_diameter_m` → `equiv_diameter`
  - `head_deadband_m` → `head_deadband`

- [ ] **Task 1.4:** Standardize time naming in `swe2d/runtime/`:
  - `t_sec` → `t_s` (pick `_s` as the standard second suffix)
  - `time_s` → `t_s` (where colloquial; keep `time_s` in kwargs/API for readability)

- [ ] **Task 1.5:** Standardize flow naming in `swe2d/boundary_and_forcing/`:
  - `q_cms` → `boundary_flow`
  - `flow_cms` → `flow` (local vars only, not dict keys)

- [ ] **Task 1.6:** Run full test suite to verify no breakage

  ```bash
  find . -type d -name __pycache__ -exec rm -rf {} +
  mamba run -n qgis_stable python3 -m unittest -v \
    tests.test_workbench_gui \
    tests.test_workbench_imports \
    tests.test_workbench_persistence \
    tests.test_swe2d_results_queries \
    tests.test_gpkg_persistence \
    tests.test_gpkg_line_results_roundtrip
  ```

---

### Phase 2: Data Contract Migration (MEDIUM risk — persistence)

Change dictionary keys and GeoPackage column names. Must include backward-compatible read logic for existing GeoPackage files.

**Scope:** `results/data.py`, `services/gpkg_persistence_service.py`, `services/results_render_service.py`, `services/line_sampling_service.py`, other services and views that reference these dict keys

**Approach:**
1. Define a key mapping in one place
2. Use the new keys everywhere
3. Add backward-compat read in `gpkg_persistence_service.py` to detect old keys

- [ ] **Task 2.1:** Create key mapping module `swe2d/results/key_map.py`

  ```python
  """Mapping from old (legacy) result keys to standardized keys."""

  RESULT_KEY_MAP: dict[str, str] = {
      "depth_m": "depth",
      "velocity_ms": "velocity",
      "wse_m": "wse",
      "bed_m": "bed",
      "station_m": "station",
      "flow_cms": "flow",
      "flow_qn": "flow_norm",
      "wet_frac": "wetted_fraction",
      "fr": "froude",
      "t_s": "time_s",  # only if we standardize to time_s
  }

  LEGACY_KEYS: frozenset[str] = frozenset(RESULT_KEY_MAP.keys())

  def standardize_key(key: str) -> str:
      return RESULT_KEY_MAP.get(key, key)

  def is_legacy_key(key: str) -> bool:
      return key in LEGACY_KEYS
  ```

- [ ] **Task 2.2:** Update `swe2d/results/data.py` — change all dict keys to new names

  ```python
  # OLD: for key in ("depth_m", "velocity_ms", "wse_m", "bed_m", "flow_cms", "wet_frac", "fr"):
  # NEW: for key in ("depth", "velocity", "wse", "bed", "flow", "wetted_fraction", "froude"):
  ```

- [ ] **Task 2.3:** Update `swe2d/services/gpkg_persistence_service.py` — write new column names, read both old and new

  ```python
  # When writing: use standardized keys
  # When reading: accept old column names, map to new keys via key_map
  ```

- [ ] **Task 2.4:** Update `swe2d/services/results_render_service.py` — label maps and rendering logic

- [ ] **Task 2.5:** Update `swe2d/services/line_sampling_service.py` — internal dict keys

- [ ] **Task 2.6:** Update `swe2d/results/structure_service.py` — key references

- [ ] **Task 2.7:** Update `swe2d/results/profile_service.py` — key references

- [ ] **Task 2.8:** Update `swe2d/cli/headless_runner.py` — TS/profile key names

- [ ] **Task 2.9:** Update `swe2d/runtime/run_finalizer.py` — result dict construction

- [ ] **Task 2.10:** Update `swe2d/workbench/services/line_sampling_service.py` — workbench-specific wrapping

- [ ] **Task 2.11:** Update `swe2d/workbench/views/studio_viewer_pg.py` — display key references

- [ ] **Task 2.12:** Update `swe2d/workbench/views/studio_viewer_profile_pg.py` — display key references

- [ ] **Task 2.13:** Update tests — all test files that reference old key names

- [ ] **Task 2.14:** Run full test suite to verify no breakage

  ```bash
  find . -type d -name __pycache__ -exec rm -rf {} +
  mamba run -n qgis_stable python3 -m unittest -v \
    tests.test_workbench_gui \
    tests.test_workbench_imports \
    tests.test_workbench_persistence \
    tests.test_swe2d_results_queries \
    tests.test_gpkg_persistence \
    tests.test_gpkg_line_results_roundtrip \
    tests.test_line_results_plot \
    tests.test_workbench_gpkg_service
  ```

---

### Phase 3: C++/CUDA Variable Renames (MEDIUM risk — bindings must match Python)

**Scope:** `cpp/src/swe2d_gpu.cuh`, `swe2d_gpu.cu`, `swe2d_solver.hpp`, `swe2d_solver.cpp`, `swe2d_bindings.cpp`, `pipe1d.cuh`, `pipe1d.cu`

**Risk:** Medium — C++ variable renames must exactly match Python-side names, especially in pybind11 bindings.

- [ ] **Task 3.1:** Rename device-side variable members in `swe2d_gpu.cuh`
  - `d_cell_source_mps` → `d_cell_source`
  - `d_external_source_mps` → `d_external_source`
  - `d_stage_cell_source_mps` → `d_stage_cell_source`
  - `d_rain_hg_cum_mm` → `d_rainfall_cumulative` (rain stays in mm — property is documented)
  - `d_rain_cum_mm` → `d_rainfall_cumulative_cell`
  - `d_rain_excess_cum_mm` → `d_rainfall_excess_cell`
  - `cell_area_m2` → `cell_area`
  - `culvert_area_m2` → `culvert_area`
  - `inlet_flow_cms` → `inlet_flow`
  - `structure_flow_cms` → `structure_flow`
  - `bridge_flow_cms` → `bridge_flow`
  - `d_lm_station_m` → `d_lm_station`
  - `d_hg_time_s` → `d_hg_time`
  - `d_rain_hg_time_s` → `d_rain_hg_time`
  - `rain_update_interval_s` → `rain_update_interval`
  - `mm_to_model_depth` → keep (the name describes what it IS, not a value unit)
  - `model_to_ft` → keep (same logic)
  - `k_mann` → `mannings_factor`

- [ ] **Task 3.2:** Rename local variables in `swe2d_gpu.cu`
  - `hg_cum_mm`, `cum_rain_mm`, `cum_excess_mm`, `s_mm`, etc. → drop `_mm` suffix
  - `rate_mm_per_s` → `rate_mm_per_sec` if kept, or compute in model units
  - `hg_time_s` → `hg_time`
  - `q_cfs` → keep (truly in CFS, well-scoped to culvert hydraulics)

- [ ] **Task 3.3:** Rename in `swe2d_solver.hpp` / `swe2d_solver.cpp`
  - Match renaming from `swe2d_gpu.cuh`

- [ ] **Task 3.4:** Update pybind11 bindings in `swe2d_bindings.cpp`
  - Python argument names must match renamed C++ variables
  - Python-side callers must pass renamed kwargs

- [ ] **Task 3.5:** Update Python C++ binding callers
  - `swe2d/runtime/backend.py` — kwargs to C++ calls
  - `swe2d/runtime/coupling.py` — structure flow kwargs
  - `swe2d/runtime/runtime_setup_configurator.py` — rain setup kwargs

- [ ] **Task 3.6:** Build and test

  ```bash
  cd $REPO_ROOT/build
  cmake .. -DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_BUILD_TYPE=Release
  cmake --build . -j$(nproc)
  ```

---

### Phase 4: Python API + CLI Renames (LOW risk — public surface)

- [ ] **Task 4.1:** Update `swe2d/units.py` parameter names
  - `flow_si_to_model(flow_cms)` → `flow_si_to_model(flow_si_m3_per_s)`
  - `rain_si_to_model(rain_rate_mps)` → `rain_si_to_model(rain_rate_si_m_per_s)`

- [ ] **Task 4.2:** Update `swe2d/workbench/services/unit_conversion_service.py`
  - `rain_mm_to_model_depth()` → `rainfall_mm_to_model_depth()` (more descriptive)
  - `rain_rate_si_to_model(rain_rate_mps)` → follow units.py rename

- [ ] **Task 4.3:** Update CLI argument names in `swe2d/cli/headless_runner.py`
  - Internal variable renames to match convention

---

### Phase 5: Fix Redundant Round-Trip Conversions & Display Bugs

The existing `units.py` API correctly handles **model-unit-aware** conversion (SI ↔ model units). However, two real issues were found during audit:

**Issue A — Redundant round-trip (Finding 1 from audit):** The constant rain rate path goes `mm/hr → m/s → model_units/s → mm/s`, where `_model_per_si_m` is multiplied in (`run_options_builder.py:160` via `rain_si_to_model`) then immediately divided back out (`runtime_setup_configurator.py:76` via division by `mm_to_model_depth = 1.0e-3 * _model_per_si_m`). The `/1000.0` (mm→m) is also undone by the `1.0e-3` (m→mm). The chain simplifies to `rain_rate_mmhr / 3600.0`. Fix: bypass the round-trip and compute directly in mm/s for the hyetograph.

**Issue B — Rain intensity display off by 1000× in both SI and USC (Finding 2, confirmed bug):** `high_perf_viewer.py:619-622` takes `overlay_cell_rain_rate_mps` (m/s) and displays it. The label says "mm/hr". The SI path does `* 3600` (m/hr) — should be `* 3,600,000` (mm/hr). The USC path does `* 3600 / 25.4` — should be `* 3600 * 1000 / 25.4` (in/hr).

**Scope:** 2 real bugs + add inline comments to trivial metric prefix conversions

- [ ] **Task 5.1:** Fix the rain rate round-trip in constant-rate path
  - Currently: `run_options_builder.py:160` converts mm/hr→m/s then applies `rain_si_to_model()` (×`_model_per_si_m`), then `runtime_setup_configurator.py:76` divides back by `mm_to_model_depth` (`1.0e-3 * _model_per_si_m`) to get mm/s for hyetograph construction
  - Fix: skip the m/s and model-unit intermediates. Compute mm/s directly:
    ```python
    # run_options_builder.py
    # rain_rate_mmhr is in mm/hr — convert to mm/s for hyetograph
    rate_mm_s = rain_rate_mmhr / 3600.0
    ```
    Pass `rate_mm_s` directly instead of going through model units.

- [ ] **Task 5.2:** Fix rain intensity display conversion in `swe2d/results/high_perf_viewer.py:619-622`
  - `overlay_cell_rain_rate_mps` is in m/s (SI)
  - SI display (mm/hr as labeled):
    ```python
    # m/s → mm/hr: × 3600 s/hr × 1000 mm/m
    vals = overlay_cell_rain_rate_mps * 3_600_000.0
    ```
  - USC display (in/hr):
    ```python
    # m/s → in/hr: × 3600 s/hr × 1000 mm/m / 25.4 mm/in
    vals = overlay_cell_rain_rate_mps * 3600.0 * 1000.0 / 25.4
    ```

- [ ] **Task 5.3:** Add inline comments to trivial metric prefix conversions in `rainfall_hydrology.py:67-88`
  - Keep the conversions as-is (they are correct), but add a comment per conversion:
    ```python
    if "in" in u:
        return value * 25.4       # in → mm
    if "cm" in u:
        return value * 10.0        # cm → mm
    if "m" in u and "mm" not in u:
        return value * 1000.0      # m → mm
    ```

- [ ] **Task 5.4:** Add inline comment to `rainfall_hydrology.py:594`
  - Currently: `rate_mps = (excess_mm / 1000.0) / dt_s`
  - Change to:
    ```python
    rate_mps = (excess_mm / 1000.0) / dt_s  # mm → m, then /dt gives m/s
    ```

- [ ] **Task 5.5:** Add inline comments to `run_options_builder.py:160` temporal conversions
  ```python
  rain_rate_mmhr / 1000.0 / 3600.0  # mm/hr → m/s (metric prefix /3600 for hr→s)
  ```

- [ ] **Task 5.6:** Add inline comments to `s→hr` display conversions
  ```python
  t_s / 3600.0  # seconds → hours for display
  ```

- [ ] **Task 5.7:** Verify all s→hr display labels are correct (e.g., do plots label the axis "hr"?)

- [ ] **Task 5.8:** Add regression test for rain intensity display conversions
  ```python
  # Test that 1 m/s → 3600 mm/hr (SI) and 141732 in/hr (USC)
  ```

---

### Phase 6: Verification & Final Testing (CRITICAL — gate for completion)

- [ ] **Task 6.1:** Full test suite

  ```bash
  find . -type d -name __pycache__ -exec rm -rf {} +
  mamba run -n qgis_stable python3 -m unittest -v
  ```

- [ ] **Task 6.2:** Build C++/CUDA

  ```bash
  cd $REPO_ROOT/build
  cmake --build . -j$(nproc)
  ```

---

## Summary Statistics

| Category | Count | Risk |
|----------|-------|------|
| `_m` suffix (model units, misnamed) | ~500+ | HIGH — lies about units in USC mode |
| `_s` / `_sec` naming inconsistency | ~200+ | MEDIUM — consistency |
| `_ms` / `_mps` naming inconsistency | ~150+ | MEDIUM — consistency |
| `_cms` / `_cfs` naming inconsistency | ~100+ | MEDIUM — consistency |
| `_mm` suffix (rainfall, correct) | ~100 | LOW — actually in mm |
| `_ft` suffix (culvert, correct) | ~50 | LOW — actually in feet |
| Data contract dict keys | ~12 keys × ~30 files | HIGH — migration requires backward compat |
| C++/CUDA variable names | ~80+ | MEDIUM — must match Python bindings |
| **TOTAL** | **~1000+** | |

## Verification Gate (after every Phase)

```bash
find . -type d -name __pycache__ -exec rm -rf {} +
mamba run -n qgis_stable python3 -m unittest -v \
    tests.test_workbench_gui \
    tests.test_workbench_imports \
    tests.test_workbench_persistence \
    tests.test_swe2d_results_queries \
    tests.test_gpkg_persistence \
    tests.test_gpkg_line_results_roundtrip \
    tests.test_line_results_plot \
    tests.test_workbench_gpkg_service
```

## Cross-Review Rule

Every code change produced by one subagent must be reviewed by a different subagent before the phase is marked complete.
