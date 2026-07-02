# Results Persistence & Rendering Bug Fix Specification

**Created:** 2026-07-01  
**Status:** Completed  
**Priority:** Critical/High

## 1. Problem Statement

The results persistence and rendering path for plotted views had several bugs that caused:
- Missing data when loading timeseries (wet_frac, fr fields not loaded)
- Silent failures making debugging difficult
- Potential crashes on NULL database values

## 2. Issues Fixed

### Issue 1: SQL Placeholder Mismatch — NOT A BUG
**Location:** `swe2d/services/gpkg_persistence_service.py:792`

**Finding:** The INSERT statement correctly has 12 placeholders matching 12 values. No fix needed.

---

### Issue 2: Missing wet_frac and fr in Load — FIXED ✅
**Location:** `swe2d/services/gpkg_persistence_service.py:950-965`

The `load_baked_line_timeseries` function only loaded 6 fields but should load 8.

**Fix Applied:** Added `wet_frac_blob, fr_blob` to SELECT query and added `wet_frac` and `fr` to return dict with NULL safety.

---

### Issue 3: Missing NULL Check in load_baked_timesteps — FIXED ✅
**Location:** `swe2d/services/gpkg_persistence_service.py:1082-1085`

If row exists but times_blob is NULL, `np.frombuffer(None)` would fail.

**Fix Applied:** Added NULL check before calling frombuffer.

---

### Issue 4: Silent Exception Handlers — FIXED ✅
**Locations:** 
- `swe2d/workbench/views/studio_viewer_plot.py:178`
- `swe2d/results/data.py:507`
- `swe2d/results/db_utils.py:64,81`

These silently swallowed exceptions, making debugging impossible.

**Fix Applied:** Added proper logging with error details.

---

### Issue 5: Live Data Path Missing Fields — NOT A BUG
**Location:** `swe2d/services/gpkg_persistence_service.py:928-929`

**Finding:** The live data path already correctly includes wet_frac and fr in the loop. No fix needed.

---

## 3. Summary of Changes

| File | Changes |
|------|---------|
| `swe2d/services/gpkg_persistence_service.py` | Added wet_frac/fr to load, added NULL check |
| `swe2d/workbench/views/studio_viewer_plot.py` | Added error logging |
| `swe2d/results/data.py` | Added error logging |
| `swe2d/results/db_utils.py` | Added error logging |

---

## 4. Testing

After implementing fixes:
1. Run a simulation and verify line results are persisted to GPKG
2. Load the results and verify wet_frac and fr data is present
3. Check logs for any previously silent errors
4. Verify no crashes on edge cases (NULL values, empty tables)
