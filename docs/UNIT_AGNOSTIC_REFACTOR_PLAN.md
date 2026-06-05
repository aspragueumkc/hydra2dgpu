# Unit-Agnostic Refactor Plan

> **Status**: Phases 1–5 COMPLETE. Phases 6–8 REMAINING.
> See `docs/UNIT_AGNOSTIC_REFACTOR_COMPLETED.md` for done items.

## Goal

Make all code (except culvert empiricism) unit-agnostic. The CRS defines the length unit; all conversions derive from a single `LENGTH_SCALE` factor (SI meters → model units). No variable names shall carry unit labels (`_m`, `_ft`, `_cms`, `_cfs`). Constants use system labels (`SI_FACTOR`, `USC_FACTOR`). Where needed, dimensional comments use `// L³T⁻¹` notation.

---

## Phase 1: Centralize Conversion Constants

**Goal**: Single source of truth for all unit conversions.

### 1.1 Create `swe2d/units.py`

```python
"""
Unit system constants derived from the project CRS.
All conversions flow from a single LENGTH_SCALE (SI meters → model units).
"""

# ── Length ──
SI_M_PER_MODEL = 1.0          # m / model-length (set at runtime from CRS)
MODEL_PER_SI_M = 1.0          # model-length / m
SI_M2_PER_MODEL_AREA = 1.0    # m² / model-length²
SI_M3_PER_MODEL_VOLUME = 1.0  # m³ / model-length³

# ── Derived (do not edit directly) ──
USC_FT_PER_SI_M = 3.280839895013123
SI_M_PER_USC_FT = 0.3048
SI_M2_PER_USC_FT2 = 0.09290304
SI_M3_PER_USC_FT3 = 0.028316846592
USC_FT3_PER_SI_M3 = 35.31466672148859  # CFS_PER_CMS

# ── Gravity ──
SI_GRAVITY_MPS2 = 9.80665
USC_GRAVITY_FTPS2 = 32.17404855643045

# ── Manning ──
SI_MANNING_FACTOR = 1.0       # n in SI
USC_MANNING_FACTOR = 1.486    # 1.49 / n^(1/2) → 1.486

def configure(length_scale_si_to_model: float):
    """Call once at startup with CRS-derived length scale."""
    global SI_M_PER_MODEL, MODEL_PER_SI_M, SI_M2_PER_MODEL_AREA, SI_M3_PER_MODEL_VOLUME
    SI_M_PER_MODEL = length_scale_si_to_model
    MODEL_PER_SI_M = 1.0 / length_scale_si_to_model
    SI_M2_PER_MODEL_AREA = length_scale_si_to_model ** 2
    SI_M3_PER_MODEL_VOLUME = length_scale_si_to_model ** 3
```

### 1.2 Update all files to import from `swe2d.units`

| File | Current | Replace with |
|------|---------|--------------|
| `swe2d/runtime/coupling.py` | `self._length_scale`, `self._model_to_ft` | `units.SI_M_PER_MODEL`, `units.USC_FT_PER_SI_M / units.SI_M_PER_MODEL` |
| `swe2d/extensions/structures.py` | `_FT_PER_M = 3.28...`, `_CFS_PER_CMS = 35.31...` | `units.USC_FT_PER_SI_M`, `units.USC_FT3_PER_SI_M3` |
| `swe2d_workbench_qt.py` | `return 3.28...` in `_length_scale_si_to_model` | `units.configure(crs_scale)` then use `units.SI_M_PER_MODEL` |
| `cpp/src/swe2d_bindings.cpp` | `FT_PER_M`, `CFS_PER_CMS` | Pass as parameters or use shared header |
| `cpp/src/swe2d_gpu.cu` | `35.314...` inline | `USC_FT3_PER_SI_M3` constant |
| `tests/test_swe2d_culvert_validation.py` | `_FT_PER_M = 3.28...` | `from swe2d.units import USC_FT_PER_SI_M` |

---

## Phase 2: Remove Unit Labels from Variable Names

### 2.1 Rename in Python

| Current Name | New Name | File |
|-------------|----------|------|
| `cell_bed_m` (property) | `cell_bed` | `coupling.py` |
| `cell_area_m2` (param) | `cell_area` | `coupling.py` |
| `available_head_up_m` | `available_head_up` | `structures.py` |
| `tailwater_depth_m` | `tailwater_depth` | `structures.py` |
| `flow_cms` | `flow` | `structures.py` |
| `inlet_control_flow_cms` | `inlet_control_flow` | `structures.py` |
| `outlet_control_flow_cms` | `outlet_control_flow` | `structures.py` |
| `orifice_cap_cms` | `orifice_cap` | `structures.py` |
| `manning_cap_cms` | `manning_cap` | `structures.py` |
| `embankment_flow_cms` | `embankment_flow` | `structures.py` |
| `inlet_invert_elev_m` | `inlet_invert_elev` | `structures.py` |
| `outlet_invert_elev_m` | `outlet_invert_elev` | `structures.py` |
| `upstream_wse_m` | `upstream_wse` | `structures.py` |
| `downstream_wse_m` | `downstream_wse` | `structures.py` |
| `culvert_area_m2` (SoA field) | `culvert_area` | `coupling.py` + SoA dataclass |
| `structure_flow_cms` (GPU output) | `structure_flow` | `swe2d_gpu.cu` |
| `source_rate_mps` (GPU output) | `source_rate` | `swe2d_gpu.cu` |
| `gravity_mps2` (param) | `gravity` | `coupling.py`, `bindings.cpp` |
| `_length_scale` | `_si_m_per_model` | `coupling.py` |
| `_model_to_ft` | Remove — use `units.USC_FT_PER_SI_M / units.SI_M_PER_MODEL` | `coupling.py` |

### 2.2 Rename in C++

| Current Name | New Name | File |
|-------------|----------|------|
| `cell_area_m2` | `cell_area` | `swe2d_gpu.cu`, `swe2d_bindings.cpp` |
| `structure_flow_cms` | `structure_flow` | `swe2d_gpu.cu` |
| `structure_flow_cms_out` | `structure_flow_out` | `swe2d_gpu.cu` |
| `source_rate_mps` | `source_rate` | `swe2d_gpu.cu` |
| `gravity_mps2` | `gravity` | `swe2d_bindings.cpp`, `swe2d_gpu.cu` |
| `available_head_up_ft` | `available_head_up` | `swe2d_gpu.cu` (comment: `// L`) |
| `tailwater_depth_ft` | `tailwater_depth` | `swe2d_gpu.cu` (comment: `// L`) |
| `length_ft` | `length` | `swe2d_gpu.cu` (comment: `// L`) |
| `slope_ftft` | `slope` | `swe2d_gpu.cu` (comment: `// L/L`) |
| `q_hint_cfs` | `q_hint` | `swe2d_gpu.cu` (comment: `// L³T⁻¹`) |
| `q_inlet_cfs` | `q_inlet` | `swe2d_gpu.cu` |
| `q_inlet_cms` | Remove intermediate — keep everything in one unit system | `swe2d_gpu.cu` |
| `FT_PER_M` | `USC_FT_PER_SI_M` | `swe2d_bindings.cpp` |
| `CFS_PER_CMS` | `USC_FT3_PER_SI_M3` | `swe2d_bindings.cpp` |
| `BW2D_GRAVITY_FTPS2` | `USC_GRAVITY` | `swe2d_bindings.cpp` |

### 2.3 Add dimensional comments for key variables

```c++
double q;              // L³T⁻¹ (flow)
double h;              // L   (head)
double a;              // L²  (area)
double slope;          // L/L (dimensionless slope)
double roughness_n;    // L⁻⅓T (Manning's n)
double gravity;        // LT⁻² (gravitational acceleration)
double source_rate;    // LT⁻¹ (depth rate)
```

---

## Phase 3: Fix Critical Unit Bugs

### 3.1 `culvert_area_m2` Not Scaled

**File**: `swe2d/runtime/coupling.py` line 349

**Current**:
```python
culvert_area_m2[i] = float(st.metadata.get("culvert_area_m2", ...) or 0.0)
```

**Fix**:
```python
# Area scales as L² — convert from model units to computation units
_area_raw = float(st.metadata.get("culvert_area_m2") or st.metadata.get("area_m2") or 0.0)
culvert_area[i] = _area_raw * (m2ft * m2ft)  # L²
```

### 3.2 Mixed Units in GPU Orifice Flow

**File**: `cpp/src/swe2d_gpu.cu` line ~3159

**Current**: `area` comes from `culvert_area_m2[i]` (unscaled, effectively m²), but head and gravity are in feet.

**Fix**: Ensure `culvert_area` in SoA is already scaled to the computation unit system (feet² for USC, m² for SI) before it reaches the kernel. The `pack_structures_soa` fix in 3.1 handles this.

### 3.3 Same in CPU Path

**File**: `cpp/src/swe2d_bindings.cpp` lines ~873-880

**Fix**: Same as 3.2 — the SoA packing fix propagates correctly.

### 3.4 GPU Source Rate Cell Area Scaling

**File**: `swe2d/runtime/coupling.py` lines ~673, ~1275

**Current**: `cell_area` uploaded to GPU in model units. GPU divides flow by `cell_area` to get source rate. Flows are in SI (CMS), so `cell_area` must be in SI (m²) for correct `m/s` output.

**Fix** (already applied):
```python
cell_area_si = np.asarray(self.cell_area) / (units.SI_M_PER_MODEL ** 2)
```

### 3.5 Slope Unit Consistency

**File**: `swe2d/runtime/coupling.py` line ~352

**Current**:
```python
culvert_slope[i] = float(st.metadata.get("culvert_slope", 0.0) or 0.0)
```

**Analysis**: Slope is dimensionless (L/L). Whether the model is SI or USC, the slope value is the same. No conversion needed. ✓ (No bug)

---

## Phase 4: Documentation & Comments

### 4.1 Add module docstring to `swe2d/units.py`

### 4.2 Add dimensional comments to all C++ kernel parameters

### 4.3 Update `pack_structures_soa` docstring

```python
def pack_structures_soa(cfg, n_cells, compute_length_scale):
    """
    Pack hydraulic structure metadata into SoA arrays for GPU/CPU computation.
    
    Input metadata is in MODEL UNITS (SI or USC depending on CRS).
    Output SoA arrays are in COMPUTATION UNITS:
      - SI model: computation = SI (meters, m², m³/s)
      - USC model: computation = USC (feet, ft², ft³/s)
    
    The computation unit system is determined by compute_length_scale
    (USC_FT_PER_SI_M / SI_M_PER_MODEL).
    
    All length values [L] are multiplied by compute_length_scale.
    All area values [L²] are multiplied by compute_length_scale².
    """
```

---

## Phase 5: Test Updates

### 5.1 Update test imports to use `swe2d.units`

### 5.2 Add unit-agnostic test for `pack_structures_soa`

### 5.3 Add dimensional consistency assertions

```python
def test_pack_structures_soa_dimensional_consistency():
    """Verify area scales as L², lengths as L."""
    cfg = _make_test_cfg()
    soa_si = pack_structures_soa(cfg, 100, compute_length_scale=USC_FT_PER_SI_M)
    soa_usc = pack_structures_soa(cfg, 100, compute_length_scale=1.0)
    
    # Lengths: SI should be 3.28× USC
    assert soa_si.diameter[0] == pytest.approx(soa_usc.diameter[0] * USC_FT_PER_SI_M)
    # Areas: SI should be 10.76× USC
    assert soa_si.culvert_area[0] == pytest.approx(soa_usc.culvert_area[0] * USC_FT_PER_SI_M ** 2)
```

---

## Implementation Order

| Step | Files | Risk | Effort |
|------|-------|------|--------|
| 1. Create `swe2d/units.py` | 1 new file | Low | Small |
| 2. Fix `culvert_area_m2` scaling | `coupling.py` | **Critical** | Small |
| 3. Centralize C++ constants | `swe2d_gpu.cuh` (new header) | Low | Small |
| 4. Rename Python variables | `structures.py`, `coupling.py` | Medium | Medium |
| 5. Rename C++ variables | `swe2d_gpu.cu`, `swe2d_bindings.cpp` | Medium | Large |
| 6. Update callers | `swe2d_workbench_qt.py`, `model_and_run_methods.py` | Medium | Medium |
| 7. Update tests | `test_swe2d_culvert_validation.py`, etc. | Medium | Medium |
| 8. Add dimensional comments | C++ files | Low | Small |

---

## Files Touched

- **NEW**: `swe2d/units.py`
- **NEW**: `cpp/src/swe2d_units.cuh`
- `swe2d/runtime/coupling.py`
- `swe2d/extensions/structures.py`
- `swe2d_workbench_qt.py`
- `swe2d/workbench/extracted/model_and_run_methods.py`
- `cpp/src/swe2d_gpu.cu`
- `cpp/src/swe2d_bindings.cpp`
- `cpp/src/swe2d_gpu.cuh`
- `tests/test_swe2d_culvert_validation.py`
- `tests/test_swe2d_drainage_structures.py`

---

## Phase 6: Remove Backward-Compatibility Aliases ⏳

The Phase 2 rename added `_cms`/`_m`/`_m2`/`_mps2` backward-compat aliases everywhere.
These aliases are **now the primary source of unit confusion** — they suggest SI units
(`_m`, `_cms`) but the values they alias are often in model units (feet for USC CRS).

### 6.1 Python dict-key aliases in `structures.py`

**Current**: `structure_details()` adds both `"flow"` AND `"flow_cms"` to every dict.
The `_cms` key implies SI m³/s, but the value is actually in model³/s (ft³/s for USC).

**Remove**: The `_ALIASES` dict and the loop that adds aliased keys.
All callers should use the canonical names (`flow`, `inlet_control_flow`, etc.).

**Files to update**:
| File | Change |
|------|--------|
| `swe2d/extensions/structures.py` | Remove `_ALIASES` dict and alias loop |
| `swe2d_workbench_qt.py` | Update all `.get("flow_cms")` → `.get("flow")` etc. |
| `swe2d/results/panel.py` | Update `flow_cms` dict keys |
| `tests/test_swe2d_culvert_validation.py` | Update `.get("flow_cms")` → `.get("flow")` etc. |
| `tests/test_swe2d_drainage_structures.py` | Update dict key accesses |

### 6.2 Python property aliases in `extension_models.py`

**Remove**: All backward-compat properties on `PipeNetworkState` and `CouplingDiagnostics`:

```python
# REMOVE these properties from PipeNetworkState:
@property node_depth_m → just use node_depth
@property node_depth_m.setter
@property link_flow_cms → just use link_flow
@property link_flow_cms.setter

# REMOVE these properties from CouplingDiagnostics:
@property max_node_depth_m → use max_node_depth
@property max_link_flow_cms → use max_link_flow
@property net_node_inflow_cms → use net_node_inflow
@property total_capture_cms → use total_capture
@property total_surcharge_cms → use total_surcharge
# ... and their setters
```

**Files to update**:
| File | Change |
|------|--------|
| `swe2d/extensions/extension_models.py` | Remove property aliases |
| `swe2d_workbench_qt.py` | Change `.node_depth_m` → `.node_depth` etc. |
| `swe2d/runtime/coupling.py` | Change `.link_flow_cms` → `.link_flow` etc. |
| `tests/*.py` | Update any property accesses |

### 6.3 Python property aliases in `coupling.py`

**Remove**: Backward-compat kwargs and properties on `SWE2DCouplingController`:

```python
# REMOVE: cell_area_m2 property (alias for cell_area)
# REMOVE: cell_bed_m property (alias for cell_bed)
# REMOVE: cell_area_m2= keyword arg in __init__
# REMOVE: cell_bed_m= keyword arg in __init__
```

**Files to update**:
| File | Change |
|------|--------|
| `swe2d/runtime/coupling.py` | Remove `cell_area_m2` / `cell_bed_m` kwargs and properties |
| `swe2d_workbench_qt.py` | Change `cell_area_m2=` → `cell_area=` etc. |
| `swe2d/workbench/extracted/model_and_run_methods.py` | Change `cell_area_m2=` → `cell_area=` etc. |
| `tests/test_swe2d_drainage_structures.py` | Change `cell_area_m2=` → `cell_area=` etc. |

### 6.4 C++ `py::arg()` backward-compat strings ✅ **COMPLETE**

**Status**: Done. All `py::arg()` strings in `swe2d_bindings.cpp` and parameter
names in `swe2d_gpu_redistribute.cu` have been renamed to canonical forms.
Python callers verified to use positional args for these bindings (no breakage).
C++ compilation required after changes.

**Completed renames**:
| Old py::arg | New py::arg |
|---|---|
| `cell_area_m2` | `cell_area` |
| `culvert_area_m2` | `culvert_area` |
| `inlet_flow_cms` | `inlet_flow` |
| `structure_flow_cms` | `structure_flow` |
| `bridge_flow_cms` | `bridge_flow` |
| `bridge_opening_width_m` | `bridge_opening_width` |
| `head_deadband_m` | `head_deadband` |
| `gravity_mps2` | `gravity` |
| `source_mps` | `external_source` |

**Files modified**:
| File | Changes |
|------|---------|
| `cpp/src/swe2d_bindings.cpp` | All 23+ `py::arg()` strings, lambda params, error messages |
| `cpp/src/swe2d_gpu_redistribute.cu` | `struct_flow_cms→struct_flow`, `structure_flow_cms→structure_flow`, `cell_area_m2→cell_area`, `source_rate_mps→source_rate`, `source_rate_mps_inout→source_rate_inout` |

### 6.5 Rename model-unit parameters with misleading `_cms` / `_m` suffixes ✅ **COMPLETE**

These are the last holdouts where the *parameter name* implies a specific unit
but the value is in model units:

| Current Name | Location | New Name | Notes |
|-------------|----------|----------|-------|
| `cell_flow_cms` | `extension_models.py:convert_cell_flows_to_depth_rates()` | `cell_flow` | |
| `cell_area_m2` | `extension_models.py:convert_cell_flows_to_depth_rates()` | `cell_area` | |
| `max_flow_cms` | `extension_models.py:compute_weir_flow()` | `max_flow` (keep both args with union logic) | Already takes `max_flow` also |
| `max_flow_cms` | `extension_models.py:compute_orifice_flow()` | same pattern | |
| `_culvert_outlet_control_flow_cms` | `structures.py` | `_culvert_outlet_control_flow` | |
| `node_depth_m` | `extension_models.py:PipeNetworkState` | `node_depth` | Already the canonical field |
| `link_flow_cms` | `extension_models.py:PipeNetworkState` | `link_flow` | Already the canonical field |
| `head_up_m` / `head_down_m` | `extension_models.py:compute_orifice_flow()` | `head_up` / `head_down` | |
| `area_m2` | `extension_models.py:compute_orifice_flow()` | `area` | |
| `depth_m` / `diameter_m` | `extension_models.py:circular_section_from_depth()` | `depth` / `diameter` | |
| `_model_to_ft()` | `structures.py` helper | Already internal, rename to `_model_length_to_ft()` | |
| `inv_m2ft` | `structures.py:_structure_detail()` | `inv_model_to_ft` | |

---

## Phase 7: Fix Native Flow Unit Inconsistency ⏳

### 7.1 The Core Problem

The C++ kernel `swe2d_compute_structure_flows_kernel` returns **mixed units**:
- **Culvert (type 2)**: Returns CMS (m³/s) after converting from internal CFS
- **Non-culvert (types 1/3/4/5)**: Returns CFS (ft³/s)

This makes `_last_native_structure_flows` a mixed-unit array. The Python path
(`structure_details()`) returns mixed units too:
- **Culvert**: Returns CMS
- **Non-culvert**: Returns model³/s

### 7.2 The Fix

**Option A (Recommended)**: Make the kernel return all flows in model³/s.

In `swe2d_compute_structure_flows_kernel`, after computing the final flow
for each structure type, convert to model³/s by dividing by
`USC_FT3_PER_SI_M3 / model_per_ft_cubed`. Since the kernel receives
`m2ft` via Python, add a conversion factor parameter.

For the kernel return path:
```c++
// After weir/orifice/bridge/pump flow computation (currently in CFS):
structure_flow[i] = sign * q / ft3_per_model3;  // CFS → model³/s

// For culvert (currently returns CMS):
structure_flow[i] = sign * q * m3_per_model3;  // CMS → model³/s
```

Where `ft3_per_model3 = USC_FT3_PER_SI_M3 / si_m_per_model³` and
`m3_per_model3 = si_m_per_model³`.

**Option B (Minimal)**: Accept mixed units and document the contract clearly.
The viewer-side fix (Phase 6 / already done) handles this at the display layer,
and `_structure_source_rate_from_flows()` needs type-aware conversion.

### 7.3 Files

| File | Change |
|------|--------|
| `cpp/src/swe2d_gpu.cu` | Add model conversion factor param, apply to all return paths |
| `cpp/src/swe2d_bindings.cpp` | Pass `model_per_ft3` / `si_m3_per_model_volume` as new param |
| `swe2d/runtime/coupling.py` | Pass conversion args to native kernel; update `_native_structure_flows()` |
| `swe2d/runtime/coupling.py` | Fix `_structure_source_rate_from_flows()` for mixed units |
| `swe2d_workbench_qt.py` | Simplify `_sample_coupling_object_metrics()` (remove `_cfs_to_model` branch) |

---

## Phase 8: Remove `convert_cell_flows_to_depth_rates` `_cms` / `_m2` Suffixes ⏳

### 8.1 `extension_models.py:function convert_cell_flows_to_depth_rates`

Rename `cell_flow_cms` → `cell_flow` and `cell_area_m2` → `cell_area`.
Update docstring to say "model-length³/T flow" and "model-length² area" instead of
"m³/s" and "m²".

### 8.2 `DrainageCouplingState` aliases

Remove `node_depth_m` / `link_flow_cms` / `max_node_depth_m` / `max_link_flow_cms`
/ `net_node_inflow_cms` / `total_capture_cms` / `total_surcharge_cms` property
aliases. These all suggest SI units but their values are in model units.

### 8.3 Results schema

The GeoPackage schema uses column names like `station_m`, `elev_m`, `flow_cms`
in line/mesh results. These are in model units. Consider adding a `unit_system`
metadata field or a schema version column, and rename columns in a future
schema migration.

---

## Priority Order for Phases 6–8

| Priority | Phase | Scope | Risk | Impact |
|----------|-------|-------|------|--------|
| **6.1** | Remove dict-key aliases | `structures.py` + callers | Medium | Eliminates primary confusion source |
| **6.2** | Remove `PipeNetworkState` prop aliases | `extension_models.py` + callers | Low | Clean API |
| **6.3** | Remove `CouplingController` prop aliases | `coupling.py` + callers | Medium | Clean API |
| **6.5** | Rename `_cms`/`_m` params | Wide | Low | Consistency |
| **7.2** | Fix native flow units | C++ kernel + Python | **High** | Correctness fix |
| **6.4** | Update C++ `py::arg` strings | `swe2d_bindings.cpp` | Low | Must follow 6.3 |
| **8.1** | Rename `convert_cell_flows_...` params | `extension_models.py` | Low | Consistency |
