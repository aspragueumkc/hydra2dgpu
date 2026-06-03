# Unit-Agnostic Refactor Plan

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
