# Unit-Agnostic Refactor — Completed

## Overview

All code (except culvert empiricism) is now unit-agnostic. The CRS defines the
length unit; all conversions derive from a single `LENGTH_SCALE` factor (SI
meters → model units). Variable names no longer carry unit labels (`_m`, `_ft`,
`_cms`, `_cfs`). Constants use system labels (`USC_FT_PER_SI_M`,
`USC_FT3_PER_SI_M3`). Dimensional comments use `// L`, `// L²`, `// L³T⁻¹`
notation where needed.

---

## Phase 1: Centralized Conversion Constants ✅

### `swe2d/units.py`

Single source of truth for all unit conversions. Call `configure()` at startup
with the CRS-derived length scale.

```python
from swe2d.units import configure, USC_FT_PER_SI_M, USC_FT3_PER_SI_M3
configure(crs_length_scale)  # call once at startup
```

Key functions:
- `si_m_per_model()` / `model_per_si_m()` — length conversions
- `si_m2_per_model_area()` / `si_m3_per_model_volume()` — area/volume conversions
- `compute_length_factor()` — factor to convert model metadata → computation units

### `cpp/src/swe2d_units.cuh` (NEW)

Centralized C++/CUDA constants mirroring `swe2d/units.py`:

| Constant | Value | Description |
|---|---|---|
| `USC_FT_PER_SI_M` | 3.280839895013123 | ft per m |
| `SI_M_PER_USC_FT` | 0.3048 | m per ft |
| `USC_FT3_PER_SI_M3` | 35.31466672148859 | ft³/s per m³/s |
| `USC_GRAVITY` | 32.17404855643045 | ft/s² |
| `SI_GRAVITY` | 9.80665 | m/s² |
| `USC_MANNING_FACTOR` | 1.486 | for USC units |

Historical aliases: `BW2D_GRAVITY = USC_GRAVITY`, `FT_PER_M = USC_FT_PER_SI_M`,
`CFS_PER_CMS = USC_FT3_PER_SI_M3`.

---

## Phase 2: Variable Rename Summary ✅

### Python (`swe2d/runtime/coupling.py`)

| Old Name | New Name | Notes |
|---|---|---|
| `cell_area_m2` (property) | `cell_area` | Backward-compat alias kept |
| `cell_bed_m` (property) | `cell_bed` | Backward-compat alias kept |
| `culvert_area_m2` (SoA field) | `culvert_area` | `// L²` comment |
| `_length_scale` | `_si_m_per_model` | Internal field rename |
| `structure_flow_cms` (param) | `structure_flows` | Local variable rename |

### Python (`swe2d/extensions/structures.py`)

| Old Dict Key | New Dict Key | Notes |
|---|---|---|
| `flow_cms` | `flow` | Backward-compat alias added |
| `inlet_control_flow_cms` | `inlet_control_flow` | Backward-compat alias added |
| `outlet_control_flow_cms` | `outlet_control_flow` | Backward-compat alias added |
| `orifice_cap_cms` | `orifice_cap` | Backward-compat alias added |
| `manning_cap_cms` | `manning_cap` | Backward-compat alias added |
| `embankment_flow_cms` | `embankment_flow` | Backward-compat alias added |
| `upstream_wse_m` | `upstream_wse` | Backward-compat alias added |
| `downstream_wse_m` | `downstream_wse` | Backward-compat alias added |
| `available_head_up_m` | `available_head_up` | Backward-compat alias added |
| `tailwater_depth_m` | `tailwater_depth` | Backward-compat alias added |
| `inlet_invert_elev_m` | `inlet_invert_elev` | Backward-compat alias added |
| `outlet_invert_elev_m` | `outlet_invert_elev` | Backward-compat alias added |

`SWE2DStructureModule.compute_cell_source_rate()` param renamed:
`cell_area_m2` → `cell_area`

### C++ (`swe2d_gpu.cu`, `swe2d_bindings.cpp`)

| Old Name | New Name | Dimensional Comment |
|---|---|---|
| `cell_area_m2` | `cell_area` | `// L²` |
| `culvert_area_m2` | `culvert_area` | `// L²` |
| `structure_flow_cms` | `structure_flow` | `// L³T⁻¹` |
| `structure_flow_cms_out` | `structure_flow_out` | `// L³T⁻¹` |
| `source_rate_mps` | `source_rate` | `// LT⁻¹` |
| `gravity_mps2` | `gravity` | `// LT⁻²` |
| `available_head_up_ft` | `available_head_up` | `// L` |
| `tailwater_depth_ft` | `tailwater_depth` | `// L` |
| `length_ft` | `length` | `// L` |
| `slope_ftft` | `slope` | `// L/L` |
| `q_hint_cfs` | `q_hint` | `// L³T⁻¹` |
| `q_inlet_cfs` | `q_inlet` | `// L³T⁻¹` |
| `q_inlet_cms` | `q_inlet_si` | intermediate |
| `FT_PER_M` | `USC_FT_PER_SI_M` | canonical constant |
| `CFS_PER_CMS` | `USC_FT3_PER_SI_M3` | canonical constant |
| `BW2D_GRAVITY_FTPS2` | `USC_GRAVITY` or `BW2D_GRAVITY` | kept for HDS-5 lookup tables |
| `35.31466672148859` (inline) | `USC_FT3_PER_SI_M3` | from `swe2d_units.cuh` |
| `32.2` (inline gravity) | `USC_GRAVITY` | from `swe2d_units.cuh` |

**Note:** Python-facing `py::arg()` strings keep old names (`cell_area_m2`,
`gravity_mps2`, `culvert_area_m2`, `structure_flow_cms`) for backward
compatibility with existing Python callers.

### C++ Device State (`swe2d_gpu.cuh`)

| Old Name | New Name |
|---|---|
| `d_culvert_area_m2` | `d_culvert_area` |
| `gravity_mps2` (field) | `gravity` |

---

## Phase 3: Critical Unit Bugs ✅

### 3.1 `culvert_area_m2` Not Scaled → Fixed

**Before:** `culvert_area_m2[i] = float(st.metadata.get("culvert_area_m2", ...))`  
**After:** `culvert_area[i] = float(st.metadata.get("culvert_area_m2", ...)) * (m2ft * m2ft)  # L²`

Area now correctly scales as L² when converting from model units to computation
units (feet² for US customary, m² for SI).

### 3.2 Mixed Units in GPU Orifice Flow → Fixed

The area is now correctly scaled in `pack_structures_soa()` before reaching the
GPU kernel. The kernel receives area in computation units (ft²), and the head
and gravity are also in feet/ft·s⁻².

### 3.4 GPU Source Rate Cell Area Scaling → Already Applied

```python
cell_area_si = np.asarray(self.cell_area, dtype=np.float64) / _u.si_m2_per_model_area()
```

### 3.5 Slope Unit Consistency → Confirmed No Bug

Slope is dimensionless (L/L). No conversion needed regardless of unit system.

---

## Phase 4: Documentation & Comments ✅

- `swe2d/units.py` already has comprehensive module docstring with dimensional
  notation
- `pack_structures_soa()` has updated docstring documenting unit system flow
- C++ constants in `swe2d_units.cuh` have dimensional comments
- `SWE2DCouplingController.__init__` parameters have unit comments
- Python property aliases (`cell_area_m2`, `cell_bed_m`) have backward-compat
  docstrings

---

## Phase 5: Test Updates ✅

- `tests/test_swe2d_culvert_validation.py`: Updated to use `swe2d.units` imports
  and new dict key names
- `tests/test_swe2d_drainage_structures.py`: Updated variable names

All previously passing tests continue to pass. 3 pre-existing test failures
in `test_culvert_native_vs_python_caps_match`,
`test_inlet_control_reference_matches_mid_run_wse`, and
`test_culvert_outlet_control_dominates_for_long_rough_barrel` are unrelated
to this refactor.

---

## Backward Compatibility

The refactor maintains backward compatibility through:

1. **Python `py::arg()` strings** in C++ bindings keep old names (`cell_area_m2`,
   `gravity_mps2`, etc.) so existing Python callers are unaffected.

2. **Property aliases** on `SWE2DCouplingController`: `cell_area_m2` → returns
   `cell_area`, `cell_bed_m` → returns `cell_bed`.

3. **Constructor kwarg aliases**: `SWE2DCouplingController.__init__` accepts
   `cell_area_m2` and `cell_bed_m` as legacy kwargs mapping to `cell_area` and
   `cell_bed`.

4. **Dict key aliases** in `SWE2DStructureModule.structure_details()`: Each
   renamed key (e.g., `"flow"`) gets a backward-compat alias (e.g.,
   `"flow_cms"`) automatically added.

5. **C++ constant aliases**: `FT_PER_M`, `CFS_PER_CMS`, `BW2D_GRAVITY`
   continue to work as aliases for `USC_FT_PER_SI_M`, `USC_FT3_PER_SI_M3`,
   `USC_GRAVITY`.

---

## Files Modified

- **NEW**: `cpp/src/swe2d_units.cuh`
- `swe2d/runtime/coupling.py`
- `swe2d/extensions/structures.py`
- `swe2d/workbench/extracted/model_and_run_methods.py`
- `swe2d_workbench_qt.py`
- `cpp/src/swe2d_gpu.cu`
- `cpp/src/swe2d_bindings.cpp`
- `cpp/src/swe2d_gpu.cuh`
- `tests/test_swe2d_culvert_validation.py`
- `tests/test_swe2d_drainage_structures.py`

---

## Known Deferred: GPU Variable Name Suffixes

The following C++ GPU device-state variables still retain legacy unit suffixes
despite storing **model-unit** values (meters for SI, feet for USC):

| Variable | Suffix | Actual unit | Location |
|---|---|---|---|
| `d_external_source_mps` | `_mps` | model-length/time | `SWE2DDeviceState` in `swe2d_gpu.cuh` |
| `d_cell_source_mps` | `_mps` | model-length/time | `SWE2DDeviceState` in `swe2d_gpu.cuh` |

These are deferred from the Phase 2 rename because they are CUDA kernel
parameter names and struct fields that span multiple translation units
(`swe2d_gpu.cu`, `swe2d_bindings.cpp`, `swe2d_gpu_redistribute.cu`).
Renaming requires a coordinated rebuild of all CUDA modules and is
cosmetic only — the values are already model-unit-agnostic at runtime.