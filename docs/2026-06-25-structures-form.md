# Structures Attribute Form Cleanup Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clean up the QGIS structures attribute form — conditional field visibility by structure type, visible defaults, value maps for culvert code, and NULL invert defaults that fall back to bed elevation at solve time.

**Architecture:** QGIS attribute forms support Python init code, groups with visibility expressions, and value maps — no .ui file needed. A Python function file (swe2d_structures_form.py) is installed on the layer's edit form init. The solver's pack_structures_soa checks for NULL inverts and defaults to bed elevation with a log message.

**Tech Stack:** QGIS Python API (QgsEditFormConfig, QgsAttributeEditorGroup, QgsDefaultValue), numpy

**File dependency map:**
- Task 1 (create): `swe2d/workbench/forms/swe2d_structures_form.py`
- Task 2 (modify): `swe2d/workbench/views/topology_tab_view.py`
- Task 3 (modify): `swe2d/workbench/services/structure_config_service.py`
- Task 4 (modify): `swe2d/runtime/coupling.py`

Task 1+2 are dependent (form file must exist to be registered). Task 3+4 are independent of each other.

---

### Task 1: Create Python form init for structures layer

Create `swe2d/workbench/forms/swe2d_structures_form.py` — a QGIS attribute form init Python file. QGIS calls `form_open(dialog, layer, feature)` when the feature form opens.

The form groups fields by structure type using QgsAttributeEditorGroup with visibility expressions. When structure_type changes (QGIS layer-level on-the-fly via the attribute table dropdown), the visible field groups swap automatically.

**Create:** `swe2d/workbench/forms/swe2d_structures_form.py`

```python
"""QGIS attribute form init script for structures layer.
Installed on the layer via setInitCodeFunction / setInitFilePath.
Groups fields by structure type using visibility expressions so only
relevant fields appear for the selected structure type.
"""
from qgis.PyQt.QtWidgets import QComboBox
from qgis.gui import QgsEditFormConfig, QgsAttributeEditorGroup, QgsAttributeEditorField

CULVERT = 2
BRIDGE = 4
WEIR = 1
GATE = 3
PUMP = 5

_CULVERT_CODE_MAP = {
    0: "— Select culvert code —",
    1: "Circular concrete, square edge w/ headwall",
    2: "Circular concrete, groove end w/ headwall",
    3: "Circular concrete, groove end projecting",
    4: "Circular concrete, mitred to slope",
    5: "Circular concrete, beveled ring",
    6: "Circular concrete, beveled ring (smoother)",
    7: "Circular CMP, projecting",
    8: "Circular CMP, projecting (different edge)",
    9: "Circular CMP, mitered to slope",
    10: "Circular CMP, mitered to slope (alt)",
    11: "Circular CMP, beveled end (thin wall)",
    12: "Circular CMP, groove end in headwall",
    13: "Circular CMP, groove end in headwall (alt)",
    14: "Circular CMP, headwall (square edge)",
    15: "Circular CMP, headwall (groove end)",
    16: "Circular CMP, headwall (thin wall projecting)",
    17: "Rectangular box, 30-75deg wingwall flares",
    18: "Rectangular box, 90deg headwall w/ chamfers",
    19: "Rectangular box, 0deg wingwall flares",
    20: "Rectangular box, 45deg wingwall flares",
    21: "Rectangular box, 18-33deg wingwall flares",
    22: "Rectangular box, 0deg wingwall flares (thick)",
    23: "Rectangular box, 30deg wingwall flares (thick)",
    24: "Rectangular box, 45deg wingwall flares (thick)",
    25: "Rectangular box, 0deg wingwall flares (thick alt)",
    26: "Rectangular box, beveled edge (1:1)",
    27: "Circular concrete, square edge w/ headwall (form-1 alt)",
    28: "Circular concrete, groove end w/ headwall (form-1 alt)",
    29: "Circular concrete, groove end projecting (form-1 alt)",
    30: "Circular CMP, projecting (form-1 alt)",
    31: "Circular CMP, mitered to slope (form-1 alt)",
    32: "Circular CMP, beveled end thin wall (form-1 alt)",
    33: "Circular CMP, groove end in headwall (form-1 alt)",
    34: "Circular CMP, headwall square edge (form-1 alt)",
    35: "Circular CMP, headwall groove end (form-1 alt)",
    36: "Circular CMP, beveled ring (form-1 alt)",
    37: "Circular CMP, beveled ring thick (form-1 alt)",
    38: "Circular concrete, beveled ring (form-1 alt)",
    39: "Circular pipe, beveled ring (thin wall)",
    40: "Circular pipe, beveled ring (thick wall)",
    41: "Circular pipe, 45deg beveled ring",
    42: "Circular pipe, 33.7deg beveled ring",
    43: "Circular pipe, 45deg bevel (offset)",
    44: "Circular pipe, 33.7deg bevel (offset)",
    45: "Circular CMP, prefab end section (safety)",
    46: "Circular CMP, prefab end section (alt)",
    47: "Arch CMP, 2-3-1 fill (soffit thickness 0.0625)",
    48: "Arch CMP, 2-3-1 fill (soffit varying)",
    49: "Arch CMP, 2-3-1 fill projecting (soffit varying)",
    50: "Arch CMP, 2-2-1 fill (soffit thickness 0.0625)",
    51: "Pipe arch CMP, 0.75x0.75 fill (soffit thickness 0.0625)",
    52: "Pipe arch CMP, 0.75x0.75 fill projecting",
    53: "Pipe arch CMP, 0.75x0.75 fill (soffit varying)",
    54: "Horizontal ellipse, concrete (form-2)",
    55: "Horizontal ellipse, corrugated metal (form-2)",
    56: "Arch CMP, 2-3-1 fill premium (form-2)",
    57: "Horizontal ellipse, special shape (form-2)",
}

# Fields that are common to all structure types
_COMMON_FIELDS = {"structure_id", "structure_type", "crest_elev", "enabled"}

# Field groups per type — only these fields are visible for each type
_TYPE_FIELDS = {
    CULVERT: {
        "culvert_shape", "culvert_code", "culvert_rise", "culvert_span",
        "culvert_area_m2", "culvert_barrels", "culvert_slope",
        "diameter", "length", "roughness_n",
        "inlet_invert_elev", "outlet_invert_elev",
        "entrance_loss_k", "exit_loss_k",
        "embankment_enabled", "embankment_crest_elev",
        "embankment_overflow_width", "embankment_weir_coeff",
    },
    BRIDGE: {
        "width", "length", "deck_soffit_elev", "deck_top_elev",
        "model_top_elev", "under_layers", "over_layers",
        "inlet_loss_k", "outlet_loss_k",
        "pier_count", "pier_width", "face_flux_depth_safety",
    },
    WEIR: {
        "width", "embankment_enabled", "embankment_crest_elev",
        "embankment_overflow_width", "embankment_weir_coeff",
    },
    GATE: {
        "width", "height", "opening",
    },
    PUMP: {
        "q_pump", "max_flow", "min_head_diff", "max_head_diff",
    },
}


def form_open(dialog, layer, feature):
    """Called by QGIS when opening a feature's attribute form.

    Groups fields by structure type and sets visibility expressions
    so that only the relevant fields show for the current type.
    """
    cfg = layer.editFormConfig()
    cfg.setLayout(cfg.DragAndDropLayout)
    cfg.setLayout(cfg.EditableGridLayout)

    # Clear all existing groups (keep individual field configs)
    for i in reversed(range(cfg.tabCount())):
        cfg.removeTab(i)

    # Build one tab per structure type
    for type_val, type_name, field_set in [
        (CULVERT, "Culvert", _TYPE_FIELDS[CULVERT]),
        (BRIDGE, "Bridge", _TYPE_FIELDS[BRIDGE]),
        (WEIR, "Weir", _TYPE_FIELDS[WEIR]),
        (GATE, "Gate", _TYPE_FIELDS[GATE]),
        (PUMP, "Pump", _TYPE_FIELDS[PUMP]),
    ]:
        tab = cfg.addTab(type_name)
        for fname in sorted(field_set):
            field_idx = layer.fields().lookupField(fname)
            if field_idx < 0:
                continue
            editor = QgsAttributeEditorField(fname, field_idx, tab)
            editor.setVisibilityExpression(
                f'"structure_type" = {type_val}'
            )
            tab.addChildElement(editor)

    layer.setEditFormConfig(cfg)

    # Wire culvert_code combo to its value map
    _init_culvert_code_combo(dialog, layer)

    # Set default values on new features
    _set_defaults(layer, feature)


def _culvert_code_value_map(layer):
    """Return a QGIS value map config map for culvert_code."""
    from qgis.core import QgsEditorWidgetSetup
    config = {"map": {}}
    for code, desc in _CULVERT_CODE_MAP.items():
        config["map"][desc] = code
    setup = QgsEditorWidgetSetup("ValueMap", config)
    return setup


def _init_culvert_code_combo(dialog, layer):
    """Set up the culvert_code field with a value map."""
    field_idx = layer.fields().lookupField("culvert_code")
    if field_idx < 0:
        return
    setup = _culvert_code_value_map(layer)
    layer.setEditorWidgetSetup(field_idx, setup)


def _set_defaults(layer, feature):
    """Set visible defaults on newly created features."""
    if feature.id() < 0:
        pass  # New feature — QGIS default values handle this from the field settings
```

- [ ] **Step 1: Create the form init file**

Create file `swe2d/workbench/forms/__init__.py` (empty) and `swe2d/workbench/forms/swe2d_structures_form.py` with the code above.

- [ ] **Step 2: Verify the file is importable**

```bash
python -c "import sys; sys.path.insert(0, '.'); from swe2d.workbench.forms.swe2d_structures_form import form_open; print('OK')"
```

Expected: OK

- [ ] **Step 3: Commit**

```bash
git add swe2d/workbench/forms/
git commit -m "feat: create QGIS attribute form init for structures layer

- Form groups fields by structure type with visibility expressions
- Only culvert, bridge, weir, gate, or pump fields visible per type
- culvert_code gets a descriptive value map (1-57 HDS-5 codes)
- Common fields (structure_id, type, crest_elev, enabled) always visible"
```

---

### Task 2: Register form init on structures layer

Update `topology_tab_view.py` to install the Python form init file on the structures layer when the layer is created or loaded.

**Modify:** `swe2d/workbench/views/topology_tab_view.py`

- [ ] **Step 1: Find the structures layer config section**

In `swe2d/workbench/views/topology_tab_view.py`, find where the structures layer is configured — look for `_is_structures` block inside `_configure_swe2d_layer_editors`. This is around lines 1267-1316.

- [ ] **Step 2: Register the form init file**

After the existing structure-type value map setup (after line 1269 or wherever `_STRUCTURE_TYPE_VALUE_MAP` is set), add:

```python
            # Register Python form init for conditional field visibility
            import os
            from qgis.PyQt.QtCore import QgsAttributeEditorGroup, QgsAttributeEditorField
            form_py = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "workbench", "forms", "swe2d_structures_form.py"
            )
            if os.path.exists(form_py):
                cfg.setInitCodePath(form_py)
                cfg.setInitFunction("form_open")
```

This replaces the old .ui file path registration (which was to a non-existent file). Remove or leave the dead .ui path — either way works since it was guarded by `os.path.exists`.

- [ ] **Step 3: Set default values on common fields**

In the same `_is_structures` block, set default values so new features have sensible defaults:

```python
            # Default values for new features
            _set_default_value(layer, "structure_type", 1)
            _set_default_value(layer, "crest_elev", 0.0)
            _set_default_value(layer, "enabled", 1)
            _set_default_value(layer, "roughness_n", 0.035)
            _set_default_value(layer, "length", 30.0)
            _set_default_value(layer, "entrance_loss_k", 0.5)
            _set_default_value(layer, "exit_loss_k", 1.0)
            _set_default_value(layer, "culvert_barrels", 1)
```

Add a helper function at module scope in `topology_tab_view.py`:

```python
def _set_default_value(layer, field_name: str, value) -> None:
    """Set a QGIS default value on a field for new features."""
    from qgis.core import QgsDefaultValue
    field_idx = layer.fields().lookupField(field_name)
    if field_idx >= 0:
        layer.setDefaultValueDefinition(field_idx, QgsDefaultValue(repr(value)))
```

- [ ] **Step 4: Remove the dead .ui file path**

Remove the block that attempts to load `swe2d_structures_culvert_form.ui` (the non-existent file). It's dead code — the `os.path.exists` guard made it a no-op.

- [ ] **Step 5: Commit**

```bash
git add swe2d/workbench/views/topology_tab_view.py
git commit -m "feat: register structures Python form init on the layer

- Register swe2d_structures_form.py as the attribute form init
- Set default values on common fields for new features
- Remove dead .ui file path (file never existed)"
```

---

### Task 3: Handle NULL invert elevations in structure config service

When `inlet_invert_elev` or `outlet_invert_elev` is NULL in the QGIS feature, the `_ReadVisitor` in `structure_config_service.py` currently reads it as `None` and stores it in metadata. Update the packing step to pass this through as NULL, and update `pack_structures_soa` in Task 4 to handle it.

**Modify:** `swe2d/workbench/services/structure_config_service.py`

- [ ] **Step 1: Find where invert elevations are read from the feature**

In `swe2d/workbench/services/structure_config_service.py`, find the `_ReadVisitor` or equivalent logic that reads culvert fields from a QGIS feature. Look for `inlet_invert_elev` and `outlet_invert_elev` reads (likely lines 112-127).

Currently they read the values and store them in metadata. Ensure that when the value is NULL (Python `None`), it is stored as `None` in metadata rather than being silently converted to 0.0 or `crest_elev`.

The current code at structure_config_service.py:112-127 reads fields and stores them in a metadata dict. Verify that NULL QGIS values produce `None` in Python (they do — QGIS `attribute()` returns `None` for NULL values). If the code currently converts them to a numeric default, change it to pass `None` through.

- [ ] **Step 2: Commit**

```bash
git add swe2d/workbench/services/structure_config_service.py
git commit -m "fix: pass through NULL invert elevations from QGIS features

- inlet_invert_elev and outlet_invert_elev now store None
  in metadata when the feature has NULL values
- Downstream pack_structures_soa handles NULL fallback"
```

---

### Task 4: Handle NULL inverts in pack_structures_soa

When `inlet_invert_elev` or `outlet_invert_elev` is NULL in the structure metadata, fall back to the cell bed elevation at solve time. This requires passing `cell_bed` to `pack_structures_soa` and doing the lookup when inverts are None.

**Modify:** `swe2d/runtime/coupling.py`

- [ ] **Step 1: Add cell_bed parameter to pack_structures_soa**

Find `pack_structures_soa` definition (around line 360). Add a `cell_bed: Optional[np.ndarray] = None` parameter.

- [ ] **Step 2: Update invert elevation fallback logic**

Change the invert elevation default logic from:

```python
inlet_invert_elev[i] = _meta_float(st.metadata, "inlet_invert_elev", st.crest_elev)
outlet_invert_elev[i] = _meta_float(st.metadata, "outlet_invert_elev", inlet_invert_elev[i])
```

to:

```python
raw_inlet = st.metadata.get("inlet_invert_elev")
raw_outlet = st.metadata.get("outlet_invert_elev")
if raw_inlet is not None:
    inlet_invert_elev[i] = float(raw_inlet)
elif cell_bed is not None and st.upstream_cell >= 0 and st.upstream_cell < len(cell_bed):
    inlet_invert_elev[i] = float(cell_bed[st.upstream_cell])
    log_fn(f"inlet_invert_elev not set for structure {st.structure_id} — defaulting to bed elevation {cell_bed[st.upstream_cell]:.3f}")
else:
    inlet_invert_elev[i] = st.crest_elev
    log_fn(f"inlet_invert_elev not set for structure {st.structure_id} — defaulting to crest elevation {st.crest_elev:.3f}")

if raw_outlet is not None:
    outlet_invert_elev[i] = float(raw_outlet)
elif cell_bed is not None and st.downstream_cell >= 0 and st.downstream_cell < len(cell_bed):
    outlet_invert_elev[i] = float(cell_bed[st.downstream_cell])
    log_fn(f"outlet_invert_elev not set for structure {st.structure_id} — defaulting to bed elevation {cell_bed[st.downstream_cell]:.3f}")
else:
    outlet_invert_elev[i] = inlet_invert_elev[i]
```

This establishes the fallback chain: explicit value → bed elevation → crest elevation → inlet value. The `log_fn` prints a clear message when the fallback triggers.

- [ ] **Step 3: Update the caller to pass cell_bed**

Find where `pack_structures_soa` is called (likely near where the coupling controller is initialized). Pass the cell_bed array. The cell_bed array should be in **original (pre-RCMK) ordering** since structure cell indices are in original order (from the exploration findings).

- [ ] **Step 4: Commit**

```bash
git add swe2d/runtime/coupling.py
git commit -m "feat: NULL invert elevations fall back to bed elevation

- pack_structures_soa accepts cell_bed array
- NULL inlet_invert_elev defaults to upstream cell's bed elevation
- NULL outlet_invert_elev defaults to downstream cell's bed elevation
- Falls back to crest elevation if cell_bed is unavailable
- Logs a message when fallback is triggered"
```

---

### Task 5: Set NULL defaults for invert fields in layer setup

In `topology_tab_view.py`, the invert elevation fields should default to NULL (not 0.0) so the solver's fallback logic triggers. QGIS defaults to NULL unless a default is set.

- [ ] **Step 1: Verify invert fields have no default value**

In the `_is_structures` block of `_configure_swe2d_layer_editors`, do NOT call `_set_default_value` for `inlet_invert_elev` or `outlet_invert_elev`. They should remain NULL by default.

- [ ] **Step 2: Verify no hardcoded NULL-to-zero conversion**

In `structure_config_service.py`, ensure that `None` is passed through for invert fields rather than being converted to 0.0 or crest_elev. This was handled in Task 3.

- [ ] **Step 3: Commit**

```bash
git add swe2d/workbench/views/topology_tab_view.py
git commit -m "fix: invert elevation fields default to NULL (no default value set)

- inlet_invert_elev and outlet_invert_elev have no QGIS default value
- NULL values trigger solver fallback to bed elevation with log message"
```

---

### Self-Review Checklist

1. **Spec coverage:**
   - Conditional field visibility by structure type? → Task 1 (form_open with visibility expressions)
   - Fields hidden or shown by type? → Task 1 (_TYPE_FIELDS dict per type)
   - Default values visible to user when feature created? → Task 2 (_set_default_value for common fields)
   - Value map for culvert code → Task 1 (_CULVERT_CODE_MAP)
   - NULL invert defaults → Task 4 (log fallback to bed elevation)
   - Default invert = bed elevation with log → Task 4 (cell_bed lookup)

2. **No placeholders** — all code is complete in tasks.

3. **Type consistency** — field names match the QGIS layer schema from topology_template_service.py and mesh_controller.py.

