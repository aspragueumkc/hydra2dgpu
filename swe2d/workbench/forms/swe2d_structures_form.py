"""QGIS attribute form init script for structures layer.
Installed on the layer via setInitCodePath / setInitFunction.
Groups fields by structure type using visibility expressions so only
relevant fields appear for the selected structure type.
"""
from qgis.core import QgsEditFormConfig, QgsAttributeEditorField, QgsAttributeEditorContainer, QgsOptionalExpression, QgsExpression

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
    """Called by QGIS when opening a feature's attribute form."""
    _init_culvert_code_combo(dialog, layer)


def _set_visibility(container, expression):
    expr = QgsOptionalExpression()
    expr.setData(QgsExpression(expression))
    container.setVisibilityExpression(expr)


def _init_culvert_code_combo(dialog, layer):
    """Set up the culvert_code field with a value map."""
    from qgis.core import QgsEditorWidgetSetup
    field_idx = layer.fields().lookupField("culvert_code")
    if field_idx < 0:
        return
    config = {"map": {}}
    for code, desc in _CULVERT_CODE_MAP.items():
        config["map"][desc] = code
    setup = QgsEditorWidgetSetup("ValueMap", config)
    layer.setEditorWidgetSetup(field_idx, setup)
