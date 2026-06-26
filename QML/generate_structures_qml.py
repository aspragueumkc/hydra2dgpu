"""Generate QGIS QML style file for structures layer.

Run from QGIS Python console or with qgis_process:
    qgis_process run python://QML/generate_structures_qml.py

Output: QML/structures.qml

Applying to a GPKG:
    Place structures.qml next to the GPKG (same base name), or
    Layer → Properties → Load Style → structures.qml
"""
import os, sys, tempfile, json

QML_DIR = os.path.dirname(os.path.abspath(__file__))
PLUGIN_DIR = os.path.dirname(QML_DIR)

sys.path.insert(0, PLUGIN_DIR)
from swe2d.workbench.forms.swe2d_structures_form import _CULVERT_CODE_MAP

try:
    from qgis.core import (
        QgsApplication, QgsVectorLayer, QgsField, QgsDefaultValue,
        QgsEditorWidgetSetup, QgsEditFormConfig, QgsProject,
        QgsVectorFileWriter, QgsCoordinateReferenceSystem,
    )
    from qgis.PyQt.QtCore import QVariant
except ImportError:
    print("This script must be run inside QGIS (Python console or qgis_process).")
    sys.exit(1)

QgsApplication.setPrefixPath("", True)
QgsApplication.initQgis()

SCHEMA = [
    ("structure_id", "string", 64),
    ("structure_type", "integer"),
    ("crest_elev", "double"),
    ("enabled", "integer"),
    ("width", "double"),
    ("height", "double"),
    ("diameter", "double"),
    ("culvert_shape", "string", 32),
    ("culvert_code", "integer"),
    ("culvert_rise", "double"),
    ("culvert_span", "double"),
    ("culvert_area_m2", "double"),
    ("culvert_barrels", "integer"),
    ("culvert_slope", "double"),
    ("inlet_invert_elev", "double"),
    ("outlet_invert_elev", "double"),
    ("entrance_loss_k", "double"),
    ("exit_loss_k", "double"),
    ("embankment_enabled", "integer"),
    ("embankment_crest_elev", "double"),
    ("embankment_overflow_width", "double"),
    ("embankment_weir_coeff", "double"),
    ("length", "double"),
    ("roughness_n", "double"),
    ("inlet_loss_k", "double"),
    ("outlet_loss_k", "double"),
    ("coeff", "double"),
    ("cd", "double"),
    ("opening", "double"),
    ("q_pump", "double"),
    ("max_flow", "double"),
    ("face_flux_depth_safety", "double"),
    ("deck_soffit_elev", "double"),
    ("deck_top_elev", "double"),
    ("model_top_elev", "double"),
    ("under_layers", "integer"),
    ("over_layers", "integer"),
    ("pier_count", "integer"),
    ("pier_width", "double"),
]

OUTPUT_QML = os.path.join(QML_DIR, "structures.qml")

# Create a temporary GPKG with the structures schema
tmp_gpkg = os.path.join(tempfile.mkdtemp(), "tmp_structures.gpkg")
uri_parts = [f"{name}={{{type_}}}" if type_ == "integer" else f"{name}={type_}" for name, type_, *_ in SCHEMA]
uri = f"{tmp_gpkg}?{','.join([p for p in uri_parts])}"
layer = QgsVectorLayer(f"MultiLineString?crs=EPSG:4326&field={'&'.join(uri_parts)}", "structures", "memory")
assert layer.isValid(), "Failed to create temporary layer"

dp = layer.dataProvider()
dp.addAttributes([QgsField(name, QVariant.Int if t == "integer" else QVariant.Double if "double" in t else QVariant.String, len=str(length) if t == "string" else "") for name, t, *rest in SCHEMA])
layer.updateFields()

# Apply form init
form_py = os.path.join(PLUGIN_DIR, "swe2d", "workbench", "forms", "swe2d_structures_form.py")
cfg = layer.editFormConfig()
cfg.setInitCodePath(form_py)
cfg.setInitFunction("form_open")
cfg.setLayout(cfg.TabLayout)

# Default values
defaults = {
    "structure_type": 1,
    "crest_elev": 0.0,
    "enabled": 1,
    "roughness_n": 0.035,
    "length": 30.0,
    "entrance_loss_k": 0.5,
    "exit_loss_k": 1.0,
    "culvert_barrels": 1,
}
for name, val in defaults.items():
    idx = layer.fields().lookupField(name)
    if idx >= 0:
        layer.setDefaultValueDefinition(idx, QgsDefaultValue(repr(val)))

# Culvert code value map
vm_config = {"map": {}}
for code, desc in _CULVERT_CODE_MAP.items():
    vm_config["map"][desc] = code
idx = layer.fields().lookupField("culvert_code")
if idx >= 0:
    layer.setEditorWidgetSetup(idx, QgsEditorWidgetSetup("ValueMap", vm_config))

# Structure type value map
type_map = {"Weir": 1, "Culvert": 2, "Gate": 3, "Bridge": 4, "Pump": 5}
idx = layer.fields().lookupField("structure_type")
if idx >= 0:
    layer.setEditorWidgetSetup(idx, QgsEditorWidgetSetup("ValueMap", {"map": type_map}))

# Save QML
layer.saveNamedStyle(OUTPUT_QML)
print(f"Saved: {OUTPUT_QML}")
print(f"  Form init path: {form_py}")
print(f"  Init function: form_open")
print(f"  To apply to a GPKG: place structures.qml next to the GPKG and open in QGIS")
print(f"  Or: Layer Properties → Load Style → {OUTPUT_QML}")
