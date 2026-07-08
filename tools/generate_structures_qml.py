"""Generate QGIS QML style file for structures layer.

Usage: python QML/generate_structures_qml.py

Output: QML/structures.qml

Self-contained QML with all widget configs (value maps, constraints,
defaults, aliases, tab layout with type visibility). No Python form
init file needed. Drop next to any GPKG (same base name) and QGIS
auto-applies the style. Or: Layer Properties → Load Style.
"""
import os, xml.sax.saxutils as saxutils

QML_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_QML = os.path.join(QML_DIR, "structures.qml")

V = saxutils.escape

_CC = [
    (0, "— Select culvert code —"), (1, "Circular concrete, square edge w/ headwall"),
    (2, "Circular concrete, groove end w/ headwall"),
    (3, "Circular concrete, groove end projecting"),
    (4, "Circular concrete, mitred to slope"),
    (5, "Circular concrete, beveled ring"),
    (6, "Circular concrete, beveled ring (smoother)"),
    (7, "Circular CMP, projecting"),
    (8, "Circular CMP, projecting (different edge)"),
    (9, "Circular CMP, mitered to slope"),
    (10, "Circular CMP, mitered to slope (alt)"),
    (11, "Circular CMP, beveled end (thin wall)"),
    (12, "Circular CMP, groove end in headwall"),
    (13, "Circular CMP, groove end in headwall (alt)"),
    (14, "Circular CMP, headwall (square edge)"),
    (15, "Circular CMP, headwall (groove end)"),
    (16, "Circular CMP, headwall (thin wall projecting)"),
    (17, "Rectangular box, 30-75deg wingwall flares"),
    (18, "Rectangular box, 90deg headwall w/ chamfers"),
    (19, "Rectangular box, 0deg wingwall flares"),
    (20, "Rectangular box, 45deg wingwall flares"),
    (21, "Rectangular box, 18-33deg wingwall flares"),
    (22, "Rectangular box, 0deg wingwall flares (thick)"),
    (23, "Rectangular box, 30deg wingwall flares (thick)"),
    (24, "Rectangular box, 45deg wingwall flares (thick)"),
    (25, "Rectangular box, 0deg wingwall flares (thick alt)"),
    (26, "Rectangular box, beveled edge (1:1)"),
    (27, "Circular concrete, square edge w/ headwall (form-1 alt)"),
    (28, "Circular concrete, groove end w/ headwall (form-1 alt)"),
    (29, "Circular concrete, groove end projecting (form-1 alt)"),
    (30, "Circular CMP, projecting (form-1 alt)"),
    (31, "Circular CMP, mitered to slope (form-1 alt)"),
    (32, "Circular CMP, beveled end thin wall (form-1 alt)"),
    (33, "Circular CMP, groove end in headwall (form-1 alt)"),
    (34, "Circular CMP, headwall square edge (form-1 alt)"),
    (35, "Circular CMP, headwall groove end (form-1 alt)"),
    (36, "Circular CMP, beveled ring (form-1 alt)"),
    (37, "Circular CMP, beveled ring thick (form-1 alt)"),
    (38, "Circular concrete, beveled ring (form-1 alt)"),
    (39, "Circular pipe, beveled ring (thin wall)"),
    (40, "Circular pipe, beveled ring (thick wall)"),
    (41, "Circular pipe, 45deg beveled ring"),
    (42, "Circular pipe, 33.7deg beveled ring"),
    (43, "Circular pipe, 45deg bevel (offset)"),
    (44, "Circular pipe, 33.7deg bevel (offset)"),
    (45, "Circular CMP, prefab end section (safety)"),
    (46, "Circular CMP, prefab end section (alt)"),
    (47, "Arch CMP, 2-3-1 fill (soffit thickness 0.0625)"),
    (48, "Arch CMP, 2-3-1 fill (soffit varying)"),
    (49, "Arch CMP, 2-3-1 fill projecting (soffit varying)"),
    (50, "Arch CMP, 2-2-1 fill (soffit thickness 0.0625)"),
    (51, "Pipe arch CMP, 0.75x0.75 fill (soffit thickness 0.0625)"),
    (52, "Pipe arch CMP, 0.75x0.75 fill projecting"),
    (53, "Pipe arch CMP, 0.75x0.75 fill (soffit varying)"),
    (54, "Horizontal ellipse, concrete (form-2)"),
    (55, "Horizontal ellipse, corrugated metal (form-2)"),
    (56, "Arch CMP, 2-3-1 fill premium (form-2)"),
    (57, "Horizontal ellipse, special shape (form-2)"),
]

def _entries(items):
    return "\n".join(f'              <value pair="{V(desc)}">{code}</value>' for code, desc in items)

def _tab(name, type_val, *fields):
    fxml = "\n".join(f'      <attributeEditorField index="-1" name="{f}"/>' for f in fields)
    return f"""    <attributeEditorContainer name="{name}" visibilityExpressionEnabled="1" visibilityExpression="&quot;structure_type&quot; = {type_val}" groupBox="0">
{fxml}
    </attributeEditorContainer>"""

TAB_LIST = [
    ("Weir", 1, "width", "embankment_enabled", "embankment_crest_elev",
     "embankment_overflow_width", "embankment_weir_coeff"),
    ("Culvert", 2, "culvert_shape", "culvert_code", "culvert_rise", "culvert_span",
     "culvert_area_m2", "culvert_barrels", "culvert_slope", "diameter", "length",
     "roughness_n", "inlet_invert_elev", "outlet_invert_elev", "entrance_loss_k",
     "exit_loss_k", "embankment_enabled", "embankment_crest_elev",
     "embankment_overflow_width", "embankment_weir_coeff"),
    ("Gate", 3, "width", "height", "opening"),
    ("Bridge", 4, "width", "length", "deck_soffit_elev", "deck_top_elev",
     "model_top_elev", "under_layers", "over_layers", "inlet_loss_k", "outlet_loss_k",
     "pier_count", "pier_width", "face_flux_depth_safety"),
    ("Pump", 5, "q_pump", "max_flow", "min_head_diff", "max_head_diff"),
]

ALIASES = [
    ("culvert_shape", "Culvert Shape"), ("culvert_code", "FHWA Culvert Code"),
    ("culvert_rise", "Culvert Rise"), ("culvert_span", "Culvert Span"),
    ("culvert_area_m2", "Override Area"), ("culvert_barrels", "Barrel Count"),
    ("culvert_slope", "Culvert Slope"), ("inlet_invert_elev", "Inlet Invert Elev."),
    ("outlet_invert_elev", "Outlet Invert Elev."),
    ("entrance_loss_k", "Entrance Loss K"), ("exit_loss_k", "Exit Loss K"),
    ("embankment_enabled", "Enable Embankment Overflow"),
    ("embankment_crest_elev", "Embankment Crest Elev."),
    ("embankment_overflow_width", "Overflow Width"),
    ("embankment_weir_coeff", "Weir Coefficient"),
]

CONSTRAINTS = [
    ("structure_id", 'length(trim("structure_id")) > 0'),
    ("structure_type", '"structure_type" IN (1,2,3,4,5)'),
    ("enabled", '"enabled" IS NULL OR "enabled" IN (0,1)'),
    ("culvert_code", '"culvert_code" IS NULL OR "culvert_code" >= 1'),
    ("culvert_rise", '"culvert_rise" IS NULL OR "culvert_rise" > 0'),
    ("culvert_span", '"culvert_span" IS NULL OR "culvert_span" > 0'),
    ("culvert_area_m2", '"culvert_area_m2" IS NULL OR "culvert_area_m2" > 0'),
    ("culvert_barrels", '"culvert_barrels" IS NULL OR "culvert_barrels" >= 1'),
    ("length", '"length" IS NULL OR "length" > 0'),
    ("roughness_n", '"roughness_n" IS NULL OR "roughness_n" > 0'),
    ("entrance_loss_k", '"entrance_loss_k" IS NULL OR "entrance_loss_k" >= 0'),
    ("exit_loss_k", '"exit_loss_k" IS NULL OR "exit_loss_k" >= 0'),
    ("embankment_enabled", '"embankment_enabled" IS NULL OR "embankment_enabled" IN (0,1)'),
    ("embankment_overflow_width", '"embankment_overflow_width" IS NULL OR "embankment_overflow_width" >= 0'),
    ("embankment_weir_coeff", '"embankment_weir_coeff" IS NULL OR "embankment_weir_coeff" > 0'),
]

ALL_FIELDS_IN_TABS = sorted({f for _, _, *fs in TAB_LIST for f in fs})

qml = f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.44.0" editorLayout="tablayout">
  <fieldConfiguration>
    <field name="structure_id">
      <editWidget type="TextEdit">
        <config><Option type="Map">
          <Option value="0" name="IsMultiline" type="int"/>
          <Option value="0" name="UseHtml" type="int"/>
        </Option></config>
      </editWidget>
    </field>
    <field name="structure_type">
      <editWidget type="ValueMap">
        <config><Option type="Map">
          <Option name="map" type="List">
            <Option value="1" name="Weir" type="int"/>
            <Option value="2" name="Culvert" type="int"/>
            <Option value="3" name="Gate" type="int"/>
            <Option value="4" name="Bridge" type="int"/>
            <Option value="5" name="Pump" type="int"/>
          </Option>
        </Option></config>
      </editWidget>
    </field>
    <field name="enabled">
      <editWidget type="ValueMap">
        <config><Option type="Map">
          <Option name="map" type="List">
            <Option value="1" name="Yes" type="int"/>
            <Option value="0" name="No" type="int"/>
          </Option>
        </Option></config>
      </editWidget>
    </field>
    <field name="culvert_code">
      <editWidget type="ValueMap">
        <config><Option type="Map">
          <Option name="map" type="List">
{_entries(_CC)}
          </Option>
        </Option></config>
      </editWidget>
    </field>
    <field name="culvert_shape">
      <editWidget type="ValueMap">
        <config><Option type="Map">
          <Option name="map" type="List">
            <Option value="circular" name="Circular" type="string"/>
            <Option value="box" name="Box" type="string"/>
            <Option value="rectangular" name="Rectangular" type="string"/>
          </Option>
        </Option></config>
      </editWidget>
    </field>
  </fieldConfiguration>
  <aliases>
{chr(10).join(f'    <alias field="{f}" index="-1" name="{V(a)}"/>' for f, a in ALIASES)}
  </aliases>
  <defaults>
    <default field="structure_type" applyOnUpdate="0" expression="1"/>
    <default field="crest_elev" applyOnUpdate="0" expression="0.0"/>
    <default field="enabled" applyOnUpdate="0" expression="1"/>
    <default field="roughness_n" applyOnUpdate="0" expression="0.035"/>
    <default field="length" applyOnUpdate="0" expression="30.0"/>
    <default field="entrance_loss_k" applyOnUpdate="0" expression="0.5"/>
    <default field="exit_loss_k" applyOnUpdate="0" expression="1.0"/>
    <default field="culvert_barrels" applyOnUpdate="0" expression="1"/>
  </defaults>
  <constraints>
{chr(10).join(f'    <constraint field="{f}" exp_strength="0" constraints="3" desc="" expression="{V(e)}"/>' for f, e in CONSTRAINTS)}
  </constraints>
  <editorlayout>tablayout</editorlayout>
  <attributeEditorForm>
{chr(10).join(_tab(*t) for t in TAB_LIST)}
  </attributeEditorForm>
</qgis>"""

with open(OUTPUT_QML, "w") as f:
    f.write(qml)

print(f"Saved: {OUTPUT_QML}")
print(f"All widget configs embedded (value maps, constraints, defaults, aliases, tabs)")
