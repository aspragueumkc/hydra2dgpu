"""Generate QGIS QML style file for structures layer.

Usage: python QML/generate_structures_qml.py

Output: QML/structures.qml

The QML is written directly as XML — no QGIS installation required.
Drop structures.qml next to any GPKG (same base name) and QGIS
auto-loads the style. Or: Layer Properties → Load Style.
"""
import os, xml.sax.saxutils as saxutils

QML_DIR = os.path.dirname(os.path.abspath(__file__))
PLUGIN_DIR = os.path.dirname(QML_DIR)

# Load form init code to embed it in the QML
form_init_path = os.path.join(PLUGIN_DIR, "swe2d", "workbench", "forms", "swe2d_structures_form.py")
with open(form_init_path) as f:
    INIT_CODE = f.read()

OUTPUT_QML = os.path.join(QML_DIR, "structures.qml")

CULVERT_CODE_MAP_XML = "\n".join(
    f'              <value pair="{saxutils.escape(desc)}">{code}</value>'
    for code, desc in [
        (0, "— Select culvert code —"),
        (1, "Circular concrete, square edge w/ headwall"),
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
)

qml = f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.44.0" styleCategories="AllStyleCategories" editorLayout="tablayout">
  <fieldConfiguration>
    <field name="structure_id">
      <editWidget type="TextEdit">
        <config>
          <Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option>
        </config>
      </editWidget>
    </field>
    <field name="structure_type">
      <editWidget type="ValueMap">
        <config>
          <Option type="Map">
            <Option name="map" type="List">
              <Option value="1" name="Weir" type="int"/>
              <Option value="2" name="Culvert" type="int"/>
              <Option value="3" name="Gate" type="int"/>
              <Option value="4" name="Bridge" type="int"/>
              <Option value="5" name="Pump" type="int"/>
            </Option>
          </Option>
        </config>
      </editWidget>
    </field>
    <field name="crest_elev">
      <editWidget type="TextEdit">
        <config>
          <Option type="Map">
            <Option value="0" name="IsMultiline" type="int"/>
            <Option value="0" name="UseHtml" type="int"/>
          </Option>
        </config>
      </editWidget>
    </field>
    <field name="enabled">
      <editWidget type="ValueMap">
        <config>
          <Option type="Map">
            <Option name="map" type="List">
              <Option value="1" name="Yes" type="int"/>
              <Option value="0" name="No" type="int"/>
            </Option>
          </Option>
        </config>
      </editWidget>
    </field>
    <field name="culvert_code">
      <editWidget type="ValueMap">
        <config>
          <Option type="Map">
            <Option name="map" type="List">
{CULVERT_CODE_MAP_XML}
            </Option>
          </Option>
        </config>
      </editWidget>
    </field>
  </fieldConfiguration>
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
  <editorlayout>tablayout</editorlayout>
  <editforminit>{saxutils.escape(INIT_CODE)}</editforminit>
  <editforminitinitcode>form_open</editforminitinitcode>
  <editforminitcodesource>1</editforminitcodesource>
</qgis>"""

with open(OUTPUT_QML, "w") as f:
    f.write(qml)

print(f"Saved: {OUTPUT_QML}")
print(f"To apply: Layer Properties → Load Style → {OUTPUT_QML}")
print(f"Or place {OUTPUT_QML} next to any GPKG with the same base name")
