"""Generate swe2d_drainage_inlets.qml with HEC-22 value maps and aliases.
"""
import os

_QML_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "QML")


def _esc(v):
    s = str(v)
    return s.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")


def _qt(v):
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "double"
    return "QString"


def _val_map(items):
    entries = "\n".join(
        '              <Option type="Map">\n'
        '                <Option value="{}" name="{}" type="{}"/>\n'
        '              </Option>'.format(_esc(str(val)), _esc(label), _qt(val))
        for label, val in items
    )
    return (
        '          <config>\n'
        '            <Option type="Map">\n'
        '              <Option name="map" type="List">\n'
        f'{entries}\n'
        '              </Option>\n'
        '            </Option>\n'
        '          </config>'
    )


# (name, widget, config, alias, constr_exp)
F = [
    ("inlet_type_id", "TextEdit", None, "Inlet Type ID", 'length(trim("inlet_type_id")) > 0'),
    ("name", "TextEdit", None, "Name", ""),
    ("inlet_type", "ValueMap", _val_map([
        ("Grate", "grate"),
        ("Curb-opening", "curb"),
        ("Slotted drain", "slotted"),
        ("Combined grate + curb", "combo"),
        ("Custom (legacy)", "custom"),
    ]), "Inlet Type", '"inlet_type" IN (\'grate\',\'curb\',\'slotted\',\'combo\',\'custom\')'),
    ("grate_length", "TextEdit", None, "Grate Length", ""),
    ("grate_width", "TextEdit", None, "Grate Width", ""),
    ("grate_type", "ValueMap", _val_map([
        ("Generic (use open fraction)", -1),
        ("HEC-22 P-1-1/8", 0),
        ("HEC-22 P-1-1/4", 1),
        ("HEC-22 P-1-1/2", 2),
        ("HEC-22 P-1-3/4", 3),
        ("HEC-22 P-1-7/8", 4),
        ("HEC-22 Reticuline", 5),
        ("HEC-22 Hinged", 6),
        ("HEC-22 Fish (deprecated)", 7),
    ]), "Grate Type (HEC-22)", '"grate_type" >= -1 AND "grate_type" <= 7'),
    ("grate_open_frac", "TextEdit", None, "Grate Open Fraction", ""),
    ("curb_length", "TextEdit", None, "Curb Length", ""),
    ("curb_height", "TextEdit", None, "Curb Height", ""),
    ("curb_throat", "ValueMap", _val_map([
        ("Vertical", 0),
        ("Horizontal", 1),
        ("Inclined", 2),
    ]), "Curb Throat Orientation", '"curb_throat" IN (0,1,2)'),
    ("slot_length", "TextEdit", None, "Slot Length", ""),
    ("slot_width", "TextEdit", None, "Slot Width", ""),
    ("weir_length", "TextEdit", None, "Weir Length (legacy)", ""),
    ("orifice_area", "TextEdit", None, "Orifice Area (legacy)", ""),
    ("coeff_weir", "TextEdit", None, "Weir Coefficient", ""),
    ("coeff_orifice", "TextEdit", None, "Orifice Coefficient", ""),
    ("max_capture", "TextEdit", None, "Max Capture", ""),
    ("description", "TextEdit", None, "Description", ""),
]

_FIELDS = [(n, w, c, a, e) for n, w, c, a, e in F]


def _field_xml(name, widget, config):
    cfg = config or '<config>\n            <Option/>\n          </config>'
    return (
        f'    <field configurationFlags="NoFlag" name="{name}">\n'
        f'      <editWidget type="{widget}">\n'
        f'{cfg}\n'
        f'      </editWidget>\n'
        f'    </field>'
    )


def generate():
    lines = []
    lines.append('<!DOCTYPE qgis PUBLIC \'http://mrcc.com/qgis.dtd\' \'SYSTEM\'>')
    lines.append('<qgis version="3.34.4" styleCategories="Fields|Forms|AttributeTable">')
    lines.append('  <fieldConfiguration>')
    for n, w, c, *_ in _FIELDS:
        lines.append(_field_xml(n, w, c))
    lines.append('  </fieldConfiguration>')

    lines.append('  <aliases>')
    for i, (n, *_, a, _) in enumerate(_FIELDS):
        lines.append(f'    <alias index="{i}" field="{n}" name="{a}"/>')
    lines.append('  </aliases>')

    lines.append('  <defaults>')
    for n, *_ in _FIELDS:
        lines.append(f'    <default expression="" field="{n}" applyOnUpdate="0"/>')
    lines.append('  </defaults>')

    lines.append('  <constraints>')
    for n, *_, e in _FIELDS:
        s = "4" if e else "0"
        lines.append(f'    <constraint notnull_strength="0" exp_strength="{s}" constraints="{s}" field="{n}" unique_strength="0"/>')
    lines.append('  </constraints>')

    lines.append('  <constraintExpressions>')
    for n, *_, e in _FIELDS:
        lines.append(f'    <constraint exp="{_esc(e)}" field="{n}" desc=""/>')
    lines.append('  </constraintExpressions>')

    lines.append('  <editform></editform>')
    lines.append('  <editforminit></editforminit>')
    lines.append('  <editforminitcodesource>0</editforminitcodesource>')
    lines.append('  <editforminitfilepath></editforminitfilepath>')
    lines.append('  <editforminitcode><![CDATA[]]></editforminitcode>')
    lines.append('  <featformsuppress>0</featformsuppress>')
    lines.append('  <editorlayout>tablayout</editorlayout>')
    lines.append('  <editable>')
    for n, *_ in _FIELDS:
        lines.append(f'    <field name="{n}"/>')
    lines.append('  </editable>')
    lines.append('  <labelOnTop>')
    lines.append('  </labelOnTop>')
    lines.append('  <reuseLastValue>')
    lines.append('  </reuseLastValue>')
    lines.append('  <dataDefinedFieldProperties/>')
    lines.append('  <widgets/>')
    lines.append('</qgis>')

    return "\n".join(lines)


if __name__ == "__main__":
    qml = generate()
    out_path = os.path.join(_QML_DIR, "swe2d_drainage_inlets.qml")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(qml)
    print(f"  ✓ {out_path}  ({len(qml)} bytes)")
