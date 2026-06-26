"""Generate QGIS QML style files for all 18 SWE2D model layers.

Usage:  python QML/generate_all_layer_qmls.py

Output: 18 .qml files in QML/  — one per model layer.

Each QML matches the exact XML format produced by
``QgsVectorLayer.saveNamedStyle()`` for editor widget config
(ValueMap, Range, constraints, aliases, defaults).

Adding default symbology (fill colours, line styles, markers) to these
QMLs is just a matter of editing the generated XML — they are standalone
files that can be committed to version control and customised per build.
"""
from __future__ import annotations

import functools
import os
import xml.sax.saxutils as saxutils

QML_DIR = os.path.dirname(os.path.abspath(__file__))

# Escape for XML attribute values — must handle &, <, >, and "
_V_ESCAPE = functools.partial(saxutils.escape, entities={'"': '&quot;'})


def _qtype(value) -> str:
    """Return the QGIS QML ``type`` attribute for a Python value."""
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "double"
    return "string"


# ── Widget config builders ────────────────────────────────────────────────
# These produce <config> blocks in the EXACT format QGIS itself writes.


def _value_map_config(items: list) -> str:
    """ValueMap config — QGIS wraps each entry in ``<Option type="Map">``."""
    entries = "\n".join(
        '              <Option type="Map">\n'
        '                <Option value="{}" name="{}" type="{}"/>\n'
        '              </Option>'.format(
            _V_ESCAPE(str(val)), _V_ESCAPE(label), _qtype(val),
        )
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


def _range_config(min_v, max_v, step) -> str:
    """Range widget config in QGIS canonical format."""
    return (
        '          <config>\n'
        '            <Option type="Map">\n'
        f'              <Option value="{min_v}" name="Min" type="int"/>\n'
        f'              <Option value="{max_v}" name="Max" type="int"/>\n'
        f'              <Option value="{step}" name="Step" type="int"/>\n'
        '            </Option>\n'
        '          </config>'
    )


_PLAIN_CONFIG = (
    '          <config>\n'
    '            <Option/>\n'
    '          </config>'
)


_TEXT_EDIT_CONFIG = (
    '          <config>\n'
    '            <Option type="Map">\n'
    '              <Option value="0" name="IsMultiline" type="int"/>\n'
    '              <Option value="0" name="UseHtml" type="int"/>\n'
    '            </Option>\n'
    '          </config>'
)


# ── Building blocks ────────────────────────────────────────────────────────


def _field_block(name: str, widget_type: str = "",
                 config: str | None = None) -> str:
    """``<field configurationFlags="NoFlag" name="...">`` block."""
    cfg = config if config is not None else _PLAIN_CONFIG
    return (
        f'    <field configurationFlags="NoFlag" name="{_V_ESCAPE(name)}">\n'
        f'      <editWidget type="{widget_type}">\n'
        f'{cfg}\n'
        '      </editWidget>\n'
        '    </field>'
    )


def _constraint_entry(field: str, has_expr: bool) -> str:
    """Entry inside ``<constraints>`` block."""
    bits = "4" if has_expr else "0"
    es = "1" if has_expr else "0"
    return (
        f'    <constraint notnull_strength="0" exp_strength="{es}" '
        f'constraints="{bits}" field="{_V_ESCAPE(field)}" unique_strength="0"/>'
    )


def _constraint_expr(field: str, expr: str) -> str:
    """Entry inside ``<constraintExpressions>`` block."""
    return (
        f'    <constraint exp="{_V_ESCAPE(expr)}" field="{_V_ESCAPE(field)}" desc=""/>'
    )


def _alias_entry(field_name: str, index: int, alias: str) -> str:
    return f'    <alias index="{index}" field="{_V_ESCAPE(field_name)}" name="{_V_ESCAPE(alias)}"/>'


def _default_entry(field: str, expr: str, apply_on_update: str = "0") -> str:
    return (
        f'    <default expression="{_V_ESCAPE(expr)}" '
        f'field="{_V_ESCAPE(field)}" applyOnUpdate="{apply_on_update}"/>'
    )


QGIS_VERSION = "3.34.4"


def build_qml(geom_type: str | None,
              fields: list,
              constraint_exprs: list | None = None,
              aliases: list | None = None,
              defaults: list | None = None) -> str:
    """Assemble a QML document matching ``QgsVectorLayer.saveNamedStyle()``.

    Parameters
    ----------
    geom_type:
        ``"Point"``, ``"LineString"``, ``"Polygon"``, or ``None`` (table).
    fields:
        List of ``(name, widget_type, widget_config_or_None)``.
        Pass ``None`` for config to get a plain TextEdit.
    constraint_exprs:
        ``[(field, expression), ...]`` — only fields WITH constraints.
        All other fields get an empty entry automatically.
    aliases:
        ``[(index, alias), ...]`` — only fields WITH non-empty aliases.
    defaults:
        ``[(field, expression, applyOnUpdate), ...]`` — only fields WITH defaults.
    """
    field_names = [f[0] for f in fields]
    n_fields = len(field_names)

    # Build maps
    constraint_map = dict(constraint_exprs or [])
    # aliases arrive as [(index, alias), ...]; convert to {field_name: alias}
    alias_map = {}
    if aliases:
        for idx, alias in aliases:
            if 0 <= idx < len(field_names):
                alias_map[field_names[idx]] = alias
    default_map = {d[0]: (d[1], d[2]) for d in (defaults or [])}

    # Sections
    field_xml = "\n".join(_field_block(n, w, c) for n, w, c in fields)

    constraint_entries = "\n".join(
        _constraint_entry(n, n in constraint_map) for n in field_names
    )

    expr_entries = "\n".join(
        _constraint_expr(n, constraint_map.get(n, "")) for n in field_names
    )

    alias_entries = "\n".join(
        _alias_entry(n, i, alias_map.get(n, "")) for i, n in enumerate(field_names)
    )

    default_entries = "\n".join(
        _default_entry(n, *default_map.get(n, ("", "0"))) for n in field_names
    )

    editable_entries = "\n".join(
        f'    <field name="{_V_ESCAPE(n)}"/>' for n in field_names
    )

    return (
        '<!DOCTYPE qgis PUBLIC \'http://mrcc.com/qgis.dtd\' \'SYSTEM\'>\n'
        f'<qgis version="{QGIS_VERSION}"'
        ' styleCategories="Fields|Forms|AttributeTable">\n'
        '  <fieldConfiguration>\n'
        f'{field_xml}\n'
        '  </fieldConfiguration>\n'
        '  <aliases>\n'
        f'{alias_entries}\n'
        '  </aliases>\n'
        '  <defaults>\n'
        f'{default_entries}\n'
        '  </defaults>\n'
        '  <constraints>\n'
        f'{constraint_entries}\n'
        '  </constraints>\n'
        '  <constraintExpressions>\n'
        f'{expr_entries}\n'
        '  </constraintExpressions>\n'
        '  <editform></editform>\n'
        '  <editforminit></editforminit>\n'
        '  <editforminitcodesource>0</editforminitcodesource>\n'
        '  <editforminitfilepath></editforminitfilepath>\n'
        '  <editforminitcode><![CDATA[]]></editforminitcode>\n'
        '  <featformsuppress>0</featformsuppress>\n'
        '  <editorlayout>tablayout</editorlayout>\n'
        '  <editable>\n'
        f'{editable_entries}\n'
        '  </editable>\n'
        '  <labelOnTop>\n'
        '  </labelOnTop>\n'
        '  <reuseLastValue>\n'
        '  </reuseLastValue>\n'
        '  <dataDefinedFieldProperties/>\n'
        '  <widgets/>\n'
        '</qgis>\n'
    )


# ── Value map definitions ─────────────────────────────────────────────────

ARC_MODE_ITEMS = [
    ("Hard embed arcs", "hard_embed"),
    ("Soft arc size hint", "soft_size_hint"),
    ("Disable arc influence", "disabled"),
]

CELL_TYPE_ITEMS = [
    ("Triangular", "triangular"),
    ("Quadrilateral", "quadrilateral"),
    ("Cartesian", "cartesian"),
    ("Channel generator", "channel_generator"),
    ("Empty", "empty"),
]

BC_TYPE_ITEMS = [
    ("Wall (zero normal flux)", 1),
    ("Inflow Q (total discharge)", 2),
    ("Stage (prescribed WSE)", 3),
    ("Normal Depth (prescribed depth)", 6),
    ("Normal Depth (friction slope Sf)", 7),
    ("Timeseries Flow Q", 102),
    ("Timeseries Stage", 103),
    ("Open (zero-gradient)", 4),
    ("Reflecting", 5),
]

DRAIN_NODE_TYPE_ITEMS = [
    ("Junction", "junction"),
    ("Outfall", "outfall"),
    ("Storage", "storage"),
    ("Inlet", "inlet"),
    ("Pipe end", "pipe_end"),
]

DRAIN_LINK_TYPE_ITEMS = [
    ("Conduit", "conduit"),
    ("Short lateral (simplified)", "lateral_simple"),
    ("Pump", "pump"),
    ("Weir", "weir"),
    ("Orifice", "orifice"),
    ("Culvert (HDS-5)", "culvert"),
]

DRAIN_LINK_SHAPE_ITEMS = [
    ("Circular", "circular"),
    ("Box", "box"),
    ("Pipe arch", "pipe_arch"),
    ("Custom area", "custom"),
]

RAIN_GAGE_UNITS_ITEMS = [
    ("mm/hr", "mm/hr"),
    ("in/hr", "in/hr"),
    ("mm", "mm"),
    ("in", "in"),
]

HYETOGRAPH_VALUE_TYPE_ITEMS = [
    ("Intensity", "intensity"),
    ("Incremental depth", "incremental"),
    ("Cumulative depth", "cumulative"),
]

STRUCTURE_TYPE_ITEMS = [
    ("Weir", 1),
    ("Culvert", 2),
    ("Gate", 3),
    ("Bridge", 4),
    ("Pump", 5),
]


# ── Layer definitions ─────────────────────────────────────────────────────

# Each entry: (file_name, geom_type, [field_tuples], [constraints], [aliases], [defaults])

LAYERS = []

# 1. swe2d_topo_nodes
LAYERS.append(("swe2d_topo_nodes", "Point", [
    ("node_id", "Range", _range_config(0, 2147483647, 1)),
], None, [
    (0, "Node ID"),
], None))

# 2. swe2d_topo_arcs
LAYERS.append(("swe2d_topo_arcs", "LineString", [
    ("arc_id", "Range", _range_config(0, 2147483647, 1)),
    ("node0", "Range", _range_config(0, 2147483647, 1)),
    ("node1", "Range", _range_config(0, 2147483647, 1)),
    ("use_global_arc_ctrl", "ValueMap", _value_map_config([
        ("Use global control", 1),
        ("Per-arc override", 0),
    ])),
    ("arc_mode_override", "ValueMap", _value_map_config(ARC_MODE_ITEMS)),
    ("arc_soft_size_override", "TextEdit", None),
    ("arc_soft_dist_override", "TextEdit", None),
], [
    ("use_global_arc_ctrl", '"use_global_arc_ctrl" IS NULL OR "use_global_arc_ctrl" IN (0,1)'),
    ("arc_mode_override", '"arc_mode_override" IS NULL OR "arc_mode_override" IN (\'hard_embed\',\'soft_size_hint\',\'disabled\')'),
    ("arc_soft_size_override", '"arc_soft_size_override" IS NULL OR "arc_soft_size_override" > 0'),
    ("arc_soft_dist_override", '"arc_soft_dist_override" IS NULL OR "arc_soft_dist_override" > 0'),
], [
    (0, "Arc ID"),
    (1, "From Node"),
    (2, "To Node"),
    (3, "Use Global Arc Ctrl"),
    (4, "Arc Mode Override"),
    (5, "Arc Soft Size Override"),
    (6, "Arc Soft Dist Override"),
], None))

# 3. swe2d_topo_regions
LAYERS.append(("swe2d_topo_regions", "Polygon", [
    ("region_id", "Range", _range_config(1, 2147483647, 1)),
    ("target_size", "TextEdit", None),
    ("cell_type", "ValueMap", _value_map_config(CELL_TYPE_ITEMS)),
    ("edge_len_1", "TextEdit", None),
    ("edge_len_2", "TextEdit", None),
    ("edge_len_3", "TextEdit", None),
    ("edge_len_4", "TextEdit", None),
], [
    ("cell_type", '"cell_type" IN (\'triangular\',\'quadrilateral\',\'cartesian\',\'channel_generator\',\'empty\')'),
    ("target_size", '"target_size" > 0'),
    ("edge_len_1", '"edge_len_1" IS NULL OR "edge_len_1" > 0'),
    ("edge_len_2", '"edge_len_2" IS NULL OR "edge_len_2" > 0'),
    ("edge_len_3", '"edge_len_3" IS NULL OR "edge_len_3" > 0'),
    ("edge_len_4", '"edge_len_4" IS NULL OR "edge_len_4" > 0'),
], [
    (0, "Region ID"),
    (1, "Target Size"),
    (2, "Cell Type"),
    (3, "Edge Length 1"),
    (4, "Edge Length 2"),
    (5, "Edge Length 3"),
    (6, "Edge Length 4"),
], None))

# 4. swe2d_topo_constraints
LAYERS.append(("swe2d_topo_constraints", "Polygon", [
    ("constraint_id", "Range", _range_config(1, 2147483647, 1)),
    ("target_size", "TextEdit", None),
    ("cell_type", "ValueMap", _value_map_config(CELL_TYPE_ITEMS)),
    ("edge_len_1", "TextEdit", None),
    ("edge_len_2", "TextEdit", None),
    ("edge_len_3", "TextEdit", None),
    ("edge_len_4", "TextEdit", None),
], [
    ("cell_type", '"cell_type" IN (\'triangular\',\'quadrilateral\',\'cartesian\',\'channel_generator\',\'empty\')'),
    ("target_size", '"target_size" > 0'),
    ("edge_len_1", '"edge_len_1" IS NULL OR "edge_len_1" > 0'),
    ("edge_len_2", '"edge_len_2" IS NULL OR "edge_len_2" > 0'),
    ("edge_len_3", '"edge_len_3" IS NULL OR "edge_len_3" > 0'),
    ("edge_len_4", '"edge_len_4" IS NULL OR "edge_len_4" > 0'),
], [
    (0, "Constraint ID"),
    (1, "Target Size"),
    (2, "Cell Type"),
    (3, "Edge Length 1"),
    (4, "Edge Length 2"),
    (5, "Edge Length 3"),
    (6, "Edge Length 4"),
], None))

# 5. swe2d_topo_quad_edges
LAYERS.append(("swe2d_topo_quad_edges", "LineString", [
    ("region_id", "Range", _range_config(1, 2147483647, 1)),
    ("edge_id", "ValueMap", _value_map_config([
        ("Edge 1 (left)", 1),
        ("Edge 2 (right)", 2),
        ("Edge 3 (bottom)", 3),
        ("Edge 4 (top)", 4),
    ])),
    ("target_size", "TextEdit", None),
    ("n_layers", "Range", _range_config(0, 1000, 1)),
    ("first_height", "TextEdit", None),
    ("growth_rate", "TextEdit", None),
], [
    ("region_id", '"region_id" >= 0'),
    ("edge_id", '"edge_id" IN (1,2,3,4)'),
    ("target_size", '"target_size" IS NULL OR "target_size" > 0'),
    ("n_layers", '"n_layers" >= 0'),
    ("first_height", '"first_height" IS NULL OR "first_height" > 0'),
    ("growth_rate", '"growth_rate" IS NULL OR "growth_rate" > 0'),
], [
    (0, "Region ID"),
    (1, "Edge ID"),
    (2, "Target Size"),
    (3, "Number of Layers"),
    (4, "First Cell Height"),
    (5, "Growth Rate"),
], None))

# 6. swe2d_manning_zones
LAYERS.append(("swe2d_manning_zones", "Polygon", [
    ("zone_id", "Range", _range_config(1, 2147483647, 1)),
    ("n_mann", "TextEdit", None),
    ("priority", "Range", _range_config(0, 100, 1)),
], [
    ("n_mann", '"n_mann" >= 0'),
    ("priority", '"priority" >= 0'),
], [
    (0, "Zone ID"),
    (1, "Manning's n"),
    (2, "Priority"),
], [
    ("n_mann", "0.035", "0"),
]))

# 7. swe2d_bc_lines
LAYERS.append(("swe2d_bc_lines", "LineString", [
    ("bc_type", "ValueMap", _value_map_config(BC_TYPE_ITEMS)),
    ("bc_value", "TextEdit", None),
    ("priority", "Range", _range_config(0, 100, 1)),
    ("hydrograph", "TextEdit", None),
    ("hydrograph_id", "TextEdit", None),
    ("hydrograph_layer", "TextEdit", None),
], [
    ("bc_type", '"bc_type" IN (1,2,3,4,5,6,7,102,103)'),
    ("priority", '"priority" >= 0'),
], [
    (0, "BC Type"),
    (1, "BC Value"),
    (2, "Priority"),
    (3, "Hydrograph"),
    (4, "Hydrograph ID"),
    (5, "Hydrograph Layer"),
], None))

# 8. swe2d_sample_lines
LAYERS.append(("swe2d_sample_lines", "LineString", [
    ("line_id", "Range", _range_config(1, 2147483647, 1)),
    ("name", "TextEdit", None),
    ("enabled", "ValueMap", _value_map_config([
        ("Yes", 1),
        ("No", 0),
    ])),
    ("priority", "Range", _range_config(0, 100, 1)),
], [
    ("line_id", '"line_id" IS NULL OR "line_id" >= 0'),
    ("enabled", '"enabled" IS NULL OR "enabled" IN (0,1)'),
    ("priority", '"priority" IS NULL OR "priority" >= 0'),
], [
    (0, "Line ID"),
    (1, "Name"),
    (2, "Enabled"),
    (3, "Priority"),
], None))

# 9. swe2d_rain_gages
LAYERS.append(("swe2d_rain_gages", "Point", [
    ("gage_id", "TextEdit", None),
    ("name", "TextEdit", None),
    ("hyetograph_id", "TextEdit", None),
    ("units", "ValueMap", _value_map_config(RAIN_GAGE_UNITS_ITEMS)),
    ("priority", "Range", _range_config(0, 100, 1)),
], [
    ("gage_id", 'length(trim("gage_id")) > 0'),
    ("hyetograph_id", 'length(trim("hyetograph_id")) > 0'),
    ("units", '"units" IS NULL OR "units" IN (\'mm/hr\',\'in/hr\',\'mm\',\'in\')'),
], [
    (0, "Gage ID"),
    (1, "Name"),
    (2, "Hyetograph ID"),
    (3, "Units"),
    (4, "Priority"),
], None))

# 10. swe2d_storm_areas
LAYERS.append(("swe2d_storm_areas", "Polygon", [
    ("storm_id", "Range", _range_config(1, 2147483647, 1)),
    ("name", "TextEdit", None),
    ("priority", "Range", _range_config(0, 100, 1)),
], None, [
    (0, "Storm ID"),
    (1, "Name"),
    (2, "Priority"),
], None))

# 11. swe2d_cn_zones
LAYERS.append(("swe2d_cn_zones", "Polygon", [
    ("zone_id", "Range", _range_config(1, 2147483647, 1)),
    ("cn", "Range", _range_config(1, 100, 1)),
    ("priority", "Range", _range_config(0, 100, 1)),
], [
    ("cn", '"cn" >= 1 AND "cn" <= 100'),
    ("priority", '"priority" >= 0'),
], [
    (0, "Zone ID"),
    (1, "Curve Number"),
    (2, "Priority"),
], None))

# 12. swe2d_hyetographs (table)
LAYERS.append(("swe2d_hyetographs", None, [
    ("hyetograph_id", "TextEdit", None),
    ("Time", "TextEdit", None),
    ("Value", "TextEdit", None),
    ("value_type", "ValueMap", _value_map_config(HYETOGRAPH_VALUE_TYPE_ITEMS)),
    ("units", "ValueMap", _value_map_config(RAIN_GAGE_UNITS_ITEMS)),
    ("description", "TextEdit", None),
], [
    ("hyetograph_id", 'length(trim("hyetograph_id")) > 0'),
    ("Time", 'length(trim("Time")) > 0'),
    ("Value", '"Value" >= 0'),
    ("value_type", '"value_type" IS NULL OR "value_type" IN (\'intensity\',\'incremental\',\'cumulative\')'),
    ("units", '"units" IS NULL OR "units" IN (\'mm/hr\',\'in/hr\',\'mm\',\'in\')'),
], [
    (0, "Hyetograph ID"),
    (1, "Time"),
    (2, "Value"),
    (3, "Value Type"),
    (4, "Units"),
    (5, "Description"),
], None))

# 13. swe2d_hydrographs (table)
LAYERS.append(("swe2d_hydrographs", None, [
    ("hydrograph_id", "TextEdit", None),
    ("bc_type", "Range", _range_config(1, 103, 1)),
    ("Time", "TextEdit", None),
    ("Value", "TextEdit", None),
    ("description", "TextEdit", None),
], None, [
    (0, "Hydrograph ID"),
    (1, "BC Type"),
    (2, "Time"),
    (3, "Value"),
    (4, "Description"),
], None))

# 14. swe2d_drainage_nodes
LAYERS.append(("swe2d_drainage_nodes", "Point", [
    ("node_id", "TextEdit", None),
    ("invert_elev", "TextEdit", None),
    ("max_depth", "TextEdit", None),
    ("rim_elev", "TextEdit", None),
    ("crest_elev", "TextEdit", None),
    ("node_type", "ValueMap", _value_map_config(DRAIN_NODE_TYPE_ITEMS)),
    ("surface_area", "TextEdit", None),
    ("outfall_area", "TextEdit", None),
    ("zero_storage", "ValueMap", _value_map_config([
        ("No", 0),
        ("Yes", 1),
    ])),
], [
    ("node_id", 'length(trim("node_id")) > 0'),
    ("node_type", '"node_type" IN (\'junction\',\'outfall\',\'storage\',\'inlet\',\'pipe_end\')'),
    ("max_depth", '"max_depth" IS NULL OR "max_depth" > 0'),
    ("rim_elev", '"rim_elev" IS NULL OR "rim_elev" >= "invert_elev"'),
    ("crest_elev", '"crest_elev" IS NULL OR "crest_elev" >= "invert_elev"'),
    ("surface_area", '"surface_area" IS NULL OR "surface_area" > 0'),
    ("outfall_area", '"outfall_area" IS NULL OR "outfall_area" > 0'),
    ("zero_storage", '"zero_storage" IS NULL OR "zero_storage" IN (0,1)'),
], [
    (0, "Node ID"),
    (1, "Invert Elevation"),
    (2, "Max Depth"),
    (3, "Rim Elevation"),
    (4, "Crest Elevation"),
    (5, "Node Type"),
    (6, "Surface Area"),
    (7, "Outfall Area"),
    (8, "Zero Storage"),
], None))

# 15. swe2d_drainage_links
LAYERS.append(("swe2d_drainage_links", "LineString", [
    ("link_id", "TextEdit", None),
    ("from_node", "TextEdit", None),
    ("to_node", "TextEdit", None),
    ("link_type", "ValueMap", _value_map_config(DRAIN_LINK_TYPE_ITEMS)),
    ("link_shape", "ValueMap", _value_map_config(DRAIN_LINK_SHAPE_ITEMS)),
    ("length", "TextEdit", None),
    ("roughness_n", "TextEdit", None),
    ("diameter", "TextEdit", None),
    ("span", "TextEdit", None),
    ("rise", "TextEdit", None),
    ("area_m2", "TextEdit", None),
    ("equiv_diameter_m", "TextEdit", None),
    ("max_flow", "TextEdit", None),
    ("cd", "TextEdit", None),
    ("culvert_shape", "TextEdit", None),
    ("culvert_code", "Range", _range_config(0, 57, 1)),
    ("culvert_rise", "TextEdit", None),
    ("culvert_span", "TextEdit", None),
    ("culvert_area_m2", "TextEdit", None),
    ("culvert_barrels", "Range", _range_config(1, 10, 1)),
    ("culvert_slope", "TextEdit", None),
    ("inlet_invert_elev", "TextEdit", None),
    ("outlet_invert_elev", "TextEdit", None),
    ("entrance_loss_k", "TextEdit", None),
    ("exit_loss_k", "TextEdit", None),
    ("inlet_loss_k", "TextEdit", None),
    ("outlet_loss_k", "TextEdit", None),
], [
    ("link_id", 'length(trim("link_id")) > 0'),
    ("from_node", 'length(trim("from_node")) > 0'),
    ("to_node", 'length(trim("to_node")) > 0'),
    ("link_type", '"link_type" IN (\'conduit\',\'lateral_simple\',\'pump\',\'weir\',\'orifice\',\'culvert\')'),
    ("link_shape", '"link_shape" IS NULL OR "link_shape" IN (\'circular\',\'box\',\'pipe_arch\',\'custom\')'),
    ("length", '"length" IS NULL OR "length" > 0'),
    ("roughness_n", '"roughness_n" IS NULL OR "roughness_n" > 0'),
    ("diameter", '"diameter" IS NULL OR "diameter" > 0'),
    ("span", '"span" IS NULL OR "span" > 0'),
    ("rise", '"rise" IS NULL OR "rise" > 0'),
    ("area_m2", '"area_m2" IS NULL OR "area_m2" > 0'),
], [
    (0, "Link ID"),
    (1, "From Node"),
    (2, "To Node"),
    (3, "Link Type"),
    (4, "Link Shape"),
    (5, "Length"),
    (6, "Roughness n"),
    (7, "Diameter"),
    (8, "Span"),
    (9, "Rise"),
    (10, "Area (m²)"),
    (11, "Equivalent Diameter"),
    (12, "Max Flow"),
    (13, "Discharge Coefficient"),
    (14, "Culvert Shape"),
    (15, "Culvert Code"),
    (16, "Culvert Rise"),
    (17, "Culvert Span"),
    (18, "Culvert Area"),
    (19, "Culvert Barrels"),
    (20, "Culvert Slope"),
    (21, "Inlet Invert Elev."),
    (22, "Outlet Invert Elev."),
    (23, "Entrance Loss K"),
    (24, "Exit Loss K"),
    (25, "Inlet Loss K"),
    (26, "Outlet Loss K"),
], None))

# 16. swe2d_drainage_inlets (table)
LAYERS.append(("swe2d_drainage_inlets", None, [
    ("inlet_type_id", "TextEdit", None),
    ("name", "TextEdit", None),
    ("weir_length", "TextEdit", None),
    ("orifice_area", "TextEdit", None),
    ("coeff_weir", "TextEdit", None),
    ("coeff_orifice", "TextEdit", None),
    ("max_capture", "TextEdit", None),
    ("description", "TextEdit", None),
], [
    ("inlet_type_id", 'length(trim("inlet_type_id")) > 0'),
    ("weir_length", '"weir_length" IS NULL OR "weir_length" > 0'),
    ("orifice_area", '"orifice_area" IS NULL OR "orifice_area" > 0'),
    ("coeff_weir", '"coeff_weir" IS NULL OR "coeff_weir" > 0'),
    ("coeff_orifice", '"coeff_orifice" IS NULL OR "coeff_orifice" > 0'),
    ("max_capture", '"max_capture" IS NULL OR "max_capture" > 0'),
], [
    (0, "Inlet Type ID"),
    (1, "Name"),
    (2, "Weir Length"),
    (3, "Orifice Area"),
    (4, "Weir Coefficient"),
    (5, "Orifice Coefficient"),
    (6, "Max Capture"),
    (7, "Description"),
], None))

# 17. swe2d_drainage_node_inlets (table)
LAYERS.append(("swe2d_drainage_node_inlets", None, [
    ("node_id", "TextEdit", None),
    ("inlet_type_id", "TextEdit", None),
    ("inlet_count", "Range", _range_config(0, 1000, 1)),
    ("crest_offset", "TextEdit", None),
    ("description", "TextEdit", None),
], [
    ("node_id", 'length(trim("node_id")) > 0'),
    ("inlet_type_id", 'length(trim("inlet_type_id")) > 0'),
    ("inlet_count", '"inlet_count" IS NULL OR "inlet_count" > 0'),
], [
    (0, "Node ID"),
    (1, "Inlet Type ID"),
    (2, "Inlet Count"),
    (3, "Crest Offset"),
    (4, "Description"),
], None))

# 18. swe2d_structures
LAYERS.append(("swe2d_structures", "LineString", [
    ("structure_id", "TextEdit", None),
    ("structure_type", "ValueMap", _value_map_config(STRUCTURE_TYPE_ITEMS)),
    ("crest_elev", "TextEdit", None),
    ("enabled", "ValueMap", _value_map_config([
        ("Yes", 1),
        ("No", 0),
    ])),
    ("width", "TextEdit", None),
    ("height", "TextEdit", None),
    ("diameter", "TextEdit", None),
    ("culvert_shape", "TextEdit", None),
    ("culvert_code", "Range", _range_config(0, 57, 1)),
    ("culvert_rise", "TextEdit", None),
    ("culvert_span", "TextEdit", None),
    ("culvert_area_m2", "TextEdit", None),
    ("culvert_barrels", "Range", _range_config(1, 10, 1)),
    ("culvert_slope", "TextEdit", None),
    ("inlet_invert_elev", "TextEdit", None),
    ("outlet_invert_elev", "TextEdit", None),
    ("entrance_loss_k", "TextEdit", None),
    ("exit_loss_k", "TextEdit", None),
    ("embankment_enabled", "ValueMap", _value_map_config([
        ("No", 0),
        ("Yes", 1),
    ])),
    ("embankment_crest_elev", "TextEdit", None),
    ("embankment_overflow_width", "TextEdit", None),
    ("embankment_weir_coeff", "TextEdit", None),
    ("length", "TextEdit", None),
    ("roughness_n", "TextEdit", None),
    ("coeff", "TextEdit", None),
    ("cd", "TextEdit", None),
    ("opening", "TextEdit", None),
    ("q_pump", "TextEdit", None),
    ("max_flow", "TextEdit", None),
    ("inlet_loss_k", "TextEdit", None),
    ("outlet_loss_k", "TextEdit", None),
    ("stacked_enabled", "ValueMap", _value_map_config([
        ("No", 0),
        ("Yes", 1),
    ])),
    ("use_redistribution", "ValueMap", _value_map_config([
        ("No", 0),
        ("Yes", 1),
    ])),
    ("influence_width", "TextEdit", None),
    ("upstream_buffer", "TextEdit", None),
    ("downstream_buffer", "TextEdit", None),
    ("deck_soffit_elev", "TextEdit", None),
    ("deck_top_elev", "TextEdit", None),
    ("model_top_elev", "TextEdit", None),
    ("under_layers", "Range", _range_config(0, 10, 1)),
    ("over_layers", "Range", _range_config(0, 10, 1)),
    ("pier_count", "Range", _range_config(0, 20, 1)),
    ("pier_width", "TextEdit", None),
], [
    ("structure_id", 'length(trim("structure_id")) > 0'),
    ("structure_type", '"structure_type" IN (1,2,3,4,5)'),
    ("enabled", '"enabled" IS NULL OR "enabled" IN (0,1)'),
], [
    (0, "Structure ID"),
    (1, "Structure Type"),
    (2, "Crest Elevation"),
    (3, "Enabled"),
    (4, "Width"),
    (5, "Height"),
    (6, "Diameter"),
    (7, "Culvert Shape"),
    (8, "Culvert Code"),
    (9, "Culvert Rise"),
    (10, "Culvert Span"),
    (11, "Culvert Area"),
    (12, "Barrels"),
    (13, "Culvert Slope"),
    (14, "Inlet Invert Elev."),
    (15, "Outlet Invert Elev."),
    (16, "Entrance Loss K"),
    (17, "Exit Loss K"),
    (18, "Embankment Enabled"),
    (19, "Embankment Crest Elev."),
    (20, "Overflow Width"),
    (21, "Weir Coefficient"),
    (22, "Length"),
    (23, "Roughness n"),
    (24, "Coefficient"),
    (25, "Discharge Coeff."),
    (26, "Opening"),
    (27, "Pump Flow"),
    (28, "Max Flow"),
    (29, "Inlet Loss K"),
    (30, "Outlet Loss K"),
    (31, "Stacked Enabled"),
    (32, "Use Redistribution"),
    (33, "Influence Width"),
    (34, "Upstream Buffer"),
    (35, "Downstream Buffer"),
    (36, "Deck Soffit Elev."),
    (37, "Deck Top Elev."),
    (38, "Model Top Elev."),
    (39, "Under Layers"),
    (40, "Over Layers"),
    (41, "Pier Count"),
    (42, "Pier Width"),
], [
    ("structure_type", "1", "0"),
    ("crest_elev", "0.0", "0"),
    ("enabled", "1", "0"),
    ("roughness_n", "0.035", "0"),
    ("length", "30.0", "0"),
    ("entrance_loss_k", "0.5", "0"),
    ("exit_loss_k", "1.0", "0"),
    ("culvert_barrels", "1", "0"),
]))


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    """Generate QML files for all 18 layers."""
    os.makedirs(QML_DIR, exist_ok=True)
    for name, geom_type, fields, constraints, aliases, defaults in LAYERS:
        qml = build_qml(geom_type, fields, constraints, aliases, defaults)
        path = os.path.join(QML_DIR, f"{name}.qml")
        with open(path, "w") as f:
            f.write(qml)
        print(f"  ✓ {name}.qml  ({len(qml)} bytes)")


if __name__ == "__main__":
    main()
