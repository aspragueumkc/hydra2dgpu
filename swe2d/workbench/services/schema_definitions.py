"""Single source of truth for all 18 SWE2D model layer schemas.

Centralises field definitions, geometry types, and geometry column names
that were previously duplicated across ``mesh_controller.py``,
``topology_template_service.py``, ``model_gpkg_loader_service.py``,
and ``gpkg_layer_styles_service.py``.

Usage::

    from swe2d.workbench.services.schema_definitions import (
        LAYER_SCHEMAS,
        create_memory_layer,
        get_layer_names,
        get_geom_column,
    )

    lyr = create_memory_layer("swe2d_topo_regions", crs_auth="EPSG:4326")
"""
from __future__ import annotations

# ── Field type helpers ─────────────────────────────────────────────────────
# QGIS memory-layer URI type strings
_INT = "integer"
_DBL = "double"
_STR = lambda n: f"string({n})"  # noqa: E731


# ── Layer schema definitions ──────────────────────────────────────────────
# Each entry: {
#     "geom": geometry type string or None for tables,
#     "fields": [(field_name, qgis_type_string), ...],
#     "geom_col": geometry column name or ""
# }

LAYER_SCHEMAS: dict[str, dict] = {
    # ── Topology layers ───────────────────────────────────────────────────
    "swe2d_topo_nodes": {
        "geom": "Point",
        "geom_col": "geom",
        "fields": [
            ("node_id", _INT),
        ],
    },
    "swe2d_topo_arcs": {
        "geom": "LineString",
        "geom_col": "geom",
        "fields": [
            ("arc_id", _INT),
            ("node0", _INT),
            ("node1", _INT),
            ("region_id", _INT),
            ("arc_role", _STR(24)),
            ("use_global_arc_ctrl", _INT),
            ("arc_mode_override", _STR(24)),
            ("arc_soft_size_override", _DBL),
            ("arc_soft_dist_override", _DBL),
        ],
    },
    "swe2d_topo_regions": {
        "geom": "Polygon",
        "geom_col": "geom",
        "fields": [
            ("region_id", _INT),
            ("target_size", _DBL),
            ("cell_type", _STR(32)),
            ("channel_generator_type", _STR(32)),
            ("edge_len_1", _DBL),
            ("edge_len_2", _DBL),
            ("edge_len_3", _DBL),
            ("edge_len_4", _DBL),
        ],
    },
    "swe2d_topo_constraints": {
        "geom": "Polygon",
        "geom_col": "geom",
        "fields": [
            ("constraint_id", _INT),
            ("target_size", _DBL),
            ("cell_type", _STR(32)),
            ("edge_len_1", _DBL),
            ("edge_len_2", _DBL),
            ("edge_len_3", _DBL),
            ("edge_len_4", _DBL),
        ],
    },
    "swe2d_topo_quad_edges": {
        "geom": "LineString",
        "geom_col": "geom",
        "fields": [
            ("region_id", _INT),
            ("edge_id", _INT),
            ("target_size", _DBL),
            ("n_layers", _INT),
            ("first_height", _DBL),
            ("growth_rate", _DBL),
        ],
    },
    # ── Manning / BC / sample lines ──────────────────────────────────────
    "swe2d_manning_zones": {
        "geom": "Polygon",
        "geom_col": "geom",
        "fields": [
            ("zone_id", _INT),
            ("n_mann", _DBL),
            ("priority", _INT),
        ],
    },
    "swe2d_bc_lines": {
        "geom": "LineString",
        "geom_col": "geom",
        "fields": [
            ("bc_type", _INT),
            ("bc_value", _DBL),
            ("priority", _INT),
            ("hydrograph", _STR(1024)),
            ("hydrograph_id", _STR(64)),
            ("hydrograph_layer", _STR(128)),
        ],
    },
    "swe2d_sample_lines": {
        "geom": "LineString",
        "geom_col": "geom",
        "fields": [
            ("line_id", _INT),
            ("name", _STR(128)),
            ("enabled", _INT),
            ("priority", _INT),
        ],
    },
    # ── Rainfall / hydrology ─────────────────────────────────────────────
    "swe2d_rain_gages": {
        "geom": "Point",
        "geom_col": "geom",
        "fields": [
            ("gage_id", _STR(64)),
            ("name", _STR(128)),
            ("hyetograph_id", _STR(64)),
            ("units", _STR(32)),
            ("priority", _INT),
        ],
    },
    "swe2d_storm_areas": {
        "geom": "Polygon",
        "geom_col": "geom",
        "fields": [
            ("storm_id", _INT),
            ("name", _STR(128)),
            ("priority", _INT),
        ],
    },
    "swe2d_cn_zones": {
        "geom": "Polygon",
        "geom_col": "geom",
        "fields": [
            ("zone_id", _INT),
            ("cn", _DBL),
            ("priority", _INT),
        ],
    },
    "swe2d_hyetographs": {
        "geom": None,
        "geom_col": "",
        "fields": [
            ("hyetograph_id", _STR(64)),
            ("Time", _STR(32)),
            ("Value", _DBL),
            ("value_type", _STR(24)),
            ("units", _STR(24)),
            ("description", _STR(256)),
        ],
    },
    "swe2d_hydrographs": {
        "geom": None,
        "geom_col": "",
        "fields": [
            ("hydrograph_id", _STR(64)),
            ("bc_type", _INT),
            ("Time", _STR(32)),
            ("Value", _DBL),
            ("description", _STR(256)),
        ],
    },
    # ── Drainage network ─────────────────────────────────────────────────
    "swe2d_drainage_nodes": {
        "geom": "Point",
        "geom_col": "geom",
        "fields": [
            ("node_id", _STR(64)),
            ("invert_elev", _DBL),
            ("max_depth", _DBL),
            ("rim_elev", _DBL),
            ("crest_elev", _DBL),
            ("node_type", _STR(32)),
            ("surface_area", _DBL),
            ("outfall_area", _DBL),
            ("zero_storage", _INT),
        ],
    },
    "swe2d_drainage_links": {
        "geom": "LineString",
        "geom_col": "geom",
        "fields": [
            ("link_id", _STR(64)),
            ("from_node", _STR(64)),
            ("to_node", _STR(64)),
            ("link_type", _STR(32)),
            ("link_shape", _STR(32)),
            ("length", _DBL),
            ("roughness_n", _DBL),
            ("diameter", _DBL),
            ("span", _DBL),
            ("rise", _DBL),
            ("area_m2", _DBL),
            ("equiv_diameter_m", _DBL),
            ("max_flow", _DBL),
            ("cd", _DBL),
            ("culvert_shape", _STR(32)),
            ("culvert_code", _INT),
            ("culvert_rise", _DBL),
            ("culvert_span", _DBL),
            ("culvert_area_m2", _DBL),
            ("culvert_barrels", _INT),
            ("culvert_slope", _DBL),
            ("inlet_invert_elev", _DBL),
            ("outlet_invert_elev", _DBL),
            ("entrance_loss_k", _DBL),
            ("exit_loss_k", _DBL),
            ("inlet_loss_k", _DBL),
            ("outlet_loss_k", _DBL),
        ],
    },
    "swe2d_drainage_inlets": {
        "geom": None,
        "geom_col": "",
        "fields": [
            ("inlet_type_id", _STR(64)),
            ("name", _STR(128)),
            ("weir_length", _DBL),
            ("orifice_area", _DBL),
            ("coeff_weir", _DBL),
            ("coeff_orifice", _DBL),
            ("max_capture", _DBL),
            ("description", _STR(256)),
        ],
    },
    "swe2d_drainage_node_inlets": {
        "geom": None,
        "geom_col": "",
        "fields": [
            ("node_id", _STR(64)),
            ("inlet_type_id", _STR(64)),
            ("inlet_count", _DBL),
            ("crest_offset", _DBL),
            ("description", _STR(256)),
        ],
    },
    # ── Structures ───────────────────────────────────────────────────────
    "swe2d_structures": {
        "geom": "LineString",
        "geom_col": "geom",
        "fields": [
            ("structure_id", _STR(64)),
            ("structure_type", _INT),
            ("crest_elev", _DBL),
            ("enabled", _INT),
            ("width", _DBL),
            ("height", _DBL),
            ("diameter", _DBL),
            ("culvert_shape", _STR(32)),
            ("culvert_code", _INT),
            ("culvert_rise", _DBL),
            ("culvert_span", _DBL),
            ("culvert_area_m2", _DBL),
            ("culvert_barrels", _INT),
            ("culvert_slope", _DBL),
            ("inlet_invert_elev", _DBL),
            ("outlet_invert_elev", _DBL),
            ("entrance_loss_k", _DBL),
            ("exit_loss_k", _DBL),
            ("embankment_enabled", _INT),
            ("embankment_crest_elev", _DBL),
            ("embankment_overflow_width", _DBL),
            ("embankment_weir_coeff", _DBL),
            ("length", _DBL),
            ("roughness_n", _DBL),
            ("coeff", _DBL),
            ("cd", _DBL),
            ("opening", _DBL),
            ("q_pump", _DBL),
            ("max_flow", _DBL),
            ("inlet_loss_k", _DBL),
            ("outlet_loss_k", _DBL),
            ("stacked_enabled", _INT),
            ("use_redistribution", _INT),
            ("influence_width", _DBL),
            ("upstream_buffer", _DBL),
            ("downstream_buffer", _DBL),
            ("deck_soffit_elev", _DBL),
            ("deck_top_elev", _DBL),
            ("model_top_elev", _DBL),
            ("under_layers", _INT),
            ("over_layers", _INT),
            ("pier_count", _INT),
            ("pier_width", _DBL),
        ],
    },
}

# ── Display-name mapping (TitleCase) ──────────────────────────────────────
# Maps swe2d_* names to the SWE2D_* display names used by topology templates.
_LAYER_DISPLAY_NAMES: dict[str, str] = {
    "swe2d_topo_nodes": "SWE2D_Topo_Nodes",
    "swe2d_topo_arcs": "SWE2D_Topo_Arcs",
    "swe2d_topo_regions": "SWE2D_Topo_Regions",
    "swe2d_topo_constraints": "SWE2D_Topo_Constraints",
    "swe2d_topo_quad_edges": "SWE2D_Topo_Quad_Edges",
    "swe2d_manning_zones": "SWE2D_Manning_Zones",
    "swe2d_bc_lines": "SWE2D_BC_Lines",
    "swe2d_sample_lines": "SWE2D_Sample_Lines",
    "swe2d_rain_gages": "SWE2D_Rain_Gages",
    "swe2d_storm_areas": "SWE2D_Storm_Areas",
    "swe2d_cn_zones": "SWE2D_CN_Zones",
    "swe2d_hyetographs": "SWE2D_Hyetographs",
    "swe2d_hydrographs": "SWE2D_Hydrographs",
    "swe2d_drainage_nodes": "SWE2D_Drainage_Nodes",
    "swe2d_drainage_links": "SWE2D_Drainage_Links",
    "swe2d_drainage_inlets": "SWE2D_Drainage_Inlets",
    "swe2d_drainage_node_inlets": "SWE2D_Drainage_Node_Inlets",
    "swe2d_structures": "SWE2D_Structures",
}


# ── Public API ────────────────────────────────────────────────────────────

def get_layer_names() -> list[str]:
    """Return the ordered list of all 18 model layer keys."""
    return list(LAYER_SCHEMAS.keys())


def get_display_name(layer_key: str) -> str:
    """Return the TitleCase display name for a ``swe2d_*`` layer key."""
    return _LAYER_DISPLAY_NAMES.get(layer_key, layer_key)


def get_geom_column(layer_key: str) -> str:
    """Return the geometry column name for a layer, or ``""`` for tables."""
    schema = LAYER_SCHEMAS.get(layer_key)
    if schema is None:
        return ""
    return schema.get("geom_col", "")


def build_uri(layer_key: str, crs_auth: str = "EPSG:4326") -> str:
    """Build a QGIS memory-layer URI string for the given layer.

    Example::

        ``"Point?crs=EPSG:4326&field=node_id:integer"``
    """
    schema = LAYER_SCHEMAS.get(layer_key)
    if schema is None:
        raise KeyError(f"Unknown layer: {layer_key}")

    geom = schema["geom"]
    field_parts = []
    for fname, ftype in schema["fields"]:
        field_parts.append(f"&field={fname}:{ftype}")

    if geom is not None:
        return f"{geom}?crs={crs_auth}{''.join(field_parts)}"
    else:
        return f"None?{''.join(field_parts).lstrip('&')}"


def create_memory_layer(layer_key: str, crs_auth: str = "EPSG:4326") -> "QgsVectorLayer":
    """Create an in-memory ``QgsVectorLayer`` from the canonical schema.

    Args:
        layer_key: One of the ``swe2d_*`` keys from ``LAYER_SCHEMAS``.
        crs_auth: CRS authority string, e.g. ``"EPSG:4326"``.

    Returns:
        A ``QgsVectorLayer`` with provider type ``"memory"``.
    """
    from qgis.core import QgsVectorLayer

    uri = build_uri(layer_key, crs_auth)
    display_name = get_display_name(layer_key)
    return QgsVectorLayer(uri, display_name, "memory")
