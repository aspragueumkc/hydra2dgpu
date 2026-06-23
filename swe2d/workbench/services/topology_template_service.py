"""Topology template layer definitions and creation.

Defines the field schemas for all 14 SWE2D topology template layers used
during mesh generation.  Zero Qt — no PyQt5 imports.
"""
from __future__ import annotations


def _vector_layer(uri: str, name: str) -> "QgsVectorLayer":
    """vector layer."""
    from qgis.core import QgsVectorLayer
    return QgsVectorLayer(uri, name, "memory")


def create_topology_template_layers(crs_auth: str = "EPSG:4326"):
    """Create 14 in-memory QgsVectorLayers matching the SWE2D topology template.

    Returns list of (layer_name, QgsVectorLayer) tuples.
    """
    nodes = _vector_layer(
        f"Point?crs={crs_auth}&field=node_id:integer",
        "SWE2D_Topo_Nodes",
    )
    arcs = _vector_layer(
        f"LineString?crs={crs_auth}&field=arc_id:integer&field=node0:integer&field=node1:integer"
        "&field=region_id:integer&field=arc_role:string(24)"
        "&field=use_global_arc_ctrl:integer&field=arc_mode_override:string(24)"
        "&field=arc_soft_size_override:double&field=arc_soft_dist_override:double",
        "SWE2D_Topo_Arcs",
    )
    regions = _vector_layer(
        f"Polygon?crs={crs_auth}&field=region_id:integer&field=target_size:double"
        "&field=cell_type:string(32)&field=channel_generator_type:string(32)"
        "&field=edge_len_1:double&field=edge_len_2:double"
        "&field=edge_len_3:double&field=edge_len_4:double",
        "SWE2D_Topo_Regions",
    )
    constraints = _vector_layer(
        f"Polygon?crs={crs_auth}&field=constraint_id:integer&field=target_size:double"
        "&field=cell_type:string(32)&field=edge_len_1:double&field=edge_len_2:double"
        "&field=edge_len_3:double&field=edge_len_4:double",
        "SWE2D_Topo_Constraints",
    )
    quad_edges = _vector_layer(
        f"LineString?crs={crs_auth}&field=region_id:integer&field=edge_id:integer"
        "&field=target_size:double&field=n_layers:integer"
        "&field=first_height:double&field=growth_rate:double",
        "SWE2D_Topo_Quad_Edges",
    )
    manning = _vector_layer(
        f"Polygon?crs={crs_auth}&field=zone_id:integer&field=n_mann:double&field=priority:integer",
        "SWE2D_Manning_Zones",
    )
    bc_lines = _vector_layer(
        f"LineString?crs={crs_auth}&field=bc_type:integer&field=bc_value:double"
        "&field=priority:integer&field=hydrograph:string(1024)"
        "&field=hydrograph_id:string(64)&field=hydrograph_layer:string(128)",
        "SWE2D_BC_Lines",
    )
    sample_lines = _vector_layer(
        f"LineString?crs={crs_auth}&field=line_id:integer&field=name:string(128)"
        "&field=enabled:integer&field=priority:integer",
        "SWE2D_Sample_Lines",
    )
    drainage_nodes = _vector_layer(
        f"Point?crs={crs_auth}&field=node_id:string(64)&field=invert_elev:double"
        "&field=max_depth:double&field=rim_elev:double&field=crest_elev:double"
        "&field=node_type:string(32)&field=surface_area:double&field=outfall_area:double"
        "&field=zero_storage:integer",
        "SWE2D_Drainage_Nodes",
    )
    drainage_links = _vector_layer(
        f"LineString?crs={crs_auth}&field=link_id:string(64)&field=from_node:string(64)"
        "&field=to_node:string(64)&field=link_type:string(32)&field=link_shape:string(32)"
        "&field=length:double&field=roughness_n:double&field=diameter:double"
        "&field=span:double&field=rise:double&field=area_m2:double"
        "&field=equiv_diameter_m:double&field=max_flow:double&field=cd:double"
        "&field=culvert_shape:string(32)&field=culvert_code:integer"
        "&field=culvert_rise:double&field=culvert_span:double&field=culvert_area_m2:double"
        "&field=culvert_barrels:integer&field=culvert_slope:double"
        "&field=inlet_invert_elev:double&field=outlet_invert_elev:double"
        "&field=entrance_loss_k:double&field=exit_loss_k:double"
        "&field=inlet_loss_k:double&field=outlet_loss_k:double",
        "SWE2D_Drainage_Links",
    )
    drainage_inlets = _vector_layer(
        "None?field=inlet_type_id:string(64)&field=name:string(128)"
        "&field=weir_length:double&field=orifice_area:double"
        "&field=coeff_weir:double&field=coeff_orifice:double"
        "&field=max_capture:double&field=description:string(256)",
        "SWE2D_Drainage_Inlets",
    )
    drainage_node_inlets = _vector_layer(
        "None?field=node_id:string(64)&field=inlet_type_id:string(64)"
        "&field=inlet_count:double&field=crest_offset:double"
        "&field=description:string(256)",
        "SWE2D_Drainage_Node_Inlets",
    )
    structures = _vector_layer(
        f"LineString?crs={crs_auth}&field=structure_id:string(64)"
        "&field=structure_type:integer&field=crest_elev:double&field=enabled:integer"
        "&field=width:double&field=height:double&field=diameter:double"
        "&field=culvert_shape:string(32)&field=culvert_code:integer"
        "&field=culvert_rise:double&field=culvert_span:double"
        "&field=culvert_area_m2:double&field=culvert_barrels:integer"
        "&field=culvert_slope:double&field=inlet_invert_elev:double"
        "&field=outlet_invert_elev:double&field=entrance_loss_k:double"
        "&field=exit_loss_k:double&field=embankment_enabled:integer"
        "&field=embankment_crest_elev:double&field=embankment_overflow_width:double"
        "&field=embankment_weir_coeff:double&field=length:double"
        "&field=roughness_n:double&field=coeff:double&field=cd:double"
        "&field=opening:double&field=q_pump:double&field=max_flow:double"
        "&field=use_redistribution:integer&field=influence_width:double",
        "SWE2D_Structures",
    )
    hydro_tbl = _vector_layer(
        "None?field=hydrograph_id:string(64)&field=bc_type:integer"
        "&field=Time:string(32)&field=Value:double&field=description:string(256)",
        "SWE2D_Hydrographs",
    )

    return [
        ("SWE2D_Topo_Nodes", nodes),
        ("SWE2D_Topo_Arcs", arcs),
        ("SWE2D_Topo_Regions", regions),
        ("SWE2D_Topo_Constraints", constraints),
        ("SWE2D_Topo_Quad_Edges", quad_edges),
        ("SWE2D_Manning_Zones", manning),
        ("SWE2D_BC_Lines", bc_lines),
        ("SWE2D_Sample_Lines", sample_lines),
        ("SWE2D_Drainage_Nodes", drainage_nodes),
        ("SWE2D_Drainage_Links", drainage_links),
        ("SWE2D_Drainage_Inlets", drainage_inlets),
        ("SWE2D_Drainage_Node_Inlets", drainage_node_inlets),
        ("SWE2D_Structures", structures),
        ("SWE2D_Hydrographs", hydro_tbl),
    ]
