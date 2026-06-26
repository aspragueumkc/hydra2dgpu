"""Topology template layer definitions and creation.

Delegates to ``schema_definitions.py`` for canonical field schemas.
Zero Qt — no PyQt5 imports.
"""
from __future__ import annotations

from swe2d.workbench.services.schema_definitions import (
    LAYER_SCHEMAS,
    create_memory_layer,
    get_display_name,
)

# The 14 layers created by topology templates — a subset of the full 18.
# rain_gages, storm_areas, cn_zones, hyetographs are excluded because
# they are not needed for mesh generation.
_TOPOLOGY_TEMPLATE_KEYS = [
    "swe2d_topo_nodes",
    "swe2d_topo_arcs",
    "swe2d_topo_regions",
    "swe2d_topo_constraints",
    "swe2d_topo_quad_edges",
    "swe2d_manning_zones",
    "swe2d_bc_lines",
    "swe2d_sample_lines",
    "swe2d_drainage_nodes",
    "swe2d_drainage_links",
    "swe2d_drainage_inlets",
    "swe2d_drainage_node_inlets",
    "swe2d_structures",
    "swe2d_hydrographs",
]


def create_topology_template_layers(crs_auth: str = "EPSG:4326"):
    """Create 14 in-memory QgsVectorLayers from the canonical schema.

    Returns list of (display_name, QgsVectorLayer) tuples.
    """
    layers = []
    for key in _TOPOLOGY_TEMPLATE_KEYS:
        lyr = create_memory_layer(key, crs_auth)
        layers.append((get_display_name(key), lyr))
    return layers
