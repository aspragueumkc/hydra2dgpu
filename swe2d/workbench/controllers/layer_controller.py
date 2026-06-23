"""Layer management controller — refresh/autopopulate combos via View protocol.

Calls ``view.populate_layer_combo()`` etc. — never touches widgets directly.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from qgis.core import QgsProject

from swe2d.workbench.controllers.protocols_controller import LayerView


class LayerController:
    """Manages layer combo boxes: refresh, autopopulate, and selection.

    Receives a ``LayerView`` protocol.
    """

    def __init__(self, view: LayerView):
        self._view = view

    def refresh_layer_combos(self) -> None:
        """Refresh all layer combos from the current QGIS project."""
        v = self._view
        layers = list(QgsProject.instance().mapLayers().values())
        v.populate_layer_combo("nodes_layer_combo", layers, "point")
        v.populate_layer_combo("cells_layer_combo", layers, "polygon")
        v.populate_layer_combo("terrain_layer_combo", layers, "raster")
        v.populate_layer_combo("manning_layer_combo", layers, "polygon")
        v.populate_layer_combo("cn_layer_combo", layers, "polygon")
        v.populate_layer_combo("rain_gage_layer_combo", layers, "point")
        v.populate_layer_combo("hyetograph_layer_combo", layers)
        v.populate_layer_combo("sample_lines_layer_combo", layers, "line")
        v.populate_layer_combo("drain_nodes_layer_combo", layers, "point")
        v.populate_layer_combo("drain_links_layer_combo", layers, "line")
        v.populate_layer_combo("drain_inlets_layer_combo", layers)
        v.populate_layer_combo("drain_node_inlets_layer_combo", layers)
        v.populate_layer_combo("structures_layer_combo", layers, "line")
        v.populate_layer_combo("bc_lines_layer_combo", layers, "line")
        v.populate_layer_combo("layer_group_combo", layers)
        self._refresh_topo_layer_combos(layers)

    def _refresh_topo_layer_combos(self, layers: List) -> None:
        """Refresh topology-specific layer combos from the given layer list."""
        v = self._view
        for attr in ("topo_nodes_combo", "topo_arcs_combo", "topo_regions_combo",
                     "topo_constraints_combo", "topo_quad_edges_combo"):
            combo = v.get_topo_combo(attr)
            if combo is not None:
                v.populate_layer_combo(attr, layers)

    def autopopulate_layer_combos_from_group(self) -> None:
        """Match layers from a QGIS group to known combo names."""
        v = self._view
        group_name = v.get_combo_current_text("layer_group_combo")
        if not group_name:
            return
        root = QgsProject.instance().layerTreeRoot()
        group = root.findGroup(group_name)
        if group is None:
            return
        group_layers = {str(lyr.name()): lyr for lyr in group.layerOrder()}

        _NAME_MAP: Dict[str, List[str]] = {
            "nodes_layer_combo": ["nodes", "topo_nodes", "mesh_nodes"],
            "cells_layer_combo": ["cells", "topo_cells"],
            "terrain_layer_combo": ["terrain", "dem", "dtm", "elevation"],
            "manning_layer_combo": ["manning", "mannings_n", "roughness"],
            "cn_layer_combo": ["cn", "curve_number"],
            "rain_gage_layer_combo": ["rain_gage", "rainfall_gage"],
            "hyetograph_layer_combo": ["hyetograph", "rainfall_data"],
            "sample_lines_layer_combo": ["sample_line", "cross_section"],
            "drain_nodes_layer_combo": ["drain_node", "drainage_node"],
            "drain_links_layer_combo": ["drain_link", "drainage_link"],
            "drain_inlets_layer_combo": ["drain_inlet", "inlet_type"],
            "drain_node_inlets_layer_combo": ["drain_node_inlet"],
            "structures_layer_combo": ["structure", "hydraulic_structure"],
            "bc_lines_layer_combo": ["bc_line", "boundary_line", "bc_boundary"],
        }
        for combo_attr, keywords in _NAME_MAP.items():
            for kw in keywords:
                matched = [name for name in group_layers if kw in name.lower()]
                if matched:
                    v.select_layer_in_combo(combo_attr, group_layers[matched[0]].id())
                    break
