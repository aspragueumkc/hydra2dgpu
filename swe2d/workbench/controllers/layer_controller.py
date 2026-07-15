"""Layer management controller — refresh/autopopulate combos via View protocol.

Calls ``view.populate_layer_combo()`` etc. — never touches widgets directly.
"""

from __future__ import annotations

from typing import List

from qgis.core import QgsProject

from swe2d.workbench.controllers.protocols_controller import LayerView


class LayerController:
    """Manages layer combo boxes: refresh and selection.

    Receives a ``LayerView`` protocol.
    """

    def __init__(self, view: LayerView):
        self._view = view

    def refresh_layer_combos(self) -> None:
        """Refresh all layer combos from the current QGIS project."""
        v = self._view
        layers = list(QgsProject.instance().mapLayers().values())
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
        v.populate_layer_combo("internal_flow_layer_combo", layers, "polygon")
        v.populate_layer_combo("storm_area_layer_combo", layers, "polygon")
        self._refresh_topo_layer_combos(layers)

    def _refresh_topo_layer_combos(self, layers: List) -> None:
        """Refresh topology-specific layer combos from the given layer list."""
        v = self._view
        for attr in ("topo_nodes_combo", "topo_arcs_combo", "topo_regions_combo",
                     "topo_constraints_combo", "topo_quad_edges_combo"):
            combo = v.get_topo_combo(attr)
            if combo is not None:
                v.populate_layer_combo(attr, layers)
        # Elevation combo accepts raster + PointZ layers only.
        v.populate_elevation_combo(layers)

