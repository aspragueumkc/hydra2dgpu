"""Layer management controller — refresh/autopopulate combos via View protocol.

Calls ``view.populate_layer_combo()`` etc. — never touches widgets directly.
"""
from __future__ import annotations

from typing import Dict, List

from qgis.core import QgsProject

from swe2d.workbench.controllers.protocols_controller import LayerView


# combo_attr -> swe2d_* layer name. Used by ``auto_select_model_layers()`` to
# default each layer combo to the matching layer from a freshly created (or
# freshly loaded) 2D model GeoPackage.
_DEFAULT_MODEL_LAYER_BY_COMBO: Dict[str, str] = {
    "manning_layer_combo": "swe2d_manning_zones",
    "cn_layer_combo": "swe2d_cn_zones",
    "rain_gage_layer_combo": "swe2d_rain_gages",
    "hyetograph_layer_combo": "swe2d_hyetographs",
    "sample_lines_layer_combo": "swe2d_sample_lines",
    "drain_nodes_layer_combo": "swe2d_drainage_nodes",
    "drain_links_layer_combo": "swe2d_drainage_links",
    "drain_inlets_layer_combo": "swe2d_drainage_inlets",
    "drain_node_inlets_layer_combo": "swe2d_drainage_node_inlets",
    "structures_layer_combo": "swe2d_structures",
    "bc_lines_layer_combo": "swe2d_bc_lines",
    "internal_flow_layer_combo": "swe2d_internal_flow_sources",
    "storm_area_layer_combo": "swe2d_storm_areas",
    "topo_nodes_combo": "swe2d_topo_nodes",
    "topo_arcs_combo": "swe2d_topo_arcs",
    "topo_regions_combo": "swe2d_topo_regions",
    "topo_constraints_combo": "swe2d_topo_constraints",
    "topo_quad_edges_combo": "swe2d_topo_quad_edges",
}


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

    def auto_select_model_layers(self) -> None:
        """Default each layer combo to its matching ``swe2d_*`` layer.

        For each combo in ``_DEFAULT_MODEL_LAYER_BY_COMBO``, find the layer
        with the matching name in the current project and set the combo to
        it. Safe to call when a combo's expected layer isn't in the project
        (no-op for that combo). Call after ``refresh_layer_combos()``.
        """
        v = self._view
        layers_by_name = {
            lyr.name(): lyr for lyr in QgsProject.instance().mapLayers().values()
        }
        for combo_attr, expected_name in _DEFAULT_MODEL_LAYER_BY_COMBO.items():
            layer = layers_by_name.get(expected_name)
            if layer is None:
                continue
            v.select_layer_in_combo(combo_attr, layer.id())

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