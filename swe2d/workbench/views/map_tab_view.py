"""Map tab view — owns its own widget references.

QWidget subclass for the Map tab in the Studio workbench. Mirrors the
structure previously created by ``SWE2DWorkbenchStudioDialog._build_map_tab_page_fallback``
plus the four ``_bind_map_tab_*_controls`` binders in studio_dialog.py and
``swe2d.workbench.extracted.results_and_ui_methods``.

Object names and widget types are preserved so that existing
``findChild()`` calls elsewhere in the codebase keep working.
"""
from __future__ import annotations

from qgis.PyQt import QtWidgets


_BC_OPTIONS = [
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


class MapTabView(QtWidgets.QWidget):
    """View for the Map tab.

    Houses four QToolBox pages.  Every widget is created here as a direct
    instance attribute with a stable ``objectName``.

    Data page ("Load Layers") - layer selection combos + group buttons:
        nodes_layer_combo, cells_layer_combo, terrain_layer_combo,
        manning_layer_combo, cn_layer_combo, rain_gage_layer_combo,
        hyetograph_layer_combo, sample_lines_layer_combo,
        drain_nodes_layer_combo, drain_links_layer_combo,
        drain_inlets_layer_combo, drain_node_inlets_layer_combo,
        structures_layer_combo, bc_lines_layer_combo

    Actions page ("Mesh Setup") - mesh I/O: REMOVED. The mesh I/O
    buttons were moved to the Mesh Generation tab as the
    "Import/Export" page (top, before Layer Setup). See
    TopologyTabView._build_import_export_page.

    Utilities page ("Utilities") - helpers:
        open_model_gpkg_explorer_btn, open_run_log_viewer_btn,


    Results page (overlay controls) — added later by the controller;
        see ``swe2d.workbench.views.results_view`` for the overlay helpers.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        """Build the toolbox with Data and Utilities pages (Mesh Setup moved)."""
        root_layout = QtWidgets.QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        toolbox = QtWidgets.QToolBox()
        toolbox.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding
        )

        self._build_data_page(toolbox)
        self._build_tools_page(toolbox)

        root_layout.addWidget(toolbox)

    def _build_data_page(self, toolbox: QtWidgets.QToolBox) -> None:
        """Build the Load Layers page with layer selection combos."""
        page = QtWidgets.QWidget()
        page.setObjectName("map_data_page")
        data_layout = QtWidgets.QGridLayout(page)
        data_layout.setObjectName("map_data_layout")
        data_layout.setContentsMargins(0, 0, 0, 0)

        for attr in [
            "nodes_layer_combo",
            "cells_layer_combo",
            "terrain_layer_combo",
            "manning_layer_combo",
            "cn_layer_combo",
            "rain_gage_layer_combo",
            "hyetograph_layer_combo",
            "sample_lines_layer_combo",
            "drain_nodes_layer_combo",
            "drain_links_layer_combo",
            "drain_inlets_layer_combo",
            "drain_node_inlets_layer_combo",
            "structures_layer_combo",
            "bc_lines_layer_combo",
        ]:
            widget = QtWidgets.QComboBox()
            widget.setObjectName(attr)
            setattr(self, attr, widget)

        for row, label, attr in [
            (0, "Nodes layer:", "nodes_layer_combo"),
            (1, "Cells layer:", "cells_layer_combo"),
            (2, "Terrain raster:", "terrain_layer_combo"),
            (3, "Manning polygons:", "manning_layer_combo"),
            (4, "CN polygons:", "cn_layer_combo"),
            (5, "Rain gages (points):", "rain_gage_layer_combo"),
            (6, "Rain hyetographs (table):", "hyetograph_layer_combo"),
            (7, "Sample lines layer:", "sample_lines_layer_combo"),
            (8, "Drainage nodes layer:", "drain_nodes_layer_combo"),
            (9, "Drainage links layer:", "drain_links_layer_combo"),
            (10, "Drainage inlet types (table):", "drain_inlets_layer_combo"),
            (11, "Drainage node-inlets (table):", "drain_node_inlets_layer_combo"),
            (12, "Hydraulic structures layer:", "structures_layer_combo"),
            (13, "BC lines layer:", "bc_lines_layer_combo"),
        ]:
            widget = getattr(self, attr)
            if data_layout.indexOf(widget) < 0:
                data_layout.addWidget(QtWidgets.QLabel(label), row, 0)
                data_layout.addWidget(widget, row, 1)
        data_layout.setRowStretch(14, 1)

        # ── Tooltips for all Data page widgets ──────────────────────
        self.nodes_layer_combo.setToolTip(
            "QGIS point layer containing mesh node coordinates. "
            "Required for mesh construction. The 'node_id' field must be present."
        )
        self.cells_layer_combo.setToolTip(
            "QGIS polygon/multipolygon layer defining mesh cell geometry. "
            "Each cell has a 'cell_id' and references 'node_id' values. "
            "Required for mesh construction."
        )
        self.terrain_layer_combo.setToolTip(
            "Digital elevation model (DEM) raster layer used to assign node bed elevations. "
            "Select a raster then use 'Assign Node Z From Terrain' on the Mesh Setup tab."
        )
        self.manning_layer_combo.setToolTip(
            "Polygon layer with Manning's n values for spatially varying roughness. "
            "Field must contain a numeric roughness column. Leave empty for uniform n "
            "set in the Model tab."
        )
        self.cn_layer_combo.setToolTip(
            "Polygon layer containing SCS Curve Number values for runoff computation. "
            "Required when infiltration method is SCS Curve Number."
        )
        self.rain_gage_layer_combo.setToolTip(
            "Point layer defining rain gauge locations. Each gauge should have an ID "
            "matching entries in the hyetograph table layer."
        )
        self.hyetograph_layer_combo.setToolTip(
            "Table layer containing precipitation hyetographs. Columns: time (hours) "
            "and rainfall intensity (mm/hr or in/hr) for each gauge."
        )
        self.sample_lines_layer_combo.setToolTip(
            "Line layer for sampling flow results along cross-sections during simulation. "
            "Results are saved at the line output interval specified in the Run tab."
        )
        self.drain_nodes_layer_combo.setToolTip(
            "Point layer for drainage network nodes (manholes, junctions). "
            "Used for 1D-2D coupled drainage simulations."
        )
        self.drain_links_layer_combo.setToolTip(
            "Line layer for drainage network links (pipes, channels). "
            "Connects drain nodes for 1D-2D coupled drainage."
        )
        self.drain_inlets_layer_combo.setToolTip(
            "Table layer defining inlet types (grate, curb, combination) "
            "and their hydraulic capture curves."
        )
        self.drain_node_inlets_layer_combo.setToolTip(
            "Table layer mapping drain nodes to inlet types from the inlet types table. "
            "Defines which inlets are connected to which nodes."
        )
        self.structures_layer_combo.setToolTip(
            "Line layer for hydraulic structures (weirs, orifices, bridges, culverts, pumps). "
            "Each structure must have a type field and geometry."
        )
        self.bc_lines_layer_combo.setToolTip(
            "Line layer for boundary condition segments. "
            "Each segment defines a BC type (inflow, stage, normal depth, etc.) "
            "assigned via the default BC type combo or per-segment attributes."
        )

        for attr in [
            "drain_nodes_layer_combo",
            "drain_links_layer_combo",
            "drain_inlets_layer_combo",
            "drain_node_inlets_layer_combo",
            "structures_layer_combo",
            "bc_lines_layer_combo",
        ]:
            c = getattr(self, attr)
            if c.count() == 0:
                c.addItem("(none)", None)

        toolbox.addItem(page, "Layers")

    def _build_tools_page(self, toolbox: QtWidgets.QToolBox) -> None:
        """Build the Utilities page with explorer and log viewer buttons."""
        page = QtWidgets.QWidget()
        page.setObjectName("map_tools_page")
        tools_layout = QtWidgets.QGridLayout(page)
        tools_layout.setObjectName("map_tools_layout")
        tools_layout.setContentsMargins(0, 0, 0, 0)

        for attr, text, tip in [
            ("open_model_gpkg_explorer_btn", "Open Model GeoPackage Explorer",
             "Browse model GeoPackage tables and open matching viewers; "
             "rename/delete model result tables."),
            ("open_run_log_viewer_btn", "Open Run Log Viewer",
             "View, search, and export the current model run log. "
             "Shows solver output, timestep diagnostics, and error messages."),
        ]:
            btn = QtWidgets.QPushButton(text)
            btn.setObjectName(attr)
            setattr(self, attr, btn)
            if tip:
                btn.setToolTip(tip)

        for attr in [
            "open_model_gpkg_explorer_btn",
            "open_run_log_viewer_btn",
        ]:
            w = getattr(self, attr)
            row = 0 if attr == "open_model_gpkg_explorer_btn" else 1
            if tools_layout.indexOf(w) < 0:
                tools_layout.addWidget(w, row, 0, 1, 2)

        tools_layout.setRowStretch(3, 1)

        toolbox.addItem(page, "Utilities")


