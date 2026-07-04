"""Tests for MapTabView."""
import unittest
from qgis.PyQt.QtWidgets import (
    QApplication,
    QWidget,
    QComboBox,
    QPushButton,
    QCheckBox,
    QDoubleSpinBox,
    QLabel,
    QSpinBox,
)

_app = None


def _ensure_app():
    global _app
    if _app is None:
        _app = QApplication.instance() or QApplication([])


class TestMapTabView(unittest.TestCase):
    def setUp(self):
        _ensure_app()

    def test_view_imports(self):
        from swe2d.workbench.views.map_tab_view import MapTabView
        self.assertIsNotNone(MapTabView)

    def test_view_is_qwidget(self):
        from swe2d.workbench.views.map_tab_view import MapTabView
        view = MapTabView()
        self.assertIsInstance(view, QWidget)

    def test_view_has_nodes_layer_combo(self):
        from swe2d.workbench.views.map_tab_view import MapTabView
        view = MapTabView()
        self.assertIsInstance(view.nodes_layer_combo, QComboBox)

    def test_view_has_cells_layer_combo(self):
        from swe2d.workbench.views.map_tab_view import MapTabView
        view = MapTabView()
        self.assertIsInstance(view.cells_layer_combo, QComboBox)

    def test_view_has_terrain_layer_combo(self):
        from swe2d.workbench.views.map_tab_view import MapTabView
        view = MapTabView()
        self.assertIsInstance(view.terrain_layer_combo, QComboBox)

    def test_view_has_manning_layer_combo(self):
        from swe2d.workbench.views.map_tab_view import MapTabView
        view = MapTabView()
        self.assertIsInstance(view.manning_layer_combo, QComboBox)

    def test_view_has_cn_layer_combo(self):
        from swe2d.workbench.views.map_tab_view import MapTabView
        view = MapTabView()
        self.assertIsInstance(view.cn_layer_combo, QComboBox)

    def test_view_has_rain_gage_layer_combo(self):
        from swe2d.workbench.views.map_tab_view import MapTabView
        view = MapTabView()
        self.assertIsInstance(view.rain_gage_layer_combo, QComboBox)

    def test_view_has_hyetograph_layer_combo(self):
        from swe2d.workbench.views.map_tab_view import MapTabView
        view = MapTabView()
        self.assertIsInstance(view.hyetograph_layer_combo, QComboBox)

    def test_view_has_sample_lines_layer_combo(self):
        from swe2d.workbench.views.map_tab_view import MapTabView
        view = MapTabView()
        self.assertIsInstance(view.sample_lines_layer_combo, QComboBox)

    def test_view_has_drain_nodes_layer_combo(self):
        from swe2d.workbench.views.map_tab_view import MapTabView
        view = MapTabView()
        self.assertIsInstance(view.drain_nodes_layer_combo, QComboBox)

    def test_view_has_drain_links_layer_combo(self):
        from swe2d.workbench.views.map_tab_view import MapTabView
        view = MapTabView()
        self.assertIsInstance(view.drain_links_layer_combo, QComboBox)

    def test_view_has_drain_inlets_layer_combo(self):
        from swe2d.workbench.views.map_tab_view import MapTabView
        view = MapTabView()
        self.assertIsInstance(view.drain_inlets_layer_combo, QComboBox)

    def test_view_has_drain_node_inlets_layer_combo(self):
        from swe2d.workbench.views.map_tab_view import MapTabView
        view = MapTabView()
        self.assertIsInstance(view.drain_node_inlets_layer_combo, QComboBox)

    def test_view_has_structures_layer_combo(self):
        from swe2d.workbench.views.map_tab_view import MapTabView
        view = MapTabView()
        self.assertIsInstance(view.structures_layer_combo, QComboBox)

    def test_view_has_export_mesh_layers_btn(self):
        from swe2d.workbench.views.map_tab_view import MapTabView
        view = MapTabView()
        self.assertIsInstance(view.export_mesh_layers_btn, QPushButton)

    def test_view_has_import_mesh_layers_btn(self):
        from swe2d.workbench.views.map_tab_view import MapTabView
        view = MapTabView()
        self.assertIsInstance(view.import_mesh_layers_btn, QPushButton)

    def test_view_has_open_model_gpkg_explorer_btn(self):
        from swe2d.workbench.views.map_tab_view import MapTabView
        view = MapTabView()
        self.assertIsInstance(view.open_model_gpkg_explorer_btn, QPushButton)

    def test_view_has_open_run_log_viewer_btn(self):
        from swe2d.workbench.views.map_tab_view import MapTabView
        view = MapTabView()
        self.assertIsInstance(view.open_run_log_viewer_btn, QPushButton)

    def test_widgets_have_object_names(self):
        """Object names are preserved for findChild compatibility."""
        from swe2d.workbench.views.map_tab_view import MapTabView
        view = MapTabView()
        expected_names = [
            ("nodes_layer_combo", "nodes_layer_combo"),
            ("cells_layer_combo", "cells_layer_combo"),
            ("terrain_layer_combo", "terrain_layer_combo"),
            ("manning_layer_combo", "manning_layer_combo"),
            ("cn_layer_combo", "cn_layer_combo"),
            ("rain_gage_layer_combo", "rain_gage_layer_combo"),
            ("hyetograph_layer_combo", "hyetograph_layer_combo"),
            ("sample_lines_layer_combo", "sample_lines_layer_combo"),
            ("drain_nodes_layer_combo", "drain_nodes_layer_combo"),
            ("drain_links_layer_combo", "drain_links_layer_combo"),
            ("drain_inlets_layer_combo", "drain_inlets_layer_combo"),
            ("drain_node_inlets_layer_combo", "drain_node_inlets_layer_combo"),
            ("structures_layer_combo", "structures_layer_combo"),
            ("export_mesh_layers_btn", "export_mesh_layers_btn"),
            ("import_mesh_layers_btn", "import_mesh_layers_btn"),
            ("open_model_gpkg_explorer_btn", "open_model_gpkg_explorer_btn"),
            ("open_run_log_viewer_btn", "open_run_log_viewer_btn"),
        ]
        for attr, expected_name in expected_names:
            widget = getattr(view, attr)
            self.assertEqual(widget.objectName(), expected_name, f"{attr} object name mismatch")

    def test_view_is_standalone(self):
        """The view should be testable without a parent dialog."""
        from swe2d.workbench.views.map_tab_view import MapTabView
        view = MapTabView()
        view.nodes_layer_combo.addItem("Test Layer", "test_id")
        self.assertEqual(view.nodes_layer_combo.count(), 1)
        view.deleteLater()

    def test_view_has_toolbox(self):
        """The view uses a QToolBox for the three sub-pages."""
        from swe2d.workbench.views.map_tab_view import MapTabView
        from qgis.PyQt.QtWidgets import QToolBox
        view = MapTabView()
        toolboxes = view.findChildren(QToolBox)
        self.assertEqual(len(toolboxes), 1)
        self.assertEqual(toolboxes[0].count(), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
