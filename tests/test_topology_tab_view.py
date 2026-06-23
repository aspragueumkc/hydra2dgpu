"""Tests for TopologyTabView."""
import unittest
from qgis.PyQt.QtWidgets import (
    QApplication, QComboBox, QDoubleSpinBox, QPushButton, QLabel, QWidget
)

_app = None


def _ensure_app():
    global _app
    if _app is None:
        _app = QApplication.instance() or QApplication([])


class TestTopologyTabView(unittest.TestCase):
    def setUp(self):
        _ensure_app()

    def test_view_imports(self):
        from swe2d.workbench.views.topology_tab_view import TopologyTabView
        self.assertIsNotNone(TopologyTabView)

    def test_view_is_qwidget(self):
        from swe2d.workbench.views.topology_tab_view import TopologyTabView
        view = TopologyTabView()
        self.assertIsInstance(view, QWidget)

    def test_view_has_nodes_combo(self):
        from swe2d.workbench.views.topology_tab_view import TopologyTabView
        view = TopologyTabView()
        self.assertIsInstance(view.topo_nodes_combo, QComboBox)

    def test_view_has_arcs_combo(self):
        from swe2d.workbench.views.topology_tab_view import TopologyTabView
        view = TopologyTabView()
        self.assertIsInstance(view.topo_arcs_combo, QComboBox)

    def test_view_has_regions_combo(self):
        from swe2d.workbench.views.topology_tab_view import TopologyTabView
        view = TopologyTabView()
        self.assertIsInstance(view.topo_regions_combo, QComboBox)

    def test_view_has_constraints_combo(self):
        from swe2d.workbench.views.topology_tab_view import TopologyTabView
        view = TopologyTabView()
        self.assertIsInstance(view.topo_constraints_combo, QComboBox)

    def test_view_has_quad_edges_combo(self):
        from swe2d.workbench.views.topology_tab_view import TopologyTabView
        view = TopologyTabView()
        self.assertIsInstance(view.topo_quad_edges_combo, QComboBox)

    def test_view_has_export_template_btn(self):
        from swe2d.workbench.views.topology_tab_view import TopologyTabView
        view = TopologyTabView()
        self.assertIsInstance(view.topo_export_template_btn, QPushButton)

    def test_view_has_backend_combo(self):
        from swe2d.workbench.views.topology_tab_view import TopologyTabView
        view = TopologyTabView()
        self.assertIsInstance(view.topo_backend_combo, QComboBox)

    def test_view_has_default_size_spin(self):
        from swe2d.workbench.views.topology_tab_view import TopologyTabView
        view = TopologyTabView()
        self.assertIsInstance(view.topo_default_size_spin, QDoubleSpinBox)

    def test_view_has_default_cell_type_combo(self):
        from swe2d.workbench.views.topology_tab_view import TopologyTabView
        view = TopologyTabView()
        self.assertIsInstance(view.topo_default_cell_type_combo, QComboBox)

    def test_view_has_gmsh_controls_widget(self):
        from swe2d.workbench.views.topology_tab_view import TopologyTabView
        view = TopologyTabView()
        self.assertIsInstance(view.topo_gmsh_controls_widget, QWidget)
        self.assertFalse(view.topo_gmsh_controls_widget.isVisible())

    def test_view_has_quality_controls_widget(self):
        from swe2d.workbench.views.topology_tab_view import TopologyTabView
        view = TopologyTabView()
        self.assertIsInstance(view.topo_quality_controls_widget, QWidget)
        self.assertFalse(view.topo_quality_controls_widget.isVisible())

    def test_view_has_controls_summary_lbl(self):
        from swe2d.workbench.views.topology_tab_view import TopologyTabView
        view = TopologyTabView()
        self.assertIsInstance(view.topo_controls_summary_lbl, QLabel)
        self.assertTrue(view.topo_controls_summary_lbl.wordWrap())

    def test_view_has_validate_btn(self):
        from swe2d.workbench.views.topology_tab_view import TopologyTabView
        view = TopologyTabView()
        self.assertIsInstance(view.topo_validate_btn, QPushButton)
        self.assertTrue(view.topo_validate_btn.isEnabled())

    def test_view_has_edit_regions_btn(self):
        from swe2d.workbench.views.topology_tab_view import TopologyTabView
        view = TopologyTabView()
        self.assertIsInstance(view.topo_edit_regions_btn, QPushButton)
        self.assertTrue(view.topo_edit_regions_btn.isEnabled())

    def test_view_has_edit_quad_edges_btn(self):
        from swe2d.workbench.views.topology_tab_view import TopologyTabView
        view = TopologyTabView()
        self.assertIsInstance(view.topo_edit_quad_edges_btn, QPushButton)
        self.assertTrue(view.topo_edit_quad_edges_btn.isEnabled())

    def test_view_has_generate_btn(self):
        from swe2d.workbench.views.topology_tab_view import TopologyTabView
        view = TopologyTabView()
        self.assertIsInstance(view.topo_generate_btn, QPushButton)
        self.assertTrue(view.topo_generate_btn.isEnabled())

    def test_view_has_terminate_btn(self):
        from swe2d.workbench.views.topology_tab_view import TopologyTabView
        view = TopologyTabView()
        self.assertIsInstance(view.topo_terminate_btn, QPushButton)
        self.assertFalse(view.topo_terminate_btn.isEnabled())

    def test_widgets_have_object_names(self):
        """Object names are preserved for findChild compatibility."""
        from swe2d.workbench.views.topology_tab_view import TopologyTabView
        view = TopologyTabView()
        expected_names = [
            (view.topo_nodes_combo, "topo_nodes_combo"),
            (view.topo_arcs_combo, "topo_arcs_combo"),
            (view.topo_regions_combo, "topo_regions_combo"),
            (view.topo_constraints_combo, "topo_constraints_combo"),
            (view.topo_quad_edges_combo, "topo_quad_edges_combo"),
            (view.topo_export_template_btn, "topo_export_template_btn"),
            (view.topo_backend_combo, "topo_backend_combo"),
            (view.topo_default_size_spin, "topo_default_size_spin"),
            (view.topo_default_cell_type_combo, "topo_default_cell_type_combo"),
            (view.topo_gmsh_controls_widget, "topo_gmsh_controls_widget"),
            (view.topo_quality_controls_widget, "topo_quality_controls_widget"),
            (view.topo_controls_summary_lbl, "topo_controls_summary_lbl"),
            (view.topo_validate_btn, "topo_validate_btn"),
            (view.topo_edit_regions_btn, "topo_edit_regions_btn"),
            (view.topo_edit_quad_edges_btn, "topo_edit_quad_edges_btn"),
            (view.topo_generate_btn, "topo_generate_btn"),
            (view.topo_terminate_btn, "topo_terminate_btn"),
        ]
        for widget, expected_name in expected_names:
            self.assertEqual(widget.objectName(), expected_name)

    def test_view_is_standalone(self):
        """The view should be testable without a parent dialog."""
        from swe2d.workbench.views.topology_tab_view import TopologyTabView
        view = TopologyTabView()
        view.topo_default_size_spin.setValue(2.5)
        self.assertEqual(view.topo_default_size_spin.value(), 2.5)
        view.topo_nodes_combo.addItem("nodes_layer")
        self.assertEqual(view.topo_nodes_combo.count(), 1)
        view.deleteLater()


if __name__ == "__main__":
    unittest.main(verbosity=2)
