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

    def test_view_has_algorithm_page(self):
        from swe2d.workbench.views.topology_tab_view import TopologyTabView
        view = TopologyTabView()
        page = view.findChild(QWidget, "topo_algo_page")
        self.assertIsInstance(page, QWidget)
        self.assertTrue(view._toolbox.isItemEnabled(view._algo_idx))

    def test_view_has_arcs_page(self):
        from swe2d.workbench.views.topology_tab_view import TopologyTabView
        view = TopologyTabView()
        page = view.findChild(QWidget, "topo_arcs_page")
        self.assertIsInstance(page, QWidget)

    def test_view_has_sizing_page(self):
        from swe2d.workbench.views.topology_tab_view import TopologyTabView
        view = TopologyTabView()
        page = view.findChild(QWidget, "topo_sizing_page")
        self.assertIsInstance(page, QWidget)

    def test_view_has_threading_page(self):
        from swe2d.workbench.views.topology_tab_view import TopologyTabView
        view = TopologyTabView()
        page = view.findChild(QWidget, "topo_threading_page")
        self.assertIsInstance(page, QWidget)

    def test_view_has_transfinite_page(self):
        from swe2d.workbench.views.topology_tab_view import TopologyTabView
        view = TopologyTabView()
        page = view.findChild(QWidget, "topo_transfinite_page")
        self.assertIsInstance(page, QWidget)

    def test_view_has_quality_page(self):
        from swe2d.workbench.views.topology_tab_view import TopologyTabView
        view = TopologyTabView()
        page = view.findChild(QWidget, "topo_quality_page")
        self.assertIsInstance(page, QWidget)
        self.assertTrue(view._toolbox.isItemEnabled(view._quality_idx))

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
            (view.topo_backend_combo, "topo_backend_combo"),
            (view.topo_default_size_spin, "topo_default_size_spin"),
            (view.topo_default_cell_type_combo, "topo_default_cell_type_combo"),
            (view.findChild(QWidget, "topo_algo_page"), "topo_algo_page"),
            (view.findChild(QWidget, "topo_arcs_page"), "topo_arcs_page"),
            (view.findChild(QWidget, "topo_sizing_page"), "topo_sizing_page"),
            (view.findChild(QWidget, "topo_threading_page"), "topo_threading_page"),
            (view.findChild(QWidget, "topo_transfinite_page"), "topo_transfinite_page"),
            (view.findChild(QWidget, "topo_quality_page"), "topo_quality_page"),
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

    def test_arcs_page_title_is_plain_text(self):
        from swe2d.workbench.views.topology_tab_view import TopologyTabView
        view = TopologyTabView()
        self.assertEqual(view._toolbox.itemText(view._arcs_idx), "Arcs and Interfaces")

    def test_non_gmsh_pages_are_disabled_and_suffixed(self):
        from swe2d.workbench.views.topology_tab_view import TopologyTabView
        view = TopologyTabView()
        view.topo_backend_combo.setCurrentIndex(view.topo_backend_combo.findData("structured"))
        view.update_control_summary()
        for idx in (view._algo_idx, view._arcs_idx, view._sizing_idx,
                    view._threading_idx, view._transfinite_idx, view._quality_idx):
            self.assertFalse(view._toolbox.isItemEnabled(idx))
            self.assertIn("(Gmsh only)", view._toolbox.itemText(idx))


if __name__ == "__main__":
    unittest.main(verbosity=2)
