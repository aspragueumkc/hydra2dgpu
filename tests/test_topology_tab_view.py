"""Tests for TopologyTabView."""
import unittest
from qgis.PyQt.QtWidgets import (
    QApplication, QComboBox, QDoubleSpinBox, QPushButton, QLabel, QLineEdit, QWidget
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
        # Pages are always enabled regardless of backend selection;
        # update_control_summary() must be called to sync state after init.
        view.update_control_summary()
        self.assertTrue(view._toolbox.isItemEnabled(view._algo_idx))

    def test_view_has_mesh_def_page(self):
        """Arcs + Sizing + Transfinite are combined into a single
        Mesh Definition page."""
        from swe2d.workbench.views.topology_tab_view import TopologyTabView
        view = TopologyTabView()
        page = view.findChild(QWidget, "topo_mesh_def_page")
        self.assertIsInstance(page, QWidget)

    def test_mesh_def_page_has_three_sub_sections(self):
        """The Mesh Definition page has three QGroupBox sub-sections."""
        from qgis.PyQt.QtWidgets import QGroupBox
        from swe2d.workbench.views.topology_tab_view import TopologyTabView
        view = TopologyTabView()
        for section_name in ("topo_arcs_section", "topo_sizing_section",
                             "topo_transfinite_section"):
            section = view.findChild(QGroupBox, section_name)
            self.assertIsInstance(
                section, QGroupBox,
                f"missing sub-section: {section_name}",
            )

    def test_legacy_arcs_sizing_transfinite_pages_removed(self):
        """The old standalone page widgets should no longer exist."""
        from swe2d.workbench.views.topology_tab_view import TopologyTabView
        view = TopologyTabView()
        for page_name in ("topo_arcs_page", "topo_sizing_page",
                          "topo_transfinite_page"):
            page = view.findChild(QWidget, page_name)
            self.assertIsNone(page, f"legacy page still present: {page_name}")

    def test_threading_page_is_removed(self):
        """The standalone Threading page was removed; threading widgets
        now live on the Algorithm page."""
        from swe2d.workbench.views.topology_tab_view import TopologyTabView
        view = TopologyTabView()
        page = view.findChild(QWidget, "topo_threading_page")
        self.assertIsNone(page)

    def test_threading_form_is_algo_form(self):
        """topo_threading_form is an alias of topo_algo_form so the
        legacy control-builder can keep targeting it."""
        from swe2d.workbench.views.topology_tab_view import TopologyTabView
        view = TopologyTabView()
        self.assertIs(view.topo_threading_form, view.topo_algo_form)

    def test_view_has_quality_page(self):
        from swe2d.workbench.views.topology_tab_view import TopologyTabView
        view = TopologyTabView()
        page = view.findChild(QWidget, "topo_quality_page")
        self.assertIsInstance(page, QWidget)
        # Pages are always enabled regardless of backend selection;
        # update_control_summary() must be called to sync state after init.
        view.update_control_summary()
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
            (view.findChild(QWidget, "topo_mesh_def_page"), "topo_mesh_def_page"),
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
        """The legacy ``_arcs_idx`` attribute now points at the combined
        Mesh Definition page; the title is 'Mesh Definition'."""
        from swe2d.workbench.views.topology_tab_view import TopologyTabView
        view = TopologyTabView()
        self.assertEqual(view._toolbox.itemText(view._arcs_idx), "Mesh Definition")

    def test_quality_page_renamed_to_quality_loop(self):
        """Quality page is now titled 'Quality Loop'."""
        from swe2d.workbench.views.topology_tab_view import TopologyTabView
        view = TopologyTabView()
        self.assertEqual(view._toolbox.itemText(view._quality_idx), "Quality Loop")

    def test_gmsh_only_pages_always_enabled_suffixes_when_structured(self):
        """Gmsh-only pages are always enabled; they get '(Gmsh only)' suffix when structured backend is selected."""
        from swe2d.workbench.views.topology_tab_view import TopologyTabView
        view = TopologyTabView()
        view.topo_backend_combo.setCurrentIndex(view.topo_backend_combo.findData("structured"))
        view.update_control_summary()
        for idx in (view._algo_idx, view._mesh_def_idx, view._quality_idx):
            self.assertTrue(view._toolbox.isItemEnabled(idx))
            self.assertIn("(Gmsh only)", view._toolbox.itemText(idx))


class TestTopologyTabFilter(unittest.TestCase):
    """Tests for the topo_search / topo_show_advanced_chk filter (mirrors ModelTabView)."""

    def setUp(self):
        _ensure_app()

    def _make_view(self):
        from swe2d.workbench.views.topology_tab_view import TopologyTabView
        return TopologyTabView()

    def test_topology_tab_has_search_filter(self):
        view = self._make_view()
        self.assertIsInstance(view.topo_search, QLineEdit)
        self.assertEqual(view.topo_search.objectName(), "topo_search")

    def test_topology_tab_has_advanced_toggle(self):
        from qgis.PyQt.QtWidgets import QCheckBox
        view = self._make_view()
        self.assertIsInstance(view.topo_show_advanced_chk, QCheckBox)
        self.assertEqual(view.topo_show_advanced_chk.objectName(), "topo_show_advanced_chk")
        self.assertFalse(view.topo_show_advanced_chk.isChecked())

    def test_topology_tab_uses_filterable_registry(self):
        from swe2d.workbench.views.widget_filter_helper import FilterableRowRegistry
        view = self._make_view()
        self.assertIsInstance(view._filterable, FilterableRowRegistry)
        # The view pre-registers the layer-setup and general-page widgets
        # before _populate_gmsh_quality_controls runs, so _filterable
        # should be non-empty.
        self.assertGreater(len(view._filterable), 0)

    def test_filter_hides_non_matching_rows(self):
        view = self._make_view()
        # Build all Gmsh controls so the filter covers them too
        view._populate_gmsh_quality_controls()
        view.show()
        view.topo_show_advanced_chk.setChecked(True)
        view.topo_search.setText("cfl")
        view._filter_topology_tab()
        # No "cfl" matches expected on the topology tab; just check
        # the filter ran without error and at least one widget is
        # hidden (no widget on the topology tab has "cfl" in its blob).
        from swe2d.workbench.views.widget_filter_helper import FilterableRowRegistry
        hidden = sum(
            1 for _g, _l, w, _a in view._filterable
            if not view._filterable.filter_visible(w)
        )
        self.assertGreater(hidden, 0, "Filter should hide non-matching rows")

    def test_advanced_toggle_shows_advanced_rows(self):
        view = self._make_view()
        view._populate_gmsh_quality_controls()
        view.show()
        advanced_rows = [(_g, _l, w) for _g, _l, w, _a in view._filterable if _a]
        # Sanity: at least one advanced widget must be registered for the
        # toggle to be useful.
        self.assertGreater(
            len(advanced_rows), 0,
            "Topology tab must register advanced widgets for the filter "
            "toggle to be non-trivial."
        )

        # With advanced hidden, every advanced row should be invisible.
        view.topo_show_advanced_chk.setChecked(False)
        view._filter_topology_tab()
        for _g, _l, w in advanced_rows:
            self.assertFalse(
                view._filterable.filter_visible(w),
                f"{w.objectName()} should be hidden when advanced toggle is off",
            )

        # With advanced shown, every advanced row should be visible.
        view.topo_show_advanced_chk.setChecked(True)
        view._filter_topology_tab()
        for _g, _l, w in advanced_rows:
            self.assertTrue(
                view._filterable.filter_visible(w),
                f"{w.objectName()} should be visible when advanced toggle is on",
            )

    def test_text_search_filters_widgets(self):
        view = self._make_view()
        view._populate_gmsh_quality_controls()
        view.show()
        view.topo_show_advanced_chk.setChecked(True)
        view.topo_search.setText("gmsh")
        view._filter_topology_tab()
        visible = sum(
            1 for _g, _l, w, _a in view._filterable
            if view._filterable.filter_visible(w)
        )
        # 'gmsh' appears in many tooltips — at least some rows should remain.
        self.assertGreater(visible, 0)
        # And every visible row's search blob should contain "gmsh".
        for _g, _l, w, _a in view._filterable:
            if view._filterable.filter_visible(w):
                blob = str(w.property("filter_search_blob") or "").lower()
                self.assertIn("gmsh", blob)

    def test_filter_with_no_text_shows_everything(self):
        view = self._make_view()
        view._populate_gmsh_quality_controls()
        view.show()
        # With empty search text AND advanced shown, every row should be visible.
        view.topo_search.setText("")
        view.topo_show_advanced_chk.setChecked(True)
        view._filter_topology_tab()
        all_visible = all(
            view._filterable.filter_visible(w)
            for _g, _l, w, _a in view._filterable
        )
        self.assertTrue(
            all_visible,
            "Empty filter with advanced on should show every registered row",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
