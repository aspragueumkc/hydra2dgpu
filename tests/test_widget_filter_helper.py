"""Tests for the FilterableRowRegistry helper and the Output page
integration with the Model tab filter."""
from __future__ import annotations

import unittest

from qgis.PyQt.QtWidgets import QApplication as _QApp

_test_app = _QApp.instance()
if _test_app is None:
    _test_app = _QApp([])


class TestFilterableRowRegistry(unittest.TestCase):
    def setUp(self):
        from qgis.PyQt import QtWidgets

        # Make a clean widget parent to hold the test widgets
        self.parent = QtWidgets.QWidget()
        self.registry_add = __import__(
            "swe2d.workbench.views.widget_filter_helper",
            fromlist=["FilterableRowRegistry"],
        ).FilterableRowRegistry

    def _mk_checkbox(self, text, tooltip="", obj_name=""):
        from qgis.PyQt import QtWidgets
        cb = QtWidgets.QCheckBox(text)
        cb.setToolTip(tooltip)
        cb.setObjectName(obj_name)
        return cb

    def test_registry_empty_by_default(self):
        reg = self.registry_add()
        self.assertEqual(len(reg), 0)

    def test_register_increments_length(self):
        from qgis.PyQt import QtWidgets
        reg = self.registry_add()
        cb = self._mk_checkbox("Save mesh", obj_name="save_mesh_chk")
        reg.add(cb, label_text=cb.text(), tooltip=cb.toolTip())
        self.assertEqual(len(reg), 1)
        self.assertTrue(reg.is_registered(cb))

    def test_filter_hides_widget_when_no_text_match(self):
        reg = self.registry_add()
        cb = self._mk_checkbox("Save mesh", obj_name="save_mesh_chk")
        reg.add(cb, label_text=cb.text(), tooltip=cb.toolTip())
        reg.apply_filter("coupling", show_advanced=False)
        self.assertFalse(reg.filter_visible(cb))
        reg.apply_filter("", show_advanced=False)
        self.assertTrue(reg.filter_visible(cb))
        reg.apply_filter("mesh", show_advanced=False)
        self.assertTrue(reg.filter_visible(cb))

    def test_filter_matches_tooltip(self):
        reg = self.registry_add()
        cb = self._mk_checkbox(
            "Save coupling",
            tooltip="Save drainage/structure coupling time series results",
            obj_name="save_coupling_chk",
        )
        reg.add(cb, label_text=cb.text(), tooltip=cb.toolTip())
        reg.apply_filter("drainage", show_advanced=False)
        self.assertTrue(reg.filter_visible(cb))

    def test_filter_matches_object_name(self):
        reg = self.registry_add()
        cb = self._mk_checkbox("Foo", obj_name="save_log_chk")
        reg.add(cb, label_text=cb.text(), tooltip=cb.toolTip())
        reg.apply_filter("save_log", show_advanced=False)
        self.assertTrue(reg.filter_visible(cb))

    def test_advanced_widget_hidden_when_toggle_off(self):
        reg = self.registry_add()
        cb = self._mk_checkbox("Beta")
        reg.add(cb, label_text=cb.text(), advanced=True)
        reg.apply_filter("", show_advanced=False)
        self.assertFalse(reg.filter_visible(cb))
        reg.apply_filter("", show_advanced=True)
        self.assertTrue(reg.filter_visible(cb))

    def test_group_visibility_follows_children(self):
        """A QGroupBox is shown iff at least one registered child is shown."""
        from qgis.PyQt import QtWidgets
        reg = self.registry_add()
        group = QtWidgets.QGroupBox("Group A")
        cb1 = self._mk_checkbox("Save mesh", obj_name="save_mesh_chk")
        cb2 = self._mk_checkbox("Save line", obj_name="save_line_chk")
        reg.add(cb1, label_text=cb1.text(), group=group)
        reg.add(cb2, label_text=cb2.text(), group=group)
        # No filter — both visible, group visible
        reg.apply_filter("", show_advanced=False)
        self.assertTrue(reg.filter_visible(group))
        # Filter that matches neither — both hidden, group hidden
        reg.apply_filter("xyzzy", show_advanced=False)
        self.assertFalse(reg.filter_visible(group))
        # Filter matches only cb1 — group still visible
        reg.apply_filter("mesh", show_advanced=False)
        self.assertTrue(reg.filter_visible(group))

    def test_label_widget_toggles_with_control(self):
        from qgis.PyQt import QtWidgets
        reg = self.registry_add()
        label = QtWidgets.QLabel("Save mesh:")
        cb = self._mk_checkbox("Save mesh", obj_name="save_mesh_chk")
        reg.add(cb, label_widget=label, label_text=label.text())
        reg.apply_filter("coupling", show_advanced=False)
        self.assertFalse(reg.filter_visible(cb))
        self.assertFalse(reg.filter_visible(label))
        reg.apply_filter("mesh", show_advanced=False)
        self.assertTrue(reg.filter_visible(cb))
        self.assertTrue(reg.filter_visible(label))


class TestOutputPageFilterIntegration(unittest.TestCase):
    """The Output page (Storage moved from ResultsToolbox) must respond
    to the Simulation-tab param_search filter."""

    @classmethod
    def setUpClass(cls):
        _ensure_app()

    def test_output_checkboxes_are_registered(self):
        from swe2d.workbench.views.model_tab_view import ModelTabView
        view = ModelTabView()
        for chk in (
            view.extended_outputs_chk,
            view.save_mesh_chk,
            view.save_line_chk,
            view.save_coupling_chk,
            view.save_max_only_chk,
            view.save_log_chk,
        ):
            with self.subTest(chk=chk.objectName()):
                self.assertTrue(
                    view._filterable.is_registered(chk),
                    f"{chk.objectName()} not registered with _filterable",
                )

    def test_filter_hides_all_output_checkboxes_for_unmatched_text(self):
        from swe2d.workbench.views.model_tab_view import ModelTabView
        view = ModelTabView()
        view.param_search.setText("zzz_no_such_text_zzz")
        for chk in (
            view.extended_outputs_chk,
            view.save_mesh_chk,
            view.save_line_chk,
            view.save_coupling_chk,
            view.save_max_only_chk,
            view.save_log_chk,
        ):
            with self.subTest(chk=chk.objectName()):
                self.assertFalse(view._filterable.filter_visible(chk))

    def test_filter_shows_checkbox_by_label_substring(self):
        from swe2d.workbench.views.model_tab_view import ModelTabView
        view = ModelTabView()
        view.param_search.setText("coupling")
        self.assertTrue(view._filterable.filter_visible(view.save_coupling_chk))
        # Other output checkboxes don't match "coupling" and should be hidden
        self.assertFalse(view._filterable.filter_visible(view.save_mesh_chk))
        self.assertFalse(view._filterable.filter_visible(view.save_line_chk))

    def test_filter_shows_checkbox_by_object_name(self):
        from swe2d.workbench.views.model_tab_view import ModelTabView
        view = ModelTabView()
        view.param_search.setText("save_log_chk")
        self.assertTrue(view._filterable.filter_visible(view.save_log_chk))

    def test_clearing_filter_shows_everything(self):
        from swe2d.workbench.views.model_tab_view import ModelTabView
        view = ModelTabView()
        view.param_search.setText("coupling")
        # Sanity: some widgets are now hidden
        self.assertFalse(view._filterable.filter_visible(view.save_mesh_chk))
        view.param_search.setText("")
        for chk in (
            view.extended_outputs_chk,
            view.save_mesh_chk,
            view.save_line_chk,
            view.save_coupling_chk,
            view.save_max_only_chk,
            view.save_log_chk,
        ):
            with self.subTest(chk=chk.objectName()):
                self.assertTrue(view._filterable.filter_visible(chk))


def _ensure_app():
    if _QApp.instance() is None:
        _QApp([])


if __name__ == "__main__":
    unittest.main(verbosity=2)