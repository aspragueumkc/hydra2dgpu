#!/usr/bin/env python3
"""Behavioral tests for swe2d/workbench/views/studio_viewer_pg.py.

Covers (per docs/specs/2026-08-02-gui-test-coverage-design.md §3-§4,
pattern P2 — real headless QGIS, no mocks):

- ``_unit_labels`` in both unit systems (explicit + configured fallback)
- ``_label_for_var`` / ``_var_from_label`` round-trip identity for every
  supported var key (flow, depth, wse, velocity) in both unit systems,
  plus the unknown-key contract of both functions
- ``_c2q`` colour conversion
- ``PGTimeSeriesWidget``: construct, feed a real time series baked into a
  real results GPKG via the production writer, drive the real data API
  (``set_data`` / ``refresh`` / metric switch) and assert on the pyqtgraph
  curve data (``curve.getData()`` — never pixels), table population, and
  offscreen render via ``grab_non_empty``.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path and os.path.isdir(_REPO_ROOT):
    sys.path.insert(0, _REPO_ROOT)

from tests.qgis_real_env import (
    delete_widgets_now,
    ensure_qgis_app,
    grab_non_empty,
    make_temp_results_gpkg,
    requires_qgis,
)

_HAVE_PG = importlib.util.find_spec("pyqtgraph") is not None

# The complete set of var keys the module supports (from _label_for_var).
_VAR_KEYS = ("flow", "depth", "wse", "velocity")

_SI_LABELS = {"len": "m", "flow": "m³/s", "vel": "m/s"}
_USC_LABELS = {"len": "ft", "flow": "ft³/s", "vel": "ft/s"}

# Deterministic baked line time-series fixture.
_LINE_ID = 3
_OTHER_LINE_ID = 7  # second line so the element-id combo has >1 entries
_TIMES_S = np.array([0.0, 10.0, 20.0, 30.0], dtype=np.float64)
_DEPTH = np.array([1.0, 1.1, 1.2, 1.3], dtype=np.float64)
_VELOCITY = np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float64)
_WSE = np.array([11.0, 11.1, 11.2, 11.3], dtype=np.float64)
_BED = np.array([10.0, 10.0, 10.0, 10.0], dtype=np.float64)
_FLOW = np.array([2.5, 3.0, 3.5, 4.0], dtype=np.float64)
_WET_FRAC = np.ones_like(_TIMES_S)
_FR = np.zeros_like(_TIMES_S)

_VAR_ARRAYS = {
    "flow": _FLOW,
    "depth": _DEPTH,
    "wse": _WSE,
    "velocity": _VELOCITY,
}


@requires_qgis
class TestUnitLabels(unittest.TestCase):
    """_unit_labels / _label_for_var / _var_from_label contracts."""

    @classmethod
    def setUpClass(cls):
        ensure_qgis_app()
        from swe2d.workbench.views import studio_viewer_pg as mod

        cls.mod = mod

    # -- _unit_labels ---------------------------------------------------

    def test_unit_labels_si_explicit(self):
        self.assertEqual(self.mod._unit_labels("m"), _SI_LABELS)

    def test_unit_labels_usc_explicit(self):
        self.assertEqual(self.mod._unit_labels("ft"), _USC_LABELS)

    def test_unit_labels_case_and_whitespace_insensitive(self):
        self.assertEqual(self.mod._unit_labels(" FT "), _USC_LABELS)

    def test_unit_labels_configured_fallback_si(self):
        from swe2d import units as _u

        try:
            _u.configure(1.0)  # metric CRS
            self.assertEqual(self.mod._unit_labels(""), _SI_LABELS)
        finally:
            _u.configure(1.0)

    def test_unit_labels_configured_fallback_usc(self):
        from swe2d import units as _u

        try:
            _u.configure(0.3048)  # US-foot CRS
            self.assertEqual(self.mod._unit_labels(""), _USC_LABELS)
        finally:
            _u.configure(1.0)

    # -- _label_for_var / _var_from_label -------------------------------

    def test_label_for_var_si(self):
        self.assertEqual(self.mod._label_for_var("flow", "m"), "Flow (m³/s)")
        self.assertEqual(self.mod._label_for_var("depth", "m"), "Depth (m)")
        self.assertEqual(self.mod._label_for_var("wse", "m"), "WSE (m)")
        self.assertEqual(
            self.mod._label_for_var("velocity", "m"), "Velocity (m/s)"
        )

    def test_label_for_var_usc(self):
        self.assertEqual(self.mod._label_for_var("flow", "ft"), "Flow (ft³/s)")
        self.assertEqual(self.mod._label_for_var("depth", "ft"), "Depth (ft)")
        self.assertEqual(self.mod._label_for_var("wse", "ft"), "WSE (ft)")
        self.assertEqual(
            self.mod._label_for_var("velocity", "ft"), "Velocity (ft/s)"
        )

    def test_label_roundtrip_identity_every_var_key_both_unit_systems(self):
        for unit in ("m", "ft"):
            for key in _VAR_KEYS:
                with self.subTest(unit=unit, key=key):
                    label = self.mod._label_for_var(key, unit)
                    self.assertEqual(self.mod._var_from_label(label), key)

    def test_label_for_var_unknown_key_returns_key_verbatim(self):
        self.assertEqual(
            self.mod._label_for_var("bogus_metric", "m"), "bogus_metric"
        )

    def test_var_from_label_unknown_label_defaults_to_flow(self):
        self.assertEqual(self.mod._var_from_label("Totally Bogus"), "flow")

    # -- _c2q -------------------------------------------------------------

    def test_c2q_produces_qcolor_with_matching_rgb(self):
        color = self.mod._c2q((31, 119, 180))
        self.assertEqual(tuple(color.getRgb()[:3]), (31, 119, 180))


@requires_qgis
@unittest.skipUnless(_HAVE_PG, "pyqtgraph required for PGTimeSeriesWidget")
class TestPGTimeSeriesWidget(unittest.TestCase):
    """Drive the real widget against a real baked results GPKG."""

    @classmethod
    def setUpClass(cls):
        ensure_qgis_app()
        from swe2d.workbench.views import studio_viewer_pg as mod

        cls.mod = mod

    def setUp(self):
        from swe2d.results.data import SWE2DResultsData
        from swe2d.services.gpkg_persistence_service import persist_baked_line_ts

        # Real results GPKG via the production writer, plus a baked line
        # time series via the production line-TS writer (same run).
        self._gpkg_ctx = make_temp_results_gpkg(n_cells=4, n_timesteps=3)
        self._gpkg_path = self._gpkg_ctx.__enter__()
        self.addCleanup(self._gpkg_ctx.__exit__, None, None, None)
        persist_baked_line_ts(
            gpkg_path=self._gpkg_path,
            run_id="hydra_test_run",
            line_id=_LINE_ID,
            line_name="Test Line",
            times=_TIMES_S,
            depth=_DEPTH,
            velocity=_VELOCITY,
            wse=_WSE,
            bed=_BED,
            flow=_FLOW,
            wet_frac=_WET_FRAC,
            fr=_FR,
        )

        # Real data layer discovered through the production read path.
        self._data = SWE2DResultsData()
        added_paths, added_runs = self._data.add_results_files([self._gpkg_path])
        self.assertGreater(added_runs, 0, "production reader found no runs")
        self.assertTrue(self._data.get_enabled_run_records())

        self._widget = self.mod.PGTimeSeriesWidget()
        self._widget.resize(800, 600)
        self.addCleanup(delete_widgets_now, self._widget)
        self._widget.set_data(result_data=self._data, length_unit="m")

    # -- helpers --------------------------------------------------------

    def _curve_xy(self):
        self.assertGreaterEqual(
            len(self._widget._plot_items), 1, "no curve plotted"
        )
        x, y = self._widget._plot_items[0].getData()
        self.assertIsNotNone(x)
        self.assertIsNotNone(y)
        return np.asarray(x), np.asarray(y)

    # -- tests ----------------------------------------------------------

    def test_construct_protocol_properties(self):
        widget = self.mod.PGTimeSeriesWidget()
        self.addCleanup(delete_widgets_now, widget)
        import pyqtgraph as pg

        self.assertEqual(widget.mode, "Time Series")
        self.assertIsInstance(widget.canvas, pg.PlotWidget)
        self.assertIsNone(widget.fig)
        self.assertEqual(widget.selected_metric, "flow")
        self.assertEqual(widget.selected_element_id, "")

    def test_set_data_populates_element_ids_from_gpkg(self):
        combo = self._widget._element_id_combo
        ids = [combo.itemData(i) for i in range(combo.count())]
        self.assertIn(_LINE_ID, ids)
        self.assertEqual(self._widget.selected_element_id, str(_LINE_ID))

    def test_selected_element_id_setter_syncs_combo(self):
        """Setting ``selected_element_id`` to a known combo value must move
        the combo's current index to match.  Mirrors the contract that
        ``selected_metric`` already follows (covered by
        ``test_variable_switch_via_property_updates_curve`` above).

        Seed a second line ts so the combo has at least two entries — the
        property setter must demonstrably move the combo, not just write
        to an internal field.
        """
        from swe2d.services.gpkg_persistence_service import persist_baked_line_ts

        persist_baked_line_ts(
            gpkg_path=self._gpkg_path,
            run_id="hydra_test_run",
            line_id=_OTHER_LINE_ID,
            line_name="Other Line",
            times=_TIMES_S,
            depth=_DEPTH,
            velocity=_VELOCITY,
            wse=_WSE,
            bed=_BED,
            flow=_FLOW,
            wet_frac=_WET_FRAC,
            fr=_FR,
        )
        # Re-bind result_data so the new line is picked up.
        from swe2d.results.data import SWE2DResultsData

        data = SWE2DResultsData()
        added_paths, _ = data.add_results_files([self._gpkg_path])
        self.assertEqual(added_paths, 1)
        self._widget.set_data(result_data=data, length_unit="m")
        combo = self._widget._element_id_combo
        ids = [combo.itemData(i) for i in range(combo.count())]
        self.assertIn(_LINE_ID, ids)
        self.assertIn(_OTHER_LINE_ID, ids)

        # Park the combo on the OTHER id first to prove the setter moves it.
        other_idx = ids.index(_OTHER_LINE_ID)
        combo.setCurrentIndex(other_idx)
        self.assertEqual(combo.currentData(), _OTHER_LINE_ID)

        # Drive the property setter; the combo must follow.
        self._widget.selected_element_id = str(_LINE_ID)
        self.assertEqual(combo.currentData(), _LINE_ID)
        self.assertEqual(self._widget.selected_element_id, str(_LINE_ID))

    def test_selected_element_id_setter_unknown_value_no_op(self):
        """Unknown element id leaves the combo untouched (no exception)."""
        combo = self._widget._element_id_combo
        before_idx = combo.currentIndex()
        self._widget.selected_element_id = "99999_does_not_exist"
        self.assertEqual(combo.currentIndex(), before_idx)
        self.assertEqual(
            self._widget.selected_element_id, "99999_does_not_exist",
            "setter must still record the value even when no combo entry exists",
        )

    def test_selected_element_id_setter_empty_clears(self):
        """Falsy element id is stored as "" and does not move the combo."""
        self._widget.selected_element_id = ""
        self.assertEqual(self._widget.selected_element_id, "")

    def test_refresh_plots_flow_curve_from_real_gpkg(self):
        self._widget.refresh()
        x, y = self._curve_xy()
        np.testing.assert_allclose(x, _TIMES_S / 3600.0)
        np.testing.assert_allclose(y, _FLOW)
        axis_label = (
            self._widget.canvas.plotItem.getAxis("left").labelText
        )
        self.assertEqual(axis_label, "Flow (m³/s)")

    def test_variable_switch_via_property_updates_curve(self):
        self._widget.refresh()
        for key in _VAR_KEYS:
            with self.subTest(key=key):
                self._widget.selected_metric = key
                x, y = self._curve_xy()
                np.testing.assert_allclose(x, _TIMES_S / 3600.0)
                np.testing.assert_allclose(y, _VAR_ARRAYS[key])

    def test_variable_switch_via_combo_updates_curve(self):
        self._widget.refresh()
        combo = self._widget._metric_combo
        idx = combo.findData("depth")
        self.assertGreaterEqual(idx, 0)
        combo.setCurrentIndex(idx)  # fires _on_metric_changed → refresh
        self.assertEqual(self._widget.selected_metric, "depth")
        x, y = self._curve_xy()
        np.testing.assert_allclose(y, _DEPTH)
        axis_label = (
            self._widget.canvas.plotItem.getAxis("left").labelText
        )
        self.assertEqual(axis_label, "Depth (m)")

    def test_length_unit_override_ft_axis_label(self):
        self._widget.set_data(result_data=self._data, length_unit="ft")
        self._widget.refresh()
        axis_label = (
            self._widget.canvas.plotItem.getAxis("left").labelText
        )
        self.assertEqual(axis_label, "Flow (ft³/s)")

    def test_table_toggle_populates_table(self):
        self._widget.refresh()
        self._widget.show_table_toggle.setChecked(True)
        table = self._widget._table_widget
        # Offscreen harness: isHidden() is False once explicitly shown,
        # even though isVisible() stays False under a hidden parent.
        self.assertFalse(table.isHidden())
        self.assertEqual(table.rowCount(), _TIMES_S.size)
        headers = [
            table.horizontalHeaderItem(j).text()
            for j in range(table.columnCount())
        ]
        self.assertIn("t_s", headers)
        self.assertIn("flow", headers)

    def test_grab_non_empty_after_data_set(self):
        self._widget.refresh()
        self.assertTrue(
            grab_non_empty(self._widget),
            "widget rendered empty after real time series was plotted",
        )


if __name__ == "__main__":
    unittest.main()
