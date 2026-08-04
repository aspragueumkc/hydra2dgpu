"""Characterization tests for the GeoPackage inspection widgets.

Covers ``swe2d/workbench/dialogs/gpkg_plot_tab.py`` (``GpkgPlotTab``) and
``swe2d/workbench/dialogs/gpkg_array_viewer_widget.py`` (``ArrayViewerWidget``)
per Task D.4 of ``docs/plans/2026-08-02-gui-test-coverage.md``.

Pattern P2 (dialog workflow) per
``docs/specs/2026-08-02-gui-test-coverage-design.md`` §3-§4: real widgets
driven against a real results GeoPackage written by the production results
writer (``make_temp_results_gpkg`` — run ``hydra_test_run`` /
``hydra_test_mesh`` with 3 timesteps × 4 cells of h/hu/hv plus max-tracking).
Plot assertions inspect matplotlib artist data, never pixels.  No mocks.

Note on QMessageBox paths: ``GpkgPlotTab._render_plot`` pops a modal warning
when the selected columns are not extractable (e.g. TEXT columns).  These
tests always select real BLOB columns before clicking Plot so no modal
dialog can block the offscreen test run.
"""

import unittest

import numpy as np

from tests.qgis_real_env import (
    delete_widgets_now,
    ensure_qgis_app,
    grab_non_empty,
    make_temp_results_gpkg,
    requires_qgis,
)

from swe2d.workbench.dialogs.gpkg_array_viewer_widget import ArrayViewerWidget
from swe2d.workbench.dialogs.gpkg_plot_tab import FigureCanvasQt, GpkgPlotTab
from swe2d.workbench.services.numpy_blob_service import deserialize_blob_to_array

_RESULTS_TABLE = "swe2d_baked_results"
_RUN_PK = {"run_id": "hydra_test_run"}

# Ground truth mirrored from make_temp_results_gpkg (n_cells=4, n_timesteps=3):
#   times[k]        = 10.0 * k
#   h[k, :]         = 1.0 + 0.1 * k + linspace(0.0, 0.5, 4)
#   max_h           = h at the last timestep (h grows monotonically in k)
_N_CELLS = 4
_N_TIMESTEPS = 3
_TIMES = np.arange(_N_TIMESTEPS, dtype=np.float64) * 10.0
_H = np.array(
    [1.0 + 0.1 * k + np.linspace(0.0, 0.5, _N_CELLS) for k in range(_N_TIMESTEPS)]
)
_MAX_H = _H[-1]

_skip_no_mpl = unittest.skipUnless(
    FigureCanvasQt is not None, "matplotlib Qt backend not available"
)


def _decode(gpkg_path, column):
    """Decode a BLOB column through the production service; fail loudly."""
    arr = deserialize_blob_to_array(gpkg_path, _RESULTS_TABLE, column, **_RUN_PK)
    if arr is None:
        raise AssertionError(
            f"production decoder returned None for {_RESULTS_TABLE}.{column}"
        )
    return arr


def _select_combo_by_data(combo, data_value):
    """Set a GpkgPlotTab combo to the item whose userData == data_value."""
    idx = combo.findData(data_value)
    if idx < 0:
        raise AssertionError(
            f"combo has no item with data {data_value!r}; "
            f"items: {[combo.itemData(i) for i in range(combo.count())]}"
        )
    combo.setCurrentIndex(idx)


@requires_qgis
class TestGpkgPlotTab(unittest.TestCase):
    """GpkgPlotTab against a real results GPKG (pattern P2)."""

    @classmethod
    def setUpClass(cls):
        ensure_qgis_app()

    def setUp(self):
        self._tab = GpkgPlotTab()

    def tearDown(self):
        delete_widgets_now(self._tab)
        self._tab = None

    def test_set_table_lists_real_columns(self):
        with make_temp_results_gpkg() as gpkg:
            self._tab.set_table(gpkg, _RESULTS_TABLE)

            import os
            self.assertEqual(self._tab._gpkg_lbl.text(), os.path.basename(gpkg))
            self.assertEqual(self._tab._table_lbl.text(), _RESULTS_TABLE)

            x_items = {
                self._tab._x_combo.itemData(i): self._tab._x_combo.itemText(i)
                for i in range(self._tab._x_combo.count())
            }
            y_items = {
                self._tab._y_combo.itemData(i): self._tab._y_combo.itemText(i)
                for i in range(self._tab._y_combo.count())
            }
            # Real columns discovered from the real table — both combos identical.
            self.assertEqual(x_items, y_items)
            # Blob columns carry their dimensionality suffix.
            self.assertEqual(x_items["times_blob"], "times_blob [1D]")
            self.assertEqual(x_items["h_blob"], "h_blob [2D]")
            self.assertEqual(x_items["max_h_blob"], "max_h_blob [1D]")
            # Numeric metadata columns carry the [num] suffix.
            self.assertEqual(x_items["n_cells"], "n_cells [num]")
            self.assertEqual(x_items["n_timesteps"], "n_timesteps [num]")
            # The run-identifying text column is listed too (real schema).
            self.assertIn("run_id", x_items)

            # Both combos have a selection → Plot is enabled.
            self.assertTrue(self._tab._plot_btn.isEnabled())

    def test_set_table_without_target_clears_combos(self):
        with make_temp_results_gpkg() as gpkg:
            self._tab.set_table(gpkg, _RESULTS_TABLE)
            self.assertGreater(self._tab._x_combo.count(), 0)
            self._tab.set_table("", "")
            self.assertEqual(self._tab._x_combo.count(), 0)
            self.assertEqual(self._tab._y_combo.count(), 0)
            self.assertFalse(self._tab._plot_btn.isEnabled())

    def test_selecting_2d_column_reveals_slice_controls(self):
        with make_temp_results_gpkg() as gpkg:
            self._tab.set_table(gpkg, _RESULTS_TABLE)
            _select_combo_by_data(self._tab._y_combo, "h_blob")
            self.assertTrue(self._tab._slice_widget.isVisibleTo(self._tab))
            # Slice range covers every stored timestep (n_timesteps - 1).
            self.assertEqual(self._tab._slice_spin.minimum(), 0)
            self.assertEqual(self._tab._slice_spin.maximum(), _N_TIMESTEPS - 1)

    @_skip_no_mpl
    def test_plot_line_curve_matches_stored_arrays(self):
        with make_temp_results_gpkg() as gpkg:
            self._tab.set_table(gpkg, _RESULTS_TABLE)
            _select_combo_by_data(self._tab._x_combo, "max_h_blob")
            _select_combo_by_data(self._tab._y_combo, "h_blob")
            self._tab._slice_spin.setValue(0)
            self._tab._plot_type_combo.setCurrentText("Line")
            self._tab._plot_btn.click()

            self.assertIsNotNone(self._tab._canvas)
            lines = self._tab._canvas.figure.axes[0].lines
            self.assertEqual(len(lines), 1)
            np.testing.assert_allclose(lines[0].get_xdata(), _MAX_H)
            np.testing.assert_allclose(lines[0].get_ydata(), _H[0, :])
            # Axis labels track the plotted columns.
            self.assertEqual(
                self._tab._canvas.figure.axes[0].get_xlabel(), "max_h_blob"
            )
            self.assertEqual(
                self._tab._canvas.figure.axes[0].get_ylabel(), "h_blob"
            )
            # Internal plot state records the selection (drives CSV export).
            self.assertEqual(self._tab._x_col, "max_h_blob")
            self.assertEqual(self._tab._y_col, "h_blob")

    @_skip_no_mpl
    def test_plot_2d_slice_spin_selects_timestep_row(self):
        with make_temp_results_gpkg() as gpkg:
            self._tab.set_table(gpkg, _RESULTS_TABLE)
            _select_combo_by_data(self._tab._x_combo, "max_hu_blob")
            _select_combo_by_data(self._tab._y_combo, "hu_blob")
            self._tab._plot_type_combo.setCurrentText("Line")
            for row in range(_N_TIMESTEPS):
                self._tab._slice_spin.setValue(row)
                self._tab._plot_btn.click()
                line = self._tab._canvas.figure.axes[0].lines[0]
                expected_hu = 0.01 * (row + 1) * np.arange(
                    1, _N_CELLS + 1, dtype=np.float64
                )
                np.testing.assert_allclose(line.get_ydata(), expected_hu)

    @_skip_no_mpl
    def test_grab_non_empty_after_plot(self):
        with make_temp_results_gpkg() as gpkg:
            self._tab.set_table(gpkg, _RESULTS_TABLE)
            _select_combo_by_data(self._tab._x_combo, "times_blob")
            _select_combo_by_data(self._tab._y_combo, "times_blob")
            self._tab._plot_btn.click()
            self._tab.resize(640, 480)
            self.assertTrue(grab_non_empty(self._tab))


@requires_qgis
class TestArrayViewerWidget(unittest.TestCase):
    """ArrayViewerWidget browsing real BLOB data (pattern P2)."""

    @classmethod
    def setUpClass(cls):
        ensure_qgis_app()

    def setUp(self):
        self._viewer = ArrayViewerWidget()

    def tearDown(self):
        delete_widgets_now(self._viewer)
        self._viewer = None

    def test_show_2d_blob_decodes_shape_dtype_and_cells(self):
        with make_temp_results_gpkg() as gpkg:
            h = _decode(gpkg, "h_blob")
            self.assertEqual(h.shape, (_N_TIMESTEPS, _N_CELLS))
            self.assertEqual(h.dtype, np.float64)

            self._viewer.show_array(h, "h")

            # Info label reports the decoded dtype and shape.
            self.assertEqual(
                self._viewer._info_lbl.text(),
                f"h: float64[{_N_TIMESTEPS}×{_N_CELLS}]",
            )
            # Slice controls visible with ranges matching the array dims.
            self.assertTrue(self._viewer._slice_widget.isVisibleTo(self._viewer))
            self.assertEqual(self._viewer._slice_row_spin.maximum(), _N_TIMESTEPS - 1)
            self.assertEqual(self._viewer._slice_col_spin.maximum(), _N_CELLS - 1)

            # Full 2D table: timesteps as rows, cells as columns.
            table = self._viewer._data_table
            self.assertEqual(table.rowCount(), _N_TIMESTEPS)
            self.assertEqual(table.columnCount(), _N_CELLS)
            for i in range(_N_TIMESTEPS):
                for j in range(_N_CELLS):
                    self.assertEqual(
                        table.item(i, j).text(), f"{h[i, j]:.6g}"
                    )

    def test_row_slice_navigation_updates_displayed_array(self):
        with make_temp_results_gpkg() as gpkg:
            h = _decode(gpkg, "h_blob")
            self._viewer.show_array(h, "h")

            # Move to timestep row 1 → table switches to that 1D slice.
            self._viewer._slice_row_spin.setValue(1)
            table = self._viewer._data_table
            self.assertEqual(table.rowCount(), _N_CELLS)
            self.assertEqual(table.columnCount(), 2)
            self.assertEqual(
                table.horizontalHeaderItem(1).text(), "h [0]"
            )
            for j in range(_N_CELLS):
                self.assertEqual(table.item(j, 0).text(), str(j))
                self.assertEqual(table.item(j, 1).text(), f"{h[1, j]:.6g}")

            # Column spin changes the displayed slice label (production
            # contract: row spin picks the data row, col spin labels it).
            self._viewer._slice_col_spin.setValue(2)
            self.assertEqual(table.horizontalHeaderItem(1).text(), "h [2]")
            self.assertEqual(table.item(0, 1).text(), f"{h[1, 0]:.6g}")

    def test_get_slice_for_plot_tracks_row_spin(self):
        with make_temp_results_gpkg() as gpkg:
            h = _decode(gpkg, "h_blob")
            self._viewer.show_array(h, "h")
            for row in range(_N_TIMESTEPS):
                self._viewer._slice_row_spin.setValue(row)
                np.testing.assert_array_equal(
                    self._viewer.get_slice_for_plot(), h[row, :]
                )

            times = _decode(gpkg, "times_blob")
            self._viewer.show_array(times, "times")
            np.testing.assert_array_equal(
                self._viewer.get_slice_for_plot(), times
            )

            self._viewer.show_array(None)
            self.assertIsNone(self._viewer.get_slice_for_plot())

    def test_show_1d_blob_populates_index_value_table(self):
        with make_temp_results_gpkg() as gpkg:
            times = _decode(gpkg, "times_blob")
            self.assertEqual(times.shape, (_N_TIMESTEPS,))

            self._viewer.show_array(times, "times")
            self.assertEqual(
                self._viewer._info_lbl.text(), f"times: float64[{_N_TIMESTEPS}]"
            )
            # 1D arrays hide the slice controls.
            self.assertFalse(self._viewer._slice_widget.isVisibleTo(self._viewer))
            table = self._viewer._data_table
            self.assertEqual(table.columnCount(), 2)
            self.assertEqual(table.rowCount(), _N_TIMESTEPS)
            self.assertEqual(table.horizontalHeaderItem(0).text(), "Index")
            self.assertEqual(table.horizontalHeaderItem(1).text(), "times")
            for i in range(_N_TIMESTEPS):
                self.assertEqual(table.item(i, 0).text(), str(i))
                self.assertEqual(table.item(i, 1).text(), f"{times[i]:.6g}")

    def test_show_none_clears_viewer(self):
        with make_temp_results_gpkg() as gpkg:
            self._viewer.show_array(_decode(gpkg, "h_blob"), "h")
            self._viewer.show_array(None)
            self.assertEqual(self._viewer._info_lbl.text(), "No array selected")
            self.assertEqual(self._viewer._data_table.rowCount(), 0)
            self.assertEqual(self._viewer._data_table.columnCount(), 0)
            self.assertFalse(
                self._viewer._slice_widget.isVisibleTo(self._viewer)
            )

    def test_show_combined_prepends_index_column(self):
        with make_temp_results_gpkg() as gpkg:
            times = _decode(gpkg, "times_blob")
            combined = np.column_stack([times, times * 2.0])
            self._viewer.show_combined(["times", "doubled"], combined)

            self.assertEqual(
                self._viewer._info_lbl.text(),
                f"Combined: times[{_N_TIMESTEPS}], doubled[{_N_TIMESTEPS}]",
            )
            table = self._viewer._data_table
            self.assertEqual(table.columnCount(), 3)
            self.assertEqual(table.rowCount(), _N_TIMESTEPS)
            self.assertEqual(table.horizontalHeaderItem(0).text(), "Index")
            self.assertEqual(table.horizontalHeaderItem(1).text(), "times")
            self.assertEqual(table.horizontalHeaderItem(2).text(), "doubled")
            for i in range(_N_TIMESTEPS):
                self.assertEqual(table.item(i, 0).text(), str(i))
                self.assertEqual(table.item(i, 1).text(), f"{combined[i, 0]:.6g}")
                self.assertEqual(table.item(i, 2).text(), f"{combined[i, 1]:.6g}")

            # A 1D payload falls back to plain show_array (production path).
            self._viewer.show_combined(["times"], times)
            self.assertEqual(
                self._viewer._info_lbl.text(), f"times: float64[{_N_TIMESTEPS}]"
            )

            # Empty input clears the viewer.
            self._viewer.show_combined([], None)
            self.assertEqual(self._viewer._info_lbl.text(), "No array selected")

    def test_grab_non_empty_with_array_shown(self):
        with make_temp_results_gpkg() as gpkg:
            self._viewer.show_array(_decode(gpkg, "h_blob"), "h")
            self._viewer.resize(480, 360)
            self.assertTrue(grab_non_empty(self._viewer))


if __name__ == "__main__":
    unittest.main()
