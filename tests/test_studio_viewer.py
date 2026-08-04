#!/usr/bin/env python3
"""Behavioral tests for the HYDRA2D View viewer stack.

Covers (Task C.2, spec docs/specs/2026-08-02-gui-test-coverage-design.md §3-§4,
pattern P2):
  - swe2d/workbench/views/studio_viewer.py      (SWE2DStudioViewer)
  - swe2d/workbench/views/studio_viewer_plot.py (PlotViewWidget)

Characterization tests against the real production contract, driven through
real widgets under headless QGIS (tests/qgis_real_env.py).  No mocks for
Qgs* types; no synthetic PyQt5.

Production contract notes (discovered while writing these tests):
  - ``SWE2DStudioViewer`` owns three plot tabs (Mesh / Time Series /
    Profile).  It has no run-loading or timestep controls of its own —
    run loading and timestep stepping live in ``SWE2DResultsData``
    (``swe2d/results/data.py``); the viewer merely fans
    ``set_mesh_data``/``set_result_data``/``set_h_min`` out to its plot
    widgets.  Timestep assertions therefore read the shared data object's
    state and the Time Series widget's displayed vline, per the real
    wiring (``PGTimeSeriesWidget.refresh`` positions its ``_vline`` from
    ``result_data.current_time_sec``).
  - ``PlotViewWidget.set_render_fn`` stores a callback that ``refresh()``
    never invokes — ``refresh()`` always calls
    ``swe2d.plotting.viewer_plots.render_viewer_figure``.  The callback is
    dead surface area; tests characterize the real path.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

# Ensure repo root and build dir are on sys.path for all discovery modes
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BUILD_DIR = os.path.join(_REPO_ROOT, "build")
for _p in (_REPO_ROOT, _BUILD_DIR):
    if _p not in sys.path and os.path.isdir(_p):
        sys.path.insert(0, _p)

from tests.qgis_real_env import (
    delete_widgets_now,
    ensure_qgis_app,
    grab_non_empty,
    make_temp_results_gpkg,
    requires_qgis,
)

# Qt binding pinning: both pyqtgraph and matplotlib's ``backend_qt5agg``
# resolve their Qt binding at import time, and both prefer PySide6 when it
# is installed (it is, in qgis_stable, as a transitive dep).  Production
# code parents their widgets under real PyQt5 QGIS widgets, so a PySide6
# pick crashes at runtime.  QGIS itself is a PyQt5 app — pin both to PyQt5
# to mirror the real runtime; this is not a test shim.
os.environ.setdefault("QT_API", "pyqt5")
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt5")

try:  # loud skip marker only — never a silent fallback
    import pyqtgraph  # noqa: F401

    _HAVE_PG = True
except ImportError:
    _HAVE_PG = False


def _triangle_mesh() -> dict:
    """Minimal real mesh dict satisfying the mesh_render_service contract."""
    return {
        "node_x": np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
        "node_y": np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
        "cell_nodes": np.asarray([[0, 1, 2]], dtype=np.int32),
    }


@requires_qgis
class TestSWE2DStudioViewer(unittest.TestCase):
    """SWE2DStudioViewer — construct, load run, timestep stepping (P2)."""

    @classmethod
    def setUpClass(cls) -> None:
        ensure_qgis_app()

    def setUp(self) -> None:
        from swe2d.workbench.views.studio_viewer import SWE2DStudioViewer

        self.viewer = SWE2DStudioViewer()
        self.viewer.resize(640, 480)

    def tearDown(self) -> None:
        delete_widgets_now(self.viewer)
        self.viewer = None

    # ------------------------------------------------------------------

    def test_construct_builds_three_plot_tabs(self) -> None:
        tabs = self.viewer.tab_widget
        self.assertEqual(tabs.count(), 3)
        self.assertEqual(
            [tabs.tabText(i) for i in range(tabs.count())],
            ["Mesh", "Time Series", "Profile"],
        )
        widgets = self.viewer.plot_widgets
        self.assertEqual(
            sorted(widgets.keys()), ["Mesh", "Profile", "Time Series"]
        )
        for mode in widgets:
            self.assertIs(tabs.widget(list(widgets).index(mode)), widgets[mode])
        self.assertIs(self.viewer.current_widget, tabs.currentWidget())

    def test_set_mesh_data_propagates_to_all_tabs(self) -> None:
        mesh = _triangle_mesh()
        self.viewer.set_mesh_data(mesh)
        for mode, w in self.viewer.plot_widgets.items():
            self.assertIs(
                w._mesh_data, mesh, f"tab {mode!r} did not receive mesh data"
            )

    def test_set_h_min_propagates_to_all_tabs(self) -> None:
        self.viewer.set_h_min(0.01)
        for mode, w in self.viewer.plot_widgets.items():
            self.assertAlmostEqual(
                w._h_min, 0.01, msg=f"tab {mode!r} did not receive h_min"
            )

    def test_load_results_run_and_step_timesteps(self) -> None:
        """Load a real results GPKG through the production load path.

        Run discovery: ``SWE2DResultsData.add_results_files`` →
        ``collect_runs_from_gpkg`` (production reader); timestep union via
        ``_rebuild_timestep_union`` exactly as ``studio_results_panel``
        does after adding files.
        """
        from swe2d.results.data import SWE2DResultsData

        with make_temp_results_gpkg(n_cells=4, n_timesteps=3) as gpkg:
            data = SWE2DResultsData()
            added_paths, added_runs = data.add_results_files([gpkg])
            self.assertEqual(added_paths, 1)
            self.assertEqual(added_runs, 1)
            data._rebuild_timestep_union()

            # The run appears: 3 deterministic timesteps at 0/10/20 s.
            self.assertEqual(data.frame_count, 3)
            np.testing.assert_allclose(data.all_timesteps, [0.0, 10.0, 20.0])
            enabled = data.get_enabled_run_records()
            self.assertEqual(len(enabled), 1)
            self.assertEqual(enabled[0].run_id, "hydra_test_run")

            # Hand the run to the viewer — every tab gets the same object.
            self.viewer.set_result_data(data)
            ts_widget = self.viewer.plot_widgets["Time Series"]
            for mode, w in self.viewer.plot_widgets.items():
                self.assertIs(
                    w._result_data,
                    data,
                    f"tab {mode!r} did not receive result data",
                )

            if not _HAVE_PG:
                self.skipTest(
                    "pyqtgraph not available — Time Series vline assertion "
                    "requires the pg widget"
                )

            # Initial frame: vline sits at t=0 hr.
            ts_widget.refresh()
            self.assertIsNotNone(ts_widget._vline)
            self.assertAlmostEqual(ts_widget._vline.value(), 0.0)

            # Step the timestep control (the data-layer API the temporal
            # dock drives) → the displayed time indicator moves to the new
            # timestep.  Data API assertion, not pixels.
            data.set_current_time(10.0)
            self.assertAlmostEqual(data.current_time_sec, 10.0)
            ts_widget.refresh()
            self.assertAlmostEqual(
                ts_widget._vline.value(),
                10.0 / 3600.0,
                msg="Time Series vline did not move to the stepped timestep",
            )

            data.step_forward()
            self.assertAlmostEqual(data.current_time_sec, 20.0)
            ts_widget.refresh()
            self.assertAlmostEqual(ts_widget._vline.value(), 20.0 / 3600.0)

    def test_tab_switch_refreshes_and_grab_non_empty(self) -> None:
        self.viewer.set_mesh_data(_triangle_mesh())
        self.viewer.refresh()  # viewer-level fan-out must not raise
        self.viewer.show()
        for idx in range(self.viewer.tab_widget.count()):
            self.viewer.tab_widget.setCurrentIndex(idx)
        # Back to Mesh tab: wireframe rendered by the real render service.
        self.viewer.tab_widget.setCurrentIndex(0)
        mesh_widget = self.viewer.plot_widgets["Mesh"]
        self.assertTrue(
            grab_non_empty(mesh_widget),
            "Mesh tab produced an empty grab after rendering real mesh data",
        )


@requires_qgis
class TestPlotViewWidget(unittest.TestCase):
    """PlotViewWidget — construct, render real data, table toggle (P2)."""

    @classmethod
    def setUpClass(cls) -> None:
        ensure_qgis_app()

    def setUp(self) -> None:
        from swe2d.workbench.views.studio_viewer_plot import (
            PlotViewWidget,
            _HAVE_MPL,
        )

        if not _HAVE_MPL:
            self.skipTest("matplotlib Qt backend not available")
        self.widget = PlotViewWidget(mode="Mesh")
        self.widget.resize(640, 480)

    def tearDown(self) -> None:
        if self.widget is not None:
            delete_widgets_now(self.widget)
            self.widget = None

    # ------------------------------------------------------------------

    def test_construct(self) -> None:
        self.assertEqual(self.widget.mode, "Mesh")
        self.assertIsNotNone(self.widget.fig)
        self.assertIsNotNone(self.widget.canvas)
        self.assertIsNotNone(self.widget.show_table_toggle)
        self.assertFalse(self.widget.show_table_toggle.isChecked())
        self.assertTrue(self.widget._table_widget.isHidden())
        # selected_metric round-trips through its property
        self.assertEqual(self.widget.selected_metric, "flow")
        self.widget.selected_metric = "depth"
        self.assertEqual(self.widget.selected_metric, "depth")

    def test_refresh_renders_mesh_wireframe(self) -> None:
        self.widget.set_data(mesh_data=_triangle_mesh())
        self.widget.refresh()
        axes = self.widget.fig.axes
        self.assertEqual(len(axes), 1, "refresh must produce exactly one axes")
        # triplot draws one Line2D per triangle edge group — real line art,
        # asserted through the figure's data API (not pixels).
        self.assertGreater(
            len(axes[0].lines), 0, "mesh wireframe produced no line art"
        )
        self.assertEqual(axes[0].get_title(), "Generated mesh")
        self.widget.show()
        self.assertTrue(
            grab_non_empty(self.widget),
            "PlotViewWidget grab is empty after rendering real mesh data",
        )

    def test_refresh_without_mesh_shows_placeholder(self) -> None:
        self.widget.refresh()
        axes = self.widget.fig.axes
        self.assertEqual(len(axes), 1)
        texts = [t.get_text() for t in axes[0].texts]
        self.assertIn("No mesh loaded", texts)
        self.assertEqual(len(axes[0].lines), 0)

    def test_table_toggle_shows_and_hides_table(self) -> None:
        # The widget is never shown in this test, so assert the explicit
        # visibility flag (isHidden) rather than effective on-screen
        # visibility (isVisible), which is False for hidden ancestors.
        toggle = self.widget.show_table_toggle
        toggle.setChecked(True)
        self.assertFalse(self.widget._table_widget.isHidden())
        # Mesh mode populates nothing (production early-return).
        self.assertEqual(self.widget._table_widget.rowCount(), 0)
        toggle.setChecked(False)
        self.assertTrue(self.widget._table_widget.isHidden())


if __name__ == "__main__":
    unittest.main()
