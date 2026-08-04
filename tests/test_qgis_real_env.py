"""Self-tests for the shared real-QGIS harness fixture helpers.

Every helper in ``tests/qgis_real_env.py`` is exercised once through the
production code paths it wraps, per
``docs/specs/2026-08-02-gui-test-coverage-design.md`` §4.
"""

import os
import sqlite3
import unittest

from tests.qgis_real_env import (
    delete_widgets_now,
    ensure_qgis_app,
    grab_non_empty,
    make_memory_layer,
    make_temp_model_gpkg,
    make_temp_results_gpkg,
    requires_qgis,
)


@requires_qgis
class TestMakeMemoryLayer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ensure_qgis_app()

    def test_valid_layer_feature_count_and_fields(self):
        from qgis.PyQt.QtCore import QVariant

        layer = make_memory_layer(
            geometry="Point",
            fields=[("id", QVariant.Int), ("name", QVariant.String)],
            features=[
                ("POINT(1 1)", (1, "alpha")),
                ("POINT(2 2)", (2, "beta")),
            ],
        )
        self.assertTrue(layer.isValid())
        self.assertEqual(layer.featureCount(), 2)
        self.assertEqual([f.name() for f in layer.fields()], ["id", "name"])
        attrs = sorted(f.attribute("name") for f in layer.getFeatures())
        self.assertEqual(attrs, ["alpha", "beta"])

    def test_real_geometry_predicate(self):
        from qgis.core import QgsGeometry

        polys = make_memory_layer(
            geometry="Polygon",
            features=[("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))", ())],
        )
        points = make_memory_layer(
            geometry="Point",
            features=[("POINT(5 5)", ()), ("POINT(50 50)", ())],
        )
        poly_geom = next(polys.getFeatures()).geometry()
        self.assertIsInstance(poly_geom, QgsGeometry)
        inside, outside = [f.geometry() for f in points.getFeatures()]
        self.assertTrue(poly_geom.contains(inside))
        self.assertFalse(poly_geom.contains(outside))

    def test_misuse_raises_loudly(self):
        from qgis.PyQt.QtCore import QVariant

        with self.assertRaises(RuntimeError):
            make_memory_layer(geometry="NotAGeometry")
        with self.assertRaises(ValueError):
            make_memory_layer(features=[("POINT(NOT WKT)", ())])
        with self.assertRaises(ValueError):
            make_memory_layer(
                fields=[("id", QVariant.Int)],
                features=[("POINT(0 0)", (1, "too", "many"))],
            )


@requires_qgis
class TestMakeTempModelGpkg(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ensure_qgis_app()

    def test_opens_with_production_loader(self):
        from swe2d.workbench.services.model_gpkg_loader_service import (
            get_model_gpkg_layer_names,
            load_layers_from_gpkg,
        )

        with make_temp_model_gpkg() as path:
            self.assertTrue(os.path.exists(path))
            names = get_model_gpkg_layer_names()
            self.assertGreaterEqual(len(names), 18)
            layers = load_layers_from_gpkg(path)
            self.assertEqual(
                sorted(layers), sorted(names),
                "production loader must return every canonical model layer",
            )
            for lname, lyr in layers.items():
                self.assertTrue(lyr.isValid(), f"layer {lname} invalid")

    def test_expected_core_tables_present(self):
        with make_temp_model_gpkg() as path:
            conn = sqlite3.connect(path)
            try:
                tables = {
                    r[0]
                    for r in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                }
            finally:
                conn.close()
        # OGC metadata tables
        for core in ("gpkg_contents", "gpkg_geometry_columns", "gpkg_spatial_ref_sys"):
            self.assertIn(core, tables)
        # Canonical model tables (written under their display names).
        for model_table in ("SWE2D_Topo_Nodes", "SWE2D_Topo_Arcs", "SWE2D_Structures"):
            self.assertIn(model_table, tables)


@requires_qgis
class TestMakeTempResultsGpkg(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ensure_qgis_app()

    def test_production_reader_returns_written_run(self):
        import numpy as np
        from swe2d.services.gpkg_persistence_service import (
            collect_baked_runs_from_gpkg,
            load_baked_snapshot,
            load_baked_timesteps,
        )

        with make_temp_results_gpkg(n_cells=4, n_timesteps=3) as path:
            runs = collect_baked_runs_from_gpkg(path)
            self.assertEqual(len(runs), 1)
            run = runs[0]
            self.assertEqual(run["n_cells"], 4)
            self.assertEqual(run["n_timesteps"], 3)
            run_id = run["run_id"]

            times = load_baked_timesteps(path, run_id)
            self.assertEqual(times.shape, (3,))
            np.testing.assert_allclose(times, [0.0, 10.0, 20.0])

            snap = load_baked_snapshot(path, run_id, 10.0)
            self.assertIsNotNone(snap)
            self.assertEqual(snap["t_s"], 10.0)
            for key in ("h", "hu", "hv"):
                self.assertEqual(snap[key].shape, (4,))
            # Readback matches the deterministic fixture values exactly.
            np.testing.assert_allclose(
                snap["h"], 1.0 + 0.1 * 1 + np.linspace(0.0, 0.5, 4)
            )
            np.testing.assert_allclose(snap["hu"], 0.01 * 2 * np.arange(1, 5))

    def test_misuse_raises_loudly(self):
        with self.assertRaises(ValueError):
            with make_temp_results_gpkg(n_cells=0):
                pass
        with self.assertRaises(ValueError):
            with make_temp_results_gpkg(n_timesteps=-1):
                pass


@requires_qgis
class TestGrabNonEmpty(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ensure_qgis_app()

    def test_content_widget_is_non_empty(self):
        from qgis.PyQt.QtWidgets import QLabel

        label = QLabel("HYDRA depth 1.23 m")
        label.resize(200, 40)
        try:
            self.assertTrue(grab_non_empty(label))
        finally:
            delete_widgets_now(label)

    def test_blank_widget_is_empty(self):
        from qgis.PyQt.QtWidgets import QWidget

        blank = QWidget()
        blank.resize(200, 40)
        try:
            self.assertFalse(grab_non_empty(blank))
        finally:
            delete_widgets_now(blank)

    def test_misuse_raises_loudly(self):
        with self.assertRaises(ValueError):
            grab_non_empty(None)
        from qgis.PyQt.QtWidgets import QWidget

        zero = QWidget()  # never resized → 0x0 (or 640x480 default? verify)
        zero.setFixedSize(0, 0)
        try:
            with self.assertRaises(ValueError):
                grab_non_empty(zero)
        finally:
            delete_widgets_now(zero)


@requires_qgis
class TestDeleteWidgetsNow(unittest.TestCase):
    """Self-test for ``delete_widgets_now`` (offscreen QPA delivery).

    ``delete_widgets_now`` is the test-harness replacement for bare
    ``deleteLater()`` because under the offscreen QPA,
    ``QApplication.processEvents()`` alone does NOT deliver
    ``DeferredDelete`` events — torn-down widgets stay alive (and visible)
    and leak into subsequent test files.  This test pins the contract:

    1. ``delete_widgets_now(w)`` delivers ``DeferredDelete`` *synchronously*
       so the wrapper's C++ object is dead before the call returns.
    2. ``None`` entries are skipped silently (matches the helper's docstring).
    3. A batch of widgets all reach the dead state in a single call.
    """

    @classmethod
    def setUpClass(cls):
        ensure_qgis_app()

    def _is_dead(self, widget) -> bool:
        """C++ object identity check via the prescribed objectName liveness
        pattern (AGENTS.md PyQt5 widget liveness rule)."""
        try:
            widget.objectName()
        except RuntimeError:
            return True
        return False

    def test_single_widget_is_destroyed_before_return(self):
        from qgis.PyQt.QtWidgets import QWidget

        w = QWidget()
        w.setObjectName("delete_widgets_now_single")
        delete_widgets_now(w)
        self.assertTrue(
            self._is_dead(w),
            "delete_widgets_now must deliver DeferredDelete synchronously; "
            "wrapper is still alive after the call",
        )

    def test_batch_destroyed_in_single_call(self):
        from qgis.PyQt.QtWidgets import QWidget

        widgets = [QWidget() for _ in range(5)]
        for i, w in enumerate(widgets):
            w.setObjectName(f"delete_widgets_now_batch_{i}")
        delete_widgets_now(*widgets)
        for w in widgets:
            self.assertTrue(
                self._is_dead(w),
                "every widget in the batch must reach the dead state "
                "after a single delete_widgets_now() call",
            )

    def test_none_entries_skipped_silently(self):
        from qgis.PyQt.QtWidgets import QWidget

        w = QWidget()
        w.setObjectName("delete_widgets_now_none_test")
        # Mix a None in; it must be skipped without raising.
        delete_widgets_now(None, w, None)
        self.assertTrue(self._is_dead(w))

    def test_no_widgets_is_noop(self):
        # No widgets passed — must not raise, must not crash.
        delete_widgets_now()


if __name__ == "__main__":
    unittest.main()
