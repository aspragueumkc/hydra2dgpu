"""Regression tests for QGIS WKB enum handling in internal flow geometry helpers."""
from __future__ import annotations

import unittest

import numpy as np


class TestInternalFlowQGISGeometry(unittest.TestCase):
    """Ensure QgsWkbTypes geometryType receives an enum, not a raw int."""

    @classmethod
    def setUpClass(cls):
        try:
            from qgis.core import QgsApplication
            app = QgsApplication.instance()
            if app is None:
                cls._app = QgsApplication([], False)
                cls._app.initQgis()
            else:
                cls._app = None
        except Exception:
            cls._app = None

    @classmethod
    def tearDownClass(cls):
        if cls._app is not None:
            cls._app.exitQgis()

    def setUp(self):
        try:
            from qgis.core import (
                QgsGeometry, QgsPointXY, QgsWkbTypes,
            )
            self._qgis_ok = True
            self.QgsGeometry = QgsGeometry
            self.QgsPointXY = QgsPointXY
            self.QgsWkbTypes = QgsWkbTypes
        except Exception:
            self._qgis_ok = False

    def test_polygon_maps_indices_with_qgis_wkb_enum(self):
        if not self._qgis_ok:
            self.skipTest("QGIS core not available")

        from swe2d.boundary_and_forcing.internal_flow_qgis_geometry import (
            internal_flow_geom_to_indices_weights_qgis,
        )

        geom = self.QgsGeometry.fromWkt(
            "POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))"
        )
        cx = np.array([2.0, 5.0, 12.0], dtype=np.float64)
        cy = np.array([2.0, 5.0, 5.0], dtype=np.float64)

        result = internal_flow_geom_to_indices_weights_qgis(
            geom,
            cx,
            cy,
            qgs_wkb_types=self.QgsWkbTypes,
            qgs_geometry_cls=self.QgsGeometry,
            qgs_pointxy_cls=self.QgsPointXY,
        )

        self.assertIsNotNone(result)
        idx, wt = result
        self.assertEqual(set(int(i) for i in idx), {0, 1})
        self.assertEqual(len(wt), len(idx))


if __name__ == "__main__":
    unittest.main()
