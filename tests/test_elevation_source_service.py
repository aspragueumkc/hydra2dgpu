"""Regression tests for elevation source assignment service."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import numpy as np


class TestElevationSourceService(unittest.TestCase):
    """Ensure mesh node_z is assigned from raster/PointZ elevation sources."""

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
                QgsFeature,
                QgsGeometry,
                QgsPoint,
                QgsVectorLayer,
            )
            self._qgis_ok = True
            self.QgsFeature = QgsFeature
            self.QgsGeometry = QgsGeometry
            self.QgsPoint = QgsPoint
            self.QgsVectorLayer = QgsVectorLayer
        except Exception:
            self._qgis_ok = False

    def _make_point_z_layer(self, points):
        """Create an in-memory PointZ layer from a list of (x, y, z)."""
        layer = self.QgsVectorLayer("PointZ?crs=EPSG:4326", "z_points", "memory")
        provider = layer.dataProvider()
        features = []
        for x, y, z in points:
            feat = self.QgsFeature()
            feat.setGeometry(self.QgsGeometry(self.QgsPoint(x, y, z)))
            features.append(feat)
        provider.addFeatures(features)
        layer.updateExtents()
        return layer

    def test_assign_node_z_from_pointz_vector_layer(self):
        if not self._qgis_ok:
            self.skipTest("QGIS core not available")

        from swe2d.workbench.services.elevation_source_service import (
            assign_node_z_from_elevation_source,
        )

        layer = self._make_point_z_layer([
            (0.0, 0.0, 10.0),
            (1.0, 0.0, 20.0),
        ])
        mesh_data = {
            "node_x": np.array([0.0, 1.0, 0.5], dtype=np.float64),
            "node_y": np.array([0.0, 0.0, 0.0], dtype=np.float64),
        }

        ok = assign_node_z_from_elevation_source(mesh_data, layer)

        self.assertTrue(ok)
        self.assertIn("node_z", mesh_data)
        np.testing.assert_allclose(
            mesh_data["node_z"],
            [10.0, 20.0, 15.0],
        )

    def test_auto_assign_node_z_from_view_elevation_source(self):
        view = MagicMock()
        view.get_topo_elevation_layer_id.return_value = "layer-1"
        view._log = MagicMock()

        layer = MagicMock()
        layer.name.return_value = "DEM"

        with patch("qgis.core.QgsProject") as mock_project:
            mock_project.instance.return_value.mapLayer.return_value = layer
            with patch(
                "swe2d.workbench.services.elevation_source_service."
                "assign_node_z_from_elevation_source",
                return_value=True,
            ) as mock_assign:
                from swe2d.workbench.services.elevation_source_service import (
                    auto_assign_node_z_from_view_elevation_source,
                )
                mesh_data = {}
                ok = auto_assign_node_z_from_view_elevation_source(view, mesh_data)

        self.assertTrue(ok)
        view._log.assert_called_with(
            "Assigned node z from elevation source: DEM"
        )
        mock_assign.assert_called_once()


if __name__ == "__main__":
    unittest.main()
