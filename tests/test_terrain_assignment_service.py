"""Tests for the pure-numpy terrain/elevation interpolation functions.

CLI path uses these directly; QGIS GUI path uses the adapter in
swe2d.services.qgis_terrain_interpolator which wraps the same inputs.
"""
import unittest

import numpy as np

from swe2d.services.terrain_assignment_service import (
    assign_node_z_from_layer_features,
    idw_interpolate_points,
    sample_raster_at_nodes,
)


class TestSampleRasterAtNodes(unittest.TestCase):
    def test_basic_sampling(self):
        # 3x3 raster with origin at (0, 0), pixel size 1m.
        # Raster values 1..9 left-to-right, top-to-bottom.
        raster = np.arange(1, 10, dtype=np.float64).reshape(3, 3)
        geo = (0.0, 1.0, 0.0, 3.0, 0.0, -1.0)  # origin (0,3), px=1, py=-1
        # Sampling at cell centers: (0.5, 2.5) -> row 0 col 0 -> 1
        # (1.5, 1.5) -> row 1 col 1 -> 5
        # (2.5, 0.5) -> row 2 col 2 -> 9
        nodes_x = np.array([0.5, 1.5, 2.5])
        nodes_y = np.array([2.5, 1.5, 0.5])
        z = sample_raster_at_nodes(nodes_x, nodes_y, raster, geo, default_z=-1.0)
        np.testing.assert_array_equal(z, [1.0, 5.0, 9.0])

    def test_outside_extent_returns_default(self):
        raster = np.ones((2, 2), dtype=np.float64)
        geo = (0.0, 1.0, 0.0, 2.0, 0.0, -1.0)
        nodes_x = np.array([10.0, -1.0])
        nodes_y = np.array([0.0, 0.0])
        z = sample_raster_at_nodes(nodes_x, nodes_y, raster, geo, default_z=-99.0)
        np.testing.assert_array_equal(z, [-99.0, -99.0])


class TestIdwInterpolatePoints(unittest.TestCase):
    def test_colocated_returns_source_z(self):
        node_x = np.array([1.0])
        node_y = np.array([0.0])
        point_x = np.array([0.0, 1.0, 2.0])
        point_y = np.array([0.0, 0.0, 0.0])
        point_z = np.array([10.0, 20.0, 30.0])
        z = idw_interpolate_points(node_x, node_y, point_x, point_y, point_z)
        self.assertAlmostEqual(z[0], 20.0)

    def test_empty_source_returns_default(self):
        node_x = np.array([1.0])
        node_y = np.array([0.0])
        z = idw_interpolate_points(
            node_x, node_y,
            np.array([]), np.array([]), np.array([]),
            default_z=42.0,
        )
        self.assertAlmostEqual(z[0], 42.0)

    def test_idw_weighted_average(self):
        # One node at (1, 0), sources at x=0 and x=2.
        # Distances: d2=1, d2=1 -> weights equal -> z = (10+30)/2 = 20
        node_x = np.array([1.0])
        node_y = np.array([0.0])
        point_x = np.array([0.0, 2.0])
        point_y = np.array([0.0, 0.0])
        point_z = np.array([10.0, 30.0])
        z = idw_interpolate_points(node_x, node_y, point_x, point_y, point_z)
        self.assertAlmostEqual(z[0], 20.0)

    def test_closer_source_dominates(self):
        # Node at (1, 0), sources at x=0 (z=0) and x=10 (z=100).
        # Distances 1 and 81 -> weights 1 and 1/81 -> z ≈ 0.012...
        node_x = np.array([1.0])
        node_y = np.array([0.0])
        point_x = np.array([0.0, 10.0])
        point_y = np.array([0.0, 0.0])
        point_z = np.array([0.0, 100.0])
        z = idw_interpolate_points(node_x, node_y, point_x, point_y, point_z)
        self.assertLess(z[0], 1.5)  # very close to 0
        self.assertGreater(z[0], 0.0)

    def test_chunked_consistency(self):
        # Many nodes, ensure chunking produces the same answer as no chunking
        np.random.seed(0)
        point_x = np.random.uniform(0, 100, 500)
        point_y = np.random.uniform(0, 100, 500)
        point_z = np.random.uniform(0, 10, 500)
        node_x = np.random.uniform(0, 100, 2000)
        node_y = np.random.uniform(0, 100, 2000)

        z_chunked = idw_interpolate_points(node_x, node_y,
                                            point_x, point_y, point_z)
        # Spot-check a few values against a manual 4-NN compute
        for i in [0, 100, 1500, 1999]:
            d2 = (point_x - node_x[i]) ** 2 + (point_y - node_y[i]) ** 2
            k = 4
            idx = np.argpartition(d2, k - 1)[:k]
            d2n = d2[idx]
            zn = point_z[idx]
            zero = d2n[0] == 0.0
            if zero:
                expected = zn[0]
            else:
                w = 1.0 / np.maximum(d2n, 1e-30) ** 2
                expected = (w * zn).sum() / w.sum()
            self.assertAlmostEqual(z_chunked[i], expected, places=10)


class TestAssignNodeZFromLayerFeatures(unittest.TestCase):
    def test_updates_node_z_in_place(self):
        node_z = np.array([0.0, 0.0, 0.0, 0.0])
        features = [
            {"node_id": 0, "bed_z": 100.0},
            {"node_id": 2, "bed_z": 200.0},
            {"node_id": 99, "bed_z": 999.0},  # out-of-range -> ignored
        ]
        updated = assign_node_z_from_layer_features(node_z, features)
        self.assertEqual(updated, 2)
        np.testing.assert_array_equal(node_z, [100.0, 0.0, 200.0, 0.0])


if __name__ == "__main__":
    unittest.main()