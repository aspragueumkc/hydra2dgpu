"""Tests for swe2d.workbench.mesh_service — pure numpy mesh computation, no Qt."""

import unittest
import numpy as np

from swe2d.core.mesh_service import assign_node_z_from_terrain


# ---------------------------------------------------------------------------
# assign_node_z_from_terrain
# ---------------------------------------------------------------------------


class TestAssignNodeZFromTerrain(unittest.TestCase):
    def _simple_raster(self):
        raster_data = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ], dtype=np.float64)
        # origin=(0, 3), dx=1, dy=-1 => pixel (col,row) = (x, 3-y)
        transform = (0.0, 1.0, 0.0, 3.0, 0.0, -1.0)
        return raster_data, transform

    def test_samples_at_node_coordinates(self):
        raster, transform = self._simple_raster()
        node_coords = np.array([
            [0.5, 2.5],  # col=0, row=0 => 1.0
            [1.5, 1.5],  # col=1, row=1 => 5.0
            [2.5, 0.5],  # col=2, row=2 => 9.0
        ], dtype=np.float64)
        result = assign_node_z_from_terrain(node_coords, raster, transform)
        expected = np.array([1.0, 5.0, 9.0], dtype=np.float64)
        np.testing.assert_allclose(result, expected)

    def test_outside_extent_uses_default(self):
        raster, transform = self._simple_raster()
        node_coords = np.array([
            [10.0, 10.0],
            [-5.0, -5.0],
        ], dtype=np.float64)
        result = assign_node_z_from_terrain(
            node_coords, raster, transform, default_z=-999.0,
        )
        expected = np.array([-999.0, -999.0], dtype=np.float64)
        np.testing.assert_allclose(result, expected)

    def test_mixed_inside_and_outside(self):
        raster, transform = self._simple_raster()
        node_coords = np.array([
            [0.5, 2.5],   # inside => 1.0
            [100.0, 0.0], # outside => default
            [2.5, 0.5],   # inside => 9.0
        ], dtype=np.float64)
        result = assign_node_z_from_terrain(
            node_coords, raster, transform, default_z=0.0,
        )
        expected = np.array([1.0, 0.0, 9.0], dtype=np.float64)
        np.testing.assert_allclose(result, expected)

    def test_empty_nodes_returns_empty(self):
        raster, transform = self._simple_raster()
        node_coords = np.empty((0, 2), dtype=np.float64)
        result = assign_node_z_from_terrain(node_coords, raster, transform)
        self.assertEqual(result.shape, (0,))
        self.assertEqual(result.dtype, np.float64)

    def test_nearest_neighbor_sampling(self):
        raster_data = np.array([
            [10.0, 20.0],
            [30.0, 40.0],
        ], dtype=np.float64)
        transform = (0.0, 2.0, 0.0, 4.0, 0.0, -2.0)
        # pixel (0,0) covers x=[0,2), y=[2,4)
        node_coords = np.array([
            [0.1, 3.9],  # col=0, row=0 => 10.0
            [0.1, 2.1],  # col=0, row=0 => 10.0
            [1.9, 3.9],  # col=0, row=0 => 10.0
            [2.1, 3.9],  # col=1, row=0 => 20.0
        ], dtype=np.float64)
        result = assign_node_z_from_terrain(node_coords, raster_data, transform, default_z=-1.0)
        expected = np.array([10.0, 10.0, 10.0, 20.0], dtype=np.float64)
        np.testing.assert_allclose(result, expected)


# NOTE: TestBuildLineSamplingMap and TestSampleLineMetrics classes were
# removed in Task 8 of the canonical sample-line sampling plan. They
# targeted ``build_line_sampling_map_numpy`` and ``sample_line_metrics``,
# both of which were legacy duplicate builders superseded by
# ``build_canonical_line_sampling_map``. Equivalent coverage lives in
# ``tests/test_sample_line_canonical.py`` and
# ``tests/test_swe2d_gpu_line_flow_reference.py``.


if __name__ == "__main__":
    unittest.main()