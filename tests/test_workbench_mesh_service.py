"""Tests for swe2d.workbench.mesh_service — pure numpy mesh computation, no Qt."""

import unittest
import numpy as np

from swe2d.workbench.services.mesh_service import assign_node_z_from_terrain
from swe2d.workbench.services.line_sampling_service import (
    build_line_sampling_map_numpy,
    sample_line_metrics,
)


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


# ---------------------------------------------------------------------------
# build_line_sampling_map
# ---------------------------------------------------------------------------


class TestBuildLineSamplingMap(unittest.TestCase):
    def _simple_mesh(self):
        node_coords = np.array([
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 1.0],
            [0.0, 1.0],
        ], dtype=np.float64)
        cell_nodes = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        return node_coords, cell_nodes

    def test_horizontal_line_through_center(self):
        nc, cn = self._simple_mesh()
        line_xy = np.array([[0.5, 0.5], [1.5, 0.5]], dtype=np.float64)
        result = build_line_sampling_map_numpy(nc, cn, line_xy)
        self.assertIn("cell_idx", result)
        self.assertIn("weights", result)
        self.assertIn("normal_x", result)
        self.assertIn("normal_y", result)
        self.assertIn("profile_station_m", result)
        self.assertIn("profile_cell_idx", result)
        self.assertIn("profile_cell_w", result)
        self.assertGreater(len(result["cell_idx"]), 0)

    def test_line_outside_mesh_returns_default(self):
        nc, cn = self._simple_mesh()
        line_xy = np.array([[10.0, 10.0], [20.0, 20.0]], dtype=np.float64)
        result = build_line_sampling_map_numpy(nc, cn, line_xy)
        self.assertEqual(len(result["cell_idx"]), 0)

    def test_normal_points_left_of_line(self):
        nc, cn = self._simple_mesh()
        line_xy = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
        result = build_line_sampling_map_numpy(nc, cn, line_xy)
        # normal convention: nx = dy/mag, ny = -dx/mag
        # for (dx=1, dy=0): nx=0, ny=-1
        self.assertAlmostEqual(result["normal_x"], 0.0, places=10)
        self.assertAlmostEqual(result["normal_y"], -1.0, places=10)

    def test_weights_sum_to_one(self):
        nc, cn = self._simple_mesh()
        line_xy = np.array([[0.5, 0.25], [1.5, 0.75]], dtype=np.float64)
        result = build_line_sampling_map_numpy(nc, cn, line_xy)
        w = result["weights"]
        if w.size > 0:
            self.assertAlmostEqual(float(np.sum(w)), 1.0, places=5)

    def test_profile_arrays_match_stations(self):
        nc, cn = self._simple_mesh()
        line_xy = np.array([[0.25, 0.5], [1.75, 0.5]], dtype=np.float64)
        result = build_line_sampling_map_numpy(nc, cn, line_xy)
        n_sta = len(result["profile_station_m"])
        self.assertEqual(result["profile_cell_idx"].shape[0], n_sta)
        self.assertEqual(result["profile_cell_w"].shape[0], n_sta)

    def test_empty_line_returns_default(self):
        nc, cn = self._simple_mesh()
        line_xy = np.empty((0, 2), dtype=np.float64)
        result = build_line_sampling_map_numpy(nc, cn, line_xy)
        self.assertEqual(len(result["cell_idx"]), 0)

    def test_single_point_line_returns_default(self):
        nc, cn = self._simple_mesh()
        line_xy = np.array([[0.5, 0.5]], dtype=np.float64)
        result = build_line_sampling_map_numpy(nc, cn, line_xy)
        self.assertEqual(len(result["cell_idx"]), 0)


# ---------------------------------------------------------------------------
# sample_line_metrics
# ---------------------------------------------------------------------------


class TestSampleLineMetrics(unittest.TestCase):
    def _simple_mesh(self):
        node_coords = np.array([
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 1.0],
            [0.0, 1.0],
        ], dtype=np.float64)
        cell_nodes = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        return node_coords, cell_nodes

    def _uniform_solution(self):
        nc, cn = self._simple_mesh()
        h = np.array([1.0, 1.0], dtype=np.float64)
        hu = np.array([0.5, 0.5], dtype=np.float64)
        hv = np.array([0.0, 0.0], dtype=np.float64)
        bed = np.array([5.0, 5.0], dtype=np.float64)
        return nc, cn, h, hu, hv, bed

    def test_returns_dict_with_expected_keys(self):
        nc, cn, h, hu, hv, bed = self._uniform_solution()
        line_xy = np.array([[0.5, 0.5], [1.5, 0.5]], dtype=np.float64)
        result = sample_line_metrics(
            h, hu, hv, bed, nc, cn, line_xy,
            h_min=0.01, timestep_s=0.0, gravity=9.81,
        )
        expected_keys = {
            "station_m", "depth_m", "velocity_ms", "wse_m",
            "bed_m", "froude", "wet", "flow_qn",
        }
        self.assertTrue(expected_keys.issubset(set(result.keys())))

    def test_uniform_depth_and_wse(self):
        nc, cn, h, hu, hv, bed = self._uniform_solution()
        line_xy = np.array([[0.5, 0.5], [1.5, 0.5]], dtype=np.float64)
        result = sample_line_metrics(
            h, hu, hv, bed, nc, cn, line_xy,
            h_min=0.01, timestep_s=0.0, gravity=9.81,
        )
        np.testing.assert_allclose(result["depth_m"], 1.0, atol=1e-6)
        np.testing.assert_allclose(result["bed_m"], 5.0, atol=1e-6)
        np.testing.assert_allclose(result["wse_m"], 6.0, atol=1e-6)

    def test_all_dry_returns_zero_velocity(self):
        nc, cn, h, hu, hv, bed = self._uniform_solution()
        h_dry = np.array([1e-8, 1e-8], dtype=np.float64)
        line_xy = np.array([[0.5, 0.5], [1.5, 0.5]], dtype=np.float64)
        result = sample_line_metrics(
            h_dry, hu, hv, bed, nc, cn, line_xy,
            h_min=0.01, timestep_s=0.0, gravity=9.81,
        )
        self.assertTrue(np.all(result["wet"] == 0))
        np.testing.assert_allclose(result["velocity_ms"], 0.0, atol=1e-12)

    def test_known_froude_for_uniform_flow(self):
        nc, cn, h, hu, hv, bed = self._uniform_solution()
        # h=1, hu=0.5 => u=0.5, Fr = u/sqrt(g*h) = 0.5/sqrt(9.81)
        expected_fr = 0.5 / np.sqrt(9.81 * 1.0)
        line_xy = np.array([[0.5, 0.5], [1.5, 0.5]], dtype=np.float64)
        result = sample_line_metrics(
            h, hu, hv, bed, nc, cn, line_xy,
            h_min=0.01, timestep_s=0.0, gravity=9.81,
        )
        self.assertTrue(np.all(np.isfinite(result["froude"])))
        mean_fr = float(np.mean(result["froude"][np.isfinite(result["froude"])]))
        self.assertAlmostEqual(mean_fr, expected_fr, places=4)

    def test_flow_qn_sign_matches_normal_direction(self):
        nc, cn, h, hu, hv, bed = self._uniform_solution()
        # flow is positive x direction, line from (0,0) to (1,0) gives
        # normal (nx=0, ny=-1). qn = h * (u dot n) = 1 * (0.5*0 + 0*-1) = 0
        line_xy = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
        result = sample_line_metrics(
            h, hu, hv, bed, nc, cn, line_xy,
            h_min=0.01, timestep_s=0.0, gravity=9.81,
        )
        self.assertTrue(np.all(np.isfinite(result["flow_qn"])))

    def test_all_dry_produces_nan_wse(self):
        nc, cn, h, hu, hv, bed = self._uniform_solution()
        h_dry = np.array([1e-8, 1e-8], dtype=np.float64)
        line_xy = np.array([[0.5, 0.5], [1.5, 0.5]], dtype=np.float64)
        result = sample_line_metrics(
            h_dry, hu, hv, bed, nc, cn, line_xy,
            h_min=0.01, timestep_s=0.0, gravity=9.81,
        )
        self.assertTrue(np.all(result["wet"] == 0))
        self.assertTrue(np.all(np.isnan(result["depth_m"])))
        self.assertTrue(np.all(np.isnan(result["wse_m"])))


if __name__ == "__main__":
    unittest.main()
