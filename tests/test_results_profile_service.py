"""Tests for swe2d.results.profile_service — pure numpy data logic, no Qt."""

import unittest
import numpy as np

from swe2d.results.profile_service import compute_profile_data, compute_profile_fill

# ---------------------------------------------------------------------------
# compute_profile_data
# ---------------------------------------------------------------------------


class TestComputeProfileDataSmoke(unittest.TestCase):
    """Smoke tests: returns correct structure for a minimal mesh."""

    def _simple_mesh(self):
        # 4 nodes forming a unit square, 2 triangles
        node_coords = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ], dtype=np.float64)
        cell_nodes = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        h = np.array([0.5, 0.5], dtype=np.float64)
        hu = np.array([0.1, 0.0], dtype=np.float64)
        hv = np.array([0.0, 0.1], dtype=np.float64)
        bed = np.array([10.0, 10.0], dtype=np.float64)
        return node_coords, cell_nodes, h, hu, hv, bed

    def test_returns_dict_with_expected_keys(self):
        nc, cn, h, hu, hv, bed = self._simple_mesh()
        profile_line = np.array([[0.25, 0.5], [0.75, 0.5]], dtype=np.float64)
        result = compute_profile_data(
            h, hu, hv, bed, nc, cn, profile_line, h_min=0.01, n_samples=5,
        )
        expected_keys = {
            "distance", "depth", "velocity", "wse", "bed",
            "froude", "wet_fraction", "flow_qn",
        }
        self.assertEqual(expected_keys, set(result.keys()))

    def test_all_arrays_have_expected_length(self):
        nc, cn, h, hu, hv, bed = self._simple_mesh()
        profile_line = np.array([[0.25, 0.5], [0.75, 0.5]], dtype=np.float64)
        for n in (3, 10, 101):
            result = compute_profile_data(
                h, hu, hv, bed, nc, cn, profile_line, h_min=0.01, n_samples=n,
            )
            for k, v in result.items():
                self.assertEqual(
                    n, len(v), f"Key={k} length mismatch for n_samples={n}",
                )

    def test_all_outputs_finite(self):
        nc, cn, h, hu, hv, bed = self._simple_mesh()
        profile_line = np.array([[0.25, 0.5], [0.75, 0.5]], dtype=np.float64)
        result = compute_profile_data(
            h, hu, hv, bed, nc, cn, profile_line, h_min=0.01, n_samples=10,
        )
        for k, v in result.items():
            self.assertTrue(
                np.all(np.isfinite(v)),
                f"Key={k} has non-finite values: {v}",
            )

    def test_distance_starts_at_zero_and_monotonic(self):
        nc, cn, h, hu, hv, bed = self._simple_mesh()
        profile_line = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float64)
        result = compute_profile_data(
            h, hu, hv, bed, nc, cn, profile_line, h_min=0.01, n_samples=10,
        )
        dist = result["distance"]
        self.assertAlmostEqual(0.0, dist[0])
        self.assertAlmostEqual(2.0, dist[-1], places=4)
        # monotonic
        self.assertTrue(np.all(np.diff(dist) >= 0))


class TestComputeProfileDataValues(unittest.TestCase):
    """Verify known values for a simple uniform mesh."""

    def test_uniform_depth_and_wse(self):
        node_coords = np.array([
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 1.0],
            [0.0, 1.0],
        ], dtype=np.float64)
        # 2 triangles
        cell_nodes = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        h = np.array([1.0, 1.0], dtype=np.float64)
        hu = np.array([0.5, 0.5], dtype=np.float64)
        hv = np.array([0.0, 0.0], dtype=np.float64)
        bed = np.array([5.0, 5.0], dtype=np.float64)
        profile_line = np.array([[0.5, 0.5], [1.5, 0.5]], dtype=np.float64)

        result = compute_profile_data(
            h, hu, hv, bed, node_coords, cell_nodes,
            profile_line, h_min=0.01, n_samples=20,
        )
        # All sample points should see depth=1.0, wse=6.0, bed=5.0
        for k, expected in [("depth", 1.0), ("bed", 5.0)]:
            self.assertTrue(
                np.allclose(result[k], expected, atol=1e-6),
                f"Key={k}: expected {expected}, got {result[k]}",
            )
        self.assertTrue(
            np.allclose(result["wse"], 6.0, atol=1e-6),
            f"wse: {result['wse']}",
        )

    def test_velocity_computation(self):
        node_coords = np.array([
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 1.0],
            [0.0, 1.0],
        ], dtype=np.float64)
        cell_nodes = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        h = np.array([2.0, 2.0], dtype=np.float64)
        hu = np.array([3.0, 3.0], dtype=np.float64)  # u = 1.5
        hv = np.array([4.0, 4.0], dtype=np.float64)  # v = 2.0
        bed = np.array([10.0, 10.0], dtype=np.float64)
        profile_line = np.array([[0.5, 0.5], [1.5, 0.5]], dtype=np.float64)

        result = compute_profile_data(
            h, hu, hv, bed, node_coords, cell_nodes,
            profile_line, h_min=0.01, n_samples=20,
        )
        # velocity = sqrt(1.5^2 + 2.0^2) = 2.5
        expected_vel = 2.5
        self.assertTrue(
            np.allclose(result["velocity"], expected_vel, atol=1e-6),
            f"velocity: {result['velocity']} (expected {expected_vel})",
        )

    def test_froude_number(self):
        g = 9.81
        node_coords = np.array([
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 1.0],
            [0.0, 1.0],
        ], dtype=np.float64)
        cell_nodes = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        h = np.array([1.0, 1.0], dtype=np.float64)
        hu = np.array([1.0, 1.0], dtype=np.float64)  # u = 1.0
        hv = np.array([0.0, 0.0], dtype=np.float64)
        bed = np.array([0.0, 0.0], dtype=np.float64)
        profile_line = np.array([[0.5, 0.5], [1.5, 0.5]], dtype=np.float64)

        result = compute_profile_data(
            h, hu, hv, bed, node_coords, cell_nodes,
            profile_line, h_min=0.01, n_samples=10,
        )
        # Fr = velocity / sqrt(g * depth) = 1.0 / sqrt(9.81 * 1.0) ~ 0.319
        expected_fr = 1.0 / np.sqrt(g * 1.0)
        self.assertTrue(
            np.allclose(result["froude"], expected_fr, atol=1e-5),
            f"froude: {result['froude']} (expected {expected_fr})",
        )

    def test_wet_fraction_is_one_for_deep_water(self):
        nc, cn, h, hu, hv, bed = self._simple_mesh()
        profile_line = np.array([[0.25, 0.5], [0.75, 0.5]], dtype=np.float64)
        result = compute_profile_data(
            h, hu, hv, bed, nc, cn, profile_line, h_min=0.01, n_samples=10,
        )
        self.assertTrue(np.all(result["wet_fraction"] == 1.0))

    def test_dry_cells_give_zero_velocity(self):
        node_coords = np.array([
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 1.0],
            [0.0, 1.0],
        ], dtype=np.float64)
        cell_nodes = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        h = np.array([0.0, 0.0], dtype=np.float64)
        hu = np.array([0.0, 0.0], dtype=np.float64)
        hv = np.array([0.0, 0.0], dtype=np.float64)
        bed = np.array([10.0, 10.0], dtype=np.float64)
        profile_line = np.array([[0.5, 0.5], [1.5, 0.5]], dtype=np.float64)

        result = compute_profile_data(
            h, hu, hv, bed, node_coords, cell_nodes,
            profile_line, h_min=0.01, n_samples=10,
        )
        self.assertTrue(np.all(result["wet_fraction"] == 0.0))
        self.assertTrue(np.allclose(result["velocity"], 0.0))

    @staticmethod
    def _simple_mesh():
        node_coords = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ], dtype=np.float64)
        cell_nodes = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        h = np.array([0.5, 0.5], dtype=np.float64)
        hu = np.array([0.1, 0.0], dtype=np.float64)
        hv = np.array([0.0, 0.1], dtype=np.float64)
        bed = np.array([10.0, 10.0], dtype=np.float64)
        return node_coords, cell_nodes, h, hu, hv, bed


class TestComputeProfileDataEdgeCases(unittest.TestCase):
    def test_empty_profile_line(self):
        nc = np.empty((0, 2), dtype=np.float64)
        cn = np.empty((0, 3), dtype=np.int32)
        h = hu = hv = bed = np.empty(0, dtype=np.float64)
        profile_line = np.empty((0, 2), dtype=np.float64)
        result = compute_profile_data(
            h, hu, hv, bed, nc, cn, profile_line, h_min=0.01, n_samples=0,
        )
        for v in result.values():
            self.assertEqual(0, len(v))

    def test_single_point_profile_line(self):
        nc, cn, h, hu, hv, bed = self._simple_mesh()
        profile_line = np.array([[0.5, 0.5]], dtype=np.float64)
        result = compute_profile_data(
            h, hu, hv, bed, nc, cn, profile_line, h_min=0.01, n_samples=1,
        )
        for v in result.values():
            self.assertEqual(1, len(v))
            self.assertTrue(np.isfinite(v).all())

    def test_negative_h_min_same_as_zero(self):
        nc, cn, h, hu, hv, bed = self._simple_mesh()
        profile_line = np.array([[0.25, 0.5], [0.75, 0.5]], dtype=np.float64)
        r1 = compute_profile_data(
            h, hu, hv, bed, nc, cn, profile_line, h_min=-0.1, n_samples=10,
        )
        r2 = compute_profile_data(
            h, hu, hv, bed, nc, cn, profile_line, h_min=0.0, n_samples=10,
        )
        for k in r1:
            np.testing.assert_array_equal(r1[k], r2[k])

    @staticmethod
    def _simple_mesh():
        node_coords = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ], dtype=np.float64)
        cell_nodes = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        h = np.array([0.5, 0.5], dtype=np.float64)
        hu = np.array([0.1, 0.0], dtype=np.float64)
        hv = np.array([0.0, 0.1], dtype=np.float64)
        bed = np.array([10.0, 10.0], dtype=np.float64)
        return node_coords, cell_nodes, h, hu, hv, bed


# ---------------------------------------------------------------------------
# compute_profile_fill
# ---------------------------------------------------------------------------


class TestComputeProfileFill(unittest.TestCase):
    def test_returns_array(self):
        points = {
            "station_m": np.array([0.0, 1.0, 2.0], dtype=np.float64),
            "wse_m": np.array([5.0, 5.5, 6.0], dtype=np.float64),
            "bed_m": np.array([4.0, 4.0, 4.0], dtype=np.float64),
            "depth_m": np.array([1.0, 1.5, 2.0], dtype=np.float64),
            "velocity_ms": np.array([0.5, 1.0, 1.5], dtype=np.float64),
            "wet": np.array([1, 1, 1], dtype=np.int32),
            "fr": np.array([0.1, 0.2, 0.3], dtype=np.float64),
        }
        result = compute_profile_fill(points, fill_by="velocity_ms", wse_render="clipped")
        self.assertIsInstance(result, np.ndarray)

    def test_fill_values_midpoint_of_adjacent_pairs(self):
        points = {
            "station_m": np.array([0.0, 1.0, 2.0], dtype=np.float64),
            "wse_m": np.array([5.0, 5.5, 6.0], dtype=np.float64),
            "bed_m": np.array([4.0, 4.0, 4.0], dtype=np.float64),
            "depth_m": np.array([1.0, 1.5, 2.0], dtype=np.float64),
            "velocity_ms": np.array([1.0, 3.0, 5.0], dtype=np.float64),
            "wet": np.array([1, 1, 1], dtype=np.int32),
            "fr": np.array([0.1, 0.2, 0.3], dtype=np.float64),
        }
        result = compute_profile_fill(points, fill_by="velocity_ms", wse_render="clipped")
        # 3 points -> 2 segments: midpoints should be (1+3)/2=2 and (3+5)/2=4
        self.assertEqual(2, len(result))
        self.assertAlmostEqual(2.0, result[0])
        self.assertAlmostEqual(4.0, result[1])

    def test_fill_by_depth_clipped_render(self):
        points = {
            "station_m": np.array([0.0, 1.0, 2.0], dtype=np.float64),
            "wse_m": np.array([5.0, 5.5, 6.0], dtype=np.float64),
            "bed_m": np.array([4.0, 4.0, 4.0], dtype=np.float64),
            "depth_m": np.array([1.0, 1.5, 2.0], dtype=np.float64),
            "velocity_ms": np.array([0.5, 1.0, 1.5], dtype=np.float64),
            "wet": np.array([1, 1, 1], dtype=np.int32),
            "fr": np.array([0.1, 0.2, 0.3], dtype=np.float64),
        }
        result = compute_profile_fill(points, fill_by="depth_m", wse_render="clipped")
        self.assertEqual(2, len(result))
        self.assertAlmostEqual(1.25, result[0])
        self.assertAlmostEqual(1.75, result[1])

    def test_raw_render_uses_all_finite_points(self):
        # raw mode doesn't use wet mask, just finite check
        points = {
            "station_m": np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64),
            "wse_m": np.array([5.0, np.nan, 6.0, 7.0], dtype=np.float64),
            "bed_m": np.array([4.0, 4.0, 4.0, 4.0], dtype=np.float64),
            "depth_m": np.array([1.0, 1.5, 2.0, 2.5], dtype=np.float64),
            "wet": np.array([1, 0, 1, 1], dtype=np.int32),
            "fr": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64),
        }
        result = compute_profile_fill(points, fill_by="depth_m", wse_render="raw")
        # Segments: (0-1) has wse_m[1]=nan -> skip; (1-2) has wse_m[1]=nan -> skip;
        # (2-3) both finite -> include
        self.assertEqual(1, len(result))
        self.assertAlmostEqual(2.25, result[0])

    def test_clipped_skips_dry_segments(self):
        points = {
            "station_m": np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64),
            "wse_m": np.array([5.0, 5.5, 6.0, 6.5], dtype=np.float64),
            "bed_m": np.array([4.0, 4.0, 4.0, 4.0], dtype=np.float64),
            "depth_m": np.array([1.0, 0.0, 1.5, 2.0], dtype=np.float64),
            "wet": np.array([1, 0, 1, 1], dtype=np.int32),
            "fr": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64),
        }
        result = compute_profile_fill(points, fill_by="fr", wse_render="clipped")
        # wet=[1,0,1,1]; segments: (0,1) has wet[1]=0 -> skip, (1,2) has wet[1]=0 -> skip,
        # (2,3) both wet -> include. Midpoint of fr[2]=0.3 & fr[3]=0.4 = 0.35
        self.assertEqual(1, len(result))
        self.assertAlmostEqual(0.35, result[0])

    def test_all_dry_returns_empty(self):
        points = {
            "station_m": np.array([0.0, 1.0, 2.0], dtype=np.float64),
            "wse_m": np.array([5.0, 5.5, 6.0], dtype=np.float64),
            "bed_m": np.array([4.0, 4.0, 4.0], dtype=np.float64),
            "depth_m": np.array([0.0, 0.0, 0.0], dtype=np.float64),
            "wet": np.array([0, 0, 0], dtype=np.int32),
            "fr": np.array([0.0, 0.0, 0.0], dtype=np.float64),
        }
        result = compute_profile_fill(points, fill_by="depth_m", wse_render="clipped")
        self.assertEqual(0, len(result))

    def test_less_than_two_points_returns_empty(self):
        points = {
            "station_m": np.array([0.0], dtype=np.float64),
            "wse_m": np.array([5.0], dtype=np.float64),
            "bed_m": np.array([4.0], dtype=np.float64),
            "depth_m": np.array([1.0], dtype=np.float64),
            "wet": np.array([1], dtype=np.int32),
        }
        result = compute_profile_fill(points, fill_by="depth_m", wse_render="clipped")
        self.assertEqual(0, len(result))


if __name__ == "__main__":
    unittest.main()
