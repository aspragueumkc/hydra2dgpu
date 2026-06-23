#!/usr/bin/env python3
"""Tests for swe2d/results/overlay_service.py — pure numpy overlay computations."""

import unittest
import numpy as np

from swe2d.results.overlay_service import (
    compute_velocity_magnitude,
    compute_wse,
    compute_froude,
)


class TestComputeVelocityMagnitude(unittest.TestCase):
    def test_basic_velocity_magnitude(self):
        hu = np.array([3.0, 0.0, -4.0], dtype=np.float64)
        hv = np.array([4.0, 0.0, 3.0], dtype=np.float64)
        h = np.array([2.0, 1.0, 5.0], dtype=np.float64)
        expected = np.array([2.5, 0.0, 1.0], dtype=np.float64)
        result = compute_velocity_magnitude(hu, hv, h)
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_zero_depth_gives_zero_speed(self):
        hu = np.array([5.0, 0.0], dtype=np.float64)
        hv = np.array([0.0, 3.0], dtype=np.float64)
        h = np.array([0.0, 0.0], dtype=np.float64)
        result = compute_velocity_magnitude(hu, hv, h)
        self.assertEqual(result.shape, (2,))
        self.assertTrue(np.all(np.isfinite(result)))
        np.testing.assert_allclose(result, [0.0, 0.0], atol=1e-15)

    def test_negative_depth_treated_as_zero(self):
        hu = np.array([2.0], dtype=np.float64)
        hv = np.array([0.0], dtype=np.float64)
        h = np.array([-1.0], dtype=np.float64)
        result = compute_velocity_magnitude(hu, hv, h)
        self.assertTrue(np.isfinite(result[0]))
        self.assertEqual(result[0], 0.0)

    def test_empty_arrays(self):
        hu = np.empty(0, dtype=np.float64)
        hv = np.empty(0, dtype=np.float64)
        h = np.empty(0, dtype=np.float64)
        result = compute_velocity_magnitude(hu, hv, h)
        self.assertEqual(result.shape, (0,))

    def test_assumed_formula_equal_magnitude(self):
        hu = np.array([1.5, 0.0], dtype=np.float64)
        hv = np.array([2.0, 0.0], dtype=np.float64)
        h = np.array([0.5, 1.0], dtype=np.float64)
        u = hu / np.maximum(h, 1e-12)
        v = hv / np.maximum(h, 1e-12)
        expected = np.sqrt(u * u + v * v)
        result = compute_velocity_magnitude(hu, hv, h)
        np.testing.assert_allclose(result, expected, rtol=1e-12)


class TestComputeWSE(unittest.TestCase):
    def test_basic_wse(self):
        h = np.array([1.0, 2.5, 0.0], dtype=np.float64)
        bed = np.array([100.0, 101.0, 99.5], dtype=np.float64)
        expected = np.array([101.0, 103.5, 99.5], dtype=np.float64)
        result = compute_wse(h, bed)
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_zero_depth(self):
        h = np.array([0.0, 0.0], dtype=np.float64)
        bed = np.array([50.0, -10.0], dtype=np.float64)
        result = compute_wse(h, bed)
        np.testing.assert_allclose(result, bed, rtol=1e-12)

    def test_negative_h(self):
        h = np.array([-0.5], dtype=np.float64)
        bed = np.array([100.0], dtype=np.float64)
        result = compute_wse(h, bed)
        np.testing.assert_allclose(result, [99.5], rtol=1e-12)

    def test_empty_arrays(self):
        h = np.empty(0, dtype=np.float64)
        bed = np.empty(0, dtype=np.float64)
        result = compute_wse(h, bed)
        self.assertEqual(result.shape, (0,))


class TestComputeFroude(unittest.TestCase):
    def test_basic_froude(self):
        hu = np.array([3.0], dtype=np.float64)
        hv = np.array([4.0], dtype=np.float64)
        h = np.array([2.0], dtype=np.float64)
        g = 9.81
        speed = 2.5  # sqrt((3/2)^2 + (4/2)^2) = sqrt(2.25 + 4) = sqrt(6.25) = 2.5
        expected = speed / np.sqrt(g * 2.0)
        result = compute_froude(hu, hv, h, g, h_min=0.001)
        self.assertAlmostEqual(float(result[0]), float(expected), places=12)

    def test_dry_cell_returns_zero(self):
        hu = np.array([5.0, 0.0], dtype=np.float64)
        hv = np.array([0.0, 3.0], dtype=np.float64)
        h = np.array([0.0, 1e-6], dtype=np.float64)
        g = 9.81
        result = compute_froude(hu, hv, h, g, h_min=0.01)
        h_safe = np.maximum(np.abs(h), 0.01)
        speed0 = 0.0
        speed1 = np.sqrt((0.0 / h_safe[1]) ** 2 + (3.0 / h_safe[1]) ** 2)
        expected1 = speed1 / np.sqrt(g * h_safe[1])
        self.assertEqual(result[0], 0.0)
        self.assertAlmostEqual(float(result[1]), float(expected1), places=12)

    def test_subcritical_supercritical(self):
        hu = np.array([0.5, 5.0], dtype=np.float64)
        hv = np.array([0.0, 0.0], dtype=np.float64)
        h = np.array([1.0, 0.5], dtype=np.float64)
        g = 9.81
        result = compute_froude(hu, hv, h, g, h_min=0.001)
        h_safe = np.maximum(np.abs(h), 0.001)
        v = np.abs(hu) / h_safe
        expected = v / np.sqrt(g * h_safe)
        np.testing.assert_allclose(result, expected, rtol=1e-12)
        self.assertLess(result[0], 1.0)
        self.assertGreater(result[1], 1.0)

    def test_empty_arrays(self):
        hu = np.empty(0, dtype=np.float64)
        hv = np.empty(0, dtype=np.float64)
        h = np.empty(0, dtype=np.float64)
        result = compute_froude(hu, hv, h, g=9.81, h_min=0.001)
        self.assertEqual(result.shape, (0,))


if __name__ == "__main__":
    unittest.main(verbosity=2)
