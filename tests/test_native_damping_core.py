#!/usr/bin/env python3
"""Parity checks for optional native adaptive damping core."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unsteady_model import _adaptive_damping_scale_core


class TestNativeDampingCore(unittest.TestCase):
    def test_native_damping_scale_matches_python(self):
        try:
            import backwater_native
        except Exception:
            self.skipTest('native module not built/importable in test environment')

        bed = np.array([100.0, 99.8, 99.5], dtype=float)
        z_iter = np.array([100.2, 100.05, 99.65], dtype=float)
        q_iter = np.array([35.0, 120.0, 15.0], dtype=float)
        dz_raw = np.array([0.4, -0.2, 0.08], dtype=float)
        dq_raw = np.array([60.0, -120.0, 30.0], dtype=float)

        py_scale = _adaptive_damping_scale_core(bed, z_iter, q_iter, dz_raw, dq_raw)
        native_scale = backwater_native.adaptive_damping_scale(bed, z_iter, q_iter, dz_raw, dq_raw, 0.001)

        self.assertAlmostEqual(native_scale, py_scale, places=12)


if __name__ == '__main__':
    unittest.main()