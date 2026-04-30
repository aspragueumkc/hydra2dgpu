#!/usr/bin/env python3
"""Tests for optional native backend integration."""

import os
import sys
import unittest

import numpy as np

# Make sure the plugin directory is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unsteady_model import _solve_banded


class TestNativeBackendToggle(unittest.TestCase):
    def _full_to_banded_5(self, a):
        n = a.shape[0]
        ab = np.zeros((5, n), dtype=float)
        for k in range(-2, 3):
            row_start = max(0, -k)
            col_start = max(0, k)
            length = n - abs(k)
            ab_row = 2 - k
            for i in range(length):
                ab[ab_row, col_start + i] = a[row_start + i, col_start + i]
        return ab

    def test_solver_runs_without_native_module(self):
        old = os.environ.get("BACKWATER_USE_CPP_SOLVER")
        os.environ["BACKWATER_USE_CPP_SOLVER"] = "1"
        try:
            # Symmetric, diagonally dominant pentadiagonal matrix
            a = np.array([
                [5.0, -1.0, 0.5, 0.0, 0.0],
                [-1.0, 5.0, -1.0, 0.5, 0.0],
                [0.5, -1.0, 5.0, -1.0, 0.5],
                [0.0, 0.5, -1.0, 5.0, -1.0],
                [0.0, 0.0, 0.5, -1.0, 5.0],
            ], dtype=float)
            rhs = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=float)
            ab = self._full_to_banded_5(a)

            x = _solve_banded(ab, rhs)
            x_ref = np.linalg.solve(a, rhs)
            self.assertTrue(np.allclose(x, x_ref, rtol=1e-8, atol=1e-10))
        finally:
            if old is None:
                os.environ.pop("BACKWATER_USE_CPP_SOLVER", None)
            else:
                os.environ["BACKWATER_USE_CPP_SOLVER"] = old


if __name__ == "__main__":
    unittest.main()
