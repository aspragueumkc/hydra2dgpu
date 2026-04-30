#!/usr/bin/env python3
"""Parity checks for HP2 native batch node-property evaluation."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unsteady_model import (
    _compute_node_properties,
    _build_hydraulic_tables,
    _hydraulic_bed_elevation,
    WETTING_DEPTH_FT,
)
from backwater_model import CrossSection


def _make_sections():
    """Return 4 simple test cross-sections with channel + overbank geometry."""
    specs = [
        ('100', 100.0, 99.0),
        ('200',  99.5, 98.5),
        ('300',  99.0, 98.0),
        ('400',  98.5, 97.5),
    ]
    sections = []
    for rs, lob_el, ch_el in specs:
        sections.append(CrossSection(
            river_station=rs,
            geometry=[
                (-20.0, lob_el),
                (0.0,   ch_el),
                (20.0,  ch_el),
                (40.0,  lob_el),
            ],
            left_bank_station=-20.0,
            right_bank_station=40.0,
            n_lob=0.05,
            n_ch=0.035,
            n_rob=0.05,
            L_lob_to_next=500.0,
            L_ch_to_next=500.0,
            L_rob_to_next=500.0,
        ))
    return sections


class TestNativeNodeProperties(unittest.TestCase):
    def setUp(self):
        try:
            import backwater_native  # noqa: F401
        except Exception:
            self.skipTest('native module not built/importable')

        self.sections = _make_sections()
        N = len(self.sections)
        self.dx = [500.0] * (N - 1)
        self.z_n = np.array([100.2, 99.7, 99.2, 98.7], dtype=np.float64)
        self.Q_n = np.array([120.0, 120.0, 120.0, 120.0], dtype=np.float64)
        self.bed_elevations = np.array(
            [_hydraulic_bed_elevation(xs) for xs in self.sections], dtype=np.float64
        )
        self.hydraulic_tables = _build_hydraulic_tables(self.sections, dz=0.01, padding=2.0)

    def _python_result(self):
        """Compute reference result using Python path (native disabled)."""
        old = os.environ.get('BACKWATER_USE_CPP_SOLVER', None)
        os.environ['BACKWATER_USE_CPP_SOLVER'] = '0'
        try:
            return _compute_node_properties(
                self.sections,
                self.dx,
                self.z_n,
                self.Q_n,
                hydraulic_tables=self.hydraulic_tables,
                bed_elevations=None,  # Force Python path
            )
        finally:
            if old is None:
                del os.environ['BACKWATER_USE_CPP_SOLVER']
            else:
                os.environ['BACKWATER_USE_CPP_SOLVER'] = old

    def _native_result(self):
        """Compute result using native HP2 path."""
        old = os.environ.get('BACKWATER_USE_CPP_SOLVER', None)
        os.environ['BACKWATER_USE_CPP_SOLVER'] = '1'
        try:
            return _compute_node_properties(
                self.sections,
                self.dx,
                self.z_n,
                self.Q_n,
                hydraulic_tables=self.hydraulic_tables,
                bed_elevations=self.bed_elevations,
            )
        finally:
            if old is None:
                del os.environ['BACKWATER_USE_CPP_SOLVER']
            else:
                os.environ['BACKWATER_USE_CPP_SOLVER'] = old

    def test_node_properties_parity(self):
        """Native batch node-property eval must match Python path to 1e-10."""
        py = self._python_result()
        na = self._native_result()

        names = ['reach_lengths', 'area', 'conveyance', 'top_width', 'velocity', 'alpha', 'dkdz']
        for name, arr_py, arr_na in zip(names, py, na):
            arr_py = np.asarray(arr_py, dtype=np.float64)
            arr_na = np.asarray(arr_na, dtype=np.float64)
            self.assertEqual(arr_py.shape, arr_na.shape,
                             msg=f'{name}: shape mismatch {arr_py.shape} vs {arr_na.shape}')
            if not np.allclose(arr_py, arr_na, atol=1e-10, rtol=1e-8):
                max_err = np.max(np.abs(arr_py - arr_na))
                self.fail(
                    f'{name}: max abs error {max_err:.3e}\n  Python: {arr_py}\n  Native: {arr_na}'
                )

    def test_node_properties_shapes(self):
        """Native result arrays have correct shapes (N-1) and (N,)."""
        na = self._native_result()
        N = len(self.sections)
        reach_len, area, conv, tw, vel, alpha, dkdz = na
        self.assertEqual(len(reach_len), N - 1, 'reach_lengths shape')
        for name, arr in [('area', area), ('conv', conv), ('tw', tw),
                          ('vel', vel), ('alpha', alpha), ('dkdz', dkdz)]:
            self.assertEqual(len(arr), N, f'{name} shape')

    def test_node_properties_physical(self):
        """Native results satisfy basic physical constraints."""
        na = self._native_result()
        reach_len, area, conv, tw, vel, alpha, dkdz = na
        np.testing.assert_array_less(0.0, area, err_msg='area must be positive')
        np.testing.assert_array_less(0.0, conv, err_msg='conveyance must be positive')
        np.testing.assert_array_less(0.0, tw, err_msg='top-width must be positive')
        np.testing.assert_array_less(0.0, reach_len, err_msg='reach lengths must be positive')
        self.assertTrue(np.all(alpha >= 1.0), f'alpha >= 1 failed: {alpha}')


if __name__ == '__main__':
    unittest.main()
