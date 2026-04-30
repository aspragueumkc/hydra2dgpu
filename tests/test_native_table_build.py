#!/usr/bin/env python3
"""Parity checks for native section hydraulic-table construction."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backwater_model import CrossSection
from unsteady_model import _build_section_hydraulic_table


class TestNativeTableBuild(unittest.TestCase):
    def setUp(self):
        self.xs = CrossSection(
            river_station='150',
            geometry=[(-30.0, 101.0), (-10.0, 99.0), (0.0, 98.4), (10.0, 98.5), (30.0, 101.2)],
            left_bank_station=-10.0,
            right_bank_station=10.0,
            n_lob=0.055,
            n_ch=0.035,
            n_rob=0.06,
            L_lob_to_next=500.0,
            L_ch_to_next=500.0,
            L_rob_to_next=500.0,
        )

    def test_native_table_builder_matches_python(self):
        try:
            import backwater_native
        except Exception:
            self.skipTest('native module not built/importable in test environment')

        if not hasattr(backwater_native, 'build_section_hydraulic_table_cpp'):
            self.skipTest('native module missing build_section_hydraulic_table_cpp entrypoint')

        old = os.environ.get('BACKWATER_USE_CPP_SOLVER')
        try:
            os.environ['BACKWATER_USE_CPP_SOLVER'] = '0'
            py_table = _build_section_hydraulic_table(self.xs, dz=0.05, padding=3.0)

            os.environ['BACKWATER_USE_CPP_SOLVER'] = '1'
            native_table = _build_section_hydraulic_table(self.xs, dz=0.05, padding=3.0)
        finally:
            if old is None:
                os.environ.pop('BACKWATER_USE_CPP_SOLVER', None)
            else:
                os.environ['BACKWATER_USE_CPP_SOLVER'] = old

        arrays = (
            'z_values',
            'A_lob_raw', 'T_lob_raw', 'K_lob_raw',
            'A_ch', 'T_ch', 'K_ch',
            'A_rob_raw', 'T_rob_raw', 'K_rob_raw',
            'K_total_raw', 'dK_dz_raw',
        )
        for name in arrays:
            self.assertTrue(
                np.allclose(getattr(native_table, name), getattr(py_table, name), rtol=1e-10, atol=1e-12),
                msg=name,
            )

        self.assertAlmostEqual(native_table.left_activation_elev, py_table.left_activation_elev, places=12)
        self.assertAlmostEqual(native_table.right_activation_elev, py_table.right_activation_elev, places=12)


if __name__ == '__main__':
    unittest.main()
