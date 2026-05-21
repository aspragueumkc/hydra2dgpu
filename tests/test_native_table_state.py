#!/usr/bin/env python3
"""Parity checks for optional native hydraulic table-state kernel."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unsteady_model import SectionHydraulicTable, _unsteady_section_state_from_table
from hydra_1d import CrossSection


class TestNativeTableState(unittest.TestCase):
    def setUp(self):
        self.xs = CrossSection(
            river_station='100',
            geometry=[(-20.0, 100.0), (-5.0, 98.0), (5.0, 98.0), (20.0, 100.0)],
            left_bank_station=-5.0,
            right_bank_station=5.0,
            n_lob=0.05,
            n_ch=0.035,
            n_rob=0.05,
            L_ch_to_next=500.0,
            L_lob_to_next=500.0,
            L_rob_to_next=500.0,
        )
        z_values = np.array([98.0, 99.0, 100.0, 101.0], dtype=float)
        self.table = SectionHydraulicTable(
            z_values=z_values,
            A_lob_raw=np.array([0.0, 1.0, 2.0, 3.0], dtype=float),
            T_lob_raw=np.array([0.0, 1.0, 1.2, 1.4], dtype=float),
            K_lob_raw=np.array([0.0, 4.0, 8.0, 12.0], dtype=float),
            A_ch=np.array([2.0, 3.0, 4.0, 5.0], dtype=float),
            T_ch=np.array([5.0, 5.5, 6.0, 6.5], dtype=float),
            K_ch=np.array([20.0, 24.0, 28.0, 32.0], dtype=float),
            A_rob_raw=np.array([0.0, 0.8, 1.6, 2.4], dtype=float),
            T_rob_raw=np.array([0.0, 0.9, 1.1, 1.3], dtype=float),
            K_rob_raw=np.array([0.0, 3.5, 7.0, 10.5], dtype=float),
            K_total_raw=np.array([20.0, 31.5, 43.0, 54.5], dtype=float),
            dK_dz_raw=np.array([11.5, 11.5, 11.5, 11.5], dtype=float),
            left_activation_elev=98.5,
            right_activation_elev=98.75,
        )

    def test_native_table_state_matches_python_path(self):
        try:
            import hydra_native
        except Exception:
            self.skipTest('native module not built/importable in test environment')

        self.assertTrue(hasattr(hydra_native, 'solve_table_state'))

        old = os.environ.get('BACKWATER_USE_CPP_SOLVER')
        try:
            os.environ['BACKWATER_USE_CPP_SOLVER'] = '0'
            py_state = _unsteady_section_state_from_table(self.xs, self.table, z=99.25, Q_total=150.0, overbank_ramp_depth=0.25)

            os.environ['BACKWATER_USE_CPP_SOLVER'] = '1'
            native_state = _unsteady_section_state_from_table(self.xs, self.table, z=99.25, Q_total=150.0, overbank_ramp_depth=0.25)
        finally:
            if old is None:
                os.environ.pop('BACKWATER_USE_CPP_SOLVER', None)
            else:
                os.environ['BACKWATER_USE_CPP_SOLVER'] = old

        fields = (
            'alpha', 'A_lob', 'A_ch', 'A_rob', 'T_lob', 'T_ch', 'T_rob',
            'K_lob', 'K_ch', 'K_rob', 'Q_lob', 'Q_ch', 'Q_rob',
            'A_t', 'T_t', 'K_t', 'V_t', 'left_activation_factor', 'right_activation_factor',
        )
        for field in fields:
            self.assertAlmostEqual(getattr(native_state, field), getattr(py_state, field), places=10, msg=field)


if __name__ == '__main__':
    unittest.main()
