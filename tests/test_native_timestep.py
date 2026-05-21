#!/usr/bin/env python3
"""Parity checks for optional native full-timestep binding."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unsteady_model import _compute_node_properties, _assemble_system_core, _solve_banded, _apply_adaptive_damping, _regularized_wse, WETTING_DEPTH_FT
from hydra_1d import CrossSection, ModelInput


class TestNativeTimestep(unittest.TestCase):
    def setUp(self):
        """Create a simple 3-section test model."""
        self.sections_us_to_ds = [
            CrossSection(
                river_station='100',
                geometry=[(0.0, 100.0), (20.0, 99.0), (40.0, 100.0)],
                left_bank_station=0.0,
                right_bank_station=20.0,
                n_lob=0.05,
                n_ch=0.035,
                n_rob=0.05,
                L_lob_to_next=500.0,
                L_ch_to_next=500.0,
                L_rob_to_next=500.0,
            ),
            CrossSection(
                river_station='200',
                geometry=[(0.0, 99.5), (20.0, 98.5), (40.0, 99.5)],
                left_bank_station=0.0,
                right_bank_station=20.0,
                n_lob=0.05,
                n_ch=0.035,
                n_rob=0.05,
                L_lob_to_next=500.0,
                L_ch_to_next=500.0,
                L_rob_to_next=500.0,
            ),
            CrossSection(
                river_station='300',
                geometry=[(0.0, 99.0), (20.0, 98.0), (40.0, 99.0)],
                left_bank_station=0.0,
                right_bank_station=20.0,
                n_lob=0.05,
                n_ch=0.035,
                n_rob=0.05,
                L_lob_to_next=500.0,
                L_ch_to_next=500.0,
                L_rob_to_next=500.0,
            ),
        ]

        # Simple initial state
        self.z_n = np.array([100.5, 100.0, 99.5], dtype=np.float64)
        self.Q_n = np.array([100.0, 100.0, 100.0], dtype=np.float64)
        self.dx = [500.0, 500.0]
        self.bed_elevations = np.array([100.0, 99.5, 99.0], dtype=np.float64)

        self.dt = 30.0
        self.theta = 0.6
        self.Q_upstream_next = 110.0
        self.ds_bc = 'normal_depth'
        self.ds_bc_value = 0.001
        self.ds_bc_ramp_factor = 1.0
        self.max_iter = 3
        self.tol = 1e-4

    def test_native_timestep_matches_python(self):
        """Verify native timestep binding matches Python Newton iterations."""
        try:
            import hydra_native
        except Exception:
            self.skipTest('native module not built/importable in test environment')

        # Ensure native is enabled
        os.environ['BACKWATER_USE_CPP_SOLVER'] = '1'

        # Python-side computation
        reach_lengths, area_values, conveyance_values, top_width_values, velocity_values, alpha_values, dkdz_values = _compute_node_properties(
            self.sections_us_to_ds,
            self.dx,
            self.z_n,
            self.Q_n,
            hydraulic_tables=None,
            overbank_ramp_depth=WETTING_DEPTH_FT,
        )

        # Run Python Newton loop
        z_py = np.array(self.z_n, dtype=np.float64)
        Q_py = np.array(self.Q_n, dtype=np.float64)
        for _inner in range(self.max_iter):
            ab, rhs_vec = _assemble_system_core(
                reach_lengths,
                z_py,
                Q_py,
                area_values,
                conveyance_values,
                top_width_values,
                velocity_values,
                alpha_values,
                dkdz_values,
                self.dt,
                self.theta,
                self.Q_upstream_next,
                False,  # ds_is_stage
                self.ds_bc_value,
                self.ds_bc_ramp_factor,
            )
            delta = _solve_banded(ab, rhs_vec)
            dz_raw = delta[0::2]
            dQ_raw = delta[1::2]
            dz, dQ, damping = _apply_adaptive_damping(self.bed_elevations, z_py, Q_py, dz_raw, dQ_raw)
            z_py = z_py + dz
            Q_py = Q_py + dQ

            # Enforce minimum depth
            for i in range(len(self.sections_us_to_ds)):
                z_py[i] = _regularized_wse(self.sections_us_to_ds[i], z_py[i], WETTING_DEPTH_FT)

            max_update = max(np.max(np.abs(dz)), np.max(np.abs(dQ)))
            if max_update < self.tol:
                break

        # Run native timestep
        z_native, Q_native, inner_iters, max_error, converged = hydra_native.run_one_timestep_unsteady_1d_cpp(
            np.array(self.z_n, dtype=np.float64),
            np.array(self.Q_n, dtype=np.float64),
            reach_lengths,
            self.bed_elevations,
            area_values,
            conveyance_values,
            top_width_values,
            velocity_values,
            alpha_values,
            dkdz_values,
            self.dt,
            self.theta,
            self.Q_upstream_next,
            False,  # ds_is_stage
            self.ds_bc_value,
            self.ds_bc_ramp_factor,
            self.max_iter,
            self.tol,
            WETTING_DEPTH_FT,
        )

        # Compare
        self.assertTrue(np.allclose(z_native, z_py, rtol=1e-6, atol=1e-8),
                        f"z mismatch: native={z_native}, python={z_py}")
        self.assertTrue(np.allclose(Q_native, Q_py, rtol=1e-6, atol=1e-8),
                        f"Q mismatch: native={Q_native}, python={Q_py}")


if __name__ == '__main__':
    unittest.main()
