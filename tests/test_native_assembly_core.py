#!/usr/bin/env python3
"""Parity checks for optional native unsteady assembly core."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unsteady_model import _assemble_system_core


class TestNativeAssemblyCore(unittest.TestCase):
    def test_native_assembly_core_matches_python(self):
        try:
            import hydra_native
        except Exception:
            self.skipTest('native module not built/importable in test environment')

        reach_lengths = np.array([420.0, 395.0], dtype=float)
        z_values = np.array([101.8, 101.4, 101.1], dtype=float)
        q_values = np.array([240.0, 232.0, 225.0], dtype=float)
        area_values = np.array([145.0, 142.0, 139.0], dtype=float)
        conveyance_values = np.array([5100.0, 4980.0, 4860.0], dtype=float)
        top_width_values = np.array([85.0, 82.0, 80.0], dtype=float)
        velocity_values = q_values / area_values
        alpha_values = np.array([1.08, 1.07, 1.06], dtype=float)
        dkdz_values = np.array([180.0, 176.0, 171.0], dtype=float)

        py_ab, py_rhs = _assemble_system_core(
            reach_lengths=reach_lengths,
            z_values=z_values,
            q_values=q_values,
            area_values=area_values,
            conveyance_values=conveyance_values,
            top_width_values=top_width_values,
            velocity_values=velocity_values,
            alpha_values=alpha_values,
            dkdz_values=dkdz_values,
            dt=30.0,
            theta=0.6,
            q_upstream_next=250.0,
            ds_is_stage=False,
            ds_bc_value=0.0012,
            ds_bc_ramp_factor=0.75,
        )
        native_ab, native_rhs = hydra_native.assemble_system_core(
            reach_lengths,
            z_values,
            q_values,
            area_values,
            conveyance_values,
            top_width_values,
            velocity_values,
            alpha_values,
            dkdz_values,
            30.0,
            0.6,
            250.0,
            False,
            0.0012,
            0.75,
        )

        self.assertTrue(np.allclose(native_ab, py_ab, rtol=1e-10, atol=1e-12))
        self.assertTrue(np.allclose(native_rhs, py_rhs, rtol=1e-10, atol=1e-12))


if __name__ == '__main__':
    unittest.main()