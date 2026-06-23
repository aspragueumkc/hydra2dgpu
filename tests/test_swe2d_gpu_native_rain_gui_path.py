"""Test that the native rain CN path works when Thiessen gages are configured.

Simulates the exact GUI workflow from run_controller.py: ThiessenRainCNForcing
built from mock gauge data → SWE2DRunSetupConfigurator.configure_native_rain_cn_forcing()
→ backend.set_rain_cn_forcing_native() → GPU kernel evaluates SCS-CN each step.
"""

import os
import sys
import unittest

import numpy as np

from swe2d.boundary_and_forcing.rainfall_hydrology import (
    Hyetograph,
    ThiessenRainCNForcing,
)
from swe2d.runtime.backend import SWE2DBackend, swe2d_available
from swe2d.runtime.runtime_setup_configurator import SWE2DRunSetupConfigurator
from tests._swe2d_test_helpers import _make_gmsh_triangle_mesh


def _load_module():
    try:
        import hydra_swe2d
        return hydra_swe2d
    except Exception:
        return None


def _gpu_available():
    mod = _load_module()
    if mod is None:
        return False
    try:
        return bool(mod.swe2d_gpu_available())
    except Exception:
        return False


def _gmsh_available():
    try:
        import gmsh
        return True
    except Exception:
        return False


_has_rain_cn_native = None


def _native_rain_cn_available():
    global _has_rain_cn_native
    if _has_rain_cn_native is not None:
        return _has_rain_cn_native
    mod = _load_module()
    if mod is None:
        _has_rain_cn_native = False
        return False
    try:
        _has_rain_cn_native = hasattr(mod, "swe2d_solver_set_rain_cn_forcing")
        return bool(_has_rain_cn_native)
    except Exception:
        _has_rain_cn_native = False
        return False


@unittest.skipUnless(swe2d_available(), "hydra_swe2d not built")
@unittest.skipUnless(_gpu_available(), "CUDA GPU not available")
@unittest.skipUnless(_gmsh_available(), "gmsh not installed")
@unittest.skipUnless(_native_rain_cn_available(), "set_rain_cn_forcing_native not in module")
class TestNativeRainCNGuiPath(unittest.TestCase):
    LX = 200.0
    LY = 80.0
    SIZE = 15.0

    def _build_backend(self, h0_scalar: float = 0.0) -> SWE2DBackend:
        node_x, node_y, node_z, cell_nodes, _, _ = _make_gmsh_triangle_mesh(
            self.LX, self.LY, self.SIZE,
        )
        backend = SWE2DBackend()
        backend.build_mesh(node_x, node_y, node_z, cell_nodes)
        n_cells = backend.n_cells
        backend.initialize(
            h0=np.full((n_cells,), float(h0_scalar), dtype=np.float64),
            hu0=np.zeros((n_cells,), dtype=np.float64),
            hv0=np.zeros((n_cells,), dtype=np.float64),
            n_mann=0.03,
            cfl=0.45,
            h_min=1.0e-6,
            dt_fixed=-1.0,
            dt_max=0.25,
            momentum_cap_min_speed=50.0,
            momentum_cap_celerity_mult=20.0,
            depth_cap=1.0e6,
            max_rel_depth_increase=2.0,
            shallow_damping_depth=1.0e-4,
            gpu_diag_sync_interval_steps=1,
            spatial_discretization=0,
        )
        return backend

    def _make_thiessen_forcing(self, n_cells: int) -> ThiessenRainCNForcing:
        hg = Hyetograph(
            times_s=np.array([0.0, 3600.0, 7200.0], dtype=np.float64),
            cumulative_mm=np.array([0.0, 50.0, 80.0], dtype=np.float64),
        )
        cell_to_gauge = np.zeros(n_cells, dtype=np.int32)
        cn = np.full(n_cells, 75.0, dtype=np.float64)
        return ThiessenRainCNForcing(
            cell_to_gauge=cell_to_gauge,
            gauge_hyetographs={0: hg},
            curve_number=cn,
            ia_ratio=0.2,
            infiltration_method="scs_cn",
        )

    def test_native_rain_cn_configured_and_applies_excess(self):
        backend = self._build_backend(h0_scalar=0.0)
        n_cells = backend.n_cells
        try:
            thiessen = self._make_thiessen_forcing(n_cells)
            configurator = SWE2DRunSetupConfigurator()
            payload = thiessen.build_native_preprocessed_payload()
            self.assertIn("cell_gage_idx", payload)
            self.assertIn("gage_offsets", payload)
            self.assertIn("hg_time_s", payload)
            self.assertIn("hg_cum_mm", payload)
            self.assertIn("cn", payload)
            self.assertIn("ia_ratio", payload)
            self.assertEqual(payload["cell_gage_idx"].shape[0], n_cells)
            self.assertEqual(float(payload["cn"][0]), 100.0)
            self.assertEqual(float(payload["ia_ratio"][0]), 0.0)

            mm_to_model_depth = 1.0e-3
            res = configurator.configure_native_rain_cn_forcing(
                backend=backend,
                thiessen_forcing=thiessen,
                mm_to_model_depth=mm_to_model_depth,
            )
            self.assertTrue(bool(res.get("configured", False)))
            self.assertEqual(str(res.get("infiltration_method", "")), "scs_cn")
            self.assertGreater(int(res.get("groups", 0)), 0)

            h, hu, hv = backend.get_state()
            self.assertAlmostEqual(float(np.sum(h)), 0.0)

            diags = backend.run(
                t_end=1800.0,
                dt_request=-1.0,
            )
            if len(diags) > 0:
                self.assertTrue(bool(diags[-1].get("gpu_active", False)))

            h, hu, hv = backend.get_state()
            self.assertTrue(np.isfinite(h).all())
            self.assertTrue(np.isfinite(hu).all())
            self.assertTrue(np.isfinite(hv).all())
            total_depth = float(np.sum(h))
            self.assertGreater(total_depth, 0.0,
                               "Native rain CN kernel should have added water depth")
        finally:
            backend.destroy()

    def test_native_rain_cn_identity_cn_produces_same_excess(self):
        backend = self._build_backend(h0_scalar=0.0)
        n_cells = backend.n_cells
        try:
            hg = Hyetograph(
                times_s=np.array([0.0, 1800.0, 3600.0], dtype=np.float64),
                cumulative_mm=np.array([0.0, 25.0, 50.0], dtype=np.float64),
            )
            cell_to_gauge = np.zeros(n_cells, dtype=np.int32)
            cn = np.full(n_cells, 75.0, dtype=np.float64)
            thiessen = ThiessenRainCNForcing(
                cell_to_gauge=cell_to_gauge,
                gauge_hyetographs={0: hg},
                curve_number=cn,
                ia_ratio=0.2,
                infiltration_method="scs_cn",
            )

            configurator = SWE2DRunSetupConfigurator()
            mm_to_model_depth = 1.0e-3
            res = configurator.configure_native_rain_cn_forcing(
                backend=backend,
                thiessen_forcing=thiessen,
                mm_to_model_depth=mm_to_model_depth,
            )
            self.assertTrue(bool(res.get("configured", False)))

            backend.run(t_end=3600.0, dt_request=-1.0)

            h, hu, hv = backend.get_state()
            self.assertTrue(np.isfinite(h).all())

            total_depth_model = float(np.sum(h))
            # Excess per cell ≈ 9.3mm for CN=75 with 50mm rain.
            # Summed across n_cells, total equivalent depth is n_cells * mm.
            max_possible_mm = 50.0 * n_cells  # raw rainfall * cells
            actual_mm = total_depth_model / mm_to_model_depth
            self.assertGreater(actual_mm, 1.0,
                               "Expected measurable excess rainfall volume")
            self.assertLess(actual_mm, max_possible_mm * 2.0,
                            f"Excess ({actual_mm:.0f} mm) should not wildly exceed raw rainfall ({max_possible_mm:.0f} mm)")
        finally:
            backend.destroy()

    def test_native_rain_cn_zero_excess_when_no_rain(self):
        backend = self._build_backend(h0_scalar=0.0)
        n_cells = backend.n_cells
        try:
            hg = Hyetograph(
                times_s=np.array([0.0, 3600.0], dtype=np.float64),
                cumulative_mm=np.array([0.0, 0.0], dtype=np.float64),
            )
            cell_to_gauge = np.zeros(n_cells, dtype=np.int32)
            cn = np.full(n_cells, 75.0, dtype=np.float64)
            thiessen = ThiessenRainCNForcing(
                cell_to_gauge=cell_to_gauge,
                gauge_hyetographs={0: hg},
                curve_number=cn,
                ia_ratio=0.2,
                infiltration_method="scs_cn",
            )

            configurator = SWE2DRunSetupConfigurator()
            res = configurator.configure_native_rain_cn_forcing(
                backend=backend,
                thiessen_forcing=thiessen,
                mm_to_model_depth=1.0e-3,
            )
            self.assertTrue(bool(res.get("configured", False)))

            diags = backend.run(
                t_end=1800.0,
                dt_request=-1.0,
                source_rate_callback=None,
                use_native_source_injection=False,
            )
            h, hu, hv = backend.get_state()
            total_depth = float(np.sum(h))
            self.assertAlmostEqual(total_depth, 0.0, places=6,
                                   msg="No rain → no water added")
        finally:
            backend.destroy()


if __name__ == "__main__":
    unittest.main(verbosity=2)
