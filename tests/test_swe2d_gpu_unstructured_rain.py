"""GPU unstructured rain-on-grid stability regression tests.

These tests target the exact runtime path used by production GPU runs:
- unstructured gmsh mesh
- CUDA solver active
- native external source injection (device-resident rain source)
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from swe2d.runtime.backend import SWE2DBackend, swe2d_available
from tests.test_swe2d_unstructured import _make_gmsh_triangle_mesh


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
        import gmsh  # noqa: F401

        return True
    except Exception:
        return False


@unittest.skipUnless(swe2d_available(), "hydra_swe2d not built")
@unittest.skipUnless(_gpu_available(), "CUDA GPU not available")
@unittest.skipUnless(_gmsh_available(), "gmsh not installed")
class TestGPUUnstructuredRainOnGrid(unittest.TestCase):
    LX = 200.0
    LY = 80.0
    SIZE = 12.0

    def _build_backend(self, h0_scalar: float = 0.0, godunov_mode: int = 0) -> SWE2DBackend:
        node_x, node_y, node_z, cell_nodes, _, _ = _make_gmsh_triangle_mesh(
            self.LX,
            self.LY,
            self.SIZE,
        )

        backend = SWE2DBackend(use_gpu=True)
        backend.build_mesh(node_x, node_y, node_z, cell_nodes)
        n_cells = backend.n_cells
        h0 = np.full((n_cells,), float(h0_scalar), dtype=np.float64)
        hu0 = np.zeros((n_cells,), dtype=np.float64)
        hv0 = np.zeros((n_cells,), dtype=np.float64)

        backend.initialize(
            h0=h0,
            hu0=hu0,
            hv0=hv0,
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
            godunov_mode=int(godunov_mode),
        )
        return backend

    def test_uniform_rain_native_injection_stays_finite(self):
        backend = self._build_backend(h0_scalar=0.0)
        try:
            rain_mps = 2.0e-4  # 720 mm/hr stress test

            def src_cb(_t, _dt, h, _hu, _hv):
                return np.full_like(h, rain_mps, dtype=np.float64)

            diags = backend.run(
                t_end=60.0,
                dt_request=-1.0,
                source_rate_callback=src_cb,
                use_native_source_injection=True,
            )

            self.assertGreater(len(diags), 0)
            self.assertTrue(bool(diags[-1].get("gpu_active", False)))

            h, hu, hv = backend.get_state()
            self.assertTrue(np.isfinite(h).all())
            self.assertTrue(np.isfinite(hu).all())
            self.assertTrue(np.isfinite(hv).all())
            self.assertGreaterEqual(float(np.min(h)), 0.0)

            max_courant = max(float(d.get("max_courant", 0.0)) for d in diags)
            self.assertTrue(np.isfinite(max_courant), f"non-finite max_courant: {max_courant}")
            self.assertLess(max_courant, 10.0, f"run showed Courant blow-up: {max_courant:.3e}")
        finally:
            backend.destroy()

    def test_pulsed_extreme_rain_native_injection_stays_finite(self):
        backend = self._build_backend(h0_scalar=0.0)
        try:
            base = 5.0e-5   # 180 mm/hr
            pulse = 5.0e-4  # 1800 mm/hr short stress pulse

            def src_cb(t, _dt, h, _hu, _hv):
                rate = pulse if 15.0 <= float(t) <= 30.0 else base
                return np.full_like(h, rate, dtype=np.float64)

            diags = backend.run(
                t_end=90.0,
                dt_request=-1.0,
                source_rate_callback=src_cb,
                use_native_source_injection=True,
            )

            self.assertGreater(len(diags), 0)
            self.assertTrue(bool(diags[-1].get("gpu_active", False)))

            h, hu, hv = backend.get_state()
            self.assertTrue(np.isfinite(h).all())
            self.assertTrue(np.isfinite(hu).all())
            self.assertTrue(np.isfinite(hv).all())
            self.assertGreaterEqual(float(np.min(h)), 0.0)

            max_courant = max(float(d.get("max_courant", 0.0)) for d in diags)
            self.assertTrue(np.isfinite(max_courant), f"non-finite max_courant: {max_courant}")
            self.assertLess(max_courant, 20.0, f"pulse rain triggered Courant blow-up: {max_courant:.3e}")
        finally:
            backend.destroy()

    def test_uniform_rain_native_injection_rollout_mode_stays_finite(self):
        backend = self._build_backend(h0_scalar=0.0, godunov_mode=1)
        try:
            rain_mps = 2.0e-4

            def src_cb(_t, _dt, h, _hu, _hv):
                return np.full_like(h, rain_mps, dtype=np.float64)

            diags = backend.run(
                t_end=60.0,
                dt_request=-1.0,
                source_rate_callback=src_cb,
                use_native_source_injection=True,
            )

            self.assertGreater(len(diags), 0)
            self.assertTrue(bool(diags[-1].get("gpu_active", False)))

            h, hu, hv = backend.get_state()
            self.assertTrue(np.isfinite(h).all())
            self.assertTrue(np.isfinite(hu).all())
            self.assertTrue(np.isfinite(hv).all())
            self.assertGreaterEqual(float(np.min(h)), 0.0)

            max_courant = max(float(d.get("max_courant", 0.0)) for d in diags)
            self.assertTrue(np.isfinite(max_courant), f"non-finite max_courant: {max_courant}")
            self.assertLess(max_courant, 10.0, f"rollout rain run showed Courant blow-up: {max_courant:.3e}")
        finally:
            backend.destroy()


if __name__ == "__main__":
    unittest.main(verbosity=2)
