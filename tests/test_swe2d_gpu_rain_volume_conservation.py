"""
Rain volume conservation test for all spatial / temporal schemes.

Applies a constant external source (rain) to a flat, closed box and checks
that the final water volume equals the integrated source input to tight
tolerance. This guards against RK stage bookkeeping over- or under-counting
source contributions.
"""

import os
import unittest
import numpy as np

from tests._swe2d_test_helpers import (
    _load_module,
    _gpu_available,
    _make_rect_mesh,
    _build_mesh,
)


class TestGPURainVolumeConservation(unittest.TestCase):
    """Closed box with constant rain: volume must match source integral."""

    @classmethod
    def setUpClass(cls):
        cls.mod = _load_module()

    def _run_case(self, spatial_scheme: int, temporal_order: int) -> dict:
        """Run a closed box with constant rain and return diagnostics."""
        if self.mod is None:
            self.skipTest("hydra_swe2d not built")
        if not _gpu_available():
            self.skipTest("CUDA GPU not available")

        mod = self.mod
        Lx, Ly = 100.0, 100.0
        nx, ny = 5, 5
        node_x, node_y, node_z, cell_nodes = _make_rect_mesh(nx, ny, Lx, Ly)
        n_cells = cell_nodes.size // 3

        # Closed domain: swe2d_build_mesh classifies unspecified boundary edges as WALL
        mesh = _build_mesh(mod, node_x, node_y, node_z, cell_nodes)

        # Compute cell areas from node geometry
        areas = np.empty(n_cells, dtype=np.float64)
        for c in range(n_cells):
            nodes = cell_nodes[3*c:3*c+3]
            x = node_x[nodes]
            y = node_y[nodes]
            areas[c] = 0.5 * abs(
                (x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0])
            )
        total_area = float(areas.sum())

        h0 = np.zeros(n_cells, dtype=np.float64)
        hu0 = np.zeros(n_cells, dtype=np.float64)
        hv0 = np.zeros(n_cells, dtype=np.float64)

        rain_rate = 1.0e-4  # m/s
        dt_fixed = 1.0      # s
        t_end = 10.0        # s

        solver = mod.swe2d_create_solver(
            mesh, h0, hu0, hv0,
            g=9.81,
            n_mann=0.03,
            h_min=1.0e-6,
            cfl=0.9,
            dt_max=dt_fixed,
            dt_fixed=dt_fixed,
            dt_initial=dt_fixed,
            use_gpu=True,
            temporal_order=temporal_order,
            spatial_scheme=spatial_scheme,
            max_rel_depth_increase=0.0,  # disable per-step depth cap so source volume is exact
            tiny_mode=0,
        )

        source = np.full(n_cells, rain_rate, dtype=np.float64)
        mod.swe2d_solver_set_external_sources(solver, source)

        t = 0.0
        step = 0
        while t < t_end and step < 1000:
            diag = mod.swe2d_step(solver, -1.0)
            dt = diag["dt"]
            t += dt
            step += 1

        h, _, _ = mod.swe2d_get_state(solver)
        final_volume = float((h * areas).sum())
        expected_volume = rain_rate * total_area * t
        rel_err = abs(final_volume - expected_volume) / max(expected_volume, 1.0e-12)

        mod.swe2d_destroy(solver)

        return {
            "t": t,
            "steps": step,
            "final_volume": final_volume,
            "expected_volume": expected_volume,
            "rel_err": rel_err,
        }

    def _check(self, spatial_scheme: int, temporal_order: int):
        info = self._run_case(spatial_scheme, temporal_order)
        self.assertLess(
            info["rel_err"],
            1.0e-6,
            f"spatial={spatial_scheme} temporal={temporal_order}: "
            f"final_volume={info['final_volume']:.6e} "
            f"expected={info['expected_volume']:.6e} "
            f"rel_err={info['rel_err']:.6e}",
        )

    # ── Spatial scheme sweep with RK2 (baseline) ────────────────────────────
    def test_spatial0_rain_volume_rk2(self):
        self._check(0, 2)

    def test_spatial1_rain_volume_rk2(self):
        self._check(1, 2)

    def test_spatial2_rain_volume_rk2(self):
        self._check(2, 2)

    def test_spatial3_rain_volume_rk2(self):
        self._check(3, 2)

    def test_spatial4_rain_volume_rk2(self):
        self._check(4, 2)

    def test_spatial6_rain_volume_rk2(self):
        self._check(6, 2)

    # ── Spatial scheme sweep with RK3 (the fixed path) ───────────────────────
    def test_spatial0_rain_volume_rk3(self):
        self._check(0, 3)

    def test_spatial1_rain_volume_rk3(self):
        self._check(1, 3)

    def test_spatial2_rain_volume_rk3(self):
        self._check(2, 3)

    def test_spatial3_rain_volume_rk3(self):
        self._check(3, 3)

    def test_spatial4_rain_volume_rk3(self):
        self._check(4, 3)

    def test_spatial6_rain_volume_rk3(self):
        self._check(6, 3)

    # ── Forward-Euler sanity check ─────────────────────────────────────────
    def test_spatial0_rain_volume_euler(self):
        self._check(0, 1)


if __name__ == "__main__":
    unittest.main()
