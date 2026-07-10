"""
GPU lake-at-rest validation — steep island (complex bed orography).

Reference: reference/anuga_validation_tests/analytical_exact/lake_at_rest_steep_island/

Physical setup
--------------
A 2000 m × 5 m channel with a steep, multi-slope island carved into the
bed elevation (peak 6.0 m). Initial stage = 4.5 m everywhere. All
boundaries are reflective (WALL). No friction. The exact solution is
stage = 4.5 m for all time: the water surface must remain flat.

Test strategy
--------------
Set h0 = max(4.5 − bed, 0). Run to t = 5 s. Compute the L∞ departure of
the final stage (h + bed) from 4.5 m across all cells.

Tolerance: L∞ error < 1 × 10⁻⁶ m (machine precision for lake at rest).
"""

import unittest
import numpy as np

from tests._swe2d_test_helpers import _make_rect_mesh


# Inline bed function from numerical_steep_island.py (avoids ANUGA-side
# domain-construction side effects at import time).
def _bed_elevation(x, y):
    z = np.zeros(len(x), dtype=np.float64)
    for i in range(len(x)):
        xi = float(x[i])
        if 0 <= xi < 200.0:
            z[i] = -0.01 * (xi - 200.0) + 4.0
        elif 200.0 <= xi < 300.0:
            z[i] = -0.02 * (xi - 200.0) + 4.0
        elif 300.0 <= xi < 400.0:
            z[i] = -0.01 * (xi - 300.0) + 2.0
        elif 400.0 <= xi < 550.0:
            z[i] = (-1.0 / 75.0) * (xi - 400.0) + 2.0
        elif 550.0 <= xi < 700.0:
            z[i] = (1.0 / 11250.0) * (xi - 550.0) * (xi - 550.0)
        elif 700.0 <= xi < 800.0:
            z[i] = 0.03 * (xi - 700.0)
        elif 800.0 <= xi < 900.0:
            z[i] = -0.03 * (xi - 800.0) + 3.0
        elif 900.0 <= xi < 1000.0:
            z[i] = 6.0
        elif 1000.0 <= xi < 1400.0:
            z[i] = (-1.0 / 20000.0) * (xi - 1000.0) * (xi - 1400.0)
        elif 1400.0 <= xi < 1500.0:
            z[i] = 0.0
        elif 1500.0 <= xi < 1700.0:
            z[i] = 3.0
        elif 1700.0 <= xi < 1800.0:
            z[i] = -0.03 * (xi - 1700.0) + 3.0
        else:
            z[i] = (4.5 / 40000.0) * (xi - 1800.0) * (xi - 1800.0) + 2.0
    return z


def _load_module():
    try:
        import hydra_swe2d
        return hydra_swe2d
    except ImportError:
        return None


def _gpu_available():
    mod = _load_module()
    if mod is None:
        return False
    try:
        return mod.swe2d_gpu_available()
    except Exception:
        return False


@unittest.skipUnless(_load_module() is not None, "hydra_swe2d not built")
@unittest.skipUnless(_gpu_available(), "CUDA GPU not available")
class TestGPULakeAtRestSteepIsland(unittest.TestCase):
    anuga_reference = (
        "reference/anuga_validation_tests/analytical_exact/"
        "lake_at_rest_steep_island/"
    )
    NX = 2000
    NY = 5
    LX = 2000.0
    LY = 5.0
    STAGE = 4.5
    T_END = 5.0

    def _build(self, spatial_scheme: int = 0):
        mod = _load_module()
        node_x, node_y, node_z, cell_nodes = _make_rect_mesh(
            self.NX, self.NY, self.LX, self.LY, zb_func=_bed_elevation
        )

        # All boundaries default to reflective wall (empty BC arrays).
        mesh = mod.swe2d_build_mesh(
            node_x,
            node_y,
            node_z,
            cell_nodes,
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float64),
        )
        info = mod.swe2d_mesh_info(mesh)
        n_cells = info["n_cells"]
        nx_p1 = self.NX + 1

        # Centroids (original order)
        cell_cx = np.empty(n_cells)
        cell_cy = np.empty(n_cells)
        cell_zb = np.empty(n_cells)
        for ci in range(n_cells):
            row, col = divmod(ci // 2, self.NX)
            if ci % 2 == 0:
                n = [row * nx_p1 + col, row * nx_p1 + col + 1, (row + 1) * nx_p1 + col + 1]
            else:
                n = [row * nx_p1 + col, (row + 1) * nx_p1 + col + 1, (row + 1) * nx_p1 + col]
            cell_cx[ci] = float(np.mean(node_x[n]))
            cell_cy[ci] = float(np.mean(node_y[n]))
            cell_zb[ci] = float(np.mean(node_z[n]))

        # Initial condition: stage = self.STAGE, but capped so h ≥ 0
        h0 = np.maximum(self.STAGE - cell_zb, 0.0).astype(np.float64)

        cfl = 0.4 if spatial_scheme == 8 else 0.45
        solver = mod.swe2d_create_solver(
            mesh, h0, n_mann=0.0, cfl=cfl, dt_max=0.5, use_gpu=True, g=9.8,
            spatial_scheme=spatial_scheme,
        )

        perm = mod.swe2d_get_cell_perm(mesh)
        zb_p = cell_zb[perm]
        return mod, mesh, solver, zb_p

    def _run_to_end(self, spatial_scheme: int = 0):
        mod, mesh, solver, zb_p = self._build(spatial_scheme)
        t = 0.0
        last_diag = None
        while t < self.T_END:
            last_diag = mod.swe2d_step(solver, -1.0)
            t += last_diag["dt"]
        h, hu, hv = mod.swe2d_get_state(solver)
        mod.swe2d_destroy(solver)
        return h, zb_p, last_diag

    def test_stability(self):
        h, _, last_diag = self._run_to_end()
        self.assertTrue(last_diag["gpu_active"])
        self.assertTrue(np.all(np.isfinite(h)))
        self.assertTrue(np.all(h >= -1e-12))

    def test_linf_error_lake_at_rest(self):
        h, zb_p, _ = self._run_to_end()
        wet = h > 1e-6
        if not np.any(wet):
            self.skipTest("All cells are dry — cannot test lake at rest")
        stage = h[wet] + zb_p[wet]
        linf = float(np.max(np.abs(stage - self.STAGE)))
        limit = 1e-6
        self.assertLess(
            linf,
            limit,
            msg=f"Steep island L∞ stage error on wet cells {linf:.3e} m exceeds limit ({limit:.1e} m)",
        )

    def test_new_schemes_stability(self):
        """Sweep schemes 5, 6, 8 — must remain stable (no NaN, no negative depth)."""
        for scheme, name in [(5, "Barth-Jespersen"), (6, "WENO3"), (8, "MP5")]:
            h, _, last_diag = self._run_to_end(spatial_scheme=scheme)
            self.assertTrue(last_diag["gpu_active"], f"GPU inactive for {name}")
            self.assertTrue(np.all(np.isfinite(h)), f"NaN/Inf depth for {name}")
            self.assertTrue(np.all(h >= -1e-10), f"Negative depth for {name}: min={h.min():.4e}")
