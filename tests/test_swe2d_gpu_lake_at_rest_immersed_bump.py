"""
GPU lake-at-rest validation — smooth immersed bump.

Reference: reference/anuga_validation_tests/analytical_exact/lake_at_rest_immersed_bump/

Physical setup
--------------
A 25 m × 5 m channel with a smooth parabolic bump in the middle:
z(x) = max(0, 0.2 − 0.05·(x−10)²). Initial stage = 0.5 m everywhere,
so the bump crest (z_max = 0.2) is fully submerged. All boundaries
reflective (WALL). No friction. The exact solution is stage = 0.5 m
for all time.

Test strategy
--------------
Set h0 = max(0.5 − bed, 0). Run to t = 5 s. Compute the L∞ departure of
the final stage (h + bed) from 0.5 m across all cells.

Tolerance: L∞ error < 1 × 10⁻⁶ m (machine precision for lake at rest).
"""

import unittest
import numpy as np

from tests._swe2d_test_helpers import _make_rect_mesh


def _bed_elevation(x, y):
    z = np.zeros(len(x), dtype=np.float64)
    mask = (x >= 8.0) & (x <= 12.0)
    z[mask] = 0.2 - 0.05 * (x[mask] - 10.0) ** 2
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
class TestGPULakeAtRestImmersedBump(unittest.TestCase):
    anuga_reference = (
        "reference/anuga_validation_tests/analytical_exact/"
        "lake_at_rest_immersed_bump/"
    )
    NX = 25
    NY = 5
    LX = 25.0
    LY = 5.0
    STAGE = 0.5
    T_END = 5.0

    def _build(self):
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

        # Centroids and cell-bed elevation (original order)
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

        # Initial condition: stage = self.STAGE, capped so h ≥ 0
        h0 = np.maximum(self.STAGE - cell_zb, 0.0).astype(np.float64)

        solver = mod.swe2d_create_solver(
            mesh, h0, n_mann=0.0, cfl=0.45, dt_max=0.5, use_gpu=True, g=9.8
        )

        perm = mod.swe2d_get_cell_perm(mesh)
        zb_p = cell_zb[perm]
        return mod, mesh, solver, zb_p

    def _run_to_end(self):
        mod, mesh, solver, zb_p = self._build()
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
        stage = h + zb_p
        linf = float(np.max(np.abs(stage - self.STAGE)))
        limit = 1e-6
        self.assertLess(
            linf,
            limit,
            msg=f"Immersed bump L∞ stage error {linf:.3e} m exceeds limit ({limit:.1e} m)",
        )
