"""
GPU-only 1D Stoker dam-break analytical comparison on a 2D structured mesh.

Setup
-----
Domain : 1 000 m × 50 m rectangle.
IC     : h_L = 2.0 m for x ≤ 500 m, h_R = 0.5 m for x > 500 m; u = v = 0.
Bed    : flat (zb = 0).
BCs    : all walls (default).
Run    : t = 10 s.
Metric : L∞ error in h(x, t=10) versus Stoker exact solution, < 20 % of h_L.
"""

import unittest
import numpy as np

from tests._swe2d_test_helpers import _make_rect_mesh, stoker_dam_break


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
class TestGPUDamBreak1D(unittest.TestCase):
    NX = 100
    NY = 5
    LX = 1000.0
    LY = 50.0
    H_L = 2.0
    H_R = 0.5
    T_END = 10.0

    def test_stoker_linf_error_gpu(self):
        mod = _load_module()
        node_x, node_y, node_z, cell_nodes = _make_rect_mesh(
            self.NX, self.NY, self.LX, self.LY)

        mesh = mod.swe2d_build_mesh(
            node_x, node_y, node_z, cell_nodes,
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float64))

        info = mod.swe2d_mesh_info(mesh)
        n_cells = info["n_cells"]
        nx_p1 = self.NX + 1
        cell_cx = np.zeros(n_cells)
        for ci in range(n_cells):
            row, col = divmod(ci // 2, self.NX)
            stride = nx_p1
            if ci % 2 == 0:
                nodes = [row * stride + col,
                         row * stride + col + 1,
                         (row + 1) * stride + col + 1]
            else:
                nodes = [row * stride + col,
                         (row + 1) * stride + col + 1,
                         (row + 1) * stride + col]
            cell_cx[ci] = np.mean(node_x[nodes])

        h0 = np.where(cell_cx <= self.LX / 2.0, self.H_L, self.H_R)

        solver = mod.swe2d_create_solver(
            mesh, h0, n_mann=0.0, cfl=0.45, dt_max=0.5, use_gpu=True)

        t = 0.0
        last_diag = None
        while t < self.T_END:
            last_diag = mod.swe2d_step(solver, -1.0)
            t += last_diag["dt"]

        self.assertTrue(last_diag["gpu_active"])
        h, hu, hv = mod.swe2d_get_state(solver)
        mod.swe2d_destroy(solver)

        mid_row = self.NY // 2
        start = mid_row * self.NX * 2
        end = start + self.NX * 2
        cx_strip = cell_cx[start:end]
        h_strip = h[start:end]

        x_shifted = cx_strip - self.LX / 2.0
        h_exact = stoker_dam_break(x_shifted, self.T_END, self.H_L, self.H_R)
        linf = np.max(np.abs(h_strip - h_exact))
        limit = 0.20 * self.H_L
        self.assertLess(linf, limit,
            msg=f"GPU dam-break L∞ error {linf:.4f} m exceeds limit ({limit:.4f} m)")
