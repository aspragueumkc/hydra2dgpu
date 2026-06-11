"""
GPU-only well-balanced lake-at-rest test.

A flat rectangular domain with a sinusoidal bed perturbation is initialised
with constant free-surface elevation (h + zb = const).  After N steps the
free-surface should remain constant to machine precision (< 1e-10 m).

This directly tests the hydrostatic reconstruction and bed-slope correction
on the GPU solver path.
"""

import unittest
import numpy as np

from tests._swe2d_test_helpers import _make_rect_mesh


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
class TestGPULakeAtRest(unittest.TestCase):
    NX, NY = 20, 10
    LX, LY = 200.0, 100.0
    ETA0   = 1.0
    A_BED  = 0.3
    N_MANN = 0.000
    N_STEPS = 100

    def _zb(self, x, y):
        return self.A_BED * np.sin(np.pi * x / self.LX) * np.cos(np.pi * y / self.LY)

    def test_free_surface_constant_gpu(self):
        mod = _load_module()
        node_x, node_y, node_z, cell_nodes = _make_rect_mesh(
            self.NX, self.NY, self.LX, self.LY, zb_func=self._zb)

        mesh = mod.swe2d_build_mesh(
            node_x, node_y, node_z, cell_nodes,
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float64))

        info = mod.swe2d_mesh_info(mesh)
        n_cells = info["n_cells"]

        stride = self.NX + 1
        zb_cell = np.zeros(n_cells)
        for ci in range(n_cells):
            row, col = divmod(ci // 2, self.NX)
            if ci % 2 == 0:
                nodes = [row * stride + col,
                         row * stride + col + 1,
                         (row + 1) * stride + col + 1]
            else:
                nodes = [row * stride + col,
                         (row + 1) * stride + col + 1,
                         (row + 1) * stride + col]
            zb_cell[ci] = np.mean(node_z[nodes])

        h0 = np.maximum(0.0, self.ETA0 - zb_cell)

        solver = mod.swe2d_create_solver(
            mesh, h0, n_mann=self.N_MANN, cfl=0.45, dt_max=5.0, use_gpu=True)

        for _ in range(self.N_STEPS):
            mod.swe2d_step(solver, -1.0)

        h, hu, hv = mod.swe2d_get_state(solver)
        mod.swe2d_destroy(solver)

        eta = h + zb_cell
        wet = h > 1e-6
        self.assertTrue(wet.any(), "All cells are dry after lake-at-rest test!")
        deviation = np.max(np.abs(eta[wet] - self.ETA0))
        self.assertLess(deviation, 1e-10,
            msg=f"GPU lake-at-rest free-surface deviation: {deviation:.2e} m (limit 1e-10)")
