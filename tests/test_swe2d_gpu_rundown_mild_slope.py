"""
GPU numerical-only validation — rundown on a mild slope.

Reference: reference/anuga_validation_tests/analytical_exact/rundown_mild_slope/

No analytical solution used. Tests stable flow down a mild linear slope
driven by a fixed stage boundary at the upstream end and an open (transmissive)
downstream boundary.

Physical setup
--------------
100 m × 10 m channel (narrow, fast to run). Linear bed slope:
    z(x) = −x/10   (descending from 0 to −10)
Initial condition: thin water layer following the bed + 0.01 m:
    stage = z(x) + 0.01   →   h0 ≈ 0.01 m everywhere.
Boundary conditions:
  - left  (x = 0):   STAGE = 0.09266 (normal depth from Manning equation)
  - right (x = Lx):  OPEN (transmissive, free outflow)
  - top / bottom:     WALL (reflective)

Manning n = 0.03. T_END = 10.0 s.

Checks
------
1. GPU active, all depths finite and non-negative (stability).
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


def _channel_bc_arrays(nx, ny, left_type, left_val, right_type, right_val):
    """BC arrays for a channel: custom left/right, walls top/bottom."""
    stride = nx + 1
    n0, n1, tp, vl = [], [], [], []
    for j in range(ny):
        n0.append(j * stride)
        n1.append((j + 1) * stride)
        tp.append(left_type)
        vl.append(float(left_val))
    for j in range(ny):
        n0.append(j * stride + nx)
        n1.append((j + 1) * stride + nx)
        tp.append(right_type)
        vl.append(float(right_val))
    for i in range(nx):
        n0.append(i)
        n1.append(i + 1)
        tp.append(1)
        vl.append(0.0)
    top0 = ny * stride
    for i in range(nx):
        n0.append(top0 + i)
        n1.append(top0 + i + 1)
        tp.append(1)
        vl.append(0.0)
    return (
        np.array(n0, dtype=np.int32),
        np.array(n1, dtype=np.int32),
        np.array(tp, dtype=np.int32),
        np.array(vl, dtype=np.float64),
    )


@unittest.skipUnless(_load_module() is not None, "hydra_swe2d not built")
@unittest.skipUnless(_gpu_available(), "CUDA GPU not available")
class TestGPURundownMildSlope(unittest.TestCase):
    anuga_reference = (
        "reference/anuga_validation_tests/analytical_exact/"
        "rundown_mild_slope/"
    )
    NX = 50
    NY = 5
    LX = 100.0
    LY = 10.0
    T_END = 10.0

    def _build(self):
        mod = _load_module()

        # Bed: z(x) = -x/10  (linear slope, 0 at x=0, -10 at x=100)
        def bed_func(x, y):
            return -x / 10.0

        node_x, node_y, node_z, cell_nodes = _make_rect_mesh(
            self.NX, self.NY, self.LX, self.LY, zb_func=bed_func
        )

        # BC: left STAGE(3)=0.09266, right OPEN(4), walls top/bottom
        bc = _channel_bc_arrays(self.NX, self.NY, 3, 0.09266, 4, 0.0)

        mesh = mod.swe2d_build_mesh(node_x, node_y, node_z, cell_nodes, *bc)
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

        # Initial stage = z(x) + 0.01 m → thin water layer everywhere
        initial_stage = cell_zb + 0.01
        h0 = np.maximum(initial_stage - cell_zb, 0.0).astype(np.float64)

        solver = mod.swe2d_create_solver(
            mesh, h0, n_mann=0.03, cfl=0.45, dt_max=0.5, use_gpu=True, g=9.8
        )

        perm = mod.swe2d_get_cell_perm(mesh)
        zb_p = cell_zb[perm]
        cx_p = cell_cx[perm]
        cy_p = cell_cy[perm]
        return mod, mesh, solver, zb_p, cx_p, cy_p

    def _run_to_end(self):
        mod, mesh, solver, zb_p, cx_p, cy_p = self._build()
        t = 0.0
        last_diag = None
        while t < self.T_END:
            last_diag = mod.swe2d_step(solver, -1.0)
            t += last_diag["dt"]
        h, hu, hv = mod.swe2d_get_state(solver)
        mod.swe2d_destroy(solver)
        return h, zb_p, cx_p, cy_p, last_diag

    def test_stability(self):
        """GPU active, all depths finite and non-negative."""
        h, _, _, _, last_diag = self._run_to_end()
        self.assertTrue(last_diag["gpu_active"])
        self.assertTrue(np.all(np.isfinite(h)))
        self.assertTrue(np.all(h >= -1e-12))


if __name__ == "__main__":
    unittest.main(verbosity=2)
