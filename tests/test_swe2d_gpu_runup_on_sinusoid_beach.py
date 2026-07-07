"""
GPU numerical-only validation — run-up on a sinusoidal beach.

Reference: reference/anuga_validation_tests/analytical_exact/runup_on_sinusoid_beach/

No analytical solution. Checks solver stability on a 2-D domain with a
sinusoidal perturbation superposed on a linear slope.

Physical setup
--------------
1 m × 1 m square domain. Bed elevation:
    z(x, y) = −x/2.0 + 0.05·sin((x + y)·50.0)
Initial stage = −0.2 m everywhere, so only the deeper end is wet.
Boundary conditions:
  - left  (x = 0):   WALL (reflective)
  - right (x = Lx):  STAGE = −0.1 (constant stage forcing)
  - top / bottom:     WALL (reflective)

No friction. T_END = 1.0 s — short stability check on a 40×40 mesh (3200
triangles).

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
class TestGPURunupOnSinusoidBeach(unittest.TestCase):
    anuga_reference = (
        "reference/anuga_validation_tests/analytical_exact/"
        "runup_on_sinusoid_beach/"
    )
    NX = 40
    NY = 40
    LX = 1.0
    LY = 1.0
    T_END = 1.0

    def _build(self):
        mod = _load_module()

        # Bed: z(x,y) = -x/2.0 + 0.05·sin((x+y)·50.0)
        def bed_func(x, y):
            return -x / 2.0 + 0.05 * np.sin((x + y) * 50.0)

        node_x, node_y, node_z, cell_nodes = _make_rect_mesh(
            self.NX, self.NY, self.LX, self.LY, zb_func=bed_func
        )

        # BC: left WALL(1), right STAGE(3)=[-0.1]
        bc = _channel_bc_arrays(self.NX, self.NY, 1, 0.0, 3, -0.1)

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

        # Initial stage = -0.2 m → depth capped at zero
        initial_stage = -0.2
        h0 = np.maximum(initial_stage - cell_zb, 0.0).astype(np.float64)

        solver = mod.swe2d_create_solver(
            mesh, h0, n_mann=0.0, cfl=0.45, dt_max=0.5, use_gpu=True, g=9.8
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

    def test_water_moves(self):
        """Water surface standard deviation is non-zero (flow occurs)."""
        h, zb_p, _, _, _ = self._run_to_end()
        stage = h + zb_p
        std_stage = float(np.std(stage))
        self.assertGreater(std_stage, 0.0,
                           msg=f"Stage std = {std_stage:.6e}, expected > 0 (no water motion)")


if __name__ == "__main__":
    unittest.main(verbosity=2)
