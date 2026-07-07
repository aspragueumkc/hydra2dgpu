"""
GPU numerical-only validation — trapezoidal channel flow with floodplain.

Reference: reference/anuga_validation_tests/analytical_exact/trapezoidal_channel/

No analytical solution used. Tests stable subcritical channel flow driven by
a prescribed inflow at the upstream boundary and free outflow downstream, with
Manning friction on a flat bed.

Physical setup
--------------
800 m × 14 m channel (floodplain width). Flat bed at z = 0 m.
Manning n = 0.03. Initial depth = 0.65 m everywhere.
Boundary conditions:
  - left  (x = 0):   INFLOW_Q = 0.5 m²/s (prescribed discharge)
  - right (x = Lx):  OPEN (transmissive, free outflow)
  - top / bottom:     WALL (reflective)

T_END = 50.0 s — enough time for the inflow to propagate partway down the
channel and establish approximate steady flow in the upstream section.

Checks
------
1. GPU active, all depths finite and non-negative (stability).
2. Depth standard deviation is non-zero after flow develops (non-trivial
   spatial structure from inflow BC).
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
class TestGPUTrapezoidalChannel(unittest.TestCase):
    anuga_reference = (
        "reference/anuga_validation_tests/analytical_exact/"
        "trapezoidal_channel/"
    )
    NX = 160
    NY = 3
    LX = 800.0
    LY = 14.0
    T_END = 50.0

    def _build(self):
        mod = _load_module()

        # Flat bed at z = 0
        def bed_func(x, y):
            return np.zeros_like(x)

        node_x, node_y, node_z, cell_nodes = _make_rect_mesh(
            self.NX, self.NY, self.LX, self.LY, zb_func=bed_func
        )

        # BC: left INFLOW_Q(2)=0.5, right OPEN(4), walls top/bottom
        bc = _channel_bc_arrays(self.NX, self.NY, 2, 0.5, 4, 0.0)

        mesh = mod.swe2d_build_mesh(node_x, node_y, node_z, cell_nodes, *bc)
        info = mod.swe2d_mesh_info(mesh)
        n_cells = info["n_cells"]
        nx_p1 = self.NX + 1

        # Centroids (original order)
        cell_cx = np.empty(n_cells)
        cell_cy = np.empty(n_cells)
        for ci in range(n_cells):
            row, col = divmod(ci // 2, self.NX)
            if ci % 2 == 0:
                n = [row * nx_p1 + col, row * nx_p1 + col + 1, (row + 1) * nx_p1 + col + 1]
            else:
                n = [row * nx_p1 + col, (row + 1) * nx_p1 + col + 1, (row + 1) * nx_p1 + col]
            cell_cx[ci] = float(np.mean(node_x[n]))
            cell_cy[ci] = float(np.mean(node_y[n]))

        # Initial depth = 0.65 m everywhere
        h0 = np.full(n_cells, 0.65, dtype=np.float64)

        solver = mod.swe2d_create_solver(
            mesh, h0, n_mann=0.03, cfl=0.45, dt_max=0.5, use_gpu=True, g=9.8
        )

        perm = mod.swe2d_get_cell_perm(mesh)
        cx_p = cell_cx[perm]
        cy_p = cell_cy[perm]
        return mod, mesh, solver, cx_p, cy_p

    def _run_to_end(self):
        mod, mesh, solver, cx_p, cy_p = self._build()
        t = 0.0
        last_diag = None
        while t < self.T_END:
            last_diag = mod.swe2d_step(solver, -1.0)
            t += last_diag["dt"]
        h, hu, hv = mod.swe2d_get_state(solver)
        mod.swe2d_destroy(solver)
        return h, cx_p, cy_p, last_diag

    def test_stability(self):
        """GPU active, all depths finite and non-negative."""
        h, _, _, last_diag = self._run_to_end()
        self.assertTrue(last_diag["gpu_active"])
        self.assertTrue(np.all(np.isfinite(h)))
        self.assertTrue(np.all(h >= -1e-12))

    def test_flow_develops(self):
        """Depth standard deviation is non-zero after flow develops."""
        h, cx_p, _, _ = self._run_to_end()
        # Interior cells (middle 80 % of channel, excluding BC influence)
        mask = (cx_p > 0.1 * self.LX) & (cx_p < 0.9 * self.LX)
        h_interior = h[mask]
        std_h = float(np.std(h_interior))
        self.assertGreater(
            std_h, 0.0,
            msg=f"Interior depth std = {std_h:.6e}, expected > 0 (no flow structure)"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
