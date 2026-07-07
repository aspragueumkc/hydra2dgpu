"""
GPU parabolic basin oscillation validation (Thacker-type 1D canal).

Reference: reference/anuga_validation_tests/analytical_exact/parabolic_basin/
Original ANUGA setup: Lx=40 m ([-20, 20]), Ly=2 m, dx=0.2 m,
D0=4 m, L_parab=10 m, A=2 m, g=9.8.

Physical setup
--------------
Parabolic basin: z(x) = D0 * (x/L_parab)^2, D0=4 m, L_parab=10 m.
Initial water surface is set from the Thacker analytical solution at t=0.
All boundaries are reflective walls (no BC arrays needed). The water
surface oscillates within the parabolic basin.

Test strategy
-------------
Run to t=1.0 s (short run, captures initial oscillation phase). Compare
SWE2D GPU water depth against the Thacker analytical solution at final
time. The GPU solver applies a RCMK cell permutation, so we use
`swe2d_get_cell_perm` to align cell coordinates with the returned state.

Tolerance: L1 error < 10% of D0 (0.4 m). This is deliberately loose since
1 s is a short simulation time and the initial transient may not fully
settle into the analytical periodic orbit.
"""

import unittest
import numpy as np

from tests._swe2d_test_helpers import _make_rect_mesh
from tests._anuga_importer import import_anuga_module


_analytical = import_anuga_module(
    "reference/anuga_validation_tests/analytical_exact/"
    "parabolic_basin/analytical_parabolic_basin.py"
)


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


D0 = 4.0
L_PARAB = 10.0
A = 2.0


@unittest.skipUnless(_load_module() is not None, "hydra_swe2d not built")
@unittest.skipUnless(_gpu_available(), "CUDA GPU not available")
class TestGPUParabolicBasin(unittest.TestCase):
    anuga_reference = "reference/anuga_validation_tests/analytical_exact/parabolic_basin/"
    NX = 200
    NY = 10
    LX = 40.0
    LY = 2.0
    T_END = 1.0

    def _build(self):
        mod = _load_module()
        node_x, node_y, _, cell_nodes = _make_rect_mesh(self.NX, self.NY, self.LX, self.LY)

        # Center mesh at x=0
        node_x = node_x - self.LX / 2.0

        # Parabolic bed: z = D0 * (x/L_parab)**2
        node_z = D0 * (node_x / L_PARAB) ** 2

        # All walls — empty BC arrays
        mesh = mod.swe2d_build_mesh(
            node_x, node_y, node_z, cell_nodes,
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
        for ci in range(n_cells):
            row, col = divmod(ci // 2, self.NX)
            if ci % 2 == 0:
                n = [row * nx_p1 + col, row * nx_p1 + col + 1, (row + 1) * nx_p1 + col + 1]
            else:
                n = [row * nx_p1 + col, (row + 1) * nx_p1 + col + 1, (row + 1) * nx_p1 + col]
            cell_cx[ci] = float(np.mean(node_x[n]))
            cell_cy[ci] = float(np.mean(node_y[n]))

        # Initial condition from analytical at t=0
        _, h0_raw, _, _ = _analytical.analytic_cannal(
            cell_cx, 0.0, D0=D0, L=L_PARAB, A=A, g=9.8
        )
        h0 = np.maximum(h0_raw.astype(np.float64), 1e-12)

        perm = mod.swe2d_get_cell_perm(mesh)
        cx_p = cell_cx[perm]
        cy_p = cell_cy[perm]

        solver = mod.swe2d_create_solver(
            mesh, h0, n_mann=0.0, cfl=0.45, dt_max=0.5, use_gpu=True, g=9.8
        )
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
        h, _, _, last_diag = self._run_to_end()
        self.assertTrue(last_diag["gpu_active"])
        self.assertTrue(np.all(np.isfinite(h)))
        self.assertTrue(np.all(h >= -1e-12))

    def test_l1_error_vs_anuga(self):
        h, cx_p, cy_p, _ = self._run_to_end()
        # Use all cells for comparison
        order = np.argsort(cx_p)
        cx_sort = cx_p[order]
        h_sort = h[order]
        _, h_exact, _, _ = _analytical.analytic_cannal(
            cx_sort, self.T_END, D0=D0, L=L_PARAB, A=A, g=9.8
        )
        l1 = float(np.mean(np.abs(h_sort - h_exact)))
        limit = 0.10 * D0  # 10 % of D0 (loose for short run)
        self.assertLess(
            l1,
            limit,
            msg=f"Parabolic basin L1 error {l1:.6f} m exceeds limit ({limit:.4f} m)",
        )
