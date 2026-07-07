"""
GPU subcritical depth-expansion validation (steady state over a step drop).

Reference: reference/anuga_validation_tests/analytical_exact/subcritical_depth_expansion/
Original ANUGA setup: L=25 m, dx=0.1 m, W=3*dx=0.3 m, qA=1.0, hx=1.0 m.

Physical setup
--------------
Subcritical flow over a step drop in bed elevation from z=0.2 to z=0.0,
centered at x=10 m with e_w=2.0 m transition width. Discharge qA=1.0 m^2/s.
Downstream reference depth hx=1.0 m. The analytical solution is obtained
by solving the Bernoulli equation at each x location.

Test strategy
-------------
Initialise with the analytical depth profile from ``analytic_sol``.
Apply stage BCs (left and right) from analytical stage (h+z) at boundaries,
walls on top/bottom. Run to t=200 s. Compare SWE2D GPU depth along the
channel centreline against the ANUGA analytical solution.

Tolerance: L1 error < 2% of reference depth (0.02 m).
"""

import unittest
import numpy as np

from tests._swe2d_test_helpers import _make_rect_mesh
from tests._anuga_importer import import_anuga_module


_analytical = import_anuga_module(
    "reference/anuga_validation_tests/analytical_exact/"
    "subcritical_depth_expansion/analytical_depth_expansion.py"
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


BC_WALL = 1
BC_STAGE = 3


def _channel_bc_arrays(nx, ny, left_type, left_val, right_type, right_val):
    """BC arrays: left/right STAGE (3), walls top/bottom (1)."""
    stride = nx + 1
    n0, n1, tp, vl = [], [], [], []
    for j in range(ny):
        n0.append(j * stride); n1.append((j + 1) * stride); tp.append(left_type); vl.append(float(left_val))
    for j in range(ny):
        n0.append(j * stride + nx); n1.append((j + 1) * stride + nx); tp.append(right_type); vl.append(float(right_val))
    for i in range(nx):
        n0.append(i); n1.append(i + 1); tp.append(BC_WALL); vl.append(0.0)
    top0 = ny * stride
    for i in range(nx):
        n0.append(top0 + i); n1.append(top0 + i + 1); tp.append(BC_WALL); vl.append(0.0)
    return (
        np.array(n0, dtype=np.int32),
        np.array(n1, dtype=np.int32),
        np.array(tp, dtype=np.int32),
        np.array(vl, dtype=np.float64),
    )


def _bed_elevation(x, y):
    """Bed: step from 0.2 to 0.0 centred at x=10 with e_w=2.0 transition."""
    e_w = 2.0
    z = np.zeros_like(x)
    z = np.where(x <= 10.0 - e_w / 2.0, 0.2, z)
    mask = (x > 10.0 - e_w / 2.0) & (x < 10.0 + e_w / 2.0)
    z = np.where(mask, 0.2 - 0.2 * (x - (10.0 - e_w / 2.0)) / e_w, z)
    return z


@unittest.skipUnless(_load_module() is not None, "hydra_swe2d not built")
@unittest.skipUnless(_gpu_available(), "CUDA GPU not available")
class TestGPUSubcriticalDepthExpansion(unittest.TestCase):
    anuga_reference = "reference/anuga_validation_tests/analytical_exact/subcritical_depth_expansion/"
    NX = 250
    NY = 3
    LX = 25.0
    LY = 0.3
    Q = 1.0
    H_REF = 1.0
    T_END = 200.0

    def _build(self):
        mod = _load_module()
        node_x, node_y, node_z, cell_nodes = _make_rect_mesh(
            self.NX, self.NY, self.LX, self.LY,
            zb_func=_bed_elevation,
        )

        # Centroids (original order)
        n_cells = self.NX * self.NY * 2
        nx_p1 = self.NX + 1
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

        # Initial condition from analytical solution
        h_analytic, _ = _analytical.analytic_sol(cell_cx)
        h0 = np.maximum(h_analytic.astype(np.float64), 1e-12)

        # BC: stage from analytical at boundaries
        h_left, z_left = _analytical.analytic_sol(np.array([0.0]))
        stage_left = float(h_left[0] + z_left[0])
        h_right, z_right = _analytical.analytic_sol(np.array([self.LX]))
        stage_right = float(h_right[0] + z_right[0])

        bc_n0, bc_n1, bc_tp, bc_vl = _channel_bc_arrays(
            self.NX, self.NY,
            left_type=BC_STAGE, left_val=stage_left,
            right_type=BC_STAGE, right_val=stage_right,
        )

        mesh = mod.swe2d_build_mesh(node_x, node_y, node_z, cell_nodes, bc_n0, bc_n1, bc_tp, bc_vl)

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
        strip_tol = self.LY * 0.15
        mask = np.abs(cy_p - self.LY / 2.0) < strip_tol
        order = np.argsort(cx_p[mask])
        cx_strip = cx_p[mask][order]
        h_strip = h[mask][order]
        h_exact, _ = _analytical.analytic_sol(cx_strip)
        l1 = float(np.mean(np.abs(h_strip - h_exact)))
        limit = 0.02 * self.H_REF  # 2 % of reference depth
        self.assertLess(
            l1,
            limit,
            msg=f"Subcritical depth-expansion L1 error {l1:.6f} m exceeds limit ({limit:.4f} m)",
        )
