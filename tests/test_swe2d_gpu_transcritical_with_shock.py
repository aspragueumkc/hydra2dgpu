"""
GPU transcritical flow over a bump with a standing shock (steady state).

Reference: reference/anuga_validation_tests/analytical_exact/transcritical_with_shock/
Refs: Houghton & Kasahara (1968), Delestre et al (2012, SWASHES).

Physical setup
--------------
Steady transcritical flow (q = 0.18 m²/s) passes over a smooth bump in a
25 m channel. The flow accelerates from subcritical to supercritical over
the bump crest and recovers through a standing hydraulic shock on the
downstream face. Bump: z(x) = max(0, 0.2 − 0.05·(x−10)²). Upstream stage
≈ 0.4137 m; downstream tailwater stage = 0.33 m.

Test strategy
--------------
Initialise with the analytical depth profile from ``analytic_sol``. Apply
inflow Q at the left boundary and stage BC at the right. Run to t = 100 s
allowing the shock to stabilise. Compare centreline depth against the
ANUGA analytical solution.

Tolerance: L1 error < 3 % of reference depth (0.5 m) → 0.015 m.
"""

import unittest
import numpy as np

from tests._swe2d_test_helpers import _make_rect_mesh
from tests._anuga_importer import import_anuga_module


_analytical = import_anuga_module(
    "reference/anuga_validation_tests/analytical_exact/"
    "transcritical_with_shock/analytical_with_shock.py"
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


def _channel_bc_inflow_stage(nx: int, ny: int, q_left: float, stage_right: float):
    """BC arrays: left INFLOW_Q (2), right STAGE (3), walls top/bottom."""
    stride = nx + 1
    n0, n1, tp, val = [], [], [], []
    for j in range(ny):
        n0.append(j * stride)
        n1.append((j + 1) * stride)
        tp.append(2)
        val.append(float(q_left))
    for j in range(ny):
        n0.append(j * stride + nx)
        n1.append((j + 1) * stride + nx)
        tp.append(3)
        val.append(float(stage_right))
    for i in range(nx):
        n0.append(i)
        n1.append(i + 1)
        tp.append(1)
        val.append(0.0)
    top0 = ny * stride
    for i in range(nx):
        n0.append(top0 + i)
        n1.append(top0 + i + 1)
        tp.append(1)
        val.append(0.0)
    return (
        np.array(n0, dtype=np.int32),
        np.array(n1, dtype=np.int32),
        np.array(tp, dtype=np.int32),
        np.array(val, dtype=np.float64),
    )


@unittest.skipUnless(_load_module() is not None, "hydra_swe2d not built")
@unittest.skipUnless(_gpu_available(), "CUDA GPU not available")
class TestGPUTranscriticalWithShock(unittest.TestCase):
    anuga_reference = (
        "reference/anuga_validation_tests/analytical_exact/transcritical_with_shock/"
    )
    NX = 250
    NY = 3
    LX = 25.0
    LY = 0.3
    Q = 0.18
    STAGE_RIGHT = 0.33
    T_END = 100.0

    def _build(self):
        mod = _load_module()
        node_x, node_y, _, cell_nodes = _make_rect_mesh(
            self.NX, self.NY, self.LX, self.LY
        )

        # Bed elevation — bump
        node_z = np.zeros_like(node_x)
        mask = (node_x >= 8.0) & (node_x <= 12.0)
        node_z[mask] = 0.2 - 0.05 * (node_x[mask] - 10.0) ** 2

        bc = _channel_bc_inflow_stage(self.NX, self.NY, self.Q, self.STAGE_RIGHT)

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

        # Initial condition from analytical solution
        h_analytic, _ = _analytical.analytic_sol(cell_cx)
        h0 = np.maximum(h_analytic.astype(np.float64), 1e-12)

        solver = mod.swe2d_create_solver(
            mesh, h0, n_mann=0.0, cfl=0.45, dt_max=0.5, use_gpu=True, g=9.8
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
        limit = 0.03 * 0.5  # 3 % of reference depth scale
        self.assertLess(
            l1,
            limit,
            msg=f"Transcritical w/ shock L1 error {l1:.6f} m exceeds limit ({limit:.4f} m)",
        )
