"""
GPU-only A/B regression: orthogonal vs non-orthogonal mesh for channel flow.

Case definition (identical physics in both runs):
- Constant inflow (INFLOW_Q)
- Outlet normal-depth slope BC (NORMAL_DEPTH_SLOPE)
- Manning n = 0.02
- Same topology, BC assignment, dt controls, and initial condition
- Only interior node positions are perturbed in the non-orthogonal mesh
"""

import unittest
import numpy as np

from tests._swe2d_test_helpers import (
    _make_tri_channel_mesh,
    _channel_bc_edges,
    _manning_normal_depth,
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


def _run_case(mod, node_x, node_y, node_z, cell_nodes, bc_n0, bc_n1, bc_type, bc_val, h0, n_mann):
    mesh = mod.swe2d_build_mesh(node_x, node_y, node_z, cell_nodes, bc_n0, bc_n1, bc_type, bc_val)

    solver = mod.swe2d_create_solver(
        mesh, h0.copy(), n_mann=float(n_mann), cfl=0.45, dt_max=0.5,
        spatial_scheme=0, use_gpu=True)

    for _ in range(240):
        mod.swe2d_step(solver, -1.0)

    h, hu, hv = mod.swe2d_get_state(solver)
    mod.swe2d_destroy(solver)
    return h, hu, hv


@unittest.skipUnless(_load_module() is not None, "hydra_swe2d not built")
@unittest.skipUnless(_gpu_available(), "CUDA GPU not available")
class TestGPUChannelOrthVsNonOrth(unittest.TestCase):
    LX = 800.0
    LY = 20.0
    NX = 64
    NY = 8
    S0 = 1.0e-3
    N_MANN = 0.02
    Q_IN = 0.8

    def test_nonorth_matches_orthogonal_channel_solution_gpu(self):
        mod = _load_module()

        h_n = _manning_normal_depth(self.Q_IN, self.N_MANN, self.S0)
        n_cells = 2 * self.NX * self.NY
        h0 = np.full(n_cells, h_n, dtype=np.float64)

        bc_n0, bc_n1, bc_tp, bc_vl = _channel_bc_edges(self.NX, self.NY, self.Q_IN, self.S0)

        node_x_o, node_y_o, node_z_o, cell_nodes_o = _make_tri_channel_mesh(
            self.NX, self.NY, self.LX, self.LY, self.S0, skew_amp=0.0)

        skew_amp = 0.25 * (self.LX / self.NX)
        node_x_n, node_y_n, node_z_n, cell_nodes_n = _make_tri_channel_mesh(
            self.NX, self.NY, self.LX, self.LY, self.S0, skew_amp=skew_amp)

        h_o, hu_o, hv_o = _run_case(
            mod, node_x_o, node_y_o, node_z_o, cell_nodes_o,
            bc_n0, bc_n1, bc_tp, bc_vl, h0, self.N_MANN)

        h_n, hu_n, hv_n = _run_case(
            mod, node_x_n, node_y_n, node_z_n, cell_nodes_n,
            bc_n0, bc_n1, bc_tp, bc_vl, h0, self.N_MANN)

        self.assertTrue(np.isfinite(h_o).all() and np.isfinite(hu_o).all() and np.isfinite(hv_o).all())
        self.assertTrue(np.isfinite(h_n).all() and np.isfinite(hu_n).all() and np.isfinite(hv_n).all())

        rel_l2_h = float(np.linalg.norm(h_n - h_o) / max(np.linalg.norm(h_o), 1.0e-12))
        rel_l2_hu = float(np.linalg.norm(hu_n - hu_o) / max(np.linalg.norm(hu_o), 1.0e-12))

        q_o = float(np.mean(hu_o))
        q_n = float(np.mean(hu_n))

        self.assertLess(rel_l2_h, 0.10, f"Depth mismatch too large: rel_l2_h={rel_l2_h:.4f}")
        self.assertLess(rel_l2_hu, 0.15, f"Momentum mismatch too large: rel_l2_hu={rel_l2_hu:.4f}")
        self.assertGreater(q_o, 0.0, f"Orth mesh mean discharge should stay positive, got q={q_o:.4f}")
        self.assertGreater(q_n, 0.0, f"Non-orth mesh mean discharge should stay positive, got q={q_n:.4f}")
        self.assertLess(abs(q_n - q_o) / max(abs(q_o), 1.0e-12), 0.08,
            f"Orth/non-orth mean discharge mismatch too large: q_orth={q_o:.4f}, q_nonorth={q_n:.4f}")
