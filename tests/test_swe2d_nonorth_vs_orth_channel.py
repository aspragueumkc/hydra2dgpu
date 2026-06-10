"""
A/B regression: orthogonal vs non-orthogonal mesh for the same channel-flow case.

Case definition (identical physics in both runs):
- Constant inflow (INFLOW_Q)
- Outlet normal-depth slope BC (NORMAL_DEPTH_SLOPE)
- Manning n = 0.02
- Same topology, BC assignment, dt controls, and initial condition
- Only interior node positions are perturbed in the non-orthogonal mesh
"""

import os
import sys
import unittest

import numpy as np




def _load_module():
    try:
        import hydra_swe2d
        return hydra_swe2d
    except ImportError:
        return None


def _manning_normal_depth(q, n_mann, s0):
    # Wide-rectangular Manning relation: q = (1/n) h^(5/3) sqrt(S)
    return (q * n_mann / np.sqrt(s0)) ** (3.0 / 5.0)


def _make_tri_channel_mesh(nx, ny, lx, ly, s0, skew_amp=0.0):
    """
    Build a triangular channel mesh over [0,lx]x[0,ly].

    Non-orthogonality is introduced by perturbing interior node x positions,
    while keeping all boundary nodes fixed so BC geometry remains identical.
    """
    xs = np.linspace(0.0, lx, nx + 1)
    ys = np.linspace(0.0, ly, ny + 1)
    x_base, y_base = np.meshgrid(xs, ys)
    x = x_base.copy()
    y = y_base.copy()

    if skew_amp > 0.0:
        # Smooth interior shear-like perturbation; boundaries remain unchanged.
        bump = np.sin(np.pi * x_base / lx) * np.sin(np.pi * y_base / ly)
        x += float(skew_amp) * bump

    node_x = x.ravel().astype(np.float64)
    node_y = y.ravel().astype(np.float64)

    # Keep bed identical as a function of streamwise coordinate in the
    # reference (orthogonal) channel definition.
    node_z = (s0 * (lx - x_base)).ravel().astype(np.float64)

    stride = nx + 1
    cells = []
    for j in range(ny):
        for i in range(nx):
            n00 = j * stride + i
            n10 = j * stride + i + 1
            n01 = (j + 1) * stride + i
            n11 = (j + 1) * stride + i + 1
            cells.extend([n00, n10, n11])
            cells.extend([n00, n11, n01])

    cell_nodes = np.asarray(cells, dtype=np.int32)

    return node_x, node_y, node_z, cell_nodes


def _channel_bc_edges(nx, ny, q_in, s0):
    """
    Explicitly provide all boundary edges to keep BC setup identical.

    BC types:
    - left:   INFLOW_Q (2), value=q_in
    - right:  NORMAL_DEPTH_SLOPE (7), value=s0
    - top/bot WALL (1), value=0
    """
    stride = nx + 1

    n0 = []
    n1 = []
    tp = []
    val = []

    # Left boundary, bottom->top
    for j in range(ny):
        n0.append(j * stride)
        n1.append((j + 1) * stride)
        tp.append(2)
        val.append(float(q_in))

    # Right boundary, bottom->top
    for j in range(ny):
        n0.append(j * stride + nx)
        n1.append((j + 1) * stride + nx)
        tp.append(7)
        val.append(float(s0))

    # Bottom boundary, left->right
    for i in range(nx):
        n0.append(i)
        n1.append(i + 1)
        tp.append(1)
        val.append(0.0)

    # Top boundary, left->right
    top0 = ny * stride
    for i in range(nx):
        n0.append(top0 + i)
        n1.append(top0 + i + 1)
        tp.append(1)
        val.append(0.0)

    return (
        np.asarray(n0, dtype=np.int32),
        np.asarray(n1, dtype=np.int32),
        np.asarray(tp, dtype=np.int32),
        np.asarray(val, dtype=np.float64),
    )


def _run_case(mod, node_x, node_y, node_z, cell_nodes, bc_n0, bc_n1, bc_type, bc_val, h0, n_mann):
    mesh = mod.swe2d_build_mesh(node_x, node_y, node_z, cell_nodes, bc_n0, bc_n1, bc_type, bc_val)

    solver = mod.swe2d_create_solver(
        mesh,
        h0.copy(),
        n_mann=float(n_mann),
        cfl=0.45,
        dt_max=0.5,
        spatial_scheme=0,
        use_gpu=False,
    )

    for _ in range(240):
        mod.swe2d_step(solver, -1.0)

    h, hu, hv = mod.swe2d_get_state(solver)
    mod.swe2d_destroy(solver)

    return h, hu, hv


@unittest.skipUnless(_load_module() is not None, "hydra_swe2d not built")
class TestChannelOrthVsNonOrth(unittest.TestCase):
    """
    Compare orthogonal and non-orthogonal meshes for identical channel physics.
    """

    LX = 800.0
    LY = 20.0
    NX = 64
    NY = 8

    S0 = 1.0e-3
    N_MANN = 0.02
    Q_IN = 0.8  # unit discharge [m^2/s]

    def test_nonorth_matches_orthogonal_channel_solution(self):
        mod = _load_module()

        h_n = _manning_normal_depth(self.Q_IN, self.N_MANN, self.S0)
        n_cells = 2 * self.NX * self.NY
        h0 = np.full(n_cells, h_n, dtype=np.float64)

        bc_n0, bc_n1, bc_tp, bc_vl = _channel_bc_edges(self.NX, self.NY, self.Q_IN, self.S0)

        node_x_o, node_y_o, node_z_o, cell_nodes_o = _make_tri_channel_mesh(
            self.NX, self.NY, self.LX, self.LY, self.S0, skew_amp=0.0
        )

        # Perturb interior nodes by ~25% of dx to create non-orthogonality.
        skew_amp = 0.25 * (self.LX / self.NX)
        node_x_n, node_y_n, node_z_n, cell_nodes_n = _make_tri_channel_mesh(
            self.NX, self.NY, self.LX, self.LY, self.S0, skew_amp=skew_amp
        )

        h_o, hu_o, hv_o = _run_case(
            mod,
            node_x_o,
            node_y_o,
            node_z_o,
            cell_nodes_o,
            bc_n0,
            bc_n1,
            bc_tp,
            bc_vl,
            h0,
            self.N_MANN,
        )

        h_n, hu_n, hv_n = _run_case(
            mod,
            node_x_n,
            node_y_n,
            node_z_n,
            cell_nodes_n,
            bc_n0,
            bc_n1,
            bc_tp,
            bc_vl,
            h0,
            self.N_MANN,
        )

        self.assertTrue(np.isfinite(h_o).all() and np.isfinite(hu_o).all() and np.isfinite(hv_o).all())
        self.assertTrue(np.isfinite(h_n).all() and np.isfinite(hu_n).all() and np.isfinite(hv_n).all())

        # Compare state fields directly (same topology/cell ordering).
        rel_l2_h = float(np.linalg.norm(h_n - h_o) / max(np.linalg.norm(h_o), 1.0e-12))
        rel_l2_hu = float(np.linalg.norm(hu_n - hu_o) / max(np.linalg.norm(hu_o), 1.0e-12))

        # Interior-mean discharge consistency to prescribed inflow.
        q_o = float(np.mean(hu_o))
        q_n = float(np.mean(hu_n))

        self.assertLess(rel_l2_h, 0.10, f"Depth mismatch too large: rel_l2_h={rel_l2_h:.4f}")
        self.assertLess(rel_l2_hu, 0.15, f"Momentum mismatch too large: rel_l2_hu={rel_l2_hu:.4f}")
        self.assertGreater(q_o, 0.0, f"Orth mesh mean discharge should stay positive, got q={q_o:.4f}")
        self.assertGreater(q_n, 0.0, f"Non-orth mesh mean discharge should stay positive, got q={q_n:.4f}")
        self.assertLess(abs(q_n - q_o) / max(abs(q_o), 1.0e-12), 0.08,
                f"Orth/non-orth mean discharge mismatch too large: q_orth={q_o:.4f}, q_nonorth={q_n:.4f}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
