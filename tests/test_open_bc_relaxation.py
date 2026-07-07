import unittest
import gc
import numpy as np
from swe2d.runtime.backend import SWE2DBackend, BCType

_NODE_X = np.array([0.0, 100.0, 200.0, 300.0, 400.0, 500.0], dtype=np.float64)
_NODE_Y = np.array([0.0, 0.0, 0.0, 100.0, 100.0, 100.0], dtype=np.float64)
_NODE_Z = np.zeros(6, dtype=np.float64)
_CELL_NODES = np.array([[0, 1, 3], [1, 4, 3], [1, 2, 4], [2, 5, 4]], dtype=np.int32)
_BC_N0 = np.array([0, 2, 0, 3], dtype=np.int32)
_BC_N1 = np.array([3, 5, 1, 4], dtype=np.int32)
_BC_TP = np.array([BCType.WALL, BCType.OPEN, BCType.OPEN, BCType.OPEN], dtype=np.int32)
_BC_VL = np.zeros(4, dtype=np.float64)

def _make_backend():
    b = SWE2DBackend()
    b.build_mesh(_NODE_X, _NODE_Y, _NODE_Z, _CELL_NODES,
                 bc_edge_node0=_BC_N0, bc_edge_node1=_BC_N1,
                 bc_edge_type=_BC_TP, bc_edge_val=_BC_VL)
    return b, _BC_N0, _BC_N1, _BC_TP, _BC_VL


class TestOpenBCRelaxation(unittest.TestCase):

    def tearDown(self):
        gc.collect()

    def test_relaxation_zero_no_change(self):
        h0 = np.zeros(4, dtype=np.float64)
        h0[0] = 1.0

        b1, *_ = _make_backend()
        b1.initialize(h0=h0, n_mann=0.03, h_min=1e-4, cfl=0.45, dt_max=1.0,
                       open_bc_relaxation=0.0)
        d1 = b1.step()

        b2, *_ = _make_backend()
        b2.initialize(h0=h0, n_mann=0.03, h_min=1e-4, cfl=0.45, dt_max=1.0)
        d2 = b2.step()

        self.assertAlmostEqual(d1["dt"], d2["dt"])
        h1, hu1, hv1 = b1.get_state()
        h2, hu2, hv2 = b2.get_state()
        np.testing.assert_array_almost_equal(h1, h2)
        np.testing.assert_array_almost_equal(hu1, hu2)
        np.testing.assert_array_almost_equal(hv1, hv2)

    def test_relaxation_affects_open_only(self):
        h0 = np.zeros(4, dtype=np.float64)
        h0[0] = 1.0

        bc_tp = _BC_TP.copy()
        bc_tp[0] = BCType.WALL
        bc_tp[1] = BCType.OPEN

        b1 = SWE2DBackend()
        b1.build_mesh(_NODE_X, _NODE_Y, _NODE_Z, _CELL_NODES,
                       bc_edge_node0=_BC_N0, bc_edge_node1=_BC_N1,
                       bc_edge_type=bc_tp, bc_edge_val=_BC_VL)
        b1.initialize(h0=h0, n_mann=0.03, h_min=1e-4, cfl=0.45, dt_max=1.0,
                       open_bc_relaxation=0.5)
        b1.step()
        h_relax, _, _ = b1.get_state()

        b2 = SWE2DBackend()
        b2.build_mesh(_NODE_X, _NODE_Y, _NODE_Z, _CELL_NODES,
                       bc_edge_node0=_BC_N0, bc_edge_node1=_BC_N1,
                       bc_edge_type=bc_tp, bc_edge_val=_BC_VL)
        b2.initialize(h0=h0, n_mann=0.03, h_min=1e-4, cfl=0.45, dt_max=1.0,
                       open_bc_relaxation=0.0)
        b2.step()
        h_no_relax, _, _ = b2.get_state()

        wall_cell = 0
        self.assertAlmostEqual(h_relax[wall_cell], h_no_relax[wall_cell])

    def test_relaxation_per_edge_override(self):
        h0 = np.zeros(4, dtype=np.float64)
        h0[0] = 1.0

        b1, bc_n0, bc_n1, bc_tp, bc_vl = _make_backend()
        b1.initialize(h0=h0, n_mann=0.03, h_min=1e-4, cfl=0.45, dt_max=1.0,
                       open_bc_relaxation=0.0)
        relax = np.full(bc_n0.size, 0.5, dtype=np.float64)
        b1.set_boundary_relaxation(bc_n0, bc_n1, relax)
        b1.step()
        h1, _, _ = b1.get_state()

        b2, *_ = _make_backend()
        b2.initialize(h0=h0, n_mann=0.03, h_min=1e-4, cfl=0.45, dt_max=1.0,
                       open_bc_relaxation=0.5)
        b2.step()
        h2, _, _ = b2.get_state()

        np.testing.assert_array_almost_equal(h1, h2)
