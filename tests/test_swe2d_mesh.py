"""
test_swe2d_mesh.py
Validates unstructured mesh construction and edge classification.
"""

import unittest
import sys
import os
import numpy as np

# Allow importing from parent directory when run standalone
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _load_module():
    try:
        import backwater_swe2d
        return backwater_swe2d
    except ImportError as e:
        return None


@unittest.skipUnless(_load_module() is not None, "backwater_swe2d not built")
class TestSWE2DMeshRectangle(unittest.TestCase):
    """Build a simple 2-triangle rectangle and verify mesh properties."""

    def setUp(self):
        self.mod = _load_module()
        # Rectangle: [0,2] x [0,1], two triangles
        #   node layout:
        #   3──2
        #   |\ |
        #   | \|
        #   0──1
        self.node_x = np.array([0.0, 2.0, 2.0, 0.0])
        self.node_y = np.array([0.0, 0.0, 1.0, 1.0])
        self.node_z = np.zeros(4)
        # Two CCW triangles
        self.cell_nodes = np.array([0, 1, 2,   0, 2, 3], dtype=np.int32)

    def _build(self, bc_n0=None, bc_n1=None, bc_tp=None, bc_vl=None):
        if bc_n0 is None:
            bc_n0 = np.empty(0, dtype=np.int32)
            bc_n1 = np.empty(0, dtype=np.int32)
            bc_tp = np.empty(0, dtype=np.int32)
            bc_vl = np.empty(0, dtype=np.float64)
        return self.mod.swe2d_build_mesh(
            self.node_x, self.node_y, self.node_z,
            self.cell_nodes,
            bc_n0, bc_n1, bc_tp, bc_vl)

    def test_node_cell_counts(self):
        mesh = self._build()
        info = self.mod.swe2d_mesh_info(mesh)
        self.assertEqual(info["n_nodes"], 4)
        self.assertEqual(info["n_cells"], 2)

    def test_edge_count(self):
        mesh = self._build()
        info = self.mod.swe2d_mesh_info(mesh)
        # 2 triangles: 3 shared edges + 4 boundary edges = 5 total
        # But the diagonal is shared → interior (1); 4 boundary edges
        # Total unique edges = 5
        self.assertEqual(info["n_edges"], 5)

    def test_repr(self):
        mesh = self._build()
        r = repr(mesh)
        self.assertIn("SWE2DMeshHandle", r)
        self.assertIn("nodes=4", r)
        self.assertIn("cells=2", r)

    def test_degenerate_cell_raises(self):
        # Collinear nodes → zero area
        bad_nodes = np.array([0, 1, 2], dtype=np.int32)
        bad_x = np.array([0.0, 1.0, 2.0])
        bad_y = np.array([0.0, 0.0, 0.0])
        bad_z = np.zeros(3)
        with self.assertRaises(RuntimeError):
            self.mod.swe2d_build_mesh(
                bad_x, bad_y, bad_z, bad_nodes,
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.float64))

    def test_bc_classification(self):
        # Mark bottom edge (node 0→1) as INFLOW_Q
        bc_n0 = np.array([0], dtype=np.int32)
        bc_n1 = np.array([1], dtype=np.int32)
        bc_tp = np.array([2], dtype=np.int32)  # INFLOW_Q
        bc_vl = np.array([5.0])
        # Should build without error
        mesh = self._build(bc_n0, bc_n1, bc_tp, bc_vl)
        info = self.mod.swe2d_mesh_info(mesh)
        self.assertEqual(info["n_cells"], 2)


@unittest.skipUnless(_load_module() is not None, "backwater_swe2d not built")
class TestSWE2DMeshLarger(unittest.TestCase):
    """Verify mesh correctness for a larger structured triangulation."""

    def _make_structured_tri_mesh(self, nx, ny, Lx, Ly):
        """Generate a structured triangular mesh over [0,Lx] x [0,Ly]."""
        xs = np.linspace(0.0, Lx, nx + 1)
        ys = np.linspace(0.0, Ly, ny + 1)
        Xg, Yg = np.meshgrid(xs, ys)
        node_x = Xg.ravel()
        node_y = Yg.ravel()
        node_z = np.zeros_like(node_x)

        cells = []
        stride = nx + 1
        for j in range(ny):
            for i in range(nx):
                n00 = j * stride + i
                n10 = j * stride + i + 1
                n01 = (j + 1) * stride + i
                n11 = (j + 1) * stride + i + 1
                # Lower triangle
                cells.extend([n00, n10, n11])
                # Upper triangle
                cells.extend([n00, n11, n01])

        return node_x, node_y, node_z, np.array(cells, dtype=np.int32)

    def test_euler_formula(self):
        """V - E + F = 1 for a planar mesh (with boundary = 1 connected component)."""
        mod = _load_module()
        node_x, node_y, node_z, cell_nodes = self._make_structured_tri_mesh(4, 4, 100.0, 50.0)
        mesh = mod.swe2d_build_mesh(
            node_x, node_y, node_z, cell_nodes,
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float64))
        info = mod.swe2d_mesh_info(mesh)
        V = info["n_nodes"]
        E = info["n_edges"]
        F = info["n_cells"]
        # Euler: V - E + F = 1 for disk topology
        self.assertEqual(V - E + F, 1,
            msg=f"Euler formula failed: V={V} E={E} F={F} => V-E+F={V-E+F}")

@unittest.skipUnless(_load_module() is not None, "backwater_swe2d not built")
class TestSWE2DMeshPolygon(unittest.TestCase):
    """Verify native polygon-cell mesh construction path."""

    def test_single_quad_polygon(self):
        mod = _load_module()
        node_x = np.array([0.0, 2.0, 2.0, 0.0], dtype=np.float64)
        node_y = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
        node_z = np.zeros(4, dtype=np.float64)

        # One polygon cell with four vertices.
        cell_face_offsets = np.array([0, 4], dtype=np.int32)
        cell_face_nodes = np.array([0, 1, 2, 3], dtype=np.int32)

        mesh = mod.swe2d_build_mesh_poly(
            node_x,
            node_y,
            node_z,
            cell_face_offsets,
            cell_face_nodes,
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float64),
        )
        info = mod.swe2d_mesh_info(mesh)
        self.assertEqual(info["n_nodes"], 4)
        self.assertEqual(info["n_cells"], 1)
        self.assertEqual(info["n_edges"], 4)


if __name__ == "__main__":
    unittest.main()
