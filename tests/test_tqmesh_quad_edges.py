import os
import sys
import unittest
import numpy as np

here = os.path.dirname(os.path.dirname(__file__))
if here not in sys.path:
    sys.path.insert(0, here)

build_dir = os.path.join(here, "build")
if build_dir not in sys.path:
    sys.path.insert(0, build_dir)

from swe2d_meshing import ConceptualArc, ConceptualModel, ConceptualRegion, QuadEdgeControl, TQMeshBackend, GmshBackend


class TestTQMeshQuadEdges(unittest.TestCase):
    def test_multi_vertex_quad_edges_generate_quads(self):
        try:
            import backwater_tqmesh  # noqa: F401
        except ImportError:
            self.skipTest("backwater_tqmesh module is not built")

        region = ConceptualRegion(
            region_id=1,
            ring_xy=[(0.0, 0.0), (100.0, 0.0), (100.0, 50.0), (0.0, 50.0)],
            default_size=10.0,
            default_cell_type="quadrilateral",
            edge_lengths=[10.0, 10.0, 10.0, 10.0],
        )
        quad_edges = [
            QuadEdgeControl(1, 1, [(0.0, 0.0), (30.0, 0.0), (70.0, 0.0), (100.0, 0.0)], 10.0, 2, 5.0, 1.0),
            QuadEdgeControl(1, 2, [(100.0, 0.0), (100.0, 15.0), (100.0, 35.0), (100.0, 50.0)], 10.0, 2, 5.0, 1.0),
            QuadEdgeControl(1, 3, [(100.0, 50.0), (60.0, 50.0), (25.0, 50.0), (0.0, 50.0)], 10.0, 2, 5.0, 1.0),
            QuadEdgeControl(1, 4, [(0.0, 50.0), (0.0, 30.0), (0.0, 10.0), (0.0, 0.0)], 10.0, 2, 5.0, 1.0),
        ]
        model = ConceptualModel(nodes=[], arcs=[], regions=[region], constraints=[], quad_edges=quad_edges)

        mesh = TQMeshBackend().generate(model)

        quads = sum(1 for cell_type in mesh.cell_type.tolist() if cell_type == "quadrilateral")
        tris = sum(1 for cell_type in mesh.cell_type.tolist() if cell_type == "triangular")

        self.assertGreater(quads, 0)
        self.assertGreater(len(mesh.node_x), 0)
        self.assertGreater(quads, tris)

    def test_adjacent_regions_share_interface_edges(self):
        try:
            import backwater_tqmesh  # noqa: F401
        except ImportError:
            self.skipTest("backwater_tqmesh module is not built")

        regions = [
            ConceptualRegion(
                region_id=1,
                ring_xy=[(0.0, 0.0), (50.0, 0.0), (50.0, 50.0), (0.0, 50.0)],
                default_size=10.0,
                default_cell_type="triangular",
            ),
            ConceptualRegion(
                region_id=2,
                ring_xy=[(50.0, 0.0), (100.0, 0.0), (100.0, 50.0), (50.0, 50.0)],
                default_size=10.0,
                default_cell_type="triangular",
            ),
        ]
        model = ConceptualModel(nodes=[], arcs=[], regions=regions, constraints=[], quad_edges=[])

        mesh = TQMeshBackend().generate(model)

        edge_regions = {}
        offs = mesh.cell_face_offsets.astype(np.int32)
        nodes = mesh.cell_face_nodes.astype(np.int32)
        region_ids = mesh.region_id.astype(np.int32)
        for face_idx in range(offs.size - 1):
            poly = nodes[int(offs[face_idx]):int(offs[face_idx + 1])]
            rid = int(region_ids[face_idx])
            for i in range(poly.size):
                a = int(poly[i])
                b = int(poly[(i + 1) % poly.size])
                edge = (a, b) if a < b else (b, a)
                edge_regions.setdefault(edge, set()).add(rid)

        shared_edges = [edge for edge, rids in edge_regions.items() if rids == {1, 2}]
        self.assertGreater(len(shared_edges), 0)


class TestGmshConformingInterfaces(unittest.TestCase):
    """Verify the Gmsh backend produces conforming meshes at region boundaries."""

    def _try_gmsh(self):
        try:
            import gmsh  # noqa: F401
        except ImportError:
            self.skipTest("gmsh not available")

    def _shared_edge_count(self, mesh):
        """Count edges that are referenced by cells from two different regions."""
        offs = mesh.cell_face_offsets.astype(np.int32)
        nodes = mesh.cell_face_nodes.astype(np.int32)
        region_ids = mesh.region_id.astype(np.int32)
        edge_regions: dict = {}
        for face_idx in range(offs.size - 1):
            poly = nodes[int(offs[face_idx]):int(offs[face_idx + 1])]
            rid = int(region_ids[face_idx])
            for i in range(poly.size):
                a = int(poly[i])
                b = int(poly[(i + 1) % poly.size])
                edge = (min(a, b), max(a, b))
                edge_regions.setdefault(edge, set()).add(rid)
        return sum(1 for rids in edge_regions.values() if len(rids) > 1)

    def _count_edges_on_segment(self, mesh, start_xy, end_xy, tol=1.0e-6):
        offs = mesh.cell_face_offsets.astype(np.int32)
        nodes = mesh.cell_face_nodes.astype(np.int32)
        node_x = mesh.node_x.astype(np.float64)
        node_y = mesh.node_y.astype(np.float64)
        x0, y0 = float(start_xy[0]), float(start_xy[1])
        x1, y1 = float(end_xy[0]), float(end_xy[1])
        seg_dx = x1 - x0
        seg_dy = y1 - y0
        seg_len2 = seg_dx * seg_dx + seg_dy * seg_dy
        if seg_len2 <= 0.0:
            return 0

        def _point_on_segment(px, py):
            cross = (px - x0) * seg_dy - (py - y0) * seg_dx
            if abs(cross) > tol:
                return False
            dot = (px - x0) * seg_dx + (py - y0) * seg_dy
            if dot < -tol or dot > seg_len2 + tol:
                return False
            return True

        count = 0
        seen = set()
        for face_idx in range(offs.size - 1):
            poly = nodes[int(offs[face_idx]):int(offs[face_idx + 1])]
            for i in range(poly.size):
                a = int(poly[i])
                b = int(poly[(i + 1) % poly.size])
                edge = (min(a, b), max(a, b))
                if edge in seen:
                    continue
                seen.add(edge)
                ax = float(node_x[a])
                ay = float(node_y[a])
                bx = float(node_x[b])
                by = float(node_y[b])
                if _point_on_segment(ax, ay) and _point_on_segment(bx, by):
                    count += 1
        return count

    def test_tri_tri_interface_is_conforming(self):
        """Two adjacent triangular regions must share interface edges."""
        self._try_gmsh()
        regions = [
            ConceptualRegion(
                region_id=1,
                ring_xy=[(0.0, 0.0), (50.0, 0.0), (50.0, 50.0), (0.0, 50.0)],
                default_size=15.0,
                default_cell_type="triangular",
            ),
            ConceptualRegion(
                region_id=2,
                ring_xy=[(50.0, 0.0), (100.0, 0.0), (100.0, 50.0), (50.0, 50.0)],
                default_size=15.0,
                default_cell_type="triangular",
            ),
        ]
        model = ConceptualModel(nodes=[], arcs=[], regions=regions, constraints=[], quad_edges=[])
        mesh = GmshBackend().generate(model)
        shared = self._shared_edge_count(mesh)
        self.assertGreater(shared, 0, "tri-tri: no shared interface edges found — mesh is non-conforming")

    def test_tri_quad_interface_is_conforming(self):
        """Triangular region adjacent to a quadrilateral region must share interface edges."""
        self._try_gmsh()
        regions = [
            ConceptualRegion(
                region_id=1,
                ring_xy=[(0.0, 0.0), (50.0, 0.0), (50.0, 50.0), (0.0, 50.0)],
                default_size=15.0,
                default_cell_type="triangular",
            ),
            ConceptualRegion(
                region_id=2,
                ring_xy=[(50.0, 0.0), (100.0, 0.0), (100.0, 50.0), (50.0, 50.0)],
                default_size=15.0,
                default_cell_type="quadrilateral",
            ),
        ]
        model = ConceptualModel(nodes=[], arcs=[], regions=regions, constraints=[], quad_edges=[])
        mesh = GmshBackend().generate(model)
        shared = self._shared_edge_count(mesh)
        self.assertGreater(shared, 0, "tri-quad: no shared interface edges found — mesh is non-conforming")

    def test_tri_cartesian_interface_is_conforming(self):
        """Triangular region adjacent to a cartesian (structured-quad) region must share interface edges."""
        self._try_gmsh()
        regions = [
            ConceptualRegion(
                region_id=1,
                ring_xy=[(0.0, 0.0), (50.0, 0.0), (50.0, 50.0), (0.0, 50.0)],
                default_size=15.0,
                default_cell_type="triangular",
            ),
            ConceptualRegion(
                region_id=2,
                ring_xy=[(50.0, 0.0), (100.0, 0.0), (100.0, 50.0), (50.0, 50.0)],
                default_size=15.0,
                default_cell_type="cartesian",
                edge_lengths=[15.0, 15.0, 15.0, 15.0],
            ),
        ]
        model = ConceptualModel(nodes=[], arcs=[], regions=regions, constraints=[], quad_edges=[])
        mesh = GmshBackend().generate(model)
        shared = self._shared_edge_count(mesh)
        self.assertGreater(shared, 0, "tri-cartesian: no shared interface edges found — mesh is non-conforming")

    def test_multivertex_breakline_is_embedded_as_polyline_segments(self):
        """A topo_arc polyline must survive as piecewise linear embedded mesh edges."""
        self._try_gmsh()
        region = ConceptualRegion(
            region_id=1,
            ring_xy=[(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)],
            default_size=12.0,
            default_cell_type="triangular",
        )
        arc = ConceptualArc(
            arc_id=1,
            points_xy=[(20.0, 20.0), (50.0, 20.0), (50.0, 80.0), (80.0, 80.0)],
        )
        model = ConceptualModel(nodes=[], arcs=[arc], regions=[region], constraints=[], quad_edges=[])

        mesh = GmshBackend().generate(model)

        self.assertGreater(self._count_edges_on_segment(mesh, (20.0, 20.0), (50.0, 20.0)), 0)
        self.assertGreater(self._count_edges_on_segment(mesh, (50.0, 20.0), (50.0, 80.0)), 0)
        self.assertGreater(self._count_edges_on_segment(mesh, (50.0, 80.0), (80.0, 80.0)), 0)


if __name__ == "__main__":
    unittest.main()

