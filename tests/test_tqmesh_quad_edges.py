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

from swe2d_meshing import ConceptualModel, ConceptualRegion, QuadEdgeControl, TQMeshBackend


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


if __name__ == "__main__":
    unittest.main()
