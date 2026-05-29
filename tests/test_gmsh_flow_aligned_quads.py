import unittest

from swe2d.mesh.meshing import (
    ConceptualModel,
    ConceptualRegion,
    QuadEdgeControl,
    _harmonize_transfinite_shared_quad_interfaces,
    _gmsh_flow_align_region_preflight,
    _gmsh_flow_aligned_curve_counts,
    _gmsh_interface_coincidence_report,
)


class TestGmshFlowAlignedCurveCounts(unittest.TestCase):
    def test_counts_follow_edge_spacing_and_match_opposites(self):
        controls = [
            QuadEdgeControl(region_id=1, edge_id=1, points_xy=[(0.0, 0.0), (100.0, 0.0)], target_size=10.0),
            QuadEdgeControl(region_id=1, edge_id=2, points_xy=[(100.0, 0.0), (100.0, 50.0)], target_size=10.0),
            QuadEdgeControl(region_id=1, edge_id=3, points_xy=[(100.0, 50.0), (0.0, 50.0)], target_size=20.0),
            QuadEdgeControl(region_id=1, edge_id=4, points_xy=[(0.0, 50.0), (0.0, 0.0)], target_size=None),
        ]

        counts = _gmsh_flow_aligned_curve_counts(controls, fallback_size=10.0)

        self.assertEqual(counts, [11, 6, 11, 6])

    def test_invalid_control_count_returns_none(self):
        controls = [
            QuadEdgeControl(region_id=1, edge_id=1, points_xy=[(0.0, 0.0), (1.0, 0.0)], target_size=1.0),
            QuadEdgeControl(region_id=1, edge_id=2, points_xy=[(1.0, 0.0), (1.0, 1.0)], target_size=1.0),
            QuadEdgeControl(region_id=1, edge_id=3, points_xy=[(1.0, 1.0), (0.0, 1.0)], target_size=1.0),
        ]
        self.assertIsNone(_gmsh_flow_aligned_curve_counts(controls, fallback_size=1.0))


class TestGmshFlowAlignedPreflight(unittest.TestCase):
    def test_preflight_accepts_valid_quad_region(self):
        controls = [
            QuadEdgeControl(region_id=4, edge_id=1, points_xy=[(0.0, 0.0), (100.0, 0.0)], target_size=10.0),
            QuadEdgeControl(region_id=4, edge_id=2, points_xy=[(100.0, 0.0), (100.0, 40.0)], target_size=10.0),
            QuadEdgeControl(region_id=4, edge_id=3, points_xy=[(100.0, 40.0), (0.0, 40.0)], target_size=10.0),
            QuadEdgeControl(region_id=4, edge_id=4, points_xy=[(0.0, 40.0), (0.0, 0.0)], target_size=10.0),
        ]

        diag = _gmsh_flow_align_region_preflight(
            region_id=4,
            cell_type="quadrilateral",
            curve_tags=[11, 12, 13, 14],
            edge_controls=controls,
            fallback_size=10.0,
        )

        self.assertTrue(diag["eligible"])
        self.assertFalse(diag["fallback"])
        self.assertEqual(diag["transfinite_counts"], [11, 5, 11, 5])
        self.assertEqual(diag["reasons"], [])

    def test_preflight_rejects_missing_four_curve_surface(self):
        controls = [
            QuadEdgeControl(region_id=5, edge_id=1, points_xy=[(0.0, 0.0), (10.0, 0.0)], target_size=2.0),
            QuadEdgeControl(region_id=5, edge_id=2, points_xy=[(10.0, 0.0), (10.0, 8.0)], target_size=2.0),
            QuadEdgeControl(region_id=5, edge_id=3, points_xy=[(10.0, 8.0), (0.0, 8.0)], target_size=2.0),
            QuadEdgeControl(region_id=5, edge_id=4, points_xy=[(0.0, 8.0), (0.0, 0.0)], target_size=2.0),
        ]

        diag = _gmsh_flow_align_region_preflight(
            region_id=5,
            cell_type="quadrilateral",
            curve_tags=[1, 2, 3],
            edge_controls=controls,
            fallback_size=2.0,
        )

        self.assertFalse(diag["eligible"])
        self.assertTrue(diag["fallback"])
        self.assertIn("surface-must-have-4-curves-got-3", diag["reasons"])


class TestGmshTransfiniteInterfaceHarmonization(unittest.TestCase):
    def test_shared_interface_uses_denser_chain_and_densifies_opposite_subset(self):
        # Region 4 (right block) intentionally starts with a sparse shared edge,
        # while region 5 (left block) provides a denser representation.
        region_quad_setups = {
            4: (
                [(10.0, 0.0), (20.0, 0.0), (20.0, 10.0), (10.0, 10.0)],
                [
                    QuadEdgeControl(4, 1, [(10.0, 0.0), (20.0, 0.0)], 5.0),
                    QuadEdgeControl(4, 2, [(20.0, 0.0), (20.0, 10.0)], 5.0),
                    QuadEdgeControl(4, 3, [(20.0, 10.0), (10.0, 10.0)], 5.0),
                    QuadEdgeControl(4, 4, [(10.0, 10.0), (10.0, 0.0)], 5.0),
                ],
            ),
            5: (
                [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
                [
                    QuadEdgeControl(5, 1, [(0.0, 0.0), (10.0, 0.0)], 5.0),
                    QuadEdgeControl(5, 2, [(10.0, 0.0), (10.0, 2.0), (10.0, 4.0), (10.0, 6.0), (10.0, 8.0), (10.0, 10.0)], 1.0),
                    QuadEdgeControl(5, 3, [(10.0, 10.0), (0.0, 10.0)], 5.0),
                    QuadEdgeControl(5, 4, [(0.0, 10.0), (0.0, 0.0)], 5.0),
                ],
            ),
        }
        region_cell_types = {4: "quadrilateral", 5: "quadrilateral"}
        all_region_rings = {
            1: [(10.0, 5.0), (12.0, 5.0), (12.0, 7.0), (10.0, 7.0)],
            4: [(10.0, 0.0), (20.0, 0.0), (20.0, 10.0), (10.0, 10.0)],
            5: [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
        }

        min_nodes, stats = _harmonize_transfinite_shared_quad_interfaces(
            region_quad_setups=region_quad_setups,
            region_cell_types=region_cell_types,
            gmsh_quad_full_region_flow_align=True,
            all_region_rings=all_region_rings,
            opposite_subset_start_frac=0.30,
            opposite_subset_end_frac=0.70,
            opposite_subset_density_scale=0.50,
        )

        edge4_r4 = next(edge for edge in region_quad_setups[4][1] if edge.edge_id == 4)
        edge2_r5 = next(edge for edge in region_quad_setups[5][1] if edge.edge_id == 2)
        self.assertEqual(len(edge4_r4.points_xy), len(edge2_r5.points_xy))
        self.assertEqual(edge4_r4.points_xy[0], (10.0, 10.0))
        self.assertEqual(edge4_r4.points_xy[-1], (10.0, 0.0))

        self.assertGreaterEqual(int(min_nodes.get((4, 4), 0)), len(edge4_r4.points_xy))
        self.assertGreaterEqual(int(min_nodes.get((5, 2), 0)), len(edge2_r5.points_xy))
        self.assertGreater(int(stats.get("junction_points_inserted", 0)), 0)
        self.assertTrue(any(abs(float(p[1]) - 5.0) <= 1.0e-6 for p in edge2_r5.points_xy))

        opp_r4 = next(edge for edge in region_quad_setups[4][1] if edge.edge_id == 2)
        opp_r5 = next(edge for edge in region_quad_setups[5][1] if edge.edge_id == 4)
        self.assertGreater(len(opp_r4.points_xy), 2)
        self.assertGreater(len(opp_r5.points_xy), 2)
        self.assertGreaterEqual(int(stats.get("opposite_subset_requests", 0)), 1)

    def test_subset_containment_groups_t_junction_and_densifies_container_subset(self):
        region_quad_setups = {
            4: (
                [(10.0, 0.0), (20.0, 0.0), (20.0, 10.0), (10.0, 10.0)],
                [
                    QuadEdgeControl(4, 1, [(10.0, 0.0), (20.0, 0.0)], 5.0),
                    QuadEdgeControl(4, 2, [(20.0, 0.0), (20.0, 10.0)], 5.0),
                    QuadEdgeControl(4, 3, [(20.0, 10.0), (10.0, 10.0)], 5.0),
                    QuadEdgeControl(4, 4, [(10.0, 10.0), (10.0, 0.0)], 5.0),
                ],
            ),
            5: (
                [(0.0, 4.0), (10.0, 4.0), (10.0, 6.0), (0.0, 6.0)],
                [
                    QuadEdgeControl(5, 1, [(0.0, 4.0), (10.0, 4.0)], 1.0),
                    QuadEdgeControl(5, 2, [(10.0, 4.0), (10.0, 5.0), (10.0, 6.0)], 0.5),
                    QuadEdgeControl(5, 3, [(10.0, 6.0), (0.0, 6.0)], 1.0),
                    QuadEdgeControl(5, 4, [(0.0, 6.0), (0.0, 4.0)], 1.0),
                ],
            ),
        }
        region_cell_types = {4: "quadrilateral", 5: "quadrilateral"}
        debug_capture = {}

        min_nodes, stats = _harmonize_transfinite_shared_quad_interfaces(
            region_quad_setups=region_quad_setups,
            region_cell_types=region_cell_types,
            gmsh_quad_full_region_flow_align=True,
            all_region_rings=None,
            subset_containment_enable=True,
            subset_containment_high_overlap=0.95,
            subset_containment_min_overlap=0.02,
            subset_containment_max_length_ratio=0.35,
            debug_capture=debug_capture,
        )

        self.assertGreaterEqual(int(stats.get("shared_groups", 0)), 1)
        self.assertGreaterEqual(int(stats.get("subset_containment_requests", 0)), 1)

        pair_45 = next(
            (
                p
                for p in debug_capture.get("pair_debug", [])
                if {int(p.get("region_i", -1)), int(p.get("region_j", -1))} == {4, 5}
                and bool(p.get("pass_subset_containment", False))
                and bool(p.get("grouped", False))
            ),
            None,
        )
        self.assertIsNotNone(pair_45)
        self.assertEqual(str(pair_45.get("grouped_by", "")), "overlap_subset_containment")

        edge4_r4 = next(edge for edge in region_quad_setups[4][1] if edge.edge_id == 4)
        self.assertGreater(len(edge4_r4.points_xy), 2)
        self.assertEqual(edge4_r4.points_xy[0], (10.0, 10.0))
        self.assertEqual(edge4_r4.points_xy[-1], (10.0, 0.0))
        self.assertEqual(int(min_nodes.get((4, 4), 0)), 0)

    def test_curve_counts_respect_min_nodes_floor(self):
        controls = [
            QuadEdgeControl(region_id=1, edge_id=1, points_xy=[(0.0, 0.0), (10.0, 0.0)], target_size=5.0),
            QuadEdgeControl(region_id=1, edge_id=2, points_xy=[(10.0, 0.0), (10.0, 8.0)], target_size=4.0),
            QuadEdgeControl(region_id=1, edge_id=3, points_xy=[(10.0, 8.0), (0.0, 8.0)], target_size=5.0),
            QuadEdgeControl(region_id=1, edge_id=4, points_xy=[(0.0, 8.0), (0.0, 0.0)], target_size=4.0),
        ]

        counts = _gmsh_flow_aligned_curve_counts(
            controls,
            fallback_size=5.0,
            min_nodes=[15, 0, 0, 0],
        )

        self.assertIsNotNone(counts)
        self.assertEqual(counts[0], 15)
        self.assertEqual(counts[2], 15)

    def test_singleton_transfinite_edge_splits_against_non_transfinite_neighbor_rings(self):
        region_quad_setups = {
            4: (
                [(10.0, 0.0), (20.0, 0.0), (20.0, 10.0), (10.0, 10.0)],
                [
                    QuadEdgeControl(4, 1, [(10.0, 0.0), (20.0, 0.0)], 5.0),
                    QuadEdgeControl(4, 2, [(20.0, 0.0), (20.0, 10.0)], 5.0),
                    QuadEdgeControl(4, 3, [(20.0, 10.0), (10.0, 10.0)], 5.0),
                    QuadEdgeControl(4, 4, [(10.0, 10.0), (10.0, 0.0)], 5.0),
                ],
            ),
        }
        region_cell_types = {1: "triangular", 4: "quadrilateral"}
        all_region_rings = {
            1: [(8.0, 2.0), (10.0, 2.0), (10.0, 8.0), (8.0, 8.0)],
            4: [(10.0, 0.0), (20.0, 0.0), (20.0, 10.0), (10.0, 10.0)],
        }

        min_nodes, stats = _harmonize_transfinite_shared_quad_interfaces(
            region_quad_setups=region_quad_setups,
            region_cell_types=region_cell_types,
            gmsh_quad_full_region_flow_align=True,
            all_region_rings=all_region_rings,
        )

        edge4_r4 = next(edge for edge in region_quad_setups[4][1] if edge.edge_id == 4)
        self.assertGreater(len(edge4_r4.points_xy), 2)
        self.assertTrue(any(abs(float(p[1]) - 2.0) <= 1.0e-6 for p in edge4_r4.points_xy))
        self.assertTrue(any(abs(float(p[1]) - 8.0) <= 1.0e-6 for p in edge4_r4.points_xy))
        self.assertEqual(int(min_nodes.get((4, 4), 0)), 0)
        self.assertGreaterEqual(int(stats.get("singleton_external_junction_edges", 0)), 1)

    def test_singleton_transfinite_edge_splits_with_offset_non_transfinite_neighbor(self):
        region_quad_setups = {
            4: (
                [(10.0, 0.0), (20.0, 0.0), (20.0, 10.0), (10.0, 10.0)],
                [
                    QuadEdgeControl(4, 1, [(10.0, 0.0), (20.0, 0.0)], 20.0),
                    QuadEdgeControl(4, 2, [(20.0, 0.0), (20.0, 10.0)], 20.0),
                    QuadEdgeControl(4, 3, [(20.0, 10.0), (10.0, 10.0)], 20.0),
                    QuadEdgeControl(4, 4, [(10.0, 10.0), (10.0, 0.0)], 20.0),
                ],
            ),
        }
        region_cell_types = {1: "triangular", 4: "quadrilateral"}
        # Neighbor ring is slightly offset from the transfinite edge (x=10.8).
        # Harmonization should still project/split on the interface chain.
        all_region_rings = {
            1: [(8.0, 2.0), (10.8, 2.0), (10.8, 8.0), (8.0, 8.0)],
            4: [(10.0, 0.0), (20.0, 0.0), (20.0, 10.0), (10.0, 10.0)],
        }

        _min_nodes, stats = _harmonize_transfinite_shared_quad_interfaces(
            region_quad_setups=region_quad_setups,
            region_cell_types=region_cell_types,
            gmsh_quad_full_region_flow_align=True,
            all_region_rings=all_region_rings,
        )

        edge4_r4 = next(edge for edge in region_quad_setups[4][1] if edge.edge_id == 4)
        self.assertGreater(len(edge4_r4.points_xy), 2)
        self.assertTrue(any(abs(float(p[1]) - 2.0) <= 1.0e-6 for p in edge4_r4.points_xy))
        self.assertTrue(any(abs(float(p[1]) - 8.0) <= 1.0e-6 for p in edge4_r4.points_xy))
        self.assertGreaterEqual(int(stats.get("singleton_external_junction_edges", 0)), 1)


class TestGmshInterfaceCoincidenceReport(unittest.TestCase):
    def test_report_detects_aligned_shared_interface_with_split_parameterization(self):
        regions = [
            ConceptualRegion(
                region_id=4,
                ring_xy=[(10.0, 0.0), (20.0, 0.0), (20.0, 10.0), (10.0, 10.0)],
                default_size=2.0,
                default_cell_type="quadrilateral",
            ),
            ConceptualRegion(
                region_id=5,
                ring_xy=[(0.0, 0.0), (10.0, 0.0), (10.0, 5.0), (10.0, 10.0), (0.0, 10.0)],
                default_size=2.0,
                default_cell_type="triangular",
            ),
        ]
        model = ConceptualModel(nodes=[], arcs=[], regions=regions, constraints=[], quad_edges=[])

        report = _gmsh_interface_coincidence_report(model)
        pair = next((r for r in report if {int(r["region_a"]), int(r["region_b"])} == {4, 5}), None)

        self.assertIsNotNone(pair)
        self.assertGreater(float(pair["overlap_ab"]), 0.20)
        self.assertGreater(float(pair["overlap_ba"]), 0.20)
        self.assertLessEqual(float(pair["endpoint_delta_max"]), 2.0 * float(pair["near_tol"]))

    def test_report_flags_partial_overlap_as_high_overlap_delta(self):
        regions = [
            ConceptualRegion(
                region_id=4,
                ring_xy=[(10.0, 0.0), (20.0, 0.0), (20.0, 10.0), (10.0, 10.0)],
                default_size=2.0,
                default_cell_type="quadrilateral",
            ),
            ConceptualRegion(
                region_id=3,
                ring_xy=[(0.0, 0.0), (10.0, 0.0), (10.0, 4.0), (0.0, 4.0)],
                default_size=2.0,
                default_cell_type="triangular",
            ),
        ]
        model = ConceptualModel(nodes=[], arcs=[], regions=regions, constraints=[], quad_edges=[])

        report = _gmsh_interface_coincidence_report(model)
        pair = next((r for r in report if {int(r["region_a"]), int(r["region_b"])} == {3, 4}), None)

        self.assertIsNotNone(pair)
        self.assertGreater(max(float(pair["overlap_ab"]), float(pair["overlap_ba"])), 0.05)
        self.assertLess(max(float(pair["overlap_ab"]), float(pair["overlap_ba"])), 0.90)


if __name__ == "__main__":
    unittest.main()
