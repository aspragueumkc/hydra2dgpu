import unittest
import numpy as np

from swe2d_backend import SWE2DBackend, swe2d_available
from swe2d_coupling import SWE2DCouplingController, pack_coupling_soa
from swe2d_drainage_network import SWE2DUrbanDrainageModule
from swe2d_extensions import (
    DrainageLink,
    DrainageNode,
    HydraulicStructure,
    HydraulicStructureConfig,
    InletExchange,
    PipeNetworkConfig,
    StructureType,
)
from swe2d_structures import SWE2DStructureModule


class TestSWE2DDrainageStructures(unittest.TestCase):
    def _build_simple_network(self):
        nodes = [
            DrainageNode(node_id="N0", x=0.0, y=0.0, invert_elev=0.0, max_depth=3.0, metadata={"surface_area_m2": 10.0}),
            DrainageNode(node_id="N1", x=10.0, y=0.0, invert_elev=0.0, max_depth=3.0),
        ]
        links = [
            DrainageLink(
                link_id="L0",
                from_node_id="N0",
                to_node_id="N1",
                length_m=10.0,
                roughness_n=0.013,
                diameter_m=1.0,
            )
        ]
        inlets = [
            InletExchange(
                inlet_id="I0",
                cell_id=0,
                node_id="N0",
                crest_elev=0.5,
                width_m=1.0,
                coefficient=0.62,
                max_capture_cms=1.0,
            )
        ]
        cfg = PipeNetworkConfig(enabled=True, nodes=nodes, links=links, inlets=inlets)
        mod = SWE2DUrbanDrainageModule(cfg)
        mod.initialize()
        return mod

    def test_network_step_moves_head_down_gradient(self):
        mod = self._build_simple_network()
        mod.state.node_depth_m["N0"] = 1.5
        mod.state.node_depth_m["N1"] = 0.5

        diag = mod.solve_network_step(1.0)

        self.assertIn("max_link_flow_cms", diag)
        self.assertGreater(diag["max_link_flow_cms"], 0.0)
        self.assertLess(mod.state.node_depth_m["N0"], 1.5)
        self.assertGreater(mod.state.node_depth_m["N1"], 0.5)

    def test_surface_exchange_capture_then_surcharge(self):
        mod = self._build_simple_network()

        mod.state.node_depth_m["N0"] = 0.1
        sinks, sources = mod.exchange_step(1.0, [1.8])
        self.assertGreater(sinks[0], 0.0)
        self.assertEqual(sources[0], 0.0)

        mod.state.node_depth_m["N0"] = 2.0
        sinks, sources = mod.exchange_step(1.0, [0.6])
        self.assertGreater(sources[0], 0.0)

    def test_surface_exchange_depth_rate_matches_node_storage_change(self):
        mod = self._build_simple_network()
        mod.state.node_depth_m["N0"] = 0.1
        node_area = 10.0
        cell_area = [5.0]
        depth0 = mod.state.node_depth_m["N0"]

        src_rate = mod.surface_exchange_source_rate(1.0, [1.8], cell_area)
        removed_surface_volume = -src_rate[0] * cell_area[0] * 1.0
        added_node_volume = (mod.state.node_depth_m["N0"] - depth0) * node_area

        self.assertGreater(removed_surface_volume, 0.0)
        self.assertAlmostEqual(removed_surface_volume, added_node_volume, places=10)

    def test_structure_module_culvert_directionality(self):
        structure = HydraulicStructure(
            structure_id="C0",
            structure_type=StructureType.CULVERT,
            upstream_cell=0,
            downstream_cell=1,
            crest_elev=0.0,
            metadata={
                "diameter_m": 1.0,
                "length_m": 12.0,
                "roughness_n": 0.013,
                "cd": 0.75,
            },
        )
        cfg = HydraulicStructureConfig(enabled=True, structures=[structure])
        mod = SWE2DStructureModule(cfg)

        flux_forward = mod.compute_flux_adjustments(1.0, [2.0, 1.0])
        self.assertGreater(flux_forward["total_structure_flow_cms"], 0.0)

        flux_reverse = mod.compute_flux_adjustments(1.0, [1.0, 2.0])
        self.assertGreater(flux_reverse["total_structure_flow_cms"], 0.0)

    def test_structure_source_rates_are_conservative(self):
        structure = HydraulicStructure(
            structure_id="C0",
            structure_type=StructureType.CULVERT,
            upstream_cell=0,
            downstream_cell=1,
            crest_elev=0.0,
            metadata={
                "diameter_m": 1.0,
                "length_m": 12.0,
                "roughness_n": 0.013,
                "cd": 0.75,
            },
        )
        cfg = HydraulicStructureConfig(enabled=True, structures=[structure])
        mod = SWE2DStructureModule(cfg)

        q_cells = mod.compute_cell_source_terms_cms(1.0, [2.0, 1.0])
        rates = mod.compute_cell_source_rate(1.0, [2.0, 1.0], [4.0, 4.0])

        self.assertAlmostEqual(sum(q_cells), 0.0, places=12)
        self.assertAlmostEqual((rates[0] + rates[1]) * 4.0, 0.0, places=12)

    def test_coupling_controller_combines_modules(self):
        drainage = self._build_simple_network()
        drainage.state.node_depth_m["N0"] = 0.1
        structure = HydraulicStructure(
            structure_id="C0",
            structure_type=StructureType.CULVERT,
            upstream_cell=0,
            downstream_cell=1,
            crest_elev=0.0,
            metadata={
                "diameter_m": 1.0,
                "length_m": 12.0,
                "roughness_n": 0.013,
                "cd": 0.75,
            },
        )
        structures = SWE2DStructureModule(HydraulicStructureConfig(enabled=True, structures=[structure]))
        controller = SWE2DCouplingController(
            cell_area_m2=[5.0, 5.0],
            cell_bed_m=[0.0, 0.0],
            drainage=drainage,
            structures=structures,
        )

        src = controller.compute_source_rates(
            t_s=0.0,
            dt_s=1.0,
            h=np.asarray([1.8, 1.0], dtype=np.float64),
            hu=np.asarray([0.0, 0.0], dtype=np.float64),
            hv=np.asarray([0.0, 0.0], dtype=np.float64),
        )

        self.assertEqual(src.size, 2)
        self.assertIn("drainage", controller.last_diag.component_sums_mps)
        self.assertIn("structures", controller.last_diag.component_sums_mps)
        self.assertGreater(controller.last_diag.drainage_max_link_flow_cms, 0.0)
        self.assertGreater(controller.last_diag.structure_total_flow_cms, 0.0)

    def test_pack_coupling_soa_shapes_and_indices(self):
        drainage = self._build_simple_network().cfg
        structure = HydraulicStructure(
            structure_id="C0",
            structure_type=StructureType.CULVERT,
            upstream_cell=0,
            downstream_cell=1,
            crest_elev=0.0,
            metadata={"diameter_m": 1.0, "cd": 0.75},
        )
        structures_cfg = HydraulicStructureConfig(enabled=True, structures=[structure])

        soa = pack_coupling_soa(
            n_cells=2,
            pipe_network=drainage,
            hydraulic_structures=structures_cfg,
        )

        self.assertEqual(soa.n_cells, 2)
        self.assertIsNotNone(soa.drainage)
        self.assertIsNotNone(soa.structures)
        self.assertEqual(int(soa.drainage.node_x.size), 2)
        self.assertEqual(int(soa.drainage.link_from.size), 1)
        self.assertEqual(int(soa.drainage.inlet_cell.size), 1)
        self.assertEqual(int(soa.drainage.link_from[0]), 0)
        self.assertEqual(int(soa.drainage.link_to[0]), 1)
        self.assertEqual(int(soa.structures.structure_type.size), 1)
        self.assertEqual(int(soa.structures.upstream_cell[0]), 0)
        self.assertEqual(int(soa.structures.downstream_cell[0]), 1)

    def test_coupling_controller_accepts_explicit_cpu_loop(self):
        drainage = self._build_simple_network()
        structure = HydraulicStructure(
            structure_id="C0",
            structure_type=StructureType.CULVERT,
            upstream_cell=0,
            downstream_cell=1,
            crest_elev=0.0,
            metadata={"diameter_m": 1.0, "length_m": 12.0, "roughness_n": 0.013, "cd": 0.75},
        )
        structures = SWE2DStructureModule(HydraulicStructureConfig(enabled=True, structures=[structure]))
        controller = SWE2DCouplingController(
            cell_area_m2=[5.0, 5.0],
            cell_bed_m=[0.0, 0.0],
            drainage=drainage,
            structures=structures,
            coupling_loop="cpu",
        )
        src = controller.compute_source_rates(
            t_s=0.0,
            dt_s=1.0,
            h=np.asarray([1.8, 1.0], dtype=np.float64),
            hu=np.asarray([0.0, 0.0], dtype=np.float64),
            hv=np.asarray([0.0, 0.0], dtype=np.float64),
        )
        self.assertEqual(src.size, 2)

    def test_coupling_controller_rejects_invalid_loop_mode(self):
        with self.assertRaises(ValueError):
            SWE2DCouplingController(
                cell_area_m2=[1.0],
                cell_bed_m=[0.0],
                coupling_loop="invalid",
            )

    @unittest.skipUnless(swe2d_available(), "native SWE2D backend not available")
    def test_backend_run_with_coupling_controller(self):
        backend = SWE2DBackend(use_gpu=False)
        node_x = np.asarray([0.0, 1.0, 1.0, 0.0], dtype=np.float64)
        node_y = np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
        node_z = np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        cell_nodes = np.asarray([0, 1, 2, 0, 2, 3], dtype=np.int32)
        backend.build_mesh(node_x, node_y, node_z, cell_nodes)

        drainage = self._build_simple_network()
        drainage.state.node_depth_m["N0"] = 0.1
        structure = HydraulicStructure(
            structure_id="C0",
            structure_type=StructureType.CULVERT,
            upstream_cell=0,
            downstream_cell=1,
            crest_elev=0.0,
            metadata={
                "diameter_m": 1.0,
                "length_m": 12.0,
                "roughness_n": 0.013,
                "cd": 0.75,
            },
        )
        structures = SWE2DStructureModule(HydraulicStructureConfig(enabled=True, structures=[structure]))
        controller = SWE2DCouplingController(
            cell_area_m2=backend.cell_areas(),
            cell_bed_m=np.zeros(backend.n_cells, dtype=np.float64),
            drainage=drainage,
            structures=structures,
        )

        backend.initialize(
            h0=np.asarray([1.8, 1.0], dtype=np.float64),
            hu0=np.asarray([0.0, 0.0], dtype=np.float64),
            hv0=np.asarray([0.0, 0.0], dtype=np.float64),
            dt_fixed=0.05,
            dt_max=0.05,
        )
        backend.run(
            t_end=0.05,
            dt_request=0.05,
            source_rate_callback=controller.source_rate_callback(),
        )
        h, _, _ = backend.get_state()
        self.assertTrue(np.all(np.isfinite(h)))
        self.assertGreater(controller.last_diag.structure_total_flow_cms, 0.0)
        self.assertGreater(controller.last_diag.drainage_max_link_flow_cms, 0.0)
        backend.destroy()

    @unittest.skipUnless(swe2d_available(), "native SWE2D backend not available")
    def test_backend_cell_area_cache_and_source_callback(self):
        backend = SWE2DBackend(use_gpu=False)
        node_x = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        node_y = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        node_z = np.asarray([0.0, 0.0, 0.0], dtype=np.float64)
        cell_nodes = np.asarray([0, 1, 2], dtype=np.int32)
        backend.build_mesh(node_x, node_y, node_z, cell_nodes)
        areas = backend.cell_areas()
        self.assertEqual(areas.size, 1)
        self.assertAlmostEqual(float(areas[0]), 0.5, places=12)

        backend.initialize(
            h0=np.asarray([0.0], dtype=np.float64),
            hu0=np.asarray([0.0], dtype=np.float64),
            hv0=np.asarray([0.0], dtype=np.float64),
            dt_fixed=0.1,
            dt_max=0.1,
        )
        backend.run(
            t_end=0.1,
            dt_request=0.1,
            source_rate_callback=lambda t, dt, h, hu, hv: np.asarray([0.2], dtype=np.float64),
        )
        h, _, _ = backend.get_state()
        self.assertAlmostEqual(float(h[0]), 0.02, places=12)
        backend.destroy()


if __name__ == "__main__":
    unittest.main()
