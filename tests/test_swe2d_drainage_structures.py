import unittest
import numpy as np

from swe2d_backend import SWE2DBackend, swe2d_available, swe2d_gpu_available
from swe2d_coupling import SWE2DCouplingController, pack_coupling_soa
from swe2d_drainage_network import SWE2DUrbanDrainageModule
from swe2d_extensions import (
    DrainageLink,
    DrainageNode,
    DrainageSolverMode,
    HydraulicStructure,
    HydraulicStructureConfig,
    InletExchange,
    OutfallExchange,
    PipeEndExchange,
    PipeNetworkConfig,
    StructureType,
)
from swe2d_structures import SWE2DStructureModule

try:
    import swe2d_workbench_qt as _wbqt
    _HAVE_WORKBENCH = True
except Exception:
    _wbqt = None
    _HAVE_WORKBENCH = False


class TestSWE2DDrainageStructures(unittest.TestCase):
    def _build_simple_network(self):
        nodes = [
            DrainageNode(node_id="N0", x=0.0, y=0.0, invert_elev=0.0, max_depth=3.0, metadata={"surface_area": 10.0}),
            DrainageNode(node_id="N1", x=10.0, y=0.0, invert_elev=0.0, max_depth=3.0),
        ]
        links = [
            DrainageLink(
                link_id="L0",
                from_node_id="N0",
                to_node_id="N1",
                length=10.0,
                roughness_n=0.013,
                diameter=1.0,
            )
        ]
        inlets = [
            InletExchange(
                inlet_id="I0",
                cell_id=0,
                node_id="N0",
                crest_elev=0.5,
                width=1.0,
                coefficient=0.62,
                max_capture=1.0,
            )
        ]
        cfg = PipeNetworkConfig(enabled=True, nodes=nodes, links=links, inlets=inlets)
        mod = SWE2DUrbanDrainageModule(cfg)
        mod.initialize()
        return mod

    def _build_outfall_network(self, zero_storage=False):
        nodes = [
            DrainageNode(
                node_id="O0",
                x=0.0,
                y=0.0,
                invert_elev=0.0,
                max_depth=3.0,
                node_type="outfall",
                metadata={"surface_area": 8.0},
            ),
        ]
        outfalls = [
            OutfallExchange(
                outfall_id="O0",
                cell_id=0,
                node_id="O0",
                invert_elev=0.0,
                diameter=1.0,
                coefficient=0.82,
                max_flow=None,
                zero_storage=bool(zero_storage),
            )
        ]
        cfg = PipeNetworkConfig(enabled=True, nodes=nodes, links=[], inlets=[], outfalls=outfalls)
        mod = SWE2DUrbanDrainageModule(cfg)
        mod.initialize()
        return mod

    def _build_pipe_end_network(self):
        nodes = [
            DrainageNode(
                node_id="P0",
                x=0.0,
                y=0.0,
                invert_elev=0.0,
                max_depth=3.0,
                node_type="pipe_end",
                metadata={"surface_area": 8.0},
            )
        ]
        pipe_ends = [
            PipeEndExchange(
                pipe_end_id="PE0",
                cell_id=0,
                node_id="P0",
                invert_elev=0.0,
                diameter=1.0,
                inlet_loss_k=0.7,
                outlet_loss_k=1.2,
            )
        ]
        cfg = PipeNetworkConfig(enabled=True, nodes=nodes, links=[], inlets=[], outfalls=[], pipe_ends=pipe_ends)
        mod = SWE2DUrbanDrainageModule(cfg)
        mod.initialize()
        return mod

    def test_network_step_moves_head_down_gradient(self):
        mod = self._build_simple_network()
        mod.state.node_depth["N0"] = 1.5
        mod.state.node_depth["N1"] = 0.5

        diag = mod.solve_network_step(1.0)

        self.assertIn("max_link_flow", diag)
        self.assertGreater(diag["max_link_flow"], 0.0)
        self.assertLess(mod.state.node_depth["N0"], 1.5)
        self.assertGreater(mod.state.node_depth["N1"], 0.5)

    def test_dynamic_mode_uses_simplified_link_for_lateral_simple(self):
        nodes = [
            DrainageNode(node_id="N0", x=0.0, y=0.0, invert_elev=0.0, max_depth=3.0, metadata={"surface_area": 10.0}),
            DrainageNode(node_id="N1", x=10.0, y=0.0, invert_elev=0.0, max_depth=3.0, metadata={"surface_area": 10.0}),
        ]
        link = DrainageLink(
            link_id="L0",
            from_node_id="N0",
            to_node_id="N1",
            link_type="lateral_simple",
            length=10.0,
            roughness_n=0.013,
            diameter=1.0,
        )
        cfg = PipeNetworkConfig(
            enabled=True,
            nodes=nodes,
            links=[link],
            inlets=[],
            solver_mode=DrainageSolverMode.DYNAMIC,
        )
        mod = SWE2DUrbanDrainageModule(cfg)
        mod.initialize()
        mod.state.node_depth["N0"] = 1.5
        mod.state.node_depth["N1"] = 0.5

        expected_q = mod._estimate_link_flow(link)
        mod.solve_network_step(1.0)

        self.assertAlmostEqual(mod.state.link_flow["L0"], expected_q, places=12)

    def test_adaptive_substeps_increase_for_stiff_network(self):
        mod = self._build_simple_network()
        mod.cfg.solver_mode = DrainageSolverMode.DYNAMIC
        mod.cfg.coupling_substeps = 1
        mod.cfg.max_coupling_substeps = 64
        mod._node_area["N0"] = 1.0
        mod._node_area["N1"] = 1.0
        mod.state.node_depth["N0"] = 2.0
        mod.state.node_depth["N1"] = 0.0
        mod.state.link_flow["L0"] = 5.0

        diag = mod.solve_network_step(1.0)

        self.assertGreater(diag["substeps_used"], 1)

    def test_dynamic_deadband_reduces_small_gradient_flow_response(self):
        nodes = [
            DrainageNode(node_id="N0", x=0.0, y=0.0, invert_elev=0.0, max_depth=3.0, metadata={"surface_area": 10.0}),
            DrainageNode(node_id="N1", x=10.0, y=0.0, invert_elev=0.0, max_depth=3.0, metadata={"surface_area": 10.0}),
        ]
        link = DrainageLink(
            link_id="L0",
            from_node_id="N0",
            to_node_id="N1",
            length=10.0,
            roughness_n=0.013,
            diameter=1.0,
        )

        cfg_no_deadband = PipeNetworkConfig(
            enabled=True,
            nodes=nodes,
            links=[link],
            inlets=[],
            solver_mode=DrainageSolverMode.DYNAMIC,
        )
        mod_no_deadband = SWE2DUrbanDrainageModule(cfg_no_deadband)
        mod_no_deadband.initialize()
        mod_no_deadband.state.node_depth["N0"] = 1.00
        mod_no_deadband.state.node_depth["N1"] = 0.95
        mod_no_deadband.state.link_flow["L0"] = 2.0

        cfg_deadband = PipeNetworkConfig(
            enabled=True,
            nodes=nodes,
            links=[link],
            inlets=[],
            solver_mode=DrainageSolverMode.DYNAMIC,
        )
        cfg_deadband.head_deadband_m = 0.1
        cfg_deadband.dynamic_flow_relaxation = 0.5
        mod_deadband = SWE2DUrbanDrainageModule(cfg_deadband)
        mod_deadband.initialize()
        mod_deadband.state.node_depth["N0"] = 1.00
        mod_deadband.state.node_depth["N1"] = 0.95
        mod_deadband.state.link_flow["L0"] = 2.0

        q_no_deadband = mod_no_deadband._dynamic_link_flow_update(link, 1.0)
        q_deadband = mod_deadband._dynamic_link_flow_update(link, 1.0)

        self.assertLessEqual(abs(q_deadband), abs(q_no_deadband))

    def test_node_depth_cap_matches_rim_derived_max_depth(self):
        # This mirrors the parser's auto-assignment intent: max_depth = rim_elev - invert_elev.
        invert = 0.2
        rim = 0.9
        derived_max_depth = rim - invert
        nodes = [
            DrainageNode(node_id="UP", x=0.0, y=0.0, invert_elev=0.0, max_depth=5.0, metadata={"surface_area": 10.0}),
            DrainageNode(
                node_id="DN",
                x=10.0,
                y=0.0,
                invert_elev=invert,
                max_depth=derived_max_depth,
                rim_elev=rim,
                metadata={"surface_area": 1.0},
            ),
        ]
        links = [
            DrainageLink(
                link_id="L0",
                from_node_id="UP",
                to_node_id="DN",
                length=10.0,
                roughness_n=0.013,
                diameter=1.0,
            )
        ]
        cfg = PipeNetworkConfig(
            enabled=True,
            nodes=nodes,
            links=links,
            inlets=[],
            solver_mode=DrainageSolverMode.DYNAMIC,
        )
        mod = SWE2DUrbanDrainageModule(cfg)
        mod.initialize()
        mod.state.node_depth["UP"] = 3.0
        mod.state.node_depth["DN"] = 0.0

        mod.solve_network_step(1.0)

        self.assertLessEqual(mod.state.node_depth["DN"], derived_max_depth)
        self.assertAlmostEqual(mod.state.node_depth["DN"], derived_max_depth, places=12)

    def test_surface_exchange_capture_then_surcharge(self):
        mod = self._build_simple_network()

        mod.state.node_depth["N0"] = 0.1
        sinks, sources = mod.exchange_step(1.0, [1.8])
        self.assertGreater(sinks[0], 0.0)
        self.assertEqual(sources[0], 0.0)

        mod.state.node_depth["N0"] = 2.0
        sinks, sources = mod.exchange_step(1.0, [0.6])
        self.assertGreater(sources[0], 0.0)

    def test_surface_exchange_depth_rate_matches_node_storage_change(self):
        mod = self._build_simple_network()
        mod.state.node_depth["N0"] = 0.1
        node_area = 10.0
        cell_area = [5.0]
        depth0 = mod.state.node_depth["N0"]

        src_rate = mod.surface_exchange_source_rate(1.0, [1.8], cell_area)
        removed_surface_volume = -src_rate[0] * cell_area[0] * 1.0
        added_node_volume = (mod.state.node_depth["N0"] - depth0) * node_area

        self.assertGreater(removed_surface_volume, 0.0)
        self.assertAlmostEqual(removed_surface_volume, added_node_volume, places=10)

    def test_surface_exchange_capture_is_limited_by_available_surface_volume(self):
        mod = self._build_simple_network()
        mod.state.node_depth["N0"] = 0.0

        src_rate = mod.surface_exchange_source_rate(
            1.0,
            [0.6],
            [1.0],
            cell_depth_m=[0.01],
        )

        removed_surface_volume = -src_rate[0] * 1.0 * 1.0
        self.assertAlmostEqual(removed_surface_volume, 0.01, places=12)

    def test_surface_exchange_capture_is_limited_by_remaining_node_storage(self):
        mod = self._build_simple_network()
        mod.state.node_depth["N0"] = 2.99

        src_rate = mod.surface_exchange_source_rate(
            1.0,
            [4.0],
            [10.0],
            cell_depth_m=[10.0],
        )

        removed_surface_volume = -src_rate[0] * 10.0 * 1.0
        self.assertAlmostEqual(removed_surface_volume, 0.1, places=12)
        self.assertAlmostEqual(mod.state.node_depth["N0"], 3.0, places=12)

    def test_structure_module_culvert_directionality(self):
        structure = HydraulicStructure(
            structure_id="C0",
            structure_type=StructureType.CULVERT,
            upstream_cell=0,
            downstream_cell=1,
            crest_elev=0.0,
            metadata={
                "diameter": 1.0,
                "length": 12.0,
                "roughness_n": 0.013,
                "cd": 0.75,
            },
        )
        cfg = HydraulicStructureConfig(enabled=True, structures=[structure])
        mod = SWE2DStructureModule(cfg)

        flux_forward = mod.compute_flux_adjustments(1.0, [2.0, 1.0])
        self.assertGreater(flux_forward["total_structure_flow"], 0.0)

        flux_reverse = mod.compute_flux_adjustments(1.0, [1.0, 2.0])
        self.assertGreater(flux_reverse["total_structure_flow"], 0.0)

    def test_structure_source_rates_are_conservative(self):
        structure = HydraulicStructure(
            structure_id="C0",
            structure_type=StructureType.CULVERT,
            upstream_cell=0,
            downstream_cell=1,
            crest_elev=0.0,
            metadata={
                "diameter": 1.0,
                "length": 12.0,
                "roughness_n": 0.013,
                "cd": 0.75,
            },
        )
        cfg = HydraulicStructureConfig(enabled=True, structures=[structure])
        mod = SWE2DStructureModule(cfg)

        q_cells = mod.compute_cell_source_terms(1.0, [2.0, 1.0])
        rates = mod.compute_cell_source_rate(1.0, [2.0, 1.0], [4.0, 4.0])

        self.assertAlmostEqual(sum(q_cells), 0.0, places=12)
        self.assertAlmostEqual((rates[0] + rates[1]) * 4.0, 0.0, places=12)

    def test_coupling_controller_combines_modules(self):
        drainage = self._build_simple_network()
        drainage.state.node_depth["N0"] = 0.1
        structure = HydraulicStructure(
            structure_id="C0",
            structure_type=StructureType.CULVERT,
            upstream_cell=0,
            downstream_cell=1,
            crest_elev=0.0,
            metadata={
                "diameter": 1.0,
                "length": 12.0,
                "roughness_n": 0.013,
                "cd": 0.75,
            },
        )
        structures = SWE2DStructureModule(HydraulicStructureConfig(enabled=True, structures=[structure]))
        controller = SWE2DCouplingController(
            cell_area=[5.0, 5.0],
            cell_bed=[0.0, 0.0],
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
        self.assertIn("drainage", controller.last_diag.component_sums)
        self.assertIn("structures", controller.last_diag.component_sums)
        self.assertGreater(controller.last_diag.drainage_max_link_flow, 0.0)
        self.assertGreater(controller.last_diag.structure_total_flow, 0.0)

    def test_pack_coupling_soa_shapes_and_indices(self):
        drainage = self._build_simple_network().cfg
        structure = HydraulicStructure(
            structure_id="C0",
            structure_type=StructureType.CULVERT,
            upstream_cell=0,
            downstream_cell=1,
            crest_elev=0.0,
            metadata={"diameter": 1.0, "cd": 0.75},
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

    def test_pack_coupling_soa_inlet_alias_none_falls_back_to_length_and_orifice(self):
        nodes = [
            DrainageNode(node_id="N0", x=0.0, y=0.0, invert_elev=0.0, max_depth=3.0),
            DrainageNode(node_id="N1", x=10.0, y=0.0, invert_elev=0.0, max_depth=3.0),
        ]
        links = [
            DrainageLink(
                link_id="L0",
                from_node_id="N0",
                to_node_id="N1",
                length=10.0,
                roughness_n=0.013,
                diameter=1.0,
            )
        ]
        # Mirror QGIS-built inlet objects where alias fields are present but None.
        inlet = InletExchange(
            inlet_id="I0",
            cell_id=0,
            node_id="N0",
            crest_elev=0.5,
            length=6.0,
            coeff_orifice=0.75,
            max_capture=1.0,
            width=None,
            coefficient=None,
        )
        cfg = PipeNetworkConfig(enabled=True, nodes=nodes, links=links, inlets=[inlet])

        soa = pack_coupling_soa(n_cells=2, pipe_network=cfg, hydraulic_structures=None)

        self.assertIsNotNone(soa.drainage)
        self.assertEqual(float(soa.drainage.inlet_width[0]), 6.0)
        self.assertAlmostEqual(float(soa.drainage.inlet_coefficient[0]), 0.75, places=12)

    def test_coupling_controller_accepts_explicit_cpu_loop(self):
        drainage = self._build_simple_network()
        structure = HydraulicStructure(
            structure_id="C0",
            structure_type=StructureType.CULVERT,
            upstream_cell=0,
            downstream_cell=1,
            crest_elev=0.0,
            metadata={"diameter": 1.0, "length": 12.0, "roughness_n": 0.013, "cd": 0.75},
        )
        structures = SWE2DStructureModule(HydraulicStructureConfig(enabled=True, structures=[structure]))
        controller = SWE2DCouplingController(
            cell_area=[5.0, 5.0],
            cell_bed=[0.0, 0.0],
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
                cell_area=[1.0],
                cell_bed=[0.0],
                coupling_loop="invalid",
            )

    def test_outfall_exchange_injects_surface_source_cpu(self):
        mod = self._build_outfall_network()
        mod.state.node_depth["O0"] = 1.2

        sinks, sources = mod.exchange_step(1.0, [0.2])

        self.assertEqual(len(sinks), 1)
        self.assertEqual(len(sources), 1)
        self.assertGreater(sources[0], 0.0)
        self.assertEqual(sinks[0], 0.0)

    def test_outfall_exchange_zero_storage_keeps_node_depth_at_invert(self):
        mod = self._build_outfall_network(zero_storage=True)

        mod.state.node_depth["O0"] = 1.2
        sinks, sources = mod.exchange_step(1.0, [0.2])
        self.assertGreater(sinks[0], 0.0)
        self.assertEqual(sources[0], 0.0)
        self.assertAlmostEqual(mod.state.node_depth["O0"], 0.0, places=12)

        mod.state.node_depth["O0"] = 0.0
        sinks, sources = mod.exchange_step(1.0, [1.2])
        self.assertGreater(sinks[0], 0.0)
        self.assertEqual(sources[0], 0.0)
        self.assertAlmostEqual(mod.state.node_depth["O0"], 0.0, places=12)

    def test_gpu_path_matches_cpu_for_outfall_surface_exchange(self):
        class _FakeNativeModule:
            @staticmethod
            def swe2d_gpu_compute_coupling_sources(cell_area, inlet_cell, inlet_flow, struct_up, struct_dn, struct_q):
                _ = (struct_up, struct_dn, struct_q)
                src = np.zeros_like(cell_area, dtype=np.float64)
                for i in range(int(inlet_cell.size)):
                    ci = int(inlet_cell[i])
                    if ci < 0 or ci >= int(src.size):
                        continue
                    area = max(float(cell_area[ci]), 1.0e-12)
                    # Kernel convention: positive inlet_flow removes surface water.
                    src[ci] -= float(inlet_flow[i]) / area
                return src

        drainage_cpu = self._build_outfall_network()
        drainage_gpu = self._build_outfall_network()
        drainage_cpu.state.node_depth["O0"] = 1.2
        drainage_gpu.state.node_depth["O0"] = 1.2

        cpu_controller = SWE2DCouplingController(
            cell_area=[5.0],
            cell_bed=[0.0],
            drainage=drainage_cpu,
            structures=None,
            coupling_loop="cpu",
        )
        gpu_controller = SWE2DCouplingController(
            cell_area=[5.0],
            cell_bed=[0.0],
            drainage=drainage_gpu,
            structures=None,
            coupling_loop="cpu",
        )

        h = np.asarray([0.2], dtype=np.float64)
        hu = np.asarray([0.0], dtype=np.float64)
        hv = np.asarray([0.0], dtype=np.float64)
        src_cpu = cpu_controller.compute_source_rates(0.0, 1.0, h, hu, hv)
        src_gpu = gpu_controller._compute_source_rates_cuda(_FakeNativeModule(), 0.0, 1.0, h)

        self.assertEqual(src_cpu.size, 1)
        self.assertEqual(src_gpu.size, 1)
        self.assertGreater(src_cpu[0], 0.0)
        self.assertAlmostEqual(float(src_cpu[0]), float(src_gpu[0]), places=10)

    def test_cuda_coupling_path_uses_gpu_drainage_step_when_enabled(self):
        class _FakeNativeModule:
            called = False

            @staticmethod
            def swe2d_gpu_drainage_step(
                cell_wse,
                cell_area,
                node_invert_elev,
                node_max_depth,
                node_surface_area,
                link_from,
                link_to,
                link_length,
                link_roughness_n,
                link_diameter,
                link_max_flow,
                inlet_cell,
                inlet_node,
                inlet_crest_elev,
                inlet_width,
                inlet_coefficient,
                inlet_max_capture,
                outfall_cell,
                outfall_node,
                outfall_invert_elev,
                outfall_diameter,
                outfall_coefficient,
                outfall_max_flow,
                outfall_zero_storage,
                pipe_end_cell,
                pipe_end_node,
                pipe_end_invert_elev,
                pipe_end_diameter,
                pipe_end_area,
                pipe_end_inlet_loss_k,
                pipe_end_outlet_loss_k,
                cell_depth,
                gpu_node_depth,
                gpu_link_flow,
                dt_s,
                gravity,
                solver_mode,
                head_deadband_m,
                dynamic_flow_relaxation,
            ):
                _ = (
                    cell_wse,
                    node_invert_elev,
                    node_max_depth,
                    node_surface_area,
                    link_from,
                    link_to,
                    link_length,
                    link_roughness_n,
                    link_diameter,
                    link_max_flow,
                    inlet_node,
                    inlet_crest_elev,
                    inlet_width,
                    inlet_coefficient,
                    inlet_max_capture,
                    outfall_cell,
                    outfall_node,
                    outfall_invert_elev,
                    outfall_diameter,
                    outfall_coefficient,
                    outfall_max_flow,
                    outfall_zero_storage,
                    pipe_end_cell,
                    pipe_end_node,
                    pipe_end_invert_elev,
                    pipe_end_diameter,
                    pipe_end_area,
                    pipe_end_inlet_loss_k,
                    pipe_end_outlet_loss_k,
                    cell_depth,
                    dt_s,
                    gravity,
                    solver_mode,
                    head_deadband_m,
                    dynamic_flow_relaxation,
                )
                _FakeNativeModule.called = True
                q_cell = np.zeros_like(cell_area, dtype=np.float64)
                q_cell[int(inlet_cell[0])] = -0.25
                diag = {
                    "max_node_depth": float(gpu_node_depth[0]),
                    "max_link_flow": 0.0,
                    "limiter_events": 0.0,
                    "limiter_volume_m3": 0.0,
                }
                return gpu_node_depth.copy(), gpu_link_flow.copy(), q_cell, diag

            @staticmethod
            def swe2d_gpu_compute_coupling_sources(cell_area, inlet_cell, inlet_flow, struct_up, struct_dn, struct_q):
                _ = (struct_up, struct_dn, struct_q)
                src = np.zeros_like(cell_area, dtype=np.float64)
                for i in range(int(inlet_cell.size)):
                    ci = int(inlet_cell[i])
                    if ci < 0 or ci >= int(src.size):
                        continue
                    area = max(float(cell_area[ci]), 1.0e-12)
                    src[ci] -= float(inlet_flow[i]) / area
                return src

        drainage = self._build_simple_network()
        drainage.state.node_depth["N0"] = 0.4
        controller = SWE2DCouplingController(
            cell_area=[5.0, 5.0],
            cell_bed=[0.0, 0.0],
            drainage=drainage,
            structures=None,
            coupling_loop="cuda",
            drainage_solver_backend="gpu",
        )
        controller._native_cuda_module = lambda: _FakeNativeModule()

        src = controller.compute_source_rates(
            t_s=0.0,
            dt_s=1.0,
            h=np.asarray([1.0, 0.0], dtype=np.float64),
            hu=np.zeros(2, dtype=np.float64),
            hv=np.zeros(2, dtype=np.float64),
        )

        self.assertTrue(_FakeNativeModule.called)
        self.assertEqual(src.size, 2)
        self.assertAlmostEqual(float(src[0]), -0.25 / 5.0, places=12)
        self.assertAlmostEqual(float(src[1]), 0.0, places=12)
        self.assertAlmostEqual(controller.last_diag.drainage_max_node_depth, 0.4, places=12)

    def test_cuda_coupling_passes_pipe_end_arrays_to_gpu_drainage_step(self):
        class _FakeNativeModule:
            called = False

            @staticmethod
            def swe2d_gpu_drainage_step(
                cell_wse,
                cell_area,
                node_invert_elev,
                node_max_depth,
                node_surface_area,
                link_from,
                link_to,
                link_length,
                link_roughness_n,
                link_diameter,
                link_max_flow,
                inlet_cell,
                inlet_node,
                inlet_crest_elev,
                inlet_width,
                inlet_coefficient,
                inlet_max_capture,
                outfall_cell,
                outfall_node,
                outfall_invert_elev,
                outfall_diameter,
                outfall_coefficient,
                outfall_max_flow,
                outfall_zero_storage,
                pipe_end_cell,
                pipe_end_node,
                pipe_end_invert_elev,
                pipe_end_diameter,
                pipe_end_area,
                pipe_end_inlet_loss_k,
                pipe_end_outlet_loss_k,
                cell_depth,
                gpu_node_depth,
                gpu_link_flow,
                dt_s,
                gravity,
                solver_mode,
                head_deadband_m,
                dynamic_flow_relaxation,
            ):
                _ = (
                    cell_wse,
                    node_invert_elev,
                    node_max_depth,
                    node_surface_area,
                    link_from,
                    link_to,
                    link_length,
                    link_roughness_n,
                    link_diameter,
                    link_max_flow,
                    inlet_cell,
                    inlet_node,
                    inlet_crest_elev,
                    inlet_width,
                    inlet_coefficient,
                    inlet_max_capture,
                    outfall_cell,
                    outfall_node,
                    outfall_invert_elev,
                    outfall_diameter,
                    outfall_coefficient,
                    outfall_max_flow,
                    outfall_zero_storage,
                    cell_depth,
                    gpu_link_flow,
                    dt_s,
                    gravity,
                    solver_mode,
                    head_deadband_m,
                    dynamic_flow_relaxation,
                )
                _FakeNativeModule.called = True
                assert int(pipe_end_cell.size) == 1
                assert int(pipe_end_node.size) == 1
                assert float(pipe_end_invert_elev[0]) == 0.0
                assert float(pipe_end_diameter[0]) == 1.0
                assert abs(float(pipe_end_inlet_loss_k[0]) - 0.7) < 1.0e-12
                assert abs(float(pipe_end_outlet_loss_k[0]) - 1.2) < 1.0e-12
                q_cell = np.zeros_like(cell_area, dtype=np.float64)
                diag = {
                    "max_node_depth": float(gpu_node_depth[0]),
                    "max_link_flow": 0.0,
                    "limiter_events": 0.0,
                    "limiter_volume_m3": 0.0,
                }
                return gpu_node_depth.copy(), gpu_link_flow.copy(), q_cell, diag

            @staticmethod
            def swe2d_gpu_compute_coupling_sources(cell_area, inlet_cell, inlet_flow, struct_up, struct_dn, struct_q):
                _ = (inlet_cell, inlet_flow, struct_up, struct_dn, struct_q)
                return np.zeros_like(cell_area, dtype=np.float64)

        drainage = self._build_pipe_end_network()
        drainage.state.node_depth["P0"] = 0.2
        controller = SWE2DCouplingController(
            cell_area=[5.0],
            cell_bed=[0.0],
            drainage=drainage,
            structures=None,
            coupling_loop="cuda",
            drainage_solver_backend="gpu",
        )
        controller._native_cuda_module = lambda: _FakeNativeModule()

        src = controller.compute_source_rates(
            t_s=0.0,
            dt_s=1.0,
            h=np.asarray([1.0], dtype=np.float64),
            hu=np.zeros(1, dtype=np.float64),
            hv=np.zeros(1, dtype=np.float64),
        )

        self.assertTrue(_FakeNativeModule.called)
        self.assertEqual(src.size, 1)

    @unittest.skipUnless(swe2d_available() and swe2d_gpu_available(), "native SWE2D CUDA backend not available")
    def test_backend_gpu_run_combines_rain_and_drainage_sources(self):
        backend = SWE2DBackend(use_gpu=True)
        node_x = np.asarray([0.0, 1.0, 1.0, 0.0], dtype=np.float64)
        node_y = np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
        node_z = np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        cell_nodes = np.asarray([0, 1, 2, 0, 2, 3], dtype=np.int32)
        backend.build_mesh(node_x, node_y, node_z, cell_nodes)

        drainage = self._build_simple_network()
        drainage.state.node_depth["N0"] = 0.1
        controller = SWE2DCouplingController(
            cell_area=backend.cell_areas(),
            cell_bed=np.zeros(backend.n_cells, dtype=np.float64),
            drainage=drainage,
            structures=None,
            coupling_loop="cuda",
            drainage_solver_backend="gpu",
        )

        backend.initialize(
            h0=np.asarray([0.2, 0.1], dtype=np.float64),
            hu0=np.zeros(2, dtype=np.float64),
            hv0=np.zeros(2, dtype=np.float64),
            dt_fixed=0.05,
            dt_max=0.05,
        )

        rain_rate = 0.01

        def combined_source_callback(t_s, dt_s, h, hu, hv):
            drainage_src = controller.compute_source_rates(t_s, dt_s, h, hu, hv)
            return drainage_src + rain_rate

        backend.run(
            t_end=0.1,
            dt_request=0.05,
            source_rate_callback=combined_source_callback,
        )
        h, _, _ = backend.get_state()
        self.assertTrue(np.all(np.isfinite(h)))
        self.assertTrue(backend.gpu_active())
        self.assertGreater(controller.last_diag.drainage_max_link_flow, 0.0)
        self.assertGreater(float(np.max(h)), 0.1)
        backend.destroy()

    @unittest.skipUnless(swe2d_available() and swe2d_gpu_available(), "native SWE2D CUDA backend not available")
    def test_backend_gpu_run_combines_rain_and_drainage_sources_rollout_mode(self):
        backend = SWE2DBackend(use_gpu=True)
        node_x = np.asarray([0.0, 1.0, 1.0, 0.0], dtype=np.float64)
        node_y = np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
        node_z = np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        cell_nodes = np.asarray([0, 1, 2, 0, 2, 3], dtype=np.int32)
        backend.build_mesh(node_x, node_y, node_z, cell_nodes)

        drainage = self._build_simple_network()
        drainage.state.node_depth["N0"] = 0.1
        controller = SWE2DCouplingController(
            cell_area=backend.cell_areas(),
            cell_bed=np.zeros(backend.n_cells, dtype=np.float64),
            drainage=drainage,
            structures=None,
            coupling_loop="cuda",
            drainage_solver_backend="gpu",
        )

        backend.initialize(
            h0=np.asarray([0.2, 0.1], dtype=np.float64),
            hu0=np.zeros(2, dtype=np.float64),
            hv0=np.zeros(2, dtype=np.float64),
            dt_fixed=0.05,
            dt_max=0.05,
            godunov_mode=1,
        )

        rain_rate = 0.01

        def combined_source_callback(t_s, dt_s, h, hu, hv):
            drainage_src = controller.compute_source_rates(t_s, dt_s, h, hu, hv)
            return drainage_src + rain_rate

        backend.run(
            t_end=0.1,
            dt_request=0.05,
            source_rate_callback=combined_source_callback,
        )
        h, _, _ = backend.get_state()
        self.assertTrue(np.all(np.isfinite(h)))
        self.assertTrue(backend.gpu_active())
        self.assertGreater(controller.last_diag.drainage_max_link_flow, 0.0)
        self.assertGreater(float(np.max(h)), 0.1)
        backend.destroy()

    @unittest.skipUnless(swe2d_available(), "native SWE2D backend not available")
    def test_backend_run_with_coupling_controller(self):
        backend = SWE2DBackend(use_gpu=False)
        node_x = np.asarray([0.0, 1.0, 1.0, 0.0], dtype=np.float64)
        node_y = np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
        node_z = np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        cell_nodes = np.asarray([0, 1, 2, 0, 2, 3], dtype=np.int32)
        backend.build_mesh(node_x, node_y, node_z, cell_nodes)

        drainage = self._build_simple_network()
        drainage.state.node_depth["N0"] = 0.1
        structure = HydraulicStructure(
            structure_id="C0",
            structure_type=StructureType.CULVERT,
            upstream_cell=0,
            downstream_cell=1,
            crest_elev=0.0,
            metadata={
                "diameter": 1.0,
                "length": 12.0,
                "roughness_n": 0.013,
                "cd": 0.75,
            },
        )
        structures = SWE2DStructureModule(HydraulicStructureConfig(enabled=True, structures=[structure]))
        controller = SWE2DCouplingController(
            cell_area=backend.cell_areas(),
            cell_bed=np.zeros(backend.n_cells, dtype=np.float64),
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
        self.assertGreater(controller.last_diag.structure_total_flow, 0.0)
        self.assertGreater(controller.last_diag.drainage_max_link_flow, 0.0)
        backend.destroy()

    @unittest.skipUnless(swe2d_available(), "native SWE2D backend not available")
    def test_backend_run_with_coupling_controller_native_injection(self):
        backend = SWE2DBackend(use_gpu=False)
        node_x = np.asarray([0.0, 1.0, 0.0, 1.0], dtype=np.float64)
        node_y = np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
        node_z = np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        cell_nodes = np.asarray([0, 1, 2, 1, 3, 2], dtype=np.int32)
        backend.build_mesh(node_x, node_y, node_z, cell_nodes)

        drainage = self._build_simple_network()
        drainage.state.node_depth["N0"] = 0.1
        structure = HydraulicStructure(
            structure_id="C0",
            structure_type=StructureType.CULVERT,
            upstream_cell=0,
            downstream_cell=1,
            crest_elev=0.0,
            metadata={
                "diameter": 1.0,
                "length": 12.0,
                "roughness_n": 0.013,
                "cd": 0.75,
            },
        )
        structures = SWE2DStructureModule(HydraulicStructureConfig(enabled=True, structures=[structure]))
        controller = SWE2DCouplingController(
            cell_area=backend.cell_areas(),
            cell_bed=np.zeros(backend.n_cells, dtype=np.float64),
            drainage=drainage,
            structures=structures,
        )

        backend.initialize(
            h0=np.asarray([0.2, 0.1], dtype=np.float64),
            hu0=np.zeros(2, dtype=np.float64),
            hv0=np.zeros(2, dtype=np.float64),
            dt_fixed=0.1,
            dt_max=0.1,
        )

        backend.run(
            t_end=0.5,
            dt_request=0.1,
            source_rate_callback=controller.source_rate_callback(),
            use_native_source_injection=True,
        )
        h, _, _ = backend.get_state()
        self.assertTrue(np.all(np.isfinite(h)))
        self.assertGreater(controller.last_diag.structure_total_flow, 0.0)
        self.assertGreater(controller.last_diag.drainage_max_link_flow, 0.0)
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

    @unittest.skipUnless(swe2d_available(), "native SWE2D backend not available")
    def test_daylighted_pipe_horizontal_reservoir_to_reservoir(self):
        """
        Two-reservoir horizontal pipe exchange test (US-customary units internally converted to SI).
        Tests routed, head-driven flow through the connected drainage link between
        two daylighted pipe ends.

        Setup (US-customary):
          - Reservoir A: WSE = 10 ft (3.048 m), area = 1 sq-ft (0.0929 m²)
          - Reservoir B: WSE = 6 ft (1.8288 m), area = 1 sq-ft
          - Pipe: diameter = 12 in (0.3048 m), length = 100 ft (30.48 m), n = 0.013
          - Head difference: 4 ft = 1.2192 m
          - Expected orifice flow: Q = Cd * A * sqrt(2*g*dH)
            with Cd=0.82, A=π*r²=0.0729 m², g=9.81, dH=1.2192:
            Q ≈ 0.82 * 0.0729 * sqrt(19.62 * 1.2192) ≈ 0.260 m³/s
        """
        # Convert US-customary to SI
        ft_to_m = 0.3048
        sqft_to_m2 = 0.092903

        # Reservoir water-surface elevations (ft → m)
        wse_a_m = 10.0 * ft_to_m  # 3.048 m
        wse_b_m = 6.0 * ft_to_m   # 1.8288 m

        # Pipe properties (ft/in → m)
        pipe_diameter_m = 12.0 * ft_to_m / 12.0  # 1 ft to m
        pipe_length_m = 100.0 * ft_to_m
        roughness_n = 0.013

        # Node area (surface storage areas at reservoirs, sq-ft → m²)
        node_area_m2 = 1.0 * sqft_to_m2

        # Pipe end invert elevations (set at bottom of each reservoir)
        invert_a_m = 0.0
        invert_b_m = 0.0

        nodes = [
            DrainageNode(
                node_id="reservoir_a",
                x=0.0,
                y=0.0,
                invert_elev=invert_a_m,
                max_depth=5.0,
                node_type="pipe_end",
                metadata={"surface_area": node_area_m2},
            ),
            DrainageNode(
                node_id="reservoir_b",
                x=pipe_length_m,
                y=0.0,
                invert_elev=invert_b_m,
                max_depth=5.0,
                node_type="pipe_end",
                metadata={"surface_area": node_area_m2},
            ),
        ]

        # Pipe link (active routed link between daylighted ends)
        links = [
            DrainageLink(
                link_id="pipe_ab",
                from_node_id="reservoir_a",
                to_node_id="reservoir_b",
                length=pipe_length_m,
                roughness_n=roughness_n,
                diameter=pipe_diameter_m,
            )
        ]

        # Two daylighted pipe ends (surface-head BCs to routed link)
        pipe_ends = [
            PipeEndExchange(
                pipe_end_id="pe_a",
                cell_id=0,  # 2D surface cell A
                node_id="reservoir_a",
                invert_elev=invert_a_m,
                diameter=pipe_diameter_m,
                coefficient=0.82,  # FHWA outlet loss coefficient
                max_flow=None,
            ),
            PipeEndExchange(
                pipe_end_id="pe_b",
                cell_id=1,  # 2D surface cell B
                node_id="reservoir_b",
                invert_elev=invert_b_m,
                diameter=pipe_diameter_m,
                coefficient=0.82,
                max_flow=None,
            ),
        ]

        cfg = PipeNetworkConfig(
            enabled=True,
            nodes=nodes,
            links=links,
            inlets=[],
            outfalls=[],
            pipe_ends=pipe_ends,
            gravity=9.81,
            solver_mode=DrainageSolverMode.EGL,
        )

        mod = SWE2DUrbanDrainageModule(cfg)
        mod.initialize()

        # Set initial reservoir surface elevations as 2D cell WSE
        cell_wse = np.array([wse_a_m, wse_b_m], dtype=np.float64)

        # Run one exchange step (dt=1.0 s)
        dt_s = 1.0
        sinks, sources = mod.exchange_step(dt=dt_s, cell_wse=cell_wse)

        # Expected behavior: routed cross-pipe transfer from A -> B.
        self.assertEqual(len(sinks), 2, "Should have 2 cells")
        self.assertEqual(len(sources), 2, "Should have 2 cells")

        self.assertGreater(sinks[0], 0.0, "Cell A should discharge to the routed pipe")
        self.assertGreater(sources[1], 0.0, "Cell B should receive routed inflow from the pipe")
        self.assertLessEqual(sinks[1], 1.0e-12, "Cell B should not be a net sink in A->B transfer")
        self.assertLessEqual(sources[0], 1.0e-12, "Cell A should not be a net source in A->B transfer")

        # Cross-pipe transfer should be approximately conservative over the step.
        self.assertAlmostEqual(sinks[0], sources[1], delta=1.0e-3)

    @unittest.skipUnless(swe2d_available(), "native SWE2D backend not available")
    def test_daylighted_pipe_sloped_channel_to_channel(self):
        """
        Sloped pipe exchange test between two channel cross-sections (US-customary → SI).
        Tests routed daylighted pipe-end behavior through the connected link.

        Setup (US-customary):
          - Channel A: WSE = 3 ft (0.9144 m), area = 10 sq-ft (0.929 m²)
          - Channel B: WSE = 2 ft (0.6096 m), area = 10 sq-ft
          - Pipe: diameter = 2 ft (0.6096 m), length = 50 ft (15.24 m), slope = 2% (0.02)
          - Both channels above their respective pipe inverts → both drain
        """
        # Convert US-customary to SI
        ft_to_m = 0.3048
        sqft_to_m2 = 0.092903

        # Channel water-surface elevations
        wse_chan_a_m = 3.0 * ft_to_m  # 0.9144 m
        wse_chan_b_m = 2.0 * ft_to_m  # 0.6096 m

        # Pipe properties
        pipe_diameter_m = 2.0 * ft_to_m  # 0.6096 m
        pipe_length_m = 50.0 * ft_to_m  # 15.24 m
        pipe_slope = 0.02  # 2%
        roughness_n = 0.013

        # Channel node areas
        node_area_m2 = 10.0 * sqft_to_m2

        # Pipe inverts: downstream lower due to slope
        invert_upstream_m = 1.0 * ft_to_m  # 0.3048 m
        invert_drop_m = pipe_slope * pipe_length_m  # ~0.305 m
        invert_downstream_m = invert_upstream_m - invert_drop_m

        nodes = [
            DrainageNode(
                node_id="channel_a",
                x=0.0,
                y=0.0,
                invert_elev=invert_upstream_m,
                max_depth=5.0,
                node_type="pipe_end",
                metadata={"surface_area": node_area_m2},
            ),
            DrainageNode(
                node_id="channel_b",
                x=pipe_length_m,
                y=0.0,
                invert_elev=invert_downstream_m,
                max_depth=5.0,
                node_type="pipe_end",
                metadata={"surface_area": node_area_m2},
            ),
        ]

        links = [
            DrainageLink(
                link_id="sloped_pipe",
                from_node_id="channel_a",
                to_node_id="channel_b",
                length=pipe_length_m,
                roughness_n=roughness_n,
                diameter=pipe_diameter_m,
            )
        ]

        pipe_ends = [
            PipeEndExchange(
                pipe_end_id="pe_chan_a",
                cell_id=0,
                node_id="channel_a",
                invert_elev=invert_upstream_m,
                diameter=pipe_diameter_m,
                coefficient=0.82,
                max_flow=None,
            ),
            PipeEndExchange(
                pipe_end_id="pe_chan_b",
                cell_id=1,
                node_id="channel_b",
                invert_elev=invert_downstream_m,
                diameter=pipe_diameter_m,
                coefficient=0.82,
                max_flow=None,
            ),
        ]

        cfg = PipeNetworkConfig(
            enabled=True,
            nodes=nodes,
            links=links,
            inlets=[],
            outfalls=[],
            pipe_ends=pipe_ends,
            gravity=9.81,
            solver_mode=DrainageSolverMode.EGL,
        )

        mod = SWE2DUrbanDrainageModule(cfg)
        mod.initialize()

        cell_wse = np.array([wse_chan_a_m, wse_chan_b_m], dtype=np.float64)

        dt_s = 1.0
        sinks, sources = mod.exchange_step(dt=dt_s, cell_wse=cell_wse)

        # Channel A is higher than B; routed transfer should go A -> B.
        self.assertEqual(len(sinks), 2)
        self.assertEqual(len(sources), 2)

        self.assertGreater(sinks[0], 0.0, "Channel A should discharge")
        self.assertGreater(sources[1], 0.0, "Channel B should receive routed inflow")
        self.assertLessEqual(sinks[1], 1.0e-12, "Channel B should not be a net sink in A->B transfer")
        self.assertLessEqual(sources[0], 1.0e-12, "Channel A should not be a net source in A->B transfer")
        self.assertAlmostEqual(sinks[0], sources[1], delta=1.0e-3)

        # Verify that pipe-end nodes are registered in _outfall_exchange_nodes
        # (i.e., they are recognized as daylighted endpoints without separate inlet/outfall objects)
        self.assertIn("channel_a", mod._outfall_exchange_nodes)
        self.assertIn("channel_b", mod._outfall_exchange_nodes)

    @unittest.skipUnless(swe2d_available(), "native SWE2D backend not available")
    def test_daylighted_pipe_end_loss_coefficients_reduce_transfer(self):
        nodes = [
            DrainageNode(node_id="n0", x=0.0, y=0.0, invert_elev=0.0, max_depth=3.0, node_type="pipe_end", metadata={"surface_area": 0.5}),
            DrainageNode(node_id="n1", x=10.0, y=0.0, invert_elev=0.0, max_depth=3.0, node_type="pipe_end", metadata={"surface_area": 0.5}),
        ]
        links = [
            DrainageLink(
                link_id="l0",
                from_node_id="n0",
                to_node_id="n1",
                length=10.0,
                roughness_n=0.013,
                diameter=0.5,
            )
        ]

        def _run_case(k_in: float, k_out: float):
            cfg = PipeNetworkConfig(
                enabled=True,
                nodes=nodes,
                links=links,
                inlets=[],
                outfalls=[],
                pipe_ends=[
                    PipeEndExchange(pipe_end_id="pe0", cell_id=0, node_id="n0", invert_elev=0.0, diameter=0.5, inlet_loss_k=k_in, outlet_loss_k=k_out),
                    PipeEndExchange(pipe_end_id="pe1", cell_id=1, node_id="n1", invert_elev=0.0, diameter=0.5, inlet_loss_k=k_in, outlet_loss_k=k_out),
                ],
                gravity=9.81,
                solver_mode=DrainageSolverMode.EGL,
            )
            mod = SWE2DUrbanDrainageModule(cfg)
            mod.initialize()
            cell_wse = np.asarray([2.0, 1.0], dtype=np.float64)
            # Warm-up establishes directional link flow used by end-loss head correction.
            mod.exchange_step(dt=1.0, cell_wse=cell_wse)
            sinks, sources = mod.exchange_step(dt=1.0, cell_wse=cell_wse)
            return float(sinks[0]), float(sources[1])

        q_sink_low, q_src_low = _run_case(0.0, 0.0)
        q_sink_high, q_src_high = _run_case(2.0, 2.0)

        self.assertGreater(q_sink_low, 0.0)
        self.assertGreater(q_src_low, 0.0)
        self.assertGreater(q_sink_low, q_sink_high, "Higher end-loss coefficients should reduce routed transfer")
        self.assertGreater(q_src_low, q_src_high, "Higher end-loss coefficients should reduce routed transfer")


@unittest.skipUnless(_HAVE_WORKBENCH, "workbench module unavailable")
class TestSWE2DExternalSourceApplication(unittest.TestCase):
    class _Spin:
        def __init__(self, val):
            self._val = float(val)

        def value(self):
            return self._val

    class _Harness:
        def __init__(self, cell_area, h_min=1.0e-6):
            self._cell_area = np.asarray(cell_area, dtype=np.float64)
            self.h_min_spin = TestSWE2DExternalSourceApplication._Spin(h_min)

        def _mesh_cell_areas(self):
            return self._cell_area

    class _BackendStub:
        def __init__(self, n_cells, n_cells_as_callable=False):
            self._state = (
                np.zeros((int(n_cells),), dtype=np.float64),
                np.zeros((int(n_cells),), dtype=np.float64),
                np.zeros((int(n_cells),), dtype=np.float64),
            )
            if n_cells_as_callable:
                self.n_cells = lambda: int(n_cells)
            else:
                self.n_cells = int(n_cells)
            self.native_calls = []

        def set_external_sources_native(self, src):
            self.native_calls.append(None if src is None else np.asarray(src, dtype=np.float64).copy())

        def get_state(self):
            h, hu, hv = self._state
            return h.copy(), hu.copy(), hv.copy()

        def set_state(self, h, hu, hv):
            self._state = (
                np.asarray(h, dtype=np.float64).copy(),
                np.asarray(hu, dtype=np.float64).copy(),
                np.asarray(hv, dtype=np.float64).copy(),
            )

    class _BackendFallbackStub(_BackendStub):
        def set_external_sources_native(self, src):
            raise RuntimeError("native source injection unavailable")

    def test_apply_external_sources_rain_only_updates_depth(self):
        harness = self._Harness(cell_area=[2.0, 3.0], h_min=1.0e-6)
        backend = self._BackendStub(n_cells=2, n_cells_as_callable=False)

        _wbqt.SWE2DWorkbenchDialog._apply_external_sources(
            harness,
            backend,
            dt_step=2.0,
            rain_rate_model=0.01,
            cell_source_model=None,
            coupled_source_rate=None,
            prefer_native_injection=False,
        )

        h, hu, hv = backend.get_state()
        np.testing.assert_allclose(h, np.asarray([0.02, 0.02], dtype=np.float64), rtol=0.0, atol=1.0e-12)
        np.testing.assert_allclose(hu, 0.0, rtol=0.0, atol=1.0e-12)
        np.testing.assert_allclose(hv, 0.0, rtol=0.0, atol=1.0e-12)

    def test_apply_external_sources_drainage_only_uses_coupled_source(self):
        harness = self._Harness(cell_area=[1.0, 1.0], h_min=1.0e-6)
        backend = self._BackendStub(n_cells=2, n_cells_as_callable=True)
        backend.set_state(
            np.asarray([0.2, 0.1], dtype=np.float64),
            np.asarray([0.0, 0.0], dtype=np.float64),
            np.asarray([0.0, 0.0], dtype=np.float64),
        )

        _wbqt.SWE2DWorkbenchDialog._apply_external_sources(
            harness,
            backend,
            dt_step=1.0,
            rain_rate_model=0.0,
            cell_source_model=None,
            coupled_source_rate=np.asarray([-0.10, 0.05], dtype=np.float64),
            prefer_native_injection=False,
        )

        h, _, _ = backend.get_state()
        np.testing.assert_allclose(h, np.asarray([0.10, 0.15], dtype=np.float64), rtol=0.0, atol=1.0e-12)

    def test_apply_external_sources_combines_terms_and_falls_back_from_native(self):
        harness = self._Harness(cell_area=[2.0, 4.0], h_min=1.0e-6)
        backend = self._BackendFallbackStub(n_cells=2, n_cells_as_callable=False)

        _wbqt.SWE2DWorkbenchDialog._apply_external_sources(
            harness,
            backend,
            dt_step=2.0,
            rain_rate_model=0.02,
            cell_source_model=np.asarray([0.2, 0.0], dtype=np.float64),
            coupled_source_rate=np.asarray([-0.01, 0.03], dtype=np.float64),
            prefer_native_injection=True,
        )

        h, _, _ = backend.get_state()
        # src = rain + (Qcell/area) + coupled = [0.11, 0.05] m/s; dt=2 s
        np.testing.assert_allclose(h, np.asarray([0.22, 0.10], dtype=np.float64), rtol=0.0, atol=1.0e-12)

    @unittest.skipUnless(swe2d_available(), "native SWE2D backend not available")
    def test_backend_native_source_injection_mode(self):
        backend = SWE2DBackend(use_gpu=False)
        node_x = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        node_y = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        node_z = np.asarray([0.0, 0.0, 0.0], dtype=np.float64)
        cell_nodes = np.asarray([0, 1, 2], dtype=np.int32)
        backend.build_mesh(node_x, node_y, node_z, cell_nodes)

        backend.initialize(
            h0=np.asarray([0.0], dtype=np.float64),
            hu0=np.asarray([0.0], dtype=np.float64),
            hv0=np.asarray([0.0], dtype=np.float64),
            dt_fixed=0.1,
            dt_max=0.1,
        )

        backend.run(
            t_end=0.2,
            dt_request=0.1,
            source_rate_callback=lambda t, dt, h, hu, hv: np.asarray([0.2], dtype=np.float64),
            use_native_source_injection=True,
        )
        h, _, _ = backend.get_state()
        # Native-injection mode applies callback output on the subsequent step.
        self.assertAlmostEqual(float(h[0]), 0.02, places=12)
        backend.destroy()


if __name__ == "__main__":
    unittest.main()
