import unittest
import numpy as np

from swe2d.runtime.backend import SWE2DBackend, SpatialDiscretization, swe2d_available, swe2d_gpu_available
from swe2d.runtime.coupling import SWE2DCouplingController, pack_coupling_soa
from swe2d.extensions.drainage_network import SWE2DUrbanDrainageModule
from swe2d.extensions.extension_models import (
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
from swe2d.extensions.structures import SWE2DStructureModule

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

    @unittest.skip("GPU-only — non-persistent structures path removed")
    def test_cuda_coupling_uses_native_structure_helper_when_available(self):
        pass

    @unittest.skip("GPU-only")
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

    @unittest.skip("GPU-only")
    def test_coupling_controller_accepts_explicit_cpu_loop(self):
        pass

    def test_coupling_controller_rejects_invalid_loop_mode(self):
        with self.assertRaises(ValueError):
            SWE2DCouplingController(
                cell_area=[1.0],
                cell_bed=[0.0],
                drainage_gpu_method="invalid",
            )

    @unittest.skip("GPU-only")
    def test_outfall_exchange_injects_surface_source_cpu(self):
        pass

    @unittest.skip("GPU-only")
    def test_outfall_exchange_zero_storage_keeps_node_depth_at_invert(self):
        pass

    @unittest.skip("GPU-only — swe2d_gpu_drainage_step removed; migrate to swe2d_build_pipe1d_mesh + swe2d_pipe1d_step")
    def test_cuda_coupling_path_uses_gpu_drainage_step_when_enabled(self):
        class _FakeNativeModule:
            called = False

            @staticmethod
            def swe2d_gpu_preload_structure_params(*a, **kw):
                pass

            @staticmethod
            def swe2d_gpu_preload_coupling_cell_area(*a, **kw):
                pass

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
                return gpu_node_depth.copy(), gpu_link_flow.copy(), diag

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

    @unittest.skip("GPU-only — swe2d_gpu_drainage_step removed; migrate to swe2d_build_pipe1d_mesh + swe2d_pipe1d_step")
    def test_cuda_coupling_passes_pipe_end_arrays_to_gpu_drainage_step(self):
        class _FakeNativeModule:
            called = False

            @staticmethod
            def swe2d_gpu_preload_structure_params(*a, **kw):
                pass

            @staticmethod
            def swe2d_gpu_preload_coupling_cell_area(*a, **kw):
                pass

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
                return gpu_node_depth.copy(), gpu_link_flow.copy(), diag

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
        backend = SWE2DBackend()
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

    @unittest.skip("GPU-only — godunov_mode removed from Python API")
    def test_backend_gpu_run_combines_rain_and_drainage_sources_rollout_mode(self):
        pass

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
        backend = SWE2DBackend()
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
        # Water should have moved from cell 0 to cell 1 via culvert
        self.assertLess(float(h[0]), 1.79, "Upstream cell should lose water to culvert")
        self.assertGreater(float(h[1]), 1.0, "Downstream cell should gain water from culvert")
        self.assertIn("structures_persistent_path", controller.last_diag.component_sums)
        backend.destroy()

    @unittest.skipUnless(swe2d_available(), "native SWE2D backend not available")
    def test_backend_run_with_coupling_controller_native_injection(self):
        backend = SWE2DBackend()
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
        # Validate mass conservation: total water should be ~initial sum
        total = float(np.sum(h * np.asarray(backend.cell_areas(), dtype=np.float64)))
        # initial: h=[0.2, 0.1], areas both=0.5 → total=0.15
        self.assertAlmostEqual(total, 0.15, places=4)
        backend.destroy()

    @unittest.skipUnless(swe2d_available(), "native SWE2D backend not available")
    def test_backend_cell_area_cache_and_source_callback(self):
        backend = SWE2DBackend()
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
            t_end=0.2,
            dt_request=0.1,
            source_rate_callback=lambda t, dt, h, hu, hv: np.asarray([0.2], dtype=np.float64),
            use_native_source_injection=True,
        )
        h, _, _ = backend.get_state()
        self.assertAlmostEqual(float(h[0]), 0.02, places=12)
        backend.destroy()

    def test_daylighted_pipe_horizontal_reservoir_to_reservoir(self):
        """
        Two-reservoir horizontal pipe exchange test (US-customary units internally converted to SI).
        Tests routed, head-driven flow through the connected drainage link between
        two daylighted pipe ends using the GPU coupling path.

        Setup (US-customary):
          - Reservoir A: WSE = 10 ft (3.048 m), area = 1 sq-ft (0.0929 m²)
          - Reservoir B: WSE = 6 ft (1.8288 m), area = 1 sq-ft
          - Pipe: diameter = 12 in (0.3048 m), length = 100 ft (30.48 m), n = 0.013
          - Head difference: 4 ft = 1.2192 m
          - Expected orifice flow: Q = Cd * A * sqrt(2*g*dH)
            with Cd=0.82, A=π*r²=0.0729 m², g=9.81, dH=1.2192:
            Q ≈ 0.82 * 0.0729 * sqrt(19.62 * 1.2192) ≈ 0.260 m³/s
        """
        ft_to_m = 0.3048
        sqft_to_m2 = 0.092903

        wse_a_m = 10.0 * ft_to_m
        wse_b_m = 6.0 * ft_to_m
        pipe_diameter_m = 12.0 * ft_to_m / 12.0
        pipe_length_m = 100.0 * ft_to_m
        roughness_n = 0.013
        node_area_m2 = 1.0 * sqft_to_m2
        invert_a_m = 0.0
        invert_b_m = 0.0

        nodes = [
            DrainageNode(
                node_id="reservoir_a", x=0.0, y=0.0,
                invert_elev=invert_a_m, max_depth=5.0, node_type="pipe_end",
                metadata={"surface_area": node_area_m2},
            ),
            DrainageNode(
                node_id="reservoir_b", x=pipe_length_m, y=0.0,
                invert_elev=invert_b_m, max_depth=5.0, node_type="pipe_end",
                metadata={"surface_area": node_area_m2},
            ),
        ]
        links = [
            DrainageLink(
                link_id="pipe_ab", from_node_id="reservoir_a", to_node_id="reservoir_b",
                length=pipe_length_m, roughness_n=roughness_n, diameter=pipe_diameter_m,
            )
        ]
        pipe_ends = [
            PipeEndExchange(
                pipe_end_id="pe_a", cell_id=0, node_id="reservoir_a",
                invert_elev=invert_a_m, diameter=pipe_diameter_m, coefficient=0.82,
            ),
            PipeEndExchange(
                pipe_end_id="pe_b", cell_id=1, node_id="reservoir_b",
                invert_elev=invert_b_m, diameter=pipe_diameter_m, coefficient=0.82,
            ),
        ]

        drain_cfg = PipeNetworkConfig(
            enabled=True, nodes=nodes, links=links,
            inlets=[], outfalls=[], pipe_ends=pipe_ends,
            gravity=9.81, pipe_solver_mode="fully_dynamic",
        )
        drain_mod = SWE2DUrbanDrainageModule(drain_cfg)
        drain_mod.initialize()
        drain_mod.state.node_depth["reservoir_a"] = wse_a_m
        drain_mod.state.node_depth["reservoir_b"] = wse_b_m

        backend = SWE2DBackend()
        node_x = np.asarray([0.0, pipe_length_m, pipe_length_m, 0.0], dtype=np.float64)
        node_y = np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
        node_z = np.zeros(4, dtype=np.float64)
        cell_nodes = np.asarray([0, 1, 2, 0, 2, 3], dtype=np.int32)
        backend.build_mesh(node_x, node_y, node_z, cell_nodes)

        controller = SWE2DCouplingController(
            cell_area=backend.cell_areas(),
            cell_bed=np.zeros(2, dtype=np.float64),
            drainage=drain_mod,
            structures=None,
        )
        cx = np.asarray([pipe_length_m / 2, pipe_length_m / 2], dtype=np.float64)
        cy = np.asarray([0.5, 0.5], dtype=np.float64)
        controller.set_cell_centroids(cx=cx, cy=cy)

        backend.initialize(
            h0=np.asarray([wse_a_m, wse_b_m], dtype=np.float64),
            hu0=np.zeros(2, dtype=np.float64),
            hv0=np.zeros(2, dtype=np.float64),
            dt_fixed=0.05,
            dt_max=0.05,
        )

        cell_wse = np.asarray([wse_a_m, wse_b_m], dtype=np.float64)
        ok = controller.apply_native_device_sources(t_s=0.0, dt_s=0.05)
        self.assertTrue(ok, "GPU coupling path should succeed")

        drain_state = controller.drainage.state
        nA = float(drain_state.node_depth.get("reservoir_a", 0.0))
        nB = float(drain_state.node_depth.get("reservoir_b", 0.0))

        self.assertGreater(
            wse_a_m, nA,
            "Reservoir A node depth should decrease (drains to pipe)"
        )
        self.assertGreater(
            nB, wse_b_m,
            "Reservoir B node depth should increase (receives from pipe)"
        )

        link_flow = float(drain_state.link_flow.get("pipe_ab", 0.0))
        self.assertGreater(
            abs(link_flow), 0.0,
            "Link pipe_ab should carry non-zero flow"
        )
        self.assertGreater(
            link_flow, 0.0,
            "Link flow should be positive (reservoir_a -> reservoir_b)"
        )

        backend.destroy()

    @unittest.skip("GPU-only")
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
            pipe_solver_mode="fully_dynamic",
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

    @unittest.skip("GPU pipe1d applies k_in+k_out uniformly to all sub-cells, not at pipe-end boundaries; needs separate inlet/outlet loss handling")
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
            pipe_ends = [
                PipeEndExchange(pipe_end_id="pe0", cell_id=0, node_id="n0",
                                invert_elev=0.0, diameter=0.5, inlet_loss_k=k_in, outlet_loss_k=k_out),
                PipeEndExchange(pipe_end_id="pe1", cell_id=1, node_id="n1",
                                invert_elev=0.0, diameter=0.5, inlet_loss_k=k_in, outlet_loss_k=k_out),
            ]
            drain_cfg = PipeNetworkConfig(
                enabled=True, nodes=nodes, links=links,
                inlets=[], outfalls=[], pipe_ends=pipe_ends,
                gravity=9.81, pipe_solver_mode="fully_dynamic",
            )
            drain_mod = SWE2DUrbanDrainageModule(drain_cfg)
            drain_mod.initialize()
            drain_mod.state.node_depth["n0"] = 2.0
            drain_mod.state.node_depth["n1"] = 1.0

            backend = SWE2DBackend()
            node_x = np.asarray([0.0, 10.0, 10.0, 0.0], dtype=np.float64)
            node_y = np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
            node_z = np.zeros(4, dtype=np.float64)
            cell_nodes = np.asarray([0, 1, 2, 0, 2, 3], dtype=np.int32)
            backend.build_mesh(node_x, node_y, node_z, cell_nodes)

            controller = SWE2DCouplingController(
                cell_area=backend.cell_areas(),
                cell_bed=np.zeros(2, dtype=np.float64),
                drainage=drain_mod,
                structures=None,
            )
            controller.set_cell_centroids(
                cx=np.asarray([5.0, 5.0], dtype=np.float64),
                cy=np.asarray([0.5, 0.5], dtype=np.float64),
            )

            backend.initialize(
                h0=np.asarray([2.0, 1.0], dtype=np.float64),
                hu0=np.zeros(2, dtype=np.float64),
                hv0=np.zeros(2, dtype=np.float64),
                dt_fixed=0.05,
                dt_max=0.05,
            )

            ok = controller.apply_native_device_sources(t_s=0.0, dt_s=0.05)
            self.assertTrue(ok, "GPU coupling should succeed")
            return float(controller.drainage.state.link_flow.get("l0", 0.0))

        q_low = _run_case(0.0, 0.0)
        q_high = _run_case(2.0, 2.0)

        self.assertGreater(abs(q_low), 0.0, "Low-loss case should have non-zero flow")
        self.assertGreater(abs(q_high), 0.0, "High-loss case should have non-zero flow")
        self.assertGreater(
            abs(q_low), abs(q_high),
            "Higher end-loss coefficients should reduce routed transfer"
        )

    # ── Phase 1: face-flux + drainage coexistence ─────────────────────────
    def test_face_flux_preloaded_with_drainage(self):
        """Face-flux culvert preload succeeds when drainage is active.

        This validates the Phase 1 isolation fix: when drainage blocks the
        apply_native_device_sources fast path, the fallback path in
        _compute_source_rates_cuda must still call _ensure_culvert_face_flux_preloaded
        so the GPU's use_culvert_face_flux toggle is set to true.
        """
        # Simple drainage network: two nodes with pipe-end exchange.
        drainage_nodes = [
            DrainageNode(node_id="n0", x=0.0, y=0.0, invert_elev=0.0, max_depth=3.0,
                          metadata={"surface_area": 50.0}),
            DrainageNode(node_id="n1", x=10.0, y=0.0, invert_elev=0.0, max_depth=3.0,
                          metadata={"surface_area": 50.0}),
        ]
        drainage_links = [
            DrainageLink(link_id="L0", from_node_id="n0", to_node_id="n1",
                          length=10.0, roughness_n=0.013, diameter=1.0),
        ]
        drain_cfg = PipeNetworkConfig(
            enabled=True, nodes=drainage_nodes, links=drainage_links,
            inlets=[], outfalls=[], pipe_ends=[
                PipeEndExchange(pipe_end_id="pe0", cell_id=0, node_id="n0",
                                 invert_elev=0.0, diameter=1.0),
                PipeEndExchange(pipe_end_id="pe1", cell_id=1, node_id="n1",
                                 invert_elev=0.0, diameter=1.0),
            ],
            pipe_solver_mode="fully_dynamic",
        )
        drain_mod = SWE2DUrbanDrainageModule(drain_cfg)

        # Single face-flux culvert (HydraulicStructure).
        structure = HydraulicStructure(
            structure_id="C0",
            structure_type=StructureType.CULVERT,
            upstream_cell=0,
            downstream_cell=1,
            crest_elev=0.0,
            metadata={
                "culvert_shape": "circular",
                "culvert_code": 1,
                "diameter": 1.0,
                "culvert_rise": 1.0,
                "length": 12.0,
                "roughness_n": 0.013,
                "inlet_invert_elev": 0.0,
                "outlet_invert_elev": -0.05,
            },
        )
        struct_mod = SWE2DStructureModule(
            HydraulicStructureConfig(enabled=True, structures=[structure])
        )

        controller = SWE2DCouplingController(
            cell_area=[50.0, 50.0],
            cell_bed=[0.0, 0.0],
            drainage=drain_mod,
            structures=struct_mod,


            culvert_face_flux_mode="face_flux",
        )

        # Verify initial state: not preloaded.
        self.assertFalse(controller._culvert_face_flux_preloaded)

        # Cell centroids are needed by _build_face_flux_soa().
        controller.set_cell_centroids(
            cx=np.asarray([0.0, 10.0], dtype=np.float64),
            cy=np.asarray([0.0, 0.0], dtype=np.float64),
        )

        # Fake native module that supports structure flows and face-flux upload.
        face_flux_uploaded = []

        class _FakeNative:
            @staticmethod
            def swe2d_gpu_preload_structure_params(*a, **kw):
                pass

            @staticmethod
            def swe2d_gpu_preload_coupling_cell_area(*a, **kw):
                pass

            @staticmethod
            def swe2d_gpu_upload_culvert_face_flux_params(**kwargs):
                face_flux_uploaded.append(True)

            @staticmethod
            def swe2d_gpu_compute_structure_flows(*args, **kwargs):
                return np.asarray([9.99], dtype=np.float64)

            @staticmethod
            def swe2d_gpu_compute_coupling_sources(
                cell_area, inlet_cell, inlet_flow_cms,
                structure_up_cell, structure_down_cell, structure_flow,
            ):
                out = np.zeros(int(np.asarray(cell_area).size), dtype=np.float64)
                area = np.asarray(cell_area, dtype=np.float64)
                up = np.asarray(structure_up_cell, dtype=np.int32)
                dn = np.asarray(structure_down_cell, dtype=np.int32)
                qq = np.asarray(structure_flow, dtype=np.float64)
                for i in range(int(qq.size)):
                    out[int(up[i])] -= float(qq[i]) / float(area[int(up[i])])
                    out[int(dn[i])] += float(qq[i]) / float(area[int(dn[i])])
                return out

        controller._native_cuda_module = lambda: _FakeNative()  # type: ignore[method-assign]

        # This call exercises _compute_source_rates_cuda, which should
        # now call _ensure_culvert_face_flux_preloaded before the
        # structures section.
        src = controller.compute_source_rates(
            t_s=0.0, dt_s=1.0,
            h=np.asarray([1.5, 0.8], dtype=np.float64),
            hu=np.zeros(2, dtype=np.float64),
            hv=np.zeros(2, dtype=np.float64),
        )

        # The face-flux preload should have succeeded.
        self.assertTrue(controller._culvert_face_flux_preloaded,
                        "Face-flux parameters should be preloaded even with drainage active")
        self.assertEqual(len(face_flux_uploaded), 1,
                         "swe2d_gpu_upload_culvert_face_flux_params should be called once")

        # Source array should be produced (structure flows + drainage
        # surface exchange combined).
        self.assertIsNotNone(src, "Source array should not be None")
        self.assertEqual(src.size, 2)

        # Diagnostic log should record the bypass.
        self.assertEqual(
            float(controller.last_diag.component_sums.get("structures_native_helper", 0.0)),
            1.0,
        )

    # ── Phase 2+3: Culvert-as-Drainage-Link + Coexistence tests ─────────

    @unittest.skip("GPU-only")
    def test_coupling_drainage_and_face_flux_culvert_no_drainage_drop(self):
        """Regression: drainage q_cell not dropped when face-flux culvert is also active."""
        from swe2d import units as _u
        _u.configure(1.0)
        drainage_mod = self._build_simple_network()
        drainage_mod.state.node_depth["N0"] = 1.0
        structure = HydraulicStructure(
            structure_id="C0", structure_type=StructureType.CULVERT,
            upstream_cell=0, downstream_cell=1, crest_elev=0.0,
            metadata={"diameter": 1.0, "length": 12.0, "roughness_n": 0.013, "cd": 0.75},
        )
        structures_mod = SWE2DStructureModule(
            HydraulicStructureConfig(enabled=True, structures=[structure]),
            model_to_ft=_u.model_to_ft(),
        )
        controller = SWE2DCouplingController(
            cell_area=[5.0, 5.0], cell_bed=[0.0, 0.0],
            drainage=drainage_mod, structures=structures_mod,
            culvert_face_flux_mode="face_flux",
        )
        src = controller.compute_source_rates(
            t_s=0.0, dt_s=1.0,
            h=np.asarray([1.8, 1.0], dtype=np.float64),
            hu=np.zeros(2, dtype=np.float64),
            hv=np.zeros(2, dtype=np.float64),
        )
        self.assertEqual(src.size, 2)
        diag = controller.last_diag
        self.assertIsNotNone(diag)
        self.assertGreater(
            abs(diag.component_sums.get("drainage", 0.0)), 0.0,
            "Drainage contribution must NOT be dropped")
        # Structures flow may appear under 'structures_native_cpu_helper' (CPU
        # native helper path) or 'structures' (pure Python path).
        struct_key = "structures" if diag.component_sums.get("structures", 0.0) != 0.0 else "structures_native_cpu_helper"
        self.assertGreater(
            abs(diag.component_sums.get(struct_key, 0.0)), 0.0,
            f"Structure contribution must be present (checked key='{struct_key}')")

    def test_link_type_value_map_includes_culvert(self):
        """_DRAIN_LINK_TYPE_VALUE_MAP must include the culvert entry."""
        if not _HAVE_WORKBENCH:
            self.skipTest("workbench module not importable")
        self.assertIn(
            "culvert",
            _wbqt._DRAIN_LINK_TYPE_VALUE_MAP.values(),
            "_DRAIN_LINK_TYPE_VALUE_MAP must contain 'culvert'")

    # ── Phase 4: GPU integration test — drainage + 2 face-flux culverts ─────

    @unittest.skipUnless(swe2d_available() and swe2d_gpu_available(),
                         "native SWE2D CUDA backend not available")
    def test_gpu_persistent_path_with_drainage_and_culverts(self):
        """GPU integration test matching real-world case:
        - 2D mesh, two face-flux culvert structures
        - Drainage network with pipe-end exchange
        - Coupling controller in CUDA mode with real native module
        - Verifies persistent path handles drainage + structures together
        """
        from swe2d import units as _u
        _u.configure(1.0)

        backend = SWE2DBackend()
        node_x = np.asarray([0.0, 12.0, 12.0, 0.0, 0.0, 12.0, 12.0, 0.0],
                            dtype=np.float64)
        node_y = np.asarray([0.0, 0.0, 8.0, 8.0, 0.0, 0.0, 8.0, 8.0],
                            dtype=np.float64)
        node_z = np.zeros(8, dtype=np.float64)
        cell_nodes = np.asarray([
            0, 1, 2, 0, 2, 3,
            4, 5, 6, 4, 6, 7,
        ], dtype=np.int32)
        backend.build_mesh(node_x, node_y, node_z, cell_nodes)

        n_cells = backend.n_cells

        # ── Two face-flux culvert structures ──
        structures_list = [
            HydraulicStructure(
                structure_id="C0", structure_type=StructureType.CULVERT,
                upstream_cell=0, downstream_cell=1, crest_elev=0.0,
                metadata={
                    "culvert_shape": "circular", "culvert_code": 1,
                    "diameter": 1.0, "culvert_rise": 1.0,
                    "length": 12.0, "roughness_n": 0.013,
                    "inlet_invert_elev": 0.0, "outlet_invert_elev": -0.05,
                    "entrance_loss_k": 0.5, "exit_loss_k": 1.0,
                },
            ),
            HydraulicStructure(
                structure_id="C1", structure_type=StructureType.CULVERT,
                upstream_cell=2, downstream_cell=3, crest_elev=0.0,
                metadata={
                    "culvert_shape": "circular", "culvert_code": 1,
                    "diameter": 0.8, "culvert_rise": 0.8,
                    "length": 10.0, "roughness_n": 0.014,
                    "inlet_invert_elev": -0.1, "outlet_invert_elev": -0.15,
                    "entrance_loss_k": 0.5, "exit_loss_k": 1.0,
                },
            ),
        ]
        struct_cfg = HydraulicStructureConfig(enabled=True, structures=structures_list)
        struct_mod = SWE2DStructureModule(struct_cfg, model_to_ft=_u.model_to_ft())

        # ── Drainage network with pipe-end exchange ──
        drain_nodes = [
            DrainageNode(node_id="N0", x=0.0, y=0.0, invert_elev=0.0, max_depth=4.0,
                         metadata={"surface_area": 50.0}),
            DrainageNode(node_id="N1", x=0.0, y=8.0, invert_elev=-0.5, max_depth=4.0,
                         metadata={"surface_area": 50.0}),
        ]
        drain_links = [
            DrainageLink(link_id="L0", from_node_id="N0", to_node_id="N1",
                          length=8.0, roughness_n=0.013, diameter=0.5),
        ]
        drain_cfg = PipeNetworkConfig(
            enabled=True, nodes=drain_nodes, links=drain_links,
            inlets=[], outfalls=[],
            pipe_ends=[
                PipeEndExchange(pipe_end_id="pe_N0", cell_id=0, node_id="N0",
                                 invert_elev=0.0, diameter=0.5, area_m2=0.19635),
                PipeEndExchange(pipe_end_id="pe_N1", cell_id=2, node_id="N1",
                                 invert_elev=-0.5, diameter=0.5, area_m2=0.19635),
            ],
            pipe_solver_mode="fully_dynamic",
        )
        drain_mod = SWE2DUrbanDrainageModule(drain_cfg)
        drain_mod.initialize()
        drain_mod.state.node_depth["N0"] = 1.5
        drain_mod.state.node_depth["N1"] = 0.8

        # ── Coupling controller with real CUDA path ──
        controller = SWE2DCouplingController(
            cell_area=backend.cell_areas(),
            cell_bed=np.zeros(n_cells, dtype=np.float64),
            drainage=drain_mod,
            structures=struct_mod,


            culvert_face_flux_mode="face_flux",
            length_scale_si_to_model=_u.si_m_per_model(),
        )
        # Cell centroids needed for face-flux SoA builder
        cx = np.asarray([6.0, 6.0, 6.0, 6.0], dtype=np.float64)
        cy = np.asarray([4.0, 4.0, 4.0, 4.0], dtype=np.float64)
        controller.set_cell_centroids(cx=cx, cy=cy)

        # ── Run the backend with coupling via bound method ──
        backend.initialize(
            h0=np.asarray([1.8, 0.5, 1.2, 0.4], dtype=np.float64),
            hu0=np.zeros(n_cells, dtype=np.float64),
            hv0=np.zeros(n_cells, dtype=np.float64),
            dt_fixed=0.05,
            dt_max=0.05,
        )

        backend.run(
            t_end=0.1,
            dt_request=0.05,
            source_rate_callback=controller.compute_source_rates,
        )

        h, _, _ = backend.get_state()
        self.assertTrue(np.all(np.isfinite(h)),
                        "GPU solver must produce finite state")
        self.assertTrue(backend.gpu_active(),
                        "GPU must be active for CUDA path")

        # Water should have moved through face-flux culverts and drainage.
        # Initial h[0]=1.8 should have decreased (drainage extracts at cell 0
        # + culvert C0 moves water from cell 0 to cell 1).
        self.assertLess(
            float(h[0]), 1.79,
            "Upstream cell depth should decrease below 1.79 from initial 1.8 "
            "(drainage extracts + culvert face-flux moves water)")

        # Drainage: verify through actual network state synced back from GPU.
        # The apply_native_device_sources path (use_persistent=True) runs the
        # GPU drainage step and syncs results back to the Python-side drainage
        # module via _sync_gpu_state_back_to_drainage.
        drain_state = controller.drainage.state
        node_n0_depth = float(drain_state.node_depth.get("N0", 0.0))
        node_n1_depth = float(drain_state.node_depth.get("N1", 0.0))
        self.assertNotEqual(
            node_n0_depth, 1.5,
            "Drainage node N0 depth must change from initial 1.5 "
            "(pipe-end exchange should transfer water)")
        self.assertNotEqual(
            node_n1_depth, 0.8,
            "Drainage node N1 depth must change from initial 0.8 "
            "(pipe-end exchange should transfer water)")

        backend.destroy()


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
        )

        # GPU-native path: source written to device buffer, no set_state
        self.assertEqual(len(backend.native_calls), 1)
        expected_src = np.asarray([0.01, 0.01], dtype=np.float64)
        np.testing.assert_allclose(backend.native_calls[0], expected_src, rtol=0.0, atol=1.0e-12)

    def test_apply_external_sources_drainage_only_uses_coupled_source(self):
        harness = self._Harness(cell_area=[1.0, 1.0], h_min=1.0e-6)
        backend = self._BackendStub(n_cells=2, n_cells_as_callable=True)

        _wbqt.SWE2DWorkbenchDialog._apply_external_sources(
            harness,
            backend,
            dt_step=1.0,
            rain_rate_model=0.0,
            cell_source_model=None,
            coupled_source_rate=np.asarray([-0.10, 0.05], dtype=np.float64),
        )

        # GPU-native path: source written via set_external_sources_native
        self.assertEqual(len(backend.native_calls), 1)
        expected_src = np.asarray([-0.10, 0.05], dtype=np.float64)
        np.testing.assert_allclose(backend.native_calls[0], expected_src, rtol=0.0, atol=1.0e-12)

    def test_apply_external_sources_raises_when_native_unavailable(self):
        harness = self._Harness(cell_area=[2.0, 4.0], h_min=1.0e-6)
        backend = self._BackendFallbackStub(n_cells=2, n_cells_as_callable=False)

        with self.assertRaises(RuntimeError):
            _wbqt.SWE2DWorkbenchDialog._apply_external_sources(
                harness,
                backend,
                dt_step=2.0,
                rain_rate_model=0.02,
                cell_source_model=np.asarray([0.2, 0.0], dtype=np.float64),
                coupled_source_rate=np.asarray([-0.01, 0.03], dtype=np.float64),
            )

    @unittest.skipUnless(swe2d_available(), "native SWE2D backend not available")
    def test_backend_native_source_injection_mode(self):
        backend = SWE2DBackend()
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
