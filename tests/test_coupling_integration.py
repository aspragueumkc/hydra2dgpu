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

    def test_coupling_controller_rejects_invalid_loop_mode(self):
        with self.assertRaises(ValueError):
            SWE2DCouplingController(
                cell_area=[1.0],
                cell_bed=[0.0],
                drainage_gpu_method="invalid",
            )

    def test_link_type_value_map_includes_culvert(self):
        """_DRAIN_LINK_TYPE_VALUE_MAP must include the culvert entry."""
        if not _HAVE_WORKBENCH:
            self.skipTest("workbench module not importable")
        self.assertIn(
            "culvert",
            _wbqt._DRAIN_LINK_TYPE_VALUE_MAP.values(),
            "_DRAIN_LINK_TYPE_VALUE_MAP must contain 'culvert'")


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


if __name__ == "__main__":
    unittest.main()
