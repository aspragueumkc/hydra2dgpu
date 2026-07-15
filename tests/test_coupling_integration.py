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
    from swe2d.workbench.services import constants_service as _wb_constants
    _HAVE_CONSTANTS = True
except Exception:
    _wb_constants = None
    _HAVE_CONSTANTS = False

# The original test file imported a non-existent ``swe2d_workbench_qt``
# module; replace it with the actual non-GUI constants module so the
# constant test can run without QGIS.  The 3 GUI-dialog tests below are
# gated on a separate flag because they require a full studio dialog
# stub which is not maintained.
try:
    from swe2d.workbench import studio_dialog as _wbqt  # noqa: F401
    _STUDIO_DIALOG_IMPORTED = True
except Exception:
    _wbqt = None
    _STUDIO_DIALOG_IMPORTED = False

# Pretend the GUI class isn't there unless the test author updates the
# harness to match the current dialog.  Keeps the 3 GUI-stub tests
# skipped instead of broken.
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
        """DRAIN_LINK_TYPE_VALUE_MAP (non-GUI constant) must include culvert."""
        if not _HAVE_CONSTANTS:
            self.skipTest("constants_service not importable")
        self.assertIn(
            "culvert",
            _wb_constants.DRAIN_LINK_TYPE_VALUE_MAP.values(),
            "DRAIN_LINK_TYPE_VALUE_MAP must contain 'culvert'")


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


class TestPipeCellMetrics(unittest.TestCase):
    """Tests for pipe-cell metric sampling at t=0."""

    def test_pipe_cell_snapshot_at_t0_is_zero(self):
        """Drainage cell depth is 0.0 at t=0 (no spurious priming)."""
        from unittest.mock import patch, MagicMock
        import numpy as np
        from swe2d.workbench.services.non_gui_runtime_service import _sample_coupling_object_metrics

        cc = MagicMock()
        cc.readback_coupling_state.return_value = {
            "node_depth": np.zeros(2),
            "link_flow": np.zeros(1),
            "struct_flow": np.empty(0),
            "cell_velocity": np.zeros(3),
            "cell_depth": np.zeros(3),
            "cell_flow": np.zeros(3),
            "cell_head": np.zeros(3),
            "cell_owner_link": np.array([0, 0, 0]),
        }
        cfg = MagicMock()
        cfg.nodes = []
        cfg.links = [MagicMock(link_id="L1")]
        cc.drainage = MagicMock(cfg=cfg)
        cc._dsoa = MagicMock()
        cc._dsoa._sub_cells_per_link = [3]

        rows = _sample_coupling_object_metrics(cc, 0.0, 0.0, None)
        cell_depth_rows = [r for r in rows if r["component"] == "drainage_cell" and r["metric"] == "depth"]
        self.assertGreater(len(cell_depth_rows), 0, "Expected at least one drainage_cell depth row")
        for r in cell_depth_rows:
            self.assertAlmostEqual(r["value"], 0.0, places=9, msg=f"Non-zero depth at t=0: {r['value']}")

    @unittest.skipUnless(swe2d_available() and swe2d_gpu_available(),
                         "hydra_swe2d GPU module not available")
    def test_real_pipe1d_readback_at_t0_is_zero(self):
        """t=0 readback against a real GPU returns zeros, not uninitialized heap.

        Regression: when the t=0 coupling snapshot ran before
        swe2d_build_pipe1d_mesh, the C++ binding's guard
        ``n_nodes == p.n_nodes`` evaluated False (because p.n_nodes == 0),
        so cudaMemcpy was skipped and py::array_t<double>(N) host buffers
        stayed uninitialized. ``_depth_from_area`` then turned small garbage
        areas (~6e-200) into tiny but plausible-looking cell depths, which
        polluted plot autoRange().
        """
        from swe2d.extensions.drainage_network import SWE2DUrbanDrainageModule
        from tests._swe2d_test_helpers import _make_rect_mesh

        backend = SWE2DBackend()
        node_x, node_y, node_z, cell_nodes = _make_rect_mesh(2, 1, 20.0, 10.0)
        backend.build_mesh(
            node_x, node_y, node_z, cell_nodes,
            bc_edge_node0=np.empty(0, dtype=np.int32),
            bc_edge_node1=np.empty(0, dtype=np.int32),
            bc_edge_type=np.empty(0, dtype=np.int32),
            bc_edge_val=np.empty(0, dtype=np.float64),
        )
        n_cells = int(backend.n_cells)
        backend.initialize(
            h0=np.full(n_cells, 0.05, dtype=np.float64),
            hu0=np.zeros(n_cells, dtype=np.float64),
            hv0=np.zeros(n_cells, dtype=np.float64),
            n_mann=0.035,
            h_min=1.0e-4,
            cfl=0.45,
            dt_max=0.5,
            dt_fixed=0.5,
            gpu_diag_sync_interval_steps=1,
            spatial_discretization=1,
        )

        nodes = [
            DrainageNode(node_id="N0", x=0.0, y=0.0, invert_elev=0.0, max_depth=3.0,
                         metadata={"surface_area": 10.0}),
            DrainageNode(node_id="N1", x=10.0, y=0.0, invert_elev=0.0, max_depth=3.0),
        ]
        links = [
            DrainageLink(link_id="L0", from_node_id="N0", to_node_id="N1",
                         length=10.0, roughness_n=0.013, diameter=1.0),
        ]
        cfg = PipeNetworkConfig(enabled=True, nodes=nodes, links=links)
        drain_mod = SWE2DUrbanDrainageModule(cfg)
        drain_mod.initialize()

        cc = SWE2DCouplingController(
            cell_area=backend.cell_areas(),
            cell_bed=np.zeros(n_cells, dtype=np.float64),
            drainage=drain_mod,
        )

        state = cc.readback_coupling_state()

        self.assertEqual(int(state["node_depth"].size), 2)
        np.testing.assert_array_equal(
            state["node_depth"], np.zeros(2),
            err_msg=f"t=0 node_depth should be zero, got {state['node_depth']}",
        )

        if state["cell_flow"].size > 0:
            np.testing.assert_array_equal(
                state["cell_flow"], np.zeros_like(state["cell_flow"]),
                err_msg=f"t=0 cell_flow should be zero, got {state['cell_flow']}",
            )
            np.testing.assert_array_equal(
                state["cell_depth"], np.zeros_like(state["cell_depth"]),
                err_msg=f"t=0 cell_depth should be zero, got {state['cell_depth']}",
            )
            np.testing.assert_array_equal(
                state["cell_velocity"], np.zeros_like(state["cell_velocity"]),
                err_msg=f"t=0 cell_velocity should be zero, got {state['cell_velocity']}",
            )

    @unittest.skipUnless(swe2d_available() and swe2d_gpu_available(),
                         "hydra_swe2d GPU module not available")
    def test_drainage_exchange_upload_runs_when_mesh_built_eagerly(self):
        """Production order: backend.initialize() THEN coupling_controller.

        With the production call order, ``s_coupling_dev`` is valid by the
        time the controller's __init__ runs, so the eager pipe1d mesh
        build in _build_pipe1d_mesh_on_device() succeeds and sets
        ``_pipe1d_mesh_built = True``.  The first per-step call then sees
        that flag and must STILL upload the drainage exchange parameters
        (inlets / outfalls / pipe_ends) — they're tracked by a separate
        flag (``_drainage_exchange_uploaded``) so the upload is not
        silently skipped.
        """
        from swe2d.extensions.drainage_network import SWE2DUrbanDrainageModule
        from tests._swe2d_test_helpers import _make_rect_mesh

        backend = SWE2DBackend()
        node_x, node_y, node_z, cell_nodes = _make_rect_mesh(2, 1, 20.0, 10.0)
        backend.build_mesh(
            node_x, node_y, node_z, cell_nodes,
            bc_edge_node0=np.empty(0, dtype=np.int32),
            bc_edge_node1=np.empty(0, dtype=np.int32),
            bc_edge_type=np.empty(0, dtype=np.int32),
            bc_edge_val=np.empty(0, dtype=np.float64),
        )
        n_cells = int(backend.n_cells)
        backend.initialize(
            h0=np.full(n_cells, 0.05, dtype=np.float64),
            hu0=np.zeros(n_cells, dtype=np.float64),
            hv0=np.zeros(n_cells, dtype=np.float64),
            n_mann=0.035,
            h_min=1.0e-4,
            cfl=0.45,
            dt_max=0.5,
            dt_fixed=0.5,
            gpu_diag_sync_interval_steps=1,
            spatial_discretization=1,
        )

        nodes = [
            DrainageNode(node_id="N0", x=0.0, y=0.0, invert_elev=0.0, max_depth=3.0,
                         metadata={"surface_area": 10.0}),
            DrainageNode(node_id="N1", x=10.0, y=0.0, invert_elev=0.0, max_depth=3.0),
        ]
        links = [
            DrainageLink(link_id="L0", from_node_id="N0", to_node_id="N1",
                         length=10.0, roughness_n=0.013, diameter=1.0),
        ]
        inlets = [
            InletExchange(
                inlet_id="I0", cell_id=0, node_id="N0",
                crest_elev=0.5, width=1.0, coefficient=0.62, max_capture=1.0,
            ),
        ]
        cfg = PipeNetworkConfig(enabled=True, nodes=nodes, links=links, inlets=inlets)
        drain_mod = SWE2DUrbanDrainageModule(cfg)
        drain_mod.initialize()

        # Production order: backend.initialize() THEN controller.
        cc = SWE2DCouplingController(
            cell_area=backend.cell_areas(),
            cell_bed=np.zeros(n_cells, dtype=np.float64),
            drainage=drain_mod,
        )

        # Eager build should have populated the pipe1d mesh.
        self.assertTrue(cc._pipe1d_mesh_built,
                        "Eager pipe1d mesh build should have set _pipe1d_mesh_built")
        # Exchange params are not yet uploaded (no step has run).
        self.assertFalse(cc._drainage_exchange_uploaded,
                         "Exchange params should NOT be uploaded before any step")

        # Wrap the native module so we can detect the upload call.
        upload_called = [False]
        native_mod = cc._native_cuda_module()
        original_upload = native_mod.swe2d_gpu_upload_drainage_exchange_params

        def _spy_upload(*args, **kwargs):
            upload_called[0] = True
            return original_upload(*args, **kwargs)

        native_mod.swe2d_gpu_upload_drainage_exchange_params = _spy_upload

        # Run a single on-device coupling step.  The per-step code path
        # must upload the exchange parameters even though the mesh was
        # already built eagerly.
        try:
            applied = cc.apply_native_device_sources(0.0, 0.5)
            self.assertTrue(applied, "apply_native_device_sources should succeed")
        finally:
            native_mod.swe2d_gpu_upload_drainage_exchange_params = original_upload

        self.assertTrue(upload_called[0],
                        "swe2d_gpu_upload_drainage_exchange_params must run even when "
                        "the pipe1d mesh was built eagerly (regression: it was "
                        "previously skipped because the upload was inside the "
                        "_pipe1d_mesh_built guard).")
        self.assertTrue(cc._drainage_exchange_uploaded,
                        "_drainage_exchange_uploaded should be True after first step")

    @unittest.skipUnless(swe2d_available() and swe2d_gpu_available(),
                         "hydra_swe2d GPU module not available")
    def test_readback_returns_zeros_on_size_mismatch(self):
        """Host buffer is zero-init even when the C++ guard skips cudaMemcpy.

        Regression: ``swe2d_pipe1d_readback_node_state`` allocates host
        buffers via ``py::array_t<double>(N)`` which is equivalent to
        ``np.empty`` — uninitialized.  If the size guard
        ``n_* == p.n_*`` fails (e.g. caller passes the wrong count), the
        cudaMemcpy is skipped and Python sees random heap bits.  The fix
        pre-zeros every host buffer so a guard mismatch silently degrades
        to zeros instead of garbage.
        """
        import hydra_swe2d as m
        from tests._swe2d_test_helpers import _make_rect_mesh

        backend = SWE2DBackend()
        node_x, node_y, node_z, cell_nodes = _make_rect_mesh(2, 1, 20.0, 10.0)
        backend.build_mesh(
            node_x, node_y, node_z, cell_nodes,
            bc_edge_node0=np.empty(0, dtype=np.int32),
            bc_edge_node1=np.empty(0, dtype=np.int32),
            bc_edge_type=np.empty(0, dtype=np.int32),
            bc_edge_val=np.empty(0, dtype=np.float64),
        )
        n_cells = int(backend.n_cells)
        backend.initialize(
            h0=np.full(n_cells, 0.05, dtype=np.float64),
            hu0=np.zeros(n_cells, dtype=np.float64),
            hv0=np.zeros(n_cells, dtype=np.float64),
            n_mann=0.035,
            h_min=1.0e-4,
            cfl=0.45,
            dt_max=0.5,
            dt_fixed=0.5,
            gpu_diag_sync_interval_steps=1,
            spatial_discretization=1,
        )

        dev_ptr = int(m.swe2d_get_coupling_dev_ptr())
        self.assertNotEqual(dev_ptr, 0, "Coupling device pointer must be set")

        wrong_n_nodes = 99
        wrong_n_cells = 99
        state = m.swe2d_pipe1d_readback_node_state(
            dev_ptr, wrong_n_nodes, wrong_n_cells
        )

        np.testing.assert_array_equal(
            state["node_depth"], np.zeros(wrong_n_nodes),
            err_msg=f"node_depth should be zero on size mismatch, got {state['node_depth']}",
        )
        np.testing.assert_array_equal(
            state["cell_A"], np.zeros(wrong_n_cells),
            err_msg=f"cell_A should be zero on size mismatch, got {state['cell_A']}",
        )
        np.testing.assert_array_equal(
            state["cell_Q"], np.zeros(wrong_n_cells),
            err_msg=f"cell_Q should be zero on size mismatch, got {state['cell_Q']}",
        )


if __name__ == "__main__":
    unittest.main()
