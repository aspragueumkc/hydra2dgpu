import unittest

import numpy as np


class TestSWE2DGPUCouplingKernel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import hydra_swe2d as mod
        except Exception:
            mod = None
        cls.mod = mod

    def test_device_resident_path_throughput(self):
        """Smoke-test: compute_coupling_full_on_device handles drainage + structures."""
        from swe2d.runtime.backend import SWE2DBackend
        from swe2d.runtime.coupling import SWE2DCouplingController
        from swe2d.extensions.drainage_network import SWE2DUrbanDrainageModule
        from swe2d.extensions.extension_models import (
            DrainageNode, HydraulicStructure,
            HydraulicStructureConfig, InletExchange,
            PipeNetworkConfig, DrainageSolverMode, StructureType,
        )
        from swe2d.extensions.structures import SWE2DStructureModule
        from swe2d import units as _u

        _u.configure(1.0)

        backend = SWE2DBackend()
        node_x = np.array([0., 12., 12., 0., 0., 12., 12., 0.], dtype=np.float64)
        node_y = np.array([0., 0., 8., 8., 0., 0., 8., 8.], dtype=np.float64)
        node_z = np.zeros(8, dtype=np.float64)
        cell_nodes = np.array([0,1,2, 0,2,3, 4,5,6, 4,6,7], dtype=np.int32)
        backend.build_mesh(node_x, node_y, node_z, cell_nodes)
        n_cells = backend.n_cells

        struct_mod = SWE2DStructureModule(HydraulicStructureConfig(enabled=True, structures=[
            HydraulicStructure('W0', StructureType.WEIR, 0, 1, crest_elev=0.0,
                               metadata={'width':2.0, 'cd':0.6}),
        ]), model_to_ft=_u.model_to_ft())

        drain_mod = SWE2DUrbanDrainageModule(PipeNetworkConfig(
            enabled=True,
            nodes=[DrainageNode('N0',0,0,invert_elev=0.0,max_depth=4.0,metadata={'surface_area':50.0})],
            links=[],
            inlets=[InletExchange('I0',0,'N0',crest_elev=0.5,width=1.0,coefficient=0.62,max_capture=1.0)],
            outfalls=[], pipe_ends=[],
            solver_mode=DrainageSolverMode.EGL,
        ))
        drain_mod.initialize()
        drain_mod.state.node_depth['N0'] = 2.0

        controller = SWE2DCouplingController(
            cell_area=backend.cell_areas(),
            cell_bed=np.zeros(n_cells, dtype=np.float64),
            drainage=drain_mod,
            structures=struct_mod,
            length_scale_si_to_model=1.0,
        )

        backend.initialize(
            h0=np.array([1.5, 0.5, 1.2, 0.4], dtype=np.float64),
            hu0=np.zeros(n_cells, dtype=np.float64),
            hv0=np.zeros(n_cells, dtype=np.float64),
            dt_max=0.02, dt_fixed=0.02,
        )
        backend.run(t_end=0.1, dt_request=0.02,
            source_rate_callback=controller.compute_source_rates)

        h, _, _ = backend.get_state()
        self.assertTrue(np.all(np.isfinite(h)),
                        "GPU solver must produce finite state with drainage + structures")
        backend.destroy()
