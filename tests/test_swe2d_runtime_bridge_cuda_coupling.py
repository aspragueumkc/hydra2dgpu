import unittest

import numpy as np

from swe2d.extensions.extension_models import (
    HydraulicStructure,
    HydraulicStructureConfig,
    StructureType,
)
from swe2d.extensions.structures import SWE2DStructureModule
from swe2d.mesh.bridge_stacked_mesh import BridgeStackedPlan
from swe2d.runtime.bridge_stacked_runtime import bridge_stacked_source_scale
from swe2d.runtime.coupling import SWE2DCouplingController


class TestSWE2DRuntimeBridgeCudaCoupling(unittest.TestCase):
    def test_bridge_cuda_feature_flag_routes_bridge_structures_to_bridge_helper(self):
        bridge_cfg = HydraulicStructureConfig(
            enabled=True,
            structures=[
                HydraulicStructure(
                    structure_id="B0",
                    structure_type=StructureType.BRIDGE,
                    upstream_cell=0,
                    downstream_cell=1,
                    crest_elev=0.0,
                    metadata={
                        "width": 2.0,
                        "height": 1.0,
                        "opening": 1.0,
                        "coeff": 0.6,
                        "inlet_loss_k": 0.8,
                        "outlet_loss_k": 1.1,
                    },
                ),
                HydraulicStructure(
                    structure_id="W0",
                    structure_type=StructureType.WEIR,
                    upstream_cell=1,
                    downstream_cell=0,
                    crest_elev=0.0,
                    metadata={"width": 1.0, "coeff": 1.7},
                ),
            ],
        )
        structures = SWE2DStructureModule(bridge_cfg)
        controller = SWE2DCouplingController(
            cell_area=[5.0, 5.0],
            cell_bed=[0.0, 0.0],
            structures=structures,
            coupling_loop="cuda",
            bridge_cuda_coupling=True,
            bridge_stacked_coupling_mode="legacy_scalar",
        )

        calls = {"generic": None, "bridge": None}

        class _FakeNativeModule:
            @staticmethod
            def swe2d_gpu_available():
                return True

            @staticmethod
            def swe2d_gpu_compute_coupling_sources(cell_area, inlet_cell, inlet_flow, struct_up, struct_dn, struct_q):
                calls["generic"] = {
                    "struct_up": np.asarray(struct_up, dtype=np.int32),
                    "struct_dn": np.asarray(struct_dn, dtype=np.int32),
                    "struct_q": np.asarray(struct_q, dtype=np.float64),
                }
                return np.zeros_like(np.asarray(cell_area, dtype=np.float64))

            @staticmethod
            def swe2d_gpu_compute_bridge_coupling_sources(cell_area, bridge_up_cell, bridge_down_cell, bridge_flow_cms, bridge_loss_k_upstream, bridge_loss_k_downstream, bridge_opening_width_m, dt_s):
                calls["bridge"] = {
                    "bridge_up_cell": np.asarray(bridge_up_cell, dtype=np.int32),
                    "bridge_down_cell": np.asarray(bridge_down_cell, dtype=np.int32),
                    "bridge_flow_cms": np.asarray(bridge_flow_cms, dtype=np.float64),
                    "bridge_loss_k_upstream": np.asarray(bridge_loss_k_upstream, dtype=np.float64),
                    "bridge_loss_k_downstream": np.asarray(bridge_loss_k_downstream, dtype=np.float64),
                    "bridge_opening_width_m": float(bridge_opening_width_m),
                    "dt_s": float(dt_s),
                }
                out = np.zeros_like(np.asarray(cell_area, dtype=np.float64))
                out[0] = -0.1 / float(cell_area[0])
                out[1] = 0.1 / float(cell_area[1])
                return out

        controller._native_cuda_module = lambda: _FakeNativeModule()
        controller.bridge_stacked_plans = [
            BridgeStackedPlan(
                structure_id="B0",
                selected_cells=np.asarray([0, 1], dtype=np.int32),
                streamwise_m=np.asarray([0.0, 1.0], dtype=np.float64),
                transverse_m=np.asarray([0.0, 0.0], dtype=np.float64),
                layer_bottom_m=np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
                layer_top_m=np.asarray([0.5, 1.0, 1.5], dtype=np.float64),
                layer_role=np.asarray([0, 0, 2], dtype=np.int32),
                opening_fraction=np.asarray([1.0, 0.5], dtype=np.float64),
                effective_opening_width_m=1.5,
            )
        ]

        src = controller.compute_source_rates(
            t_s=0.0,
            dt_s=0.05,
            h=np.asarray([1.0, 0.5], dtype=np.float64),
            hu=np.zeros(2, dtype=np.float64),
            hv=np.zeros(2, dtype=np.float64),
        )

        self.assertIsNotNone(calls["generic"])
        self.assertIsNotNone(calls["bridge"])
        self.assertEqual(calls["generic"]["struct_up"].size, 1)
        self.assertEqual(calls["generic"]["struct_dn"].size, 1)
        self.assertEqual(calls["bridge"]["bridge_up_cell"].tolist(), [0])
        self.assertEqual(calls["bridge"]["bridge_down_cell"].tolist(), [1])
        scale = bridge_stacked_source_scale(controller.bridge_stacked_plans[0])
        self.assertAlmostEqual(float(src[0]), (-0.1 / 5.0) * scale, places=12)
        self.assertAlmostEqual(float(src[1]), (0.1 / 5.0) * scale, places=12)
        self.assertIn("bridges", controller.last_diag.component_sums_mps)


if __name__ == "__main__":
    unittest.main()