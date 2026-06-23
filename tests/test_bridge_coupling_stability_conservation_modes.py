import unittest

import numpy as np

from swe2d.extensions.extension_models import HydraulicStructure, HydraulicStructureConfig, StructureType
from swe2d.mesh.bridge_stacked_mesh import BridgeStackedPlan
from swe2d.runtime.coupling import SWE2DCouplingController


class _FakeStructureModule:
    def __init__(self, cfg: HydraulicStructureConfig):
        self.cfg = cfg

    def structure_flows(self, cell_wse: np.ndarray) -> np.ndarray:
        out = []
        for st in self.cfg.structures:
            if int(st.structure_type) == int(StructureType.BRIDGE):
                up = int(st.upstream_cell)
                dn = int(st.downstream_cell)
                # Bridge-like rating surrogate: discharge responds to head drop.
                head = max(float(cell_wse[up] - cell_wse[dn]), 0.0)
                out.append(22.0 * np.sqrt(max(head, 0.0)))
            else:
                out.append(0.0)
        return np.asarray(out, dtype=np.float64)

    def compute_structure_fluxes(self, dt_s: float, cell_wse: np.ndarray):
        q = self.structure_flows(cell_wse)
        return {"total_structure_flow": float(np.sum(q))}


class _FakeNativeModule:
    @staticmethod
    def swe2d_gpu_available():
        return True

    @staticmethod
    def swe2d_gpu_compute_coupling_sources(cell_area, inlet_cell, inlet_flow, struct_up, struct_dn, struct_q):
        # Keep generic structure source neutral for this bridge-focused test.
        return np.zeros_like(np.asarray(cell_area, dtype=np.float64))

    @staticmethod
    def swe2d_gpu_compute_bridge_coupling_sources(
        cell_area,
        bridge_up_cell,
        bridge_down_cell,
        bridge_flow_cms,
        bridge_loss_k_upstream,
        bridge_loss_k_downstream,
        bridge_opening_width_m,
        dt_s,
    ):
        area = np.asarray(cell_area, dtype=np.float64)
        out = np.zeros_like(area)
        up = int(np.asarray(bridge_up_cell, dtype=np.int32)[0])
        dn = int(np.asarray(bridge_down_cell, dtype=np.int32)[0])
        q = float(np.asarray(bridge_flow_cms, dtype=np.float64)[0])
        out[up] = -q / max(float(area[up]), 1.0e-12)
        out[dn] = q / max(float(area[dn]), 1.0e-12)
        return out


def _build_controller(mode: str) -> SWE2DCouplingController:
    n_cells = 12
    area = np.asarray([45.0, 47.0, 50.0, 52.0, 53.0, 52.0, 51.0, 50.0, 48.0, 47.0, 46.0, 45.0], dtype=np.float64)
    bed = np.linspace(101.0, 99.8, n_cells, dtype=np.float64)
    cfg = HydraulicStructureConfig(
        enabled=True,
        structures=[
            HydraulicStructure(
                structure_id="bridge_main",
                structure_type=StructureType.BRIDGE,
                upstream_cell=2,
                downstream_cell=7,
                crest_elev=100.0,
                metadata={
                    "width": 12.0,
                    "height": 3.5,
                    "opening": 0.85,
                    "inlet_loss_k": 0.9,
                    "outlet_loss_k": 1.1,
                },
            )
        ],
    )
    structures = _FakeStructureModule(cfg)
    ctrl = SWE2DCouplingController(
        cell_area=area,
        cell_bed=bed,
        structures=structures,
        coupling_loop="cuda",
        bridge_cuda_coupling=True,
        bridge_stacked_coupling_mode=mode,
    )
    ctrl._native_cuda_module = lambda: _FakeNativeModule()
    ctrl.bridge_stacked_plans = [
        BridgeStackedPlan(
            structure_id="bridge_main",
            selected_cells=np.asarray([2, 3, 4, 5, 6, 7], dtype=np.int32),
            streamwise_m=np.asarray([0.0, 10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float64),
            transverse_m=np.asarray([-4.0, -2.0, -1.0, 1.0, 2.0, 4.0], dtype=np.float64),
            layer_bottom_m=np.asarray([0.0, 1.5, 2.8], dtype=np.float64),
            layer_top_m=np.asarray([1.5, 2.8, 4.5], dtype=np.float64),
            layer_role=np.asarray([0, 0, 2], dtype=np.int32),
            opening_fraction=np.asarray([1.0, 0.85, 0.7, 0.6, 0.8, 1.0], dtype=np.float64),
            effective_opening_width_m=9.5,
        )
    ]
    return ctrl


class TestBridgeCouplingStabilityConservationModes(unittest.TestCase):
    def _run_transient_case(self, mode: str):
        ctrl = _build_controller(mode)
        area = np.asarray(ctrl.cell_area, dtype=np.float64)
        max_abs_src = 0.0
        nonzero_counts = []

        for k in range(90):
            t = float(k) * 5.0
            h = np.full(area.size, 2.1, dtype=np.float64)
            # Hydrograph-like transient at bridge approach/exit cells.
            h[2] = 2.6 + 0.35 * np.sin(2.0 * np.pi * t / 900.0)
            h[7] = 1.95 + 0.20 * np.sin(2.0 * np.pi * (t - 120.0) / 900.0)
            src = ctrl.compute_source_rates(
                t_s=t,
                dt_s=5.0,
                h=h,
                hu=np.zeros_like(h),
                hv=np.zeros_like(h),
            )
            self.assertTrue(np.all(np.isfinite(src)))
            q_net = float(np.sum(src * area))
            # Net bridge source should remain conservative.
            self.assertAlmostEqual(q_net, 0.0, places=10)
            max_abs_src = max(max_abs_src, float(np.max(np.abs(src))))
            nonzero_counts.append(int(np.count_nonzero(np.abs(src) > 1.0e-12)))

        self.assertLess(max_abs_src, 2.0)
        return nonzero_counts

    def test_legacy_mode_stable_and_conservative(self):
        nonzero_counts = self._run_transient_case("legacy_scalar")
        # Legacy helper injects/extracts on the bridge endpoint pair only.
        self.assertTrue(all(c <= 2 for c in nonzero_counts))

    def test_phase3_mode_stable_conservative_and_spatially_distributed(self):
        nonzero_counts = self._run_transient_case("phase3_spatial")
        # Phase 3 redistribution should activate multiple corridor cells.
        self.assertGreater(max(nonzero_counts), 2)


if __name__ == "__main__":
    unittest.main()
