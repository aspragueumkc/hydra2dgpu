import unittest
import numpy as np

from swe2d.extensions.extension_models import TemporalScheme, BedFrictionModel
from swe2d.runtime.backend import SWE2DBackend


class TestSWE2DIMEXSubcycling(unittest.TestCase):
    """Regression tests for the operator-split IMEX friction path
    (source_imex_split=True)."""

    def _build_backend(self) -> SWE2DBackend:
        b = SWE2DBackend()
        node_x = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        node_y = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        node_z = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        cell_nodes = np.array([0, 1, 2], dtype=np.int32)
        b.build_mesh(node_x, node_y, node_z, cell_nodes)
        return b

    def test_imex_split_runs(self):
        b = self._build_backend()
        h0 = np.array([1.0], dtype=np.float64)
        hu0 = np.array([1.0], dtype=np.float64)
        hv0 = np.array([0.5], dtype=np.float64)
        b.initialize(
            h0, hu0=hu0, hv0=hv0,
            n_mann=0.035,
            cfl=0.45,
            temporal_scheme=TemporalScheme.EULER_1ST,
            bed_friction_model=BedFrictionModel.MANNING,
            source_imex_split=True,
        )
        diag = b.step(-1.0)
        if not bool(diag.get("gpu_active", False)):
            self.skipTest("GPU path not active in current environment")
        self.assertGreaterEqual(diag.get("dt", -1.0), 0.0)

    def test_imex_split_no_nans(self):
        b = self._build_backend()
        h0 = np.array([1.0], dtype=np.float64)
        hu0 = np.array([2.0], dtype=np.float64)
        hv0 = np.array([1.0], dtype=np.float64)
        b.initialize(
            h0, hu0=hu0, hv0=hv0,
            n_mann=0.050,
            cfl=0.45,
            temporal_scheme=TemporalScheme.EULER_1ST,
            bed_friction_model=BedFrictionModel.MANNING,
            source_imex_split=True,
        )
        diag = b.step(-1.0)
        if not bool(diag.get("gpu_active", False)):
            self.skipTest("GPU path not active in current environment")
        h, hu, hv = b.get_state()
        self.assertTrue(np.all(np.isfinite(h)), "h has NaN/inf")
        self.assertTrue(np.all(np.isfinite(hu)), "hu has NaN/inf")
        self.assertTrue(np.all(np.isfinite(hv)), "hv has NaN/inf")

    def test_imex_split_without_momentum_init_runs(self):
        b = self._build_backend()
        h0 = np.array([1.0], dtype=np.float64)
        b.initialize(
            h0,
            n_mann=0.035,
            cfl=0.45,
            temporal_scheme=TemporalScheme.SSP_RK2,
            bed_friction_model=BedFrictionModel.MANNING,
            source_imex_split=True,
        )
        diag = b.step(-1.0)
        if not bool(diag.get("gpu_active", False)):
            self.skipTest("GPU path not active in current environment")
        self.assertGreaterEqual(diag.get("dt", -1.0), 0.0)


if __name__ == "__main__":
    unittest.main()
