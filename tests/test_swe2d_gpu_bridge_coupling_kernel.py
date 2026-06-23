import unittest

import numpy as np


class TestSWE2DGPUBridgeCouplingKernel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import hydra_swe2d as mod
        except Exception:
            mod = None
        cls.mod = mod

    def _require_cuda_kernel(self):
        if self.mod is None:
            self.skipTest("hydra_swe2d module not available")
        if not hasattr(self.mod, "swe2d_gpu_compute_bridge_coupling_sources"):
            self.skipTest("swe2d_gpu_compute_bridge_coupling_sources not available")
        if not bool(self.mod.swe2d_gpu_available()):
            self.skipTest("CUDA GPU not available for bridge kernel test")

    def test_bridge_source_kernel_attentuates_with_loss(self):
        self._require_cuda_kernel()

        cell_area = np.asarray([5.0, 5.0], dtype=np.float64)
        bridge_up = np.asarray([0], dtype=np.int32)
        bridge_dn = np.asarray([1], dtype=np.int32)
        bridge_flow = np.asarray([2.0], dtype=np.float64)
        low_k = np.asarray([0.0], dtype=np.float64)
        high_k = np.asarray([2.0], dtype=np.float64)

        base = self.mod.swe2d_gpu_compute_bridge_coupling_sources(
            cell_area,
            bridge_up,
            bridge_dn,
            bridge_flow,
            low_k,
            low_k,
            2.0,
            0.05,
        )
        lossy = self.mod.swe2d_gpu_compute_bridge_coupling_sources(
            cell_area,
            bridge_up,
            bridge_dn,
            bridge_flow,
            high_k,
            high_k,
            2.0,
            0.05,
        )

        self.assertEqual(base.shape[0], 2)
        self.assertEqual(lossy.shape[0], 2)
        self.assertLess(np.abs(lossy[0]), np.abs(base[0]))
        self.assertLess(np.abs(lossy[1]), np.abs(base[1]))
        self.assertAlmostEqual(float(np.dot(base, cell_area)), 0.0, places=12)
        self.assertAlmostEqual(float(np.dot(lossy, cell_area)), 0.0, places=12)


if __name__ == "__main__":
    unittest.main()