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

    def _require_cuda_kernel(self):
        if self.mod is None:
            self.skipTest("hydra_swe2d module not available")
        if not hasattr(self.mod, "swe2d_gpu_compute_coupling_sources"):
            self.skipTest("swe2d_gpu_compute_coupling_sources not available")
        if not bool(self.mod.swe2d_gpu_available()):
            self.skipTest("CUDA GPU not available for kernel test")

    def test_coupling_source_kernel_mass_balance(self):
        self._require_cuda_kernel()

        cell_area = np.asarray([5.0, 5.0, 10.0], dtype=np.float64)

        inlet_cell = np.asarray([0, 2, -1], dtype=np.int32)
        inlet_flow = np.asarray([1.0, 0.5, 99.0], dtype=np.float64)

        structure_up = np.asarray([0, 1], dtype=np.int32)
        structure_dn = np.asarray([1, 2], dtype=np.int32)
        structure_flow = np.asarray([0.4, -0.2], dtype=np.float64)

        src = self.mod.swe2d_gpu_compute_coupling_sources(
            cell_area,
            inlet_cell,
            inlet_flow,
            structure_up,
            structure_dn,
            structure_flow,
        )

        self.assertEqual(src.shape[0], 3)

        expected = np.asarray([-0.28, 0.12, -0.07], dtype=np.float64)
        self.assertTrue(np.allclose(src, expected, atol=1.0e-12))

        # Structures conserve volume internally; net sink should equal inlet capture.
        net_q = float(np.dot(src, cell_area))
        self.assertAlmostEqual(net_q, -1.5, places=12)


if __name__ == "__main__":
    unittest.main()
