import os
import sys
import unittest

import numpy as np

# Ensure workspace root is importable when running this file directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from swe2d.extensions.extension_models import TemporalScheme
from swe2d.runtime.backend import SWE2DBackend


class TestSWE2DTinyModeDispatch(unittest.TestCase):
    def _build_backend(self) -> SWE2DBackend:
        b = SWE2DBackend(use_gpu=True)
        node_x = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        node_y = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        node_z = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        cell_nodes = np.array([0, 1, 2], dtype=np.int32)
        b.build_mesh(node_x, node_y, node_z, cell_nodes)
        return b

    def test_tiny_fused_effective_for_single_stage_gpu_step(self):
        b = self._build_backend()
        b.initialize(
            np.array([0.1], dtype=np.float64),
            temporal_scheme=TemporalScheme.EULER_1ST,
            tiny_mode=2,
            tiny_cell_threshold=16,
            tiny_edge_threshold=32,
            tiny_wet_cell_threshold=16,
        )
        diag = b.step(-1.0)

        if not bool(diag.get("gpu_active", False)):
            self.skipTest("GPU path not active in current environment")
        if "tiny_mode_requested" not in diag:
            self.skipTest("Native extension diagnostics do not include tiny-mode fields (rebuild required)")

        self.assertEqual(int(diag["tiny_mode_requested"]), 2)
        self.assertEqual(int(diag["tiny_mode_selected"]), 2)
        self.assertEqual(int(diag["tiny_mode_effective"]), 2)
        self.assertFalse(bool(diag["tiny_mode_fallback"]))

    def test_tiny_fused_falls_back_for_rk2_path(self):
        b = self._build_backend()
        b.initialize(
            np.array([0.1], dtype=np.float64),
            temporal_scheme=TemporalScheme.SSP_RK2,
            tiny_mode=2,
            tiny_cell_threshold=16,
            tiny_edge_threshold=32,
            tiny_wet_cell_threshold=16,
        )
        diag = b.step(-1.0)

        if not bool(diag.get("gpu_active", False)):
            self.skipTest("GPU path not active in current environment")
        if "tiny_mode_requested" not in diag:
            self.skipTest("Native extension diagnostics do not include tiny-mode fields (rebuild required)")

        self.assertEqual(int(diag["tiny_mode_requested"]), 2)
        self.assertEqual(int(diag["tiny_mode_selected"]), 2)
        self.assertEqual(int(diag["tiny_mode_effective"]), 0)
        self.assertTrue(bool(diag["tiny_mode_fallback"]))

    def test_tiny_persistent_effective_for_single_stage_gpu_step(self):
        b = self._build_backend()
        b.initialize(
            np.array([0.1], dtype=np.float64),
            temporal_scheme=TemporalScheme.EULER_1ST,
            tiny_mode=3,
            tiny_cell_threshold=16,
            tiny_edge_threshold=32,
            tiny_wet_cell_threshold=16,
            tiny_persistent_chunk_substeps=4,
        )
        diag = b.step(-1.0)

        if not bool(diag.get("gpu_active", False)):
            self.skipTest("GPU path not active in current environment")
        if "tiny_mode_requested" not in diag:
            self.skipTest("Native extension diagnostics do not include tiny-mode fields (rebuild required)")

        self.assertEqual(int(diag["tiny_mode_requested"]), 3)
        self.assertEqual(int(diag["tiny_mode_selected"]), 3)
        self.assertEqual(int(diag["tiny_mode_effective"]), 3)
        self.assertFalse(bool(diag["tiny_mode_fallback"]))

    def test_tiny_persistent_effective_for_rk2_path(self):
        b = self._build_backend()
        b.initialize(
            np.array([0.1], dtype=np.float64),
            temporal_scheme=TemporalScheme.SSP_RK2,
            tiny_mode=3,
            tiny_cell_threshold=16,
            tiny_edge_threshold=32,
            tiny_wet_cell_threshold=16,
            tiny_persistent_chunk_substeps=4,
        )
        diag = b.step(-1.0)

        if not bool(diag.get("gpu_active", False)):
            self.skipTest("GPU path not active in current environment")
        if "tiny_mode_requested" not in diag:
            self.skipTest("Native extension diagnostics do not include tiny-mode fields (rebuild required)")

        self.assertEqual(int(diag["tiny_mode_requested"]), 3)
        self.assertEqual(int(diag["tiny_mode_selected"]), 3)
        self.assertEqual(int(diag["tiny_mode_effective"]), 3)
        self.assertFalse(bool(diag["tiny_mode_fallback"]))

    def test_tiny_persistent_effective_for_rk4_path(self):
        b = self._build_backend()
        b.initialize(
            np.array([0.1], dtype=np.float64),
            temporal_scheme=TemporalScheme.CLASSIC_RK4,
            tiny_mode=3,
            tiny_cell_threshold=16,
            tiny_edge_threshold=32,
            tiny_wet_cell_threshold=16,
            tiny_persistent_chunk_substeps=4,
        )
        diag = b.step(-1.0)

        if not bool(diag.get("gpu_active", False)):
            self.skipTest("GPU path not active in current environment")
        if "tiny_mode_requested" not in diag:
            self.skipTest("Native extension diagnostics do not include tiny-mode fields (rebuild required)")

        self.assertEqual(int(diag["tiny_mode_requested"]), 3)
        self.assertEqual(int(diag["tiny_mode_selected"]), 3)
        self.assertEqual(int(diag["tiny_mode_effective"]), 3)
        self.assertFalse(bool(diag["tiny_mode_fallback"]))

    def test_tiny_persistent_effective_for_rk4_graph_path(self):
        b = self._build_backend()
        b.initialize(
            np.array([0.1], dtype=np.float64),
            temporal_scheme=TemporalScheme.GRAPH_SAFE_RK4,
            tiny_mode=3,
            tiny_cell_threshold=16,
            tiny_edge_threshold=32,
            tiny_wet_cell_threshold=16,
            tiny_persistent_chunk_substeps=4,
        )
        diag = b.step(-1.0)

        if not bool(diag.get("gpu_active", False)):
            self.skipTest("GPU path not active in current environment")
        if "tiny_mode_requested" not in diag:
            self.skipTest("Native extension diagnostics do not include tiny-mode fields (rebuild required)")

        self.assertEqual(int(diag["tiny_mode_requested"]), 3)
        self.assertEqual(int(diag["tiny_mode_selected"]), 3)
        self.assertEqual(int(diag["tiny_mode_effective"]), 3)
        self.assertFalse(bool(diag["tiny_mode_fallback"]))

    def test_tiny_persistent_effective_for_rk5_graph_path(self):
        b = self._build_backend()
        b.initialize(
            np.array([0.1], dtype=np.float64),
            temporal_scheme=TemporalScheme.GRAPH_SAFE_RK5,
            tiny_mode=3,
            tiny_cell_threshold=16,
            tiny_edge_threshold=32,
            tiny_wet_cell_threshold=16,
            tiny_persistent_chunk_substeps=4,
        )
        diag = b.step(-1.0)

        if not bool(diag.get("gpu_active", False)):
            self.skipTest("GPU path not active in current environment")
        if "tiny_mode_requested" not in diag:
            self.skipTest("Native extension diagnostics do not include tiny-mode fields (rebuild required)")

        self.assertEqual(int(diag["tiny_mode_requested"]), 3)
        self.assertEqual(int(diag["tiny_mode_selected"]), 3)
        self.assertEqual(int(diag["tiny_mode_effective"]), 3)
        self.assertFalse(bool(diag["tiny_mode_fallback"]))


if __name__ == "__main__":
    unittest.main()
