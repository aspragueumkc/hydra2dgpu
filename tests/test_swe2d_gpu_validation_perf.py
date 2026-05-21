"""
Dedicated GPU validation + high-performance benchmark suite.

This suite is GPU-only by design and focuses on practical runtime health:
- positivity/finite-state checks
- diagnostic sanity (CFL and residual outputs)
- optional throughput benchmark

The benchmark test is off by default and can be enabled with:
    BACKWATER_RUN_GPU_PERF=1
"""

import os
import sys
import time
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _load_module():
    try:
        import hydra_swe2d
        return hydra_swe2d
    except ImportError:
        return None


def _gpu_available():
    mod = _load_module()
    if mod is None:
        return False
    try:
        return mod.swe2d_gpu_available()
    except Exception:
        return False


def _make_rect_mesh(nx, ny, lx, ly):
    xs = np.linspace(0.0, lx, nx + 1)
    ys = np.linspace(0.0, ly, ny + 1)
    xg, yg = np.meshgrid(xs, ys)
    node_x = xg.ravel().copy()
    node_y = yg.ravel().copy()
    node_z = np.zeros_like(node_x)
    cells = []
    stride = nx + 1
    for j in range(ny):
        for i in range(nx):
            n00 = j * stride + i
            n10 = j * stride + i + 1
            n01 = (j + 1) * stride + i
            n11 = (j + 1) * stride + i + 1
            cells.extend([n00, n10, n11])
            cells.extend([n00, n11, n01])
    return node_x, node_y, node_z, np.array(cells, dtype=np.int32)


@unittest.skipUnless(_load_module() is not None, "hydra_swe2d not built")
@unittest.skipUnless(_gpu_available(), "CUDA GPU not available")
class TestSWE2DGPUValidationPerf(unittest.TestCase):
    NX, NY = 80, 40
    LX, LY = 1600.0, 800.0
    N_STEPS = 120

    def setUp(self):
        self.mod = _load_module()
        node_x, node_y, node_z, cell_nodes = _make_rect_mesh(
            self.NX, self.NY, self.LX, self.LY
        )
        self.mesh = self.mod.swe2d_build_mesh(
            node_x,
            node_y,
            node_z,
            cell_nodes,
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float64),
        )
        n_cells = self.mod.swe2d_mesh_info(self.mesh)["n_cells"]
        rng = np.random.default_rng(7)
        self.h0 = rng.uniform(0.20, 1.80, n_cells)

    def _run_gpu(self, n_steps):
        solver = self.mod.swe2d_create_solver(
            self.mesh,
            self.h0.copy(),
            n_mann=0.030,
            cfl=0.45,
            dt_max=1.0,
            use_gpu=True,
        )
        diag = None
        for _ in range(n_steps):
            diag = self.mod.swe2d_step(solver, -1.0)
        h, hu, hv = self.mod.swe2d_get_state(solver)
        self.mod.swe2d_destroy(solver)
        return h, hu, hv, diag

    def test_gpu_runtime_diagnostics_sane(self):
        h, hu, hv, diag = self._run_gpu(self.N_STEPS)
        self.assertTrue(diag["gpu_active"], "Last step did not report GPU active")
        self.assertGreater(diag["dt"], 0.0)

        # GPU kernel currently returns -1 for some host reductions not yet
        # computed on device (wet_cells/max/min/mass). Accept either sentinel
        # mode or a fully-populated non-negative value.
        wet_cells = int(diag["wet_cells"])
        self.assertTrue(wet_cells == -1 or (0 <= wet_cells <= h.size))

        self.assertTrue(np.isfinite(diag["max_courant"]))
        self.assertGreaterEqual(diag["max_courant"], 0.0)
        self.assertTrue(np.isfinite(diag["max_depth_residual"]))
        self.assertGreaterEqual(diag["max_depth_residual"], 0.0)
        self.assertTrue(np.isfinite(diag["max_wse_elev_error"]))
        self.assertGreaterEqual(diag["max_wse_elev_error"], 0.0)

        self.assertTrue(np.all(np.isfinite(h)))
        self.assertTrue(np.all(np.isfinite(hu)))
        self.assertTrue(np.all(np.isfinite(hv)))
        self.assertGreaterEqual(float(np.min(h)), -1.0e-10)

    def test_gpu_godunov_rollout_runtime_sane(self):
        solver = self.mod.swe2d_create_solver(
            self.mesh,
            self.h0.copy(),
            n_mann=0.030,
            cfl=0.45,
            dt_max=1.0,
            temporal_order=2,
            spatial_scheme=0,
            godunov_mode=1,
            use_gpu=True,
        )
        diag = None
        for _ in range(80):
            diag = self.mod.swe2d_step(solver, -1.0)
        h, hu, hv = self.mod.swe2d_get_state(solver)
        self.mod.swe2d_destroy(solver)

        self.assertTrue(diag["gpu_active"], "Godunov rollout run did not stay on GPU")
        self.assertTrue(np.all(np.isfinite(h)))
        self.assertTrue(np.all(np.isfinite(hu)))
        self.assertTrue(np.all(np.isfinite(hv)))
        self.assertGreaterEqual(float(np.min(h)), -1.0e-10)
        self.assertLess(float(np.max(h)), 1.0e6)

    @unittest.skipUnless(
        os.environ.get("BACKWATER_RUN_GPU_PERF", "0") == "1",
        "Set BACKWATER_RUN_GPU_PERF=1 to enable throughput benchmark",
    )
    def test_gpu_step_throughput_benchmark(self):
        n_steps = 400
        start = time.perf_counter()
        _, _, _, diag = self._run_gpu(n_steps)
        elapsed = time.perf_counter() - start
        steps_per_sec = n_steps / max(elapsed, 1.0e-12)

        self.assertTrue(diag["gpu_active"], "Benchmark run did not stay on GPU")
        self.assertGreater(steps_per_sec, 5.0, f"GPU throughput too low: {steps_per_sec:.2f} steps/s")


if __name__ == "__main__":
    unittest.main()
