"""
test_swe2d_gpu.py
Legacy CPU vs GPU coarse parity smoke test.

SWE2D development is GPU-first.  This file is retained only as a broad
compatibility envelope so catastrophic drifts are still visible when someone
touches the fallback CPU path.

Strict bitwise or near-bitwise CPU/GPU parity is not a project objective.
Do not treat failures here as a reason to avoid GPU-focused numerical or
performance improvements unless the task explicitly requires CPU fallback work.

Skipped automatically when:
    - hydra_swe2d is not built, OR
    - swe2d_gpu_available() returns False
"""

import unittest
import sys
import os
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


def _make_rect_mesh(nx, ny, Lx, Ly):
    xs = np.linspace(0.0, Lx, nx + 1)
    ys = np.linspace(0.0, Ly, ny + 1)
    Xg, Yg = np.meshgrid(xs, ys)
    node_x = Xg.ravel().copy()
    node_y = Yg.ravel().copy()
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


def _run_solver(mod, mesh, h0, n_steps, use_gpu, n_mann_cell=None):
    solver = mod.swe2d_create_solver(
        mesh, h0.copy(),
        n_mann_cell=n_mann_cell,
        n_mann=0.030,
        cfl=0.45, dt_max=2.0,
        use_gpu=use_gpu)
    diag = None
    for _ in range(n_steps):
        diag = mod.swe2d_step(solver, -1.0)
    h, hu, hv = mod.swe2d_get_state(solver)
    mod.swe2d_destroy(solver)
    return h, hu, hv, diag


@unittest.skipUnless(_load_module() is not None, "hydra_swe2d not built")
@unittest.skipUnless(_gpu_available(), "CUDA GPU not available")
class TestGPUCPUParity(unittest.TestCase):
    """
    GPU vs CPU coarse smoke parity. This is not a strict numerical-equivalence
    test and only guards against large regressions.
    """

    NX, NY = 20, 10
    LX, LY = 200.0, 50.0
    N_STEPS = 50
    TOLERANCE = 5e-1

    def setUp(self):
        self.mod = _load_module()
        node_x, node_y, node_z, cell_nodes = _make_rect_mesh(
            self.NX, self.NY, self.LX, self.LY)

        self.mesh = self.mod.swe2d_build_mesh(
            node_x, node_y, node_z, cell_nodes,
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float64))

        n_cells = self.mod.swe2d_mesh_info(self.mesh)["n_cells"]
        rng = np.random.default_rng(42)
        self.h0 = rng.uniform(0.5, 2.0, n_cells)

    def test_h_parity(self):
        h_cpu, _, _, _ = _run_solver(
            self.mod, self.mesh, self.h0, self.N_STEPS, use_gpu=False)
        h_gpu, _, _, _ = _run_solver(
            self.mod, self.mesh, self.h0, self.N_STEPS, use_gpu=True)
        diff = np.max(np.abs(h_cpu - h_gpu))
        self.assertLess(diff, self.TOLERANCE,
            msg=f"GPU/CPU h coarse parity: max|diff| = {diff:.2e} (limit {self.TOLERANCE:.1e})")

    def test_hu_parity(self):
        _, hu_cpu, _, _ = _run_solver(
            self.mod, self.mesh, self.h0, self.N_STEPS, use_gpu=False)
        _, hu_gpu, _, _ = _run_solver(
            self.mod, self.mesh, self.h0, self.N_STEPS, use_gpu=True)
        diff = np.max(np.abs(hu_cpu - hu_gpu))
        self.assertLess(diff, self.TOLERANCE,
            msg=f"GPU/CPU hu coarse parity: max|diff| = {diff:.2e} (limit {self.TOLERANCE:.1e})")

    def test_hv_parity(self):
        _, _, hv_cpu, _ = _run_solver(
            self.mod, self.mesh, self.h0, self.N_STEPS, use_gpu=False)
        _, _, hv_gpu, _ = _run_solver(
            self.mod, self.mesh, self.h0, self.N_STEPS, use_gpu=True)
        diff = np.max(np.abs(hv_cpu - hv_gpu))
        self.assertLess(diff, self.TOLERANCE,
            msg=f"GPU/CPU hv coarse parity: max|diff| = {diff:.2e} (limit {self.TOLERANCE:.1e})")

    def test_gpu_diagnostic_flag(self):
        _, _, _, diag = _run_solver(
            self.mod, self.mesh, self.h0, self.N_STEPS, use_gpu=True)
        self.assertTrue(diag["gpu_active"],
            "GPU was requested but last step ran on CPU")

    def test_spatial_manning_parity(self):
        n_cells = self.mod.swe2d_mesh_info(self.mesh)["n_cells"]
        n_mann_cell = np.linspace(0.02, 0.08, n_cells, dtype=np.float64)

        h_cpu, hu_cpu, hv_cpu, _ = _run_solver(
            self.mod, self.mesh, self.h0, self.N_STEPS, use_gpu=False,
            n_mann_cell=n_mann_cell)
        h_gpu, hu_gpu, hv_gpu, diag = _run_solver(
            self.mod, self.mesh, self.h0, self.N_STEPS, use_gpu=True,
            n_mann_cell=n_mann_cell)

        self.assertTrue(diag["gpu_active"],
            "GPU was requested with spatial Manning but last step ran on CPU")
        self.assertLess(np.max(np.abs(h_cpu - h_gpu)), self.TOLERANCE)
        self.assertLess(np.max(np.abs(hu_cpu - hu_gpu)), self.TOLERANCE)
        self.assertLess(np.max(np.abs(hv_cpu - hv_gpu)), self.TOLERANCE)


if __name__ == "__main__":
    unittest.main()
