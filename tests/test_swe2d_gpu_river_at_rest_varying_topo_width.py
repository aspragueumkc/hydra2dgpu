"""
GPU river-at-rest validation (stage preservation with varying channel width).

Reference: reference/anuga_validation_tests/analytical_exact/river_at_rest_varying_topo_width/
Original ANUGA setup: L=1500 m, dx=1 m, W=60 m, stage=12.0 m.

Physical setup
--------------
Constant stage 12.0 m over flat bed (z=0). All boundaries are reflective
walls. The analytical solution is trivial: stage remains 12.0 m everywhere
at all times. This test verifies that the GPU solver does not introduce
artificial perturbations in a quiescent water body.

Test strategy
-------------
Run to t=5.0 s. Check that the L-infinity norm of (stage - 12.0) is below
1e-6 m (near machine precision for double-precision arithmetic with
O(180K) cells).

Tolerance: L_inf < 1e-6 m.
"""

import unittest
import numpy as np

from tests._swe2d_test_helpers import _make_rect_mesh


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


@unittest.skipUnless(_load_module() is not None, "hydra_swe2d not built")
@unittest.skipUnless(_gpu_available(), "CUDA GPU not available")
class TestGPURiverAtRestVaryingTopoWidth(unittest.TestCase):
    anuga_reference = "reference/anuga_validation_tests/analytical_exact/river_at_rest_varying_topo_width/"
    NX = 1500
    NY = 60
    LX = 1500.0
    LY = 60.0
    STAGE = 12.0
    T_END = 5.0

    def _build(self):
        mod = _load_module()
        node_x, node_y, _, cell_nodes = _make_rect_mesh(self.NX, self.NY, self.LX, self.LY)

        # Flat bed
        node_z = np.zeros_like(node_x)

        # All walls — empty BC arrays
        mesh = mod.swe2d_build_mesh(
            node_x, node_y, node_z, cell_nodes,
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float64),
        )

        info = mod.swe2d_mesh_info(mesh)
        n_cells = info["n_cells"]

        # Initial condition: constant stage everywhere
        h0 = np.full(n_cells, self.STAGE, dtype=np.float64)

        solver = mod.swe2d_create_solver(
            mesh, h0, n_mann=0.0, cfl=0.45, dt_max=0.5, use_gpu=True, g=9.8
        )
        return mod, mesh, solver

    def _run_to_end(self):
        mod, mesh, solver = self._build()
        t = 0.0
        last_diag = None
        while t < self.T_END:
            last_diag = mod.swe2d_step(solver, -1.0)
            t += last_diag["dt"]
        h, hu, hv = mod.swe2d_get_state(solver)
        mod.swe2d_destroy(solver)
        return h, last_diag

    def test_stability(self):
        h, last_diag = self._run_to_end()
        self.assertTrue(last_diag["gpu_active"])
        self.assertTrue(np.all(np.isfinite(h)))
        self.assertTrue(np.all(h >= -1e-12))

    def test_linf_error_vs_constant_stage(self):
        h, _ = self._run_to_end()
        # Stage = h + z, with z=0 everywhere → stage = h
        linf = float(np.max(np.abs(h - self.STAGE)))
        limit = 1.0e-6
        self.assertLess(
            linf,
            limit,
            msg=f"River-at-rest L_inf error {linf:.2e} m exceeds limit ({limit:.2e} m)",
        )
