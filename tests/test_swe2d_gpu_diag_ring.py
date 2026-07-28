"""Phase 4.1 — DiagRecord device ring buffer tests.

Requires a built ``hydra_swe2d`` with the Phase 4.1 binding
(``swe2d_gpu_push_diag`` + ``swe2d_gpu_read_latest_diag``) + CUDA GPU.
Auto-skipped otherwise via the conftest hooks in tests/conftest.py.
"""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np
import pytest


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BUILD_DIR = os.environ.get("HYDRA_BUILD_DIR") or os.path.join(_REPO_ROOT, "build")
for _p in (_REPO_ROOT, _BUILD_DIR):
    if _p not in sys.path and os.path.isdir(_p):
        sys.path.insert(0, _p)


def _load_module():
    try:
        import hydra_swe2d
        return hydra_swe2d
    except ImportError:
        return None


def _binding_present(mod):
    return mod is not None and all(
        hasattr(mod, a) for a in
        ("swe2d_gpu_push_diag", "swe2d_gpu_read_latest_diag",
         "swe2d_gpu_init_diag_ring", "swe2d_gpu_clear_diag")
    )


def _gpu_available():
    mod = _load_module()
    if mod is None or not _binding_present(mod):
        return False
    try:
        return mod.swe2d_gpu_available()
    except Exception:
        return False


from tests._swe2d_test_helpers import _make_rect_mesh  # noqa: E402


@pytest.mark.solver
@pytest.mark.gpu
@unittest.skipUnless(_load_module() is not None, "hydra_swe2d not built")
@unittest.skipUnless(_binding_present(_load_module()),
                     "Phase 4.1 diag ring bindings not in .so (rebuild C++)")
@unittest.skipUnless(_gpu_available(), "CUDA GPU not available")
class TestGPUDiagRing(unittest.TestCase):
    """Verify the DiagRecord ring buffer round-trips on device."""

    NX = 10
    NY = 4
    LX = 100.0
    LY = 40.0

    def _make_solver(self):
        mod = _load_module()
        node_x, node_y, node_z, cell_nodes = _make_rect_mesh(
            self.NX, self.NY, self.LX, self.LY,
        )
        mesh = mod.swe2d_build_mesh(
            node_x, node_y, node_z, cell_nodes,
            np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float64),
        )
        info = mod.swe2d_mesh_info(mesh)
        n_cells = info["n_cells"]
        # Force diag sync every step so swe2d_step auto-pushes on every call.
        solver = mod.swe2d_create_solver(
            mesh, np.full(n_cells, 1.0, dtype=np.float64),
            n_mann=0.0, cfl=0.45, dt_max=0.5, use_gpu=True,
            gpu_diag_sync_interval_steps=1,
        )
        return mod, solver

    def test_push_and_read_latest(self):
        """Push a DiagRecord manually; read it back; verify fields round-trip."""
        mod, solver = self._make_solver()
        try:
            mod.swe2d_gpu_init_diag_ring(64)
            mod.swe2d_gpu_clear_diag()

            mod.swe2d_gpu_push_diag(
                t_s=12.5, dt_used=0.045, gpu_active=1, wet_cells=80,
                max_courant=0.42, max_wse_error=0.0001, mass_total=1.23e6,
            )
            rec = mod.swe2d_gpu_read_latest_diag()
            self.assertEqual(rec["t_s"], 12.5)
            self.assertEqual(rec["dt_used"], 0.045)
            self.assertEqual(rec["gpu_active"], 1)
            self.assertEqual(rec["wet_cells"], 80)
            self.assertEqual(rec["max_courant"], 0.42)
            self.assertEqual(rec["max_wse_error"], 0.0001)
            self.assertEqual(rec["mass_total"], 1.23e6)
        finally:
            mod.swe2d_gpu_shutdown_diag_ring()
            mod.swe2d_destroy(solver)

    def test_multiple_pushes_latest_is_last(self):
        """Push 3 records; latest is the third."""
        mod, solver = self._make_solver()
        try:
            mod.swe2d_gpu_init_diag_ring(64)
            mod.swe2d_gpu_clear_diag()
            for i, t in enumerate([1.0, 2.0, 3.0]):
                mod.swe2d_gpu_push_diag(
                    t_s=t, dt_used=0.01, gpu_active=1, wet_cells=i,
                    max_courant=0.1 * i, max_wse_error=0.0, mass_total=1.0,
                )
            rec = mod.swe2d_gpu_read_latest_diag()
            self.assertEqual(rec["t_s"], 3.0)
            self.assertEqual(rec["wet_cells"], 2)
        finally:
            mod.swe2d_gpu_shutdown_diag_ring()
            mod.swe2d_destroy(solver)

    def test_clear_resets_ring(self):
        """Clear ring → read returns nothing useful (or zeros from stale buffer)."""
        mod, solver = self._make_solver()
        try:
            mod.swe2d_gpu_init_diag_ring(64)
            mod.swe2d_gpu_push_diag(
                t_s=99.0, dt_used=0.0, gpu_active=1, wet_cells=0,
                max_courant=0.0, max_wse_error=0.0, mass_total=0.0,
            )
            mod.swe2d_gpu_clear_diag()
            # Push a fresh record; latest should be the fresh one, not 99.0
            mod.swe2d_gpu_push_diag(
                t_s=1.0, dt_used=0.01, gpu_active=1, wet_cells=10,
                max_courant=0.2, max_wse_error=0.0, mass_total=100.0,
            )
            rec = mod.swe2d_gpu_read_latest_diag()
            self.assertEqual(rec["t_s"], 1.0)
            self.assertEqual(rec["mass_total"], 100.0)
        finally:
            mod.swe2d_gpu_shutdown_diag_ring()
            mod.swe2d_destroy(solver)

    def test_swe2d_step_auto_pushes_to_ring(self):
        """swe2d_step auto-pushes a DiagRecord per step (Phase 4.1 wiring)."""
        mod, solver = self._make_solver()
        try:
            mod.swe2d_gpu_init_diag_ring(64)
            mod.swe2d_gpu_clear_diag()

            # Run a few steps — diag ring should fill automatically.
            for _ in range(3):
                mod.swe2d_step(solver, -1.0)

            rec = mod.swe2d_gpu_read_latest_diag()
            # t_s should have advanced (sum of dt_used) > 0
            self.assertGreater(rec["t_s"], 0.0)
            # dt_used should be a real positive timestep
            self.assertGreater(rec["dt_used"], 0.0)
            # gpu_active must be 1 (we used GPU)
            self.assertEqual(rec["gpu_active"], 1)
            # CFL must be in valid range
            self.assertGreaterEqual(rec["max_courant"], 0.0)
            self.assertLessEqual(rec["max_courant"], 10.0)
        finally:
            mod.swe2d_gpu_shutdown_diag_ring()
            mod.swe2d_destroy(solver)


if __name__ == "__main__":
    unittest.main()