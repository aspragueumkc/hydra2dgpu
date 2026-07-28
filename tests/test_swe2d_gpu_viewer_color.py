"""Phase 2 — GPU color kernel smoke test.

Requires a built `hydra_swe2d` module with the Phase 2 binding
``swe2d_gpu_render_field_to_rgba`` + CUDA GPU.  Auto-skipped otherwise
via the conftest hooks in tests/conftest.py (markers ``solver`` + ``gpu``).
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
    """The Phase 2 binding must exist on the freshly built .so."""
    if mod is None:
        return False
    return hasattr(mod, "swe2d_gpu_render_field_to_rgba")


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
                     "swe2d_gpu_render_field_to_rgba not in binding (rebuild C++)")
@unittest.skipUnless(_gpu_available(), "CUDA GPU not available")
class TestGPUViewerColor(unittest.TestCase):
    """Verify the Phase 2 CUDA color kernel renders a single frame."""

    NX = 10
    NY = 4
    LX = 100.0
    LY = 40.0

    def test_color_kernel_renders_one_frame(self):
        mod = _load_module()
        node_x, node_y, node_z, cell_nodes = _make_rect_mesh(
            self.NX, self.NY, self.LX, self.LY,
        )
        mesh = mod.swe2d_build_mesh(
            node_x, node_y, node_z, cell_nodes,
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float64),
        )
        info = mod.swe2d_mesh_info(mesh)
        n_cells = info["n_cells"]
        h0 = np.full(n_cells, 1.0, dtype=np.float64)
        solver = mod.swe2d_create_solver(
            mesh, h0, n_mann=0.0, cfl=0.45, dt_max=0.5, use_gpu=True,
        )
        try:
            cell_x = np.zeros(n_cells, dtype=np.float64)
            cell_y = np.zeros(n_cells, dtype=np.float64)
            for ci in range(n_cells):
                n0, n1, n2 = cell_nodes[3*ci:3*ci+3]
                cell_x[ci] = (node_x[n0] + node_x[n1] + node_x[n2]) / 3.0
                cell_y[ci] = (node_y[n0] + node_y[n1] + node_y[n2]) / 3.0

            # Build a simple grayscale ramp colormap (256 entries).
            lut = np.zeros(256 * 4, dtype=np.uint8)
            for i in range(256):
                lut[4*i + 0] = i
                lut[4*i + 1] = i
                lut[4*i + 2] = i
                lut[4*i + 3] = 255

            result = mod.swe2d_gpu_render_field_to_rgba(
                solver, "h", 0.0, 1.0,
                320, 180, lut,
                cell_x, cell_y,
                0.0, self.LX, 0.0, self.LY,
            )
            arr = np.asarray(result["image"])
            self.assertEqual(
                arr.shape, (180, 320, 4),
                f"unexpected shape: {arr.shape}",
            )
            # At least some pixels rendered (alpha > 0).
            n_rendered = int((arr[..., 3] > 0).sum())
            self.assertGreater(
                n_rendered, 0,
                f"no rendered cells (shape={arr.shape})",
            )
            # Background pixels remain zero (the kernel only writes cell pixels).
            self.assertEqual(int((arr[..., 3] == 0).sum()) > 0, True)
        finally:
            mod.swe2d_destroy(solver)

    def test_color_kernel_unknown_field_raises(self):
        mod = _load_module()
        node_x, node_y, node_z, cell_nodes = _make_rect_mesh(4, 2, 40.0, 20.0)
        mesh = mod.swe2d_build_mesh(
            node_x, node_y, node_z, cell_nodes,
            np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float64),
        )
        info = mod.swe2d_mesh_info(mesh)
        n_cells = info["n_cells"]
        solver = mod.swe2d_create_solver(
            mesh, np.zeros(n_cells, dtype=np.float64),
            n_mann=0.0, cfl=0.45, dt_max=0.5, use_gpu=True,
        )
        try:
            cell_x = np.zeros(n_cells, dtype=np.float64)
            cell_y = np.zeros(n_cells, dtype=np.float64)
            lut = np.zeros(256 * 4, dtype=np.uint8)
            with self.assertRaises(RuntimeError) as ctx:
                mod.swe2d_gpu_render_field_to_rgba(
                    solver, "not-a-field", 0.0, 1.0,
                    100, 100, lut, cell_x, cell_y,
                    0.0, 40.0, 0.0, 20.0,
                )
            self.assertIn("unknown field_key", str(ctx.exception))
        finally:
            mod.swe2d_destroy(solver)


if __name__ == "__main__":
    unittest.main()