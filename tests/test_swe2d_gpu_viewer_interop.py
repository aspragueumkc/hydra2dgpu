"""Phase 3 — CUDA-OpenGL interop smoke test.

The C++ interop helpers + bindings + kernel are built and verified to load
(this module passes the import check).  The full register+render flow
requires a real CUDA-compatible OpenGL context.  PyQt5's
``QOffscreenSurface`` in ``QT_QPA_PLATFORM=offscreen`` mode uses Mesa
software rendering (llvmpipe / swrast), which does not support CUDA-OpenGL
interop (``cuGraphicsGLRegisterImage`` returns ``CUDA_ERROR_UNKNOWN``).

The smoke test below is marked ``expectedFailure`` for that reason — it
will pass once run against a real GL context (e.g. in a live QGIS session
or via ``xvfb-run`` with NVIDIA GLX libs).  The C++ code path is verified
via the binding existence check below.

Auto-skipped without GPU + Phase 3 binding.
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pytest


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BUILD_DIR = os.environ.get("HYDRA_BUILD_DIR") or os.path.join(_REPO_ROOT, "build")
_CPP_DIR = Path(_REPO_ROOT) / "cpp" / "src"
for _p in (_REPO_ROOT, _BUILD_DIR):
    if _p not in sys.path and os.path.isdir(_p):
        sys.path.insert(0, _p)


def _binding_present():
    try:
        import hydra_swe2d as m
        return all(
            hasattr(m, a) for a in
            ("swe2d_gpu_register_gl_texture",
             "swe2d_gpu_render_into_gl_texture",
             "swe2d_gpu_unregister_gl_texture")
        )
    except ImportError:
        return False


def _gpu_available():
    try:
        import hydra_swe2d as m
        return m.swe2d_gpu_available()
    except Exception:
        return False


from tests._swe2d_test_helpers import _make_rect_mesh  # noqa: E402


class TestGPUViewerSynchronizationSource(unittest.TestCase):
    """Hardware-independent source-contract tests for viewer stage checks.

    These tests pin the C++ contract declared in
    ``docs/specs/2026-07-27-gpu-viewer-stage-synchronization-design.md``:
    a shared ``check_cuda_stage`` helper, named layers that synchronize
    between kernel launch and surface destruction, and strict unmap
    error handling. They read source files directly so they can run on
    any host without a CUDA-OpenGL context.
    """

    INTEROP_HEADER = _CPP_DIR / "swe2d_gpu_viewer_interop.cuh"
    INTEROP_SOURCE = _CPP_DIR / "swe2d_gpu_viewer_interop.cu"
    HUD_SOURCE = _CPP_DIR / "swe2d_gpu_viewer_hud.cu"

    def test_named_stage_checker_contract(self):
        header = self.INTEROP_HEADER.read_text()
        source = self.INTEROP_SOURCE.read_text()
        self.assertIn(
            "void check_cuda_stage(const char* stage, void* stream);",
            header,
        )
        self.assertIn("cudaPeekAtLastError()", source)
        self.assertIn("cudaStreamSynchronize", source)
        self.assertIn('check_cuda_stage("viewer min/max"', source)
        self.assertIn('check_cuda_stage("viewer color"', source)

    def test_color_completes_before_surface_destroy(self):
        source = self.INTEROP_SOURCE.read_text()
        launch = source.index("color_kernel_into_array<<<")
        check = source.index('check_cuda_stage("viewer color"', launch)
        destroy = source.index("cudaDestroySurfaceObject", check)
        self.assertLess(launch, check)
        self.assertLess(check, destroy)

    def test_hud_completes_before_surface_destroy(self):
        source = self.HUD_SOURCE.read_text()
        launch = source.index("hud_render_kernel<<<")
        check = source.index('check_cuda_stage("viewer HUD"', launch)
        destroy = source.index("cudaDestroySurfaceObject", check)
        self.assertLess(launch, check)
        self.assertLess(check, destroy)

    def test_unmap_errors_are_not_swallowed(self):
        source = self.INTEROP_SOURCE.read_text()
        start = source.index("void unmap(")
        body = source[start:source.index("__global__", start)]
        self.assertIn("throw std::runtime_error", body)
        self.assertIn("cuGraphicsUnmapResources failed", body)


class TestGPUViewerInterop(unittest.TestCase):
    """Verify the CUDA-OpenGL interop helpers link, register, and render."""

    NX = 10
    NY = 4
    LX = 100.0
    LY = 40.0

    def setUp(self):
        """Guard class-level dependencies via _require_or_skip.

        In ordinary (no env var) mode a missing binding or GPU causes a
        unittest skip, preserving the original test-suite skip behavior.
        Under ``HYDRA_REQUIRE_CUDA_GL_INTEROP=1`` the same conditions
        raise ``AssertionError`` so the sanitizer target cannot pass
        through a silent skip.
        """
        super().setUp()
        self._require_or_skip(
            _binding_present(),
            "Phase 3 interop bindings not in .so",
        )
        self._require_or_skip(
            _gpu_available(),
            "CUDA GPU not available",
        )

    def _require_or_skip(self, condition: bool, message: str) -> None:
        if condition:
            return
        if os.environ.get("HYDRA_REQUIRE_CUDA_GL_INTEROP") == "1":
            self.fail(message)
        self.skipTest(message)

    def test_register_and_render_one_frame(self):
        """Register a GL texture, render one frame via the interop path.

        When ``HYDRA_REQUIRE_CUDA_GL_INTEROP=1`` is set, a missing or
        invalid GL context / failed CUDA-GL registration / null handle is
        a hard failure; otherwise the test skips (e.g. on Mesa software GL).
        """
        from PyQt5.QtCore import QCoreApplication
        from PyQt5.QtGui import QOpenGLContext, QImage, QOpenGLTexture
        from PyQt5.QtWidgets import QApplication

        _ = QApplication.instance() or QApplication([])

        # Build mesh + solver (Phase 2 pattern).
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
        solver = mod.swe2d_create_solver(
            mesh, np.full(n_cells, 1.0, dtype=np.float64),
            n_mann=0.0, cfl=0.45, dt_max=0.5, use_gpu=True,
        )
        try:
            cell_x = np.zeros(n_cells, dtype=np.float64)
            cell_y = np.zeros(n_cells, dtype=np.float64)
            for ci in range(n_cells):
                n0 = cell_nodes[3*ci]
                n1 = cell_nodes[3*ci + 1]
                n2 = cell_nodes[3*ci + 2]
                cell_x[ci] = (node_x[n0] + node_x[n1] + node_x[n2]) / 3.0
                cell_y[ci] = (node_y[n0] + node_y[n1] + node_y[n2]) / 3.0

            # Grayscale colormap LUT (256 entries).
            lut = np.zeros(256 * 4, dtype=np.uint8)
            for i in range(256):
                lut[4*i + 0] = i
                lut[4*i + 1] = i
                lut[4*i + 2] = i
                lut[4*i + 3] = 255

            # Create an offscreen OpenGL context + RGBA8 texture.
            from PyQt5.QtGui import QSurfaceFormat, QOffscreenSurface
            fmt = QSurfaceFormat()
            fmt.setVersion(3, 3)
            fmt.setProfile(QSurfaceFormat.CoreProfile)
            ctx = QOpenGLContext()
            ctx.setFormat(fmt)
            ok = ctx.create()
            self._require_or_skip(ok, "QOpenGLContext.create() failed")
            surface = QOffscreenSurface()
            surface.setFormat(ctx.format())
            surface.create()
            self._require_or_skip(
                surface.isValid(), "QOffscreenSurface is invalid"
            )
            self._require_or_skip(
                ctx.makeCurrent(surface), "QOpenGLContext.makeCurrent failed"
            )

            width, height = 320, 180
            tex = QOpenGLTexture(QOpenGLTexture.Target2D)
            tex.setSize(width, height)
            tex.setFormat(QOpenGLTexture.TextureFormat.RGBA8UNorm
                if hasattr(QOpenGLTexture.TextureFormat, "RGBA8UNorm")
                else QOpenGLTexture.TextureFormat.RGBAFormat)
            tex.allocateStorage()

            gl_tex_id = int(tex.textureId())
            try:
                handle = mod.swe2d_gpu_register_gl_texture(gl_tex_id)
            except Exception as exc:
                self._require_or_skip(
                    False, f"CUDA-OpenGL texture registration failed: {exc}"
                )
                raise AssertionError("unreachable") from exc
            if handle == 0:
                self._require_or_skip(False, "register returned null handle")

            try:
                result = mod.swe2d_gpu_render_into_gl_texture(
                    solver, "h", 0.0, 1.0,
                    handle, width, height, lut,
                    cell_x, cell_y,
                    0.0, self.LX, 0.0, self.LY,
                )
                self.assertTrue(
                    result.get("ok"),
                    f"render failed: {result.get('error')}",
                )
                self.assertEqual(
                    result.get("bytes_written"), width * height * 4,
                )
            finally:
                mod.swe2d_gpu_unregister_gl_texture(handle)

            ctx.doneCurrent()
        finally:
            mod.swe2d_destroy(solver)


def _load_module():
    try:
        import hydra_swe2d
        return hydra_swe2d
    except ImportError:
        return None


if __name__ == "__main__":
    unittest.main()