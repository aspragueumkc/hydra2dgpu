"""Phase 3 — GPUViewerGLWidget construction + lifecycle tests.

Uses pytest-qt's qtbot fixture (per docs/plans/2026-07-26-gpu-direct-viewer.md
Task 1.2). Covers:
- Widget creation
- Field selector state (default + set)
- Field validation (invalid raises)
- Timer starts on construction
- closeEvent stops the timer

The full GL init + render path is covered by
tests/test_swe2d_gpu_viewer_interop.py (xfailed in offscreen mode).
"""
from __future__ import annotations

import os
import sys
import unittest

import pytest


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BUILD_DIR = os.environ.get("HYDRA_BUILD_DIR") or os.path.join(_REPO_ROOT, "build")
for _p in (_REPO_ROOT, _BUILD_DIR):
    if _p not in sys.path and os.path.isdir(_p):
        sys.path.insert(0, _p)


def _gpu_available():
    try:
        import hydra_swe2d as m
        return m.swe2d_gpu_available()
    except Exception:
        return False


@pytest.mark.gpu
@unittest.skipUnless(_gpu_available(), "CUDA GPU not available")
class TestGPUViewerGLWidget:
    """Widget construction + lifecycle tests."""

    def test_default_field_is_depth(self, qtbot):
        from swe2d.workbench.views.gpu_viewer_gl_widget import GPUViewerGLWidget
        from unittest.mock import MagicMock
        mesh_data = {
            "cell_x": [], "cell_y": [], "cell_bed": [],
            "node_x": [], "node_y": [], "cell_nodes": [],
        }
        w = GPUViewerGLWidget(reader=MagicMock(), mesh_data=mesh_data)
        qtbot.addWidget(w)
        assert w.get_field() == "depth"

    def test_set_field_persists(self, qtbot):
        from swe2d.workbench.views.gpu_viewer_gl_widget import GPUViewerGLWidget
        from unittest.mock import MagicMock
        mesh_data = {"cell_x": [], "cell_y": [], "cell_bed": [],
                     "node_x": [], "node_y": [], "cell_nodes": []}
        w = GPUViewerGLWidget(reader=MagicMock(), mesh_data=mesh_data)
        qtbot.addWidget(w)
        # Only "depth" is supported now (speed field was removed in 4.x
        # cleanup — see docs/plans/2026-07-26-gpu-direct-viewer-phase4-hud.md).
        w.set_field("depth")
        assert w.get_field() == "depth"

    def test_set_field_invalid_raises(self, qtbot):
        from swe2d.workbench.views.gpu_viewer_gl_widget import GPUViewerGLWidget
        from unittest.mock import MagicMock
        mesh_data = {"cell_x": [], "cell_y": [], "cell_bed": [],
                     "node_x": [], "node_y": [], "cell_nodes": []}
        w = GPUViewerGLWidget(reader=MagicMock(), mesh_data=mesh_data)
        qtbot.addWidget(w)
        with pytest.raises(ValueError):
            w.set_field("not-a-field")

    def test_field_options_constant(self, qtbot):
        from swe2d.workbench.views.gpu_viewer_gl_widget import GPUViewerGLWidget
        from unittest.mock import MagicMock
        mesh_data = {"cell_x": [], "cell_y": [], "cell_bed": [],
                     "node_x": [], "node_y": [], "cell_nodes": []}
        w = GPUViewerGLWidget(reader=MagicMock(), mesh_data=mesh_data)
        qtbot.addWidget(w)
        assert w.FIELD_OPTIONS == ["depth"]

    def test_timer_stops_on_close(self, qtbot):
        from swe2d.workbench.views.gpu_viewer_gl_widget import GPUViewerGLWidget
        from unittest.mock import MagicMock
        mesh_data = {"cell_x": [], "cell_y": [], "cell_bed": [],
                     "node_x": [], "node_y": [], "cell_nodes": []}
        w = GPUViewerGLWidget(reader=MagicMock(), mesh_data=mesh_data)
        qtbot.addWidget(w)
        assert w._timer.isActive()
        w.close()
        assert not w._timer.isActive()

    def test_default_dimensions(self, qtbot):
        from swe2d.workbench.views.gpu_viewer_gl_widget import GPUViewerGLWidget
        from unittest.mock import MagicMock
        mesh_data = {"cell_x": [], "cell_y": [], "cell_bed": [],
                     "node_x": [], "node_y": [], "cell_nodes": []}
        w = GPUViewerGLWidget(reader=MagicMock(), mesh_data=mesh_data)
        qtbot.addWidget(w)
        assert w._width == w.DEFAULT_WIDTH
        assert w._height == w.DEFAULT_HEIGHT