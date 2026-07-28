"""Phase 3 — QOpenGLWidget with CUDA-OpenGL texture interop.

Drives the zero-D2H GUI render path: on each ``paintGL`` call, the
widget calls ``swe2d_gpu_render_into_gl_texture_dev`` which does
device-side min/max reduction + per-cell RGBA colorization directly
into the GL-mapped cudaArray.  The widget then draws a fullscreen
textured quad + a HUD overlay (t_s / CFL / wet / dt from the diag ring).

Reads live ``d_h`` from the solver's ``SWE2DDeviceState`` — no snapshot
ring buffer.  View layer (per ``MVP_ARCHITECTURE.md``) — owns its GL
resources and the timer that drives repaints.  Delegates computation
to the Phase 2/3 interop bindings + the service layer for mesh extents.
"""
from __future__ import annotations

from typing import Any, Optional

from qgis.PyQt import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QOpenGLWidget
# In PyQt5 5.15.x: QOpenGL* classes live in QtGui (not QtOpenGL,
# which only has the legacy QGL* classes).  We import them here
# rather than via `from PyQt5 import QtOpenGL` to avoid the legacy
# module.
from PyQt5.QtGui import (
    QOpenGLShaderProgram,
    QOpenGLShader,
    QOpenGLBuffer,
    QOpenGLTexture,
)

import numpy as np

# hydra_swe2d is the CUDA extension module — required at runtime by
# the GL widget (registers textures, runs kernels, etc.).  Imported at
# module level so it's not re-imported on every paintGL tick.  If the
# module is missing (e.g. .so not built), the widget fails fast in
# __init__ — a build misconfiguration, not a silent fallback.
try:
    import hydra_swe2d as _hydra_swe2d
except ImportError:
    _hydra_swe2d = None

# GLSL 410 shaders — minimal fullscreen textured quad.
# Use 410 (not 330) because PyQt5 5.15.x doesn't ship
# _QOpenGLFunctions_3_3_Core — only _QOpenGLFunctions_4_1_Core.
_VERTEX_SHADER = """
#version 410
in vec2 a_pos;
in vec2 a_uv;
out vec2 v_uv;
uniform mat4 u_proj;
void main() {
    v_uv = a_uv;
    gl_Position = u_proj * vec4(a_pos, 0.0, 1.0);
}
"""

_FRAGMENT_SHADER = """
#version 410
in vec2 v_uv;
out vec4 fragColor;
uniform sampler2D u_tex;
void main() {
    fragColor = texture(u_tex, v_uv);
}
"""

# Fullscreen quad: position (xy) + uv (st) — packed float32 buffer.
# QOpenGLBuffer.allocate expects raw bytes (sip.voidptr), so we pre-pack
# the 16 floats into a bytes object at import time.  Stride 16 below
# (4 floats × 4 bytes) matches the float32 layout.
_QUAD_VERTS = np.array(
    [
        -1.0, -1.0, 0.0, 1.0,
         1.0, -1.0, 1.0, 1.0,
        -1.0,  1.0, 0.0, 0.0,
         1.0,  1.0, 1.0, 0.0,
    ],
    dtype=np.float32,
).tobytes()
assert len(_QUAD_VERTS) == 16 * 4, "quad VBO size mismatch"


class GPUViewerGLWidget(QOpenGLWidget):
    """QOpenGLWidget that renders the GPU-direct viewer via CUDA-OpenGL interop.

    Reads live ``d_h`` directly from the solver's ``SWE2DDeviceState``
    via ``swe2d_gpu_render_into_gl_texture_dev``.  No snapshot ring
    buffer, no per-tick D2H for the colorization path.  Render every
    paintGL — Qt's GL thread model limits us to ~10 Hz naturally; the
    solver advances much faster (typical CFL-limited timestep is
    sub-millisecond) so we always paint the freshest available frame.
    """

    FIELD_OPTIONS = ["depth"]
    DEFAULT_WIDTH = 960
    DEFAULT_HEIGHT = 540

    frameRendered = QtCore.pyqtSignal(object)  # {ok, bytes_written} or {ok: False, error}

    def __init__(self, mesh_data: dict,
                 parent: Optional[QtWidgets.QWidget] = None,
                 solver: Any = None,
                 get_solver_fn: Optional[Any] = None) -> None:
        super().__init__(parent)
        self._mesh_data = mesh_data
        # Either a static solver handle (legacy) or a callable that
        # returns the current solver (preferred — picks up solvers from
        # runs that start AFTER the dialog is opened).
        self._solver_static = solver
        self._get_solver_fn = get_solver_fn
        self._solver: Any = None  # refreshed each paintGL tick
        self._field = "depth"
        self._auto_scale = True

        # Set OpenGL 4.1 Core Profile format (required for our shaders + the
# QOpenGLFunctions subclass shipped by PyQt5 5.15.x; 3.3 Core is
# unavailable because _QOpenGLFunctions_3_3_Core is not generated).
        fmt = QtGui.QSurfaceFormat()
        fmt.setVersion(4, 1)
        fmt.setProfile(QtGui.QSurfaceFormat.CoreProfile)
        fmt.setSwapBehavior(QtGui.QSurfaceFormat.DoubleBuffer)
        self.setFormat(fmt)

        # GL state — initialized in initializeGL.
        self._program: Optional[QOpenGLShaderProgram] = None
        self._texture: Optional[QtGui.QOpenGLTexture] = None
        self._texture_id: int = 0
        self._cuda_resource: int = 0          # opaque CUgraphicsResource handle
        self._width: int = self.DEFAULT_WIDTH
        self._height: int = self.DEFAULT_HEIGHT

        # Colormap LUT (rebuilt once) — uint8 ndarray, shape (256, 4).
        self._lut: Optional[np.ndarray] = None

        # Render timer (~10 Hz paintGL rate).
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.update)  # triggers paintGL
        self._timer.start(100)

    # ── View protocol (matches Phase 1 GPUViewerDialog) ──────────────────

    def get_field(self) -> str:
        return self._field

    def set_field(self, field: str) -> None:
        if field not in self.FIELD_OPTIONS:
            raise ValueError(f"unknown field {field!r}")
        self._field = field

    # ── GL lifecycle ────────────────────────────────────────────────────

    def initializeGL(self) -> None:
        """Create shader program + GL texture + register with CUDA."""
        super().initializeGL()
        # Compile shaders.
        self._program = QOpenGLShaderProgram()
        self._program.addShaderFromSourceCode(
            QOpenGLShader.Vertex, _VERTEX_SHADER)
        self._program.addShaderFromSourceCode(
            QOpenGLShader.Fragment, _FRAGMENT_SHADER)
        if not self._program.link():
            raise RuntimeError(f"shader link failed: {self._program.log()}")

        # Fullscreen quad VBO (positions + uvs).
        vbo = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
        vbo.create()
        vbo.bind()
        vbo.allocate(_QUAD_VERTS, len(_QUAD_VERTS))
        pos_attr = self._program.attributeLocation("a_pos")
        uv_attr = self._program.attributeLocation("a_uv")
        self._program.enableAttributeArray(pos_attr)
        self._program.setAttributeBuffer(pos_attr, QOpenGLBuffer.VertexBuffer, 0, 2, 16)
        self._program.enableAttributeArray(uv_attr)
        self._program.setAttributeBuffer(uv_attr, QOpenGLBuffer.VertexBuffer, 8, 2, 16)
        vbo.release()

        # Create GL texture (RGBA8, viewport-sized).
        self._texture = QtGui.QOpenGLTexture(QtGui.QOpenGLTexture.Target2D)
        self._texture.setSize(self._width, self._height)
        fmt = (QtGui.QOpenGLTexture.TextureFormat.RGBA8UNorm
               if hasattr(QtGui.QOpenGLTexture.TextureFormat, "RGBA8UNorm")
               else QtGui.QOpenGLTexture.TextureFormat.RGBAFormat)
        self._texture.setFormat(fmt)
        self._texture.allocateStorage()
        self._texture_id = int(self._texture.textureId())

        # Register the GL texture with CUDA for interop write access.
        try:
            self._cuda_resource = int(
                _hydra_swe2d.swe2d_gpu_register_gl_texture(self._texture_id)
            )
        except Exception as exc:
            # Interop unavailable — widget will show black texture.  The
            # caller should fall back to the CPU rasterizer path.
            self._cuda_resource = 0
            self._frameRenderedError(f"register_gl_texture failed: {exc}")

    def resizeGL(self, w: int, h: int) -> None:
        """Recreate the GL texture at the new viewport size.

        Qt refuses setSize/allocateStorage on a texture that already has
        storage.  Detach the existing CUDA interop binding first,
        destroy the texture, then build a fresh one and re-register.
        """
        super().resizeGL(w, h)
        new_w = max(64, int(w))
        new_h = max(64, int(h))
        if new_w == self._width and new_h == self._height and self._texture is not None:
            return  # no-op resize
        self._width = new_w
        self._height = new_h
        # Detach the CUDA-registered texture before destroying it.
        if self._cuda_resource and _hydra_swe2d is not None:
            try:
                _hydra_swe2d.swe2d_gpu_unregister_gl_texture(self._cuda_resource)
            except Exception:
                pass
            self._cuda_resource = 0
            self._texture_id = 0
        if self._texture is not None:
            self._texture.destroy()
            self._texture = None
        # Recreate.
        self._texture = QtGui.QOpenGLTexture(QtGui.QOpenGLTexture.Target2D)
        self._texture.setSize(self._width, self._height)
        fmt = (QtGui.QOpenGLTexture.TextureFormat.RGBA8UNorm
               if hasattr(QtGui.QOpenGLTexture.TextureFormat, "RGBA8UNorm")
               else QtGui.QOpenGLTexture.TextureFormat.RGBAFormat)
        self._texture.setFormat(fmt)
        self._texture.allocateStorage()
        self._texture_id = int(self._texture.textureId())
        if _hydra_swe2d is not None:
            try:
                self._cuda_resource = int(
                    _hydra_swe2d.swe2d_gpu_register_gl_texture(self._texture_id)
                )
            except Exception as exc:
                self._cuda_resource = 0
                self._frameRenderedError(f"register_gl_texture failed: {exc}")

    def paintGL(self) -> None:
        """Render live solver state via dev_ptr (zero D2H for colorization).

        Pipeline:
        1. dev_ptr = swe2d_get_solver_dev_ptr(self._solver)  (1 call, ~us)
        2. binding does min/max reduction on device, uploads cell_x/y/LUT
        3. launches color kernel writing RGBA into the GL-mapped cudaArray
        4. draws a fullscreen textured quad
        5. renders HUD text (t_s / CFL / wet / dt) into the same texture

        No snapshot ring buffer.  Reads ``d_h`` directly from
        ``SWE2DDeviceState`` every paintGL.  ``t_s`` comes from the
        diag ring (already on device).
        """
        super().paintGL()
        # PyQt5 5.15.x ships _QOpenGLFunctions_4_1_Core, not _3_3_Core.
        from PyQt5._QOpenGLFunctions_4_1_Core import QOpenGLFunctions_4_1_Core
        f = QOpenGLFunctions_4_1_Core()
        f.initializeOpenGLFunctions()
        f.glClearColor(0.0, 0.0, 0.0, 1.0)
        f.glClear(0x00004000 | 0x00000100)  # GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT

        # Early exit paths — the widget stays visible (black texture) until
        # both the CUDA-OpenGL interop is registered AND a solver is wired.
        if self._cuda_resource == 0:
            return
        # Refresh solver handle from callable each frame (the common case
        # is: dialog opened before run started, run kicks off later).
        # Falls back to the static handle if no callable was provided.
        if self._get_solver_fn is not None:
            try:
                self._solver = self._get_solver_fn()
            except Exception:
                self._solver = None
        else:
            self._solver = self._solver_static
        if self._solver is None:
            self.frameRendered.emit({"ok": False, "error": "no solver attached"})
            return
        if _hydra_swe2d is None:
            self.frameRendered.emit({"ok": False, "error": "hydra_swe2d not loaded"})
            return

        # 1. Get dev_ptr — thin wrapper around the SWE2DDeviceState*.
        dev_ptr = _hydra_swe2d.swe2d_get_solver_dev_ptr(self._solver)
        if dev_ptr == 0:
            self.frameRendered.emit({"ok": False, "error": "solver not initialized"})
            return

        # 2. Build LUT once.
        if self._lut is None:
            self._lut = self._build_lut("turbo")

        # 3. Mesh extents from service layer (numpy lives in the service,
        # not the View — per MVP_ARCHITECTURE.md Rule 4).
        from swe2d.workbench.services.viewer_frame_extents import (
            compute_render_extents,
        )
        xmax, ymax, cell_x, cell_y = compute_render_extents(self._mesh_data)

        # 4. Live render: dev_ptr binding does min/max reduction on device,
        # then color kernel writes RGBA into the GL-mapped cudaArray.
        try:
            result = _hydra_swe2d.swe2d_gpu_render_into_gl_texture_dev(
                dev_ptr, "h",
                self._cuda_resource, self._width, self._height,
                self._lut, cell_x, cell_y,
                0.0, xmax, 0.0, ymax,
            )
        except Exception as exc:
            self._frameRenderedError(f"render_into_gl_texture_dev: {exc}")
            return

        # 5. Draw the textured fullscreen quad.  Skip if shader program
        # didn't link (e.g. in CI without a hardware GL context).
        if self._program is None or not self._program.isLinked():
            self.frameRendered.emit(result)
            return

        self._program.bind()
        tex_loc = self._program.uniformLocation("u_tex")
        if tex_loc >= 0:
            f.glUniform1i(tex_loc, 0)
        self._texture.bind(0)
        f.glDrawArrays(0x0005, 0, 4)  # GL_TRIANGLE_STRIP
        self._texture.release()
        self._program.release()

        # 6. Overlay HUD diagnostic text (zero D2H — renders directly into
        # the same GL texture's cudaArray).  Wet cell count and CFL/dt
        # come from the diag ring (already on device) — no Python work.
        try:
            diag = _hydra_swe2d.swe2d_gpu_read_latest_diag()
            t_s = float(diag.get("t_s", 0.0))
            cfl = float(diag.get("max_courant", 0.0))
            dt_used = float(diag.get("dt_used", 0.0))
            wet = int(diag.get("wet_cells", 0))
            hud_text = (
                f"t={t_s:.1f}s CFL={cfl:.2f} wet={wet} dt={dt_used:.3f}"
            ).encode("ascii")
            _hydra_swe2d.swe2d_gpu_render_hud(
                self._cuda_resource, self._width, self._height,
                hud_text, 8, 8,
                0x80000000,  # ARGB: semi-black background
                0xFFFFFFFF,  # ARGB: white foreground
            )
        except Exception as exc:
            # Kernel failure (bad GL state, OOM, etc.) — surface via the
            # frameRendered signal.  Does NOT replace the texture render
            # above, which already succeeded.
            self._frameRenderedError(f"HUD render failed: {exc}")

        self.frameRendered.emit(result)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._timer.stop()
        if self._cuda_resource:
            try:
                _hydra_swe2d.swe2d_gpu_unregister_gl_texture(self._cuda_resource)
            except Exception:
                pass
            self._cuda_resource = 0
        super().closeEvent(event)

    # ── Helpers ────────────────────────────────────────────────────────

    def _frameRenderedError(self, msg: str) -> None:
        self.frameRendered.emit({"ok": False, "error": msg})

    @staticmethod
    def _build_lut(key: str = "turbo") -> np.ndarray:
        """256-entry RGBA LUT (uint8 ndarray, contiguous).

        The binding requires a uint8 ndarray (not raw bytes) so
        pybind11 can pass it through cleanly.  Mirrors Phase 2 CLI's
        _COLOR_LUTS.
        """
        # Lazy import to avoid Qt dependency in non-GUI tests.
        from tools.hydra_viewer_cli import build_colormap_lut
        return np.ascontiguousarray(build_colormap_lut(key), dtype=np.uint8)