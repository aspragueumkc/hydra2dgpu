"""Headless CUDA-OpenGL interop test via EGL_EXT_platform_device.

PyQt5's ``QOffscreenSurface`` always uses Mesa software rendering (llvmpipe /
swrast), which cannot do CUDA-GL interop. This test bypasses Qt/X entirely
and goes straight to the NVIDIA GPU through:

    EGL → libglvnd → libnvidia-eglcore → /dev/dri/renderD128 → nvidia_drm → RTX 3080

The trick: ``EGL_KHR_no_config_context`` skips the (broken on device platform)
config selection, and ``EGL_KHR_surfaceless_context`` skips the surface. The
result is a real NVIDIA OpenGL context bound to ``/dev/dri/renderD128`` with
full CUDA-GL interop, no X server / RDP / Wayland required.

Auto-skipped without:
  * Phase 3 interop binding present in the .so
  * A CUDA-compatible GPU
  * An NVIDIA EGL device (``EGL_NV_device_cuda``)
  * The ``render`` group membership on ``/dev/dri/renderD128``

Under ``HYDRA_REQUIRE_CUDA_GL_INTEROP=1`` the same conditions are hard
failures (no silent skip), matching the existing viewer-interop test.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys
import unittest

import numpy as np


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BUILD_DIR = os.environ.get("HYDRA_BUILD_DIR") or os.path.join(_REPO_ROOT, "build")
for _p in (_REPO_ROOT, _BUILD_DIR):
    if _p not in sys.path and os.path.isdir(_p):
        sys.path.insert(0, _p)

from tests._swe2d_test_helpers import _make_rect_mesh  # noqa: E402


# ── EGL constants used by the test ──────────────────────────────────────────
EGL_NO_DISPLAY = ctypes.c_void_p()
EGL_NO_CONTEXT = ctypes.c_void_p()
EGL_NO_SURFACE = ctypes.c_void_p()
EGL_OPENGL_API = 0x30A2
EGL_NONE = 0x3038
EGL_PLATFORM_DEVICE_EXT = 0x313F
EGL_DRM_RENDER_NODE_FILE_EXT = 0x3377


def _load_egl():
    try:
        return ctypes.CDLL("libEGL.so.1", ctypes.RTLD_GLOBAL)
    except OSError:
        return None


def _load_gl():
    path = ctypes.util.find_library("GL")
    if not path:
        return None
    try:
        return ctypes.CDLL(path, ctypes.RTLD_GLOBAL)
    except OSError:
        return None


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


def _find_nvidia_device(libegl):
    """Return the EGLDeviceEXT for the first NVIDIA-CUDA-capable device, or None.

    Looks up the ``eglQueryDevicesEXT`` / ``eglQueryDeviceStringEXT`` /
    ``eglGetPlatformDisplayEXT`` symbols via ``eglGetProcAddress`` since they
    live in the ``EGL_EXT_device_enumeration`` extension, not the core ABI.
    """
    EGLint = ctypes.c_int32
    EGLDeviceEXT = ctypes.c_void_p

    libegl.eglGetProcAddress.argtypes = [ctypes.c_char_p]
    libegl.eglGetProcAddress.restype = ctypes.c_void_p

    def _resolve(name, argtypes, restype):
        addr = libegl.eglGetProcAddress(name.encode())
        if not addr:
            return None
        return ctypes.CFUNCTYPE(restype, *argtypes)(addr)

    qdevs = _resolve("eglQueryDevicesEXT",
                     [EGLint,
                      ctypes.POINTER(EGLDeviceEXT),
                      ctypes.POINTER(EGLint)],
                     ctypes.c_uint32)
    qstr = _resolve("eglQueryDeviceStringEXT",
                    [EGLDeviceEXT, EGLint],
                    ctypes.c_char_p)
    if not (qdevs and qstr):
        return None

    devices = (EGLDeviceEXT * 8)()
    num = EGLint()
    qdevs(8, devices, ctypes.byref(num))
    if num.value == 0:
        return None

    # Prefer the device that exposes EGL_NV_device_cuda (it can do CUDA-GL).
    # Fall back to the first device that resolves to /dev/dri/renderD128.
    fallback = None
    for i in range(num.value):
        ext = qstr(devices[i], 0x3055)  # EGL_EXTENSIONS
        rn = qstr(devices[i], EGL_DRM_RENDER_NODE_FILE_EXT)
        if ext and b"EGL_NV_device_cuda" in ext:
            return devices[i]
        if fallback is None and rn and b"renderD" in rn:
            fallback = devices[i]
    return fallback


def _create_nvidia_gl_context(libegl, device):
    """Return (display, context) on the NVIDIA device, no config, no surface.

    Requires ``EGL_KHR_no_config_context`` (skip config selection) and
    ``EGL_KHR_surfaceless_context`` (skip surface creation). Both are
    available in NVIDIA's EGL 1.5 (driver ≥ 470).
    """
    EGLint = ctypes.c_int32
    EGLDisplay = ctypes.c_void_p
    EGLContext = ctypes.c_void_p

    libegl.eglGetProcAddress.argtypes = [ctypes.c_char_p]
    libegl.eglGetProcAddress.restype = ctypes.c_void_p

    def _resolve(name, argtypes, restype):
        addr = libegl.eglGetProcAddress(name.encode())
        if not addr:
            return None
        return ctypes.CFUNCTYPE(restype, *argtypes)(addr)

    get_platform_display = _resolve("eglGetPlatformDisplayEXT",
                                    [EGLint, ctypes.c_void_p,
                                     ctypes.POINTER(EGLint)],
                                    EGLDisplay)
    if not get_platform_display:
        return None, None

    display = get_platform_display(EGL_PLATFORM_DEVICE_EXT, device, None)
    if not display:
        return None, None

    libegl.eglInitialize.argtypes = [EGLDisplay,
                                     ctypes.POINTER(EGLint),
                                     ctypes.POINTER(EGLint)]
    libegl.eglInitialize.restype = ctypes.c_uint32
    major = EGLint()
    minor = EGLint()
    if not libegl.eglInitialize(display, ctypes.byref(major), ctypes.byref(minor)):
        return None, None

    # Confirm the device actually exposes no_config_context + surfaceless.
    # Set argtypes/restype BEFORE calling — without them ctypes returns int.
    libegl.eglQueryString.argtypes = [EGLDisplay, EGLint]
    libegl.eglQueryString.restype = ctypes.c_char_p
    disp_ext = libegl.eglQueryString(display, 0x3055)
    if not disp_ext or b"EGL_KHR_no_config_context" not in disp_ext:
        libegl.eglTerminate.argtypes = [EGLDisplay]
        libegl.eglTerminate(display)
        return None, None

    libegl.eglBindAPI.argtypes = [EGLint]
    libegl.eglBindAPI.restype = ctypes.c_uint32
    libegl.eglBindAPI(EGL_OPENGL_API)

    libegl.eglCreateContext.argtypes = [EGLDisplay, EGLContext, EGLContext,
                                        ctypes.POINTER(EGLint)]
    libegl.eglCreateContext.restype = EGLContext
    ctx_attribs = (EGLint * 5)(0x3098, 3, 0x30FB, 3, EGL_NONE)  # MAJOR=3, MINOR=3
    ctx = libegl.eglCreateContext(display, EGL_NO_CONTEXT, EGL_NO_CONTEXT,
                                  ctx_attribs)
    if not ctx or ctx == EGL_NO_CONTEXT:
        libegl.eglTerminate.argtypes = [EGLDisplay]
        libegl.eglTerminate(display)
        return None, None

    libegl.eglMakeCurrent.argtypes = [EGLDisplay, ctypes.c_void_p,
                                      ctypes.c_void_p, EGLContext]
    libegl.eglMakeCurrent.restype = ctypes.c_uint32
    if not libegl.eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx):
        libegl.eglTerminate.argtypes = [EGLDisplay]
        libegl.eglTerminate(display)
        return None, None

    return display, ctx


class TestGPUViewerInteropHeadlessEGL(unittest.TestCase):
    """Headless CUDA-GL interop via EGL_EXT_platform_device + DRM.

    Drives the same worktree interop pipeline as
    ``TestGPUViewerInterop::test_register_and_render_one_frame`` but
    without Qt, without X, without a display. Skips gracefully when no
    NVIDIA EGL device is reachable via ``/dev/dri/renderD128``.
    """

    NX = 10
    NY = 4
    LX = 100.0
    LY = 40.0
    W = 320
    H = 180

    def setUp(self):
        super().setUp()
        self._require_or_skip(
            _binding_present(),
            "Phase 3 interop bindings not in .so",
        )
        self._require_or_skip(
            _gpu_available(),
            "CUDA GPU not available",
        )
        libegl = _load_egl()
        self._require_or_skip(
            libegl is not None,
            "libEGL.so.1 not loadable",
        )
        libgl = _load_gl()
        self._require_or_skip(
            libgl is not None,
            "libGL not loadable",
        )

        device = _find_nvidia_device(libegl)
        self._require_or_skip(
            device is not None,
            "no NVIDIA EGL device (EGL_NV_device_cuda or renderD128)",
        )

        display, ctx = _create_nvidia_gl_context(libegl, device)
        self._require_or_skip(
            display is not None and ctx is not None,
            "could not create headless NVIDIA EGL context "
            "(EGL_KHR_no_config_context + EGL_KHR_surfaceless_context)",
        )

        # Confirm we really got NVIDIA, not Mesa-software-in-disguise.
        glGetString = libgl.glGetString
        glGetString.argtypes = [ctypes.c_uint]
        glGetString.restype = ctypes.c_char_p
        renderer = glGetString(0x1F01)
        self._require_or_skip(
            renderer is not None and b"NVIDIA" in renderer,
            f"headless EGL context is not NVIDIA (renderer={renderer!r})",
        )

        self._libegl = libegl
        self._libgl = libgl
        self._display = display
        self._ctx = ctx
        self._renderer = renderer

    def tearDown(self):
        # Release the EGL context + display (best-effort).
        libegl = getattr(self, "_libegl", None)
        display = getattr(self, "_display", None)
        if libegl and display:
            try:
                libegl.eglMakeCurrent.argtypes = [ctypes.c_void_p,
                                                  ctypes.c_void_p,
                                                  ctypes.c_void_p,
                                                  ctypes.c_void_p]
                libegl.eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE,
                                      EGL_NO_CONTEXT)
                libegl.eglTerminate.argtypes = [ctypes.c_void_p]
                libegl.eglTerminate(display)
            except Exception:
                pass

    def _require_or_skip(self, condition: bool, message: str) -> None:
        if condition:
            return
        if os.environ.get("HYDRA_REQUIRE_CUDA_GL_INTEROP") == "1":
            self.fail(message)
        self.skipTest(message)

    def _gen_texture(self):
        """Allocate a 320×180 GL_TEXTURE_2D RGBA8 texture, return its ID."""
        libgl = self._libgl
        EGLint = ctypes.c_int32

        libgl.glGenTextures.argtypes = [EGLint,
                                       ctypes.POINTER(ctypes.c_uint)]
        libgl.glGenTextures.restype = None
        libgl.glBindTexture.argtypes = [ctypes.c_uint, ctypes.c_uint]
        libgl.glBindTexture.restype = None
        libgl.glTexImage2D.argtypes = [ctypes.c_uint, EGLint, EGLint, EGLint,
                                       EGLint, EGLint, ctypes.c_uint,
                                       ctypes.c_uint, ctypes.c_void_p]
        libgl.glTexImage2D.restype = None
        libgl.glDeleteTextures.argtypes = [EGLint,
                                          ctypes.POINTER(ctypes.c_uint)]
        libgl.glDeleteTextures.restype = None

        tex = ctypes.c_uint()
        libgl.glGenTextures(1, ctypes.byref(tex))
        libgl.glBindTexture(0x0DE1, tex)  # GL_TEXTURE_2D
        libgl.glTexImage2D(0x0DE1, 0, 0x1908, self.W, self.H, 0,
                           0x1908, 0x1401, None)  # RGBA8, UNSIGNED_BYTE
        libgl.glBindTexture(0x0DE1, ctypes.c_uint(0))
        return tex

    def test_register_render_unregister_one_frame(self):
        """Register a GL texture, render one frame, unregister. End-to-end."""
        import hydra_swe2d as mod

        tex = self._gen_texture()
        try:
            handle = mod.swe2d_gpu_register_gl_texture(int(tex.value))
            self.assertNotEqual(handle, 0, "register returned null handle")

            # Build the same small rect mesh + solver the Qt-based test uses.
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
            solver = mod.swe2d_create_solver(
                mesh, np.full(info["n_cells"], 1.0, dtype=np.float64),
                n_mann=0.0, cfl=0.45, dt_max=0.5, use_gpu=True,
            )
            try:
                cell_x = np.zeros(info["n_cells"], dtype=np.float64)
                cell_y = np.zeros(info["n_cells"], dtype=np.float64)
                for ci in range(info["n_cells"]):
                    n0 = cell_nodes[3*ci]
                    n1 = cell_nodes[3*ci + 1]
                    n2 = cell_nodes[3*ci + 2]
                    cell_x[ci] = (node_x[n0] + node_x[n1] + node_x[n2]) / 3.0
                    cell_y[ci] = (node_y[n0] + node_y[n1] + node_y[n2]) / 3.0

                lut = np.zeros(256 * 4, dtype=np.uint8)
                for i in range(256):
                    lut[4*i + 0] = i
                    lut[4*i + 1] = i
                    lut[4*i + 2] = i
                    lut[4*i + 3] = 255

                result = mod.swe2d_gpu_render_into_gl_texture(
                    solver, "h", 0.0, 1.0,
                    handle, self.W, self.H, lut,
                    cell_x, cell_y,
                    0.0, self.LX, 0.0, self.LY,
                )
                self.assertTrue(
                    result.get("ok"),
                    f"render failed: {result.get('error')}",
                )
                self.assertEqual(result.get("bytes_written"), self.W * self.H * 4)
            finally:
                mod.swe2d_destroy(solver)

            # Unregister. Exception ⇒ fail (catches the swallowed-error
            # regression we just fixed in the pybind bindings).
            try:
                mod.swe2d_gpu_unregister_gl_texture(handle)
            except Exception as exc:
                self.fail(f"unregister raised: {exc!r}")
        finally:
            self._libgl.glDeleteTextures(1, ctypes.byref(tex))

    def test_register_unregister_cycles_have_no_leak(self):
        """Register/unregister 5 times — catches unmap-on-unmap / dbl-free."""
        import hydra_swe2d as mod
        tex = self._gen_texture()
        try:
            for _ in range(5):
                handle = mod.swe2d_gpu_register_gl_texture(int(tex.value))
                self.assertNotEqual(handle, 0)
                mod.swe2d_gpu_unregister_gl_texture(handle)
        finally:
            self._libgl.glDeleteTextures(1, ctypes.byref(tex))


if __name__ == "__main__":
    unittest.main()