"""Standalone live GPU-direct viewer dialog.

Wraps ``GPUViewerGLWidget`` (QOpenGLWidget + CUDA-OpenGL interop) for
a zero-D2H render of the active solver state.  Reads ``d_h`` directly
from ``SWE2DDeviceState`` via the dev_ptr path — no snapshot ring
buffer, no CPU rasterizer fallback.  The high-perf canvas overlay
covers the "snapshot-based CPU rasterizer" use case; this dialog is
exclusively the GPU-direct live viewer.

Per ``MVP_ARCHITECTURE.md`` this is a View layer module: it owns its
widgets, exposes protocol methods, and delegates all computation to
services (``compute_render_extents``) + the C++ interop bindings.
"""
from __future__ import annotations

from typing import Any, Optional

from qgis.PyQt import QtCore, QtGui, QtWidgets


class GPUViewerDialog(QtWidgets.QDialog):
    """Floating GL dialog showing live solver state at ~10 Hz."""

    FIELD_OPTIONS = ["depth"]

    frameRendered = QtCore.pyqtSignal(object)

    def __init__(self, mesh_data: Optional[dict] = None,
                 parent: Optional[QtWidgets.QWidget] = None,
                 solver: Any = None,
                 get_solver_fn: Optional[Any] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("GPU Live Viewer")
        self.resize(960, 600)
        self._mesh_data = mesh_data or {}
        # Either a static solver handle (legacy / one-shot) or a callable
        # returning the live solver.  The callable form lets the widget
        # pick up a SimulationWorker's solver handle when a run starts
        # AFTER the dialog was opened (common workflow: open the viewer,
        # then click Run).
        self._solver_static = solver
        self._get_solver_fn = get_solver_fn
        self._field = "depth"
        self._auto_scale = True

        layout = QtWidgets.QVBoxLayout(self)

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("Field:"))
        self._field_combo = QtWidgets.QComboBox()
        self._field_combo.addItems(self.FIELD_OPTIONS)
        self._field_combo.currentTextChanged.connect(self._on_field_changed)
        controls.addWidget(self._field_combo)
        self._auto_scale_cb = QtWidgets.QCheckBox("Auto-scale")
        self._auto_scale_cb.setChecked(True)
        self._auto_scale_cb.toggled.connect(self._on_auto_scale_toggled)
        controls.addWidget(self._auto_scale_cb)
        self._status_label = QtWidgets.QLabel("waiting for run…")
        # Selectable so users can copy error messages out (e.g. the CUDA
        # interop registration error includes the driver code, which is
        # useful for debugging).
        self._status_label.setTextInteractionFlags(
            QtCore.Qt.TextSelectableByMouse | QtCore.Qt.TextSelectableByKeyboard
        )
        self._status_label.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred
        )
        controls.addWidget(self._status_label)
        controls.addStretch(1)
        layout.addLayout(controls)

        # GL widget — owns its own timer (paints at ~10 Hz).  No CPU
        # fallback: this dialog is GPU-direct only.  The high-perf
        # canvas overlay handles the "snapshot CPU rasterizer" path.
        self._gl_widget = None
        try:
            from swe2d.workbench.views.gpu_viewer_gl_widget import (
                GPUViewerGLWidget,
            )
            self._gl_widget = GPUViewerGLWidget(
                mesh_data=self._mesh_data,
                get_solver_fn=get_solver_fn,
            )
            self._gl_widget.frameRendered.connect(self._on_frame_rendered)
            layout.addWidget(self._gl_widget, 1)
        except Exception as exc:
            # GL init failed — surface to the user and let the dialog
            # close.  Cannot fall back to CPU rasterizer (out of scope).
            self._status_label.setText(f"GL init failed: {exc}")
            QtWidgets.QMessageBox.warning(
                self,
                "HYDRA2DGPU",
                f"GPU Direct Viewer requires CUDA-OpenGL interop: {exc}",
            )

        if self._gl_widget is None:
            # Widget couldn't be created — disable controls, allow close.
            self._field_combo.setEnabled(False)
            self._auto_scale_cb.setEnabled(False)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self._gl_widget is not None:
            try:
                self._gl_widget.close()
            except Exception:
                pass
        super().closeEvent(event)

    def get_field(self) -> str:
        return self._field

    def set_field(self, field: str) -> None:
        if field not in self.FIELD_OPTIONS:
            raise ValueError(f"unknown field {field!r}")
        self._field = field
        idx = self.FIELD_OPTIONS.index(field)
        self._field_combo.setCurrentIndex(idx)
        if self._gl_widget is not None:
            self._gl_widget.set_field(field)

    def _on_field_changed(self, text: str) -> None:
        self._field = text

    def _on_auto_scale_toggled(self, checked: bool) -> None:
        self._auto_scale = checked

    def _on_frame_rendered(self, payload: dict) -> None:
        """Forward GL widget's frameRendered signal + update status label."""
        if isinstance(payload, dict):
            t_s = payload.get("t_s", 0.0)
            ok = payload.get("ok", False)
            if ok:
                self._status_label.setText(f"live (GL): t={t_s:.1f}s")
            else:
                err = payload.get("error", "unknown error")
                self._status_label.setText(f"GL err: {err}")
        self.frameRendered.emit(payload)