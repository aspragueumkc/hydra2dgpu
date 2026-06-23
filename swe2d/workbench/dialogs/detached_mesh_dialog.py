#!/usr/bin/env python3
"""Detached mesh view dialog.

The dialog uses the pure-Python ``mesh_render_service`` to render the
mesh view to a RGB ``numpy.ndarray`` and displays it in a Qt label via
``QImage``. No matplotlib Qt backend is required.

The dialog takes data sources as zero-arg getter callables so the
display always reflects the live state of the parent dialog (mesh and
result data may be re-imported at any time).
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

import numpy as np

from qgis.PyQt import QtCore, QtGui, QtWidgets

from swe2d.workbench.services.mesh_render_service import render_workbench_mesh_view

logger_wb = logging.getLogger(__name__)


MeshDataGetter = Callable[[], Optional[Dict[str, Any]]]
ResultDataGetter = Callable[[], Optional[Dict[str, Any]]]
HMinGetter = Callable[[], float]


class SWE2DDetachedMeshViewDialog(QtWidgets.QDialog):
    def __init__(
        self,
        mesh_data_fn: Optional[MeshDataGetter] = None,
        result_data_fn: Optional[ResultDataGetter] = None,
        h_min_fn: Optional[HMinGetter] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("2D SWE Mesh View")
        self.resize(980, 720)
        self._mesh_data_fn = mesh_data_fn if callable(mesh_data_fn) else (lambda: None)
        self._result_data_fn = result_data_fn if callable(result_data_fn) else (lambda: None)
        self._h_min_fn = h_min_fn if callable(h_min_fn) else (lambda: 1.0e-6)

        root = QtWidgets.QVBoxLayout(self)
        header = QtWidgets.QHBoxLayout()
        header.addWidget(QtWidgets.QLabel("View:"))
        self.view_mode_combo = QtWidgets.QComboBox()
        self.view_mode_combo.addItem("Mesh", "mesh")
        self.view_mode_combo.addItem("Depth", "depth")
        self.view_mode_combo.addItem("Velocity magnitude", "velocity")
        header.addWidget(self.view_mode_combo)
        header.addStretch(1)
        self.refresh_btn = QtWidgets.QPushButton("Refresh")
        header.addWidget(self.refresh_btn)
        root.addLayout(header)

        self._image_label = QtWidgets.QLabel()
        self._image_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._image_label.setMinimumSize(640, 480)
        self._image_label.setStyleSheet("background-color: #1e1e1e;")
        root.addWidget(self._image_label, stretch=1)

        self._status_label = QtWidgets.QLabel("")
        self._status_label.setStyleSheet("color: #888888;")
        root.addWidget(self._status_label)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        root.addWidget(buttons)

        self.refresh_btn.clicked.connect(self.refresh_view)
        self.view_mode_combo.currentIndexChanged.connect(self.refresh_view)
        self.refresh_view()

    def refresh_view(self) -> None:
        """Re-render the mesh/results view using current mode and data."""
        mesh_data = self._mesh_data_fn()
        if mesh_data is None:
            self._status_label.setText("No mesh loaded")
            self._image_label.clear()
            return
        mode = str(self.view_mode_combo.currentData() or "mesh")
        try:
            image = render_workbench_mesh_view(
                mesh_data=mesh_data,
                result_data=self._result_data_fn(),
                mode=mode,
                h_min=float(self._h_min_fn()),
            )
        except Exception as exc:
            logger_wb.warning("Detached mesh view render failed", exc_info=True)
            self._status_label.setText(f"Render failed: {exc}")
            self._image_label.clear()
            return

        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
            self._status_label.setText("Renderer returned unexpected image shape")
            self._image_label.clear()
            return

        h, w, _ = image.shape
        bytes_per_line = int(w) * 3
        qimage = QtGui.QImage(
            np.ascontiguousarray(image).tobytes(),
            int(w),
            int(h),
            bytes_per_line,
            QtGui.QImage.Format.Format_RGB888,
        )
        pixmap = QtGui.QPixmap.fromImage(qimage)
        self._image_label.setPixmap(
            pixmap.scaled(
                self._image_label.size(),
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
        )
        self._status_label.setText(f"Rendered {w}x{h} ({mode})")
