#!/usr/bin/env python3
"""Generic detachable container dialog."""

from __future__ import annotations

from qgis.PyQt import QtWidgets


class SWE2DDetachedPanelDialog(QtWidgets.QDialog):
    """Generic detachable container with automatic reattach callback."""

    def __init__(self, title: str, content_widget: QtWidgets.QWidget, on_reattach=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(str(title or "Detached Panel"))
        self.resize(760, 620)
        self._on_reattach = on_reattach
        self._content_widget = content_widget
        self._reattached = False

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        if self._content_widget is not None:
            root.addWidget(self._content_widget, stretch=1)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        self._reattach_btn = btns.addButton("Reattach", QtWidgets.QDialogButtonBox.ButtonRole.ActionRole)
        self._reattach_btn.clicked.connect(self._reattach_and_close)
        btns.rejected.connect(self.reject)
        btns.accepted.connect(self.accept)
        root.addWidget(btns)

    def _reattach_and_close(self) -> None:
        """Call the reattach callback and close the dialog."""
        self._reattach_once()
        self.close()

    def _reattach_once(self) -> None:
        """Fire the reattach callback once (idempotent)."""
        if self._reattached:
            return
        self._reattached = True
        if callable(self._on_reattach):
            try:
                self._on_reattach()
            except Exception:
                self._log("[WARNING] Unexpected Exception silently caught — review this handler")

    def closeEvent(self, event):
        """Fire reattach callback on close, then delegate to parent."""
        self._reattach_once()
        super().closeEvent(event)
