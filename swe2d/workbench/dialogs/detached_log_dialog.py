#!/usr/bin/env python3
"""Detached runtime log viewer dialog."""

from __future__ import annotations

from qgis.PyQt import QtWidgets


class SWE2DDetachedRuntimeLogDialog(QtWidgets.QDialog):
    def __init__(self, initial_text: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("2D SWE Runtime Log")
        self.resize(920, 620)
        root = QtWidgets.QVBoxLayout(self)
        self.text = QtWidgets.QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setPlainText(str(initial_text or ""))
        root.addWidget(self.text, stretch=1)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        root.addWidget(buttons)

    def append_text(self, msg: str) -> None:
        """Append a message to the log text widget."""
        self.text.appendPlainText(str(msg))

    def set_text(self, text: str) -> None:
        """Replace the entire log text content."""
        self.text.setPlainText(str(text or ""))
