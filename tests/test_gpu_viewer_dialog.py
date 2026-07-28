from __future__ import annotations
import os
import sys
import unittest
"""Tests for swe2d.workbench.views.gpu_viewer_dialog.

The dialog is GL-only (no CPU rasterizer fallback, no reader).  These
tests verify dialog-level state (field selector, status label, close
semantics) without requiring a live GPU/GL context or pytest-qt's
qtbot fixture — they use QApplication directly so they survive the
``wrap_pytest_style`` unittest conversion.
"""

import numpy as np


# Ensure repo root on sys.path so swe2d imports work
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# Construct QApplication at import time — required before any QWidget
# construction.  setUpClass can race with test discovery in some runners.
from PyQt5.QtWidgets import QApplication  # noqa: E402
_app = QApplication.instance() or QApplication(sys.argv)


def _mesh_data():
    """Minimal mesh-data dict — empty arrays; tests don't actually render."""
    return {
        "cell_x": np.array([], dtype=np.float64),
        "cell_y": np.array([], dtype=np.float64),
        "cell_bed": np.array([], dtype=np.float64),
        "node_x": np.array([], dtype=np.float64),
        "node_y": np.array([], dtype=np.float64),
        "cell_nodes": np.array([], dtype=np.int32),
    }


class TestGPUViewerDialog(unittest.TestCase):
    """Dialog-level state tests (no live GPU/GL required)."""

    def setUp(self):
        from swe2d.workbench.views.gpu_viewer_dialog import GPUViewerDialog
        self.dlg = GPUViewerDialog(mesh_data=_mesh_data())

    def tearDown(self):
        try:
            self.dlg.close()
        except Exception:
            pass
        self.dlg.deleteLater()

    def test_default_field_is_depth(self):
        self.assertEqual(self.dlg.get_field(), "depth")

    def test_set_field_persists(self):
        self.dlg.set_field("depth")
        self.assertEqual(self.dlg.get_field(), "depth")

    def test_set_field_invalid_raises(self):
        with self.assertRaises(ValueError):
            self.dlg.set_field("not-a-field")

    def test_set_field_updates_combobox(self):
        self.dlg.set_field("depth")
        self.assertEqual(self.dlg._field_combo.currentText(), "depth")

    def test_auto_scale_toggle(self):
        self.assertTrue(self.dlg._auto_scale)
        self.dlg._auto_scale_cb.setChecked(False)
        self.assertFalse(self.dlg._auto_scale)

    def test_close_does_not_raise(self):
        # Should not raise — closeEvent handles missing gl_widget gracefully.
        self.dlg.close()

    def test_no_solver_shows_initial_status(self):
        """When solver=None, the status label has some initial text."""
        # The exact text depends on whether GL init succeeded in the
        # headless test env.  Either "waiting for run…" (GL ok) or
        # "GL init failed: …" (no GL).  Just verify it's not empty.
        self.assertNotEqual(self.dlg._status_label.text(), "")


if __name__ == "__main__":
    unittest.main()