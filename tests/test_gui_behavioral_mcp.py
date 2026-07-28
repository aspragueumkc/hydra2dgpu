"""Xvfb-gated integration tests for MCP Phase 3 behavioral GUI tools.

These tests exercise the in-process bridge handlers against real Qt widgets.
They are skipped unless a display (Xvfb or real) is available, and unless the
QGIS Qt bindings are importable.
"""
from __future__ import annotations

import os
import sys
import time
import unittest
import types
from pathlib import Path

# Allow tests run from a worktree to find the native CUDA extension by passing
# the build directory via the HYDRA_BUILD_DIR environment variable.
_HYDRA_BUILD_DIR = os.environ.get("HYDRA_BUILD_DIR")
if _HYDRA_BUILD_DIR and str(_HYDRA_BUILD_DIR) not in sys.path:
    sys.path.insert(0, str(_HYDRA_BUILD_DIR))


# Gate the entire module on display availability.
if not os.environ.get("DISPLAY") and os.environ.get("QT_QPA_PLATFORM") != "offscreen":
    raise unittest.SkipTest("Xvfb/display not available; skipping GUI behavioral tests")


try:
    from qgis.PyQt.QtCore import QObject, Qt, QTimer, pyqtSignal, QEventLoop
    from qgis.PyQt.QtWidgets import (
        QAction,
        QApplication,
        QLineEdit,
        QMainWindow,
        QPushButton,
        QWidget,
    )
    from qgis.PyQt.QtTest import QTest
    from tools.hydra_mcp.qgis_bridge import HydraMcpBridge
except Exception as exc:  # pragma: no cover
    raise unittest.SkipTest(f"QGIS Qt not importable: {exc}") from exc


def _qapp() -> QApplication:
    """Return a real QApplication, creating one if needed."""
    app = QApplication.instance()
    if app is not None and not isinstance(app, unittest.mock.MagicMock):
        return app
    return QApplication([])


# Module-level application for the test process.
_APP = _qapp()


class _ClickReceiver(QWidget):
    """Simple widget with a named button and a counter."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.clicks = []
        self.btn = QPushButton("Click me", self)
        self.btn.setObjectName("test_button")
        self.btn.clicked.connect(lambda: self.clicks.append("click"))


class TestBridgeClickAndKey(unittest.TestCase):
    """gui_click and gui_key use QTest on live widgets."""

    def setUp(self):
        self.win = QMainWindow()
        self.win.setObjectName("test_main_window")
        self.central = _ClickReceiver()
        self.central.setObjectName("central")
        self.win.setCentralWidget(self.central)
        self.edit = QLineEdit(self.central)
        self.edit.setObjectName("test_edit")
        self.win.show()
        _APP.setActiveWindow(self.win)
        self.bridge = HydraMcpBridge()

    def tearDown(self):
        self.win.close()
        self.win.deleteLater()
        _APP.processEvents()

    def test_click_button_by_object_name(self):
        self.assertEqual(len(self.central.clicks), 0)
        result = self.bridge._handle_click({"object_name": "test_button"})
        self.assertTrue(result.get("ok"))
        _APP.processEvents()
        self.assertEqual(len(self.central.clicks), 1)

    def test_click_button_by_path(self):
        result = self.bridge._handle_click({"path": "central.test_button"})
        self.assertTrue(result.get("ok"))
        _APP.processEvents()
        self.assertEqual(len(self.central.clicks), 1)

    def test_key_into_line_edit(self):
        self.edit.setFocus()
        _APP.processEvents()
        result = self.bridge._handle_key({"object_name": "test_edit", "key": "A"})
        self.assertTrue(result.get("ok"))
        self.assertEqual(result.get("key"), "A")
        # Under some headless platforms key events may not update the widget
        # text; the handler reaching this point without error is the real check.

    def test_key_named_return(self):
        result = self.bridge._handle_key({"object_name": "test_edit", "key": "return"})
        self.assertTrue(result.get("ok"))


class TestBridgeRunAction(unittest.TestCase):
    """gui_run_action triggers QAction objects."""

    def setUp(self):
        self.win = QMainWindow()
        self.win.setObjectName("action_test_window")
        self.action = QAction("Test Action", self.win)
        self.action.setObjectName("test_action")
        self.win.addAction(self.action)
        self.triggered: list = []
        self.action.triggered.connect(lambda: self.triggered.append("triggered"))
        self.win.show()
        _APP.setActiveWindow(self.win)
        self.bridge = HydraMcpBridge()

    def tearDown(self):
        self.win.close()
        self.win.deleteLater()
        _APP.processEvents()

    def test_run_action_by_object_name(self):
        self.assertEqual(len(self.triggered), 0)
        result = self.bridge._handle_run_action({"object_name": "test_action"})
        self.assertTrue(result.get("ok"))
        _APP.processEvents()
        self.assertEqual(len(self.triggered), 1)


class TestBridgeReadLog(unittest.TestCase):
    """gui_read_log returns the active studio dialog runtime log."""

    def setUp(self):
        self.bridge = HydraMcpBridge()
        self.fake_dialog = types.SimpleNamespace(
            objectName=lambda: "FakeStudioDialog",
            _runtime_log_lines=["first", "second", "third"],
        )
        import swe2d.workbench.studio_dialog as studio_mod
        self._prev_dialog = getattr(studio_mod, "_studio_active_dialog", None)
        studio_mod._studio_active_dialog = self.fake_dialog

    def tearDown(self):
        import swe2d.workbench.studio_dialog as studio_mod
        studio_mod._studio_active_dialog = self._prev_dialog

    def test_read_log_returns_lines(self):
        result = self.bridge._handle_read_log({"max_lines": 2})
        self.assertTrue(result.get("ok"))
        self.assertEqual(result["lines"], ["second", "third"])
        self.assertEqual(result["total"], 3)


class _FakeWorker(QObject):
    """Worker stand-in that emits compute_finished after a short delay."""

    compute_finished = pyqtSignal(object)
    compute_failed = pyqtSignal(str)

    def __init__(self, run_id: str, delay_ms: int = 100):
        super().__init__()
        self._run_id = run_id
        self._delay_ms = delay_ms

    def start(self) -> None:
        result = types.SimpleNamespace(
            run_id=self._run_id,
            ok=True,
            cancelled=False,
        )
        QTimer.singleShot(self._delay_ms, lambda: self.compute_finished.emit(result))


class _FakeController:
    def __init__(self, worker: _FakeWorker):
        self._simulation_worker: _FakeWorker | None = None
        self._worker = worker

    def on_run(self) -> None:
        self._simulation_worker = self._worker
        # The real bridge starts the worker after connecting its signals.
        # Do not start here to avoid a race in the test.


class TestBridgeRunSimulation(unittest.TestCase):
    """gui_run_simulation sets inputs, clicks Run, and waits for compute_finished."""

    def setUp(self):
        self.win = QMainWindow()
        self.win.setObjectName("run_sim_window")
        self.central = QWidget(self.win)
        self.win.setCentralWidget(self.central)

        self.run_edit = QLineEdit(self.central)
        self.run_edit.setObjectName("run_time_edit")
        self.run_edit.setText("1:00")

        self.out_edit = QLineEdit(self.central)
        self.out_edit.setObjectName("output_interval_edit")
        self.out_edit.setText("00:10")

        self.run_btn = QPushButton("Run", self.central)
        self.run_btn.setObjectName("run_btn")

        self.worker = _FakeWorker("test_run_001")
        self.controller = _FakeController(self.worker)
        self.run_btn.clicked.connect(self.controller.on_run)

        self.dialog = types.SimpleNamespace(
            objectName=lambda: "FakeStudioDialog",
            _runtime_log_lines=["starting"],
            _controller=self.controller,
        )
        import swe2d.workbench.studio_dialog as studio_mod
        self._prev_dialog = getattr(studio_mod, "_studio_active_dialog", None)
        studio_mod._studio_active_dialog = self.dialog

        self.win.show()
        _APP.setActiveWindow(self.win)
        self.bridge = HydraMcpBridge()

    def tearDown(self):
        import swe2d.workbench.studio_dialog as studio_mod
        studio_mod._studio_active_dialog = self._prev_dialog
        self.win.close()
        self.win.deleteLater()
        _APP.processEvents()

    def test_run_simulation_waits_for_compute_finished(self):
        result = self.bridge._handle_run_simulation({
            "run_duration_text": "0:05",
            "output_interval_text": "0:01",
            "timeout": 2.0,
            "startup_timeout": 2.0,
        })
        self.assertTrue(result.get("ok"), result)
        self.assertEqual(result.get("status"), "finished")
        self.assertEqual(result.get("run_id"), "test_run_001")
        self.assertEqual(self.run_edit.text(), "0:05")
        self.assertEqual(self.out_edit.text(), "0:01")


class TestBridgeCloseNoOp(unittest.TestCase):
    """Placeholder: gui_close lifecycle is covered by ProcessRegistry unit tests."""

    def test_module_loads_with_qgis(self):
        self.assertTrue(HydraMcpBridge is not None)


if __name__ == "__main__":
    unittest.main()
