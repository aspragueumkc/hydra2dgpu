#!/usr/bin/env python3
"""Tests for workbench misc modules (Task E.2 of
docs/plans/2026-08-02-gui-test-coverage.md):

- ``swe2d/workbench/workbench_api.py`` — view-plotter registry
  (``register_view_plotter`` / ``get_view_plotter`` / ``list_view_plotters``
  / ``clear_view_plotters``).
- ``swe2d/workbench/startup_state.py`` — ``initialize_workbench_startup_state``.
- ``swe2d/workbench/post_init.py`` — ``run_workbench_post_bootstrap_setup``.
- ``swe2d/workbench/dialogs/widget_inspector.py`` — ``arm`` / ``_ClickWatcher``
  / ``_probe``.

All tests run against the real headless QGIS harness (tests/qgis_real_env.py).
Patterns P1 (pure round-trip) and P2 (drive real widgets) per
docs/specs/2026-08-02-gui-test-coverage-design.md §4.
"""

from __future__ import annotations

import concurrent.futures
import importlib.util
import os
import sys
import unittest

# Ensure repo root is on sys.path for all discovery modes
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path and os.path.isdir(_REPO_ROOT):
    sys.path.insert(0, _REPO_ROOT)

from tests.qgis_real_env import (
    delete_widgets_now,
    ensure_qgis_app,
    requires_qgis,
)


def _wait_until(cond, timeout_ms: int = 8000, step_ms: int = 25) -> bool:
    """Pump the event loop until *cond* is true or the timeout expires."""
    from qgis.PyQt.QtTest import QTest

    waited = 0
    while waited < timeout_ms:
        if cond():
            return True
        QTest.qWait(step_ms)
        waited += step_ms
    return bool(cond())


# ---------------------------------------------------------------------------
# workbench_api — view plotter registry (P1)
# ---------------------------------------------------------------------------


@requires_qgis
class TestViewPlotterRegistry(unittest.TestCase):
    """Registry round-trip for the module-level ``_VIEW_PLOTTERS`` dict.

    The registry is module-global, so each test snapshots and restores it
    to avoid polluting other suites.
    """

    def setUp(self) -> None:
        from swe2d.workbench import workbench_api

        self._api = workbench_api
        self._saved = dict(workbench_api._VIEW_PLOTTERS)
        workbench_api.clear_view_plotters()

    def tearDown(self) -> None:
        self._api._VIEW_PLOTTERS.clear()
        self._api._VIEW_PLOTTERS.update(self._saved)

    def test_register_get_list_round_trip(self) -> None:
        def plotter(fig, mesh_data, result_data, mode, h_min) -> None:
            return None

        self._api.register_view_plotter("Depth", plotter)
        self.assertIs(self._api.get_view_plotter("Depth"), plotter)
        self.assertIn("Depth", self._api.list_view_plotters())

    def test_get_unregistered_name_returns_none(self) -> None:
        # Contract: get_view_plotter returns None for unknown names.
        self.assertIsNone(self._api.get_view_plotter("no such view mode"))

    def test_re_register_same_name_overwrites(self) -> None:
        # Contract (workbench_api.py:150 — plain dict assignment): a second
        # registration under the same name silently overwrites the first.
        def first(fig, mesh_data, result_data, mode, h_min) -> None:
            return None

        def second(fig, mesh_data, result_data, mode, h_min) -> None:
            return None

        self._api.register_view_plotter("Mesh", first)
        self._api.register_view_plotter("Mesh", second)
        self.assertIs(self._api.get_view_plotter("Mesh"), second)
        self.assertEqual(self._api.list_view_plotters().count("Mesh"), 1)

    def test_clear_view_plotters_empties_registry(self) -> None:
        def plotter(fig, mesh_data, result_data, mode, h_min) -> None:
            return None

        self._api.register_view_plotter("A", plotter)
        self._api.register_view_plotter("B", plotter)
        self._api.clear_view_plotters()
        self.assertEqual(self._api.list_view_plotters(), [])


# ---------------------------------------------------------------------------
# startup_state — initialize_workbench_startup_state (P2)
# ---------------------------------------------------------------------------


def _make_startup_dialog_class():
    """Build a minimal REAL QDialog carrying the one hook the initializer
    connects to (``_poll_topology_mesh_future``).  Not a mock — a real
    QObject subclass, mirroring what the production dialog provides."""
    from qgis.PyQt.QtWidgets import QDialog

    class _StartupDialog(QDialog):
        def __init__(self) -> None:
            super().__init__()
            self.poll_calls = 0

        def _poll_topology_mesh_future(self) -> None:
            self.poll_calls += 1

    return _StartupDialog


@requires_qgis
class TestInitializeWorkbenchStartupState(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        ensure_qgis_app()

    def setUp(self) -> None:
        self._dialog = _make_startup_dialog_class()()

    def tearDown(self) -> None:
        # Shut down the real ThreadPoolExecutor and timer the initializer
        # installed so they don't leak into other suites.
        pool = getattr(self._dialog, "_topology_mesh_thread_pool", None)
        if pool is not None:
            pool.shutdown(wait=False)
        timer = getattr(self._dialog, "_topology_mesh_timer", None)
        if timer is not None:
            timer.stop()
        delete_widgets_now(self._dialog)

    def _run_initializer(self) -> None:
        from qgis.PyQt import QtCore

        from swe2d.workbench.startup_state import initialize_workbench_startup_state
        from swe2d.workbench.studio_dialog import _try_import_matplotlib_qt

        initialize_workbench_startup_state(
            self._dialog,
            qtcore_module=QtCore,
            concurrent_futures_module=concurrent.futures,
            try_import_matplotlib_qt=_try_import_matplotlib_qt,
        )

    def test_documented_state_fields(self) -> None:
        from qgis.PyQt.QtCore import QTimer

        from swe2d import units as _u

        self._run_initializer()
        dlg = self._dialog

        # Scalar startup state (startup_state.py:22-46)
        self.assertIsNone(dlg._backend)
        self.assertIs(dlg._cancel_requested, False)
        self.assertIsNone(dlg._mesh_data)
        self.assertIsNone(dlg._result_data)
        self.assertEqual(dlg._snapshot_mesh_fingerprint, "")
        self.assertEqual(dlg._model_gpkg_path, "")
        self.assertEqual(dlg._runtime_log_lines, [])
        self.assertEqual(dlg._unit_system, "SI")
        self.assertEqual(dlg._length_unit_name, "m")
        self.assertEqual(dlg._gravity, _u.gravity())
        self.assertEqual(dlg._topology_mesh_run_mode, "full")
        self.assertIs(dlg._topology_mesh_auto_fallback_used, False)
        self.assertEqual(dlg._topology_mesh_options, {})
        self.assertIs(dlg._initial_layer_restore_pending, True)
        self.assertIs(dlg._overlay_no_data_warned, False)

        # Latest-run bookkeeping starts empty
        self.assertEqual(dlg._line_results_latest_run_id, "")
        self.assertEqual(dlg._coupling_results_latest_db_path, "")
        self.assertEqual(dlg._run_log_latest_run_id, "")

        # Real timer + thread pool installed
        self.assertIsInstance(dlg._topology_mesh_timer, QTimer)
        self.assertEqual(dlg._topology_mesh_timer.interval(), 120)
        self.assertIsInstance(
            dlg._topology_mesh_thread_pool, concurrent.futures.ThreadPoolExecutor
        )
        self.assertIsNone(dlg._topology_mesh_future)

        # Timeout contract: env override honoured, clamped to >= 30 s
        self.assertGreaterEqual(dlg._topology_mesh_timeout_sec, 30.0)
        self.assertEqual(
            dlg._topology_mesh_active_timeout_sec, dlg._topology_mesh_timeout_sec
        )

        # Topology-mesh state dict mirrors the scalar fields
        state = dlg._topology_mesh_state
        self.assertIsNone(state["topology_mesh_future"])
        self.assertIsNone(state["topology_mesh_started_at"])
        self.assertEqual(state["topology_mesh_poll_count"], 0)
        self.assertIs(state["topology_mesh_auto_fallback_used"], False)
        self.assertEqual(state["topology_mesh_progress_last_seq"], -1)

        # Detached-dialog bookkeeping + run-component slots
        self.assertEqual(dlg._detached_panel_dialogs, [])
        self.assertIsNone(dlg._run_orchestrator)
        self.assertIsNone(dlg._run_controller)
        self.assertIsNone(dlg._backend_initializer)
        self.assertIsNone(dlg._run_finalizer)
        self.assertIsNone(dlg._run_lifecycle)
        self.assertIsNone(dlg._last_run_request)

    @unittest.skipUnless(
        importlib.util.find_spec("matplotlib") is not None,
        "matplotlib not installed",
    )
    def test_matplotlib_import_hook_populated(self) -> None:
        self._run_initializer()
        dlg = self._dialog
        self.assertTrue(dlg._have_mpl)
        self.assertIsNotNone(dlg._FigureCanvas)
        self.assertIsNotNone(dlg._Figure)

    def test_timer_timeout_wired_to_poll_method(self) -> None:
        # Behaviorally verify the QTimer connection: fire the timer's
        # timeout signal and assert the dialog hook ran.
        self._run_initializer()
        self.assertEqual(self._dialog.poll_calls, 0)
        self._dialog._topology_mesh_timer.timeout.emit()
        self.assertEqual(self._dialog.poll_calls, 1)


# ---------------------------------------------------------------------------
# post_init — run_workbench_post_bootstrap_setup (P2)
# ---------------------------------------------------------------------------


def _make_post_bootstrap_dialog_class():
    """Minimal REAL QDialog providing the four hooks post_init calls.

    The hooks record invocations so tests can assert call counts — these
    are real methods on a real widget, not stand-ins for Qgs* types.
    """
    from qgis.PyQt.QtWidgets import QDialog

    class _PostBootstrapDialog(QDialog):
        def __init__(self) -> None:
            super().__init__()
            self.logs = []
            self.hook_calls = {
                "workbench_state": 0,
                "save_state": 0,
                "unit_system": 0,
            }

        def _connect_project_workbench_state_signals(self) -> None:
            self.hook_calls["workbench_state"] += 1

        def _connect_project_save_state_signals(self) -> None:
            self.hook_calls["save_state"] += 1

        def _update_unit_system_from_crs(self) -> None:
            self.hook_calls["unit_system"] += 1

        def _log(self, msg: str) -> None:
            self.logs.append(str(msg))

    return _PostBootstrapDialog


@requires_qgis
class TestRunWorkbenchPostBootstrapSetup(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        ensure_qgis_app()

    def setUp(self) -> None:
        self._dialog = _make_post_bootstrap_dialog_class()()

    def tearDown(self) -> None:
        delete_widgets_now(self._dialog)

    def _run_setup(self) -> None:
        from swe2d.mesh.gmsh_backend import _gmsh_available
        from swe2d.runtime.backend import swe2d_gpu_available
        from swe2d.workbench.post_init import run_workbench_post_bootstrap_setup

        run_workbench_post_bootstrap_setup(
            self._dialog,
            swe2d_gpu_available_fn=swe2d_gpu_available,
            gmsh_available_fn=_gmsh_available,
        )

    def test_documented_post_bootstrap_state(self) -> None:
        import numpy as np

        from swe2d.runtime.backend import swe2d_gpu_available

        dlg = self._dialog
        dlg._initial_layer_restore_pending = True  # set by startup_state
        self._run_setup()

        # Signal hooks invoked exactly once each
        self.assertEqual(dlg.hook_calls["workbench_state"], 1)
        self.assertEqual(dlg.hook_calls["save_state"], 1)
        self.assertEqual(dlg.hook_calls["unit_system"], 1)

        # Layer-restore gate cleared
        self.assertIs(dlg._initial_layer_restore_pending, False)

        # Two log lines emitted: GPU backend + Gmsh availability
        self.assertEqual(len(dlg.logs), 2)
        expected_gpu = "available" if swe2d_gpu_available() else "unavailable"
        self.assertEqual(dlg.logs[0], f"GPU backend: {expected_gpu}")
        self.assertTrue(dlg.logs[1].startswith("Meshing: Gmsh "))

        # Overlay state initialized (post_init.py:32-59)
        self.assertIsNone(dlg._results_panel)
        self.assertIsNone(dlg._high_perf_canvas_overlay_item)
        self.assertIs(dlg._high_perf_canvas_overlay_enabled, False)
        for attr, dtype in (
            ("_high_perf_overlay_cell_x", np.float64),
            ("_high_perf_overlay_cell_y", np.float64),
            ("_high_perf_overlay_cell_bed", np.float64),
            ("_high_perf_overlay_node_x", np.float64),
            ("_high_perf_overlay_node_y", np.float64),
            ("_high_perf_overlay_cell_nodes", np.int32),
        ):
            arr = getattr(dlg, attr)
            self.assertIsInstance(arr, np.ndarray, attr)
            self.assertEqual(arr.dtype, dtype, attr)
            self.assertEqual(arr.size, 0, attr)
        self.assertIsNone(dlg._velocity_vectors_layer_id)
        self.assertEqual(dlg._velocity_overlay_sources, [])
        self.assertEqual(dlg._velocity_overlay_layer_ids, {})
        self.assertEqual(dlg._velocity_overlay_refresh_token, 0)
        self.assertEqual(dlg._velocity_overlay_perf_log_every, 30)
        self.assertEqual(dlg._streamline_overlay_perf_log_every, 30)
        self.assertIsNone(dlg._three_d_patch_surface_layer_id)
        self.assertIsNone(dlg._three_d_patch_last_spec)

    def test_double_call_is_safe(self) -> None:
        # Contract: the function is a plain state initializer with no
        # once-guard; a second call must not raise and must leave the
        # documented state consistent (hooks simply re-run).
        dlg = self._dialog
        self._run_setup()
        self._run_setup()
        self.assertEqual(dlg.hook_calls["workbench_state"], 2)
        self.assertEqual(dlg.hook_calls["save_state"], 2)
        self.assertEqual(dlg.hook_calls["unit_system"], 2)
        self.assertIs(dlg._initial_layer_restore_pending, False)
        self.assertIsNone(dlg._results_panel)
        self.assertIs(dlg._high_perf_canvas_overlay_enabled, False)
        self.assertEqual(len(dlg.logs), 4)


# ---------------------------------------------------------------------------
# widget_inspector — arm / _ClickWatcher / _probe (P2)
# ---------------------------------------------------------------------------


@requires_qgis
class TestWidgetInspector(unittest.TestCase):
    """Drive the real polling click-watcher against a real widget.

    ``_probe`` reports through ``QMessageBox.information`` (a modal static
    call); the test substitutes a plain recording class for the module's
    ``QMessageBox`` name so the report text can be asserted and the test
    never blocks on a modal dialog.  This is a capture shim for a blocking
    dialog, not a mock of any Qgs* type.
    """

    @classmethod
    def setUpClass(cls) -> None:
        ensure_qgis_app()

    def setUp(self) -> None:
        from qgis.PyQt.QtWidgets import QPushButton, QWidget

        from swe2d.workbench.dialogs import widget_inspector as wi

        self._wi = wi
        # Reset module global so no stale poller from another suite leaks in.
        if wi._POLLER is not None:
            wi._POLLER._timer.stop()
            wi._POLLER = None

        # Real widget under test.  QCursor.setPos works on the offscreen
        # QPA (verified empirically), so park the window at a fixed spot
        # and move the cursor over the button's center.
        self._window = QWidget()
        self._window.setObjectName("inspector_test_window")
        self._button = QPushButton("probe me", self._window)
        self._button.setObjectName("inspector_probe_target")
        self._button.setGeometry(0, 0, 220, 120)
        self._window.setGeometry(100, 100, 220, 120)
        self._window.show()
        from qgis.PyQt.QtGui import QCursor
        from qgis.PyQt.QtWidgets import QApplication

        QCursor.setPos(self._button.mapToGlobal(self._button.rect().center()))
        QApplication.processEvents()

        # Capture shim for the modal report dialog.
        self._captured = []
        wi_mod = self._wi

        class _CaptureMessageBox:
            @staticmethod
            def information(parent, title, text):
                self._captured.append((title, text))

        self._orig_msgbox = wi_mod.QMessageBox
        wi_mod.QMessageBox = _CaptureMessageBox

    def tearDown(self) -> None:
        wi = self._wi
        wi.QMessageBox = self._orig_msgbox
        if wi._POLLER is not None:
            wi._POLLER._timer.stop()
            wi._POLLER = None
        self._window.close()
        delete_widgets_now(self._window)

    def test_arm_creates_click_watcher_after_settle_delay(self) -> None:
        wi = self._wi
        wi.arm()
        # arm() schedules _start via QTimer.singleShot(800, ...) — wait for it.
        self.assertTrue(
            _wait_until(lambda: wi._POLLER is not None, timeout_ms=3000),
            "arm() never created the click-watcher poller",
        )
        poller = wi._POLLER
        self.assertIsInstance(poller, wi._ClickWatcher)
        self.assertTrue(poller._timer.isActive())
        self.assertEqual(poller._timer.interval(), 80)
        # With no button held, the state machine advances IDLE -> ARMED.
        self.assertTrue(
            _wait_until(lambda: poller._state == wi._ClickWatcher.STATE_ARMED),
            "click watcher never reached STATE_ARMED",
        )

    def test_armed_click_probes_real_widget_identity(self) -> None:
        """Deterministic drive of the click-watcher's state machine.

        Instead of waiting on the 80 ms QTimer to fire and observe Qt's
        global mouse-button state, this test drives ``_poll()`` directly
        after every deterministic QTest mouse event.  The integration
        check that the timer-driven path still works is kept separately
        in ``test_armed_click_timer_driven_path``.
        """
        from qgis.PyQt import QtWidgets
        from qgis.PyQt.QtCore import Qt
        from qgis.PyQt.QtTest import QTest

        wi = self._wi
        wi.arm()
        # Wait once for the arm() settle delay (800 ms) to spawn _POLLER.
        self.assertTrue(
            _wait_until(
                lambda: wi._POLLER is not None
                and wi._POLLER._state == wi._ClickWatcher.STATE_ARMED,
                timeout_ms=3000,
            ),
            "click watcher never armed",
        )
        poller = wi._POLLER

        # Deterministic QTest press — drives the global Qt button state.
        QTest.mousePress(self._button, Qt.LeftButton)
        # Drive _poll() directly: the next observation advances
        # ARMED -> PRESSED without depending on the 80 ms QTimer firing.
        poller._poll()
        self.assertEqual(poller._state, wi._ClickWatcher.STATE_PRESSED)

        # Deterministic QTest release — same story.
        QTest.mouseRelease(self._button, Qt.LeftButton)
        poller._poll()
        self.assertFalse(
            poller._timer.isActive(),
            "poller must stop its own timer on the PRESSED→release transition",
        )
        # _probe is scheduled via QTimer.singleShot(0, _probe); one
        # processEvents pass delivers it.
        QtWidgets.QApplication.processEvents()
        self.assertTrue(
            len(self._captured) > 0,
            "_probe must fire after the deterministic release-driven poll",
        )

        title, text = self._captured[0]
        self.assertEqual(title, "Widget Inspector")
        self.assertIn("Class: QPushButton", text)
        self.assertIn('ObjectName: "inspector_probe_target"', text)

    def test_armed_click_timer_driven_path(self) -> None:
        """Integration check: the timer-driven poll loop does observe the
        state transitions without any direct ``_poll()`` calls.

        Kept as a small sanity check so a regression that breaks the
        80 ms QTimer (the production loop path) is still caught.  The
        deterministic direct-poll path above is the primary contract.
        """
        from qgis.PyQt.QtCore import Qt
        from qgis.PyQt.QtTest import QTest

        wi = self._wi
        wi.arm()
        self.assertTrue(
            _wait_until(
                lambda: wi._POLLER is not None
                and wi._POLLER._state == wi._ClickWatcher.STATE_ARMED,
                timeout_ms=3000,
            ),
            "click watcher never armed",
        )
        poller = wi._POLLER

        QTest.mousePress(self._button, Qt.LeftButton)
        try:
            self.assertTrue(
                _wait_until(
                    lambda: poller._state == wi._ClickWatcher.STATE_PRESSED,
                    timeout_ms=2000,
                ),
                "timer-driven poll loop never observed the press",
            )
            QTest.mouseRelease(self._button, Qt.LeftButton)
            self.assertTrue(
                _wait_until(
                    lambda: len(self._captured) > 0, timeout_ms=3000
                ),
                "timer-driven release→_probe never reported",
            )
        finally:
            # Cleanup: ensure no button state lingers into the next test.
            QTest.mouseRelease(self._button, Qt.LeftButton)


if __name__ == "__main__":
    unittest.main()
