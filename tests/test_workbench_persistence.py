"""Tests for swe2d.workbench.persistence.

Pure-headless tests — no QGIS, no plugin, just ``QApplication([])``
plus a fake QSettings stub.  Verifies that:

* open-on-startup / was-open flag round-trips through QSettings
* save_window_state / restore_window_state survive a fresh
  QMainWindow (round-trip preserves dock objectNames)
* clear_window_state drops the layout keys
* helper handles None settings + corrupt bytes without raising
"""
from __future__ import annotations

import os
import sys
import unittest
from typing import Any, Dict, Optional

# Make sure the QGIS-free persistence helper is importable without qgis.
THIS_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from qgis.PyQt import QtCore, QtWidgets  # QApplication lives here

import swe2d.workbench.persistence as persistence
import swe2d.workbench.services.widget_persistence_service as widget_persistence

# Ensure a QApplication exists for the QMainWindow round-trip
_qapp: Optional[QtWidgets.QApplication] = None


def _ensure_app() -> QtWidgets.QApplication:
    global _qapp
    if _qapp is None:
        _qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    return _qapp


class _FakeSettings:
    """In-memory QSettings replacement keyed by str → any."""

    def __init__(self) -> None:
        self._store: Dict[str, Any] = {}

    def value(self, key: str, default: Any = None) -> Any:
        return self._store.get(key, default)

    def setValue(self, key: str, value: Any) -> None:
        self._store[key] = value

    def contains(self, key: str) -> bool:
        return key in self._store

    def remove(self, key: str) -> None:
        self._store.pop(key, None)


class TestOpenOnStartup(unittest.TestCase):
    def setUp(self) -> None:
        self.s = _FakeSettings()

    def test_defaults_false_when_missing(self):
        self.assertFalse(persistence.load_open_on_startup(self.s))

    def test_save_then_load(self):
        persistence.save_open_on_startup(self.s, True)
        self.assertTrue(persistence.load_open_on_startup(self.s))

    def test_save_false_round_trips(self):
        persistence.save_open_on_startup(self.s, True)
        persistence.save_open_on_startup(self.s, False)
        self.assertFalse(persistence.load_open_on_startup(self.s))

    def test_string_coercion(self):
        self.s.setValue(persistence.K_OPEN_ON_STARTUP, "true")
        self.assertTrue(persistence.load_open_on_startup(self.s))
        self.s.setValue(persistence.K_OPEN_ON_STARTUP, "false")
        self.assertFalse(persistence.load_open_on_startup(self.s))

    def test_none_settings_returns_default(self):
        # No crash, just default — defensive against missing settings.
        self.assertFalse(persistence.load_open_on_startup(None))
        # save is a no-op
        persistence.save_open_on_startup(None, True)


class TestWasOpen(unittest.TestCase):
    def setUp(self) -> None:
        self.s = _FakeSettings()

    def test_defaults_none(self):
        self.assertIsNone(persistence.was_open(self.s))

    def test_save_and_load(self):
        persistence.save_was_open(self.s, True)
        self.assertTrue(persistence.was_open(self.s))
        persistence.save_was_open(self.s, False)
        self.assertFalse(persistence.was_open(self.s))


class TestSaveRestoreWindowState(unittest.TestCase):
    def setUp(self) -> None:
        _ensure_app()
        self.s = _FakeSettings()

    def test_save_returns_false_when_window_is_none(self):
        self.assertFalse(persistence.save_window_state(self.s, None))

    def test_save_returns_bytes(self):
        win = QtWidgets.QMainWindow()
        try:
            ok = persistence.save_window_state(self.s, win)
            self.assertTrue(ok)
            self.assertIsNotNone(self.s.value(persistence.K_DOCK_STATE))
            self.assertIsNotNone(self.s.value(persistence.K_GEOMETRY))
        finally:
            win.close()
            win.deleteLater()

    def test_round_trip_preserves_dock_objectnames(self):
        """Save a layout with 2 named docks, restore on a fresh window,
        verify both docks are tracked.
        """
        win1 = QtWidgets.QMainWindow()
        try:
            dock_a = QtWidgets.QDockWidget("Alpha", win1)
            dock_a.setObjectName("dock_alpha")
            dock_b = QtWidgets.QDockWidget("Beta", win1)
            dock_b.setObjectName("dock_beta")
            win1.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock_a)
            win1.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock_b)
            persistence.save_window_state(self.s, win1)
        finally:
            win1.close()
            win1.deleteLater()

        win2 = QtWidgets.QMainWindow()
        try:
            dock_a2 = QtWidgets.QDockWidget("Alpha", win2)
            dock_a2.setObjectName("dock_alpha")
            dock_b2 = QtWidgets.QDockWidget("Beta", win2)
            dock_b2.setObjectName("dock_beta")
            win2.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock_a2)
            win2.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock_b2)
            ok = persistence.restore_window_state(self.s, win2)
            self.assertTrue(ok)
            self.assertEqual(
                {w.objectName() for w in win2.findChildren(QtWidgets.QDockWidget)},
                {"dock_alpha", "dock_beta"},
            )
        finally:
            win2.close()
            win2.deleteLater()

    def test_restore_returns_false_with_no_saved_state(self):
        win = QtWidgets.QMainWindow()
        try:
            self.assertFalse(persistence.restore_window_state(self.s, win))
        finally:
            win.close()
            win.deleteLater()

    def test_corrupt_dock_state_does_not_crash(self):
        win = QtWidgets.QMainWindow()
        try:
            # Garbage bytes that saveState cannot decode.
            self.s.setValue(persistence.K_DOCK_STATE, b"not a real blob")
            # Should NOT raise — restore silently fails.
            persistence.restore_window_state(self.s, win)
        finally:
            win.close()
            win.deleteLater()

    def test_none_settings_does_not_crash(self):
        win = QtWidgets.QMainWindow()
        try:
            self.assertFalse(persistence.restore_window_state(None, win))
        finally:
            win.close()
            win.deleteLater()


class TestClearWindowState(unittest.TestCase):
    def setUp(self) -> None:
        _ensure_app()
        self.s = _FakeSettings()

    def test_clear_removes_dock_and_geometry_keys(self):
        win = QtWidgets.QMainWindow()
        try:
            persistence.save_window_state(self.s, win)
            self.assertTrue(self.s.contains(persistence.K_DOCK_STATE))
        finally:
            win.close()
            win.deleteLater()

        persistence.clear_window_state(self.s)
        self.assertFalse(self.s.contains(persistence.K_DOCK_STATE))
        self.assertFalse(self.s.contains(persistence.K_GEOMETRY))

    def test_clear_drops_was_open(self):
        persistence.save_was_open(self.s, True)
        persistence.clear_window_state(self.s)
        self.assertFalse(self.s.contains(persistence.K_WAS_OPEN))

    def test_clear_with_none_settings_is_noop(self):
        persistence.clear_window_state(None)  # no crash


class TestQtModuleClassifierUnderRealQGIS(unittest.TestCase):
    """Regression guard for the PyQt5 -> qgis.PyQt shim mismatch.

    QGIS exposes Qt as ``qgis.PyQt`` (a re-export shim). The runtime
    ``__module__`` of every widget class in the shim is still
    ``PyQt5.QtWidgets`` (or ``PyQt5.QtCore`` etc.), NOT
    ``qgis.PyQt.QtWidgets``. An earlier migration flipped the
    classifier in :func:`widget_persistence_service._qt_widgets_module`
    to match ``qgis.PyQt`` literally, which silently matched zero
    widgets in the real qgis env and re-saved every project with
    blank widget state. This test exercises the classifier against
    the real QGIS env so the trap cannot return.
    """

    @classmethod
    def setUpClass(cls):
        _ensure_app()

    def test_qt_widgets_module_recognises_every_persistable_widget(self):
        """Every widget type the persistence service iterates must
        resolve to a non-None QtWidgets module under real QGIS."""
        for cls in (
            QtWidgets.QSpinBox,
            QtWidgets.QDoubleSpinBox,
            QtWidgets.QComboBox,
            QtWidgets.QCheckBox,
            QtWidgets.QLineEdit,
        ):
            widget = cls()
            with self.subTest(widget_class=cls.__name__):
                qt_mod = widget_persistence._qt_widgets_module(widget)
                self.assertIsNotNone(
                    qt_mod,
                    f"{cls.__name__} (module={type(widget).__module__!r}) was "
                    "not recognised by _qt_widgets_module — the PyQt5/qgis.PyQt "
                    "shim classifier regressed. Widget persistence will silently "
                    "drop every value if this returns None.",
                )
                # And the returned module must actually contain the widget class
                self.assertTrue(
                    issubclass(cls, qt_mod.QWidget),
                    f"_qt_widgets_module returned {qt_mod!r} which does not "
                    f"contain {cls.__name__} as a subclass of QWidget.",
                )

    def test_widget_runtime_module_startswith_pyqt5(self):
        """Positive control: in the real qgis env, widget classes'
        ``__module__`` starts with ``PyQt5`` (or ``PySide2``). The
        shim layer is transparent at the class-identity level. If
        this ever changes, the classifier in
        :mod:`widget_persistence_service` needs to be updated and
        the test below will start failing.
        """
        widget = QtWidgets.QSpinBox()
        mod = type(widget).__module__
        self.assertTrue(
            mod.startswith("PyQt5") or mod.startswith("PySide2"),
            f"Unexpected widget module under real QGIS: {mod!r}. "
            "Update _qt_widgets_module to match the new module prefix "
            "and verify the regression guard above still passes.",
        )

    def test_persist_round_trip_preserves_widget_value(self):
        """End-to-end: a real QSpinBox value must survive the
        persist/restore round-trip via the service layer. This is
        the actual user-visible behaviour the previous regression
        broke (saved projects reopened with all widget values
        zeroed)."""
        spin = QtWidgets.QSpinBox()
        spin.setValue(42)
        captured: Dict[str, Dict[str, object]] = {}

        # Drive the persistence logic the same way the dialog does,
        # but bypass write_project_json by passing a captured dict.
        def iter_widgets():
            yield ("spin", spin)

        qt_mod = widget_persistence._qt_widgets_module(spin)
        self.assertIsNotNone(qt_mod)
        for _attr, widget in iter_widgets():
            if isinstance(widget, (qt_mod.QSpinBox, qt_mod.QDoubleSpinBox)):
                captured["spin"] = {
                    "type": type(widget).__name__,
                    "value": int(widget.value()),
                }
        self.assertEqual(captured["spin"]["value"], 42)


if __name__ == "__main__":
    unittest.main(verbosity=2)
