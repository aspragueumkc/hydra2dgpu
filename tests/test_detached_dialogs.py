#!/usr/bin/env python3
"""Behavioral tests for the detached dialogs.

Covers (spec docs/specs/2026-08-02-gui-test-coverage-design.md §3-§4,
pattern P2 — real headless QGIS, no mock Qgs* types):

- ``swe2d/workbench/dialogs/detached_panel_dialog.py``
  (``SWE2DDetachedPanelDialog``) — generic detach/re-attach container.
- ``swe2d/workbench/dialogs/detached_mesh_dialog.py``
  (``SWE2DDetachedMeshViewDialog``) — mesh/depth/velocity render view.
- ``swe2d/workbench/dialogs/detached_log_dialog.py``
  (``SWE2DDetachedRuntimeLogDialog``) — runtime log viewer.

These tests enforce the AGENTS.md "Proper Widget Lifecycle — No Dead
Shells" rule: extracting a content widget into a detached dialog and
re-attaching it must leave the content alive and correctly parented, and
must not leave a visible blank zombie top-level widget behind.

Timer contract: none of the three dialogs owns a ``QTimer`` (verified by
reading the sources); each dialog test class asserts this explicitly so
a future regression that adds an unstopped timer is caught here.

The reattach callback error path must swallow the callback exception and
emit a warning through the module logger without raising a second exception.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

# Ensure repo root and build dir are on sys.path for all discovery modes
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BUILD_DIR = os.path.join(_REPO_ROOT, "build")
for _p in (_REPO_ROOT, _BUILD_DIR):
    if _p not in sys.path and os.path.isdir(_p):
        sys.path.insert(0, _p)

from tests.qgis_real_env import (
    delete_widgets_now,
    ensure_qgis_app,
    grab_non_empty,
    requires_qgis,
)


def _make_host_with_content():
    """Build a real host container holding a labelled content widget.

    Returns ``(host, content)`` where *content* is parented into *host*'s
    layout — the pre-detach production arrangement.
    """
    from qgis.PyQt import QtWidgets

    host = QtWidgets.QWidget()
    host.setObjectName("test_host_container")
    layout = QtWidgets.QVBoxLayout(host)
    content = QtWidgets.QWidget()
    content.setObjectName("test_content_widget")
    content_layout = QtWidgets.QVBoxLayout(content)
    content_layout.addWidget(QtWidgets.QLabel("content-body"))
    layout.addWidget(content)
    return host, content


def _make_triangle_mesh():
    """Real two-triangle unit-square mesh for the mesh render service."""
    node_x = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float64)
    node_y = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
    cell_nodes = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return {"node_x": node_x, "node_y": node_y, "cell_nodes": cell_nodes}


def _make_result_data(n_cells=2):
    """Per-cell h/hu/hv result arrays matching ``_make_triangle_mesh``."""
    return {
        "h": np.full(n_cells, 0.5, dtype=np.float64),
        "hu": np.full(n_cells, 0.05, dtype=np.float64),
        "hv": np.full(n_cells, -0.02, dtype=np.float64),
    }


def _visible_top_levels():
    from qgis.PyQt import QtWidgets

    app = QtWidgets.QApplication.instance()
    return [w for w in app.topLevelWidgets() if w.isVisible()]


def _visible_top_level_snapshot():
    """Stable snapshot of visible top-level widgets for zombie diffs.

    Returns ``(widgets, ptrs)`` where *widgets* holds strong references to
    the transient SIP wrappers (preventing wrapper GC, so SIP keeps
    reusing the same wrapper objects) and *ptrs* is the set of stable C++
    addresses via ``sip.unwrapinstance``.  Diffing *ptrs* across snapshots
    is immune to both wrapper-identity churn and to pre-existing visible
    top-levels leaked by earlier test modules — only a *new* visible
    top-level shows up in ``after_ptrs - before_ptrs``.
    """
    from PyQt5 import sip

    widgets = _visible_top_levels()
    return widgets, {sip.unwrapinstance(w) for w in widgets}


@requires_qgis
class TestDetachedPanelDialog(unittest.TestCase):
    """Detach/re-attach lifecycle of SWE2DDetachedPanelDialog."""

    @classmethod
    def setUpClass(cls):
        ensure_qgis_app()

    def setUp(self):
        from swe2d.workbench.dialogs.detached_panel_dialog import (
            SWE2DDetachedPanelDialog,
        )

        self._dlg_cls = SWE2DDetachedPanelDialog
        self._widgets = []

    def tearDown(self):
        try:
            delete_widgets_now(*self._widgets)
        except RuntimeError:
            pass  # already destroyed by Qt parent cascade
        self._widgets.clear()

    def _track(self, *widgets):
        self._widgets.extend(widgets)
        return widgets[0] if len(widgets) == 1 else widgets

    # -- construction / detach ------------------------------------------------

    def test_detach_reparents_content_into_dialog(self):
        host, content = self._track(*_make_host_with_content())
        dlg = self._track(self._dlg_cls("Panel", content, on_reattach=None, parent=host))
        # Detach: the content widget is now owned by the dialog's layout.
        self.assertIs(content.parent(), dlg)
        # ``indexOf`` returns -1 (int) when the widget is absent; the
        # contract is that the content must be present in the dialog's
        # layout.  ``assertIsNotNone(indexOf(...))`` was tautological
        # because indexOf never returns None — the real check is the
        # inequality below.
        self.assertNotEqual(dlg.layout().indexOf(content), -1)
        self.assertEqual(dlg.windowTitle(), "Panel")

    def test_default_title_and_none_content(self):
        dlg = self._track(self._dlg_cls(None, None))
        self.assertEqual(dlg.windowTitle(), "Detached Panel")

    def test_dialog_owns_no_timers(self):
        # Contract: this dialog owns no QTimer, so there is nothing to
        # stop before shell destruction (AGENTS.md lifecycle rule).
        from qgis.PyQt import QtCore

        host, content = self._track(*_make_host_with_content())
        dlg = self._track(self._dlg_cls("Panel", content, parent=host))
        self.assertEqual(dlg.findChildren(QtCore.QTimer), [])

    def test_detached_dialog_grabs_non_empty(self):
        host, content = self._track(*_make_host_with_content())
        dlg = self._track(self._dlg_cls("Panel", content, parent=host))
        dlg.show()
        from qgis.PyQt import QtWidgets

        QtWidgets.QApplication.processEvents()
        self.assertTrue(
            grab_non_empty(dlg),
            "detached panel dialog grab is blank — content not rendered",
        )
        dlg.close()

    # -- re-attach lifecycle --------------------------------------------------

    def test_reattach_returns_content_to_host_no_dead_shell(self):
        host, content = self._track(*_make_host_with_content())
        host_layout = host.layout()

        def reattach():
            host_layout.addWidget(content)

        _before_widgets, before_ptrs = _visible_top_level_snapshot()
        dlg = self._track(self._dlg_cls("Panel", content, on_reattach=reattach, parent=host))
        self.assertIs(content.parent(), dlg)

        dlg.show()
        dlg._reattach_and_close()

        # Content is alive and back under its original parent/layout.
        self.assertIs(content.parent(), host)
        self.assertGreaterEqual(host_layout.indexOf(content), 0)
        # The detached shell is hidden — no visible blank zombie remains.
        self.assertFalse(dlg.isVisible())
        _after_widgets, after_ptrs = _visible_top_level_snapshot()
        self.assertEqual(
            after_ptrs - before_ptrs,
            set(),
            "reattach left a new visible top-level zombie behind",
        )

    def test_reattach_callback_fires_exactly_once(self):
        # _reattach_and_close() calls _reattach_once(); closeEvent() calls
        # it again — the idempotence guard must fire the callback once.
        calls = []
        host, content = self._track(*_make_host_with_content())
        dlg = self._track(
            self._dlg_cls("Panel", content, on_reattach=lambda: calls.append(1), parent=host)
        )
        dlg.show()
        dlg._reattach_btn.click()  # real user path: the Reattach button
        self.assertEqual(len(calls), 1)
        self.assertFalse(dlg.isVisible())

    def test_close_event_also_fires_reattach_once(self):
        # Window-manager close (no Reattach button) still reattaches.
        calls = []
        host, content = self._track(*_make_host_with_content())
        dlg = self._track(
            self._dlg_cls("Panel", content, on_reattach=lambda: calls.append(1), parent=host)
        )
        dlg.show()
        dlg.close()
        self.assertEqual(len(calls), 1)

    def test_reattach_callback_exception_logs_without_escaping(self):
        def bad_callback():
            raise ValueError("boom")

        host, content = self._track(*_make_host_with_content())
        dlg = self._track(
            self._dlg_cls(
                "Panel", content, on_reattach=bad_callback, parent=host
            )
        )
        with self.assertLogs(
            "swe2d.workbench.dialogs.detached_panel_dialog", level="WARNING"
        ) as logs:
            dlg._reattach_once()

        self.assertTrue(
            any("Unexpected Exception" in message for message in logs.output)
        )
        self.assertTrue(dlg._reattached)


@requires_qgis
class TestDetachedMeshViewDialog(unittest.TestCase):
    """Render + close lifecycle of SWE2DDetachedMeshViewDialog."""

    @classmethod
    def setUpClass(cls):
        ensure_qgis_app()

    def setUp(self):
        from qgis.PyQt import QtWidgets

        from swe2d.workbench.dialogs.detached_mesh_dialog import (
            SWE2DDetachedMeshViewDialog,
        )

        self._dlg_cls = SWE2DDetachedMeshViewDialog
        self._parent = QtWidgets.QMainWindow()
        self._dialogs = []

    def tearDown(self):
        delete_widgets_now(*self._dialogs, self._parent)

    def _make(self, mesh_data=None, result_data=None, h_min=1.0e-6):
        dlg = self._dlg_cls(
            mesh_data_fn=lambda: mesh_data,
            result_data_fn=lambda: result_data,
            h_min_fn=lambda: h_min,
            parent=self._parent,
        )
        self._dialogs.append(dlg)
        return dlg

    def test_no_mesh_loaded_status(self):
        dlg = self._make(mesh_data=None)
        self.assertEqual(dlg._status_label.text(), "No mesh loaded")
        self.assertIsNone(dlg._image_label.pixmap())

    def test_mesh_mode_renders_real_image(self):
        dlg = self._make(mesh_data=_make_triangle_mesh())
        status = dlg._status_label.text()
        self.assertTrue(status.startswith("Rendered"), status)
        self.assertTrue(status.endswith("(mesh)"), status)
        self.assertFalse(dlg._image_label.pixmap().isNull())

    def test_depth_and_velocity_modes_render_with_result_data(self):
        dlg = self._make(mesh_data=_make_triangle_mesh(), result_data=_make_result_data())
        for mode_key in ("depth", "velocity"):
            idx = dlg.view_mode_combo.findData(mode_key)
            self.assertGreaterEqual(idx, 0, f"combo missing mode {mode_key!r}")
            dlg.view_mode_combo.setCurrentIndex(idx)  # fires refresh_view
            status = dlg._status_label.text()
            self.assertTrue(
                status.startswith("Rendered") and status.endswith(f"({mode_key})"),
                status,
            )
            self.assertFalse(dlg._image_label.pixmap().isNull())

    def test_refresh_button_re_renders(self):
        dlg = self._make(mesh_data=_make_triangle_mesh())
        dlg.refresh_btn.click()
        self.assertTrue(dlg._status_label.text().startswith("Rendered"))

    def test_live_getter_reflects_updated_mesh(self):
        # The dialog reads through zero-arg getters; swapping the underlying
        # data and refreshing must reflect the new state (production contract
        # documented in the module docstring).
        state = {"mesh": None}
        dlg = self._dlg_cls(
            mesh_data_fn=lambda: state["mesh"],
            result_data_fn=lambda: None,
            h_min_fn=lambda: 1.0e-6,
            parent=self._parent,
        )
        self._dialogs.append(dlg)
        self.assertEqual(dlg._status_label.text(), "No mesh loaded")
        state["mesh"] = _make_triangle_mesh()
        dlg.refresh_view()
        self.assertTrue(dlg._status_label.text().startswith("Rendered"))

    def test_render_failure_sets_status_and_clears_image(self):
        # Malformed mesh data (missing required "node_x" key) must surface
        # as a loud "Render failed:" status, not a crash or a stale image.
        # (Out-of-range cell indices do NOT reach this path: the renderer
        # degrades to a "Cannot build mesh triangulation" placeholder and
        # still returns an image — verified while writing this test.)
        bad_mesh = {
            "node_y": np.array([0.0, 1.0], dtype=np.float64),
            "cell_nodes": np.array([[0, 1, 0]], dtype=np.int64),
        }
        with self.assertLogs(
            "swe2d.workbench.dialogs.detached_mesh_dialog", level="WARNING"
        ) as logs:
            dlg = self._make(mesh_data=bad_mesh)
        self.assertTrue(
            any("Detached mesh view render failed" in message for message in logs.output)
        )
        self.assertTrue(
            dlg._status_label.text().startswith("Render failed:"),
            dlg._status_label.text(),
        )
        self.assertIsNone(dlg._image_label.pixmap())

    def test_rendered_dialog_grabs_non_empty(self):
        dlg = self._make(mesh_data=_make_triangle_mesh(), result_data=_make_result_data())
        dlg.show()
        from qgis.PyQt import QtWidgets

        QtWidgets.QApplication.processEvents()
        self.assertTrue(
            grab_non_empty(dlg._image_label),
            "mesh view image label grab is blank after successful render",
        )
        dlg.close()

    def test_dialog_owns_no_timers(self):
        from qgis.PyQt import QtCore

        dlg = self._make(mesh_data=_make_triangle_mesh())
        self.assertEqual(dlg.findChildren(QtCore.QTimer), [])

    def test_close_hides_dialog_no_visible_zombie(self):
        _before_widgets, before_ptrs = _visible_top_level_snapshot()
        dlg = self._make(mesh_data=_make_triangle_mesh())
        dlg.show()
        self.assertTrue(dlg.isVisible())
        dlg.close()
        self.assertFalse(dlg.isVisible())
        _after_widgets, after_ptrs = _visible_top_level_snapshot()
        self.assertEqual(
            after_ptrs - before_ptrs,
            set(),
            "close left a new visible top-level zombie behind",
        )


@requires_qgis
class TestDetachedRuntimeLogDialog(unittest.TestCase):
    """Text round-trip + close lifecycle of SWE2DDetachedRuntimeLogDialog."""

    @classmethod
    def setUpClass(cls):
        ensure_qgis_app()

    def setUp(self):
        from qgis.PyQt import QtWidgets

        from swe2d.workbench.dialogs.detached_log_dialog import (
            SWE2DDetachedRuntimeLogDialog,
        )

        self._dlg_cls = SWE2DDetachedRuntimeLogDialog
        self._parent = QtWidgets.QMainWindow()
        self._dialogs = []

    def tearDown(self):
        delete_widgets_now(*self._dialogs, self._parent)

    def _make(self, initial_text=""):
        dlg = self._dlg_cls(initial_text=initial_text, parent=self._parent)
        self._dialogs.append(dlg)
        return dlg

    def test_initial_text_roundtrip(self):
        dlg = self._make(initial_text="line1\nline2")
        self.assertEqual(dlg.text.toPlainText(), "line1\nline2")
        self.assertTrue(dlg.text.isReadOnly())

    def test_append_and_set_text(self):
        dlg = self._make(initial_text="start")
        dlg.append_text("more")
        self.assertEqual(dlg.text.toPlainText(), "start\nmore")
        dlg.set_text("replaced")
        self.assertEqual(dlg.text.toPlainText(), "replaced")
        dlg.set_text(None)  # production coerces None to ""
        self.assertEqual(dlg.text.toPlainText(), "")

    def test_log_dialog_grabs_non_empty(self):
        dlg = self._make(initial_text="runtime log body")
        dlg.show()
        from qgis.PyQt import QtWidgets

        QtWidgets.QApplication.processEvents()
        self.assertTrue(
            grab_non_empty(dlg),
            "runtime log dialog grab is blank with text content",
        )
        dlg.close()

    def test_dialog_owns_no_timers(self):
        from qgis.PyQt import QtCore

        dlg = self._make("x")
        self.assertEqual(dlg.findChildren(QtCore.QTimer), [])

    def test_close_hides_dialog_no_visible_zombie(self):
        _before_widgets, before_ptrs = _visible_top_level_snapshot()
        dlg = self._make("log")
        dlg.show()
        self.assertTrue(dlg.isVisible())
        dlg.close()
        self.assertFalse(dlg.isVisible())
        _after_widgets, after_ptrs = _visible_top_level_snapshot()
        self.assertEqual(
            after_ptrs - before_ptrs,
            set(),
            "close left a new visible top-level zombie behind",
        )

    def test_finished_signal_fires_on_close(self):
        # Production wiring (studio_dialog._open_detached_runtime_log)
        # removes the dialog from its tracked list on ``finished`` — the
        # signal must fire so no stale reference (dead shell) is tracked.
        fired = []
        dlg = self._make("log")
        dlg.finished.connect(lambda result: fired.append(result))
        dlg.show()
        dlg.close()
        self.assertEqual(len(fired), 1)


if __name__ == "__main__":
    unittest.main()
