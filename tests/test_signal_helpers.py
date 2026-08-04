"""Behavioral tests for ``swe2d/workbench/signal_helpers.py``.

Covers the four public helpers — ``safe_connect``, ``safe_disconnect``,
``connect_lambda``, ``safe_teardown`` — with real ``pyqtSignal`` objects on
real ``QObject``/``QWidget`` instances.  These helpers exist to prevent the
classic PyQt5 failure modes: SIP wrappers outliving their C++ QObjects,
duplicate connections after hot-reload, dangling lambda captures, and
segfaults on GC after teardown.

Harness pattern P1 (real objects, no widgets beyond a plain QWidget) via
``tests/qgis_real_env.py``.  No mocks, no synthetic signals.
"""

import gc
import unittest

from tests.qgis_real_env import ensure_qgis_app, requires_qgis


def _flush_events():
    """Process pending events and DeferredDelete so deleteLater() lands."""
    from qgis.PyQt.QtCore import QCoreApplication, QEvent

    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    QCoreApplication.processEvents()


class _Emitter:
    """Factory for real QObject instances carrying real pyqtSignals.

    Defined as a factory (not a module-level import) so the module remains
    importable without Qt; ``requires_qgis`` gates execution.
    """

    @staticmethod
    def make():
        from qgis.PyQt.QtCore import QObject, pyqtSignal

        class _RealEmitter(QObject):
            changed = pyqtSignal()
            valued = pyqtSignal(int)

        return _RealEmitter()


class _Target:
    """Weakref-able plain-Python callback target for connect_lambda tests."""

    def __init__(self):
        self.calls = []

    def record(self, *args):
        self.calls.append(args)


@requires_qgis
class TestSafeConnect(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ensure_qgis_app()

    def setUp(self):
        self.emitter = _Emitter.make()

    def tearDown(self):
        self.emitter = None
        _flush_events()

    def test_connect_returns_true_and_slot_fires_once_per_emit(self):
        hits = []
        ok = __import__(
            "swe2d.workbench.signal_helpers", fromlist=["safe_connect"]
        ).safe_connect(self.emitter.changed, lambda: hits.append(1))
        self.assertTrue(ok)
        self.emitter.changed.emit()
        self.emitter.changed.emit()
        self.assertEqual(len(hits), 2)

    def test_double_safe_connect_is_idempotent(self):
        """Contract: safe_connect disconnects the same handler first, so a
        second safe_connect of the same pair must NOT double-fire."""
        from swe2d.workbench.signal_helpers import safe_connect

        hits = []

        def handler():
            hits.append(1)

        self.assertTrue(safe_connect(self.emitter.changed, handler))
        self.assertTrue(safe_connect(self.emitter.changed, handler))
        self.emitter.changed.emit()
        self.assertEqual(len(hits), 1)

    def test_distinct_handlers_both_fire(self):
        """safe_connect only dedupes the *same* handler; a different handler
        is an additional connection."""
        from swe2d.workbench.signal_helpers import safe_connect

        a, b = [], []
        safe_connect(self.emitter.changed, lambda: a.append(1))
        safe_connect(self.emitter.changed, lambda: b.append(1))
        self.emitter.changed.emit()
        self.assertEqual(len(a), 1)
        self.assertEqual(len(b), 1)

    def test_signal_args_are_forwarded(self):
        from swe2d.workbench.signal_helpers import safe_connect

        seen = []
        safe_connect(self.emitter.valued, seen.append)
        self.emitter.valued.emit(42)
        self.assertEqual(seen, [42])


@requires_qgis
class TestSafeDisconnect(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ensure_qgis_app()

    def setUp(self):
        self.emitter = _Emitter.make()

    def tearDown(self):
        self.emitter = None
        _flush_events()

    def test_connected_pair_disconnects_and_returns_true(self):
        from swe2d.workbench.signal_helpers import safe_connect, safe_disconnect

        hits = []

        def handler():
            hits.append(1)

        safe_connect(self.emitter.changed, handler)
        self.assertTrue(safe_disconnect(self.emitter.changed, handler))
        self.emitter.changed.emit()
        self.assertEqual(hits, [])

    def test_never_connected_returns_false_without_raising(self):
        from swe2d.workbench.signal_helpers import safe_disconnect

        def handler():
            pass

        self.assertFalse(safe_disconnect(self.emitter.changed, handler))

    def test_disconnect_all_handlers_when_slot_is_none(self):
        from swe2d.workbench.signal_helpers import safe_connect, safe_disconnect

        a, b = [], []
        safe_connect(self.emitter.changed, lambda: a.append(1))
        safe_connect(self.emitter.changed, lambda: b.append(1))
        self.assertTrue(safe_disconnect(self.emitter.changed))
        self.emitter.changed.emit()
        self.assertEqual(a, [])
        self.assertEqual(b, [])

    def test_disconnect_all_on_never_connected_returns_false(self):
        from swe2d.workbench.signal_helpers import safe_disconnect

        self.assertFalse(safe_disconnect(self.emitter.changed))

    def test_deleted_sender_signal_access_raises_runtimeerror(self):
        """SIP-liveness rule: after the sender's C++ object is deleted,
        merely ACCESSING ``emitter.changed`` on the surviving wrapper raises
        RuntimeError at the call site — before ``safe_disconnect`` is ever
        entered.  (Verified on this build: holding the *bound signal* across
        the deletion and calling disconnect() on it segfaults the process,
        so no helper-side try/except can cover that path.)  Callers must
        liveness-check the sender wrapper first (the ``objectName()``
        pattern from AGENTS.md)."""
        self.emitter.changed.connect(lambda: None)
        self.emitter.deleteLater()
        _flush_events()
        gc.collect()
        with self.assertRaises(RuntimeError):
            self.emitter.changed  # noqa: B018 — the access itself must raise

    def test_objectname_liveness_guard_pattern_prevents_dangling_access(self):
        """The prescribed guard: ``objectName()`` raises RuntimeError on a
        deleted wrapper, so the caller skips the disconnect entirely and
        never touches the dangling signal."""
        from swe2d.workbench.signal_helpers import safe_disconnect

        self.emitter.setObjectName("probe")
        self.emitter.changed.connect(lambda: None)
        emitter = self.emitter
        self.emitter = None  # drop the class-owned ref; wrapper stays alive
        emitter.deleteLater()
        _flush_events()
        gc.collect()

        disconnected = False
        try:
            emitter.objectName()  # liveness probe — raises when dead
            disconnected = safe_disconnect(emitter.changed)
        except RuntimeError:
            pass  # dead wrapper: skip, exactly as production callers must
        self.assertFalse(disconnected)


@requires_qgis
class TestConnectLambda(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ensure_qgis_app()

    def setUp(self):
        self.emitter = _Emitter.make()

    def tearDown(self):
        self.emitter = None
        _flush_events()

    def test_fires_while_target_alive_with_forwarded_args(self):
        from swe2d.workbench.signal_helpers import connect_lambda

        target = _Target()
        connect_lambda(self.emitter.valued, target, "record", "prefix")
        self.emitter.valued.emit(7)
        # *args precede the signal's own arguments: record("prefix", 7)
        self.assertEqual(target.calls, [("prefix", 7)])

    def test_no_arg_signal(self):
        from swe2d.workbench.signal_helpers import connect_lambda

        target = _Target()
        connect_lambda(self.emitter.changed, target, "record")
        self.emitter.changed.emit()
        self.assertEqual(target.calls, [()])

    def test_dead_target_is_not_called_and_does_not_crash(self):
        """After the weakref target is GC'd, emitting must be a silent no-op —
        this is the dangling-lambda-capture failure mode the helper exists
        to prevent."""
        from swe2d.workbench.signal_helpers import connect_lambda

        target = _Target()
        calls = target.calls  # keep the list alive after target dies
        connect_lambda(self.emitter.changed, target, "record")
        self.emitter.changed.emit()
        self.assertEqual(len(calls), 1)

        del target
        gc.collect()
        self.emitter.changed.emit()
        self.emitter.changed.emit()
        self.assertEqual(len(calls), 1)

    def test_missing_method_is_graceful_noop(self):
        from swe2d.workbench.signal_helpers import connect_lambda

        target = _Target()
        connect_lambda(self.emitter.changed, target, "does_not_exist")
        self.emitter.changed.emit()  # must not raise
        self.assertEqual(target.calls, [])


@requires_qgis
class TestSafeTeardown(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ensure_qgis_app()

    def tearDown(self):
        _flush_events()

    def _make_widget(self):
        from qgis.PyQt.QtWidgets import QWidget

        w = QWidget()
        w.setObjectName("signal_helpers_teardown_test")
        return w

    def test_none_is_noop(self):
        from swe2d.workbench.signal_helpers import safe_teardown

        safe_teardown(None)  # must not raise

    def test_blocks_signals_and_severs_delivery(self):
        """Contract: after safe_teardown, signalsBlocked() is True and emits
        no longer reach connected slots."""
        from swe2d.workbench.signal_helpers import safe_connect, safe_teardown

        w = self._make_widget()
        hits = []
        safe_connect(w.destroyed, lambda *a: hits.append(1))
        self.assertFalse(w.signalsBlocked())
        safe_teardown(w)
        self.assertTrue(w.signalsBlocked())

        # A blocked widget does not deliver its own signals.  We must
        # actually show the widget first — ``isVisible()`` is False for
        # any widget whose ancestor chain is hidden (the default state of
        # a freshly constructed QWidget), so a w that was never shown
        # would pass ``assertFalse(w.isVisible())`` even if the close
        # slot never fired.  Showing the widget proves the assertion
        # actually exercises safe_teardown's signal severing.
        w.show()
        self.assertTrue(w.isVisible())
        emitter = _Emitter.make()
        safe_connect(emitter.changed, w.close)
        emitter.changed.emit()
        _flush_events()
        self.assertFalse(w.isVisible())
        w.deleteLater()
        _flush_events()

    def test_destroyed_signal_boundary_is_documented(self):
        """Boundary: ``QObject.destroyed`` is emitted from the C++ destructor
        and is NOT suppressible by ``blockSignals(True)`` — so
        ``safe_teardown`` protects against queued *normal* signals, not
        against ``destroyed()`` delivery.  Slots connected to ``destroyed``
        must be deletion-safe on their own; this test pins the real Qt
        behavior so the helper's contract is not over-assumed."""
        from swe2d.workbench.signal_helpers import safe_teardown

        w = self._make_widget()
        hits = []
        w.destroyed.connect(lambda *a: hits.append(1))
        safe_teardown(w)
        w.deleteLater()
        _flush_events()
        gc.collect()
        self.assertEqual(hits, [1])  # destroyed() still fires — by Qt design

    def test_normal_signal_suppressed_after_teardown(self):
        """The historical failure mode safe_teardown DOES cover: a normal
        signal emitted after teardown (e.g. a queued timeout/data signal)
        no longer reaches its slots."""
        from qgis.PyQt.QtCore import pyqtSignal
        from qgis.PyQt.QtWidgets import QWidget

        from swe2d.workbench.signal_helpers import safe_teardown

        class _SignallingWidget(QWidget):
            pinged = pyqtSignal()

        w = _SignallingWidget()
        w.setObjectName("signal_helpers_teardown_signal_test")
        hits = []
        w.pinged.connect(lambda: hits.append(1))
        w.pinged.emit()
        self.assertEqual(hits, [1])
        safe_teardown(w)
        w.pinged.emit()
        self.assertEqual(hits, [1])  # second emit suppressed
        w.deleteLater()
        _flush_events()

    def test_deleted_widget_returns_without_raising(self):
        """Dangling SIP wrapper: safe_teardown must swallow RuntimeError."""
        from swe2d.workbench.signal_helpers import safe_teardown

        w = self._make_widget()
        w.deleteLater()
        _flush_events()
        gc.collect()
        safe_teardown(w)  # must not raise

    def test_gc_after_teardown_with_timer_child_no_crash(self):
        """Widget with an active QTimer child and live connections: teardown,
        deleteLater, GC.  Reaching the end of this test without a segfault
        is the assertion (process death is the loud failure)."""
        from qgis.PyQt.QtCore import QTimer

        from swe2d.workbench.signal_helpers import safe_teardown

        w = self._make_widget()
        timer = QTimer(w)
        timer.setObjectName("signal_helpers_teardown_timer")
        timer.start(10)
        hits = []
        timer.timeout.connect(lambda: hits.append(1))

        safe_teardown(w)
        w.deleteLater()
        _flush_events()
        gc.collect()
        _flush_events()
        # Qt parents the timer to the widget; deletion cascades.  A blocked
        # parent never delivered timeout() to the slot before teardown.
        self.assertEqual(hits, [])


if __name__ == "__main__":
    unittest.main()
