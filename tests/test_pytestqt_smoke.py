from __future__ import annotations
import unittest
"""Smoke test: verify pytest-qt's qtbot fixture works in this env.

Used by the pytest-qt dep bootstrap (Phase 1.2a of
docs/plans/2026-07-26-gpu-direct-viewer.md).
"""

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QLabel


def test_qtbot_waits_for_signal(qtbot):
    label = QLabel("start")
    qtbot.addWidget(label)
    QTimer.singleShot(50, lambda: label.setText("done"))
    qtbot.waitUntil(lambda: label.text() == "done", timeout=1000)
    assert label.text() == "done"


def test_qtbot_waitSignal(qtbot):
    from PyQt5.QtCore import pyqtSignal
    from PyQt5.QtWidgets import QWidget

    class Emitter(QWidget):
        fired = pyqtSignal(int)

    e = Emitter()
    qtbot.addWidget(e)
    QTimer.singleShot(50, lambda: e.fired.emit(42))

    with qtbot.waitSignal(e.fired, timeout=1000) as blocker:
        pass

    assert blocker.signal_triggered
    assert blocker.args == [42]

class _PytestStyleWrapper(unittest.TestCase):
    """Auto-generated wrapper for module-level test functions.

    Created by tools/wrap_pytest_style.py so that pytest-style tests
    (def test_* at module level) become visible to `python3 -m unittest`.
    Each module-level test is attached as a staticmethod so it can be
    discovered and run as a unittest TestCase.
    """
__wrapped_funcs = []
for _name, _obj in list(globals().items()):
    if _name.startswith("test_") and callable(_obj) and not isinstance(_obj, type):
        setattr(_PytestStyleWrapper, _name, staticmethod(_obj))
        __wrapped_funcs.append(_name)
for _name in __wrapped_funcs:
    del globals()[_name]
del _name, _obj, __wrapped_funcs
