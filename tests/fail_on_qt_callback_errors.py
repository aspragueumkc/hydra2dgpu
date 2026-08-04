"""Fail-fast collector for uncaught Qt callback errors in GUI tests.

Importing this module installs both Python's exception hook and Qt's message
handler before unittest imports the GUI modules.  Callback exceptions are
printed immediately and retained so a test suite cannot pass while Qt reports
an uncaught exception asynchronously.
"""
from __future__ import annotations

import sys
import traceback
from typing import Any

_CALLBACK_ERRORS: list[tuple[str, str]] = []
_PREVIOUS_EXCEPTHOOK = None
_PREVIOUS_QT_HANDLER = None
_INSTALLED = False

_QT_ERROR_MARKERS = (
    "traceback",
    "exception",
    "uncaught",
    "typeerror",
    "runtimeerror",
    "qvariant",
    "unable to convert",
)


def callback_errors() -> tuple[tuple[str, str], ...]:
    """Return captured ``(source, formatted_message)`` records."""
    return tuple(_CALLBACK_ERRORS)


def clear_callback_errors() -> None:
    """Clear records, primarily for isolated collector self-tests."""
    _CALLBACK_ERRORS.clear()


def record_python_exception(exc_type, exc_value, exc_tb) -> None:
    """Record and print one exception delivered through ``sys.excepthook``."""
    formatted = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    _CALLBACK_ERRORS.append(("sys.excepthook", formatted))
    print("[qt-callback-error] uncaught Python callback exception:", file=sys.stderr)
    print(formatted, file=sys.stderr, end="")


def _is_qt_error(message_type: Any, text: str) -> bool:
    from qgis.PyQt import QtCore

    if message_type in (QtCore.QtCriticalMsg, QtCore.QtFatalMsg):
        return True
    lowered = text.lower()
    return any(marker in lowered for marker in _QT_ERROR_MARKERS)


def handle_qt_message(message_type: Any, context: Any, message: str) -> None:
    """Record Qt messages that indicate an uncaught callback failure."""
    text = str(message)
    if _is_qt_error(message_type, text):
        _CALLBACK_ERRORS.append(("Qt message", text))
        print(f"[qt-callback-error] {text}", file=sys.stderr)


def _excepthook(exc_type, exc_value, exc_tb) -> None:
    record_python_exception(exc_type, exc_value, exc_tb)
    # Our output replaces the default hook; preserve a non-default host hook
    # (for example QGIS's error dialog hook) without duplicating stderr output.
    previous = _PREVIOUS_EXCEPTHOOK
    if previous is not None and previous not in (sys.__excepthook__, _excepthook):
        previous(exc_type, exc_value, exc_tb)


def _qt_message_handler(message_type, context, message) -> None:
    handle_qt_message(message_type, context, message)
    previous = _PREVIOUS_QT_HANDLER
    if previous is not None and previous is not _qt_message_handler:
        previous(message_type, context, message)
    else:
        # Installing a handler disables Qt's default stderr handler when there
        # is no previous Python handler; retain ordinary Qt diagnostics.
        print(str(message), file=sys.stderr)


def install_callback_error_collector() -> None:
    """Install the hooks exactly once."""
    global _INSTALLED, _PREVIOUS_EXCEPTHOOK, _PREVIOUS_QT_HANDLER
    if _INSTALLED:
        return

    from qgis.PyQt import QtCore

    _PREVIOUS_EXCEPTHOOK = sys.excepthook
    sys.excepthook = _excepthook
    _PREVIOUS_QT_HANDLER = QtCore.qInstallMessageHandler(_qt_message_handler)
    _INSTALLED = True


def fail_if_callback_errors() -> None:
    """Exit with status 1 after printing a concise captured-error summary."""
    if not _CALLBACK_ERRORS:
        return
    print(
        f"[qt-callback-error] {len(_CALLBACK_ERRORS)} uncaught Qt callback "
        "error(s) captured; failing gui-behavioral stage",
        file=sys.stderr,
    )
    raise SystemExit(1)


def run_unittest(test_names: list[str]) -> int:
    """Run named unittest modules and apply the callback-error gate."""
    if not test_names:
        raise ValueError("at least one unittest module is required")

    import unittest

    suite = unittest.defaultTestLoader.loadTestsFromNames(test_names)
    result = unittest.TextTestRunner(verbosity=1).run(suite)

    # Flush deferred Qt callbacks before deciding whether the stage is clean.
    from qgis.PyQt import QtCore, QtWidgets

    app = QtWidgets.QApplication.instance()
    if app is not None:
        app.processEvents()
        app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
        app.processEvents()

    if not result.wasSuccessful():
        if _CALLBACK_ERRORS:
            print(
                f"[qt-callback-error] {len(_CALLBACK_ERRORS)} callback error(s) "
                "captured while unittest also failed",
                file=sys.stderr,
            )
        return 1

    try:
        fail_if_callback_errors()
    except SystemExit as exc:
        return int(exc.code or 1)
    return 0


install_callback_error_collector()


if __name__ == "__main__":
    raise SystemExit(run_unittest(sys.argv[1:]))
