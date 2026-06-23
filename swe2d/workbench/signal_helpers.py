"""Signal connection safety helpers for PyQt5-based UI components.

Provides idempotent connect/disconnect, weakref-based lambdas, and
safe teardown for QWidget lifecycle management.
"""

from __future__ import annotations

import weakref
from typing import Any, Callable, Optional


def safe_disconnect(signal_obj, slot: Optional[Callable] = None) -> bool:
    """Disconnect a handler from a signal, ignoring ``TypeError``/``RuntimeError``.

    Parameters
    ----------
    signal_obj:
        The PyQt5 signal object (e.g. ``button.clicked``).
    slot:
        Optional specific handler to disconnect.  If ``None``, all handlers
        on the signal are disconnected.

    Returns
    -------
    bool
        ``True`` if the handler was previously connected, ``False`` otherwise.
    """
    try:
        if slot is not None:
            signal_obj.disconnect(slot)
        else:
            signal_obj.disconnect()
        return True
    except (TypeError, RuntimeError):
        return False


def safe_connect(signal_obj, handler: Callable) -> bool:
    """Connect a handler to a signal, safely disconnecting first.

    Disconnects any previous connection of *handler* first (idempotent),
    then connects.  Prevents duplicate connections when ``_build_ui()`` is
    called multiple times (e.g. during development hot-reload).

    Parameters
    ----------
    signal_obj:
        The PyQt5 signal object.
    handler:
        The callable to connect.

    Returns
    -------
    bool
        ``True`` if the connection succeeded.
    """
    safe_disconnect(signal_obj, handler)
    try:
        signal_obj.connect(handler)
        return True
    except (TypeError, RuntimeError):
        return False


def connect_lambda(signal_obj, weak_obj: object, method_name: str, *args: Any) -> None:
    """Connect a signal to a method using a weak reference.

    The lambda captures *weak_obj* via ``weakref.ref`` so the callback
    gracefully does nothing (instead of crashing) if the target object is
    garbage-collected before the signal fires.

    Parameters
    ----------
    signal_obj:
        The PyQt5 signal object.
    weak_obj:
        The object whose method will be called (captured weakly).
    method_name:
        Name of the method to call on *weak_obj*.
    *args:
        Positional arguments forwarded to the method.
    """
    ref = weakref.ref(weak_obj)

    def _handler(*sig_args: Any) -> None:
        """Weak-reference lambda: call the method only if the target object is still alive."""
        if obj is not None:
            m = getattr(obj, method_name, None)
            if m is not None:
                m(*args, *sig_args)

    safe_connect(signal_obj, _handler)


def safe_teardown(widget) -> None:
    """Prepare a widget for deletion by blocking signals.

    Call this **before** ``widget.deleteLater()`` to prevent queued
    signals from firing on a destroyed C++ object.

    Parameters
    ----------
    widget:
        A ``QWidget`` subclass (or ``None``).
    """
    if widget is None:
        return
    try:
        widget.blockSignals(True)
    except RuntimeError:
        pass


__all__ = [
    "connect_lambda",
    "safe_connect",
    "safe_disconnect",
    "safe_teardown",
]
