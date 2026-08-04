"""HYDRA MCP bridge — lives inside a live QGIS process (Phase 2.A + 2.B + 2.C).

Starts a ``QLocalServer`` on a unique per-user socket, writes a 0600 token
file, and accepts length-prefixed JSON-RPC requests.  Handlers run on the Qt
GUI thread (``QLocalServer`` signals are queued on the thread that owns the
server), so widget access is legal without extra synchronization.

Injected into QGIS via::

    qgis --code tools/hydra_mcp/qgis_bridge.py

or auto-started when the plugin loads with ``HYDRA_MCP_BRIDGE=1``.
"""
from __future__ import annotations

import json
import os
import secrets
import sys
import tempfile
import time
import getpass
from pathlib import Path
from typing import Any, Dict, List, Optional

from qgis.PyQt.QtCore import QObject, QPoint, Qt, pyqtSignal, QTimer, QEventLoop
from qgis.PyQt.QtNetwork import QLocalServer, QLocalSocket
from qgis.PyQt.QtTest import QTest
from qgis.PyQt.QtWidgets import (
    QAbstractButton,
    QAction,
    QApplication,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QLabel,
    QLineEdit,
    QMainWindow,
    QSpinBox,
    QTextEdit,
    QWidget,
)

# Boolean coercion for QCheckBox.setValue — accept truthy strings as well
# as the Python ``bool`` type.  ``bool("false")`` is ``True`` because the
# string is non-empty, so we need a string-aware coercion here.
_BOOL_TRUE_STRINGS = frozenset({"true", "1", "yes", "on"})


def _coerce_bool(value: Any) -> bool:
    """Coerce *value* to a bool honouring the standard truthy strings.

    ``True``/``False`` pass through unchanged.  Any other type is
    converted via ``str(value).strip().lower()`` and accepted only if
    the result is in ``_BOOL_TRUE_STRINGS``.  This means ``"false"``,
    ``"no"``, ``"0"``, ``"off"`` and unknown strings are all ``False``,
    and the previous ``bool("false") == True`` bug is fixed.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in _BOOL_TRUE_STRINGS

# Make the repo importable when this script is executed via ``qgis --code``.
# The repo root is two directories above tools/hydra_mcp/.  When launched
# via ``qgis --code <script>`` QGIS exec's the script in a frame where
# ``__file__`` is not defined, so fall back to ``tools/hydra_mcp`` walking
# up from CWD.
try:
    _BRIDGE_FILE = Path(__file__).resolve()
except NameError:
    _BRIDGE_FILE = Path(os.path.abspath(__name__ or "qgis_bridge"))
    if not _BRIDGE_FILE.is_file():
        _BRIDGE_FILE = Path(os.getcwd()) / "tools" / "hydra_mcp" / "qgis_bridge.py"
_REPO_ROOT = _BRIDGE_FILE.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Re-use the wire-format helpers from the client module; they are pure Python.
from tools.hydra_mcp.bridge_client import (
    FrameTooLargeError,
    MAX_FRAME_BYTES,
    decode_messages,
    encode_message,
)

from swe2d.workbench.devtools.widget_walker import (
    walk_widget_tree,
    WidgetNode,
    find_widget_by_path,
)

# Phase 2.C: screenshot helper (standalone, no module-level Qt needed).
from tools.hydra_mcp.widget_screenshot import capture_widget_screenshot

BRIDGE_VERSION = "2.C"


def _token_dir() -> Path:
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    if runtime:
        return Path(runtime)
    return Path(tempfile.gettempdir())


def _node_to_dict(node: WidgetNode) -> Dict[str, Any]:
    return {
        "object_name": node.object_name,
        "class_name": node.class_name,
        "widget_id": node.widget_id,
        "parent_id": node.parent_id,
        "text": node.text,
        "depth": node.depth,
    }


def _widget_info(widget: QWidget) -> Optional[Dict[str, Any]]:
    """Return a serializable dict of a live widget's key properties."""
    try:
        object_name = widget.objectName()
        class_name = type(widget).__name__
        geometry = widget.geometry()
        is_visible = widget.isVisible()
    except RuntimeError:
        return None
    return {
        "object_name": object_name,
        "class_name": class_name,
        "widget_id": id(widget),
        "geometry": {
            "x": geometry.x(),
            "y": geometry.y(),
            "width": geometry.width(),
            "height": geometry.height(),
        },
        "is_visible": is_visible,
    }


# ── Phase 3: behavioral helpers ────────────────────────────────────────────────


_KEY_NAME_MAP: Dict[str, Qt.Key] = {
    "return": Qt.Key_Return,
    "enter": Qt.Key_Enter,
    "tab": Qt.Key_Tab,
    "backspace": Qt.Key_Backspace,
    "delete": Qt.Key_Delete,
    "escape": Qt.Key_Escape,
    "space": Qt.Key_Space,
    "left": Qt.Key_Left,
    "right": Qt.Key_Right,
    "up": Qt.Key_Up,
    "down": Qt.Key_Down,
    "home": Qt.Key_Home,
    "end": Qt.Key_End,
    "pageup": Qt.Key_PageUp,
    "pagedown": Qt.Key_PageDown,
}


def _resolve_key(key: str) -> Qt.Key:
    """Map a key name or single character to a Qt.Key enum."""
    if isinstance(key, int):
        return Qt.Key(key)
    s = str(key).strip()
    if not s:
        raise ValueError("key is empty")
    mapped = _KEY_NAME_MAP.get(s.lower())
    if mapped is not None:
        return mapped
    if len(s) == 1 and s.isascii():
        # Qt.Key enum values for ASCII letters/digits match their code points.
        return Qt.Key(ord(s.upper()))
    raise ValueError(f"unsupported key: {s!r}")


def _find_active_studio_dialog() -> Optional[Any]:
    """Return the active SWE2D workbench dialog if one exists."""
    for mod_name in (
        "swe2d.workbench.studio_dialog",
        "swe2d.workbench.views.studio_host_methods",
    ):
        try:
            mod = __import__(mod_name, fromlist=["_studio_active_dialog"])
        except Exception:
            continue
        dlg = getattr(mod, "_studio_active_dialog", None)
        if dlg is not None:
            try:
                _ = dlg.objectName()
            except RuntimeError:
                continue
            return dlg
    return None


def _find_widget_by_name(name: str, root: Optional[QWidget] = None) -> Optional[QWidget]:
    """Find a widget anywhere under *root* (or any top-level window) by objectName."""
    if root is not None:
        try:
            found = root.findChildren(QWidget, name)
        except (RuntimeError, TypeError):
            found = []
        return found[0] if found else None

    app = QApplication.instance()
    if app is None:
        return None
    for top in app.topLevelWidgets():
        try:
            found = top.findChildren(QWidget, name)
        except (RuntimeError, TypeError):
            continue
        if found:
            return found[0]
    return None


def _resolve_widget(params: Dict[str, Any]) -> QWidget:
    """Resolve a widget from params['path'] or params['object_name']."""
    path_str = params.get("path")
    object_name = params.get("object_name")
    if not path_str and not object_name:
        raise RuntimeError("either path or object_name is required")

    if path_str:
        root_name = params.get("root_object_name")
        root = _resolve_root_widget(root_name)
        if root is None:
            raise RuntimeError(
                f"root widget '{root_name}' not found" if root_name
                else "no root widget available"
            )
        from swe2d.workbench.devtools.widget_walker import find_widget_by_path
        path_parts = [p for p in path_str.split(".") if p]
        widget = find_widget_by_path(root, path_parts)
        if widget is not None:
            return widget

    if object_name:
        widget = _find_widget_by_name(object_name)
        if widget is not None:
            return widget

    raise RuntimeError(f"widget not found for path={path_str!r} object_name={object_name!r}")



def get_widget_value(widget: QWidget) -> Dict[str, Any]:
    """Read the current value from *widget*.

    Supports: QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox, QLineEdit,
    QTextEdit, QLabel.  Uses ``isinstance`` so subclasses (e.g. a project
    subclass of QSpinBox) are accepted — the previous exact-class
    comparison silently rejected them.
    """
    try:
        if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            return {
                "ok": True,
                "type": type(widget).__name__,
                "value": widget.value(),
            }
        if isinstance(widget, QCheckBox):
            return {
                "ok": True,
                "type": type(widget).__name__,
                "value": widget.isChecked(),
            }
        if isinstance(widget, QComboBox):
            return {
                "ok": True,
                "type": type(widget).__name__,
                "value": widget.currentText(),
            }
        if isinstance(widget, QLineEdit):
            return {
                "ok": True,
                "type": type(widget).__name__,
                "value": widget.text(),
            }
        if isinstance(widget, QTextEdit):
            return {
                "ok": True,
                "type": type(widget).__name__,
                "value": widget.toPlainText(),
            }
        if isinstance(widget, QLabel):
            return {
                "ok": True,
                "type": type(widget).__name__,
                "value": widget.text(),
            }
        return {
            "ok": False,
            "error": f"unsupported widget type: {type(widget).__name__}",
        }
    except RuntimeError as exc:
        return {"ok": False, "error": str(exc)}


def set_widget_value(widget: QWidget, value: Any) -> Dict[str, Any]:
    """Write *value* into *widget*.

    Supports: QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox, QLineEdit,
    QTextEdit.  Uses ``isinstance`` (not exact-class) so subclasses are
    accepted.
    """
    try:
        if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            widget.setValue(value)
            return {"ok": True}
        if isinstance(widget, QCheckBox):
            # ``_coerce_bool`` honours ``"false"``/``"no"``/``"0"``/``"off"``;
            # the previous ``bool(value)`` returned ``True`` for any
            # non-empty string, including ``"false"``.
            widget.setChecked(_coerce_bool(value))
            return {"ok": True}
        if isinstance(widget, QComboBox):
            idx = widget.findText(str(value))
            if idx < 0:
                return {"ok": False, "error": f"no item matching {value!r}"}
            widget.setCurrentIndex(idx)
            return {"ok": True}
        if isinstance(widget, QLineEdit):
            widget.setText(str(value))
            return {"ok": True}
        if isinstance(widget, QTextEdit):
            widget.setPlainText(str(value))
            return {"ok": True}
        return {
            "ok": False,
            "error": f"unsupported widget type: {type(widget).__name__}",
        }
    except RuntimeError as exc:
        return {"ok": False, "error": str(exc)}


def _resolve_root_widget(root_name: Optional[str]) -> Optional[QWidget]:
    """Find the requested root widget from the QApplication top-level windows.

    Special root names:
    * ``"studio"`` — the active HYDRA2D Studio workbench dialog, if any.
    """
    if root_name == "studio":
        return _find_active_studio_dialog()

    app = QApplication.instance()
    if app is None:
        return None

    top_levels = []
    for w in app.topLevelWidgets():
        try:
            oname = w.objectName()
        except RuntimeError:
            continue
        top_levels.append(w)
        if root_name is not None and oname == root_name:
            return w

    if root_name is not None:
        return None

    active = app.activeWindow()
    if active is not None and active.isVisible():
        return active
    for w in top_levels:
        try:
            if isinstance(w, QMainWindow) and w.isVisible():
                return w
        except RuntimeError:
            continue
    for w in top_levels:
        try:
            if w.findChildren(QWidget, ""):
                return w
        except RuntimeError:
            continue
    return top_levels[0] if top_levels else None


class HydraMcpBridge(QObject):
    """In-process bridge exposing live GUI introspection handlers."""

    # Emitted when the bridge is ready (useful for tests that drive the loop).
    ready = pyqtSignal(str, str)  # socket_name, token_path

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._user = getpass.getuser()
        self._pid = os.getpid()
        self._rand = secrets.token_hex(4)
        self.socket_name = f"hydra_mcp_bridge_{self._user}_{self._pid}_{self._rand}"
        self.token = secrets.token_urlsafe(32)
        self.token_path = _token_dir() / f"hydra_mcp_bridge_{self._user}_{self._pid}_{self._rand}.json"

        self._server = QLocalServer(self)
        self._server.newConnection.connect(self._on_new_connection)
        self._busy = False

        # Per-socket read buffers and liveness tracking.
        self._buffers: Dict[int, bytes] = {}
        self._authenticated: Dict[int, bool] = {}
        self._handling: Dict[int, bool] = {}
        self._sockets: Dict[int, QLocalSocket] = {}

    def start(self) -> bool:
        """Write the token file and start listening."""
        self._write_token_file()
        if not self._server.listen(self.socket_name):
            # A stale socket may be left from a crashed process; remove it and retry.
            err = self._server.errorString()
            QLocalServer.removeServer(self.socket_name)
            if not self._server.listen(self.socket_name):
                raise RuntimeError(
                    f"Could not start bridge server on {self.socket_name}: {err}"
                )
        print(
            f"HYDRA_MCP_BRIDGE_READY {self.socket_name} {self.token_path}",
            flush=True,
        )
        self.ready.emit(self.socket_name, str(self.token_path))
        self._wire_shutdown_cleanup()
        return True

    def is_alive(self) -> bool:
        """Return ``True`` if the bridge object and its server are still alive."""
        try:
            _ = self.objectName()
        except RuntimeError:
            return False
        return self._server.isListening()

    def stop(self) -> None:
        """Stop listening, remove the socket, and delete the token file.

        Safe to call on an already-dead bridge; any ``RuntimeError`` from
        the SIP wrapper is swallowed.
        """
        try:
            self._server.close()
        except RuntimeError:
            pass
        try:
            QLocalServer.removeServer(self.socket_name)
        except (RuntimeError, TypeError):
            pass
        try:
            self._cleanup_token_file()
        except Exception:
            pass

    def _write_token_file(self) -> None:
        payload = {
            "socket_name": self.socket_name,
            "token": self.token,
            "pid": self._pid,
            "version": BRIDGE_VERSION,
        }
        self.token_path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic, mode-0600 creation: ``O_CREAT|O_EXCL`` raises ``FileExistsError``
        # if the path is already there (closes the symlink/precreate race that
        # ``write_text`` opens), and the 0o600 mode argument is applied at
        # create-time so the file is never visible with broader permissions.
        payload_bytes = json.dumps(payload).encode("utf-8")
        # ``we_created`` tracks whether the failure path is allowed to
        # unlink the token file.  ``O_EXCL`` failing with FileExistsError
        # means the pre-existing file belongs to another live bridge and
        # must not be removed.  Any other failure (os.open succeeded but
        # fdopen/write failed, or os.open raised something other than
        # FileExistsError after creating the file) means WE created the
        # file and the partial write should be cleaned up.
        we_created = True
        fd: Optional[int] = None
        try:
            try:
                fd = os.open(
                    str(self.token_path),
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o600,
                )
            except FileExistsError:
                # Foreign file — do not unlink on the way out.
                we_created = False
                raise
            try:
                with os.fdopen(fd, "wb") as fh:
                    fh.write(payload_bytes)
                    fh.flush()
                    try:
                        os.fsync(fh.fileno())
                    except OSError:
                        # Some filesystems (notably tmpfs on some kernels) reject
                        # fsync; the file is still on disk and atomically private.
                        pass
            except Exception:
                # ``fd`` may still be open if os.fdopen failed; close it
                # before unlinking so the unlink isn't blocked on Windows.
                try:
                    os.close(fd)
                except OSError:
                    pass
                raise
        except Exception:
            if we_created:
                # Clean up the partial write so a failed create does not
                # leave a zero-byte file with a valid mode that could be
                # picked up by auto-discovery.
                try:
                    os.unlink(self.token_path)
                except OSError:
                    pass
            raise

    def _cleanup_token_file(self) -> None:
        """Best-effort removal of the token file.

        Connected to ``destroyed`` and ``QApplication.aboutToQuit`` so the
        token file is removed when the bridge object goes away.  Without
        this, ``$XDG_RUNTIME_DIR`` accumulates stale files (the
        ``pid+random`` suffix does not collide, so discovery will keep
        returning the freshest file but old ones pile up indefinitely).
        """
        try:
            os.unlink(self.token_path)
        except FileNotFoundError:
            pass
        except OSError:
            pass

    def _wire_shutdown_cleanup(self) -> None:
        """Hook ``destroyed`` and ``aboutToQuit`` to token-file cleanup.

        ``destroyed`` covers the bridge's own destruction (e.g. plugin
        unload in a future Phase); ``aboutToQuit`` covers the normal
        QGIS shutdown path.  Both must be installed because either can
        fire first.
        """
        try:
            self.destroyed.connect(self._cleanup_token_file)
        except (RuntimeError, TypeError):
            pass
        app = QApplication.instance()
        if app is not None:
            try:
                app.aboutToQuit.connect(self._cleanup_token_file)
            except (RuntimeError, TypeError):
                pass

    def _on_new_connection(self) -> None:
        socket = self._server.nextPendingConnection()
        if socket is None:
            return
        sid = id(socket)
        self._buffers[sid] = b""
        self._authenticated[sid] = False
        # Hold a strong Python reference to the socket: PyQt5 keeps signal
        # connections on the SIP wrapper, and a wrapper that is garbage
        # collected silently drops its connections (the C++ socket itself is
        # parented to the server and survives).  The old per-connection
        # lambda kept the wrapper alive implicitly via its closure; bound
        # methods do not, so we must.  Dropped in ``_on_disconnected``.
        self._sockets[sid] = socket
        # Bound-method connections, NOT lambdas: PyQt5 destroys a lambda slot
        # (and its PyQtSlotProxy receiver) when the connection dies, and a
        # signal dispatched re-entrantly during a long handler (e.g. the
        # full-tree widget walk) then calls the freed lambda -> SIGSEGV at
        # COPY_FREE_VARS.  A bound method lives on the bridge itself, which
        # outlives every socket, so the slot can never be freed mid-dispatch.
        # ``sender()`` identifies which socket emitted the signal.
        socket.readyRead.connect(self._on_socket_ready_read)
        socket.disconnected.connect(self._on_socket_disconnected)

    def _on_socket_ready_read(self) -> None:
        try:
            socket = self.sender()
        except RuntimeError:
            return
        if socket is None or not isinstance(socket, QLocalSocket):
            return
        try:
            self._on_ready_read(socket)
        except RuntimeError:
            # The socket was destroyed between the signal and the handler;
            # drop its bookkeeping instead of crashing.
            self._buffers.pop(id(socket), None)
            self._authenticated.pop(id(socket), None)

    def _on_socket_disconnected(self) -> None:
        try:
            socket = self.sender()
        except RuntimeError:
            return
        if socket is None or not isinstance(socket, QLocalSocket):
            return
        self._on_disconnected(socket)

    def _on_ready_read(self, socket: QLocalSocket) -> None:
        sid = id(socket)
        self._buffers[sid] += bytes(socket.readAll())
        if self._handling.get(sid):
            # A nested readyRead (signals are delivered re-entrantly while a
            # long handler such as the widget walk is running) must NOT
            # process here: two interleaved invocations would interleave
            # their ``_send`` writes on the same socket and corrupt the
            # length-prefixed frame stream.  The outer invocation re-decodes
            # the appended bytes below.
            return
        self._handling[sid] = True
        try:
            while True:
                try:
                    messages, self._buffers[sid] = decode_messages(
                        self._buffers[sid]
                    )
                except FrameTooLargeError as exc:
                    # A peer declared a frame larger than the protocol
                    # limit.  Reject the connection: we refuse to allocate
                    # ``exc.length`` bytes, and the buffered bytes are an
                    # incomplete prefix of an oversize frame, so we drop the
                    # buffer too.
                    self._send_error(
                        socket, None, -32002,
                        f"frame too large: {exc.length} bytes exceeds "
                        f"MAX_FRAME_BYTES={exc.max_bytes}",
                    )
                    self._buffers.pop(sid, None)
                    socket.abort()
                    return
                if not messages:
                    break
                for msg in messages:
                    self._handle_message(socket, msg)
                # Requests that arrived re-entrantly while handling were
                # buffered (see the busy-flag return above); re-decode so
                # they are processed by this single invocation.
            # One final re-decode: a nested readyRead may have appended
            # bytes between the last decode and the loop exit.  No further
            # Qt calls happen below, so nothing can append after this.
            try:
                messages, self._buffers[sid] = decode_messages(
                    self._buffers[sid]
                )
            except FrameTooLargeError:
                self._buffers.pop(sid, None)
                socket.abort()
                return
            for msg in messages:
                self._handle_message(socket, msg)
        finally:
            self._handling[sid] = False

    def _on_disconnected(self, socket: QLocalSocket) -> None:
        sid = id(socket)
        self._buffers.pop(sid, None)
        self._authenticated.pop(sid, None)
        self._handling.pop(sid, None)
        self._sockets.pop(sid, None)
        socket.deleteLater()

    def _send(self, socket: QLocalSocket, obj: Dict[str, Any]) -> None:
        data = encode_message(obj)
        socket.write(data)
        socket.flush()

    def _send_error(
        self,
        socket: QLocalSocket,
        msg_id: Any,
        code: int,
        message: str,
    ) -> None:
        self._send(socket, {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {"code": code, "message": message},
        })

    def _send_result(self, socket: QLocalSocket, msg_id: Any, result: Any) -> None:
        self._send(socket, {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": result,
        })

    def _handle_message(self, socket: QLocalSocket, msg: Dict[str, Any]) -> None:
        sid = id(socket)
        msg_id = msg.get("id")
        method = msg.get("method")
        params = msg.get("params") or {}

        if not isinstance(params, dict):
            self._send_error(socket, msg_id, -32602, "params must be an object")
            return

        token = params.get("token")
        if token != self.token:
            self._send_error(socket, msg_id, -32001, "invalid or missing token")
            socket.disconnectFromServer()
            return

        # Token is valid from here on.
        self._authenticated[sid] = True

        if method == "ping":
            self._send_result(socket, msg_id, {
                "version": BRIDGE_VERSION,
                "socket_name": self.socket_name,
                "token_path": str(self.token_path),
                "pid": self._pid,
            })
            return

        # Single-flight: reject concurrent GUI requests.
        if self._busy:
            self._send_error(
                socket, msg_id, -32000,
                "bridge is busy processing another request; retry shortly"
            )
            return

        self._busy = True
        try:
            if method == "get_widget_tree":
                result = self._handle_get_widget_tree(params)
            elif method == "find_widget":
                result = self._handle_find_widget(params)
            elif method == "describe_widget":
                result = self._handle_describe_widget(params)
            elif method == "get_value":

                result = self._handle_get_value(params)
            elif method == "set_value":
                result = self._handle_set_value(params)
            elif method == "screenshot":
                result = self._handle_screenshot(params)
            elif method == "click_widget":
                result = self._handle_click(params)
            elif method == "key_press":
                result = self._handle_key(params)
            elif method == "run_action":
                result = self._handle_run_action(params)
            elif method == "read_log":
                result = self._handle_read_log(params)
            elif method == "run_simulation":
                result = self._handle_run_simulation(params)
            elif method == "resize_main_window":
                result = self._handle_resize_main_window(params)
            elif method == "resize_docks":
                result = self._handle_resize_docks(params)
            elif method == "set_toolbox_page":
                result = self._handle_set_toolbox_page(params)
            elif method == "force_dock_size":
                result = self._handle_force_dock_size(params)
            elif method == "list_dock_widgets":
                result = self._handle_list_dock_widgets(params)
            elif method == "set_studio_dock_tab":
                result = self._handle_set_studio_dock_tab(params)
            elif method == "list_dock_tab_pages":
                result = self._handle_list_dock_tab_pages(params)
            else:
                self._send_error(socket, msg_id, -32601, f"unknown method: {method}")
                return
        except Exception as exc:  # never leak a traceback to the wire
            self._send_error(socket, msg_id, -32603, f"handler error: {exc}")
            return
        finally:
            self._busy = False

        self._send_result(socket, msg_id, result)

    def _handle_get_widget_tree(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        root_name = params.get("root_object_name")
        root = self._resolve_root(root_name)
        if root is None:
            raise RuntimeError(
                f"root widget '{root_name}' not found" if root_name
                else "no root widget available"
            )
        nodes = walk_widget_tree(root)
        return [_node_to_dict(n) for n in nodes]

    def _handle_find_widget(self, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # Phase 2.B: dot-separated path from a root widget.
        path_str = params.get("path")
        if path_str:
            root_name = params.get("root_object_name")
            root = self._resolve_root(root_name)
            if root is None:
                raise RuntimeError(
                    f"root widget '{root_name}' not found" if root_name
                    else "no root widget available"
                )
            path_parts = [p for p in path_str.split(".") if p]
            widget = find_widget_by_path(root, path_parts)
            if widget is None:
                return None
            return _widget_info(widget)

        # Phase 2.A: search by object_name across all top-level widgets.
        object_name = params.get("object_name")
        if not object_name:
            raise RuntimeError("object_name is required")
        app = QApplication.instance()
        if app is None:
            raise RuntimeError("no QApplication instance")
        for widget in app.topLevelWidgets():
            try:
                nodes = walk_widget_tree(widget)
            except RuntimeError:
                continue
            for node in nodes:
                if node.object_name == object_name:
                    return _node_to_dict(node)
        return None

    def _handle_describe_widget(self, params: Dict[str, Any]) -> Dict[str, Any]:
        path_str = params.get("path")
        if not path_str:
            raise RuntimeError("path is required")
        root = self._resolve_root(params.get("root_object_name"))
        if root is None:
            raise RuntimeError("no root widget available")
        widget = find_widget_by_path(root, [p for p in path_str.split(".") if p])
        if widget is None:
            return None
        info = _widget_info(widget)
        if info is None:
            return None
        info.update({
            "enabled": widget.isEnabled(),
            "visible": widget.isVisible(),
            "toolTip": widget.toolTip(),
            "statusTip": widget.statusTip(),
            "whatsThis": widget.whatsThis(),
            "accessibleName": widget.accessibleName(),
            "accessibleDescription": widget.accessibleDescription(),
        })
        return info

    def _handle_get_value(self, params: Dict[str, Any]) -> Dict[str, Any]:
        path_str = params.get("path")
        if not path_str:
            raise RuntimeError("path is required")
        root_name = params.get("root_object_name")
        root = self._resolve_root(root_name)
        if root is None:
            raise RuntimeError(
                f"root widget '{root_name}' not found" if root_name
                else "no root widget available"
            )
        path_parts = [p for p in path_str.split(".") if p]
        widget = find_widget_by_path(root, path_parts)
        if widget is None:
            raise RuntimeError(f"widget not found at path '{path_str}'")
        return get_widget_value(widget)

    def _handle_set_value(self, params: Dict[str, Any]) -> Dict[str, Any]:
        path_str = params.get("path")
        if not path_str:
            raise RuntimeError("path is required")
        value = params.get("value")
        root_name = params.get("root_object_name")
        root = self._resolve_root(root_name)
        if root is None:
            raise RuntimeError(
                f"root widget '{root_name}' not found" if root_name
                else "no root widget available"
            )
        path_parts = [p for p in path_str.split(".") if p]
        widget = find_widget_by_path(root, path_parts)
        if widget is None:
            raise RuntimeError(f"widget not found at path '{path_str}'")
        return set_widget_value(widget, value)

    def _handle_screenshot(self, params: Dict[str, Any]) -> Dict[str, Any]:
        format = str(params.get("format", "png")).lower()
        if format not in ("png", "jpg", "jpeg"):
            raise RuntimeError(f"unsupported format: {format}")
        widget = self._resolve_screenshot_target(params)
        return capture_widget_screenshot(widget, format)

    def _resolve_screenshot_target(
        self, params: Dict[str, Any]
    ) -> Optional[QWidget]:
        """Resolve a screenshot target to a live QWidget.

        Mutually-exclusive modes (consistent with the
        ``gui_screenshot(target=...)`` API):

        * ``target="dialog"`` (default): the active QGIS window.
        * ``target="studio"``: the active HYDRA2D Studio workbench dialog.
        * ``target="dock"``: the first ``QDockWidget`` visible in any
          top-level window.
        * ``target="canvas"``: the QGIS map canvas (``qgis.gui.QgsMapCanvas``).
          Falls back to ``None`` when QGIS is not available (the caller
          treats ``None`` as "widget not available" and returns an error).

        Legacy mode: ``path=...`` is a dot-separated objectName path
        resolved through ``find_widget_by_path`` against the active
        root.  ``target`` and ``path`` are mutually exclusive; ``target``
        wins when both are present.
        """
        target = params.get("target")
        path_str = params.get("path")

        if target:
            t = str(target).strip().lower()
            if t == "dialog":
                app = QApplication.instance()
                return app.activeWindow() if app is not None else None
            if t == "studio":
                return _find_active_studio_dialog()
            if t == "dock":
                app = QApplication.instance()
                if app is None:
                    return None
                for top in app.topLevelWidgets():
                    # ``findChildren(QDockWidget)`` (no name) matches by
                    # type only; passing an empty name string returns an
                    # empty list under PyQt5, so we use the no-name form.
                    found = top.findChildren(QDockWidget)
                    if found:
                        return found[0]
                return None
            if t == "canvas":
                try:
                    from qgis.gui import QgsMapCanvas
                except Exception:
                    return None
                # ``QgsMapCanvas`` is a singleton-like widget; iterate
                # top-level widgets and pick the first one whose type is
                # (a subclass of) ``QgsMapCanvas``.
                app = QApplication.instance()
                if app is None:
                    return None
                for top in app.topLevelWidgets():
                    if isinstance(top, QgsMapCanvas):
                        return top
                return None
            raise RuntimeError(
                f"unknown screenshot target {target!r}; expected one of: "
                "dialog, studio, dock, canvas"
            )

        # Legacy: dot-separated widget path.
        if not path_str:
            raise RuntimeError(
                "either target='dialog'|'dock'|'canvas' or path=<objectName.path> "
                "is required"
            )
        root_name = params.get("root_object_name")
        root = self._resolve_root(root_name)
        if root is None:
            raise RuntimeError(
                f"root widget '{root_name}' not found" if root_name
                else "no root widget available"
            )
        path_parts = [p for p in path_str.split(".") if p]
        return find_widget_by_path(root, path_parts)

    # ── Phase 3: behavioral GUI handlers ────────────────────────────────────────

    def _handle_click(self, params: Dict[str, Any]) -> Dict[str, Any]:
        widget = _resolve_widget(params)
        try:
            if not widget.isVisible():
                # Clicking an invisible widget is allowed but noted.
                pass
        except RuntimeError as exc:
            raise RuntimeError(f"widget access error: {exc}")

        # Optional coordinate click.  Values in [0.0, 1.0] are treated as
        # relative to the widget rectangle; larger values are absolute pixels.
        x = params.get("x")
        y = params.get("y")
        if x is not None and y is not None:
            try:
                x = float(x)
                y = float(y)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"x and y must be numeric: {exc}")
            rect = widget.rect()
            if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                pos = QPoint(int(x * rect.width()), int(y * rect.height()))
            else:
                pos = QPoint(int(x), int(y))
            QTest.mouseClick(widget, Qt.LeftButton, pos=pos)
        else:
            # Use the widget's center; QTest accepts QPoint(0,0) for simple buttons.
            QTest.mouseClick(widget, Qt.LeftButton)
        return {"ok": True, "class_name": type(widget).__name__,
                "object_name": widget.objectName()}

    def _handle_key(self, params: Dict[str, Any]) -> Dict[str, Any]:
        widget = _resolve_widget(params)
        key = _resolve_key(params.get("key", ""))
        QTest.keyClick(widget, key)
        return {"ok": True, "key": params.get("key"),
                "class_name": type(widget).__name__,
                "object_name": widget.objectName()}

    def _handle_run_action(self, params: Dict[str, Any]) -> Dict[str, Any]:
        object_name = params.get("object_name")
        text = params.get("text")
        if not object_name and not text:
            raise RuntimeError("either object_name or text is required")

        app = QApplication.instance()
        if app is None:
            raise RuntimeError("no QApplication instance")

        actions: List[QAction] = []
        for top in app.topLevelWidgets():
            try:
                actions.extend(top.findChildren(QAction))
                actions.extend(top.actions())
            except (RuntimeError, TypeError):
                continue

        for action in actions:
            try:
                if object_name and action.objectName() == object_name:
                    action.trigger()
                    return {"ok": True, "object_name": object_name}
                if text and action.text() == text:
                    action.trigger()
                    return {"ok": True, "text": text}
            except RuntimeError:
                continue
        raise RuntimeError(f"action not found: object_name={object_name!r} text={text!r}")

    def _handle_read_log(self, params: Dict[str, Any]) -> Dict[str, Any]:
        dialog = _find_active_studio_dialog()
        if dialog is None:
            raise RuntimeError("no active HYDRA workbench dialog")
        lines = getattr(dialog, "_runtime_log_lines", [])
        max_lines = int(params.get("max_lines", 1000))
        tail = lines[-max_lines:] if max_lines > 0 else list(lines)
        return {"ok": True, "lines": tail, "total": len(lines)}

    def _handle_resize_main_window(
        self, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        width = int(params.get("width", 0))
        height = int(params.get("height", 0))
        if width <= 0 or height <= 0:
            raise RuntimeError("width and height must be positive integers")
        main_win = QApplication.instance().activeWindow()
        if main_win is None:
            raise RuntimeError("no active window")
        main_win.resize(width, height)
        return {"ok": True, "width": width, "height": height}

    def _handle_resize_docks(self, params: Dict[str, Any]) -> Dict[str, Any]:
        dock_name = params.get("dock_name")
        width = int(params.get("width", 0))
        height = int(params.get("height", 0))
        orientation = params.get("orientation", "auto")
        if not dock_name:
            raise RuntimeError("dock_name is required")
        if width <= 0 and height <= 0:
            raise RuntimeError("width and/or height must be positive")
        app = QApplication.instance()
        if app is None:
            raise RuntimeError("no QApplication instance")
        target = None
        for top in app.topLevelWidgets():
            try:
                found = top.findChildren(QDockWidget, dock_name)
            except (RuntimeError, TypeError):
                continue
            if found:
                target = found[0]
                break
        if target is None:
            raise RuntimeError(f"dock '{dock_name}' not found")
        main_win = target.parent()
        if main_win is None or not isinstance(main_win, QMainWindow):
            raise RuntimeError("dock has no QMainWindow parent")
        orient = (
            Qt.Horizontal
            if orientation == "horizontal"
            else Qt.Vertical
            if orientation == "vertical"
            else Qt.Horizontal
            if target.width() >= target.height()
            else Qt.Vertical
        )
        if orient == Qt.Horizontal:
            main_win.resizeDocks([target], [width], Qt.Horizontal)
        else:
            main_win.resizeDocks([target], [height], Qt.Vertical)
        return {
            "ok": True,
            "dock_name": dock_name,
            "width": target.width(),
            "height": target.height(),
        }

    def _handle_set_toolbox_page(
        self, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        path_str = params.get("path")
        if not path_str:
            raise RuntimeError("path is required")
        index = int(params.get("index", 0))
        root_name = params.get("root_object_name")
        root = self._resolve_root(root_name)
        if root is None:
            raise RuntimeError(
                f"root widget '{root_name}' not found" if root_name
                else "no root widget available"
            )
        path_parts = [p for p in path_str.split(".") if p]
        widget = find_widget_by_path(root, path_parts)
        if widget is None:
            raise RuntimeError(f"widget not found at path '{path_str}'")
        from qgis.PyQt.QtWidgets import QToolBox
        if not isinstance(widget, QToolBox):
            raise RuntimeError(
                f"widget at '{path_str}' is {type(widget).__name__}, not QToolBox"
            )
        n = widget.count()
        if not (0 <= index < n):
            raise RuntimeError(f"index {index} out of range (0..{n - 1})")
        widget.setCurrentIndex(index)
        return {
            "ok": True,
            "path": path_str,
            "index": index,
            "label": widget.itemText(index),
            "count": n,
        }

    def _handle_force_dock_size(
        self, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        dock_name = params.get("dock_name")
        width = int(params.get("width", 0))
        height = int(params.get("height", 0))
        if not dock_name:
            raise RuntimeError("dock_name is required")
        if width <= 0 or height <= 0:
            raise RuntimeError("width and height must be positive integers")
        app = QApplication.instance()
        if app is None:
            raise RuntimeError("no QApplication instance")
        target = None
        for top in app.topLevelWidgets():
            try:
                found = top.findChildren(QDockWidget, dock_name)
            except (RuntimeError, TypeError):
                continue
            if found:
                target = found[0]
                break
        if target is None:
            raise RuntimeError(f"dock '{dock_name}' not found")
        if not target.isVisible():
            target.show()
        target.setMinimumSize(width, height)
        target.resize(width, height)
        main_win = target.parent()
        if main_win is not None and isinstance(main_win, QMainWindow):
            if target.width() < width or target.height() < height:
                main_win.resizeDocks(
                    [target],
                    [width],
                    Qt.Horizontal
                    if target.width() >= target.height()
                    else Qt.Vertical,
                )
        return {
            "ok": True,
            "dock_name": dock_name,
            "width": target.width(),
            "height": target.height(),
        }

    def _handle_list_dock_widgets(
        self, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Return all child widgets of a dock, with class+objectName+text.

        Uses QWidget.findChildren recursively (depth-first). Includes
        unnamed widgets; layouts are filtered out. Used to discover
        the QTabWidget / QToolBox instances hidden inside a
        QDockWidget when the widget_walker stops at the dock boundary.
        """
        dock_name = params.get("dock_name")
        if not dock_name:
            raise RuntimeError("dock_name is required")
        max_depth = int(params.get("max_depth", 10))
        target = None
        app = QApplication.instance()
        if app is not None:
            for top in app.topLevelWidgets():
                try:
                    found = top.findChildren(QDockWidget, dock_name)
                except (RuntimeError, TypeError):
                    continue
                if found:
                    target = found[0]
                    break
        if target is None:
            raise RuntimeError(f"dock '{dock_name}' not found")
        widgets: List[Dict[str, Any]] = []

        def _walk(w: QWidget, depth: int) -> None:
            if depth > max_depth:
                return
            try:
                cls = type(w).__name__
                if cls in {
                    "QStackedLayout", "QGridLayout", "QFormLayout",
                    "QBoxLayout", "QLayout",
                }:
                    return
                widgets.append({
                    "object_name": w.objectName() or "",
                    "class_name": cls,
                    "widget_id": id(w),
                    "text": w.windowTitle() if isinstance(w, QDockWidget) else (
                        w.text() if hasattr(w, "text") and callable(w.text)
                        and not isinstance(w, QAbstractButton)
                        else ""
                    ),
                    "depth": depth,
                })
            except RuntimeError:
                return
            try:
                children = list(w.children())
            except (RuntimeError, TypeError):
                children = []
            for c in children:
                if not isinstance(c, QWidget):
                    continue
                _walk(c, depth + 1)

        _walk(target, 0)
        return {
            "ok": True,
            "dock_name": dock_name,
            "count": len(widgets),
            "widgets": widgets,
        }

    def _handle_set_studio_dock_tab(
        self, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Open a specific page in a QTabWidget or QToolBox inside a dock.

        ``widget_id`` identifies the target tab widget (from
        ``list_dock_widgets`` output). ``index`` is the page index.
        Also accepts a friendly alternative: ``tab_label`` matches
        the page text via ``tabText`` / ``itemText``.
        """
        dock_name = params.get("dock_name")
        widget_id = int(params.get("widget_id", -1))
        if not dock_name:
            raise RuntimeError("dock_name is required")
        if widget_id <= 0:
            raise RuntimeError("widget_id is required")
        app = QApplication.instance()
        if app is None:
            raise RuntimeError("no QApplication instance")
        target_dock = None
        for top in app.topLevelWidgets():
            try:
                found = top.findChildren(QDockWidget, dock_name)
            except (RuntimeError, TypeError):
                continue
            if found:
                target_dock = found[0]
                break
        if target_dock is None:
            raise RuntimeError(f"dock '{dock_name}' not found")
        target_tab = None
        for w in target_dock.findChildren(QWidget):
            if id(w) == widget_id:
                target_tab = w
                break
        if target_tab is None:
            raise RuntimeError(
                f"widget with id {widget_id} not found under dock '{dock_name}'"
            )
        from qgis.PyQt.QtWidgets import QTabWidget, QToolBox, QStackedWidget
        if not isinstance(target_tab, (QTabWidget, QToolBox, QStackedWidget)):
            raise RuntimeError(
                f"widget is {type(target_tab).__name__}, not a tab container"
            )
        if "index" in params:
            index = int(params["index"])
        elif "tab_label" in params:
            label = str(params["tab_label"])
            index = -1
            for i in range(target_tab.count()):
                getter = (
                    target_tab.tabText
                    if isinstance(target_tab, QTabWidget)
                    else target_tab.itemText
                )
                if getter(i) == label:
                    index = i
                    break
            if index < 0:
                raise RuntimeError(
                    f"no tab with label {label!r}; "
                    f"available: "
                    f"{[target_tab.tabText(i) if isinstance(target_tab, QTabWidget) else target_tab.itemText(i) for i in range(target_tab.count())]}"
                )
        else:
            raise RuntimeError("either index or tab_label is required")
        n = target_tab.count()
        if not (0 <= index < n):
            raise RuntimeError(f"index {index} out of range (0..{n - 1})")
        target_tab.setCurrentIndex(index)
        getter = (
            target_tab.tabText
            if isinstance(target_tab, QTabWidget)
            else target_tab.itemText
        )
        return {
            "ok": True,
            "dock_name": dock_name,
            "widget_id": widget_id,
            "class_name": type(target_tab).__name__,
            "index": index,
            "label": getter(index),
            "count": n,
        }

    def _handle_list_dock_tab_pages(
        self, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Return the page labels of a QTabWidget or QToolBox inside a dock."""
        dock_name = params.get("dock_name")
        widget_id = int(params.get("widget_id", -1))
        if not dock_name:
            raise RuntimeError("dock_name is required")
        if widget_id <= 0:
            raise RuntimeError("widget_id is required")
        app = QApplication.instance()
        if app is None:
            raise RuntimeError("no QApplication instance")
        target_dock = None
        for top in app.topLevelWidgets():
            try:
                found = top.findChildren(QDockWidget, dock_name)
            except (RuntimeError, TypeError):
                continue
            if found:
                target_dock = found[0]
                break
        if target_dock is None:
            raise RuntimeError(f"dock '{dock_name}' not found")
        target_tab = None
        for w in target_dock.findChildren(QWidget):
            if id(w) == widget_id:
                target_tab = w
                break
        if target_tab is None:
            raise RuntimeError(
                f"widget with id {widget_id} not found under dock '{dock_name}'"
            )
        from qgis.PyQt.QtWidgets import QTabWidget, QToolBox
        if not isinstance(target_tab, (QTabWidget, QToolBox)):
            raise RuntimeError(
                f"widget is {type(target_tab).__name__}, not a tab container"
            )
        getter = (
            target_tab.tabText
            if isinstance(target_tab, QTabWidget)
            else target_tab.itemText
        )
        current = target_tab.currentIndex()
        pages = [
            {"index": i, "label": getter(i)}
            for i in range(target_tab.count())
        ]
        return {
            "ok": True,
            "dock_name": dock_name,
            "widget_id": widget_id,
            "class_name": type(target_tab).__name__,
            "count": target_tab.count(),
            "current_index": current,
            "current_label": getter(current) if 0 <= current < target_tab.count() else "",
            "pages": pages,
        }

    def _handle_run_simulation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        dialog = _find_active_studio_dialog()
        if dialog is None:
            raise RuntimeError("no active HYDRA workbench dialog")

        run_duration_text = params.get("run_duration_text")
        output_interval_text = params.get("output_interval_text")

        if run_duration_text is not None:
            w = _find_widget_by_name("run_time_edit")
            if w is None:
                raise RuntimeError("run_time_edit widget not found")
            result = set_widget_value(w, str(run_duration_text))
            if not result.get("ok"):
                raise RuntimeError(result.get("error", "failed to set run duration"))

        if output_interval_text is not None:
            w = _find_widget_by_name("output_interval_edit")
            if w is None:
                raise RuntimeError("output_interval_edit widget not found")
            result = set_widget_value(w, str(output_interval_text))
            if not result.get("ok"):
                raise RuntimeError(result.get("error", "failed to set output interval"))

        run_btn = _find_widget_by_name("run_btn")
        if run_btn is None:
            raise RuntimeError("run_btn not found")

        controller = getattr(dialog, "_controller", None)
        if controller is None:
            raise RuntimeError("workbench controller not available")

        # Click the Run button and wait for the worker to appear.
        QTest.mouseClick(run_btn, Qt.LeftButton)
        app = QApplication.instance()
        if app is not None:
            app.processEvents()

        worker = None
        deadline = time.monotonic() + float(params.get("startup_timeout", 10.0))
        while time.monotonic() < deadline:
            worker = getattr(controller, "_simulation_worker", None)
            if worker is not None:
                break
            if app is not None:
                app.processEvents()
            time.sleep(0.05)

        if worker is None:
            raise RuntimeError(
                "Run button clicked but no simulation worker started; "
                "check that a mesh is loaded and inputs are valid."
            )

        # Wait for completion/timeout via a local event loop.
        loop = QEventLoop()
        status = {"result": "timeout"}
        timeout_ms = int(float(params.get("timeout", 60.0)) * 1000)
        timer = QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(loop.quit)

        def _on_finished(result: Any) -> None:
            status["result"] = "finished"
            status["run_id"] = getattr(result, "run_id", "")
            loop.quit()

        def _on_failed(message: str) -> None:
            status["result"] = "failed"
            status["message"] = str(message)
            loop.quit()

        try:
            worker.compute_finished.connect(_on_finished)
        except (RuntimeError, TypeError):
            pass
        try:
            worker.compute_failed.connect(_on_failed)
        except (RuntimeError, TypeError):
            pass

        # If the worker has not been started yet, start it now that we are
        # listening for its completion signals.
        start_fn = getattr(worker, "start", None)
        if callable(start_fn):
            try:
                start_fn()
            except RuntimeError:
                pass

        timer.start(timeout_ms)
        loop.exec_()
        timer.stop()

        # Best-effort disconnect to avoid leaking slots across calls.
        try:
            worker.compute_finished.disconnect(_on_finished)
        except (RuntimeError, TypeError):
            pass
        try:
            worker.compute_failed.disconnect(_on_failed)
        except (RuntimeError, TypeError):
            pass

        return {"ok": True, "status": status.get("result"),
                "run_id": status.get("run_id", ""),
                "message": status.get("message", "")}

    def _resolve_root(self, root_name: Optional[str]) -> Optional[QWidget]:
        return _resolve_root_widget(root_name)

    # ── GPU Direct Viewer handlers (Phase 1) ──────────────────────────────

_HYDRA_MCP_BRIDGE_INSTANCE: Optional["HydraMcpBridge"] = None


def bootstrap_bridge_if_needed() -> Optional[HydraMcpBridge]:
    """Start the HYDRA MCP bridge from inside a live QGIS process.

    Safe to call repeatedly.  If a bridge instance already exists and is
    alive, it is returned.  This is the Python-console bootstrap entry point
    for the live agent-assisted modeling workflow when ``HYDRA_MCP_BRIDGE``
    was not set at QGIS launch time.

    Returns:
        The running :class:`HydraMcpBridge` instance, or ``None`` if the
        bridge could not be created.
    """
    global _HYDRA_MCP_BRIDGE_INSTANCE
    if _HYDRA_MCP_BRIDGE_INSTANCE is not None:
        if _HYDRA_MCP_BRIDGE_INSTANCE.is_alive():
            return _HYDRA_MCP_BRIDGE_INSTANCE
        _HYDRA_MCP_BRIDGE_INSTANCE = None

    os.environ["HYDRA_MCP_BRIDGE"] = "1"
    bridge = HydraMcpBridge()
    bridge.start()
    _HYDRA_MCP_BRIDGE_INSTANCE = bridge
    return bridge


def restart_hydra_mcp_bridge() -> Optional[HydraMcpBridge]:
    """Stop any existing bridge and start a fresh one.

    This is the GUI menu entry point. It guarantees a new socket/token pair,
    which is useful when an attached agent has lost its token file or when
    the bridge's QObject has been destroyed by QGIS lifecycle events.

    Returns:
        The new running :class:`HydraMcpBridge` instance, or ``None`` if the
        bridge could not be created.
    """
    global _HYDRA_MCP_BRIDGE_INSTANCE
    existing = _HYDRA_MCP_BRIDGE_INSTANCE
    if existing is not None and existing.is_alive():
        try:
            existing.stop()
        except Exception:
            pass
    _HYDRA_MCP_BRIDGE_INSTANCE = None
    return bootstrap_bridge_if_needed()


def _ensure_qgis_pyqt_for_standalone() -> None:
    """When run outside QGIS, redirect qgis.PyQt to the real PyQt5.

    The widget_walker imports from qgis.PyQt; this lets the bridge script
    function as a standalone test target without a full QGIS install.
    """
    if "qgis.PyQt" in sys.modules:
        return
    try:
        import qgis.PyQt  # noqa: F401
    except Exception:
        import types
        qgis_pkg = types.ModuleType("qgis")
        qgis_pyqt = types.ModuleType("qgis.PyQt")
        qgis_pkg.PyQt = qgis_pyqt
        sys.modules["qgis"] = qgis_pkg
        sys.modules["qgis.PyQt"] = qgis_pyqt
        for sub in ("QtCore", "QtWidgets", "QtGui", "QtNetwork", "QtTest"):
            src = __import__(f"PyQt5.{sub}")
            m = types.ModuleType(f"qgis.PyQt.{sub}")
            for k, v in vars(getattr(src, sub)).items():
                if not k.startswith("_"):
                    setattr(m, k, v)
            sys.modules[f"qgis.PyQt.{sub}"] = m
            setattr(qgis_pyqt, sub, m)


if __name__ == "__main__":
    _ensure_qgis_pyqt_for_standalone()

    # When this script is injected via ``qgis --code ...`` the file runs as
    # ``__main__`` INSIDE a real QGIS process.  The plugin has already
    # autostarted the bridge (HYDRA_MCP_BRIDGE=1); starting a second one and
    # showing the standalone test window would shadow QgisApp as the active
    # window, so every root-less ``get_widget_tree`` would walk the 2-node
    # test window instead of the real UI.  Only run the standalone path when
    # the script is executed with plain Python (qgis.core is not importable).
    try:
        import qgis.core  # noqa: F401
    except ImportError:
        pass
    else:
        raise SystemExit(0)

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    # Build a tiny widget tree so standalone smoke tests have something to walk.
    window = QWidget()
    window.setObjectName("HydraMcpBridgeTestWindow")
    window.setWindowTitle("HYDRA MCP Bridge Test")
    child = QWidget(window)
    child.setObjectName("HydraMcpBridgeChild")
    window.show()

    bridge = HydraMcpBridge()
    bridge.start()

    # Keep the event loop alive until QGIS/quit.
    sys.exit(app.exec_())
