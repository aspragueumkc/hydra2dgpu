"""QLocalSocket client for the HYDRA MCP QGIS bridge.

This module is intentionally importable even when PyQt5/Qt is missing: the
MCP server must be able to start and serve Tier A tools in headless
environments.  Qt-dependent code is deferred to method calls, and
``BridgeClient`` raises a clear runtime error if instantiated without Qt.
"""
from __future__ import annotations

import json
import os
import stat
import struct
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tools.hydra_mcp.workspace import WorkspacePath, WorkspacePathError, default_workspace

# Pure-Python framing helpers are exposed at module level so the bridge
# (qgis_bridge.py) can import them without duplicating the wire format.


# Maximum accepted frame payload (16 MiB).  Frames whose length prefix exceeds
# this are rejected immediately to prevent an unauthenticated peer from
# forcing unbounded buffering.  Screenshot blobs can be several MiB after
# base64 encoding, so the cap is kept well above typical PNG/JPEG captures.
MAX_FRAME_BYTES: int = 16 << 20


class FrameTooLargeError(ValueError):
    """Raised by ``decode_messages`` when a frame exceeds ``MAX_FRAME_BYTES``.

    The exception carries the offending length and the configured cap so
    callers (server-side bridge, client-side ``BridgeClient``) can surface a
    precise error and drop the offending socket without leaking data.
    """

    def __init__(self, length: int, max_bytes: int = MAX_FRAME_BYTES):
        self.length = length
        self.max_bytes = max_bytes
        super().__init__(
            f"frame length {length} exceeds MAX_FRAME_BYTES={max_bytes}"
        )


def encode_message(obj: Dict[str, Any]) -> bytes:
    """Return length-prefixed UTF-8 JSON bytes for *obj*.

    Raises:
        ValueError: if the encoded payload exceeds ``MAX_FRAME_BYTES``.
    """
    payload = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    if len(payload) > MAX_FRAME_BYTES:
        raise ValueError(
            f"encoded payload {len(payload)} bytes exceeds "
            f"MAX_FRAME_BYTES={MAX_FRAME_BYTES}"
        )
    return struct.pack("!I", len(payload)) + payload


def decode_messages(data: bytes) -> Tuple[List[Dict[str, Any]], bytes]:
    """Parse all complete length-prefixed JSON messages in *data*.

    Returns ``(messages, leftover)``.  *leftover* is buffered until more data
    arrives.  Raises :class:`FrameTooLargeError` if the length prefix of any
    frame in *data* declares a payload larger than ``MAX_FRAME_BYTES`` —
    this prevents an unauthenticated peer from forcing the local side to
    buffer or decode a multi-gigabyte frame.
    """
    messages: List[Dict[str, Any]] = []
    while len(data) >= 4:
        length = struct.unpack("!I", data[:4])[0]
        if length > MAX_FRAME_BYTES:
            # Drop the length-prefixed header but keep anything that follows;
            # the caller should close the offending socket.  We only buffer
            # what we've already accepted — never allocate ``length`` bytes.
            raise FrameTooLargeError(length)
        if len(data) < 4 + length:
            break
        payload = data[4 : 4 + length]
        messages.append(json.loads(payload.decode("utf-8")))
        data = data[4 + length :]
    return messages, data


try:
    from PyQt5 import QtCore, QtNetwork

    _QT_AVAILABLE = True
except Exception:  # pragma: no cover - exercised on headless / uv envs
    _QT_AVAILABLE = False


class BridgeClient:
    """Connect to a running HYDRA MCP QGIS bridge and call JSON-RPC handlers."""

    def __init__(
        self,
        token_path: Optional[str] = None,
        socket_name: Optional[str] = None,
        token: Optional[str] = None,
        timeout: float = 30.0,
    ):
        if not _QT_AVAILABLE:
            raise RuntimeError(
                "BridgeClient requires PyQt5/QtNetwork (Qt is not available in "
                "this interpreter)."
            )
        self.timeout_ms = int(timeout * 1000)
        self._socket = QtNetwork.QLocalSocket()
        self._socket_path: Optional[str] = None

        self.token: str = ""
        self.socket_name: str = ""

        if token_path is not None:
            self._load_token_file(Path(token_path))
        elif socket_name is not None and token is not None:
            self.socket_name = socket_name
            self.token = token
        else:
            # Final fallback: discover the newest token file for the current user.
            self._discover_token_file()

        if not self.socket_name:
            raise RuntimeError(
                "Could not determine bridge socket name. Provide token_path "
                "or set HYDRA_MCP_BRIDGE_SOCKET / HYDRA_MCP_BRIDGE_TOKEN."
            )
        if not self.token:
            raise RuntimeError(
                "Could not determine bridge token. Provide token_path or set "
                "HYDRA_MCP_BRIDGE_TOKEN."
            )

    def _load_token_file(self, path: Path) -> None:
        # Explicit user-supplied token_path must live inside the workspace —
        # this is the channel the agent controls directly, so we apply the
        # same containment rule as other filesystem inputs.  Auto-discovered
        # files in /tmp / XDG_RUNTIME_DIR use a different validation path
        # (ownership + mode) and are accepted even though they sit outside
        # the workspace, because that is where the bridge writes them by
        # design.  Discovery calls ``_parse_token_file`` directly so it
        # bypasses this containment check.
        try:
            contained = default_workspace().resolve_under(path)
        except WorkspacePathError as exc:
            raise RuntimeError(
                f"token_path {str(path)!r} rejected: {exc}"
            ) from exc
        raw = json.loads(contained.read_text(encoding="utf-8"))
        self.socket_name = str(raw.get("socket_name", ""))
        self.token = str(raw.get("token", ""))
        self._socket_path = str(contained)

    def _parse_token_file(self, path: Path) -> None:
        """Parse a token file directly, bypassing ``WorkspacePath`` containment.

        Reserved for callers that have already validated the path by other
        means — currently only :meth:`_discover_token_file`, which checks
        ownership and mode 0o600 before this is called.  Auto-discovered
        token files live in ``$XDG_RUNTIME_DIR`` or ``/tmp`` by design;
        running them through ``resolve_under`` would reject every
        candidate in the default workspace, breaking the no-token-path
        code path that is the only way to reach a Phase 2 session.
        """
        raw = json.loads(path.read_text(encoding="utf-8"))
        self.socket_name = str(raw.get("socket_name", ""))
        self.token = str(raw.get("token", ""))
        self._socket_path = str(path)

    def _discover_token_file(self) -> None:
        """Look for a token file written by a bridge started by this user."""
        token = os.environ.get("HYDRA_MCP_BRIDGE_TOKEN", "")
        socket_name = os.environ.get("HYDRA_MCP_BRIDGE_SOCKET", "")
        if socket_name and token:
            self.socket_name = socket_name
            self.token = token
            return

        # Scan well-known directories for the most recently modified token file.
        dirs: List[Path] = []
        runtime = os.environ.get("XDG_RUNTIME_DIR")
        if runtime:
            dirs.append(Path(runtime))
        dirs.append(Path(tempfile.gettempdir()))

        candidates: List[Path] = []
        for d in dirs:
            if not d.exists():
                continue
            for f in d.glob("hydra_mcp_bridge_*.json"):
                try:
                    st = f.stat()
                except (OSError, AttributeError):
                    continue
                if st.st_uid != os.getuid():
                    continue
                # Defense-in-depth: refuse auto-discovered token files whose
                # mode is broader than 0600 (a leftover from before the
                # atomic-create fix).  User-supplied token_paths go through
                # WorkspacePath instead; this only constrains what discovery
                # is willing to pick up.
                if stat.S_IMODE(st.st_mode) & 0o077:
                    continue
                candidates.append(f)
        if candidates:
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            # IMPORTANT: parse directly, not through ``_load_token_file``.
            # ``_load_token_file`` enforces workspace containment which
            # would reject every file in ``/tmp`` / ``$XDG_RUNTIME_DIR``
            # — the directories the bridge writes to by design.  We have
            # already validated ownership and mode above, so the
            # containment check is unnecessary *and* wrong here.
            self._parse_token_file(candidates[0])

    def connect(self) -> None:
        """Connect to the bridge socket with timeout."""
        self._socket.connectToServer(self.socket_name)
        if not self._socket.waitForConnected(self.timeout_ms):
            err = self._socket.errorString()
            raise RuntimeError(f"Could not connect to bridge {self.socket_name}: {err}")

    def close(self) -> None:
        """Close the socket gracefully."""
        if self._socket.state() != QtNetwork.QLocalSocket.UnconnectedState:
            self._socket.disconnectFromServer()
            self._socket.waitForDisconnected(1000)

    def _send(self, obj: Dict[str, Any]) -> None:
        data = encode_message(obj)
        written = self._socket.write(data)
        if written < 0 or not self._socket.waitForBytesWritten(self.timeout_ms):
            raise RuntimeError("Failed to write to bridge socket")

    def _call(self, method: str, **params: Any) -> Dict[str, Any]:
        """Make a JSON-RPC request and return the result payload."""
        if self._socket.state() != QtNetwork.QLocalSocket.ConnectedState:
            self.connect()

        request_id = int(time.time() * 1000) & 0xFFFFFFFF
        params["token"] = self.token
        request = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": request_id,
        }
        self._send(request)

        deadline = time.monotonic() + (self.timeout_ms / 1000.0)
        buffer = b""
        while time.monotonic() < deadline:
            if self._socket.waitForReadyRead(100):
                buffer += bytes(self._socket.readAll())
                try:
                    messages, buffer = decode_messages(buffer)
                except FrameTooLargeError as exc:
                    # Server-side enforcement: surface the same cap the
                    # server applies.  Abort the socket so we do not
                    # accumulate an oversized buffer.
                    self._socket.abort()
                    raise RuntimeError(
                        f"frame too large: {exc.length} bytes exceeds "
                        f"MAX_FRAME_BYTES={exc.max_bytes}"
                    ) from exc
                for msg in messages:
                    if msg.get("id") != request_id:
                        continue
                    if "error" in msg:
                        raise RuntimeError(
                            f"Bridge error {msg['error'].get('code')}: "
                            f"{msg['error'].get('message')}"
                        )
                    return msg.get("result", {})
        raise RuntimeError(f"Timeout waiting for bridge response to {method}")

    def ping(self) -> Dict[str, Any]:
        """Bridge handshake: returns version and token path."""
        return self._call("ping")

    def get_widget_tree(self, root: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return the live widget tree from the bridge as a list of dicts."""
        return self._call("get_widget_tree", root_object_name=root)

    def find_widget(self, name: str) -> Optional[Dict[str, Any]]:
        """Return the single widget node whose objectName matches *name*."""
        return self._call("find_widget", object_name=name)

    def get_value(self, path: str, root: Optional[str] = None) -> Dict[str, Any]:
        """Return the current value of the widget at *path*.

        Args:
            path: Dot-separated widget-objectName path from the root widget.
            root: Optional ``objectName`` of the widget to use as the tree root.
                Omit to auto-detect (prefers active window).
        """
        return self._call("get_value", path=path, root_object_name=root)

    def set_value(
        self, path: str, value: Any, root: Optional[str] = None
    ) -> Dict[str, Any]:
        """Set the value of the widget at *path*.

        Args:
            path: Dot-separated widget-objectName path from the root widget.
            value: The new value (type must match the widget class).
            root: Optional ``objectName`` of the widget to use as the tree root.
                Omit to auto-detect (prefers active window).
        """
        return self._call("set_value", path=path, value=value, root_object_name=root)

    def screenshot(
        self,
        path: Optional[str] = None,
        format: str = "png",
        root: Optional[str] = None,
        target: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Capture a screenshot of a top-level widget.

        Args:
            path: Dot-separated widget-objectName path from the root widget.
                Legacy mode; ignored when *target* is provided.
            format: Image format — ``"png"`` (default) or ``"jpg"`` / ``"jpeg"``.
            root: Optional ``objectName`` of the widget to use as the tree root.
                Omit to auto-detect (prefers active window).
            target: One of ``"dialog"``, ``"dock"``, ``"canvas"``.  When set,
                the bridge resolves the target to a live QWidget and bypasses
                the ``path``-based resolution.
        """
        return self._call(
            "screenshot", path=path, format=format,
            root_object_name=root, target=target,
        )

    # ── Phase 3: behavioral GUI actions ───────────────────────────────────────

    def click_widget(
        self,
        path: Optional[str] = None,
        object_name: Optional[str] = None,
        root: Optional[str] = None,
        x: Optional[float] = None,
        y: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Click a widget using QTest.

        Args:
            x, y: Optional click position. Values in [0.0, 1.0] are relative
                to the widget rectangle; larger values are absolute pixel
                offsets from the widget's top-left corner.
        """
        return self._call(
            "click_widget", path=path, object_name=object_name,
            root_object_name=root, x=x, y=y,
        )

    def key_press(
        self,
        key: str,
        path: Optional[str] = None,
        object_name: Optional[str] = None,
        root: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send a key press to a widget using QTest."""
        return self._call(
            "key_press", key=key, path=path, object_name=object_name,
            root_object_name=root,
        )

    def run_action(
        self,
        object_name: Optional[str] = None,
        text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Trigger a QAction by objectName or text."""
        return self._call(
            "run_action", object_name=object_name, text=text,
        )

    def read_log(self, max_lines: int = 1000) -> Dict[str, Any]:
        """Read the active workbench runtime log."""
        return self._call("read_log", max_lines=max_lines)

    def run_simulation(
        self,
        run_duration_text: Optional[str] = None,
        output_interval_text: Optional[str] = None,
        timeout: float = 60.0,
        startup_timeout: float = 10.0,
    ) -> Dict[str, Any]:
        """Set inputs, click Run, and wait for compute_finished."""
        return self._call(
            "run_simulation",
            run_duration_text=run_duration_text,
            output_interval_text=output_interval_text,
            timeout=timeout,
            startup_timeout=startup_timeout,
        )

    def __enter__(self) -> "BridgeClient":
        self.connect()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
