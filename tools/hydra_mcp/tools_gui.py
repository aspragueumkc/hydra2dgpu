"""Tier B live-GUI tools for the HYDRA MCP server (Phase 2.A + 2.B).

These tools drive a live QGIS session through the QGIS bridge
(``qgis_bridge.py``).  They degrade gracefully — returning a clear error —
when no bridge session is active.

Architecture
------------
The bridge is a ``QLocalServer`` that runs inside a live QGIS process.
It is injected via the actual QGIS desktop launcher (the ``qgis`` shell
script — NOT plain ``python3``)::

    qgis --noplugins --code tools/hydra_mcp/qgis_bridge.py

or auto-started when the HYDRA plugin loads with ``HYDRA_MCP_BRIDGE=1``.
The MCP server (this process) connects to the bridge as a ``QLocalSocket``
client (``bridge_client.BridgeClient``) and forwards JSON-RPC calls.

Token discovery
---------------
The bridge writes a 0600 JSON file at startup containing the socket name,
token, and version.  Three ways to locate it:

1. Explicit ``token_path`` argument to each tool.
2. ``HYDRA_MCP_BRIDGE_TOKEN`` / ``HYDRA_MCP_BRIDGE_SOCKET`` env vars (set by
   the bridge startup output line
   ``HYDRA_MCP_BRIDGE_READY <socket> <token_path>``).
3. Auto-discovery: scan ``$XDG_RUNTIME_DIR`` / ``/tmp`` for the newest
   ``hydra_mcp_bridge_*.json`` owned by the current user.

All public functions return structured JSON (``{"ok": true, ...}`` or
``{"ok": false, "error": ...}``) — never a traceback.
"""
from __future__ import annotations

import json
import os
import queue
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Make the repo importable regardless of the caller's PYTHONPATH.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# BridgeClient is importable even without Qt (raises at instantiation).
from tools.hydra_mcp import bridge_client
from tools.hydra_mcp.workspace import WorkspacePath, WorkspacePathError, default_workspace

# Deferred: BridgeClient needs PyQt5 — the tool itself may be called from
# a headless env that has numpy but not PyQt5.  The actual call is wrapped
# in a helper so the module loads without PyQt5.
_BridgeClient: Optional[type] = None


def _get_bridge_client(
    token_path: Optional[str] = None,
    timeout: float = 30.0,
) -> bridge_client.BridgeClient:
    global _BridgeClient
    if _BridgeClient is None:
        _BridgeClient = bridge_client.BridgeClient
    return _BridgeClient(token_path=token_path, timeout=timeout)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _bridge_not_available(token_path: Optional[str] = None) -> Dict[str, Any]:
    msg = (
        "No active QGIS bridge session found. "
        "Start one with gui_launch(mode='offscreen') or gui_launch(mode='xvfb'), "
        "or set HYDRA_MCP_BRIDGE_TOKEN and HYDRA_MCP_BRIDGE_SOCKET env vars."
    )
    if token_path:
        msg += f"  (token_path={token_path})"
    return {"ok": False, "error": msg}


def _err(message: str, **context: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {"ok": False, "error": message}
    out.update(context)
    return out


# ── Phase 3: launched-process lifecycle registry ───────────────────────────────


class _ProcessRegistry:
    """Track QGIS processes launched by ``gui_launch`` and shut them down safely.

    Shutdown escalates from SIGTERM to SIGKILL so a hung QGIS process never
    leaks after a test or agent session ends.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._procs: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        token_path: str,
        proc: subprocess.Popen,
        session_id: str = "",
        mode: str = "",
    ) -> None:
        with self._lock:
            self._procs[str(token_path)] = {
                "proc": proc,
                "pid": proc.pid,
                "session_id": session_id,
                "mode": mode,
            }

    def get(self, token_path: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._procs.get(str(token_path))

    def terminate(
        self,
        token_path: str,
        term_timeout: float = 5.0,
        kill_timeout: float = 2.0,
    ) -> Dict[str, Any]:
        """Send SIGTERM, wait, then SIGKILL if still alive."""
        with self._lock:
            record = self._procs.pop(str(token_path), None)
        if record is None:
            return {"ok": False, "error": f"No launched process for token_path={token_path}"}

        proc = record["proc"]
        pid = record["pid"]

        def _is_alive() -> bool:
            if proc is not None:
                return proc.poll() is None
            try:
                os.kill(pid, 0)
                return True
            except OSError:
                return False

        if not _is_alive():
            return {"ok": True, "pid": pid, "action": "already_exited"}

        # Phase 1: SIGTERM
        try:
            if proc is not None:
                proc.terminate()
            else:
                os.kill(pid, signal.SIGTERM)
        except OSError:
            pass

        deadline = time.monotonic() + float(term_timeout)
        while time.monotonic() < deadline:
            if not _is_alive():
                return {"ok": True, "pid": pid, "action": "terminated"}
            time.sleep(0.05)

        # Phase 2: SIGKILL escalation
        try:
            if proc is not None:
                proc.kill()
            else:
                os.kill(pid, signal.SIGKILL)
        except OSError:
            pass

        deadline = time.monotonic() + float(kill_timeout)
        while time.monotonic() < deadline:
            if not _is_alive():
                return {"ok": True, "pid": pid, "action": "killed"}
            time.sleep(0.05)

        return {
            "ok": False,
            "pid": pid,
            "action": "still_running",
            "error": "Process survived SIGKILL",
        }


_PROCESS_REGISTRY = _ProcessRegistry()


def gui_launch(
    mode: str = "offscreen",
    project: Optional[str] = None,
    timeout: float = 60.0,
) -> Dict[str, Any]:
    """Launch a QGIS instance with the HYDRA MCP bridge injected.

    This starts a new QGIS process with the bridge script loaded via
    ``--code``, waits for the bridge to write its token file, and returns
    the session metadata so subsequent ``gui_widget_tree`` / ``gui_find_widget``
    calls know where to connect.

    Args:
        mode: Launch mode — one of:
            - ``"offscreen"``: ``QT_QPA_PLATFORM=offscreen`` (no display,
              suitable for automated testing on a headless machine).
            - ``"xvfb"``: Xvfb virtual display (requires Xvfb installed;
              suitable for CI without a real GPU).
            - ``"display"``: wait for a bridge token file from a running
              QGIS session.  Start QGIS with ``HYDRA_MCP_BRIDGE=1`` to
              auto-start the bridge, or inject it via the Python console
              bootstrap (see README).
        project: Optional path to a ``.qgs`` / ``.qgz`` QGIS project to open
            on startup.  Pass ``None`` to open a blank QGIS session.
        timeout: Seconds to wait for the bridge token file to appear.
            The bridge typically starts within 5–10 s even on a slow machine.
            In ``display`` mode this timeout also applies while polling for
            an already-running bridge's token file.

    Returns:
        ``{"ok": true, session_id, socket_name, token_path, mode, pid}``
        on success; ``{"ok": false, "error": ...}`` on failure.

    Example output::

        {
            "ok": true,
            "session_id": "aaron_12345_abc123",
            "socket_name": "hydra_mcp_bridge_aaron_12345_abc123",
            "token_path": "/run/user/1000/hydra_mcp_bridge_aaron_12345_abc123.json",
            "mode": "offscreen",
            "pid": 54321
        }
    """
    mode = str(mode).strip().lower()
    if mode not in ("offscreen", "xvfb", "display"):
        return _err(
            f"Invalid mode {mode!r}. Expected one of: offscreen, xvfb, display."
        )

    # In display mode the bridge is assumed to already be running;
    # wait for the token file written by the plugin auto-start hook
    # (or manual console bootstrap) to appear and become connectable.
    if mode == "display":
        return _wait_for_active_bridge(timeout=timeout)

    # Locate the actual `qgis` desktop binary (NOT the Python interpreter).
    # The bridge is injected via `qgis --code <script>`, so we need the launcher
    # script that boots the QGIS application — passing `--code` to plain
    # `python3` would never work.
    qgis_binary = _find_qgis_binary()
    if qgis_binary is None:
        return _err(
            "QGIS desktop binary not found. "
            "Ensure QGIS is installed (e.g. via the OS package manager or a "
            "conda env named 'qgis_stable') and that the `qgis` launcher is on "
            "PATH. You can override the location with the QGIS_BINARY "
            "environment variable."
        )

    # Resolve the bridge script path relative to the repo root.
    bridge_script = _REPO_ROOT / "tools" / "hydra_mcp" / "qgis_bridge.py"
    if not bridge_script.exists():
        return _err(
            f"Bridge script not found: {bridge_script}. "
            "Ensure the repo is fully checked out."
        )

    # Build the launch command and env. The helper:
    #   * inherits (or constructs) ``QGIS_PLUGINPATH`` so the symlinked
    #     ``hydra2dgpu`` plugin loads,
    #   * sets ``HYDRA_MCP_BRIDGE=1`` so ``hydra_plugin.py`` autostarts the
    #     MCP bridge,
    #   * omits ``--noplugins`` so plugin loading actually happens.
    # See ``_build_launch_env_and_cmd`` and the P0.4 fix commit for context.
    env, cmd, build_err = _build_launch_env_and_cmd(
        mode=mode,
        qgis_binary=qgis_binary,
        project=project,
        bridge_script=bridge_script,
    )
    if build_err is not None:
        return build_err

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            # Merge stderr into stdout so the single reader thread in
            # _wait_for_bridge_ready drains both streams for the life of the
            # session — a chatty QGIS would otherwise block on a full stderr
            # pipe buffer and silently hang.
            stderr=subprocess.STDOUT,
            env=env,
        )
    except OSError as exc:
        return _err(f"Failed to spawn QGIS: {exc}")

    pid = proc.pid

    # Wait for the bridge to print its readiness line on stdout. A background
    # thread drains the pipe line-by-line; we block on a queue with the
    # configured timeout. This avoids the `/proc/<pid>/fd/1` brittleness of
    # the previous implementation and prevents pipe-buffer deadlock when QGIS
    # prints a lot before the ready line.
    try:
        discovered_socket, discovered_token_path = _wait_for_bridge_ready(
            proc, timeout=timeout
        )
    except RuntimeError as exc:
        # Kill/reap the child FIRST: read() blocks until EOF, which never
        # arrives while the child is still alive (the timeout case) — the
        # previous drain-before-kill order hung forever here.
        try:
            proc.kill()
            proc.wait(timeout=5)
        except Exception:
            pass
        # Now safe to drain — the pipes hit EOF once the child is reaped.
        stdout_tail = ""
        stderr_tail = ""
        try:
            stdout_tail = (proc.stdout.read() or b"").decode(errors="replace")[:500] if proc.stdout else ""
            stderr_tail = (proc.stderr.read() or b"").decode(errors="replace")[:500] if proc.stderr else ""
        except Exception:
            pass
        return _err(
            str(exc),
            stdout=stdout_tail,
            stderr=stderr_tail,
        )

    session_id = Path(discovered_token_path).stem.replace("hydra_mcp_bridge_", "")

    _PROCESS_REGISTRY.register(
        discovered_token_path, proc, session_id=session_id, mode=mode
    )

    return {
        "ok": True,
        "session_id": session_id,
        "socket_name": discovered_socket,
        "token_path": discovered_token_path,
        "mode": mode,
        "pid": pid,
        "note": (
            "Pass token_path to gui_widget_tree() / gui_find_widget() "
            "to connect to this session."
        ),
    }


def _find_qgis_plugin_path() -> Optional[str]:
    """Locate the canonical QGIS ``share/qgis/python/plugins`` directory.

    Mirrors the well-known conda-env locations used by
    :func:`_find_qgis_binary` so the symlinked ``hydra2dgpu`` plugin is
    loaded even when QGIS is launched outside the env's shell.  Returns
    ``None`` when no candidate directory exists, in which case the caller
    should skip ``QGIS_PLUGINPATH`` setup.
    """
    home = Path.home()
    for candidate in (
        os.environ.get("QGIS_PLUGINPATH"),
        str(home / "miniforge3" / "envs" / "qgis_stable" / "share" / "qgis" / "python" / "plugins"),
        str(home / "anaconda3" / "envs" / "qgis_stable" / "share" / "qgis" / "python" / "plugins"),
        str(home / "micromamba" / "envs" / "qgis_stable" / "share" / "qgis" / "python" / "plugins"),
        "/opt/conda/envs/qgis_stable/share/qgis/python/plugins",
    ):
        if candidate and os.path.isdir(candidate):
            return candidate
    return None


# Launch-path settings — extracted so unit tests can assert the (env, cmd)
# pair that ``gui_launch`` builds without spawning QGIS.
_HYDRA_PLUGINPATH_FALLBACK: Optional[str] = _find_qgis_plugin_path()


def _build_launch_env_and_cmd(
    *,
    mode: str,
    qgis_binary: str,
    project: Optional[str] = None,
    bridge_script: Optional[Path] = None,
) -> Tuple[Dict[str, str], List[str], Optional[Dict[str, Any]]]:
    """Return ``(env, cmd, error_or_None)`` for launching QGIS with the bridge.

    The values returned here drive every tier-B Phase 2.A tool —
    ``gui_widget_tree`` only succeeds when ``HYDRA_MCP_BRIDGE_READY`` is
    printed by the bridge, which in turn only happens when the
    ``hydra2dgpu`` plugin loads AND sees ``HYDRA_MCP_BRIDGE=1``.

    Returns ``(env, cmd, None)`` on success, or ``({}, [], error_dict)``
    when a precondition (Xvfb, project file) is missing — the caller
    returns ``error_dict`` verbatim to the MCP client.
    """
    env = dict(os.environ)
    # Honour an inherited QGIS_PLUGINPATH so callers can override; fall back
    # to the canonical qgis_stable plugin directory so the symlinked
    # ``hydra2dgpu`` plugin is loaded by QGIS at boot.
    if "QGIS_PLUGINPATH" not in env and _HYDRA_PLUGINPATH_FALLBACK:
        env["QGIS_PLUGINPATH"] = _HYDRA_PLUGINPATH_FALLBACK
    # Auto-start the MCP bridge from the plugin's QTimer hook in
    # ``hydra_plugin.py:152-155``. Without this env var the gate
    # short-circuits and no bridge is created.
    env["HYDRA_MCP_BRIDGE"] = "1"

    if mode == "offscreen":
        env["QT_QPA_PLATFORM"] = "offscreen"
    elif mode == "xvfb":
        if "DISPLAY" not in env:
            xvfb = _start_xvfb()
            if xvfb is None:
                return {}, [], _err(
                    "mode='xvfb' requires Xvfb to be installed. "
                    "Install with: sudo apt install xvfb"
                )
            env["DISPLAY"] = xvfb

    cmd = [qgis_binary, "--noversioncheck"]
    if project:
        try:
            project_resolved = default_workspace().resolve_under(project)
        except WorkspacePathError as exc:
            return {}, [], _err(str(exc))
        if not os.path.isfile(str(project_resolved)):
            return {}, [], _err(f"Project file not found: {project}")
        cmd.extend(["--project", str(project_resolved)])
    else:
        empty_project = _REPO_ROOT / "tests" / "mocks" / "empty_project.qgs"
        if empty_project.exists():
            cmd.extend(["--project", str(empty_project)])
    if bridge_script is not None:
        cmd.extend(["--code", str(bridge_script)])
    return env, cmd, None


def _find_qgis_binary() -> Optional[str]:
    """Locate the QGIS desktop launcher script.

    Returns the path to the ``qgis`` shell-script wrapper that bootstraps
    the QGIS application (and its embedded Python). The launch command is
    ``[qgis_binary, --noplugins, --project, ..., --code, bridge_script]`` —
    NOT ``python3`` plus QGIS-only flags.

    Lookup order:
      1. ``QGIS_BINARY`` environment variable (explicit override).
      2. Well-known conda env locations (``qgis_stable`` env).
      3. ``shutil.which("qgis")`` — finds anything on ``PATH`` named ``qgis``.
      4. Common system-package locations (``/usr/bin/qgis``).
    """
    if "QGIS_BINARY" in os.environ:
        candidate = os.environ["QGIS_BINARY"]
        # If the user explicitly set QGIS_BINARY, honor it strictly: return
        # it when it exists, otherwise return None so the caller reports a
        # clear "QGIS binary not found" error rather than silently falling
        # through to a different binary.
        return candidate if os.path.exists(candidate) else None

    # Try conda env first — explicit paths to avoid relying on PATH.
    home = Path.home()
    for candidate in (
        str(home / "miniforge3" / "envs" / "qgis_stable" / "bin" / "qgis"),
        str(home / "anaconda3" / "envs" / "qgis_stable" / "bin" / "qgis"),
        str(home / "micromamba" / "envs" / "qgis_stable" / "bin" / "qgis"),
        "/opt/conda/envs/qgis_stable/bin/qgis",
    ):
        if os.path.exists(candidate):
            return candidate

    # Then search PATH for the qgis launcher script.
    for exe in ("qgis", "qgis_stable", "qgis-dev"):
        path = shutil.which(exe)
        if path:
            return path

    # Last resort: common system-package locations.
    for candidate in ("/usr/bin/qgis", "/usr/local/bin/qgis"):
        if os.path.exists(candidate):
            return candidate

    return None


def _start_xvfb(display: str = ":99", resolution: str = "1280x1024x24") -> Optional[str]:
    """Start Xvfb and return the display string, or None on failure."""
    try:
        proc = subprocess.Popen(
            ["Xvfb", display, "-screen", "0", resolution],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(1)
        if proc.poll() is None:
            return display
    except OSError:
        pass
    return None


def _wait_for_bridge_ready(
    proc: "subprocess.Popen[bytes]", timeout: float
) -> tuple[str, str]:
    """Block until *proc* prints ``HYDRA_MCP_BRIDGE_READY <socket> <token>``.

    A daemon thread reads ``proc.stdout`` line-by-line (stderr is merged into
    stdout by the caller via ``subprocess.STDOUT``). The main thread waits up
    to ``timeout`` seconds for the ready tuple; if the child dies early or the
    timeout expires, a ``RuntimeError`` is raised.

    After the ready line is seen the reader thread keeps draining the pipe to
    EOF (discarding lines). QGIS routinely logs Qt warnings; without a
    drainer the child blocks in ``write()`` once the 64 KiB OS pipe buffer
    fills and the live session silently hangs.

    This replaces the previous ``/proc/<pid>/fd/1`` polling approach, which was
    Linux-only, brittle (the fd path resolves to the pipe's write end and is
    not reliably readable), and risked buffer-fill deadlock when stdout and
    stderr were piped but never drained.
    """
    if proc.stdout is None:
        raise RuntimeError("QGIS subprocess stdout was not piped; cannot detect bridge readiness.")

    ready_queue: "queue.Queue[tuple[str, str]]" = queue.Queue()

    def _reader() -> None:
        try:
            for raw_line in iter(proc.stdout.readline, b""):
                text = raw_line.decode("utf-8", errors="replace").strip()
                if not text:
                    continue
                if text.startswith("HYDRA_MCP_BRIDGE_READY"):
                    parts = text.split()
                    if len(parts) >= 3:
                        ready_queue.put((parts[1], parts[2]))
                        # Do NOT return: keep draining the pipe to EOF so a
                        # warning-chatty QGIS never blocks on a full pipe
                        # buffer for the rest of the session.
        except (OSError, ValueError):
            # Pipe closed or invalid state — fall through; the main thread
            # will detect process death via the timeout.
            pass

    reader_thread = threading.Thread(target=_reader, name="hydra-mcp-bridge-reader", daemon=True)
    reader_thread.start()

    # Poll the queue and process liveness until timeout.
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rc = proc.poll()
        if rc is not None:
            raise RuntimeError(
                f"QGIS exited prematurely with code {rc} before announcing "
                "the bridge readiness line."
            )
        try:
            return ready_queue.get(timeout=0.25)
        except queue.Empty:
            continue

    raise RuntimeError(
        f"Timeout ({timeout}s) waiting for bridge to start. "
        "Check that QGIS launched successfully and the bridge script "
        "printed HYDRA_MCP_BRIDGE_READY to stdout."
    )


def _wait_for_active_bridge(timeout: float = 60.0) -> Dict[str, Any]:
    """Poll for an already-running bridge up to *timeout* seconds.

    The user's QGIS may still be starting; this waits for the token file
    written by the plugin auto-start hook (or the manual console bootstrap)
    to appear and become connectable.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            cli = _get_bridge_client()
            cli.connect()
            info = cli.ping()
            return {
                "ok": True,
                "session_id": Path(info.get("token_path", "")).stem.replace(
                    "hydra_mcp_bridge_", ""
                ),
                "socket_name": info.get("socket_name"),
                "token_path": info.get("token_path"),
                "mode": "display",
                "version": info.get("version"),
                "note": "Connected to an existing bridge session.",
            }
        except Exception:
            time.sleep(0.25)
    return _bridge_not_available()


def _discover_active_bridge() -> Dict[str, Any]:
    """Look for an already-running bridge and return its session info."""
    try:
        cli = _get_bridge_client()
        cli.connect()
        info = cli.ping()
        return {
            "ok": True,
            "session_id": Path(info.get("token_path", "")).stem.replace(
                "hydra_mcp_bridge_", ""
            ),
            "socket_name": info.get("socket_name"),
            "token_path": info.get("token_path"),
            "mode": "display",
            "version": info.get("version"),
            "note": "Connected to an existing bridge session.",
        }
    except Exception as exc:
        return _bridge_not_available()


# ── Tool: gui_widget_tree ───────────────────────────────────────────────────


def gui_widget_tree(
    root: Optional[str] = None,
    token_path: Optional[str] = None,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    """Return the live widget tree from the active QGIS session.

    The tree is returned as a flat list of widget nodes (matching the
    structure of ``swe2d/workbench/devtools/widget_walker.WidgetNode``) sorted
    depth-first.  The root node itself is always the first entry.

    Args:
        root: Optional ``objectName`` of the widget to use as the tree root.
            Omit to auto-detect: prefers the active window, then the first
            top-level widget with children.
        token_path: Path to the bridge token file.  Omit to auto-discover
            from env vars or the most recently created token file.
        timeout: Seconds to wait for a response from the bridge.

    Returns:
        ``{"ok": true, "nodes": [...]}`` with a list of node dicts, each::

            {
                "object_name": "meshTabRunDuration",
                "class_name": "QDoubleSpinBox",
                "widget_id": 140234567890,
                "parent_id": 140234567889,
                "text": "3600.0",
                "depth": 2
            }

        Or ``{"ok": false, "error": ...}`` if no bridge session is active
        or the bridge returned an error.
    """
    try:
        cli = _get_bridge_client(token_path=token_path, timeout=timeout)
        with cli:
            nodes = cli.get_widget_tree(root=root)
        return {"ok": True, "nodes": nodes, "root_object_name": root}
    except RuntimeError as exc:
        msg = str(exc)
        if any(kw in msg.lower() for kw in ("not available", "could not connect",
                                               "pyqt5", "qnetwork")):
            return _bridge_not_available(token_path)
        return _err(f"Bridge communication error: {exc}")
    except Exception as exc:
        return _err(f"gui_widget_tree failed: {exc}")


# ── Tool: gui_find_widget ───────────────────────────────────────────────────


def gui_find_widget(
    name: str,
    token_path: Optional[str] = None,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    """Return the single widget node whose ``objectName`` matches *name*.

    Searches across all top-level widgets in the QGIS session.

    Args:
        name: The ``objectName`` string to search for (exact match).
        token_path: Path to the bridge token file.  Omit to auto-discover.
        timeout: Seconds to wait for a response from the bridge.

    Returns:
        ``{"ok": true, "node": {...}}`` with the matching node dict
        (same shape as ``gui_widget_tree`` nodes), or
        ``{"ok": false, "error": "widget not found"}`` if no widget with
        that ``objectName`` exists in the current session.
    """
    if not name:
        return _err("name is required (the widget's objectName to search for).")

    try:
        cli = _get_bridge_client(token_path=token_path, timeout=timeout)
        with cli:
            node = cli.find_widget(name=name)
        if node is None:
            return {
                "ok": False,
                "error": f"No widget with objectName='{name}' found in the current session.",
            }
        return {"ok": True, "node": node}
    except RuntimeError as exc:
        msg = str(exc)
        if any(kw in msg.lower() for kw in ("not available", "could not connect",
                                               "pyqt5", "qnetwork")):
            return _bridge_not_available(token_path)
        return _err(f"Bridge communication error: {exc}")
    except Exception as exc:
        return _err(f"gui_find_widget failed: {exc}")


# ── Tool: gui_find_widget (Phase 2.B — dot-separated path) ───────────────────


def gui_find_widget_by_path(
    path: str,
    token_path: Optional[str] = None,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    """Return the widget at a dot-separated path and its key properties.

    Args:
        path: Dot-separated ``objectName`` path from the root widget,
            e.g. ``"central_container.simulation_tab.cfl_spin"``.
        token_path: Path to the bridge token file.  Omit to auto-discover.
        timeout: Seconds to wait for a response from the bridge.

    Returns:
        ``{"ok": true, "widget": {"object_name", "class_name", "widget_id",
        "geometry": {"x", "y", "width", "height"}, "is_visible"}}``
        or ``{"ok": false, "error": "..."}``.
    """
    if not path:
        return _err("path is required (e.g. 'central_container.simulation_tab.cfl_spin').")

    try:
        cli = _get_bridge_client(token_path=token_path, timeout=timeout)
        with cli:
            result = cli._call("find_widget", path=path)
        if result is None:
            return {
                "ok": False,
                "error": f"No widget found at path '{path}'.",
            }
        return {"ok": True, "widget": result}
    except RuntimeError as exc:
        msg = str(exc)
        if any(kw in msg.lower() for kw in ("not available", "could not connect",
                                               "pyqt5", "qnetwork")):
            return _bridge_not_available(token_path)
        return _err(f"Bridge communication error: {exc}")
    except Exception as exc:
        return _err(f"gui_find_widget_by_path failed: {exc}")


# ── Tool: gui_get_value ───────────────────────────────────────────────────────


def gui_get_value(
    path: str,
    token_path: Optional[str] = None,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    """Read the current value of the widget at *path*.

    Supports: QSpinBox (int), QDoubleSpinBox (float), QCheckBox (bool),
    QComboBox (str), QLineEdit (str), QTextEdit (str), QLabel (str).

    Args:
        path: Dot-separated ``objectName`` path, e.g.
            ``"studio_window.simulation_tab.run_duration"``.
        token_path: Path to the bridge token file.  Omit to auto-discover.
        timeout: Seconds to wait for a response from the bridge.

    Returns:
        ``{"ok": true, "type": "QDoubleSpinBox", "value": 3600.0}``
        or ``{"ok": false, "error": "widget not found at path ..."}``.
    """
    if not path:
        return _err("path is required.")

    try:
        cli = _get_bridge_client(token_path=token_path, timeout=timeout)
        with cli:
            result = cli.get_value(path=path)
        if not result.get("ok", False):
            return {"ok": False, "error": result.get("error", "unknown error")}
        return {"ok": True, "type": result["type"], "value": result["value"]}
    except RuntimeError as exc:
        msg = str(exc)
        if any(kw in msg.lower() for kw in ("not available", "could not connect",
                                               "pyqt5", "qnetwork")):
            return _bridge_not_available(token_path)
        return _err(f"Bridge communication error: {exc}")
    except Exception as exc:
        return _err(f"gui_get_value failed: {exc}")


# ── Tool: gui_set_value ───────────────────────────────────────────────────────


def gui_set_value(
    path: str,
    value: Any,
    token_path: Optional[str] = None,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    """Set the value of the widget at *path*.

    Supports: QSpinBox (int), QDoubleSpinBox (float), QCheckBox (bool),
    QComboBox (str — matches ``currentText``), QLineEdit (str),
    QTextEdit (str).

    Args:
        path: Dot-separated ``objectName`` path, e.g.
            ``"studio_window.simulation_tab.cfl_spin"``.
        value: The new value. Type must match the widget class.
        token_path: Path to the bridge token file.  Omit to auto-discover.
        timeout: Seconds to wait for a response from the bridge.

    Returns:
        ``{"ok": true}`` on success, or
        ``{"ok": false, "error": "..."}``.
    """
    if not path:
        return _err("path is required.")

    try:
        cli = _get_bridge_client(token_path=token_path, timeout=timeout)
        with cli:
            result = cli.set_value(path=path, value=value)
        if not result.get("ok", False):
            return {"ok": False, "error": result.get("error", "unknown error")}
        return {"ok": True}
    except RuntimeError as exc:
        msg = str(exc)
        if any(kw in msg.lower() for kw in ("not available", "could not connect",
                                               "pyqt5", "qnetwork")):
            return _bridge_not_available(token_path)
        return _err(f"Bridge communication error: {exc}")
    except Exception as exc:
        return _err(f"gui_set_value failed: {exc}")


# ── Tool: gui_screenshot ───────────────────────────────────────────────────────


def gui_screenshot(
    path: Optional[str] = None,
    format: str = "png",
    target: Optional[str] = None,
    token_path: Optional[str] = None,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    """Capture a screenshot of a widget in the live QGIS session.

    Args:
        path: Dot-separated ``objectName`` path, e.g.
            ``"studio_window.simulation_tab"``.  Used when *target* is
            ``None`` (the default).
        format: Image format — ``"png"`` (default) or ``"jpg"`` / ``"jpeg"``.
            JPEG uses quality=85.
        target: Which top-level widget to screenshot — one of:
            - ``"dialog"``: the active QGIS main window.
            - ``"dock"``: the first ``QDockWidget`` found.
            - ``"canvas"``: the QGIS map canvas (``QgsMapCanvas``).
            When *target* is given, *path* is ignored.  ``None`` (the
            default) means "use path if given, otherwise reject".
            The MCP surface applies its own ``"dialog"`` default; the
            ``tools_gui`` function leaves the choice to the caller so
            an explicit path is not silently overridden.
        token_path: Path to the bridge token file.  Omit to auto-discover.
        timeout: Seconds to wait for a response from the bridge.

    Returns:
        ``{"ok": true, "image_b64": "...", "format": "png", "width": 800,
        "height": 600}`` on success, or ``{"ok": false, "error": "..."}``.
    """
    target_explicit = target is not None
    if not target_explicit and not path:
        return _err(
            "target is required (one of: dialog, dock, canvas), or pass a "
            "widget path."
        )

    fmt = str(format).lower()
    if fmt not in ("png", "jpg", "jpeg"):
        return _err(
            f"Invalid format {fmt!r}. Expected one of: png, jpg, jpeg."
        )

    target_norm: Optional[str] = None
    if target_explicit:
        target_norm = str(target).strip().lower()
        if target_norm not in ("dialog", "dock", "canvas"):
            return _err(
                f"Invalid target {target!r}. Expected one of: dialog, dock, canvas."
            )

    try:
        cli = _get_bridge_client(token_path=token_path, timeout=timeout)
        with cli:
            result = cli.screenshot(
                path=path, format=fmt, target=target_norm,
            )
        if not result.get("ok", False):
            return {"ok": False, "error": result.get("error", "unknown error")}
        return {
            "ok": True,
            "image_b64": result["image_b64"],
            "format": result["format"],
            "width": result["width"],
            "height": result["height"],
        }
    except RuntimeError as exc:
        msg = str(exc)
        if any(kw in msg.lower() for kw in ("not available", "could not connect",
                                               "pyqt5", "qnetwork")):
            return _bridge_not_available(token_path)
        return _err(f"Bridge communication error: {exc}")
    except Exception as exc:
        return _err(f"gui_screenshot failed: {exc}")


# ── Phase 3: behavioral GUI tools ────────────────────────────────────────────


def _call_bridge_method(
    method: str,
    token_path: Optional[str] = None,
    timeout: float = 30.0,
    **params: Any,
) -> Dict[str, Any]:
    """Forward a behavioral RPC call to the bridge."""
    try:
        cli = _get_bridge_client(token_path=token_path, timeout=timeout)
        with cli:
            result = getattr(cli, method)(**params)
        if not result.get("ok", False):
            return {"ok": False, "error": result.get("error", "unknown error")}
        return {"ok": True, **{k: v for k, v in result.items() if k != "ok"}}
    except RuntimeError as exc:
        msg = str(exc)
        if any(kw in msg.lower() for kw in ("not available", "could not connect",
                                               "pyqt5", "qnetwork")):
            return _bridge_not_available(token_path)
        return _err(f"Bridge communication error: {exc}")
    except Exception as exc:
        return _err(f"{method} failed: {exc}")


def gui_click(
    path: Optional[str] = None,
    object_name: Optional[str] = None,
    x: Optional[float] = None,
    y: Optional[float] = None,
    token_path: Optional[str] = None,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    """Click a widget in the live QGIS session using QTest.

    Args:
        path: Dot-separated ``objectName`` path from the active window.
        object_name: Alternative to *path* — search recursively for this
            ``objectName`` (useful for dock widgets with deep nesting).
        x, y: Optional click position. Values in [0.0, 1.0] are relative
            to the widget rectangle; larger values are absolute pixel offsets
            from the widget's top-left corner.
        token_path: Path to the bridge token file.  Omit to auto-discover.
        timeout: Seconds to wait for a response from the bridge.

    Returns:
        ``{"ok": true}`` on success, or ``{"ok": false, "error": ...}``.
    """
    if not path and not object_name:
        return _err("either path or object_name is required")
    return _call_bridge_method(
        "click_widget", token_path=token_path, timeout=timeout,
        path=path, object_name=object_name, x=x, y=y,
    )


def gui_key(
    key: str,
    path: Optional[str] = None,
    object_name: Optional[str] = None,
    token_path: Optional[str] = None,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    """Send a key press to a widget using QTest.

    Args:
        key: Key name (e.g. ``"return"``, ``"a"``, ``"delete"``) or single
            character.
        path: Dot-separated ``objectName`` path.
        object_name: Alternative to *path*.
        token_path: Path to the bridge token file.  Omit to auto-discover.
        timeout: Seconds to wait for a response from the bridge.
    """
    if not key:
        return _err("key is required")
    return _call_bridge_method(
        "key_press", token_path=token_path, timeout=timeout,
        key=key, path=path, object_name=object_name,
    )


def gui_run_action(
    object_name: Optional[str] = None,
    text: Optional[str] = None,
    token_path: Optional[str] = None,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    """Trigger a QAction by objectName or text.

    Args:
        object_name: Exact ``objectName`` of the QAction.
        text: Exact menu/toolbar text of the QAction.
        token_path: Path to the bridge token file.  Omit to auto-discover.
        timeout: Seconds to wait for a response from the bridge.
    """
    if not object_name and not text:
        return _err("either object_name or text is required")
    return _call_bridge_method(
        "run_action", token_path=token_path, timeout=timeout,
        object_name=object_name, text=text,
    )


def gui_read_log(
    max_lines: int = 1000,
    token_path: Optional[str] = None,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    """Read the active HYDRA workbench runtime log.

    Args:
        max_lines: Maximum number of recent log lines to return.
        token_path: Path to the bridge token file.  Omit to auto-discover.
        timeout: Seconds to wait for a response from the bridge.

    Returns:
        ``{"ok": true, "lines": [...], "total": N}`` or error.
    """
    return _call_bridge_method(
        "read_log", token_path=token_path, timeout=timeout,
        max_lines=max_lines,
    )


def gui_run_simulation(
    run_duration_text: Optional[str] = None,
    output_interval_text: Optional[str] = None,
    timeout: float = 60.0,
    startup_timeout: float = 10.0,
    token_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Set run inputs, click Run, and wait for compute_finished.

    Args:
        run_duration_text: Optional duration text (e.g. ``"0:30"``) to set
            in ``run_time_edit`` before clicking Run.
        output_interval_text: Optional output interval text (e.g. ``"00:10"``).
        timeout: Seconds to wait for the simulation to finish after it starts.
        startup_timeout: Seconds to wait for the worker thread to appear.
        token_path: Path to the bridge token file.  Omit to auto-discover.
    """
    try:
        # The bridge needs enough time for startup + the simulation wait.
        bridge_timeout = float(timeout) + float(startup_timeout) + 10.0
        cli = _get_bridge_client(token_path=token_path, timeout=bridge_timeout)
        with cli:
            result = cli.run_simulation(
                run_duration_text=run_duration_text,
                output_interval_text=output_interval_text,
                timeout=timeout,
                startup_timeout=startup_timeout,
            )
        if not result.get("ok", False):
            return {"ok": False, "error": result.get("error", "unknown error")}
        return {"ok": True, **{k: v for k, v in result.items() if k != "ok"}}
    except RuntimeError as exc:
        msg = str(exc)
        if any(kw in msg.lower() for kw in ("not available", "could not connect",
                                               "pyqt5", "qnetwork")):
            return _bridge_not_available(token_path)
        return _err(f"Bridge communication error: {exc}")
    except Exception as exc:
        return _err(f"gui_run_simulation failed: {exc}")


def gui_close(
    token_path: Optional[str] = None,
    timeout: float = 10.0,
) -> Dict[str, Any]:
    """Shut down a QGIS session launched by ``gui_launch``.

    Sends SIGTERM first, then escalates to SIGKILL if the process does not
    exit within *timeout* seconds.

    Args:
        token_path: Token path returned by ``gui_launch``.  Omit to terminate
            the most recently launched session.
        timeout: Total seconds to wait for graceful termination before
            escalating to SIGKILL.

    Returns:
        ``{"ok": true, "pid": ..., "action": "terminated"|"killed"}`` or
        ``{"ok": false, "error": ...}``.
    """
    target_token = token_path
    if not target_token:
        # Fall back to the most recently registered token_path.
        with _PROCESS_REGISTRY._lock:
            if not _PROCESS_REGISTRY._procs:
                return _err("No launched process to close. Start one with gui_launch.")
            target_token = next(reversed(list(_PROCESS_REGISTRY._procs.keys())))

    record = _PROCESS_REGISTRY.get(target_token)
    if record is None:
        return _err(f"No launched process for token_path={target_token}")

    # Reserve half the budget for SIGTERM, half for SIGKILL.
    term_timeout = max(1.0, float(timeout) * 0.5)
    kill_timeout = max(0.5, float(timeout) - term_timeout)
    return _PROCESS_REGISTRY.terminate(
        target_token, term_timeout=term_timeout, kill_timeout=kill_timeout
    )
