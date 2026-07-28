"""Audit-only adapters for bounded GUI evidence."""
from __future__ import annotations

import base64
from typing import Any, Dict, Optional

from tools.hydra_mcp import tools_gui
from tools.hydra_mcp.workspace import WorkspacePathError, default_workspace


def gui_dump_dock(
    object_name: str,
    token_path: Optional[str] = None,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    """Return the widget subtree rooted at a named dock."""
    if not object_name:
        return tools_gui._err("object_name is required")
    found = tools_gui.gui_find_widget(
        name=object_name, token_path=token_path, timeout=timeout
    )
    if not found.get("ok"):
        return found
    tree = tools_gui.gui_widget_tree(
        root=object_name, token_path=token_path, timeout=timeout
    )
    if not tree.get("ok"):
        return tree
    return {
        "ok": True,
        "dock_object_name": object_name,
        "nodes": tree["nodes"],
    }


def gui_screenshot_path(
    out_path: str,
    path: Optional[str] = None,
    format: str = "png",
    target: Optional[str] = None,
    token_path: Optional[str] = None,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    """Capture a GUI screenshot and write it under the workspace."""
    try:
        resolved = default_workspace().resolve_under(out_path)
    except WorkspacePathError as exc:
        return tools_gui._err(str(exc))

    captured = tools_gui.gui_screenshot(
        path=path,
        format=format,
        target=target,
        token_path=token_path,
        timeout=timeout,
    )
    if not captured.get("ok"):
        return captured
    try:
        image = base64.b64decode(captured["image_b64"], validate=True)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_bytes(image)
    except (KeyError, ValueError, OSError) as exc:
        return tools_gui._err(f"gui_screenshot_path failed writing output: {exc}")
    return {
        "ok": True,
        "path": str(resolved),
        "format": captured["format"],
        "width": captured["width"],
        "height": captured["height"],
    }


def gui_describe_widget(
    path: str,
    token_path: Optional[str] = None,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    """Return complete bridge-side metadata for a widget path."""
    if not path:
        return tools_gui._err("path is required")
    try:
        cli = tools_gui._get_bridge_client(token_path=token_path, timeout=timeout)
        with cli:
            result = cli._call("describe_widget", path=path)
        if result is None:
            return tools_gui._err(f"No widget found at path '{path}'.")
        return {"ok": True, "widget": result}
    except RuntimeError as exc:
        msg = str(exc)
        if any(
            kw in msg.lower()
            for kw in ("not available", "could not connect", "pyqt5", "qnetwork")
        ):
            return tools_gui._bridge_not_available(token_path)
        return tools_gui._err(f"Bridge communication error: {exc}")
    except Exception as exc:
        return tools_gui._err(f"gui_describe_widget failed: {exc}")
