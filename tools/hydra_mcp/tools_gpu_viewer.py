"""MCP tools for the GPU Direct Viewer.

Four tools:
- ``gpu_viewer_open``        : open the standalone ``GPUViewerDialog``
- ``gpu_viewer_set_field``   : change the field (``depth`` / ``speed``)
- ``gpu_viewer_read_snapshot``: return latest (t_s, h, hu, hv) as plain data
- ``gpu_viewer_screenshot``  : PNG screenshot of the viewer window

Usage from an MCP agent:

    gpu_viewer_open()
    gpu_viewer_set_field("speed")
    snap = gpu_viewer_read_snapshot()        # {t_s, n_cells, h_b64, ...}
    gpu_viewer_screenshot("/tmp/viewer.png")
"""
from __future__ import annotations

from typing import Any, Dict, Optional


def _get_bridge_client(*args, **kwargs):
    """Lazy import of the bridge client so tools are importable without QGIS."""
    from tools.hydra_mcp.bridge_client import get_bridge_client
    return get_bridge_client(*args, **kwargs)


def _err(message: str, **context: Any) -> Dict[str, Any]:
    out = {"ok": False, "error": message}
    out.update(context)
    return out


def gpu_viewer_open(token_path: Optional[str] = None) -> Dict[str, Any]:
    """Open the GPUViewerDialog on top of the running studio window.

    The dialog uses a snapshot reader that pulls from the GPU device ring
    buffer.  If no run is in progress, the reader returns empty dicts and
    the dialog shows "Waiting for first snapshot…".
    """
    client = _get_bridge_client(token_path=token_path)
    if client is None:
        return _err("no active QGIS bridge session", token_path=token_path)
    try:
        return client.call("gpu_viewer_open")
    except Exception as exc:
        return _err(f"gpu_viewer_open failed: {exc}")


def gpu_viewer_set_field(
    field: str, token_path: Optional[str] = None
) -> Dict[str, Any]:
    """Change the field on the open viewer. Valid fields: ``'depth'``, ``'speed'``."""
    if field not in ("depth", "speed"):
        return _err(f"invalid field {field!r}; must be 'depth' or 'speed'")
    client = _get_bridge_client(token_path=token_path)
    if client is None:
        return _err("no active QGIS bridge session", token_path=token_path)
    try:
        return client.call("gpu_viewer_set_field", {"field": field})
    except Exception as exc:
        return _err(f"gpu_viewer_set_field failed: {exc}")


def gpu_viewer_read_snapshot(token_path: Optional[str] = None) -> Dict[str, Any]:
    """Read the latest live snapshot.

    Returns ``{ok: True, t_s, n_cells, h_b64, hu_b64, hv_b64}`` where the
    arrays are base64-encoded ``float64`` bytes (small enough to survive
    JSON RPC for typical meshes).

    Use ``base64.b64decode(snap['h_b64'])`` then ``np.frombuffer(...,
    dtype=np.float64)`` to recover the array client-side.
    """
    client = _get_bridge_client(token_path=token_path)
    if client is None:
        return _err("no active QGIS bridge session", token_path=token_path)
    try:
        return client.call("gpu_viewer_read_snapshot")
    except Exception as exc:
        return _err(f"gpu_viewer_read_snapshot failed: {exc}")


def gpu_viewer_screenshot(
    out_path: str,
    format: str = "png",
    token_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Screenshot the open GPUViewerDialog to *out_path*."""
    import os as _os
    _os.makedirs(_os.path.dirname(_os.path.abspath(out_path)) or ".", exist_ok=True)
    client = _get_bridge_client(token_path=token_path)
    if client is None:
        return _err("no active QGIS bridge session", token_path=token_path)
    try:
        return client.call(
            "gpu_viewer_screenshot",
            {"out_path": out_path, "format": format},
        )
    except Exception as exc:
        return _err(f"gpu_viewer_screenshot failed: {exc}")