"""Widget screenshot helper for the HYDRA MCP bridge (Phase 2.C).

This module is intentionally importable without a real Qt environment — it
only needs ``qgis.PyQt.QtWidgets.QWidget`` (and the QtCore classes imported
below) at call time, not at import time. This allows the
``capture_widget_screenshot`` function to be unit-tested with plain-Python
mock widgets.
"""
from __future__ import annotations

import base64
from typing import TYPE_CHECKING, Any, Dict, Optional

# Import QtCore lazily — we still want the module importable without Qt so
# that the MCP server can run in a headless env for Tier A tools. The actual
# Qt classes are looked up on demand inside capture_widget_screenshot().
try:
    from PyQt5 import QtCore  # type: ignore[import-not-found]
    _QT_AVAILABLE = True
except Exception:  # pragma: no cover - exercised on headless / uv envs
    QtCore = None  # type: ignore[assignment]
    _QT_AVAILABLE = False

if TYPE_CHECKING:
    from qgis.PyQt.QtWidgets import QWidget


def capture_widget_screenshot(
    widget: Optional["QWidget"], format: str = "png"
) -> Dict[str, Any]:
    """Capture a screenshot of *widget* and return base64-encoded image data.

    Args:
        widget: The QWidget to capture. May be None.
        format: Image format — ``"png"`` (default) or ``"jpg"`` / ``"jpeg"``.
            JPEG uses quality=85.

    Returns:
        ``{"ok": true, "image_b64": "...", "format": "...", "width": N, "height": M}``
        on success; ``{"ok": false, "error": "..."}`` on failure (None widget,
        not visible, capture error).
    """
    if widget is None:
        return {"ok": False, "error": "widget not available"}
    try:
        if not widget.isVisible():
            return {"ok": False, "error": "widget not available"}

        fmt_lower = format.lower()
        if fmt_lower == "png":
            qt_format = "PNG"
            canonical = "png"
        elif fmt_lower in ("jpg", "jpeg"):
            qt_format = "JPEG"
            canonical = "jpg"  # normalize jpeg → jpg
        else:
            return {"ok": False, "error": f"unsupported format: {format}"}

        pixmap = widget.grab()

        if not _QT_AVAILABLE:
            return {
                "ok": False,
                "error": (
                    "PyQt5.QtCore is not importable in this interpreter; cannot "
                    "encode a real QPixmap to PNG/JPEG."
                ),
            }

        # Use QBuffer (a QIODevice) to hold the encoded bytes — PyQt5's
        # QPixmap.save() accepts a filename (str) or any QIODevice, NOT a
        # generic Python file-like such as io.BytesIO. The previous
        # implementation called pixmap.save(io.BytesIO(), "PNG"), which
        # silently failed for real QPixmap instances.
        buffer = QtCore.QBuffer()
        buffer.open(QtCore.QIODevice.WriteOnly)
        try:
            if qt_format == "JPEG":
                # Quality is the 3rd positional argument; values are clamped
                # to [0, 100] by Qt.
                saved = pixmap.save(buffer, qt_format, 85)
            else:
                saved = pixmap.save(buffer, qt_format)
        finally:
            buffer.close()

        if not saved:
            return {
                "ok": False,
                "error": f"QPixmap.save() returned False for format {qt_format!r}.",
            }

        image_bytes: bytes = bytes(buffer.data())
        return {
            "ok": True,
            "image_b64": base64.b64encode(image_bytes).decode("ascii"),
            "format": canonical,
            "width": pixmap.width(),
            "height": pixmap.height(),
        }
    except RuntimeError as exc:
        return {"ok": False, "error": str(exc)}
    except Exception as exc:
        return {"ok": False, "error": f"screenshot capture failed: {exc}"}
