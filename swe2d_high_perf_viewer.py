"""swe2d_high_perf_viewer.py

Shared high-performance SWE2D frame rendering + QGIS canvas overlay helpers.

The core renderer uses vectorized NumPy rasterization of unstructured cell
fields, and the canvas overlay item draws georeferenced frames directly over
the map canvas extent.
"""

from __future__ import annotations

import time
from typing import Optional, Sequence, Tuple

import numpy as np

try:
    from qgis.PyQt import QtCore, QtGui
except Exception:
    from PyQt5 import QtCore, QtGui

try:
    from qgis.core import QgsPointXY
except Exception:
    QgsPointXY = None

try:
    from qgis.gui import QgsMapCanvasItem
except Exception:
    QgsMapCanvasItem = None


def _build_color_lut(stops: Sequence[Tuple[float, Tuple[int, int, int]]]) -> np.ndarray:
    x = np.asarray([float(s[0]) for s in stops], dtype=np.float64)
    r = np.asarray([float(s[1][0]) for s in stops], dtype=np.float64)
    g = np.asarray([float(s[1][1]) for s in stops], dtype=np.float64)
    b = np.asarray([float(s[1][2]) for s in stops], dtype=np.float64)
    xi = np.linspace(0.0, 1.0, 256, dtype=np.float64)
    lut = np.zeros((256, 3), dtype=np.uint8)
    lut[:, 0] = np.clip(np.interp(xi, x, r), 0.0, 255.0).astype(np.uint8)
    lut[:, 1] = np.clip(np.interp(xi, x, g), 0.0, 255.0).astype(np.uint8)
    lut[:, 2] = np.clip(np.interp(xi, x, b), 0.0, 255.0).astype(np.uint8)
    return lut


_COLOR_LUTS = {
    "turbo": _build_color_lut(
        [
            (0.0, (48, 18, 59)),
            (0.20, (50, 100, 220)),
            (0.40, (41, 187, 236)),
            (0.60, (124, 234, 87)),
            (0.80, (250, 205, 32)),
            (1.0, (180, 4, 38)),
        ]
    ),
    "viridis": _build_color_lut(
        [
            (0.0, (68, 1, 84)),
            (0.25, (59, 82, 139)),
            (0.50, (33, 145, 140)),
            (0.75, (94, 201, 98)),
            (1.0, (253, 231, 37)),
        ]
    ),
    "plasma": _build_color_lut(
        [
            (0.0, (13, 8, 135)),
            (0.25, (126, 3, 167)),
            (0.50, (203, 71, 119)),
            (0.75, (248, 149, 64)),
            (1.0, (240, 249, 33)),
        ]
    ),
    "gray": _build_color_lut([(0.0, (0, 0, 0)), (1.0, (255, 255, 255))]),
}


def render_unstructured_snapshot_image(
    cell_x: np.ndarray,
    cell_y: np.ndarray,
    cell_bed: Optional[np.ndarray],
    timesteps: Sequence[Tuple[float, np.ndarray, np.ndarray, np.ndarray]],
    current_time_s: float,
    field_key: str = "depth",
    cmap_key: str = "turbo",
    resolution: Tuple[int, int] = (960, 540),
    auto_contrast: bool = True,
) -> dict:
    """Rasterize unstructured cell data into a QImage for high-FPS display.

    Returns a dictionary with keys:
    - ok: bool
    - image: QImage
    - extent: (xmin, xmax, ymin, ymax)
    - frame_idx, frame_count, time_s, n_cells
    - vmin, vmax, render_ms
    - message
    """
    t0 = time.perf_counter()
    out = {
        "ok": False,
        "image": QtGui.QImage(),
        "extent": (0.0, 1.0, 0.0, 1.0),
        "frame_idx": 0,
        "frame_count": int(len(timesteps or [])),
        "time_s": float(current_time_s),
        "n_cells": 0,
        "vmin": 0.0,
        "vmax": 1.0,
        "render_ms": 0.0,
        "message": "",
    }

    x_all = np.asarray(cell_x, dtype=np.float64).ravel()
    y_all = np.asarray(cell_y, dtype=np.float64).ravel()
    if x_all.size <= 0 or y_all.size <= 0:
        out["message"] = "No mesh/cell data available."
        out["render_ms"] = (time.perf_counter() - t0) * 1000.0
        return out

    x_min = float(np.nanmin(x_all))
    x_max = float(np.nanmax(x_all))
    y_min = float(np.nanmin(y_all))
    y_max = float(np.nanmax(y_all))
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
        x_min, x_max = 0.0, 1.0
    if not np.isfinite(y_min) or not np.isfinite(y_max) or y_max <= y_min:
        y_min, y_max = 0.0, 1.0
    out["extent"] = (x_min, x_max, y_min, y_max)

    ts_list = list(timesteps or [])
    if not ts_list:
        out["message"] = "No snapshots available yet."
        out["render_ms"] = (time.perf_counter() - t0) * 1000.0
        return out

    times_s = np.asarray([float(ts[0]) for ts in ts_list], dtype=np.float64)
    idx = int(np.argmin(np.abs(times_s - float(current_time_s))))
    t_s, h_raw, hu_raw, hv_raw = ts_list[idx]
    out["frame_idx"] = idx
    out["time_s"] = float(t_s)

    h_arr = np.asarray(h_raw, dtype=np.float64).ravel()
    hu_arr = np.asarray(hu_raw, dtype=np.float64).ravel()
    hv_arr = np.asarray(hv_raw, dtype=np.float64).ravel()
    bed_arr = np.asarray(cell_bed, dtype=np.float64).ravel() if cell_bed is not None else np.empty(0, dtype=np.float64)

    n = min(
        int(x_all.size),
        int(y_all.size),
        int(h_arr.size),
        int(hu_arr.size),
        int(hv_arr.size),
    )
    out["n_cells"] = n
    if n <= 0:
        out["message"] = "No mesh/snapshot overlap for rendering."
        out["render_ms"] = (time.perf_counter() - t0) * 1000.0
        return out

    x = x_all[:n]
    y = y_all[:n]
    h = h_arr[:n]
    hu = hu_arr[:n]
    hv = hv_arr[:n]

    mode = str(field_key or "depth").lower()
    if mode == "speed":
        safe_h = np.maximum(h, 1.0e-6)
        wet = h > 1.0e-6
        vals = np.zeros_like(h)
        vals[wet] = np.sqrt((hu[wet] / safe_h[wet]) ** 2 + (hv[wet] / safe_h[wet]) ** 2)
    elif mode == "wse":
        if bed_arr.size >= n:
            vals = h + bed_arr[:n]
        else:
            vals = h.copy()
    else:
        vals = h.copy()

    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(vals)
    if not np.any(valid):
        out["message"] = "No finite values at this frame."
        out["render_ms"] = (time.perf_counter() - t0) * 1000.0
        return out

    x = x[valid]
    y = y[valid]
    vals = vals[valid]

    w = max(32, int(resolution[0]))
    h_img = max(32, int(resolution[1]))
    x_span = max(1.0e-12, x_max - x_min)
    y_span = max(1.0e-12, y_max - y_min)

    ix = np.clip(((x - x_min) / x_span * (w - 1)).astype(np.int32), 0, w - 1)
    iy = np.clip(((y_max - y) / y_span * (h_img - 1)).astype(np.int32), 0, h_img - 1)
    pix = (iy * w + ix).astype(np.int64)

    wh = int(w * h_img)
    sum_vals = np.bincount(pix, weights=vals, minlength=wh).astype(np.float64)
    cnt_vals = np.bincount(pix, minlength=wh).astype(np.float64)
    grid = np.full(wh, np.nan, dtype=np.float64)
    has_data = cnt_vals > 0.0
    grid[has_data] = sum_vals[has_data] / cnt_vals[has_data]
    grid = grid.reshape((h_img, w))

    finite = np.isfinite(grid)
    if not np.any(finite):
        out["message"] = "No rasterized values at this frame."
        out["render_ms"] = (time.perf_counter() - t0) * 1000.0
        return out

    v = grid[finite]
    if bool(auto_contrast):
        vmin = float(np.percentile(v, 2.0))
        vmax = float(np.percentile(v, 98.0))
    else:
        vmin = float(np.nanmin(v))
        vmax = float(np.nanmax(v))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0

    norm = np.zeros_like(grid, dtype=np.float64)
    norm[finite] = np.clip((grid[finite] - vmin) / (vmax - vmin), 0.0, 1.0)
    idx_img = np.clip((norm * 255.0).astype(np.int32), 0, 255)

    cmap = _COLOR_LUTS.get(str(cmap_key or "turbo").lower(), _COLOR_LUTS["turbo"])
    rgb = cmap[idx_img]
    alpha = np.zeros((h_img, w), dtype=np.uint8)
    alpha[finite] = 255
    rgba = np.dstack((rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2], alpha))
    rgba = np.ascontiguousarray(rgba)

    qimg = QtGui.QImage(rgba.data, w, h_img, 4 * w, QtGui.QImage.Format_RGBA8888)
    out["image"] = qimg.copy()
    out["ok"] = True
    out["vmin"] = vmin
    out["vmax"] = vmax
    out["render_ms"] = (time.perf_counter() - t0) * 1000.0
    return out


if QgsMapCanvasItem is not None and QgsPointXY is not None:

    class SWE2DHighPerfCanvasOverlayItem(QgsMapCanvasItem):
        """Map-canvas overlay item for high-performance rasterized SWE2D frames."""

        def __init__(self, canvas):
            super().__init__(canvas)
            self._image = QtGui.QImage()
            self._extent = (0.0, 1.0, 0.0, 1.0)
            self._opacity = 0.65
            self.setZValue(9999.0)
            self.setVisible(False)

        def set_frame(
            self,
            image: QtGui.QImage,
            extent: Tuple[float, float, float, float],
            opacity: float = 0.65,
        ):
            self._image = image if image is not None else QtGui.QImage()
            self._extent = tuple(extent) if extent is not None else (0.0, 1.0, 0.0, 1.0)
            self._opacity = max(0.0, min(1.0, float(opacity)))
            self.setVisible(not self._image.isNull())
            self.update()

        def clear(self):
            self._image = QtGui.QImage()
            self.setVisible(False)
            self.update()

        def boundingRect(self):
            c = self.canvas()
            if c is None:
                return QtCore.QRectF()
            return QtCore.QRectF(0.0, 0.0, float(c.width()), float(c.height()))

        def paint(self, painter, option, widget):
            if self._image.isNull():
                return
            x_min, x_max, y_min, y_max = self._extent
            if x_max <= x_min or y_max <= y_min:
                return
            p0 = self.toCanvasCoordinates(QgsPointXY(float(x_min), float(y_max)))
            p1 = self.toCanvasCoordinates(QgsPointXY(float(x_max), float(y_min)))
            target = QtCore.QRectF(
                QtCore.QPointF(float(p0.x()), float(p0.y())),
                QtCore.QPointF(float(p1.x()), float(p1.y())),
            ).normalized()
            if target.width() <= 1.0 or target.height() <= 1.0:
                return

            painter.save()
            painter.setOpacity(self._opacity)
            painter.drawImage(target, self._image)
            painter.restore()

else:

    class SWE2DHighPerfCanvasOverlayItem:  # type: ignore[no-redef]
        """Fallback placeholder when QGIS canvas item APIs are unavailable."""

        def __init__(self, canvas):
            raise RuntimeError("QGIS canvas overlay APIs are unavailable in this environment.")
