"""High-performance unstructured mesh renderer for SWE2D.

Shared high-performance SWE2D frame rendering + QGIS canvas overlay helpers.

The core renderer uses vectorized NumPy rasterization of unstructured cell
fields, and the canvas overlay item draws georeferenced frames directly over
the map canvas extent.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Ensure the native overlay extension (.so) built under ./build/ is findable
# regardless of how this module is imported (package or top-level).
_here = os.path.dirname(os.path.abspath(__file__))
_plugin_root = os.path.abspath(os.path.dirname(_here))
for _candidate in (
    os.path.join(_plugin_root, "build"),
    os.path.join(_plugin_root, "build", "Release"),
    os.path.join(_plugin_root, "build", "Debug"),
    os.path.join(_here, "build"),
    os.path.join(_here, "build", "Release"),
    os.path.join(_here, "build", "Debug"),
):
    if os.path.isdir(_candidate) and _candidate not in sys.path:
        sys.path.insert(0, _candidate)

try:
    import hydra_overlay as _hydra_overlay
except Exception:
    _hydra_overlay = None
    logger.warning(
        "hydra_overlay C++ backend import failed — falling back to pure-Python "
        "NumPy rasterization. Rebuild the native module with `cmake --build build` "
        "to restore GPU-accelerated rendering.",
        exc_info=True,
    )

QgsPointXY = None  # ponytail: deferred to conditional block below
QgsMapCanvasItem = None  # ponytail: deferred to conditional block below


def _build_color_lut(stops: Sequence[Tuple[float, Tuple[int, int, int]]]) -> np.ndarray:
    """build color lut."""
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
    "magma": _build_color_lut(
        [
            (0.0, (0, 0, 4)),
            (0.25, (84, 15, 109)),
            (0.50, (187, 55, 84)),
            (0.75, (249, 142, 8)),
            (1.0, (252, 253, 191)),
        ]
    ),
    "cividis": _build_color_lut(
        [
            (0.0, (0, 34, 78)),
            (0.25, (44, 81, 110)),
            (0.50, (86, 122, 119)),
            (0.75, (132, 168, 115)),
            (1.0, (253, 231, 55)),
        ]
    ),
    "inferno": _build_color_lut(
        [
            (0.0, (0, 0, 4)),
            (0.25, (87, 15, 109)),
            (0.50, (187, 55, 84)),
            (0.75, (249, 142, 8)),
            (1.0, (252, 255, 164)),
        ]
    ),
    "terrain": _build_color_lut(
        [
            (0.0, (50, 90, 150)),
            (0.20, (70, 140, 170)),
            (0.40, (110, 160, 90)),
            (0.60, (170, 140, 90)),
            (0.80, (205, 185, 145)),
            (1.0, (245, 245, 245)),
        ]
    ),
    "ocean": _build_color_lut(
        [
            (0.0, (3, 35, 76)),
            (0.25, (12, 72, 135)),
            (0.50, (35, 126, 180)),
            (0.75, (82, 173, 196)),
            (1.0, (173, 222, 228)),
        ]
    ),
}


def _shift_or_default(arr: np.ndarray, dy: int, dx: int, default: float) -> np.ndarray:
    """shift or default."""
    out = np.full_like(arr, default)
    y0 = max(0, -dy)
    y1 = arr.shape[0] - max(0, dy)
    x0 = max(0, -dx)
    x1 = arr.shape[1] - max(0, dx)
    if y1 > y0 and x1 > x0:
        out[y0 + dy:y1 + dy, x0 + dx:x1 + dx] = arr[y0:y1, x0:x1]
    return out


def _box_dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """box dilate."""
    if radius <= 0:
        return mask.copy()
    outs = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            outs.append(_shift_or_default(mask, dy, dx, False))
    return np.logical_or.reduce(outs)


def _flood_fill_outside(mask: np.ndarray) -> np.ndarray:
    """flood fill outside."""
    h, w = mask.shape
    outside = np.zeros((h, w), dtype=bool)
    stack = []

    for x in range(w):
        if not mask[0, x]:
            outside[0, x] = True
            stack.append((0, x))
        if not mask[h - 1, x]:
            outside[h - 1, x] = True
            stack.append((h - 1, x))
    for y in range(h):
        if not mask[y, 0]:
            outside[y, 0] = True
            stack.append((y, 0))
        if not mask[y, w - 1]:
            outside[y, w - 1] = True
            stack.append((y, w - 1))

    while stack:
        y, x = stack.pop()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if ny < 0 or ny >= h or nx < 0 or nx >= w:
                continue
            if mask[ny, nx] or outside[ny, nx]:
                continue
            outside[ny, nx] = True
            stack.append((ny, nx))

    return outside


def _nearest_fill(mask_known: np.ndarray, values: np.ndarray) -> np.ndarray:
    """nearest fill."""
    h, w = values.shape
    out = values.copy()
    src_y = np.full((h, w), -1, dtype=np.int32)
    src_x = np.full((h, w), -1, dtype=np.int32)
    dist = np.full((h, w), 1.0e30, dtype=np.float64)

    ys, xs = np.where(mask_known)
    src_y[ys, xs] = ys.astype(np.int32)
    src_x[ys, xs] = xs.astype(np.int32)
    dist[ys, xs] = 0.0

    # Two sweeps approximate nearest-neighbor propagation in O(H*W).
    for y in range(h):
        for x in range(w):
            if y > 0 and src_y[y - 1, x] >= 0:
                cand = dist[y - 1, x] + 1.0
                if cand < dist[y, x]:
                    dist[y, x] = cand
                    src_y[y, x] = src_y[y - 1, x]
                    src_x[y, x] = src_x[y - 1, x]
            if x > 0 and src_y[y, x - 1] >= 0:
                cand = dist[y, x - 1] + 1.0
                if cand < dist[y, x]:
                    dist[y, x] = cand
                    src_y[y, x] = src_y[y, x - 1]
                    src_x[y, x] = src_x[y, x - 1]

    for y in range(h - 1, -1, -1):
        for x in range(w - 1, -1, -1):
            if y < h - 1 and src_y[y + 1, x] >= 0:
                cand = dist[y + 1, x] + 1.0
                if cand < dist[y, x]:
                    dist[y, x] = cand
                    src_y[y, x] = src_y[y + 1, x]
                    src_x[y, x] = src_x[y + 1, x]
            if x < w - 1 and src_y[y, x + 1] >= 0:
                cand = dist[y, x + 1] + 1.0
                if cand < dist[y, x]:
                    dist[y, x] = cand
                    src_y[y, x] = src_y[y, x + 1]
                    src_x[y, x] = src_x[y, x + 1]

    missing = ~mask_known
    my, mx = np.where(missing)
    vy = src_y[my, mx]
    vx = src_x[my, mx]
    ok = (vy >= 0) & (vx >= 0)
    out[my[ok], mx[ok]] = out[vy[ok], vx[ok]]
    return out


def _smooth_wse_grid_nodal_eta(
    tri_node_x: np.ndarray,
    tri_node_y: np.ndarray,
    tri_nodes: np.ndarray,
    tri_eta: np.ndarray,
    tri_wet: np.ndarray,
    width: int,
    height: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> Optional[np.ndarray]:
    """Build a smooth WSE raster by reconstructing nodal eta from triangle values.

    Each triangle contributes its eta to all three vertices (area-weighted), then
    eta is linearly interpolated over the triangulation.
    """
    try:
        from matplotlib.tri import Triangulation, LinearTriInterpolator
    except Exception:
        return None

    try:
        n_nodes = int(tri_node_x.size)
        tri_idx = np.asarray(tri_nodes, dtype=np.int32).reshape((-1, 3))
        tri_eta = np.asarray(tri_eta, dtype=np.float64).ravel()
        tri_wet = np.asarray(tri_wet, dtype=bool).ravel()
        n_tri = int(tri_idx.shape[0])
        if n_nodes <= 0 or n_tri <= 0 or tri_eta.size < n_tri or tri_wet.size < n_tri:
            return None
        if int(np.max(tri_idx)) >= n_nodes or int(np.min(tri_idx)) < 0:
            return None

        active = np.isfinite(tri_eta[:n_tri]) & tri_wet[:n_tri]
        if not np.any(active):
            return None

        tri_idx_a = tri_idx[active]
        tri_eta_a = tri_eta[:n_tri][active]

        x0 = tri_node_x[tri_idx_a[:, 0]]
        y0 = tri_node_y[tri_idx_a[:, 0]]
        x1 = tri_node_x[tri_idx_a[:, 1]]
        y1 = tri_node_y[tri_idx_a[:, 1]]
        x2 = tri_node_x[tri_idx_a[:, 2]]
        y2 = tri_node_y[tri_idx_a[:, 2]]
        tri_area = 0.5 * np.abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))
        tri_area = np.maximum(tri_area, 1.0e-12)

        node_sum = np.zeros(n_nodes, dtype=np.float64)
        node_w = np.zeros(n_nodes, dtype=np.float64)
        for col in range(3):
            v = tri_idx_a[:, col]
            node_sum += np.bincount(v, weights=tri_eta_a * tri_area, minlength=n_nodes).astype(np.float64)
            node_w += np.bincount(v, weights=tri_area, minlength=n_nodes).astype(np.float64)

        node_eta = np.full(n_nodes, np.nan, dtype=np.float64)
        known_nodes = node_w > 0.0
        node_eta[known_nodes] = node_sum[known_nodes] / np.maximum(node_w[known_nodes], 1.0e-12)

        if np.count_nonzero(np.isfinite(node_eta)) < 3:
            return None

        tri_full = Triangulation(tri_node_x, tri_node_y, tri_idx)
        # Mask triangles outside active/wet region to avoid extrapolating across dry zones.
        tri_mask = ~active
        if tri_mask.size == tri_full.triangles.shape[0]:
            tri_full.set_mask(tri_mask)

        interp = LinearTriInterpolator(tri_full, node_eta)
        gx = np.linspace(float(x_min), float(x_max), int(width), dtype=np.float64)
        gy = np.linspace(float(y_max), float(y_min), int(height), dtype=np.float64)
        xx, yy = np.meshgrid(gx, gy)
        zz = interp(xx, yy)
        grid = np.asarray(np.ma.filled(zz, np.nan), dtype=np.float64)
        if grid.shape != (int(height), int(width)):
            return None
        return grid
    except Exception:
        return None


def _interp_grid(field: np.ndarray, x: float, y: float) -> float:
    """interp grid."""
    h, w = field.shape
    if x < 0.0 or y < 0.0 or x >= (w - 1) or y >= (h - 1):
        return float("nan")
    x0 = int(x)
    y0 = int(y)
    tx = float(x - x0)
    ty = float(y - y0)
    f00 = field[y0, x0]
    f10 = field[y0, x0 + 1]
    f01 = field[y0 + 1, x0]
    f11 = field[y0 + 1, x0 + 1]
    if not (np.isfinite(f00) and np.isfinite(f10) and np.isfinite(f01) and np.isfinite(f11)):
        return float("nan")
    return float(
        (1.0 - tx) * (1.0 - ty) * f00
        + tx * (1.0 - ty) * f10
        + (1.0 - tx) * ty * f01
        + tx * ty * f11
    )


def _draw_velocity_overlays(
    image,
    shell_mask: np.ndarray,
    seed_mask: np.ndarray,
    u_grid: np.ndarray,
    v_grid: np.ndarray,
    speed_grid: np.ndarray,
    show_arrows: bool,
    arrow_stride_px: int,
    arrow_scale_px: float,
    arrow_length_scale: float,
    arrow_head_length_scale: float,
    arrow_head_width_scale: float,
    arrow_min_speed: float,
    show_streamlines: bool,
    streamline_backend: str,
    streamline_seed_count: int,
    streamline_steps: int,
    streamline_step_px: float,
) -> None:
    """draw velocity overlays."""
    try:
        from qgis.PyQt import QtCore, QtGui
    except Exception:
        from PyQt5 import QtCore, QtGui
    painter = QtGui.QPainter(image)
    painter.setRenderHint(QtGui.QPainter.Antialiasing, True)

    h, w = shell_mask.shape

    if show_streamlines:
        pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 165))
        pen.setWidthF(1.1)
        painter.setPen(pen)
        used_compiled = False
        if _hydra_overlay is not None:
            try:
                traces = _hydra_overlay.advect_streamlines(
                    np.asarray(u_grid, dtype=np.float64),
                    np.asarray(v_grid, dtype=np.float64),
                    np.asarray(speed_grid, dtype=np.float64),
                    np.asarray(shell_mask, dtype=np.uint8),
                    np.asarray(seed_mask, dtype=np.uint8),
                    int(streamline_seed_count),
                    int(streamline_steps),
                    float(streamline_step_px),
                    float(arrow_min_speed),
                    str(streamline_backend or "auto"),
                )
                counts = np.asarray(traces.get("counts", []), dtype=np.int32).ravel()
                xy = np.asarray(traces.get("xy", []), dtype=np.float64)
                if counts.size > 0 and xy.ndim == 3 and xy.shape[0] == counts.size and xy.shape[2] == 2:
                    for it, c in enumerate(counts.tolist()):
                        npt = max(0, int(c))
                        if npt < 2:
                            continue
                        poly = [QtCore.QPointF(float(xy[it, j, 0]), float(xy[it, j, 1])) for j in range(npt)]
                        painter.drawPolyline(QtGui.QPolygonF(poly))
                    used_compiled = True
            except Exception:
                used_compiled = False

        if not used_compiled:
            ys, xs = np.where(seed_mask & np.isfinite(speed_grid) & (speed_grid >= float(arrow_min_speed)))
            if ys.size > 0:
                step_pick = max(1, int(ys.size / max(1, int(streamline_seed_count))))
                seed_idx = np.arange(0, ys.size, step_pick, dtype=np.int64)[: max(1, int(streamline_seed_count))]
                for i in seed_idx:
                    y = float(ys[i])
                    x = float(xs[i])
                    pts = [QtCore.QPointF(x, y)]
                    for _ in range(max(2, int(streamline_steps))):
                        ui = _interp_grid(u_grid, x, y)
                        vi = _interp_grid(v_grid, x, y)
                        si = _interp_grid(speed_grid, x, y)
                        if not (np.isfinite(ui) and np.isfinite(vi) and np.isfinite(si)):
                            break
                        if si < float(arrow_min_speed):
                            break
                        x += float(streamline_step_px) * (ui / max(1.0e-8, si))
                        y -= float(streamline_step_px) * (vi / max(1.0e-8, si))
                        if x < 1.0 or x > (w - 2) or y < 1.0 or y > (h - 2):
                            break
                        iy = int(round(y))
                        ix = int(round(x))
                        if iy < 0 or iy >= h or ix < 0 or ix >= w or not shell_mask[iy, ix]:
                            break
                        pts.append(QtCore.QPointF(x, y))
                    if len(pts) >= 2:
                        painter.drawPolyline(QtGui.QPolygonF(pts))

    if show_arrows:
        pen = QtGui.QPen(QtGui.QColor(0, 0, 0, 190))
        pen.setWidthF(1.0)
        painter.setPen(pen)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255, 190))
        painter.setBrush(brush)

        stride = max(8, int(arrow_stride_px))
        base = max(6.0, float(arrow_scale_px))
        length_scale = max(0.2, float(arrow_length_scale))
        head_len_scale = max(0.2, float(arrow_head_length_scale))
        head_w_scale = max(0.2, float(arrow_head_width_scale))
        for iy in range(stride // 2, h, stride):
            for ix in range(stride // 2, w, stride):
                if not shell_mask[iy, ix]:
                    continue
                s = float(speed_grid[iy, ix])
                if not np.isfinite(s) or s < float(arrow_min_speed):
                    continue
                u = float(u_grid[iy, ix])
                v = float(v_grid[iy, ix])
                if not (np.isfinite(u) and np.isfinite(v)):
                    continue
                dn = max(1.0e-8, np.hypot(u, v))
                ux = u / dn
                uy = -v / dn
                ln = min(2.5 * base, max(0.8 * base, base * (0.4 + 0.9 * s)))
                ln = ln * length_scale
                x0 = float(ix)
                y0 = float(iy)
                x1 = x0 + ux * ln
                y1 = y0 + uy * ln
                painter.drawLine(QtCore.QPointF(x0, y0), QtCore.QPointF(x1, y1))

                ah = max(3.0, 0.25 * ln * head_len_scale)
                ax = -uy
                ay = ux
                hw = ah * 0.6 * head_w_scale
                p1 = QtCore.QPointF(x1, y1)
                p2 = QtCore.QPointF(x1 - ux * ah + ax * hw, y1 - uy * ah + ay * hw)
                p3 = QtCore.QPointF(x1 - ux * ah - ax * hw, y1 - uy * ah - ay * hw)
                painter.drawPolygon(QtGui.QPolygonF([p1, p2, p3]))

    painter.end()


def _draw_scalar_legend(
    image,
    cmap: np.ndarray,
    vmin: float,
    vmax: float,
    label: str,
) -> None:
    """draw scalar legend."""
    try:
        from qgis.PyQt import QtCore, QtGui
    except Exception:
        from PyQt5 import QtCore, QtGui
    if image is None or image.isNull() or cmap is None or cmap.shape[0] < 2:
        return
    if not (np.isfinite(vmin) and np.isfinite(vmax)):
        return

    painter = QtGui.QPainter(image)
    painter.setRenderHint(QtGui.QPainter.Antialiasing, True)

    w = image.width()
    h = image.height()
    bar_h = max(96, int(round(0.42 * h)))
    bar_w = 16
    pad = 12
    box_w = 140
    box_h = bar_h + 32
    x0 = max(0, w - box_w - pad)
    y0 = max(0, pad)

    painter.setPen(QtCore.Qt.PenStyle.NoPen)
    painter.setBrush(QtGui.QColor(0, 0, 0, 120))
    painter.drawRoundedRect(QtCore.QRectF(float(x0), float(y0), float(box_w), float(box_h)), 6.0, 6.0)

    bar_x = x0 + 12
    bar_y = y0 + 18
    for i in range(bar_h):
        t = 1.0 - (float(i) / max(1.0, float(bar_h - 1)))
        k = int(max(0, min(255, round(t * 255.0))))
        c = cmap[k]
        painter.setPen(QtGui.QColor(int(c[0]), int(c[1]), int(c[2]), 255))
        painter.drawLine(bar_x, bar_y + i, bar_x + bar_w, bar_y + i)

    painter.setPen(QtGui.QColor(255, 255, 255, 230))
    painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
    painter.drawRect(bar_x, bar_y, bar_w, bar_h)

    font = painter.font()
    font.setPointSize(max(8, font.pointSize()))
    painter.setFont(font)
    txt_x = bar_x + bar_w + 8
    painter.drawText(QtCore.QPointF(float(txt_x), float(bar_y + 9)), f"{vmax:.3g}")
    painter.drawText(QtCore.QPointF(float(txt_x), float(bar_y + bar_h - 2)), f"{vmin:.3g}")
    painter.drawText(QtCore.QPointF(float(bar_x), float(y0 + box_h - 6)), str(label or "Field"))
    painter.end()


def draw_scalar_legend_on_painter(
    painter,
    canvas_w: int,
    canvas_h: int,
    cmap: np.ndarray,
    vmin: float,
    vmax: float,
    label: str,
) -> None:
    """draw scalar legend on painter."""
    try:
        from qgis.PyQt import QtCore, QtGui
    except Exception:
        from PyQt5 import QtCore, QtGui
    if painter is None or cmap is None or cmap.shape[0] < 2:
        return
    if not (np.isfinite(vmin) and np.isfinite(vmax)):
        return

    w = max(32, int(canvas_w))
    h = max(32, int(canvas_h))
    bar_h = max(96, int(round(0.42 * h)))
    bar_w = 16
    pad = 12
    box_w = 140
    box_h = bar_h + 32
    x0 = max(0, w - box_w - pad)
    y0 = max(0, pad)

    painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
    painter.setPen(QtCore.Qt.PenStyle.NoPen)
    painter.setBrush(QtGui.QColor(0, 0, 0, 120))
    painter.drawRoundedRect(QtCore.QRectF(float(x0), float(y0), float(box_w), float(box_h)), 6.0, 6.0)

    bar_x = x0 + 12
    bar_y = y0 + 18
    for i in range(bar_h):
        t = 1.0 - (float(i) / max(1.0, float(bar_h - 1)))
        k = int(max(0, min(255, round(t * 255.0))))
        c = cmap[k]
        painter.setPen(QtGui.QColor(int(c[0]), int(c[1]), int(c[2]), 255))
        painter.drawLine(bar_x, bar_y + i, bar_x + bar_w, bar_y + i)

    painter.setPen(QtGui.QColor(255, 255, 255, 230))
    painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
    painter.drawRect(bar_x, bar_y, bar_w, bar_h)

    font = painter.font()
    font.setPointSize(max(8, font.pointSize()))
    painter.setFont(font)
    txt_x = bar_x + bar_w + 8
    painter.drawText(QtCore.QPointF(float(txt_x), float(bar_y + 9)), f"{vmax:.3g}")
    painter.drawText(QtCore.QPointF(float(txt_x), float(bar_y + bar_h - 2)), f"{vmin:.3g}")
    painter.drawText(QtCore.QPointF(float(bar_x), float(y0 + box_h - 6)), str(label or "Field"))


def render_unstructured_snapshot_image(
    cell_x: np.ndarray,
    cell_y: np.ndarray,
    cell_bed: Optional[np.ndarray],
    timesteps: Sequence[Tuple[float, np.ndarray, np.ndarray, np.ndarray]],
    current_time_s: float,
    field_key: str = "depth",
    wse_render_mode: str = "cell",
    gravity: float = 9.81,
    courant_cell_size: float = 0.0,
    courant_dt: float = 0.0,
    manning_n: float = 0.035,
    cmap_key: str = "turbo",
    resolution: Tuple[int, int] = (960, 540),
    auto_contrast: bool = True,
    show_velocity_arrows: bool = False,
    arrow_stride_px: int = 28,
    arrow_scale_px: float = 14.0,
    arrow_length_scale: float = 1.0,
    arrow_head_length_scale: float = 1.0,
    arrow_head_width_scale: float = 1.0,
    arrow_min_speed: float = 0.02,
    show_streamlines: bool = False,
    streamline_backend: str = "auto",
    streamline_seed_count: int = 40,
    streamline_steps: int = 22,
    streamline_step_px: float = 4.0,
    visible_extent_world: Optional[Tuple[float, float, float, float]] = None,
    render_extent_world: Optional[Tuple[float, float, float, float]] = None,
    node_x: Optional[np.ndarray] = None,
    node_y: Optional[np.ndarray] = None,
    cell_nodes: Optional[np.ndarray] = None,
    tri_to_cell: Optional[np.ndarray] = None,
    show_legend: bool = True,
    legend_label: str = "",
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
    try:
        from qgis.PyQt import QtCore, QtGui
    except Exception:
        from PyQt5 import QtCore, QtGui
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
        "backend": "numpy",
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

    if render_extent_world is not None:
        try:
            rx_min, rx_max, ry_min, ry_max = [float(v) for v in render_extent_world]
            if rx_max < rx_min:
                rx_min, rx_max = rx_max, rx_min
            if ry_max < ry_min:
                ry_min, ry_max = ry_max, ry_min
            if np.isfinite(rx_min) and np.isfinite(rx_max) and (rx_max - rx_min) > 1.0e-12:
                x_min = rx_min
                x_max = rx_max
            if np.isfinite(ry_min) and np.isfinite(ry_max) and (ry_max - ry_min) > 1.0e-12:
                y_min = ry_min
                y_max = ry_max
        except Exception:
            pass
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
    wet_all = h > 1.0e-6

    mode = str(field_key or "depth").lower()
    if mode == "speed":
        safe_h = np.maximum(h, 1.0e-6)
        vals = np.zeros_like(h)
        vals[wet_all] = np.sqrt((hu[wet_all] / safe_h[wet_all]) ** 2 + (hv[wet_all] / safe_h[wet_all]) ** 2)
    elif mode == "wse":
        if bed_arr.size >= n:
            vals = h + bed_arr[:n]
        else:
            vals = h.copy()
    elif mode == "froude":
        g = float(gravity)
        safe_h = np.maximum(h, 1.0e-6)
        v = np.zeros_like(h)
        v[wet_all] = np.sqrt((hu[wet_all] / safe_h[wet_all]) ** 2 + (hv[wet_all] / safe_h[wet_all]) ** 2)
        vals = np.where(wet_all, v / np.sqrt(np.maximum(g * safe_h, 1.0e-12)), 0.0)
    elif mode == "courant":
        safe_h = np.maximum(h, 1.0e-6)
        v = np.zeros_like(h)
        v[wet_all] = np.sqrt((hu[wet_all] / safe_h[wet_all]) ** 2 + (hv[wet_all] / safe_h[wet_all]) ** 2)
        cell_size = max(float(courant_cell_size), 1.0e-6) if courant_cell_size > 0 else 1.0
        dt = max(float(courant_dt), 1.0e-6) if courant_dt > 0 else 1.0
        vals = np.where(wet_all, v * dt / cell_size, 0.0)
    elif mode == "shear_stress":
        g = float(gravity)
        n_val = max(float(manning_n), 1.0e-6)
        safe_h = np.maximum(h, 1.0e-6)
        v = np.zeros_like(h)
        v[wet_all] = np.sqrt((hu[wet_all] / safe_h[wet_all]) ** 2 + (hv[wet_all] / safe_h[wet_all]) ** 2)
        rho = 1000.0
        vals = np.where(wet_all, rho * g * n_val * n_val * v * v / np.maximum(safe_h ** (1.0 / 3.0), 1.0e-12), 0.0)
    else:
        vals = h.copy()

    vals_cell = np.asarray(vals, dtype=np.float64).ravel()

    if render_extent_world is not None:
        in_view = (x_all[:n] >= x_min) & (x_all[:n] <= x_max) & (y_all[:n] >= y_min) & (y_all[:n] <= y_max)
        if np.any(in_view):
            wet_all = wet_all & in_view

    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(vals) & wet_all
    if not np.any(valid):
        out["message"] = "No wetted values at this frame."
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
    sum_vals = None
    cnt_vals = None
    tri_raster_used = False
    tri_valid = None

    tri_node_x = np.asarray(node_x if node_x is not None else np.empty(0), dtype=np.float64).ravel()
    tri_node_y = np.asarray(node_y if node_y is not None else np.empty(0), dtype=np.float64).ravel()
    tri_nodes = np.asarray(cell_nodes if cell_nodes is not None else np.empty(0), dtype=np.int32).ravel()

    tri_n = 0
    tri_mismatch_reason = ""
    if tri_nodes.size > 0:
        if (tri_nodes.size % 3) != 0:
            tri_mismatch_reason = "cell_nodes length is not divisible by 3"
        else:
            tri_n = int(tri_nodes.size // 3)
            if tri_n != int(vals_cell.size):
                tri_mismatch_reason = (
                    f"triangle/value count mismatch (n_tri={tri_n}, n_values={int(vals_cell.size)})"
                )

    # If a tri_to_cell map is given (fan triangulation of quad/polygon mesh), expand
    # per-cell arrays to per-triangle so the tri-fill rasterizer gets a 1:1 mapping.
    _tc_map = None
    if tri_to_cell is not None and tri_n > 0 and tri_mismatch_reason:
        _tc_cand = np.asarray(tri_to_cell, dtype=np.int32).ravel()
        if (
            _tc_cand.size == tri_n
            and int(vals_cell.size) > 0
            and int(_tc_cand.max()) < int(vals_cell.size)
        ):
            _tc_map = _tc_cand
            tri_mismatch_reason = ""  # resolved by expansion

    # Per-triangle versions of cell-valued arrays (same as originals if no map).
    _vals_cell_tri = vals_cell[_tc_map] if _tc_map is not None else vals_cell
    _wet_all_tri = wet_all[_tc_map] if _tc_map is not None else wet_all
    _h_tri = h[_tc_map] if _tc_map is not None else h
    _hu_tri = hu[_tc_map] if _tc_map is not None else hu
    _hv_tri = hv[_tc_map] if _tc_map is not None else hv

    if _hydra_overlay is not None and tri_node_x.size > 0 and tri_node_y.size > 0 and tri_nodes.size >= 3:
        if tri_mismatch_reason:
            out["message"] = (
                "Tri-fill disabled: "
                + str(tri_mismatch_reason)
                + "; using centroid accumulation fallback."
            )
            tri_raster_used = False
        elif tri_n > 0 and vals_cell.size > 0:
            tri_valid = int(tri_n)
            try:
                tri_wet = np.asarray(_wet_all_tri[:tri_valid], dtype=bool)
                tri_idx = np.nonzero(tri_wet)[0].astype(np.int32)
                if tri_idx.size <= 0:
                    raise RuntimeError("No wetted triangles for scalar rasterization")
                tri_nodes_wet = tri_nodes.reshape((-1, 3))[tri_idx].reshape(-1)
                vals_wet = _vals_cell_tri[:tri_valid][tri_idx]
                acc_tri = _hydra_overlay.rasterize_tri_mesh_accum(
                    tri_node_x,
                    tri_node_y,
                    tri_nodes_wet,
                    vals_wet,
                    None,
                    None,
                    int(w),
                    int(h_img),
                    float(x_min),
                    float(x_max),
                    float(y_min),
                    float(y_max),
                )
                sum_vals = np.asarray(acc_tri["sum_scalar"], dtype=np.float64).reshape(-1)
                cnt_vals = np.asarray(acc_tri["count"], dtype=np.float64).reshape(-1)
                out["backend"] = "hydra_overlay_tri_fill"
                tri_raster_used = True
            except Exception:
                sum_vals = None
                cnt_vals = None
                tri_raster_used = False

    if _hydra_overlay is not None and not tri_raster_used:
        try:
            # Compiled accumulation path (CPU now; API is CUDA/OpenGL-ready).
            acc = _hydra_overlay.rasterize_unstructured_accum(
                x.astype(np.float64, copy=False),
                y.astype(np.float64, copy=False),
                vals.astype(np.float64, copy=False),
                None,
                None,
                int(w),
                int(h_img),
                float(x_min),
                float(x_max),
                float(y_min),
                float(y_max),
            )
            sum_vals = np.asarray(acc["sum_scalar"], dtype=np.float64).reshape(-1)
            cnt_vals = np.asarray(acc["count"], dtype=np.float64).reshape(-1)
            out["backend"] = "hydra_overlay"
        except Exception:
            sum_vals = None
            cnt_vals = None

    if sum_vals is None or cnt_vals is None:
        sum_vals = np.bincount(pix, weights=vals, minlength=wh).astype(np.float64)
        cnt_vals = np.bincount(pix, minlength=wh).astype(np.float64)
    grid = np.full(wh, np.nan, dtype=np.float64)
    has_data = cnt_vals > 0.0
    grid[has_data] = sum_vals[has_data] / cnt_vals[has_data]
    grid = grid.reshape((h_img, w))

    smooth_grid = None
    if mode == "wse" and str(wse_render_mode or "cell").lower() in ("nodal", "smooth", "smoothed"):
        smooth_grid = _smooth_wse_grid_nodal_eta(
            tri_node_x=tri_node_x,
            tri_node_y=tri_node_y,
            tri_nodes=tri_nodes,
            tri_eta=_vals_cell_tri,
            tri_wet=_wet_all_tri,
            width=int(w),
            height=int(h_img),
            x_min=float(x_min),
            x_max=float(x_max),
            y_min=float(y_min),
            y_max=float(y_max),
        )

    # Build shell mask and fill to avoid sparse center-point appearance.
    mask_known = has_data.reshape((h_img, w))
    if np.any(mask_known):
        finalized = None
        if _hydra_overlay is not None:
            try:
                finalized = _hydra_overlay.finalize_scalar_field(
                    sum_vals.reshape((h_img, w)),
                    cnt_vals.reshape((h_img, w)),
                    0,
                )
            except Exception:
                finalized = None

        if finalized is not None:
            grid = np.asarray(finalized["grid"], dtype=np.float64)
            mask_known = np.asarray(finalized["known_mask"], dtype=np.uint8).astype(bool)
            shell_mask = np.asarray(finalized["shell_mask"], dtype=np.uint8).astype(bool)
        else:
            spacing = np.sqrt((w * h_img) / max(1.0, float(np.count_nonzero(mask_known))))
            dil_radius = int(max(1, min(4, round(0.55 * spacing))))
            shell_seed = _box_dilate(mask_known, dil_radius)
            outside = _flood_fill_outside(shell_seed)
            shell_mask = ~outside
            grid = _nearest_fill(mask_known, grid)
    else:
        shell_mask = np.zeros((h_img, w), dtype=bool)

    if smooth_grid is not None:
        grid = smooth_grid
        mask_known = np.isfinite(grid)
        shell_mask = mask_known.copy()
        out["backend"] = str(out.get("backend", "numpy")) + "+wse_nodal"

    # Expose the raw scalar field and mask so callers that need data values
    # (e.g. GeoTIFF export) can avoid re‑rendering.
    out["grid"] = grid.copy()
    out["grid_mask"] = np.isfinite(grid) & shell_mask

    finite = np.isfinite(grid) & shell_mask
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
    img_out = qimg.copy()

    if bool(show_velocity_arrows or show_streamlines):
        h_valid = h[valid]
        hu_valid = hu[valid]
        hv_valid = hv[valid]
        safe_h = np.maximum(h_valid, 1.0e-6)
        u_cell = np.zeros_like(h_valid)
        v_cell = np.zeros_like(h_valid)
        wet = h_valid > 1.0e-6
        u_cell[wet] = hu_valid[wet] / safe_h[wet]
        v_cell[wet] = hv_valid[wet] / safe_h[wet]

        sum_u = None
        sum_v = None
        if _hydra_overlay is not None and tri_raster_used and tri_valid is not None and tri_valid > 0:
            try:
                safe_h_all = np.maximum(_h_tri[:tri_valid], 1.0e-6)
                wet_all_tri_v = _h_tri[:tri_valid] > 1.0e-6
                u_cell_all = np.zeros(tri_valid, dtype=np.float64)
                v_cell_all = np.zeros(tri_valid, dtype=np.float64)
                u_cell_all[wet_all_tri_v] = _hu_tri[:tri_valid][wet_all_tri_v] / safe_h_all[wet_all_tri_v]
                v_cell_all[wet_all_tri_v] = _hv_tri[:tri_valid][wet_all_tri_v] / safe_h_all[wet_all_tri_v]
                tri_idx = np.nonzero(wet_all_tri_v)[0].astype(np.int32)
                if tri_idx.size <= 0:
                    raise RuntimeError("No wetted triangles for velocity rasterization")
                tri_nodes_wet = tri_nodes.reshape((-1, 3))[tri_idx].reshape(-1)
                acc_uv_tri = _hydra_overlay.rasterize_tri_mesh_accum(
                    tri_node_x,
                    tri_node_y,
                    tri_nodes_wet,
                    _vals_cell_tri[:tri_valid][tri_idx],
                    u_cell_all[tri_idx],
                    v_cell_all[tri_idx],
                    int(w),
                    int(h_img),
                    float(x_min),
                    float(x_max),
                    float(y_min),
                    float(y_max),
                )
                sum_u = np.asarray(acc_uv_tri["sum_u"], dtype=np.float64).reshape(-1)
                sum_v = np.asarray(acc_uv_tri["sum_v"], dtype=np.float64).reshape(-1)
            except Exception:
                sum_u = None
                sum_v = None

        if _hydra_overlay is not None and sum_u is None:
            try:
                acc_uv = _hydra_overlay.rasterize_unstructured_accum(
                    x.astype(np.float64, copy=False),
                    y.astype(np.float64, copy=False),
                    vals.astype(np.float64, copy=False),
                    u_cell.astype(np.float64, copy=False),
                    v_cell.astype(np.float64, copy=False),
                    int(w),
                    int(h_img),
                    float(x_min),
                    float(x_max),
                    float(y_min),
                    float(y_max),
                )
                sum_u = np.asarray(acc_uv["sum_u"], dtype=np.float64).reshape(-1)
                sum_v = np.asarray(acc_uv["sum_v"], dtype=np.float64).reshape(-1)
            except Exception:
                sum_u = None
                sum_v = None

        if sum_u is None or sum_v is None:
            sum_u = np.bincount(pix, weights=u_cell, minlength=wh).astype(np.float64)
            sum_v = np.bincount(pix, weights=v_cell, minlength=wh).astype(np.float64)
        u_grid = np.full((h_img, w), np.nan, dtype=np.float64)
        v_grid = np.full((h_img, w), np.nan, dtype=np.float64)
        has_data_2d = has_data.reshape((h_img, w))
        u_grid[has_data_2d] = (sum_u[has_data] / cnt_vals[has_data])
        v_grid[has_data_2d] = (sum_v[has_data] / cnt_vals[has_data])
        if _hydra_overlay is not None:
            try:
                km = mask_known.astype(np.uint8, copy=False)
                u_grid = np.asarray(_hydra_overlay.nearest_fill(u_grid, km), dtype=np.float64)
                v_grid = np.asarray(_hydra_overlay.nearest_fill(v_grid, km), dtype=np.float64)
            except Exception:
                u_grid = _nearest_fill(mask_known, u_grid)
                v_grid = _nearest_fill(mask_known, v_grid)
        else:
            u_grid = _nearest_fill(mask_known, u_grid)
            v_grid = _nearest_fill(mask_known, v_grid)
        speed_grid = np.sqrt(np.maximum(0.0, u_grid * u_grid + v_grid * v_grid))
        speed_grid[~shell_mask] = np.nan

        seed_mask = shell_mask
        if visible_extent_world is not None:
            try:
                vx_min, vx_max, vy_min, vy_max = [float(v) for v in visible_extent_world]
                if vx_max < vx_min:
                    vx_min, vx_max = vx_max, vx_min
                if vy_max < vy_min:
                    vy_min, vy_max = vy_max, vy_min

                ix0 = int(np.floor((vx_min - x_min) / x_span * (w - 1)))
                ix1 = int(np.ceil((vx_max - x_min) / x_span * (w - 1)))
                iy0 = int(np.floor((y_max - vy_max) / y_span * (h_img - 1)))
                iy1 = int(np.ceil((y_max - vy_min) / y_span * (h_img - 1)))

                ix0 = max(0, min(w - 1, ix0))
                ix1 = max(0, min(w - 1, ix1))
                iy0 = max(0, min(h_img - 1, iy0))
                iy1 = max(0, min(h_img - 1, iy1))

                if ix1 < ix0:
                    ix0, ix1 = ix1, ix0
                if iy1 < iy0:
                    iy0, iy1 = iy1, iy0

                vis_mask = np.zeros_like(shell_mask, dtype=bool)
                vis_mask[iy0:iy1 + 1, ix0:ix1 + 1] = True
                seed_mask = shell_mask & vis_mask
            except Exception:
                seed_mask = shell_mask

        _draw_velocity_overlays(
            image=img_out,
            shell_mask=shell_mask,
            seed_mask=seed_mask,
            u_grid=u_grid,
            v_grid=v_grid,
            speed_grid=speed_grid,
            show_arrows=bool(show_velocity_arrows),
            arrow_stride_px=int(arrow_stride_px),
            arrow_scale_px=float(arrow_scale_px),
            arrow_length_scale=float(arrow_length_scale),
            arrow_head_length_scale=float(arrow_head_length_scale),
            arrow_head_width_scale=float(arrow_head_width_scale),
            arrow_min_speed=float(arrow_min_speed),
            show_streamlines=bool(show_streamlines),
            streamline_backend=str(streamline_backend or "auto"),
            streamline_seed_count=int(streamline_seed_count),
            streamline_steps=int(streamline_steps),
            streamline_step_px=float(streamline_step_px),
        )

    if bool(show_legend):
        label = str(legend_label or "").strip()
        if not label:
            if mode == "speed":
                label = "Velocity"
            elif mode == "wse":
                label = "Water Surface"
            else:
                label = "Depth"
        _draw_scalar_legend(img_out, cmap, vmin, vmax, label)

    out["image"] = img_out
    out["ok"] = True
    out["vmin"] = vmin
    out["vmax"] = vmax
    out["render_ms"] = (time.perf_counter() - t0) * 1000.0
    return out


if True:  # ponytail: always define block; Qt imported inside functions that need it
    try:
        from qgis.PyQt import QtCore, QtGui
    except Exception:
        from PyQt5 import QtCore, QtGui
    try:
        from qgis.core import QgsPointXY
        from qgis.gui import QgsMapCanvasItem
    except Exception:
        QgsPointXY = None
        QgsMapCanvasItem = None

    class SWE2DHighPerfCanvasOverlayItem(QgsMapCanvasItem):
        """Map-canvas overlay item for high-performance rasterized SWE2D frames."""

        def __init__(self, canvas):
            super().__init__(canvas)
            self._canvas_ref = canvas
            self._image = QtGui.QImage()
            self._extent = (0.0, 1.0, 0.0, 1.0)
            self._opacity = 0.65
            self._legend_enabled = False
            self._legend_label = ""
            self._legend_vmin = 0.0
            self._legend_vmax = 1.0
            self._legend_cmap_key = "turbo"
            self._face_segments = np.empty((0, 4), dtype=np.float64)
            self._station_point = None
            self._station_label = ""
            self.setZValue(9999.0)
            self.setVisible(False)

        def _canvas(self):
            """Resolve map canvas across QGIS/PyQt API variations."""
            c = getattr(self, "_canvas_ref", None)
            if c is not None:
                return c
            try:
                getter = getattr(super(), "canvas", None)
                if callable(getter):
                    c = getter()
                    if c is not None:
                        return c
            except Exception:
                pass
            for attr_name in ("mMapCanvas", "mapCanvas", "_mapCanvas"):
                c = getattr(self, attr_name, None)
                if c is not None:
                    return c
            return None

        def set_frame(
            self,
            image: QtGui.QImage,
            extent: Tuple[float, float, float, float],
            opacity: float = 0.65,
        ):
            """Set frame."""
            self._image = image if image is not None else QtGui.QImage()
            self._extent = tuple(extent) if extent is not None else (0.0, 1.0, 0.0, 1.0)
            self._opacity = max(0.0, min(1.0, float(opacity)))
            self.setVisible(not self._image.isNull())
            self.update()

        def clear(self):
            """clear."""
            self._image = QtGui.QImage()
            self._face_segments = np.empty((0, 4), dtype=np.float64)
            self._station_point = None
            self._station_label = ""
            self.setVisible(False)
            self.update()

        def set_face_segments(self, segments_world: np.ndarray):
            """Set face segments."""
            seg = np.asarray(segments_world if segments_world is not None else np.empty((0, 4)), dtype=np.float64)
            if seg.ndim != 2 or seg.shape[1] != 4:
                seg = np.empty((0, 4), dtype=np.float64)
            self._face_segments = seg
            self.update()

        def set_station_indicator(self, world_point: Optional[Tuple[float, float]], label: str = ""):
            """Set station indicator."""
            if world_point is None:
                self._station_point = None
                self._station_label = ""
            else:
                try:
                    self._station_point = (float(world_point[0]), float(world_point[1]))
                    self._station_label = str(label or "")
                except Exception:
                    self._station_point = None
                    self._station_label = ""
            self.update()

        def set_legend(self, enabled: bool, cmap_key: str, vmin: float, vmax: float, label: str):
            """Set legend."""
            self._legend_enabled = bool(enabled)
            self._legend_cmap_key = str(cmap_key or "turbo").strip().lower() or "turbo"
            self._legend_vmin = float(vmin)
            self._legend_vmax = float(vmax)
            self._legend_label = str(label or "")
            self.update()

        def boundingRect(self):
            """Return the bounding rect of this canvas item."""
            c = self._canvas()
            if c is None:
                return QtCore.QRectF()
            return QtCore.QRectF(0.0, 0.0, float(c.width()), float(c.height()))

        def paint(self, painter, option, widget):
            """Paint the overlay item on the map canvas."""
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

            # Highlight selected finite-volume faces used for line-flux sampling.
            if self._face_segments.size > 0:
                try:
                    painter.setOpacity(1.0)
                    pen = QtGui.QPen(QtGui.QColor(220, 40, 40, 230))
                    pen.setWidthF(2.2)
                    painter.setPen(pen)
                    for seg in self._face_segments:
                        x0, y0, x1, y1 = [float(v) for v in seg]
                        p0 = self.toCanvasCoordinates(QgsPointXY(x0, y0))
                        p1 = self.toCanvasCoordinates(QgsPointXY(x1, y1))
                        painter.drawLine(
                            QtCore.QPointF(float(p0.x()), float(p0.y())),
                            QtCore.QPointF(float(p1.x()), float(p1.y())),
                        )
                except Exception:
                    pass

            # Station marker synced from line-profile hover.
            if self._station_point is not None:
                try:
                    sp = self.toCanvasCoordinates(QgsPointXY(float(self._station_point[0]), float(self._station_point[1])))
                    sx = float(sp.x())
                    sy = float(sp.y())
                    painter.setOpacity(1.0)
                    ring_pen = QtGui.QPen(QtGui.QColor(255, 226, 46, 240))
                    ring_pen.setWidthF(2.0)
                    painter.setPen(ring_pen)
                    painter.setBrush(QtGui.QColor(255, 226, 46, 80))
                    painter.drawEllipse(QtCore.QPointF(sx, sy), 5.0, 5.0)
                    painter.drawLine(QtCore.QPointF(sx - 8.0, sy), QtCore.QPointF(sx + 8.0, sy))
                    painter.drawLine(QtCore.QPointF(sx, sy - 8.0), QtCore.QPointF(sx, sy + 8.0))
                    if self._station_label:
                        painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 235)))
                        painter.drawText(QtCore.QPointF(sx + 8.0, sy - 8.0), self._station_label)
                except Exception:
                    pass

            if self._legend_enabled:
                cmap = _COLOR_LUTS.get(self._legend_cmap_key, _COLOR_LUTS.get("turbo"))
                if cmap is not None:
                    draw_scalar_legend_on_painter(
                        painter,
                        int(self.boundingRect().width()),
                        int(self.boundingRect().height()),
                        cmap,
                        float(self._legend_vmin),
                        float(self._legend_vmax),
                        str(self._legend_label),
                    )
            painter.restore()

else:

    class SWE2DHighPerfCanvasOverlayItem:  # type: ignore[no-redef]
        """Fallback placeholder when QGIS canvas item APIs are unavailable."""

        def __init__(self, canvas):
            raise RuntimeError("QGIS canvas overlay APIs are unavailable in this environment.")
