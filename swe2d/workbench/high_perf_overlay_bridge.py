#!/usr/bin/env python3
"""High-performance overlay bridge helpers for the workbench dialog."""

from __future__ import annotations

from typing import Any

import numpy as np


def sync_high_perf_overlay_data(dialog: Any) -> None:
    """Refresh cached cell-center and bed arrays used by the canvas overlay."""
    if not dialog._snapshot_timesteps:
        dialog._high_perf_overlay_cell_x = np.empty(0, dtype=np.float64)
        dialog._high_perf_overlay_cell_y = np.empty(0, dtype=np.float64)
        dialog._high_perf_overlay_cell_bed = np.empty(0, dtype=np.float64)
        dialog._refresh_high_perf_canvas_overlay(None)
        return

    try:
        cx, cy = dialog._mesh_cell_centroids()
        bed = dialog._mesh_cell_min_bed()
        dialog._high_perf_overlay_cell_x = np.asarray(cx, dtype=np.float64)
        dialog._high_perf_overlay_cell_y = np.asarray(cy, dtype=np.float64)
        dialog._high_perf_overlay_cell_bed = np.asarray(bed, dtype=np.float64)
    except Exception as exc:
        dialog._log(f"[HighPerf Overlay] Data sync failed: {exc}")

    dialog._refresh_high_perf_canvas_overlay(None)


def update_high_perf_overlay_time(dialog: Any, t_s: float) -> None:
    """Update overlay rendering at a specific simulation time."""
    dialog._refresh_high_perf_canvas_overlay(float(t_s))


def destroy_high_perf_canvas_overlay_item(dialog: Any) -> None:
    """Detach overlay canvas item and clear dialog-held references."""
    item = getattr(dialog, "_high_perf_canvas_overlay_item", None)
    dialog._high_perf_canvas_overlay_item = None
    dialog._high_perf_canvas_overlay_enabled = False
    if item is None:
        return
    try:
        canvas = dialog._resolve_map_canvas()
        if canvas is not None and hasattr(canvas, "scene"):
            canvas.scene().removeItem(item)
    except Exception:
        pass
