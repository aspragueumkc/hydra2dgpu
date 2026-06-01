#!/usr/bin/env python3
"""High-performance overlay bridge helpers for the workbench dialog."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


def mesh_fingerprint_from_arrays(node_x: np.ndarray, node_y: np.ndarray, cell_nodes: np.ndarray) -> str:
    """Build a stable mesh fingerprint from node/cell topology arrays."""
    nx = np.asarray(node_x, dtype=np.float64).ravel()
    ny = np.asarray(node_y, dtype=np.float64).ravel()
    tri = np.asarray(cell_nodes, dtype=np.int32).ravel()
    if nx.size <= 0 or ny.size <= 0 or tri.size <= 0:
        return ""

    n_nodes = int(min(nx.size, ny.size))
    n_tri = int(tri.size // 3)
    if n_nodes <= 0 or n_tri <= 0:
        return ""

    sample_nodes = min(n_nodes, 4096)
    sample_tri = min(tri.size, 12288)
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(nx[:sample_nodes], dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(ny[:sample_nodes], dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(tri[:sample_tri], dtype=np.int32).tobytes())
    h.update(f"|n_nodes={n_nodes}|n_tri={n_tri}|".encode("ascii"))
    return h.hexdigest()


def mesh_fingerprint_from_mesh_data(mesh: Any) -> str:
    """Build fingerprint from workbench mesh-data dictionary."""
    md = mesh or {}
    return mesh_fingerprint_from_arrays(
        np.asarray(md.get("node_x", np.empty(0)), dtype=np.float64),
        np.asarray(md.get("node_y", np.empty(0)), dtype=np.float64),
        np.asarray(md.get("cell_nodes", np.empty(0)), dtype=np.int32),
    )


def sync_high_perf_overlay_data(dialog: Any) -> None:
    """Refresh cached cell-center and bed arrays used by the canvas overlay."""
    if not dialog._snapshot_timesteps:
        dialog._high_perf_overlay_cell_x = np.empty(0, dtype=np.float64)
        dialog._high_perf_overlay_cell_y = np.empty(0, dtype=np.float64)
        dialog._high_perf_overlay_cell_bed = np.empty(0, dtype=np.float64)
        dialog._high_perf_overlay_node_x = np.empty(0, dtype=np.float64)
        dialog._high_perf_overlay_node_y = np.empty(0, dtype=np.float64)
        dialog._high_perf_overlay_cell_nodes = np.empty(0, dtype=np.int32)
        dialog._high_perf_overlay_tri_to_cell = np.empty(0, dtype=np.int32)
        dialog._high_perf_overlay_mesh_fingerprint = ""
        dialog._refresh_high_perf_canvas_overlay(None)
        return

    try:
        cx, cy = dialog._mesh_cell_centroids()
        bed = dialog._mesh_cell_solver_bed()
        dialog._high_perf_overlay_cell_x = np.asarray(cx, dtype=np.float64)
        dialog._high_perf_overlay_cell_y = np.asarray(cy, dtype=np.float64)
        dialog._high_perf_overlay_cell_bed = np.asarray(bed, dtype=np.float64)
        mesh = getattr(dialog, "_mesh_data", {}) or {}
        dialog._high_perf_overlay_node_x = np.asarray(mesh.get("node_x", np.empty(0)), dtype=np.float64).ravel()
        dialog._high_perf_overlay_node_y = np.asarray(mesh.get("node_y", np.empty(0)), dtype=np.float64).ravel()
        # Build fan triangulation from polygon CSR when available, keeping a
        # tri_to_cell index map so the renderer can expand per-cell values to
        # per-triangle (needed for quad/polygon meshes after roundtrip import).
        raw_cell_nodes = np.asarray(mesh.get("cell_nodes", np.empty(0)), dtype=np.int32).ravel()
        if "cell_face_offsets" in mesh and "cell_face_nodes" in mesh:
            offs = np.asarray(mesh["cell_face_offsets"], dtype=np.int32).ravel()
            faces = np.asarray(mesh["cell_face_nodes"], dtype=np.int32).ravel()
            tri_list = []
            tc_list = []
            for ci in range(int(offs.size) - 1):
                s = int(offs[ci])
                e = int(offs[ci + 1])
                ns = faces[s:e]
                # Fan triangulation from first vertex
                for k in range(1, int(ns.size) - 1):
                    tri_list.append([int(ns[0]), int(ns[k]), int(ns[k + 1])])
                    tc_list.append(ci)
            if tri_list:
                dialog._high_perf_overlay_cell_nodes = np.asarray(tri_list, dtype=np.int32).ravel()
                dialog._high_perf_overlay_tri_to_cell = np.asarray(tc_list, dtype=np.int32)
            else:
                dialog._high_perf_overlay_cell_nodes = raw_cell_nodes
                dialog._high_perf_overlay_tri_to_cell = np.empty(0, dtype=np.int32)
        else:
            dialog._high_perf_overlay_cell_nodes = raw_cell_nodes
            dialog._high_perf_overlay_tri_to_cell = np.empty(0, dtype=np.int32)
        dialog._high_perf_overlay_mesh_fingerprint = mesh_fingerprint_from_arrays(
            dialog._high_perf_overlay_node_x,
            dialog._high_perf_overlay_node_y,
            dialog._high_perf_overlay_cell_nodes,
        )
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
