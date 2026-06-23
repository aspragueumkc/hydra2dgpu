"""Pure-Python mesh data preparation service for the workbench overlay pipeline.

Extracted from
``SWE2DWorkbenchStudioDialog._apply_overlay_frame``,
``SWE2DWorkbenchStudioDialog._reset_runtime_snapshot_overlay_cache``,
and the high-perf overlay bridge in
``swe2d.workbench.high_perf_overlay_bridge`` (Task 4 of
docs/STUDIO_GUI_FULL_MIGRATION_PLAN_2026-06-16.md).

The service owns the numpy bundles that drive the high-perf canvas
overlay (cell x, cell y, cell bed, node x, node y, cell nodes,
``tri_to_cell`` index map, mesh fingerprint). It is pure Python — it
does not touch Qt. The dialog and bridge call this service instead of
constructing numpy arrays inline.

The ``overlay_frame_inputs`` and ``overlay_frame_is_valid`` helpers
separate "what came out of the renderer" from "what Qt needs to draw
it" so the dialog can fail loudly on a null ``QImage`` without the
service needing a Qt dependency.

NO SILENT FALLBACKS:
    * ``create_empty_overlay_arrays`` returns a fresh dict on every call
      — no shared mutable state.
    * ``prepare_overlay_arrays`` does not invent cell/node data when
      the mesh dict is ``None``; the arrays stay empty.
    * ``overlay_frame_is_valid`` treats a ``None`` image as invalid
      and a qimage whose ``isNull()`` raises as invalid (deleted C++
      object).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

__all__ = [
    "create_empty_overlay_arrays",
    "prepare_overlay_arrays",
    "overlay_frame_inputs",
    "overlay_frame_is_valid",
]


OverlayArrays = Dict[str, Any]


def create_empty_overlay_arrays() -> OverlayArrays:
    """Return a canonical "no data yet" numpy bundle for the overlay.

    The returned dict contains the keys the dialog / bridge read:
        ``cell_x``, ``cell_y``, ``cell_bed`` — per-cell arrays
        ``node_x``, ``node_y`` — per-node arrays
        ``cell_nodes`` — flat triangle vertex indices (``(N, 3)`` int32)
        ``tri_to_cell`` — per-triangle cell index (``(N,)`` int32)
        ``mesh_fingerprint`` — stable hash of the node/cell topology

    Every call returns a fresh dict with freshly-allocated arrays, so
    callers can mutate the result without affecting other consumers.
    """
    return {
        "cell_x": np.empty(0, dtype=np.float64),
        "cell_y": np.empty(0, dtype=np.float64),
        "cell_bed": np.empty(0, dtype=np.float64),
        "node_x": np.empty(0, dtype=np.float64),
        "node_y": np.empty(0, dtype=np.float64),
        "cell_nodes": np.empty(0, dtype=np.int32),
        "tri_to_cell": np.empty(0, dtype=np.int32),
        "mesh_fingerprint": "",
    }


def prepare_overlay_arrays(
    mesh_data: Optional[Dict[str, Any]],
    cell_centroids_x: np.ndarray,
    cell_centroids_y: np.ndarray,
    cell_bed: np.ndarray,
) -> OverlayArrays:
    """Build the populated overlay arrays from mesh + per-cell data.

    Parameters
    ----------
    mesh_data : dict or None
        Optional mesh geometry dict. ``None`` or empty dict is allowed;
        the node arrays stay empty and the cell arrays come purely
        from the centroid / bed parameters.
    cell_centroids_x, cell_centroids_y : np.ndarray
        Per-cell centroid coordinates in the mesh CRS.
    cell_bed : np.ndarray
        Per-cell bed elevation (model units).

    Returns
    -------
    dict
        Same shape as :func:`create_empty_overlay_arrays`, with the
        arrays populated. When ``mesh_data`` carries ``cell_face_offsets``
        and ``cell_face_nodes``, polygon cells are fan-triangulated and
        ``tri_to_cell`` is populated. Otherwise the raw ``cell_nodes``
        is used and ``tri_to_cell`` stays empty.
    """
    result = create_empty_overlay_arrays()

    result["cell_x"] = np.asarray(cell_centroids_x, dtype=np.float64)
    result["cell_y"] = np.asarray(cell_centroids_y, dtype=np.float64)
    result["cell_bed"] = np.asarray(cell_bed, dtype=np.float64)

    mesh = mesh_data or {}
    result["node_x"] = np.asarray(
        mesh.get("node_x", np.empty(0)), dtype=np.float64,
    ).ravel()
    result["node_y"] = np.asarray(
        mesh.get("node_y", np.empty(0)), dtype=np.float64,
    ).ravel()

    raw_cell_nodes = np.asarray(
        mesh.get("cell_nodes", np.empty(0)), dtype=np.int32,
    ).ravel()

    if "cell_face_offsets" in mesh and "cell_face_nodes" in mesh:
        triangles, tri_to_cell = _fan_triangulate_polygons(
            np.asarray(mesh["cell_face_offsets"], dtype=np.int32).ravel(),
            np.asarray(mesh["cell_face_nodes"], dtype=np.int32).ravel(),
        )
        if triangles.size > 0:
            result["cell_nodes"] = triangles.ravel()
            result["tri_to_cell"] = tri_to_cell
        else:
            result["cell_nodes"] = raw_cell_nodes
    else:
        result["cell_nodes"] = raw_cell_nodes

    return result


def _fan_triangulate_polygons(
    cell_face_offsets: np.ndarray,
    cell_face_nodes: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fan-triangulate polygon cells (CSR-style) into triangles.

    For each cell ``c`` with face nodes ``[v0, v1, v2, ..., vN-1]``,
    emit triangles ``(v0, v1, v2), (v0, v2, v3), ..., (v0, vN-2, vN-1)``.

    Parameters
    ----------
    cell_face_offsets : np.ndarray
        CSR-style offsets. ``cell_face_offsets[c]`` is the start of cell
        ``c``'s face in ``cell_face_nodes``; ``cell_face_offsets[c+1]``
        is the end. Length = ``num_cells + 1``.
    cell_face_nodes : np.ndarray
        Flat int32 array of all cell face vertices, concatenated.

    Returns
    -------
    tuple ``(triangles, tri_to_cell)``
        ``triangles`` has shape ``(num_tri, 3)`` int32 vertex indices.
        ``tri_to_cell`` has shape ``(num_tri,)`` int32 cell indices.
        Both are empty arrays if the input produced no triangles.
    """
    offs = np.asarray(cell_face_offsets, dtype=np.int32).ravel()
    faces = np.asarray(cell_face_nodes, dtype=np.int32).ravel()
    tri_list = []
    tc_list = []
    for ci in range(int(offs.size) - 1):
        s = int(offs[ci])
        e = int(offs[ci + 1])
        ns = faces[s:e]
        for k in range(1, int(ns.size) - 1):
            tri_list.append([int(ns[0]), int(ns[k]), int(ns[k + 1])])
            tc_list.append(ci)
    if not tri_list:
        empty_tri = np.empty((0, 3), dtype=np.int32)
        empty_tc = np.empty(0, dtype=np.int32)
        return empty_tri, empty_tc
    return (
        np.asarray(tri_list, dtype=np.int32),
        np.asarray(tc_list, dtype=np.int32),
    )


def overlay_frame_inputs(
    frame: Dict[str, Any],
    default_opacity: float = 1.0,
) -> Tuple[Any, Tuple[float, float, float, float], float]:
    """Extract ``(qimage, extent, opacity)`` from a rendered frame dict.

    The QImage is treated opaquely (no Qt methods called); ``extent``
    falls back to ``(0.0, 1.0, 0.0, 1.0)`` if missing; ``opacity`` is
    the caller-supplied default (no per-frame override is honored here
    — the dialog owns the opacity slider).

    Parameters
    ----------
    frame : dict
        Renderer output dict. May contain ``image`` (QImage-like) and
        ``extent`` (4-tuple of floats).
    default_opacity : float
        Opacity the dialog wants to use for this frame.

    Returns
    -------
    tuple
        ``(qimage, extent, opacity)``.
    """
    qimage = frame.get("image", None)
    extent = frame.get("extent", (0.0, 1.0, 0.0, 1.0))
    return qimage, tuple(extent), float(default_opacity)


def overlay_frame_is_valid(qimage: Any) -> bool:
    """Check whether a QImage is valid for overlay rendering.

    Rules:
        * ``None`` → invalid.
        * Object with an ``isNull()`` method that returns ``True`` →
          invalid.
        * Object with an ``isNull()`` method that **raises** (deleted
          C++ object) → invalid.
        * Otherwise → valid.

    This helper is the single point where the dialog decides whether
    to apply a frame. It is in the service (not the dialog) so the
    decision is testable without spinning up QGIS.
    """
    if qimage is None:
        return False
    is_null = getattr(qimage, "isNull", None)
    if is_null is None:
        return True
    try:
        return not bool(is_null())
    except Exception:
        return False
