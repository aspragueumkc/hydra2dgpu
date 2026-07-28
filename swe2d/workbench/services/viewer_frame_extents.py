"""Service layer for viewer frame geometry.

Pure-Python / Qt-free.  Returns per-frame mesh extents and cell centroid
arrays consumed by the color kernel binding (cell_x / cell_y are uploaded
to device per frame).  No field-value computation here — the GPU
viewer computes vmin/vmax on device via ``compute_field_minmax_into_dev``
inside ``swe2d_gpu_render_into_gl_texture_dev``.  Per MVP_ARCHITECTURE.md
Rule 4: the View never does numpy math on mesh data; the service does.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def compute_render_extents(
    mesh_data: dict,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Return ``(xmax, ymax, cell_x, cell_y)`` for a single render.

    If the mesh dict already has precomputed ``cell_x`` / ``cell_y``
    (e.g. mesh baked directly with centroids), they're used as-is.
    Otherwise, derive centroids from the ``cell_nodes`` topology plus
    ``node_x`` / ``node_y`` (the standard ``load_baked_mesh`` shape).

    cell_x / cell_y are float64 ndarrays uploaded to device by the color
    kernel binding each paintGL call (no caching yet — see
    ``swe2d_gpu_render_into_gl_texture_dev``).  ``xmax`` / ``ymax`` are
    floats used to map world coords → pixel coords in the color kernel.

    Returns ``(1.0, 1.0, empty, empty)`` for a mesh with no cells — the
    GPU path treats n_cells==0 as a no-op and the GUI shows a blank
    texture.
    """
    cell_x = np.asarray(
        mesh_data.get("cell_x", np.empty(0, dtype=np.float64)),
        dtype=np.float64,
    ).ravel()
    cell_y = np.asarray(
        mesh_data.get("cell_y", np.empty(0, dtype=np.float64)),
        dtype=np.float64,
    ).ravel()
    if cell_x.size == 0 or cell_y.size == 0:
        # Derive centroids from mesh topology (standard load_baked_mesh
        # representation: node_x/node_y + cell_nodes).
        cell_nodes = mesh_data.get("cell_nodes")
        node_x = mesh_data.get("node_x")
        node_y = mesh_data.get("node_y")
        if (
            cell_nodes is not None
            and node_x is not None
            and node_y is not None
            and len(node_x) > 0
        ):
            from swe2d.services.mesh_computation_service import (
                mesh_cell_centroids,
            )
            cx, cy = mesh_cell_centroids(mesh_data)
            if cx.size and cy.size:
                cell_x = cx
                cell_y = cy
    xmax = float(cell_x.max()) if cell_x.size else 1.0
    ymax = float(cell_y.max()) if cell_y.size else 1.0
    return xmax, ymax, cell_x, cell_y