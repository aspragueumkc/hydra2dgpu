"""Mesh rendering service.

Pure-Python mesh view rendering extracted from
``SWE2DWorkbenchStudioDialog._render_workbench_mesh_view``.

The service takes mesh data + result data + mode + ``h_min`` as plain
parameters (no Qt, no widget reads) and returns a rendered image as a
RGB ``numpy.ndarray`` of shape ``(H, W, 3)`` and dtype ``uint8``.

Supported modes:

* ``"mesh"``     — wireframe of the cell triangulation
* ``"depth"``    — color-filled depth (h)
* ``"velocity"`` — color-filled velocity magnitude
                   sqrt((hu/h)^2 + (hv/h)^2) with ``h`` masked by ``h_min``

Mesh geometry may be supplied two ways:

* ``cell_nodes`` — flat ``(Nc, 3)`` int array of triangle vertex indices
* ``cell_face_offsets`` + ``cell_face_nodes`` — per-cell face vertex
  lists (e.g. quad/poly cells). The service fans the faces out into
  triangles and tracks a ``tri_to_cell`` index map so per-cell values
  (depth, velocity) are correctly broadcast onto the rendered triangles.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")

from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402


MeshData = Dict[str, Any]
ResultData = Dict[str, Any]
RGBImage = np.ndarray


def render_workbench_mesh_view(
    mesh_data: Optional[MeshData],
    result_data: Optional[ResultData],
    mode: str,
    h_min: float = 1.0e-6,
    figsize: Tuple[float, float] = (6.4, 4.2),
) -> RGBImage:
    """Render the workbench mesh view to a RGB image array.

    Parameters
    ----------
    mesh_data : dict or None
        Mesh geometry. Must contain ``node_x`` and ``node_y`` arrays.
        Provide either ``cell_nodes`` (``(Nc, 3)`` int) or both
        ``cell_face_offsets`` and ``cell_face_nodes``. If ``None``, a
        "No mesh loaded" placeholder is rendered.
    result_data : dict or None
        Per-cell result arrays ``h``, ``hu``, ``hv``. Required for
        ``"depth"`` and ``"velocity"`` modes; ignored for ``"mesh"``.
    mode : str
        One of ``"mesh"``, ``"depth"``, ``"velocity"``.
    h_min : float
        Depth threshold below which velocity magnitude is set to 0.
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    np.ndarray
        RGB image as a ``(H, W, 3)`` ``uint8`` array.
    """
    fig = Figure(figsize=figsize, tight_layout=True)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    render_mesh_view_to_axes(ax, mesh_data, result_data, mode, h_min)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal", adjustable="box")

    canvas.draw()
    buf = np.asarray(canvas.buffer_rgba())
    return buf[:, :, :3]


def plot_mesh_view_on_figure(
    fig: matplotlib.figure.Figure,
    mesh_data: Optional[MeshData],
    result_data: Optional[ResultData],
    mode: str,
    h_min: float = 1.0e-6,
) -> None:
    """Standalone plot function: render mesh view onto an existing Figure.

    Completely decoupled from Qt — takes a ``matplotlib.figure.Figure``
    and plots the mesh view onto it.  Can be called from any Qt backend
    (FigureCanvasQTAgg), a detached PNG export, or a headless script.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to plot onto.  The caller owns the canvas lifecycle.
    mesh_data, result_data, mode, h_min
        Forwarded to ``render_mesh_view_to_axes``.
    """
    fig.clear()
    ax = fig.add_subplot(111)
    render_mesh_view_to_axes(ax, mesh_data, result_data, mode, h_min)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])


def render_mesh_view_to_axes(
    ax: Any,
    mesh_data: Optional[MeshData],
    result_data: Optional[ResultData],
    mode: str,
    h_min: float,
) -> None:
    """Render the workbench mesh view onto an existing matplotlib axes."""
    if mesh_data is None:
        ax.text(
            0.5, 0.5, "No mesh loaded",
            ha="center", va="center", transform=ax.transAxes,
        )
        return

    figure = getattr(ax, "figure", None)
    node_x = mesh_data["node_x"]
    node_y = mesh_data["node_y"]

    tri, tri_to_cell = _build_triangulation(mesh_data, node_x, node_y)

    if tri is None:
        ax.text(
            0.5, 0.5, "Cannot build mesh triangulation",
            ha="center", va="center", transform=ax.transAxes,
        )
        return

    if mode == "mesh" or result_data is None:
        ax.triplot(tri, color="black", linewidth=0.3)
        ax.set_title("Generated mesh")
        return

    vals_cell = _compute_cell_values(result_data, mode, h_min)

    if tri_to_cell is not None and len(tri_to_cell) != len(vals_cell):
        vals_tri = vals_cell[tri_to_cell]
    else:
        vals_tri = vals_cell

    cmap = "viridis" if mode == "depth" else "plasma"
    label = "Depth" if mode == "depth" else "Velocity magnitude"
    title = "Final depth" if mode == "depth" else "Final velocity magnitude"
    tpc = ax.tripcolor(tri, facecolors=vals_tri, cmap=cmap, edgecolors="none")
    if figure is not None:
        figure.colorbar(tpc, ax=ax, label=label)
    ax.set_title(title)


def _build_triangulation(
    mesh_data: MeshData,
    node_x: np.ndarray,
    node_y: np.ndarray,
) -> Tuple[Optional[mtri.Triangulation], Optional[np.ndarray]]:
    """Build a matplotlib triangulation from mesh data.

    Returns the triangulation and (if face-based) the ``tri_to_cell``
    index array. Returns ``(None, None)`` if no valid triangulation can
    be built.
    """
    use_face_offsets = (
        "cell_face_offsets" in mesh_data
        and "cell_face_nodes" in mesh_data
    )

    if use_face_offsets:
        offs = np.asarray(mesh_data["cell_face_offsets"], dtype=np.int32).ravel()
        faces = np.asarray(mesh_data["cell_face_nodes"], dtype=np.int32).ravel()
        tri_list = []
        tc_list = []
        for ci in range(int(offs.size) - 1):
            s = int(offs[ci])
            e = int(offs[ci + 1])
            ns = faces[s:e]
            for k in range(1, int(ns.size) - 1):
                tri_list.append([int(ns[0]), int(ns[k]), int(ns[k + 1])])
                tc_list.append(ci)
        if tri_list:
            triangles = np.asarray(tri_list, dtype=np.int32)
            tri_to_cell = np.asarray(tc_list, dtype=np.int32)
            return mtri.Triangulation(node_x, node_y, triangles), tri_to_cell

    try:
        if "cell_nodes" not in mesh_data:
            return None, None
        triangles = np.asarray(mesh_data["cell_nodes"], dtype=np.int32).reshape((-1, 3))
        return mtri.Triangulation(node_x, node_y, triangles), None
    except (ValueError, KeyError, IndexError):
        return None, None


def _compute_cell_values(
    result_data: ResultData,
    mode: str,
    h_min: float,
) -> np.ndarray:
    """Compute per-cell values for the requested rendering mode."""
    if mode == "depth":
        return np.asarray(result_data["h"], dtype=np.float64)

    h_raw = np.asarray(result_data["h"], dtype=np.float64)
    h_safe = np.maximum(h_raw, 1.0e-12)
    hu = np.asarray(result_data["hu"], dtype=np.float64)
    hv = np.asarray(result_data["hv"], dtype=np.float64)
    return np.where(
        h_raw > h_min,
        np.sqrt((hu / h_safe) ** 2 + (hv / h_safe) ** 2),
        0.0,
    )
