"""Viewer plotting service — matplotlib rendering for all 5 plot modes.

Each function accepts a matplotlib Figure and data, modifies the Figure
in place, and returns it. No Qt imports. Callable from CLI to render
directly to file::

    fig = Figure(figsize=(6.4, 4.2))
    render_viewer_figure(fig, mesh_data, None, "Mesh", 1e-6)
    fig.savefig("mesh.png")
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from swe2d.workbench.services.mesh_render_service import plot_mesh_view_on_figure
from swe2d.workbench.services.results_render_service import (
    render_timeseries_on_figure,
    render_profile_on_figure,
    render_structures_on_figure,
    render_network_on_figure,
)


def render_viewer_figure(
    fig: Any,
    mesh_data: Optional[Dict[str, np.ndarray]],
    result_data: Any,
    mode: str,
    h_min: float,
    selected_element_id: str = "",
    selected_metric: str = "flow",
    length_unit: str = "",
) -> Any:
    """Render a figure for the given viewer tab mode.

    Dispatches to the appropriate internal renderer based on *mode*.
    Returns the (modified) figure.
    """
    dispatch = {
        "Mesh": plot_mesh_view_on_figure,
        "Time Series": render_timeseries_on_figure,
        "Profile": render_profile_on_figure,
        "Structure": render_structures_on_figure,
        "Network": render_network_on_figure,
    }
    renderer = dispatch.get(mode)
    if renderer is None:
        fig.clear()
        fig.text(0.5, 0.5, f"Unknown mode: {mode}", ha="center", va="center", color="gray")
        return fig

    # ponytail: plot_mesh_view_on_figure doesn't accept extra kwargs; others do
    if mode == "Mesh":
        renderer(fig=fig, mesh_data=mesh_data, result_data=result_data, mode="mesh", h_min=h_min)
    else:
        renderer(
            fig=fig,
            mesh_data=mesh_data,
            result_data=result_data,
            mode=mode,
            h_min=h_min,
            selected_element_id=selected_element_id,
            selected_metric=selected_metric,
            length_unit=length_unit,
        )
    return fig
