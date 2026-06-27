"""Viewer plotting service — matplotlib mesh wireframe rendering.

Only the Mesh tab uses this path (pyqtgraph handles Time Series and Profile).
Colored depth/velocity rendering is handled by the high-perf canvas overlay.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from swe2d.services.mesh_render_service import plot_mesh_view_on_figure


def render_viewer_figure(
    fig: Any,
    mesh_data: Optional[Dict[str, np.ndarray]],
    result_data: Any = None,
    mode: str = "Mesh",
    h_min: float = 1.0e-6,
    **kwargs,
) -> Any:
    """Render a mesh wireframe figure for the Mesh tab.

    Returns the (modified) figure.
    """
    plot_mesh_view_on_figure(
        fig=fig, mesh_data=mesh_data,
        result_data=None,
        mode="mesh",
        h_min=h_min,
    )
    return fig
