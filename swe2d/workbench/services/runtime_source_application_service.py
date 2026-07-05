"""Qt-free application of runtime external source terms."""

from __future__ import annotations

from typing import Optional

import numpy as np


def _apply_external_sources_logic(
    backend,
    dt_step: float,
    rain_rate_model,
    cell_source_model: Optional[np.ndarray],
    coupled_source_rate: Optional[np.ndarray],
    mesh_cell_areas: Optional[np.ndarray],
    max_source_rate: float,
    h_min: float,
    max_rel_depth_increase: float,
    max_source_depth_step: float,
    shallow_damping_depth: float,
    momentum_cap_min_speed: float,
    momentum_cap_celerity_mult: float,
) -> None:
    """Apply external source terms without touching Qt widgets.

    All parameters are forwarded to
    :func:`swe2d.boundary_and_forcing.runtime_source_logic.apply_external_sources`.
    Some parameters (``h_min``, ``max_rel_depth_increase``, ``max_source_depth_step``,
    ``shallow_damping_depth``, ``momentum_cap_min_speed``,
    ``momentum_cap_celerity_mult``) are currently passed through for callers that
    already construct the argument list; the underlying function may use them in
    future source-term limiting.
    """
    from swe2d.boundary_and_forcing.runtime_source_logic import (
        apply_external_sources as _logic,
    )

    _logic(
        backend=backend,
        dt_step=dt_step,
        rain_rate_model=rain_rate_model,
        cell_source_model=cell_source_model,
        coupled_source_rate=coupled_source_rate,
        mesh_cell_areas=mesh_cell_areas,
        max_source_rate=max_source_rate,
        h_min=h_min,
        max_rel_depth_increase=max_rel_depth_increase,
        max_source_depth_step=max_source_depth_step,
        shallow_damping_depth=shallow_damping_depth,
        momentum_cap_min_speed=momentum_cap_min_speed,
        momentum_cap_celerity_mult=momentum_cap_celerity_mult,
    )
