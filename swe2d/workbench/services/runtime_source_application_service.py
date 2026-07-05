"""Qt-free application of runtime external source terms."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from swe2d.boundary_and_forcing.bc_logic import (
    EdgeHydrographMap,
    Hydrograph,
    _bc_side_classification,
    distribute_total_flow_to_unit_q,
)


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


def _distribute_total_flow_to_unit_q_logic(
    edge_n0: np.ndarray,
    edge_n1: np.ndarray,
    bc_type_step: np.ndarray,
    bc_val_step: np.ndarray,
    bc_type_template: np.ndarray,
    side_hydrographs: Dict[str, Hydrograph],
    node_x: np.ndarray,
    node_y: np.ndarray,
    node_z: np.ndarray,
    progressive: bool,
    edge_hydrographs: EdgeHydrographMap = None,
    edge_groups: Optional[Dict[int, str]] = None,
    *,
    _side_idx: Optional[np.ndarray] = None,
    _edge_len: Optional[np.ndarray] = None,
    _edge_z: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Distribute total flow BC values to unit discharge per edge without Qt.

    All parameters are forwarded to
    :func:`swe2d.boundary_and_forcing.bc_logic.distribute_total_flow_to_unit_q`.
    Optional pre-computed geometry invariants (``_side_idx``, ``_edge_len``,
    ``_edge_z``) are used when all three are supplied; otherwise they are
    computed from the mesh.
    """
    if _side_idx is None or _edge_len is None or _edge_z is None:
        side_idx, edge_len, edge_z, *_ = _bc_side_classification(
            edge_n0, edge_n1, node_x, node_y, node_z,
        )
    else:
        side_idx, edge_len, edge_z = _side_idx, _edge_len, _edge_z
    return distribute_total_flow_to_unit_q(
        edge_n0=edge_n0, edge_n1=edge_n1,
        bc_type_step=bc_type_step, bc_val_step=bc_val_step,
        bc_type_template=bc_type_template,
        side_hydrographs=side_hydrographs,
        node_x=node_x, node_y=node_y, node_z=node_z,
        progressive=progressive,
        ts_flow_code=102,
        edge_hydrographs=edge_hydrographs,
        edge_groups=edge_groups,
        _side_idx=side_idx,
        _edge_len=edge_len,
        _edge_z=edge_z,
    )
