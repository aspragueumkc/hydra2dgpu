"""Native boundary hydrograph setup seam for SWE2D workbench.

Phase 13 goal: extract native BC hydrograph pre-processing and upload from
`_on_run` into a reusable helper module.

The preprocessing now lives in
``swe2d.boundary_and_forcing.native_bc_forcing.BoundaryHydrographConfigurator``.
This module remains a thin facade that delegates to it and uploads the
resulting payload to the backend.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from swe2d.boundary_and_forcing.native_bc_forcing import (
    BoundaryHydrographConfigurator,
)


class SWE2DNativeBoundaryHydrographConfigurator:
    """Build and upload edge hydrograph forcing payloads for native runtime."""

    def configure(
        self,
        *,
        backend: Any,
        bc_n0: np.ndarray,
        bc_n1: np.ndarray,
        bc_tp: np.ndarray,
        side_hydrographs: Dict[str, Tuple[np.ndarray, np.ndarray]],
        edge_hydrographs: Dict[int, Tuple[int, Tuple[np.ndarray, np.ndarray]]],
        node_x: np.ndarray,
        node_y: np.ndarray,
        node_z: np.ndarray,
        inflow_q_bc_type: int,
        progressive: bool,
        ts_flow_code: int = 102,
        ts_stage_code: int = 103,
    ) -> Dict[str, Any]:
        """configure."""
        configurator = BoundaryHydrographConfigurator(
            bc_n0=bc_n0,
            bc_n1=bc_n1,
            bc_tp=bc_tp,
            node_x=node_x,
            node_y=node_y,
            node_z=node_z,
            side_hydrographs=side_hydrographs,
            edge_hydrographs=edge_hydrographs,
            inflow_q_bc_type=inflow_q_bc_type,
            progressive=progressive,
            ts_flow_code=ts_flow_code,
            ts_stage_code=ts_stage_code,
        )
        payload = configurator.build_payload()

        if not payload["native_bc_forcing"]:
            return {
                "native_bc_forcing": False,
                "configured_edges": 0,
                "skipped_progressive": False,
            }

        edge_rows = payload["edge_rows"]
        n_edges = int(edge_rows.size)
        edge_index = np.empty(n_edges, dtype=np.int32)
        for j in range(n_edges):
            bi = int(edge_rows[j])
            a = int(bc_n0[bi])
            b = int(bc_n1[bi])
            keyn = (a, b) if a < b else (b, a)
            edge_index[j] = int(backend._boundary_edge_index_by_nodes[keyn])

        backend.set_boundary_hydrographs_native(
            edge_index=edge_index,
            bc_type=payload["bc_type_native"],
            offsets=payload["offsets_native"],
            time_s=payload["time_s_native"],
            value=payload["value_native"],
        )

        uploaded_progressive = False
        progressive_data = payload.get("progressive_data")
        if progressive_data is not None:
            backend.set_progressive_bc_data(
                n_groups=progressive_data["n_groups"],
                n_edges_total=progressive_data["n_edges_total"],
                group_offsets=progressive_data["group_offsets"],
                edge_hg_idx=progressive_data["edge_hg_idx"],
                edge_len=progressive_data["edge_len"],
                edge_cum_len=progressive_data["edge_cum_len"],
                group_peak_q=progressive_data["group_peak_q"],
                group_total_len=progressive_data["group_total_len"],
            )
            uploaded_progressive = True

        return {
            "native_bc_forcing": True,
            "configured_edges": int(n_edges),
            "skipped_progressive": False,
            "progressive_uploaded": uploaded_progressive,
            "n_prog_edges": progressive_data["n_edges_total"] if progressive_data is not None else 0,
        }
