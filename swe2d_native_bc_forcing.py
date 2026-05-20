#!/usr/bin/env python3
"""Native boundary hydrograph setup seam for SWE2D workbench.

Phase 13 goal: extract native BC hydrograph pre-processing and upload from
`_on_run` into a reusable helper module.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


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
        inflow_q_bc_type: int,
        progressive: bool,
    ) -> Dict[str, Any]:
        xmin = float(np.min(node_x))
        xmax = float(np.max(node_x))
        ymin = float(np.min(node_y))
        ymax = float(np.max(node_y))
        mx = 0.5 * (node_x[bc_n0] + node_x[bc_n1])
        my = 0.5 * (node_y[bc_n0] + node_y[bc_n1])
        d = np.vstack([
            np.abs(mx - xmin),
            np.abs(mx - xmax),
            np.abs(my - ymin),
            np.abs(my - ymax),
        ])
        side_idx = np.argmin(d, axis=0)
        side_names = ["left", "right", "bottom", "top"]
        edge_len = np.hypot(node_x[bc_n1] - node_x[bc_n0], node_y[bc_n1] - node_y[bc_n0])

        def _is_flow(tp_val: int) -> bool:
            return int(tp_val) == int(inflow_q_bc_type)

        any_flow_hg = False
        edge_rows: List[int] = []
        edge_types: List[int] = []
        edge_hgs: List[Tuple[np.ndarray, np.ndarray]] = []

        for bi in range(bc_n0.size):
            hg_info = edge_hydrographs.get(int(bi)) if edge_hydrographs else None
            if hg_info is not None:
                tp_i, hg_i = hg_info
                edge_rows.append(int(bi))
                edge_types.append(int(tp_i))
                edge_hgs.append(hg_i)
                any_flow_hg = any_flow_hg or _is_flow(int(tp_i))
                continue
            side = side_names[int(side_idx[bi])]
            if side in side_hydrographs:
                tp_i = int(bc_tp[bi])
                edge_rows.append(int(bi))
                edge_types.append(tp_i)
                edge_hgs.append(side_hydrographs[side])
                any_flow_hg = any_flow_hg or _is_flow(tp_i)

        if not edge_rows:
            return {
                "native_bc_forcing": False,
                "configured_edges": 0,
                "skipped_progressive": False,
            }

        if progressive and any_flow_hg:
            return {
                "native_bc_forcing": False,
                "configured_edges": 0,
                "skipped_progressive": True,
            }

        edge_index = np.empty(len(edge_rows), dtype=np.int32)
        bc_type_native = np.asarray(edge_types, dtype=np.int32)
        offsets = [0]
        t_all: List[np.ndarray] = []
        v_all: List[np.ndarray] = []

        flow_scale: Dict[int, float] = {}
        for bi in edge_rows:
            key = -1000000 - bi
            hg_info = edge_hydrographs.get(int(bi)) if edge_hydrographs else None
            if hg_info is None:
                key = int(side_idx[bi])
            if key not in flow_scale:
                if key < 0:
                    total_len = max(float(edge_len[bi]), 1.0e-9)
                else:
                    mask = side_idx == key
                    total_len = max(float(np.sum(edge_len[mask])), 1.0e-9)
                flow_scale[key] = total_len

        for j, bi in enumerate(edge_rows):
            a = int(bc_n0[bi])
            b = int(bc_n1[bi])
            keyn = (a, b) if a < b else (b, a)
            edge_index[j] = int(backend._boundary_edge_index_by_nodes[keyn])
            t_i, v_i = edge_hgs[j]
            t_i = np.asarray(t_i, dtype=np.float64).ravel()
            v_i = np.asarray(v_i, dtype=np.float64).ravel()
            if int(bc_type_native[j]) == int(inflow_q_bc_type):
                flow_key = -1000000 - bi
                if not (edge_hydrographs and int(bi) in edge_hydrographs):
                    flow_key = int(side_idx[bi])
                v_i = v_i / max(flow_scale.get(flow_key, 1.0), 1.0e-9)
            t_all.append(t_i)
            v_all.append(v_i)
            offsets.append(offsets[-1] + int(t_i.size))

        time_s_native = np.concatenate(t_all).astype(np.float64, copy=False)
        value_native = np.concatenate(v_all).astype(np.float64, copy=False)
        offsets_native = np.asarray(offsets, dtype=np.int32)
        backend.set_boundary_hydrographs_native(
            edge_index=edge_index,
            bc_type=bc_type_native,
            offsets=offsets_native,
            time_s=time_s_native,
            value=value_native,
        )

        return {
            "native_bc_forcing": True,
            "configured_edges": int(len(edge_rows)),
            "skipped_progressive": False,
        }
