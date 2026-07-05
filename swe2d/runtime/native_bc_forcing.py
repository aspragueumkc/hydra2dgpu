"""Native boundary hydrograph setup seam for SWE2D workbench.

Phase 13 goal: extract native BC hydrograph pre-processing and upload from
`_on_run` into a reusable helper module.
"""

from __future__ import annotations

import math
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
        edge_z = 0.5 * (node_z[bc_n0] + node_z[bc_n1])

        def _is_flow(tp_val: int) -> bool:
            """is flow."""
            return int(tp_val) == int(inflow_q_bc_type) or int(tp_val) == int(ts_flow_code)

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

        edge_index = np.empty(len(edge_rows), dtype=np.int32)
        bc_type_native = np.asarray(edge_types, dtype=np.int32)
        offsets = [0]
        t_all: List[np.ndarray] = []
        v_all: List[np.ndarray] = []

        # Group edges by hydrograph identity (shared tuple object) for length
        # scaling.  Edge-based hydrographs (from BC line layer features) share
        # the same hydrograph tuple across all edges of a feature; side-based
        # hydrographs share the per-side hydrograph.  Summing edge lengths per
        # group gives the correct divisor for total-Q→unit-q conversion.
        # Previously each edge-based hydrograph edge used its own length as the
        # divisor, producing N× the intended total flow (N = edges per feature).
        flow_scale: Dict[Any, float] = {}
        for bi in edge_rows:
            hg_info = edge_hydrographs.get(int(bi)) if edge_hydrographs else None
            if hg_info is not None:
                key = ("edge_hg", id(hg_info[1]))
            else:
                key = ("side_hg", int(side_idx[bi]))
            flow_scale[key] = flow_scale.get(key, 0.0) + float(edge_len[bi])

        # ── Progressive edge group pre-computation ───────────────────────────
        # When progressive is True and flow hydrographs exist, group edges by
        # (side, hydrograph_id), sort by bed elevation, and compute the metadata
        # needed for on-device progressive distribution.  The GPU kernel
        # handles the per-step Q→q conversion using these pre-computed tables,
        # eliminating the Python fallback path entirely.
        need_progressive = bool(progressive and any_flow_hg)
        # Progressive group data structures — uploaded to GPU for the on-device
        # kernel.  Each element's position in the flat arrays ties its
        # sorted rank, edge length, and cumulative length for the serial-group
        # kernel that processes one block per group.
        prog_group_offsets: List[int] = [0]
        prog_group_edge_hg_idx: List[int] = []   # index into edge_rows/hg arrays
        prog_group_edge_len: List[float] = []
        prog_group_cum_len: List[float] = []
        prog_group_peak_q_vals: List[float] = []
        prog_group_total_len_vals: List[float] = []
        if need_progressive:
            prog_groups: Dict[Any, Dict[str, Any]] = {}
            for j, bi in enumerate(edge_rows):
                if not _is_flow(int(bc_type_native[j])):
                    continue
                side = side_names[int(side_idx[bi])]
                hg = edge_hgs[j]
                hg_info_local = edge_hydrographs.get(int(bi)) if edge_hydrographs else None
                if hg_info_local is not None:
                    key = ("edge_hg", id(hg))
                else:
                    key = ("side_hg", side)
                if key not in prog_groups:
                    hg_arr = np.asarray(hg[1], dtype=np.float64).ravel()
                    pk = float(np.max(np.abs(hg_arr))) if hg_arr.size > 0 else 0.0
                    prog_groups[key] = {
                        "hg_indices": [],
                        "elevations": [],
                        "lengths": [],
                        "peak_q": pk,
                    }
                grp = prog_groups[key]
                grp["hg_indices"].append(j)
                grp["elevations"].append(float(edge_z[bi]))
                grp["lengths"].append(float(edge_len[bi]))
                # Peak Q = max of all hydrographs in group (they share the same values)
                hg_arr_t = np.asarray(hg[1], dtype=np.float64).ravel()
                grp["peak_q"] = max(grp["peak_q"], float(np.max(np.abs(hg_arr_t))) if hg_arr_t.size > 0 else 0.0)

            # Flatten groups in sorted-by-elevation order with cumulative lengths
            for key, grp in prog_groups.items():
                n = len(grp["hg_indices"])
                elev = np.asarray(grp["elevations"], dtype=np.float64)
                order = np.argsort(elev, kind="stable")
                cum = 0.0
                for pos in order:
                    j = int(grp["hg_indices"][pos])
                    elen = float(grp["lengths"][pos])
                    cum += elen
                    prog_group_edge_hg_idx.append(j)
                    prog_group_edge_len.append(elen)
                    prog_group_cum_len.append(cum)
                prog_group_peak_q_vals.append(float(grp["peak_q"]))
                prog_group_total_len_vals.append(cum)
                prog_group_offsets.append(len(prog_group_edge_hg_idx))

        for j, bi in enumerate(edge_rows):
            a = int(bc_n0[bi])
            b = int(bc_n1[bi])
            keyn = (a, b) if a < b else (b, a)
            edge_index[j] = int(backend._boundary_edge_index_by_nodes[keyn])
            t_i, v_i = edge_hgs[j]
            t_i = np.asarray(t_i, dtype=np.float64).ravel()
            v_i = np.asarray(v_i, dtype=np.float64).ravel()
            # Convert time-varying BC types (102/103) to static types (2/3)
            native_tp = int(bc_type_native[j])
            if native_tp == int(ts_flow_code):
                bc_type_native[j] = int(inflow_q_bc_type)  # 102 → 2
            elif native_tp == int(ts_stage_code):
                bc_type_native[j] = 3  # 103 → 3
            if _is_flow(int(bc_type_native[j])):
                if need_progressive:
                    # Upload raw Q — GPU kernel does progressive distribution
                    pass  # v_i unchanged
                else:
                    # Non-progressive flow: divide by group total length
                    hg_info_local = edge_hydrographs.get(int(bi)) if edge_hydrographs else None
                    if hg_info_local is not None:
                        flow_key = ("edge_hg", id(hg_info_local[1]))
                    else:
                        flow_key = ("side_hg", int(side_idx[bi]))
                    total_len = flow_scale.get(flow_key, float(edge_len[bi]))
                    v_i = v_i / max(total_len, 1.0e-9)
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

        # Upload progressive BC group metadata when needed.
        n_groups = len(prog_group_peak_q_vals) if need_progressive else 0
        uploaded_progressive = False
        if n_groups > 0:
            backend.set_progressive_bc_data(
                n_groups=n_groups,
                n_edges_total=len(prog_group_edge_hg_idx),
                group_offsets=np.asarray(prog_group_offsets, dtype=np.int32),
                edge_hg_idx=np.asarray(prog_group_edge_hg_idx, dtype=np.int32),
                edge_len=np.asarray(prog_group_edge_len, dtype=np.float64),
                edge_cum_len=np.asarray(prog_group_cum_len, dtype=np.float64),
                group_peak_q=np.asarray(prog_group_peak_q_vals, dtype=np.float64),
                group_total_len=np.asarray(prog_group_total_len_vals, dtype=np.float64),
            )
            uploaded_progressive = True

        return {
            "native_bc_forcing": True,
            "configured_edges": int(len(edge_rows)),
            "skipped_progressive": False,
            "progressive_uploaded": uploaded_progressive,
            "n_prog_edges": len(prog_group_edge_hg_idx) if need_progressive else 0,
        }
