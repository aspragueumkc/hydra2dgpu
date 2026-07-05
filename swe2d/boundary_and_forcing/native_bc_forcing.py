"""Pure-logic BC configurator extracted from swe2d.runtime.native_bc_forcing.

The ``BoundaryHydrographConfigurator`` performs all preprocessing (side
detection, hydrograph grouping, progressive-BC sorting, code conversion)
and returns a payload dict.  The runtime shim in
``swe2d.runtime.native_bc_forcing`` applies this payload to the backend.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class BoundaryHydrographConfigurator:
    """Build a payload dict describing how boundary edges should be forced."""

    def __init__(
        self,
        *,
        bc_n0: np.ndarray,
        bc_n1: np.ndarray,
        bc_tp: np.ndarray,
        node_x: np.ndarray,
        node_y: np.ndarray,
        node_z: np.ndarray,
        side_hydrographs: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
        edge_hydrographs: Optional[Dict[int, Tuple[int, Tuple[np.ndarray, np.ndarray]]]] = None,
        inflow_q_bc_type: int = 2,
        progressive: bool = False,
        ts_flow_code: int = 102,
        ts_stage_code: int = 103,
    ) -> None:
        self.bc_n0 = np.asarray(bc_n0, dtype=np.int32).ravel()
        self.bc_n1 = np.asarray(bc_n1, dtype=np.int32).ravel()
        self.bc_tp = np.asarray(bc_tp, dtype=np.int32).ravel()
        self.node_x = np.asarray(node_x, dtype=np.float64).ravel()
        self.node_y = np.asarray(node_y, dtype=np.float64).ravel()
        self.node_z = np.asarray(node_z, dtype=np.float64).ravel()
        self.side_hydrographs = side_hydrographs or {}
        self.edge_hydrographs = edge_hydrographs or {}
        self.inflow_q_bc_type = int(inflow_q_bc_type)
        self.progressive = bool(progressive)
        self.ts_flow_code = int(ts_flow_code)
        self.ts_stage_code = int(ts_stage_code)

    # ------------------------------------------------------------------
    # Side classification
    # ------------------------------------------------------------------

    def _classify_sides(self) -> np.ndarray:
        """Return array of side indices (0=left, 1=right, 2=bottom, 3=top) for each edge."""
        mx = 0.5 * (self.node_x[self.bc_n0] + self.node_x[self.bc_n1])
        my = 0.5 * (self.node_y[self.bc_n0] + self.node_y[self.bc_n1])
        xmin, xmax = float(np.min(self.node_x)), float(np.max(self.node_x))
        ymin, ymax = float(np.min(self.node_y)), float(np.max(self.node_y))
        d = np.vstack([
            np.abs(mx - xmin),
            np.abs(mx - xmax),
            np.abs(my - ymin),
            np.abs(my - ymax),
        ])
        return np.argmin(d, axis=0)

    @staticmethod
    def _side_name(side_idx: int) -> str:
        return ["left", "right", "bottom", "top"][int(side_idx)]

    def _is_flow(self, tp_val: int) -> bool:
        return int(tp_val) == self.inflow_q_bc_type or int(tp_val) == self.ts_flow_code

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_payload(self) -> Dict[str, Any]:
        """Return a preprocessing payload for native BC hydrograph setup.

        The payload contains everything the runtime needs to upload boundary
        hydrographs to the backend except the backend-specific edge-index
        mapping.
        """
        side_idx = self._classify_sides()
        side_names_arr = np.array(["left", "right", "bottom", "top"])
        side_by_edge = side_names_arr[side_idx]
        edge_len = np.hypot(
            self.node_x[self.bc_n1] - self.node_x[self.bc_n0],
            self.node_y[self.bc_n1] - self.node_y[self.bc_n0],
        )
        edge_z = 0.5 * (self.node_z[self.bc_n0] + self.node_z[self.bc_n1])

        edge_rows: List[int] = []
        edge_types: List[int] = []
        edge_hgs: List[Tuple[np.ndarray, np.ndarray]] = []
        any_flow_hg = False

        for bi in range(self.bc_n0.size):
            hg_info = self.edge_hydrographs.get(int(bi))
            if hg_info is not None:
                tp_i, hg_i = hg_info
                edge_rows.append(int(bi))
                edge_types.append(int(tp_i))
                edge_hgs.append(hg_i)
                any_flow_hg = any_flow_hg or self._is_flow(int(tp_i))
                continue
            side_name = side_by_edge[int(bi)]
            if side_name in self.side_hydrographs:
                tp_i = int(self.bc_tp[bi])
                edge_rows.append(int(bi))
                edge_types.append(tp_i)
                edge_hgs.append(self.side_hydrographs[side_name])
                any_flow_hg = any_flow_hg or self._is_flow(tp_i)

        if not edge_rows:
            return {
                "native_bc_forcing": False,
                "configured_edges": 0,
                "skipped_progressive": False,
                "edge_rows": np.empty(0, dtype=np.int32),
                "bc_type_native": np.empty(0, dtype=np.int32),
                "offsets_native": np.array([0], dtype=np.int32),
                "time_s_native": np.empty(0, dtype=np.float64),
                "value_native": np.empty(0, dtype=np.float64),
                "side_by_edge": side_by_edge,
                "progressive_data": None,
            }

        bc_type_native = np.asarray(edge_types, dtype=np.int32)

        # Group edges by hydrograph identity for length scaling.
        flow_scale: Dict[Any, float] = {}
        for bi in edge_rows:
            hg_info = self.edge_hydrographs.get(int(bi))
            if hg_info is not None:
                key = ("edge_hg", id(hg_info[1]))
            else:
                key = ("side_hg", int(side_idx[int(bi)]))
            flow_scale[key] = flow_scale.get(key, 0.0) + float(edge_len[bi])

        # Progressive edge group pre-computation.
        need_progressive = bool(self.progressive and any_flow_hg)
        prog_group_offsets: List[int] = [0]
        prog_group_edge_hg_idx: List[int] = []
        prog_group_edge_len: List[float] = []
        prog_group_cum_len: List[float] = []
        prog_group_peak_q_vals: List[float] = []
        prog_group_total_len_vals: List[float] = []

        if need_progressive:
            prog_groups: Dict[Any, Dict[str, Any]] = {}
            for j, bi in enumerate(edge_rows):
                if not self._is_flow(int(bc_type_native[j])):
                    continue
                side_name = side_by_edge[int(bi)]
                hg = edge_hgs[j]
                hg_info = self.edge_hydrographs.get(int(bi))
                if hg_info is not None:
                    key = ("edge_hg", id(hg))
                else:
                    key = ("side_hg", side_name)
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
                hg_arr_t = np.asarray(hg[1], dtype=np.float64).ravel()
                grp["peak_q"] = max(
                    grp["peak_q"],
                    float(np.max(np.abs(hg_arr_t))) if hg_arr_t.size > 0 else 0.0,
                )

            for key, grp in prog_groups.items():
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

        offsets = [0]
        t_all: List[np.ndarray] = []
        v_all: List[np.ndarray] = []

        for j, bi in enumerate(edge_rows):
            native_tp = int(bc_type_native[j])
            if native_tp == self.ts_flow_code:
                bc_type_native[j] = self.inflow_q_bc_type
            elif native_tp == self.ts_stage_code:
                bc_type_native[j] = 3

            t_i, v_i = edge_hgs[j]
            t_i = np.asarray(t_i, dtype=np.float64).ravel()
            v_i = np.asarray(v_i, dtype=np.float64).ravel()

            if self._is_flow(int(bc_type_native[j])):
                if not need_progressive:
                    hg_info = self.edge_hydrographs.get(int(bi))
                    if hg_info is not None:
                        flow_key = ("edge_hg", id(hg_info[1]))
                    else:
                        flow_key = ("side_hg", int(side_idx[int(bi)]))
                    total_len = flow_scale.get(flow_key, float(edge_len[bi]))
                    v_i = v_i / max(total_len, 1.0e-9)

            t_all.append(t_i)
            v_all.append(v_i)
            offsets.append(offsets[-1] + int(t_i.size))

        payload: Dict[str, Any] = {
            "native_bc_forcing": True,
            "configured_edges": int(len(edge_rows)),
            "skipped_progressive": False,
            "edge_rows": np.asarray(edge_rows, dtype=np.int32),
            "bc_type_native": bc_type_native,
            "offsets_native": np.asarray(offsets, dtype=np.int32),
            "time_s_native": np.concatenate(t_all).astype(np.float64, copy=False),
            "value_native": np.concatenate(v_all).astype(np.float64, copy=False),
            "side_by_edge": side_by_edge,
            "progressive_data": None,
        }

        if need_progressive:
            n_groups = len(prog_group_peak_q_vals)
            if n_groups > 0:
                payload["progressive_data"] = {
                    "n_groups": n_groups,
                    "n_edges_total": len(prog_group_edge_hg_idx),
                    "group_offsets": np.asarray(prog_group_offsets, dtype=np.int32),
                    "edge_hg_idx": np.asarray(prog_group_edge_hg_idx, dtype=np.int32),
                    "edge_len": np.asarray(prog_group_edge_len, dtype=np.float64),
                    "edge_cum_len": np.asarray(prog_group_cum_len, dtype=np.float64),
                    "group_peak_q": np.asarray(prog_group_peak_q_vals, dtype=np.float64),
                    "group_total_len": np.asarray(prog_group_total_len_vals, dtype=np.float64),
                }

        return payload
