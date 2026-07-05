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
        edge_nodes: np.ndarray,
        node_coords: np.ndarray,
        edge_groups: np.ndarray,
        bc_codes_input: Optional[np.ndarray] = None,
        ts_flow_code: int = 102,
        ts_stage_code: int = 103,
        inflow_q_bc_type: int = 2,
    ) -> None:
        self.edge_nodes = np.asarray(edge_nodes, dtype=np.int32)
        self.node_coords = np.asarray(node_coords, dtype=np.float64)
        self.edge_groups = np.asarray(edge_groups, dtype=np.int32)
        self.bc_codes_input = (
            np.asarray(bc_codes_input, dtype=np.int32)
            if bc_codes_input is not None
            else None
        )
        self.ts_flow_code = int(ts_flow_code)
        self.ts_stage_code = int(ts_stage_code)
        self.inflow_q_bc_type = int(inflow_q_bc_type)

    # ------------------------------------------------------------------
    # Side classification
    # ------------------------------------------------------------------

    def _classify_sides(self) -> np.ndarray:
        """Return array of side names ('left'/'right'/'bottom'/'top') for each edge."""
        n0 = self.edge_nodes[:, 0]
        n1 = self.edge_nodes[:, 1]
        coords = self.node_coords
        mx = 0.5 * (coords[n0, 0] + coords[n1, 0])
        my = 0.5 * (coords[n0, 1] + coords[n1, 1])
        xmin, xmax = float(np.min(coords[:, 0])), float(np.max(coords[:, 0]))
        ymin, ymax = float(np.min(coords[:, 1])), float(np.max(coords[:, 1]))
        d = np.vstack([
            np.abs(mx - xmin),
            np.abs(mx - xmax),
            np.abs(my - ymin),
            np.abs(my - ymax),
        ])
        side_idx = np.argmin(d, axis=0)
        names = np.array(["left", "right", "bottom", "top"])
        return names[side_idx]

    # ------------------------------------------------------------------
    # BC code conversion
    # ------------------------------------------------------------------

    def _convert_bc_codes(self) -> np.ndarray:
        """Convert time-varying BC codes (102/103) to native static codes (2/3)."""
        if self.bc_codes_input is None:
            return np.zeros(len(self.edge_nodes), dtype=np.int32)
        out = self.bc_codes_input.copy()
        out[out == self.ts_flow_code] = self.inflow_q_bc_type  # 102 → 2
        out[out == self.ts_stage_code] = 3  # 103 → 3
        return out

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_payload(self) -> Dict[str, Any]:
        """Return a dict with side classifications and converted BC codes."""
        side_by_edge = self._classify_sides()
        bc_codes_output = self._convert_bc_codes()
        return {
            "side_by_edge": side_by_edge,
            "bc_codes_output": bc_codes_output,
        }
