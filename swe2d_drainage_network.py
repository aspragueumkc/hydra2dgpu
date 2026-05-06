"""Urban drainage and SWMM-style coupling skeleton for SWE2D."""

from __future__ import annotations

from typing import Dict, List, Sequence

from swe2d_extensions import (
    CouplingDiagnostics,
    DrainageCouplingEngine,
    DrainageLink,
    DrainageNode,
    InletExchange,
    PipeNetworkConfig,
    circular_area_from_diameter,
    convert_cell_flows_to_depth_rates,
    compute_orifice_flow,
    compute_pipe_manning_capacity_full,
)


class SWE2DUrbanDrainageModule(DrainageCouplingEngine):
    """
    Placeholder module for 2D surface <-> 1D network coupling.

    Roadmap targets:
    - CPU reference coupling with SWMM-compatible equations
    - optional CUDA acceleration for conduit/link solves
    - two-way exchange at inlets, outfalls, surcharge nodes
    """

    def _node_by_id(self, node_id: str) -> DrainageNode:
        idx = self._node_index.get(node_id, -1)
        if idx < 0:
            raise KeyError(f"Unknown drainage node '{node_id}'")
        return self.cfg.nodes[idx]

    def _node_head_m(self, node: DrainageNode) -> float:
        d = max(0.0, float(self.state.node_depth_m.get(node.node_id, 0.0)))
        return float(node.invert_elev) + d

    def _estimate_link_flow(self, link: DrainageLink) -> float:
        n0 = self._node_by_id(link.from_node_id)
        n1 = self._node_by_id(link.to_node_id)
        h0 = self._node_head_m(n0)
        h1 = self._node_head_m(n1)
        dh = h0 - h1
        if abs(dh) <= 1.0e-12:
            return 0.0

        diameter = float(link.diameter_m or link.metadata.get("diameter_m", 0.0) or 0.0)
        area = float(link.metadata.get("area_m2", 0.0) or 0.0)
        if area <= 0.0 and diameter > 0.0:
            area = circular_area_from_diameter(diameter)
        if area <= 0.0:
            return 0.0

        length = max(1.0, float(link.length_m or 1.0))
        slope = max(1.0e-6, abs(dh) / length)
        q_orifice = compute_orifice_flow(h0, h1, area, discharge_coeff=float(link.metadata.get("cd", 0.75)))
        q_cap = compute_pipe_manning_capacity_full(
            diameter_m=max(diameter, float(link.metadata.get("equiv_diameter_m", 0.0))),
            slope_m_per_m=slope,
            roughness_n=float(link.roughness_n),
        )
        q_mag = abs(q_orifice)
        if q_cap > 0.0:
            q_mag = min(q_mag, q_cap)
        if link.max_flow_cms is not None:
            q_mag = min(q_mag, max(0.0, float(link.max_flow_cms)))
        return q_mag if dh >= 0.0 else -q_mag

    def solve_network_step(self, dt: float) -> Dict[str, float]:
        dt_s = max(1.0e-6, float(dt))
        if not self._node_index:
            self.initialize()

        node_net_q: Dict[str, float] = {n.node_id: 0.0 for n in self.cfg.nodes}
        max_q = 0.0
        for link in self.cfg.links:
            q = self._estimate_link_flow(link)
            self.state.link_flow_cms[link.link_id] = q
            node_net_q[link.from_node_id] -= q
            node_net_q[link.to_node_id] += q
            max_q = max(max_q, abs(q))

        max_depth = 0.0
        for node in self.cfg.nodes:
            area = max(1.0, float(self._node_area_m2.get(node.node_id, 50.0)))
            d0 = max(0.0, float(self.state.node_depth_m.get(node.node_id, 0.0)))
            d1 = d0 + (node_net_q[node.node_id] * dt_s / area)
            d1 = min(max(0.0, d1), max(0.0, float(node.max_depth)))
            self.state.node_depth_m[node.node_id] = d1
            max_depth = max(max_depth, d1)

        diag = CouplingDiagnostics(
            dt_s=dt_s,
            net_node_inflow_cms=float(sum(node_net_q.values())),
            max_node_depth_m=max_depth,
            max_link_flow_cms=max_q,
        )
        return {
            "dt": diag.dt_s,
            "net_node_inflow_cms": diag.net_node_inflow_cms,
            "max_node_depth_m": diag.max_node_depth_m,
            "max_link_flow_cms": diag.max_link_flow_cms,
        }

    def exchange_step(self, dt: float, cell_wse: Sequence[float]):
        dt_s = max(1.0e-6, float(dt))
        if not self._node_index:
            self.initialize()
        n_cells = len(cell_wse)
        sinks = [0.0] * n_cells
        sources = [0.0] * n_cells

        for inlet in self.cfg.inlets:
            ci = int(inlet.cell_id)
            if ci < 0 or ci >= n_cells:
                continue
            try:
                node = self._node_by_id(inlet.node_id)
            except KeyError:
                continue

            wse_surface = float(cell_wse[ci])
            wse_node = self._node_head_m(node)
            capture_head = max(0.0, wse_surface - max(wse_node, float(inlet.crest_elev)))
            if capture_head > 0.0:
                area_capture = max(0.0, float(inlet.width_m)) * max(0.01, capture_head)
                q_capture = compute_orifice_flow(
                    head_up_m=wse_surface,
                    head_down_m=wse_node,
                    area_m2=area_capture,
                    discharge_coeff=float(inlet.coefficient),
                    max_flow_cms=inlet.max_capture_cms,
                )
                q_capture = max(0.0, q_capture)
                sinks[ci] += q_capture
                node_area = max(1.0, float(self._node_area_m2.get(node.node_id, 50.0)))
                d = self.state.node_depth_m.get(node.node_id, 0.0)
                self.state.node_depth_m[node.node_id] = min(
                    float(node.max_depth),
                    max(0.0, float(d) + q_capture * dt_s / node_area),
                )

            surcharge_head = max(0.0, wse_node - max(wse_surface, float(inlet.crest_elev)))
            if surcharge_head > 0.0:
                area_relief = max(0.0, float(inlet.width_m)) * max(0.01, surcharge_head)
                q_relief = compute_orifice_flow(
                    head_up_m=wse_node,
                    head_down_m=wse_surface,
                    area_m2=area_relief,
                    discharge_coeff=float(inlet.coefficient),
                    max_flow_cms=inlet.max_capture_cms,
                )
                q_relief = max(0.0, q_relief)
                sources[ci] += q_relief
                node_area = max(1.0, float(self._node_area_m2.get(node.node_id, 50.0)))
                d = self.state.node_depth_m.get(node.node_id, 0.0)
                self.state.node_depth_m[node.node_id] = max(0.0, float(d) - q_relief * dt_s / node_area)
        return sinks, sources

    def apply_surface_exchange(self, dt: float, cell_wse: Sequence[float]) -> List[float]:
        sinks, sources = self.exchange_step(dt=dt, cell_wse=cell_wse)
        if not sinks and not sources:
            return [0.0] * len(cell_wse)
        combined = [0.0] * len(cell_wse)
        for i, v in enumerate(sinks):
            if i < len(combined):
                combined[i] -= v
        for i, v in enumerate(sources):
            if i < len(combined):
                combined[i] += v
        return combined

    def surface_exchange_source_rate(
        self,
        dt: float,
        cell_wse: Sequence[float],
        cell_area_m2: Sequence[float],
    ) -> List[float]:
        """Return per-cell depth-rate sources [m/s] for 2D coupling."""
        net_flow = self.apply_surface_exchange(dt=dt, cell_wse=cell_wse)
        return convert_cell_flows_to_depth_rates(net_flow, cell_area_m2)


__all__ = [
    "DrainageNode",
    "DrainageLink",
    "InletExchange",
    "PipeNetworkConfig",
    "SWE2DUrbanDrainageModule",
]
