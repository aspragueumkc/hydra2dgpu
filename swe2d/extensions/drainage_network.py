"""Urban drainage integration for SWE2D — GPU-only computational path.

The class SWE2DUrbanDrainageModule serves as a config/state carrier for the
native CUDA drainage solver.  All hydraulic computation (EGL, diffusion-wave,
dynamic-wave, HDS-5 culvert, inlet/outfall exchange) runs on-device.

The Python methods that survive here are only those that the GPU coupling
controller still calls at runtime:
  - _adaptive_substep_count       — determines substep count for GPU iterative mode
  - _use_simplified_link_model    — used by _adaptive_substep_count for DYNAMIC mode
  - _node_area_m2                 — lookup helper for _adaptive_substep_count
  - _adaptive_depth_fraction      — substep heuristic
  - _adaptive_wave_courant        — substep heuristic
  - _max_adaptive_substeps        — substep cap
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from swe2d import units as _u
from swe2d.extensions.extension_models import (
    DrainageLink,
    DrainageNode,
    DrainageSolverMode,
    InletExchange,
    InletType,
    NodeInletAssignment,
    OutfallExchange,
    PipeEndExchange,
    PipeNetworkConfig,
    PipeNetworkState,
)


class DrainageCouplingEngine:
    """Skeleton orchestrator for 2D surface <-> 1D pipe-network exchange."""

    def __init__(self, cfg: PipeNetworkConfig):
        self.cfg = cfg
        self.state = PipeNetworkState()
        self._node_index: Dict[str, int] = {}
        self._node_area: Dict[str, float] = {}
        self._links_from: Dict[str, List[DrainageLink]] = {}
        self._links_to: Dict[str, List[DrainageLink]] = {}
        self._outfall_exchange_nodes: set = set()

    def initialize(self) -> None:
        """Build node/link indices and initialise transient state."""
        self._node_index = {n.node_id: i for i, n in enumerate(self.cfg.nodes)}
        self._node_area = {
            n.node_id: max(
                1.0,
                float(n.metadata.get("surface_area", n.metadata.get("surface_area_m2", 50.0))),
            )
            for n in self.cfg.nodes
        }
        self._links_from = {n.node_id: [] for n in self.cfg.nodes}
        self._links_to = {n.node_id: [] for n in self.cfg.nodes}
        for lnk in self.cfg.links:
            if lnk.from_node_id in self._links_from:
                self._links_from[lnk.from_node_id].append(lnk)
            if lnk.to_node_id in self._links_to:
                self._links_to[lnk.to_node_id].append(lnk)
            self.state.link_flow.setdefault(lnk.link_id, 0.0)
        for n in self.cfg.nodes:
            self.state.node_depth.setdefault(n.node_id, 0.0)
        self._outfall_exchange_nodes = (
            {o.node_id for o in self.cfg.outfalls}
            | {p.node_id for p in getattr(self.cfg, "pipe_ends", [])}
        )
        return None


def build_drainage_config_from_json(
    drainage_data: Optional[Dict[str, Any]],
    n_cells: int,
) -> Optional[PipeNetworkConfig]:
    """Build PipeNetworkConfig from a JSON object.

    Expected format:
    {
        "gravity": 9.81,
        "head_deadband_m": 0.001,
        "dynamic_flow_relaxation": 1.0,
        "solver_mode": 0,
        "coupling_substeps": 1,
        "nodes": [
            {"id": "n1", "type": "inlet", "invert": 8.0, "y_max": 10.0, "area": 10.0,
             "surcharge_depth": 1.0, "initial_depth": 0.0}
        ],
        "links": [
            {"from": "n1", "to": "n2", "length": 100.0, "diameter": 1.0,
             "roughness": 0.013, "max_flow": -1.0}
        ],
        "inlets": [
            {"node_id": "n1", "inlet_cell": 100, "flow_rate": 0.5}
        ],
        "outfalls": [
            {"node_id": "n2", "invert": 3.0}
        ]
    }
    """
    if not drainage_data:
        return None

    data = drainage_data
    nodes_raw = data.get("nodes", [])
    links_raw = data.get("links", [])
    if not nodes_raw or not links_raw:
        return None

    nodes: List[DrainageNode] = []
    for i, n in enumerate(nodes_raw):
        nid = str(n["id"])
        ntype = str(n.get("type", "junction")).lower()
        nodes.append(DrainageNode(
            node_id=nid,
            x=float(n.get("x", 0.0)),
            y=float(n.get("y", 0.0)),
            node_type=ntype,
            invert_elev=float(n.get("invert", 0.0)),
            max_depth=float(n.get("y_max", 10.0)),
            crest_elev=n.get("crest_elev"),
            rim_elev=n.get("rim_elev"),
        ))

    links: List[DrainageLink] = []
    for l in links_raw:
        links.append(DrainageLink(
            link_id=str(l.get("id", f"link_{len(links)}")),
            from_node_id=str(l["from"]),
            to_node_id=str(l["to"]),
            length=float(l.get("length", 100.0)),
            diameter=float(l.get("diameter", 1.0)),
            roughness_n=float(l.get("roughness", 0.013)),
            max_flow=float(l.get("max_flow", -1.0)),
        ))

    inlets_raw = data.get("inlets", [])
    inlets: List[InletExchange] = [
        InletExchange(
            inlet_id=str(i.get("inlet_id", f"in_{idx}")),
            cell_id=int(i.get("cell_id", i.get("inlet_cell", 0))),
            node_id=str(i.get("node_id", "")),
            crest_elev=float(i.get("crest_elev", 0.0)),
            length=float(i.get("length", 1.0)),
            area=float(i.get("area", 0.0)),
            coeff_weir=float(i.get("coeff_weir", 1.70)),
            coeff_orifice=float(i.get("coeff_orifice", 0.62)),
            max_capture=float(i["max_capture"]) if "max_capture" in i and i["max_capture"] is not None else None,
            inlet_type=str(i.get("inlet_type", "custom")),
            grate_length=float(i.get("grate_length", 0.0)),
            grate_width=float(i.get("grate_width", 0.0)),
            grate_type=int(i.get("grate_type", -1)),
            grate_open_frac=float(i.get("grate_open_frac", 1.0)),
            curb_length=float(i.get("curb_length", 0.0)),
            curb_height=float(i.get("curb_height", 0.0)),
            curb_throat=int(i.get("curb_throat", 0)),
            slot_length=float(i.get("slot_length", 0.0)),
            slot_width=float(i.get("slot_width", 0.0)),
        )
        for idx, i in enumerate(inlets_raw)
    ]

    inlet_types: List[InletType] = [
        InletType(inlet_type_id=str(t.get("inlet_type_id", f"it_{idx}")))
        for idx, t in enumerate(data.get("inlet_types", []))
    ]

    node_inlets: List[NodeInletAssignment] = [
        NodeInletAssignment(
            node_id=str(n.get("node_id", "")),
            inlet_type_id=str(n.get("inlet_type_id", "")),
            multiplier=float(n.get("multiplier", 1.0)),
            crest_offset=float(n.get("crest_offset", 0.0)),
        )
        for n in data.get("node_inlets", [])
    ]

    outfalls_raw = data.get("outfalls", [])
    outfalls: List[OutfallExchange] = [
        OutfallExchange(
            outfall_id=str(o.get("outfall_id", f"out_{idx}")),
            cell_id=int(o.get("cell_id", 0)),
            node_id=str(o.get("node_id", "")),
            invert_elev=float(o.get("invert_elev", o.get("invert", 0.0))),
            area_m2=float(o.get("area_m2", 0.0)),
            diameter=float(o.get("diameter", 0.0)),
            coefficient=float(o.get("coefficient", 0.82)),
            max_flow=float(o["max_flow"]) if "max_flow" in o and o["max_flow"] is not None else None,
            zero_storage=bool(o.get("zero_storage", False)),
        )
        for idx, o in enumerate(outfalls_raw)
    ]

    return PipeNetworkConfig(
        nodes=nodes,
        links=links,
        inlets=inlets,
        inlet_types=inlet_types,
        node_inlets=node_inlets,
        outfalls=outfalls,
        pipe_ends=[],
        gravity=float(data.get("gravity", 9.81)),
        head_deadband_m=float(data.get("head_deadband_m", 1.0e-3)),
        dynamic_flow_relaxation=float(data.get("dynamic_flow_relaxation", 1.0)),
        pipe_solver_mode=str(data.get("pipe_solver_mode", "diffusion_wave")),
        coupling_substeps=int(data.get("coupling_substeps", 1)),
    )


class SWE2DUrbanDrainageModule(DrainageCouplingEngine):
    """
    Urban drainage solver: 2D surface <-> 1D pipe-network coupling.

    All hydraulics run on-device via the native CUDA module.  This Python
    class holds the PipeNetworkConfig and node/link state, and provides
    helper methods used by the GPU coupling controller to determine
    sub-stepping and solver parameters.
    """

    def _node_area_m2(self, node_id: str) -> float:
        """node area m2"""
        return max(1.0, float(self._node_area.get(node_id, 50.0)))

    def _use_simplified_link_model(self, link: DrainageLink) -> bool:
        """use simplified link model"""
        t = str(link.link_type or "").strip().lower()
        if t in {"lateral_simple", "lateral", "short_lateral"}:
            return True
        md = link.metadata or {}
        return bool(md.get("simplified", False) or md.get("ignore_inertia", False))

    def _adaptive_depth_fraction(self) -> float:
        """adaptive depth fraction"""
        return min(1.0, max(1.0e-3, float(getattr(self.cfg, "adaptive_depth_fraction", 0.2))))

    def _adaptive_wave_courant(self) -> float:
        """adaptive wave courant"""
        return max(1.0e-3, float(getattr(self.cfg, "adaptive_wave_courant", 0.5)))

    def _max_adaptive_substeps(self) -> int:
        """max adaptive substeps"""
        return max(1, int(getattr(self.cfg, "max_coupling_substeps", 64)))

    def _adaptive_substep_count(self, dt_s: float, solver_mode: DrainageSolverMode) -> int:
        """adaptive substep count"""
        if dt_s <= 0.0 or not self.cfg.nodes:
            return 1

        node_abs_q: Dict[str, float] = {n.node_id: 0.0 for n in self.cfg.nodes}
        dt_limit = float("inf")
        g = max(1.0e-6, float(getattr(self.cfg, "gravity", _u.gravity())))

        for link in self.cfg.links:
            q_est = abs(float(self.state.link_flow.get(link.link_id, 0.0)))
            node_abs_q[link.from_node_id] = node_abs_q.get(link.from_node_id, 0.0) + q_est
            node_abs_q[link.to_node_id] = node_abs_q.get(link.to_node_id, 0.0) + q_est

            if solver_mode == DrainageSolverMode.DYNAMIC and not self._use_simplified_link_model(link):
                diameter = float(link.diameter or link.metadata.get("diameter", 0.0) or 0.0)
                length = max(1.0, float(link.length or 1.0))
                if diameter > 0.0:
                    wave_celerity = math.sqrt(g * max(1.0e-3, diameter))
                    if wave_celerity > 0.0:
                        dt_limit = min(dt_limit, self._adaptive_wave_courant() * length / wave_celerity)

        for node in self.cfg.nodes:
            q_sum = node_abs_q.get(node.node_id, 0.0)
            if q_sum <= 0.0:
                continue
            area = self._node_area_m2(node.node_id)
            max_depth = max(0.0, float(node.max_depth))
            allowed_depth_change = max(1.0e-2, min(5.0e-2, self._adaptive_depth_fraction() * max(max_depth, 0.1)))
            dt_limit = min(dt_limit, area * allowed_depth_change / q_sum)

        if not math.isfinite(dt_limit) or dt_limit <= 0.0:
            return 1
        return min(self._max_adaptive_substeps(), max(1, int(math.ceil(dt_s / dt_limit))))


__all__ = [
    "DrainageNode",
    "DrainageLink",
    "DrainageSolverMode",
    "InletExchange",
    "OutfallExchange",
    "PipeEndExchange",
    "PipeNetworkConfig",
    "SWE2DUrbanDrainageModule",
    "build_drainage_config_from_json",
]
