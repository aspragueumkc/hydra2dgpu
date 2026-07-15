"""Urban drainage integration for SWE2D — GPU-only computational path.

The class SWE2DUrbanDrainageModule serves as a config/state carrier for the
native CUDA drainage solver.  All hydraulic computation (EGL, diffusion-wave,
dynamic-wave, HDS-5 culvert, inlet/outfall exchange) runs on-device.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

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
        meta: Dict[str, float] = {}
        if n.get("surface_area") is not None:
            meta["surface_area"] = float(n["surface_area"])
        if n.get("outfall_area") is not None:
            meta["outfall_area"] = float(n["outfall_area"])
        nodes.append(DrainageNode(
            node_id=nid,
            x=float(n.get("x", 0.0)),
            y=float(n.get("y", 0.0)),
            node_type=ntype,
            invert_elev=float(n.get("invert", 0.0)),
            max_depth=float(n.get("y_max", 10.0)),
            crest_elev=n.get("crest_elev"),
            rim_elev=n.get("rim_elev"),
            metadata=meta,
        ))

    links: List[DrainageLink] = []
    for l in links_raw:
        meta: Dict[str, float] = {}
        if l.get("area_m2") is not None and float(l["area_m2"]) > 0:
            meta["area_m2"] = float(l["area_m2"])
        if l.get("equiv_diameter_m") is not None and float(l["equiv_diameter_m"]) > 0:
            meta["equiv_diameter_m"] = float(l["equiv_diameter_m"])
        links.append(DrainageLink(
            link_id=str(l.get("id", f"link_{len(links)}")),
            from_node_id=str(l["from"]),
            to_node_id=str(l["to"]),
            link_type=str(l.get("link_type", "conduit")),
            length=float(l.get("length", 100.0)),
            roughness_n=float(l.get("roughness", 0.013)),
            diameter=float(l.get("diameter", 0.0)) or None,
            link_shape=str(l.get("link_shape", "circular")),
            width=float(l["span"]) if l.get("span") is not None else None,
            height=float(l["rise"]) if l.get("rise") is not None else None,
            max_flow=float(l.get("max_flow", -1.0)),
            entrance_loss_k=float(l.get("entrance_loss_k", 0.5)),
            exit_loss_k=float(l.get("exit_loss_k", 1.0)),
            max_cell_length=float(l.get("max_cell_length", 0.0)),
            metadata=meta,
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
    class holds the PipeNetworkConfig and node/link state.
    """


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
