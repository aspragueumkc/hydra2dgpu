"""Hydraulic structures skeleton module for SWE2D."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence

from swe2d.extensions.extension_models import (
    HydraulicStructure,
    HydraulicStructureConfig,
    HydraulicStructureEngine,
    StructureType,
    convert_cell_flows_to_depth_rates,
    compute_orifice_flow,
    compute_weir_flow,
)
from swe2d import units as _u


def _signed_weir_flow(
    upstream_wse: float,
    downstream_wse: float,
    crest_elev: float,
    width: float,
    coeff: float,
) -> float:
    """signed weir flow"""
    return compute_weir_flow(
        upstream_wse=upstream_wse,
        downstream_wse=downstream_wse,
        crest_elev=crest_elev,
        width=width,
        coeff=coeff,
    )


class SWE2DStructureModule(HydraulicStructureEngine):
    """
    Structure dispatcher for future embedded hydraulic controls.

    Supported skeleton types:
    - weirs
    - culverts
    - gates
    - bridges
    - pumps
    """

    def __init__(self, cfg: HydraulicStructureConfig, model_to_ft: float = 1.0):
        super().__init__(cfg)
        _ = model_to_ft  # retained for API compat; all hydraulics run on-device
        self._last_structure_details: List[Dict[str, Any]] = []

    def _structure_detail(self, structure: HydraulicStructure, cell_wse: Sequence[float]) -> Dict[str, Any]:
        """structure detail"""
        if not structure.enabled:
            return {"flow": 0.0, "active": False, "control_mode": "disabled"}
        iu = int(structure.upstream_cell)
        idn = int(structure.downstream_cell)
        if iu < 0 or idn < 0 or iu >= len(cell_wse) or idn >= len(cell_wse):
            return {"flow": 0.0, "active": False, "control_mode": "invalid_cells"}

        wu = float(cell_wse[iu])
        wd = float(cell_wse[idn])
        dz = float(structure.crest_elev)
        md = structure.metadata
        max_q = md.get("max_flow")
        g = max(1.0e-6, float(getattr(self.cfg, "gravity", _u.gravity())))

        detail: Dict[str, Any] = {
            "structure_id": str(structure.structure_id),
            "structure_type": str(structure.structure_type.name).lower(),
            "upstream_cell": iu,
            "downstream_cell": idn,
            "upstream_wse": wu,
            "downstream_wse": wd,
            "flow": 0.0,
            "active": True,
            "control_mode": "none",
        }

        if structure.structure_type == StructureType.CULVERT:
            detail.update({"control_mode": "culvert_gpu", "flow": 0.0})
            return detail

        if structure.structure_type == StructureType.WEIR:
            q = _signed_weir_flow(
                upstream_wse=wu,
                downstream_wse=wd,
                crest_elev=dz,
                width=float(md.get("width", 1.0)),
                coeff=float(md.get("coeff", 1.7)),
            )
            if max_q is not None:
                q = max(-float(max_q), min(float(max_q), q))
            detail.update({"control_mode": "weir", "flow": q})
            return detail

        if structure.structure_type == StructureType.GATE:
            opening = max(0.0, min(1.0, float(md.get("opening", 1.0))))
            width = max(0.0, float(md.get("width", 1.0)))
            height = max(0.0, float(md.get("height", 1.0)))
            area = opening * width * height
            q = compute_orifice_flow(
                head_up=wu,
                head_down=wd,
                area=area,
                discharge_coeff=float(md.get("cd", 0.67)),
                g=g,
                max_flow=float(max_q) if max_q is not None else None,
            )
            detail.update({"control_mode": "gate", "flow": q})
            return detail

        if structure.structure_type == StructureType.PUMP:
            q = max(0.0, float(md.get("q_pump", 0.0)))
            if wu >= wd:
                detail.update({"control_mode": "pump", "flow": q})
                return detail
            detail.update({"control_mode": "pump", "flow": -q})
            return detail

        if structure.structure_type == StructureType.BRIDGE:
            width = max(0.0, float(md.get("width", 0.0)))
            height = max(0.0, float(md.get("height", 0.0)))
            opening = max(0.0, min(1.0, float(md.get("opening", 1.0))))
            area = opening * width * height
            if area <= 0.0:
                return 0.0

            k_up = max(0.0, float(md.get("inlet_loss_k", md.get("coeff", 0.5))))
            k_dn = max(0.0, float(md.get("outlet_loss_k", md.get("coeff", 0.5))))
            loss_scale = max(1.0e-6, 1.0 + k_up + k_dn)
            dh = wu - wd
            if abs(dh) <= 1.0e-12:
                detail.update({"control_mode": "bridge", "flow": 0.0})
                return detail
            q = area * math.sqrt(max(0.0, 2.0 * g * abs(dh))) / loss_scale
            if max_q is not None:
                q = min(q, max(0.0, float(max_q)))
            detail.update({"control_mode": "bridge", "flow": q if dh >= 0.0 else -q})
            return detail

        detail.update({"flow": 0.0, "active": False, "control_mode": "unsupported"})
        return detail

    def _structure_flow(self, structure: HydraulicStructure, cell_wse: Sequence[float]) -> float:
        """structure flow"""
        return float(self._structure_detail(structure, cell_wse).get("flow", 0.0))

    def structure_details(self, cell_wse: Sequence[float]) -> List[Dict[str, Any]]:
        """
        structure details.

        Parameters
        ----------
        cell_wse : Sequence[float]
            Description of cell_wse.

        Returns
        -------
        List[Dict[str, Any]]
        """
        details = [self._structure_detail(st, cell_wse) for st in self.cfg.structures]
        self._last_structure_details = [dict(d) for d in details]
        return details

    @property
    def last_structure_details(self) -> List[Dict[str, Any]]:
        """
        last structure details.

        Returns
        -------
        List[Dict[str, Any]]
        """
        return [dict(d) for d in self._last_structure_details]

    def compute_structure_fluxes(self, dt: float, cell_wse: Sequence[float]) -> Dict[str, float]:
        """
        compute structure fluxes.

        Parameters
        ----------
        dt : float
            Description of dt.
        cell_wse : Sequence[float]
            Description of cell_wse.

        Returns
        -------
        Dict[str, float]
        """
        _ = dt
        total_q = 0.0
        details = self.structure_details(cell_wse)
        culvert_count = 0.0
        embankment_total = 0.0
        for detail in details:
            total_q += abs(float(detail.get("flow", 0.0)))
            if str(detail.get("structure_type", "")) == "culvert":
                culvert_count += 1.0
                embankment_total += abs(float(detail.get("embankment_flow", 0.0) or 0.0))
        return {
            "active_structures": float(sum(1 for s in self.cfg.structures if s.enabled)),
            "total_structure_flow": float(total_q),
            "culvert_count": culvert_count,
            "culvert_embankment_total_flow": float(embankment_total),
        }

    def structure_flows(self, cell_wse: Sequence[float]) -> List[float]:
        """Return signed flow for each configured structure (upstream -> downstream positive)."""
        return [float(d.get("flow", 0.0)) for d in self.structure_details(cell_wse)]

    def compute_cell_source_terms(self, dt: float, cell_wse: Sequence[float]) -> List[float]:
        """
        compute cell source terms.

        Parameters
        ----------
        dt : float
            Description of dt.
        cell_wse : Sequence[float]
            Description of cell_wse.

        Returns
        -------
        List[float]
        """
        _ = dt
        net = [0.0] * len(cell_wse)
        for st in self.cfg.structures:
            q = self._structure_flow(st, cell_wse)
            if q == 0.0:
                continue
            iu = int(st.upstream_cell)
            idn = int(st.downstream_cell)
            if 0 <= iu < len(net):
                net[iu] -= q
            if 0 <= idn < len(net):
                net[idn] += q
        return net

    def compute_cell_source_rate(
        self,
        dt: float,
        cell_wse: Sequence[float],
        cell_area: Sequence[float],
    ) -> List[float]:
        """
        compute cell source rate.

        Parameters
        ----------
        dt : float
            Description of dt.
        cell_wse : Sequence[float]
            Description of cell_wse.
        cell_area : Sequence[float]
            Description of cell_area.

        Returns
        -------
        List[float]
        """
        net = self.compute_cell_source_terms(dt=dt, cell_wse=cell_wse)
        return convert_cell_flows_to_depth_rates(net, cell_area)

    def compute_flux_adjustments(self, dt: float, cell_wse: Sequence[float]) -> Dict[str, float]:
        """
        compute flux adjustments.

        Parameters
        ----------
        dt : float
            Description of dt.
        cell_wse : Sequence[float]
            Description of cell_wse.

        Returns
        -------
        Dict[str, float]
        """
        fluxes = self.compute_structure_fluxes(dt=dt, cell_wse=cell_wse)
        if not cell_wse:
            fluxes["net_source"] = 0.0
            return fluxes

        net = self.compute_cell_source_terms(dt=dt, cell_wse=cell_wse)

        fluxes["net_source"] = float(sum(net))
        fluxes["max_abs_cell_source"] = float(max((abs(v) for v in net), default=0.0))
        return fluxes


__all__ = [
    "StructureType",
    "HydraulicStructure",
    "HydraulicStructureConfig",
    "SWE2DStructureModule",
]
