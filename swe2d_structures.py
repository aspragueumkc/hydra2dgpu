"""Hydraulic structures skeleton module for SWE2D."""

from __future__ import annotations

from typing import Dict, Sequence

from swe2d_extensions import (
    HydraulicStructure,
    HydraulicStructureConfig,
    HydraulicStructureEngine,
    StructureType,
    circular_area_from_diameter,
    convert_cell_flows_to_depth_rates,
    compute_orifice_flow,
    compute_pipe_manning_capacity_full,
    compute_weir_flow,
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

    def _structure_flow(self, structure: HydraulicStructure, cell_wse: Sequence[float]) -> float:
        if not structure.enabled:
            return 0.0
        iu = int(structure.upstream_cell)
        idn = int(structure.downstream_cell)
        if iu < 0 or idn < 0 or iu >= len(cell_wse) or idn >= len(cell_wse):
            return 0.0

        wu = float(cell_wse[iu])
        wd = float(cell_wse[idn])
        dz = float(structure.crest_elev)
        md = structure.metadata
        max_q = md.get("max_flow")
        g = max(1.0e-6, float(getattr(self.cfg, "gravity", 9.81)))

        if structure.structure_type == StructureType.CULVERT:
            diameter = float(md.get("diameter", 0.0) or 0.0)
            area = float(md.get("area_m2", 0.0) or 0.0)
            if area <= 0.0 and diameter > 0.0:
                area = circular_area_from_diameter(diameter)
            if area <= 0.0:
                return 0.0
            q_orifice = compute_orifice_flow(
                head_up_m=wu,
                head_down_m=wd,
                area_m2=area,
                discharge_coeff=float(md.get("cd", 0.75)),
                g=g,
                max_flow=float(max_q) if max_q is not None else None,
            )
            length = max(1.0, float(md.get("length", 1.0)))
            slope = max(1.0e-6, abs(wu - wd) / length)
            q_cap = compute_pipe_manning_capacity_full(
                diameter_m=max(diameter, float(md.get("equiv_diameter_m", 0.0) or 0.0)),
                slope_m_per_m=slope,
                roughness_n=float(md.get("roughness_n", 0.013)),
            )
            if q_cap > 0.0:
                q_orifice = (1.0 if q_orifice >= 0.0 else -1.0) * min(abs(q_orifice), q_cap)
            return q_orifice

        if structure.structure_type == StructureType.WEIR:
            return compute_weir_flow(
                upstream_wse_m=wu,
                downstream_wse_m=wd,
                crest_elev_m=dz,
                width_m=float(md.get("width", 1.0)),
                coeff=float(md.get("coeff", 1.7)),
                max_flow=float(max_q) if max_q is not None else None,
            )

        if structure.structure_type == StructureType.GATE:
            opening = max(0.0, min(1.0, float(md.get("opening", 1.0))))
            width = max(0.0, float(md.get("width", 1.0)))
            height = max(0.0, float(md.get("height", 1.0)))
            area = opening * width * height
            return compute_orifice_flow(
                head_up_m=wu,
                head_down_m=wd,
                area_m2=area,
                discharge_coeff=float(md.get("cd", 0.67)),
                g=g,
                max_flow=float(max_q) if max_q is not None else None,
            )

        if structure.structure_type == StructureType.PUMP:
            q = max(0.0, float(md.get("q_pump", 0.0)))
            if wu >= wd:
                return q
            return -q

        return 0.0

    def compute_structure_fluxes(self, dt: float, cell_wse: Sequence[float]) -> Dict[str, float]:
        _ = dt
        total_q = 0.0
        for st in self.cfg.structures:
            q = self._structure_flow(st, cell_wse)
            total_q += abs(q)
        return {
            "active_structures": float(sum(1 for s in self.cfg.structures if s.enabled)),
            "total_structure_flow": float(total_q),
        }

    def structure_flows(self, cell_wse: Sequence[float]) -> List[float]:
        """Return signed flow for each configured structure (upstream -> downstream positive)."""
        return [float(self._structure_flow(st, cell_wse)) for st in self.cfg.structures]

    def structure_flows_cms(self, cell_wse: Sequence[float]) -> List[float]:
        """Backward-compatible alias for older callers."""
        return self.structure_flows(cell_wse)

    def compute_cell_source_terms(self, dt: float, cell_wse: Sequence[float]) -> List[float]:
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

    def compute_cell_source_terms_cms(self, dt: float, cell_wse: Sequence[float]) -> List[float]:
        """Backward-compatible alias for older callers."""
        return self.compute_cell_source_terms(dt=dt, cell_wse=cell_wse)

    def compute_cell_source_rate(
        self,
        dt: float,
        cell_wse: Sequence[float],
        cell_area_m2: Sequence[float],
    ) -> List[float]:
        net = self.compute_cell_source_terms(dt=dt, cell_wse=cell_wse)
        return convert_cell_flows_to_depth_rates(net, cell_area_m2)

    def compute_flux_adjustments(self, dt: float, cell_wse: Sequence[float]) -> Dict[str, float]:
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
