"""Hydraulic structures skeleton module for SWE2D."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence

from culvert_routine import CircularXsect, RectangularXsect, direct_step_culvert_upstream_energy, inlet_controlled_flow
from swe2d.extensions.extension_models import (
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
from swe2d import units as _u


def _m_to_ft(value_m: float) -> float:
    return float(value_m) * _u.USC_FT_PER_SI_M


def _ft_to_m(value_ft: float) -> float:
    return float(value_ft) / _u.USC_FT_PER_SI_M


def _cms_to_cfs(value_cms: float) -> float:
    return float(value_cms) * _u.USC_FT3_PER_SI_M3


def _cfs_to_cms(value_cfs: float) -> float:
    return float(value_cfs) / _u.USC_FT3_PER_SI_M3


def _culvert_xsect_from_metadata(md: Dict[str, float]):
    shape = str(md.get("culvert_shape", md.get("shape", "circular")) or "circular").strip().lower()
    code = int(round(float(md.get("culvert_code", 1.0))))
    rise_m = float(md.get("culvert_rise", md.get("height", md.get("diameter", 0.0))) or 0.0)
    span_m = float(md.get("culvert_span", md.get("width", rise_m)) or 0.0)
    diameter_m = float(md.get("diameter", rise_m) or 0.0)

    if shape in ("box", "rect", "rectangular"):
        width_ft = max(1.0e-6, _m_to_ft(span_m))
        height_ft = max(1.0e-6, _m_to_ft(rise_m))
        return RectangularXsect(width_ft=width_ft, height_ft=height_ft, culvert_code=code)

    return CircularXsect(diameter_ft=max(1.0e-6, _m_to_ft(diameter_m)), culvert_code=code)


def _culvert_outlet_control_flow_cms(
    *,
    xsect,
    available_head_up_ft: float,
    tailwater_depth_ft: float,
    length_ft: float,
    slope_ftft: float,
    roughness_n: float,
    entrance_loss_k: float,
    exit_loss_k: float,
    q_hint_cfs: float,
) -> float:
    if available_head_up_ft <= 0.0:
        return 0.0

    max_q = max(1.0, q_hint_cfs * 2.0)

    def required_head_ft(q_cfs: float) -> float:
        if q_cfs <= 0.0:
            return 0.0
        e_up_ft, y_up_ft, _mode = direct_step_culvert_upstream_energy(
            xsect=xsect,
            Q=q_cfs,
            n_value=max(1.0e-6, roughness_n),
            slope=max(1.0e-6, slope_ftft),
            length=max(1.0, length_ft),
            tailwater_depth=max(0.0, tailwater_depth_ft),
        )
        area_ft2 = max(xsect.area(max(1.0e-6, min(y_up_ft, xsect.yFull))), 1.0e-9)
        vel_ft_s = q_cfs / area_ft2
        hv_loss = (max(0.0, entrance_loss_k) + max(0.0, exit_loss_k)) * (vel_ft_s * vel_ft_s) / (2.0 * 32.2)
        return float(e_up_ft + hv_loss)

    q_lo = 0.0
    f_lo = -available_head_up_ft
    q_hi = max(1.0, q_hint_cfs * 2.0)
    f_hi = required_head_ft(q_hi) - available_head_up_ft
    for _ in range(12):
        if f_hi >= 0.0:
            break
        q_lo, f_lo = q_hi, f_hi
        q_hi *= 2.0
        f_hi = required_head_ft(q_hi) - available_head_up_ft
    if f_hi < 0.0:
        return _cfs_to_cms(q_hi)

    # Illinois algorithm: secant with stalling-side damping
    side = 0
    for _ in range(16):
        denom = f_hi - f_lo
        if abs(denom) < 1.0e-30:
            break
        q_mid = (q_lo * f_hi - q_hi * f_lo) / denom
        if q_mid <= q_lo or q_mid >= q_hi:
            q_mid = 0.5 * (q_lo + q_hi)
        f_mid = required_head_ft(q_mid) - available_head_up_ft
        if abs(f_mid) < 1.0e-8 * available_head_up_ft:
            return _cfs_to_cms(max(0.0, q_mid))
        if f_lo * f_mid < 0.0:
            q_hi, f_hi = q_mid, f_mid
            if side == 1:
                f_lo *= 0.5
            side = 1
        else:
            q_lo, f_lo = q_mid, f_mid
            if side == 0:
                f_hi *= 0.5
            side = 0
    return _cfs_to_cms(max(0.0, 0.5 * (q_lo + q_hi)))


def _signed_weir_flow(
    upstream_wse_m: float,
    downstream_wse_m: float,
    crest_elev_m: float,
    width_m: float,
    coeff: float,
) -> float:
    return compute_weir_flow(
        upstream_wse_m=upstream_wse_m,
        downstream_wse_m=downstream_wse_m,
        crest_elev_m=crest_elev_m,
        width_m=width_m,
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

    def __init__(self, cfg: HydraulicStructureConfig):
        super().__init__(cfg)
        self._last_structure_details: List[Dict[str, Any]] = []

    def _structure_detail(self, structure: HydraulicStructure, cell_wse: Sequence[float]) -> Dict[str, Any]:
        if not structure.enabled:
            return {"flow_cms": 0.0, "active": False, "control_mode": "disabled"}
        iu = int(structure.upstream_cell)
        idn = int(structure.downstream_cell)
        if iu < 0 or idn < 0 or iu >= len(cell_wse) or idn >= len(cell_wse):
            return {"flow_cms": 0.0, "active": False, "control_mode": "invalid_cells"}

        wu = float(cell_wse[iu])
        wd = float(cell_wse[idn])
        dz = float(structure.crest_elev)
        md = structure.metadata
        max_q = md.get("max_flow")
        g = max(1.0e-6, float(getattr(self.cfg, "gravity", 9.81)))

        detail: Dict[str, Any] = {
            "structure_id": str(structure.structure_id),
            "structure_type": str(structure.structure_type.name).lower(),
            "upstream_cell": iu,
            "downstream_cell": idn,
            "upstream_wse_m": wu,
            "downstream_wse_m": wd,
            "flow_cms": 0.0,
            "active": True,
            "control_mode": "none",
        }

        if structure.structure_type == StructureType.CULVERT:
            inlet_invert = float(md.get("inlet_invert_elev", dz))
            outlet_invert = float(md.get("outlet_invert_elev", inlet_invert))
            sign = 1.0 if wu >= wd else -1.0
            upstream_wse = wu if sign >= 0.0 else wd
            downstream_wse = wd if sign >= 0.0 else wu
            upstream_invert = inlet_invert if sign >= 0.0 else outlet_invert
            downstream_invert = outlet_invert if sign >= 0.0 else inlet_invert
            available_head_up_m = max(0.0, upstream_wse - upstream_invert)
            tailwater_depth_m = max(0.0, downstream_wse - downstream_invert)
            length_m = max(0.1, float(md.get("length", 1.0) or 1.0))
            slope_mpm = float(md.get("culvert_slope", (upstream_invert - downstream_invert) / length_m if length_m > 0.0 else 0.0))
            slope_mpm = max(1.0e-6, abs(slope_mpm))
            roughness_n = max(1.0e-6, float(md.get("roughness_n", 0.013) or 0.013))
            _ent_raw = md.get("inlet_loss_k") if "inlet_loss_k" in md else md.get("entrance_loss_k")
            entrance_loss_k = float(_ent_raw) if _ent_raw is not None else 0.5
            _ext_raw = md.get("outlet_loss_k") if "outlet_loss_k" in md else md.get("exit_loss_k")
            exit_loss_k = float(_ext_raw) if _ext_raw is not None else 1.0
            detail.update(
                {
                    "sign": sign,
                    "inlet_invert_elev_m": inlet_invert,
                    "outlet_invert_elev_m": outlet_invert,
                    "available_head_up_m": available_head_up_m,
                    "tailwater_depth_m": tailwater_depth_m,
                    "culvert_slope": slope_mpm,
                    "entrance_loss_k": entrance_loss_k,
                    "exit_loss_k": exit_loss_k,
                }
            )

            xsect = _culvert_xsect_from_metadata(md)
            q_inlet_cfs, _dqh, _condition, _yr = inlet_controlled_flow(
                xsect,
                max(1.0e-6, slope_mpm),
                max(0.0, _m_to_ft(available_head_up_m)),
            )
            q_inlet = _cfs_to_cms(q_inlet_cfs)
            detail["inlet_control_flow_cms"] = q_inlet

            diameter = float(md.get("diameter", md.get("culvert_rise", 0.0)) or 0.0)
            area = float(md.get("culvert_area_m2", md.get("area_m2", 0.0)) or 0.0)
            if area <= 0.0 and diameter > 0.0 and str(md.get("culvert_shape", "circular")).strip().lower() in ("circular", "pipe", "round"):
                area = circular_area_from_diameter(diameter)
            q_orifice = 0.0
            if area > 0.0:
                q_orifice = abs(
                    compute_orifice_flow(
                        head_up_m=available_head_up_m,
                        head_down_m=tailwater_depth_m,
                        area_m2=area,
                        discharge_coeff=float(md.get("cd", 0.75)),
                        g=g,
                        max_flow=float(max_q) if max_q is not None else None,
                    )
                )
            detail["orifice_cap_cms"] = q_orifice
            q_manning_cap = compute_pipe_manning_capacity_full(
                diameter_m=max(diameter, float(md.get("equiv_diameter_m", 0.0) or 0.0)),
                slope_m_per_m=slope_mpm,
                roughness_n=roughness_n,
            ) if diameter > 0.0 else 0.0
            detail["manning_cap_cms"] = q_manning_cap
            q_outlet = _culvert_outlet_control_flow_cms(
                xsect=xsect,
                available_head_up_ft=max(0.0, _m_to_ft(available_head_up_m)),
                tailwater_depth_ft=max(0.0, _m_to_ft(tailwater_depth_m)),
                length_ft=max(0.1, _m_to_ft(length_m)),
                slope_ftft=max(1.0e-6, slope_mpm),
                roughness_n=roughness_n,
                entrance_loss_k=entrance_loss_k,
                exit_loss_k=exit_loss_k,
                q_hint_cfs=max(q_inlet_cfs, _cms_to_cfs(max(q_orifice, q_manning_cap, 0.0))),
            )
            detail["outlet_control_flow_cms"] = q_outlet
            q_culvert = max(0.0, min(q_inlet, q_outlet if q_outlet > 0.0 else q_inlet))
            control_mode = "inlet_control"
            if q_outlet > 0.0 and q_outlet < q_culvert + 1.0e-12:
                control_mode = "outlet_control"
            if q_orifice > 0.0:
                if q_culvert > 0.0 and q_orifice < q_culvert - 1.0e-12:
                    control_mode = "orifice_cap"
                q_culvert = min(q_culvert, q_orifice) if q_culvert > 0.0 else q_orifice
            if q_manning_cap > 0.0:
                if q_culvert > 0.0 and q_manning_cap < q_culvert - 1.0e-12:
                    control_mode = "manning_cap"
                q_culvert = min(q_culvert, q_manning_cap) if q_culvert > 0.0 else q_manning_cap

            q_emb = 0.0
            if int(round(float(md.get("embankment_enabled", 0.0) or 0.0))) > 0:
                q_emb = abs(_signed_weir_flow(
                    upstream_wse_m=upstream_wse,
                    downstream_wse_m=downstream_wse,
                    crest_elev_m=float(md.get("embankment_crest_elev", md.get("road_crest_elev", dz)) or dz),
                    width_m=float(md.get("embankment_overflow_width", md.get("road_overflow_width", md.get("width", 1.0))) or 1.0),
                    coeff=float(md.get("embankment_weir_coeff", md.get("road_weir_coeff", 1.7)) or 1.7),
                ))
                q_culvert += q_emb

            detail["embankment_flow_cms"] = q_emb

            barrels = max(1.0, float(md.get("culvert_barrels", 1.0) or 1.0))
            q_culvert *= barrels
            if max_q is not None:
                q_culvert = min(q_culvert, max(0.0, float(max_q)))
                if q_culvert >= max(0.0, float(max_q)) - 1.0e-12:
                    control_mode = "max_flow"
            detail["barrels"] = barrels
            detail["control_mode"] = control_mode
            detail["flow_cms"] = q_culvert if sign >= 0.0 else -q_culvert
            return detail

        if structure.structure_type == StructureType.WEIR:
            q = _signed_weir_flow(
                upstream_wse_m=wu,
                downstream_wse_m=wd,
                crest_elev_m=dz,
                width_m=float(md.get("width", 1.0)),
                coeff=float(md.get("coeff", 1.7)),
            )
            if max_q is not None:
                q = max(-float(max_q), min(float(max_q), q))
            detail.update({"control_mode": "weir", "flow_cms": q})
            return detail

        if structure.structure_type == StructureType.GATE:
            opening = max(0.0, min(1.0, float(md.get("opening", 1.0))))
            width = max(0.0, float(md.get("width", 1.0)))
            height = max(0.0, float(md.get("height", 1.0)))
            area = opening * width * height
            q = compute_orifice_flow(
                head_up_m=wu,
                head_down_m=wd,
                area_m2=area,
                discharge_coeff=float(md.get("cd", 0.67)),
                g=g,
                max_flow=float(max_q) if max_q is not None else None,
            )
            detail.update({"control_mode": "gate", "flow_cms": q})
            return detail

        if structure.structure_type == StructureType.PUMP:
            q = max(0.0, float(md.get("q_pump", 0.0)))
            if wu >= wd:
                detail.update({"control_mode": "pump", "flow_cms": q})
                return detail
            detail.update({"control_mode": "pump", "flow_cms": -q})
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
                detail.update({"control_mode": "bridge", "flow_cms": 0.0})
                return detail
            q = area * math.sqrt(max(0.0, 2.0 * g * abs(dh))) / loss_scale
            if max_q is not None:
                q = min(q, max(0.0, float(max_q)))
            detail.update({"control_mode": "bridge", "flow_cms": q if dh >= 0.0 else -q})
            return detail

        detail.update({"flow_cms": 0.0, "active": False, "control_mode": "unsupported"})
        return detail

    def _structure_flow(self, structure: HydraulicStructure, cell_wse: Sequence[float]) -> float:
        return float(self._structure_detail(structure, cell_wse).get("flow_cms", 0.0))

    def structure_details(self, cell_wse: Sequence[float]) -> List[Dict[str, Any]]:
        details = [self._structure_detail(st, cell_wse) for st in self.cfg.structures]
        self._last_structure_details = [dict(d) for d in details]
        return details

    @property
    def last_structure_details(self) -> List[Dict[str, Any]]:
        return [dict(d) for d in self._last_structure_details]

    def compute_structure_fluxes(self, dt: float, cell_wse: Sequence[float]) -> Dict[str, float]:
        _ = dt
        total_q = 0.0
        details = self.structure_details(cell_wse)
        culvert_count = 0.0
        embankment_total = 0.0
        for detail in details:
            total_q += abs(float(detail.get("flow_cms", 0.0)))
            if str(detail.get("structure_type", "")) == "culvert":
                culvert_count += 1.0
                embankment_total += abs(float(detail.get("embankment_flow_cms", 0.0) or 0.0))
        return {
            "active_structures": float(sum(1 for s in self.cfg.structures if s.enabled)),
            "total_structure_flow": float(total_q),
            "culvert_count": culvert_count,
            "culvert_embankment_total_flow": float(embankment_total),
        }

    def structure_flows(self, cell_wse: Sequence[float]) -> List[float]:
        """Return signed flow for each configured structure (upstream -> downstream positive)."""
        return [float(d.get("flow_cms", 0.0)) for d in self.structure_details(cell_wse)]

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
