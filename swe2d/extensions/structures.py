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


def _model_to_ft(value_model: float, model_to_ft: float = 1.0) -> float:
    """Convert a model-unit value to feet.

    For a foot-based model (model_to_ft=1.0) this is a no-op.
    For an SI model (model_to_ft=3.28084) this converts meters to feet.
    """
    return float(value_model) * float(model_to_ft)


def _model3_to_cfs(value_model3: float, model_to_ft: float = 1.0) -> float:
    """Convert a model-unit-volume/s flow to ft³/s.

    For foot model (model_to_ft=1.0): value already in ft³/s → no change.
    For SI model (model_to_ft=3.28): value in m³/s → multiply by ft³/m³.
    """
    v = float(value_model3)
    scale = float(model_to_ft) ** 3 / _u.USC_FT3_PER_SI_M3  # model³/s → ft³/s
    return v * scale if abs(scale - 1.0) > 1e-12 else v


def _culvert_xsect_from_metadata(md: Dict[str, float], model_to_ft: float = 1.0):
    shape = str(md.get("culvert_shape", md.get("shape", "circular")) or "circular").strip().lower()
    code = int(round(float(md.get("culvert_code", 1.0))))
    # GPKG stores dimensions in model units (feet for US model, meters for SI).
    # Convert to feet using the model_to_ft factor.
    rise_model = float(md.get("culvert_rise", md.get("height", md.get("diameter", 0.0))) or 0.0)
    span_model = float(md.get("culvert_span", md.get("width", rise_model)) or 0.0)
    diameter_model = float(md.get("diameter", rise_model) or 0.0)

    if shape in ("box", "rect", "rectangular"):
        width_ft = max(1.0e-6, _model_to_ft(span_model, model_to_ft))
        height_ft = max(1.0e-6, _model_to_ft(rise_model, model_to_ft))
        return RectangularXsect(width_ft=width_ft, height_ft=height_ft, culvert_code=code)

    return CircularXsect(diameter_ft=max(1.0e-6, _model_to_ft(diameter_model, model_to_ft)), culvert_code=code)


def _culvert_outlet_control_flow(
    *,
    xsect,
    available_head_up_ft: float,  # L  (upstream head in feet)
    tailwater_depth_ft: float,   # L  (tailwater depth in feet)
    length_ft: float,            # L  (culvert barrel length in feet)
    slope_ftft: float,           # L/L (dimensionless slope)
    roughness_n: float,
    entrance_loss_k: float,
    exit_loss_k: float,
    q_hint_cfs: float,           # L³T⁻¹ (flow hint in ft³/s)
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
    upstream_wse: float,
    downstream_wse: float,
    crest_elev: float,
    width: float,
    coeff: float,
) -> float:
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
        self._model_to_ft = float(model_to_ft)
        self._last_structure_details: List[Dict[str, Any]] = []

    def _structure_detail(self, structure: HydraulicStructure, cell_wse: Sequence[float]) -> Dict[str, Any]:
        m2ft = float(self._model_to_ft)
        inv_m2ft = 1.0 / m2ft if m2ft > 0 else 1.0
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
        g = max(1.0e-6, float(getattr(self.cfg, "gravity", 9.81)))

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
            inlet_invert = float(md.get("inlet_invert_elev", dz))
            outlet_invert = float(md.get("outlet_invert_elev", inlet_invert))
            sign = 1.0 if wu >= wd else -1.0
            upstream_wse = wu if sign >= 0.0 else wd
            downstream_wse = wd if sign >= 0.0 else wu
            upstream_invert = inlet_invert if sign >= 0.0 else outlet_invert
            downstream_invert = outlet_invert if sign >= 0.0 else inlet_invert

            # Convert model-unit values → feet for the culvert hydraulics routines
            available_head_up_ft = max(0.0, _model_to_ft(upstream_wse - upstream_invert, m2ft))
            tailwater_depth_ft = max(0.0, _model_to_ft(downstream_wse - downstream_invert, m2ft))
            length_ft = max(0.1, _model_to_ft(float(md.get("length", 1.0) or 1.0), m2ft))
            slope_ftft = float(md.get("culvert_slope", (upstream_invert - downstream_invert) / max(length_ft, 0.1) if length_ft > 0.0 else 0.0))
            slope_ftft = max(1.0e-6, abs(slope_ftft))
            roughness_n = max(1.0e-6, float(md.get("roughness_n", 0.013) or 0.013))
            _ent_raw = md.get("inlet_loss_k") if "inlet_loss_k" in md else md.get("entrance_loss_k")
            entrance_loss_k = float(_ent_raw) if _ent_raw is not None else 0.5
            _ext_raw = md.get("outlet_loss_k") if "outlet_loss_k" in md else md.get("exit_loss_k")
            exit_loss_k = float(_ext_raw) if _ext_raw is not None else 1.0
            detail.update(
                {
                    "sign": sign,
                    "inlet_invert_elev": inlet_invert,
                    "outlet_invert_elev": outlet_invert,
                    "available_head_up": available_head_up_ft * inv_m2ft,
                    "tailwater_depth": tailwater_depth_ft * inv_m2ft,
                    "culvert_slope": slope_ftft,
                    "entrance_loss_k": entrance_loss_k,
                    "exit_loss_k": exit_loss_k,
                }
            )

            # Cross-section with model-unit-aware conversion to feet
            xsect = _culvert_xsect_from_metadata(md, m2ft)

            # Inlet control — routines expect feet
            q_inlet_cfs, _dqh, _condition, _yr = inlet_controlled_flow(
                xsect,
                max(1.0e-6, slope_ftft),
                max(0.0, available_head_up_ft),
            )
            q_inlet = q_inlet_cfs / _u.USC_FT3_PER_SI_M3
            detail["inlet_control_flow"] = q_inlet

            # Orifice capacity
            diameter_model = float(md.get("diameter", md.get("culvert_rise", 0.0)) or 0.0)
            diameter_ft = _model_to_ft(diameter_model, m2ft)
            area_m2_input = float(md.get("culvert_area_m2", md.get("area_m2", 0.0)) or 0.0)
            shape = str(md.get("culvert_shape", "circular")).strip().lower()
            if area_m2_input <= 0.0 and diameter_ft > 0.0 and shape in ("circular", "pipe", "round"):
                area_m2_input = circular_area_from_diameter(diameter_ft * _u.SI_M_PER_USC_FT)
            q_orifice = 0.0
            if area_m2_input > 0.0:
                q_orifice = abs(
                    compute_orifice_flow(
                        head_up=available_head_up_ft * inv_m2ft,
                        head_down=tailwater_depth_ft * inv_m2ft,
                        area=area_m2_input,
                        discharge_coeff=float(md.get("cd", 0.75)),
                        g=g,
                        max_flow=float(max_q) if max_q is not None else None,
                    )
                )
            detail["orifice_cap"] = q_orifice

            # Manning capacity — use correct geometry based on shape
            q_manning_cap = 0.0
            is_rect = hasattr(xsect, 'width_ft') and xsect.width_ft > 0
            if is_rect:
                # Rectangular culvert: A = w*h, P = 2(w+h), R = A/P
                area_ft2 = xsect.width_ft * xsect.yFull
                perim_ft = 2.0 * (xsect.width_ft + xsect.yFull)
                if perim_ft > 0:
                    rh_ft = area_ft2 / perim_ft
                    q_manning_cap_cfs = (1.486 / roughness_n) * area_ft2 * (rh_ft ** (2.0 / 3.0)) * (slope_ftft ** 0.5)
                    q_manning_cap = q_manning_cap_cfs / _u.USC_FT3_PER_SI_M3
            elif hasattr(xsect, 'radius_ft') and xsect.radius_ft > 0:
                # Circular culvert — delegate to existing function
                q_manning_cap = compute_pipe_manning_capacity_full(
                    diameter_m=_ft_to_m(2.0 * xsect.radius_ft),
                    slope_m_per_m=slope_ftft,
                    roughness_n=roughness_n,
                )
            detail["manning_cap"] = q_manning_cap

            # Outlet control
            q_hint_cfs = max(q_inlet_cfs, _model3_to_cfs(max(q_orifice, q_manning_cap, 0.0), m2ft))
            q_outlet = _culvert_outlet_control_flow(
                xsect=xsect,
                available_head_up_ft=max(0.0, available_head_up_ft),
                tailwater_depth_ft=max(0.0, tailwater_depth_ft),
                length_ft=max(0.1, length_ft),
                slope_ftft=max(1.0e-6, slope_ftft),
                roughness_n=roughness_n,
                entrance_loss_k=entrance_loss_k,
                exit_loss_k=exit_loss_k,
                q_hint_cfs=q_hint_cfs,
            )
            detail["outlet_control_flow"] = q_outlet

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
                    upstream_wse=upstream_wse,
                    downstream_wse=downstream_wse,
                    crest_elev=float(md.get("embankment_crest_elev", md.get("road_crest_elev", dz)) or dz),
                    width=float(md.get("embankment_overflow_width", md.get("road_overflow_width", md.get("width", 1.0))) or 1.0),
                    coeff=float(md.get("embankment_weir_coeff", md.get("road_weir_coeff", 1.7)) or 1.7),
                ))
                q_culvert += q_emb
            detail["embankment_flow"] = q_emb

            barrels = max(1.0, float(md.get("culvert_barrels", 1.0) or 1.0))
            q_culvert *= barrels
            if max_q is not None:
                q_culvert = min(q_culvert, max(0.0, float(max_q)))
                if q_culvert >= max(0.0, float(max_q)) - 1.0e-12:
                    control_mode = "max_flow"
            detail["barrels"] = barrels
            detail["control_mode"] = control_mode
            detail["flow"] = q_culvert if sign >= 0.0 else -q_culvert
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
        return float(self._structure_detail(structure, cell_wse).get("flow", 0.0))

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
        net = self.compute_cell_source_terms(dt=dt, cell_wse=cell_wse)
        return convert_cell_flows_to_depth_rates(net, cell_area)

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
