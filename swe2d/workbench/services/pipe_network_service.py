from __future__ import annotations

"""Pipe-network service: node/link geometry, topology validation, and SWMM import."""

import math
from typing import Callable, Dict, List, Optional

import numpy as np

from swe2d.extensions.extension_models import (
    DrainageLink,
    DrainageNode,
    InletExchange,
    InletType,
    NodeInletAssignment,
    OutfallExchange,
    PipeEndExchange,
    PipeNetworkConfig,
)


def _opt_float(value, fallback=None, log_fn=None):
    """opt float."""
    if value in (None, ""):
        return fallback
    try:
        return float(value)
    except Exception as e:
        if log_fn is not None:
            log_fn(f"[ERROR] opt_float conversion: {e}")
        return fallback


def _opt_bool(value, fallback=False, log_fn=None):
    """opt bool."""
    if value in (None, ""):
        return fallback
    if value is True:
        return True
    if value is False:
        return False
    sval = str(value).strip().lower()
    if sval in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if sval in {"0", "false", "f", "no", "n", "off"}:
        return False
    try:
        return float(value) != 0.0
    except Exception as e:
        if log_fn is not None:
            log_fn(f"[ERROR] opt_bool float conversion: {e}")
        return fallback


def _ellipse_perimeter(a: float, b: float) -> float:
    """ellipse perimeter."""
    if a <= 0.0 or b <= 0.0:
        return 0.0
    return math.pi * (
        3.0 * (a + b) - math.sqrt(max(0.0, (3.0 * a + b) * (a + 3.0 * b)))
    )


def build_pipe_network_config(
    *,
    mesh_data: Optional[dict],
    node_layer=None,
    link_layer=None,
    inlet_layer=None,
    node_inlet_layer=None,
    cell_min_bed: Optional[np.ndarray] = None,
    nearest_cell_fn: Optional[Callable[[float, float], int]] = None,
    gravity: float = 9.81,
    config: Optional[dict] = None,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Optional[PipeNetworkConfig]:
    """Build a PipeNetworkConfig from resolved layer data.

    All widget access, combo resolution, and spatial lookups are handled by the caller.
    This function contains zero Qt imports and zero widget access.

    Parameters
    ----------
    mesh_data : dict or None
        Mesh data dictionary with node_x, node_y, node_z keys.
    node_layer : QgsVectorLayer or None
        Node features layer.
    link_layer : QgsVectorLayer or None
        Link features layer.
    inlet_layer : QgsVectorLayer or None
        Inlet features layer.
    node_inlet_layer : QgsVectorLayer or None
        Node-inlet assignment layer.
    cell_min_bed : np.ndarray or None
        Minimum bed elevation per cell, pre-computed.
    nearest_cell_fn : callable or None
        Function (x, y) -> int returning nearest cell index.
    gravity : float
        Gravity value.
    config : dict or None
        Drainage coupling config from widgets. Keys:
        solver_mode, coupling_substeps, max_coupling_substeps,
        gpu_method, head_deadband, dynamic_relaxation,
        adaptive_depth_fraction, adaptive_wave_courant,
        implicit_iters, implicit_relax.
    log_fn : callable or None
        Logging function.
    """
    _log = log_fn or (lambda msg: None)

    if mesh_data is None or node_layer is None or link_layer is None:
        return None

    if cell_min_bed is None:
        _log("[ERROR] build_pipe_network_config: cell_min_bed is None")
        return None
    if nearest_cell_fn is None:
        _log("[ERROR] build_pipe_network_config: nearest_cell_fn is None")
        return None
    if config is None:
        config = {}

    node_fields = set(node_layer.fields().names())
    nodes: List[DrainageNode] = []
    node_by_id: Dict[str, DrainageNode] = {}
    node_cell_by_id: Dict[str, int] = {}
    node_zero_storage_by_id: Dict[str, bool] = {}

    for ft in node_layer.getFeatures():
        geom = ft.geometry()
        if geom is None or geom.isEmpty():
            continue
        try:
            pt = geom.asPoint()
        except Exception as e:
            _log(f"[ERROR] node feature geometry asPoint: {e}")
            continue
        node_id = str(
            ft["node_id"] if "node_id" in node_fields else ft.id()
        ).strip()
        if not node_id:
            continue
        x = float(pt.x())
        y = float(pt.y())
        invert = _opt_float(
            ft["invert_elev"] if "invert_elev" in node_fields else None,
            0.0,
            log_fn=_log,
        )
        node_type = str(
            ft["node_type"] if "node_type" in node_fields else "junction"
        ).strip().lower() or "junction"
        ci = nearest_cell_fn(x, y)
        bed_here = (
            float(cell_min_bed[ci])
            if ci >= 0 and ci < int(cell_min_bed.size)
            else invert
        )
        rim = _opt_float(
            ft["rim_elev"] if "rim_elev" in node_fields else None,
            None,
            log_fn=_log,
        )
        if rim is None:
            rim = max(invert, bed_here)
        max_depth = _opt_float(
            ft["max_depth"] if "max_depth" in node_fields else None,
            None,
            log_fn=_log,
        )
        if max_depth is None:
            if node_type == "outfall":
                max_depth = 10.0
            else:
                max_depth = max(0.1, float(rim) - float(invert))
        crest = _opt_float(
            ft["crest_elev"] if "crest_elev" in node_fields else None,
            None,
            log_fn=_log,
        )
        if crest is None:
            crest = float(invert if node_type == "outfall" else rim)

        node = DrainageNode(
            node_id=node_id,
            x=x,
            y=y,
            invert_elev=float(invert),
            max_depth=float(max_depth),
            crest_elev=float(crest),
            rim_elev=float(rim),
            node_type=node_type,
            metadata={
                "surface_area": float(
                    ft["surface_area"]
                    if "surface_area" in node_fields
                    and ft["surface_area"] not in (None, "")
                    else 50.0
                ),
                "outfall_area_m2": float(
                    ft["outfall_area"]
                    if "outfall_area" in node_fields
                    and ft["outfall_area"] not in (None, "")
                    else 0.0
                ),
            },
        )
        nodes.append(node)
        node_by_id[node_id] = node
        node_cell_by_id[node_id] = int(ci)
        node_zero_storage_by_id[node_id] = _opt_bool(
            ft["zero_storage"] if "zero_storage" in node_fields else None,
            False,
            log_fn=_log,
        )
    if not nodes:
        return None

    link_fields = set(link_layer.fields().names())
    links: List[DrainageLink] = []
    links_missing_capacity: List[str] = []

    for ft in link_layer.getFeatures():
        geom = ft.geometry()
        if geom is None or geom.isEmpty():
            continue
        link_id = str(
            ft["link_id"] if "link_id" in link_fields else ft.id()
        ).strip()
        from_node = str(
            ft["from_node"] if "from_node" in link_fields else ""
        ).strip()
        to_node = str(
            ft["to_node"] if "to_node" in link_fields else ""
        ).strip()
        if not link_id or not from_node or not to_node:
            continue

        link_shape = str(
            ft["link_shape"] if "link_shape" in link_fields else ""
        ).strip().lower()
        if link_shape in ("", "none", "null"):
            link_shape = "circular"

        diameter_val = None
        for nm in ("diameter", "diameter_m", "equiv_diameter", "equiv_diameter_m"):
            if nm in link_fields and ft[nm] not in (None, ""):
                try:
                    d_try = float(ft[nm])
                    if d_try > 0.0:
                        diameter_val = d_try
                        break
                except Exception as e:
                    _log(f"[ERROR] link diameter float conversion: {e}")

        area_val = None
        for nm in ("area_m2", "area", "cross_area"):
            if nm in link_fields and ft[nm] not in (None, ""):
                try:
                    a_try = float(ft[nm])
                    if a_try > 0.0:
                        area_val = a_try
                        break
                except Exception as e:
                    _log(f"[ERROR] link area float conversion: {e}")

        span_val = None
        for nm in ("span", "span_m", "width", "width_m"):
            if nm in link_fields and ft[nm] not in (None, ""):
                try:
                    s_try = float(ft[nm])
                    if s_try > 0.0:
                        span_val = s_try
                        break
                except Exception as e:
                    _log(f"[ERROR] link span float conversion: {e}")

        rise_val = None
        for nm in ("rise", "rise_m", "height", "height_m"):
            if nm in link_fields and ft[nm] not in (None, ""):
                try:
                    r_try = float(ft[nm])
                    if r_try > 0.0:
                        rise_val = r_try
                        break
                except Exception as e:
                    _log(f"[ERROR] link rise float conversion: {e}")

        equiv_d_val = None
        for nm in ("equiv_diameter_m", "equiv_diameter"):
            if nm in link_fields and ft[nm] not in (None, ""):
                try:
                    eq_try = float(ft[nm])
                    if eq_try > 0.0:
                        equiv_d_val = eq_try
                        break
                except Exception as e:
                    _log(f"[ERROR] link equiv_diameter float conversion: {e}")

        if area_val is None or area_val <= 0.0:
            if (
                link_shape == "circular"
                and diameter_val is not None
                and diameter_val > 0.0
            ):
                area_val = 0.25 * math.pi * float(diameter_val) * float(diameter_val)
            elif link_shape in ("box", "rectangular", "rect") and span_val is not None and rise_val is not None:
                area_val = float(span_val) * float(rise_val)
            elif link_shape == "pipe_arch" and span_val is not None and rise_val is not None:
                area_val = 0.25 * math.pi * float(span_val) * float(rise_val)

        if equiv_d_val is None or equiv_d_val <= 0.0:
            if diameter_val is not None and diameter_val > 0.0:
                equiv_d_val = float(diameter_val)
            elif area_val is not None and area_val > 0.0:
                if link_shape in ("box", "rectangular", "rect") and span_val is not None and rise_val is not None:
                    perim = 2.0 * (float(span_val) + float(rise_val))
                    if perim > 0.0:
                        equiv_d_val = 4.0 * float(area_val) / perim
                elif link_shape == "pipe_arch" and span_val is not None and rise_val is not None:
                    perim = _ellipse_perimeter(
                        0.5 * float(span_val), 0.5 * float(rise_val)
                    )
                    if perim > 0.0:
                        equiv_d_val = 4.0 * float(area_val) / perim
                if equiv_d_val is None or equiv_d_val <= 0.0:
                    equiv_d_val = math.sqrt(4.0 * float(area_val) / math.pi)

        if (
            diameter_val is None or diameter_val <= 0.0
        ) and equiv_d_val is not None and equiv_d_val > 0.0:
            diameter_val = float(equiv_d_val)

        if (
            (diameter_val is None or diameter_val <= 0.0)
            and (area_val is None or area_val <= 0.0)
            and (equiv_d_val is None or equiv_d_val <= 0.0)
        ):
            links_missing_capacity.append(link_id)

        link_type_raw = str(
            ft["link_type"] if "link_type" in link_fields else "conduit"
        ).strip() or "conduit"
        is_culvert = link_type_raw.lower() == "culvert"

        culvert_shape_val = None
        if is_culvert:
            raw_shape = str(ft.get("culvert_shape", "") or "").strip().lower()
            if raw_shape in ("", "none", "null"):
                raw_shape = "circular"
            culvert_shape_val = raw_shape

        culvert_code_val = 1
        if is_culvert and "culvert_code" in link_fields:
            try:
                culvert_code_val = int(round(float(ft["culvert_code"])))
            except Exception as e:
                _log(f"[ERROR] culvert_code conversion: {e}")

        culvert_rise_val = None
        if is_culvert and "culvert_rise" in link_fields:
            try:
                rv = float(ft["culvert_rise"])
                if rv > 0.0:
                    culvert_rise_val = rv
            except Exception as e:
                _log(f"[ERROR] culvert_rise float conversion: {e}")

        culvert_span_val = None
        if is_culvert and "culvert_span" in link_fields:
            try:
                sv = float(ft["culvert_span"])
                if sv > 0.0:
                    culvert_span_val = sv
            except Exception as e:
                _log(f"[ERROR] culvert_span float conversion: {e}")

        inlet_invert_val = None
        if is_culvert and "inlet_invert_elev" in link_fields:
            try:
                inlet_invert_val = float(ft["inlet_invert_elev"])
            except Exception as e:
                _log(f"[ERROR] inlet_invert_elev float conversion: {e}")

        outlet_invert_val = None
        if is_culvert and "outlet_invert_elev" in link_fields:
            try:
                outlet_invert_val = float(ft["outlet_invert_elev"])
            except Exception as e:
                _log(f"[ERROR] outlet_invert_elev float conversion: {e}")

        entrance_loss_val = 0.5
        if is_culvert:
            for cand in ("entrance_loss_k", "inlet_loss_k", "entry_loss_k"):
                if cand in link_fields and ft[cand] not in (None, ""):
                    try:
                        entrance_loss_val = float(ft[cand])
                        break
                    except Exception as e:
                        _log(f"[ERROR] entrance_loss_k float conversion: {e}")

        exit_loss_val = 1.0
        if is_culvert:
            for cand in ("exit_loss_k", "outlet_loss_k"):
                if cand in link_fields and ft[cand] not in (None, ""):
                    try:
                        exit_loss_val = float(ft[cand])
                        break
                    except Exception as e:
                        _log(f"[ERROR] exit_loss_k float conversion: {e}")

        barrel_count_val = 1
        if is_culvert and "culvert_barrels" in link_fields:
            try:
                bc = int(round(float(ft["culvert_barrels"])))
                if bc >= 1:
                    barrel_count_val = bc
            except Exception as e:
                _log(f"[ERROR] culvert_barrels int conversion: {e}")

        links.append(
            DrainageLink(
                link_id=link_id,
                from_node_id=from_node,
                to_node_id=to_node,
                link_type=link_type_raw,
                length=(
                    float(ft["length"])
                    if "length" in link_fields and ft["length"] not in (None, "")
                    else float(geom.length())
                ),
                roughness_n=float(
                    ft["roughness_n"]
                    if "roughness_n" in link_fields
                    and ft["roughness_n"] not in (None, "")
                    else 0.013
                ),
                diameter=diameter_val,
                max_flow=(
                    float(ft["max_flow"])
                    if "max_flow" in link_fields and ft["max_flow"] not in (None, "")
                    else None
                ),
                culvert_shape=culvert_shape_val,
                culvert_code=culvert_code_val,
                culvert_rise=culvert_rise_val,
                culvert_span=culvert_span_val,
                inlet_invert_elev=inlet_invert_val,
                outlet_invert_elev=outlet_invert_val,
                entrance_loss_k=entrance_loss_val,
                exit_loss_k=exit_loss_val,
                barrel_count=barrel_count_val,
                cd=float(
                    ft["cd"]
                    if "cd" in link_fields and ft["cd"] not in (None, "")
                    else 0.75
                ),
                metadata={
                    "area_m2": float(area_val) if area_val is not None else 0.0,
                    "equiv_diameter_m": (
                        float(equiv_d_val) if equiv_d_val is not None else 0.0
                    ),
                    "cd": float(
                        ft["cd"]
                        if "cd" in link_fields and ft["cd"] not in (None, "")
                        else 0.75
                    ),
                    "entry_loss_k": float(
                        ft["entry_loss_k"]
                        if "entry_loss_k" in link_fields
                        and ft["entry_loss_k"] not in (None, "")
                        else 0.5
                    ),
                    "exit_loss_k": float(
                        ft["exit_loss_k"]
                        if "exit_loss_k" in link_fields
                        and ft["exit_loss_k"] not in (None, "")
                        else 1.0
                    ),
                    "pipe_end_inlet_loss_k": float(
                        ft["pipe_end_inlet_loss_k"]
                        if "pipe_end_inlet_loss_k" in link_fields
                        and ft["pipe_end_inlet_loss_k"] not in (None, "")
                        else (
                            ft["inlet_loss_k"]
                            if "inlet_loss_k" in link_fields
                            and ft["inlet_loss_k"] not in (None, "")
                            else 0.5
                        )
                    ),
                    "pipe_end_outlet_loss_k": float(
                        ft["pipe_end_outlet_loss_k"]
                        if "pipe_end_outlet_loss_k" in link_fields
                        and ft["pipe_end_outlet_loss_k"] not in (None, "")
                        else (
                            ft["outlet_loss_k"]
                            if "outlet_loss_k" in link_fields
                            and ft["outlet_loss_k"] not in (None, "")
                            else 1.0
                        )
                    ),
                    "link_shape": link_shape,
                    "span_m": float(span_val) if span_val is not None else 0.0,
                    "rise_m": float(rise_val) if rise_val is not None else 0.0,
                    "culvert_shape": culvert_shape_val or "circular",
                    "culvert_code": float(culvert_code_val),
                    "culvert_rise": (
                        float(culvert_rise_val) if culvert_rise_val is not None else 0.0
                    ),
                    "culvert_span": (
                        float(culvert_span_val) if culvert_span_val is not None else 0.0
                    ),
                    "inlet_invert_elev": (
                        float(inlet_invert_val) if inlet_invert_val is not None else 0.0
                    ),
                    "outlet_invert_elev": (
                        float(outlet_invert_val) if outlet_invert_val is not None else 0.0
                    ),
                    "entrance_loss_k": entrance_loss_val,
                    "exit_loss_k": exit_loss_val,
                    "culvert_barrels": float(barrel_count_val),
                },
            )
        )
    if not links:
        return None

    if links_missing_capacity:
        preview = ", ".join(links_missing_capacity[:8])
        suffix = "" if len(links_missing_capacity) <= 8 else f", ... (+{len(links_missing_capacity) - 8} more)"
        _log(
            "Drainage warning: link(s) missing hydraulic geometry (diameter/area/equiv_diameter/shape dimensions); "
            "link flow will stay zero for these IDs: "
            f"{preview}{suffix}"
        )

    inlets: List[InletExchange] = []
    inlet_types: List[InletType] = []
    node_inlets: List[NodeInletAssignment] = []
    inlet_types_by_id: Dict[str, InletType] = {}

    if inlet_layer is not None:
        inlet_fields = set(inlet_layer.fields().names())
        has_new_inlet_schema = "inlet_type_id" in inlet_fields
        if has_new_inlet_schema:
            for ft in inlet_layer.getFeatures():
                inlet_type_id = str(
                    ft["inlet_type_id"] if "inlet_type_id" in inlet_fields else ""
                ).strip()
                if not inlet_type_id:
                    continue
                inlet_type = InletType(
                    inlet_type_id=inlet_type_id,
                    name=str(
                        ft["name"]
                        if "name" in inlet_fields and ft["name"] not in (None, "")
                        else inlet_type_id
                    ),
                    length=float(
                        ft["weir_length"]
                        if "weir_length" in inlet_fields
                        and ft["weir_length"] not in (None, "")
                        else 1.0
                    ),
                    area=float(
                        ft["orifice_area"]
                        if "orifice_area" in inlet_fields
                        and ft["orifice_area"] not in (None, "")
                        else 0.0
                    ),
                    coeff_weir=float(
                        ft["coeff_weir"]
                        if "coeff_weir" in inlet_fields
                        and ft["coeff_weir"] not in (None, "")
                        else 1.70
                    ),
                    coeff_orifice=float(
                        ft["coeff_orifice"]
                        if "coeff_orifice" in inlet_fields
                        and ft["coeff_orifice"] not in (None, "")
                        else 0.62
                    ),
                    max_capture=(
                        float(ft["max_capture"])
                        if "max_capture" in inlet_fields
                        and ft["max_capture"] not in (None, "")
                        else None
                    ),
                )
                inlet_types.append(inlet_type)
                inlet_types_by_id[inlet_type_id] = inlet_type

            if node_inlet_layer is not None:
                assign_fields = set(node_inlet_layer.fields().names())
                for ft in node_inlet_layer.getFeatures():
                    node_id = str(
                        ft["node_id"] if "node_id" in assign_fields else ""
                    ).strip()
                    inlet_type_id = str(
                        ft["inlet_type_id"] if "inlet_type_id" in assign_fields else ""
                    ).strip()
                    if not node_id or not inlet_type_id:
                        continue
                    node_inlets.append(
                        NodeInletAssignment(
                            node_id=node_id,
                            inlet_type_id=inlet_type_id,
                            multiplier=float(
                                ft["inlet_count"]
                                if "inlet_count" in assign_fields
                                and ft["inlet_count"] not in (None, "")
                                else 1.0
                            ),
                            crest_offset=float(
                                ft["crest_offset"]
                                if "crest_offset" in assign_fields
                                and ft["crest_offset"] not in (None, "")
                                else 0.0
                            ),
                        )
                    )

            for a in node_inlets:
                if a.node_id not in node_by_id:
                    continue
                it = inlet_types_by_id.get(a.inlet_type_id)
                if it is None:
                    continue
                node = node_by_id[a.node_id]
                crest = float(
                    (node.crest_elev if node.crest_elev is not None else node.invert_elev)
                    + a.crest_offset
                )
                inlets.append(
                    InletExchange(
                        inlet_id=f"{a.node_id}:{a.inlet_type_id}",
                        cell_id=int(
                            node_cell_by_id.get(
                                a.node_id, nearest_cell_fn(node.x, node.y)
                            )
                        ),
                        node_id=a.node_id,
                        crest_elev=crest,
                        length=max(0.0, float(it.length)) * max(0.0, float(a.multiplier)),
                        area=max(0.0, float(it.area)) * max(0.0, float(a.multiplier)),
                        coeff_weir=max(0.0, float(it.coeff_weir)),
                        coeff_orifice=max(0.0, float(it.coeff_orifice)),
                        max_capture=it.max_capture,
                    )
                )
        else:
            for ft in inlet_layer.getFeatures():
                geom = ft.geometry()
                if geom is None or geom.isEmpty():
                    continue
                try:
                    pt = geom.asPoint()
                except Exception as e:
                    _log(f"[ERROR] inlet geometry asPoint: {e}")
                    try:
                        c = geom.centroid()
                        pt = (
                            c.asPoint()
                            if c is not None and not c.isEmpty()
                            else None
                        )
                    except Exception as e2:
                        _log(f"[ERROR] inlet centroid asPoint: {e2}")
                        pt = None
                if pt is None:
                    continue
                node_id = str(
                    ft["node_id"] if "node_id" in inlet_fields else ""
                ).strip()
                if not node_id:
                    continue
                inlets.append(
                    InletExchange(
                        inlet_id=str(
                            ft["inlet_id"] if "inlet_id" in inlet_fields else ft.id()
                        ).strip(),
                        cell_id=nearest_cell_fn(float(pt.x()), float(pt.y())),
                        node_id=node_id,
                        crest_elev=float(
                            ft["crest_elev"]
                            if "crest_elev" in inlet_fields
                            and ft["crest_elev"] not in (None, "")
                            else 0.0
                        ),
                        length=float(
                            ft["width"]
                            if "width" in inlet_fields and ft["width"] not in (None, "")
                            else 1.0
                        ),
                        area=float(
                            ft["area"]
                            if "area" in inlet_fields and ft["area"] not in (None, "")
                            else 0.0
                        ),
                        coeff_weir=float(
                            ft["coeff_weir"]
                            if "coeff_weir" in inlet_fields
                            and ft["coeff_weir"] not in (None, "")
                            else 1.70
                        ),
                        coeff_orifice=float(
                            ft["coefficient"]
                            if "coefficient" in inlet_fields
                            and ft["coefficient"] not in (None, "")
                            else 0.62
                        ),
                        max_capture=(
                            float(ft["max_capture"])
                            if "max_capture" in inlet_fields
                            and ft["max_capture"] not in (None, "")
                            else None
                        ),
                    )
                )

    outfalls: List[OutfallExchange] = []
    _node_connected_area: dict = {}
    _node_connected_diameter: dict = {}
    for lnk in links:
        area_lnk = float(lnk.metadata.get("area_m2", 0.0) or 0.0)
        d_lnk = float(lnk.diameter or 0.0)
        if d_lnk <= 0.0:
            d_lnk = float(lnk.metadata.get("equiv_diameter_m", 0.0) or 0.0)
        if d_lnk <= 0.0:
            if area_lnk > 0.0:
                d_lnk = math.sqrt(4.0 * area_lnk / math.pi)
        if area_lnk <= 0.0 and d_lnk > 0.0:
            area_lnk = 0.25 * math.pi * d_lnk * d_lnk
        for nid in (lnk.from_node_id, lnk.to_node_id):
            cur_a = float(_node_connected_area.get(nid, 0.0))
            if area_lnk > cur_a:
                _node_connected_area[nid] = area_lnk
            cur = float(_node_connected_diameter.get(nid, 0.0))
            if d_lnk > cur:
                _node_connected_diameter[nid] = d_lnk

    outfalls_missing_capacity: List[str] = []
    for node in nodes:
        if str(node.node_type).strip().lower() != "outfall":
            continue
        cell_id = nearest_cell_fn(float(node.x), float(node.y))
        area_outfall = max(
            0.0, float(node.metadata.get("outfall_area_m2", 0.0) or 0.0)
        )
        if area_outfall <= 0.0:
            area_outfall = max(
                0.0, float(_node_connected_area.get(node.node_id, 0.0) or 0.0)
            )
        diameter = float(_node_connected_diameter.get(node.node_id, 0.0) or 0.0)
        if diameter <= 0.0 and area_outfall > 0.0:
            diameter = math.sqrt(4.0 * area_outfall / math.pi)
        if area_outfall <= 0.0 and diameter <= 0.0:
            outfalls_missing_capacity.append(str(node.node_id))
        outfalls.append(
            OutfallExchange(
                outfall_id=node.node_id,
                cell_id=cell_id,
                node_id=node.node_id,
                invert_elev=float(node.invert_elev),
                area_m2=area_outfall,
                diameter=diameter,
                coefficient=0.82,
                max_flow=None,
                zero_storage=bool(
                    node_zero_storage_by_id.get(node.node_id, False)
                ),
            )
        )
    if outfalls_missing_capacity:
        preview = ", ".join(outfalls_missing_capacity[:8])
        suffix = "" if len(outfalls_missing_capacity) <= 8 else f", ... (+{len(outfalls_missing_capacity) - 8} more)"
        _log(
            "Drainage warning: outfall node(s) missing outfall_area and connected link capacity; "
            f"outfall exchange will stay zero for IDs: {preview}{suffix}"
        )

    pipe_ends: List[PipeEndExchange] = []
    pipe_end_link_types = {
        "pipe_end", "pipe-end", "daylighted_pipe", "daylighted", "daylight_pipe",
        "culvert",
    }
    pipe_end_nodes = {
        str(n.node_id)
        for n in nodes
        if str(n.node_type).strip().lower() == "pipe_end"
    }
    assigned_pipe_end_nodes: set = set()
    for lnk in links:
        ltype = str(lnk.link_type or "").strip().lower()
        if (
            ltype not in pipe_end_link_types
            and str(lnk.from_node_id) not in pipe_end_nodes
            and str(lnk.to_node_id) not in pipe_end_nodes
        ):
            continue

        for nid in (str(lnk.from_node_id), str(lnk.to_node_id)):
            if nid in assigned_pipe_end_nodes:
                continue
            node = node_by_id.get(nid)
            if node is None:
                continue
            if str(node.node_type).strip().lower() != "pipe_end":
                continue

            cell_id = int(
                node_cell_by_id.get(
                    nid, nearest_cell_fn(float(node.x), float(node.y))
                )
            )
            diameter = float(
                lnk.diameter or lnk.metadata.get("equiv_diameter_m", 0.0) or 0.0
            )
            area_pipe = max(0.0, float(lnk.metadata.get("area_m2", 0.0) or 0.0))
            if area_pipe <= 0.0 and diameter > 0.0:
                area_pipe = 0.25 * math.pi * diameter * diameter

            pipe_ends.append(
                PipeEndExchange(
                    pipe_end_id=f"pipe_end:{nid}",
                    cell_id=cell_id,
                    node_id=nid,
                    invert_elev=float(node.invert_elev),
                    diameter=diameter,
                    area_m2=area_pipe,
                    coefficient=float(lnk.metadata.get("cd", 0.82) or 0.82),
                    max_flow=lnk.max_flow,
                    inlet_loss_k=float(
                        lnk.metadata.get(
                            "pipe_end_inlet_loss_k",
                            lnk.metadata.get("entry_loss_k", 0.5),
                        )
                        or 0.5
                    ),
                    outlet_loss_k=float(
                        lnk.metadata.get(
                            "pipe_end_outlet_loss_k",
                            lnk.metadata.get("exit_loss_k", 1.0),
                        )
                        or 1.0
                    ),
                )
            )
            assigned_pipe_end_nodes.add(nid)

    solver_mode = int(config.get("solver_mode", 0))
    solver_mode_name = str(config.get("solver_mode_name", ""))
    coupling_substeps = int(config.get("coupling_substeps", 1))
    max_coupling_substeps = int(config.get("max_coupling_substeps", 1))
    gpu_method = str(config.get("gpu_method", "auto"))
    head_deadband = float(config.get("head_deadband", 0.001))
    dynamic_relaxation = float(config.get("dynamic_relaxation", 0.7))
    adaptive_depth_fraction = float(config.get("adaptive_depth_fraction", 0.5))
    adaptive_wave_courant = float(config.get("adaptive_wave_courant", 0.45))
    implicit_iters = int(config.get("implicit_iters", 3))
    implicit_relax = float(config.get("implicit_relax", 0.8))

    pipe_solver_mode = "diffusion_wave" if solver_mode != 2 else "fully_dynamic"

    _log(
        f"Drainage coupling configured: nodes={len(nodes)}, links={len(links)}, "
        f"inlets={len(inlets)}, inlet_types={len(inlet_types)}, node_inlets={len(node_inlets)}, "
        f"outfalls={len(outfalls)}, pipe_ends={len(pipe_ends)}, gravity={gravity:.3f}, mode={solver_mode_name}, "
        f"substeps={coupling_substeps}, max_substeps={max_coupling_substeps}, "
        f"gpu_method={gpu_method}, deadband={head_deadband:.4g}, relax={dynamic_relaxation:.3f}"
    )
    return PipeNetworkConfig(
        enabled=True,
        nodes=nodes,
        links=links,
        inlet_types=inlet_types,
        node_inlets=node_inlets,
        inlets=inlets,
        outfalls=outfalls,
        pipe_ends=pipe_ends,
        gravity=gravity,
        pipe_solver_mode=pipe_solver_mode,
        coupling_substeps=coupling_substeps,
        max_coupling_substeps=max_coupling_substeps,
        head_deadband_m=head_deadband,
        dynamic_flow_relaxation=dynamic_relaxation,
        adaptive_depth_fraction=adaptive_depth_fraction,
        adaptive_wave_courant=adaptive_wave_courant,
        implicit_coupling_iterations=implicit_iters,
        implicit_coupling_relaxation=implicit_relax,
    )
