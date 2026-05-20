from __future__ import annotations

import math
from typing import Callable, Dict, Iterable, Optional, Tuple

import numpy as np


def apply_bc_layer_overrides_qgis(
    *,
    mesh_data,
    have_qgis_core: bool,
    bc_lines_layer_combo,
    combo_layer_fn: Callable[[object, str], Optional[object]],
    edge_n0: np.ndarray,
    edge_n1: np.ndarray,
    bc_type: np.ndarray,
    bc_val: np.ndarray,
    qgs_geometry_cls,
    qgs_pointxy_cls,
    log_fn: Callable[[str], None],
) -> Tuple[np.ndarray, np.ndarray]:
    if mesh_data is None or not have_qgis_core:
        return bc_type, bc_val
    if bc_lines_layer_combo is None:
        return bc_type, bc_val

    bc_layer = combo_layer_fn(bc_lines_layer_combo, "vector")
    if bc_layer is None:
        return bc_type, bc_val

    fields = set(bc_layer.fields().names())
    type_field = None
    for cand in ("bc_type", "type", "bc"):
        if cand in fields:
            type_field = cand
            break
    val_field = None
    for cand in ("bc_value", "value", "bc_val"):
        if cand in fields:
            val_field = cand
            break
    prio_field = "priority" if "priority" in fields else None

    if type_field is None:
        log_fn("BC polyline layer selected but no bc_type/type field found; skipping overrides.")
        return bc_type, bc_val

    node_x = mesh_data["node_x"]
    node_y = mesh_data["node_y"]

    features = []
    for ft in bc_layer.getFeatures():
        geom = ft.geometry()
        if geom is None or geom.isEmpty():
            continue
        try:
            t = int(ft[type_field])
        except Exception:
            continue
        v = 0.0
        if val_field is not None:
            try:
                v = float(ft[val_field])
            except Exception:
                v = 0.0
        pr = 0
        if prio_field is not None:
            try:
                pr = int(ft[prio_field])
            except Exception:
                pr = 0
        features.append((pr, geom, t, v))

    if not features:
        return bc_type, bc_val

    features.sort(key=lambda x: x[0], reverse=True)
    applied = 0
    for i in range(edge_n0.size):
        x0 = float(node_x[edge_n0[i]])
        y0 = float(node_y[edge_n0[i]])
        x1 = float(node_x[edge_n1[i]])
        y1 = float(node_y[edge_n1[i]])
        tol = math.hypot(x1 - x0, y1 - y0) * 0.5
        mid = qgs_geometry_cls.fromPointXY(qgs_pointxy_cls(0.5 * (x0 + x1), 0.5 * (y0 + y1)))
        for _, g, t, v in features:
            if mid.distance(g) < tol:
                changed = (int(bc_type[i]) != int(t)) or (not np.isclose(float(bc_val[i]), float(v)))
                bc_type[i] = int(t)
                bc_val[i] = float(v)
                if changed:
                    applied += 1
                break

    if applied:
        log_fn(f"BC line static overrides applied to {applied}/{edge_n0.size} boundary edges from '{bc_layer.name()}'.")

    return bc_type, bc_val


def collect_bc_layer_hydrographs_qgis(
    *,
    mesh_data,
    have_qgis_core: bool,
    bc_lines_layer_combo,
    combo_layer_fn: Callable[[object, str], Optional[object]],
    iter_project_layers_fn: Callable[[], Iterable[object]],
    hydrograph_from_layer_fn: Callable[..., Optional[Tuple[np.ndarray, np.ndarray]]],
    parse_hydrograph_text_fn: Callable[[str], Optional[Tuple[np.ndarray, np.ndarray]]],
    edge_n0: np.ndarray,
    edge_n1: np.ndarray,
    ts_flow_code: int,
    ts_stage_code: int,
    qgs_vector_layer_cls,
    qgs_geometry_cls,
    qgs_pointxy_cls,
    log_fn: Callable[[str], None],
) -> Dict[int, Tuple[int, Tuple[np.ndarray, np.ndarray]]]:
    edge_hydro: Dict[int, Tuple[int, Tuple[np.ndarray, np.ndarray]]] = {}
    if mesh_data is None or not have_qgis_core:
        return edge_hydro
    if bc_lines_layer_combo is None:
        return edge_hydro

    bc_layer = combo_layer_fn(bc_lines_layer_combo, "vector")
    if bc_layer is None:
        return edge_hydro

    fields = set(bc_layer.fields().names())
    type_field = "bc_type" if "bc_type" in fields else ("type" if "type" in fields else None)
    if type_field is None:
        return edge_hydro
    prio_field = "priority" if "priority" in fields else None

    hydro_field = None
    for cand in ("hydrograph", "hydrograph_text", "hydro", "hg"):
        if cand in fields:
            hydro_field = cand
            break

    hgid_field = "hydrograph_id" if "hydrograph_id" in fields else None
    hlyr_field = "hydrograph_layer" if "hydrograph_layer" in fields else None

    hydro_lookup: Dict[str, str] = {}
    if hgid_field is not None:
        hydro_layers = [
            lyr
            for lyr in iter_project_layers_fn()
            if isinstance(lyr, qgs_vector_layer_cls) and str(lyr.name()).lower() in ("swe2d_hydrographs",)
        ]
        if hydro_layers:
            hlyr = hydro_layers[0]
            hfields = set(hlyr.fields().names())
            if "hydrograph_id" in hfields and "hydrograph" in hfields:
                for hft in hlyr.getFeatures():
                    hid = str(hft["hydrograph_id"] or "").strip()
                    htxt = str(hft["hydrograph"] or "").strip()
                    if hid and htxt:
                        hydro_lookup[hid] = htxt

    if hydro_field is None and hgid_field is None:
        return edge_hydro

    node_x = mesh_data["node_x"]
    node_y = mesh_data["node_y"]

    features = []
    for ft in bc_layer.getFeatures():
        geom = ft.geometry()
        if geom is None or geom.isEmpty():
            continue
        try:
            t = int(ft[type_field])
        except Exception:
            continue
        if t not in (int(ts_flow_code), int(ts_stage_code)):
            continue

        raw_h = str(ft[hydro_field] or "").strip() if hydro_field is not None else ""
        ref_layer = str(ft[hlyr_field] or "").strip() if hlyr_field is not None else ""

        if not raw_h and hgid_field is not None:
            hid = str(ft[hgid_field] or "").strip()
            if hid in hydro_lookup:
                raw_h = hydro_lookup[hid]

        if not raw_h and (ref_layer or (hydro_field is not None and str(ft[hydro_field] or "").strip())):
            layer_ref = ref_layer or str(ft[hydro_field] or "").strip()
            target_layer = None
            for lyr in iter_project_layers_fn():
                if not isinstance(lyr, qgs_vector_layer_cls):
                    continue
                try:
                    if lyr.id() == layer_ref or str(lyr.name()) == layer_ref:
                        target_layer = lyr
                        break
                except Exception:
                    continue
            if target_layer is not None:
                hid = str(ft[hgid_field] or "").strip() if hgid_field is not None else ""
                hg_layer = hydrograph_from_layer_fn(target_layer, hydrograph_id=hid, bc_type=t)
                if hg_layer is not None:
                    pr = 0
                    if prio_field is not None:
                        try:
                            pr = int(ft[prio_field])
                        except Exception:
                            pr = 0
                    features.append((pr, geom, t, hg_layer))
                    continue

        if not raw_h:
            continue

        try:
            hg = parse_hydrograph_text_fn(raw_h)
        except Exception:
            continue
        if hg is None:
            continue

        pr = 0
        if prio_field is not None:
            try:
                pr = int(ft[prio_field])
            except Exception:
                pr = 0
        features.append((pr, geom, t, hg))

    if not features:
        return edge_hydro

    features.sort(key=lambda x: x[0], reverse=True)
    for i in range(edge_n0.size):
        x0 = float(node_x[edge_n0[i]])
        y0 = float(node_y[edge_n0[i]])
        x1 = float(node_x[edge_n1[i]])
        y1 = float(node_y[edge_n1[i]])
        tol = math.hypot(x1 - x0, y1 - y0) * 0.5
        mid = qgs_geometry_cls.fromPointXY(qgs_pointxy_cls(0.5 * (x0 + x1), 0.5 * (y0 + y1)))
        for _pr, g, t, hg in features:
            if mid.distance(g) < tol:
                edge_hydro[i] = (t, hg)
                break

    if edge_hydro:
        log_fn(f"BC line hydrographs applied to {len(edge_hydro)} boundary edges.")
    return edge_hydro


def collect_bc_layer_edge_groups_qgis(
    *,
    mesh_data,
    have_qgis_core: bool,
    bc_lines_layer_combo,
    combo_layer_fn: Callable[[object, str], Optional[object]],
    edge_n0: np.ndarray,
    edge_n1: np.ndarray,
    qgs_geometry_cls,
    qgs_pointxy_cls,
) -> Dict[int, str]:
    edge_groups: Dict[int, str] = {}
    if mesh_data is None or not have_qgis_core:
        return edge_groups
    if bc_lines_layer_combo is None:
        return edge_groups

    bc_layer = combo_layer_fn(bc_lines_layer_combo, "vector")
    if bc_layer is None:
        return edge_groups

    fields = set(bc_layer.fields().names())
    name_field = "name" if "name" in fields else None
    prio_field = "priority" if "priority" in fields else None

    node_x = mesh_data["node_x"]
    node_y = mesh_data["node_y"]

    features = []
    for ft in bc_layer.getFeatures():
        geom = ft.geometry()
        if geom is None or geom.isEmpty():
            continue
        pr = 0
        if prio_field is not None:
            try:
                pr = int(ft[prio_field])
            except Exception:
                pr = 0
        nm = ""
        if name_field is not None:
            try:
                nm = str(ft[name_field] or "").strip()
            except Exception:
                nm = ""
        if not nm:
            try:
                nm = f"feature_{int(ft.id())}"
            except Exception:
                nm = "feature"
        features.append((pr, geom, nm))

    if not features:
        return edge_groups

    features.sort(key=lambda x: x[0], reverse=True)
    for i in range(edge_n0.size):
        x0 = float(node_x[edge_n0[i]])
        y0 = float(node_y[edge_n0[i]])
        x1 = float(node_x[edge_n1[i]])
        y1 = float(node_y[edge_n1[i]])
        tol = math.hypot(x1 - x0, y1 - y0) * 0.5
        mid = qgs_geometry_cls.fromPointXY(qgs_pointxy_cls(0.5 * (x0 + x1), 0.5 * (y0 + y1)))
        for _pr, g, nm in features:
            if mid.distance(g) < tol:
                edge_groups[i] = f"bc_line:{nm}"
                break

    return edge_groups
