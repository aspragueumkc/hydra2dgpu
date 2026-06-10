from __future__ import annotations

import logging
import math
from typing import Callable, Dict, Iterable, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _edge_midpoint_distance_to_feature(
    *,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    feature_geom,
    qgs_geometry_cls,
    qgs_pointxy_cls,
) -> float:
    mid = qgs_geometry_cls.fromPointXY(qgs_pointxy_cls(0.5 * (x0 + x1), 0.5 * (y0 + y1)))
    return float(mid.distance(feature_geom))


def _edge_matches_feature(
    *,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    feature_geom,
    qgs_geometry_cls,
    qgs_pointxy_cls,
) -> bool:
    edge_len = float(math.hypot(x1 - x0, y1 - y0))
    dist = _edge_midpoint_distance_to_feature(
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        feature_geom=feature_geom,
        qgs_geometry_cls=qgs_geometry_cls,
        qgs_pointxy_cls=qgs_pointxy_cls,
    )

    # Tight midpoint criterion to keep assignments local to the intended edge set.
    tol_local = max(1.0e-9, 0.10 * edge_len)
    if dist <= tol_local:
        return True

    try:
        edge_geom = qgs_geometry_cls.fromPolylineXY([
            qgs_pointxy_cls(x0, y0),
            qgs_pointxy_cls(x1, y1),
        ])
        if bool(edge_geom.intersects(feature_geom)):
            # Intersection-only can include endpoint touches at corners; require
            # the midpoint to still be reasonably close before accepting.
            return bool(dist <= max(1.0e-9, 0.25 * edge_len))
    except Exception as e:
        logger.warning("_edge_matches_feature: %s", e, exc_info=True)
        pass

    return False


def _edge_intersects_feature(
    *,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    feature_geom,
    qgs_geometry_cls,
    qgs_pointxy_cls,
) -> bool:
    try:
        edge_geom = qgs_geometry_cls.fromPolylineXY([
            qgs_pointxy_cls(x0, y0),
            qgs_pointxy_cls(x1, y1),
        ])
        return bool(edge_geom.intersects(feature_geom))
    except Exception as e:
        logger.warning("_edge_intersects_feature: %s", e, exc_info=True)
        return False


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
            raw_t = ft[type_field]
            if raw_t is None or (isinstance(raw_t, str) and raw_t.strip().upper() == 'NULL'):
                continue
            t = int(raw_t)
        except Exception as e:
            log_fn(f"[ERROR] BC type field parse failed: {e}")
            continue
        v = 0.0
        if val_field is not None:
            try:
                raw_val = ft[val_field]
                if raw_val is None or (isinstance(raw_val, str) and raw_val.strip().upper() == 'NULL'):
                    v = 0.0
                else:
                    v = float(str(raw_val))
            except Exception as e:
                log_fn(f"[ERROR] BC value field parse failed: {e}")
                v = 0.0
        pr = 0
        if prio_field is not None:
            try:
                raw_pr = ft[prio_field]
                if raw_pr is None or (isinstance(raw_pr, str) and raw_pr.strip().upper() == 'NULL'):
                    pr = 0
                else:
                    pr = int(raw_pr)
            except Exception as e:
                log_fn(f"[ERROR] BC priority field parse failed: {e}")
                pr = 0
        features.append((pr, geom, t, v))

    if not features:
        return bc_type, bc_val

    applied = 0
    for i in range(edge_n0.size):
        x0 = float(node_x[edge_n0[i]])
        y0 = float(node_y[edge_n0[i]])
        x1 = float(node_x[edge_n1[i]])
        y1 = float(node_y[edge_n1[i]])
        best = None
        best_key = None
        for pr, g, t, v in features:
            if not _edge_matches_feature(
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                feature_geom=g,
                qgs_geometry_cls=qgs_geometry_cls,
                qgs_pointxy_cls=qgs_pointxy_cls,
            ):
                continue
            dist = _edge_midpoint_distance_to_feature(
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                feature_geom=g,
                qgs_geometry_cls=qgs_geometry_cls,
                qgs_pointxy_cls=qgs_pointxy_cls,
            )
            key = (int(pr), -float(dist))
            if best is None or best_key is None or key > best_key:
                best = (t, v)
                best_key = key
        if best is not None:
            t, v = best
            changed = (int(bc_type[i]) != int(t)) or (not np.isclose(float(bc_val[i]), float(v)))
            bc_type[i] = int(t)
            bc_val[i] = float(v)
            if changed:
                applied += 1

    if applied:
        log_fn(f"BC line static overrides applied to {applied}/{edge_n0.size} boundary edges from '{bc_layer.name()}'.")
    elif features:
        # Fallback path for legacy/offset geometries where strict midpoint
        # proximity misses all intended edges.
        fallback_applied = 0
        for i in range(edge_n0.size):
            x0 = float(node_x[edge_n0[i]])
            y0 = float(node_y[edge_n0[i]])
            x1 = float(node_x[edge_n1[i]])
            y1 = float(node_y[edge_n1[i]])
            best = None
            best_key = None
            for pr, g, t, v in features:
                if not _edge_intersects_feature(
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    feature_geom=g,
                    qgs_geometry_cls=qgs_geometry_cls,
                    qgs_pointxy_cls=qgs_pointxy_cls,
                ):
                    continue
                dist = _edge_midpoint_distance_to_feature(
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    feature_geom=g,
                    qgs_geometry_cls=qgs_geometry_cls,
                    qgs_pointxy_cls=qgs_pointxy_cls,
                )
                key = (int(pr), -float(dist))
                if best is None or best_key is None or key > best_key:
                    best = (t, v)
                    best_key = key
            if best is not None:
                t, v = best
                changed = (int(bc_type[i]) != int(t)) or (not np.isclose(float(bc_val[i]), float(v)))
                bc_type[i] = int(t)
                bc_val[i] = float(v)
                if changed:
                    fallback_applied += 1
        if fallback_applied:
            log_fn(
                "BC line static overrides fallback applied via geometry intersections "
                f"to {fallback_applied}/{edge_n0.size} boundary edges from '{bc_layer.name()}'."
            )

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
    canonical_hydro_layer = None
    if hgid_field is not None:
        hydro_layers = [
            lyr
            for lyr in iter_project_layers_fn()
            if isinstance(lyr, qgs_vector_layer_cls) and str(lyr.name()).lower() in ("swe2d_hydrographs",)
        ]
        if hydro_layers:
            hlyr = hydro_layers[0]
            canonical_hydro_layer = hlyr
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

    def _resolve_layer_hydrograph(layer_ref: str, hid: str, bc_t: int):
        if not layer_ref:
            return None
        target_layer = None
        for lyr in iter_project_layers_fn():
            if not isinstance(lyr, qgs_vector_layer_cls):
                continue
            try:
                if lyr.id() == layer_ref or str(lyr.name()) == layer_ref:
                    target_layer = lyr
                    break
            except Exception as e:
                logger.warning("_resolve_layer_hydrograph layer matching failed: %s", e)
                continue
        if target_layer is None:
            return None
        return hydrograph_from_layer_fn(target_layer, hydrograph_id=hid, bc_type=bc_t)

    features = []
    for ft in bc_layer.getFeatures():
        geom = ft.geometry()
        if geom is None or geom.isEmpty():
            continue
        try:
            t = int(ft[type_field])
        except Exception as e:
            log_fn(f"[ERROR] Hydrograph BC type field parse failed: {e}")
            continue
        if t not in (int(ts_flow_code), int(ts_stage_code)):
            continue

        raw_h = str(ft[hydro_field] or "").strip() if hydro_field is not None else ""
        hid = str(ft[hgid_field] or "").strip() if hgid_field is not None else ""
        ref_layer = str(ft[hlyr_field] or "").strip() if hlyr_field is not None else ""

        if not raw_h and hgid_field is not None:
            if hid in hydro_lookup:
                raw_h = hydro_lookup[hid]

        # Prefer explicit referenced table/layer when provided.
        if ref_layer:
            hg_layer = _resolve_layer_hydrograph(ref_layer, hid, t)
            if hg_layer is not None:
                pr = 0
                if prio_field is not None:
                    try:
                        pr = int(ft[prio_field])
                    except Exception as e:
                        log_fn(f"[ERROR] Hydrograph priority field parse failed: {e}")
                        pr = 0
                features.append((pr, geom, t, hg_layer))
                continue

        if not raw_h:
            if hid and canonical_hydro_layer is not None:
                hg_layer = hydrograph_from_layer_fn(canonical_hydro_layer, hydrograph_id=hid, bc_type=t)
                if hg_layer is not None:
                    pr = 0
                    if prio_field is not None:
                        try:
                            pr = int(ft[prio_field])
                        except Exception as e:
                            log_fn(f"[ERROR] Hydrograph priority field parse (canonical): {e}")
                            pr = 0
                    features.append((pr, geom, t, hg_layer))
            continue

        try:
            hg = parse_hydrograph_text_fn(raw_h)
        except Exception as e:
            log_fn(f"[ERROR] Hydrograph text parse failed: {e}")
            hg = None
        if hg is None:
            # If parsing failed, treat hydrograph field as a layer reference token.
            hg_layer = _resolve_layer_hydrograph(raw_h, hid, t)
            if hg_layer is None and hid and canonical_hydro_layer is not None:
                hg_layer = hydrograph_from_layer_fn(canonical_hydro_layer, hydrograph_id=hid, bc_type=t)
            if hg_layer is not None:
                pr = 0
                if prio_field is not None:
                    try:
                        pr = int(ft[prio_field])
                    except Exception as e:
                        log_fn(f"[ERROR] Hydrograph priority field parse (layer ref): {e}")
                        pr = 0
                features.append((pr, geom, t, hg_layer))
            continue

        pr = 0
        if prio_field is not None:
            try:
                pr = int(ft[prio_field])
            except Exception as e:
                log_fn(f"[ERROR] Hydrograph priority field parse (text): {e}")
                pr = 0
        features.append((pr, geom, t, hg))

    if not features:
        return edge_hydro

    for i in range(edge_n0.size):
        x0 = float(node_x[edge_n0[i]])
        y0 = float(node_y[edge_n0[i]])
        x1 = float(node_x[edge_n1[i]])
        y1 = float(node_y[edge_n1[i]])
        best = None
        best_key = None
        for pr, g, t, hg in features:
            if not _edge_matches_feature(
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                feature_geom=g,
                qgs_geometry_cls=qgs_geometry_cls,
                qgs_pointxy_cls=qgs_pointxy_cls,
            ):
                continue
            dist = _edge_midpoint_distance_to_feature(
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                feature_geom=g,
                qgs_geometry_cls=qgs_geometry_cls,
                qgs_pointxy_cls=qgs_pointxy_cls,
            )
            key = (int(pr), -float(dist))
            if best is None or best_key is None or key > best_key:
                best = (t, hg)
                best_key = key
        if best is not None:
            t, hg = best
            edge_hydro[i] = (t, hg)

    if not edge_hydro and features:
        for i in range(edge_n0.size):
            x0 = float(node_x[edge_n0[i]])
            y0 = float(node_y[edge_n0[i]])
            x1 = float(node_x[edge_n1[i]])
            y1 = float(node_y[edge_n1[i]])
            best = None
            best_key = None
            for pr, g, t, hg in features:
                if not _edge_intersects_feature(
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    feature_geom=g,
                    qgs_geometry_cls=qgs_geometry_cls,
                    qgs_pointxy_cls=qgs_pointxy_cls,
                ):
                    continue
                dist = _edge_midpoint_distance_to_feature(
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    feature_geom=g,
                    qgs_geometry_cls=qgs_geometry_cls,
                    qgs_pointxy_cls=qgs_pointxy_cls,
                )
                key = (int(pr), -float(dist))
                if best is None or best_key is None or key > best_key:
                    best = (t, hg)
                    best_key = key
            if best is not None:
                t, hg = best
                edge_hydro[i] = (t, hg)

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
            except Exception as e:
                logger.warning("BC edge group priority field parse failed: %s", e)
                pr = 0
        try:
            nm = f"feature_{int(ft.id())}"
        except Exception as e:
            logger.warning("BC edge group feature ID parse failed: %s", e)
            nm = "feature"
        features.append((pr, geom, nm))

    if not features:
        return edge_groups

    for i in range(edge_n0.size):
        x0 = float(node_x[edge_n0[i]])
        y0 = float(node_y[edge_n0[i]])
        x1 = float(node_x[edge_n1[i]])
        y1 = float(node_y[edge_n1[i]])
        best_nm = None
        best_key = None
        for pr, g, nm in features:
            if not _edge_matches_feature(
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                feature_geom=g,
                qgs_geometry_cls=qgs_geometry_cls,
                qgs_pointxy_cls=qgs_pointxy_cls,
            ):
                continue
            dist = _edge_midpoint_distance_to_feature(
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                feature_geom=g,
                qgs_geometry_cls=qgs_geometry_cls,
                qgs_pointxy_cls=qgs_pointxy_cls,
            )
            key = (int(pr), -float(dist))
            if best_nm is None or best_key is None or key > best_key:
                best_nm = nm
                best_key = key
        if best_nm is not None:
            edge_groups[i] = f"bc_line:{best_nm}"

    if not edge_groups and features:
        for i in range(edge_n0.size):
            x0 = float(node_x[edge_n0[i]])
            y0 = float(node_y[edge_n0[i]])
            x1 = float(node_x[edge_n1[i]])
            y1 = float(node_y[edge_n1[i]])
            best_nm = None
            best_key = None
            for pr, g, nm in features:
                if not _edge_intersects_feature(
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    feature_geom=g,
                    qgs_geometry_cls=qgs_geometry_cls,
                    qgs_pointxy_cls=qgs_pointxy_cls,
                ):
                    continue
                dist = _edge_midpoint_distance_to_feature(
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    feature_geom=g,
                    qgs_geometry_cls=qgs_geometry_cls,
                    qgs_pointxy_cls=qgs_pointxy_cls,
                )
                key = (int(pr), -float(dist))
                if best_nm is None or best_key is None or key > best_key:
                    best_nm = nm
                    best_key = key
            if best_nm is not None:
                edge_groups[i] = f"bc_line:{best_nm}"

    return edge_groups
