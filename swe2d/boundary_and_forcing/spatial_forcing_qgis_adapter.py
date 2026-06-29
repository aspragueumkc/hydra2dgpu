from __future__ import annotations

"""QGIS adapter for spatially-distributed forcing (Manning, CN, rain gages)."""

import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def build_spatial_manning_array_qgis(
    *,
    mesh_data,
    have_qgis_core: bool,
    manning_layer_combo,
    combo_layer_fn: Callable[[object, str], Optional[object]],
    mesh_cell_centroids_fn: Callable[[], Tuple[np.ndarray, np.ndarray]],
    default_n: float,
    qgs_geometry_cls,
    qgs_pointxy_cls,
    log_fn: Callable[[str], None],
) -> Optional[np.ndarray]:
    """
    build spatial manning array qgis.

    Parameters
    ----------
    mesh_data
        Description of mesh_data.
    have_qgis_core : bool
        Description of have_qgis_core.
    manning_layer_combo
        Description of manning_layer_combo.
    combo_layer_fn : Callable[[object, str], Optional[object]]
        Description of combo_layer_fn.
    mesh_cell_centroids_fn : Callable[[], Tuple[np.ndarray, np.ndarray]]
        Description of mesh_cell_centroids_fn.
    default_n : float
        Description of default_n.
    qgs_geometry_cls
        Description of qgs_geometry_cls.
    qgs_pointxy_cls
        Description of qgs_pointxy_cls.
    log_fn : Callable[[str], None]
        Description of log_fn.

    Returns
    -------
    Optional[np.ndarray]
    """
    if mesh_data is None or not have_qgis_core:
        return None
    if manning_layer_combo is None:
        return None

    lyr = combo_layer_fn(manning_layer_combo, "vector")
    if lyr is None:
        return None

    fields = set(lyr.fields().names())
    n_field = None
    for cand in ("n_mann", "manning_n", "manning", "n"):
        if cand in fields:
            n_field = cand
            break
    if n_field is None:
        log_fn("Manning layer selected but no n_mann/manning_n/manning/n field found; using global n.")
        return None

    prio_field = "priority" if "priority" in fields else None
    cx, cy = mesh_cell_centroids_fn()
    nvals = np.full(cx.shape[0], float(default_n), dtype=np.float64)

    features = []
    for ft in lyr.getFeatures():
        g = ft.geometry()
        if g is None or g.isEmpty():
            continue
        try:
            n = float(ft[n_field])
        except Exception as e:
            log_fn(f"[ERROR] Manning n field parse failed: {e}")
            continue
        pr = 0
        if prio_field is not None:
            try:
                pr = int(ft[prio_field])
            except Exception as e:
                log_fn(f"[ERROR] Manning priority field parse failed: {e}")
                pr = 0
        features.append((pr, g, n))

    if not features:
        return None

    features.sort(key=lambda x: x[0], reverse=True)
    applied = 0
    for i in range(cx.shape[0]):
        p = qgs_geometry_cls.fromPointXY(qgs_pointxy_cls(float(cx[i]), float(cy[i])))
        for _, g, n in features:
            if g.contains(p) or g.intersects(p):
                nvals[i] = n
                applied += 1
                break

    log_fn(f"Spatial Manning applied to {applied}/{cx.shape[0]} cells from '{lyr.name()}'.")
    return nvals


def build_spatial_cn_array_qgis(
    *,
    mesh_data,
    have_qgis_core: bool,
    cn_layer_combo,
    combo_layer_fn: Callable[[object, str], Optional[object]],
    mesh_cell_centroids_fn: Callable[[], Tuple[np.ndarray, np.ndarray]],
    default_cn: float,
    qgs_geometry_cls,
    qgs_pointxy_cls,
    log_fn: Callable[[str], None],
) -> np.ndarray:
    """
    build spatial cn array qgis.

    Parameters
    ----------
    mesh_data
        Description of mesh_data.
    have_qgis_core : bool
        Description of have_qgis_core.
    cn_layer_combo
        Description of cn_layer_combo.
    combo_layer_fn : Callable[[object, str], Optional[object]]
        Description of combo_layer_fn.
    mesh_cell_centroids_fn : Callable[[], Tuple[np.ndarray, np.ndarray]]
        Description of mesh_cell_centroids_fn.
    default_cn : float
        Description of default_cn.
    qgs_geometry_cls
        Description of qgs_geometry_cls.
    qgs_pointxy_cls
        Description of qgs_pointxy_cls.
    log_fn : Callable[[str], None]
        Description of log_fn.

    Returns
    -------
    np.ndarray
    """
    cx, cy = mesh_cell_centroids_fn()
    cnvals = np.full(cx.shape[0], float(default_cn), dtype=np.float64)

    if mesh_data is None or not have_qgis_core:
        return cnvals
    if cn_layer_combo is None:
        return cnvals

    lyr = combo_layer_fn(cn_layer_combo, "vector")
    if lyr is None:
        return cnvals

    fields = set(lyr.fields().names())
    cn_field = None
    for cand in ("cn", "curve_number", "CN"):
        if cand in fields:
            cn_field = cand
            break
    if cn_field is None:
        log_fn("CN layer selected but no cn/curve_number field found; using default CN.")
        return cnvals

    prio_field = "priority" if "priority" in fields else None
    features = []
    for ft in lyr.getFeatures():
        g = ft.geometry()
        if g is None or g.isEmpty():
            continue
        try:
            cn = float(ft[cn_field])
        except Exception as e:
            log_fn(f"[ERROR] CN field parse failed: {e}")
            continue
        pr = 0
        if prio_field is not None:
            try:
                pr = int(ft[prio_field])
            except Exception as e:
                log_fn(f"[ERROR] CN priority field parse failed: {e}")
                pr = 0
        features.append((pr, g, float(np.clip(cn, 1.0, 100.0))))

    if not features:
        return cnvals

    features.sort(key=lambda x: x[0], reverse=True)
    applied = 0
    for i in range(cx.shape[0]):
        p = qgs_geometry_cls.fromPointXY(qgs_pointxy_cls(float(cx[i]), float(cy[i])))
        for _, g, cn in features:
            if g.contains(p) or g.intersects(p):
                cnvals[i] = cn
                applied += 1
                break

    log_fn(f"Spatial CN applied to {applied}/{cx.shape[0]} cells from '{lyr.name()}'.")
    return cnvals


def build_thiessen_rain_cn_forcing_qgis(
    *,
    mesh_data,
    have_qgis_core: bool,
    thiessen_rain_cn_forcing_cls,
    gauge_cls,
    build_hyetograph_fn,
    assign_cells_to_nearest_gauge_fn,
    inspect_hyetograph_rows_fn,
    use_spatial_rain_cn: bool,
    rain_gage_layer_combo,
    hyetograph_layer_combo,
    storm_area_layer_combo,
    combo_layer_fn: Callable[[object, str], Optional[object]],
    mesh_cell_centroids_fn: Callable[[], Tuple[np.ndarray, np.ndarray]],
    boundary_buffer_cells_fn: Callable[[int], np.ndarray],
    build_spatial_cn_array_fn: Callable[[], np.ndarray],
    ia_ratio: float,
    infiltration_method: str,
    rain_boundary_buffer_rings: int,
    qgs_wkb_types,
    qgs_geometry_cls,
    qgs_pointxy_cls,
    log_fn: Callable[[str], None],
):
    """
    build thiessen rain cn forcing qgis.

    Parameters
    ----------
    mesh_data
        Description of mesh_data.
    have_qgis_core : bool
        Description of have_qgis_core.
    thiessen_rain_cn_forcing_cls
        Description of thiessen_rain_cn_forcing_cls.
    gauge_cls
        Description of gauge_cls.
    build_hyetograph_fn
        Description of build_hyetograph_fn.
    assign_cells_to_nearest_gauge_fn
        Description of assign_cells_to_nearest_gauge_fn.
    inspect_hyetograph_rows_fn
        Description of inspect_hyetograph_rows_fn.
    use_spatial_rain_cn : bool
        Description of use_spatial_rain_cn.
    rain_gage_layer_combo
        Description of rain_gage_layer_combo.
    hyetograph_layer_combo
        Description of hyetograph_layer_combo.
    storm_area_layer_combo
        Description of storm_area_layer_combo.
    combo_layer_fn : Callable[[object, str], Optional[object]]
        Description of combo_layer_fn.
    mesh_cell_centroids_fn : Callable[[], Tuple[np.ndarray, np.ndarray]]
        Description of mesh_cell_centroids_fn.
    boundary_buffer_cells_fn : Callable[[int], np.ndarray]
        Description of boundary_buffer_cells_fn.
    build_spatial_cn_array_fn : Callable[[], np.ndarray]
        Description of build_spatial_cn_array_fn.
    ia_ratio : float
        Description of ia_ratio.
    infiltration_method : str
        Description of infiltration_method.
    rain_boundary_buffer_rings : int
        Description of rain_boundary_buffer_rings.
    qgs_wkb_types
        Description of qgs_wkb_types.
    qgs_geometry_cls
        Description of qgs_geometry_cls.
    qgs_pointxy_cls
        Description of qgs_pointxy_cls.
    log_fn : Callable[[str], None]
        Description of log_fn.
    """
    if (
        mesh_data is None
        or not have_qgis_core
        or thiessen_rain_cn_forcing_cls is None
        or gauge_cls is None
        or build_hyetograph_fn is None
        or assign_cells_to_nearest_gauge_fn is None
    ):
        return None
    if not bool(use_spatial_rain_cn):
        return None
    if rain_gage_layer_combo is None or hyetograph_layer_combo is None:
        return None

    gage_layer = combo_layer_fn(rain_gage_layer_combo, "vector")
    hyetograph_layer = combo_layer_fn(hyetograph_layer_combo, "vector")
    if gage_layer is None or hyetograph_layer is None:
        return None

    gage_fields = set(gage_layer.fields().names())
    gid_field = "gage_id" if "gage_id" in gage_fields else None
    hyid_field = "hyetograph_id" if "hyetograph_id" in gage_fields else None
    if gid_field is None or hyid_field is None:
        log_fn("Rain gage layer missing gage_id/hyetograph_id fields; skipping Thiessen rain forcing.")
        return None

    hy_field_names = list(hyetograph_layer.fields().names())
    hy_fields = set(hy_field_names)
    hy_id_field = "hyetograph_id" if "hyetograph_id" in hy_fields else None

    time_field = None
    for cand in ("Time", "time", "Time_hr", "time_hr", "Time_min", "time_min", "minutes", "Minutes", "t_min"):
        if cand in hy_fields:
            time_field = cand
            break

    value_field = None
    for cand in (
        "Value",
        "value",
        "Rain",
        "rain",
        "rainfall",
        "Rainfall",
        "Incremental_Rainfall_in",
        "incremental_rainfall_in",
        "rain_in",
        "rain_mm",
    ):
        if cand in hy_fields:
            value_field = cand
            break

    if value_field is None:
        for name in hy_field_names:
            ln = str(name).lower()
            if "value" in ln or "rain" in ln or "hyeto" in ln:
                value_field = name
                break
    if hy_id_field is None or time_field is None or value_field is None:
        log_fn("Hyetograph table missing hyetograph_id/Time/Value fields; skipping Thiessen rain forcing.")
        return None

    hy_rows_by_id: Dict[str, List[Dict[str, object]]] = {}
    time_field_l = str(time_field or "").lower()
    value_field_l = str(value_field or "").lower()

    inferred_value_type = "intensity"
    if "increment" in value_field_l or "depth" in value_field_l:
        inferred_value_type = "incremental_depth"
    elif "cum" in value_field_l:
        inferred_value_type = "cumulative_depth"

    inferred_units = "mm/hr"
    if "in/hr" in value_field_l:
        inferred_units = "in/hr"
    elif "mm/hr" in value_field_l:
        inferred_units = "mm/hr"
    elif "_in" in value_field_l or "inch" in value_field_l:
        inferred_units = "in"
    elif "_mm" in value_field_l:
        inferred_units = "mm"

    for ft in hyetograph_layer.getFeatures():
        try:
            hy_id = str(ft[hy_id_field] or "").strip()
        except Exception as e:
            log_fn(f"[ERROR] Hyetograph ID field parse failed: {e}")
            hy_id = ""
        if not hy_id:
            continue

        time_value = ft[time_field]
        if "min" in time_field_l and isinstance(time_value, (int, float)):
            time_value = f"{float(time_value)} min"
        elif "hr" in time_field_l and isinstance(time_value, (int, float)):
            time_value = f"{float(time_value)} hr"

        row = {
            "Time": time_value,
            "Value": ft[value_field],
            "value_type": ft["value_type"] if "value_type" in hy_fields else inferred_value_type,
            "units": ft["units"] if "units" in hy_fields else inferred_units,
        }
        hy_rows_by_id.setdefault(hy_id, []).append(row)

    if inspect_hyetograph_rows_fn is not None:
        for hy_id in sorted(hy_rows_by_id.keys()):
            rows = hy_rows_by_id.get(hy_id, [])
            diag = inspect_hyetograph_rows_fn(rows)
            log_fn(
                "Hyetograph parse: "
                f"id='{hy_id}', rows={int(diag.get('n_rows', 0))}, valid={int(diag.get('n_valid', 0))}, "
                f"mode={diag.get('mode', 'unknown')}, units={diag.get('units', 'unknown')}, "
                f"t=[{float(diag.get('t_start_s', 0.0)):.1f},{float(diag.get('t_end_s', 0.0)):.1f}] s, "
                f"dt_med={float(diag.get('dt_median_s', 0.0)):.1f} s, "
                f"total_depth={float(diag.get('total_depth_mm', 0.0)):.3f} mm"
            )
            for w in list(diag.get("warnings", [])):
                log_fn(f"Hyetograph parse warning (id='{hy_id}'): {w}")

    gauges = []
    hy_by_gauge_index: Dict[int, object] = {}
    for ft in gage_layer.getFeatures():
        geom = ft.geometry()
        if geom is None or geom.isEmpty():
            continue
        try:
            pt = geom.asPoint()
        except Exception as e:
            log_fn(f"[ERROR] Gauge geometry asPoint failed: {e}")
            continue
        gauge_id = str(ft[gid_field] or "").strip()
        hy_id = str(ft[hyid_field] or "").strip()
        if not gauge_id or not hy_id:
            continue
        hy = build_hyetograph_fn(hy_rows_by_id.get(hy_id, []))
        if hy is None:
            continue
        gauges.append(gauge_cls(gauge_id=gauge_id, x=float(pt.x()), y=float(pt.y()), hyetograph_id=hy_id))
        hy_by_gauge_index[len(gauges) - 1] = hy

    if not gauges:
        return None

    cell_x, cell_y = mesh_cell_centroids_fn()
    cell_to_gauge = assign_cells_to_nearest_gauge_fn(cell_x, cell_y, gauges)
    if cell_to_gauge is None:
        return None
    cell_to_gauge = np.asarray(cell_to_gauge, dtype=np.int32).copy()

    storm_area_layer = combo_layer_fn(storm_area_layer_combo, "vector") if storm_area_layer_combo is not None else None
    if storm_area_layer is not None:
        in_storm = np.zeros(cell_to_gauge.shape[0], dtype=bool)
        for ft in storm_area_layer.getFeatures():
            geom = ft.geometry()
            if geom is None or geom.isEmpty():
                continue
            try:
                wkb_type = int(geom.wkbType())
            except Exception as e:
                log_fn(f"[ERROR] WKB type parse failed: {e}")
                wkb_type = -1
            if qgs_wkb_types.geometryType(wkb_type) == qgs_wkb_types.GeometryType.PolygonGeometry:
                for i in range(cell_x.shape[0]):
                    if in_storm[i]:
                        continue
                    p = qgs_geometry_cls.fromPointXY(qgs_pointxy_cls(float(cell_x[i]), float(cell_y[i])))
                    if geom.contains(p) or geom.intersects(p):
                        in_storm[i] = True
            else:
                rp = geom.centroid().asPoint() if not geom.centroid().isEmpty() else None
                if rp is None:
                    continue
                dx = cell_x - float(rp.x())
                dy = cell_y - float(rp.y())
                in_storm[int(np.argmin(dx * dx + dy * dy))] = True

        if np.any(in_storm):
            excluded_count = int(np.count_nonzero(~in_storm))
            cell_to_gauge[~in_storm] = -1
            log_fn(
                f"Thiessen storm-area mask active: included {int(np.count_nonzero(in_storm))} cell(s), "
                f"excluded {excluded_count} outside '{storm_area_layer.name()}'."
            )
        else:
            cell_to_gauge[:] = -1
            log_fn(
                f"Thiessen storm-area mask active: no cell centroids intersected '{storm_area_layer.name()}'; "
                "rainfall forcing disabled by mask."
            )

    excluded = boundary_buffer_cells_fn(int(rain_boundary_buffer_rings))
    if excluded.size > 0:
        cell_to_gauge[excluded] = -1
        log_fn(
            f"Thiessen rain boundary buffer active: excluded {excluded.size} cell(s) "
            f"across {int(rain_boundary_buffer_rings)} boundary ring(s)."
        )

    cnvals = build_spatial_cn_array_fn()
    forcing = thiessen_rain_cn_forcing_cls(
        cell_to_gauge=cell_to_gauge,
        gauge_hyetographs=hy_by_gauge_index,
        curve_number=cnvals,
        ia_ratio=float(ia_ratio),
        infiltration_method=str(infiltration_method),
    )
    log_fn(
        f"Thiessen rain forcing active: gauges={len(gauges)}, "
        f"cells={cell_to_gauge.shape[0]}, infiltration={str(infiltration_method)}, Ia/S={float(ia_ratio):.3f}, "
        f"cn_range=[{float(np.min(cnvals)):.1f}, {float(np.max(cnvals)):.1f}]"
    )
    return forcing
