"""Pure-Python, Qt-free service for SWE2D structure/network/line geometry logic.

Provides structure record loading, filtering, line geometry extraction, and
structure profile overlay resolution without any Qt dependency.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np

from swe2d.results.db_utils import open_ro, table_exists
from swe2d.results.queries import _find_prefixed_or_default_table

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional QGIS imports (graceful fallback)
# ---------------------------------------------------------------------------

try:
    from qgis.core import QgsVectorLayer, QgsFeatureRequest
except ImportError:
    QgsVectorLayer = None
    QgsFeatureRequest = None

_HAVE_QGIS = QgsVectorLayer is not None


def _line_layer_uri(gpkg_path: str, layer_name: str) -> str:
    """line layer uri."""
    return f"{gpkg_path}|layername={layer_name}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_structure_records(
    gpkg_path: str, run_id: str,
) -> List[Dict[str, Any]]:
    """Load all coupling structure records for *run_id* from *gpkg_path*.

    Returns a list of dicts with keys:
        ``t_s``, ``component``, ``metric``, ``object_id``, ``object_name``, ``value``

    Returns an empty list on any error.
    """
    if not gpkg_path or not run_id:
        return []
    conn = open_ro(gpkg_path)
    if conn is None:
        return []
    try:
        ct = _find_prefixed_or_default_table(conn, "swe2d_coupling_results")
        if not ct:
            return []
        cur = conn.execute(
            f'SELECT t_s, component, metric, object_id, object_name, value '
            f'FROM "{ct}" WHERE run_id = ? ORDER BY t_s',
            (str(run_id),),
        )
        cols = ["t_s", "component", "metric", "object_id", "object_name", "value"]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
    except Exception as exc:
        logger.debug("[STRUCT] Failed to load structure records: %s", exc)
        return []
    finally:
        conn.close()


def load_line_geometry(
    gpkg_path: str, line_id: int, line_name: str,
) -> np.ndarray:
    """Load profile line vertex coordinates as an ``(N, 2)`` numpy array.

    Each row is ``(x, y)`` in the layer's CRS.
    Returns an empty float64 array on error or if not found.
    """
    if not _HAVE_QGIS or not gpkg_path or line_id < 0:
        return np.empty((0, 2), dtype=np.float64)
    try:
        layer_name_table = "swe2d_sample_lines"
        layer = QgsVectorLayer(
            _line_layer_uri(gpkg_path, layer_name_table),
            layer_name_table, "ogr",
        )
        if layer is None or not layer.isValid():
            return np.empty((0, 2), dtype=np.float64)

        fields = set(layer.fields().names())
        for ft in layer.getFeatures():
            match = False
            try:
                if "line_id" in fields and int(ft["line_id"]) == int(line_id):
                    match = True
            except Exception:
                pass
            if not match and line_name and "name" in fields:
                try:
                    if str(ft["name"] or "") == line_name:
                        match = True
                except Exception:
                    pass
            if not match:
                continue

            geom = ft.geometry()
            if geom is None or geom.isEmpty():
                continue

            # Store geometry object reference for reuse
            raw = geom.constGet() if hasattr(geom, 'constGet') else geom
            n = raw.vertexCount() if hasattr(raw, 'vertexCount') else 0
            if n < 2:
                continue
            coords = np.empty((n, 2), dtype=np.float64)
            for i in range(n):
                coords[i, 0] = float(raw.xAt(i))
                coords[i, 1] = float(raw.yAt(i))
            return coords

        return np.empty((0, 2), dtype=np.float64)
    except Exception as exc:
        logger.debug("[STRUCT] Failed to load line geometry: %s", exc)
        return np.empty((0, 2), dtype=np.float64)


def load_bound_layer_name(
    gpkg_path: str, role: str, default_name: str,
) -> str:
    """Look up a layer name bound to *role* from the GPKG bindings table.

    Returns *default_name* on any error.
    """
    if not gpkg_path:
        return str(default_name)
    conn = open_ro(gpkg_path)
    if conn is None:
        return str(default_name)
    try:
        if not table_exists(conn, "swe2d_layer_bindings"):
            return str(default_name)
        cur = conn.execute(
            "SELECT layer_name FROM swe2d_layer_bindings WHERE role = ?",
            (str(role),),
        )
        row = cur.fetchone()
        if row is None or not row[0]:
            return str(default_name)
        return str(row[0])
    except Exception as exc:
        logger.debug("[STRUCT] Failed to load bound layer name: %s", exc)
        return str(default_name)
    finally:
        if conn is not None:
            conn.close()


def filter_structure_records(
    records: List[Dict[str, Any]],
    metric_col: str,
    min_val: float,
) -> List[Dict[str, Any]]:
    """Return records where ``record[metric_col] >= min_val``.

    Records missing *metric_col* are excluded.
    """
    out: List[Dict[str, Any]] = []
    for rec in records:
        val = rec.get(metric_col)
        if val is None:
            continue
        try:
            if float(val) >= float(min_val):
                out.append(rec)
        except (ValueError, TypeError):
            continue
    return out


def resolve_structure_profile_overlays(
    gpkg_path: str, run_ids: List[str],
) -> Dict[str, List[Dict[str, Any]]]:
    """Resolve structure profile overlay data for multiple runs.

    For each run in *run_ids*, loads structure records, computes station
    positions along the sample line, and returns an overlay dict per run::

        {
            run_id: [
                {
                    "run_id": str,
                    "run_label": str,
                    "object_id": str,
                    "flow_cms": float,
                    "station_m": float,
                    "elev_m": float,
                    "placement": "geometry" | "fallback" | "unplaced",
                },
            ],
        }

    Returns an empty dict on error or if no data is available.
    """
    if not gpkg_path or not run_ids or not _HAVE_QGIS:
        return {}

    line_id = -1
    line_name = ""
    # Load line geometry from the first run that has data
    # (all runs share the same sample line for a given GPKG)
    for rid in run_ids:
        records = load_structure_records(gpkg_path, rid)
        if records:
            break

    if not records:
        return {}

    line_geom = _load_profile_line_geom(gpkg_path, line_id, line_name)
    if line_geom is None or line_geom.isEmpty():
        # Fall back to station-less mode
        return _resolve_overlays_fallback(gpkg_path, run_ids)

    layer_name = load_bound_layer_name(
        gpkg_path, "hydraulic_structures", "swe2d_structures",
    )
    layer = QgsVectorLayer(
        _line_layer_uri(gpkg_path, layer_name), layer_name, "ogr",
    )
    if layer is None or not layer.isValid():
        return {}

    fields = set(layer.fields().names())
    if "structure_id" not in fields:
        return {}

    out: Dict[str, List[Dict[str, Any]]] = {}
    for run_id in run_ids:
        struct_rows = load_structure_records(gpkg_path, run_id)
        if not struct_rows:
            out[run_id] = []
            continue

        flow_by_id = {
            str(r.get("object_id", "")): float(r.get("value", 0.0))
            for r in struct_rows
            if str(r.get("object_id", ""))
        }

        overlays: List[Dict[str, Any]] = []
        for ft in layer.getFeatures():
            sid = str(ft["structure_id"] or "")
            if sid not in flow_by_id:
                continue
            geom = ft.geometry()
            if geom is None or geom.isEmpty():
                continue

            station_m = float("nan")
            try:
                inter = geom.intersection(line_geom)
                if inter is not None and not inter.isEmpty():
                    station_m = float(line_geom.lineLocatePoint(inter.centroid()))
            except Exception as exc:
                logger.debug("[STRUCT] Failed to compute station: %s", exc)

            if not np.isfinite(station_m):
                try:
                    centroid = geom.centroid()
                    if centroid is not None and not centroid.isEmpty():
                        nearest = line_geom.nearestPoint(centroid)
                        if nearest is not None and not nearest.isEmpty():
                            station_m = float(line_geom.lineLocatePoint(nearest))
                except Exception as exc:
                    logger.debug("[STRUCT] Failed fallback station: %s", exc)

            crest = float("nan")
            try:
                if "crest_elev" in fields and ft["crest_elev"] not in (None, ""):
                    crest = float(ft["crest_elev"])
            except Exception as exc:
                logger.debug("[STRUCT] Failed to read crest: %s", exc)

            overlays.append({
                "run_id": str(run_id),
                "run_label": str(run_id),
                "object_id": sid,
                "flow_cms": float(flow_by_id[sid]),
                "station_m": station_m,
                "elev_m": crest,
                "placement": "geometry" if np.isfinite(station_m) else "fallback",
            })

        # Add unplaced structures
        placed_ids = {o["object_id"] for o in overlays}
        for r in struct_rows:
            sid = str(r.get("object_id", ""))
            if sid in placed_ids:
                continue
            overlays.append({
                "run_id": str(run_id),
                "run_label": str(run_id),
                "object_id": sid,
                "flow_cms": float(r.get("value", 0.0)),
                "station_m": float("nan"),
                "elev_m": float("nan"),
                "placement": "unplaced",
            })

        out[run_id] = overlays

    return out


def _load_profile_line_geom(gpkg_path: str, line_id: int, line_name: str):
    """Load profile line as a QgsGeometry (internal helper)."""
    if not _HAVE_QGIS or not gpkg_path:
        return None
    try:
        layer_name_table = "swe2d_sample_lines"
        layer = QgsVectorLayer(
            _line_layer_uri(gpkg_path, layer_name_table),
            layer_name_table, "ogr",
        )
        if layer is None or not layer.isValid():
            return None
        fields = set(layer.fields().names())
        for ft in layer.getFeatures():
            match = False
            try:
                if "line_id" in fields and int(ft["line_id"]) == int(line_id):
                    match = True
            except Exception:
                pass
            if not match and line_name and "name" in fields:
                try:
                    if str(ft["name"] or "") == line_name:
                        match = True
                except Exception:
                    pass
            if not match:
                continue
            geom = ft.geometry()
            if geom is not None and not geom.isEmpty():
                return geom
        return None
    except Exception as exc:
        logger.debug("[STRUCT] Failed to load profile line geom: %s", exc)
        return None


def _resolve_overlays_fallback(
    gpkg_path: str, run_ids: List[str],
) -> Dict[str, List[Dict[str, Any]]]:
    """Fallback overlay resolution when line geometry is unavailable."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    for run_id in run_ids:
        struct_rows = load_structure_records(gpkg_path, run_id)
        overlays = [
            {
                "run_id": str(run_id),
                "run_label": str(run_id),
                "object_id": str(r.get("object_id", "")),
                "flow_cms": float(r.get("value", 0.0)),
                "station_m": float("nan"),
                "elev_m": float("nan"),
                "placement": "unplaced",
            }
            for r in struct_rows
            if r.get("object_id")
        ]
        out[run_id] = overlays
    return out


def load_structure_overlay_data(
    gpkg_path: str,
    run_ids: List[str],
    t_sec: float,
    line_id: int = -1,
    line_name: str = "",
) -> List[Dict[str, Any]]:
    """Load structure overlay data for plot selection.

    Returns a list of dicts with keys:
        object_id, object_name, flow_cms, station_m, elev_m, placement

    station_m and elev_m are computed for profile view.
    x, y are NOT populated (structure overlay is 1D).
    """
    if not gpkg_path or not run_ids:
        return []

    records = []
    for rid in run_ids:
        struct_rows = load_structure_records(gpkg_path, rid)
        if not struct_rows:
            continue

        # Load line geometry for station computation
        line_geom = _load_profile_line_geom(gpkg_path, line_id, line_name)
        if line_geom is None or line_geom.isEmpty():
            # Fallback: station-less mode
            for r in struct_rows:
                records.append({
                    "object_id": str(r.get("object_id", "")),
                    "object_name": str(r.get("object_name", "") or ""),
                    "flow_cms": float(r.get("value", 0.0)),
                    "station_m": float("nan"),
                    "elev_m": float("nan"),
                    "placement": "unplaced",
                })
            continue

        layer_name = load_bound_layer_name(
            gpkg_path, "hydraulic_structures", "swe2d_structures",
        )
        layer = QgsVectorLayer(
            _line_layer_uri(gpkg_path, layer_name), layer_name, "ogr",
        )
        if layer is None or not layer.isValid():
            continue

        fields = set(layer.fields().names())
        if "structure_id" not in fields:
            continue

        # Find nearest timestep
        from swe2d.results.queries import find_nearest_timestep
        t = find_nearest_timestep(gpkg_path, rid, line_id, t_sec)

        # Load structure flows at this timestep
        from swe2d.results.queries import load_structure_flows_at_time
        struct_flows = load_structure_flows_at_time(gpkg_path, rid, t, t_tol=1.0)

        flow_by_id = {
            str(r.get("object_id", "")): float(r.get("value", 0.0))
            for r in struct_flows
        }

        # Compute stations
        for ft in layer.getFeatures():
            sid = str(ft["structure_id"] or "")
            if sid not in flow_by_id:
                continue

            geom = ft.geometry()
            if geom is None or geom.isEmpty():
                continue

            station_m = float("nan")
            try:
                inter = geom.intersection(line_geom)
                if inter is not None and not inter.isEmpty():
                    station_m = float(line_geom.lineLocatePoint(inter.centroid()))
            except Exception:
                pass

            if not np.isfinite(station_m):
                try:
                    centroid = geom.centroid()
                    if centroid is not None and not centroid.isEmpty():
                        nearest = line_geom.nearestPoint(centroid)
                        if nearest is not None and not nearest.isEmpty():
                            station_m = float(line_geom.lineLocatePoint(nearest))
                except Exception:
                    pass

            crest = float("nan")
            if "crest_elev" in fields and ft["crest_elev"] not in (None, ""):
                crest = float(ft["crest_elev"])

            records.append({
                "object_id": sid,
                "object_name": str(ft.get("name", "") or ""),
                "flow_cms": float(flow_by_id[sid]),
                "station_m": station_m,
                "elev_m": crest,
                "placement": "geometry" if np.isfinite(station_m) else "unplaced",
            })

    return records
