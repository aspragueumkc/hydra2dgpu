"""Core GPKG I/O helpers: read forcing data directly from GeoPackage without QGIS.

Each function mirrors a QGIS-layer-reader in the workbench but uses sqlite3
or ``QgsVectorLayer`` against a file GPKG.  Returns the same Python objects
(numpy arrays, ThiessenRainCNForcing, etc.) so the existing runtime pipeline
works unchanged.  These helpers live in ``swe2d.core`` so the CLI, GUI, and
runtime can all consume them without cross-layer imports.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
from typing import Any, Callable, Dict, List, Optional, Tuple

from osgeo import ogr

import numpy as np

from swe2d.mesh.mesh_runtime_logic import mesh_cell_centroids

from swe2d.boundary_and_forcing.rainfall_hydrology import (
    Hyetograph,
    ThiessenRainCNForcing,
    build_hyetograph,
)



logger = logging.getLogger(__name__)


class MeshLoadError(FileNotFoundError):
    """Raised when a configured baked mesh cannot be loaded safely."""


def query_mesh_from_gpkg(gpkg_path: str, mesh_name: str) -> Optional[Dict[str, Any]]:
    """Load a baked mesh, returning ``None`` only when its named row is absent."""
    if not os.path.isfile(gpkg_path):
        raise MeshLoadError(
            f"Mesh spec key 'mesh' references missing GeoPackage: {gpkg_path!r}"
        )

    try:
        connection = sqlite3.connect(gpkg_path)
        try:
            has_mesh_table = connection.execute(
                "SELECT 1 FROM sqlite_master "
                "WHERE type='table' AND name='swe2d_baked_mesh'"
            ).fetchone()
        finally:
            connection.close()
    except sqlite3.Error as exc:
        raise MeshLoadError(
            f"Mesh spec key 'mesh' could not inspect GeoPackage {gpkg_path!r}: {exc}"
        ) from exc
    if has_mesh_table is None:
        raise MeshLoadError(
            "Mesh spec key 'mesh' references a GeoPackage with missing table "
            f"'swe2d_baked_mesh': {gpkg_path!r}"
        )

    try:
        from swe2d.services.gpkg_persistence_service import load_baked_mesh
    except ImportError as exc:
        raise MeshLoadError(
            "Mesh spec key 'mesh' cannot load because required module "
            f"{(exc.name or 'swe2d.services.gpkg_persistence_service')!r} is unavailable"
        ) from exc

    try:
        result = load_baked_mesh(gpkg_path, mesh_name)
    except Exception as exc:
        raise MeshLoadError(
            f"Mesh spec key 'mesh' could not read table 'swe2d_baked_mesh' "
            f"from {gpkg_path!r}: {exc}"
        ) from exc
    if result is None:
        return None
    blob, db_crs_wkt = result

    try:
        from hydra_swe2d import swe2d_deserialize_mesh
    except ImportError as exc:
        raise MeshLoadError(
            "Mesh spec key 'mesh' cannot load because required module "
            f"{(exc.name or 'hydra_swe2d')!r} is unavailable"
        ) from exc

    try:
        pm = swe2d_deserialize_mesh(blob)
        # Prefer CRS from the BLOB (new format), fall back to DB column.
        crs_wkt = str(pm.crs_wkt or "").strip() or str(db_crs_wkt or "").strip()
        out = {
            "node_x": np.asarray(pm.node_x, dtype=np.float64),
            "node_y": np.asarray(pm.node_y, dtype=np.float64),
            "node_z": np.asarray(pm.node_z, dtype=np.float64),
            "cell_nodes": (
                np.asarray(pm.cell_face_nodes, dtype=np.int32)
                if pm.cell_face_nodes is not None
                else np.empty(0, dtype=np.int32)
            ),
            "crs_wkt": crs_wkt,
        }
        # BC type/values are not baked; only boundary topology is restored.
        if pm.edge_n0 is not None and pm.edge_n1 is not None:
            n0_all = np.asarray(pm.edge_n0, dtype=np.int32)
            n1_all = np.asarray(pm.edge_n1, dtype=np.int32)
            bc_all = (
                np.asarray(pm.edge_bc, dtype=np.int32)
                if pm.edge_bc is not None
                else np.zeros_like(n0_all, dtype=np.int32)
            )
            boundary_mask = bc_all != 0
            out["bc_edge_node0"] = n0_all[boundary_mask]
            out["bc_edge_node1"] = n1_all[boundary_mask]
        if pm.cell_face_offsets is not None:
            out["cell_face_offsets"] = np.asarray(pm.cell_face_offsets, dtype=np.int32)
            out["cell_face_nodes"] = (
                np.asarray(pm.cell_face_nodes, dtype=np.int32)
                if pm.cell_face_nodes is not None
                else np.empty(0, dtype=np.int32)
            )
        return out
    except Exception as exc:
        raise MeshLoadError(
            f"Mesh spec key 'mesh' contains a corrupt baked-mesh BLOB for "
            f"mesh {mesh_name!r} in {gpkg_path!r}: {exc}"
        ) from exc



_GPKG_ENV_SIZES = (0, 32, 48, 64)


def _geom_from_blob(raw: bytes):
    """Parse standard WKB from a GPKG geometry blob using GDAL/OGR.

    GPKG Binary header (OGC 12-128r12):
        GP (2) + version (1) + flags (1) + srs_id (4) + optional envelope
    Flags bits 1-2 encode envelope type:
        0=none, 1=xy(32B), 2=xyz(48B), 3=xyzm(64B)
    """
    if len(raw) < 5:
        return None
    if raw[:2] == b'GP':
        flags = raw[3]
        env_type = (flags >> 1) & 0x3
        offset = 8 + _GPKG_ENV_SIZES[env_type]
    elif raw[0] in (0, 1):
        offset = 0
    else:
        offset = 4
    wkb = raw[offset:]
    return ogr.CreateGeometryFromWkb(wkb)


def _parse_wkb_linestring(data) -> List[Tuple[float, float]]:
    """Parse a WKB LINESTRING from GPKG geometry blob using GDAL/OGR."""
    if data is None:
        return []
    raw = bytes(data)
    geom = _geom_from_blob(raw)
    if geom is None or geom.GetGeometryType() != ogr.wkbLineString:
        return []
    return [(geom.GetX(i), geom.GetY(i)) for i in range(geom.GetPointCount())]


def _parse_wkb_point(data) -> List[float]:
    """Parse a WKB POINT from GPKG geometry blob using GDAL/OGR."""
    if data is None:
        return [0.0, 0.0]
    raw = bytes(data)
    geom = _geom_from_blob(raw)
    if geom is None or geom.GetGeometryType() not in (ogr.wkbPoint, ogr.wkbPoint25D):
        return [0.0, 0.0]
    pt = geom.GetPoint()
    return [pt[0], pt[1]]


def _find_geom_column(table: str, conn: sqlite3.Connection) -> Optional[str]:
    """Return the name of the first geometry column in *table*, or None."""
    cur = conn.cursor()
    cur.execute(f'PRAGMA table_info("{table}")')
    for row in cur.fetchall():
        col_type = str(row[2]).upper()
        if col_type in ("POINT", "LINESTRING", "POLYGON", "MULTIPOINT",
                        "MULTILINESTRING", "MULTIPOLYGON", "GEOMETRYCOLLECTION"):
            return str(row[1])
    return None


def _parse_wkt_linestring_coords(wkt: str) -> List[Tuple[float, float]]:
    """Parse a WKT LINESTRING(x1 y1, x2 y2, ...) and return (x, y) vertex list."""
    wkt = wkt.strip()
    if "(" not in wkt:
        return []
    coords_str = wkt.split("(")[-1].split(")")[0]
    pairs = coords_str.split(",")
    coords = []
    for p in pairs:
        parts = p.strip().split()
        if len(parts) >= 2:
            coords.append((float(parts[0]), float(parts[1])))
    return coords




def query_hyetograph_rows(
    conn: sqlite3.Connection,
    hyetograph_table: str,
    hyetograph_id_field: str = "hyetograph_id",
    time_field: str = "Time",
    value_field: str = "Value",
    value_type_field: str = "value_type",
    units_field: str = "units",
) -> Dict[str, List[Dict[str, Any]]]:
    """Read hyetograph rows grouped by hyetograph_id.

    Returns dict mapping hyetograph_id -> list of row dicts for build_hyetograph().
    """
    cur = conn.cursor()
    cur.execute(f"SELECT DISTINCT \"{hyetograph_id_field}\" FROM \"{hyetograph_table}\"")
    ids = [r[0] for r in cur.fetchall()]
    result: Dict[str, List[Dict[str, Any]]] = {}
    for hid in ids:
        cur.execute(
            f"SELECT \"{time_field}\", \"{value_field}\", "
            f"\"{value_type_field}\", \"{units_field}\" "
            f"FROM \"{hyetograph_table}\" "
            f"WHERE \"{hyetograph_id_field}\" = ? ORDER BY rowid",
            (hid,),
        )
        rows = []
        for time_val, value_val, vt, u in cur.fetchall():
            rows.append({
                "Time": str(time_val),
                "Value": float(value_val),
                "value_type": str(vt),
                "units": str(u),
            })
        result[str(hid)] = rows
    return result


def query_gauge_layer(
    conn: sqlite3.Connection,
    gauge_table: str,
) -> List[Dict[str, Any]]:
    """Read gauge positions from a rain gage layer table.

    Expected schema (from schema_definitions.py):
        gage_id TEXT, hyetograph_id TEXT, geom POINT
    """
    cur = conn.cursor()
    cur.execute(
        'SELECT "gage_id", "hyetograph_id", "geom" '
        f'FROM "{gauge_table}" ORDER BY rowid'
    )
    result = []
    for r in cur.fetchall():
        xy = _parse_wkb_point(r[2])
        x_val, y_val = xy[0], xy[1]
        if x_val is None:
            continue
        result.append({
            "gauge_id": str(r[0]),
            "hyetograph_id": str(r[1] or r[0]),
            "x": x_val,
            "y": y_val,
        })
    return result


def query_cn_grid(
    conn: sqlite3.Connection,
    cn_table: str,
    cn_field: str = "cn",
    ia_ratio_field: str = "ia_ratio",
) -> Tuple[np.ndarray, float]:
    """Read per-cell curve number array and Ia ratio from a CN raster table."""
    cur = conn.cursor()
    try:
        cur.execute(f"SELECT \"{cn_field}\" FROM \"{cn_table}\" ORDER BY rowid")
        cn = np.array([float(r[0]) for r in cur.fetchall()], dtype=np.float64)
    except Exception as exc:
        from swe2d.core.builder import BuildRunContextError

        raise BuildRunContextError(
            f"spec key 'rain_cn' could not read field {cn_field!r} "
            f"from table {cn_table!r}: {exc}"
        ) from exc
    try:
        cur.execute(f"SELECT \"{ia_ratio_field}\" FROM \"{cn_table}\" LIMIT 1")
        row = cur.fetchone()
        ia_ratio = float(row[0]) if row else 0.2
    except Exception as exc:
        from swe2d.core.builder import BuildRunContextError

        raise BuildRunContextError(
            f"spec key 'rain_cn' could not read field {ia_ratio_field!r} "
            f"from table {cn_table!r}: {exc}"
        ) from exc
    return cn, ia_ratio



# delegates to the SAME ``build_*_qgis`` / ``build_*_from_layer`` logic the
# GUI dialog calls, so the CLI path stays byte-for-byte equivalent to the
# GUI path (only the layer source differs — file GPKG vs map canvas).
#
# This is the pattern ``boundary_qgis_adapter.apply_bc_layer_overrides_from_gpkg``
# already established (line 328).  Mirror it.


def _open_gpkg_layer(gpkg_path: str, table_name: str, name: str = "layer"):
    """Open a GPKG table as a ``QgsVectorLayer`` via the ogr provider.

    Returns ``None`` (with a log warning) if the layer cannot be loaded
    or the table name is empty.
    """
    if not gpkg_path or not table_name:
        return None
    try:
        from qgis.core import QgsVectorLayer
    except ImportError:
        return None
    uri = f"{gpkg_path}|layername={table_name}"
    layer = QgsVectorLayer(uri, name, "ogr")
    if not layer.isValid():
        return None
    return layer


def build_internal_flow_forcing_from_gpkg(
    *,
    gpkg_path: str,
    table_name: str,
    mesh_data: Dict[str, Any],
    requested_field_name: str = "src_value",
    hydrograph_table: str = "SWE2D_Hydrographs",
    log_fn: Optional[Callable] = None,
) -> Optional[Dict[str, Any]]:
    """Load internal flow source forcing from a GPKG polygon/line layer.

    Mirrors :func:`build_internal_flow_forcing_qgis` exactly: opens the
    polygon/line layer as ``QgsVectorLayer``, walks ``lyr.getFeatures()``,
    optionally resolves hydrograph references from ``SWE2D_Hydrographs``,
    and returns a ``{"base_q", "dynamic_terms", "layer_name"}`` dict.

    Returns ``None`` if the layer has no flow field, no geometry, or
    cannot be opened.
    """
    if log_fn is None:
        log_fn = logger.info
    layer = _open_gpkg_layer(gpkg_path, table_name, "internal_flow_sources")
    if layer is None:
        log_fn(f"Internal flow sources layer '{table_name}' not loaded from {gpkg_path}")
        return None

    fields = set(layer.fields().names())
    field_name = None
    for cand in (requested_field_name, "src_value", "q_cms", "flow_cms", "q", "flow"):
        if cand in fields:
            field_name = cand
            break
    if field_name is None:
        log_fn(f"Internal flow sources layer '{table_name}' missing flow field; skipping.")
        return None

    hydro_field = None
    for cand in ("hydrograph", "hydrograph_text", "hydro", "hg"):
        if cand in fields:
            hydro_field = cand
            break
    hgid_field = "hydrograph_id" if "hydrograph_id" in fields else None

    # Load the canonical hydrograph table if it exists in the same GPKG.
    hydro_layer = _open_gpkg_layer(gpkg_path, hydrograph_table, "swe2d_hydrographs") if hgid_field else None

    def _iter_project_layers_fn():
        if hydro_layer is not None and hydro_layer.isValid():
            return [hydro_layer]
        return []

    def _combo_layer_fn(combo, layer_type):
        return layer

    def _resolve_hydrograph_via_canonical(hid):
        if not hid or hydro_layer is None:
            return None
        from swe2d.boundary_and_forcing.hydrograph_logic import hydrograph_from_layer as _logic
        try:
            from qgis.core import QgsVectorLayer
            return _logic(hydro_layer, hydrograph_id=hid, bc_type=None,
                          parse_time_hours_fn=_parse_time_hours_native,
                          vector_layer_type=QgsVectorLayer)
        except Exception:
            return None

    from swe2d.mesh.mesh_runtime_logic import mesh_cell_centroids as _cctds
    cx_local, cy_local = _ctds_safe(mesh_data)

    def _geometry_to_indices_weights_qgis(geom, cx_l, cy_l):
        try:
            from qgis.core import QgsWkbTypes, QgsGeometry, QgsPointXY
        except ImportError:
            return None
        from swe2d.boundary_and_forcing.internal_flow_qgis_geometry import (
            internal_flow_geom_to_indices_weights_qgis as _logic,
        )
        return _logic(
            geom, cx_l, cy_l,
            qgs_wkb_types=QgsWkbTypes,
            qgs_geometry_cls=QgsGeometry,
            qgs_pointxy_cls=QgsPointXY,
        )

    from swe2d.core.internal_flow_qgis_adapter import (
        build_internal_flow_forcing_qgis as _gui_logic,
    )
    try:
        from qgis.core import (
            QgsVectorLayer as _QVL, QgsWkbTypes as _QWT,
            QgsGeometry as _QG, QgsPointXY as _QPXY,
        )
    except ImportError:
        return None

    return _gui_logic(
        mesh_data=mesh_data, have_qgis_core=True,
        internal_flow_layer_combo="__gpkg_layer__",
        combo_layer_fn=_combo_layer_fn,
        requested_field_name=field_name,
        iter_project_layers_fn=_iter_project_layers_fn,
        mesh_cell_centroids_fn=lambda: (cx_local, cy_local),
        parse_hydrograph_text_fn=_parse_hydrograph_text_native,
        hydrograph_from_layer_fn=lambda lyr, hydrograph_id="", bc_type=None:
            _hydrograph_from_layer_native(lyr, hydrograph_id=hydrograph_id, bc_type=bc_type),
        qgs_vector_layer_cls=_QVL,
        qgs_wkb_types=_QWT,
        qgs_geometry_cls=_QG,
        qgs_pointxy_cls=_QPXY,
        log_fn=log_fn,
    )


def _ctds_safe(mesh_data):
    from swe2d.mesh.mesh_runtime_logic import mesh_cell_centroids as _cctds
    cx, cy = _cctds(mesh_data)
    return np.asarray(cx, dtype=np.float64), np.asarray(cy, dtype=np.float64)


def build_thiessen_rain_cn_forcing_from_gpkg(
    *,
    gpkg_path: str,
    hyetograph_table: str,
    gauge_table: str,
    n_cells: int,
    mesh_data: Dict[str, Any],
    cn_table: Optional[str] = None,
    cn_field: str = "cn",
    infiltration_method: str = "scs_cn",
    ia_ratio: float = 0.2,
    use_spatial_rain_cn: bool = True,
    log_fn: Optional[Callable] = None,
) -> Any:
    """GPKG shim for ``build_thiessen_rain_cn_forcing_qgis``.

    Opens the gauge layer, hyetograph table, and optional CN grid from
    GeoPackage via ``QgsVectorLayer`` and delegates to the QGIS adapter
    with the same arguments the GUI dialog uses.

    NOTE: NOT dead.  Phase 3.6 mistakenly listed this function for deletion,
    but ``swe2d.core.builder.build_run_context`` (CLI builder) calls it for
    every spec that has a ``hyetograph`` block.  Restored here; if a future
    refactor wants to inline the body into ``builder.py``, delete it then.
    """
    if log_fn is None:
        log_fn = logger.info
    gauge_layer = _open_gpkg_layer(gpkg_path, gauge_table, "rain_gages")
    hyeto_layer = _open_gpkg_layer(gpkg_path, hyetograph_table, "hyetographs")
    if gauge_layer is None or hyeto_layer is None:
        return None
    cn_layer = (_open_gpkg_layer(gpkg_path, cn_table, "cn")
                if cn_table and use_spatial_rain_cn else None)

    try:
        from qgis.core import (
            QgsVectorLayer as _QVL, QgsGeometry as _QG,
            QgsPointXY as _QPXY, QgsWkbTypes as _QWT,
        )
    except ImportError:
        return None

    cell_centroids_fn = lambda: _ctds_safe(mesh_data)

    def boundary_buffer_cells_fn(n_rings):
        from swe2d.mesh.mesh_runtime_logic import boundary_buffer_cells as _logic
        return _logic(mesh_data, n_rings)

    def build_spatial_cn_array_fn():
        if cn_layer is None:
            return None
        from swe2d.core.spatial_forcing_qgis_adapter import (
            build_spatial_cn_array_qgis as _logic,
        )
        return _logic(
            mesh_data=mesh_data, have_qgis_core=True,
            cn_layer_combo="__gpkg_layer__",
            combo_layer_fn=lambda combo, layer_type: cn_layer,
            mesh_cell_centroids_fn=cell_centroids_fn,
            default_cn=80.0,
            qgs_geometry_cls=_QG, qgs_pointxy_cls=_QPXY,
            log_fn=log_fn,
        )

    from swe2d.core.spatial_forcing_qgis_adapter import (
        build_thiessen_rain_cn_forcing_qgis as _gui_logic,
    )
    from swe2d.boundary_and_forcing.rainfall_hydrology import (
        Gauge as _Gauge,
        ThiessenRainCNForcing as _ThiessenRainCNForcing,
        build_hyetograph as _build_hyetograph,
        assign_cells_to_nearest_gauge as _assign_cells_to_nearest_gauge,
        inspect_hyetograph_rows as _inspect_hyetograph_rows,
    )

    return _gui_logic(
        mesh_data=mesh_data, have_qgis_core=True,
        thiessen_rain_cn_forcing_cls=_ThiessenRainCNForcing,
        gauge_cls=_Gauge,
        build_hyetograph_fn=_build_hyetograph,
        assign_cells_to_nearest_gauge_fn=_assign_cells_to_nearest_gauge,
        inspect_hyetograph_rows_fn=_inspect_hyetograph_rows,
        use_spatial_rain_cn=bool(use_spatial_rain_cn) and (cn_layer is not None),
        rain_gage_layer_combo="__gpkg_layer__",
        hyetograph_layer_combo="__gpkg_layer__",
        storm_area_layer_combo=None,
        combo_layer_fn=lambda combo, layer_type: (
            gauge_layer if combo == "rain_gage" else hyeto_layer
        ),
        mesh_cell_centroids_fn=cell_centroids_fn,
        boundary_buffer_cells_fn=boundary_buffer_cells_fn,
        build_spatial_cn_array_fn=build_spatial_cn_array_fn,
        ia_ratio=float(ia_ratio),
        infiltration_method=str(infiltration_method),
        rain_boundary_buffer_rings=2,
        qgs_wkb_types=_QWT,
        qgs_geometry_cls=_QG,
        qgs_pointxy_cls=_QPXY,
        log_fn=log_fn,
    )


def _parse_time_hours_native(token):
    from swe2d.boundary_and_forcing.hydrograph_logic import parse_time_hours as _logic
    return _logic(token)


def _parse_hydrograph_text_native(text):
    from swe2d.boundary_and_forcing.hydrograph_logic import (
        parse_hydrograph_text as _logic,
    )
    return _logic(text, parse_time_hours_fn=_parse_time_hours_native)


def _hydrograph_from_layer_native(layer, hydrograph_id="", bc_type=None):
    from swe2d.boundary_and_forcing.hydrograph_logic import (
        hydrograph_from_layer as _logic,
    )
    try:
        from qgis.core import QgsVectorLayer
        return _logic(layer, hydrograph_id=hydrograph_id, bc_type=bc_type,
                      parse_time_hours_fn=_parse_time_hours_native,
                      vector_layer_type=QgsVectorLayer)
    except Exception:
        return None





def build_hydraulic_structure_config_from_gpkg(
    *,
    gpkg_path: str,
    structures_table: str,
    mesh_data: Dict[str, Any],
    log_fn: Optional[Callable] = None,
) -> Any:
    """GPKG shim for ``build_hydraulic_structure_config_from_layer``.

    Opens the structures GPKG layer and delegates to the same builder the
    GUI dialog calls.
    """
    if log_fn is None:
        log_fn = logger.info
    structures_layer = _open_gpkg_layer(gpkg_path, structures_table, "structures")
    if structures_layer is None:
        return None

    try:
        from swe2d.extensions.extension_models import (
            HydraulicStructureConfig as _HSConfig,
            StructureType as _SType,
            HydraulicStructure as _HS,
        )
    except ImportError:
        return None

    from swe2d.core.structure_config_service import (
        build_hydraulic_structure_config_from_layer as _gui_logic,
    )
    return _gui_logic(
        mesh_data=mesh_data, have_qgis_core=True,
        hydraulic_structure_config_cls=_HSConfig,
        structure_type_cls=_SType,
        hydraulic_structure_cls=_HS,
        structures_layer=structures_layer,
    )


def collect_bc_layer_hydrographs_from_gpkg(
    *,
    gpkg_path: str,
    bc_table: str,
    mesh_data: Dict[str, Any],
    edge_n0: np.ndarray,
    edge_n1: np.ndarray,
    hydrograph_table: str = "SWE2D_Hydrographs",
    log_fn: Optional[Callable] = None,
) -> Dict[Any, Tuple[int, Tuple[np.ndarray, np.ndarray]]]:
    """GPKG shim for ``collect_bc_layer_hydrographs_qgis``.

    Opens bc_lines + hydrographs as ``QgsVectorLayer`` and delegates to
    the same GUI logic.
    """
    if log_fn is None:
        log_fn = logger.info
    bc_layer = _open_gpkg_layer(gpkg_path, bc_table, "bc_lines")
    if bc_layer is None:
        return {}

    hydro_layer = _open_gpkg_layer(gpkg_path, hydrograph_table, "swe2d_hydrographs")

    try:
        from qgis.core import (
            QgsVectorLayer as _QVL, QgsGeometry as _QG, QgsPointXY as _QPXY,
        )
    except ImportError:
        return {}

    def _iter_project_layers_fn():
        return [hydro_layer] if hydro_layer is not None and hydro_layer.isValid() else []

    def _combo_layer_fn(combo, layer_type):
        return bc_layer if layer_type == "vector" else None

    from swe2d.core.boundary_qgis_adapter import (
        collect_bc_layer_hydrographs_qgis as _gui_logic,
    )
    from swe2d.core.constants_service import (
        BC_TS_FLOW, BC_TS_STAGE,
    )
    return _gui_logic(
        mesh_data=mesh_data, have_qgis_core=True,
        bc_lines_layer_combo="__gpkg_bc__",
        hydrograph_source_layer_combo="__gpkg_hydro__",
        combo_layer_fn=_combo_layer_fn,
        iter_project_layers_fn=_iter_project_layers_fn,
        hydrograph_from_layer_fn=lambda lyr, hydrograph_id="", bc_type=None:
            _hydrograph_from_layer_native(lyr, hydrograph_id=hydrograph_id, bc_type=bc_type),
        parse_hydrograph_text_fn=_parse_hydrograph_text_native,
        edge_n0=edge_n0, edge_n1=edge_n1,
        ts_flow_code=BC_TS_FLOW, ts_stage_code=BC_TS_STAGE,
        qgs_vector_layer_cls=_QVL, qgs_geometry_cls=_QG, qgs_pointxy_cls=_QPXY,
        log_fn=log_fn,
    )


def collect_bc_layer_edge_groups_from_gpkg(
    *,
    gpkg_path: str,
    bc_table: str,
    mesh_data: Dict[str, Any],
    edge_n0: np.ndarray,
    edge_n1: np.ndarray,
    log_fn: Optional[Callable] = None,
) -> Dict[int, str]:
    """GPKG shim for ``collect_bc_layer_edge_groups_qgis``."""
    if log_fn is None:
        log_fn = logger.info
    bc_layer = _open_gpkg_layer(gpkg_path, bc_table, "bc_lines")
    if bc_layer is None:
        return {}

    try:
        from qgis.core import (
            QgsGeometry as _QG, QgsPointXY as _QPXY,
        )
    except ImportError:
        return {}

    def _combo_layer_fn(combo, layer_type):
        return bc_layer if layer_type == "vector" else None

    from swe2d.core.boundary_qgis_adapter import (
        collect_bc_layer_edge_groups_qgis as _gui_logic,
    )
    return _gui_logic(
        mesh_data=mesh_data, have_qgis_core=True,
        bc_lines_layer_combo="__gpkg_bc__",
        combo_layer_fn=_combo_layer_fn,
        edge_n0=edge_n0, edge_n1=edge_n1,
        qgs_geometry_cls=_QG, qgs_pointxy_cls=_QPXY,
    )


def build_line_sampling_map_from_gpkg(
    *,
    gpkg_path: str,
    sample_lines_table: str,
    mesh_data: Dict[str, Any],
    log_fn: Optional[Callable] = None,
) -> List[Dict[str, Any]]:
    """GPKG adapter for the canonical plain-data sample-line service.

    Extracts each enabled sample feature as a plain record
    (``line_id``, ``line_name``, ``enabled``, ``points``) and feeds it
    directly to ``build_canonical_line_sampling_map`` alongside the
    canonical ``mesh_cell_records_from_mesh_data`` adapter output.
    No ``QgsGeometry`` polygon callback wiring — the tuple-vs-QGIS
    mismatch from spec §2/§12 is gone.

    Returns the canonical map list, or ``[]`` when the layer cannot be
    opened.  Malformed geometry raises a typed error identifying the
    table and feature ID (spec §10).
    """
    if log_fn is None:
        log_fn = logger.info
    layer = _open_gpkg_layer(gpkg_path, sample_lines_table, "sample_lines")
    if layer is None:
        return []

    from swe2d.core.builder import BuildRunContextError
    from swe2d.mesh.mesh_runtime_logic import (
        mesh_cell_records_from_mesh_data as _cell_records,
    )
    from swe2d.services.line_sampling_service import (
        build_canonical_line_sampling_map as _canonical_logic,
    )

    fields = set(layer.fields().names())
    id_field = "line_id" if "line_id" in fields else None
    name_field = "name" if "name" in fields else None
    enabled_field = "enabled" if "enabled" in fields else None

    sample_lines: List[Dict[str, Any]] = []
    for ft in layer.getFeatures():
        feat_id = ft.id()
        try:
            geom = ft.geometry()
        except Exception as exc:
            raise BuildRunContextError(
                f"spec key 'sample_lines' table {sample_lines_table!r} "
                f"feature {feat_id!r}: failed to read geometry: {exc}"
            ) from exc
        if geom is None or geom.isEmpty():
            continue
        try:
            n_points = list(geom.asPolyline())
            if not n_points:
                parts = geom.asMultiPolyline()
                n_points = list(parts[0]) if len(parts) == 1 else []
            if len(n_points) < 2:
                continue
            raw_pts = [
                (float(point.x()), float(point.y())) for point in n_points
            ]
        except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
            raise BuildRunContextError(
                f"spec key 'sample_lines' table {sample_lines_table!r} "
                f"feature {feat_id!r}: failed to extract line vertices: {exc}"
            ) from exc
        pts_arr = np.asarray(raw_pts, dtype=np.float64)
        try:
            line_id = int(ft[id_field]) if id_field is not None else int(feat_id)
        except Exception:
            line_id = int(feat_id)
        if name_field is not None:
            try:
                raw_name = ft[name_field]
            except Exception:
                raw_name = None
            line_name = str(raw_name) if raw_name not in (None, "") else ""
        else:
            line_name = ""
        enabled = True
        if enabled_field is not None:
            try:
                enabled = int(ft[enabled_field]) > 0
            except Exception as exc:
                raise BuildRunContextError(
                    f"spec key 'sample_lines' table {sample_lines_table!r} "
                    f"feature {feat_id!r}: invalid 'enabled' field: {exc}"
                ) from exc
        sample_lines.append({
            "line_id": line_id,
            "line_name": line_name,
            "enabled": bool(enabled),
            "points": pts_arr,
        })

    mesh_cells = _cell_records(mesh_data)

    try:
        return _canonical_logic(
            sample_lines=sample_lines,
            mesh_cells=mesh_cells,
        )
    except (TypeError, ValueError) as exc:
        raise BuildRunContextError(
            f"spec key 'sample_lines' table {sample_lines_table!r}: "
            f"invalid source geometry: {exc}"
        ) from exc









# ── QGIS-backed GPKG helpers ───────────────────────────────────────────────────
# These helpers open a GPKG table as a ``QgsVectorLayer`` and delegate to the
# same pure-logic builders the GUI calls.  They live in ``swe2d.core`` so the
# canonical RunContext builder can stay Qt-free while still supporting the
# drainage and structures data-sources blocks.


def _build_drainage_config_from_gpkg_layers(
    *,
    mesh_data: Dict[str, Any],
    drainage_gpkg: str,
    nodes_layer: str,
    links_layer: str,
    inlets_layer: Optional[str] = None,
    node_inlets_layer: Optional[str] = None,
    cell_min_bed: np.ndarray,
    gravity: float,
    config: Dict[str, Any],
    log_fn,
) -> Any:
    """Build drainage/pipe-network config from GPKG layers via QGIS ogr provider.

    This helper is intentionally in ``swe2d.core`` rather than the workbench
    adapter so the canonical builder can import it without depending on a
    GUI layer.  It is only called from ``build_run_context`` when the spec
    contains a ``drainage`` block.
    """
    try:
        from qgis.core import QgsVectorLayer
    except ImportError:
        log_fn("[Drainage] QGIS not available — cannot load drainage layers from GPKG")
        return None

    nl_uri = f"{drainage_gpkg}|layername={nodes_layer}"
    ll_uri = f"{drainage_gpkg}|layername={links_layer}"
    node_layer = QgsVectorLayer(nl_uri, "drain_nodes", "ogr")
    link_layer = QgsVectorLayer(ll_uri, "drain_links", "ogr")
    inlet_layer = None
    if inlets_layer:
        il_uri = f"{drainage_gpkg}|layername={inlets_layer}"
        inlet_layer = QgsVectorLayer(il_uri, "drain_inlets", "ogr")
    node_inlet_layer = None
    if node_inlets_layer:
        ni_uri = f"{drainage_gpkg}|layername={node_inlets_layer}"
        node_inlet_layer = QgsVectorLayer(ni_uri, "drain_node_inlets", "ogr")

    if not (node_layer.isValid() and link_layer.isValid()):
        log_fn(f"[Drainage] Drainage layers invalid in {drainage_gpkg}")
        return None

    from swe2d.mesh.mesh_runtime_logic import nearest_cell_index, mesh_cell_centroids
    cell_cx, cell_cy = mesh_cell_centroids(mesh_data)

    def _nearest_cell(x, y):
        return nearest_cell_index(x, y, cell_cx, cell_cy)

    from swe2d.core.pipe_network_service import (
        build_pipe_network_config as _build_pnc,
    )
    return _build_pnc(
        mesh_data=mesh_data,
        node_layer=node_layer,
        link_layer=link_layer,
        inlet_layer=inlet_layer,
        node_inlet_layer=node_inlet_layer,
        cell_min_bed=cell_min_bed,
        nearest_cell_fn=_nearest_cell,
        gravity=gravity,
        config=config,
        log_fn=log_fn,
    )
