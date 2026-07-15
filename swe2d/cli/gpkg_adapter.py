"""GPKG adapter: read forcing data directly from GeoPackage without QGIS.

Each function mirrors a QGIS-layer-reader in the workbench but uses sqlite3
directly.  Returns the same Python objects (numpy arrays, ThiessenRainCNForcing,
etc.) so the existing runtime pipeline works unchanged.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

from osgeo import ogr

import numpy as np

from swe2d.mesh.mesh_runtime_logic import mesh_cell_centroids

from swe2d.boundary_and_forcing.rainfall_hydrology import (
    Hyetograph,
    ThiessenRainCNForcing,
    build_hyetograph,
)



logger = logging.getLogger(__name__)


def query_mesh_from_gpkg(gpkg_path: str, mesh_name: str) -> Optional[Dict[str, np.ndarray]]:
    """Load mesh arrays from baked BLOB. Returns None if not found."""
    try:
        from swe2d.services.gpkg_persistence_service import load_baked_mesh
        blob = load_baked_mesh(gpkg_path, mesh_name)
        if blob is None:
            return None
        from hydra_swe2d import swe2d_deserialize_mesh
        pm = swe2d_deserialize_mesh(blob)
        out = {
            "node_x": np.asarray(pm.node_x, dtype=np.float64),
            "node_y": np.asarray(pm.node_y, dtype=np.float64),
            "node_z": np.asarray(pm.node_z, dtype=np.float64),
            "cell_nodes": np.asarray(pm.cell_face_nodes, dtype=np.int32) if pm.cell_face_nodes is not None else np.empty(0, dtype=np.int32),
        }
        # Extract the boundary edge topology (node pairs) from the BLOB.
        # BC type/values are NOT baked — they come from config
        # (default_bc_type + bc_lines override).
        if pm.edge_n0 is not None and pm.edge_n1 is not None:
            n0_all = np.asarray(pm.edge_n0, dtype=np.int32)
            n1_all = np.asarray(pm.edge_n1, dtype=np.int32)
            bc_all = np.asarray(pm.edge_bc, dtype=np.int32) if pm.edge_bc is not None else np.zeros_like(n0_all, dtype=np.int32)
            boundary_mask = bc_all != 0
            out["bc_edge_node0"] = n0_all[boundary_mask]
            out["bc_edge_node1"] = n1_all[boundary_mask]
        # Also read CRS from the baked mesh table
        try:
            _c = sqlite3.connect(gpkg_path)
            _r = _c.execute(
                "SELECT crs_wkt FROM swe2d_baked_mesh WHERE mesh_name=?", (mesh_name,)
            ).fetchone()
            if _r:
                out["crs_wkt"] = str(_r[0] or "")
            _c.close()
        except Exception:
            pass
        cfo = pm.cell_face_offsets
        if cfo is not None:
            out["cell_face_offsets"] = np.asarray(cfo, dtype=np.int32)
            out["cell_face_nodes"] = np.asarray(pm.cell_face_nodes, dtype=np.int32) if pm.cell_face_nodes is not None else np.empty(0, dtype=np.int32)
        return out
    except Exception:
        return None


def query_sample_lines_from_qgis(
    gpkg_path: str,
    table_name: str,
) -> List[Dict[str, Any]]:
    """Read sample-line features from a GPKG vector layer via sqlite3 + WKB.

    Pure sqlite3 — no QGIS dependency. Uses the same WKB parsing as BC lines.

    Returns a list of dicts, each with keys:
        line_id  — feature rowid (int)
        line_name — name field value or empty string
        line_xy  — (M, 2) float64 ndarray of vertex coordinates
    """
    if not gpkg_path or not os.path.exists(gpkg_path):
        return []
    conn = sqlite3.connect(gpkg_path)
    try:
        geom_col = _find_geom_column(table_name, conn)
        if not geom_col:
            logger.warning("No geometry column in sample-lines table '%s'", table_name)
            return []

        cols = [str(r[1]) for r in conn.execute(f'PRAGMA table_info("{table_name}")')]
        name_col = ""
        for c in cols:
            if c.lower() in ("name", "line_name", "label", "title"):
                name_col = c
                break

        sel = f'rowid, "{geom_col}"'
        if name_col:
            sel += f', "{name_col}"'
        rows = conn.execute(f'SELECT {sel} FROM "{table_name}" ORDER BY rowid').fetchall()
    finally:
        conn.close()

    lines: List[Dict[str, Any]] = []
    for row in rows:
        raw = row[1]
        coords = _parse_wkb_linestring(raw)
        if len(coords) < 2:
            continue
        xy = np.array(coords, dtype=np.float64)
        lname = ""
        if name_col and len(row) > 2 and row[2]:
            lname = str(row[2])
        lines.append({
            "line_id": int(row[0]),
            "line_name": lname,
            "line_xy": xy,
        })
    return lines


def query_bc_arrays(
    conn: sqlite3.Connection,
    bc_table: str,
    node_x: Optional[np.ndarray] = None,
    node_y: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Read boundary condition edge arrays from a BC table.

    First tries pre-split edge table format (columns: node0, node1, bc_type, bc_val).
    Falls back to geometry-based tables (LineString features), parsing WKB or WKT
    and snapping vertices to the nearest mesh node when *node_x*/*node_y* are given.

    Returns dict with keys: bc_edge_node0, bc_edge_node1,
    bc_edge_type, bc_edge_val.  Empty dict if nothing found.
    """
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info(\"{bc_table}\")")
    col_info = [(str(r[1]), str(r[2]).lower()) for r in cur.fetchall()]
    col_names = {c for c, _ in col_info}
    col_names_lower = {c.lower() for c, _ in col_info}

    # ── Path 1: Pre-split edge table ──────────────────────────────────
    if {"node0", "node1", "bc_type", "bc_val"}.issubset(col_names_lower):
        relax_col = None
        for cand in ("bc_relax", "open_bc_relax", "relax"):
            if cand in col_names_lower:
                relax_col = cand
                break
        cols = ["node0", "node1", "bc_type", "bc_val"]
        if relax_col:
            cols.append(relax_col)
        cur.execute(
            f'SELECT {", ".join(cols)} FROM "{bc_table}" ORDER BY rowid'
        )
        rows = cur.fetchall()
        if rows:
            out = {
                "bc_edge_node0": np.array([r[0] for r in rows], dtype=np.int32),
                "bc_edge_node1": np.array([r[1] for r in rows], dtype=np.int32),
                "bc_edge_type": np.array([r[2] for r in rows], dtype=np.int32),
                "bc_edge_val": np.array([r[3] for r in rows], dtype=np.float64),
            }
            if relax_col:
                out["bc_relax"] = np.array(
                    [0.0 if r[4] is None else float(r[4]) for r in rows],
                    dtype=np.float64,
                )
            return out

    # ── Path 2: Geometry-based table (LineString features) ────────────
    # Requires node_x/node_y for vertex-to-node snapping
    if node_x is None or node_y is None:
        return {}

    has_geom = any(c in ("geom", "wkb_geometry", "geometry", "shape") for c in col_names_lower)
    if not has_geom:
        return {}

    # Find geometry column - check column NAME (first element), not TYPE (second element)
    geom_col = next(c for c, _ in col_info if c.lower() in ("geom", "wkb_geometry", "geometry", "shape"))

    # Build kd-tree for nearest-node lookups
    from scipy.spatial import KDTree
    tree = KDTree(np.column_stack([np.asarray(node_x, dtype=np.float64),
                                    np.asarray(node_y, dtype=np.float64)]))

    # Find bc_type / bc_val columns - check column NAME (first element), not type (second)
    bc_col = next(c for c, _ in col_info if c.lower() in ("bc_type", "bc", "bctype", "boundary_type"))
    val_col = next(c for c, _ in col_info if c.lower() in ("bc_val", "bcvalue", "bc_value", "value", "val"))
    relax_col = next((c for c, _ in col_info if c.lower() in ("bc_relax", "open_bc_relax", "relax")), None)

    q_cols = f"\"{geom_col}\""
    if bc_col:
        q_cols += f", \"{bc_col}\""
    if val_col:
        q_cols += f", \"{val_col}\""
    if relax_col:
        q_cols += f", \"{relax_col}\""

    cur.execute(f"SELECT {q_cols} FROM \"{bc_table}\" ORDER BY rowid")

    all_n0, all_n1, all_type, all_val, all_relax = [], [], [], [], []
    for row in cur.fetchall():
        geom_raw = row[0]
        bt = int(row[1]) if bc_col and len(row) > 1 and row[1] is not None else 1
        bv = float(row[2]) if val_col and len(row) > 2 and row[2] is not None else 0.0
        br = 0.0
        if relax_col and len(row) > 3 and row[3] is not None:
            try:
                br = float(row[3])
            except Exception:
                br = 0.0

        coords = _parse_wkb_linestring(geom_raw)
        if not coords:
            coords = _parse_wkt_linestring_coords(str(geom_raw or ""))
        if len(coords) < 2:
            continue

        # Snap each vertex to nearest mesh node
        node_ids = [int(tree.query([x, y])[1]) for x, y in coords]
        for i in range(len(node_ids) - 1):
            all_n0.append(node_ids[i])
            all_n1.append(node_ids[i + 1])
            all_type.append(bt)
            all_val.append(bv)
            all_relax.append(br)

        # Close ring if start==end (LINESTRING that forms a loop)
        if len(node_ids) > 2 and node_ids[0] == node_ids[-1]:
            all_n0.append(node_ids[-2])
            all_n1.append(node_ids[-1])
            all_type.append(bt)
            all_val.append(bv)
            all_relax.append(br)

    if not all_n0:
        return {}

    return {
        "bc_edge_node0": np.array(all_n0, dtype=np.int32),
        "bc_edge_node1": np.array(all_n1, dtype=np.int32),
        "bc_edge_type": np.array(all_type, dtype=np.int32),
        "bc_edge_val": np.array(all_val, dtype=np.float64),
        "bc_relax": np.array(all_relax, dtype=np.float64),
    }


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
    except Exception as _e:
        logger.warning("query_cn_grid: could not read CN from table '%s': %s", cn_table, _e)
        cn = np.empty(0, dtype=np.float64)
    try:
        cur.execute(f"SELECT \"{ia_ratio_field}\" FROM \"{cn_table}\" LIMIT 1")
        row = cur.fetchone()
        ia_ratio = float(row[0]) if row else 0.2
    except Exception as _e:
        logger.warning("query_cn_grid: could not read ia_ratio from table '%s': %s", cn_table, _e)
        ia_ratio = 0.2
    return cn, ia_ratio



def build_forced_thiessen_from_gpkg(
    conn: sqlite3.Connection,
    n_cells: int,
    mesh_node_x: np.ndarray,
    mesh_node_y: np.ndarray,
    cell_nodes: np.ndarray,
    *,
    cell_face_offsets: Optional[np.ndarray] = None,
    hyetograph_table: str,
    gauge_table: str,
    cn_table: Optional[str] = None,
    cn_field: str = "cn",
    ia_ratio_field: str = "ia_ratio",
    hyetograph_id_field: str = "hyetograph_id",
    time_field: str = "Time",
    value_field: str = "Value",
    value_type_field: str = "value_type",
    units_field: str = "units",
    infiltration_method: str = "scs_cn",
) -> Optional[ThiessenRainCNForcing]:
    """Build ThiessenRainCNForcing directly from GPKG tables.

    Mirrors swe2d/boundary_and_forcing/spatial_forcing_qgis_adapter.py
    but reads from raw GPKG tables instead of QGIS vector layers.
    """
    gauge_rows = query_gauge_layer(conn, gauge_table)
    if not gauge_rows:
        return None

    hy_rows_by_id = query_hyetograph_rows(
        conn, hyetograph_table,
        hyetograph_id_field=hyetograph_id_field,
        time_field=time_field, value_field=value_field,
        value_type_field=value_type_field, units_field=units_field,
    )

    gauges = []
    hy_by_gauge_index: Dict[int, Hyetograph] = {}
    for gi, gr in enumerate(gauge_rows):
        hid = gr["hyetograph_id"]
        if hid not in hy_rows_by_id:
            continue
        hy = build_hyetograph(hy_rows_by_id[hid])
        gauges.append({
            "gauge_id": gr["gauge_id"],
            "x": gr["x"],
            "y": gr["y"],
            "hyetograph_id": hid,
        })
        hy_by_gauge_index[gi] = hy

    if not gauges:
        return None

    gx = np.array([g["x"] for g in gauges], dtype=np.float64)
    gy = np.array([g["y"] for g in gauges], dtype=np.float64)

    _md = {"node_x": mesh_node_x, "node_y": mesh_node_y}
    if cell_face_offsets is not None:
        _md["cell_face_offsets"] = cell_face_offsets
        _md["cell_face_nodes"] = cell_nodes
    else:
        _md["cell_nodes"] = cell_nodes
    _cx_raw, _cy_raw = mesh_cell_centroids(_md)
    cell_centroids = np.column_stack((_cx_raw, _cy_raw))
    n_cells_actual = min(cell_centroids.shape[0], n_cells)
    cx = cell_centroids[:n_cells_actual, 0]
    cy = cell_centroids[:n_cells_actual, 1]

    # Nearest-gauge assignment
    cell_to_gauge = np.full(n_cells_actual, -1, dtype=np.int32)
    for ci in range(n_cells_actual):
        dist = np.hypot(gx - cx[ci], gy - cy[ci])
        cell_to_gauge[ci] = int(np.argmin(dist))

    cn_arr, ia_ratio = query_cn_grid(
        conn, cn_table or "swe2d_rain_cn",
        cn_field=cn_field, ia_ratio_field=ia_ratio_field,
    ) if cn_table else (np.full(n_cells_actual, 75.0, dtype=np.float64), 0.2)

    if cn_arr.size != n_cells_actual:
        cn_arr = np.full(n_cells_actual, float(cn_arr.mean()) if cn_arr.size > 0 else 75.0, dtype=np.float64)

    return ThiessenRainCNForcing(
        cell_to_gauge=cell_to_gauge,
        gauge_hyetographs=hy_by_gauge_index,
        curve_number=cn_arr,
        ia_ratio=float(ia_ratio),
        infiltration_method=str(infiltration_method),
    )



def read_drainage_config_from_gpkg(
    conn: sqlite3.Connection,
    nodes_table: str,
    links_table: str,
    mesh_node_x: np.ndarray,
    mesh_node_y: np.ndarray,
    cell_nodes: np.ndarray,
    *,
    cell_face_offsets: Optional[np.ndarray] = None,
    inlets_table: Optional[str] = None,
    node_inlets_table: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Read drainage network topology from GPKG tables and compute mesh cell assignments.

    Returns an inline dict suitable for ``build_drainage_config_from_json``.
    Requires mesh topology so node->cell nearest-neighbor can be computed.
    """
    nodes_cur = conn.execute(f'PRAGMA table_info("{nodes_table}")').fetchall()
    links_cur = conn.execute(f'PRAGMA table_info("{links_table}")').fetchall()

    node_x_map: Dict[str, float] = {}
    node_y_map: Dict[str, float] = {}
    nodes_raw: List[Dict[str, Any]] = []
    _geom_col = _find_geom_column(nodes_table, conn)
    if _geom_col:
        for row in conn.execute(f'SELECT node_id, invert_elev, max_depth, rim_elev, crest_elev, node_type, surface_area, outfall_area, zero_storage, "{_geom_col}" FROM "{nodes_table}" ORDER BY rowid'):
            nid = str(row[0] or "")
            if not nid:
                continue
            coords = _parse_wkb_point(row[9])
            nx, ny = (coords[0], coords[1]) if len(coords) >= 2 else (0.0, 0.0)
            node_x_map[nid] = nx
            node_y_map[nid] = ny
            nodes_raw.append({
                "id": nid,
                "x": nx,
                "y": ny,
                "invert": float(row[1] or 0.0),
                "y_max": float(row[2] or 10.0),
                "rim_elev": float(row[3]) if row[3] is not None else None,
                "crest_elev": float(row[4]) if row[4] is not None else None,
                "type": str(row[5] or "junction").lower(),
                "surface_area": float(row[6]) if row[6] is not None else None,
                "outfall_area": float(row[7]) if row[7] is not None else None,
                "zero_storage": int(row[8]) if row[8] is not None else 0,
            })
    else:
        cols = {str(r[1]).lower() for r in nodes_cur}
        if "x" in cols and "y" in cols:
            for row in conn.execute(f'SELECT node_id, x, y, invert_elev, max_depth, rim_elev, crest_elev, node_type, surface_area, outfall_area, zero_storage FROM "{nodes_table}"'):
                nid = str(row[0] or "")
                if not nid:
                    continue
                node_x_map[nid] = float(row[1] or 0.0)
                node_y_map[nid] = float(row[2] or 0.0)
                nodes_raw.append({
                    "id": nid,
                    "x": float(row[1] or 0.0),
                    "y": float(row[2] or 0.0),
                    "invert": float(row[3] or 0.0),
                    "y_max": float(row[4] or 10.0),
                    "rim_elev": float(row[5]) if row[5] is not None else None,
                    "crest_elev": float(row[6]) if row[6] is not None else None,
                    "type": str(row[7] or "junction").lower(),
                    "surface_area": float(row[8]) if row[8] is not None else None,
                    "outfall_area": float(row[9]) if row[9] is not None else None,
                    "zero_storage": int(row[10]) if row[10] is not None else 0,
                })

    links_raw: List[Dict[str, Any]] = []
    for row in conn.execute(
        f'SELECT link_id, from_node, to_node, length, diameter, roughness_n, max_flow, link_type, '
        f'link_shape, span, rise, area_m2, equiv_diameter_m, '
        f'entrance_loss_k, exit_loss_k, max_cell_length '
        f'FROM "{links_table}"'
    ):
        fid = str(row[0] or f"link_{len(links_raw)}")
        link_entry: Dict[str, Any] = {
            "id": fid,
            "from": str(row[1] or ""),
            "to": str(row[2] or ""),
            "length": float(row[3] or 100.0),
            "diameter": float(row[4]) if row[4] is not None else 0.0,
            "roughness": float(row[5] or 0.013),
            "max_flow": float(row[6]) if row[6] is not None else -1.0,
            "link_type": str(row[7] or "conduit"),
            "link_shape": str(row[8] or "circular"),
            "span": float(row[9]) if row[9] is not None else None,
            "rise": float(row[10]) if row[10] is not None else None,
            "area_m2": float(row[11]) if row[11] is not None else None,
            "equiv_diameter_m": float(row[12]) if row[12] is not None else None,
            "entrance_loss_k": float(row[13]) if row[13] is not None else 0.5,
            "exit_loss_k": float(row[14]) if row[14] is not None else 1.0,
            "max_cell_length": float(row[15]) if row[15] is not None else 0.0,
        }
        links_raw.append(link_entry)

    if not nodes_raw or not links_raw:
        return None

    _md2 = {"node_x": mesh_node_x, "node_y": mesh_node_y}
    if cell_face_offsets is not None:
        _md2["cell_face_offsets"] = cell_face_offsets
        _md2["cell_face_nodes"] = cell_nodes
    else:
        _md2["cell_nodes"] = cell_nodes
    _cx2_raw, _cy2_raw = mesh_cell_centroids(_md2)
    cell_centroids = np.column_stack((_cx2_raw, _cy2_raw))

    from scipy.spatial import KDTree
    cell_tree = KDTree(cell_centroids)

    node_coords = np.column_stack([
        np.array([node_x_map[nid] for nid in node_x_map], dtype=np.float64),
        np.array([node_y_map[nid] for nid in node_y_map], dtype=np.float64),
    ])
    _, node_cell_arr = cell_tree.query(node_coords)
    node_cell: Dict[str, int] = {
        nid: int(node_cell_arr[i]) for i, nid in enumerate(node_x_map)
    }

    # Build lookup: node_id -> connecting link geometry (for outfall diameter derivation)
    _link_by_to_node: Dict[str, Dict[str, Any]] = {}
    for lk in links_raw:
        _link_by_to_node[lk["to"]] = lk

    outfalls: List[Dict[str, Any]] = []
    for n in nodes_raw:
        if n["type"] in ("outfall", "free_outfall"):
            # Outfalls are free-discharge boundaries; do not auto-couple them to a
            # surface cell even if the node is inside the 2D domain.
            cid = -1
            outfall_entry: Dict[str, Any] = {
                "outfall_id": n["id"],
                "cell_id": cid,
                "node_id": n["id"],
                "invert_elev": n["invert"],
            }
            if n.get("outfall_area") is not None:
                outfall_entry["area_m2"] = float(n["outfall_area"])
            elif n.get("zero_storage"):
                # Daylight outfall: derive orifice area from connecting pipe
                conn_link = _link_by_to_node.get(n["id"])
                if conn_link is not None:
                    d = float(conn_link.get("diameter") or 0.0)
                    if d <= 0.0:
                        # Try span/rise for box pipes
                        span = conn_link.get("span")
                        rise = conn_link.get("rise")
                        if span is not None and rise is not None and span > 0 and rise > 0:
                            outfall_entry["area_m2"] = float(span) * float(rise)
                        else:
                            a = conn_link.get("area_m2")
                            if a is not None and a > 0:
                                outfall_entry["area_m2"] = float(a)
                    else:
                        outfall_entry["diameter"] = float(d)
            outfalls.append(outfall_entry)

    inlets_raw: List[Dict[str, Any]] = []
    inlet_types_raw: List[Dict[str, Any]] = []
    node_inlets_raw: List[Dict[str, Any]] = []

    if inlets_table and node_inlets_table:
        conn.execute(f'PRAGMA table_info("{inlets_table}")')
        for row in conn.execute(
            f'SELECT inlet_type_id, name, inlet_type, '
            f'grate_length, grate_width, grate_type, grate_open_frac, '
            f'curb_length, curb_height, curb_throat, '
            f'slot_length, slot_width, '
            f'weir_length, orifice_area, coeff_weir, coeff_orifice, max_capture '
            f'FROM "{inlets_table}"'
        ):
            itid = str(row[0] or "")
            if not itid:
                continue
            inlet_types_raw.append({
                "inlet_type_id": itid,
                "name": str(row[1] or itid),
                "inlet_type": str(row[2] or "custom"),
                "grate_length": float(row[3]) if row[3] is not None else 0.0,
                "grate_width": float(row[4]) if row[4] is not None else 0.0,
                "grate_type": int(row[5]) if row[5] is not None else -1,
                "grate_open_frac": float(row[6]) if row[6] is not None else 1.0,
                "curb_length": float(row[7]) if row[7] is not None else 0.0,
                "curb_height": float(row[8]) if row[8] is not None else 0.0,
                "curb_throat": int(row[9]) if row[9] is not None else 0,
                "slot_length": float(row[10]) if row[10] is not None else 0.0,
                "slot_width": float(row[11]) if row[11] is not None else 0.0,
                "length": float(row[12] or 1.0),
                "area": float(row[13] or 0.0),
                "coeff_weir": float(row[14] or 1.70),
                "coeff_orifice": float(row[15] or 0.62),
                "max_capture": float(row[16]) if row[16] is not None else None,
            })

        # Build lookup from inlet_type_id to inlet type definition
        inlet_type_lookup: Dict[str, Dict[str, Any]] = {
            it["inlet_type_id"]: it for it in inlet_types_raw
        }

        conn.execute(f'PRAGMA table_info("{node_inlets_table}")')
        for row in conn.execute(f'SELECT node_id, inlet_type_id, inlet_count, crest_offset FROM "{node_inlets_table}"'):
            nid = str(row[0] or "")
            itid = str(row[1] or "")
            if not nid or not itid:
                continue
            it_def = inlet_type_lookup.get(itid, {})
            # Use rim_elev as crest_elev if available, otherwise 0
            rim = None
            for n in nodes_raw:
                if n["id"] == nid:
                    rim = n.get("rim_elev")
                    break
            inlet_entry: Dict[str, Any] = {
                "inlet_id": f"{nid}:{itid}",
                "cell_id": node_cell.get(nid, 0),
                "node_id": nid,
                "inlet_type_id": itid,
                "inlet_type": it_def.get("inlet_type", "custom"),
                "crest_elev": float(rim) if rim is not None else 0.0,
                "grate_length": it_def.get("grate_length", 0.0),
                "grate_width": it_def.get("grate_width", 0.0),
                "grate_type": it_def.get("grate_type", -1),
                "grate_open_frac": it_def.get("grate_open_frac", 1.0),
                "curb_length": it_def.get("curb_length", 0.0),
                "curb_height": it_def.get("curb_height", 0.0),
                "curb_throat": it_def.get("curb_throat", 0),
                "slot_length": it_def.get("slot_length", 0.0),
                "slot_width": it_def.get("slot_width", 0.0),
                "length": it_def.get("length", 1.0),
                "area": it_def.get("area", 0.0),
                "coeff_weir": it_def.get("coeff_weir", 1.70),
                "coeff_orifice": it_def.get("coeff_orifice", 0.62),
                "max_capture": it_def.get("max_capture"),
            }
            inlets_raw.append(inlet_entry)
            node_inlets_raw.append({
                "node_id": nid,
                "inlet_type_id": itid,
                "multiplier": float(row[2] or 1.0),
                "crest_offset": float(row[3] or 0.0),
            })

    return {
        "nodes": nodes_raw,
        "links": links_raw,
        "inlets": inlets_raw,
        "inlet_types": inlet_types_raw,
        "node_inlets": node_inlets_raw,
        "outfalls": outfalls,
    }


def load_and_configure_hydrographs(
    bc_conn: sqlite3.Connection,
    bc_table: str,
    hyd_conn: sqlite3.Connection,
    hyd_table: str,
    node_x: np.ndarray,
    node_y: np.ndarray,
    backend: Any,
    logger: Any,
) -> bool:
    """Read BC hydrographs from GPKG and configure them on the backend.

    Looks for BC lines in *bc_table* that have a non-null *hydrograph_id*,
    snaps them to mesh boundary edges, reads the time series from
    *hyd_table*, and calls ``backend.set_boundary_hydrographs_native()``.

    Returns True if any hydrographs were configured.
    """
    cur = bc_conn.cursor()
    cur.execute(f'PRAGMA table_info("{bc_table}")')
    col_info = [(str(r[1]), str(r[2]).lower()) for r in cur.fetchall()]
    col_names_lower = {c.lower() for c, _ in col_info}

    if "hydrograph_id" not in col_names_lower:
        return False

    # Identify geometry column
    geom_col = next((c for c, _ in col_info if c.lower() in ("geom", "wkb_geometry", "geometry", "shape")), None)
    if geom_col is None:
        return False

    # Read BC rows with a hydrograph_id
    cur.execute(f'SELECT "{geom_col}", "bc_type", "hydrograph_id" FROM "{bc_table}" '
                f'WHERE "hydrograph_id" IS NOT NULL AND "hydrograph_id" != \'\'')
    bc_rows = cur.fetchall()
    if not bc_rows:
        return False

    # Build KD-tree for node snapping
    from scipy.spatial import KDTree
    tree = KDTree(np.column_stack([np.asarray(node_x, dtype=np.float64),
                                    np.asarray(node_y, dtype=np.float64)]))

    # Read all hydrographs from the hydrograph table
    hcur = hyd_conn.cursor()
    hcur.execute(f'PRAGMA table_info("{hyd_table}")')
    hcols = [str(r[1]) for r in hcur.fetchall()]
    hcol_names_lower = {c.lower() for c in hcols}

    hg_col = next((c for c in hcols if c.lower() == "hydrograph_id"), None)
    time_col = next((c for c in hcols if c.lower() in ("time", "t", "time_s")), None)
    val_col = next((c for c in hcols if c.lower() in ("value", "val", "v", "flow", "q")), None)
    if not all([hg_col, time_col, val_col]):
        return False

    hcur.execute(f'SELECT "{hg_col}", "{time_col}", "{val_col}" FROM "{hyd_table}" '
                 f'ORDER BY "{hg_col}", "{time_col}"')
    hyd_rows = hcur.fetchall()
    if not hyd_rows:
        return False

    # Group hydrograph data by hydrograph_id
    hyd_series: Dict[str, list] = {}
    for row in hyd_rows:
        hid = str(row[0])
        t = float(row[1])
        v = float(row[2])
        hyd_series.setdefault(hid, []).append((t, v))

    # Map BC lines to boundary edges
    from swe2d.runtime.backend import SWE2DBackend
    edge_to_idx = getattr(backend, '_boundary_edge_index_by_nodes', {})
    if not edge_to_idx:
        logger.warning("[Hydrograph] No boundary edge index available on backend")
        return False

    hyd_edge_indices: list = []
    hyd_bc_types: list = []
    hyd_offsets: list = [0]
    hyd_times: list = []
    hyd_values: list = []

    for row in bc_rows:
        geom_raw = row[0]
        bc_type = int(row[1]) if row[1] is not None else 102
        hydro_id = str(row[2])

        series = hyd_series.get(hydro_id)
        if not series:
            logger.warning("[Hydrograph] No series found for hydrograph_id='%s'", hydro_id)
            continue

        coords = _parse_wkb_linestring(geom_raw)
        if not coords:
            coords = _parse_wkt_linestring_coords(str(geom_raw or ""))
        if len(coords) < 2:
            continue

        # Snap vertices to nearest mesh nodes
        node_ids = [int(tree.query([x, y])[1]) for x, y in coords]

        # For each segment (vertex pair), look up the boundary edge
        for i in range(len(node_ids) - 1):
            a, b = node_ids[i], node_ids[i + 1]
            key = (a, b) if a < b else (b, a)
            eidx = edge_to_idx.get(key)
            if eidx is None:
                continue

            hyd_edge_indices.append(eidx)
            hyd_bc_types.append(bc_type)
            n_pts = len(series)
            hyd_offsets.append(hyd_offsets[-1] + n_pts)
            for t, v in series:
                hyd_times.append(t)
                hyd_values.append(v)

    if not hyd_edge_indices:
        return False

    # Convert to arrays and call backend
    edge_arr = np.array(hyd_edge_indices, dtype=np.int32)
    type_arr = np.array(hyd_bc_types, dtype=np.int32)
    off_arr = np.array(hyd_offsets, dtype=np.int32)
    time_arr = np.array(hyd_times, dtype=np.float64)
    val_arr = np.array(hyd_values, dtype=np.float64)

    logger.info("[Hydrograph] Configuring %d hydrograph edges", len(edge_arr))
    backend.set_boundary_hydrographs_native(
        edge_index=edge_arr,
        bc_type=type_arr,
        offsets=off_arr,
        time_s=time_arr,
        value=val_arr,
    )
    return True


def load_hydrograph_edge_data(
    bc_conn: sqlite3.Connection,
    bc_table: str,
    hyd_conn: sqlite3.Connection,
    hyd_table: str,
    node_x: np.ndarray,
    node_y: np.ndarray,
    bc_n0: np.ndarray,
    bc_n1: np.ndarray,
    logger: Any = None,
) -> dict:
    """Read BC hydrographs from GPKG into edge_hydrographs dict format.

    Does NOT call any backend methods — returns data only.  The returned dict
    can be passed to ``RunContext.edge_hydrographs`` and will be handled by
    ``SWE2DNativeBoundaryHydrographConfigurator`` inside ``_execute()``.

    Returns ``edge_hydrographs``: ``{bc_edge_index: (bc_type, (times, values))}``
    where *bc_edge_index* is the position in the bc_n0/bc_n1 arrays.
    """
    _log = logger.info if logger else (lambda *a: None)
    cur = bc_conn.cursor()
    cur.execute(f'PRAGMA table_info("{bc_table}")')
    col_info = [(str(r[1]), str(r[2]).lower()) for r in cur.fetchall()]
    col_names_lower = {c.lower() for c, _ in col_info}

    if "hydrograph_id" not in col_names_lower:
        _log("[Hydrograph] No hydrograph_id column in bc_table")
        return {}

    geom_col = next((c for c, _ in col_info if c.lower() in ("geom", "wkb_geometry", "geometry", "shape")), None)
    if geom_col is None:
        _log("[Hydrograph] No geometry column found in bc_table")
        return {}

    cur.execute(f'SELECT "{geom_col}", "bc_type", "hydrograph_id" FROM "{bc_table}" '
                f'WHERE "hydrograph_id" IS NOT NULL AND "hydrograph_id" != \'\'')
    bc_rows = cur.fetchall()
    if not bc_rows:
        _log("[Hydrograph] No bc_lines with hydrograph_id")
        return {}

    # Build KD-tree for snapping bc_line vertices to mesh nodes
    from scipy.spatial import KDTree
    tree = KDTree(np.column_stack([np.asarray(node_x, dtype=np.float64),
                                    np.asarray(node_y, dtype=np.float64)]))

    # Build bc edge lookup: (min_node, max_node) → bc edge index
    _bc_n0 = np.asarray(bc_n0, dtype=np.int32).ravel()
    _bc_n1 = np.asarray(bc_n1, dtype=np.int32).ravel()
    bc_edge_map: Dict[tuple, int] = {}
    for ei in range(_bc_n0.size):
        a, b = int(_bc_n0[ei]), int(_bc_n1[ei])
        key = (a, b) if a < b else (b, a)
        bc_edge_map[key] = ei

    # Read all hydrographs from hydrograph table
    hcur = hyd_conn.cursor()
    hcur.execute(f'PRAGMA table_info("{hyd_table}")')
    hcols = [str(r[1]) for r in hcur.fetchall()]
    hcol_names_lower = {c.lower() for c in hcols}

    hg_col = next((c for c in hcols if c.lower() == "hydrograph_id"), None)
    time_col = next((c for c in hcols if c.lower() in ("time", "t", "time_s")), None)
    val_col = next((c for c in hcols if c.lower() in ("value", "val", "v", "flow", "q")), None)
    if not all([hg_col, time_col, val_col]):
        _log("[Hydrograph] Missing required columns in hydrograph table")
        return {}

    hcur.execute(f'SELECT "{hg_col}", "{time_col}", "{val_col}" FROM "{hyd_table}" '
                 f'ORDER BY "{hg_col}", "{time_col}"')
    hyd_rows = hcur.fetchall()
    if not hyd_rows:
        return {}

    # Group hydrograph data by hydrograph_id → (times, values)
    hyd_series: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for row in hyd_rows:
        hid = str(row[0])
        t = float(row[1])
        v = float(row[2])
        if hid not in hyd_series:
            hyd_series[hid] = ([], [])
        hyd_series[hid][0].append(t)
        hyd_series[hid][1].append(v)
    for hid in hyd_series:
        hyd_series[hid] = (
            np.array(hyd_series[hid][0], dtype=np.float64),
            np.array(hyd_series[hid][1], dtype=np.float64),
        )

    # Match bc_lines to bc edges and build edge_hydrographs
    edge_hydrographs: Dict[int, Tuple[int, Tuple[np.ndarray, np.ndarray]]] = {}
    for row in bc_rows:
        geom_raw = row[0]
        bc_type = int(row[1]) if row[1] is not None else 102
        hydro_id = str(row[2])

        series = hyd_series.get(hydro_id)
        if series is None:
            _log("[Hydrograph] No series found for hydrograph_id='%s'", hydro_id)
            continue

        coords = _parse_wkb_linestring(geom_raw)
        if not coords:
            coords = _parse_wkt_linestring_coords(str(geom_raw or ""))
        if len(coords) < 2:
            continue

        node_ids = [int(tree.query([x, y])[1]) for x, y in coords]

        for i in range(len(node_ids) - 1):
            a, b = node_ids[i], node_ids[i + 1]
            key = (a, b) if a < b else (b, a)
            eidx = bc_edge_map.get(key)
            if eidx is None:
                continue
            edge_hydrographs[int(eidx)] = (int(bc_type), series)
            break  # found a match for this bc_line, move to next

    _log("[Hydrograph] Built edge_hydrographs dict with %d entries", len(edge_hydrographs))
    return edge_hydrographs
