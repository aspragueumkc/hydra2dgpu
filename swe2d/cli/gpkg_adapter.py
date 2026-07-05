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

from swe2d.boundary_and_forcing.rainfall_hydrology import (
    Hyetograph,
    ThiessenRainCNForcing,
    build_hyetograph,
)
from swe2d.extensions.extension_models import (
    DrainageSolverMode,
    HydraulicStructure,
    HydraulicStructureConfig,
    PipeNetworkConfig,
)
from swe2d.extensions.extension_models import StructureType


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
        # Include baked boundary-condition arrays if the mesh was baked with them.
        # Filter out INTERIOR edges (bc_type==0): the C++ BLOB serializes ALL
        # edges (interior + boundary), but backend.set_boundary_conditions only
        # accepts boundary edges. Re-applying interior edges fails the lookup
        # in _boundary_edge_index_by_nodes and aborts the run.
        if pm.edge_n0 is not None and pm.edge_n1 is not None:
            n0_all = np.asarray(pm.edge_n0, dtype=np.int32)
            n1_all = np.asarray(pm.edge_n1, dtype=np.int32)
            bc_all = np.asarray(pm.edge_bc, dtype=np.int32) if pm.edge_bc is not None else np.zeros_like(n0_all, dtype=np.int32)
            vl_all = np.asarray(pm.edge_bc_val, dtype=np.float64) if pm.edge_bc_val is not None else np.zeros_like(n0_all, dtype=np.float64)
            boundary_mask = bc_all != 0
            out["bc_edge_node0"] = n0_all[boundary_mask]
            out["bc_edge_node1"] = n1_all[boundary_mask]
            out["bc_edge_type"] = bc_all[boundary_mask]
            out["bc_edge_val"] = vl_all[boundary_mask]
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
        cur.execute(
            f"SELECT node0, node1, bc_type, bc_val FROM \"{bc_table}\" ORDER BY rowid"
        )
        rows = cur.fetchall()
        if rows:
            return {
                "bc_edge_node0": np.array([r[0] for r in rows], dtype=np.int32),
                "bc_edge_node1": np.array([r[1] for r in rows], dtype=np.int32),
                "bc_edge_type": np.array([r[2] for r in rows], dtype=np.int32),
                "bc_edge_val": np.array([r[3] for r in rows], dtype=np.float64),
            }

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

    q_cols = f"\"{geom_col}\""
    if bc_col:
        q_cols += f", \"{bc_col}\""
    if val_col:
        q_cols += f", \"{val_col}\""

    cur.execute(f"SELECT {q_cols} FROM \"{bc_table}\" ORDER BY rowid")

    all_n0, all_n1, all_type, all_val = [], [], [], []
    for row in cur.fetchall():
        geom_raw = row[0]
        bt = int(row[1]) if bc_col and len(row) > 1 and row[1] is not None else 1
        bv = float(row[2]) if val_col and len(row) > 2 and row[2] is not None else 0.0

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

        # Close ring if start==end (LINESTRING that forms a loop)
        if len(node_ids) > 2 and node_ids[0] == node_ids[-1]:
            all_n0.append(node_ids[-2])
            all_n1.append(node_ids[-1])
            all_type.append(bt)
            all_val.append(bv)

    if not all_n0:
        return {}

    return {
        "bc_edge_node0": np.array(all_n0, dtype=np.int32),
        "bc_edge_node1": np.array(all_n1, dtype=np.int32),
        "bc_edge_type": np.array(all_type, dtype=np.int32),
        "bc_edge_val": np.array(all_val, dtype=np.float64),
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


def build_drainage_config_from_json(
    drainage_data: Optional[Dict[str, Any]],
    n_cells: int,
) -> Optional[PipeNetworkConfig]:
    """Build PipeNetworkConfig from a JSON object.

    Expected format:
    {
        "gravity": 9.81,
        "head_deadband_m": 0.001,
        "dynamic_flow_relaxation": 1.0,
        "solver_mode": 0,
        "coupling_substeps": 1,
        "nodes": [
            {"id": "n1", "type": "inlet", "invert": 8.0, "y_max": 10.0, "area": 10.0,
             "surcharge_depth": 1.0, "initial_depth": 0.0}
        ],
        "links": [
            {"from": "n1", "to": "n2", "length": 100.0, "diameter": 1.0,
             "roughness": 0.013, "max_flow": -1.0}
        ],
        "inlets": [
            {"node_id": "n1", "inlet_cell": 100, "flow_rate": 0.5}
        ],
        "outfalls": [
            {"node_id": "n2", "invert": 3.0}
        ]
    }
    """
    if not drainage_data:
        return None

    from swe2d.extensions.extension_models import (
        DrainageNode,
        DrainageLink,
        InletExchange,
        InletType,
        NodeInletAssignment,
        OutfallExchange,
    )

    data = drainage_data
    nodes_raw = data.get("nodes", [])
    links_raw = data.get("links", [])
    if not nodes_raw or not links_raw:
        return None

    nodes: List[DrainageNode] = []
    for i, n in enumerate(nodes_raw):
        nid = str(n["id"])
        ntype = str(n.get("type", "junction")).lower()
        nodes.append(DrainageNode(
            node_id=nid,
            x=float(n.get("x", 0.0)),
            y=float(n.get("y", 0.0)),
            node_type=ntype,
            invert_elev=float(n.get("invert", 0.0)),
            max_depth=float(n.get("y_max", 10.0)),
            crest_elev=n.get("crest_elev"),
            rim_elev=n.get("rim_elev"),
        ))

    links: List[DrainageLink] = []
    for l in links_raw:
        links.append(DrainageLink(
            link_id=str(l.get("id", f"link_{len(links)}")),
            from_node_id=str(l["from"]),
            to_node_id=str(l["to"]),
            length=float(l.get("length", 100.0)),
            diameter=float(l.get("diameter", 1.0)),
            roughness_n=float(l.get("roughness", 0.013)),
            max_flow=float(l.get("max_flow", -1.0)),
        ))

    inlets_raw = data.get("inlets", [])
    inlets: List[InletExchange] = [
        InletExchange(
            inlet_id=str(i.get("inlet_id", f"in_{idx}")),
            cell_id=int(i.get("cell_id", i.get("inlet_cell", 0))),
            node_id=str(i.get("node_id", "")),
            crest_elev=float(i.get("crest_elev", 0.0)),
            length=float(i.get("length", 1.0)),
            area=float(i.get("area", 0.0)),
            coeff_weir=float(i.get("coeff_weir", 1.70)),
            coeff_orifice=float(i.get("coeff_orifice", 0.62)),
            max_capture=float(i["max_capture"]) if "max_capture" in i and i["max_capture"] is not None else None,
        )
        for idx, i in enumerate(inlets_raw)
    ]

    inlet_types: List[InletType] = [
        InletType(inlet_type_id=str(t.get("inlet_type_id", f"it_{idx}")))
        for idx, t in enumerate(data.get("inlet_types", []))
    ]

    node_inlets: List[NodeInletAssignment] = [
        NodeInletAssignment(
            node_id=str(n.get("node_id", "")),
            inlet_type_id=str(n.get("inlet_type_id", "")),
            multiplier=float(n.get("multiplier", 1.0)),
            crest_offset=float(n.get("crest_offset", 0.0)),
        )
        for n in data.get("node_inlets", [])
    ]

    outfalls_raw = data.get("outfalls", [])
    outfalls: List[OutfallExchange] = [
        OutfallExchange(
            outfall_id=str(o.get("outfall_id", f"out_{idx}")),
            cell_id=int(o.get("cell_id", 0)),
            node_id=str(o.get("node_id", "")),
            invert_elev=float(o.get("invert_elev", o.get("invert", 0.0))),
            area_m2=float(o.get("area_m2", 0.0)),
            diameter=float(o.get("diameter", 0.0)),
            coefficient=float(o.get("coefficient", 0.82)),
            max_flow=float(o["max_flow"]) if "max_flow" in o and o["max_flow"] is not None else None,
            zero_storage=bool(o.get("zero_storage", False)),
        )
        for idx, o in enumerate(outfalls_raw)
    ]

    return PipeNetworkConfig(
        nodes=nodes,
        links=links,
        inlets=inlets,
        inlet_types=inlet_types,
        node_inlets=node_inlets,
        outfalls=outfalls,
        pipe_ends=[],
        gravity=float(data.get("gravity", 9.81)),
        head_deadband_m=float(data.get("head_deadband_m", 1.0e-3)),
        dynamic_flow_relaxation=float(data.get("dynamic_flow_relaxation", 1.0)),
        pipe_solver_mode=str(data.get("pipe_solver_mode", "diffusion_wave")),
        coupling_substeps=int(data.get("coupling_substeps", 1)),
    )


def build_structures_config_from_json(
    structures_data: Optional[Any],
    n_cells: int,
) -> Optional[HydraulicStructureConfig]:
    """Build HydraulicStructureConfig from a JSON object or array.

    Supports two input forms:

    Form A — dict with metadata wrapper (recommended):
    {
        "enabled": true,
        "control_interval_s": 1.0,
        "controller_name": "none",
        "structures": [
            {"id": "s1", "type": "culvert", ...},
            ...
        ]
    }

    Note: ``gravity`` is not read from this dict — the coupling controller
    derives it from the mesh CRS via ``_u.gravity()``.

    Form B — bare list of structure entries:
    [
        {"id": "s1", "type": "culvert", ...},
        ...
    ]

    Each entry:
    {
        "id": "s1",
        "type": "culvert",
        "upstream_cell": 100,
        "downstream_cell": 101,
        "crest_elev": 5.0,
        "metadata": {"diameter": 1.0, "length": 20.0, ...}
    }
    """
    if not structures_data:
        return None

    _from_dict = False
    if isinstance(structures_data, dict):
        struct_list = structures_data.get("structures")
        if struct_list is None:
            return None
        _from_dict = True
        _enabled = bool(structures_data.get("enabled", True))
        _control_interval_s = float(structures_data.get("control_interval_s", 1.0))
        _controller_name = str(structures_data.get("controller_name", "none"))
        structures_data = struct_list

    if not isinstance(structures_data, list):
        raise TypeError(
            f"build_structures_config_from_json: expected a list of structure dicts, "
            f"got {type(structures_data).__name__}. "
            f"If passing a dict, it must contain a 'structures' key with the list."
        )

    type_map = {
        "weir": StructureType.WEIR,
        "culvert": StructureType.CULVERT,
        "gate": StructureType.GATE,
        "bridge": StructureType.BRIDGE,
        "pump": StructureType.PUMP,
    }

    structs = []
    for s in structures_data:
        stype = type_map.get(str(s.get("type", "")).lower(), StructureType.CULVERT)
        meta = dict(s.get("metadata", {}) or {})
        # Lift top-level keys into metadata for the coupling controller
        for k in ("diameter", "length", "width", "height", "roughness_n",
                   "coefficient", "cd", "opening", "max_flow",
                   "culvert_code", "culvert_shape", "culvert_rise", "culvert_span",
                   "culvert_area", "culvert_barrels", "culvert_slope",
                   "inlet_invert_elev", "outlet_invert_elev",
                   "entrance_loss_k", "exit_loss_k",
                   "embankment_enabled", "embankment_crest_elev",
                   "embankment_overflow_width", "embankment_weir_coeff",
                   "q_pump"):
            if k in s:
                meta[k] = s[k]
        structs.append(HydraulicStructure(
            structure_id=str(s.get("id", f"s_{len(structs)}")),
            structure_type=stype,
            upstream_cell=int(s.get("upstream_cell", 0)),
            downstream_cell=int(s.get("downstream_cell", 0)),
            crest_elev=float(s.get("crest_elev", 0.0)),
            metadata=meta,
        ))

    cfg = HydraulicStructureConfig(structures=structs)
    # Always enable when the user supplied a non-empty structure list.  Without
    # this, the bare-list form (Form B) leaves HydraulicStructureConfig.enabled
    # at its default False, and the runtime silently skips structure coupling.
    if _from_dict:
        # Dict form: respect the explicit ``enabled`` key (default True).
        cfg.enabled = _enabled and (len(structs) > 0)
        cfg.control_interval_s = _control_interval_s
        cfg.controller_name = _controller_name
    else:
        # Bare-list form: enable whenever structures were supplied.
        cfg.enabled = len(structs) > 0
    return cfg


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

    cell_centroids = _compute_cell_centroids(mesh_node_x, mesh_node_y, cell_nodes, cell_face_offsets)
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


def _compute_cell_centroids(
    node_x: np.ndarray, node_y: np.ndarray, cell_nodes: np.ndarray,
    cell_face_offsets: Optional[np.ndarray] = None
) -> np.ndarray:
    """Compute cell centroids from mesh topology.

    Handles both uniform (all triangles/quads) and mixed meshes.
    For uniform triangle meshes, cell_face_offsets is not needed.
    For mixed meshes, cell_face_offsets defines start indices for each cell.
    """
    if cell_face_offsets is not None and len(cell_face_offsets) > 0:
        # Mixed mesh: use offsets to extract variable-length node lists
        n_cells = len(cell_face_offsets)
        centroids = np.empty((n_cells, 2), dtype=np.float64)
        for i in range(n_cells):
            start = cell_face_offsets[i]
            end = cell_face_offsets[i + 1] if i + 1 < len(cell_face_offsets) else len(cell_nodes)
            nodes_i = cell_nodes[start:end]
            if len(nodes_i) > 0:
                centroids[i, 0] = np.mean(node_x[nodes_i])
                centroids[i, 1] = np.mean(node_y[nodes_i])
            else:
                centroids[i, 0] = 0.0
                centroids[i, 1] = 0.0
        return centroids
    else:
        # Uniform mesh: all cells have same node count
        total = len(cell_nodes)
        # Check quads first (more common in structured meshes) then triangles
        nodes_per_cell = 4 if total % 4 == 0 else 3
        n_cells = total // nodes_per_cell
        cell_nodes_2d = cell_nodes.reshape((n_cells, nodes_per_cell))
        cx = np.mean(node_x[cell_nodes_2d], axis=1)
        cy = np.mean(node_y[cell_nodes_2d], axis=1)
        return np.column_stack((cx, cy))


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
        for row in conn.execute(f'SELECT node_id, invert_elev, max_depth, rim_elev, crest_elev, node_type, "{_geom_col}" FROM "{nodes_table}" ORDER BY rowid'):
            nid = str(row[0] or "")
            if not nid:
                continue
            coords = _parse_wkb_point(row[6])
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
            })
    else:
        cols = {str(r[1]).lower() for r in nodes_cur}
        if "x" in cols and "y" in cols:
            for row in conn.execute(f'SELECT node_id, x, y, invert_elev, max_depth, rim_elev, crest_elev, node_type FROM "{nodes_table}"'):
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
                })

    links_raw: List[Dict[str, Any]] = []
    for row in conn.execute(f'SELECT link_id, from_node, to_node, length, diameter, roughness_n, max_flow, link_type FROM "{links_table}"'):
        fid = str(row[0] or f"link_{len(links_raw)}")
        links_raw.append({
            "id": fid,
            "from": str(row[1] or ""),
            "to": str(row[2] or ""),
            "length": float(row[3] or 100.0),
            "diameter": float(row[4] or 1.0),
            "roughness": float(row[5] or 0.013),
            "max_flow": float(row[6]) if row[6] is not None else -1.0,
        })

    if not nodes_raw or not links_raw:
        return None

    cell_centroids = _compute_cell_centroids(mesh_node_x, mesh_node_y, cell_nodes, cell_face_offsets)
    n_cells_actual = min(cell_centroids.shape[0], len(nodes_raw) * 10 + 1)
    ccx = cell_centroids[:n_cells_actual, 0]
    ccy = cell_centroids[:n_cells_actual, 1]

    node_cell: Dict[str, int] = {}
    for nid, nx in node_x_map.items():
        ny = node_y_map[nid]
        dist = np.hypot(ccx - nx, ccy - ny)
        node_cell[nid] = int(np.argmin(dist))

    outfalls: List[Dict[str, Any]] = []
    for n in nodes_raw:
        if n["type"] in ("outfall", "free_outfall"):
            cid = node_cell.get(n["id"], 0)
            outfalls.append({
                "outfall_id": n["id"],
                "cell_id": cid,
                "node_id": n["id"],
                "invert_elev": n["invert"],
            })

    inlets_raw: List[Dict[str, Any]] = []
    inlet_types_raw: List[Dict[str, Any]] = []
    node_inlets_raw: List[Dict[str, Any]] = []

    if inlets_table and node_inlets_table:
        conn.execute(f'PRAGMA table_info("{inlets_table}")')
        for row in conn.execute(f'SELECT inlet_type_id, name, weir_length, orifice_area, coeff_weir, coeff_orifice, max_capture FROM "{inlets_table}"'):
            itid = str(row[0] or "")
            if not itid:
                continue
            inlet_types_raw.append({
                "inlet_type_id": itid,
                "name": str(row[1] or itid),
                "length": float(row[2] or 1.0),
                "area": float(row[3] or 0.0),
                "coeff_weir": float(row[4] or 1.70),
                "coeff_orifice": float(row[5] or 0.62),
                "max_capture": float(row[6]) if row[6] is not None else None,
            })

        conn.execute(f'PRAGMA table_info("{node_inlets_table}")')
        for row in conn.execute(f'SELECT node_id, inlet_type_id, inlet_count, crest_offset FROM "{node_inlets_table}"'):
            nid = str(row[0] or "")
            itid = str(row[1] or "")
            if not nid or not itid:
                continue
            inlets_raw.append({
                "inlet_id": f"{nid}:{itid}",
                "cell_id": node_cell.get(nid, 0),
                "node_id": nid,
                "inlet_type_id": itid,
                "crest_elev": 0.0,
            })
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
