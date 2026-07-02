"""GPKG adapter: read forcing data directly from GeoPackage without QGIS.

Each function mirrors a QGIS-layer-reader in the workbench but uses sqlite3
directly.  Returns the same Python objects (numpy arrays, ThiessenRainCNForcing,
etc.) so the existing runtime pipeline works unchanged.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import struct
from typing import Any, Dict, List, Optional, Tuple

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
        # Also read CRS from the baked mesh table
        try:
            import sqlite3
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


def _parse_wkb_linestring(data) -> List[Tuple[float, float]]:
    """Parse a WKB LINESTRING from GPKG Extended WKB format.

    GPKG stores geometry as: 4-byte srs_id + standard WKB.
    Standard WKB LINESTRING: byte_order(1) + type(4) + n_points(4) + [x(8), y(8)] * n
    Returns list of (x, y) tuples, or empty list on any error.
    """
    if data is None:
        return []
    raw = bytes(data)
    if len(raw) < 9:
        return []

    # Skip 4-byte GPKG srs_id prefix
    wkb = raw[4:]
    if len(wkb) < 9:
        return []

    # Byte order
    bo = wkb[0]
    little = (bo == 1)
    # Geometry type
    gtype = int.from_bytes(wkb[1:5], byteorder='little' if little else 'big')
    if gtype != 2:  # 2 = LINESTRING
        return []

    if len(wkb) < 9:
        return []

    n_pts = int.from_bytes(wkb[5:9], byteorder='little' if little else 'big')
    n_pts = min(n_pts, 100000)  # sanity cap

    expected_len = 9 + n_pts * 16  # 8 bytes each for x and y
    if len(wkb) < expected_len:
        return []

    coords = []
    off = 9
    for _ in range(n_pts):
        x = struct.unpack_from('<d' if little else '>d', wkb, off)[0]
        y = struct.unpack_from('<d' if little else '>d', wkb, off + 8)[0]
        coords.append((x, y))
        off += 16

    return coords


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


def apply_bc_overrides_from_gpkg(
    conn: sqlite3.Connection,
    bc_table: str,
    edge_n0: np.ndarray,
    edge_n1: np.ndarray,
    bc_type_in: np.ndarray,
    bc_val_in: np.ndarray,
    node_x: Optional[np.ndarray] = None,
    node_y: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply boundary condition overrides from a GPKG BC lines table.

    Reads BC features from *bc_table*, splits multi-vertex LINESTRINGs
    into individual edge segments, matches each segment to a mesh boundary
    edge by comparing (min_node, max_node) keys (when pre-computed node
    columns exist) or by coordinate proximity (when geometry + node coords
    are provided).

    Parameters
    ----------
    conn : sqlite3.Connection
        Open connection to the BC GPKG.
    bc_table : str
        Table name containing BC features.
    edge_n0, edge_n1 : ndarray
        Mesh boundary edge node indices.
    bc_type_in, bc_val_in : ndarray
        Default BC types and values per edge.
    node_x, node_y : ndarray, optional
        Mesh node coordinates for coordinate-based matching.

    Returns
    -------
    Tuple[ndarray, ndarray]
        Updated (bc_type, bc_val) arrays.
    """
    bc_type_out = bc_type_in.copy()
    bc_val_out = bc_val_in.copy()

    # Probe table columns
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info(\"{bc_table}\")")
    cols = [str(r[1]) for r in cur.fetchall()]
    col_lower = [c.lower() for c in cols]
    if not cols:
        return bc_type_out, bc_val_out

    # Build column map
    col_map = {}
    for c, cl in zip(cols, col_lower):
        if cl in ("node0", "node_0", "start_node", "from_node"):
            col_map["node0"] = c
        elif cl in ("node1", "node_1", "end_node", "to_node"):
            col_map["node1"] = c
        elif cl in ("bc_type", "bc", "bctype", "boundary_type"):
            col_map["bc_type"] = c
        elif cl in ("bc_val", "bcvalue", "bc_value", "value", "val"):
            col_map["bc_val"] = c

    has_node_cols = "node0" in col_map and "node1" in col_map
    bc_col = col_map.get("bc_type", "")
    val_col = col_map.get("bc_val", "")

    # Build mesh edge lookup: (min_node, max_node) -> edge_index
    edge_n0_arr = np.asarray(edge_n0, dtype=np.int32).ravel()
    edge_n1_arr = np.asarray(edge_n1, dtype=np.int32).ravel()
    edge_key_to_idx: Dict[Tuple[int, int], int] = {}
    for i in range(edge_n0_arr.size):
        a, b = int(edge_n0_arr[i]), int(edge_n1_arr[i])
        key = (a, b) if a < b else (b, a)
        edge_key_to_idx[key] = i

    # Also build node coordinate lookup for coordinate matching
    node_x_arr = np.asarray(node_x, dtype=np.float64) if node_x is not None else None
    node_y_arr = np.asarray(node_y, dtype=np.float64) if node_y is not None else None

    def _find_nearest_node(x: float, y: float, tol: float = 0.1) -> int:
        """Find the nearest mesh node to (x, y) within tolerance."""
        if node_x_arr is None or node_y_arr is None:
            return -1
        dist = np.sqrt((node_x_arr - x) ** 2 + (node_y_arr - y) ** 2)
        idx = int(np.argmin(dist))
        return idx if dist[idx] <= tol else -1

    # Read features
    if has_node_cols:
        # Pre-split edge table: each row is one mesh edge with node0/node1
        n_col = col_map["node0"]
        n2_col = col_map["node1"]
        bt_col = bc_col or ""
        bv_col = val_col or ""
        query_cols = f"\"{n_col}\", \"{n2_col}\""
        if bt_col:
            query_cols += f", \"{bt_col}\""
        if bv_col:
            query_cols += f", \"{bv_col}\""
        cur.execute(f"SELECT {query_cols} FROM \"{bc_table}\" ORDER BY rowid")
        for row in cur.fetchall():
            n0, n1 = int(row[0]), int(row[1])
            key = (n0, n1) if n0 < n1 else (n1, n0)
            idx = edge_key_to_idx.get(key)
            if idx is not None:
                if bt_col and row[2] is not None:
                    bc_type_out[idx] = int(row[2])
                if bv_col and row[3] is not None:
                    bc_val_out[idx] = float(row[3])
    else:
        # Geometry table: read LINESTRINGs, split into vertex pairs
        geom_col = "geom"
        for c in cols:
            if c.lower() in ("wkb_geometry", "geometry", "shape"):
                geom_col = c
                break
        bt_col = bc_col or ""
        bv_col = val_col or ""
        query_cols = f"\"{geom_col}\""
        if bt_col:
            query_cols += f", \"{bt_col}\""
        if bv_col:
            query_cols += f", \"{bv_col}\""
        cur.execute(f"SELECT {query_cols} FROM \"{bc_table}\" ORDER BY rowid")
        for row in cur.fetchall():
            raw = str(row[0] or "")
            if not raw.startswith("LINESTRING") and not raw.startswith("LineString"):
                continue
            vertices = _parse_linestring_coords(raw)
            if len(vertices) < 2:
                continue
            feat_type = int(row[1]) if bt_col and row[1] is not None else 1
            feat_val = float(row[2]) if bv_col and row[2] is not None else 0.0
            for i in range(len(vertices) - 1):
                n0 = _find_nearest_node(vertices[i][0], vertices[i][1])
                n1 = _find_nearest_node(vertices[i + 1][0], vertices[i + 1][1])
                if n0 < 0 or n1 < 0:
                    continue
                key = (n0, n1) if n0 < n1 else (n1, n0)
                idx = edge_key_to_idx.get(key)
                if idx is not None:
                    bc_type_out[idx] = feat_type
                    bc_val_out[idx] = feat_val

    return bc_type_out, bc_val_out


def _parse_linestring_coords(wkt: str) -> List[Tuple[float, float]]:
    """Parse LINESTRING(x1 y1, x2 y2, ...) into list of (x, y) tuples."""
    wkt = wkt.strip()
    if "(" not in wkt:
        return []
    coords_str = wkt.split("(")[-1].split(")")[0]
    result = []
    for p in coords_str.split(","):
        parts = p.strip().split()
        if len(parts) >= 2:
            result.append((float(parts[0]), float(parts[1])))
    return result



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
        raw = str(r[2] or "")
        # Parse POINT(x y) from WKT
        x_val, y_val = None, None
        if "point" in raw.lower() and "(" in raw:
            inside = raw.split("(")[-1].split(")")[0]
            parts = inside.strip().split()
            if len(parts) >= 2:
                try:
                    x_val = float(parts[0])
                    y_val = float(parts[1])
                except (TypeError, ValueError):
                    pass
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
            node_id=str(i.get("node_id", "")),
            inlet_cell=int(i.get("inlet_cell", 0)),
            flow_rate=float(i.get("flow_rate", 0.0)),
        ) for i in inlets_raw
    ]

    inlet_assignments: List[NodeInletAssignment] = []
    for i_idx, inv in enumerate(inlets_raw):
        inlet_assignments.append(NodeInletAssignment(
            inlet_index=i_idx,
            inlet_type=InletType(int(inv.get("inlet_type", 0))),
        ))

    outfalls_raw = data.get("outfalls", [])
    outfalls: List[OutfallExchange] = [
        OutfallExchange(
            node_id=str(o.get("node_id", "")),
            outfall_invert_elev=float(o.get("invert", 0.0)),
        ) for o in outfalls_raw
    ]

    return PipeNetworkConfig(
        nodes=nodes,
        links=links,
        inlets=inlets,
        inlet_types=[InletType.DEFAULT] * len(inlets_raw) if inlets_raw else [],
        node_inlets=inlet_assignments,
        outfalls=outfalls,
        pipe_ends=[],
        gravity=float(data.get("gravity", 9.81)),
        head_deadband_m=float(data.get("head_deadband_m", 1.0e-3)),
        dynamic_flow_relaxation=float(data.get("dynamic_flow_relaxation", 1.0)),
        solver_mode=DrainageSolverMode(int(data.get("solver_mode", 0))),
        coupling_substeps=int(data.get("coupling_substeps", 1)),
    )


def build_structures_config_from_json(
    structures_data: Optional[Any],
    n_cells: int,
) -> Optional[HydraulicStructureConfig]:
    """Build HydraulicStructureConfig from a JSON array.

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
    if not isinstance(structures_data, list):
        raise TypeError(
            f"build_structures_config_from_json: expected a list of structure dicts, "
            f"got {type(structures_data).__name__}. Check that the 'structures' key "
            f"in your params JSON contains an array, not a bare string."
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

    return HydraulicStructureConfig(structures=structs)


def build_forced_thiessen_from_gpkg(
    conn: sqlite3.Connection,
    n_cells: int,
    mesh_node_x: np.ndarray,
    mesh_node_y: np.ndarray,
    cell_nodes: np.ndarray,
    *,
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

    cell_centroids = _compute_cell_centroids(mesh_node_x, mesh_node_y, cell_nodes)
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
    node_x: np.ndarray, node_y: np.ndarray, cell_nodes: np.ndarray
) -> np.ndarray:
    """Compute cell centroids from mesh topology."""
    tris = cell_nodes.reshape((-1, 3))
    cx = np.mean(node_x[tris], axis=1)
    cy = np.mean(node_y[tris], axis=1)
    return np.column_stack((cx, cy))
