"""GPKG adapter: read forcing data directly from GeoPackage without QGIS.

Each function mirrors a QGIS-layer-reader in the workbench but uses sqlite3
directly.  Returns the same Python objects (numpy arrays, ThiessenRainCNForcing,
etc.) so the existing runtime pipeline works unchanged.
"""
from __future__ import annotations

import json
import logging
import sqlite3
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
from swe2d.extensions.structures import StructureType


logger = logging.getLogger(__name__)


def query_mesh_from_gpkg(gpkg_path: str, mesh_name: str) -> Optional[Dict[str, np.ndarray]]:
    """Load mesh arrays from swe2d_mesh table (delegates to persistence service)."""
    from swe2d.services.gpkg_persistence_service import load_mesh_from_geopackage
    return load_mesh_from_geopackage(gpkg_path, mesh_name)


def query_bc_arrays(conn: sqlite3.Connection, bc_table: str) -> Dict[str, np.ndarray]:
    """Read boundary condition edge arrays from a BC lines layer table.

    Expects table with columns: node0 INTEGER, node1 INTEGER, bc_type INTEGER, bc_val REAL.
    Returns dict with keys: bc_edge_node0, bc_edge_node1, bc_edge_type, bc_edge_val.
    """
    cur = conn.cursor()
    cur.execute(f"SELECT node0, node1, bc_type, bc_val FROM \"{bc_table}\" ORDER BY rowid")
    rows = cur.fetchall()
    if not rows:
        return {}
    out = {
        "bc_edge_node0": np.array([r[0] for r in rows], dtype=np.int32),
        "bc_edge_node1": np.array([r[1] for r in rows], dtype=np.int32),
        "bc_edge_type": np.array([r[2] for r in rows], dtype=np.int32),
        "bc_edge_val": np.array([r[3] for r in rows], dtype=np.float64),
    }
    return out


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
    gauge_id_field: str = "gage_id",
    hyetograph_id_field: str = "hyetograph_id",
    x_field: str = "x",
    y_field: str = "y",
) -> List[Dict[str, Any]]:
    """Read gauge positions from a rain gage layer table."""
    cur = conn.cursor()
    cur.execute(
        f"SELECT \"{gauge_id_field}\", \"{hyetograph_id_field}\", "
        f"\"{x_field}\", \"{y_field}\" FROM \"{gauge_table}\" ORDER BY rowid"
    )
    return [
        {"gauge_id": str(r[0]), "hyetograph_id": str(r[1]), "x": float(r[2]), "y": float(r[3])}
        for r in cur.fetchall()
    ]


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
    structures_data: Optional[List[Dict[str, Any]]],
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
    gauge_id_field: str = "gage_id",
    x_field: str = "x",
    y_field: str = "y",
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
    gauge_rows = query_gauge_layer(
        conn, gauge_table, gauge_id_field=gauge_id_field,
        hyetograph_id_field=hyetograph_id_field,
        x_field=x_field, y_field=y_field,
    )
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
