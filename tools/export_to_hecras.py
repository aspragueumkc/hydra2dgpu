#!/usr/bin/env python3
"""Export the most recent HYDRA2D run from a GPKG to HEC-RAS 7.0 format.

Pulls drainage geometry from the source project GPKG.

Usage:
    python tools/export_to_hecras.py <path/to/results.gpkg> [source_project.gpkg] [output_dir]
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from osgeo import ogr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "build"))

from swe2d.services.hecras_model_export_service import (
    Hecras2DFlowArea,
    HecrasBCLine,
    HecrasBoundary,
    HecrasCulvertGroup,
    HecrasPipeConduit,
    HecrasPipeNode,
    HecrasPlanParams,
    HecrasSA2DConnection,
    HecrasTopInlet,
    write_hecras_project,
)


# ---------------------------------------------------------------------------
# Mesh loading
# ---------------------------------------------------------------------------


def load_mesh(gpkg_path: str, mesh_name: str) -> Optional[Dict[str, np.ndarray]]:
    from swe2d.services.gpkg_persistence_service import load_baked_mesh
    blob = load_baked_mesh(gpkg_path, mesh_name)
    if blob is None:
        return None
    from hydra_swe2d import swe2d_deserialize_mesh
    pm = swe2d_deserialize_mesh(blob)
    out: Dict[str, np.ndarray] = {
        "node_x": np.asarray(pm.node_x, dtype=np.float64),
        "node_y": np.asarray(pm.node_y, dtype=np.float64),
        "node_z": np.asarray(pm.node_z, dtype=np.float64),
    }
    if pm.cell_face_offsets is not None:
        out["cell_face_offsets"] = np.asarray(pm.cell_face_offsets, dtype=np.int32)
        out["cell_face_nodes"] = np.asarray(pm.cell_face_nodes, dtype=np.int32)
    if pm.edge_n0 is not None and pm.edge_n1 is not None:
        n0_all = np.asarray(pm.edge_n0, dtype=np.int32)
        n1_all = np.asarray(pm.edge_n1, dtype=np.int32)
        bc_all = np.asarray(pm.edge_bc, dtype=np.int32) if pm.edge_bc is not None else np.zeros_like(n0_all, dtype=np.int32)
        vl_all = np.asarray(pm.edge_bc_val, dtype=np.float64) if pm.edge_bc_val is not None else np.zeros_like(n0_all, dtype=np.float64)
        mask = bc_all != 0
        out["bc_edge_node0"] = n0_all[mask]
        out["bc_edge_node1"] = n1_all[mask]
        out["bc_edge_type"] = bc_all[mask]
        out["bc_edge_val"] = vl_all[mask]
    return out


# ---------------------------------------------------------------------------
# Perimeter and cell centers
# ---------------------------------------------------------------------------


def extract_perimeter(mesh_data: dict) -> np.ndarray:
    n0 = mesh_data.get("bc_edge_node0")
    n1 = mesh_data.get("bc_edge_node1")
    nx = mesh_data["node_x"]
    ny = mesh_data["node_y"]
    if n0 is None or n1 is None or len(n0) == 0:
        from scipy.spatial import ConvexHull
        points = np.column_stack([nx, ny])
        hull = ConvexHull(points)
        return points[hull.vertices]

    # Build adjacency graph from boundary edges
    from collections import defaultdict
    adj: Dict[int, List[int]] = defaultdict(list)
    for a, b in zip(n0, n1):
        ai, bi = int(a), int(b)
        adj[ai].append(bi)
        adj[bi].append(ai)

    # Find connected components, pick the largest (exterior boundary)
    visited: set = set()
    components: List[List[int]] = []
    for start in adj:
        if start in visited:
            continue
        stack = [start]
        comp: List[int] = []
        while stack:
            n = stack.pop()
            if n in visited:
                continue
            visited.add(n)
            comp.append(n)
            stack.extend(adj[n])
        if len(comp) > 2:
            components.append(comp)

    if not components:
        from scipy.spatial import ConvexHull
        points = np.column_stack([nx, ny])
        hull = ConvexHull(points)
        return points[hull.vertices]

    # Take the largest component
    bound_nodes = max(components, key=len)

    # Walk the chain using proper adjacency (all degree-2 on boundary)
    start = bound_nodes[0]
    ring = [start]
    walked = {start}
    current = start
    prev = -1
    while True:
        neighbors = [n for n in adj[current] if n != prev]
        if not neighbors:
            break
        nxt = neighbors[0]
        if nxt in walked:
            # Check if this closes the ring back to start
            if nxt == start and len(ring) > 2:
                break
            # Otherwise pick the other neighbor if available
            alt = [n for n in adj[current] if n not in walked]
            if not alt:
                break
            nxt = alt[0]
        walked.add(nxt)
        ring.append(int(nxt))
        prev, current = current, nxt

    ring_np = np.array(ring, dtype=np.int32)
    return np.column_stack([nx[ring_np], ny[ring_np]])


def extract_cell_centers(mesh_data: dict) -> np.ndarray:
    nx = mesh_data["node_x"]
    ny = mesh_data["node_y"]
    if "cell_face_offsets" in mesh_data:
        off = mesh_data["cell_face_offsets"]
        cfn = mesh_data["cell_face_nodes"]
        n_cells = len(off) - 1
        centers = np.zeros((n_cells, 2), dtype=np.float64)
        for i in range(n_cells):
            s, e = int(off[i]), int(off[i + 1])
            ring = cfn[s:e]
            centers[i, 0] = float(np.mean(nx[ring]))
            centers[i, 1] = float(np.mean(ny[ring]))
        return centers
    cn = mesh_data.get("cell_nodes", np.empty((0, 3), dtype=np.int32))
    if len(cn) == 0:
        return np.zeros((0, 2))
    return np.column_stack([np.mean(nx[cn], axis=1), np.mean(ny[cn], axis=1)])


# ---------------------------------------------------------------------------
# BC classification
# ---------------------------------------------------------------------------


def classify_bc_sides(
    mesh_data: dict,
) -> Tuple[List[HecrasBCLine], List[HecrasBoundary]]:
    n0 = mesh_data.get("bc_edge_node0", np.array([], dtype=np.int32))
    n1 = mesh_data.get("bc_edge_node1", np.array([], dtype=np.int32))
    bt = mesh_data.get("bc_edge_type", np.array([], dtype=np.int32))
    bv = mesh_data.get("bc_edge_val", np.array([], dtype=np.float64))
    nx = mesh_data["node_x"]
    ny = mesh_data["node_y"]

    if len(n0) == 0:
        return [], []

    bc_lines_list: List[HecrasBCLine] = []
    boundaries: List[HecrasBoundary] = []
    type_names = {2: "Inflow", 3: "Stage", 4: "Outlet", 6: "Outflow"}

    for bc_type in sorted(set(bt)):
        mask = bt == bc_type
        e_n0 = n0[mask]
        e_n1 = n1[mask]
        e_val = bv[mask]
        coords_set: set = set()
        coords_list: List[Tuple[float, float]] = []
        for a, b in zip(e_n0, e_n1):
            for idx in (int(a), int(b)):
                coord = (float(nx[idx]), float(ny[idx]))
                if coord not in coords_set:
                    coords_set.add(coord)
                    coords_list.append(coord)
        if len(coords_list) < 2:
            continue
        coords = np.array(coords_list, dtype=np.float64)
        ordered = [coords[0]]
        remaining = list(range(1, len(coords)))
        while remaining:
            last = ordered[-1]
            dists = [np.hypot(coords[i, 0] - last[0], coords[i, 1] - last[1]) for i in remaining]
            nearest = remaining[int(np.argmin(dists))]
            ordered.append(coords[nearest])
            remaining.remove(nearest)
        ordered_arr = np.array(ordered, dtype=np.float64)

        name = type_names.get(bc_type, f"BC_{bc_type}")
        bc_lines_list.append(
            HecrasBCLine(name=name, storage_area="Perimeter 1", coordinates=ordered_arr)
        )

        flow_type = "Flow Hydrograph"
        friction_slope = 0.001
        hydro_values = None
        if bc_type == 2:
            avg_val = float(np.mean(e_val))
            if avg_val > 0:
                hydro_values = [avg_val, avg_val, avg_val, avg_val, avg_val]
            else:
                flow_type = "Normal Depth"
                friction_slope = 0.001
        elif bc_type == 3:
            flow_type = "Stage Hydrograph"
            avg_val = float(np.mean(e_val))
            hydro_values = [avg_val, avg_val, avg_val, avg_val, avg_val]
        elif bc_type in (6, 4):
            flow_type = "Normal Depth"
            friction_slope = abs(float(np.mean(e_val))) or 0.001
        else:
            continue

        boundaries.append(
            HecrasBoundary(
                bc_line_name=name,
                area_2d="Perimeter 1",
                flow_type=flow_type,
                hydrograph_values=hydro_values,
                friction_slope=friction_slope,
                interval="1HOUR",
            )
        )

    return bc_lines_list, boundaries


# ---------------------------------------------------------------------------
# Drainage nodes from source GPKG
# ---------------------------------------------------------------------------


def load_drainage_nodes(source_gpkg: str) -> Dict[str, HecrasPipeNode]:
    """Load drainage node points from source project GPKG."""
    nodes: Dict[str, HecrasPipeNode] = {}
    ds = ogr.Open(source_gpkg)
    layer = ds.GetLayerByName("SWE2D_Drainage_Nodes")
    if layer is None:
        return nodes
    for feat in layer:
        nid = str(feat.GetField("node_id"))
        g = feat.GetGeometryRef()
        node_type_map = {
            "inlet": "Junction",
            "pipe_end": "Culvert Opening",
            "outfall": "Culvert Opening",
            "junction": "Junction",
        }
        invert = float(feat.GetField("invert_elev") or 0.0)
        depth = float(feat.GetField("max_depth") or 10.0)
        rim = feat.GetField("rim_elev")
        crest = feat.GetField("crest_elev")
        ntype = str(feat.GetField("node_type") or "junction")

        nodes[nid] = HecrasPipeNode(
            name=nid[:6],
            node_type=node_type_map.get(ntype, "Junction"),
            node_status=f"{ntype.upper()} junction",
            invert_elevation=float(invert),
            depth=float(depth),
            terrain_elevation=float(rim) if rim else float(invert) + float(depth),
            x=float(g.GetX()),
            y=float(g.GetY()),
            base_area=float(feat.GetField("surface_area") or 25.0) or 25.0,
            total_connections=0,
        )
    ds = None
    return nodes


def load_drainage_links(source_gpkg: str) -> List[HecrasPipeConduit]:
    """Load drainage link polylines from source project GPKG."""
    conduits: List[HecrasPipeConduit] = []
    ds = ogr.Open(source_gpkg)
    layer = ds.GetLayerByName("SWE2D_Drainage_Links")
    if layer is None:
        return conduits
    for feat in layer:
        lid = str(feat.GetField("link_id"))
        from_node = str(feat.GetField("from_node"))
        to_node = str(feat.GetField("to_node"))
        shape = str(feat.GetField("link_shape") or "circular")
        length = float(feat.GetField("length") or 0.0)
        mann_n = float(feat.GetField("roughness_n") or 0.012)
        span = feat.GetField("span")
        rise = feat.GetField("rise")
        diameter = feat.GetField("diameter")
        entr_loss = float(feat.GetField("entrance_loss_k") or 0.5)
        exit_loss = float(feat.GetField("exit_loss_k") or 1.0)

        g = feat.GetGeometryRef()
        pts = np.array([(g.GetX(i), g.GetY(i)) for i in range(g.GetPointCount())], dtype=np.float64)

        if shape == "box":
            hec_shape = "Box"
            hec_rise = float(rise or 5.0)
            hec_span = float(span or 10.0)
        else:
            hec_shape = "Circular"
            hec_rise = float(diameter or 3.0)
            hec_span = float(diameter or 3.0)

        conduits.append(HecrasPipeConduit(
            name=lid[:9],
            us_node=from_node[:6],
            ds_node=to_node[:6],
            length=length,
            shape=hec_shape,
            rise=hec_rise,
            span=hec_span,
            manning_n=mann_n,
            entrance_loss_k=entr_loss,
            exit_loss_k=exit_loss,
            us_backflow_k=entr_loss,
            ds_backflow_k=exit_loss,
            polyline=pts,
        ))
    ds = None
    return conduits


def assign_node_connections(nodes: Dict[str, HecrasPipeNode],
                            conduits: List[HecrasPipeConduit]) -> None:
    """Count US/DS connections for each node."""
    counts: Dict[str, Tuple[int, int]] = {}
    for c in conduits:
        us = c.us_node
        ds = c.ds_node
        if us not in counts:
            counts[us] = [0, 0]
        if ds not in counts:
            counts[ds] = [0, 0]
        counts[us][1] += 1  # ds outflow
        counts[ds][0] += 1  # us inflow
    for nid, (ds_c, us_c) in counts.items():
        if nid in nodes:
            nodes[nid].us_connections = ds_c
            nodes[nid].ds_connections = us_c
            nodes[nid].total_connections = ds_c + us_c


# ---------------------------------------------------------------------------
# Structures from source GPKG
# ---------------------------------------------------------------------------


def load_structures(source_gpkg: str) -> List[HecrasSA2DConnection]:
    """Load hydraulic structures as SA/2D connections with culvert groups."""
    connections: List[HecrasSA2DConnection] = []
    ds = ogr.Open(source_gpkg)
    layer = ds.GetLayerByName("SWE2D_Structures")
    if layer is None:
        return connections

    for feat in layer:
        sid = str(feat.GetField("structure_id"))
        stype = int(feat.GetField("structure_type") or 0)
        if stype != 2:  # Only CULVERT for now
            continue

        g = feat.GetGeometryRef()
        pts = np.array([(g.GetX(i), g.GetY(i)) for i in range(g.GetPointCount())], dtype=np.float64)

        culvert_shape = str(feat.GetField("culvert_shape") or "box")
        culvert_code = int(feat.GetField("culvert_code") or 19)
        culvert_rise = float(feat.GetField("culvert_rise") or 4.0)
        culvert_span = float(feat.GetField("culvert_span") or 8.0)
        length = float(feat.GetField("length") or 70.0)
        mann_n = float(feat.GetField("roughness_n") or 0.012)
        inlet_inv = float(feat.GetField("inlet_invert_elev") or 0.0)
        outlet_inv = float(feat.GetField("outlet_invert_elev") or 0.0)
        entr_loss = float(feat.GetField("entrance_loss_k") or 0.5)
        exit_loss = float(feat.GetField("exit_loss_k") or 1.0)
        barrels = int(feat.GetField("culvert_barrels") or 1)

        shape_map = {"box": 2, "circular": 1, "pipe_arch": 3}
        hec_shape_num = shape_map.get(culvert_shape, 2)
        hec_shape_name = culvert_shape.capitalize()

        connections.append(HecrasSA2DConnection(
            name=f"Structure {sid}",
            upstream_area="Perimeter 1",
            downstream_area="Perimeter 1",
            centerline=pts,
            weir_width=culvert_span,
            weir_coef=2.6,
            culvert_groups=[
                HecrasCulvertGroup(
                    shape=hec_shape_num,
                    shape_name=hec_shape_name,
                    rise=culvert_rise,
                    span=culvert_span,
                    length=length,
                    manning_n=mann_n,
                    entrance_loss=entr_loss,
                    exit_loss=exit_loss,
                    us_invert=inlet_inv,
                    ds_invert=outlet_inv,
                    chart=8,
                    scale=3,
                    group_name=f"Group #{sid}",
                    barrels=barrels,
                    culvert_code=culvert_code,
                )
            ],
        ))
    ds = None
    return connections


# ---------------------------------------------------------------------------
# Widget state reader
# ---------------------------------------------------------------------------


def get_widget_value(widgets: dict, key: str, default=None):
    w = widgets.get(key)
    if w is None:
        return default
    return w.get("value", default)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    gpkg_path = sys.argv[1]
    source_gpkg = sys.argv[2] if len(sys.argv) > 2 else None
    output_dir = Path(sys.argv[3] if len(sys.argv) > 3 else "hecras_export")
    project_name = Path(gpkg_path).stem

    conn = sqlite3.connect(gpkg_path)

    # --- Most recent run ---
    cur = conn.execute(
        "SELECT run_id, metadata_json FROM swe2d_run_logs ORDER BY rowid DESC LIMIT 1"
    )
    r = cur.fetchone()
    if r is None:
        print("No runs found in GPKG")
        sys.exit(1)
    run_id, metadata_json = r
    print(f"Exporting run: {run_id}")

    meta = json.loads(metadata_json) if metadata_json else {}
    ws = meta.get("workbench_widget_state", {})
    widgets = ws.get("widgets", {})

    # --- Load mesh ---
    cur = conn.execute(
        "SELECT mesh_name FROM swe2d_baked_mesh ORDER BY rowid DESC LIMIT 1"
    )
    mesh_row = cur.fetchone()
    if mesh_row is None:
        print("No mesh found")
        sys.exit(1)
    mesh_name = mesh_row[0]
    conn.close()

    print(f"Loading mesh: {mesh_name}")
    mesh_data = load_mesh(gpkg_path, mesh_name)
    if mesh_data is None:
        print("Failed to load mesh")
        sys.exit(1)
    n_nodes = len(mesh_data["node_x"])
    n_cells = len(mesh_data.get("cell_face_offsets", [])) - 1
    print(f"  Nodes: {n_nodes}, Cells: {n_cells}")

    perimeter = extract_perimeter(mesh_data)
    cell_centers = extract_cell_centers(mesh_data)
    print(f"  Perimeter vertices: {len(perimeter)}, Cell centers: {len(cell_centers)}")

    bc_lines, boundaries = classify_bc_sides(mesh_data)
    print(f"  BC lines: {len(bc_lines)}, boundaries: {len(boundaries)}")
    for bl in bc_lines:
        print(f"    - {bl.name}: {len(bl.coordinates)} points")

    # --- Drainage from source GPKG ---
    pipe_nodes: List[HecrasPipeNode] = []
    pipe_conduits: List[HecrasPipeConduit] = []
    connections: List[HecrasSA2DConnection] = []

    if source_gpkg and Path(source_gpkg).exists():
        print(f"\nReading drainage from: {source_gpkg}")
        node_map = load_drainage_nodes(source_gpkg)
        pipe_conduits = load_drainage_links(source_gpkg)
        assign_node_connections(node_map, pipe_conduits)
        pipe_nodes = list(node_map.values())
        connections = load_structures(source_gpkg)
        print(f"  Pipe nodes: {len(pipe_nodes)}")
        for n in pipe_nodes:
            print(f"    {n.name}: ({n.x:.1f}, {n.y:.1f}) type={n.node_type} "
                  f"inv={n.invert_elevation} depth={n.depth}")
        print(f"  Pipe conduits: {len(pipe_conduits)}")
        for c in pipe_conduits:
            print(f"    {c.name}: {c.us_node}->{c.ds_node} {c.shape} "
                  f"{c.span}x{c.rise} L={c.length:.0f}")
        print(f"  Structures: {len(connections)}")
        for c in connections:
            for cg in c.culvert_groups:
                print(f"    {c.name}: {cg.shape_name} {cg.span}x{cg.rise} "
                      f"code={cg.culvert_code} inv={cg.us_invert}->{cg.ds_invert}")
    else:
        print("\nNo source GPKG provided; skipping drainage/structure export")

    # --- Simulation params ---
    run_duration_str = str(get_widget_value(widgets, "run_time_edit", "2:00") or "2:00")
    if ":" in run_duration_str:
        parts = run_duration_str.split(":")
        run_duration_s = float(parts[0]) * 3600 + float(parts[1]) * 60
    else:
        run_duration_s = float(run_duration_str)
    dt_val = get_widget_value(widgets, "dt_spin", 2.0)
    if dt_val is None:
        dt_val = 2.0
    dt_computation = f"{float(dt_val)}SEC"
    output_interval = str(get_widget_value(widgets, "output_interval_combo", "1MIN") or "1MIN")
    manning_n = float(get_widget_value(widgets, "n_mann_spin", 0.03) or 0.03)
    start_time = str(get_widget_value(widgets, "start_time_edit", "0000") or "0000")

    end_hours = int(run_duration_s // 3600)
    end_minutes = int((run_duration_s % 3600) // 60)
    end_time_str = f"{end_hours:02d}{end_minutes:02d}"

    plan_params = HecrasPlanParams(
        title=project_name,
        short_identifier=project_name[:60],
        simulation_date_start="01JAN2000",
        simulation_time_start=str(start_time).zfill(4) if start_time else "0000",
        simulation_date_end="01JAN2000",
        simulation_time_end=end_time_str,
        computation_interval=str(dt_computation),
        output_interval=str(output_interval),
        gravity=32.17405,
    )

    flow_area = Hecras2DFlowArea(name="Perimeter 1", manning_n=manning_n)

    # --- Write ---
    print(f"\nWriting HEC-RAS project to: {output_dir}")
    write_hecras_project(
        output_dir=output_dir,
        project_name=project_name,
        mesh_data=mesh_data,
        flow_area=flow_area,
        perimeter=perimeter,
        cell_centers=cell_centers,  # seeds for HEC-RAS mesh generation
        bc_lines=bc_lines,
        unsteady_boundaries=boundaries,
        connections=connections or None,
        pipe_nodes=pipe_nodes or None,
        pipe_conduits=pipe_conduits or None,
        plan_params=plan_params,
        is_us_customary=True,
        manning_n=manning_n,
        projection_wkt='LOCAL_CS["NAD83 / Texas Central (ftUS)"]',
    )

    print(f"\nDone! Files written to {output_dir.absolute()}")


if __name__ == "__main__":
    main()
