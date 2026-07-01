"""
Build the ANUGA Merewether mesh with gmsh:
- 1.5m interior (merewether.csv region), 3m elsewhere
- 58 building holes (from houses/*.csv)
- Building elevation bump (+3m inside houses)
- Variable friction (n=0.02 on road buffered 10m, n=0.04 elsewhere)
- Boundary: left/bottom=WALL, right/top=OPEN
"""
import csv
import os
import numpy as np
import gmsh
import matplotlib.path as mpath


def _read_csv_pts(path):
    with open(path) as f:
        return [(float(row[0]), float(row[1])) for row in csv.reader(f)]


def _buffer_polyline(points, distance):
    """Buffer a closed polyline by given distance to create a polygon."""
    n = len(points)
    normals = []
    for i in range(n):
        prev_idx = (i - 1) % n
        next_idx = (i + 1) % n
        d1x = points[i, 0] - points[prev_idx, 0]
        d1y = points[i, 1] - points[prev_idx, 1]
        d2x = points[next_idx, 0] - points[i, 0]
        d2y = points[next_idx, 1] - points[i, 1]
        n1x, n1y = -d1y, d1x
        n2x, n2y = -d2y, d2x
        len1 = np.sqrt(n1x*n1x + n1y*n1y) + 1e-10
        len2 = np.sqrt(n2x*n2x + n2y*n2y) + 1e-10
        n1x /= len1; n1y /= len1
        n2x /= len2; n2y /= len2
        nx = (n1x + n2x) / 2.0
        ny = (n1y + n2y) / 2.0
        nlen = np.sqrt(nx*nx + ny*ny) + 1e-10
        normals.append((nx/nlen, ny/nlen))
    normals = np.array(normals)
    left = points + normals * distance
    right = points - normals * distance
    return np.vstack([left, right[::-1]])


def _interp_bilinear(x, y, asc):
    cs = asc["cellsize"]
    xll = asc["xll"]
    yll = asc["yll"]
    ncols = asc["ncols"]
    nrows = asc["nrows"]
    data = asc["data"]

    col = (x - xll) / cs
    row = (y - yll) / cs
    i = int(np.clip(np.floor(col), 0, ncols - 2))
    j = int(np.clip(np.floor(row), 0, nrows - 2))

    fx = col - i
    fy = row - j

    v00 = data[nrows - 1 - j, i]
    v10 = data[nrows - 1 - j, i + 1]
    v01 = data[nrows - 2 - j, i]
    v11 = data[nrows - 2 - j, i + 1]

    nodata = asc["nodata"]
    for v in [v00, v10, v01, v11]:
        if v == nodata:
            return np.nan

    return (v00 * (1 - fx) * (1 - fy) +
            v10 * fx * (1 - fy) +
            v01 * (1 - fx) * fy +
            v11 * fx * fy)


def build_anuga_merewether_mesh(asc_path, elem_size_outer=2.0, elem_size_inner=1.0):
    """
    Build the ANUGA-matching Merewether mesh.

    Returns dict with:
        node_x, node_y, node_z, cell_nodes,
        bc_n0, bc_n1, bc_tp, bc_vl,
        cell_cx, cell_cy, cell_area,
        inlet_cells, q_rate,
        manning_n_per_cell,
        house_nodes_mask (bool array per node, True if inside any house)
    """
    base = os.path.dirname(asc_path)

    # Load ASC DEM
    with open(asc_path) as f:
        lines = f.readlines()
    ncols = int(lines[0].split()[1])
    nrows = int(lines[1].split()[1])
    xll = float(lines[2].split()[1])
    yll = float(lines[3].split()[1])
    cs = float(lines[4].split()[1])
    nodata = float(lines[5].split()[1])
    data = np.array([[float(v) for v in line.split()] for line in lines[6:]], dtype=np.float64)
    asc = {"ncols": ncols, "nrows": nrows, "xll": xll, "yll": yll,
           "cellsize": cs, "nodata": nodata, "data": data}

    # Read extent (bounding rectangle)
    extent_pts = _read_csv_pts(os.path.join(base, "extent.csv"))
    # extent_pts order: (382250,6354265), (382571,6354265), (382571,6354681), (382250,6354681)

    # Read interior region polygon
    merewether_pts = _read_csv_pts(os.path.join(base, "merewether.csv"))

    # Read all house files
    houses_dir = os.path.join(base, "houses")
    house_files = sorted([f for f in os.listdir(houses_dir)
                          if f.startswith("house") and f.endswith(".csv")])
    house_polygons = [_read_csv_pts(os.path.join(houses_dir, f)) for f in house_files]

    # Read road polygon and buffer by 10m for friction assignment
    road_pts = _read_csv_pts(os.path.join(base, "Road", "RoadPolygon.csv"))
    road_coords = np.array(road_pts)
    road_buffered = _buffer_polyline(road_coords, 10.0)
    road_path = mpath.Path(road_buffered)

    # ── Build plain OCC geometry (no house holes) ───────────────────────────────
    gmsh.initialize()
    gmsh.model.add("merewether_anuga")

    xmin, ymin = extent_pts[0]
    xmax, ymax = extent_pts[2]
    Lx = xmax - xmin
    Ly = ymax - ymin

    rect = gmsh.model.occ.addRectangle(xmin, ymin, 0, Lx, Ly)
    gmsh.model.occ.synchronize()

    # ── Mesh size fields ────────────────────────────────────────────────────────
    merew_xmin = min(p[0] for p in merewether_pts)
    merew_xmax = max(p[0] for p in merewether_pts)
    merew_ymin = min(p[1] for p in merewether_pts)
    merew_ymax = max(p[1] for p in merewether_pts)

    gmsh.model.mesh.field.add("Box", 1)
    gmsh.model.mesh.field.setNumber(1, "VIn", elem_size_inner)
    gmsh.model.mesh.field.setNumber(1, "VOut", elem_size_outer)
    gmsh.model.mesh.field.setNumber(1, "XMin", merew_xmin)
    gmsh.model.mesh.field.setNumber(1, "XMax", merew_xmax)
    gmsh.model.mesh.field.setNumber(1, "YMin", merew_ymin)
    gmsh.model.mesh.field.setNumber(1, "YMax", merew_ymax)
    gmsh.model.mesh.field.setAsBackgroundMesh(1)

    # ── Generate mesh ───────────────────────────────────────────────────────────
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.model.mesh.generate(2)

    # Extract mesh
    _, node_coords, _ = gmsh.model.mesh.getNodes()
    node_xyz = node_coords.reshape(-1, 3)

    elem_types, _, elem_node_tags = gmsh.model.mesh.getElements(dim=2)
    cell_nodes_full = None
    for etype, nodes in zip(elem_types, elem_node_tags):
        if etype == 2:
            cell_nodes_full = (nodes - 1).reshape(-1, 3).astype(np.int32)
            break

    gmsh.finalize()

    # ── Filter out cells inside house polygons (post-hoc hole removal) ─────────────
    cell_cx_full = np.mean(node_xyz[:, 0][cell_nodes_full], axis=1)
    cell_cy_full = np.mean(node_xyz[:, 1][cell_nodes_full], axis=1)

    house_paths = [mpath.Path(np.array(hp)) for hp in house_polygons]
    inside_house = np.zeros(len(cell_nodes_full), dtype=bool)
    for hp in house_paths:
        inside_house |= hp.contains_points(np.column_stack([cell_cx_full, cell_cy_full]))

    print(f"  House cells removed: {inside_house.sum()} / {len(cell_nodes_full)}")
    cell_nodes = cell_nodes_full[~inside_house]

    # ── Reindex nodes (only keep used ones) ──────────────────────────────────────────
    all_nodes_used = np.unique(cell_nodes.ravel())
    n_nodes_orig = len(node_xyz)
    reindex = np.full(n_nodes_orig, -1, dtype=np.int32)
    reindex[all_nodes_used] = np.arange(len(all_nodes_used), dtype=np.int32)
    node_xyz = node_xyz[all_nodes_used]
    cell_nodes = reindex[cell_nodes]

    node_x = node_xyz[:, 0].copy()
    node_y = node_xyz[:, 1].copy()

    # Remove sliver cells (area < 1e-5 m²)
    v1x = node_x[cell_nodes[:, 1]] - node_x[cell_nodes[:, 0]]
    v1y = node_y[cell_nodes[:, 1]] - node_y[cell_nodes[:, 0]]
    v2x = node_x[cell_nodes[:, 2]] - node_x[cell_nodes[:, 0]]
    v2y = node_y[cell_nodes[:, 2]] - node_y[cell_nodes[:, 0]]
    areas = np.abs(v1x * v2y - v1y * v2x) / 2.0
    good = areas > 1e-5
    if (~good).sum():
        print(f'  Removing {(~good).sum()} sliver cells')
        cell_nodes = cell_nodes[good]

    n_cells = cell_nodes.shape[0]
    n_nodes = len(node_x)
    print(f'  Mesh: {n_nodes} nodes, {n_cells} cells')

    # ── Fix negative-area cells (CW orientation) using shoelace formula ─────────────────────
    # The C++ backend uses shoelace: area = 0.5*(x0*y1 + x1*y2 + x2*y0 - x1*y0 - x2*y1 - x0*y2)
    # We compute signed shoelace area; if negative, reverse the node order
    tris = cell_nodes
    x0, x1, x2 = node_x[tris[:, 0]], node_x[tris[:, 1]], node_x[tris[:, 2]]
    y0, y1, y2 = node_y[tris[:, 0]], node_y[tris[:, 1]], node_y[tris[:, 2]]
    twice_area = (x0 * y1 + x1 * y2 + x2 * y0 - x1 * y0 - x2 * y1 - x0 * y2)
    neg_mask = twice_area < 0
    n_neg = neg_mask.sum()
    if n_neg > 0:
        print(f'  Fixing {n_neg} clockwise cells')
        cell_nodes[neg_mask] = cell_nodes[neg_mask, ::-1]  # reverse node order

    # ── Node elevations from ASC ────────────────────────────────────────────────
    node_z = np.array([_interp_bilinear(x, y, asc)
                        for x, y in zip(node_x, node_y)], dtype=np.float64)
    node_z = np.nan_to_num(node_z, nan=0.0)

    # ── Building elevation bump (+3m inside houses) ─────────────────────────────
    house_nodes_mask = np.zeros(n_nodes, dtype=bool)
    for hp in house_polygons:
        hp_path = mpath.Path(np.array(hp))
        for i in range(n_nodes):
            if hp_path.contains_point((node_x[i], node_y[i])):
                house_nodes_mask[i] = True
                node_z[i] += 3.0  # building floor is 3m above ground

    # ── Cell data ──────────────────────────────────────────────────────────────
    cell_cx = np.mean(node_x[cell_nodes], axis=1)
    cell_cy = np.mean(node_y[cell_nodes], axis=1)
    cell_area = np.array([
        np.sum(np.abs(np.cross(
            [node_x[cell_nodes[i, 1]] - node_x[cell_nodes[i, 0]],
             node_y[cell_nodes[i, 1]] - node_y[cell_nodes[i, 0]], 0.0],
            [node_x[cell_nodes[i, 2]] - node_x[cell_nodes[i, 0]],
             node_y[cell_nodes[i, 2]] - node_y[cell_nodes[i, 0]], 0.0]
        ))) / 2.0 for i in range(n_cells)
    ])

    # ── Manning n per cell ─────────────────────────────────────────────────────
    manning_n_per_cell = np.full(n_cells, 0.04, dtype=np.float64)
    for i in range(n_cells):
        if road_path.contains_point((cell_cx[i], cell_cy[i])):
            manning_n_per_cell[i] = 0.02

    # ── Inlet cells (circular region r=10m at inlet center) ────────────────────
    INLET_CX = 382265.0
    INLET_CY = 6354280.0
    INLET_RADIUS = 10.0
    INLET_Q = 19.7

    inlet_cells = np.zeros(n_cells, dtype=bool)
    for i in range(n_cells):
        dx = cell_cx[i] - INLET_CX
        dy = cell_cy[i] - INLET_CY
        if np.sqrt(dx*dx + dy*dy) <= INLET_RADIUS:
            inlet_cells[i] = True

    total_inlet_area = cell_area[inlet_cells].sum()
    q_rate = INLET_Q / total_inlet_area if total_inlet_area > 0 else 0.0

    # ── Boundary conditions ─────────────────────────────────────────────────────
    # BC: left (xmin) = WALL, bottom (ymin) = WALL, right (xmax) = OPEN, top (ymax) = OPEN
    TOL = elem_size_outer * 0.4

    def _is_on_xmin(x): return abs(x - xmin) < TOL
    def _is_on_xmax(x): return abs(x - xmax) < TOL
    def _is_on_ymin(y): return abs(y - ymin) < TOL
    def _is_on_ymax(y): return abs(y - ymax) < TOL

    bc_n0, bc_n1, bc_tp, bc_vl = [], [], [], []
    seen = set()

    for i in range(n_cells):
        nds = cell_nodes[i]
        for ek in range(3):
            n0 = nds[ek]
            n1 = nds[(ek + 1) % 3]
            edge_key = (min(n0, n1), max(n0, n1))
            if edge_key in seen:
                continue
            mx = (node_x[n0] + node_x[n1]) * 0.5
            my = (node_y[n0] + node_y[n1]) * 0.5

            if _is_on_xmin(mx):
                bc_n0.append(n0); bc_n1.append(n1); bc_tp.append(1); bc_vl.append(0.0)
                seen.add(edge_key)
            elif _is_on_xmax(mx):
                bc_n0.append(n0); bc_n1.append(n1); bc_tp.append(4); bc_vl.append(0.0)
                seen.add(edge_key)
            elif _is_on_ymin(my):
                bc_n0.append(n0); bc_n1.append(n1); bc_tp.append(1); bc_vl.append(0.0)
                seen.add(edge_key)
            elif _is_on_ymax(my):
                bc_n0.append(n0); bc_n1.append(n1); bc_tp.append(4); bc_vl.append(0.0)
                seen.add(edge_key)

    bc_n0 = np.array(bc_n0, dtype=np.int32)
    bc_n1 = np.array(bc_n1, dtype=np.int32)
    bc_tp = np.array(bc_tp, dtype=np.int32)
    bc_vl = np.array(bc_vl, dtype=np.float64)

    # ── Source callback ────────────────────────────────────────────────────────
    def source_rate_callback(_t, _dt, h, _hu, _hv):
        rate = np.zeros_like(h)
        rate[inlet_cells] = q_rate
        return rate

    return {
        "node_x": node_x,
        "node_y": node_y,
        "node_z": node_z,
        "cell_nodes": cell_nodes,
        "bc_n0": bc_n0,
        "bc_n1": bc_n1,
        "bc_tp": bc_tp,
        "bc_vl": bc_vl,
        "cell_cx": cell_cx,
        "cell_cy": cell_cy,
        "cell_area": cell_area,
        "cell_manning_n": manning_n_per_cell,
        "house_nodes_mask": house_nodes_mask,
        "inlet_cells": inlet_cells,
        "q_rate": q_rate,
        "source_rate_callback": source_rate_callback,
    }


if __name__ == "__main__":
    asc_path = os.path.join(
        os.path.dirname(__file__),
        "..", "reference", "anuga_validation_tests", "case_studies", "merewether", "topography1.asc"
    )
    asc_path = os.path.normpath(asc_path)
    result = build_anuga_merewether_mesh(asc_path)
    n_cells = result["cell_nodes"].shape[0]
    n_nodes = len(result["node_x"])
    print(f"Mesh: {n_nodes} nodes, {n_cells} triangles")
    print(f"Inlet cells: {result['inlet_cells'].sum()}, q_rate={result['q_rate']:.4f} m/s")
    print(f"Road cells (n=0.02): {(result['cell_manning_n'] == 0.02).sum()}")
    print(f"Non-road cells (n=0.04): {(result['cell_manning_n'] == 0.04).sum()}")
    print(f"House-elevated nodes: {result['house_nodes_mask'].sum()}")
