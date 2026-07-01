#!/usr/bin/env python3
"""Wireframe mesh comparison: plain rect mesh vs house-cell-filtered mesh.
Both with Manning n colorramp and buildings blacked out as patches."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.path as mpath
import glob, os, sys

# ── helpers ────────────────────────────────────────────────────────────────────

def load_asc_raw(path):
    with open(path) as f:
        lines = f.readlines()
    header = {}
    for line in lines[:6]:
        k, v = line.strip().split()
        header[k] = float(v)
    nrows, ncols = int(header['nrows']), int(header['ncols'])
    data = np.array([[float(v) for v in l.strip().split()] for l in lines[6:]])
    return header, data

def buffer_polyline(points, distance):
    n = len(points)
    result = []
    for i in range(n):
        p0 = points[i - 1]
        p1 = points[i]
        p2 = points[(i + 1) % n]
        v1x, v1y = p1[0]-p0[0], p1[1]-p0[1]
        v2x, v2y = p2[0]-p1[0], p2[1]-p1[1]
        L1 = np.sqrt(v1x**2 + v1y**2)
        L2 = np.sqrt(v2x**2 + v2y**2)
        if L1 < 1e-9 or L2 < 1e-9:
            result.append(p1); continue
        nx1, ny1 = -v1y/L1, v1x/L1
        nx2, ny2 = -v2y/L2, v2x/L2
        nx, ny = (nx1+nx2)/2, (ny1+ny2)/2
        L = np.sqrt(nx**2 + ny**2)
        if L < 1e-9:
            nx, ny = nx1, ny1
        else:
            nx, ny = nx/L, ny/L
        result.append([p1[0] + nx*distance, p1[1] + ny*distance])
    return np.array(result)

# ── paths ─────────────────────────────────────────────────────────────────────
ASC_PATH = 'reference/anuga_validation_tests/case_studies/merewether/topography1.asc'
HOUSES_DIR = 'reference/anuga_validation_tests/case_studies/merewether/houses/'
BASE_DIR = 'reference/anuga_validation_tests/case_studies/merewether'
MSH_PATH = 'tests/merewether_1m2m_mesh.msh'

asc_header, asc_data = load_asc_raw(ASC_PATH)
xll, yll = asc_header['xllcorner'], asc_header['yllcorner']
cs = asc_header['cellsize']
ncols, nrows = int(asc_header['ncols']), int(asc_header['nrows'])

# ── house polygons ─────────────────────────────────────────────────────────────
house_files = sorted(glob.glob(HOUSES_DIR + '*.csv'))
house_polys = [np.loadtxt(hf, delimiter=',', skiprows=1) for hf in house_files]
print(f'Loaded {len(house_polys)} house polygons')

# ── road polygon (10m buffer) ──────────────────────────────────────────────────
road_pts = np.loadtxt(os.path.join(BASE_DIR, 'Road', 'RoadPolygon.csv'), delimiter=',', skiprows=1)
road_buffered = buffer_polyline(road_pts, 10.0)
road_path = mpath.Path(road_buffered)

# ── Load pre-built mesh from build_anuga_merewether_mesh ───────────────────────
sys.path.insert(0, '.')
from tests.build_anuga_merewether_mesh import build_anuga_merewether_mesh

print('Building 1m/2m mesh with post-hoc house removal...')
res = build_anuga_merewether_mesh(ASC_PATH, elem_size_outer=2.0, elem_size_inner=1.0)

node_x_f = res['node_x']
node_y_f = res['node_y']
cell_nodes_f = res['cell_nodes']
n_mann_f = res['cell_manning_n']
cell_cx_f = res['cell_cx']
cell_cy_f = res['cell_cy']

# ── Build plain rect gmsh mesh (no OCC holes) and save as .msh ─────────────────
print('Building plain rect gmsh mesh and saving as .msh...')
import gmsh
gmsh.initialize()
gmsh.model.add('merewether_plain')
gmsh.option.setNumber('Mesh.Algorithm', 6)

# extent from extent.csv
extent = np.loadtxt(os.path.join(BASE_DIR, 'extent.csv'), delimiter=',', skiprows=1)
xmin, ymin = extent[0]
xmax, ymax = extent[2]
Lx, Ly = xmax - xmin, ymax - ymin

rect = gmsh.model.occ.addRectangle(xmin, ymin, 0, Lx, Ly)
gmsh.model.occ.synchronize()

# Mesh size field: 1m interior, 2m exterior
merew = np.loadtxt(os.path.join(BASE_DIR, 'merewether.csv'), delimiter=',', skiprows=1)
mw_xmin, mw_xmax = merew[:, 0].min(), merew[:, 0].max()
mw_ymin, mw_ymax = merew[:, 1].min(), merew[:, 1].max()

gmsh.model.mesh.field.add('Box', 1)
gmsh.model.mesh.field.setNumber(1, 'VIn', 1.0)
gmsh.model.mesh.field.setNumber(1, 'VOut', 2.0)
gmsh.model.mesh.field.setNumber(1, 'XMin', mw_xmin)
gmsh.model.mesh.field.setNumber(1, 'XMax', mw_xmax)
gmsh.model.mesh.field.setNumber(1, 'YMin', mw_ymin)
gmsh.model.mesh.field.setNumber(1, 'YMax', mw_ymax)
gmsh.model.mesh.field.setAsBackgroundMesh(1)

gmsh.model.mesh.generate(2)

# Extract mesh before finalize
_, node_coords, _ = gmsh.model.mesh.getNodes()
plain_nodes = node_coords.reshape(-1, 3)[:, :2]
elem_types, _, elem_node_tags = gmsh.model.mesh.getElements(dim=2)
plain_tris = None
for etype, nodes in zip(elem_types, elem_node_tags):
    if etype == 2:
        plain_tris = (nodes - 1).reshape(-1, 3).astype(np.int32)
        break

gmsh.write(MSH_PATH)
gmsh.finalize()
print(f'Saved {MSH_PATH}')
print(f'Plain mesh: {len(plain_nodes)} nodes, {len(plain_tris)} cells')

# Assign Manning n to plain mesh cells: check if centroid is in road buffer
plain_cx = plain_nodes[plain_tris[:, 0]][:, 0]  # rough centroid (fine for viz)
plain_cy = plain_nodes[plain_tris[:, 0]][:, 1]
in_road_plain = road_path.contains_points(np.column_stack([plain_cx, plain_cy]))
n_mann_plain = np.where(in_road_plain, 0.02, 0.04)

# ── figure ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(20, 10))
fig.suptitle('Merewether Mesh Comparison — Manning n (n=0.02 road, n=0.04 terrain) | Buildings = black', fontsize=13)

cmap = plt.cm.YlGnBu
norm = mcolors.Normalize(vmin=0.015, vmax=0.05)

def plot_mesh_panel(ax, title, node_x, node_y, cell_nodes, n_mann_arr, house_polys_patch=None):
    ax.set_title(title, fontsize=11)
    t = tri.Triangulation(node_x, node_y, cell_nodes)
    tc = ax.tripcolor(t, n_mann_arr, cmap=cmap, norm=norm, shading='flat', alpha=0.9)
    ax.triplot(t, 'w-', linewidth=0.25, alpha=0.5)

    if house_polys_patch:
        for hp in house_polys_patch:
            poly = mpatches.Polygon(hp, closed=True, facecolor='black', edgecolor='red',
                                    linewidth=0.8, zorder=10)
            ax.add_patch(poly)

    ax.set_aspect('equal')
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    ax.set_xlim(xmin - 20, xmax + 20)
    ax.set_ylim(ymin - 20, ymax + 20)
    return tc

# Panel A: plain rect mesh (buildings shown as black patches)
tc = plot_mesh_panel(
    axes[0], 'A: Plain rect mesh — 1m/2m (gmsh .msh)\nBuildings overlaid as black patches',
    plain_nodes[:, 0], plain_nodes[:, 1], plain_tris, n_mann_plain,
    house_polys_patch=house_polys
)

# Panel B: filtered mesh (house cells removed, shown as gaps + black patches)
tc2 = plot_mesh_panel(
    axes[1], 'B: Filtered mesh — house cells removed post-hoc\nBuildings overlaid as black patches',
    node_x_f, node_y_f, cell_nodes_f, n_mann_f,
    house_polys_patch=house_polys
)

cbar = fig.colorbar(tc, ax=axes.ravel().tolist(), shrink=0.55, pad=0.02)
cbar.set_label("Manning's n", fontsize=10)

plt.tight_layout()
out = 'tests/merewether_mesh_comparison.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved {out}')
plt.close()

