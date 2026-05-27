#!/usr/bin/env python3
"""Generate topology mesh from GeoPackage layers with MFEM beta post-opt enabled.

This script is intended to run inside the qgis_stable conda environment.
"""

import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from osgeo import ogr

from swe2d.mesh.meshing import (
    CellConstraint,
    ConceptualArc,
    ConceptualModel,
    ConceptualNode,
    ConceptualRegion,
    QuadEdgeControl,
    generate_face_centric_mesh,
)


def _field(feat, name, default=None):
    idx = feat.GetFieldIndex(name)
    if idx < 0:
        return default
    val = feat.GetField(idx)
    return default if val is None else val


def _line_points(geom):
    if geom is None:
        return None
    gtype = geom.GetGeometryName().upper()
    if gtype == "LINESTRING":
        return [(geom.GetX(i), geom.GetY(i)) for i in range(geom.GetPointCount())]
    if gtype == "MULTILINESTRING" and geom.GetGeometryCount() > 0:
        g0 = geom.GetGeometryRef(0)
        return [(g0.GetX(i), g0.GetY(i)) for i in range(g0.GetPointCount())]
    return None


def _polygon_rings(geom):
    if geom is None:
        return None, []
    gtype = geom.GetGeometryName().upper()
    poly = geom
    if gtype == "MULTIPOLYGON" and geom.GetGeometryCount() > 0:
        poly = geom.GetGeometryRef(0)
        gtype = poly.GetGeometryName().upper()
    if gtype != "POLYGON":
        return None, []
    ext = poly.GetGeometryRef(0)
    if ext is None:
        return None, []
    ring = [(ext.GetX(i), ext.GetY(i)) for i in range(ext.GetPointCount())]
    holes = []
    for j in range(1, poly.GetGeometryCount()):
        h = poly.GetGeometryRef(j)
        holes.append([(h.GetX(i), h.GetY(i)) for i in range(h.GetPointCount())])
    return ring, holes


def load_conceptual_from_gpkg(gpkg_path: str) -> ConceptualModel:
    ds = ogr.Open(gpkg_path)
    if ds is None:
        raise RuntimeError(f"Failed to open GeoPackage: {gpkg_path}")

    nodes = []
    lyr = ds.GetLayerByName("swe2d_topo_nodes")
    if lyr is not None:
        for i, feat in enumerate(lyr):
            geom = feat.GetGeometryRef()
            if geom is None or geom.GetGeometryName().upper() != "POINT":
                continue
            nid = int(_field(feat, "node_id", i + 1))
            nodes.append(ConceptualNode(node_id=nid, x=float(geom.GetX()), y=float(geom.GetY())))

    arcs = []
    lyr = ds.GetLayerByName("swe2d_topo_arcs")
    if lyr is not None:
        for i, feat in enumerate(lyr):
            pts = _line_points(feat.GetGeometryRef())
            arcs.append(
                ConceptualArc(
                    arc_id=int(_field(feat, "arc_id", i + 1)),
                    node0=int(_field(feat, "node0", -1)),
                    node1=int(_field(feat, "node1", -1)),
                    region_id=int(_field(feat, "region_id", -1)),
                    arc_role=str(_field(feat, "arc_role", "") or "").strip().lower() or None,
                    points_xy=pts,
                )
            )

    regions = []
    lyr = ds.GetLayerByName("swe2d_topo_regions")
    if lyr is None:
        raise RuntimeError("Layer swe2d_topo_regions not found")
    for i, feat in enumerate(lyr):
        ring, holes = _polygon_rings(feat.GetGeometryRef())
        if ring is None or len(ring) < 4:
            continue
        regions.append(
            ConceptualRegion(
                region_id=int(_field(feat, "region_id", i + 1)),
                ring_xy=[(float(x), float(y)) for x, y in ring],
                default_size=float(_field(feat, "target_size", 20.0)),
                default_cell_type=str(_field(feat, "cell_type", "triangular")).strip().lower() or "triangular",
                hole_rings=[[(float(x), float(y)) for x, y in h] for h in holes],
            )
        )

    constraints = []
    lyr = ds.GetLayerByName("swe2d_topo_constraints")
    if lyr is not None:
        for i, feat in enumerate(lyr):
            ring, _holes = _polygon_rings(feat.GetGeometryRef())
            if ring is None or len(ring) < 4:
                continue
            constraints.append(
                CellConstraint(
                    constraint_id=int(_field(feat, "constraint_id", i + 1)),
                    ring_xy=[(float(x), float(y)) for x, y in ring],
                    target_size=float(_field(feat, "target_size", 20.0)),
                    cell_type=str(_field(feat, "cell_type", "triangular")).strip().lower() or "triangular",
                )
            )

    quad_edges = []
    lyr = ds.GetLayerByName("swe2d_topo_quad_edges")
    if lyr is not None:
        for i, feat in enumerate(lyr):
            pts = _line_points(feat.GetGeometryRef())
            if pts is None or len(pts) < 2:
                continue
            target_size = _field(feat, "target_size", None)
            first_height = _field(feat, "first_height", None)
            quad_edges.append(
                QuadEdgeControl(
                    region_id=int(_field(feat, "region_id", -1)),
                    edge_id=int(_field(feat, "edge_id", i + 1)),
                    points_xy=[(float(x), float(y)) for x, y in pts],
                    target_size=None if target_size is None else float(target_size),
                    n_layers=int(_field(feat, "n_layers", 0)),
                    first_height=None if first_height is None else float(first_height),
                    growth_rate=float(_field(feat, "growth_rate", 1.0)),
                )
            )

    if not regions:
        raise RuntimeError("No valid topology regions found")

    return ConceptualModel(
        nodes=nodes,
        arcs=arcs,
        regions=regions,
        constraints=constraints,
        quad_edges=quad_edges,
    )


def save_mesh_figure(mesh, out_img: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 7), dpi=160)
    for i in range(len(mesh.cell_face_offsets) - 1):
        s = int(mesh.cell_face_offsets[i])
        e = int(mesh.cell_face_offsets[i + 1])
        poly = mesh.cell_face_nodes[s:e]
        if poly.size < 3:
            continue
        xs = mesh.node_x[poly]
        ys = mesh.node_y[poly]
        ax.plot(np.r_[xs, xs[0]], np.r_[ys, ys[0]], color="k", linewidth=0.22, alpha=0.85)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Generated Mesh from GeoPackage Topology Layers (MFEM beta option enabled)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, linewidth=0.2, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_img)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpkg", default="qgis_testing_project/swe2d_model.gpkg")
    ap.add_argument("--base-backend", default="structured", choices=["structured", "gmsh", "tqmesh", "hybrid_cpp", "mfem_opt"])
    ap.add_argument("--mfem-seed-backend", default="structured", choices=["structured", "gmsh", "tqmesh", "hybrid_cpp"])
    ap.add_argument("--mfem-preset", default="balanced_shape_size")
    ap.add_argument("--out-img", default="artifacts/mfem_beta_topo_mesh.png")
    ap.add_argument("--out-npz", default="artifacts/mfem_beta_topo_mesh.npz")
    ap.add_argument("--mfem-strict", action="store_true")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_img) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_npz) or ".", exist_ok=True)

    conceptual = load_conceptual_from_gpkg(args.gpkg)
    mesh = generate_face_centric_mesh(
        conceptual,
        backend=args.base_backend,
        options={
            "post_opt_backend": "mfem_tmop",
            "mfem_post_opt_enable": True,
            "mfem_seed_backend": args.mfem_seed_backend,
            "mfem_post_opt_preset": args.mfem_preset,
            "mfem_post_opt_strict": bool(args.mfem_strict),
            "mfem_post_opt_max_iterations": 25,
            "mfem_post_opt_quality_weight": 1.0,
            "mfem_post_opt_boundary_fit_weight": 0.35,
            "mfem_post_opt_interface_fit_weight": 0.25,
            "mfem_post_opt_lock_boundary_nodes": True,
        },
    )

    np.savez(
        args.out_npz,
        node_x=mesh.node_x,
        node_y=mesh.node_y,
        node_z=mesh.node_z,
        cell_face_offsets=mesh.cell_face_offsets,
        cell_face_nodes=mesh.cell_face_nodes,
    )
    save_mesh_figure(mesh, args.out_img)

    print(f"GPKG: {args.gpkg}")
    print(
        f"base_backend={args.base_backend} mfem_beta_enabled=True "
        f"seed_backend={args.mfem_seed_backend} preset={args.mfem_preset} strict={bool(args.mfem_strict)}"
    )
    print(
        f"regions={len(conceptual.regions)} arcs={len(conceptual.arcs)} "
        f"constraints={len(conceptual.constraints)} quad_edges={len(conceptual.quad_edges)}"
    )
    print(f"nodes={mesh.node_x.size} faces={max(0, mesh.cell_face_offsets.size - 1)}")
    print(f"mesh_npz={args.out_npz}")
    print(f"mesh_figure={args.out_img}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
