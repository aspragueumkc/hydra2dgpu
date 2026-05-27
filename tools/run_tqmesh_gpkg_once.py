#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from generate_mfem_beta_mesh_qgis_stable import load_conceptual_from_gpkg, save_mesh_figure
from swe2d.mesh.meshing import generate_face_centric_mesh


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpkg", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--disable-quad-edges", action="store_true")
    args = ap.parse_args()

    out_prefix = Path(args.out_prefix)
    out_npz = out_prefix.with_suffix(".npz")
    out_png = out_prefix.with_suffix(".png")
    out_json = out_prefix.with_suffix(".json")

    conceptual = load_conceptual_from_gpkg(args.gpkg)
    if args.disable_quad_edges:
        conceptual.quad_edges = []
    mesh = generate_face_centric_mesh(conceptual, backend="tqmesh", options={"post_opt_backend": "none"})

    np.savez(
        out_npz,
        node_x=mesh.node_x,
        node_y=mesh.node_y,
        node_z=mesh.node_z,
        cell_nodes=mesh.cell_nodes,
        cell_face_offsets=mesh.cell_face_offsets,
        cell_face_nodes=mesh.cell_face_nodes,
        cell_type=mesh.cell_type,
        region_id=mesh.region_id,
        target_size=mesh.target_size,
    )
    save_mesh_figure(mesh, str(out_png))

    summary = {
        "source": args.gpkg,
        "backend": "tqmesh",
        "nodes": int(mesh.node_x.size),
        "faces": int(max(0, mesh.cell_face_offsets.size - 1)),
        "regions": int(len(conceptual.regions)),
        "arcs": int(len(conceptual.arcs)),
        "constraints": int(len(conceptual.constraints)),
        "quad_edges": int(len(conceptual.quad_edges)),
        "outputs": {
            "npz": str(out_npz),
            "figure_png": str(out_png),
            "json": str(out_json),
        },
        "preprocess": {
            "normalize_origin": True,
            "normalization_owner": "swe2d.mesh.meshing.generate_face_centric_mesh",
            "disable_quad_edges": bool(args.disable_quad_edges),
        },
    }
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
