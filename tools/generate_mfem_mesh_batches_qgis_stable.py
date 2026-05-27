#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from swe2d.mesh.meshing import CellConstraint, ConceptualModel, ConceptualRegion, generate_face_centric_mesh
from swe2d.mesh.mfem_opt import available_mfem_presets
from tools.generate_mfem_beta_mesh_qgis_stable import load_conceptual_from_gpkg, save_mesh_figure


def _save_mesh_npz(path: Path, mesh) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        node_x=np.asarray(mesh.node_x),
        node_y=np.asarray(mesh.node_y),
        node_z=np.asarray(mesh.node_z),
        cell_nodes=np.asarray(mesh.cell_nodes),
        cell_face_offsets=np.asarray(mesh.cell_face_offsets),
        cell_face_nodes=np.asarray(mesh.cell_face_nodes),
        cell_type=np.asarray(mesh.cell_type),
        region_id=np.asarray(mesh.region_id),
        target_size=np.asarray(mesh.target_size),
    )


def _clone_model(model: ConceptualModel) -> ConceptualModel:
    return copy.deepcopy(model)


def _iter_faces(mesh):
    for i in range(len(mesh.cell_face_offsets) - 1):
        s = int(mesh.cell_face_offsets[i])
        e = int(mesh.cell_face_offsets[i + 1])
        poly = mesh.cell_face_nodes[s:e]
        if poly.size >= 3:
            yield poly


def _draw_wireframe(ax, mesh, color: str, linewidth: float, alpha: float) -> None:
    for poly in _iter_faces(mesh):
        xs = mesh.node_x[poly]
        ys = mesh.node_y[poly]
        ax.plot(np.r_[xs, xs[0]], np.r_[ys, ys[0]], color=color, linewidth=linewidth, alpha=alpha)


def _displacement(seed_mesh, opt_mesh) -> np.ndarray:
    n = min(int(seed_mesh.node_x.size), int(opt_mesh.node_x.size))
    if n <= 0:
        return np.zeros((0,), dtype=np.float64)
    dx = np.asarray(opt_mesh.node_x[:n], dtype=np.float64) - np.asarray(seed_mesh.node_x[:n], dtype=np.float64)
    dy = np.asarray(opt_mesh.node_y[:n], dtype=np.float64) - np.asarray(seed_mesh.node_y[:n], dtype=np.float64)
    return np.hypot(dx, dy)


def _save_delta_figure(path: Path, seed_mesh, opt_mesh, layout_name: str, preset: str) -> dict:
    disp = _displacement(seed_mesh, opt_mesh)
    if disp.size:
        stats = {
            "max": float(np.max(disp)),
            "mean": float(np.mean(disp)),
            "p95": float(np.percentile(disp, 95.0)),
            "p99": float(np.percentile(disp, 99.0)),
        }
    else:
        stats = {"max": 0.0, "mean": 0.0, "p95": 0.0, "p99": 0.0}

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 6), dpi=170)

    _draw_wireframe(ax0, seed_mesh, color="#909090", linewidth=0.20, alpha=0.70)
    _draw_wireframe(ax0, opt_mesh, color="#1565c0", linewidth=0.24, alpha=0.78)
    ax0.set_title(f"{layout_name} | {preset} | seed(gray) vs mfem(blue)")
    ax0.set_aspect("equal", adjustable="box")
    ax0.set_xlabel("X")
    ax0.set_ylabel("Y")
    ax0.grid(True, linewidth=0.2, alpha=0.25)

    sc = ax1.scatter(
        np.asarray(opt_mesh.node_x),
        np.asarray(opt_mesh.node_y),
        c=disp if disp.size else np.zeros_like(np.asarray(opt_mesh.node_x)),
        s=3.0,
        cmap="inferno",
        linewidths=0.0,
        alpha=0.9,
    )
    ax1.set_title(
        f"node displacement | max={stats['max']:.4g} mean={stats['mean']:.4g} p95={stats['p95']:.4g}"
    )
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.grid(True, linewidth=0.2, alpha=0.25)
    cb = fig.colorbar(sc, ax=ax1, fraction=0.046, pad=0.04)
    cb.set_label("|delta x,y|", rotation=90)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return stats


def _constraint_variant(model: ConceptualModel) -> ConceptualModel:
    variant = _clone_model(model)
    if not variant.regions:
        return variant

    outer = variant.regions[0]
    constraint_regions = list(variant.regions[1:])
    constraint_polygons = [
        CellConstraint(
            constraint_id=region.region_id,
            ring_xy=list(region.ring_xy),
            target_size=float(region.default_size),
            cell_type=region.default_cell_type,
        )
        for region in constraint_regions
    ]

    variant.regions = [
        ConceptualRegion(
            region_id=outer.region_id,
            ring_xy=list(outer.ring_xy),
            default_size=outer.default_size,
            default_cell_type=outer.default_cell_type,
            edge_lengths=list(outer.edge_lengths) if outer.edge_lengths is not None else None,
            hole_rings=list(outer.hole_rings) if outer.hole_rings is not None else None,
        )
    ]
    variant.constraints = constraint_polygons
    return variant


def _run_case(model: ConceptualModel, layout_name: str, preset: str, out_dir: Path, seed_backend: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_mesh = generate_face_centric_mesh(
        model,
        backend=seed_backend,
        options={
            "post_opt_backend": "none",
            "mfem_post_opt_enable": False,
        },
    )
    mesh = generate_face_centric_mesh(
        model,
        backend="mfem_opt",
        options={
            "mfem_seed_backend": seed_backend,
            "mfem_post_opt_enable": True,
            "mfem_post_opt_preset": preset,
            "mfem_post_opt_strict": True,
        },
    )
    stem = f"{layout_name}__{preset}"
    img_path = out_dir / f"{stem}.png"
    delta_img_path = out_dir / f"{stem}__delta.png"
    npz_path = out_dir / f"{stem}.npz"
    stats_path = out_dir / f"{stem}__stats.json"
    save_mesh_figure(mesh, str(img_path))
    _save_mesh_npz(npz_path, mesh)
    stats = _save_delta_figure(delta_img_path, seed_mesh, mesh, layout_name=layout_name, preset=preset)
    stats.update(
        {
            "layout": layout_name,
            "preset": preset,
            "seed_backend": seed_backend,
            "engine": str((mesh.quality_summary or {}).get("engine", "unknown")),
            "backend_name": str((mesh.quality_summary or {}).get("backend_name", "mfem_opt")),
        }
    )
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"saved {img_path}")
    print(f"saved {npz_path}")
    print(f"saved {delta_img_path}")
    print(f"saved {stats_path}")
    print(
        f"displacement_stats layout={layout_name} preset={preset} "
        f"max={stats['max']:.6g} mean={stats['mean']:.6g} p95={stats['p95']:.6g} engine={stats['engine']}"
    )


def _iter_presets(requested: Iterable[str] | None) -> list[str]:
    presets = available_mfem_presets()
    if requested:
        selected = [preset for preset in requested if preset in presets]
        missing = [preset for preset in requested if preset not in presets]
        if missing:
            raise SystemExit(f"Unknown presets requested: {missing}")
        return selected
    return presets


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpkg", default="qgis_testing_project/swe2d_model.gpkg")
    ap.add_argument("--out-dir", default="artifacts/mfem_batches")
    ap.add_argument("--seed-backend", default="structured", choices=["structured", "gmsh", "tqmesh", "hybrid_cpp"])
    ap.add_argument("--preset", action="append", default=None)
    args = ap.parse_args()

    base_model = load_conceptual_from_gpkg(args.gpkg)
    constraint_model = _constraint_variant(base_model)
    out_dir = Path(args.out_dir)
    presets = _iter_presets(args.preset)

    for preset in presets:
        _run_case(_clone_model(base_model), "regions", preset, out_dir, args.seed_backend)
        _run_case(_clone_model(constraint_model), "constraints", preset, out_dir, args.seed_backend)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
