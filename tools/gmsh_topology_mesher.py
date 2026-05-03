#!/usr/bin/env python3
"""Standalone topology-to-Gmsh mesher (outside QGIS/plugin runtime).

Reads topology layers from a vector datasource (typically GeoPackage), builds a
conceptual topology model compatible with swe2d_meshing, runs the pure Gmsh
backend, and writes mesh artifacts + diagnostics.

Example:
  python3 tools/gmsh_topology_mesher.py \
    --source /path/to/model.gpkg \
    --regions-layer swe2d_topo_regions \
    --constraints-layer swe2d_topo_constraints \
    --quad-edges-layer swe2d_topo_quad_edges \
    --out-prefix /tmp/topo_mesh
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import sys
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from osgeo import ogr
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "GDAL/OGR Python bindings are required. "
        "Install python3-gdal (or run from an environment with osgeo available). "
        f"Import error: {exc}"
    )


def _plugin_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _ensure_plugin_imports() -> None:
    root = _plugin_root()
    if root not in sys.path:
        sys.path.insert(0, root)


_ensure_plugin_imports()

from swe2d_meshing import (  # noqa: E402
    CellConstraint,
    ConceptualModel,
    ConceptualRegion,
    QuadEdgeControl,
    _mesh_quality_stats,
    generate_face_centric_mesh,
)


def _as_float(v, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _as_int(v, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return int(default)


def _as_str(v, default: str = "") -> str:
    if v is None:
        return default
    s = str(v).strip()
    return s if s else default


def _normalize_cell_type(v: str, default: str = "triangular") -> str:
    allowed = {"triangular", "quadrilateral", "cartesian", "empty"}
    s = _as_str(v, default).lower()
    return s if s in allowed else default


def _open_layer(ds: ogr.DataSource, layer_name: str, required: bool) -> Optional[ogr.Layer]:
    lyr = ds.GetLayerByName(layer_name)
    if lyr is None and required:
        raise ValueError(f"Layer not found: {layer_name}")
    return lyr


def _ring_from_polygon_geom(poly_geom: ogr.Geometry) -> List[Tuple[float, float]]:
    ring = poly_geom.GetGeometryRef(0)
    if ring is None:
        return []
    pts = ring.GetPoints()
    out = [(float(x), float(y)) for x, y, *_ in pts]
    if len(out) >= 2 and out[0] == out[-1]:
        out = out[:-1]
    return out


def _iter_polygon_rings(geom: ogr.Geometry) -> Iterable[List[Tuple[float, float]]]:
    if geom is None:
        return
    gname = geom.GetGeometryName().upper()
    if gname == "POLYGON":
        ring = _ring_from_polygon_geom(geom)
        if len(ring) >= 3:
            yield ring
        return
    if gname == "MULTIPOLYGON":
        for i in range(geom.GetGeometryCount()):
            pg = geom.GetGeometryRef(i)
            if pg is None:
                continue
            ring = _ring_from_polygon_geom(pg)
            if len(ring) >= 3:
                yield ring
        return


def _iter_line_coords(geom: ogr.Geometry) -> Iterable[List[Tuple[float, float]]]:
    if geom is None:
        return
    gname = geom.GetGeometryName().upper()
    if gname == "LINESTRING":
        pts = [(float(x), float(y)) for x, y, *_ in geom.GetPoints()]
        if len(pts) >= 2:
            yield pts
        return
    if gname == "MULTILINESTRING":
        for i in range(geom.GetGeometryCount()):
            ln = geom.GetGeometryRef(i)
            if ln is None:
                continue
            pts = [(float(x), float(y)) for x, y, *_ in ln.GetPoints()]
            if len(pts) >= 2:
                yield pts
        return


def load_conceptual_model(
    source: str,
    regions_layer: str,
    constraints_layer: Optional[str],
    quad_edges_layer: Optional[str],
    default_size: float,
    default_cell_type: str,
) -> ConceptualModel:
    ds = ogr.Open(source, 0)
    if ds is None:
        raise RuntimeError(f"Could not open vector source: {source}")

    regions_lyr = _open_layer(ds, regions_layer, required=True)
    constraints_lyr = _open_layer(ds, constraints_layer, required=False) if constraints_layer else None
    quad_edges_lyr = _open_layer(ds, quad_edges_layer, required=False) if quad_edges_layer else None

    regions: List[ConceptualRegion] = []
    constraints: List[CellConstraint] = []
    quad_edges: List[QuadEdgeControl] = []

    region_auto = 1
    for feat in regions_lyr:
        rid = _as_int(feat.GetField("region_id"), region_auto)
        size = max(1e-6, _as_float(feat.GetField("target_size"), default_size))
        ctype = _normalize_cell_type(feat.GetField("cell_type"), default_cell_type)

        edge_lengths = []
        for fn in ("edge_len_1", "edge_len_2", "edge_len_3", "edge_len_4"):
            val = feat.GetField(fn)
            if val is None:
                edge_lengths = []
                break
            edge_lengths.append(max(1e-6, _as_float(val, size)))
        use_edge_lengths = edge_lengths if len(edge_lengths) == 4 else None

        geom = feat.GetGeometryRef()
        if geom is None:
            continue

        part_idx = 0
        for ring in _iter_polygon_rings(geom):
            region_id = rid if part_idx == 0 else int(f"{rid}{part_idx}")
            regions.append(
                ConceptualRegion(
                    region_id=region_id,
                    ring_xy=ring,
                    default_size=size,
                    default_cell_type=ctype,
                    edge_lengths=use_edge_lengths,
                )
            )
            part_idx += 1
        region_auto += 1

    if constraints_lyr is not None:
        c_auto = 1
        for feat in constraints_lyr:
            cid = _as_int(feat.GetField("constraint_id"), c_auto)
            size = max(1e-6, _as_float(feat.GetField("target_size"), default_size))
            ctype = _normalize_cell_type(feat.GetField("cell_type"), default_cell_type)
            geom = feat.GetGeometryRef()
            if geom is None:
                continue
            part_idx = 0
            for ring in _iter_polygon_rings(geom):
                constraint_id = cid if part_idx == 0 else int(f"{cid}{part_idx}")
                constraints.append(
                    CellConstraint(
                        constraint_id=constraint_id,
                        ring_xy=ring,
                        target_size=size,
                        cell_type=ctype,
                    )
                )
                part_idx += 1
            c_auto += 1

    if quad_edges_lyr is not None:
        for feat in quad_edges_lyr:
            rid = _as_int(feat.GetField("region_id"), -1)
            eid = _as_int(feat.GetField("edge_id"), -1)
            if rid < 0 or eid not in (1, 2, 3, 4):
                continue
            target_size_val = feat.GetField("target_size")
            target_size = None if target_size_val is None else max(1e-6, _as_float(target_size_val, default_size))
            n_layers = max(0, _as_int(feat.GetField("n_layers"), 0))
            fh_val = feat.GetField("first_height")
            first_height = None if fh_val is None else max(1e-9, _as_float(fh_val, 0.0))
            growth_rate = max(1e-6, _as_float(feat.GetField("growth_rate"), 1.0))

            geom = feat.GetGeometryRef()
            if geom is None:
                continue
            for pts in _iter_line_coords(geom):
                quad_edges.append(
                    QuadEdgeControl(
                        region_id=rid,
                        edge_id=eid,
                        points_xy=pts,
                        target_size=target_size,
                        n_layers=n_layers,
                        first_height=first_height,
                        growth_rate=growth_rate,
                    )
                )

    if not regions:
        raise ValueError("No valid regions were found in regions layer")

    return ConceptualModel(nodes=[], arcs=[], regions=regions, constraints=constraints, quad_edges=quad_edges)


def _extract_tri_quad_faces(mesh) -> Tuple[np.ndarray, np.ndarray]:
    offs = np.asarray(mesh.cell_face_offsets, dtype=np.int32)
    conn = np.asarray(mesh.cell_face_nodes, dtype=np.int32)
    tris: List[List[int]] = []
    quads: List[List[int]] = []
    for i in range(int(offs.size) - 1):
        s = int(offs[i])
        e = int(offs[i + 1])
        ids = [int(v) for v in conn[s:e]]
        if len(ids) == 3:
            tris.append(ids)
        elif len(ids) == 4:
            quads.append(ids)
    tarr = np.asarray(tris, dtype=np.int32) if tris else np.empty((0, 3), dtype=np.int32)
    qarr = np.asarray(quads, dtype=np.int32) if quads else np.empty((0, 4), dtype=np.int32)
    return tarr, qarr


def write_msh2(mesh, path: str) -> int:
    """Write a basic Gmsh v2.2 ASCII mesh from MeshResult.

    Node ids are written 1-based. Polygons >4 sides are triangulated.
    """
    node_x = np.asarray(mesh.node_x, dtype=np.float64)
    node_y = np.asarray(mesh.node_y, dtype=np.float64)
    offs = np.asarray(mesh.cell_face_offsets, dtype=np.int32)
    conn = np.asarray(mesh.cell_face_nodes, dtype=np.int32)

    elems: List[Tuple[int, List[int]]] = []
    for i in range(int(offs.size) - 1):
        s = int(offs[i])
        e = int(offs[i + 1])
        ids = [int(v) + 1 for v in conn[s:e]]
        if len(ids) == 3:
            elems.append((2, ids))
        elif len(ids) == 4:
            elems.append((3, ids))
        elif len(ids) > 4:
            for k in range(1, len(ids) - 1):
                elems.append((2, [ids[0], ids[k], ids[k + 1]]))

    with open(path, "w", encoding="utf-8") as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        f.write("$Nodes\n")
        f.write(f"{node_x.size}\n")
        for i in range(node_x.size):
            f.write(f"{i + 1} {node_x[i]:.16g} {node_y[i]:.16g} 0\n")
        f.write("$EndNodes\n")
        f.write("$Elements\n")
        f.write(f"{len(elems)}\n")
        for i, (etype, ids) in enumerate(elems, start=1):
            # numTags=2, physical=0, elementary=0
            nodes_txt = " ".join(str(v) for v in ids)
            f.write(f"{i} {etype} 2 0 0 {nodes_txt}\n")
        f.write("$EndElements\n")

    return len(elems)


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_now()}] {msg}", flush=True)


def _constraint_probe_worker(result_q: mp.Queue, model: ConceptualModel, backend_name: str) -> None:
    t0 = time.perf_counter()
    try:
        mesh = generate_face_centric_mesh(model, backend=backend_name)
        result_q.put(
            {
                "status": "ok",
                "elapsed_sec": float(time.perf_counter() - t0),
                "nodes": int(np.asarray(mesh.node_x).size),
                "cells": int(np.asarray(mesh.cell_face_offsets).size - 1),
            }
        )
    except Exception as exc:
        result_q.put(
            {
                "status": "error",
                "elapsed_sec": float(time.perf_counter() - t0),
                "error": str(exc),
            }
        )


def diagnose_constraints_one_by_one(
    model: ConceptualModel,
    backend_name: str,
    timeout_sec: float,
) -> Dict[str, object]:
    results: List[Dict[str, object]] = []
    timeout_sec = max(1.0, float(timeout_sec))

    for idx, constraint in enumerate(model.constraints, start=1):
        _log(
            "mesh> diagnose-run "
            f"constraint_id={constraint.constraint_id} index={idx}/{len(model.constraints)} "
            f"cell_type={constraint.cell_type} target_size={constraint.target_size}"
        )

        probe_model = ConceptualModel(
            nodes=model.nodes,
            arcs=model.arcs,
            regions=model.regions,
            constraints=[constraint],
            quad_edges=model.quad_edges,
        )

        result_q: mp.Queue = mp.Queue()
        proc = mp.Process(target=_constraint_probe_worker, args=(result_q, probe_model, backend_name), daemon=True)
        started = time.perf_counter()
        proc.start()
        proc.join(timeout_sec)

        item: Dict[str, object] = {
            "constraint_id": int(constraint.constraint_id),
            "index": idx,
            "cell_type": str(constraint.cell_type),
            "target_size": float(constraint.target_size),
        }

        if proc.is_alive():
            proc.terminate()
            proc.join(2.0)
            item.update(
                {
                    "status": "timeout",
                    "elapsed_sec": float(time.perf_counter() - started),
                    "timeout_sec": timeout_sec,
                }
            )
        else:
            try:
                child_result = result_q.get_nowait()
                item.update(child_result)
            except queue.Empty:
                item.update(
                    {
                        "status": "error",
                        "elapsed_sec": float(time.perf_counter() - started),
                        "error": "worker exited without result",
                    }
                )

        _log(
            "mesh> diagnose-done "
            f"constraint_id={item['constraint_id']} status={item.get('status')} "
            f"elapsed={float(item.get('elapsed_sec', 0.0)):.2f}s"
        )
        results.append(item)

    bad = [r for r in results if str(r.get("status")) != "ok"]
    return {
        "constraints_total": len(model.constraints),
        "backend": backend_name,
        "timeout_sec": timeout_sec,
        "results": results,
        "bad_constraints": [int(r["constraint_id"]) for r in bad],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Standalone Gmsh topology mesher")
    p.add_argument("--source", required=True, help="Vector datasource path (e.g., .gpkg)")
    p.add_argument("--regions-layer", default="swe2d_topo_regions", help="Regions polygon layer")
    p.add_argument("--constraints-layer", default="swe2d_topo_constraints", help="Constraints polygon layer")
    p.add_argument("--quad-edges-layer", default="swe2d_topo_quad_edges", help="Quad-edge lines layer")
    p.add_argument("--no-constraints", action="store_true", help="Ignore constraints layer")
    p.add_argument("--no-quad-edges", action="store_true", help="Ignore quad-edges layer")
    p.add_argument("--default-size", type=float, default=20.0, help="Fallback target size")
    p.add_argument(
        "--default-cell-type",
        default="triangular",
        choices=["triangular", "quadrilateral", "cartesian", "empty"],
        help="Fallback cell type",
    )
    p.add_argument("--out-prefix", default="topology_mesh", help="Output file prefix")
    p.add_argument("--write-msh", action="store_true", help="Write Gmsh .msh (v2.2 ascii)")
    p.add_argument(
        "--diagnose-constraints",
        action="store_true",
        help="Run one-at-a-time constraint diagnostics with hard timeout and write diagnostic JSON",
    )
    p.add_argument(
        "--diagnose-timeout-sec",
        type=float,
        default=60.0,
        help="Per-constraint timeout in seconds for --diagnose-constraints",
    )
    p.add_argument(
        "--diagnose-output",
        default="",
        help="Optional diagnostic JSON path (default: <out-prefix>_constraints_diag.json)",
    )
    return p


def main() -> int:
    args = build_arg_parser().parse_args()

    constraints_layer = None if args.no_constraints else args.constraints_layer
    quad_edges_layer = None if args.no_quad_edges else args.quad_edges_layer

    t_all0 = time.perf_counter()

    _log("mesh> load-start")
    t0 = time.perf_counter()
    model = load_conceptual_model(
        source=args.source,
        regions_layer=args.regions_layer,
        constraints_layer=constraints_layer,
        quad_edges_layer=quad_edges_layer,
        default_size=float(args.default_size),
        default_cell_type=str(args.default_cell_type),
    )
    _log(
        "mesh> load-done "
        f"regions={len(model.regions)} constraints={len(model.constraints)} "
        f"quad_edges={len(model.quad_edges)} elapsed={time.perf_counter() - t0:.2f}s"
    )

    if args.diagnose_constraints:
        diag = diagnose_constraints_one_by_one(
            model=model,
            backend_name="gmsh",
            timeout_sec=float(args.diagnose_timeout_sec),
        )
        diag_path = args.diagnose_output.strip() or f"{args.out_prefix}_constraints_diag.json"
        with open(diag_path, "w", encoding="utf-8") as f:
            json.dump(diag, f, indent=2)
        _log(
            "mesh> diagnose-summary "
            f"total={diag['constraints_total']} bad={len(diag['bad_constraints'])} output={diag_path}"
        )
        return 0 if not diag["bad_constraints"] else 2

    _log("mesh> gmsh-start backend=gmsh")
    t1 = time.perf_counter()
    mesh = generate_face_centric_mesh(model, backend="gmsh")
    gmsh_elapsed = time.perf_counter() - t1

    tris, quads = _extract_tri_quad_faces(mesh)
    q = _mesh_quality_stats(
        np.asarray(mesh.node_x, dtype=np.float64),
        np.asarray(mesh.node_y, dtype=np.float64),
        tris,
        quads,
    )

    out_npz = f"{args.out_prefix}.npz"
    out_json = f"{args.out_prefix}.json"
    out_msh = f"{args.out_prefix}.msh"

    np.savez_compressed(
        out_npz,
        node_x=np.asarray(mesh.node_x, dtype=np.float64),
        node_y=np.asarray(mesh.node_y, dtype=np.float64),
        node_z=np.asarray(mesh.node_z, dtype=np.float64),
        cell_nodes=np.asarray(mesh.cell_nodes, dtype=np.int32),
        cell_face_offsets=np.asarray(mesh.cell_face_offsets, dtype=np.int32),
        cell_face_nodes=np.asarray(mesh.cell_face_nodes, dtype=np.int32),
        cell_type=np.asarray(mesh.cell_type),
        region_id=np.asarray(mesh.region_id, dtype=np.int32),
        target_size=np.asarray(mesh.target_size, dtype=np.float64),
    )

    msh_elems = 0
    if args.write_msh:
        msh_elems = write_msh2(mesh, out_msh)

    summary: Dict[str, object] = {
        "source": os.path.abspath(args.source),
        "regions_layer": args.regions_layer,
        "constraints_layer": constraints_layer,
        "quad_edges_layer": quad_edges_layer,
        "regions": len(model.regions),
        "constraints": len(model.constraints),
        "quad_edges": len(model.quad_edges),
        "nodes": int(np.asarray(mesh.node_x).size),
        "cells": int(np.asarray(mesh.cell_face_offsets).size - 1),
        "tris": int(tris.shape[0]),
        "quads": int(quads.shape[0]),
        "quality": {
            "min_angle_deg": float(q.get("min_angle_deg", 0.0)),
            "max_aspect_ratio": float(q.get("max_aspect_ratio", 0.0)),
            "min_area": float(q.get("min_area", 0.0)),
            "bbox_area": float(q.get("bbox_area", 0.0)),
        },
        "timing_sec": {
            "gmsh": float(gmsh_elapsed),
            "total": float(time.perf_counter() - t_all0),
        },
        "outputs": {
            "npz": os.path.abspath(out_npz),
            "json": os.path.abspath(out_json),
            "msh": os.path.abspath(out_msh) if args.write_msh else None,
            "msh_elements": msh_elems if args.write_msh else 0,
        },
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    _log(
        "mesh> gmsh-done "
        f"nodes={summary['nodes']} cells={summary['cells']} tris={summary['tris']} quads={summary['quads']} "
        f"elapsed={gmsh_elapsed:.2f}s"
    )
    _log(
        "mesh> quality "
        f"min_angle_deg={summary['quality']['min_angle_deg']:.3f} "
        f"max_aspect_ratio={summary['quality']['max_aspect_ratio']:.3f} "
        f"min_area={summary['quality']['min_area']:.6g}"
    )
    _log(f"mesh> outputs npz={out_npz} json={out_json}" + (f" msh={out_msh}" if args.write_msh else ""))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
