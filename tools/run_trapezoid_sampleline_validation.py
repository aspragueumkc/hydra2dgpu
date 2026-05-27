#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _load_bed_grid_csv(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs: List[float] = []
    ys: List[float] = []
    zs: List[float] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            xs.append(float(row["x_ft"]))
            ys.append(float(row["y_ft"]))
            zs.append(float(row["bed_z_ft"]))
    if not xs:
        raise RuntimeError(f"Bed grid CSV has no rows: {path}")
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64), np.asarray(zs, dtype=np.float64)


def _apply_bed_grid_to_nodes_layer(dlg, bed_csv_path: str) -> int:
    from qgis.core import QgsField
    from qgis.PyQt.QtCore import QVariant

    nodes_layer = dlg._combo_layer(getattr(dlg, "nodes_layer_combo", None), "vector")
    if nodes_layer is None:
        raise RuntimeError("No nodes layer selected; cannot apply bed grid.")

    field_names = set(nodes_layer.fields().names())
    if "bed_z" not in field_names:
        nodes_layer.dataProvider().addAttributes([QgsField("bed_z", QVariant.Double)])
        nodes_layer.updateFields()

    z_idx = nodes_layer.fields().indexOf("bed_z")

    bx, by, bz = _load_bed_grid_csv(bed_csv_path)
    key = {(round(float(x), 6), round(float(y), 6)): float(z) for x, y, z in zip(bx.tolist(), by.tolist(), bz.tolist())}

    updates = {}
    assigned = 0
    for ft in nodes_layer.getFeatures():
        geom = ft.geometry()
        if geom is None or geom.isEmpty():
            continue
        pt = geom.asPoint()
        k = (round(float(pt.x()), 6), round(float(pt.y()), 6))
        z = key.get(k)
        if z is None:
            # Fallback: nearest neighbor in provided bed grid.
            dx = bx - float(pt.x())
            dy = by - float(pt.y())
            i = int(np.argmin(dx * dx + dy * dy))
            z = float(bz[i])
        updates[int(ft.id())] = {int(z_idx): float(z)}
        assigned += 1

    if updates:
        nodes_layer.dataProvider().changeAttributeValues(updates)
        nodes_layer.triggerRepaint()

    dlg._pull_node_z_from_layer()
    return int(assigned)


def _query_latest_line_flow(gpkg_path: str, line_id: int) -> Tuple[str, np.ndarray, np.ndarray]:
    con = sqlite3.connect(gpkg_path)
    try:
        cur = con.cursor()
        cur.execute(
            "SELECT run_id FROM swe2d_line_results_runs ORDER BY created_utc DESC LIMIT 1"
        )
        row = cur.fetchone()
        if not row:
            raise RuntimeError("No swe2d_line_results_runs rows found after run.")
        run_id = str(row[0])

        cur.execute(
            "SELECT t_s, flow_cms FROM swe2d_line_results_ts "
            "WHERE run_id=? AND line_id=? ORDER BY t_s",
            (run_id, int(line_id)),
        )
        rows = cur.fetchall()
        if not rows:
            raise RuntimeError(f"No line results found for run_id={run_id}, line_id={line_id}.")
        t = np.asarray([float(r[0]) for r in rows], dtype=np.float64)
        q = np.asarray([float(r[1]) for r in rows], dtype=np.float64)
        return run_id, t, q
    finally:
        con.close()


def _query_boundary_forensics(gpkg_path: str, run_id: str) -> Tuple[dict, List[dict]]:
    con = sqlite3.connect(gpkg_path)
    try:
        cur = con.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='swe2d_boundary_flux_forensics_ts'"
        )
        if cur.fetchone() is None:
            return {"available": False}, []

        cur.execute(
            "SELECT COALESCE(MAX(t_s), 0.0) FROM swe2d_boundary_flux_forensics_ts WHERE run_id=?",
            (str(run_id),),
        )
        row = cur.fetchone()
        t_last = float(row[0]) if row and row[0] is not None else 0.0

        cur.execute(
            "SELECT group_name, q_effective_model, vol_effective_model "
            "FROM swe2d_boundary_flux_forensics_ts "
            "WHERE run_id=? AND ABS(t_s - ?) < 1.0e-9 "
            "ORDER BY group_name",
            (str(run_id), float(t_last)),
        )
        grp_rows = cur.fetchall()
        groups: List[dict] = []
        q_net = 0.0
        q_out = 0.0
        q_in = 0.0
        for gname, q_eff, vol_eff in grp_rows:
            qv = float(q_eff or 0.0)
            vv = float(vol_eff or 0.0)
            q_net += qv
            if qv >= 0.0:
                q_out += qv
            else:
                q_in += qv
            groups.append(
                {
                    "group_name": str(gname or ""),
                    "q_effective_model": qv,
                    "vol_effective_model": vv,
                }
            )

        return (
            {
                "available": True,
                "t_last_s": float(t_last),
                "q_net_model": float(q_net),
                "q_out_model": float(q_out),
                "q_in_model": float(q_in),
                "n_groups": int(len(groups)),
            },
            groups,
        )
    finally:
        con.close()


def _collect_line_contribution_debug(dlg, line_id: int) -> dict:
    sample_map = dlg._build_line_sampling_map()
    sm = None
    for cand in sample_map:
        try:
            if int(cand.get("line_id", -1)) == int(line_id):
                sm = cand
                break
        except Exception:
            continue
    if sm is None:
        raise RuntimeError(f"No sample_map entry for line_id={int(line_id)}")

    result_data = dlg._result_data if isinstance(getattr(dlg, "_result_data", None), dict) else None
    if not result_data:
        raise RuntimeError("No in-memory result state found after run.")

    h = np.asarray(result_data.get("h"), dtype=np.float64).ravel()
    hu = np.asarray(result_data.get("hu"), dtype=np.float64).ravel()
    hv = np.asarray(result_data.get("hv"), dtype=np.float64).ravel()
    idx = np.asarray(sm.get("cell_idx", []), dtype=np.int64).ravel()
    w = np.asarray(sm.get("weights", []), dtype=np.float64).ravel()
    flow_wx = np.asarray(sm.get("flow_wx", []), dtype=np.float64).ravel()
    flow_wy = np.asarray(sm.get("flow_wy", []), dtype=np.float64).ravel()
    station = np.asarray(sm.get("station_m", np.arange(idx.size, dtype=np.float64)), dtype=np.float64).ravel()
    if station.size != idx.size:
        station = np.linspace(0.0, float(max(0, idx.size - 1)), idx.size, dtype=np.float64)

    if idx.size == 0:
        raise RuntimeError(f"Sample line {int(line_id)} has no intersecting cells")
    if h.size == 0 or hu.size != h.size or hv.size != h.size:
        raise RuntimeError("Hydraulic state arrays are not aligned")

    h_min = float(dlg.h_min_spin.value()) if hasattr(dlg, "h_min_spin") else 1.0e-6
    hh = h[idx]
    huu = hu[idx]
    hvv = hv[idx]
    wet = hh > h_min
    safe_h = np.maximum(hh, 1.0e-12)
    uu = np.where(wet, huu / safe_h, 0.0)
    vv = np.where(wet, hvv / safe_h, 0.0)
    qn = np.where(wet, hh * (uu * float(sm.get("normal_x", 0.0)) + vv * float(sm.get("normal_y", 0.0))), 0.0)

    if flow_wx.size != idx.size or flow_wy.size != idx.size:
        raise RuntimeError("flow_wx/flow_wy are unavailable for exact line-integral diagnostics")

    contrib_exact = np.where(wet, hh * (uu * flow_wx + vv * flow_wy), 0.0)
    contrib_fallback = np.where(wet, qn * w, 0.0)

    uniq_idx, uniq_counts = np.unique(idx, return_counts=True)
    repeated_cells = int(np.sum(uniq_counts > 1))
    dup_samples = int(np.sum(np.maximum(uniq_counts - 1, 0)))

    rows: List[dict] = []
    for j in range(idx.size):
        rows.append(
            {
                "sample_i": int(j),
                "cell_idx": int(idx[j]),
                "station_m": float(station[j]),
                "wet": int(bool(wet[j])),
                "h": float(hh[j]),
                "hu": float(huu[j]),
                "hv": float(hvv[j]),
                "u": float(uu[j]),
                "v": float(vv[j]),
                "w": float(w[j]) if j < w.size else float("nan"),
                "flow_wx": float(flow_wx[j]),
                "flow_wy": float(flow_wy[j]),
                "qn": float(qn[j]),
                "contrib_exact": float(contrib_exact[j]),
                "contrib_fallback": float(contrib_fallback[j]),
            }
        )

    return {
        "line_id": int(line_id),
        "line_name": str(sm.get("line_name", "") or ""),
        "normal_x": float(sm.get("normal_x", 0.0)),
        "normal_y": float(sm.get("normal_y", 0.0)),
        "n_samples": int(idx.size),
        "n_unique_cells": int(uniq_idx.size),
        "n_repeated_cells": int(repeated_cells),
        "n_duplicate_samples": int(dup_samples),
        "sum_w": float(np.sum(w)) if w.size else 0.0,
        "sum_abs_w": float(np.sum(np.abs(w))) if w.size else 0.0,
        "sum_flow_wx": float(np.sum(flow_wx)),
        "sum_flow_wy": float(np.sum(flow_wy)),
        "sum_abs_flow_wx": float(np.sum(np.abs(flow_wx))),
        "sum_abs_flow_wy": float(np.sum(np.abs(flow_wy))),
        "q_exact": float(np.sum(contrib_exact)),
        "q_fallback": float(np.sum(contrib_fallback)),
        "rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Headless run for trapezoid sample-line flow validation")
    ap.add_argument(
        "--gpkg",
        default=str(Path(_ROOT) / "example_project" / "trapezoid_sampleline_validation.gpkg"),
        help="Path to validation GeoPackage",
    )
    ap.add_argument(
        "--bed-csv",
        default=str(Path(_ROOT) / "example_project" / "trapezoid_sampleline_bed_grid_5ft.csv"),
        help="Path to bed grid CSV",
    )
    ap.add_argument(
        "--meta-json",
        default=str(Path(_ROOT) / "example_project" / "trapezoid_sampleline_validation.json"),
        help="Path to metadata JSON with target discharge",
    )
    ap.add_argument("--run-seconds", type=float, default=120.0, help="Simulation duration in seconds")
    ap.add_argument("--line-id", type=int, default=1, help="Sample line id for validation")
    ap.add_argument(
        "--line-debug-topn",
        type=int,
        default=10,
        help="Print top-N absolute sample-line per-cell contributions",
    )
    args = ap.parse_args()

    gpkg_path = os.path.abspath(str(args.gpkg))
    bed_csv_path = os.path.abspath(str(args.bed_csv))
    meta_json_path = os.path.abspath(str(args.meta_json))

    if not os.path.exists(gpkg_path):
        raise FileNotFoundError(gpkg_path)
    if not os.path.exists(bed_csv_path):
        raise FileNotFoundError(bed_csv_path)

    target_q = float("nan")
    if os.path.exists(meta_json_path):
        with open(meta_json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        target_q = float(meta.get("hydraulics", {}).get("computed_inflow_q_cfs", float("nan")))

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from qgis.core import QgsApplication, QgsProject, QgsVectorLayer
    import swe2d_workbench_qt as wbqt
    from swe2d.mesh.meshing import generate_face_centric_mesh
    from tools.generate_mfem_beta_mesh_qgis_stable import load_conceptual_from_gpkg

    qgs = QgsApplication([], True)
    qgs.initQgis()

    try:
        dlg = wbqt.SWE2DWorkbenchDialog(parent=None, iface=None)
        dlg._load_2d_model_geopackage(path_override=gpkg_path)

        preferred = {
            "topo_nodes_combo": "swe2d_topo_nodes",
            "topo_arcs_combo": "swe2d_topo_arcs",
            "topo_regions_combo": "swe2d_topo_regions",
            "topo_constraints_combo": "swe2d_topo_constraints",
            "topo_quad_edges_combo": "swe2d_topo_quad_edges",
            "manning_layer_combo": "swe2d_manning_zones",
            "bc_lines_layer_combo": "swe2d_bc_lines",
            "sample_lines_layer_combo": "swe2d_sample_lines",
        }
        for attr, layer_name in preferred.items():
            combo = getattr(dlg, attr, None)
            if combo is None:
                continue
            try:
                dlg._set_combo_by_layer_name(combo, layer_name)
            except Exception:
                pass

        # Force structured backend so the 5-ft cartesian region setup is honored.
        try:
            if hasattr(dlg, "topo_backend_combo") and dlg.topo_backend_combo is not None:
                idx = dlg.topo_backend_combo.findData("structured")
                if idx >= 0:
                    dlg.topo_backend_combo.setCurrentIndex(idx)
        except Exception:
            pass

        conceptual = load_conceptual_from_gpkg(gpkg_path)
        mesh = generate_face_centric_mesh(conceptual, backend="structured")
        dlg._mesh_data = {
            "nx": np.array(max(2, int(round(np.sqrt(mesh.node_x.size))))),
            "ny": np.array(max(2, int(round(np.sqrt(mesh.node_x.size))))),
            "lx": np.array(max(float(np.max(mesh.node_x) - np.min(mesh.node_x)), 1.0)),
            "ly": np.array(max(float(np.max(mesh.node_y) - np.min(mesh.node_y)), 1.0)),
            "node_x": mesh.node_x,
            "node_y": mesh.node_y,
            "node_z": mesh.node_z,
            "cell_nodes": mesh.cell_nodes,
            "cell_face_offsets": mesh.cell_face_offsets,
            "cell_face_nodes": mesh.cell_face_nodes,
            "cell_type": mesh.cell_type,
            "region_id": mesh.region_id,
            "target_size": mesh.target_size,
        }
        n_faces = int(max(0, np.asarray(mesh.cell_face_offsets).size - 1))
        print(f"[val] Structured mesh prepared: nodes={int(mesh.node_x.size)} faces={n_faces}")

        # Export mesh to nodes/cells layers, then apply bed grid to nodes and pull back into mesh.
        dlg._export_mesh_to_layers()
        assigned = _apply_bed_grid_to_nodes_layer(dlg, bed_csv_path)
        print(f"[val] Assigned bed_z from CSV to {assigned} mesh node features.")

        # Ensure line/mesh results are persisted for post-run validation.
        if hasattr(dlg, "save_line_results_to_gpkg_chk") and dlg.save_line_results_to_gpkg_chk is not None:
            dlg.save_line_results_to_gpkg_chk.setChecked(True)
        if hasattr(dlg, "save_mesh_results_to_gpkg_chk") and dlg.save_mesh_results_to_gpkg_chk is not None:
            dlg.save_mesh_results_to_gpkg_chk.setChecked(True)

        run_hr = max(1.0e-6, float(args.run_seconds) / 3600.0)
        out_hr = max(1.0e-6, min(run_hr, run_hr / 4.0))
        if hasattr(dlg, "run_time_edit"):
            dlg.run_time_edit.setText(f"{run_hr:.8f}")
        if hasattr(dlg, "output_interval_edit"):
            dlg.output_interval_edit.setText(f"{out_hr:.8f}")
        if hasattr(dlg, "line_output_interval_edit"):
            dlg.line_output_interval_edit.setText(f"{out_hr:.8f}")

        print(f"[val] Running solver for {float(args.run_seconds):.1f} s simulated time...")
        dlg._on_run()
        print("[val] Run finished.")

        run_id, t_s, flow_cms = _query_latest_line_flow(gpkg_path, int(args.line_id))
        n = int(flow_cms.size)
        tail_n = max(1, n // 4)
        q_last = float(flow_cms[-1])
        q_tail = float(np.mean(flow_cms[-tail_n:]))

        print(f"[val] line_results run_id: {run_id}")
        print(f"[val] sample line id={int(args.line_id)} rows={n}")
        print(f"[val] flow_cms last={q_last:.6f} cfs")
        print(f"[val] flow_cms tail_mean(last {tail_n} samples)={q_tail:.6f} cfs")
        if math_isfinite(target_q := float(target_q)):
            rel_last = (q_last - target_q) / max(abs(target_q), 1.0e-12)
            rel_tail = (q_tail - target_q) / max(abs(target_q), 1.0e-12)
            print(f"[val] target_q={target_q:.6f} cfs")
            print(f"[val] rel_error_last={100.0 * rel_last:.3f}%")
            print(f"[val] rel_error_tail_mean={100.0 * rel_tail:.3f}%")

        bsum, bgroups = _query_boundary_forensics(gpkg_path, run_id)
        if not bool(bsum.get("available", False)):
            print("[val] boundary_forensics: table swe2d_boundary_flux_forensics_ts not found")
        else:
            print(
                "[val] boundary_forensics "
                f"t_last={float(bsum.get('t_last_s', 0.0)):.6f}s "
                f"q_net={float(bsum.get('q_net_model', 0.0)):.6f} "
                f"q_out={float(bsum.get('q_out_model', 0.0)):.6f} "
                f"q_in={float(bsum.get('q_in_model', 0.0)):.6f} "
                f"groups={int(bsum.get('n_groups', 0))}"
            )
            for row in bgroups:
                print(
                    "[val] boundary_group "
                    f"name={row['group_name']} "
                    f"q_effective={float(row['q_effective_model']):.6f} "
                    f"vol_effective={float(row['vol_effective_model']):.6f}"
                )

        dbg = _collect_line_contribution_debug(dlg, int(args.line_id))
        print(
            "[val] line_debug "
            f"line_id={int(dbg['line_id'])} "
            f"line_name={dbg['line_name']} "
            f"normal=({float(dbg['normal_x']):.6f},{float(dbg['normal_y']):.6f}) "
            f"samples={int(dbg['n_samples'])} "
            f"unique_cells={int(dbg['n_unique_cells'])} "
            f"repeated_cells={int(dbg['n_repeated_cells'])} "
            f"duplicate_samples={int(dbg['n_duplicate_samples'])}"
        )
        print(
            "[val] line_debug_weights "
            f"sum_w={float(dbg['sum_w']):.6f} "
            f"sum_abs_w={float(dbg['sum_abs_w']):.6f} "
            f"sum_flow_wx={float(dbg['sum_flow_wx']):.6f} "
            f"sum_flow_wy={float(dbg['sum_flow_wy']):.6f} "
            f"sum_abs_flow_wx={float(dbg['sum_abs_flow_wx']):.6f} "
            f"sum_abs_flow_wy={float(dbg['sum_abs_flow_wy']):.6f}"
        )
        print(
            "[val] line_debug_q "
            f"q_exact={float(dbg['q_exact']):.6f} "
            f"q_fallback={float(dbg['q_fallback']):.6f}"
        )

        top_n = max(0, int(args.line_debug_topn))
        if top_n > 0:
            rows = list(dbg.get("rows", []))
            rows.sort(key=lambda r: abs(float(r.get("contrib_exact", 0.0))), reverse=True)
            for r in rows[:top_n]:
                print(
                    "[val] line_debug_cell "
                    f"i={int(r['sample_i'])} "
                    f"cell={int(r['cell_idx'])} "
                    f"station={float(r['station_m']):.3f} "
                    f"wet={int(r['wet'])} "
                    f"h={float(r['h']):.6f} "
                    f"u={float(r['u']):.6f} "
                    f"v={float(r['v']):.6f} "
                    f"w={float(r['w']):.6f} "
                    f"flow_wx={float(r['flow_wx']):.6f} "
                    f"flow_wy={float(r['flow_wy']):.6f} "
                    f"qn={float(r['qn']):.6f} "
                    f"q_exact={float(r['contrib_exact']):.6f} "
                    f"q_fallback={float(r['contrib_fallback']):.6f}"
                )

        return 0
    finally:
        try:
            QgsProject.instance().clear()
        except Exception:
            pass
        qgs.exitQgis()


def math_isfinite(v: float) -> bool:
    return np.isfinite(np.asarray([v], dtype=np.float64))[0]


if __name__ == "__main__":
    raise SystemExit(main())
