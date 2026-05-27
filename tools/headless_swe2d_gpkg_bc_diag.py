#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from typing import Dict, List

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _side_labels(node_x: np.ndarray, node_y: np.ndarray, n0: np.ndarray, n1: np.ndarray) -> List[str]:
    xmin = float(np.min(node_x))
    xmax = float(np.max(node_x))
    ymin = float(np.min(node_y))
    ymax = float(np.max(node_y))
    mx = 0.5 * (node_x[n0] + node_x[n1])
    my = 0.5 * (node_y[n0] + node_y[n1])
    d = np.vstack([np.abs(mx - xmin), np.abs(mx - xmax), np.abs(my - ymin), np.abs(my - ymax)])
    idx = np.argmin(d, axis=0)
    names = ["left", "right", "bottom", "top"]
    return [names[int(i)] for i in idx.tolist()]


def _group_key(i: int, edge_groups: Dict[int, str], side_labels: List[str]) -> str:
    if i in edge_groups:
        return str(edge_groups[i])
    return f"side:{side_labels[i]}"


def _summarize_groups(
    *,
    bc_type: np.ndarray,
    bc_val: np.ndarray,
    edge_len: np.ndarray,
    edge_groups: Dict[int, str],
    side_labels: List[str],
) -> Dict[str, Dict[str, object]]:
    groups: Dict[str, Dict[str, object]] = {}
    flow_idx = np.where(np.asarray(bc_type, dtype=np.int32) == 2)[0]
    for i in flow_idx.tolist():
        k = _group_key(i, edge_groups, side_labels)
        g = groups.setdefault(k, {"idx": [], "q_input": []})
        g["idx"].append(int(i))
        g["q_input"].append(float(bc_val[i]))

    out: Dict[str, Dict[str, object]] = {}
    for k, g in groups.items():
        idx = np.asarray(g["idx"], dtype=np.int32)
        qv = np.asarray(g["q_input"], dtype=np.float64)
        uq = sorted({round(float(v), 12) for v in qv.tolist()})
        out[k] = {
            "n_edges": int(idx.size),
            "q_values": uq,
            "q_first": float(qv[0]) if qv.size else 0.0,
            "length": float(np.sum(edge_len[idx])) if idx.size else 0.0,
            "idx": idx,
        }
    return out


def _print_run_flux_forensics(gpkg_path: str) -> None:
    if not os.path.exists(gpkg_path):
        print(f"[diag] GeoPackage missing: {gpkg_path}")
        return
    con = sqlite3.connect(gpkg_path)
    try:
        cur = con.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='swe2d_boundary_flux_forensics_ts'"
        )
        if cur.fetchone() is None:
            print("[diag] No swe2d_boundary_flux_forensics_ts table present.")
            return

        cur.execute("SELECT run_id, MAX(t_s) FROM swe2d_boundary_flux_forensics_ts GROUP BY run_id ORDER BY MAX(t_s) DESC LIMIT 1")
        row = cur.fetchone()
        if not row:
            print("[diag] Boundary flux forensics table is empty.")
            return
        run_id = str(row[0])
        print(f"[diag] Latest forensic run_id: {run_id}")

        cur.execute(
            "SELECT group_name, COUNT(*), SUM(vol_effective_model), AVG(q_effective_model) "
            "FROM swe2d_boundary_flux_forensics_ts WHERE run_id=? GROUP BY group_name ORDER BY group_name",
            (run_id,),
        )
        rows = cur.fetchall()
        if not rows:
            print("[diag] No forensic rows for latest run.")
            return
        print("[diag] Boundary forensic groups (latest run):")
        for grp, n, vol, qavg in rows:
            print(
                f"  group={grp} rows={int(n)} vol_sum={float(vol or 0.0):.6f} q_avg={float(qavg or 0.0):.6f}"
            )
    finally:
        con.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="Headless SWE2D GeoPackage BC diagnostics")
    ap.add_argument("gpkg", help="Path to SWE2D model GeoPackage")
    ap.add_argument("--run-short", action="store_true", help="Execute a short headless solver run")
    ap.add_argument("--run-seconds", type=float, default=20.0, help="Duration for short run when --run-short is set")
    args = ap.parse_args()

    gpkg_path = os.path.abspath(str(args.gpkg))
    if not os.path.exists(gpkg_path):
        raise FileNotFoundError(gpkg_path)

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from qgis.core import QgsApplication, QgsProject, QgsVectorLayer, QgsFeature, QgsGeometry, QgsField
    from qgis.PyQt.QtCore import QVariant
    import swe2d_workbench_qt as wbqt

    qgs = QgsApplication([], True)
    qgs.initQgis()
    try:
        dlg = wbqt.SWE2DWorkbenchDialog(parent=None, iface=None)
        dlg._load_2d_model_geopackage(path_override=gpkg_path)

        # Prefer canonical SWE2D topology/boundary layers for headless meshing.
        preferred = {
            "topo_nodes_combo": "swe2d_topo_nodes",
            "topo_arcs_combo": "swe2d_topo_arcs",
            "topo_regions_combo": "swe2d_topo_regions",
            "topo_constraints_combo": "swe2d_topo_constraints",
            "topo_quad_edges_combo": "swe2d_topo_quad_edges",
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

        # Bind mesh node/cell layers if present in the GeoPackage.
        mesh_candidates = {
            "nodes_layer_combo": ["SWE2D_Mesh_Nodes", "swe2d_mesh_nodes"],
            "cells_layer_combo": ["SWE2D_Mesh_Cells", "swe2d_mesh_cells"],
        }
        for combo_attr, layer_names in mesh_candidates.items():
            combo = getattr(dlg, combo_attr, None)
            if combo is None:
                continue
            bound = False
            for lname in layer_names:
                try:
                    lyr = QgsVectorLayer(f"{gpkg_path}|layername={lname}", lname, "ogr")
                except Exception:
                    lyr = None
                if lyr is None or not lyr.isValid():
                    continue
                # Ensure the layer is in project so combo data includes it.
                try:
                    QgsProject.instance().addMapLayer(lyr)
                except Exception:
                    pass
                try:
                    dlg._refresh_layer_combos()
                except Exception:
                    pass
                try:
                    idx = combo.findData(lyr.id())
                    if idx >= 0:
                        combo.setCurrentIndex(idx)
                        bound = True
                        break
                except Exception:
                    pass
                try:
                    dlg._set_combo_by_layer_name(combo, lname)
                    bound = True
                    break
                except Exception:
                    pass
            if bound:
                print(f"[diag] Bound {combo_attr} from GeoPackage mesh layers")

        # Some model files contain arcs/regions but no explicit topo node rows.
        # Synthesize nodes from arc endpoints so topology meshing can proceed.
        topo_nodes_layer = dlg._combo_layer(getattr(dlg, "topo_nodes_combo", None), "vector")
        topo_arcs_layer = dlg._combo_layer(getattr(dlg, "topo_arcs_combo", None), "vector")
        topo_nodes_count = 0
        if topo_nodes_layer is not None:
            try:
                topo_nodes_count = int(topo_nodes_layer.featureCount())
            except Exception:
                topo_nodes_count = 0
        if topo_nodes_count <= 0 and topo_arcs_layer is not None:
            node_layer = QgsVectorLayer("Point?crs=EPSG:4326", "swe2d_topo_nodes_synth", "memory")
            pr = node_layer.dataProvider()
            pr.addAttributes([QgsField("node_id", QVariant.Int)])
            node_layer.updateFields()

            pts = []
            seen = set()
            for ft in topo_arcs_layer.getFeatures():
                g = ft.geometry()
                if g is None or g.isEmpty():
                    continue
                lines = []
                try:
                    if g.isMultipart():
                        lines = list(g.asMultiPolyline() or [])
                    else:
                        ln = g.asPolyline()
                        if ln:
                            lines = [ln]
                except Exception:
                    lines = []
                for ln in lines:
                    if not ln:
                        continue
                    for p in (ln[0], ln[-1]):
                        key = (round(float(p.x()), 9), round(float(p.y()), 9))
                        if key in seen:
                            continue
                        seen.add(key)
                        pts.append((float(p.x()), float(p.y())))

            feats = []
            for i, (x, y) in enumerate(pts, start=1):
                f = QgsFeature(node_layer.fields())
                f.setGeometry(QgsGeometry.fromWkt(f"POINT({x} {y})"))
                f.setAttribute("node_id", int(i))
                feats.append(f)
            if feats:
                pr.addFeatures(feats)
                node_layer.updateExtents()
                QgsProject.instance().addMapLayer(node_layer)
                try:
                    dlg._refresh_layer_combos()
                    combo = getattr(dlg, "topo_nodes_combo", None)
                    if combo is not None:
                        idx = combo.findData(node_layer.id())
                        if idx >= 0:
                            combo.setCurrentIndex(idx)
                        else:
                            dlg._set_combo_by_layer_name(combo, node_layer.name())
                except Exception:
                    pass
                print(f"[diag] Synthesized topo nodes from arcs: {len(feats)}")

        if dlg._mesh_data is None:
            dlg._generate_mesh_from_topology_layers()
        if dlg._mesh_data is None:
            dlg._import_mesh_from_layers()
        if dlg._mesh_data is None:
            print("[diag] Recent workbench logs:")
            for line in list(getattr(dlg, "_runtime_log_lines", []) or [])[-80:]:
                print(f"  {line}")
            print("[diag] Mesh data unavailable after load/generate/import.")
            return 2

        node_x = np.asarray(dlg._mesh_data["node_x"], dtype=np.float64)
        node_y = np.asarray(dlg._mesh_data["node_y"], dtype=np.float64)
        print(
            f"[diag] Mesh: nodes={node_x.size}, cells={int(np.asarray(dlg._mesh_data['cell_nodes']).size // 3)}"
        )

        bc_n0, bc_n1, bc_tp, bc_vl = dlg._collect_boundary_arrays()
        edge_groups = dlg._collect_bc_layer_edge_groups(bc_n0, bc_n1)
        edge_hydro = dlg._collect_bc_layer_hydrographs(bc_n0, bc_n1)
        side_hg = dlg._build_side_hydrographs()

        edge_len = np.hypot(node_x[bc_n1] - node_x[bc_n0], node_y[bc_n1] - node_y[bc_n0]).astype(np.float64)
        side_labels = _side_labels(node_x, node_y, bc_n0, bc_n1)

        static_groups = _summarize_groups(
            bc_type=bc_tp,
            bc_val=bc_vl,
            edge_len=edge_len,
            edge_groups=edge_groups,
            side_labels=side_labels,
        )
        print("[diag] Static flow BC groups before Q->q distribution:")
        if not static_groups:
            print("  (no flow-type boundary edges)")
        for grp in sorted(static_groups):
            info = static_groups[grp]
            q_vals = info["q_values"]
            print(
                f"  {grp}: edges={info['n_edges']} len={info['length']:.6f} "
                f"q_first={info['q_first']:.6f} q_values={q_vals}"
            )
            if len(q_vals) > 1:
                print("    [warn] multiple distinct total-Q values inside one group")

        bc_vl_dist = dlg._distribute_total_flow_to_unit_q(
            bc_n0,
            bc_n1,
            bc_tp,
            bc_vl,
            bc_tp,
            side_hg,
            edge_hydro,
            edge_groups,
        )
        print("[diag] Reconstructed applied total-Q per group after distribution:")
        for grp in sorted(static_groups):
            idx = static_groups[grp]["idx"]
            q_recon = float(np.sum(np.asarray(bc_vl_dist, dtype=np.float64)[idx] * edge_len[idx]))
            print(f"  {grp}: q_reconstructed={q_recon:.6f} from_edges={int(idx.size)}")

        if args.run_short:
            run_hr = max(1.0e-6, float(args.run_seconds) / 3600.0)
            out_hr = max(1.0e-6, min(run_hr, run_hr / 2.0))
            if hasattr(dlg, "save_line_results_to_gpkg_chk") and dlg.save_line_results_to_gpkg_chk is not None:
                try:
                    dlg.save_line_results_to_gpkg_chk.setChecked(True)
                except Exception:
                    pass
            if hasattr(dlg, "save_mesh_results_to_gpkg_chk") and dlg.save_mesh_results_to_gpkg_chk is not None:
                try:
                    dlg.save_mesh_results_to_gpkg_chk.setChecked(True)
                except Exception:
                    pass
            if hasattr(dlg, "save_run_log_to_gpkg_chk") and dlg.save_run_log_to_gpkg_chk is not None:
                try:
                    dlg.save_run_log_to_gpkg_chk.setChecked(True)
                except Exception:
                    pass
            if hasattr(dlg, "run_time_edit"):
                dlg.run_time_edit.setText(f"{run_hr:.8f}")
            if hasattr(dlg, "output_interval_edit"):
                dlg.output_interval_edit.setText(f"{out_hr:.8f}")
            if hasattr(dlg, "line_output_interval_edit"):
                dlg.line_output_interval_edit.setText(f"{out_hr:.8f}")

            print(f"[diag] Running headless solver for ~{float(args.run_seconds):.2f} s simulated time...")
            dlg._on_run()
            print("[diag] Run finished.")
            _print_run_flux_forensics(gpkg_path)

        print(f"[diag] Loaded layers in project: {len(QgsProject.instance().mapLayers())}")
        return 0
    finally:
        try:
            QgsProject.instance().clear()
        except Exception:
            pass
        qgs.exitQgis()


if __name__ == "__main__":
    raise SystemExit(main())
