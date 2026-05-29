#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def _set_combo_by_data(combo, data) -> bool:
    try:
        idx = combo.findData(data)
        if idx >= 0:
            combo.setCurrentIndex(idx)
            return True
    except Exception:
        pass
    return False


def _set_combo_none(combo) -> bool:
    try:
        idx = combo.findData(None)
        if idx >= 0:
            combo.setCurrentIndex(idx)
            return True
        if combo.count() > 0:
            combo.setCurrentIndex(0)
            return True
    except Exception:
        pass
    return False


def _set_combo_by_data_text(combo, data_text: str) -> bool:
    target = str(data_text or "").strip().lower()
    if not target:
        return False
    try:
        for i in range(combo.count()):
            item_data = str(combo.itemData(i) or "").strip().lower()
            if item_data == target:
                combo.setCurrentIndex(i)
                return True
    except Exception:
        pass
    return False


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    txt = str(raw).strip().lower()
    if txt in {"1", "true", "yes", "on"}:
        return True
    if txt in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return int(default)
    try:
        return int(round(float(raw)))
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _polyline_metrics(points) -> Dict[str, object]:
    pts = []
    for p in list(points or []):
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            try:
                pts.append((float(p[0]), float(p[1])))
            except Exception:
                continue
    if not pts:
        return {
            "n_vertices": 0,
            "length": 0.0,
            "bbox": None,
            "points_sha256": "",
        }
    length = 0.0
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        length += float(np.hypot(float(x1) - float(x0), float(y1) - float(y0)))
    xs = [float(p[0]) for p in pts]
    ys = [float(p[1]) for p in pts]
    pts_norm = [[round(float(x), 9), round(float(y), 9)] for x, y in pts]
    pts_txt = json.dumps(pts_norm, separators=(",", ":"), ensure_ascii=True)
    pts_sha = hashlib.sha256(pts_txt.encode("utf-8")).hexdigest()
    return {
        "n_vertices": int(len(pts)),
        "length": float(length),
        "bbox": [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))],
        "points_sha256": str(pts_sha),
    }


def _ring_metrics(ring) -> Dict[str, object]:
    pts = []
    for p in list(ring or []):
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            try:
                pts.append((float(p[0]), float(p[1])))
            except Exception:
                continue
    if len(pts) >= 2:
        if float(np.hypot(float(pts[0][0]) - float(pts[-1][0]), float(pts[0][1]) - float(pts[-1][1]))) <= 1.0e-12:
            pts = pts[:-1]
    poly = _polyline_metrics(pts)
    if len(pts) >= 2:
        poly["perimeter"] = float(poly.get("length", 0.0) + np.hypot(
            float(pts[0][0]) - float(pts[-1][0]),
            float(pts[0][1]) - float(pts[-1][1]),
        ))
    else:
        poly["perimeter"] = 0.0
    area = 0.0
    if len(pts) >= 3:
        acc = 0.0
        for i in range(len(pts)):
            x0, y0 = pts[i]
            x1, y1 = pts[(i + 1) % len(pts)]
            acc += float(x0) * float(y1) - float(x1) * float(y0)
        area = 0.5 * acc
    poly["signed_area"] = float(area)
    return poly


def _build_conceptual_from_dialog(dlg, wbqt):
    combo_layer = getattr(dlg, "_combo_layer", None)
    if not callable(combo_layer):
        raise RuntimeError("Dialog does not expose _combo_layer")

    nodes_layer = combo_layer(getattr(dlg, "topo_nodes_combo"), "vector")
    arcs_layer = combo_layer(getattr(dlg, "topo_arcs_combo"), "vector")
    regions_layer = combo_layer(getattr(dlg, "topo_regions_combo"), "vector")
    constraints_layer = combo_layer(getattr(dlg, "topo_constraints_combo"), "vector")
    quad_edges_layer = combo_layer(getattr(dlg, "topo_quad_edges_combo"), "vector")

    if getattr(getattr(dlg, "topo_nodes_combo", None), "currentData", lambda: None)() is None:
        nodes_layer = None
    if getattr(getattr(dlg, "topo_arcs_combo", None), "currentData", lambda: None)() is None:
        arcs_layer = None
    if getattr(getattr(dlg, "topo_constraints_combo", None), "currentData", lambda: None)() is None:
        constraints_layer = None
    if getattr(getattr(dlg, "topo_quad_edges_combo", None), "currentData", lambda: None)() is None:
        quad_edges_layer = None

    builder = getattr(wbqt, "conceptual_from_qgis_layers", None)
    if not callable(builder):
        raise RuntimeError("conceptual_from_qgis_layers callable not available")

    return builder(
        nodes_layer=nodes_layer,
        arcs_layer=arcs_layer,
        regions_layer=regions_layer,
        constraints_layer=constraints_layer,
        quad_edges_layer=quad_edges_layer,
        default_size=float(getattr(getattr(dlg, "topo_default_size_spin", None), "value", lambda: 20.0)()),
        default_cell_type=str(getattr(getattr(dlg, "topo_default_cell_type_combo", None), "currentText", lambda: "triangular")()),
    )


def _conceptual_digest(conceptual) -> Dict[str, object]:
    nodes = sorted(
        [
            {
                "node_id": int(getattr(n, "node_id", -1)),
                "x": round(float(getattr(n, "x", 0.0)), 9),
                "y": round(float(getattr(n, "y", 0.0)), 9),
            }
            for n in list(getattr(conceptual, "nodes", []) or [])
        ],
        key=lambda r: (int(r.get("node_id", -1)), float(r.get("x", 0.0)), float(r.get("y", 0.0))),
    )

    arcs = []
    for a in list(getattr(conceptual, "arcs", []) or []):
        pm = _polyline_metrics(getattr(a, "points_xy", None))
        arcs.append(
            {
                "arc_id": int(getattr(a, "arc_id", -1)),
                "node0": int(getattr(a, "node0", -1)),
                "node1": int(getattr(a, "node1", -1)),
                "region_id": int(getattr(a, "region_id", -1)),
                "arc_role": str(getattr(a, "arc_role", "") or ""),
                "use_global_arc_ctrl": bool(getattr(a, "use_global_arc_ctrl", True)),
                "arc_mode_override": str(getattr(a, "arc_mode_override", "") or ""),
                "arc_soft_size_override": round(float(getattr(a, "arc_soft_size_override", 0.0) or 0.0), 9),
                "arc_soft_dist_override": round(float(getattr(a, "arc_soft_dist_override", 0.0) or 0.0), 9),
                "n_vertices": int(pm.get("n_vertices", 0)),
                "length": round(float(pm.get("length", 0.0)), 9),
                "bbox": pm.get("bbox"),
                "points_sha256": str(pm.get("points_sha256", "")),
            }
        )
    arcs = sorted(arcs, key=lambda r: (int(r.get("arc_id", -1)), int(r.get("region_id", -1))))

    regions = []
    for r in list(getattr(conceptual, "regions", []) or []):
        rm = _ring_metrics(getattr(r, "ring_xy", None))
        hole_rings = list(getattr(r, "hole_rings", []) or [])
        hole_hashes = [_ring_metrics(h).get("points_sha256", "") for h in hole_rings]
        regions.append(
            {
                "region_id": int(getattr(r, "region_id", -1)),
                "default_size": round(float(getattr(r, "default_size", 0.0)), 9),
                "default_cell_type": str(getattr(r, "default_cell_type", "") or ""),
                "edge_lengths": [round(float(v), 9) for v in list(getattr(r, "edge_lengths", []) or [])],
                "ring_vertices": int(rm.get("n_vertices", 0)),
                "ring_perimeter": round(float(rm.get("perimeter", 0.0)), 9),
                "ring_signed_area": round(float(rm.get("signed_area", 0.0)), 9),
                "ring_bbox": rm.get("bbox"),
                "ring_sha256": str(rm.get("points_sha256", "")),
                "hole_count": int(len(hole_rings)),
                "hole_rings_sha256": sorted(str(v) for v in hole_hashes if str(v)),
            }
        )
    regions = sorted(regions, key=lambda r: int(r.get("region_id", -1)))

    constraints = []
    for c in list(getattr(conceptual, "constraints", []) or []):
        cm = _ring_metrics(getattr(c, "ring_xy", None))
        constraints.append(
            {
                "constraint_id": int(getattr(c, "constraint_id", -1)),
                "target_size": round(float(getattr(c, "target_size", 0.0)), 9),
                "cell_type": str(getattr(c, "cell_type", "") or ""),
                "ring_vertices": int(cm.get("n_vertices", 0)),
                "ring_perimeter": round(float(cm.get("perimeter", 0.0)), 9),
                "ring_signed_area": round(float(cm.get("signed_area", 0.0)), 9),
                "ring_bbox": cm.get("bbox"),
                "ring_sha256": str(cm.get("points_sha256", "")),
            }
        )
    constraints = sorted(constraints, key=lambda r: int(r.get("constraint_id", -1)))

    quad_edges = []
    for q in list(getattr(conceptual, "quad_edges", []) or []):
        qm = _polyline_metrics(getattr(q, "points_xy", None))
        quad_edges.append(
            {
                "region_id": int(getattr(q, "region_id", -1)),
                "edge_id": int(getattr(q, "edge_id", -1)),
                "target_size": round(float(getattr(q, "target_size", 0.0) or 0.0), 9),
                "n_layers": int(getattr(q, "n_layers", 0) or 0),
                "first_height": round(float(getattr(q, "first_height", 0.0) or 0.0), 9),
                "growth_rate": round(float(getattr(q, "growth_rate", 1.0) or 1.0), 9),
                "n_vertices": int(qm.get("n_vertices", 0)),
                "length": round(float(qm.get("length", 0.0)), 9),
                "bbox": qm.get("bbox"),
                "points_sha256": str(qm.get("points_sha256", "")),
            }
        )
    quad_edges = sorted(quad_edges, key=lambda r: (int(r.get("region_id", -1)), int(r.get("edge_id", -1))))

    canonical = {
        "nodes": nodes,
        "arcs": arcs,
        "regions": regions,
        "constraints": constraints,
        "quad_edges": quad_edges,
    }
    canonical_txt = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(canonical_txt.encode("utf-8")).hexdigest()
    return {
        "sha256": str(digest),
        "counts": {
            "nodes": int(len(nodes)),
            "arcs": int(len(arcs)),
            "regions": int(len(regions)),
            "constraints": int(len(constraints)),
            "quad_edges": int(len(quad_edges)),
        },
        "regions": regions,
        "arcs": arcs,
        "constraints": constraints,
        "quad_edges": quad_edges,
    }


def _save_mesh_png(node_x: np.ndarray, node_y: np.ndarray, cell_nodes: np.ndarray, out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    tris = np.asarray(cell_nodes, dtype=np.int32).reshape((-1, 3))
    triang = mtri.Triangulation(np.asarray(node_x, dtype=np.float64), np.asarray(node_y, dtype=np.float64), tris)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=160)
    ax.triplot(triang, color="#1f2937", linewidth=0.22, alpha=0.85)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Headless GUI Topology Mesh (Gmsh full-region flow-aligned quads)")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def _read_json_file(path: str) -> Optional[Dict[str, object]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        return None
    return None


def _clip_text(value: object, max_len: int = 220) -> str:
    txt = str(value or "").strip()
    if len(txt) <= max_len:
        return txt
    return txt[: max_len - 3] + "..."


def main() -> int:
    script_started_at = time.perf_counter()
    phase_times_s: Dict[str, float] = {}
    phase_marks_s: Dict[str, float] = {}
    timeline: List[Dict[str, object]] = []

    def _phase_start() -> float:
        return time.perf_counter()

    def _phase_add(name: str, started_at: float) -> None:
        dt = float(max(0.0, time.perf_counter() - started_at))
        phase_times_s[name] = float(phase_times_s.get(name, 0.0) + dt)

    def _phase_mark(name: str) -> None:
        phase_marks_s[name] = float(max(0.0, time.perf_counter() - script_started_at))

    def _timeline(event: str, **fields: object) -> None:
        row: Dict[str, object] = {
            "t_s": float(max(0.0, time.perf_counter() - script_started_at)),
            "event": str(event),
        }
        for k, v in fields.items():
            if v is None:
                continue
            row[str(k)] = v
        timeline.append(row)

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ["BACKWATER_GMSH_QUAD_FULL_REGION_FLOW_ALIGN"] = "1"
    _phase_mark("env_ready")

    t_phase = _phase_start()
    repo_root = Path(__file__).resolve().parents[1]
    plugin_profile_dir = Path.home() / ".local/share/QGIS/QGIS3/profiles/default/python/plugins/qgis-backwater-plugin"
    for p in (repo_root, plugin_profile_dir, plugin_profile_dir / "build"):
        p_txt = str(p)
        if p.exists() and p_txt not in sys.path:
            sys.path.insert(0, p_txt)
    _phase_add("path_setup", t_phase)

    gpkg_path = repo_root / "qgis_testing_project" / "swe3d_model.gpkg"
    if not gpkg_path.exists():
        raise FileNotFoundError(f"GeoPackage not found: {gpkg_path}")

    region_ids_raw = (os.environ.get("HEADLESS_GMSH_REGION_IDS", "4,5") or "4,5").strip()
    region_ids: List[int] = []
    for tok in region_ids_raw.replace(";", ",").split(","):
        t = tok.strip()
        if not t:
            continue
        region_ids.append(int(float(t)))
    if not region_ids:
        region_ids = [4, 5]
    region_ids = sorted(set(int(v) for v in region_ids))
    region_ids_sql = ",".join(str(int(v)) for v in region_ids)
    _timeline("region_scope", requested=str(region_ids_raw), resolved=region_ids)

    out_dir = repo_root / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_prefix = out_dir / f"headless_gui_gmsh_full_align_swe3d_{stamp}"
    out_npz = out_prefix.with_suffix(".npz")
    out_json = out_prefix.with_suffix(".json")
    out_png = out_prefix.with_suffix(".png")
    out_profile_json = out_prefix.with_suffix(".profile.json")

    t_phase = _phase_start()
    from qgis.core import QgsApplication
    from qgis.PyQt.QtWidgets import QApplication
    import swe2d_workbench_qt as wbqt
    _phase_add("imports_qgis_workbench", t_phase)

    t_phase = _phase_start()
    qgs = QgsApplication([], True)
    qgs.initQgis()
    _phase_add("qgis_init", t_phase)
    _phase_mark("qgis_initialized")

    try:
        layer_snapshot: Dict[str, object] = {}
        options_snapshot: Dict[str, object] = {}
        conceptual_digest: Dict[str, object] = {}

        t_phase = _phase_start()
        dlg = wbqt.SWE2DWorkbenchDialog(parent=None, iface=None)
        _phase_add("dialog_init", t_phase)

        t_phase = _phase_start()
        dlg._load_2d_model_geopackage(path_override=str(gpkg_path))
        _phase_add("load_geopackage", t_phase)

        t_phase = _phase_start()
        preferred_layers = {
            "topo_nodes_combo": "swe2d_topo_nodes",
            "topo_arcs_combo": "swe2d_topo_arcs",
            "topo_regions_combo": "swe2d_topo_regions",
            "topo_constraints_combo": "swe2d_topo_constraints",
            "topo_quad_edges_combo": "swe2d_topo_quad_edges",
        }
        for attr, layer_name in preferred_layers.items():
            combo = getattr(dlg, attr, None)
            if combo is None:
                continue
            try:
                dlg._set_combo_by_layer_name(combo, layer_name)
            except Exception:
                pass
        _phase_add("bind_preferred_layers", t_phase)

        t_phase = _phase_start()
        regions_layer = dlg._combo_layer(dlg.topo_regions_combo, "vector")
        if regions_layer is None:
            raise RuntimeError("Topology regions layer not resolved")
        apply_region_subset = _env_bool("HEADLESS_APPLY_REGION_SUBSET", True)
        if apply_region_subset:
            regions_layer.setSubsetString(f"region_id IN ({region_ids_sql})")
        else:
            regions_layer.setSubsetString("")

        quad_edges_layer = dlg._combo_layer(dlg.topo_quad_edges_combo, "vector")
        if quad_edges_layer is not None:
            if apply_region_subset:
                quad_edges_layer.setSubsetString(f"region_id IN ({region_ids_sql})")
            else:
                quad_edges_layer.setSubsetString("")

        subset_region_ids: List[int] = []
        try:
            for feat in regions_layer.getFeatures():
                try:
                    rid = int(float(feat["region_id"]))
                except Exception:
                    continue
                subset_region_ids.append(rid)
        except Exception:
            subset_region_ids = []

        layer_snapshot = {
            "regions_subset_sql": str(regions_layer.subsetString() or ""),
            "regions_subset_count": int(regions_layer.featureCount()),
            "regions_subset_ids": sorted(set(int(v) for v in subset_region_ids)),
            "quad_edges_subset_sql": str(quad_edges_layer.subsetString() or "") if quad_edges_layer is not None else "",
            "quad_edges_subset_count": int(quad_edges_layer.featureCount()) if quad_edges_layer is not None else 0,
        }
        _phase_add("subset_layers_and_snapshot", t_phase)
        _timeline("layer_snapshot", regions=layer_snapshot.get("regions_subset_count"), quad_edges=layer_snapshot.get("quad_edges_subset_count"))

        t_phase = _phase_start()
        topo_nodes_mode = str(os.environ.get("HEADLESS_TOPO_NODES_MODE", "none") or "none").strip().lower()
        topo_arcs_mode = str(os.environ.get("HEADLESS_TOPO_ARCS_MODE", "none") or "none").strip().lower()
        topo_constraints_mode = str(os.environ.get("HEADLESS_TOPO_CONSTRAINTS_MODE", "none") or "none").strip().lower()

        if topo_nodes_mode in {"none", "off", "disable", "disabled"}:
            if hasattr(dlg, "topo_nodes_combo") and dlg.topo_nodes_combo is not None:
                _set_combo_none(dlg.topo_nodes_combo)

        if topo_arcs_mode in {"none", "off", "disable", "disabled"}:
            if hasattr(dlg, "topo_arcs_combo") and dlg.topo_arcs_combo is not None:
                _set_combo_none(dlg.topo_arcs_combo)

        if topo_constraints_mode in {"none", "off", "disable", "disabled"}:
            if hasattr(dlg, "topo_constraints_combo") and dlg.topo_constraints_combo is not None:
                _set_combo_none(dlg.topo_constraints_combo)

        gmsh_arc_mode = str(os.environ.get("HEADLESS_GMSH_ARC_MODE", "") or "").strip().lower()
        if gmsh_arc_mode and hasattr(dlg, "topo_gmsh_arc_mode_combo") and dlg.topo_gmsh_arc_mode_combo is not None:
            _set_combo_by_data_text(dlg.topo_gmsh_arc_mode_combo, gmsh_arc_mode)

        if hasattr(dlg, "topo_backend_combo") and dlg.topo_backend_combo is not None:
            if not _set_combo_by_data(dlg.topo_backend_combo, "gmsh"):
                raise RuntimeError("Could not select gmsh backend in topology controls")

        if hasattr(dlg, "topo_gmsh_quad_full_region_flow_align_chk") and dlg.topo_gmsh_quad_full_region_flow_align_chk is not None:
            dlg.topo_gmsh_quad_full_region_flow_align_chk.setChecked(True)
        else:
            raise RuntimeError("Gmsh full-region flow-aligned checkbox control not found")

        gmsh_quality_enable = _env_bool("HEADLESS_GMSH_QUALITY_ENABLE", False)
        gmsh_smoothing = max(0, _env_int("HEADLESS_GMSH_SMOOTHING", 0))
        gmsh_optimize_iters = max(0, _env_int("HEADLESS_GMSH_OPTIMIZE_ITERS", 0))
        gmsh_verbosity = max(0, _env_int("HEADLESS_GMSH_VERBOSITY", 2))
        gmsh_quality_max_iters = max(1, _env_int("HEADLESS_GMSH_QUALITY_MAX_ITERS", 2))
        gmsh_quality_time_limit_s = max(1.0, _env_float("HEADLESS_GMSH_QUALITY_TIME_LIMIT_S", 55.0))
        gmsh_algo_switch_on_failure = _env_bool("HEADLESS_GMSH_ALGO_SWITCH_ON_FAILURE", False)
        gmsh_mesh_size_from_points = _env_bool("HEADLESS_GMSH_MESH_SIZE_FROM_POINTS", False)

        if hasattr(dlg, "topo_gmsh_quality_enable_chk") and dlg.topo_gmsh_quality_enable_chk is not None:
            dlg.topo_gmsh_quality_enable_chk.setChecked(bool(gmsh_quality_enable))
        if hasattr(dlg, "topo_gmsh_smoothing_spin") and dlg.topo_gmsh_smoothing_spin is not None:
            dlg.topo_gmsh_smoothing_spin.setValue(int(gmsh_smoothing))
        if hasattr(dlg, "topo_gmsh_optimize_iters_spin") and dlg.topo_gmsh_optimize_iters_spin is not None:
            dlg.topo_gmsh_optimize_iters_spin.setValue(int(gmsh_optimize_iters))
        if hasattr(dlg, "topo_gmsh_verbosity_spin") and dlg.topo_gmsh_verbosity_spin is not None:
            dlg.topo_gmsh_verbosity_spin.setValue(int(gmsh_verbosity))
        if hasattr(dlg, "topo_gmsh_quality_max_iters_spin") and dlg.topo_gmsh_quality_max_iters_spin is not None:
            dlg.topo_gmsh_quality_max_iters_spin.setValue(int(gmsh_quality_max_iters))
        if hasattr(dlg, "topo_gmsh_quality_time_limit_spin") and dlg.topo_gmsh_quality_time_limit_spin is not None:
            dlg.topo_gmsh_quality_time_limit_spin.setValue(float(gmsh_quality_time_limit_s))
        if hasattr(dlg, "topo_gmsh_algo_switch_on_failure_chk") and dlg.topo_gmsh_algo_switch_on_failure_chk is not None:
            dlg.topo_gmsh_algo_switch_on_failure_chk.setChecked(bool(gmsh_algo_switch_on_failure))
        if hasattr(dlg, "topo_gmsh_mesh_size_from_points_chk") and dlg.topo_gmsh_mesh_size_from_points_chk is not None:
            dlg.topo_gmsh_mesh_size_from_points_chk.setChecked(bool(gmsh_mesh_size_from_points))
        _phase_add("configure_ui_controls", t_phase)

        t_phase = _phase_start()
        opts = dlg._build_topology_meshing_options()
        if not bool(opts.get("gmsh_quad_full_region_flow_align", False)):
            raise RuntimeError("Topology options did not capture gmsh_quad_full_region_flow_align=True")
        options_snapshot = {
            "gmsh_quad_full_region_flow_align": bool(opts.get("gmsh_quad_full_region_flow_align", False)),
            "gmsh_arc_mode": str(opts.get("gmsh_arc_mode", "")),
            "gmsh_quality_enable": bool(opts.get("gmsh_quality_enable", False)),
            "gmsh_quality_max_iterations": int(opts.get("gmsh_quality_max_iterations", 0) or 0),
            "gmsh_quality_time_limit_s": float(opts.get("gmsh_quality_time_limit_s", 0.0) or 0.0),
            "gmsh_smoothing": int(opts.get("gmsh_smoothing", 0) or 0),
            "gmsh_optimize_iters": int(opts.get("gmsh_optimize_iters", 0) or 0),
            "gmsh_verbosity": int(opts.get("gmsh_verbosity", 0) or 0),
            "gmsh_algo_switch_on_failure": bool(
                opts.get("gmsh_algorithm_switch_on_failure", opts.get("gmsh_algo_switch_on_failure", False))
            ),
            "gmsh_mesh_size_from_points": bool(opts.get("gmsh_mesh_size_from_points", False)),
        }
        _phase_add("build_topology_options", t_phase)

        t_phase = _phase_start()
        conceptual = _build_conceptual_from_dialog(dlg, wbqt)
        conceptual_digest = _conceptual_digest(conceptual)
        _phase_add("build_conceptual_digest", t_phase)

        t_phase = _phase_start()
        mesh_run_started_at = time.perf_counter()
        dlg._generate_mesh_from_topology_layers()
        _phase_add("dispatch_async_meshing", t_phase)
        _phase_mark("meshing_dispatched")

        timeout_s = float(os.environ.get("HEADLESS_GMSH_TIMEOUT_S", "240"))
        wait_started_at = time.perf_counter()
        last_log_t = wait_started_at
        progress_path = str(getattr(dlg, "_topology_mesh_progress_path", "") or "").strip()
        last_progress_seq = -1
        last_progress_sig = ""
        last_status_txt = ""

        async_wait_stats: Dict[str, object] = {
            "timeout_s": float(timeout_s),
            "iterations": 0,
            "process_events_s": 0.0,
            "progress_file_read_s": 0.0,
            "sleep_s": 0.0,
            "progress_events_observed": 0,
            "status_changes_observed": 0,
            "progress_path": progress_path,
        }

        while True:
            t_proc = time.perf_counter()
            QApplication.processEvents()
            async_wait_stats["process_events_s"] = float(async_wait_stats["process_events_s"]) + float(max(0.0, time.perf_counter() - t_proc))

            fut = getattr(dlg, "_topology_mesh_future", None)
            if fut is None:
                break

            now = time.perf_counter()
            status_txt = ""
            try:
                status_txt = str(dlg.topo_status_lbl.text() or "")
            except Exception:
                status_txt = ""

            if status_txt != last_status_txt:
                _timeline("status_change", status=_clip_text(status_txt, 260))
                last_status_txt = status_txt
                async_wait_stats["status_changes_observed"] = int(async_wait_stats["status_changes_observed"]) + 1

            if progress_path:
                t_read = time.perf_counter()
                progress_payload = _read_json_file(progress_path)
                async_wait_stats["progress_file_read_s"] = float(async_wait_stats["progress_file_read_s"]) + float(max(0.0, time.perf_counter() - t_read))
                if isinstance(progress_payload, dict):
                    seq_raw = progress_payload.get("seq", -1)
                    try:
                        seq = int(seq_raw)
                    except Exception:
                        seq = -1
                    stage = str(progress_payload.get("stage", "") or "").strip()
                    attempt = progress_payload.get("attempt")
                    detail = _clip_text(progress_payload.get("detail", ""), 280)
                    sig = f"{seq}|{stage}|{attempt}|{detail}"
                    if seq > last_progress_seq or sig != last_progress_sig:
                        last_progress_seq = max(last_progress_seq, seq)
                        last_progress_sig = sig
                        evt: Dict[str, object] = {
                            "seq": int(seq),
                            "stage": stage,
                            "detail": detail,
                        }
                        if attempt is not None:
                            try:
                                evt["attempt"] = int(attempt)
                            except Exception:
                                evt["attempt"] = str(attempt)
                        if isinstance(progress_payload.get("elapsed_s"), (int, float)):
                            evt["backend_elapsed_s"] = float(progress_payload.get("elapsed_s"))
                        if isinstance(progress_payload.get("timestamp"), (int, float)):
                            evt["backend_epoch_s"] = float(progress_payload.get("timestamp"))
                        _timeline("backend_progress", **evt)
                        async_wait_stats["progress_events_observed"] = int(async_wait_stats["progress_events_observed"]) + 1

            if (now - last_log_t) >= 5.0:
                progress = getattr(dlg, "_topology_mesh_progress", None)
                print(
                    f"[headless-topo] elapsed={now - wait_started_at:7.1f}s progress={progress} status={status_txt}",
                    flush=True,
                )
                last_log_t = now

            if (time.perf_counter() - wait_started_at) > timeout_s:
                try:
                    dlg._terminate_topology_mesh_run(reason="script-timeout")
                except Exception:
                    pass
                raise TimeoutError(f"Topology meshing timed out after {timeout_s:.0f}s")

            t_sleep = time.perf_counter()
            time.sleep(0.05)
            async_wait_stats["sleep_s"] = float(async_wait_stats["sleep_s"]) + float(max(0.0, time.perf_counter() - t_sleep))
            async_wait_stats["iterations"] = int(async_wait_stats["iterations"]) + 1

        meshing_elapsed_s = float(max(0.0, time.perf_counter() - mesh_run_started_at))
        async_wait_stats["wait_elapsed_s"] = float(max(0.0, time.perf_counter() - wait_started_at))
        _phase_add("async_wait_for_meshing", wait_started_at)

        t_phase = _phase_start()
        mesh_data = getattr(dlg, "_mesh_data", None)
        if mesh_data is None:
            status_txt = ""
            try:
                status_txt = str(dlg.topo_status_lbl.text())
            except Exception:
                pass
            raise RuntimeError(f"Topology meshing finished without mesh data. status={status_txt}")

        node_x = np.asarray(mesh_data["node_x"], dtype=np.float64)
        node_y = np.asarray(mesh_data["node_y"], dtype=np.float64)
        node_z = np.asarray(mesh_data["node_z"], dtype=np.float64)
        cell_nodes = np.asarray(mesh_data["cell_nodes"], dtype=np.int32)
        cell_face_offsets = np.asarray(mesh_data["cell_face_offsets"], dtype=np.int32)
        cell_face_nodes = np.asarray(mesh_data["cell_face_nodes"], dtype=np.int32)
        cell_type = np.asarray(mesh_data["cell_type"]).astype(np.str_)
        region_id = np.asarray(mesh_data["region_id"], dtype=np.int32)
        target_size = np.asarray(mesh_data["target_size"], dtype=np.float64)
        quality_summary = mesh_data.get("quality_summary") if isinstance(mesh_data, dict) else None
        _phase_add("extract_mesh_arrays", t_phase)

        gmsh_phase_timings_s: Dict[str, float] = {}
        if isinstance(quality_summary, dict):
            candidate_timings = quality_summary.get("gmsh_phase_timings_s")
            if isinstance(candidate_timings, dict):
                gmsh_phase_timings_s = {
                    str(k): float(v)
                    for k, v in candidate_timings.items()
                    if isinstance(v, (int, float))
                }

        t_phase = _phase_start()
        np.savez_compressed(
            out_npz,
            node_x=node_x,
            node_y=node_y,
            node_z=node_z,
            cell_nodes=cell_nodes,
            cell_face_offsets=cell_face_offsets,
            cell_face_nodes=cell_face_nodes,
            cell_type=cell_type,
            region_id=region_id,
            target_size=target_size,
        )
        _phase_add("write_npz", t_phase)

        t_phase = _phase_start()
        _save_mesh_png(node_x=node_x, node_y=node_y, cell_nodes=cell_nodes, out_png=out_png)
        _phase_add("write_png", t_phase)

        n_faces = int(max(0, cell_face_offsets.size - 1))
        n_quads = int(np.sum(cell_type == "quadrilateral"))
        n_tris = int(np.sum(cell_type == "triangular"))

        _phase_mark("payload_build_start")
        payload = {
            "ok": True,
            "gpkg": str(gpkg_path),
            "region_ids": [int(v) for v in region_ids],
            "backend": "gmsh",
            "gmsh_quad_full_region_flow_align": True,
            "artifacts": {
                "npz": str(out_npz),
                "json": str(out_json),
                "png": str(out_png),
                "profile_json": str(out_profile_json),
            },
            "mesh": {
                "n_nodes": int(node_x.size),
                "n_faces": n_faces,
                "n_tris": n_tris,
                "n_quads": n_quads,
            },
            "timings_s": {
                "headless_meshing_elapsed": float(meshing_elapsed_s),
                "gmsh_phase": gmsh_phase_timings_s,
            },
            "layer_snapshot": dict(layer_snapshot),
            "options_snapshot": dict(options_snapshot),
            "conceptual_digest": dict(conceptual_digest),
            "quality_summary": quality_summary if isinstance(quality_summary, dict) else None,
            "topology_status": str(getattr(getattr(dlg, "topo_status_lbl", None), "text", lambda: "")()),
            "timestamp": stamp,
        }

        phase_times_s["total_script"] = float(max(0.0, time.perf_counter() - script_started_at))
        payload["profiling"] = {
            "script_phase_s": dict(phase_times_s),
            "script_marks_s": dict(phase_marks_s),
            "async_wait": dict(async_wait_stats),
            "timeline": list(timeline),
        }

        t_phase = _phase_start()
        out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        out_profile_json.write_text(
            json.dumps(
                {
                    "timestamp": stamp,
                    "region_ids": [int(v) for v in region_ids],
                    "profiling": payload["profiling"],
                    "timings_s": payload["timings_s"],
                    "layer_snapshot": payload["layer_snapshot"],
                    "options_snapshot": payload["options_snapshot"],
                    "conceptual_digest": payload["conceptual_digest"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        _phase_add("write_json", t_phase)

        print(json.dumps(payload, indent=2))
        return 0
    finally:
        qgs.exitQgis()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
