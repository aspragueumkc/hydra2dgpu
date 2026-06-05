#!/usr/bin/env python3
"""QGIS live polling bridge (run this inside QGIS Python Console).

This script starts a localhost-style command bridge using files in a runtime
folder. A terminal-side script can write commands; this bridge executes a
small whitelist of actions in the *current* QGIS session.

How to start from QGIS Python Console:

    exec(open('/absolute/path/to/tools/qgis_live_bridge_console.py').read(), globals())

How to stop:

    QGIS_LIVE_BRIDGE.stop()

Runtime protocol:
- command file:  <runtime_dir>/command.json
- response file: <runtime_dir>/response.json

Command JSON shape:
{
  "request_id": "uuid-string",
  "token": "shared-secret",
  "action": "ping|list_layers|get_project_info|select_layer|zoom_to_layer|trigger_action",
  "params": {...}
}
"""

import json
import os
import time
import traceback
import uuid
import hashlib
from datetime import datetime, timezone

import numpy as np

from qgis.PyQt.QtCore import QTimer
from qgis.PyQt.QtWidgets import QAction
from qgis.core import QgsProject
from qgis.utils import iface


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class QgisLiveBridge:
    def __init__(self, runtime_dir: str, token: str, poll_ms: int = 350):
        self.runtime_dir = runtime_dir
        self.token = token
        self.poll_ms = int(max(100, poll_ms))
        self.command_path = os.path.join(self.runtime_dir, "command.json")
        self.response_path = os.path.join(self.runtime_dir, "response.json")
        self.topology_mesh_gui_result_path = os.path.join(self.runtime_dir, "topology_mesh_gui_result.json")
        self.last_request_id = None
        self.timer = None
        self._active_topology_mesh_gui_run = None

    def start(self):
        os.makedirs(self.runtime_dir, exist_ok=True)
        self.timer = QTimer()
        self.timer.setInterval(self.poll_ms)
        self.timer.timeout.connect(self._poll_once)
        self.timer.start()
        print(
            "[QGIS-LIVE-BRIDGE] started",
            json.dumps(
                {
                    "runtime_dir": self.runtime_dir,
                    "command_path": self.command_path,
                    "response_path": self.response_path,
                    "poll_ms": self.poll_ms,
                },
            ),
        )

    def stop(self):
        if self.timer is not None:
            self.timer.stop()
            self.timer = None
        print("[QGIS-LIVE-BRIDGE] stopped")

    def _write_response(self, payload: dict):
        payload.setdefault("timestamp_utc", _utc_now())
        tmp_path = self.response_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, self.response_path)

    def _resolve_plugin_name(self, requested_name: str) -> str:
        import qgis.utils as qgis_utils

        plugins = getattr(qgis_utils, "plugins", {}) or {}
        known_candidates = [
            "qgis-backwater-plugin",
            "hydra",
        ]

        requested = str(requested_name or "").strip()
        if requested:
            if requested in plugins:
                return requested
            # Accept explicit names even if plugin is currently unloaded.
            return requested

        if not plugins:
            # Fall back to known local plugin IDs for this workspace.
            return known_candidates[0]
            requested_lower = requested.lower()
            for name in plugins.keys():
                if str(name).lower() == requested_lower or requested_lower in str(name).lower():
                    return str(name)
            raise ValueError(f"Plugin not found: {requested}")

        # Auto-detect HYDRA plugin instance in this workspace.
        for name, plugin in plugins.items():
            cls_name = str(getattr(plugin, "__class__", type(plugin)).__name__).lower()
            if "hydra" in cls_name:
                return str(name)
            try:
                mod_name = str(plugin.__class__.__module__).lower()
            except Exception:
                mod_name = ""
            if "hydra_plugin" in mod_name or "qgis-backwater-plugin" in mod_name:
                return str(name)

        # Fall back to known local plugin IDs for this workspace.
        return known_candidates[0]

    def _get_plugin_instance(self, requested_name: str):
        import qgis.utils as qgis_utils

        plugin_name = self._resolve_plugin_name(requested_name)
        plugins = getattr(qgis_utils, "plugins", {}) or {}
        plugin = plugins.get(plugin_name)
        if plugin is None:
            try:
                if hasattr(qgis_utils, "loadPlugin"):
                    qgis_utils.loadPlugin(plugin_name)
                if hasattr(qgis_utils, "startPlugin"):
                    qgis_utils.startPlugin(plugin_name)
            except Exception:
                pass
            plugins = getattr(qgis_utils, "plugins", {}) or {}
            plugin = plugins.get(plugin_name)
        if plugin is None:
            raise RuntimeError(f"Plugin instance not found: {plugin_name}")
        return plugin_name, plugin

    def _poll_once(self):
        self._poll_topology_mesh_gui_run()
        if not os.path.exists(self.command_path):
            return
        try:
            with open(self.command_path, "r", encoding="utf-8") as f:
                cmd = json.load(f)
        except Exception as exc:
            self._write_response(
                {
                    "ok": False,
                    "error": f"Invalid command.json: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
            return

        request_id = str(cmd.get("request_id", ""))
        if not request_id:
            self._write_response(
                {
                    "ok": False,
                    "error": "Missing request_id",
                }
            )
            return

        if request_id == self.last_request_id:
            return
        self.last_request_id = request_id

        token = str(cmd.get("token", ""))
        if token != self.token:
            self._write_response(
                {
                    "request_id": request_id,
                    "ok": False,
                    "error": "Unauthorized token",
                }
            )
            return

        action = str(cmd.get("action", "")).strip()
        params = cmd.get("params") or {}
        if not isinstance(params, dict):
            params = {}

        try:
            result = self._execute(action, params)
            self._write_response(
                {
                    "request_id": request_id,
                    "ok": True,
                    "action": action,
                    "result": result,
                }
            )
        except Exception as exc:
            self._write_response(
                {
                    "request_id": request_id,
                    "ok": False,
                    "action": action,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )

    def _json_safe(self, value):
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, dict):
            return {str(k): self._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._json_safe(v) for v in value]
        try:
            if isinstance(value, np.generic):
                return value.item()
        except Exception:
            pass
        return str(value)

    def _write_json_file(self, path: str, payload: dict):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, path)

    def _read_json_file(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _find_workbench_dialog(self, target: str = "auto"):
        import sys

        mod = sys.modules.get("swe2d_workbench_qt")
        if mod is None:
            raise RuntimeError("swe2d_workbench_qt module is not loaded in this QGIS session")

        target_l = str(target or "auto").strip().lower()
        studio_dlg = getattr(mod, "_SWE2D_STUDIO_HOST_DIALOG", None)
        demo_windows = list(getattr(mod, "_SWE2D_WORKBENCH_WINDOWS", []) or [])
        demo_dlg = demo_windows[-1] if demo_windows else None

        if target_l == "studio":
            if studio_dlg is None:
                raise RuntimeError("No active SWE2D studio dialog found")
            return studio_dlg, "studio"
        if target_l == "demo":
            if demo_dlg is None:
                raise RuntimeError("No active SWE2D demo workbench window found")
            return demo_dlg, "demo"

        if studio_dlg is not None:
            return studio_dlg, "studio"
        if demo_dlg is not None:
            return demo_dlg, "demo"
        raise RuntimeError("No active SWE2D workbench dialog found")

    def _set_combo_layer_by_name_or_none(self, dlg, combo_attr: str, value):
        combo = getattr(dlg, combo_attr, None)
        if combo is None:
            raise RuntimeError(f"Dialog has no combo: {combo_attr}")

        txt = "" if value is None else str(value).strip()
        if txt == "" or txt.lower() in {"none", "(none)", "null"}:
            idx = combo.findData(None)
            if idx < 0:
                idx = 0 if combo.count() > 0 else -1
            if idx >= 0:
                combo.setCurrentIndex(idx)
                return
            raise RuntimeError(f"Could not set {combo_attr} to None")

        set_by_name = getattr(dlg, "_set_combo_by_layer_name", None)
        if callable(set_by_name):
            ok = bool(set_by_name(combo, txt))
            if ok:
                return

        for i in range(combo.count()):
            if str(combo.itemText(i) or "").strip() == txt:
                combo.setCurrentIndex(i)
                return

        raise RuntimeError(f"Could not resolve layer '{txt}' for {combo_attr}")

    def _set_combo_data(self, combo, data_value) -> bool:
        if combo is None:
            return False
        for i in range(combo.count()):
            if combo.itemData(i) == data_value:
                combo.setCurrentIndex(i)
                return True
        return False

    def _combo_layer_snapshot(self, dlg, combo_attr: str) -> dict:
        combo = getattr(dlg, combo_attr, None)
        if combo is None:
            return {"present": False}

        snap = {
            "present": True,
            "current_index": int(combo.currentIndex()),
            "current_text": str(combo.currentText() or ""),
            "current_data": combo.currentData(),
        }

        layer = None
        combo_layer = getattr(dlg, "_combo_layer", None)
        if callable(combo_layer):
            try:
                layer = combo_layer(combo, "vector")
            except Exception:
                layer = None
        if layer is not None:
            try:
                snap.update(
                    {
                        "layer_id": str(layer.id()),
                        "layer_name": str(layer.name()),
                        "layer_source": str(layer.source()) if hasattr(layer, "source") else "",
                        "subset_sql": str(layer.subsetString() or "") if hasattr(layer, "subsetString") else "",
                        "feature_count": int(layer.featureCount()) if hasattr(layer, "featureCount") else None,
                    }
                )
            except Exception:
                pass
        return self._json_safe(snap)

    def _polyline_metrics(self, points) -> dict:
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

    def _ring_metrics(self, ring) -> dict:
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
        poly = self._polyline_metrics(pts)
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

    def _build_conceptual_from_dialog(self, dlg):
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

        import importlib
        import sys

        builder = None
        wbqt_mod = sys.modules.get("swe2d_workbench_qt")
        if wbqt_mod is not None:
            cand = getattr(wbqt_mod, "conceptual_from_qgis_layers", None)
            if callable(cand):
                builder = cand
        if builder is None:
            for mod_name in ("swe2d.mesh.meshing", "swe2d_meshing"):
                try:
                    mod = importlib.import_module(mod_name)
                except Exception:
                    continue
                cand = getattr(mod, "conceptual_from_qgis_layers", None)
                if callable(cand):
                    builder = cand
                    break
        if builder is None:
            raise RuntimeError("conceptual_from_qgis_layers callable not available")

        default_size = float(getattr(getattr(dlg, "topo_default_size_spin", None), "value", lambda: 20.0)())
        default_cell_type = str(getattr(getattr(dlg, "topo_default_cell_type_combo", None), "currentText", lambda: "triangular")())

        return builder(
            nodes_layer=nodes_layer,
            arcs_layer=arcs_layer,
            regions_layer=regions_layer,
            constraints_layer=constraints_layer,
            quad_edges_layer=quad_edges_layer,
            default_size=default_size,
            default_cell_type=default_cell_type,
        )

    def _conceptual_digest(self, conceptual) -> dict:
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
            pm = self._polyline_metrics(getattr(a, "points_xy", None))
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
            rm = self._ring_metrics(getattr(r, "ring_xy", None))
            hole_rings = list(getattr(r, "hole_rings", []) or [])
            hole_hashes = [self._ring_metrics(h).get("points_sha256", "") for h in hole_rings]
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
            cm = self._ring_metrics(getattr(c, "ring_xy", None))
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
            qm = self._polyline_metrics(getattr(q, "points_xy", None))
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

    def _topology_gui_snapshot(self, dlg) -> dict:
        snapshot = {
            "backend": str(getattr(dlg, "_topology_mesh_backend", "") or ""),
            "run_mode": str(getattr(dlg, "_topology_mesh_run_mode", "") or ""),
            "status_label": str(getattr(getattr(dlg, "topo_status_lbl", None), "text", lambda: "")() or ""),
            "active_timeout_sec": float(getattr(dlg, "_topology_mesh_active_timeout_sec", 0.0) or 0.0),
            "progress_path": str(getattr(dlg, "_topology_mesh_progress_path", "") or ""),
            "combos": {
                "topo_nodes_combo": self._combo_layer_snapshot(dlg, "topo_nodes_combo"),
                "topo_arcs_combo": self._combo_layer_snapshot(dlg, "topo_arcs_combo"),
                "topo_regions_combo": self._combo_layer_snapshot(dlg, "topo_regions_combo"),
                "topo_constraints_combo": self._combo_layer_snapshot(dlg, "topo_constraints_combo"),
                "topo_quad_edges_combo": self._combo_layer_snapshot(dlg, "topo_quad_edges_combo"),
            },
        }

        build_opts = getattr(dlg, "_build_topology_meshing_options", None)
        if callable(build_opts):
            try:
                snapshot["options"] = self._json_safe(dict(build_opts() or {}))
            except Exception as exc:
                snapshot["options_error"] = str(exc)

        try:
            conceptual = self._build_conceptual_from_dialog(dlg)
            conceptual_digest = self._conceptual_digest(conceptual)
            snapshot["conceptual_counts"] = dict(conceptual_digest.get("counts", {}))
            snapshot["conceptual_digest"] = conceptual_digest
        except Exception as exc:
            snapshot["conceptual_counts_error"] = str(exc)

        return self._json_safe(snapshot)

    def _dialog_log_text(self, dlg) -> str:
        log_view = getattr(dlg, "log_view", None)
        if log_view is not None and hasattr(log_view, "toPlainText"):
            try:
                return str(log_view.toPlainText() or "")
            except Exception:
                return ""
        return ""

    def _mesh_counts_snapshot(self, dlg) -> dict:
        mesh_data = getattr(dlg, "_mesh_data", None)
        if not isinstance(mesh_data, dict):
            return {}
        try:
            node_x = np.asarray(mesh_data.get("node_x", np.empty(0, dtype=np.float64)))
            offsets = np.asarray(mesh_data.get("cell_face_offsets", np.empty(0, dtype=np.int64)))
            cell_type = np.asarray(mesh_data.get("cell_type", np.empty(0, dtype=object)))
            out = {
                "n_nodes": int(node_x.size),
                "n_faces": int(max(0, offsets.size - 1)),
                "n_tris": int(np.sum(cell_type == "triangular")) if cell_type.size else 0,
                "n_quads": int(np.sum(cell_type == "quadrilateral")) if cell_type.size else 0,
            }
            quality_summary = mesh_data.get("quality_summary")
            if isinstance(quality_summary, dict):
                out["quality_summary"] = self._json_safe(quality_summary)
            return out
        except Exception:
            return {}

    def _build_topology_gui_run_result(self, state: dict, status: str, error: str = "") -> dict:
        dlg = state.get("dialog")
        now_log = self._dialog_log_text(dlg)
        pre_log = str(state.get("pre_log", "") or "")
        if now_log.startswith(pre_log):
            delta_log = now_log[len(pre_log):]
        else:
            delta_log = now_log[-12000:]
        delta_lines = [ln for ln in str(delta_log or "").splitlines() if ln.strip()]
        fail_lines = [ln for ln in delta_lines if "mesh> fail" in ln.lower()]
        warn_lines = [ln for ln in delta_lines if "warning" in ln.lower()]

        result = {
            "status": str(status),
            "request_id": str(state.get("request_id", "")),
            "target": str(state.get("target", "")),
            "started_utc": str(state.get("started_utc", "")),
            "finished_utc": _utc_now(),
            "elapsed_s": float(max(0.0, time.time() - float(state.get("started_epoch", time.time())))),
            "topology_status_label": str(getattr(getattr(dlg, "topo_status_lbl", None), "text", lambda: "")() or ""),
            "mesh_counts": self._mesh_counts_snapshot(dlg),
            "snapshot_before": self._json_safe(state.get("snapshot_before", {})),
            "snapshot_after": self._topology_gui_snapshot(dlg),
            "log_tail": delta_lines[-40:],
            "log_fail_lines": fail_lines[-20:],
            "log_warning_lines": warn_lines[-20:],
        }
        if error:
            result["error"] = str(error)

        mesh_counts = result.get("mesh_counts") or {}
        n_faces = int(mesh_counts.get("n_faces", 0) or 0)
        status_txt = str(result.get("topology_status_label", "") or "").lower()
        result["ok"] = bool(status == "complete" and n_faces > 0 and "failed" not in status_txt)
        return self._json_safe(result)

    def _poll_topology_mesh_gui_run(self):
        state = self._active_topology_mesh_gui_run
        if not isinstance(state, dict):
            return
        dlg = state.get("dialog")
        fut = state.get("future")
        timeout_s = float(state.get("wait_timeout_sec", 1800.0))
        elapsed_s = max(0.0, time.time() - float(state.get("started_epoch", time.time())))

        if elapsed_s > timeout_s:
            try:
                terminate = getattr(dlg, "_terminate_topology_mesh_run", None)
                if callable(terminate):
                    terminate(reason="bridge-timeout")
            except Exception:
                pass
            result = self._build_topology_gui_run_result(state, status="timeout", error=f"GUI topology run timed out after {timeout_s:.1f}s")
            self._write_json_file(self.topology_mesh_gui_result_path, result)
            self._active_topology_mesh_gui_run = None
            return

        if fut is None:
            result = self._build_topology_gui_run_result(state, status="error", error="Topology run did not start")
            self._write_json_file(self.topology_mesh_gui_result_path, result)
            self._active_topology_mesh_gui_run = None
            return

        if not fut.done():
            return

        try:
            poll_fn = getattr(dlg, "_poll_topology_mesh_future", None)
            if callable(poll_fn):
                poll_fn()
        except Exception:
            pass

        result = self._build_topology_gui_run_result(state, status="complete")
        self._write_json_file(self.topology_mesh_gui_result_path, result)
        self._active_topology_mesh_gui_run = None

    def _execute(self, action: str, params: dict):
        if action == "ping":
            return {
                "message": "pong",
                "bridge": "qgis-live-polling",
                "version": 1,
            }

        if action == "get_project_info":
            project = QgsProject.instance()
            map_layers = project.mapLayers()
            return {
                "project_file": project.fileName(),
                "layer_count": len(map_layers),
                "layer_names": [lyr.name() for lyr in map_layers.values()],
            }

        if action == "list_layers":
            project = QgsProject.instance()
            out = []
            for lyr in project.mapLayers().values():
                out.append(
                    {
                        "id": lyr.id(),
                        "name": lyr.name(),
                        "provider": lyr.providerType() if hasattr(lyr, "providerType") else "",
                        "geometry_type": int(lyr.geometryType()) if hasattr(lyr, "geometryType") else None,
                    }
                )
            return out

        if action == "describe_layer":
            name = str(params.get("name", "")).strip()
            if not name:
                raise ValueError("params.name is required")
            layers = QgsProject.instance().mapLayersByName(name)
            if not layers:
                raise ValueError(f"Layer not found: {name}")
            layer = layers[0]
            return {
                "name": layer.name(),
                "id": layer.id(),
                "feature_count": int(layer.featureCount()),
                "source": layer.source() if hasattr(layer, "source") else "",
                "fields": [f.name() for f in layer.fields()],
            }

        if action == "list_features":
            name = str(params.get("name", "")).strip()
            if not name:
                raise ValueError("params.name is required")
            limit = int(params.get("limit", 200))
            limit = max(1, min(limit, 2000))
            fields_filter = params.get("fields")
            if fields_filter is not None and not isinstance(fields_filter, list):
                raise ValueError("params.fields must be an array of field names")

            layers = QgsProject.instance().mapLayersByName(name)
            if not layers:
                raise ValueError(f"Layer not found: {name}")
            layer = layers[0]

            out = []
            for i, feat in enumerate(layer.getFeatures()):
                if i >= limit:
                    break
                attrs = {}
                if fields_filter is None:
                    field_names = [f.name() for f in layer.fields()]
                else:
                    field_names = [str(f) for f in fields_filter]
                for fname in field_names:
                    if fname in feat.fields().names():
                        val = feat[fname]
                        # Convert Qt/QGIS wrapper variants to JSON-safe Python scalars.
                        if hasattr(val, "isNull") and callable(getattr(val, "isNull")) and val.isNull():
                            attrs[fname] = None
                        elif isinstance(val, (int, float, str, bool)) or val is None:
                            attrs[fname] = val
                        else:
                            attrs[fname] = str(val)
                    else:
                        attrs[fname] = None
                out.append({"fid": int(feat.id()), "attrs": attrs})
            return {
                "name": layer.name(),
                "feature_count": int(layer.featureCount()),
                "returned": len(out),
                "features": out,
            }

        if action == "update_features":
            name = str(params.get("name", "")).strip()
            if not name:
                raise ValueError("params.name is required")
            updates = params.get("updates")
            if not isinstance(updates, list) or not updates:
                raise ValueError("params.updates must be a non-empty array")

            layers = QgsProject.instance().mapLayersByName(name)
            if not layers:
                raise ValueError(f"Layer not found: {name}")
            layer = layers[0]

            if not layer.isEditable():
                if not layer.startEditing():
                    raise RuntimeError(f"Failed to start editing: {name}")

            changed = 0
            field_index = {f.name(): i for i, f in enumerate(layer.fields())}
            for item in updates:
                if not isinstance(item, dict):
                    continue
                fid = item.get("fid")
                attrs = item.get("attrs")
                if fid is None or not isinstance(attrs, dict):
                    continue
                for fname, val in attrs.items():
                    idx = field_index.get(str(fname))
                    if idx is None:
                        continue
                    ok = layer.changeAttributeValue(int(fid), idx, val)
                    if ok:
                        changed += 1

            save_edits = bool(params.get("save", True))
            if save_edits:
                if not layer.commitChanges():
                    errs = layer.commitErrors() if hasattr(layer, "commitErrors") else []
                    layer.rollBack()
                    raise RuntimeError(f"Commit failed for {name}: {errs}")
            return {
                "name": layer.name(),
                "changed_values": changed,
                "saved": save_edits,
            }

        if action == "normalize_tqmesh_multiregion":
            regions_name = str(params.get("regions_layer", "swe2d_topo_regions")).strip()
            quad_edges_name = str(params.get("quad_edges_layer", "swe2d_topo_quad_edges")).strip()

            regions_layers = QgsProject.instance().mapLayersByName(regions_name)
            if not regions_layers:
                raise ValueError(f"Layer not found: {regions_name}")
            regions = regions_layers[0]

            quad_layers = QgsProject.instance().mapLayersByName(quad_edges_name)
            if not quad_layers:
                raise ValueError(f"Layer not found: {quad_edges_name}")
            quad_edges = quad_layers[0]

            if not regions.isEditable() and not regions.startEditing():
                raise RuntimeError(f"Failed to start editing: {regions_name}")
            if not quad_edges.isEditable() and not quad_edges.startEditing():
                raise RuntimeError(f"Failed to start editing: {quad_edges_name}")

            reg_idx = {f.name(): i for i, f in enumerate(regions.fields())}
            qe_idx = {f.name(): i for i, f in enumerate(quad_edges.fields())}
            for req in ("region_id", "cell_type"):
                if req not in reg_idx:
                    raise ValueError(f"Missing field on {regions_name}: {req}")
            for req in ("region_id", "edge_id"):
                if req not in qe_idx:
                    raise ValueError(f"Missing field on {quad_edges_name}: {req}")

            region_feats = list(regions.getFeatures())
            if not region_feats:
                raise ValueError("No region features found")

            # Preserve the currently designated quad region if exactly one exists.
            quad_tags = {"quadrilateral", "cartesian", "quad"}
            current_quad = []
            for f in region_feats:
                v = f["cell_type"]
                if str(v).strip().lower() in quad_tags:
                    current_quad.append(f)

            if len(current_quad) == 1:
                quad_fid = int(current_quad[0].id())
            elif len(current_quad) > 1:
                # Keep the largest tagged region as the single quad region.
                current_quad.sort(key=lambda ff: float(ff.geometry().area()) if ff.hasGeometry() else 0.0, reverse=True)
                quad_fid = int(current_quad[0].id())
            else:
                # If none tagged, pick largest polygon region.
                region_feats.sort(key=lambda ff: float(ff.geometry().area()) if ff.hasGeometry() else 0.0, reverse=True)
                quad_fid = int(region_feats[0].id())

            # Stable region_id assignment for deterministic multi-region behavior.
            region_feats_sorted = sorted(region_feats, key=lambda f: int(f.id()))
            fid_to_region_id = {}
            for i, f in enumerate(region_feats_sorted, start=1):
                rid = int(i)
                fid = int(f.id())
                fid_to_region_id[fid] = rid
                regions.changeAttributeValue(fid, reg_idx["region_id"], rid)
                ctype = "quadrilateral" if fid == quad_fid else "triangular"
                regions.changeAttributeValue(fid, reg_idx["cell_type"], ctype)

            quad_region_id = fid_to_region_id[quad_fid]

            # Bind all quad-edge controls to the designated quad region and force
            # edge_id coverage {1,2,3,4} using geometric ordering around centroid.
            q_feats = list(quad_edges.getFeatures())
            if len(q_feats) < 4:
                raise ValueError("Quad edges layer must contain at least 4 line features")

            # Use the 4 longest edges (best proxy when stale features exist).
            q_feats.sort(key=lambda ff: float(ff.geometry().length()) if ff.hasGeometry() else 0.0, reverse=True)
            q_pick = q_feats[:4]

            cx = 0.0
            cy = 0.0
            nmid = 0
            mids = []
            for f in q_pick:
                g = f.geometry()
                if g is None or g.isEmpty():
                    continue
                p = g.interpolate(g.length() * 0.5).asPoint()
                x = float(p.x())
                y = float(p.y())
                mids.append((f, x, y))
                cx += x
                cy += y
                nmid += 1
            if nmid != 4:
                raise ValueError("Could not compute 4 valid quad-edge midpoints")
            cx /= 4.0
            cy /= 4.0

            import math

            mids.sort(key=lambda t: math.atan2(t[2] - cy, t[1] - cx))
            for edge_id, (feat, _x, _y) in enumerate(mids, start=1):
                fid = int(feat.id())
                quad_edges.changeAttributeValue(fid, qe_idx["region_id"], int(quad_region_id))
                quad_edges.changeAttributeValue(fid, qe_idx["edge_id"], int(edge_id))

            if not regions.commitChanges():
                errs = regions.commitErrors() if hasattr(regions, "commitErrors") else []
                regions.rollBack()
                raise RuntimeError(f"Commit failed for {regions_name}: {errs}")
            if not quad_edges.commitChanges():
                errs = quad_edges.commitErrors() if hasattr(quad_edges, "commitErrors") else []
                quad_edges.rollBack()
                raise RuntimeError(f"Commit failed for {quad_edges_name}: {errs}")

            return {
                "regions_layer": regions_name,
                "quad_edges_layer": quad_edges_name,
                "region_count": len(region_feats_sorted),
                "quad_region_id": int(quad_region_id),
                "quad_region_fid": int(quad_fid),
                "quad_edge_fids": [int(f.id()) for f, _, _ in mids],
            }

        if action == "select_layer":
            name = str(params.get("name", "")).strip()
            if not name:
                raise ValueError("params.name is required")
            layers = QgsProject.instance().mapLayersByName(name)
            if not layers:
                raise ValueError(f"Layer not found: {name}")
            layer = layers[0]
            iface.setActiveLayer(layer)
            return {"selected": layer.name(), "id": layer.id()}

        if action == "zoom_to_layer":
            name = str(params.get("name", "")).strip()
            if not name:
                raise ValueError("params.name is required")
            layers = QgsProject.instance().mapLayersByName(name)
            if not layers:
                raise ValueError(f"Layer not found: {name}")
            layer = layers[0]
            iface.setActiveLayer(layer)
            iface.zoomToActiveLayer()
            return {"zoomed_to": layer.name(), "id": layer.id()}

        if action == "trigger_action":
            object_name = str(params.get("object_name", "")).strip()
            if not object_name:
                raise ValueError("params.object_name is required")
            actions = iface.mainWindow().findChildren(QAction, object_name)
            if not actions:
                raise ValueError(f"Action not found: {object_name}")
            # Trigger asynchronously so modal dialogs do not block bridge polling.
            QTimer.singleShot(0, actions[0].trigger)
            return {"triggered": object_name, "async": True}

        if action == "reload_plugin":
            import qgis.utils as qgis_utils
            import sys

            requested_name = str(params.get("plugin_name", "")).strip()
            plugin_name = self._resolve_plugin_name(requested_name)
            loaded_before = bool(qgis_utils.isPluginLoaded(plugin_name))

            # Best effort explicit unload before reload to force fresh imports.
            try:
                qgis_utils.unloadPlugin(plugin_name)
            except Exception:
                pass

            try:
                qgis_utils.reloadPlugin(plugin_name)
            except Exception:
                pass

            # Ensure plugin is loaded/started after reload cycle.
            if not qgis_utils.isPluginLoaded(plugin_name):
                try:
                    if hasattr(qgis_utils, "loadPlugin"):
                        qgis_utils.loadPlugin(plugin_name)
                    if hasattr(qgis_utils, "startPlugin"):
                        qgis_utils.startPlugin(plugin_name)
                except Exception:
                    pass

            # Clear cached SWE2D workbench modules so next open uses fresh code.
            for mod_name in list(sys.modules.keys()):
                if mod_name == "swe2d_workbench_qt" or mod_name.startswith("swe2d.workbench.extracted."):
                    try:
                        sys.modules.pop(mod_name, None)
                    except Exception:
                        pass
            loaded_after = bool(qgis_utils.isPluginLoaded(plugin_name))
            plugins = getattr(qgis_utils, "plugins", {}) or {}
            plugin = plugins.get(plugin_name)
            cls_name = str(getattr(plugin, "__class__", type(plugin)).__name__) if plugin is not None else None
            return {
                "plugin_name": plugin_name,
                "loaded_before": loaded_before,
                "loaded_after": loaded_after,
                "plugin_class": cls_name,
            }

        if action == "plugin_status":
            import qgis.utils as qgis_utils

            plugins = getattr(qgis_utils, "plugins", {}) or {}
            return {
                "loaded_plugins": sorted(str(k) for k in plugins.keys()),
            }

        if action == "open_swe2d_demo_dialog":
            import sys
            import importlib

            requested_name = str(params.get("plugin_name", "")).strip()
            plugin_name, plugin = self._get_plugin_instance(requested_name)

            run_fn = getattr(plugin, "run", None)
            if callable(run_fn):
                run_fn()

            mod = importlib.import_module("swe2d_workbench_qt")
            mod = importlib.reload(mod)

            for extracted_name in (
                "swe2d.workbench.extracted.model_and_run_methods",
                "swe2d.workbench.extracted.topology_and_io_methods",
                "swe2d.workbench.extracted.results_and_ui_methods",
                "swe2d.workbench.extracted.results_export_methods",
            ):
                extracted_mod = sys.modules.get(extracted_name)
                if extracted_mod is not None:
                    try:
                        importlib.reload(extracted_mod)
                    except Exception:
                        pass

            launch = getattr(mod, "launch_swe2d_workbench_studio", None)
            if not callable(launch):
                raise RuntimeError("launch_swe2d_workbench_studio is not available")
            launch(parent=getattr(plugin, "dock", None), iface=iface, host_mode="window")

            windows = []
            if mod is not None:
                windows = list(getattr(mod, "_SWE2D_WORKBENCH_WINDOWS", []) or [])
            return {
                "plugin_name": plugin_name,
                "demo_open_invoked": True,
                "workbench_window_count": int(len(windows)),
            }

        if action == "open_swe2d_studio_dialog":
            import importlib

            requested_name = str(params.get("plugin_name", "")).strip()
            plugin_name, plugin = self._get_plugin_instance(requested_name)

            run_fn = getattr(plugin, "run", None)
            if callable(run_fn):
                run_fn()

            mod = importlib.import_module("swe2d_workbench_qt")
            mod = importlib.reload(mod)

            launch = getattr(mod, "launch_swe2d_workbench_studio", None)
            if not callable(launch):
                raise RuntimeError("launch_swe2d_workbench_studio is not available")
            launch(parent=getattr(plugin, "dock", None), iface=iface, host_mode="dock")

            component_docks = dict(getattr(mod, "_SWE2D_STUDIO_COMPONENT_DOCKS", {}) or {})
            host_dialog = getattr(mod, "_SWE2D_STUDIO_HOST_DIALOG", None)
            return {
                "plugin_name": plugin_name,
                "studio_open_invoked": True,
                "studio_component_dock_keys": sorted(str(k) for k in component_docks.keys()),
                "studio_host_dialog": bool(host_dialog is not None),
            }

        if action == "invoke_workbench_method":
            import importlib

            method_name = str(params.get("method_name", "")).strip()
            target = str(params.get("target", "demo")).strip().lower()
            if not method_name:
                raise ValueError("params.method_name is required")

            mod = importlib.import_module("swe2d_workbench_qt")

            dlg = None
            if target == "studio":
                dlg = getattr(mod, "_SWE2D_STUDIO_HOST_DIALOG", None)
            if dlg is None:
                windows = list(getattr(mod, "_SWE2D_WORKBENCH_WINDOWS", []) or [])
                dlg = windows[-1] if windows else None
            if dlg is None:
                raise RuntimeError("No active workbench dialog found")

            fn = getattr(dlg, method_name, None)
            if not callable(fn):
                raise RuntimeError(f"Method not found on active dialog: {method_name}")

            fn()
            return {
                "target": target,
                "method": method_name,
                "invoked": True,
            }

        if action == "set_env":
            env = params.get("env")
            if not isinstance(env, dict):
                raise ValueError("params.env must be an object of key/value pairs")
            for k, v in env.items():
                os.environ[str(k)] = str(v)
            return {"set": list(env.keys())}

        if action == "run_topology_mesh_gui":
            target = str(params.get("target", "auto") or "auto")
            wait_for_completion = bool(params.get("wait_for_completion", False))
            wait_timeout_sec = float(max(30.0, float(params.get("wait_timeout_sec", 1800.0))))

            dlg, resolved_target = self._find_workbench_dialog(target)

            # Optional combo/layer overrides for deterministic GUI-run parity tests.
            layer_overrides = {
                "topo_nodes_combo": params.get("nodes_layer"),
                "topo_arcs_combo": params.get("arcs_layer"),
                "topo_regions_combo": params.get("regions_layer"),
                "topo_constraints_combo": params.get("constraints_layer"),
                "topo_quad_edges_combo": params.get("quad_edges_layer"),
            }
            for combo_attr, layer_name in layer_overrides.items():
                if layer_name is not None:
                    self._set_combo_layer_by_name_or_none(dlg, combo_attr, layer_name)

            backend_override = params.get("backend")
            if backend_override is not None:
                backend_combo = getattr(dlg, "topo_backend_combo", None)
                backend_txt = str(backend_override).strip().lower()
                if not self._set_combo_data(backend_combo, backend_txt):
                    found = False
                    if backend_combo is not None:
                        for i in range(backend_combo.count()):
                            if str(backend_combo.itemText(i) or "").strip().lower() == backend_txt:
                                backend_combo.setCurrentIndex(i)
                                found = True
                                break
                    if not found:
                        raise RuntimeError(f"Could not set backend to '{backend_override}'")

            region_subset_sql = params.get("regions_subset_sql")
            if region_subset_sql is not None:
                reg_layer = getattr(dlg, "_combo_layer", lambda *_: None)(getattr(dlg, "topo_regions_combo"), "vector")
                if reg_layer is not None and hasattr(reg_layer, "setSubsetString"):
                    reg_layer.setSubsetString(str(region_subset_sql))

            quad_subset_sql = params.get("quad_edges_subset_sql")
            if quad_subset_sql is not None:
                qe_layer = getattr(dlg, "_combo_layer", lambda *_: None)(getattr(dlg, "topo_quad_edges_combo"), "vector")
                if qe_layer is not None and hasattr(qe_layer, "setSubsetString"):
                    qe_layer.setSubsetString(str(quad_subset_sql))

            try:
                os.remove(self.topology_mesh_gui_result_path)
            except FileNotFoundError:
                pass
            except Exception:
                pass

            pre_log = self._dialog_log_text(dlg)
            snapshot_before = self._topology_gui_snapshot(dlg)

            run_fn = getattr(dlg, "_generate_mesh_from_topology_layers", None)
            if not callable(run_fn):
                raise RuntimeError("Active dialog does not expose _generate_mesh_from_topology_layers")

            run_fn()
            fut = getattr(dlg, "_topology_mesh_future", None)
            request_id = str(uuid.uuid4())

            state = {
                "request_id": request_id,
                "target": resolved_target,
                "dialog": dlg,
                "future": fut,
                "started_epoch": float(time.time()),
                "started_utc": _utc_now(),
                "wait_timeout_sec": float(wait_timeout_sec),
                "pre_log": pre_log,
                "snapshot_before": snapshot_before,
            }

            self._active_topology_mesh_gui_run = state

            if wait_for_completion:
                from qgis.PyQt.QtWidgets import QApplication

                t0 = time.time()
                while self._active_topology_mesh_gui_run is not None and (time.time() - t0) < wait_timeout_sec:
                    QApplication.processEvents()
                    self._poll_topology_mesh_gui_run()
                    time.sleep(0.05)

                if os.path.exists(self.topology_mesh_gui_result_path):
                    out = self._read_json_file(self.topology_mesh_gui_result_path)
                    out["result_path"] = self.topology_mesh_gui_result_path
                    return out

            return {
                "status": "running",
                "request_id": request_id,
                "target": resolved_target,
                "result_path": self.topology_mesh_gui_result_path,
            }

        if action == "get_topology_mesh_gui_result":
            if os.path.exists(self.topology_mesh_gui_result_path):
                data = self._read_json_file(self.topology_mesh_gui_result_path)
                data["status"] = data.get("status", "complete")
                data["result_path"] = self.topology_mesh_gui_result_path
                return self._json_safe(data)

            state = self._active_topology_mesh_gui_run
            if isinstance(state, dict):
                dlg = state.get("dialog")
                return {
                    "status": "pending",
                    "request_id": str(state.get("request_id", "")),
                    "elapsed_s": float(max(0.0, time.time() - float(state.get("started_epoch", time.time())))),
                    "topology_status_label": str(getattr(getattr(dlg, "topo_status_lbl", None), "text", lambda: "")() or ""),
                    "result_path": self.topology_mesh_gui_result_path,
                }

            return {
                "status": "idle",
                "result_path": self.topology_mesh_gui_result_path,
            }

        if action == "run_topology_mesh":
            import sys
            import threading
            import multiprocessing as _mp
            import json as _json
            import importlib
            import warnings
            import traceback as _tb

            plugin_dir = "/home/aaron/.local/share/QGIS/QGIS3/profiles/default/python/plugins/qgis-backwater-plugin"
            build_dir = os.path.join(plugin_dir, "build")
            if plugin_dir not in sys.path:
                sys.path.insert(0, plugin_dir)
            if build_dir not in sys.path:
                sys.path.insert(0, build_dir)

            regions_name = str(params.get("regions_layer", "swe2d_topo_regions"))
            constraints_name = str(params.get("constraints_layer", "swe2d_topo_constraints"))
            quad_edges_name = str(params.get("quad_edges_layer", "swe2d_topo_quad_edges"))
            backend_name = str(params.get("backend", "tqmesh"))

            def _as_bool(v, default=False):
                if v is None:
                    return bool(default)
                if isinstance(v, bool):
                    return v
                s = str(v).strip().lower()
                if s in {"1", "true", "yes", "on"}:
                    return True
                if s in {"0", "false", "no", "off"}:
                    return False
                return bool(default)

            def _as_float(v, default):
                try:
                    return float(v)
                except Exception:
                    return float(default)

            timeout_default = _as_float(
                params.get("timeout_sec", os.environ.get("BACKWATER_TOPOLOGY_MESH_TIMEOUT_SEC", "300")),
                3000.0,
            )
            timeout_sec = max(30.0, timeout_default)

            if backend_name.strip().lower() == "gmsh":
                gmsh_loop_enabled = _as_bool(
                    params.get("gmsh_quality_enable", os.environ.get("BACKWATER_GMSH_QUALITY_ENABLE", "0")),
                    False,
                )
                if gmsh_loop_enabled:
                    budget_s = max(
                        1.0,
                        _as_float(
                            params.get("gmsh_quality_time_limit_s", os.environ.get("BACKWATER_GMSH_QUALITY_TIME_LIMIT_S", "60")),
                            60.0,
                        ),
                    )
                    grace_s = max(
                        0.0,
                        _as_float(
                            params.get("gmsh_quality_timeout_grace_s", os.environ.get("BACKWATER_GMSH_QUALITY_TIMEOUT_GRACE_S", "10")),
                            10.0,
                        ),
                    )
                    timeout_sec = max(30.0, budget_s + grace_s)

            result_path = os.path.join("/tmp", "qgis-live-bridge", "topology_mesh_result.json")

            # ---------------------------------------------------------------
            # Build the ConceptualModel on the MAIN thread — QGIS layer
            # feature iteration is NOT thread-safe.
            # ---------------------------------------------------------------
            def _get_layer(name):
                ls = QgsProject.instance().mapLayersByName(name)
                return ls[0] if ls else None

            swe2d_meshing = sys.modules.get("swe2d_meshing")
            if swe2d_meshing is None:
                swe2d_meshing = importlib.import_module("swe2d_meshing")
            else:
                swe2d_meshing = importlib.reload(swe2d_meshing)

            try:
                model = swe2d_meshing.conceptual_from_qgis_layers(
                    regions_layer=_get_layer(regions_name),
                    arcs_layer=None,
                    nodes_layer=None,
                    constraints_layer=_get_layer(constraints_name),
                    quad_edges_layer=_get_layer(quad_edges_name),
                )
            except Exception as _exc:
                return {"status": "error", "error": str(_exc), "traceback": _tb.format_exc()}

            def _run_mesh_once(model=model, swe2d_meshing=swe2d_meshing, backend_name=backend_name):
                warn_list = []
                t0 = time.perf_counter()
                with warnings.catch_warnings(record=True) as w_list:
                    warnings.simplefilter("always")
                    result = swe2d_meshing.generate_face_centric_mesh(model, backend=backend_name)
                    for wi in w_list:
                        warn_list.append(str(wi.message))

                tris = []
                quads = []
                offsets = result.cell_face_offsets
                nodes = result.cell_face_nodes
                for i in range(int(offsets.size) - 1):
                    a = int(offsets[i])
                    b = int(offsets[i + 1])
                    conn = [int(v) for v in nodes[a:b]]
                    if len(conn) == 3:
                        tris.append(conn)
                    elif len(conn) == 4:
                        quads.append(conn)

                stats = swe2d_meshing._mesh_quality_stats(
                    result.node_x,
                    result.node_y,
                    np.asarray(tris, dtype=np.int32) if tris else np.empty((0, 3), dtype=np.int32),
                    np.asarray(quads, dtype=np.int32) if quads else np.empty((0, 4), dtype=np.int32),
                )
                return {
                    "ok": True,
                    "elapsed_sec": round(time.perf_counter() - t0, 3),
                    "n_nodes": int(result.node_x.size),
                    "n_cells": int(result.cell_face_offsets.size - 1),
                    "n_tris": int((result.cell_type == "triangular").sum()),
                    "n_quads": int((result.cell_type == "quadrilateral").sum()),
                    "regions_processed": int(len(model.regions)),
                    "quality": {
                        "min_angle_deg": round(float(stats.get("min_angle_deg", 0.0)), 3),
                        "max_aspect_ratio": round(float(stats.get("max_aspect_ratio", 0.0)), 3),
                        "min_area": round(float(stats.get("min_area", 0.0)), 6),
                        "bbox_area": round(float(stats.get("bbox_area", 0.0)), 6),
                    },
                    "warnings": warn_list,
                }

            def _run_mesh_with_timeout(model=model, swe2d_meshing=swe2d_meshing,
                                       backend_name=backend_name, timeout_sec=timeout_sec):
                # Gmsh can occasionally stall on pathological geometry. Run it in
                # a child process so we can enforce a hard timeout.
                if str(backend_name).strip().lower() != "gmsh":
                    return _run_mesh_once(model=model, swe2d_meshing=swe2d_meshing, backend_name=backend_name)

                ctx = _mp.get_context("fork")
                q = ctx.Queue(maxsize=1)

                def _child_runner(q=q, model=model, swe2d_meshing=swe2d_meshing, backend_name=backend_name):
                    try:
                        out = _run_mesh_once(model=model, swe2d_meshing=swe2d_meshing, backend_name=backend_name)
                        q.put({"ok": True, "out": out})
                    except Exception as exc:  # noqa: BLE001
                        q.put(
                            {
                                "ok": False,
                                "out": {
                                    "ok": False,
                                    "elapsed_sec": 0.0,
                                    "error": str(exc),
                                    "traceback": _tb.format_exc(),
                                    "warnings": [],
                                },
                            }
                        )

                p = ctx.Process(target=_child_runner, daemon=True)
                t0 = time.perf_counter()
                p.start()
                p.join(timeout=timeout_sec)
                elapsed = round(time.perf_counter() - t0, 3)

                if p.is_alive():
                    try:
                        p.terminate()
                    except Exception:
                        pass
                    try:
                        p.join(timeout=2.0)
                    except Exception:
                        pass
                    return {
                        "ok": False,
                        "elapsed_sec": elapsed,
                        "error": f"Topology meshing timed out after {timeout_sec:.0f}s",
                        "warnings": [],
                        "timeout": True,
                        "timeout_sec": float(timeout_sec),
                    }

                if not q.empty():
                    payload = q.get()
                    out = payload.get("out", {}) if isinstance(payload, dict) else {}
                    if isinstance(out, dict):
                        out.setdefault("elapsed_sec", elapsed)
                    return out

                return {
                    "ok": False,
                    "elapsed_sec": elapsed,
                    "error": "Topology meshing ended without result payload",
                    "warnings": [],
                }

            # ---------------------------------------------------------------
            # Run heavy meshing in a background thread so the bridge QTimer
            # stays responsive. Gmsh execution itself is wrapped in a child
            # process with timeout enforcement.
            # ---------------------------------------------------------------

            def _worker(model=model, swe2d_meshing=swe2d_meshing, backend_name=backend_name,
                        result_path=result_path):
                try:
                    out = _run_mesh_with_timeout(
                        model=model,
                        swe2d_meshing=swe2d_meshing,
                        backend_name=backend_name,
                        timeout_sec=timeout_sec,
                    )
                except Exception as exc:  # noqa: BLE001
                    out = {
                        "ok": False,
                        "elapsed_sec": 0.0,
                        "error": str(exc),
                        "traceback": _tb.format_exc(),
                        "warnings": [],
                    }
                except BaseException as exc:  # catch SystemExit, KeyboardInterrupt, etc.
                    out = {
                        "ok": False,
                        "elapsed_sec": 0.0,
                        "error": f"BaseException: {exc!r}",
                        "traceback": _tb.format_exc(),
                        "warnings": [],
                    }
                finally:
                    try:
                        os.makedirs(os.path.dirname(result_path), exist_ok=True)
                        with open(result_path, "w") as _f:
                            _json.dump(out, _f, indent=2)
                    except Exception:
                        pass

            # Remove any stale result from a previous run.
            try:
                os.remove(result_path)
            except FileNotFoundError:
                pass

            t = threading.Thread(target=_worker, daemon=True, name="topology-mesh")
            t.start()
            return {
                "status": "running",
                "result_path": result_path,
                "thread": t.name,
                "n_regions": len(model.regions),
                "timeout_sec": float(timeout_sec),
            }

        if action == "get_topology_mesh_result":
            result_path = os.path.join("/tmp", "qgis-live-bridge", "topology_mesh_result.json")
            if not os.path.exists(result_path):
                return {"status": "pending"}
            import json as _json
            with open(result_path) as _f:
                data = _json.load(_f)
            data["status"] = "complete"
            return data

        if action == "list_modules":
            import sys
            pattern = str(params.get("pattern", "")).lower()
            modules = sorted(k for k in sys.modules.keys() if pattern in k.lower())
            paths = {}
            for m in modules:
                mod = sys.modules[m]
                paths[m] = getattr(mod, "__file__", None)
            return {"modules": paths}

        if action == "describe_geometry":
            name = str(params.get("name", "")).strip()
            if not name:
                raise ValueError("params.name is required")
            layers = QgsProject.instance().mapLayersByName(name)
            if not layers:
                raise ValueError(f"Layer not found: {name}")
            layer = layers[0]
            rows = []
            for feat in layer.getFeatures():
                geom = feat.geometry()
                if geom is None or geom.isEmpty():
                    n_verts = 0; area = 0.0; bbox_w = bbox_h = 0.0
                else:
                    poly = geom.asPolygon()
                    n_verts = sum(len(ring) for ring in poly) if poly else 0
                    bb = geom.boundingBox()
                    bbox_w = round(bb.width(), 1); bbox_h = round(bb.height(), 1)
                    area = round(geom.area(), 1)
                rows.append({"fid": feat.id(), "n_verts": n_verts,
                             "area": area, "bbox_w": bbox_w, "bbox_h": bbox_h})
            return {"layer": name, "features": rows}

        if action == "run_swe2d_workbench":
            import sys

            mod = sys.modules.get("swe2d_workbench_qt")
            if mod is None:
                raise RuntimeError("swe2d_workbench_qt module is not loaded in this QGIS session")

            wins = getattr(mod, "_SWE2D_WORKBENCH_WINDOWS", None)
            if not isinstance(wins, list) or not wins:
                raise RuntimeError("No open SWE2D Workbench window found")

            dlg = wins[-1]
            if dlg is None:
                raise RuntimeError("Invalid SWE2D Workbench window handle")

            scheme = params.get("reconstruction_mode", None)
            if scheme is not None:
                try:
                    scheme = int(scheme)
                except Exception as exc:
                    raise ValueError(f"Invalid reconstruction_mode: {exc}")

                combo = getattr(dlg, "reconstruction_combo", None)
                if combo is None:
                    raise RuntimeError("SWE2D Workbench has no reconstruction_combo")

                idx = -1
                for i in range(combo.count()):
                    try:
                        v = int(combo.itemData(i))
                    except Exception:
                        continue
                    if v == scheme:
                        idx = i
                        break
                if idx < 0:
                    raise ValueError(f"reconstruction_mode={scheme} not found in combo")
                combo.setCurrentIndex(idx)

            log_view = getattr(dlg, "log_view", None)
            pre_log = ""
            if log_view is not None and hasattr(log_view, "toPlainText"):
                pre_log = str(log_view.toPlainText() or "")

            dlg._on_run()

            result_data = getattr(dlg, "_result_data", None)
            if not isinstance(result_data, dict) or "h" not in result_data:
                return {
                    "ok": False,
                    "error": "Run completed but no result data returned",
                }

            h = np.asarray(result_data.get("h"), dtype=np.float64)
            hu = np.asarray(result_data.get("hu"), dtype=np.float64)
            hv = np.asarray(result_data.get("hv"), dtype=np.float64)
            h_safe = np.maximum(h, 1.0e-12)
            vel = np.sqrt((hu / h_safe) ** 2 + (hv / h_safe) ** 2)

            rec_name = None
            rec_mode = None
            combo = getattr(dlg, "reconstruction_combo", None)
            if combo is not None:
                try:
                    rec_name = str(combo.currentText())
                except Exception:
                    rec_name = None
                try:
                    rec_mode = int(combo.currentData())
                except Exception:
                    rec_mode = None

            post_log = ""
            if log_view is not None and hasattr(log_view, "toPlainText"):
                post_log = str(log_view.toPlainText() or "")

            new_log = post_log[len(pre_log):].strip() if post_log.startswith(pre_log) else post_log[-4000:]
            log_tail = [ln for ln in new_log.splitlines() if ln.strip()][-12:]

            return {
                "ok": True,
                "reconstruction_mode": rec_mode,
                "reconstruction_name": rec_name,
                "n_cells": int(h.size),
                "depth_min": float(np.min(h)),
                "depth_max": float(np.max(h)),
                "velocity_max": float(np.max(vel)),
                "velocity_mean": float(np.mean(vel)),
                "gpu_active": bool(np.asarray(result_data.get("gpu_active", False)).item()),
                "last_mass_total": float(np.asarray(result_data.get("last_mass_total", -1.0)).item()),
                "log_tail": log_tail,
            }

        raise ValueError(f"Unsupported action: {action}")


def _resolve_runtime_dir() -> str:
    default_dir = os.path.join("/tmp", "qgis-live-bridge")
    return os.environ.get("QGIS_LIVE_BRIDGE_DIR", default_dir)


def _resolve_token() -> str:
    token = os.environ.get("QGIS_LIVE_BRIDGE_TOKEN", "")
    if token:
        return token
    # Fallback for quick local development; replace in production.
    return "change-me-qgis-bridge-token"


# Stop prior bridge if re-running in the console.
if "QGIS_LIVE_BRIDGE" in globals() and globals()["QGIS_LIVE_BRIDGE"] is not None:
    try:
        globals()["QGIS_LIVE_BRIDGE"].stop()
    except Exception:
        pass

QGIS_LIVE_BRIDGE = QgisLiveBridge(
    runtime_dir=_resolve_runtime_dir(),
    token=_resolve_token(),
    poll_ms=350,
)
QGIS_LIVE_BRIDGE.start()
