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
import traceback
import uuid
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
        self.last_request_id = None
        self.timer = None

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

    def _poll_once(self):
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

        if action == "set_env":
            env = params.get("env")
            if not isinstance(env, dict):
                raise ValueError("params.env must be an object of key/value pairs")
            for k, v in env.items():
                os.environ[str(k)] = str(v)
            return {"set": list(env.keys())}

        if action == "run_topology_mesh":
            import sys
            import threading
            import multiprocessing as _mp
            import json as _json
            import importlib
            import time
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
            timeout_sec = max(30.0, float(params.get("timeout_sec", os.environ.get("BACKWATER_TOPOLOGY_MESH_TIMEOUT_SEC", "300"))))
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
