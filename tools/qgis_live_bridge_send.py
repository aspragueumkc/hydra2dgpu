#!/usr/bin/env python3
"""Terminal-side sender for qgis_live_bridge_console.py.

Usage examples:

  python3 tools/qgis_live_bridge_send.py ping
  python3 tools/qgis_live_bridge_send.py list-layers
  python3 tools/qgis_live_bridge_send.py trigger-action --object-name BackwaterMenuOpenPanelAction
  python3 tools/qgis_live_bridge_send.py trigger-action --object-name BackwaterMenuUnsteadyInputDialogAction
  python3 tools/qgis_live_bridge_send.py zoom-to-layer --name cross_sections

Environment variables:
- QGIS_LIVE_BRIDGE_DIR   (default: /tmp/qgis-live-bridge)
- QGIS_LIVE_BRIDGE_TOKEN (default: change-me-qgis-bridge-token)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def send_command(runtime_dir: str, token: str, action: str, params: dict, timeout: float):
    cmd_path = os.path.join(runtime_dir, "command.json")
    resp_path = os.path.join(runtime_dir, "response.json")
    request_id = str(uuid.uuid4())

    cmd = {
        "request_id": request_id,
        "token": token,
        "action": action,
        "params": params,
        "timestamp_utc": utc_now(),
    }
    write_json_atomic(cmd_path, cmd)

    start = time.time()
    while True:
        if os.path.exists(resp_path):
            try:
                resp = read_json(resp_path)
            except Exception:
                resp = None
            if isinstance(resp, dict) and resp.get("request_id") == request_id:
                return resp

        if time.time() - start > timeout:
            raise TimeoutError(
                f"Timed out waiting for response (request_id={request_id}) at {resp_path}"
            )
        time.sleep(0.2)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Send commands to a running QGIS live bridge")
    p.add_argument(
        "--runtime-dir",
        default=os.environ.get("QGIS_LIVE_BRIDGE_DIR", "/tmp/qgis-live-bridge"),
        help="Bridge runtime directory",
    )
    p.add_argument(
        "--token",
        default=os.environ.get("QGIS_LIVE_BRIDGE_TOKEN", "change-me-qgis-bridge-token"),
        help="Shared token",
    )
    p.add_argument("--timeout", type=float, default=8.0, help="Response timeout seconds")

    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ping")
    sub.add_parser("get-project-info")
    sub.add_parser("list-layers")

    p_sel = sub.add_parser("select-layer")
    p_sel.add_argument("--name", required=True)

    p_zoom = sub.add_parser("zoom-to-layer")
    p_zoom.add_argument("--name", required=True)

    p_trig = sub.add_parser("trigger-action")
    p_trig.add_argument("--object-name", required=True)

    p_reload = sub.add_parser("reload-plugin")
    p_reload.add_argument(
        "--plugin-name",
        default="",
        help="Optional plugin name; if omitted bridge auto-detects HYDRA plugin",
    )

    p_open_demo = sub.add_parser("open-swe2d-demo-dialog")
    p_open_demo.add_argument(
        "--plugin-name",
        default="",
        help="Optional plugin name; if omitted bridge auto-detects HYDRA plugin",
    )

    p_run_topo_gui = sub.add_parser("run-topology-mesh-gui")
    p_run_topo_gui.add_argument("--target", choices=["auto", "demo", "studio"], default="auto")
    p_run_topo_gui.add_argument("--backend", default=None, help="Optional topology backend override (gmsh|tqmesh)")
    p_run_topo_gui.add_argument("--regions-layer", default=None)
    p_run_topo_gui.add_argument("--arcs-layer", default=None)
    p_run_topo_gui.add_argument("--nodes-layer", default=None)
    p_run_topo_gui.add_argument("--constraints-layer", default=None)
    p_run_topo_gui.add_argument("--quad-edges-layer", default=None)
    p_run_topo_gui.add_argument("--regions-subset-sql", default=None)
    p_run_topo_gui.add_argument("--quad-edges-subset-sql", default=None)
    p_run_topo_gui.add_argument("--wait", action="store_true", help="Wait for completion and return result payload")
    p_run_topo_gui.add_argument("--wait-timeout-sec", type=float, default=1800.0)

    sub.add_parser("get-topology-mesh-gui-result")

    p_raw = sub.add_parser("raw")
    p_raw.add_argument("--action", required=True)
    p_raw.add_argument("--params-json", default="{}")

    p_run2d = sub.add_parser("run-swe2d")
    p_run2d.add_argument(
        "--reconstruction-mode",
        type=int,
        default=None,
        help="Optional reconstruction mode override (0..4) before run",
    )

    return p


def main() -> int:
    args = parser().parse_args()

    if args.cmd == "ping":
        action = "ping"
        params = {}
    elif args.cmd == "get-project-info":
        action = "get_project_info"
        params = {}
    elif args.cmd == "list-layers":
        action = "list_layers"
        params = {}
    elif args.cmd == "select-layer":
        action = "select_layer"
        params = {"name": args.name}
    elif args.cmd == "zoom-to-layer":
        action = "zoom_to_layer"
        params = {"name": args.name}
    elif args.cmd == "trigger-action":
        action = "trigger_action"
        params = {"object_name": args.object_name}
    elif args.cmd == "reload-plugin":
        action = "reload_plugin"
        params = {}
        if args.plugin_name:
            params["plugin_name"] = args.plugin_name
    elif args.cmd == "open-swe2d-demo-dialog":
        action = "open_swe2d_demo_dialog"
        params = {}
        if args.plugin_name:
            params["plugin_name"] = args.plugin_name
    elif args.cmd == "run-topology-mesh-gui":
        action = "run_topology_mesh_gui"
        params = {
            "target": args.target,
            "wait_for_completion": bool(args.wait),
            "wait_timeout_sec": float(args.wait_timeout_sec),
        }
        if args.backend is not None:
            params["backend"] = args.backend
        if args.regions_layer is not None:
            params["regions_layer"] = args.regions_layer
        if args.arcs_layer is not None:
            params["arcs_layer"] = args.arcs_layer
        if args.nodes_layer is not None:
            params["nodes_layer"] = args.nodes_layer
        if args.constraints_layer is not None:
            params["constraints_layer"] = args.constraints_layer
        if args.quad_edges_layer is not None:
            params["quad_edges_layer"] = args.quad_edges_layer
        if args.regions_subset_sql is not None:
            params["regions_subset_sql"] = args.regions_subset_sql
        if args.quad_edges_subset_sql is not None:
            params["quad_edges_subset_sql"] = args.quad_edges_subset_sql
    elif args.cmd == "get-topology-mesh-gui-result":
        action = "get_topology_mesh_gui_result"
        params = {}
    elif args.cmd == "run-swe2d":
        action = "run_swe2d_workbench"
        params = {}
        if args.reconstruction_mode is not None:
            params["reconstruction_mode"] = int(args.reconstruction_mode)
    elif args.cmd == "raw":
        action = args.action
        try:
            params = json.loads(args.params_json)
        except Exception as exc:
            raise SystemExit(f"Invalid --params-json: {exc}")
        if not isinstance(params, dict):
            raise SystemExit("--params-json must decode to an object")
    else:
        raise SystemExit(f"Unsupported command: {args.cmd}")

    try:
        response = send_command(
            runtime_dir=args.runtime_dir,
            token=args.token,
            action=action,
            params=params,
            timeout=args.timeout,
        )
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        return 2

    print(json.dumps(response, indent=2))
    return 0 if response.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
