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
