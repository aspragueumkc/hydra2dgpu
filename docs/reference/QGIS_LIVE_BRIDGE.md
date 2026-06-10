# QGIS Live Bridge (Polling)

This bridge lets terminal tools send commands to your current QGIS session
through the Python Console process.

## Files

- Console bridge: [tools/qgis_live_bridge_console.py](../tools/qgis_live_bridge_console.py)
- Sender script: [tools/qgis_live_bridge_send.py](../tools/qgis_live_bridge_send.py)

## 1. Start bridge in QGIS Python Console

Set token/runtime dir for this QGIS process, then run bridge:

```python
import os
os.environ['QGIS_LIVE_BRIDGE_TOKEN'] = 'my-local-secret-token'
os.environ['QGIS_LIVE_BRIDGE_DIR'] = '/tmp/qgis-live-bridge'
# Adjust the path below to match your QGIS plugin installation
exec(open('<QGIS_PLUGIN_DIR>/tools/qgis_live_bridge_console.py').read(), globals())
```

You should see: `[QGIS-LIVE-BRIDGE] started ...`

To stop later:

```python
QGIS_LIVE_BRIDGE.stop()
```

## 2. Send commands from terminal

Use the same token/runtime dir:

```bash
cd <QGIS_PLUGIN_DIR>
export QGIS_LIVE_BRIDGE_TOKEN='my-local-secret-token'
export QGIS_LIVE_BRIDGE_DIR='/tmp/qgis-live-bridge'
python3 tools/qgis_live_bridge_send.py ping
```

## Supported actions

- `ping`
- `get-project-info`
- `list-layers`
- `select-layer --name <layer_name>`
- `zoom-to-layer --name <layer_name>`
- `trigger-action --object-name <QAction.objectName>`
- `raw --action <action_name> --params-json '{...}'`

## Backwater plugin examples

Open Backwater dock/panel:

```bash
python3 tools/qgis_live_bridge_send.py trigger-action --object-name BackwaterMenuOpenPanelAction
```

Open Unsteady Input window:

```bash
python3 tools/qgis_live_bridge_send.py trigger-action --object-name BackwaterMenuUnsteadyInputDialogAction
```

Run unsteady model (with values currently in GUI widgets):

```bash
python3 tools/qgis_live_bridge_send.py trigger-action --object-name BackwaterMenuRunUnsteadyAction
```

## Security notes

- Keep token non-default.
- Bridge reads local files only (`/tmp/...` by default); it is not network exposed.
- Current bridge intentionally uses a whitelist and does not execute arbitrary Python.
