# HYDRA MCP Server

An [MCP](https://modelcontextprotocol.io/) server (Python `mcp` SDK, stdio
transport) that lets an AI agent operate HYDRA — the GPU-accelerated 2D
shallow-water QGIS plugin in this repository. Full plan:
[`docs/HYDRA_MCP_SERVER_PLAN.md`](../../docs/HYDRA_MCP_SERVER_PLAN.md).

**Design principle:** the server never re-implements modeling logic. Every
tool is a thin adapter over an existing core module (`swe2d/services`,
`swe2d/results`, …). If a capability does not exist in the core, it is added
to the core first — the server inherits it. MCP-exposed behavior is by
construction identical to GUI/CLI behavior.

**Tool surface (39 tools):**

| Tier | Phase | Tools | Read-only? |
|---|---|---|---|
| A | 0 | `model_inspect`, `run_list`, `results_query` (3) | yes |
| A | 1 | model building / runs / results (19) | mixed — most write model files or spawn solver jobs |
| B | 2.A–2.C | `gui_launch` … `gui_screenshot` (7) | introspection is read-only; `gui_set_value` mutates the live GUI |
| B | 3 | `gui_click` … `gui_close` (6) | no — synthesizes input / drives the session (except `gui_read_log`) |
| C | 4 | `design_*` (4) | proposals are read-only; `design_apply_patch` edits source (disabled by default) |

## Phase 0 tool set (read-only modeling)

All tools validate inputs, return structured JSON (`{"ok": true, ...}`), and
on failure return `{"ok": false, "error": ..., ...}` with actionable context
(valid run ids / fields / timesteps) — never a traceback.

### `model_inspect(gpkg_path)`

List baked meshes, layers/tables (from `gpkg_contents`), saved simulation
configs, and simulation runs in a HYDRA model GeoPackage.

Wraps `swe2d/services/gpkg_persistence_service.py`
(`load_simulation_configs`, `collect_baked_runs_from_gpkg`); mesh/layer
listings are direct sqlite3 reads of `swe2d_baked_mesh` / `gpkg_contents`.

```json
{"gpkg_path": "/path/to/model.gpkg"}
```

### `run_list(gpkg_path)`

List simulation runs in a results GeoPackage: run id, mesh name, cell and
timestep counts, created timestamp, wallclock start/end/duration, and a
config summary when a `swe2d_run_logs` record exists.

```json
{"gpkg_path": "/path/to/results.gpkg"}
```

### `results_query(gpkg_path, run_id, field, timestep=None)`

Agent-friendly **summary** of a result field — shape, dtype, min/max/mean,
NaN count, available timesteps — never raw megabyte arrays.

- `field`: `h`, `hu`, `hv` (per-timestep snapshots) or `max_h`, `max_hu`,
  `max_hv` (per-cell GPU max tracking; `timestep` ignored).
- `timestep`: simulation time in seconds; the nearest stored snapshot is
  used. Omit to summarize the whole `(n_timesteps, n_cells)` array.

```json
{"gpkg_path": "/path/to/results.gpkg", "run_id": "run_20260724_101500",
 "field": "h", "timestep": 600.0}
```

## Phase 2.B tool set (live GUI — read/write values)

Phase 2.B adds value read/write tools driven over the same QLocalSocket bridge.
These tools take a **dot-separated widget path** from the root widget rather than
a single `objectName`, enabling precise targeting of deeply nested controls.

All value tools support these widget types:

| Widget | Get type | Get value | Set value |
|--------|----------|-----------|-----------|
| QSpinBox | `int` | `value()` | `setValue(int)` |
| QDoubleSpinBox | `float` | `value()` | `setValue(float)` |
| QCheckBox | `bool` | `isChecked()` | `setChecked(bool)` |
| QComboBox | `str` | `currentText()` | `setCurrentIndex(findText(str))` |
| QLineEdit | `str` | `text()` | `setText(str)` |
| QTextEdit | `str` | `toPlainText()` | `setPlainText(str)` |
| QLabel | `str` | `text()` | — (read-only) |

### `gui_find_widget_by_path(path, token_path=None)`

Return the widget at a dot-separated path and its key properties.

```json
{"path": "central_container.simulation_tab.cfl_spin"}
```

Response example:

```json
{
  "ok": true,
  "widget": {
    "object_name": "cfl_spin",
    "class_name": "QDoubleSpinBox",
    "widget_id": 140234567890,
    "geometry": {"x": 200, "y": 50, "width": 100, "height": 30},
    "is_visible": true
  }
}
```

### `gui_get_value(path, token_path=None)`

Read the current value of the widget at *path*.

```json
{"path": "studio_window.simulation_tab.run_duration"}
```

Response example:

```json
{"ok": true, "type": "QDoubleSpinBox", "value": 3600.0}
```

### `gui_set_value(path, value, token_path=None)`

Set the value of the widget at *path*.

```json
{"path": "studio_window.simulation_tab.cfl_spin", "value": 0.8}
```

Response:

```json
{"ok": true}
```

Or on error:

```json
{"ok": false, "error": "no item matching 'invalid_option'"}
```

### `gui_screenshot(path, format="png", token_path=None)`

Capture a screenshot of the widget at *path* and return a base64-encoded image.

```json
{"path": "studio_window.simulation_tab", "format": "png"}
```

Response example:

```json
{
  "ok": true,
  "image_b64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
  "format": "png",
  "width": 800,
  "height": 600
}
```

Or on error (widget not found, not visible, unsupported format):

```json
{"ok": false, "error": "widget not available"}
```

## Phase 2.A tool set (live GUI introspection — Phase 2.B adds read/write + Phase 2.C adds screenshot)

These tools drive a live QGIS session through the HYDRA MCP bridge.  The
bridge is a `QLocalServer` that runs **inside** the QGIS process; handlers
execute on the Qt GUI thread, making direct widget access safe and latency-free.

The bridge is injected via:

```bash
# Offscreen (no display required — recommended for automation)
QT_QPA_PLATFORM=offscreen qgis --noplugins --project /path/to/project.qgs \
    --code /path/to/repo/tools/hydra_mcp/qgis_bridge.py

# Xvfb (virtual display — CI environments)
xvfb-run -a qgis --noplugins --project /path/to/project.qgs \
    --code /path/to/repo/tools/hydra_mcp/qgis_bridge.py

# Display (attach to user's running QGIS — the bridge must already be running;
# set HYDRA_MCP_BRIDGE=1 when loading the plugin, or inject via the Python console)
```

**Token discovery:** the bridge writes a `0600` JSON token file at startup
containing the socket name, token, and version.  The server locates it via:

1. Explicit `token_path` argument (recommended for automation).
2. `HYDRA_MCP_BRIDGE_TOKEN` / `HYDRA_MCP_BRIDGE_SOCKET` env vars (set from
   the `HYDRA_MCP_BRIDGE_READY <socket> <token_path>` line the bridge prints).
3. Auto-discovery: scan `$XDG_RUNTIME_DIR` / `/tmp` for the newest
   `hydra_mcp_bridge_*.json` owned by the current user.

**Single-flight:** the bridge processes one request at a time.  Concurrent
requests receive an error; retry after a short delay.

### `gui_launch(mode="offscreen", project=None, timeout=60)`

Spawn a QGIS instance with the bridge injected and wait for the token file.
Returns session metadata (`session_id`, `socket_name`, `token_path`, `pid`).

```json
{"mode": "offscreen", "project": "/path/to/project.qgs"}
```

Response example:

```json
{
  "ok": true,
  "session_id": "aaron_12345_abc123",
  "socket_name": "hydra_mcp_bridge_aaron_12345_abc123",
  "token_path": "/run/user/1000/hydra_mcp_bridge_aaron_12345_abc123.json",
  "mode": "offscreen",
  "pid": 54321
}
```

**Modes:**

| Mode | How | Best for |
|---|---|---|
| `offscreen` | `QT_QPA_PLATFORM=offscreen` | Headless CI, no GPU/display |
| `xvfb` | Xvfb virtual display | CI behavioral gates |
| `display` | Poll for a bridge token file from a running QGIS | User live session assistance |

### Display attach / live agent-assisted modeling

Use `display` mode when a modeler already has QGIS open and you want the agent
to co-operate with their live session.

1. **Auto-start (recommended):** launch QGIS with the environment variable set:
   ```bash
   HYDRA_MCP_BRIDGE=1 qgis
   ```
   The HYDRA plugin will start the bridge automatically and write a 0600 token
   file under `$XDG_RUNTIME_DIR` (or `/tmp`).

2. **Python console bootstrap** (when QGIS is already running without the env
   var): open **Plugins ▸ Python Console** and paste:
   ```python
   import os
   if not os.environ.get("HYDRA_MCP_BRIDGE"):
       os.environ["HYDRA_MCP_BRIDGE"] = "1"
       from tools.hydra_mcp.qgis_bridge import bootstrap_bridge_if_needed
       bootstrap_bridge_if_needed()
   ```
   Watch the QGIS log for the `HYDRA_MCP_BRIDGE_READY <socket> <token_path>`
   line.

3. From the MCP client, attach to the session:
   ```json
   {"mode": "display"}
   ```
   `gui_launch(mode="display", timeout=60)` waits up to the configured timeout
   for the token file to appear, connects, and returns the same session metadata
   as the other modes.  Subsequent `gui_widget_tree` / `gui_get_value` /
   `gui_set_value` / `gui_screenshot` calls drive the live QGIS session.

### `gui_widget_tree(root=None, token_path=None)`

Return the live widget tree from the QGIS session as a flat depth-first list
of nodes.  Each node is a dict::

```json
{
  "object_name": "meshTabRunDuration",
  "class_name": "QDoubleSpinBox",
  "widget_id": 140234567890,
  "parent_id": 140234567889,
  "text": "3600.0",
  "depth": 2
}
```

```json
{"root": "SWE2DStudioDialog"}
```

### `gui_find_widget(name, token_path=None)`

Return the single widget node whose `objectName` exactly matches `name`,
or `{"ok": false, "error": "widget not found"}` if none exists.

```json
{"name": "meshTabRunDuration"}
```

## Registering in an MCP client

From the repo root:

```json
"hydra": {
  "command": "uv",
  "args": ["run", "--no-project", "--with", "mcp", "--with", "numpy", "--with", "PyQt5", "--with", "PyQt5-Qt5", "python", "tools/hydra_mcp/server.py"],
  "startupTimeoutMs": 60000,
  "disabledTools": ["design_apply_patch"]
}
```

`--no-project` matters: the repo root has a `pyproject.toml`, and without it
`uv run` enters project mode and creates an untracked `uv.lock` and a large
`.venv` in the repo root. `--with numpy` is explicit because the modeling
tools need `numpy` beyond the stdlib. `--with PyQt5 --with PyQt5-Qt5` are
required for the `gui_*` tools — the bridge client uses
`PyQt5.QtNetwork.QLocalSocket`, and without these every `gui_*` call fails
with "BridgeClient requires PyQt5/QtNetwork". `design_apply_patch` edits
source files and is disabled by default; enable it only with an explicit
approval rule.

This is already wired into `.kimi-code/mcp.json` for project development.
The server inserts the repo root into `sys.path` itself, so no `PYTHONPATH`
setup is required. Any MCP client that supports stdio servers can use the
same command (substitute `python` from the QGIS/`qgis_stable` environment if
you prefer it over `uv`).

## Layout

```
tools/hydra_mcp/
  server.py                 # FastMCP stdio server (mcp SDK) — registers all 39 tools
  bridge_client.py          # QLocalSocket client, token auth, single-flight
  qgis_bridge.py            # Injected into QGIS; QLocalServer + GUI-thread handlers
  tools_modeling.py         # Tier A read-only: inspect/list/query (Phase 0)
  tools_modeling_phase1.py  # Tier A mutating: build/run/results (Phase 1)
  tools_gui.py              # Tier B: live GUI launch/introspection/behavioral (Phases 2+3)
  tools_design.py           # Tier C: design patch tools (Phase 4)
  widget_screenshot.py      # QBuffer-based widget capture helper
  workspace.py              # Workspace containment / path resolution
  README.md                 # this file
```

## Phase 1 tool set (Tier A — model building, runs, results)

19 tools. Mutating tools write to model/results GeoPackages or spawn solver
processes; scope your client approval rules accordingly.

- `model_create(gpkg_path, crs)` — create an empty model GeoPackage.
- `mesh_generate(domain, spacing, backend="builtin")` — structured rectangular mesh (in-memory).
- `mesh_bake(gpkg_path, mesh_name, mesh_data, crs_wkt="")` — persist a mesh into the model.
- `terrain_assign(gpkg_path, mesh_name, source, method="raster")` — sample terrain onto a baked mesh.
- `bc_configure` / `rainfall_configure` / `drainage_configure` / `structures_configure` —
  store per-mesh forcing/structure configuration in the model.
- `spec_build(gpkg_path, mesh_name, run_params=None, results_gpkg_path=None)` — build a canonical `swe2d-run/2` spec (read-only).
- `spec_validate(spec)` / `spec_diff(spec_a, spec_b)` — validate / diff specs (read-only).
- `run_start(spec, job_name=None)` / `run_status(job_id)` / `run_cancel(job_id)` —
  async solver subprocess jobs.
- `run_batch(batch_spec, max_workers=0)` — batch of simulations.
- `results_timeseries(gpkg_path, run_id, line_id)` — line timeseries (read-only).
- `results_export(gpkg_path, run_id, out_path, format="csv")` — CSV summary export (writes `out_path`).
- `results_render(gpkg_path, run_id, field, timestep=None, out_path=None)` — PNG field plot.
- `results_compare(gpkg_path, run_a, run_b, field, tolerance=1e-6)` — field comparison (read-only).

## Phase 3 tool set (behavioral GUI)

6 tools driving the live session over the bridge. All are **mutating** except
`gui_read_log`: they synthesize input or trigger actions in the user's QGIS.

- `gui_click(path=None, ...)` — click a widget.
- `gui_key(key, ...)` — send a key press.
- `gui_run_action(object_name=None, ...)` — trigger a QAction.
- `gui_read_log(max_lines=1000, ...)` — read the bridge/server log (read-only).
- `gui_run_simulation(run_duration_text=None, ...)` — start a simulation from the Studio UI.
- `gui_close(token_path=None)` — close the bridged QGIS session.

## Phase 4 tool set (design)

4 tools that propose or apply source edits:

- `design_rename_widget(old, new)` — propose a patch renaming a widget `objectName`.
- `design_relabel_widget(name, text)` — propose a patch relabeling a widget.
- `design_preview_patch(edits)` — preview a unified diff for proposed edits.
- `design_apply_patch(diff)` — **apply** a patch to the source tree (disabled
  by default via `disabledTools`; enable only behind an approval rule).

The remaining phases (further behavioral gates, CI re-enable) are
specified in [`docs/HYDRA_MCP_SERVER_PLAN.md`](../../docs/HYDRA_MCP_SERVER_PLAN.md) §7.
