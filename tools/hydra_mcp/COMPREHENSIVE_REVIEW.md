# Comprehensive Review — HYDRA MCP Server (Phase 0–2)

## Scope and review basis

This review compares `docs/HYDRA_MCP_SERVER_PLAN.md` with the implementation in
`tools/hydra_mcp/`, `tests/test_hydra_mcp.py`, and `.kimi-code/mcp.json` in the
`hydra-mcp` worktree. The reviewed implementation identifies itself as Phases 0,
2.A, 2.B, and 2.C (`tools/hydra_mcp/server.py:1-9`; `tools/hydra_mcp/qgis_bridge.py:1-12`).

The plan intentionally schedules most Tier A production tools for Phase 1, advanced
Tier B behavioral tools for Phase 3, and Tier C design tools for Phase 4
(`docs/HYDRA_MCP_SERVER_PLAN.md:177-189`). Therefore, a tool marked **NOT DONE** is
not automatically a defect in the delivered phase. It is still a catalog gap that
must be closed before the single fully featured server described by the goal can
ship.

## Tool Catalog Coverage Matrix

Status meanings:

- **DONE** — registered MCP tool with the planned basic behavior.
- **PARTIAL** — a tool exists, but its contract, integration, or real-world operation
  is incomplete or defective.
- **NOT DONE** — no registered implementation exists. The complete registered set is
  only the ten names asserted in `tests/test_hydra_mcp.py:303-331` and implemented in
  `tools/hydra_mcp/server.py:42-270`.

### Tier A — production modeling and results

| Tool | Status | Notes |
|---|---|---|
| `model_create` | NOT DONE | No registration or adapter; current Tier A registrations begin with `model_inspect` (`tools/hydra_mcp/server.py:42-54`). Phase 1 work. |
| `model_inspect` | DONE | Registered and returns meshes, layers, configs, and runs (`tools/hydra_mcp/server.py:42-54`; `tools/hydra_mcp/tools_modeling.py:221-251`). It is read-only, but accepts unrestricted filesystem paths and contains direct schema queries rather than being exclusively a core adapter. |
| `mesh_generate` | NOT DONE | Not in the registered tool set (`tests/test_hydra_mcp.py:317-330`). Phase 1 work. |
| `mesh_bake` | NOT DONE | Not in the registered tool set (`tests/test_hydra_mcp.py:317-330`). Phase 1 work. |
| `terrain_assign` | NOT DONE | Not in the registered tool set (`tests/test_hydra_mcp.py:317-330`). Phase 1 work. |
| `bc_configure` | NOT DONE | Not in the registered tool set (`tests/test_hydra_mcp.py:317-330`). Phase 1 work. |
| `rainfall_configure` | NOT DONE | Not in the registered tool set (`tests/test_hydra_mcp.py:317-330`). Phase 1 work. |
| `drainage_configure` | NOT DONE | Not in the registered tool set (`tests/test_hydra_mcp.py:317-330`). Phase 1 work. |
| `structures_configure` | NOT DONE | Not in the registered tool set (`tests/test_hydra_mcp.py:317-330`). Phase 1 work. |
| `spec_build` | NOT DONE | Not in the registered tool set (`tests/test_hydra_mcp.py:317-330`). Depends on the CLI-first core work identified by the plan. |
| `spec_validate` | NOT DONE | Not in the registered tool set (`tests/test_hydra_mcp.py:317-330`). Depends on the CLI-first core work identified by the plan. |
| `spec_diff` | NOT DONE | Not in the registered tool set (`tests/test_hydra_mcp.py:317-330`). Phase 1/parity work. |
| `run_start` | NOT DONE | No async job tool exists (`tools/hydra_mcp/server.py:42-90` contains only the three Phase 0 reads). Phase 1 work. |
| `run_status` | NOT DONE | No job/status adapter exists (`tests/test_hydra_mcp.py:317-330`). Phase 1 work. |
| `run_cancel` | NOT DONE | No cancellation adapter exists (`tests/test_hydra_mcp.py:317-330`). Phase 1 work. |
| `run_batch` | NOT DONE | No batch adapter exists (`tests/test_hydra_mcp.py:317-330`). Phase 1 work. |
| `run_list` | DONE | Registered and joins baked runs with run-log metadata (`tools/hydra_mcp/server.py:57-69`; `tools/hydra_mcp/tools_modeling.py:254-320`). The direct SQL mirror is a maintainability concern. |
| `results_query` | PARTIAL | Registered, but it returns only statistics for `h`, `hu`, `hv`, and optional max-momentum fields (`tools/hydra_mcp/server.py:72-90`; `tools/hydra_mcp/tools_modeling.py:363-466`). The plan describes depth/velocity/etc. arrays; velocity is not derived/exposed, and raw or artifact-backed arrays cannot be obtained. |
| `results_timeseries` | NOT DONE | Not in the registered tool set (`tests/test_hydra_mcp.py:317-330`). Phase 1 work. |
| `results_export` | NOT DONE | Not in the registered tool set (`tests/test_hydra_mcp.py:317-330`). Phase 1 work. |
| `results_render` | NOT DONE | Not in the registered tool set (`tests/test_hydra_mcp.py:317-330`). Phase 1 work. |
| `results_compare` | NOT DONE | Not in the registered tool set (`tests/test_hydra_mcp.py:317-330`). Phase 1 work. |

Tier A coverage: **2 DONE, 1 PARTIAL, 19 NOT DONE** out of 22 planned tools.
Only the three Phase 0 deliverables were expected at this point, but
`results_query` does not fully match the catalog contract.

### Tier B — live GUI

| Tool | Status | Notes |
|---|---|---|
| `gui_launch` | PARTIAL | Registered (`tools/hydra_mcp/server.py:96-120`) but offscreen/Xvfb launch is not viable: `_find_qgis_python()` converts a QGIS executable into `python3`, then passes QGIS-only flags such as `--noplugins`, `--project`, and `--code` to that Python interpreter (`tools/hydra_mcp/tools_gui.py:142-186`, `tools/hydra_mcp/tools_gui.py:253-278`). Readiness detection also attempts to read the child pipe through `/proc/<pid>/fd/1` instead of `proc.stdout` (`tools/hydra_mcp/tools_gui.py:200-229`, `tools/hydra_mcp/tools_gui.py:298-309`). |
| `gui_widget_tree` | PARTIAL | Registered and bridged (`tools/hydra_mcp/server.py:123-146`; `tools/hydra_mcp/qgis_bridge.py:308-317`), but real hierarchy is unreliable because the shared walker uses recursive `findChildren()` at every recursion level (`swe2d/workbench/devtools/widget_walker.py:139-152`). No real QGIS/QLocalSocket test validates it. |
| `gui_find_widget` | DONE | Registered and searches exact `objectName` across top-level trees (`tools/hydra_mcp/server.py:149-168`; `tools/hydra_mcp/qgis_bridge.py:336-351`). It still depends on a bridge session that `gui_launch` cannot currently start. |
| `gui_get_value` | PARTIAL | Registered (`tools/hydra_mcp/server.py:195-216`) but uses exact Python class-name equality rather than `isinstance`, so widget subclasses are rejected (`tools/hydra_mcp/qgis_bridge.py:91-115`). Dot-path resolution is ambiguous/incorrect because the core helper recursively searches descendants and does not consume an optional root prefix (`swe2d/workbench/devtools/widget_walker.py:165-195`). The public `timeout` is ignored (`tools/hydra_mcp/tools_gui.py:478-504`). |
| `gui_set_value` | PARTIAL | Registered (`tools/hydra_mcp/server.py:219-244`) with the same path, subclass, and ignored-timeout defects (`tools/hydra_mcp/qgis_bridge.py:118-146`; `tools/hydra_mcp/tools_gui.py:521-550`). `bool(value)` also turns any nonempty string, including `"false"`, into `True` (`tools/hydra_mcp/qgis_bridge.py:129-131`). |
| `gui_click` | NOT DONE | No MCP registration or bridge handler (`tools/hydra_mcp/server.py:93-270`; `tools/hydra_mcp/qgis_bridge.py:287-299`). Phase 3 work. |
| `gui_key` | NOT DONE | No MCP registration or bridge handler (`tools/hydra_mcp/server.py:93-270`; `tools/hydra_mcp/qgis_bridge.py:287-299`). Phase 3 work. |
| `gui_run_action` | NOT DONE | No MCP registration or bridge handler (`tools/hydra_mcp/server.py:93-270`; `tools/hydra_mcp/qgis_bridge.py:287-299`). Phase 3 work. |
| `gui_screenshot` | PARTIAL | Registered, but the plan calls for `target=dialog|dock|canvas` and a PNG artifact; implementation instead requires a widget path and returns inline base64 (`tools/hydra_mcp/server.py:247-270`). More seriously, it passes `io.BytesIO` to `QPixmap.save`, whose real PyQt API expects a filename or `QIODevice`; mocks conceal this (`tools/hydra_mcp/widget_screenshot.py:31-53`; `tests/test_hydra_mcp.py:1438-1469`). It also ignores the `save()` success boolean. |
| `gui_read_log` | NOT DONE | No MCP registration or bridge handler (`tools/hydra_mcp/server.py:93-270`; `tools/hydra_mcp/qgis_bridge.py:287-299`). Phase 3 work. |
| `gui_run_simulation` | NOT DONE | No MCP registration or bridge handler (`tools/hydra_mcp/server.py:93-270`; `tools/hydra_mcp/qgis_bridge.py:287-299`). Phase 3 work. |
| `gui_save_project` | NOT DONE | No MCP registration or bridge handler (`tools/hydra_mcp/server.py:93-270`; `tools/hydra_mcp/qgis_bridge.py:287-299`). Phase 3 work. |
| `gui_close` | NOT DONE | No lifecycle tool exists; spawned QGIS and Xvfb processes cannot be cleanly closed through MCP (`tools/hydra_mcp/tools_gui.py:188-250`, `tools/hydra_mcp/tools_gui.py:282-295`). Phase 3 work, but its absence also makes current Phase 2 sessions leak-prone. |

Additional registered tool: `gui_find_widget_by_path` is not separately listed in the
plan catalog. It is a useful extension, but it inherits the path-resolution defects
above (`tools/hydra_mcp/server.py:171-192`; `tools/hydra_mcp/tools_gui.py:434-472`).

Tier B catalog coverage: **1 DONE, 5 PARTIAL, 7 NOT DONE** out of 13 planned tools.
The Phase 2 deliverables exist by name, but launch, path traversal, timeout handling,
and screenshot integration prevent Phase 2 from being considered complete.

### Tier C — design and development

| Tool | Status | Notes |
|---|---|---|
| `design_rename_widget` | NOT DONE | No `tools_design.py`, server registration, or bridge method exists (`tools/hydra_mcp/server.py:30-37`, `tools/hydra_mcp/server.py:273-274`). Phase 4 work. |
| `design_relabel_widget` | NOT DONE | No `tools_design.py`, server registration, or bridge method exists (`tools/hydra_mcp/server.py:30-37`, `tools/hydra_mcp/server.py:273-274`). Phase 4 work. |
| `design_preview_patch` | NOT DONE | No `tools_design.py`, server registration, or bridge method exists (`tools/hydra_mcp/server.py:30-37`, `tools/hydra_mcp/server.py:273-274`). Phase 4 work. |
| `design_apply_patch` | NOT DONE | No implementation exists, and the project MCP config also omits the required default-disable entry (`.kimi-code/mcp.json:8-12`). Phase 4 work. |

Tier C coverage: **0 DONE, 0 PARTIAL, 4 NOT DONE**.

## Architecture Compliance

| Requirement | Status | Evidence and assessment |
|---|---|---|
| One MIT server; no dev/prod split | PASS for current skeleton | A single `FastMCP("hydra")` instance registers Tier A and Tier B tools (`tools/hydra_mcp/server.py:30-39`). No second server or feature-gated variant was found. |
| MCP stdio transport | PASS | `FastMCP` is used and `mcp.run()` relies on its stdio default (`tools/hydra_mcp/server.py:30-39`, `tools/hydra_mcp/server.py:273-274`). No SSE or HTTP transport is present. |
| Tier A in-process over core/services | PARTIAL | The tools are in process, but `tools_modeling.py` explicitly duplicates schema reads for mesh, layer, and run-log data (`tools/hydra_mcp/tools_modeling.py:1-15`, `tools/hydra_mcp/tools_modeling.py:72-183`) instead of adding missing capability to and wrapping the core as required by the design principle. |
| Live bridge uses QLocalSocket/QLocalServer | PASS at code level | Client constructs `QLocalSocket` (`tools/hydra_mcp/bridge_client.py:68-70`); bridge constructs `QLocalServer` (`tools/hydra_mcp/qgis_bridge.py:164-170`). |
| Bridge injected with `qgis --code` | FAIL operationally | The command appends `--code qgis_bridge.py`, but invokes a derived `python3` path rather than the QGIS executable (`tools/hydra_mcp/tools_gui.py:142-186`, `tools/hydra_mcp/tools_gui.py:253-278`). Thus the apparent architecture is not executable as written. |
| Server and bridge environments support Qt/QGIS | FAIL | Project MCP config launches a fresh `uv` environment with only `mcp` and `numpy` (`.kimi-code/mcp.json:8-11`), while every live-GUI request requires `PyQt5/QtNetwork` in the MCP server process (`tools/hydra_mcp/bridge_client.py:45-67`). This contradicts the plan's same-QGIS-environment requirement and makes Tier B unavailable in the documented project configuration. |
| Length-prefixed JSON-RPC | PARTIAL | Four-byte big-endian framing is implemented (`tools/hydra_mcp/bridge_client.py:22-42`). There is no maximum frame length, malformed JSON handling, top-level object validation, or JSON-RPC version validation; malformed/unbounded input can escape the Qt slot or consume memory (`tools/hydra_mcp/qgis_bridge.py:211-216`, `tools/hydra_mcp/qgis_bridge.py:249-257`). |
| Protocol-version handshake rejects skew | FAIL | `ping` returns `BRIDGE_VERSION` (`tools/hydra_mcp/qgis_bridge.py:268-275`), but `BridgeClient.ping()` merely returns the payload and never validates it (`tools/hydra_mcp/bridge_client.py:182-184`). No explicit mismatch error exists. |
| Single in-flight GUI request | PARTIAL | `_busy` is present (`tools/hydra_mcp/qgis_bridge.py:166`, `tools/hydra_mcp/qgis_bridge.py:277-304`), but synchronous handlers execute in the same Qt event-loop callback, so another `readyRead` callback normally cannot observe `_busy=True`; it queues until the first callback returns. Requests are serialized in practice, but the documented “concurrent requests receive an error” behavior is not demonstrated and the guard is ineffective for future async handlers. `ping` also bypasses it. |
| Bridge handlers execute on GUI thread | PASS by construction, unverified | `QLocalServer` is parented to `HydraMcpBridge`, and the bridge is constructed from the injected script's main path (`tools/hydra_mcp/qgis_bridge.py:149-165`, `tools/hydra_mcp/qgis_bridge.py:465-484`). There is no runtime thread assertion or real-QGIS test. |
| Offscreen mode | FAIL | Launch executable/readiness defects block it (`tools/hydra_mcp/tools_gui.py:142-186`, `tools/hydra_mcp/tools_gui.py:200-235`). Additionally, `--noplugins` prevents the HYDRA plugin/Studio from loading (`tools/hydra_mcp/tools_gui.py:175-186`). |
| Xvfb mode | FAIL | It inherits launch defects. It uses fixed display `:99`, skips starting Xvfb whenever any `DISPLAY` already exists, discards the process handle, and has no cleanup (`tools/hydra_mcp/tools_gui.py:164-173`, `tools/hydra_mcp/tools_gui.py:282-295`). |
| Display attach | PARTIAL | `mode="display"` only discovers a token (`tools/hydra_mcp/tools_gui.py:137-140`, `tools/hydra_mcp/tools_gui.py:312-330`). No plugin code checks `HYDRA_MCP_BRIDGE=1`; the only `HydraMcpBridge()` construction is the script main block (`tools/hydra_mcp/qgis_bridge.py:465-481`). Therefore the documented automatic attach path does not exist. |
| Plain QGIS unaffected unless opted in | PASS narrowly | No normal plugin startup hook exists, so plain use is unaffected. However, the corresponding opt-in behavior is also absent, making this only half of the intended requirement (`tools/hydra_mcp/qgis_bridge.py:8-12`, `tools/hydra_mcp/qgis_bridge.py:465-481`). |
| Long work is asynchronous | NOT APPLICABLE / NOT DONE | No simulation or long-running GUI handler is implemented; handlers are synchronous (`tools/hydra_mcp/qgis_bridge.py:285-306`). This becomes mandatory in Phase 3. |

## Safety Audit

| Requirement | Status | Evidence and assessment |
|---|---|---|
| Cryptographically random per-session token | PASS | Session/socket suffix uses `secrets.token_hex`, and the auth token uses `secrets.token_urlsafe(32)` (`tools/hydra_mcp/qgis_bridge.py:155-162`). |
| Token file is mode 0600 | PARTIAL / HIGH | `chmod(0600)` is applied only after `Path.write_text()` creates/truncates the file (`tools/hydra_mcp/qgis_bridge.py:190-199`). Under a permissive umask there is a race window with broader permissions; in `/tmp`, the predictable user/PID plus 32-bit suffix also permits a symlink/precreation race. Use atomic exclusive creation with mode `0o600`, then write and rename if needed. |
| Token validated on every message | PASS | Every message reads `params.token` and compares it before dispatch, including `ping` (`tools/hydra_mcp/qgis_bridge.py:249-275`). |
| Token/session cleanup | FAIL | No shutdown handler removes `token_path` or the local-server endpoint, and `gui_close` is absent (`tools/hydra_mcp/qgis_bridge.py:190-223`, `tools/hydra_mcp/tools_gui.py:239-250`). Discovery can accumulate and select stale files (`tools/hydra_mcp/bridge_client.py:110-129`). |
| Same-machine IPC, no TCP | PASS | Only QLocalSocket/QLocalServer are used (`tools/hydra_mcp/bridge_client.py:68-70`; `tools/hydra_mcp/qgis_bridge.py:164-165`). |
| Workspace-relative file paths | FAIL / HIGH | Phase 0 tools accept any existing absolute/symlinked path and perform no workspace containment check (`tools/hydra_mcp/tools_modeling.py:53-69`). `gui_launch` likewise accepts any project path (`tools/hydra_mcp/tools_gui.py:175-180`), and GUI tools accept arbitrary token-file paths (`tools/hydra_mcp/bridge_client.py:75-99`). This directly deviates from the plan's “all paths are workspace-relative” rule. |
| Path/URI robustness | PARTIAL | SQLite is opened read-only, which is good, but a raw user path is interpolated into a SQLite URI (`tools/hydra_mcp/tools_modeling.py:62-68`, `tools/hydra_mcp/tools_modeling.py:233-238`). Filenames containing URI metacharacters such as `?` or `#` can be misparsed. Resolve/contain paths first and use a safely encoded URI or a connection approach that preserves literal filenames. |
| Subprocess command injection | PASS | Both `Popen` calls pass argument lists and do not use `shell=True`; Python's default is `shell=False` (`tools/hydra_mcp/tools_gui.py:188-194`, `tools/hydra_mcp/tools_gui.py:282-289`). Project input is a separate argument, not shell text. |
| Process lifecycle safety | FAIL / HIGH | On timeout, QGIS is killed but not waited/reaped (`tools/hydra_mcp/tools_gui.py:229-235`). On success its handle is discarded; Xvfb's handle is always discarded (`tools/hydra_mcp/tools_gui.py:188-250`, `tools/hydra_mcp/tools_gui.py:282-295`). No `gui_close` exists. This can leave zombies/orphan processes and fixed-display conflicts. |
| `design_apply_patch` disabled by default | FAIL | `.kimi-code/mcp.json` has no `disabledTools` field (`.kimi-code/mcp.json:8-12`), despite the plan requiring `disabledTools: ["design_apply_patch"]`. This must be present before the design tool is added and should be added now to prevent an unsafe future default. |
| No credentials/secrets required | PASS for server-owned credentials | No external credentials are introduced. The bridge token is the only generated secret (`tools/hydra_mcp/qgis_bridge.py:159-162`). |
| No secret/error leakage | PARTIAL | No Python traceback is deliberately serialized; bridge exceptions become a message only (`tools/hydra_mcp/qgis_bridge.py:300-302`). However, premature QGIS exit returns the first 500 bytes of stdout and stderr to the MCP caller (`tools/hydra_mcp/tools_gui.py:207-214`), and broad exception strings include absolute paths/schema details (`tools/hydra_mcp/tools_modeling.py:250-251`, `tools/hydra_mcp/tools_modeling.py:469-472`). Redact known token values and sensitive environment/path content before returning subprocess logs. |
| Message-size/DoS protection | FAIL / HIGH | The four-byte length is accepted without a cap and incomplete bodies remain buffered indefinitely (`tools/hydra_mcp/bridge_client.py:28-42`; `tools/hydra_mcp/qgis_bridge.py:211-216`). A local unauthenticated peer can declare a huge frame before token validation and force unbounded buffering. Enforce a small maximum frame before buffering/JSON decode and close offenders. |
| Invalid-token connection handling | PARTIAL | Invalid tokens are rejected, but `disconnectFromServer()` is the client-side semantic on a server-accepted socket and may not force-close it as intended (`tools/hydra_mcp/qgis_bridge.py:259-263`). Use `abort()`/`close()` after flushing the error. |

## Code Quality Findings

### CRITICAL

1. **`gui_launch` invokes the wrong executable and cannot deliver Phase 2 sessions.**
   `_find_qgis_python()` returns a `python3` path, but `gui_launch()` passes QGIS command-line
   options to it (`tools/hydra_mcp/tools_gui.py:142-186`,
   `tools/hydra_mcp/tools_gui.py:253-278`). The method name and comments obscure the
   mismatch. **Fix:** locate and invoke the `qgis` executable directly, or launch a
   tested QGIS bootstrap module through the QGIS Python environment with appropriate
   initialization. Add a successful subprocess integration test.

2. **The documented MCP configuration cannot use Tier B.** The `uv` environment installs
   only `mcp` and `numpy` (`.kimi-code/mcp.json:8-11`), but `BridgeClient` refuses to
   instantiate without PyQt5/QtNetwork (`tools/hydra_mcp/bridge_client.py:45-67`).
   **Fix:** run the server with the QGIS environment's Python as required by the plan, or
   replace the client-side Qt dependency with a carefully tested native local-socket
   implementation that preserves the bridge protocol.

3. **Real screenshot encoding is likely nonfunctional.** `QPixmap.save` is called with
   `io.BytesIO`, while real PyQt expects a filename or `QIODevice`; only permissive mocks
   are tested (`tools/hydra_mcp/widget_screenshot.py:31-53`;
   `tests/test_hydra_mcp.py:1438-1469`). **Fix:** use `QByteArray` + `QBuffer`, verify the
   boolean return from `save()`, and test with a real `QPixmap` under a `QApplication`.
   Also implement the planned artifact output and `dialog|dock|canvas` targets.

### HIGH

4. **Bridge readiness detection is broken and risks pipe deadlock.** The child is started
   with `stdout=PIPE`, but code opens `/proc/<pid>/fd/1` to search for the ready line
   instead of consuming `proc.stdout` (`tools/hydra_mcp/tools_gui.py:188-219`,
   `tools/hydra_mcp/tools_gui.py:298-309`). The descriptor is the child's write end;
   opening/reading it is nonportable and generally fails. Unconsumed stdout/stderr can
   fill and block the child. **Fix:** consume both pipes asynchronously or write readiness
   to a known secure status file/socket; never poll `/proc` this way.

5. **No real Studio is launched.** The command uses `--noplugins`, while the bridge script
   itself only creates a tiny synthetic widget tree in its `__main__` path and does not
   instantiate the HYDRA Studio (`tools/hydra_mcp/tools_gui.py:175-186`;
   `tools/hydra_mcp/qgis_bridge.py:465-481`). **Fix:** load/initialize the HYDRA plugin
   explicitly in the bootstrap or allow the plugin and deterministically open Studio.

6. **User-supplied paths are not workspace-contained.** Absolute paths, symlinks, arbitrary
   QGIS projects, and arbitrary token JSON files are accepted (`tools/hydra_mcp/tools_modeling.py:53-69`;
   `tools/hydra_mcp/tools_gui.py:175-180`; `tools/hydra_mcp/bridge_client.py:75-99`).
   **Fix:** centralize canonical path resolution, reject escape through `..` or symlinks,
   and separate allowed read/write roots and artifact roots.

7. **Token-file creation is not atomically private.** Privacy is applied after writing
   (`tools/hydra_mcp/qgis_bridge.py:190-199`). **Fix:** create with `os.open(...,
   O_CREAT|O_EXCL|O_WRONLY, 0o600)`, avoid following symlinks, fsync as appropriate, and
   clean up on shutdown.

8. **The framing layer permits unauthenticated memory exhaustion.** Frame length has no
   maximum and buffering happens before auth (`tools/hydra_mcp/bridge_client.py:28-42`;
   `tools/hydra_mcp/qgis_bridge.py:211-216`). **Fix:** define named protocol limits,
   reject oversized frames immediately, cap per-socket buffers, and test boundary and
   malformed cases.

9. **Widget paths do not reliably describe hierarchy.** `findChildren(QWidget, "")` is
   recursive; using it at each segment lets a path jump levels and choose the first of
   duplicate descendant names. It also does not account for callers including the root
   name, as the README examples do (`swe2d/workbench/devtools/widget_walker.py:165-195`;
   `tools/hydra_mcp/README.md:99-119`). **Fix:** traverse direct QObject children only,
   define whether root is included, enforce unique matches per segment, and return an
   ambiguity error.

10. **The widget tree's parent/depth model is incorrect for real Qt trees.** The walker
    calls recursive `findChildren` and then recursively visits those results, so
    descendants can be recorded as direct root children depending on object-id ordering
    (`swe2d/workbench/devtools/widget_walker.py:139-152`). **Fix:** iterate direct child
    widgets only and add a real Qt nested-tree test asserting exact parent/depth values.

11. **Public GUI timeouts are silently ignored.** Every tool accepts `timeout`, but
    `_get_bridge_client` accepts only `token_path` and constructs `BridgeClient` with its
    default (`tools/hydra_mcp/tools_gui.py:58-62`,
    `tools/hydra_mcp/tools_gui.py:336-373`,
    `tools/hydra_mcp/tools_gui.py:388-414`,
    `tools/hydra_mcp/tools_gui.py:434-458`,
    `tools/hydra_mcp/tools_gui.py:478-504`,
    `tools/hydra_mcp/tools_gui.py:521-550`,
    `tools/hydra_mcp/tools_gui.py:567-599`). **Fix:** thread the argument into
    `BridgeClient(timeout=timeout)` and test non-default values.

12. **Process/session lifecycle is unmanaged.** QGIS and Xvfb handles are discarded,
    timeout kill is not followed by `wait`, and there is no close tool
    (`tools/hydra_mcp/tools_gui.py:188-250`, `tools/hydra_mcp/tools_gui.py:282-295`).
    **Fix:** maintain an explicit single-session object with QGIS/Xvfb handles, implement
    deterministic shutdown and token/socket cleanup, and reap all children.

13. **Plugin opt-in bridge startup is documented but absent.** The only bridge
    construction is the script's `__main__` block (`tools/hydra_mcp/qgis_bridge.py:465-481`),
    despite claims that `HYDRA_MCP_BRIDGE=1` auto-starts it
    (`tools/hydra_mcp/qgis_bridge.py:8-12`; `tools/hydra_mcp/tools_gui.py:7-16`).
    **Fix:** add an explicit, isolated plugin startup hook gated on exactly that variable,
    with cleanup on plugin unload; ensure normal startup remains unchanged.

### MEDIUM

14. **Type dispatch rejects subclasses and coercion is unsafe.** `type(widget).__name__`
    equality is used instead of the `isinstance` pattern used by the persistence service
    (`tools/hydra_mcp/qgis_bridge.py:91-146`;
    `swe2d/workbench/services/widget_persistence_service.py:125-137`). `bool("false")`
    becomes true. **Fix:** use Qt base-class `isinstance`, validate JSON types strictly,
    reject bool where numeric input is expected, and report allowed values/ranges.

15. **Protocol skew is advertised but not enforced.** The bridge returns version 2.C,
    but the client never compares it (`tools/hydra_mcp/qgis_bridge.py:47-47`,
    `tools/hydra_mcp/qgis_bridge.py:268-275`; `tools/hydra_mcp/bridge_client.py:182-184`).
    **Fix:** define a protocol constant shared by both sides and fail handshake before any
    non-ping call on mismatch.

16. **Malformed protocol input can escape the handler.** `decode_messages` may raise on
    invalid UTF-8/JSON or return a non-dict; `_on_ready_read` has no guard around decoding
    or `_handle_message`, which immediately calls `.get()` (`tools/hydra_mcp/bridge_client.py:28-42`;
    `tools/hydra_mcp/qgis_bridge.py:211-216`, `tools/hydra_mcp/qgis_bridge.py:249-253`).
    **Fix:** validate framing, JSON type, `jsonrpc`, method, params, and id, then return
    standard parse/request errors without crashing a Qt slot.

17. **The MCP adapter duplicates persistence schema knowledge.** Mesh, contents, and run
    log queries are written directly in the MCP layer (`tools/hydra_mcp/tools_modeling.py:72-183`),
    contrary to the README's assertion that every tool is a thin core adapter
    (`tools/hydra_mcp/README.md:8-12`). **Fix:** move reusable listing/join APIs into
    `swe2d.services`/`swe2d.results` and call them from MCP.

18. **`results_query` is narrower than the plan contract.** It summarizes only stored
    momentum/depth blobs and intentionally never returns arrays
    (`tools/hydra_mcp/server.py:72-90`; `tools/hydra_mcp/tools_modeling.py:363-466`).
    **Fix:** clarify the contract, add velocity and other supported derived fields in the
    core, and return bounded inline data or artifact references when arrays are requested.

19. **Single-flight behavior is misleading.** `_busy` suggests concurrent calls are
    rejected, but synchronous GUI-thread dispatch makes the guard effectively unreachable
    for separate ready-read events (`tools/hydra_mcp/qgis_bridge.py:277-304`).
    **Fix:** formally choose queueing or rejection, implement it at socket acceptance/read
    boundaries, and add a two-client delayed-handler test.

20. **Token/session discovery can select stale or wrong sessions.** The newest owned file
    is chosen with no PID liveness check, version check, explicit workspace identity, or
    successful-ping fallback to older candidates (`tools/hydra_mcp/bridge_client.py:101-129`).
    **Fix:** validate mode/version/PID, ping candidates, delete stale owned files, and
    prefer explicit session IDs over “newest file.”

21. **`_BridgeClient` caching creates hidden global state and test-order dependence.** The
    first class object is cached forever (`tools/hydra_mcp/tools_gui.py:55-62`). Tests
    monkeypatch `bridge_client.BridgeClient` but do not reset `_BridgeClient`
    (`tests/test_hydra_mcp.py:429-508`), so whichever test calls the helper first can
    poison later behavior. **Fix:** remove the unnecessary cache or provide dependency
    injection/reset fixtures.

22. **Broad exception handling hides programming errors and leaks raw messages.** Public
    tools repeatedly catch `Exception` and return `str(exc)` (`tools/hydra_mcp/tools_gui.py:381-382`,
    `tools/hydra_mcp/tools_gui.py:427-428`, `tools/hydra_mcp/tools_gui.py:471-472`,
    `tools/hydra_mcp/tools_gui.py:514-515`, `tools/hydra_mcp/tools_gui.py:560-561`,
    `tools/hydra_mcp/tools_gui.py:615-616`; `tools/hydra_mcp/tools_modeling.py:250-251`).
    **Fix:** catch expected transport/input/storage exceptions specifically, log internal
    details locally, and return stable redacted error codes/messages.

23. **Screenshot output violates the artifact design and can bloat MCP messages.** Base64
    image bytes are returned inline with no size cap (`tools/hydra_mcp/widget_screenshot.py:47-53`),
    whereas the plan requires a PNG artifact. **Fix:** write under a contained artifacts
    directory, return path/MIME/dimensions, and cap dimensions/file size.

24. **QGIS launch uses developer-specific hardcoded paths.** Two absolute installation
    paths, including `/home/aaron`, are embedded (`tools/hydra_mcp/tools_gui.py:253-267`).
    **Fix:** use `QGIS_EXECUTABLE`, documented environment discovery, and `PATH`; do not
    ship a developer-home fallback.

25. **Xvfb allocation is collision-prone and mode handling is incorrect.** Fixed `:99` is
    used, and if a real `DISPLAY` exists the requested Xvfb mode silently reuses it
    (`tools/hydra_mcp/tools_gui.py:164-173`, `tools/hydra_mcp/tools_gui.py:282-295`).
    **Fix:** use `xvfb-run -a` or allocate/lock a free display and always honor the mode.

26. **Token authentication bookkeeping is dead code.** `_authenticated` is initialized,
    set, and removed but never read (`tools/hydra_mcp/qgis_bridge.py:168-170`,
    `tools/hydra_mcp/qgis_bridge.py:205-221`, `tools/hydra_mcp/qgis_bridge.py:265-266`).
    `_socket_path` is likewise assigned but never used (`tools/hydra_mcp/bridge_client.py:69-70`,
    `tools/hydra_mcp/bridge_client.py:95-100`). **Fix:** remove both or use them to enforce
    a clearly documented connection/session state model.

27. **Bridge startup publishes credentials before listening succeeds.** The token file is
    written before `listen()` and remains if both listen attempts fail
    (`tools/hydra_mcp/qgis_bridge.py:172-188`). **Fix:** bind first, atomically publish
    readiness second, and clean up on every failure path.

### LOW

28. **Magic timing/size literals are scattered.** Poll interval `0.5`, stderr truncation
    `500`, fixed Xvfb startup sleep `1`, disconnect wait `1000`, socket polling `100`,
    JPEG quality `85`, and metadata/list caps are not collected into protocol/runtime
    constants (`tools/hydra_mcp/tools_gui.py:207-227`, `tools/hydra_mcp/tools_gui.py:282-292`;
    `tools/hydra_mcp/bridge_client.py:138-180`; `tools/hydra_mcp/widget_screenshot.py:39-44`;
    `tools/hydra_mcp/tools_modeling.py:36-43`, `tools/hydra_mcp/tools_modeling.py:186-208`).
    **Fix:** name externally meaningful limits and document why they are safe.

29. **Public typing is mostly good but incomplete/inconsistent.** `BridgeClient.__init__`
    lacks `-> None` (`tools/hydra_mcp/bridge_client.py:56-62`), and `find_widget()` declares
    an optional dict while simply returning `_call()`'s dict annotation
    (`tools/hydra_mcp/bridge_client.py:150-180`, `tools/hydra_mcp/bridge_client.py:190-192`).
    **Fix:** tighten response types with `TypedDict`/protocol models and annotate all
    public methods consistently.

30. **`format` shadows a built-in.** The screenshot handler and public functions use
    `format` as a local/parameter (`tools/hydra_mcp/qgis_bridge.py:388-404`;
    `tools/hydra_mcp/widget_screenshot.py:18-46`). **Fix:** rename internal variables to
    `image_format`/`fmt` while preserving MCP wire compatibility.

31. **Invalid-token close semantics are unclear.** `disconnectFromServer()` is used on a
    server-accepted socket (`tools/hydra_mcp/qgis_bridge.py:259-263`). **Fix:** use
    `abort()` or a documented flush-then-close sequence and test that the peer observes
    EOF.

### Docstrings, type hints, TODOs, duplication, and dead-code summary

- Public server tools and module-level public adapters generally have useful docstrings
  and argument/return annotations (`tools/hydra_mcp/server.py:42-270`;
  `tools/hydra_mcp/tools_gui.py:88-616`; `tools/hydra_mcp/tools_modeling.py:221-472`).
- No `TODO` or `FIXME` markers were found in `tools/hydra_mcp/*.py`.
- Error classification logic is duplicated in every GUI wrapper, including the same
  keyword checks for bridge availability (`tools/hydra_mcp/tools_gui.py:370-382`,
  `tools/hydra_mcp/tools_gui.py:411-428`, `tools/hydra_mcp/tools_gui.py:455-472`,
  `tools/hydra_mcp/tools_gui.py:501-515`, `tools/hydra_mcp/tools_gui.py:547-561`,
  `tools/hydra_mcp/tools_gui.py:596-616`). Extract one call/translation helper.
- Widget path resolution is duplicated across four bridge handlers
  (`tools/hydra_mcp/qgis_bridge.py:319-334`, `tools/hydra_mcp/qgis_bridge.py:353-386`,
  `tools/hydra_mcp/qgis_bridge.py:388-404`). Extract a liveness-checked
  `_resolve_widget(params)` helper.
- Dead fields `_authenticated` and `_socket_path` are detailed in finding 26.

## Test Quality Findings

The requested test command was run:

```text
PYTHONPATH="$PWD" python3 -m unittest -v tests.test_hydra_mcp
Ran 63 tests in 0.082s
OK (skipped=1)
```

The apparent green result is misleading: multiple daemon test threads emitted
`OSError: [Errno 9] Bad file descriptor` tracebacks during the run, and the only MCP
registration test was skipped because the `mcp` SDK was unavailable
(`tests/test_hydra_mcp.py:303-315`).

### HIGH

1. **No successful `gui_launch` test exists.** Tests cover only invalid mode and failed
   display discovery (`tests/test_hydra_mcp.py:462-479`). They do not validate the
   executable, `--code`, environment, project, ready signal, early exit, timeout, or
   cleanup paths. This allowed the wrong-executable and `/proc` readiness defects to
   ship. **Fix:** use a controlled fake executable for command/readiness/lifecycle tests,
   plus an environment-gated real QGIS offscreen smoke test.

2. **There is no real bridge round trip.** “Socket-pair” tests use Python
   `socket.socketpair()` and handwritten mock clients/servers, not `QLocalSocket`,
   `QLocalServer`, `HydraMcpBridge`, or actual bridge handlers
   (`tests/test_hydra_mcp.py:516-675`, `tests/test_hydra_mcp.py:677-927`,
   `tests/test_hydra_mcp.py:1257-1401`). **Fix:** instantiate the real bridge under a Qt
   event loop and drive it through the real `BridgeClient`; add an offscreen QGIS layer
   above that.

3. **Security/protocol failure modes are untested.** There are no tests for invalid or
   missing token, token validation on every message, token-file mode, symlink/exclusive
   creation, oversized frames, invalid lengths, malformed UTF-8/JSON, non-object JSON,
   unknown methods, bad params, ID correlation, protocol mismatch, or two-client
   single-flight (`tests/test_hydra_mcp.py:346-426` covers only framing happy/partial
   cases and token-file loading). **Fix:** add negative wire tests before calling the
   bridge safe.

4. **Screenshot tests mock the API inaccurately.** `MockPixmap.save()` accepts a Python
   `BytesIO` by construction (`tests/test_hydra_mcp.py:1438-1469`), so it cannot detect
   incompatibility with real `QPixmap.save`. It also always returns success. **Fix:** add
   a real `QPixmap` + `QBuffer` test and explicit encode-failure tests.

5. **Thread teardown is noisy and nondeterministic.** Daemon server threads are not
   retained/joined; `tearDown` closes sockets while they are blocked in `recv`
   (`tests/test_hydra_mcp.py:523-528`, `tests/test_hydra_mcp.py:557-597`,
   `tests/test_hydra_mcp.py:1260-1265`, `tests/test_hydra_mcp.py:1320-1350`). The reviewed
   run produced multiple `Bad file descriptor` tracebacks despite `OK`. **Fix:** use
   stop events/context managers, catch expected shutdown errors, join every thread, and
   fail the test on background exceptions.

6. **MCP tool-surface verification can silently skip.** The only registration contract
   test calls `skipTest` when `mcp` is missing (`tests/test_hydra_mcp.py:303-315`).
   **Fix:** make `mcp` a test dependency in the relevant job and fail, not skip, in the
   MCP-specific suite. Also compare catalog metadata/signatures, not names only.

### MEDIUM

7. **Mocks do not enforce real auth or bridge behavior.** Handwritten socket servers
   accept any token and never exercise bridge dispatch/error payloads
   (`tests/test_hydra_mcp.py:615-635`, `tests/test_hydra_mcp.py:731-751`,
   `tests/test_hydra_mcp.py:1288-1308`). **Fix:** route test messages through
   `HydraMcpBridge._handle_message` or, preferably, its real local socket.

8. **Tests are order-dependent through global cache/monkeypatching.** Tests directly
   replace `bridge_client.BridgeClient`, while `_get_bridge_client` caches the first
   class and is not reset (`tests/test_hydra_mcp.py:429-508`;
   `tools/hydra_mcp/tools_gui.py:55-62`). **Fix:** use `unittest.mock.patch` and cleanup,
   remove the cache, and run randomized/reversed test order.

9. **Timeout behavior is not tested.** Although every GUI tool advertises a timeout,
   tests' fake clients hardcode 5-second socket waits and never assert forwarding
   (`tests/test_hydra_mcp.py:599-635`, `tests/test_hydra_mcp.py:713-767`). **Fix:** capture
   constructor arguments and test short timeout/error behavior.

10. **Widget mocks do not match Qt's recursive `findChildren` semantics.** `_MockWidget`
    returns only direct children (`tests/test_hydra_mcp.py:982-1003`), while real
    `QWidget.findChildren` defaults to recursive search. This masks the hierarchy/path
    bugs in the shared walker. **Fix:** test with real Qt objects or faithfully model
    recursion and direct-child options.

11. **No liveness/destruction race tests exist.** The plan specifically calls out SIP
    wrapper lifetime risk, but tests never delete a QObject between discovery and access.
    The bridge's `RuntimeError` catches are only inspected, not exercised
    (`tools/hydra_mcp/qgis_bridge.py:68-76`, `tools/hydra_mcp/qgis_bridge.py:91-146`).
    **Fix:** create/delete Qt widgets and assert structured errors without crashes.

12. **Coverage does not include token discovery edge cases.** Only explicit token-file
    loading and Qt absence are tested (`tests/test_hydra_mcp.py:396-426`). There are no
    tests for env pairs, one missing env value, owner filtering, stale files, malformed
    files, candidate fallback, or version/PID validation. **Fix:** isolate runtime dirs
    and cover all discovery branches.

13. **Phase 0 tests couple to hand-built internal schemas.** The run-log table is created
    directly in the fixture (`tests/test_hydra_mcp.py:76-103`). This can let both adapter
    and test agree on stale schema assumptions. **Fix:** expose/use the core writer or a
    canonical fixture generated by production code.

### LOW

14. **Duplicated test protocol clients obscure coverage.** Nearly identical mock client,
    framing, and server loops are repeated across test classes
    (`tests/test_hydra_mcp.py:599-648`, `tests/test_hydra_mcp.py:713-767`,
    `tests/test_hydra_mcp.py:1275-1318`). **Fix:** use one strict harness or the real
    client/bridge.

15. **Unused/misplaced test code reduces clarity.** `_send_request` helpers are defined
    but not used (`tests/test_hydra_mcp.py:538-555`, `tests/test_hydra_mcp.py:695-711`),
    and the widget-walker test appears inside `TestBridgeClientValueMethods` after a
    standalone string literal rather than in its own class
    (`tests/test_hydra_mcp.py:1006-1082`). **Fix:** delete dead helpers and place tests in
    focused classes.

### Tool coverage gaps in tests

- No tests exist for the 19 unimplemented Tier A tools, the seven unimplemented planned
  Tier B tools, or any Tier C tool; this follows implementation status.
- Of the implemented tools, `model_inspect`, `run_list`, and `results_query` have useful
  fixture coverage (`tests/test_hydra_mcp.py:108-300`).
- `gui_launch` has no success-path test (`tests/test_hydra_mcp.py:462-479`).
- `gui_widget_tree`, `gui_find_widget`, `gui_get_value`, `gui_set_value`,
  `gui_find_widget_by_path`, and `gui_screenshot` are covered only through handwritten
  fake socket peers, not the real bridge (`tests/test_hydra_mcp.py:516-927`,
  `tests/test_hydra_mcp.py:1257-1401`).
- No test covers bridge GUI-thread affinity, actual Qt widget subclasses, token auth,
  single-flight, protocol errors, or cleanup.
- There are no external network calls and no explicit sleeps in this test module; the
  primary flakiness comes from daemon threads, fixed five-second socket timeouts, global
  monkeypatch state, and unjoined teardown (`tests/test_hydra_mcp.py:523-528`,
  `tests/test_hydra_mcp.py:557-597`).

## Documentation Audit

### Accurate or useful content

- README correctly identifies the MCP SDK and stdio transport
  (`tools/hydra_mcp/README.md:1-6`).
- Phase 0 tool arguments and summary behavior are documented in detail
  (`tools/hydra_mcp/README.md:14-56`).
- The three session modes, token discovery, nominal 0600 token, and intended
  single-flight behavior are described (`tools/hydra_mcp/README.md:159-220`).
- The `uv --no-project` rationale is clearly explained
  (`tools/hydra_mcp/README.md:251-272`).

### Inaccuracies and omissions

1. **The README claims all tools are thin core adapters, but Phase 0 contains direct SQL
   mirrors** (`tools/hydra_mcp/README.md:8-12` versus
   `tools/hydra_mcp/tools_modeling.py:72-183`).
2. **It documents offscreen/Xvfb launch as working**, while the implementation invokes
   Python with QGIS CLI flags and uses broken readiness detection
   (`tools/hydra_mcp/README.md:165-220` versus
   `tools/hydra_mcp/tools_gui.py:142-235`).
3. **It documents display opt-in through `HYDRA_MCP_BRIDGE=1`, but no plugin hook exists**
   (`tools/hydra_mcp/README.md:176-178`;
   `tools/hydra_mcp/qgis_bridge.py:465-481`).
4. **It says the configured server can be used for GUI tools**, but the `uv` command does
   not provide PyQt5/QtNetwork (`tools/hydra_mcp/README.md:251-272`;
   `.kimi-code/mcp.json:8-11`; `tools/hydra_mcp/bridge_client.py:45-67`).
5. **Screenshot documentation claims base64 success without warning that it is unbounded
   and differs from the planned artifact contract** (`tools/hydra_mcp/README.md:133-157`).
6. **Session safety is incomplete.** There is no workspace-relative path policy,
   subprocess/log redaction policy, token cleanup behavior, session cleanup procedure,
   or design-patch approval/default-disable guidance (`tools/hydra_mcp/README.md:159-190`,
   `tools/hydra_mcp/README.md:251-287`).
7. **The README is internally inconsistent about display mode.** It presents display as
   a supported `gui_launch` mode (`tools/hydra_mcp/README.md:192-220`) and later says
   display attach is a later phase (`tools/hydra_mcp/README.md:286-287`).
8. **The layout is stale.** It omits `widget_screenshot.py` and describes
   `tools_modeling.py` as all Tier A build/run/results despite containing only Phase 0
   reads (`tools/hydra_mcp/README.md:274-284`).
9. **The existing `MCP_PHASE2_REVIEW.md` is materially over-optimistic.** It declares all
   delivered phases complete and approved (`tools/hydra_mcp/MCP_PHASE2_REVIEW.md:1-11`,
   `tools/hydra_mcp/MCP_PHASE2_REVIEW.md:180-182`) despite not detecting the broken
   executable/readiness path, missing plugin opt-in, incompatible MCP environment,
   invalid real screenshot buffer, ignored timeouts, or absent real bridge tests. Its
   claim that socket-pair tests are real full wire integration is misleading because
   they use handwritten peers (`tools/hydra_mcp/MCP_PHASE2_REVIEW.md:67-92`;
   `tests/test_hydra_mcp.py:516-675`).
10. **User-guide distribution is incomplete.** The plan requires documentation in
    `docs/USER_GUIDE.md`; this implementation documents only the tool-local README, and
    no MCP references were found in project docs during this review
    (`tools/hydra_mcp/README.md:251-287`).

## Cross-Phase Dependencies

### Before Tier A production can ship

1. Complete the CLI-first single builder/spec/executor core and expose stable core APIs
   for model creation, mesh generation/baking, terrain, forcings, drainage, structures,
   async jobs, batches, and result exports/renders. The current MCP module only has the
   Phase 0 reads (`tools/hydra_mcp/server.py:42-90`).
2. Move direct GeoPackage schema queries out of MCP and into canonical core services
   (`tools/hydra_mcp/tools_modeling.py:72-183`).
3. Implement and register the remaining 19 Tier A tools, then add contract tests that
   map every advertised tool to a real core entry point. The current name-only test
   expects only ten total tools (`tests/test_hydra_mcp.py:303-331`).
4. Establish workspace-relative path and artifact-root enforcement before adding any
   mutating or export tool (`tools/hydra_mcp/tools_modeling.py:53-69`).
5. Define bounded data/artifact semantics for result arrays, time series, rendering,
   exports, and comparisons; current `results_query` returns statistics only
   (`tools/hydra_mcp/tools_modeling.py:363-466`).
6. Add async job persistence, cancellation, restart/error semantics, GPU concurrency/MPS
   behavior, and negative tests before exposing execution tools.

### Before Tier B basic (Phase 2) can ship

1. Fix QGIS executable discovery and readiness IPC; launch the actual QGIS application
   (`tools/hydra_mcp/tools_gui.py:142-235`).
2. Make the MCP runtime capable of `QLocalSocket` use, either by using QGIS Python or by
   removing the client-side Qt dependency (`.kimi-code/mcp.json:8-11`;
   `tools/hydra_mcp/bridge_client.py:45-67`).
3. Deterministically initialize HYDRA Studio instead of launching with `--noplugins`
   (`tools/hydra_mcp/tools_gui.py:175-186`).
4. Correct widget tree/path traversal and subclass-safe value access
   (`swe2d/workbench/devtools/widget_walker.py:139-195`;
   `tools/hydra_mcp/qgis_bridge.py:91-146`).
5. Replace screenshot encoding with real Qt `QBuffer`, implement planned targets and
   artifact output, and cap output size (`tools/hydra_mcp/widget_screenshot.py:31-53`).
6. Forward timeouts, enforce protocol limits/version checks, atomically protect tokens,
   and add real QLocalSocket/QGIS tests.
7. Implement a session manager and cleanup path even if the public `gui_close` tool is
   formally Phase 3 (`tools/hydra_mcp/tools_gui.py:188-250`).

### Before Tier B advanced behavioral tools can ship

1. Complete and stabilize basic Phase 2 first; click/key/action/log/run cannot be safely
   layered on a bridge that cannot launch or resolve widgets reliably.
2. Add `gui_click`, `gui_key`, `gui_run_action`, `gui_read_log`,
   `gui_run_simulation`, `gui_save_project`, and `gui_close`; no corresponding handlers
   currently exist (`tools/hydra_mcp/qgis_bridge.py:287-299`).
3. Design async Qt-signal completion for simulations so the GUI thread remains responsive;
   current dispatch is synchronous (`tools/hydra_mcp/qgis_bridge.py:285-306`).
4. Add `tests/test_gui_behavioral_mcp.py` and a reliable Xvfb CI gate after process cleanup,
   artifact retention, and timeout semantics are deterministic.

### Before Tier C can ship

1. Add `tools_design.py` as thin adapters over existing devtools; none exists in the
   current server imports (`tools/hydra_mcp/server.py:30-37`).
2. Implement preview-before-apply, source containment, patch validation, uniqueness
   checks, stale-source detection, and auditable structured results.
3. Add `disabledTools: ["design_apply_patch"]` now, before registration
   (`.kimi-code/mcp.json:8-12`).
4. Ensure apply is client-approved and constrained to the active workspace; current path
   policy is unrestricted (`tools/hydra_mcp/tools_modeling.py:53-69`).
5. Test in a disposable worktree with malicious paths/diffs, symlinks, stale patches,
   syntax errors, and rollback behavior.

## Verdict

# BLOCKED

Phase 0's three read-only tools are useful and reasonably tested, but the delivered
Phase 2 stack is not operationally complete. The two most fundamental paths are broken:
`gui_launch` invokes Python rather than QGIS, and the project MCP configuration lacks
QtNetwork needed by `BridgeClient`. Real screenshot encoding is also incompatible with
the Qt API, no real bridge/QGIS test exists, and token/path/protocol safety has important
gaps. These are release blockers, not cosmetic review comments.

The final single fully featured server is also far from catalog completion, as expected
for Phase 0–2: 19 Tier A, seven planned Tier B, and all four Tier C tools remain
unimplemented. That future-phase incompleteness is not the reason for the **BLOCKED**
verdict; the verdict is driven by defects in the phases claimed complete.

## Prioritized Fix List

1. **CRITICAL — Invoke actual QGIS and replace `/proc` readiness polling**
   (`tools/hydra_mcp/tools_gui.py:142-235`, `tools/hydra_mcp/tools_gui.py:253-309`).
2. **CRITICAL — Make the configured MCP runtime support QtNetwork/BridgeClient**
   (`.kimi-code/mcp.json:8-11`; `tools/hydra_mcp/bridge_client.py:45-67`).
3. **CRITICAL — Replace `BytesIO` screenshot encoding with `QBuffer`, check save success,
   and implement bounded artifact output** (`tools/hydra_mcp/widget_screenshot.py:31-53`).
4. **HIGH — Launch/initialize HYDRA Studio; remove or compensate for `--noplugins`**
   (`tools/hydra_mcp/tools_gui.py:175-186`; `tools/hydra_mcp/qgis_bridge.py:465-481`).
5. **HIGH — Add real QLocalSocket bridge and offscreen-QGIS integration tests, including
   successful `gui_launch`** (`tests/test_hydra_mcp.py:462-479`,
   `tests/test_hydra_mcp.py:516-927`).
6. **HIGH — Enforce workspace-relative canonical path containment for GPKG, project,
   token, artifact, export, and future patch paths** (`tools/hydra_mcp/tools_modeling.py:53-69`;
   `tools/hydra_mcp/tools_gui.py:175-180`).
7. **HIGH — Create token files atomically at 0600 and clean them up**
   (`tools/hydra_mcp/qgis_bridge.py:190-199`).
8. **HIGH — Add maximum frame/buffer sizes before unauthenticated buffering**
   (`tools/hydra_mcp/bridge_client.py:28-42`; `tools/hydra_mcp/qgis_bridge.py:211-216`).
9. **HIGH — Fix direct-child widget traversal, root-path semantics, duplicate ambiguity,
   and tree parent/depth correctness** (`swe2d/workbench/devtools/widget_walker.py:139-195`).
10. **HIGH — Forward every advertised timeout to `BridgeClient`**
    (`tools/hydra_mcp/tools_gui.py:58-62`, `tools/hydra_mcp/tools_gui.py:336-616`).
11. **HIGH — Add deterministic QGIS/Xvfb/session shutdown and process reaping**
    (`tools/hydra_mcp/tools_gui.py:188-250`, `tools/hydra_mcp/tools_gui.py:282-295`).
12. **HIGH — Implement the `HYDRA_MCP_BRIDGE=1` plugin startup/unload hook**
    (`tools/hydra_mcp/qgis_bridge.py:8-12`, `tools/hydra_mcp/qgis_bridge.py:465-481`).
13. **HIGH — Add `disabledTools: ["design_apply_patch"]` to project config**
    (`.kimi-code/mcp.json:8-12`).
14. **HIGH — Make test thread failures fail tests; stop and join every fake server thread**
    (`tests/test_hydra_mcp.py:523-597`, `tests/test_hydra_mcp.py:1260-1390`).
15. **MEDIUM — Validate malformed JSON-RPC, IDs, params, unknown methods, invalid tokens,
    and protocol versions with stable redacted errors** (`tools/hydra_mcp/qgis_bridge.py:211-304`).
16. **MEDIUM — Use `isinstance` widget dispatch and strict JSON type/range validation**
    (`tools/hydra_mcp/qgis_bridge.py:91-146`).
17. **MEDIUM — Remove MCP-layer schema duplication by adding canonical core listing APIs**
    (`tools/hydra_mcp/tools_modeling.py:72-183`).
18. **MEDIUM — Align `results_query` with the planned depth/velocity/array contract or
    revise the catalog explicitly** (`tools/hydra_mcp/tools_modeling.py:363-466`).
19. **MEDIUM — Define and test actual queue/reject single-flight semantics with two clients**
    (`tools/hydra_mcp/qgis_bridge.py:277-304`).
20. **MEDIUM — Harden token discovery against stale/wrong/version-mismatched sessions**
    (`tools/hydra_mcp/bridge_client.py:101-129`).
21. **MEDIUM — Remove `_BridgeClient` global caching and duplicated GUI error translation**
    (`tools/hydra_mcp/tools_gui.py:55-82`, `tools/hydra_mcp/tools_gui.py:370-616`).
22. **MEDIUM — Replace broad exception-to-wire strings with specific exceptions, stable
    codes, local logging, and redaction** (`tools/hydra_mcp/tools_gui.py:329-330`,
    `tools/hydra_mcp/tools_modeling.py:250-251`).
23. **MEDIUM — Remove developer-home QGIS paths and correctly allocate Xvfb displays**
    (`tools/hydra_mcp/tools_gui.py:253-295`).
24. **MEDIUM — Publish token/readiness only after the socket listens successfully**
    (`tools/hydra_mcp/qgis_bridge.py:172-199`).
25. **MEDIUM — Make MCP registration tests mandatory with the SDK installed and assert
    schemas/signatures, not only names** (`tests/test_hydra_mcp.py:303-331`).
26. **MEDIUM — Expand token discovery, QObject liveness, real subclass, timeout, and
    negative protocol tests** (`tests/test_hydra_mcp.py:396-426`,
    `tests/test_hydra_mcp.py:929-1003`).
27. **LOW — Remove dead `_authenticated`, `_socket_path`, and unused test helpers**
    (`tools/hydra_mcp/qgis_bridge.py:168-170`; `tools/hydra_mcp/bridge_client.py:69-100`;
    `tests/test_hydra_mcp.py:538-555`).
28. **LOW — Name protocol/runtime limits and remove built-in shadowing**
    (`tools/hydra_mcp/bridge_client.py:138-180`; `tools/hydra_mcp/qgis_bridge.py:388-404`).
29. **LOW — Tighten public response typing and constructor annotations**
    (`tools/hydra_mcp/bridge_client.py:56-62`, `tools/hydra_mcp/bridge_client.py:150-192`).
30. **LOW — Correct README claims, safety guidance, layout, session support, and user-guide
    integration; supersede the existing optimistic review** (`tools/hydra_mcp/README.md:1-287`;
    `tools/hydra_mcp/MCP_PHASE2_REVIEW.md:180-182`).
31. **FUTURE PHASE — Implement the remaining Tier A, advanced Tier B, and Tier C catalog
    only after the Phase 2 foundation and safety gates above pass**
    (`tests/test_hydra_mcp.py:317-330`; `tools/hydra_mcp/qgis_bridge.py:287-299`).
