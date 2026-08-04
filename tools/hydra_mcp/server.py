"""HYDRA MCP server — stdio transport (39 tools, Phases 0 + 1 + 2 + 3 + 4).

Tool surface by tier (read-only vs mutating matters for client approval
rules — see plan §6 and tools/hydra_mcp/README.md):

- Tier A, Phase 0 (read-only): model_inspect, run_list, results_query.
- Tier A, Phase 1 (mixed): model_create, mesh_generate, mesh_bake,
  terrain_assign, bc_configure, rainfall_configure, drainage_configure,
  structures_configure, run_start, run_cancel, run_batch and results_export
  / results_render mutate model files or spawn solver jobs; spec_build,
  spec_validate, spec_diff, results_timeseries, results_compare are
  read-only.
- Tier B, Phases 2.A–2.C (GUI introspection): gui_launch, gui_widget_tree,
  gui_find_widget, gui_find_widget_by_path, gui_get_value, gui_screenshot
  are read-only; gui_set_value mutates the live GUI.
- Tier B, Phase 3 (behavioral, mutating): gui_click, gui_key,
  gui_run_action, gui_run_simulation, gui_close; gui_read_log is read-only.
- Tier C, Phase 4 (design): design_rename_widget, design_relabel_widget and
  design_preview_patch only propose patches; design_apply_patch edits the
  source tree and is disabled by default via ``disabledTools``.

See ``docs/HYDRA_MCP_SERVER_PLAN.md`` for the full plan; design principle: thin
adapters over existing core modules, no re-implemented modeling logic.

Run directly (stdio):

    python tools/hydra_mcp/server.py

Or via an MCP client config (see tools/hydra_mcp/README.md):

    uv run --with mcp python tools/hydra_mcp/server.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

# Make the repo importable regardless of the caller's PYTHONPATH.
# tools/hydra_mcp/server.py -> repo root is two levels up.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Allow the native CUDA extension to be located via env var (worktree builds).
_HYDRA_BUILD_DIR = os.environ.get("HYDRA_BUILD_DIR")
if _HYDRA_BUILD_DIR and str(_HYDRA_BUILD_DIR) not in sys.path:
    sys.path.insert(0, str(_HYDRA_BUILD_DIR))
else:
    _DEFAULT_BUILD = _REPO_ROOT / "build"
    if _DEFAULT_BUILD.exists() and str(_DEFAULT_BUILD) not in sys.path:
        sys.path.insert(0, str(_DEFAULT_BUILD))

from mcp.server.fastmcp import FastMCP

if __package__:
    from . import tools_modeling as _tools_modeling
    from . import tools_modeling_phase1 as _tools_modeling_phase1
    from . import tools_gui as _tools_gui, tools_audit as _tools_audit
    from . import tools_design as _tools_design
    from . import tools_gpu_viewer as _tools_gpu_viewer
else:  # run as a plain script: tools/hydra_mcp is already sys.path[0]
    import tools_modeling as _tools_modeling
    import tools_modeling_phase1 as _tools_modeling_phase1
    import tools_gui as _tools_gui
    import tools_audit as _tools_audit
    import tools_design as _tools_design
    import tools_gpu_viewer as _tools_gpu_viewer

mcp = FastMCP("hydra")

# GPU Direct Viewer tools are hidden by default — see
# docs/specs/2026-07-29-gpu-viewer-hide-and-user-guide-coverage.md.  Set
# HYDRA_MCP_ENABLE_GPU_VIEWER=1 to re-register them.
_GPU_VIEWER_ENABLED = str(os.environ.get("HYDRA_MCP_ENABLE_GPU_VIEWER", "")).strip().lower() in (
    "1", "true", "yes", "on",
)


@mcp.tool()
def model_inspect(gpkg_path: str) -> dict:
    """Inspect a HYDRA model GeoPackage: list baked meshes, layers/tables,
    saved simulation configs, and simulation runs.

    Args:
        gpkg_path: Path to the HYDRA model/results GeoPackage file.

    Returns:
        Dict with ok, meshes, layers, simulation_configs, runs — or a
        structured error (ok=False) describing what is wrong.
    """
    return _tools_modeling.model_inspect(gpkg_path)


@mcp.tool()
def run_list(gpkg_path: str) -> dict:
    """List simulation runs in a HYDRA results GeoPackage: run id, mesh,
    cell/timestep counts, created timestamp, wallclock duration, and a
    config summary when available.

    Args:
        gpkg_path: Path to the results GeoPackage file.

    Returns:
        Dict with ok, n_runs, runs — or a structured error (ok=False).
    """
    return _tools_modeling.run_list(gpkg_path)


@mcp.tool()
def results_query(gpkg_path: str, run_id: str, field: str,
                  timestep: float | None = None) -> dict:
    """Summarize a result field for a run (never returns raw arrays).

    Args:
        gpkg_path: Path to the results GeoPackage file.
        run_id: Run identifier (see run_list).
        field: One of h, hu, hv (per-timestep snapshots) or max_h, max_hu,
            max_hv (per-cell max tracking; timestep ignored).
        timestep: Optional simulation time in seconds; the nearest stored
            snapshot is used. Omit to summarize all timesteps at once.

    Returns:
        Dict with ok, summary (shape, dtype, min/max/mean, NaN count) and
        available timesteps — or a structured error (ok=False) listing the
        valid run ids / fields / timesteps.
    """
    return _tools_modeling.results_query(gpkg_path, run_id, field, timestep)


# ── Phase 1: Production modeling tools ──────────────────────────────────────


@mcp.tool()
def model_create(gpkg_path: str, crs: str) -> dict:
    """Create a new empty HYDRA model GeoPackage.

    Args:
        gpkg_path: Workspace-relative path for the new GeoPackage.
        crs: CRS identifier, e.g. "EPSG:4326" or a WKT string.

    Returns:
        ``{"ok": true, gpkg_path, crs}`` or a structured error.
    """
    return _tools_modeling_phase1.model_create(gpkg_path, crs)


@mcp.tool()
def mesh_generate(domain: dict, spacing: float, backend: str = "builtin") -> dict:
    """Generate a simple structured rectangular mesh.

    Args:
        domain: Dict with xmin, ymin, xmax, ymax.
        spacing: Target cell edge length.
        backend: "builtin" (default) or "gmsh".  Gmsh is not yet wired in this
            fallback helper; it returns a clear error if requested.

    Returns:
        ``{"ok": true, mesh: {...}, n_nodes, n_cells, backend}`` or error.
    """
    return _tools_modeling_phase1.mesh_generate(domain, spacing, backend)


@mcp.tool()
def mesh_bake(gpkg_path: str, mesh_name: str, mesh_data: dict, crs_wkt: str = "") -> dict:
    """Persist a generated mesh into a HYDRA model GeoPackage.

    Args:
        gpkg_path: Path to the model GeoPackage.
        mesh_name: Unique name for the baked mesh.
        mesh_data: Mesh dict as returned by ``mesh_generate``.
        crs_wkt: Optional CRS WKT string.

    Returns:
        ``{"ok": true, gpkg_path, mesh_name, n_cells}`` or error.
    """
    return _tools_modeling_phase1.mesh_bake(gpkg_path, mesh_name, mesh_data, crs_wkt)


@mcp.tool()
def terrain_assign(gpkg_path: str, mesh_name: str, source: dict, method: str = "raster") -> dict:
    """Sample a terrain source onto a baked mesh's nodes and re-bake it.

    Args:
        gpkg_path: Path to the model GeoPackage.
        mesh_name: Name of the baked mesh to modify.
        source: Dict describing the terrain source.  Supported forms:
            - {"type": "raster", "data": hex-encoded bytes, "shape": [rows, cols],
               "geo_transform": [origin_x, pixel_width, 0, origin_y, 0, pixel_height]}
            - {"type": "points", "x": [...], "y": [...], "z": [...]}
        method: "raster" (direct sampling) or "idw" (inverse-distance weighted).

    Returns:
        ``{"ok": true, n_nodes_updated}`` or error.
    """
    return _tools_modeling_phase1.terrain_assign(gpkg_path, mesh_name, source, method)


@mcp.tool()
def bc_configure(gpkg_path: str, mesh_name: str, bc_config: list) -> dict:
    """Store boundary-condition configuration for a mesh."""
    return _tools_modeling_phase1.bc_configure(gpkg_path, mesh_name, bc_config)


@mcp.tool()
def rainfall_configure(gpkg_path: str, mesh_name: str, rainfall_config: dict) -> dict:
    """Store rainfall/hyetograph configuration for a mesh."""
    return _tools_modeling_phase1.rainfall_configure(gpkg_path, mesh_name, rainfall_config)


@mcp.tool()
def drainage_configure(gpkg_path: str, mesh_name: str, drainage_config: dict) -> dict:
    """Store drainage-network configuration for a mesh."""
    return _tools_modeling_phase1.drainage_configure(gpkg_path, mesh_name, drainage_config)


@mcp.tool()
def structures_configure(gpkg_path: str, mesh_name: str, structures_config: list) -> dict:
    """Store hydraulic-structure configuration for a mesh."""
    return _tools_modeling_phase1.structures_configure(gpkg_path, mesh_name, structures_config)


@mcp.tool()
def spec_build(gpkg_path: str, mesh_name: str, run_params: dict = None, results_gpkg_path: str = None) -> dict:
    """Build a canonical swe2d-run/2 spec from a model GeoPackage."""
    return _tools_modeling_phase1.spec_build(gpkg_path, mesh_name, run_params, results_gpkg_path)


@mcp.tool()
def spec_validate(spec: dict) -> dict:
    """Validate a swe2d-run/2 spec using the canonical builder."""
    return _tools_modeling_phase1.spec_validate(spec)


@mcp.tool()
def spec_diff(spec_a: dict, spec_b: dict) -> dict:
    """Return a recursive diff of two specs."""
    return _tools_modeling_phase1.spec_diff(spec_a, spec_b)


@mcp.tool()
def run_start(spec: dict, job_name: str = None) -> dict:
    """Start an async simulation run from a swe2d-run/2 spec dict."""
    return _tools_modeling_phase1.run_start(spec, job_name)


@mcp.tool()
def run_status(job_id: str) -> dict:
    """Return the status of a running or completed job."""
    return _tools_modeling_phase1.run_status(job_id)


@mcp.tool()
def run_cancel(job_id: str) -> dict:
    """Cancel a running job."""
    return _tools_modeling_phase1.run_cancel(job_id)


@mcp.tool()
def run_batch(batch_spec: dict, max_workers: int = 0) -> dict:
    """Run a batch of simulations."""
    return _tools_modeling_phase1.run_batch(batch_spec, max_workers)


@mcp.tool()
def results_timeseries(gpkg_path: str, run_id: str, line_id: int) -> dict:
    """Load a line timeseries for a run."""
    return _tools_modeling_phase1.results_timeseries(gpkg_path, run_id, line_id)


@mcp.tool()
def results_export(gpkg_path: str, run_id: str, out_path: str, format: str = "csv") -> dict:
    """Export a run result to a simple CSV summary."""
    return _tools_modeling_phase1.results_export(gpkg_path, run_id, out_path, format)


@mcp.tool()
def results_render(gpkg_path: str, run_id: str, field: str, timestep: float = None, out_path: str = None) -> dict:
    """Render a simple 2-D field plot as a PNG artifact."""
    return _tools_modeling_phase1.results_render(gpkg_path, run_id, field, timestep, out_path)


@mcp.tool()
def results_compare(gpkg_path: str, run_a: str, run_b: str, field: str, tolerance: float = 1e-6) -> dict:
    """Compare a result field between two runs."""
    return _tools_modeling_phase1.results_compare(gpkg_path, run_a, run_b, field, tolerance)


# ── Phase 2.A: Live GUI tools ───────────────────────────────────────────────


@mcp.tool()
def gui_launch(
    mode: str = "xvfb",
    project: str | None = None,
    timeout: float = 60.0,
) -> dict:
    """Launch a QGIS instance with the HYDRA MCP bridge injected.

    Starts a new QGIS process (Xvfb, offscreen, or display mode) with the
    bridge script loaded, waits for the bridge to write its token file, and
    returns session metadata.  Pass the returned ``token_path`` to
    ``gui_widget_tree`` / ``gui_find_widget`` to drive the live session.

    Args:
        mode: ``"xvfb"`` (default; Xvfb virtual display — preferred: the
            ``offscreen`` QPA is flaky with NVIDIA GL, crashing QGIS at
            boot), ``"offscreen"`` (QT_QPA_PLATFORM=offscreen, only when
            Xvfb is unavailable), or ``"display"`` (poll for a bridge token
            file from a running QGIS; set ``HYDRA_MCP_BRIDGE=1`` at QGIS
            launch or inject the bridge via the Python console bootstrap).
        project: Optional path to a ``.qgs`` / ``.qgz`` QGIS project to open.
        timeout: Seconds to wait for the bridge token file to appear.  In
            ``display`` mode this also bounds how long the server polls for
            an existing bridge session before giving up.

    Returns:
        ``{"ok": true, session_id, socket_name, token_path, mode, pid}``
        or ``{"ok": false, "error": ...}``.
    """
    return _tools_gui.gui_launch(mode=mode, project=project, timeout=timeout)


@mcp.tool()
def gui_widget_tree(
    root: str | None = None,
    token_path: str | None = None,
    timeout: float = 30.0,
) -> dict:
    """Return the live widget tree from the active QGIS session.

    Returns a flat depth-first list of widget nodes.  Each node is a dict::

        {"object_name": "...", "class_name": "...", "widget_id": ...,
         "parent_id": ..., "text": "...", "depth": ...}

    Args:
        root: Optional ``objectName`` of the widget to use as the tree root.
            Omit to auto-detect (prefers active window).
        token_path: Path to the bridge token file.  Omit to auto-discover.
        timeout: Seconds to wait for a response from the bridge.

    Returns:
        ``{"ok": true, "nodes": [...], "root_object_name": ...}`` or
        ``{"ok": false, "error": ...}`` when no bridge session is active.
    """
    return _tools_gui.gui_widget_tree(root=root, token_path=token_path, timeout=timeout)


@mcp.tool()
def gui_find_widget(
    name: str,
    token_path: str | None = None,
    timeout: float = 30.0,
) -> dict:
    """Return the single widget node whose ``objectName`` matches *name*.

    Searches across all top-level widgets in the live QGIS session.

    Args:
        name: Exact ``objectName`` string to search for.
        token_path: Path to the bridge token file.  Omit to auto-discover.
        timeout: Seconds to wait for a response from the bridge.

    Returns:
        ``{"ok": true, "node": {...}}`` or
        ``{"ok": false, "error": "widget not found"}``.
    """
    return _tools_gui.gui_find_widget(name=name, token_path=token_path, timeout=timeout)


@mcp.tool()
def gui_find_widget_by_path(
    path: str,
    token_path: str | None = None,
    timeout: float = 30.0,
) -> dict:
    """Return the widget at a dot-separated path and its key properties.

    Args:
        path: Dot-separated ``objectName`` path from the root widget,
            e.g. ``"central_container.simulation_tab.cfl_spin"``.
        token_path: Path to the bridge token file.  Omit to auto-discover.
        timeout: Seconds to wait for a response from the bridge.

    Returns:
        ``{"ok": true, "widget": {"object_name", "class_name", "widget_id",
        "geometry": {"x", "y", "width", "height"}, "is_visible"}}``
        or ``{"ok": false, "error": "..."}``.
    """
    return _tools_gui.gui_find_widget_by_path(
        path=path, token_path=token_path, timeout=timeout
    )


@mcp.tool()
def gui_get_value(
    path: str,
    token_path: str | None = None,
    timeout: float = 30.0,
) -> dict:
    """Read the current value of the widget at *path*.

    Supports: QSpinBox (int), QDoubleSpinBox (float), QCheckBox (bool),
    QComboBox (str), QLineEdit (str), QTextEdit (str), QLabel (str).

    Args:
        path: Dot-separated ``objectName`` path, e.g.
            ``"studio_window.simulation_tab.run_duration"``.
        token_path: Path to the bridge token file.  Omit to auto-discover.
        timeout: Seconds to wait for a response from the bridge.

    Returns:
        ``{"ok": true, "type": "QDoubleSpinBox", "value": 3600.0}``
        or ``{"ok": false, "error": "..."}``.
    """
    return _tools_gui.gui_get_value(path=path, token_path=token_path, timeout=timeout)


@mcp.tool()
def gui_set_value(
    path: str,
    value: float | int | bool | str,
    token_path: str | None = None,
    timeout: float = 30.0,
) -> dict:
    """Set the value of the widget at *path*.

    Supports: QSpinBox (int), QDoubleSpinBox (float), QCheckBox (bool),
    QComboBox (str — matches ``currentText``), QLineEdit (str),
    QTextEdit (str).

    Args:
        path: Dot-separated ``objectName`` path, e.g.
            ``"studio_window.simulation_tab.cfl_spin"``.
        value: The new value. Type must match the widget class.
        token_path: Path to the bridge token file.  Omit to auto-discover.
        timeout: Seconds to wait for a response from the bridge.

    Returns:
        ``{"ok": true}`` on success, or ``{"ok": false, "error": "..."}``.
    """
    return _tools_gui.gui_set_value(
        path=path, value=value, token_path=token_path, timeout=timeout
    )


@mcp.tool()
def gui_screenshot(
    path: str | None = None,
    format: str = "png",
    target: str = "dialog",
    token_path: str | None = None,
    timeout: float = 30.0,
) -> dict:
    """Capture a screenshot of a widget in the live QGIS session.

    Args:
        path: Dot-separated ``objectName`` path, e.g.
            ``"studio_window.simulation_tab"``.  Optional when *target*
            is given; mutually compatible (target wins when both are set).
        format: Image format — ``"png"`` (default) or ``"jpg"`` / ``"jpeg"``.
            JPEG uses quality=85.
        target: Which top-level widget to screenshot — one of:
            - ``"dialog"`` (default): the active QGIS main window.
            - ``"dock"``: the first ``QDockWidget`` found.
            - ``"canvas"``: the QGIS map canvas (``QgsMapCanvas``).
        token_path: Path to the bridge token file.  Omit to auto-discover.
        timeout: Seconds to wait for a response from the bridge.

    At least one of ``path`` or ``target`` must be specified.  The
    underlying tool rejects the call with ``"target is required…"`` if
    both are missing.

    Returns:
        ``{"ok": true, "image_b64": "...", "format": "png", "width": 800,
        "height": 600}`` on success, or ``{"ok": false, "error": "..."}``.
    """
    return _tools_gui.gui_screenshot(
        path=path, format=format, target=target,
        token_path=token_path, timeout=timeout,
    )


@mcp.tool()
def gui_dump_dock(
    object_name: str,
    token_path: str | None = None,
    timeout: float = 30.0,
) -> dict:
    """Return a bounded widget subtree rooted at a named dock."""
    return _tools_audit.gui_dump_dock(
        object_name=object_name, token_path=token_path, timeout=timeout
    )


@mcp.tool()
def gui_screenshot_path(
    out_path: str,
    path: str | None = None,
    format: str = "png",
    target: str | None = None,
    token_path: str | None = None,
    timeout: float = 30.0,
) -> dict:
    """Capture a screenshot and write it to a workspace-contained path."""
    return _tools_audit.gui_screenshot_path(
        path=path, out_path=out_path, format=format, target=target,
        token_path=token_path, timeout=timeout,
    )


@mcp.tool()
def gui_describe_widget(
    path: str,
    token_path: str | None = None,
    timeout: float = 30.0,
) -> dict:
    """Return complete bridge-side metadata for a widget path."""
    return _tools_audit.gui_describe_widget(
        path=path, token_path=token_path, timeout=timeout
    )


# ── Phase 3: Behavioral GUI tools ─────────────────────────────────────────────


@mcp.tool()
def gui_click(
    path: Optional[str] = None,
    object_name: Optional[str] = None,
    token_path: Optional[str] = None,
    timeout: float = 30.0,
) -> dict:
    """Click a live Qt widget using QTest.

    Provide either a dot-separated *path* of ``objectName`` values or an
    explicit *object_name* to search for recursively.

    Args:
        path: Dot-separated objectName path (e.g. ``"dialog.run_btn"``).
        object_name: Exact objectName to search recursively.
        token_path: Path to the bridge token file.  Omit to auto-discover.
        timeout: Seconds to wait for a response from the bridge.

    Returns:
        ``{"ok": true}`` or ``{"ok": false, "error": ...}``.
    """
    return _tools_gui.gui_click(
        path=path, object_name=object_name,
        token_path=token_path, timeout=timeout,
    )


@mcp.tool()
def gui_key(
    key: str,
    path: Optional[str] = None,
    object_name: Optional[str] = None,
    token_path: Optional[str] = None,
    timeout: float = 30.0,
) -> dict:
    """Send a key press to a live Qt widget using QTest.

    Args:
        key: Key name (``"return"``, ``"delete"``, ``"a"``, etc.).
        path: Dot-separated objectName path.
        object_name: Exact objectName to search recursively.
        token_path: Path to the bridge token file.  Omit to auto-discover.
        timeout: Seconds to wait for a response from the bridge.

    Returns:
        ``{"ok": true}`` or ``{"ok": false, "error": ...}``.
    """
    return _tools_gui.gui_key(
        key=key, path=path, object_name=object_name,
        token_path=token_path, timeout=timeout,
    )


@mcp.tool()
def gui_run_action(
    object_name: Optional[str] = None,
    text: Optional[str] = None,
    token_path: Optional[str] = None,
    timeout: float = 30.0,
) -> dict:
    """Trigger a QAction by objectName or text.

    Args:
        object_name: Exact objectName of the QAction.
        text: Exact menu/toolbar text of the QAction.
        token_path: Path to the bridge token file.  Omit to auto-discover.
        timeout: Seconds to wait for a response from the bridge.

    Returns:
        ``{"ok": true}`` or ``{"ok": false, "error": ...}``.
    """
    return _tools_gui.gui_run_action(
        object_name=object_name, text=text,
        token_path=token_path, timeout=timeout,
    )


@mcp.tool()
def gui_read_log(
    max_lines: int = 1000,
    token_path: Optional[str] = None,
    timeout: float = 30.0,
) -> dict:
    """Read the active HYDRA workbench runtime log.

    Args:
        max_lines: Maximum recent log lines to return.
        token_path: Path to the bridge token file.  Omit to auto-discover.
        timeout: Seconds to wait for a response from the bridge.

    Returns:
        ``{"ok": true, "lines": [...], "total": N}`` or error.
    """
    return _tools_gui.gui_read_log(
        max_lines=max_lines, token_path=token_path, timeout=timeout,
    )


@mcp.tool()
def gui_run_simulation(
    run_duration_text: Optional[str] = None,
    output_interval_text: Optional[str] = None,
    timeout: float = 60.0,
    startup_timeout: float = 10.0,
    token_path: Optional[str] = None,
) -> dict:
    """Set run inputs, click the Run button, and await compute_finished.

    Args:
        run_duration_text: Optional duration text (e.g. ``"1:00"``) to set
            before clicking Run.
        output_interval_text: Optional output interval text.
        timeout: Seconds to wait for the simulation to finish after starting.
        startup_timeout: Seconds to wait for the worker thread to appear.
        token_path: Path to the bridge token file.  Omit to auto-discover.

    Returns:
        ``{"ok": true, "status": "finished"|"timeout"|"failed", ...}``
        or error.
    """
    return _tools_gui.gui_run_simulation(
        run_duration_text=run_duration_text,
        output_interval_text=output_interval_text,
        timeout=timeout,
        startup_timeout=startup_timeout,
        token_path=token_path,
    )


@mcp.tool()
def gui_close(
    token_path: Optional[str] = None,
    timeout: float = 10.0,
) -> dict:
    """Shut down a QGIS session launched by ``gui_launch``.

    Sends SIGTERM first, then escalates to SIGKILL if the process has not
    exited by the end of *timeout* seconds.

    Args:
        token_path: Token path returned by ``gui_launch``.  Omit to terminate
            the most recently launched session.
        timeout: Seconds to wait before escalating to SIGKILL.

    Returns:
        ``{"ok": true, "pid": ..., "action": "terminated"|"killed"}`` or
        error.
    """
    return _tools_gui.gui_close(token_path=token_path, timeout=timeout)


# ── Phase 4: Designer / structural source-editing tools ─────────────────────


@mcp.tool()
def design_rename_widget(old: str, new: str) -> dict:
    """Propose a patch that renames a widget ``objectName``.

    The new name is validated for uniqueness against all existing
    ``setObjectName`` calls in the workbench view files.  On success a
    unified diff is returned for review; apply it with ``design_apply_patch``.

    Args:
        old: Current ``objectName`` string to replace.
        new: Desired new ``objectName`` string.

    Returns:
        ``{"ok": true, patch_text, edits, file_path}`` or
        ``{"ok": false, error}``.
    """
    return _tools_design.design_rename_widget(old=old, new=new)


@mcp.tool()
def design_relabel_widget(name: str, text: str) -> dict:
    """Propose a patch that relabels a widget title/label.

    Matches ``QGroupBox("name")``, ``toolbox.addItem(page, "name")``,
    ``_add_param_row(..., "name", ...)`` and ``addRow("name", widget)``.

    Args:
        name: Current label/title string to replace.
        text: New label/title string.

    Returns:
        ``{"ok": true, patch_text, edits, file_path}`` or
        ``{"ok": false, error}``.
    """
    return _tools_design.design_relabel_widget(name=name, text=text)


@mcp.tool()
def design_preview_patch(edits: list) -> dict:
    """Preview a unified diff for a list of proposed edits.

    Each edit is a dict with ``kind``, ``file_path``, ``lineno``,
    ``old_value`` and ``new_value``.  The file is not modified.

    Args:
        edits: Non-empty list of edit dicts.

    Returns:
        ``{"ok": true, patch_text, edits}`` or ``{"ok": false, error}``.
    """
    return _tools_design.design_preview_patch(edits=edits)


@mcp.tool()
def design_apply_patch(diff: str) -> dict:
    """Apply a patch previously returned by ``design_preview_patch``.

    This tool writes to the source files under ``swe2d/workbench/views``.
    It is gated in the project dev MCP config via ``disabledTools`` so it
    requires explicit approval before it can be invoked.

    Args:
        diff: The JSON response string from ``design_preview_patch``.

    Returns:
        ``{"ok": true, files, edit_count}`` or ``{"ok": false, error}``.
    """
    return _tools_design.design_apply_patch(diff=diff)


# ── GPU Direct Viewer (Phase 1 of docs/plans/2026-07-26-gpu-direct-viewer.md) ──

if _GPU_VIEWER_ENABLED:

    @mcp.tool()
    def gpu_viewer_open(token_path: str | None = None) -> dict:
        """Open the standalone GPUViewerDialog on top of the studio window.

        The dialog uses a snapshot reader that pulls from the GPU device ring
        buffer.  If no run is in progress, the dialog shows "Waiting for first
        snapshot…" until a run starts.

        Args:
            token_path: Optional path to the bridge token file.  When omitted,
                auto-discovers the active session.

        Returns:
            ``{"ok": true}`` or a structured error.
        """
        return _tools_gpu_viewer.gpu_viewer_open(token_path=token_path)

    @mcp.tool()
    def gpu_viewer_set_field(field: str, token_path: str | None = None) -> dict:
        """Change the field on the open GPUViewerDialog.

        Args:
            field: One of ``'depth'`` or ``'speed'``.
            token_path: Optional bridge token path.

        Returns:
            ``{"ok": true, "field": "..."}`` or a structured error.
        """
        return _tools_gpu_viewer.gpu_viewer_set_field(
            field=field, token_path=token_path
        )

    @mcp.tool()
    def gpu_viewer_read_snapshot(token_path: str | None = None) -> dict:
        """Read the latest live snapshot from the open viewer.

        Returns ``{ok, t_s, n_cells, h_b64, hu_b64, hv_b64}``.  Arrays are
        base64-encoded ``float64`` bytes — decode with
        ``base64.b64decode(snap['h_b64'])`` then
        ``np.frombuffer(..., dtype=np.float64)``.

        Args:
            token_path: Optional bridge token path.

        Returns:
            Snapshot dict or a structured error.
        """
        return _tools_gpu_viewer.gpu_viewer_read_snapshot(token_path=token_path)

    @mcp.tool()
    def gpu_viewer_screenshot(
        out_path: str,
        format: str = "png",
        token_path: str | None = None,
    ) -> dict:
        """Screenshot the open GPUViewerDialog to *out_path*.

        Args:
            out_path: Destination file path (parent dirs are created).
            format: Image format (``'png'`` or ``'jpg'``).
            token_path: Optional bridge token path.

        Returns:
            ``{"ok": true, "out_path": "..."}`` or a structured error.
        """
        return _tools_gpu_viewer.gpu_viewer_screenshot(
            out_path=out_path, format=format, token_path=token_path,
        )


if __name__ == "__main__":
    mcp.run()  # stdio transport by default
