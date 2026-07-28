"""Read-only modeling tools for the HYDRA MCP server (Phase 0).

Thin adapters over the existing HYDRA core modules — primarily
``swe2d.services.gpkg_persistence_service`` (pure sqlite3 model/results
GeoPackage I/O).  Direct sqlite3 queries are used only for listings the
existing functions do not cover (mesh listing, layer listing, run-log join).

Note on imports: ``swe2d.services.gpkg_persistence_service`` is importable in
a bare Python env (numpy only).  ``swe2d.results.*`` is NOT — its package
``__init__`` pulls in the native ``hydra_overlay`` module — so run metadata
comes from ``collect_baked_runs_from_gpkg`` (the same data path that
``swe2d.results.queries.discover_line_result_runs`` and
``swe2d.results.run_service.collect_runs_from_gpkg`` delegate to) and the
``swe2d_run_logs`` table is read directly, mirroring
``swe2d.results.run_log_storage.load_run_logs_from_geopackage``.

Every public function returns a JSON-serializable dict and never raises:
errors are returned as ``{"ok": False, "error": ..., ...}`` with actionable
context (valid run ids / fields / timesteps) so the MCP client can recover.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Make the repo importable regardless of the caller's PYTHONPATH.
# tools/hydra_mcp/tools_modeling.py -> repo root is two levels up.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.hydra_mcp.workspace import WorkspacePath, WorkspacePathError, default_workspace

# Fields stored per-timestep in swe2d_baked_results (<field>_blob columns).
_SNAPSHOT_FIELDS = ("h", "hu", "hv")
# Optional per-cell GPU max-tracking fields (single array per run).
_MAX_FIELDS = ("max_h", "max_hu", "max_hv")
ALL_FIELDS = _SNAPSHOT_FIELDS + _MAX_FIELDS

# Cap on how many timestep values are echoed back in a single response.
_MAX_LISTED_TIMESTEPS = 64


def _err(message: str, **context: Any) -> Dict[str, Any]:
    """Build a structured, actionable error payload."""
    out: Dict[str, Any] = {"ok": False, "error": message}
    out.update(context)
    return out


def _validate_gpkg(gpkg_path: str) -> Optional[Dict[str, Any]]:
    """Return an error dict if *gpkg_path* is not a readable SQLite file.

    The path must resolve inside the MCP server's workspace root; escapes
    through ``..`` or symlinks targeting outside the workspace are
    rejected before any sqlite3 connection is opened.
    """
    if not gpkg_path or not str(gpkg_path).strip():
        return _err("gpkg_path is empty; provide the path to a HYDRA GeoPackage file.")
    try:
        contained = default_workspace().resolve_under(gpkg_path)
    except WorkspacePathError as exc:
        return _err(str(exc))
    gpkg_path = str(contained)
    if not os.path.exists(gpkg_path):
        return _err(f"File not found: {gpkg_path}")
    if not os.path.isfile(gpkg_path):
        return _err(f"Not a file: {gpkg_path}")
    try:
        conn = sqlite3.connect(f"file:{gpkg_path}?mode=ro", uri=True)
        try:
            conn.execute("SELECT 1 FROM sqlite_master LIMIT 1").fetchone()
        finally:
            conn.close()
    except sqlite3.Error as exc:
        return _err(f"Not a readable SQLite/GeoPackage file: {gpkg_path} ({exc})")
    return None


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


def _list_meshes(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """List baked meshes.  No existing listing function covers this table."""
    if not _table_exists(conn, "swe2d_baked_mesh"):
        return []
    rows = conn.execute(
        "SELECT mesh_name, n_nodes, n_cells, n_edges, crs_wkt, created_utc "
        "FROM swe2d_baked_mesh ORDER BY created_utc DESC"
    ).fetchall()
    return [
        {
            "mesh_name": str(r[0]),
            "n_nodes": int(r[1]),
            "n_cells": int(r[2]),
            "n_edges": int(r[3]),
            "crs_wkt": str(r[4] or ""),
            "created_utc": str(r[5] or ""),
        }
        for r in rows
    ]


def _list_layers(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """List layers/tables.  Prefers OGC gpkg_contents; falls back to sqlite_master."""
    if _table_exists(conn, "gpkg_contents"):
        rows = conn.execute(
            "SELECT table_name, data_type, COALESCE(identifier, ''), "
            "COALESCE(description, ''), srs_id FROM gpkg_contents ORDER BY table_name"
        ).fetchall()
        return [
            {
                "table_name": str(r[0]),
                "data_type": str(r[1]),
                "identifier": str(r[2]),
                "description": str(r[3]),
                "srs_id": int(r[4]) if r[4] is not None else None,
            }
            for r in rows
        ]
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    return [
        {"table_name": str(r[0]), "data_type": "unknown",
         "identifier": "", "description": "", "srs_id": None}
        for r in rows
    ]


def _summarize_configs(gpkg_path: str) -> List[Dict[str, Any]]:
    """Wrap load_simulation_configs; drop the bulky widget_state payload."""
    from swe2d.services.gpkg_persistence_service import load_simulation_configs

    out: List[Dict[str, Any]] = []
    for cfg in load_simulation_configs(gpkg_path):
        params = cfg.get("params")
        out.append({
            "config_id": str(cfg.get("config_id", "")),
            "mesh_name": str(cfg.get("mesh_name", "")),
            "created_utc": str(cfg.get("created_utc", "")),
            "run_duration_s": float(cfg.get("run_duration_s") or 0.0),
            "description": str(cfg.get("description", "")),
            "params": params if isinstance(params, dict) else {},
        })
    return out


def _load_run_logs(conn: sqlite3.Connection) -> Dict[str, Dict[str, Any]]:
    """Read swe2d_run_logs keyed by run_id (mirrors run_log_storage loader).

    Returns per-run wallclock timestamps, duration, and a small config
    summary extracted from metadata_json.  ``log_text`` is intentionally
    not returned (too large for a listing).
    """
    if not _table_exists(conn, "swe2d_run_logs"):
        return {}
    cols = {str(r[1]) for r in conn.execute("PRAGMA table_info(swe2d_run_logs)")}
    has_meta = "metadata_json" in cols
    select = (
        "SELECT run_id, created_utc, start_wallclock, end_wallclock, duration_s"
        + (", metadata_json" if has_meta else "")
        + " FROM swe2d_run_logs ORDER BY created_utc DESC"
    )
    out: Dict[str, Dict[str, Any]] = {}
    for row in conn.execute(select).fetchall():
        run_id = str(row[0] or "")
        if not run_id:
            continue
        metadata: Dict[str, Any] = {}
        if has_meta and row[5]:
            try:
                parsed = json.loads(str(row[5]))
                if isinstance(parsed, dict):
                    metadata = parsed
            except (ValueError, TypeError):
                pass
        config_summary = _summarize_run_metadata(metadata)
        out[run_id] = {
            "log_created_utc": str(row[1] or ""),
            "start_wallclock": str(row[2] or ""),
            "end_wallclock": str(row[3] or ""),
            "duration_s": float(row[4] or 0.0),
            "config_summary": config_summary,
        }
    return out


_MAX_SCALAR_STR_LEN = 120


def _summarize_run_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Truthful, bloat-free summary of a run-log metadata dict.

    Real metadata writers produce either ``{}`` (CLI/headless runs —
    ``swe2d/cli/headless_executor.py`` ``collect_run_log_metadata``) or
    ``{"workbench_widget_state": {...}}`` (workbench runs —
    ``swe2d/workbench/controllers/finalization_adapter.py``), so there is no
    fixed key schema to whitelist.  Instead report the top-level keys
    present, plus any small scalar values (bool/int/float, short strings).
    Bulky nested payloads (dicts, lists, long strings) are reported by key
    only — never echoed.
    """
    summary: Dict[str, Any] = {"metadata_keys": sorted(str(k) for k in metadata)}
    scalars: Dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, bool):
            scalars[str(key)] = value
        elif isinstance(value, (int, float)):
            scalars[str(key)] = value
        elif isinstance(value, str) and len(value) <= _MAX_SCALAR_STR_LEN:
            scalars[str(key)] = value
    if scalars:
        summary["scalars"] = scalars
    return summary


def _available_run_ids(gpkg_path: str) -> List[str]:
    from swe2d.services.gpkg_persistence_service import collect_baked_runs_from_gpkg

    return [str(r["run_id"]) for r in collect_baked_runs_from_gpkg(gpkg_path)]


def model_inspect(gpkg_path: str) -> Dict[str, Any]:
    """List meshes, layers/tables, saved simulation configs, and runs in a
    HYDRA model GeoPackage.

    Wraps swe2d.services.gpkg_persistence_service (load_simulation_configs,
    collect_baked_runs_from_gpkg); mesh and layer listings are direct
    sqlite3 reads of swe2d_baked_mesh / gpkg_contents.
    """
    bad = _validate_gpkg(gpkg_path)
    if bad is not None:
        return bad
    try:
        conn = sqlite3.connect(f"file:{gpkg_path}?mode=ro", uri=True)
        try:
            meshes = _list_meshes(conn)
            layers = _list_layers(conn)
        finally:
            conn.close()
        from swe2d.services.gpkg_persistence_service import (
            collect_baked_runs_from_gpkg,
        )
        return {
            "ok": True,
            "gpkg_path": str(gpkg_path),
            "meshes": meshes,
            "layers": layers,
            "simulation_configs": _summarize_configs(gpkg_path),
            "runs": collect_baked_runs_from_gpkg(gpkg_path),
        }
    except Exception as exc:  # never leak a traceback to the MCP client
        return _err(f"model_inspect failed for {gpkg_path}: {type(exc).__name__}: {exc}")


def run_list(gpkg_path: str) -> Dict[str, Any]:
    """List simulation runs in a results GeoPackage.

    Base run metadata (run id, mesh, cell/timestep counts, created_utc)
    comes from collect_baked_runs_from_gpkg — the same data path as
    swe2d.results.run_service.collect_runs_from_gpkg.  Each run is joined
    with its swe2d_run_logs entry (wallclock start/end, duration, config
    summary) when present.
    """
    bad = _validate_gpkg(gpkg_path)
    if bad is not None:
        return bad
    try:
        from swe2d.services.gpkg_persistence_service import (
            collect_baked_runs_from_gpkg,
        )

        runs = collect_baked_runs_from_gpkg(gpkg_path)
        conn = sqlite3.connect(f"file:{gpkg_path}?mode=ro", uri=True)
        try:
            logs = _load_run_logs(conn)
        finally:
            conn.close()

        entries: List[Dict[str, Any]] = []
        seen = set()
        for r in runs:
            run_id = str(r["run_id"])
            seen.add(run_id)
            entry: Dict[str, Any] = {
                "run_id": run_id,
                "mesh_name": str(r.get("mesh_name", "")),
                "n_cells": int(r.get("n_cells", 0)),
                "n_timesteps": int(r.get("n_timesteps", 0)),
                "created_utc": str(r.get("created_utc", "")),
                "has_lines": bool(r.get("has_lines", False)),
                "has_coupling": bool(r.get("has_coupling", False)),
            }
            entry.update(logs.get(run_id, {}))
            entries.append(entry)
        # Runs that only have a run-log record (no baked results stored).
        for run_id, log in logs.items():
            if run_id in seen:
                continue
            entry = {
                "run_id": run_id,
                "mesh_name": str(
                    log.get("config_summary", {}).get("scalars", {}).get("mesh_name", "")
                ),
                "n_cells": None,
                "n_timesteps": 0,
                "created_utc": str(log.get("log_created_utc", "")),
                "has_lines": False,
                "has_coupling": False,
                "results_stored": False,
            }
            entry.update(log)
            entries.append(entry)

        return {
            "ok": True,
            "gpkg_path": str(gpkg_path),
            "n_runs": len(entries),
            "runs": entries,
        }
    except Exception as exc:
        return _err(f"run_list failed for {gpkg_path}: {type(exc).__name__}: {exc}")


def _summarize_array(arr: Any) -> Dict[str, Any]:
    """Agent-friendly summary of a numpy array (never the raw data)."""
    import numpy as np

    a = np.asarray(arr)
    nan_count = int(np.isnan(a).sum()) if a.size else 0
    summary: Dict[str, Any] = {
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "size": int(a.size),
        "nan_count": nan_count,
    }
    if a.size and nan_count < a.size:
        summary["min"] = float(np.nanmin(a))
        summary["max"] = float(np.nanmax(a))
        summary["mean"] = float(np.nanmean(a))
    else:
        summary["min"] = summary["max"] = summary["mean"] = None
    return summary


def _timestep_listing(times: Any) -> Dict[str, Any]:
    """Return timestep info, truncating long lists."""
    import numpy as np

    t = np.asarray(times, dtype=np.float64)
    n = int(t.size)
    out: Dict[str, Any] = {"n_timesteps": n}
    if n <= _MAX_LISTED_TIMESTEPS:
        out["timesteps"] = [float(x) for x in t]
    else:
        out["timesteps_first"] = [float(x) for x in t[:5]]
        out["timesteps_last"] = [float(x) for x in t[-5:]]
        out["note"] = (
            f"{n} timesteps; only first/last 5 listed. "
            "Pass timestep=<value> to query a specific snapshot."
        )
    return out


def results_query(
    gpkg_path: str,
    run_id: str,
    field: str,
    timestep: Optional[float] = None,
) -> Dict[str, Any]:
    """Summarize a result field for a run in a baked results GeoPackage.

    Returns array statistics (shape, dtype, min/max/mean, NaN count) and the
    available timesteps — never raw megabyte arrays.  ``field`` is one of
    h, hu, hv (per-timestep snapshots) or max_h, max_hu, max_hv (per-cell
    GPU max tracking; timestep is ignored for these).  ``timestep`` selects
    the nearest snapshot; omit it to summarize the whole
    (n_timesteps, n_cells) array.
    """
    bad = _validate_gpkg(gpkg_path)
    if bad is not None:
        return bad
    try:
        import numpy as np

        conn = sqlite3.connect(f"file:{gpkg_path}?mode=ro", uri=True)
        try:
            if not _table_exists(conn, "swe2d_baked_results"):
                return _err(
                    f"No baked results table (swe2d_baked_results) in {gpkg_path}; "
                    "this file has no stored simulation runs.",
                )
            row = conn.execute(
                "SELECT n_timesteps, n_cells, times_blob, h_blob, hu_blob, hv_blob, "
                "max_h_blob, max_hu_blob, max_hv_blob "
                "FROM swe2d_baked_results WHERE run_id=?",
                (str(run_id),),
            ).fetchone()
            if row is None:
                return _err(
                    f"Run '{run_id}' not found in {gpkg_path}.",
                    available_run_ids=_available_run_ids(gpkg_path),
                )

            field = str(field or "").strip()
            n_steps, n_cells = int(row[0]), int(row[1])
            blob_idx = {"h": 3, "hu": 4, "hv": 5,
                        "max_h": 6, "max_hu": 7, "max_hv": 8}
            stored = {name for name, idx in blob_idx.items() if row[idx] is not None}
            if field not in ALL_FIELDS:
                return _err(
                    f"Unknown field '{field}'.",
                    available_fields=[f for f in ALL_FIELDS if f in stored],
                    all_supported_fields=list(ALL_FIELDS),
                )
            if field not in stored:
                return _err(
                    f"Field '{field}' was not stored for run '{run_id}' "
                    "(max_* fields require max tracking enabled at run time).",
                    available_fields=[f for f in ALL_FIELDS if f in stored],
                )

            arr = np.frombuffer(row[blob_idx[field]], dtype=np.float64)
            result: Dict[str, Any] = {
                "ok": True,
                "gpkg_path": str(gpkg_path),
                "run_id": str(run_id),
                "field": field,
                "n_cells": n_cells,
            }

            if field in _MAX_FIELDS:
                result["kind"] = "max_tracking"
                result["summary"] = _summarize_array(arr)
                return result

            times = np.frombuffer(row[2], dtype=np.float64)
            result["kind"] = "snapshot_timeseries"
            result.update(_timestep_listing(times))
            data = arr.reshape(n_steps, n_cells)

            if timestep is None:
                result["summary"] = _summarize_array(data)
                return result

            try:
                t_req = float(timestep)
            except (TypeError, ValueError):
                return _err(
                    f"Invalid timestep {timestep!r}: not a number. "
                    "Pass a simulation time in seconds (nearest stored "
                    "snapshot is used), or omit timestep to summarize all "
                    "snapshots at once.",
                    **_timestep_listing(times),
                )
            if not np.isfinite(t_req):
                return _err(
                    f"Invalid timestep {timestep!r}: NaN/inf is not a valid "
                    "simulation time.",
                    **_timestep_listing(times),
                )

            i = int(np.argmin(np.abs(times - t_req)))
            result["requested_timestep"] = t_req
            result["actual_timestep"] = float(times[i])
            result["timestep_index"] = i
            result["summary"] = _summarize_array(data[i])
            return result
        finally:
            conn.close()
    except Exception as exc:
        return _err(
            f"results_query failed for {gpkg_path}: {type(exc).__name__}: {exc}"
        )
