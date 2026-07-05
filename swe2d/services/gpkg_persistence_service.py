"""Pure-Python GeoPackage persistence service for the workbench.

Consolidates all GeoPackage/SQLite persistence logic previously scattered
across ``results_persistence_service``, ``studio_results_panel``, and inline
SQL in the dialog. This service is pure Python — it does not touch Qt.

Functions that need dialog access (e.g. ``current_line_results_storage_path``)
take the dialog as their first parameter.

"""
from __future__ import annotations

import datetime
import json
import logging
import os
import sqlite3
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    # Baked mesh & results persistence
    "persist_baked_mesh",
    "load_baked_mesh",
    "persist_baked_results",
    "load_baked_snapshot",
    "compute_max_tracking",
    "persist_baked_coupling",
    "persist_baked_coupling_batch",
    "load_baked_coupling_timeseries",
    "persist_baked_line_ts",
    "persist_baked_line_ts_batch",
    "persist_baked_line_profile",
    "persist_baked_line_profile_batch",
    "load_baked_line_timeseries",
    "load_baked_line_profile",
    "load_baked_timesteps",
    "collect_baked_runs_from_gpkg",
    # Utility functions
    "current_line_results_storage_path",
]






def current_line_results_storage_path(dialog) -> str:
    """Resolve the current GeoPackage storage path for line results.

    Priority: override edit → model GPKG path → sample lines layer GPKG → tempdir fallback.

    Parameters
    ----------
    dialog : SWE2DWorkbenchStudioDialog
        The studio dialog instance (used for widget reads and logging).

    Returns
    -------
    str
        Absolute path to a GeoPackage file.
    """
    import os, tempfile
    mtv = getattr(dialog, "_model_tab_view", None)
    if mtv is not None:
        path_edit = getattr(mtv, "results_gpkg_path_edit", None)
        if path_edit is not None:
            override_raw = str(path_edit.text() or "").strip()
            if override_raw:
                override = os.path.abspath(os.path.expanduser(override_raw))
                parent_dir = os.path.dirname(override) or "."
                if os.path.isdir(parent_dir):
                    dialog._log(f"[ResultsPath] using override: {override}")
                    return override
                else:
                    dialog._log(f"[ResultsPath] override parent dir missing: {parent_dir!r}, override={override!r}")
            else:
                dialog._log("[ResultsPath] results_gpkg_path_edit is empty")
        else:
            dialog._log("[ResultsPath] results_gpkg_path_edit not found on _model_tab_view")
    else:
        dialog._log("[ResultsPath] _model_tab_view not found")
    if dialog._model_gpkg_path and os.path.exists(dialog._model_gpkg_path):
        dialog._log(f"[ResultsPath] falling back to _model_gpkg_path: {dialog._model_gpkg_path}")
        return dialog._model_gpkg_path
    if hasattr(dialog, "_model_tab_view") and hasattr(dialog._model_tab_view, "sample_lines_layer_combo"):
        lyr = dialog._combo_layer(dialog._model_tab_view.sample_lines_layer_combo, "vector")
        if lyr is not None:
            try:
                src = str(lyr.dataProvider().dataSourceUri())
                gpkg = src.split("|", 1)[0]
                if gpkg.lower().endswith(".gpkg") and os.path.exists(gpkg):
                    return gpkg
            except Exception as e:
                dialog._log(f"[ERROR] current line results storage path failed: {e}")
    return os.path.join(tempfile.gettempdir(), "swe2d_line_results.gpkg")


def persist_baked_mesh(
    gpkg_path: str,
    mesh_name: str,
    baked_blob: bytes,
    n_nodes: int = 0,
    n_cells: int = 0,
    n_edges: int = 0,
    crs_wkt: str = "",
    log_fn: Optional[Callable[[str], None]] = None,
) -> None:
    """Save a serialized SWE2DMesh BLOB to the swe2d_baked_mesh table.

    Parameters
    ----------
    gpkg_path : str
        Path to the GeoPackage file.
    mesh_name : str
        Unique mesh name.
    baked_blob : bytes
        Serialized mesh bytes from hydra_swe2d.swe2d_serialize_mesh().
    n_nodes, n_cells, n_edges : int
        Mesh dimension counts (used for quick metadata queries without deserialization).
    crs_wkt : str
        CRS Well-Known-Text string.
    log_fn : callable, optional
        Logging callback.
    """
    if not gpkg_path or not baked_blob:
        return
    conn = sqlite3.connect(gpkg_path)
    try:
        _configure_connection(conn)
        # Ensure OGC GPKG metadata tables exist (required by GDAL/QGIS)
        _ensure_ogc_gpkg_tables(conn, crs_wkt=crs_wkt)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS swe2d_baked_mesh (
                mesh_name TEXT PRIMARY KEY,
                n_nodes INTEGER NOT NULL,
                n_cells INTEGER NOT NULL,
                n_edges INTEGER NOT NULL,
                crs_wkt TEXT DEFAULT '',
                created_utc TEXT NOT NULL,
                baked_blob BLOB NOT NULL)
        """)
        conn.execute(
            "INSERT OR REPLACE INTO swe2d_baked_mesh VALUES (?, ?, ?, ?, ?, ?, ?)",
            (mesh_name, n_nodes, n_cells, n_edges, str(crs_wkt),
             datetime.datetime.now(datetime.timezone.utc).isoformat(),
             baked_blob),
        )
        conn.commit()
        if log_fn:
            log_fn(f"Baked mesh saved: {mesh_name} ({n_nodes} nodes, {n_cells} cells, {len(baked_blob)} bytes)")
    finally:
        conn.close()


def load_baked_mesh(
    gpkg_path: str,
    mesh_name: str,
) -> Optional[bytes]:
    """Load a serialized SWE2DMesh BLOB from the swe2d_baked_mesh table.

    Parameters
    ----------
    gpkg_path : str
        Path to the GeoPackage file.
    mesh_name : str
        Mesh name to look up.

    Returns
    -------
    bytes or None
        The raw baked BLOB, or None if not found.
    """
    if not gpkg_path or not os.path.exists(gpkg_path):
        return None
    conn = sqlite3.connect(gpkg_path)
    try:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='swe2d_baked_mesh'"
        )
        if cur.fetchone() is None:
            return None
        row = conn.execute(
            "SELECT baked_blob FROM swe2d_baked_mesh WHERE mesh_name=?",
            (mesh_name,),
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def _configure_connection(conn: sqlite3.Connection) -> None:
    """Enable WAL mode and reasonable performance pragmas.

    WAL is best-effort; read-only or unsupported filesystems fall back
    to the default journal mode silently.
    """
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
    except Exception:
        pass


def _ensure_ogc_gpkg_tables(conn: sqlite3.Connection, crs_wkt: str = "") -> None:
    """Create the OGC GeoPackage metadata tables if they don't exist.

    Required for any valid .gpkg file that may be opened by GDAL/QGIS.
    """
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS spatial_ref_sys (
            srs_id INTEGER PRIMARY KEY,
            srs_name TEXT,
            srs_type TEXT,
            organization TEXT,
            organization_coordsys_id INTEGER,
            definition TEXT,
            description TEXT)
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS gpkg_contents (
            table_name TEXT NOT NULL PRIMARY KEY,
            data_type TEXT NOT NULL,
            identifier TEXT,
            description TEXT DEFAULT '',
            last_change DATETIME NOT NULL,
            min_x DOUBLE, min_y DOUBLE,
            max_x DOUBLE, max_y DOUBLE,
            srs_id INTEGER)
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS gpkg_geometry_columns (
            table_name TEXT NOT NULL,
            column_name TEXT NOT NULL,
            geometry_type_name TEXT NOT NULL,
            srs_id INTEGER NOT NULL,
            z TINYINT NOT NULL,
            m TINYINT NOT NULL,
            CONSTRAINT pk_geom_cols PRIMARY KEY (table_name, column_name))
    """)

    # Insert default SRS from the mesh CRS if not present
    if crs_wkt and not cur.execute("SELECT 1 FROM spatial_ref_sys WHERE srs_id=4326").fetchone():
        cur.execute(
            "INSERT INTO spatial_ref_sys(srs_id, srs_name, srs_type, organization, "
            "organization_coordsys_id, definition, description) "
            "VALUES(4326,'Model CRS','geodetic','EPSG',4326,?,?)",
            (str(crs_wkt), "Model CRS"),
        )
    for tbl in ("spatial_ref_sys", "gpkg_contents", "gpkg_geometry_columns"):
        if not cur.execute(
            "SELECT 1 FROM gpkg_contents WHERE table_name=?", (tbl,)
        ).fetchone():
            cur.execute(
                "INSERT INTO gpkg_contents(table_name, data_type, identifier, "
                "last_change, srs_id) VALUES(?, 'attributes', ?, ?, 4326)",
                (tbl, tbl, datetime.datetime.now(datetime.timezone.utc).isoformat()),
            )

    # ── Simulation configs table ─────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS swe2d_simulation_configs (
            config_id       TEXT PRIMARY KEY,
            mesh_name       TEXT,
            created_utc     TEXT NOT NULL,
            run_duration_s  REAL DEFAULT 0.0,
            description     TEXT DEFAULT '',
            widget_state    TEXT NOT NULL)
    """)


def persist_simulation_config(
    gpkg_path: str,
    config_id: str,
    mesh_name: str,
    run_duration_s: float,
    widget_state: Dict[str, object],
    description: str = "",
    log_fn: Optional[Callable[[str], None]] = None,
) -> None:
    """Save a simulation configuration to the GeoPackage.

    Parameters
    ----------
    gpkg_path : str
        Path to the GeoPackage file.
    config_id : str
        Unique config identifier (e.g. f"{mesh_name}_{timestamp}").
    mesh_name : str
        Name of the associated mesh.
    run_duration_s : float
        Simulation run duration in seconds.
    widget_state : dict
        Dict of widget parameter values (from collect_run_widget_params()).
    description : str, optional
        Human-readable description.
    log_fn : callable, optional
        Logging callback.
    """
    _log = log_fn or (lambda _: None)
    try:
        conn = sqlite3.connect(gpkg_path)
        _ensure_ogc_gpkg_tables(conn)
        cur = conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO swe2d_simulation_configs "
            "(config_id, mesh_name, created_utc, run_duration_s, description, widget_state) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                str(config_id),
                str(mesh_name or ""),
                datetime.datetime.now(datetime.timezone.utc).isoformat(),
                float(run_duration_s),
                str(description),
                json.dumps(widget_state, default=str),
            ),
        )
        conn.commit()
        conn.close()
        _log(f"Simulation config '{config_id}' saved to {gpkg_path}")
    except Exception as exc:
        _log(f"[WARNING] Failed to persist simulation config: {exc}")


def load_simulation_configs(
    gpkg_path: str,
    mesh_name: Optional[str] = None,
    log_fn: Optional[Callable[[str], None]] = None,
) -> List[Dict[str, object]]:
    """Load simulation configurations from a GeoPackage.

    Parameters
    ----------
    gpkg_path : str
        Path to the GeoPackage file.
    mesh_name : str, optional
        If given, filter to configs for this mesh only.
    log_fn : callable, optional
        Logging callback.

    Returns
    -------
    list of dict
        Each dict has keys: config_id, mesh_name, created_utc, run_duration_s,
        description, widget_state (parsed from JSON).
    """
    _log = log_fn or (lambda _: None)
    results: List[Dict[str, object]] = []
    try:
        conn = sqlite3.connect(gpkg_path)
        cur = conn.cursor()
        if mesh_name:
            cur.execute(
                "SELECT config_id, mesh_name, created_utc, run_duration_s, description, widget_state "
                "FROM swe2d_simulation_configs WHERE mesh_name=? ORDER BY created_utc DESC",
                (mesh_name,),
            )
        else:
            cur.execute(
                "SELECT config_id, mesh_name, created_utc, run_duration_s, description, widget_state "
                "FROM swe2d_simulation_configs ORDER BY created_utc DESC"
            )
        for row in cur.fetchall():
            ws = {}
            try:
                ws = json.loads(str(row[5] or "{}"))
            except Exception:
                pass
            results.append({
                "config_id": str(row[0]),
                "mesh_name": str(row[1]),
                "created_utc": str(row[2]),
                "run_duration_s": float(row[3]) if row[3] else 0.0,
                "description": str(row[4]),
                "widget_state": ws,
            })
        conn.close()
    except Exception as exc:
        _log(f"[WARNING] Failed to load simulation configs: {exc}")
    return results


def persist_baked_results(
    gpkg_path: str,
    run_id: str,
    mesh_name: str,
    snapshot_timesteps: List,
    max_tracking: Optional[Dict[str, np.ndarray]] = None,
    crs_wkt: str = "",
    log_fn: Optional[Callable[[str], None]] = None,
) -> None:
    """Save baked mesh results (all timesteps + optional GPU max tracking) as BLOBs.

    Parameters
    ----------
    gpkg_path : str
        Path to the GeoPackage file.
    run_id : str
        Unique run identifier.
    mesh_name : str
        Name of the mesh that produced these results.
    snapshot_timesteps : list of (t_s, h_arr, hu_arr, hv_arr)
        Each element is a tuple of (float time, ndarray h, ndarray hu, ndarray hv).
    max_tracking : dict, optional
        Optional dict with keys "max_h", "max_hu", "max_hv" — GPU per-step maxima.
    crs_wkt : str
        CRS Well-Known-Text string (passed to OGC GPKG metadata).
    log_fn : callable, optional
        Logging callback.
    """
    if not gpkg_path or not snapshot_timesteps:
        return
    n_steps = len(snapshot_timesteps)
    n_cells = int(np.asarray(snapshot_timesteps[0][1]).size)

    conn = sqlite3.connect(gpkg_path)
    try:
        _configure_connection(conn)
        # Ensure OGC GPKG metadata tables exist (required by GDAL/QGIS)
        _ensure_ogc_gpkg_tables(conn, crs_wkt=crs_wkt)
    finally:
        conn.close()

    times = np.array([float(t) for t, _, _, _ in snapshot_timesteps], dtype=np.float64)
    h_all = np.empty(n_steps * n_cells, dtype=np.float64)
    hu_all = np.empty(n_steps * n_cells, dtype=np.float64)
    hv_all = np.empty(n_steps * n_cells, dtype=np.float64)
    for i, (_, h, hu, hv) in enumerate(snapshot_timesteps):
        s, e = i * n_cells, (i + 1) * n_cells
        h_all[s:e]  = np.asarray(h, dtype=np.float64).ravel()
        hu_all[s:e] = np.asarray(hu, dtype=np.float64).ravel()
        hv_all[s:e] = np.asarray(hv, dtype=np.float64).ravel()

    max_h = max_tracking.get("max_h") if max_tracking else None
    max_hu = max_tracking.get("max_hu") if max_tracking else None
    max_hv = max_tracking.get("max_hv") if max_tracking else None

    conn = sqlite3.connect(gpkg_path)
    try:
        _configure_connection(conn)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS swe2d_baked_results (
                run_id TEXT PRIMARY KEY,
                mesh_name TEXT NOT NULL,
                n_cells INTEGER NOT NULL,
                n_timesteps INTEGER NOT NULL,
                created_utc TEXT NOT NULL,
                times_blob BLOB NOT NULL,
                h_blob BLOB NOT NULL,
                hu_blob BLOB NOT NULL,
                hv_blob BLOB NOT NULL,
                max_h_blob BLOB,
                max_hu_blob BLOB,
                max_hv_blob BLOB)
        """)
        conn.execute(
            "INSERT OR REPLACE INTO swe2d_baked_results "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, mesh_name, n_cells, n_steps,
             datetime.datetime.now(datetime.timezone.utc).isoformat(),
             times.tobytes(), h_all.tobytes(), hu_all.tobytes(), hv_all.tobytes(),
             max_h.tobytes() if max_h is not None else None,
             max_hu.tobytes() if max_hu is not None else None,
             max_hv.tobytes() if max_hv is not None else None),
        )
        conn.commit()
        if log_fn:
            log_fn(f"Baked results saved: run={run_id}, {n_steps} timesteps, {n_cells} cells")
    finally:
        conn.close()


def load_baked_snapshot(
    source: str,
    run_id: str,
    t_s: float,
) -> Optional[Dict]:
    """Load a single timestep snapshot from baked GPKG BLOBs or live data.

    Parameters
    ----------
    source : str or SWE2DResultsData
        GPKG path string, or a SWE2DResultsData-like object with
        ``_live_times``, ``_live_h``, ``_live_hu``, ``_live_hv`` numpy arrays.
    run_id : str
        Run identifier.
    t_s : float
        Target simulation time (nearest timestep is used).

    Returns
    -------
    dict or None
        Dict with keys ``t_s``, ``h``, ``hu``, ``hv``, ``cell_count``,
        or None if not found.
    """
    # Live data path — duck type check for numpy arrays on the source
    if not isinstance(source, str):
        d = source
        if not hasattr(d, '_live_times') or d._live_times is None or d._live_times.size == 0:
            return None
        times = d._live_times
        if not hasattr(d, '_live_h') or d._live_h is None or d._live_h.size == 0:
            return None
        i = int(np.argmin(np.abs(times - t_s)))
        n_cells = d._live_h.shape[1] if d._live_h.ndim == 2 else 0
        return {
            "t_s": float(times[i]),
            "h": d._live_h[i].copy(),
            "hu": d._live_hu[i].copy(),
            "hv": d._live_hv[i].copy(),
            "cell_count": n_cells,
        }
    # GPKG path
    if not source or not os.path.exists(source):
        return None
    conn = sqlite3.connect(source)
    try:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='swe2d_baked_results'"
        )
        if cur.fetchone() is None:
            return None
        row = conn.execute(
            "SELECT n_timesteps, n_cells, times_blob, h_blob, hu_blob, hv_blob "
            "FROM swe2d_baked_results WHERE run_id=?",
            (run_id,),
        ).fetchone()
        if not row:
            return None
        n_steps, n_cells = int(row[0]), int(row[1])
        times = np.frombuffer(row[2], dtype=np.float64)
        h_all = np.frombuffer(row[3], dtype=np.float64).reshape(n_steps, n_cells)
        hu_all = np.frombuffer(row[4], dtype=np.float64).reshape(n_steps, n_cells)
        hv_all = np.frombuffer(row[5], dtype=np.float64).reshape(n_steps, n_cells)
        i = int(np.argmin(np.abs(times - t_s)))
        return {
            "t_s": float(times[i]),
            "h": h_all[i].copy(),
            "hu": hu_all[i].copy(),
            "hv": hv_all[i].copy(),
            "cell_count": n_cells,
        }
    finally:
        conn.close()


def compute_max_tracking(
    source: str,
    run_id: str,
) -> Dict[str, np.ndarray]:
    """Load per-cell GPU max tracking from baked results, falling back to
    snapshot-resolution max if GPU data is not available.

    Parameters
    ----------
    source : str
        GPKG path.
    run_id : str
        Run identifier.

    Returns
    -------
    dict
        Dict with keys ``max_h``, ``max_hu``, ``max_hv`` (each np.ndarray float64).
    """
    if not source or not os.path.exists(source):
        return {"max_h": np.empty(0), "max_hu": np.empty(0), "max_hv": np.empty(0)}
    conn = sqlite3.connect(source)
    try:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='swe2d_baked_results'"
        )
        if cur.fetchone() is None:
            return {"max_h": np.empty(0), "max_hu": np.empty(0), "max_hv": np.empty(0)}
        row = conn.execute(
            "SELECT n_timesteps, n_cells, times_blob, h_blob, hu_blob, hv_blob, "
            "max_h_blob, max_hu_blob, max_hv_blob "
            "FROM swe2d_baked_results WHERE run_id=?",
            (run_id,),
        ).fetchone()
        if not row:
            return {"max_h": np.empty(0), "max_hu": np.empty(0), "max_hv": np.empty(0)}
        n_ts, n_cells = int(row[0]), int(row[1])
        # Try GPU max tracking columns first
        if row[6] is not None:
            return {
                "max_h": np.frombuffer(row[6], dtype=np.float64),
                "max_hu": np.frombuffer(row[7], dtype=np.float64),
                "max_hv": np.frombuffer(row[8], dtype=np.float64),
            }
        # Fallback: snapshot-resolution max
        logger.warning(
            "max_h_blob is NULL for run_id=%s; falling back to snapshot max "
            "(this underestimates true per-step maxima).",
            run_id,
        )
        h_all = np.frombuffer(row[3], dtype=np.float64).reshape(n_ts, n_cells)
        hu_all = np.frombuffer(row[4], dtype=np.float64).reshape(n_ts, n_cells)
        hv_all = np.frombuffer(row[5], dtype=np.float64).reshape(n_ts, n_cells)
        return {
            "max_h": np.max(h_all, axis=0),
            "max_hu": np.max(hu_all, axis=0),
            "max_hv": np.max(hv_all, axis=0),
        }
    finally:
        conn.close()


def persist_baked_coupling(
    gpkg_path: str,
    run_id: str,
    component: str,
    object_id: str,
    object_name: str,
    metric: str,
    times: np.ndarray,
    values: np.ndarray,
    log_fn: Optional[Callable[[str], None]] = None,
) -> None:
    """Save a baked coupling timeseries as BLOBs.

    Parameters
    ----------
    gpkg_path : str
        Path to the GeoPackage file.
    run_id : str
        Run identifier.
    component : str
        e.g. "drainage_node", "drainage_link", "structure".
    object_id : str
        Object identifier.
    object_name : str
        Human-readable object name.
    metric : str
        e.g. "depth", "flow", "invert", "length".
    times : ndarray
        1-D float64 array of timestamps.
    values : ndarray
        1-D float64 array of values (same length as times).
    log_fn : callable, optional
        Logging callback.
    """
    if not gpkg_path:
        return
    conn = sqlite3.connect(gpkg_path)
    try:
        _configure_connection(conn)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS swe2d_baked_coupling (
                run_id TEXT,
                component TEXT,
                object_id TEXT,
                object_name TEXT,
                metric TEXT,
                n_timesteps INTEGER,
                times_blob BLOB,
                values_blob BLOB,
                PRIMARY KEY (run_id, component, object_id, metric))
        """)
        conn.execute(
            "INSERT OR REPLACE INTO swe2d_baked_coupling "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, component, object_id, object_name, metric,
             len(times), times.tobytes(), values.tobytes()),
        )
        conn.commit()
        if log_fn:
            log_fn(f"Baked coupling saved: {component}/{object_id}/{metric} ({len(times)} steps)")
    finally:
        conn.close()


def persist_baked_coupling_batch(
    gpkg_path: str,
    run_id: str,
    coupling_items: List[Dict[str, Any]],
    log_fn: Optional[Callable[[str], None]] = None,
) -> None:
    """Save many baked coupling timeseries in a single transaction.

    Parameters
    ----------
    gpkg_path : str
        Path to the GeoPackage file.
    run_id : str
        Run identifier.
    coupling_items : list of dict
        Each dict has keys: component, object_id, object_name, metric,
        times (1-D ndarray), values (1-D ndarray).
    log_fn : callable, optional
        Logging callback.
    """
    if not gpkg_path or not coupling_items:
        return
    conn = sqlite3.connect(gpkg_path)
    try:
        _configure_connection(conn)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS swe2d_baked_coupling (
                run_id TEXT,
                component TEXT,
                object_id TEXT,
                object_name TEXT,
                metric TEXT,
                n_timesteps INTEGER,
                times_blob BLOB,
                values_blob BLOB,
                PRIMARY KEY (run_id, component, object_id, metric))
        """)
        rows = []
        for item in coupling_items:
            component = item["component"]
            object_id = item["object_id"]
            object_name = item.get("object_name", object_id)
            metric = item["metric"]
            times = np.asarray(item["times"], dtype=np.float64)
            values = np.asarray(item["values"], dtype=np.float64)
            rows.append((
                run_id, component, object_id, object_name, metric,
                len(times), times.tobytes(), values.tobytes(),
            ))
            if log_fn:
                log_fn(f"Baked coupling saved: {component}/{object_id}/{metric} ({len(times)} steps)")
        conn.executemany(
            "INSERT OR REPLACE INTO swe2d_baked_coupling "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()
    finally:
        conn.close()


def load_baked_coupling_timeseries(
    gpkg_path: str,
    run_id: str,
    component: str,
    object_id: str,
    metric: str,
) -> tuple:
    """Load a baked coupling timeseries from GPKG BLOBs.

    Parameters
    ----------
    gpkg_path : str
        Path to the GeoPackage file.
    run_id : str
        Run identifier.
    component : str
        Component name.
    object_id : str
        Object identifier.
    metric : str
        Metric name.

    Returns
    -------
    tuple of (ndarray, ndarray) or (None, None)
        (times, values) as float64 arrays, or (None, None) if not found.
    """
    if not gpkg_path or not os.path.exists(gpkg_path):
        return None, None
    conn = sqlite3.connect(gpkg_path)
    try:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='swe2d_baked_coupling'"
        )
        if cur.fetchone() is None:
            return None, None
        row = conn.execute(
            "SELECT times_blob, values_blob FROM swe2d_baked_coupling "
            "WHERE run_id=? AND component=? AND object_id=? AND metric=?",
            (run_id, component, object_id, metric),
        ).fetchone()
        if not row:
            return None, None
        return (np.frombuffer(row[0], dtype=np.float64),
                np.frombuffer(row[1], dtype=np.float64))
    finally:
        conn.close()


def persist_baked_line_ts(
    gpkg_path: str,
    run_id: str,
    line_id: int,
    line_name: str,
    times: np.ndarray,
    depth_m: np.ndarray,
    velocity_ms: np.ndarray,
    wse_m: np.ndarray,
    bed_m: np.ndarray,
    flow_cms: np.ndarray,
    wet_frac: np.ndarray,
    fr: np.ndarray,
    log_fn: Optional[Callable[[str], None]] = None,
) -> None:
    """Save baked line timeseries as BLOBs.

    Parameters
    ----------
    gpkg_path : str
        Path to the GeoPackage file.
    run_id : str
        Run identifier.
    line_id : int
        Line identifier.
    line_name : str
        Human-readable line name.
    times : ndarray
        1-D float64 array of timestamps.
    depth_m, velocity_ms, wse_m, bed_m, flow_cms, wet_frac, fr : ndarray
        1-D float64 arrays (same length as times).
    log_fn : callable, optional
        Logging callback.
    """
    if not gpkg_path:
        return
    conn = sqlite3.connect(gpkg_path)
    try:
        _configure_connection(conn)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS swe2d_baked_line_ts (
                run_id TEXT,
                line_id INTEGER,
                line_name TEXT,
                n_timesteps INTEGER,
                times_blob BLOB,
                depth_blob BLOB,
                vel_blob BLOB,
                wse_blob BLOB,
                bed_blob BLOB,
                flow_blob BLOB,
                wet_frac_blob BLOB,
                fr_blob BLOB,
                PRIMARY KEY (run_id, line_id))
        """)
        conn.execute(
            "INSERT OR REPLACE INTO swe2d_baked_line_ts "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, line_id, line_name, len(times),
             times.tobytes(),
             np.asarray(depth_m, dtype=np.float64).tobytes(),
             np.asarray(velocity_ms, dtype=np.float64).tobytes(),
             np.asarray(wse_m, dtype=np.float64).tobytes(),
             np.asarray(bed_m, dtype=np.float64).tobytes(),
             np.asarray(flow_cms, dtype=np.float64).tobytes(),
             np.asarray(wet_frac, dtype=np.float64).tobytes(),
             np.asarray(fr, dtype=np.float64).tobytes()),
        )
        conn.commit()
        if log_fn:
            log_fn(f"Baked line TS saved: line={line_id} ({len(times)} steps)")
    finally:
        conn.close()


def persist_baked_line_ts_batch(
    gpkg_path: str,
    run_id: str,
    line_items: List[Dict[str, Any]],
    log_fn: Optional[Callable[[str], None]] = None,
) -> None:
    """Save many baked line timeseries in a single transaction.

    Parameters
    ----------
    gpkg_path : str
        Path to the GeoPackage file.
    run_id : str
        Run identifier.
    line_items : list of dict
        Each dict has keys: line_id, line_name, times, depth_m,
        velocity_ms, wse_m, bed_m, flow_cms, wet_frac, fr.
    log_fn : callable, optional
        Logging callback.
    """
    if not gpkg_path or not line_items:
        return
    conn = sqlite3.connect(gpkg_path)
    try:
        _configure_connection(conn)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS swe2d_baked_line_ts (
                run_id TEXT,
                line_id INTEGER,
                line_name TEXT,
                n_timesteps INTEGER,
                times_blob BLOB,
                depth_blob BLOB,
                vel_blob BLOB,
                wse_blob BLOB,
                bed_blob BLOB,
                flow_blob BLOB,
                wet_frac_blob BLOB,
                fr_blob BLOB,
                PRIMARY KEY (run_id, line_id))
        """)
        rows = []
        for item in line_items:
            line_id = int(item["line_id"])
            line_name = str(item.get("line_name", f"line_{line_id}"))
            times = np.asarray(item["times"], dtype=np.float64)
            rows.append((
                run_id, line_id, line_name, len(times),
                times.tobytes(),
                np.asarray(item.get("depth_m", []), dtype=np.float64).tobytes(),
                np.asarray(item.get("velocity_ms", []), dtype=np.float64).tobytes(),
                np.asarray(item.get("wse_m", []), dtype=np.float64).tobytes(),
                np.asarray(item.get("bed_m", []), dtype=np.float64).tobytes(),
                np.asarray(item.get("flow_cms", []), dtype=np.float64).tobytes(),
                np.asarray(item.get("wet_frac", []), dtype=np.float64).tobytes(),
                np.asarray(item.get("fr", []), dtype=np.float64).tobytes(),
            ))
            if log_fn:
                log_fn(f"Baked line TS saved: line={line_id} ({len(times)} steps)")
        conn.executemany(
            "INSERT OR REPLACE INTO swe2d_baked_line_ts "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()
    finally:
        conn.close()


def persist_baked_line_profile(
    gpkg_path: str,
    run_id: str,
    line_id: int,
    line_name: str,
    station_m: np.ndarray,
    times: np.ndarray,
    depth_m: np.ndarray,
    velocity_ms: np.ndarray,
    wse_m: np.ndarray,
    bed_m: np.ndarray,
    flow_qn: np.ndarray,
    fr: np.ndarray,
    wet: np.ndarray,
    log_fn: Optional[Callable[[str], None]] = None,
) -> None:
    """Save baked line profiles (2-D timestep×station arrays) as BLOBs.

    Parameters
    ----------
    gpkg_path : str
        Path to the GeoPackage file.
    run_id : str
        Run identifier.
    line_id : int
        Line identifier.
    line_name : str
        Human-readable line name.
    station_m : ndarray
        1-D float64 array of station positions [n_stations].
    times : ndarray
        1-D float64 array of timestamps [n_timesteps].
    depth_m, velocity_ms, wse_m, bed_m, flow_qn, fr : ndarray
        2-D float64 arrays [n_timesteps × n_stations].
    wet : ndarray
        2-D int32 array [n_timesteps × n_stations].
    log_fn : callable, optional
        Logging callback.
    """
    if not gpkg_path:
        return
    conn = sqlite3.connect(gpkg_path)
    try:
        _configure_connection(conn)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS swe2d_baked_line_profiles (
                run_id TEXT,
                line_id INTEGER,
                line_name TEXT,
                n_stations INTEGER,
                n_timesteps INTEGER,
                station_blob BLOB,
                times_blob BLOB,
                depth_blob BLOB,
                vel_blob BLOB,
                wse_blob BLOB,
                bed_blob BLOB,
                flow_qn_blob BLOB,
                fr_blob BLOB,
                wet_blob BLOB,
                PRIMARY KEY (run_id, line_id))
        """)
        conn.execute(
            "INSERT OR REPLACE INTO swe2d_baked_line_profiles "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, line_id, line_name,
             len(station_m), len(times),
             np.asarray(station_m, dtype=np.float64).tobytes(),
             np.asarray(times, dtype=np.float64).tobytes(),
             np.asarray(depth_m, dtype=np.float64).tobytes(),
             np.asarray(velocity_ms, dtype=np.float64).tobytes(),
             np.asarray(wse_m, dtype=np.float64).tobytes(),
             np.asarray(bed_m, dtype=np.float64).tobytes(),
             np.asarray(flow_qn, dtype=np.float64).tobytes(),
             np.asarray(fr, dtype=np.float64).tobytes(),
             np.asarray(wet, dtype=np.int32).tobytes()),
        )
        conn.commit()
        if log_fn:
            log_fn(f"Baked line profile saved: line={line_id} "
                   f"({len(times)} timesteps x {len(station_m)} stations)")
    finally:
        conn.close()


def persist_baked_line_profile_batch(
    gpkg_path: str,
    run_id: str,
    profile_items: List[Dict[str, Any]],
    log_fn: Optional[Callable[[str], None]] = None,
) -> None:
    """Save many baked line profiles in a single transaction.

    Parameters
    ----------
    gpkg_path : str
        Path to the GeoPackage file.
    run_id : str
        Run identifier.
    profile_items : list of dict
        Each dict has keys: line_id, line_name, station_m, times,
        depth_m, velocity_ms, wse_m, bed_m, flow_qn, fr, wet.
    log_fn : callable, optional
        Logging callback.
    """
    if not gpkg_path or not profile_items:
        return
    conn = sqlite3.connect(gpkg_path)
    try:
        _configure_connection(conn)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS swe2d_baked_line_profiles (
                run_id TEXT,
                line_id INTEGER,
                line_name TEXT,
                n_stations INTEGER,
                n_timesteps INTEGER,
                station_blob BLOB,
                times_blob BLOB,
                depth_blob BLOB,
                vel_blob BLOB,
                wse_blob BLOB,
                bed_blob BLOB,
                flow_qn_blob BLOB,
                fr_blob BLOB,
                wet_blob BLOB,
                PRIMARY KEY (run_id, line_id))
        """)
        rows = []
        for item in profile_items:
            line_id = int(item["line_id"])
            line_name = str(item.get("line_name", f"line_{line_id}"))
            station_m = np.asarray(item["station_m"], dtype=np.float64)
            times = np.asarray(item["times"], dtype=np.float64)
            rows.append((
                run_id, line_id, line_name,
                len(station_m), len(times),
                station_m.tobytes(),
                times.tobytes(),
                np.asarray(item["depth_m"], dtype=np.float64).tobytes(),
                np.asarray(item["velocity_ms"], dtype=np.float64).tobytes(),
                np.asarray(item["wse_m"], dtype=np.float64).tobytes(),
                np.asarray(item["bed_m"], dtype=np.float64).tobytes(),
                np.asarray(item["flow_qn"], dtype=np.float64).tobytes(),
                np.asarray(item["fr"], dtype=np.float64).tobytes(),
                np.asarray(item["wet"], dtype=np.int32).tobytes(),
            ))
            if log_fn:
                log_fn(f"Baked line profile saved: line={line_id} "
                       f"({len(times)} timesteps x {len(station_m)} stations)")
        conn.executemany(
            "INSERT OR REPLACE INTO swe2d_baked_line_profiles "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()
    finally:
        conn.close()


def load_baked_line_timeseries(
    source,
    run_id: str,
    line_id: int,
) -> dict:
    """Load baked line timeseries from GPKG BLOBs or live data.

    Parameters
    ----------
    source : str or SWE2DResultsData
        GPKG path string, or a data object with ``_live_line_ts`` dict.
    run_id : str
        Run identifier.
    line_id : int
        Line identifier.

    Returns
    -------
    dict
        Dict with keys ``t_s``, ``depth_m``, ``velocity_ms``, ``wse_m``,
        ``bed_m``, ``flow_cms``, or empty dict if not found.
    """
    # Live data path
    if not isinstance(source, str):
        d = source
        # Only return live data for the run that owns it.
        if getattr(d, '_live_run_id', '') != run_id:
            return {}
        if hasattr(d, '_live_line_ts') and line_id in d._live_line_ts:
            raw = d._live_line_ts[line_id]
            # t_s comes from _live_times (mesh snapshot timesteps); other fields
            # are stored as numpy arrays matching GPKG blob layout.
            result = {"line_name": raw.get("line_name", "")}
            times = getattr(d, '_live_times', None)
            result["t_s"] = np.asarray(times, dtype=np.float64) if times is not None else np.empty(0, dtype=np.float64)
            for k in ("depth_m", "velocity_ms", "wse_m", "bed_m",
                       "flow_cms", "wet_frac", "fr"):
                v = raw.get(k)
                result[k] = np.asarray(v, dtype=np.float64) if v is not None else np.empty(0, dtype=np.float64)
            return result
        if hasattr(d, 'get_line_ts_arrays'):
            result = d.get_line_ts_arrays(run_id, line_id)
            if result:
                return dict(result)
        return {}
    # GPKG path
    if not source or not os.path.exists(source):
        return {}
    conn = sqlite3.connect(source)
    try:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='swe2d_baked_line_ts'"
        )
        if cur.fetchone() is None:
            return {}
        row = conn.execute(
            "SELECT n_timesteps, times_blob, depth_blob, vel_blob, "
            "wse_blob, bed_blob, flow_blob, wet_frac_blob, fr_blob "
            "FROM swe2d_baked_line_ts WHERE run_id=? AND line_id=?",
            (run_id, line_id),
        ).fetchone()
        if not row:
            return {}
        # Guard each blob against NULL — return empty arrays rather than
        # crashing inside np.frombuffer.  wet_frac and fr have always been
        # nullable; the others are defended defensively in case of partial
        # writes or corruption.
        def _f64(idx: int) -> np.ndarray:
            v = row[idx]
            return np.frombuffer(v, dtype=np.float64) if v is not None else np.empty(0, dtype=np.float64)

        return {
            "t_s": _f64(1),
            "depth_m": _f64(2),
            "velocity_ms": _f64(3),
            "wse_m": _f64(4),
            "bed_m": _f64(5),
            "flow_cms": _f64(6),
            "wet_frac": _f64(7),
            "fr": _f64(8),
        }
    finally:
        conn.close()


def load_baked_line_profile(
    source,
    run_id: str,
    line_id: int,
    t_sec: float,
) -> dict:
    """Load a baked line profile at a specific time from GPKG BLOBs.

    Parameters
    ----------
    source : str or SWE2DResultsData
        GPKG path string, or a data object with ``_live_line_profile`` dict.
    run_id : str
        Run identifier.
    line_id : int
        Line identifier.
    t_sec : float
        Target time (nearest timestep is used).

    Returns
    -------
    dict
        Dict with keys ``station_m``, ``wse_m``, ``bed_m``, ``depth_m``,
        ``velocity_ms``, ``flow_qn``, ``fr``, ``wet``, or empty dict if not found.
    """
    # Live data path
    if not isinstance(source, str):
        d = source
        # Only return live data for the run that owns it.
        # Other runs fall through to GPKG.
        if getattr(d, '_live_run_id', '') != run_id:
            return {}
        # Structured live storage populated by SWE2DResultsData.populate_live_line_metrics
        raw = getattr(d, '_live_line_profile', {}).get(line_id)
        if isinstance(raw, dict):
            sta = np.asarray(raw.get("station_m", np.empty(0)), dtype=np.float64)
            if sta.size == 0:
                return {}
            snap_times = np.asarray(getattr(d, '_live_times', np.empty(0)), dtype=np.float64)
            if snap_times.size == 0:
                return {}
            i = int(np.argmin(np.abs(snap_times - t_sec)))
            out = {"station_m": sta}
            for key in ("wse_m", "bed_m", "depth_m", "velocity_ms", "flow_qn", "fr"):
                arr = raw.get(key)
                if arr is None or not hasattr(arr, "ndim"):
                    out[key] = np.full(sta.size, np.nan, dtype=np.float64)
                elif arr.ndim == 2 and i < arr.shape[0]:
                    out[key] = np.asarray(arr[i], dtype=np.float64)
                else:
                    out[key] = np.full(sta.size, np.nan, dtype=np.float64)
            wet = raw.get("wet")
            if wet is not None and hasattr(wet, "ndim") and wet.ndim == 2 and i < wet.shape[0]:
                out["wet"] = np.asarray(wet[i], dtype=np.int32)
            else:
                out["wet"] = np.zeros(sta.size, dtype=np.int32)
            return out
        # Legacy hook for objects exposing get_line_profile_arrays
        if hasattr(d, 'get_line_profile_arrays'):
            result = d.get_line_profile_arrays(run_id, line_id, t_sec)
            if result:
                return dict(result)
        return {}
    # GPKG path
    if not source or not os.path.exists(source):
        return {}
    conn = sqlite3.connect(source)
    try:
        # Gracefully return empty if table doesn't exist (no legacy fallback)
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='swe2d_baked_line_profiles'"
        )
        if cur.fetchone() is None:
            return {}
        row = conn.execute(
            "SELECT n_stations, n_timesteps, station_blob, times_blob, "
            "wse_blob, bed_blob, depth_blob, vel_blob, flow_qn_blob, fr_blob, wet_blob "
            "FROM swe2d_baked_line_profiles "
            "WHERE run_id=? AND line_id=?",
            (run_id, line_id),
        ).fetchone()
        if not row:
            return {}
        n_sta, n_ts = int(row[0]), int(row[1])
        stations = np.frombuffer(row[2], dtype=np.float64)
        times = np.frombuffer(row[3], dtype=np.float64)
        i = int(np.argmin(np.abs(times - t_sec)))
        return {
            "station_m": stations,
            "wse_m": np.frombuffer(row[4], dtype=np.float64).reshape(n_ts, n_sta)[i],
            "bed_m": np.frombuffer(row[5], dtype=np.float64).reshape(n_ts, n_sta)[i],
            "depth_m": np.frombuffer(row[6], dtype=np.float64).reshape(n_ts, n_sta)[i],
            "velocity_ms": np.frombuffer(row[7], dtype=np.float64).reshape(n_ts, n_sta)[i],
            "flow_qn": np.frombuffer(row[8], dtype=np.float64).reshape(n_ts, n_sta)[i],
            "fr": np.frombuffer(row[9], dtype=np.float64).reshape(n_ts, n_sta)[i],
            "wet": np.frombuffer(row[10], dtype=np.int32).reshape(n_ts, n_sta)[i],
        }
    finally:
        conn.close()


def load_baked_timesteps(
    source: str,
    run_id: str,
) -> np.ndarray:
    """Load the times array from baked results.

    Parameters
    ----------
    source : str
        GPKG path.
    run_id : str
        Run identifier.

    Returns
    -------
    ndarray
        1-D float64 array of timesteps, or empty array if not found.
    """
    if not source or not os.path.exists(source):
        return np.empty(0, dtype=np.float64)
    conn = sqlite3.connect(source)
    try:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='swe2d_baked_results'"
        )
        if cur.fetchone() is None:
            return np.empty(0, dtype=np.float64)
        row = conn.execute(
            "SELECT times_blob FROM swe2d_baked_results WHERE run_id=?",
            (run_id,),
        ).fetchone()
        if not row or row[0] is None:
            return np.empty(0, dtype=np.float64)
        return np.frombuffer(row[0], dtype=np.float64)
    finally:
        conn.close()


def collect_baked_runs_from_gpkg(
    gpkg_path: str,
) -> List[Dict]:
    """Discover runs in a GPKG by scanning the swe2d_baked_results table.

    Parameters
    ----------
    gpkg_path : str
        Path to the GeoPackage file.

    Returns
    -------
    list of dict
        Each dict: {run_id, n_timesteps, n_cells, mesh_name, created_utc,
        has_lines (bool), has_coupling (bool)}.
    """
    if not gpkg_path or not os.path.exists(gpkg_path):
        return []
    conn = sqlite3.connect(gpkg_path)
    try:
        # Check if baked table exists — this GPKG may use old per-row tables
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='swe2d_baked_results'"
        )
        if cur.fetchone() is None:
            return []
        results = []
        for row in conn.execute(
            "SELECT run_id, mesh_name, n_cells, n_timesteps, created_utc "
            "FROM swe2d_baked_results ORDER BY created_utc DESC"
        ).fetchall():
            run_id = str(row[0])
            has_lines = (
                conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='swe2d_baked_line_ts'"
                ).fetchone() is not None
                and conn.execute(
                    "SELECT 1 FROM swe2d_baked_line_ts WHERE run_id=? LIMIT 1",
                    (run_id,),
                ).fetchone() is not None
            )
            has_coupling = (
                conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='swe2d_baked_coupling'"
                ).fetchone() is not None
                and conn.execute(
                    "SELECT 1 FROM swe2d_baked_coupling WHERE run_id=? LIMIT 1",
                    (run_id,),
                ).fetchone() is not None
            )
            results.append({
                "run_id": run_id,
                "mesh_name": str(row[1] or ""),
                "n_cells": int(row[2] or 0),
                "n_timesteps": int(row[3] or 0),
                "created_utc": str(row[4] or ""),
                "has_lines": has_lines,
                "has_coupling": has_coupling,
            })
        return results
    finally:
        conn.close()


