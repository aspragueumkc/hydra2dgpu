"""Pure-Python GeoPackage persistence service for the workbench.

Consolidates all GeoPackage/SQLite persistence logic previously scattered
across ``results_persistence_service``, ``studio_results_panel``, and inline
SQL in the dialog. This service is pure Python — it does not touch Qt.

Functions that need dialog access (e.g. ``current_line_results_storage_path``)
take the dialog as their first parameter.

NO SILENT FALLBACKS:
    * ``load_coupling_results_from_geopackage`` returns ``("", [])`` when
      the GPKG path is missing, the table does not exist, or no rows are
      found.
    * ``persist_mesh_results_to_geopackage`` returns ``None`` when the path
      or rows are empty — no phantom rows are written.
    * ``persist_conservation_forensics_to_geopackage`` returns ``None`` when
      the path or run_id are missing.
    * ``collect_run_log_metadata`` returns an empty dict on any failure.
"""
from __future__ import annotations

import datetime
import logging
import os
import sqlite3
from typing import Callable, Dict, List, Optional
import zlib

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
    "load_baked_coupling_timeseries",
    "persist_baked_line_ts",
    "persist_baked_line_profile",
    "load_baked_line_timeseries",
    "load_baked_line_profile",
    "load_baked_timesteps",
    "collect_baked_runs_from_gpkg",
    # Legacy (still referenced by workbench)
    "collect_run_log_metadata",
    "current_line_results_storage_path",
]






def persist_mesh_results_to_geopackage(
    gpkg_path: str,
    run_id: str,
    mesh_rows: List[Dict[str, object]],
    interval_s: float,
    table_name: str = "swe2d_mesh_results",
    mesh_name: str = "",
    mesh_hash: str = "",
    log_fn: Optional[Callable[[str], None]] = None,
    results_table_name_fn: Optional[Callable[[str], str]] = None,
    velocity_data_support_fn: Optional[Callable] = None,
    accumulate: bool = False,
) -> None:
    """Persist mesh snapshot results to a GeoPackage.

    Parameters
    ----------
    gpkg_path : str
        Path to the GeoPackage file.
    run_id : str
        Unique run identifier.
    mesh_rows : list of dict
        Mesh snapshot rows, each with t_s, cell_id, h, hu, hv.
    interval_s : float
        Snapshot interval in seconds.
    table_name : str, optional
        Base table name (default "swe2d_mesh_results").
    mesh_name : str, optional
        Name of the mesh that produced these results.
    mesh_hash : str, optional
        SHA-256 hash of the mesh geometry, used to associate
        results with a specific mesh on GPKG reload.
    log_fn : callable, optional
        Logging callback.
    results_table_name_fn : callable, optional
        Optional function to transform table names.
    velocity_data_support_fn : callable, optional
        Optional function to check for face-centered velocity data.
    """
    if not gpkg_path or not mesh_rows:
        return
    base_table_name = str(table_name or "swe2d_mesh_results").strip() or "swe2d_mesh_results"
    if results_table_name_fn is not None:
        try:
            base_table_name = str(results_table_name_fn(base_table_name) or base_table_name)
        except Exception:
            logger.warning("Unexpected error silently caught", exc_info=True)
    runs_table_name = f"{base_table_name}_runs"

    def _quote_ident(name: str) -> str:
        """quote ident."""
        return '"' + str(name).replace('"', '""') + '"'

    q_table = _quote_ident(base_table_name)
    q_runs = _quote_ident(runs_table_name)

    conn = sqlite3.connect(gpkg_path)
    try:
        cur = conn.cursor()
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {q_runs} (
                run_id TEXT PRIMARY KEY,
                created_utc TEXT,
                interval_s REAL,
                row_count INTEGER,
                snapshot INTEGER DEFAULT 0,
                mesh_name TEXT DEFAULT '',
                mesh_hash TEXT DEFAULT ''
            )
            """
        )
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {q_table} (
                run_id TEXT,
                t_s REAL,
                cell_id INTEGER,
                h REAL,
                hu REAL,
                hv REAL,
                PRIMARY KEY (run_id, t_s, cell_id)
            )
            """
        )
        cur.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{base_table_name}_run_t_cell "
            f"ON {q_table}(run_id, t_s, cell_id)"
        )
        if not accumulate:
            cur.execute(f"DELETE FROM {q_table} WHERE run_id = ?", (run_id,))
        cur.execute(
            f"""
            INSERT OR REPLACE INTO {q_runs}
            (run_id, created_utc, interval_s, row_count, mesh_name, mesh_hash)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                str(run_id),
                datetime.datetime.now().astimezone().replace(microsecond=0).isoformat(),
                float(interval_s),
                int(len(mesh_rows)),
                str(mesh_name or ""),
                str(mesh_hash or ""),
            ),
        )
        cur.executemany(
            f"""
            INSERT OR REPLACE INTO {q_table}
            (run_id, t_s, cell_id, h, hu, hv)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    str(run_id),
                    float(r.get("t_s", 0.0)),
                    int(r.get("cell_id", -1)),
                    float(r.get("h", 0.0)),
                    float(r.get("hu", 0.0)),
                    float(r.get("hv", 0.0)),
                )
                for r in mesh_rows
            ],
        )
        conn.commit()
        if velocity_data_support_fn is not None:
            support = velocity_data_support_fn(gpkg_path, run_id, base_table_name)
            if int(support.get("face_rows", 0)) > 0:
                log_fn(
                    "Velocity persistence check: both cell-centered and face-centered data are present "
                    f"(run_id={run_id}, cell_rows={int(support.get('cell_rows', 0))}, "
                    f"face_table={support.get('face_table')}, face_rows={int(support.get('face_rows', 0))})."
                )
            else:
                log_fn(
                    "Velocity persistence check: only cell-centered h/hu/hv rows were stored for this run; "
                    "no face-centered flux rows were found in GeoPackage tables "
                    "(swe2d_face_flux_results / swe2d_face_results / swe2d_flux_faces)."
                )
        if log_fn is not None:
            log_fn(
                f"Stored mesh snapshot results in GeoPackage: {gpkg_path} "
                f"(run_id={run_id}, table={base_table_name}, rows={len(mesh_rows)})"
            )
    finally:
        conn.close()


def _try_extract_boundary_face_flux_totals(conn: sqlite3.Connection, run_id: str) -> Dict[str, object]:
    """Try to extract boundary face flux totals from existing face tables.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open database connection.
    run_id : str
        Unique run identifier.

    Returns
    -------
    dict
        Info dict with keys: table, status, (rows, total_flux_model if found).
    """
    cur = conn.cursor()
    face_tables = ("swe2d_face_flux_results", "swe2d_face_results", "swe2d_flux_faces")
    for table_name in face_tables:
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        )
        if cur.fetchone() is None:
            continue

        cur.execute(f"PRAGMA table_info({table_name})")
        cols = [str(r[1]) for r in cur.fetchall()]
        col_set = set(cols)
        if "run_id" not in col_set:
            continue

        flux_col = ""
        for cand in ("flux_n", "flux", "q_n", "qn", "flow", "q"):
            if cand in col_set:
                flux_col = cand
                break
        if not flux_col:
            continue

        boundary_where = ""
        for cand in ("is_boundary", "boundary", "boundary_face", "at_boundary"):
            if cand in col_set:
                boundary_where = f" AND COALESCE({cand}, 0) <> 0"
                break
        if not boundary_where:
            for cand in ("nbr_cell_id", "cell_id_nbr", "cell_j", "neighbor_cell", "adj_cell_id"):
                if cand in col_set:
                    boundary_where = f" AND COALESCE({cand}, -1) < 0"
                    break
        if not boundary_where:
            return {
                "table": table_name,
                "status": "table_found_boundary_detection_unavailable",
            }

        if "t_s" in col_set:
            cur.execute(
                f"SELECT COUNT(*), COALESCE(SUM({flux_col}), 0.0) FROM {table_name} "
                "WHERE run_id = ?" + boundary_where,
                (str(run_id),),
            )
            row_count, total_flux = cur.fetchone()
            return {
                "table": table_name,
                "status": "ok",
                "rows": int(row_count or 0),
                "total_flux_model": float(total_flux or 0.0),
            }

        cur.execute(
            f"SELECT COUNT(*), COALESCE(SUM({flux_col}), 0.0) FROM {table_name} "
            "WHERE run_id = ?" + boundary_where,
            (str(run_id),),
        )
        row_count, total_flux = cur.fetchone()
        return {
            "table": table_name,
            "status": "ok_no_timestep",
            "rows": int(row_count or 0),
            "total_flux_model": float(total_flux or 0.0),
        }

    return {
        "table": "",
        "status": "table_not_found",
    }


def _compress_array(arr: np.ndarray) -> bytes:
    return zlib.compress(arr.tobytes())


def _decompress_array(data: bytes, dtype: np.dtype, shape: tuple) -> np.ndarray:
    return np.frombuffer(zlib.decompress(data), dtype=dtype).reshape(shape)


def persist_mesh_to_geopackage(
    gpkg_path: str,
    mesh_name: str,
    mesh_data: Dict[str, np.ndarray],
    crs_wkt: str = "",
    description: str = "",
    log_fn: Optional[Callable[[str], None]] = None,
) -> None:
    """Save mesh arrays to swe2d_mesh table as compressed BLOBs."""
    if not gpkg_path:
        if log_fn:
            log_fn("[GPKG] persist_mesh_to_geopackage: no gpkg_path specified")
        return
    if not mesh_data:
        if log_fn:
            log_fn("[GPKG] persist_mesh_to_geopackage: no mesh_data provided")
        return
    node_x = mesh_data.get("node_x")
    if node_x is None or node_x.size == 0:
        if log_fn:
            log_fn("[GPKG] persist_mesh_to_geopackage: node_x is empty or None")
        return
    nnodes = int(node_x.size)
    cell_nodes_arr = mesh_data.get("cell_nodes", np.empty(0))
    fo = mesh_data.get("cell_face_offsets")
    if fo is not None and fo.size > 0:
        ncells = int(fo.size) - 1
    else:
        ncells = int(cell_nodes_arr.size // 3) if cell_nodes_arr.size > 0 else 0

    # When cell_face_offsets exists, the backend expects cell_nodes to be
    # the flat array of all face vertex indices (= cell_face_nodes), not
    # the triangulated version.  Use cell_face_nodes if available.
    stored_cell_nodes = mesh_data.get("cell_face_nodes")
    if stored_cell_nodes is not None and stored_cell_nodes.size > 0 and fo is not None and fo.size > 0:
        pass  # use cell_face_nodes
    else:
        stored_cell_nodes = cell_nodes_arr

    import hashlib
    h = hashlib.sha256()
    for key in ("node_x", "node_y", "node_z",):
        arr = mesh_data.get(key)
        if arr is not None:
            h.update(arr.tobytes())
    if stored_cell_nodes is not None and stored_cell_nodes.size > 0:
        h.update(stored_cell_nodes.tobytes())
    conn = sqlite3.connect(gpkg_path)
    try:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS swe2d_mesh (
                mesh_name TEXT PRIMARY KEY,
                created_utc TEXT,
                nnodes INTEGER,
                ncells INTEGER,
                crs_wkt TEXT,
                hash TEXT,
                node_x BLOB, node_y BLOB, node_z BLOB,
                cell_nodes BLOB,
                face_offsets BLOB,
                bc_n0 BLOB, bc_n1 BLOB, bc_type BLOB, bc_val BLOB,
                description TEXT
            )
        """)
        def _b(key):
            a = mesh_data.get(key)
            return _compress_array(a) if a is not None and a.size > 0 else None
        cur.execute("DELETE FROM swe2d_mesh WHERE mesh_name = ?", (mesh_name,))
        cur.execute("""
            INSERT INTO swe2d_mesh(mesh_name, created_utc, nnodes, ncells, crs_wkt, hash,
                node_x, node_y, node_z, cell_nodes,
                face_offsets,
                bc_n0, bc_n1, bc_type, bc_val,
                description)
            VALUES(?,?,?,?,?,?,
                ?,?,?,?,
                ?,
                ?,?,?,?,
                ?)
        """, (
            mesh_name, datetime.datetime.now(datetime.timezone.utc).isoformat(),
            nnodes, ncells,
            str(crs_wkt or ""), h.hexdigest(),
            _b("node_x"), _b("node_y"), _b("node_z"),
            _compress_array(stored_cell_nodes) if stored_cell_nodes is not None and stored_cell_nodes.size > 0 else None,
            _b("cell_face_offsets"),
            _b("bc_edge_node0"), _b("bc_edge_node1"), _b("bc_edge_type"), _b("bc_edge_val"),
            str(description or ""),
        ))
        conn.commit()
        if log_fn:
            log_fn(f"Mesh '{mesh_name}' saved to {gpkg_path} ({nnodes} nodes, {ncells} cells)")
    finally:
        conn.close()


def compute_mesh_hash(mesh_data: dict) -> str:
    """Compute a deterministic SHA-256 hash from mesh geometry.

    Uses the same keys as ``persist_mesh_to_geopackage()`` so the hash
    is guaranteed to match between mesh persistence and results linking.
    """
    import hashlib
    stored_cell_nodes = mesh_data.get("cell_face_nodes")
    if stored_cell_nodes is None or stored_cell_nodes.size == 0:
        stored_cell_nodes = mesh_data.get("cell_nodes", np.empty(0))
    h = hashlib.sha256()
    for key in ("node_x", "node_y", "node_z"):
        arr = mesh_data.get(key)
        if arr is not None:
            h.update(arr.tobytes())
    if stored_cell_nodes is not None and stored_cell_nodes.size > 0:
        h.update(stored_cell_nodes.tobytes())
    return h.hexdigest()


def load_mesh_from_geopackage(
    gpkg_path: str,
    mesh_name: str,
) -> Optional[Dict[str, np.ndarray]]:
    """Load mesh arrays from swe2d_mesh table. Returns None if not found."""
    if not gpkg_path:
        return None
    conn = sqlite3.connect(gpkg_path)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='swe2d_mesh'"
        )
        if cur.fetchone() is None:
            return None
        cur.execute("SELECT node_x, node_y, node_z, cell_nodes, "
                     "face_offsets, "
                     "bc_n0, bc_n1, bc_type, bc_val "
                     "FROM swe2d_mesh WHERE mesh_name = ?", (mesh_name,))
        row = cur.fetchone()
        if row is None:
            return None
        def _ld(data, dtype):
            if data is None:
                return np.empty(0, dtype=dtype)
            try:
                return _decompress_array(data, dtype, (-1,))
            except (zlib.error, ValueError):
                return np.empty(0, dtype=dtype)
        out = {
            "node_x": _ld(row[0], np.float64),
            "node_y": _ld(row[1], np.float64),
            "node_z": _ld(row[2], np.float64),
            "cell_nodes": _ld(row[3], np.int32),
        }
        fo = _ld(row[4], np.int32) if row[4] else None
        if fo is not None and fo.size > 0:
            out["cell_face_offsets"] = fo
            # The cell_nodes column stores flat face nodes when saved
            # from mesh_data that had cell_face_nodes (see persist path).
            # Verify before aliasing — if inconsistent, cell_nodes is
            # triangulated and offsets belong to a lost face array.
            if int(fo[-1]) == int(out["cell_nodes"].size):
                out["cell_face_nodes"] = out["cell_nodes"]
        bc_n0 = _ld(row[5], np.int32)
        bc_n1 = _ld(row[6], np.int32)
        bc_tp = _ld(row[7], np.int32)
        bc_vl = _ld(row[8], np.float64)
        if bc_n0.size > 0:
            out["bc_edge_node0"] = bc_n0
            out["bc_edge_node1"] = bc_n1
            out["bc_edge_type"] = bc_tp
            out["bc_edge_val"] = bc_vl
        return out
    finally:
        conn.close()


def persist_mesh_max_results_to_geopackage(
    gpkg_path: str,
    run_id: str,
    max_results: Dict[str, np.ndarray],
    log_fn: Optional[Callable[[str], None]] = None,
) -> None:
    """Persist per-cell max results to GeoPackage (one row per cell, no t_s)."""
    if not gpkg_path or not max_results:
        return
    n = min(v.size for v in max_results.values())
    if n <= 0:
        return

    conn = sqlite3.connect(gpkg_path)
    try:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS swe2d_mesh_max_results_runs (
                run_id TEXT PRIMARY KEY,
                created_utc TEXT,
                row_count INTEGER
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS swe2d_mesh_max_results (
                run_id TEXT,
                cell_id INTEGER,
                max_h REAL,
                max_hu REAL,
                max_hv REAL,
                max_wse REAL,
                max_vel REAL,
                PRIMARY KEY (run_id, cell_id)
            )
        """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_swe2d_mesh_max_results_run "
            "ON swe2d_mesh_max_results(run_id)"
        )

        h_arr = np.asarray(max_results["max_h"], dtype=np.float64).ravel()
        hu_arr = np.asarray(max_results["max_hu"], dtype=np.float64).ravel()
        hv_arr = np.asarray(max_results["max_hv"], dtype=np.float64).ravel()
        wse_arr = np.asarray(max_results["max_wse"], dtype=np.float64).ravel()
        vel_arr = np.asarray(max_results["max_vel"], dtype=np.float64).ravel()

        def _quote_ident(name: str) -> str:
            return '"' + str(name).replace('"', '""') + '"'

        q_table = _quote_ident("swe2d_mesh_max_results")
        cur.execute(f"DELETE FROM {q_table} WHERE run_id = ?", (run_id,))
        rows = []
        for ci in range(n):
            rows.append((
                run_id, ci,
                float(h_arr[ci]) if ci < h_arr.size else 0.0,
                float(hu_arr[ci]) if ci < hu_arr.size else 0.0,
                float(hv_arr[ci]) if ci < hv_arr.size else 0.0,
                float(wse_arr[ci]) if ci < wse_arr.size else 0.0,
                float(vel_arr[ci]) if ci < vel_arr.size else 0.0,
            ))
        cur.executemany(
            f"INSERT INTO {q_table}(run_id, cell_id, max_h, max_hu, max_hv, max_wse, max_vel) "
            f"VALUES(?,?,?,?,?,?,?)", rows
        )
        cur.execute(
            "INSERT OR REPLACE INTO swe2d_mesh_max_results_runs(run_id, created_utc, row_count) "
            "VALUES(?,?,?)",
            (run_id, datetime.datetime.now(datetime.timezone.utc).isoformat(), n),
        )
        conn.commit()
        if log_fn:
            log_fn(f"Max results saved to GeoPackage: {n} cells")
    finally:
        conn.close()


def persist_conservation_forensics_to_geopackage(
    gpkg_path: str,
    run_id: str,
    storage_rows: List[Dict[str, object]],
    boundary_rows: List[Dict[str, object]],
    summary: Dict[str, object],
    source_step_rows: Optional[List[Dict[str, object]]] = None,
    log_fn: Optional[Callable[[str], None]] = None,
    results_table_name_fn: Optional[Callable[[str], str]] = None,
    length_scale_si_to_model_fn: Optional[Callable[[], float]] = None,
) -> None:
    """Persist conservation forensics results to a GeoPackage.

    Parameters
    ----------
    gpkg_path : str
        Path to the GeoPackage file.
    run_id : str
        Unique run identifier.
    storage_rows : list of dict
        Storage time-series rows.
    boundary_rows : list of dict
        Boundary flux time-series rows.
    summary : dict
        Summary statistics dict.
    source_step_rows : list of dict, optional
        Source budget step rows.
    log_fn : callable, optional
        Logging callback.
    results_table_name_fn : callable, optional
        Optional function to transform table names.
    length_scale_si_to_model_fn : callable, optional
        Optional function returning the SI-to-model length scale factor.
    """
    if not gpkg_path or not run_id:
        return

    runs_table = "swe2d_conservation_runs"
    storage_table = "swe2d_conservation_storage_ts"
    boundary_table = "swe2d_boundary_flux_forensics_ts"
    source_table = "swe2d_source_budget_forensics_ts"
    if results_table_name_fn is not None:
        try:
            runs_table = str(results_table_name_fn(runs_table) or runs_table)
            storage_table = str(results_table_name_fn(storage_table) or storage_table)
            boundary_table = str(results_table_name_fn(boundary_table) or boundary_table)
            source_table = str(results_table_name_fn(source_table) or source_table)
        except Exception:
            runs_table = "swe2d_conservation_runs"
            storage_table = "swe2d_conservation_storage_ts"
            boundary_table = "swe2d_boundary_flux_forensics_ts"
            source_table = "swe2d_source_budget_forensics_ts"

    def _q(name: str) -> str:
        """q."""
        return '"' + str(name).replace('"', '""') + '"'

    q_runs = _q(runs_table)
    q_storage = _q(storage_table)
    q_boundary = _q(boundary_table)
    q_source = _q(source_table)

    l_scale = float(length_scale_si_to_model_fn()) if length_scale_si_to_model_fn is not None else 1.0
    vol_to_si = 1.0 / (l_scale ** 3)
    flow_to_si = vol_to_si

    conn = sqlite3.connect(gpkg_path)
    try:
        cur = conn.cursor()

        def _ensure_columns(table_name: str, columns: Dict[str, str]) -> None:
            """ensure columns."""
            cur.execute(f"PRAGMA table_info({_q(table_name)})")
            existing = {str(r[1]) for r in cur.fetchall()}
            for col_name, col_type in columns.items():
                if str(col_name) in existing:
                    continue
                cur.execute(f"ALTER TABLE {_q(table_name)} ADD COLUMN {col_name} {col_type}")

        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {q_runs} (
                run_id TEXT PRIMARY KEY,
                created_utc TEXT,
                run_duration_s REAL,
                source_rain_model REAL,
                source_cell_model REAL,
                source_coupling_model REAL,
                source_total_model REAL,
                storage_start_model REAL,
                storage_end_model REAL,
                storage_delta_model REAL,
                implied_net_boundary_out_model REAL,
                avg_implied_boundary_q_model REAL,
                boundary_group_volume_sum_model REAL,
                source_total_m3 REAL,
                storage_start_m3 REAL,
                storage_end_m3 REAL,
                storage_delta_m3 REAL,
                implied_net_boundary_out_m3 REAL,
                avg_implied_boundary_q_cms REAL,
                boundary_group_volume_sum_m3 REAL,
                boundary_face_flux_table TEXT,
                boundary_face_flux_status TEXT,
                boundary_face_flux_rows INTEGER,
                boundary_face_flux_total_model REAL,
                boundary_face_flux_total_cms REAL,
                effective_net_boundary_method TEXT,
                effective_net_boundary_out_model REAL,
                effective_net_boundary_out_m3 REAL,
                effective_avg_q_model REAL,
                effective_avg_q_cms REAL,
                closure_residual_model REAL,
                closure_residual_m3 REAL
            )
            """
        )
        _ensure_columns(
            runs_table,
            {
                "boundary_face_flux_table": "TEXT",
                "boundary_face_flux_status": "TEXT",
                "boundary_face_flux_rows": "INTEGER",
                "boundary_face_flux_total_model": "REAL",
                "boundary_face_flux_total_cms": "REAL",
                "effective_net_boundary_method": "TEXT",
                "effective_net_boundary_out_model": "REAL",
                "effective_net_boundary_out_m3": "REAL",
                "effective_avg_q_model": "REAL",
                "effective_avg_q_cms": "REAL",
                "closure_residual_model": "REAL",
                "closure_residual_m3": "REAL",
            },
        )
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {q_storage} (
                run_id TEXT,
                t_s REAL,
                storage_model REAL,
                storage_delta_model REAL,
                storage_m3 REAL,
                storage_delta_m3 REAL,
                PRIMARY KEY (run_id, t_s)
            )
            """
        )
        _ensure_columns(
            storage_table,
            {
                "storage_m3": "REAL",
                "storage_delta_m3": "REAL",
            },
        )
        cur.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{storage_table}_run_t "
            f"ON {q_storage}(run_id, t_s)"
        )
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {q_boundary} (
                run_id TEXT,
                t_s REAL,
                group_name TEXT,
                q_requested_model REAL,
                q_effective_model REAL,
                vol_requested_model REAL,
                vol_effective_model REAL,
                q_requested_cms REAL,
                q_effective_cms REAL,
                vol_requested_m3 REAL,
                vol_effective_m3 REAL,
                source_note TEXT,
                PRIMARY KEY (run_id, t_s, group_name)
            )
            """
        )
        _ensure_columns(
            boundary_table,
            {
                "q_requested_model": "REAL",
                "q_effective_model": "REAL",
                "vol_requested_model": "REAL",
                "vol_effective_model": "REAL",
                "q_requested_cms": "REAL",
                "q_effective_cms": "REAL",
                "vol_requested_m3": "REAL",
                "vol_effective_m3": "REAL",
                "source_note": "TEXT",
            },
        )
        cur.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{boundary_table}_run_t_grp "
            f"ON {q_boundary}(run_id, t_s, group_name)"
        )
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {q_source} (
                run_id TEXT,
                t_s REAL,
                rain_vol_model REAL,
                cell_vol_model REAL,
                coupling_vol_model REAL,
                source_total_vol_model REAL,
                rain_vol_m3 REAL,
                cell_vol_m3 REAL,
                coupling_vol_m3 REAL,
                source_total_vol_m3 REAL,
                PRIMARY KEY (run_id, t_s)
            )
            """
        )
        _ensure_columns(
            source_table,
            {
                "rain_vol_m3": "REAL",
                "cell_vol_m3": "REAL",
                "coupling_vol_m3": "REAL",
                "source_total_vol_m3": "REAL",
            },
        )
        cur.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{source_table}_run_t "
            f"ON {q_source}(run_id, t_s)"
        )

        cur.execute(f"DELETE FROM {q_storage} WHERE run_id = ?", (str(run_id),))
        cur.execute(f"DELETE FROM {q_boundary} WHERE run_id = ?", (str(run_id),))
        cur.execute(f"DELETE FROM {q_source} WHERE run_id = ?", (str(run_id),))

        storage_batch = []
        for row in list(storage_rows or []):
            t_s = float(row.get("t_s", 0.0))
            storage_model = float(row.get("storage_model", 0.0))
            storage_delta_model = float(row.get("storage_delta_model", 0.0))
            storage_batch.append(
                (
                    str(run_id),
                    t_s,
                    storage_model,
                    storage_delta_model,
                    storage_model * vol_to_si,
                    storage_delta_model * vol_to_si,
                )
            )
        if storage_batch:
            cur.executemany(
                f"""
                INSERT OR REPLACE INTO {q_storage}
                (run_id, t_s, storage_model, storage_delta_model, storage_m3, storage_delta_m3)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                storage_batch,
            )

        boundary_batch = []
        for row in list(boundary_rows or []):
            t_s = float(row.get("t_s", 0.0))
            group_name = str(row.get("group_name", "") or "")
            q_effective_model = float(row.get("q_effective_model", 0.0))
            vol_effective_model = float(row.get("vol_effective_model", 0.0))
            # Requested values are currently equivalent to the applied BC values
            # captured at runtime; this keeps a stable schema for future split accounting.
            q_requested_model = q_effective_model
            vol_requested_model = vol_effective_model
            boundary_batch.append(
                (
                    str(run_id),
                    t_s,
                    group_name,
                    q_requested_model,
                    q_effective_model,
                    vol_requested_model,
                    vol_effective_model,
                    q_requested_model * flow_to_si,
                    q_effective_model * flow_to_si,
                    vol_requested_model * vol_to_si,
                    vol_effective_model * vol_to_si,
                    "requested_from_applied_bc_values",
                )
            )
        if boundary_batch:
            cur.executemany(
                f"""
                INSERT OR REPLACE INTO {q_boundary}
                (
                    run_id,
                    t_s,
                    group_name,
                    q_requested_model,
                    q_effective_model,
                    vol_requested_model,
                    vol_effective_model,
                    q_requested_cms,
                    q_effective_cms,
                    vol_requested_m3,
                    vol_effective_m3,
                    source_note
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                boundary_batch,
            )

        source_batch = []
        for row in list(source_step_rows or []):
            rain_vol_model = float(row.get("rain_vol_model", 0.0))
            cell_vol_model = float(row.get("cell_vol_model", 0.0))
            coupling_vol_model = float(row.get("coupling_vol_model", 0.0))
            source_total_vol_model = float(row.get("source_total_vol_model", 0.0))
            source_batch.append(
                (
                    str(run_id),
                    float(row.get("t_s", 0.0)),
                    rain_vol_model,
                    cell_vol_model,
                    coupling_vol_model,
                    source_total_vol_model,
                    rain_vol_model * vol_to_si,
                    cell_vol_model * vol_to_si,
                    coupling_vol_model * vol_to_si,
                    source_total_vol_model * vol_to_si,
                )
            )
        if source_batch:
            cur.executemany(
                f"""
                INSERT OR REPLACE INTO {q_source}
                (
                    run_id,
                    t_s,
                    rain_vol_model,
                    cell_vol_model,
                    coupling_vol_model,
                    source_total_vol_model,
                    rain_vol_m3,
                    cell_vol_m3,
                    coupling_vol_m3,
                    source_total_vol_m3
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                source_batch,
            )

        face_flux_info = _try_extract_boundary_face_flux_totals(conn, run_id)

        source_total_model = float(summary.get("source_total_model", 0.0))
        storage_start_model = float(summary.get("storage_start_model", 0.0))
        storage_end_model = float(summary.get("storage_end_model", 0.0))
        storage_delta_model = float(summary.get("storage_delta_model", 0.0))
        implied_boundary_model = float(summary.get("implied_net_boundary_out_model", 0.0))
        avg_implied_q_model = float(summary.get("avg_implied_boundary_q_model", 0.0))
        boundary_group_sum_model = float(summary.get("boundary_group_volume_sum_model", 0.0))
        closure_residual_model = float(source_total_model - storage_delta_model - implied_boundary_model)

        cur.execute(
            f"""
            INSERT OR REPLACE INTO {q_runs}
            (
                run_id,
                created_utc,
                run_duration_s,
                source_rain_model,
                source_cell_model,
                source_coupling_model,
                source_total_model,
                storage_start_model,
                storage_end_model,
                storage_delta_model,
                implied_net_boundary_out_model,
                avg_implied_boundary_q_model,
                boundary_group_volume_sum_model,
                source_total_m3,
                storage_start_m3,
                storage_end_m3,
                storage_delta_m3,
                implied_net_boundary_out_m3,
                avg_implied_boundary_q_cms,
                boundary_group_volume_sum_m3,
                boundary_face_flux_table,
                boundary_face_flux_status,
                boundary_face_flux_rows,
                boundary_face_flux_total_model,
                boundary_face_flux_total_cms,
                effective_net_boundary_method,
                effective_net_boundary_out_model,
                effective_net_boundary_out_m3,
                effective_avg_q_model,
                effective_avg_q_cms,
                closure_residual_model,
                closure_residual_m3
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(run_id),
                datetime.datetime.now().astimezone().replace(microsecond=0).isoformat(),
                float(summary.get("run_duration_s", 0.0)),
                float(summary.get("source_rain_model", 0.0)),
                float(summary.get("source_cell_model", 0.0)),
                float(summary.get("source_coupling_model", 0.0)),
                source_total_model,
                storage_start_model,
                storage_end_model,
                storage_delta_model,
                implied_boundary_model,
                avg_implied_q_model,
                boundary_group_sum_model,
                source_total_model * vol_to_si,
                storage_start_model * vol_to_si,
                storage_end_model * vol_to_si,
                storage_delta_model * vol_to_si,
                implied_boundary_model * vol_to_si,
                avg_implied_q_model * flow_to_si,
                boundary_group_sum_model * vol_to_si,
                str(face_flux_info.get("table", "") or ""),
                str(face_flux_info.get("status", "") or ""),
                int(face_flux_info.get("rows", 0) or 0),
                float(face_flux_info.get("total_flux_model", 0.0) or 0.0),
                float(face_flux_info.get("total_flux_model", 0.0) or 0.0) * flow_to_si,
                "conservation_identity",
                implied_boundary_model,
                implied_boundary_model * vol_to_si,
                avg_implied_q_model,
                avg_implied_q_model * flow_to_si,
                closure_residual_model,
                closure_residual_model * vol_to_si,
            ),
        )

        conn.commit()
        if log_fn is not None:
            log_fn(
                f"Stored conservation forensics in GeoPackage: {gpkg_path} "
                f"(run_id={run_id}, storage_rows={len(storage_batch)}, source_rows={len(source_batch)}, boundary_rows={len(boundary_batch)}, "
                f"face_flux_status={str(face_flux_info.get('status', ''))})"
            )
    finally:
        conn.close()




def collect_run_log_metadata(
    log_fn: Callable[[str], None],
    current_line_results_storage_path_fn: Optional[Callable[[], str]] = None,
    workbench_widget_state: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Collect run log metadata from the current workbench state.

    Parameters
    ----------
    log_fn : callable
        Logging callback.
    current_line_results_storage_path_fn : callable, optional
        Optional function returning the results storage path.
    workbench_widget_state : dict, optional
        Optional dict of workbench widget state.

    Returns
    -------
    dict
        Metadata dict (may be empty on failure).
    """
    metadata: Dict[str, object] = {}

    try:
        if workbench_widget_state is not None:
            metadata["workbench_widget_state"] = workbench_widget_state
    except Exception:
        log_fn(f"[WARNING] Unexpected error silently caught")

    try:
        if current_line_results_storage_path_fn is not None:
            metadata["results_gpkg_path"] = str(current_line_results_storage_path_fn() or "")
    except Exception:
        logger.warning("Failed to capture results_gpkg_path metadata", exc_info=True)

    return metadata


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
    if hasattr(dialog, "_map_tab_view") and hasattr(dialog._map_tab_view, "sample_lines_layer_combo"):
        lyr = dialog._combo_layer(dialog._map_tab_view.sample_lines_layer_combo, "vector")
        if lyr is not None:
            try:
                src = str(lyr.dataProvider().dataSourceUri())
                gpkg = src.split("|", 1)[0]
                if gpkg.lower().endswith(".gpkg") and os.path.exists(gpkg):
                    return gpkg
            except Exception as e:
                dialog._log(f"[ERROR] current line results storage path failed: {e}")
    return os.path.join(tempfile.gettempdir(), "swe2d_line_results.gpkg")


def persist_coupling_results_to_geopackage(
    gpkg_path: str,
    run_id: str,
    rows: List[Dict[str, object]],
    interval_s: float,
    results_table_name_fn: Callable[[str], str],
    log_fn: Callable[[str], None],
    accumulate: bool = False,
) -> None:
    """Persist coupling results (drainage/structures) to a GeoPackage.

    Replaces inline SQL that was previously in ``studio_results_panel`` and
    ``studio_dialog``. Takes callables instead of a dialog reference.

    Parameters
    ----------
    gpkg_path : str
        Path to the GeoPackage file.
    run_id : str
        Unique run identifier.
    rows : list of dict
        Coupling result rows with keys: t_s, component, object_id,
        object_name, metric, value.
    interval_s : float
        Coupling output interval in seconds.
    results_table_name_fn : callable
        Function to transform base table names (e.g. ``dialog._results_table_name``).
    log_fn : callable
        Logging callback (e.g. ``dialog._log``).
    """
    if not gpkg_path or not rows:
        return
    import datetime, sqlite3
    runs_table = str(results_table_name_fn("swe2d_coupling_results_runs"))
    data_table = str(results_table_name_fn("swe2d_coupling_results"))
    def _q(name: str) -> str:
        """q."""
        return '"' + str(name).replace('"', '""') + '"'
    q_runs = _q(runs_table)
    q_data = _q(data_table)
    conn = sqlite3.connect(gpkg_path)
    try:
        cur = conn.cursor()
        cur.execute(
            f"CREATE TABLE IF NOT EXISTS {q_runs} "
            "(run_id TEXT PRIMARY KEY, created_utc TEXT, interval_s REAL, row_count INTEGER, snapshot INTEGER DEFAULT 0)"
        )
        cur.execute(
            f"CREATE TABLE IF NOT EXISTS {q_data} "
            "(run_id TEXT, t_s REAL, component TEXT, object_id TEXT, object_name TEXT, "
            "metric TEXT, value REAL, "
            "PRIMARY KEY (run_id, t_s, component, object_id, metric))"
        )
        cur.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{data_table}_r_c_m_o_t "
            f"ON {q_data}(run_id, component, metric, object_id, t_s)"
        )
        if not accumulate:
            cur.execute(f"DELETE FROM {q_data} WHERE run_id = ?", (run_id,))
        cur.execute(
            f"INSERT OR REPLACE INTO {q_runs} "
            "(run_id, created_utc, interval_s, row_count) VALUES (?, ?, ?, ?)",
            (
                str(run_id),
                datetime.datetime.now().astimezone().replace(microsecond=0).isoformat(),
                float(interval_s),
                int(len(rows)),
            ),
        )
        batch = [
            (
                str(run_id),
                float(r.get("t_s", 0.0)),
                str(r.get("component", "") or ""),
                str(r.get("object_id", "") or ""),
                str(r.get("object_name", "") or ""),
                str(r.get("metric", "") or ""),
                float(r.get("value", float("nan"))),
            )
            for r in rows
        ]
        cur.executemany(
            f"INSERT OR REPLACE INTO {q_data} "
            "(run_id, t_s, component, object_id, object_name, metric, value) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            batch,
        )
        conn.commit()
        log_fn(
            f"Stored coupling results in GeoPackage: {gpkg_path} "
            f"(run_id={run_id}, rows={len(rows)})"
        )
    finally:
        conn.close()



# ═══════════════════════════════════════════════════════════════════════════════
# Baked mesh & results persistence (GPKG BLOB format)
# ═══════════════════════════════════════════════════════════════════════════════

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
        row = conn.execute(
            "SELECT baked_blob FROM swe2d_baked_mesh WHERE mesh_name=?",
            (mesh_name,),
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def persist_baked_results(
    gpkg_path: str,
    run_id: str,
    mesh_name: str,
    snapshot_timesteps: List,
    max_tracking: Optional[Dict[str, np.ndarray]] = None,
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
    log_fn : callable, optional
        Logging callback.
    """
    if not gpkg_path or not snapshot_timesteps:
        return
    n_steps = len(snapshot_timesteps)
    n_cells = int(np.asarray(snapshot_timesteps[0][1]).size)

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
                   f"({len(times)} timesteps × {len(station_m)} stations)")
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
        if hasattr(d, '_live_line_ts') and line_id in d._live_line_ts:
            return dict(d._live_line_ts[line_id])
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
        row = conn.execute(
            "SELECT n_timesteps, times_blob, depth_blob, vel_blob, "
            "wse_blob, bed_blob, flow_blob "
            "FROM swe2d_baked_line_ts WHERE run_id=? AND line_id=?",
            (run_id, line_id),
        ).fetchone()
        if not row:
            return {}
        return {
            "t_s": np.frombuffer(row[1], dtype=np.float64),
            "depth_m": np.frombuffer(row[2], dtype=np.float64),
            "velocity_ms": np.frombuffer(row[3], dtype=np.float64),
            "wse_m": np.frombuffer(row[4], dtype=np.float64),
            "bed_m": np.frombuffer(row[5], dtype=np.float64),
            "flow_cms": np.frombuffer(row[6], dtype=np.float64),
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
        or empty dict if not found.
    """
    # Live data path
    if not isinstance(source, str):
        d = source
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
        row = conn.execute(
            "SELECT n_stations, n_timesteps, station_blob, times_blob, "
            "wse_blob, bed_blob, depth_blob "
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
        row = conn.execute(
            "SELECT times_blob FROM swe2d_baked_results WHERE run_id=?",
            (run_id,),
        ).fetchone()
        return np.frombuffer(row[0], dtype=np.float64) if row else np.empty(0, dtype=np.float64)
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
        results = []
        for row in conn.execute(
            "SELECT run_id, mesh_name, n_cells, n_timesteps, created_utc "
            "FROM swe2d_baked_results ORDER BY created_utc DESC"
        ).fetchall():
            run_id = str(row[0])
            has_lines = conn.execute(
                "SELECT 1 FROM swe2d_baked_line_ts WHERE run_id=? LIMIT 1",
                (run_id,),
            ).fetchone() is not None
            has_coupling = conn.execute(
                "SELECT 1 FROM swe2d_baked_coupling WHERE run_id=? LIMIT 1",
                (run_id,),
            ).fetchone() is not None
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

