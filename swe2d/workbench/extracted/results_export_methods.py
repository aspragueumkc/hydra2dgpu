from __future__ import annotations

import datetime
import os
import sqlite3
from typing import Dict, List, Optional

from swe2d_workbench_qt import *  # type: ignore F401,F403


def load_coupling_results_from_geopackage(
    self,
    gpkg_path: str,
    run_id: Optional[str] = None,
) -> tuple[str, List[Dict[str, object]]]:
    if not gpkg_path or not os.path.exists(gpkg_path):
        return "", []
    conn = sqlite3.connect(gpkg_path)
    try:
        cur = conn.cursor()
        def _table_exists(name: str) -> bool:
            cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (str(name),),
            )
            return cur.fetchone() is not None

        data_candidates = [
            self._results_table_name("swe2d_coupling_results") if hasattr(self, "_results_table_name") else "swe2d_coupling_results",
            "swe2d_coupling_results",
        ]
        data_table = ""
        for cand in data_candidates:
            if _table_exists(cand):
                data_table = str(cand)
                break
        if not data_table:
            return "", []

        runs_candidates = [
            self._results_table_name("swe2d_coupling_results_runs") if hasattr(self, "_results_table_name") else "swe2d_coupling_results_runs",
            "swe2d_coupling_results_runs",
        ]
        runs_table = ""
        for cand in runs_candidates:
            if _table_exists(cand):
                runs_table = str(cand)
                break

        chosen = str(run_id or "").strip()
        if not chosen:
            if not runs_table:
                return "", []
            q_runs = runs_table.replace('"', '""')
            cur.execute(
                f"""
                SELECT run_id FROM \"{q_runs}\"
                ORDER BY datetime(created_utc) DESC, rowid DESC
                LIMIT 1
                """
            )
            row = cur.fetchone()
            if row is None:
                return "", []
            chosen = str(row[0])

        q_data = data_table.replace('"', '""')

        cur.execute(
            f"""
            SELECT t_s, component, object_id, object_name, metric, value
            FROM \"{q_data}\"
            WHERE run_id = ?
            ORDER BY t_s ASC, component ASC, metric ASC, object_id ASC
            """,
            (chosen,),
        )
        rows: List[Dict[str, object]] = []
        for t_s, component, object_id, object_name, metric, value in cur.fetchall():
            rows.append(
                {
                    "t_s": float(t_s),
                    "component": str(component or ""),
                    "object_id": str(object_id or ""),
                    "object_name": str(object_name or ""),
                    "metric": str(metric or ""),
                    "value": float(value),
                }
            )
        return chosen, rows
    finally:
        conn.close()


def open_coupling_results_viewer(self) -> None:
    db_path = ""
    if self._coupling_results_latest_db_path and os.path.exists(self._coupling_results_latest_db_path):
        db_path = self._coupling_results_latest_db_path
    if not db_path:
        db_path = self._current_line_results_storage_path()
    if not db_path:
        self._log("No GeoPackage available for coupling results viewer.")
        return

    run_id = self._coupling_results_latest_run_id or None
    chosen, rows = self._load_coupling_results_from_geopackage(db_path, run_id=run_id)
    if not chosen or not rows:
        self._log("No drainage/structure coupling results found in GeoPackage yet.")
        return

    dlg = SWE2DCouplingResultsViewerDialog(
        records=rows,
        run_id=chosen,
        db_path=db_path,
        length_unit=self._length_unit_name,
        flow_unit_label=self._flow_unit_label(),
        parent=self,
    )
    dlg.exec()


def persist_mesh_results_to_geopackage(
    self,
    gpkg_path: str,
    run_id: str,
    mesh_rows: List[Dict[str, object]],
    interval_s: float,
    table_name: str = "swe2d_mesh_results",
) -> None:
    if not gpkg_path or not mesh_rows:
        return
    base_table_name = str(table_name or "swe2d_mesh_results").strip() or "swe2d_mesh_results"
    if hasattr(self, "_results_table_name"):
        try:
            base_table_name = str(self._results_table_name(base_table_name) or base_table_name)
        except Exception:
            pass
    runs_table_name = f"{base_table_name}_runs"

    def _quote_ident(name: str) -> str:
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
                row_count INTEGER
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
        cur.execute(f"DELETE FROM {q_table} WHERE run_id = ?", (run_id,))
        cur.execute(
            f"""
            INSERT OR REPLACE INTO {q_runs}
            (run_id, created_utc, interval_s, row_count)
            VALUES (?, ?, ?, ?)
            """,
            (
                str(run_id),
                datetime.datetime.now().astimezone().replace(microsecond=0).isoformat(),
                float(interval_s),
                int(len(mesh_rows)),
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
        support = self._velocity_data_support_for_run(gpkg_path, run_id, base_table_name)
        if int(support.get("face_rows", 0)) > 0:
            self._log(
                "Velocity persistence check: both cell-centered and face-centered data are present "
                f"(run_id={run_id}, cell_rows={int(support.get('cell_rows', 0))}, "
                f"face_table={support.get('face_table')}, face_rows={int(support.get('face_rows', 0))})."
            )
        else:
            self._log(
                "Velocity persistence check: only cell-centered h/hu/hv rows were stored for this run; "
                "no face-centered flux rows were found in GeoPackage tables "
                "(swe2d_face_flux_results / swe2d_face_results / swe2d_flux_faces)."
            )
        self._log(
            f"Stored mesh snapshot results in GeoPackage: {gpkg_path} "
            f"(run_id={run_id}, table={base_table_name}, rows={len(mesh_rows)})"
        )
    finally:
        conn.close()


def _try_extract_boundary_face_flux_totals(conn: sqlite3.Connection, run_id: str) -> Dict[str, object]:
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


def persist_conservation_forensics_to_geopackage(
    self,
    gpkg_path: str,
    run_id: str,
    storage_rows: List[Dict[str, object]],
    boundary_rows: List[Dict[str, object]],
    summary: Dict[str, object],
    source_step_rows: Optional[List[Dict[str, object]]] = None,
) -> None:
    if not gpkg_path or not run_id:
        return

    runs_table = "swe2d_conservation_runs"
    storage_table = "swe2d_conservation_storage_ts"
    boundary_table = "swe2d_boundary_flux_forensics_ts"
    source_table = "swe2d_source_budget_forensics_ts"
    if hasattr(self, "_results_table_name"):
        try:
            runs_table = str(self._results_table_name(runs_table) or runs_table)
            storage_table = str(self._results_table_name(storage_table) or storage_table)
            boundary_table = str(self._results_table_name(boundary_table) or boundary_table)
            source_table = str(self._results_table_name(source_table) or source_table)
        except Exception:
            runs_table = "swe2d_conservation_runs"
            storage_table = "swe2d_conservation_storage_ts"
            boundary_table = "swe2d_boundary_flux_forensics_ts"
            source_table = "swe2d_source_budget_forensics_ts"

    def _q(name: str) -> str:
        return '"' + str(name).replace('"', '""') + '"'

    q_runs = _q(runs_table)
    q_storage = _q(storage_table)
    q_boundary = _q(boundary_table)
    q_source = _q(source_table)

    l_scale = float(self._length_scale_si_to_model())
    vol_to_si = 1.0 / (l_scale ** 3)
    flow_to_si = vol_to_si

    conn = sqlite3.connect(gpkg_path)
    try:
        cur = conn.cursor()

        def _ensure_columns(table_name: str, columns: Dict[str, str]) -> None:
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
        self._log(
            f"Stored conservation forensics in GeoPackage: {gpkg_path} "
            f"(run_id={run_id}, storage_rows={len(storage_batch)}, source_rows={len(source_batch)}, boundary_rows={len(boundary_batch)}, "
            f"face_flux_status={str(face_flux_info.get('status', ''))})"
        )
    finally:
        conn.close()


def collect_run_log_metadata(self) -> Dict[str, object]:
    gate_cfg = dict(getattr(self, "_swe3d_geom_gate_last_config", {}) or {})
    gate_metrics = dict(getattr(self, "_swe3d_geom_gate_last_metrics", {}) or {})
    gate_violations = [str(v) for v in (getattr(self, "_swe3d_geom_gate_last_violations", []) or [])]

    if not gate_cfg:
        def _env_bool(name: str, default: bool) -> bool:
            raw = str(os.environ.get(name, "")).strip().lower()
            if not raw:
                return bool(default)
            return raw not in ("0", "false", "no", "off")

        def _env_float(name: str, default: float) -> float:
            try:
                return float(os.environ.get(name, str(default)))
            except Exception:
                return float(default)

        def _env_int(name: str, default: int) -> int:
            try:
                return int(os.environ.get(name, str(default)))
            except Exception:
                return int(default)

        gate_cfg = {
            "strict": _env_bool("BACKWATER_SWE3D_GEOM_STRICT", False),
            "max_solid_fraction": max(0.0, min(1.0, _env_float("BACKWATER_SWE3D_GEOM_MAX_SOLID_FRACTION", 0.98))),
            "max_seed_leak_fallbacks": max(0, _env_int("BACKWATER_SWE3D_GEOM_MAX_SEED_LEAK_FALLBACKS", 0)),
        }

    metadata: Dict[str, object] = {
        "swe3d_geometry_gate": {
            "strict": bool(gate_cfg.get("strict", False)),
            "max_solid_fraction": float(gate_cfg.get("max_solid_fraction", 0.98)),
            "max_seed_leak_fallbacks": int(gate_cfg.get("max_seed_leak_fallbacks", 0)),
            "violation_count": int(len(gate_violations)),
        }
    }
    if gate_metrics:
        metadata["swe3d_geometry_gate"]["metrics"] = gate_metrics
    if gate_violations:
        metadata["swe3d_geometry_gate"]["violations"] = gate_violations

    try:
        persistable_classes = (
            getattr(QtWidgets, "QSpinBox"),
            getattr(QtWidgets, "QDoubleSpinBox"),
            getattr(QtWidgets, "QComboBox"),
            getattr(QtWidgets, "QCheckBox"),
            getattr(QtWidgets, "QLineEdit"),
        )
        widget_attrs: List[str] = []
        for attr_name, widget in vars(self).items():
            if attr_name.startswith("_"):
                continue
            if isinstance(widget, persistable_classes):
                widget_attrs.append(str(attr_name))
        metadata["workbench_widget_state"] = collect_workbench_widget_state(
            ui=self,
            widget_attrs=widget_attrs,
            qtwidgets_module=QtWidgets,
        )
    except Exception:
        pass

    try:
        metadata["results_gpkg_path"] = str(self._current_line_results_storage_path() or "")
    except Exception:
        pass
    return metadata


def open_run_log_viewer(self) -> None:
    db_path = ""
    if self._run_log_latest_db_path and os.path.exists(self._run_log_latest_db_path):
        db_path = self._run_log_latest_db_path
    if not db_path:
        db_path = self._current_line_results_storage_path()
    if not db_path:
        self._log("No GeoPackage available for run log viewer.")
        return
    records = self._load_run_logs_from_geopackage(db_path)
    if not records:
        self._log("No saved run logs found in GeoPackage yet.")
        return
    dlg = SWE2DRunLogViewerDialog(
        records=records,
        run_id=self._run_log_latest_run_id,
        db_path=db_path,
        parent=self,
        apply_run_settings_callback=self._apply_run_log_metadata_to_ui,
    )
    dlg.exec()


def export_mesh_to_hdf5(self) -> None:
    if self._mesh_data is None:
        self._on_generate_mesh()
    if self._mesh_data is None:
        return
    out_path, _ = QtWidgets.QFileDialog.getSaveFileName(
        self,
        "Save Mesh As HEC-RAS HDF5",
        "swe2d_mesh.hdf",
        "HEC-RAS HDF5 (*.hdf)",
    )
    if not out_path:
        return
    try:
        out_path = self._normalize_hecras_hdf_path(out_path)
        self._write_hecras_hdf5(out_path)
        n_nodes = int(self._mesh_data["node_x"].shape[0])
        self._log(f"Saved HEC-RAS HDF5 mesh: {out_path} (nodes={n_nodes})")
        self.layer_status_lbl.setText("Mesh saved to HEC-RAS HDF5.")
    except (OSError, RuntimeError, ValueError) as exc:
        QtWidgets.QMessageBox.critical(self, "HDF5 Export", f"Export failed:\n{exc}")


def export_results_to_hdf5(self) -> None:
    if self._mesh_data is None or not self._snapshot_timesteps:
        self._log("Run the model first (snapshots must be captured) to export HDF5 results.")
        return
    out_path, _ = QtWidgets.QFileDialog.getSaveFileName(
        self,
        "Save Results As HEC-RAS HDF5",
        "swe2d_results.hdf",
        "HEC-RAS HDF5 (*.hdf)",
    )
    if not out_path:
        return
    try:
        out_path = self._normalize_hecras_hdf_path(out_path)
        self._write_hecras_hdf5(out_path, timesteps=self._snapshot_timesteps)
        n_ts = len(self._snapshot_timesteps)
        self._log(f"Saved HEC-RAS HDF5 results: {out_path} ({n_ts} timesteps)")
        self.layer_status_lbl.setText(f"Results saved to HEC-RAS HDF5 ({n_ts} timesteps).")
    except (OSError, RuntimeError, ValueError) as exc:
        QtWidgets.QMessageBox.critical(self, "HDF5 Export", f"Export failed:\n{exc}")


def export_results_to_ugrid(self) -> None:
    if self._mesh_data is None or not self._snapshot_timesteps:
        self._log("Run the model first (snapshots must be captured) to export UGRID results.")
        return
    out_path, _ = QtWidgets.QFileDialog.getSaveFileName(
        self,
        "Save Results As UGRID NetCDF",
        "swe2d_results.nc",
        "UGRID NetCDF (*.nc)",
    )
    if not out_path:
        return
    if not out_path.lower().endswith(".nc"):
        out_path += ".nc"
    try:
        self._write_ugrid_nc(out_path, timesteps=self._snapshot_timesteps)
        n_ts = len(self._snapshot_timesteps)
        self._log(f"Saved UGRID NetCDF results: {out_path} ({n_ts} timesteps)")
        self.layer_status_lbl.setText(f"Results saved to UGRID NetCDF ({n_ts} timesteps).")
    except (OSError, RuntimeError, ValueError) as exc:
        QtWidgets.QMessageBox.critical(self, "UGRID Export", f"Export failed:\n{exc}")