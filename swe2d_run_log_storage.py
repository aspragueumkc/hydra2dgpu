from __future__ import annotations

import datetime
import json
import os
import sqlite3
from typing import Any, Dict, List, Optional


def _ensure_run_log_schema(cur: sqlite3.Cursor) -> None:
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS swe2d_run_logs (
            run_id TEXT PRIMARY KEY,
            created_utc TEXT,
            start_wallclock TEXT,
            end_wallclock TEXT,
            duration_s REAL,
            log_text TEXT,
            metadata_json TEXT
        )
        """
    )

    cur.execute("PRAGMA table_info(swe2d_run_logs)")
    cols = {str(row[1]) for row in cur.fetchall()}
    if "metadata_json" not in cols:
        cur.execute("ALTER TABLE swe2d_run_logs ADD COLUMN metadata_json TEXT")


def persist_run_log_to_geopackage(
    *,
    gpkg_path: str,
    run_id: str,
    start_wallclock: str,
    end_wallclock: str,
    duration_s: float,
    log_text: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    if not gpkg_path or not run_id:
        return False

    conn = sqlite3.connect(gpkg_path)
    try:
        cur = conn.cursor()
        _ensure_run_log_schema(cur)

        metadata_json = ""
        if metadata:
            try:
                metadata_json = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
            except Exception:
                metadata_json = json.dumps({"raw": str(metadata)}, sort_keys=True, separators=(",", ":"))

        cur.execute(
            """
            INSERT OR REPLACE INTO swe2d_run_logs
            (run_id, created_utc, start_wallclock, end_wallclock, duration_s, log_text, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(run_id),
                datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
                str(start_wallclock or ""),
                str(end_wallclock or ""),
                float(duration_s),
                str(log_text or ""),
                metadata_json,
            ),
        )
        conn.commit()
        return True
    finally:
        conn.close()


def load_run_logs_from_geopackage(*, gpkg_path: str) -> List[Dict[str, object]]:
    if not gpkg_path or not os.path.exists(gpkg_path):
        return []

    conn = sqlite3.connect(gpkg_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='swe2d_run_logs'")
        if cur.fetchone() is None:
            return []

        cur.execute("PRAGMA table_info(swe2d_run_logs)")
        cols = {str(row[1]) for row in cur.fetchall()}
        has_metadata_json = "metadata_json" in cols

        if has_metadata_json:
            cur.execute(
                """
                SELECT run_id, created_utc, start_wallclock, end_wallclock, duration_s, log_text, metadata_json
                FROM swe2d_run_logs
                ORDER BY datetime(created_utc) DESC, rowid DESC
                """
            )
        else:
            cur.execute(
                """
                SELECT run_id, created_utc, start_wallclock, end_wallclock, duration_s, log_text
                FROM swe2d_run_logs
                ORDER BY datetime(created_utc) DESC, rowid DESC
                """
            )

        rows: List[Dict[str, object]] = []
        for row in cur.fetchall():
            if has_metadata_json:
                run_id, created_utc, start_wallclock, end_wallclock, duration_s, log_text, metadata_json = row
            else:
                run_id, created_utc, start_wallclock, end_wallclock, duration_s, log_text = row
                metadata_json = ""

            metadata: Dict[str, Any] = {}
            try:
                if metadata_json:
                    parsed = json.loads(str(metadata_json))
                    if isinstance(parsed, dict):
                        metadata = parsed
            except Exception:
                metadata = {}

            rows.append(
                {
                    "run_id": str(run_id or ""),
                    "created_utc": str(created_utc or ""),
                    "start_wallclock": str(start_wallclock or ""),
                    "end_wallclock": str(end_wallclock or ""),
                    "duration_s": float(duration_s or 0.0),
                    "log_text": str(log_text or ""),
                    "metadata_json": str(metadata_json or ""),
                    "metadata": metadata,
                }
            )
        return rows
    finally:
        conn.close()
