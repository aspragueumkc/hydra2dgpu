from __future__ import annotations

import datetime
import json
import os
import sqlite3
from typing import Any, Dict, List, Optional


def _normalize_table_prefix(prefix: Optional[str]) -> str:
    """normalize table prefix."""
    raw = str(prefix or "").strip()
    if not raw:
        return ""
    chars: List[str] = []
    for ch in raw:
        if ch.isalnum() or ch == "_":
            chars.append(ch)
        else:
            chars.append("_")
    cleaned = "".join(chars).strip("_")
    if not cleaned:
        return ""
    if not (cleaned[0].isalpha() or cleaned[0] == "_"):
        cleaned = f"p_{cleaned}"
    return cleaned


def _run_logs_table_name(table_prefix: Optional[str] = None) -> str:
    """run logs table name."""
    prefix = _normalize_table_prefix(table_prefix)
    if not prefix:
        return "swe2d_run_logs"
    return f"{prefix}_swe2d_run_logs"


def _q(name: str) -> str:
    """q."""
    return '"' + str(name).replace('"', '""') + '"'


def _ensure_run_log_schema(cur: sqlite3.Cursor, table_name: str) -> None:
    """ensure run log schema."""
    q_table = _q(table_name)
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {q_table} (
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

    cur.execute(f"PRAGMA table_info({q_table})")
    cols = {str(row[1]) for row in cur.fetchall()}
    if "metadata_json" not in cols:
        cur.execute(f"ALTER TABLE {q_table} ADD COLUMN metadata_json TEXT")


def persist_run_log_to_geopackage(
    *,
    gpkg_path: str,
    run_id: str,
    start_wallclock: str,
    end_wallclock: str,
    duration_s: float,
    log_text: str,
    metadata: Optional[Dict[str, Any]] = None,
    table_prefix: Optional[str] = None,
) -> bool:
    """persist run log to geopackage."""
    if not gpkg_path or not run_id:
        return False

    conn = sqlite3.connect(gpkg_path)
    try:
        cur = conn.cursor()
        table_name = _run_logs_table_name(table_prefix)
        q_table = _q(table_name)
        _ensure_run_log_schema(cur, table_name)

        metadata_json = ""
        if metadata:
            try:
                metadata_json = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
            except Exception:
                metadata_json = json.dumps({"raw": str(metadata)}, sort_keys=True, separators=(",", ":"))

        cur.execute(
            f"""
            INSERT OR REPLACE INTO {q_table}
            (run_id, created_utc, start_wallclock, end_wallclock, duration_s, log_text, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(run_id),
                datetime.datetime.now().astimezone().replace(microsecond=0).isoformat(),
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


def load_run_logs_from_geopackage(
    *,
    gpkg_path: str,
    table_prefix: Optional[str] = None,
) -> List[Dict[str, object]]:
    """Load run logs from geopackage."""
    if not gpkg_path or not os.path.exists(gpkg_path):
        return []

    conn = sqlite3.connect(gpkg_path)
    try:
        cur = conn.cursor()
        # Prefer prefixed table when configured, but remain backward-compatible.
        table_candidates = [_run_logs_table_name(table_prefix=None)]
        prefixed = _run_logs_table_name(table_prefix)
        if prefixed not in table_candidates:
            table_candidates.insert(0, prefixed)

        table_name = ""
        for cand in table_candidates:
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (cand,))
            if cur.fetchone() is not None:
                table_name = str(cand)
                break
        if not table_name:
            return []

        q_table = _q(table_name)

        cur.execute(f"PRAGMA table_info({q_table})")
        cols = {str(row[1]) for row in cur.fetchall()}
        has_metadata_json = "metadata_json" in cols

        if has_metadata_json:
            cur.execute(
                f"""
                SELECT run_id, created_utc, start_wallclock, end_wallclock, duration_s, log_text, metadata_json
                FROM {q_table}
                ORDER BY datetime(created_utc) DESC, rowid DESC
                """
            )
        else:
            cur.execute(
                f"""
                SELECT run_id, created_utc, start_wallclock, end_wallclock, duration_s, log_text
                FROM {q_table}
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
