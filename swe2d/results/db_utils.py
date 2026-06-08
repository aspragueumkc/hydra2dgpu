"""Shared SQLite/GeoPackage read helpers for SWE2D results modules."""

from __future__ import annotations

import logging
import sqlite3
from typing import List, Optional

logger = logging.getLogger(__name__)


def open_ro(gpkg_path: str, row_factory: Optional[object] = None) -> Optional[sqlite3.Connection]:
    """Open a GeoPackage read-only and return None on failure."""
    try:
        conn = sqlite3.connect(f"file:{gpkg_path}?mode=ro", uri=True)
        if row_factory is not None:
            conn.row_factory = row_factory
        return conn
    except Exception as exc:
        logger.debug("[RESULTS] Failed to open DB read-only: %s", exc)
        return None


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    try:
        cur = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        )
        return cur.fetchone() is not None
    except Exception as exc:
        logger.debug("[RESULTS] Failed to check table exists: %s", exc)
        return False


def table_columns(conn: sqlite3.Connection, table_name: str) -> List[str]:
    try:
        cur = conn.execute(f'PRAGMA table_info("{table_name}")')
        return [str(r[1]) for r in cur.fetchall()]
    except Exception as exc:
        logger.debug("[RESULTS] Failed to get table columns: %s", exc)
        return []
