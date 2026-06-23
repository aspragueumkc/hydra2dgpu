"""Pure Python, zero-Qt GPKG table operations for the explorer dialog.

Provides low-level DDL/DML operations on GeoPackage files.  No Qt
dependency — testable without QApplication.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _quote_sqlite_ident(name: str) -> str:
    """quote sqlite ident."""
    return '"' + str(name).replace('"', '""') + '"'


def _user_table_names(cur: sqlite3.Cursor) -> List[str]:
    """user table names."""
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    names: List[str] = []
    for r in cur.fetchall():
        name = str(r[0]) if r and r[0] is not None else ""
        if name and not name.startswith("sqlite_") and not name.startswith("gpkg_") and not name.startswith("rtree_"):
            names.append(name)
    return names


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_tables(gpkg_path: str) -> List[str]:
    """List all user tables in a GeoPackage.

    Excludes ``sqlite_*``, ``gpkg_*``, and ``rtree_*`` tables.
    Returns an empty list if the file does not exist or cannot be opened.
    """
    if not gpkg_path or not os.path.exists(gpkg_path):
        return []
    try:
        conn = sqlite3.connect(gpkg_path)
        try:
            return _user_table_names(conn.cursor())
        finally:
            conn.close()
    except sqlite3.Error:
        logger.exception("Error listing tables in %s", gpkg_path)
        return []


def get_table_row_count(gpkg_path: str, table: str) -> int:
    """Count rows in a table.

    Returns 0 if the table does not exist or the file cannot be opened.
    """
    if not gpkg_path or not os.path.exists(gpkg_path):
        return 0
    try:
        conn = sqlite3.connect(gpkg_path)
        try:
            cur = conn.cursor()
            cur.execute(f"SELECT COUNT(*) FROM {_quote_sqlite_ident(table)}")
            row = cur.fetchone()
            return int(row[0]) if row and row[0] is not None else 0
        finally:
            conn.close()
    except sqlite3.Error:
        logger.exception("Error counting rows in %s.%s", gpkg_path, table)
        return 0


def rename_table(gpkg_path: str, old_name: str, new_name: str) -> None:
    """Rename a table.

    Raises ``RuntimeError`` if the old table does not exist or the new
    name already exists.
    """
    conn = sqlite3.connect(gpkg_path)
    try:
        cur = conn.cursor()

        cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (old_name,))
        if cur.fetchone() is None:
            raise RuntimeError(f"Table '{old_name}' does not exist.")

        cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (new_name,))
        if cur.fetchone() is not None:
            raise RuntimeError(f"Table '{new_name}' already exists.")

        cur.execute(
            f"ALTER TABLE {_quote_sqlite_ident(old_name)} RENAME TO {_quote_sqlite_ident(new_name)}"
        )

        for meta_tbl in ("gpkg_contents", "gpkg_geometry_columns", "gpkg_extensions"):
            cur.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (meta_tbl,)
            )
            if cur.fetchone() is None:
                continue
            try:
                cur.execute(
                    "UPDATE sqlite_master SET name=? WHERE name=?",
                    (new_name, old_name),
                )
            except sqlite3.Error:
                logger.warning("Failed to update %s metadata for %s", meta_tbl, old_name)

        conn.commit()
    except sqlite3.Error as exc:
        conn.rollback()
        raise RuntimeError(f"Failed to rename table '{old_name}': {exc}") from exc
    finally:
        conn.close()


def drop_table(gpkg_path: str, table_name: str) -> None:
    """Drop a table and clean up GeoPackage metadata.

    Safe to call for tables that do not exist (no-op).
    """
    conn = sqlite3.connect(gpkg_path)
    try:
        cur = conn.cursor()
        cur.execute(f"DROP TABLE IF EXISTS {_quote_sqlite_ident(table_name)}")

        for meta_tbl in ("gpkg_contents", "gpkg_geometry_columns", "gpkg_extensions"):
            cur.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (meta_tbl,)
            )
            if cur.fetchone() is None:
                continue
            try:
                cur.execute(
                    f"DELETE FROM {_quote_sqlite_ident(meta_tbl)} WHERE table_name=?",
                    (table_name,),
                )
            except sqlite3.Error:
                logger.warning("Failed to clean %s metadata for %s", meta_tbl, table_name)

        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE ?",
            (f"rtree_{table_name}_%",),
        )
        for row in cur.fetchall():
            t = str(row[0]) if row and row[0] is not None else ""
            if t:
                try:
                    cur.execute(f"DROP TABLE IF EXISTS {_quote_sqlite_ident(t)}")
                except sqlite3.Error:
                    logger.warning("Failed to drop rtree sidecar %s", t)

        conn.commit()
    except sqlite3.Error as exc:
        conn.rollback()
        raise RuntimeError(f"Failed to drop table '{table_name}': {exc}") from exc
    finally:
        conn.close()


def get_table_info(gpkg_path: str, table: str) -> List[Dict[str, Any]]:
    """Get column info for a table via ``PRAGMA table_info``.

    Returns a list of dicts with keys ``cid``, ``name``, ``type``,
    ``notnull``, ``dflt_value``, ``pk``.  Returns an empty list if the
    table does not exist.
    """
    try:
        conn = sqlite3.connect(gpkg_path)
        try:
            cur = conn.cursor()
            cur.execute(f"PRAGMA table_info({_quote_sqlite_ident(table)})")
            col_names = [str(d[0]) for d in cur.description]
            return [dict(zip(col_names, row)) for row in cur.fetchall()]
        finally:
            conn.close()
    except sqlite3.Error:
        logger.exception("Error getting table info for %s.%s", gpkg_path, table)
        return []


def get_table_contents(
    gpkg_path: str, table: str, limit: int = 100
) -> List[Dict[str, Any]]:
    """Get rows from a table.

    Returns a list of dicts with column names as keys.  Returns an empty
    list if the table does not exist.
    """
    try:
        conn = sqlite3.connect(gpkg_path)
        try:
            cur = conn.cursor()
            cur.execute(
                f"SELECT * FROM {_quote_sqlite_ident(table)} LIMIT ?", (int(limit),)
            )
            col_names = [str(d[0]) for d in cur.description]
            return [dict(zip(col_names, row)) for row in cur.fetchall()]
        finally:
            conn.close()
    except sqlite3.Error:
        logger.exception("Error getting table contents for %s.%s", gpkg_path, table)
        return []


def delete_run(gpkg_path: str, run_id: str) -> None:
    """Delete a run and its associated tables/data.

    Drops all tables whose name ends with ``_{run_id}`` (excluding
    ``gpkg_*``, ``sqlite_*``, ``rtree_*``), cleans up GeoPackage
    metadata, deletes the corresponding ``swe2d_run_logs`` entry, and
    runs VACUUM.

    Safe to call for run IDs that do not exist (no-op).
    """
    if not gpkg_path or not os.path.exists(gpkg_path):
        return

    conn = sqlite3.connect(gpkg_path)
    try:
        cur = conn.cursor()

        cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        all_tables = [
            str(r[0]) for r in cur.fetchall() if r and r[0] is not None
        ]

        matching = [
            t
            for t in all_tables
            if t.endswith("_" + run_id)
            and not t.startswith("gpkg_")
            and not t.startswith("sqlite_")
            and not t.startswith("rtree_")
        ]

        deleted_tables = []
        for tbl in matching:
            try:
                cur.execute(f"DROP TABLE IF EXISTS {_quote_sqlite_ident(tbl)}")

                for meta_tbl in ("gpkg_contents", "gpkg_geometry_columns", "gpkg_extensions"):
                    cur.execute(
                        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                        (meta_tbl,),
                    )
                    if cur.fetchone() is None:
                        continue
                    try:
                        cur.execute(
                            f"DELETE FROM {_quote_sqlite_ident(meta_tbl)} WHERE table_name=?",
                            (tbl,),
                        )
                    except sqlite3.Error:
                        logger.warning("Failed to clean %s metadata for %s", meta_tbl, tbl)

                cur.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE ?",
                    (f"rtree_{tbl}_%",),
                )
                for row in cur.fetchall():
                    rt = str(row[0]) if row and row[0] is not None else ""
                    if rt:
                        try:
                            cur.execute(f"DROP TABLE IF EXISTS {_quote_sqlite_ident(rt)}")
                        except sqlite3.Error:
                            logger.warning("Failed to drop rtree sidecar %s", rt)

                deleted_tables.append(tbl)
            except sqlite3.Error as exc:
                logger.error("Failed to drop table '%s': %s", tbl, exc)

        # Delete run log entries
        cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='swe2d_run_logs'")
        if cur.fetchone() is not None:
            try:
                cur.execute("DELETE FROM swe2d_run_logs WHERE run_id=?", (run_id,))
            except sqlite3.Error as exc:
                logger.error("Failed to delete run log entry for '%s': %s", run_id, exc)

        conn.commit()

        if deleted_tables:
            try:
                conn.execute("VACUUM")
            except sqlite3.Error:
                logger.warning("VACUUM failed after delete_run for %s", run_id)
    except sqlite3.Error as exc:
        conn.rollback()
        raise RuntimeError(f"Failed to delete run '{run_id}': {exc}") from exc
    finally:
        conn.close()
