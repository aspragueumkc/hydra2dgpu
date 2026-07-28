"""swe2d/workbench/services/profile_persistence_service.py

Pure-Python persistence for named network-profile chains. Stores in a new
swe2d_profile_chains table inside the user's GPKG. ChainSpec is owned by
profile_pipeline_service; this module only encodes / decodes it as a
comma-separated string with per-link orientation tokens.
"""

from __future__ import annotations

import datetime
import json
import os
import sqlite3
from typing import List, Optional

logger = __import__("logging").getLogger(__name__)

# Imported at runtime inside functions to avoid circular imports with Task 3
# (profile_pipeline_service which is created in the next batch).
_CHAIN_SPEC_CLASS = None


def _get_chain_spec_class():
    """Lazily import ChainSpec from profile_pipeline_service.

    Raised ImportError surfaces a clear 'service not yet available' instead
    of breaking module load before Task 3 lands.
    """
    global _CHAIN_SPEC_CLASS
    if _CHAIN_SPEC_CLASS is None:
        from swe2d.workbench.services.profile_pipeline_service import ChainSpec
        _CHAIN_SPEC_CLASS = ChainSpec
    return _CHAIN_SPEC_CLASS


PERSISTED_TABLE = "swe2d_profile_chains"
_SCHEMA = """
    CREATE TABLE IF NOT EXISTS swe2d_profile_chains (
        profile_id INTEGER PRIMARY KEY AUTOINCREMENT,
        profile_name TEXT UNIQUE NOT NULL,
        run_id TEXT,
        link_ids TEXT NOT NULL,
        created_utc TEXT NOT NULL,
        metadata_json TEXT DEFAULT '{}'
    )
"""


def _quote_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute(_SCHEMA)
    conn.commit()


def _encode_chain(chain) -> str:
    """Encode a ChainSpec as a comma-separated 'link,F|R,...' string."""
    parts = []
    for link_id, reverse in chain.link_specs:
        if "," in link_id:
            raise ValueError(
                f"Link id '{link_id}' contains a comma; cannot encode."
            )
        parts.append(f"{link_id},{'R' if reverse else 'F'}")
    return ",".join(parts)


def _decode_chain(s: str):
    """Decode a link_ids string into a ChainSpec."""
    ChainSpec = _get_chain_spec_class()
    link_specs = []
    if not s:
        return ChainSpec(link_specs=[])
    tokens = s.split(",")
    for index in range(0, len(tokens), 2):
        link_id = tokens[index]
        if not link_id:
            continue
        orient = tokens[index + 1] if index + 1 < len(tokens) else "F"
        link_specs.append((link_id, orient == "R"))
    return ChainSpec(link_specs=link_specs)


def save_profile(
    gpkg_path: str,
    profile_name: str,
    chain,
    run_id: Optional[str] = None,
) -> int:
    """Insert or replace a named profile. Returns profile_id."""
    if not gpkg_path or not os.path.exists(gpkg_path):
        raise ValueError(f"GeoPackage not found: {gpkg_path}")
    if not profile_name or not profile_name.strip():
        raise ValueError("profile_name must be non-empty")

    encoded = _encode_chain(chain)
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()

    conn = sqlite3.connect(gpkg_path)
    try:
        _ensure_table(conn)
        cur = conn.cursor()
        cur.execute(
            "SELECT profile_id FROM {} WHERE profile_name = ?".format(
                _quote_ident(PERSISTED_TABLE)
            ),
            (profile_name.strip(),),
        )
        existing = cur.fetchone()
        if existing is not None:
            pid = int(existing[0])
            cur.execute(
                "UPDATE {} SET run_id = ?, link_ids = ?, created_utc = ?, metadata_json = ? WHERE profile_id = ?".format(
                    _quote_ident(PERSISTED_TABLE)
                ),
                (run_id, encoded, now, "{}", pid),
            )
        else:
            cur.execute(
                "INSERT INTO {} (profile_name, run_id, link_ids, created_utc, metadata_json) VALUES (?, ?, ?, ?, ?)".format(
                    _quote_ident(PERSISTED_TABLE)
                ),
                (profile_name.strip(), run_id, encoded, now, "{}"),
            )
            pid = int(cur.lastrowid)
        conn.commit()
        return pid
    except sqlite3.IntegrityError as exc:
        raise ValueError(f"Profile name conflict: {profile_name}") from exc
    finally:
        conn.close()


def list_profiles(gpkg_path: str) -> List[dict]:
    """Return list of {profile_id, profile_name, run_id, link_ids, created_utc}."""
    if not gpkg_path or not os.path.exists(gpkg_path):
        return []
    try:
        conn = sqlite3.connect(gpkg_path)
    except sqlite3.Error:
        return []
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name = ?",
            (PERSISTED_TABLE,),
        )
        if cur.fetchone() is None:
            return []
        cur.execute(
            f"SELECT profile_id, profile_name, run_id, link_ids, created_utc "
            f"FROM {_quote_ident(PERSISTED_TABLE)} ORDER BY profile_id"
        )
        rows = []
        for r in cur.fetchall():
            metadata = {}
            try:
                metadata_row = conn.execute(
                    f"SELECT metadata_json FROM {_quote_ident(PERSISTED_TABLE)} WHERE profile_id = ?",
                    (r[0],),
                ).fetchone()
                if metadata_row and metadata_row[0]:
                    metadata = json.loads(metadata_row[0])
            except (sqlite3.Error, ValueError):
                metadata = {}
            rows.append({
                "profile_id": int(r[0]),
                "profile_name": str(r[1]),
                "run_id": str(r[2]) if r[2] is not None else None,
                "link_ids": str(r[3]),
                "created_utc": str(r[4]),
                "metadata": metadata,
            })
        return rows
    finally:
        conn.close()


def load_profile(gpkg_path: str, profile_id: int) -> "ChainSpec":
    """Load a profile chain by id. Returns ChainSpec."""
    rows = list_profiles(gpkg_path)
    for r in rows:
        if r["profile_id"] == profile_id:
            return _decode_chain(r["link_ids"])
    raise ValueError(f"Profile id {profile_id} not found in {gpkg_path}")


def delete_profile(gpkg_path: str, profile_id: int) -> None:
    """Delete a profile by id. No-op if id is unknown."""
    if not gpkg_path or not os.path.exists(gpkg_path):
        return
    try:
        conn = sqlite3.connect(gpkg_path)
    except sqlite3.Error:
        return
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name = ?",
            (PERSISTED_TABLE,),
        )
        if cur.fetchone() is None:
            return
        cur.execute(
            f"DELETE FROM {_quote_ident(PERSISTED_TABLE)} WHERE profile_id = ?",
            (profile_id,),
        )
        conn.commit()
    finally:
        conn.close()
