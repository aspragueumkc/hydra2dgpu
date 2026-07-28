#!/usr/bin/env python3
"""add_pipe1d_fields_to_existing_gpkg.py

Add the 11 new pipe1D fields to existing geopackages that pre-date the
2026-07-16 pipe1D solver rewrite.

The fields are added to the `swe2d_drainage_nodes` layer:

  Outfall BC configuration (per spec §2.8):
    - outfall_mode         (TEXT, default 'free')
    - outfall_fixed_wse    (REAL, default 0.0)
    - outfall_rating_wse   (TEXT, comma-separated WSE values, default '')
    - outfall_rating_q     (TEXT, comma-separated Q values, default '')
    - outfall_tabular_time (TEXT, comma-separated times, default '')
    - outfall_tabular_wse  (TEXT, comma-separated WSE values, default '')

  Junction surcharge overflow (per spec §2.10):
    - junction_overflow_diam      (REAL, default 0.0)
    - junction_overflow_coeff     (REAL, default 1.7)
    - junction_max_overflow_rate  (REAL, default 0.0)

  Inlet box storage-cell geometry (separate from inlet opening dims):
    - inlet_box_length (REAL, default 0.0)
    - inlet_box_width  (REAL, default 0.0)

The script is idempotent: re-running it on an already-migrated geopackage
will report "all 11 columns already exist" and exit cleanly.

Usage:
    python add_pipe1d_fields_to_existing_gpkg.py path/to/project.gpkg

Or scan a directory recursively:
    python add_pipe1d_fields_to_existing_gpkg.py --recursive path/to/dir

The script reads the layer name from gpkg_contents (looks for a layer named
`swe2d_drainage_nodes`); if absent, it lists the available layers and
exits with an error.
"""

import argparse
import sqlite3
import sys
from pathlib import Path


NEW_FIELDS = [
    # (column_name, sql_decl, default_sql_literal)
    ("outfall_mode",         "TEXT",   "'free'"),
    ("outfall_fixed_wse",    "REAL",   "0.0"),
    ("outfall_rating_wse",   "TEXT",   "''"),
    ("outfall_rating_q",     "TEXT",   "''"),
    ("outfall_tabular_time", "TEXT",   "''"),
    ("outfall_tabular_wse",  "TEXT",   "''"),
    ("junction_overflow_diam",     "REAL", "0.0"),
    ("junction_overflow_coeff",    "REAL", "1.7"),
    ("junction_max_overflow_rate", "REAL", "0.0"),
    ("inlet_box_length", "REAL", "0.0"),
    ("inlet_box_width",  "REAL", "0.0"),
]


def find_drainage_nodes_layer(conn):
    """Return the gpkg_contents table_name for the swe2d_drainage_nodes
    layer (case-insensitive), or None if the geopackage has no such layer."""
    try:
        rows = conn.execute(
            "SELECT table_name FROM gpkg_contents"
        ).fetchall()
    except sqlite3.OperationalError:
        return None
    names = {r[0]: r[0].lower() for r in rows}
    for original, lowered in names.items():
        if lowered == "swe2d_drainage_nodes":
            return original
    return None


def existing_columns(conn, table):
    """Return the set of column names currently on the table."""
    try:
        rows = conn.execute(f"PRAGMA table_info('{table}')").fetchall()
    except sqlite3.OperationalError:
        return set()
    return {r[1] for r in rows}


def add_columns(conn, table, fields):
    """ALTER TABLE to add each missing column with a default value.

    Returns a tuple (added, skipped) listing the column names.
    """
    present = existing_columns(conn, table)
    added = []
    skipped = []
    for col, decl, default in fields:
        if col in present:
            skipped.append(col)
            continue
        # SQLite ALTER TABLE ADD COLUMN with default literal.
        conn.execute(
            f'ALTER TABLE "{table}" ADD COLUMN "{col}" {decl} DEFAULT {default}'
        )
        added.append(col)
    conn.commit()
    return added, skipped


def migrate_one(path):
    """Migrate a single geopackage file. Returns a result dict."""
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "ok": False, "error": "file not found"}

    try:
        conn = sqlite3.connect(str(path))
    except sqlite3.DatabaseError as e:
        return {"path": str(path), "ok": False, "error": f"sqlite open failed: {e}"}

    try:
        table = find_drainage_nodes_layer(conn)
        if table is None:
            conn.close()
            return {"path": str(path), "ok": False,
                    "error": "no `swe2d_drainage_nodes` layer in this geopackage"}

        added, skipped = add_columns(conn, table, NEW_FIELDS)
        conn.close()
        return {"path": str(path), "ok": True,
                "table": table, "added": added, "skipped": skipped}
    except Exception as e:
        conn.close()
        return {"path": str(path), "ok": False, "error": f"{type(e).__name__}: {e}"}


def find_gpkg_files(targets, recursive):
    """Yield Path objects matching targets."""
    for t in targets:
        p = Path(t)
        if p.is_file():
            yield p
        elif p.is_dir():
            glob = "**/*.gpkg" if recursive else "*.gpkg"
            for child in sorted(p.glob(glob)):
                if child.is_file():
                    yield child


def main():
    ap = argparse.ArgumentParser(
        description="Add the 9 new pipe1D fields to existing geopackages.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("paths", nargs="+", help="GPKG file(s) or director(ies)")
    ap.add_argument("--recursive", "-r", action="store_true",
                    help="when a path is a directory, scan recursively")
    args = ap.parse_args()

    files = list(find_gpkg_files(args.paths, args.recursive))
    if not files:
        print("No .gpkg files found in the given paths.", file=sys.stderr)
        return 1

    total_added = 0
    total_skipped = 0
    failures = 0

    for f in files:
        result = migrate_one(f)
        if not result["ok"]:
            failures += 1
            print(f"FAIL  {result['path']}: {result['error']}")
            continue

        added = result["added"]
        skipped = result["skipped"]
        total_added += len(added)
        total_skipped += len(skipped)

        if added and skipped:
            print(f"OK    {result['path']}  added={len(added)} already_present={len(skipped)}")
        elif added:
            print(f"OK    {result['path']}  added={len(added)}")
        elif skipped:
            print(f"OK    {result['path']}  all 11 columns already present (no changes)")
        else:
            print(f"OK    {result['path']}")

    print()
    print(f"Summary: {len(files)} geopackage(s) processed.")
    print(f"  Columns added:        {total_added}")
    print(f"  Already present:      {total_skipped}")
    print(f"  Failures:             {failures}")
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
