#!/usr/bin/env python3
"""Copy a HYDRA model GPKG with all results tables removed.

Keeps the model/input tables (SWE2D_* + swe2d_simulation_configs + gpkg_*)
and drops the per-run results tables (swe2d_baked_results,
swe2d_baked_line_ts, swe2d_baked_line_profiles, swe2d_baked_coupling,
swe2d_run_logs, swe2d_run_replays) plus their sqlite_sequence rows.

By default the swe2d_baked_mesh table is kept as-is. Pass --keep-latest-mesh
to prune it down to the single most recently created mesh (the others are
historical BLOB snapshots, typically the bulk of the file).

Uses SQLite's backup API + DROP + VACUUM so the output file reclaims the
freed pages (the source can be ~15 GB, mostly results/mesh BLOBs).
"""
from __future__ import annotations

import argparse
import sqlite3
import sys

RESULTS_TABLES = (
    "swe2d_baked_results",
    "swe2d_baked_line_ts",
    "swe2d_baked_line_profiles",
    "swe2d_baked_coupling",
    "swe2d_run_logs",
    "swe2d_run_replays",
)


def copy_without_results(src: str, dst: str, *, keep_latest_mesh: bool) -> list[str]:
    src_c = sqlite3.connect(src)
    dst_c = sqlite3.connect(dst)
    try:
        src_c.backup(dst_c)
    finally:
        src_c.close()

    dropped: list[str] = []
    with dst_c:
        for t in RESULTS_TABLES:
            # Confirm the table exists before dropping
            row = dst_c.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (t,),
            ).fetchone()
            if row:
                dst_c.execute(f'DROP TABLE IF EXISTS "{t}"')
                dropped.append(t)
        # Clean sqlite_sequence entries for the dropped tables
        for t in dropped:
            dst_c.execute("DELETE FROM sqlite_sequence WHERE name=?", (t,))

        if keep_latest_mesh:
            keep = dst_c.execute(
                "SELECT rowid FROM swe2d_baked_mesh "
                "ORDER BY created_utc DESC LIMIT 1"
            ).fetchone()
            if keep is not None:
                n = dst_c.execute(
                    "DELETE FROM swe2d_baked_mesh WHERE rowid != ?", (keep[0],)
                ).rowcount
                dropped.append(f"swe2d_baked_mesh (kept 1 of {n + 1})")
    # Reclaim freed pages (the results BLOBs are the bulk of the file)
    dst_c.execute("VACUUM")
    dst_c.close()
    return dropped


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("src", help="source model geopackage path")
    p.add_argument("dst", help="destination (results-free) geopackage path")
    p.add_argument(
        "--keep-latest-mesh", action="store_true",
        help="prune swe2d_baked_mesh down to the single most recent mesh",
    )
    args = p.parse_args(argv)

    print(f"copying {args.src} -> {args.dst} ...")
    dropped = copy_without_results(
        args.src, args.dst, keep_latest_mesh=args.keep_latest_mesh,
    )
    print("dropped:", "; ".join(dropped) or "(none)")
    print(f"wrote {args.dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
