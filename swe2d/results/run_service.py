"""Pure-Python, Qt-free service for SWE2D results run management.

Provides run discovery, merging, filtering, and palette cycling without
any Qt dependency.  Testable without QApplication.
"""

from __future__ import annotations

import dataclasses
import os as _os
from typing import List, Set, Tuple

from swe2d.results.queries import discover_line_result_runs


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class RunRecord:
    """Metadata for a single simulation run loaded from a results GeoPackage."""
    run_id: str
    gpkg_path: str
    color: Tuple[int, int, int]
    enabled: bool = True
    label: str = ""
    has_profile: bool = False
    created_utc: str = ""

    def display_label(self) -> str:
        """Return the user-facing label (falls back to run_id if empty)."""
        return self.label or self.run_id

    @property
    def key(self) -> str:
        """Unique composite key: gpkg_path::run_id."""
        return f"{self.gpkg_path}::{self.run_id}"


# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------

_PANEL_COLORS: List[Tuple[int, int, int]] = [
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
    (188, 189, 34),
    (23, 190, 207),
]


def next_color(index: int) -> Tuple[int, int, int]:
    """Return next color from the cycling palette."""
    return _PANEL_COLORS[index % len(_PANEL_COLORS)]


# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------

def collect_runs_from_gpkg(gpkg_path: str) -> List[RunRecord]:
    """Query GPKG for run records, return list of RunRecord objects."""
    if not gpkg_path:
        return []
    runs = discover_line_result_runs(gpkg_path)
    out: List[RunRecord] = []
    gpkg_short = _os.path.basename(gpkg_path)
    for meta in runs:
        rid = str(meta.get("run_id", ""))
        if not rid:
            continue
        is_snapshot = rid.startswith("swe2d_snapshot_") or (
            "snapshot" in rid.lower()
        )
        suffix = " [snapshot]" if is_snapshot else ""
        out.append(
            RunRecord(
                run_id=rid,
                gpkg_path=gpkg_path,
                color=(0, 0, 0),
                enabled=True,
                has_profile=bool(meta.get("has_profile", False)),
                created_utc=str(meta.get("created_utc", "")),
                label=f"{gpkg_short}:{rid}{suffix}",
            )
        )
    return out


# ---------------------------------------------------------------------------
# Filter helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge_run_records(
    records: List[RunRecord],
    selected_keys: Set[str],
    manual_paths: List[str],
) -> List[RunRecord]:
    """Filter *records* to those matching *selected_keys*, deduplicate, compact colours.

    For each manual GPKG path, only runs whose key is in *selected_keys*
    survive. Runs with keys not in *selected_keys* are excluded entirely.
    """
    combined: List[RunRecord] = []
    seen: set = set()

    for gpkg in manual_paths:
        path_recs = [r for r in records if r.gpkg_path == gpkg]
        gpkg_filter = {k for k in selected_keys if k.startswith(f"{gpkg}::")}
        if gpkg_filter:
            filtered = [r for r in path_recs if r.key in gpkg_filter]
            if filtered:
                path_recs = filtered
        for rec in path_recs:
            if rec.key not in seen:
                seen.add(rec.key)
                combined.append(rec)

    for i, rec in enumerate(combined):
        rec.color = next_color(i)
        rec.enabled = True
    return combined


# ---------------------------------------------------------------------------
# Remove
# ---------------------------------------------------------------------------

def remove_selected_runs(
    records: List[RunRecord],
    selected_keys: Set[str],
    manual_paths: List[str],
) -> Tuple[List[RunRecord], List[str]]:
    """Remove runs by key, return updated (records, manual_paths).

    Manual GPKG paths that no longer have any remaining runs are dropped.
    Colours are reassigned compactly.
    """
    remaining = [r for r in records if r.key not in selected_keys]
    remaining_paths = {r.gpkg_path for r in remaining}
    updated_manual_paths = [p for p in manual_paths if p in remaining_paths]
    for i, rec in enumerate(remaining):
        rec.color = next_color(i)
    return remaining, updated_manual_paths
