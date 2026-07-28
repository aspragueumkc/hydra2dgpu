"""Results data exports with visualization modules loaded on demand."""

from __future__ import annotations

from typing import Any

from swe2d.results.data import SWE2DResultsData
from swe2d.results.db_utils import open_ro, table_columns, table_exists
from swe2d.results.queries import (
    ResultsDataset,
    discover_line_result_runs,
    find_nearest_timestep,
    load_line_ids,
    load_profile,
    load_structure_flows_at_time,
    load_timeseries,
)
__all__ = [
    "ResultsAnimationController",
    "ResultsDataset",
    "SWE2DResultsData",
    "discover_line_result_runs",
    "find_nearest_timestep",
    "load_line_ids",
    "load_profile",
    "load_structure_flows_at_time",
    "load_timeseries",
    "open_ro",
    "render_unstructured_snapshot_image",
    "table_columns",
    "table_exists",
]

_LAZY_ATTRS = {
    "ResultsAnimationController": "swe2d.results.animation",
    "render_unstructured_snapshot_image": "swe2d.results.high_perf_viewer",
}


def __getattr__(name: str) -> Any:
    """Load GUI visualization exports only when explicitly requested."""
    module_name = _LAZY_ATTRS.get(name)
    if module_name is None:
        raise AttributeError(f"module 'swe2d.results' has no attribute {name!r}")
    module = __import__(module_name, fromlist=[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
