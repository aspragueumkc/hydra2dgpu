"""Pure CSV export service — no Qt dependency."""

import csv
from typing import Any, List, Union

import numpy as np


def export_table_to_csv(
    path: Union[str, "csv._writer"], headers: List[str], rows: List[List[Any]]
) -> None:
    """Write tabular data to CSV file."""
    if isinstance(path, str):
        with open(path, "w", newline="") as f:
            _write_csv(f, headers, rows)
    else:
        _write_csv(path, headers, rows)


def _write_csv(f, headers: List[str], rows: List[List[Any]]) -> None:
    """write csv."""
    writer = csv.writer(f)
    writer.writerow(headers)
    for row in rows:
        writer.writerow(row)
