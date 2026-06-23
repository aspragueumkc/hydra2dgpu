"""Pure text parsing service for workbench hydrographs and time tokens.

Extracted from ``SWE2DWorkbenchStudioDialog._parse_time_hours`` and
``SWE2DWorkbenchStudioDialog._parse_hydrograph_text`` (Task 3 of
docs/STUDIO_GUI_FULL_MIGRATION_PLAN_2026-06-16.md).

This module is pure Python — it has zero Qt imports. The dialog will
delegate to these functions in Task 4 of the same plan. Until then,
the dialog still has its own copies.

NO SILENT FALLBACKS:
    * ``parse_time_hours`` raises ``ValueError`` on empty or invalid
      input (the old dialog method returned ``0.0`` — that is a bug,
      not a feature).
    * ``parse_hydrograph_text`` raises ``ValueError`` on empty,
      unparseable, or no-data input (the legacy helpers returned
      ``None`` — that is also a silent fallback, and is rejected here).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

__all__ = ["parse_time_hours", "parse_hydrograph_text"]


def parse_time_hours(token: str) -> float:
    """Parse a time token into hours (float).

    Accepted formats:
        * ``"0.5"`` (or any float-as-string)
        * ``"1:30"`` (HH:MM)
        * ``"1:30:30"`` (HH:MM:SS)

    Args:
        token: Raw time string from a workbench widget.

    Returns:
        Time in hours as a Python ``float``.

    Raises:
        ValueError: If ``token`` is empty, whitespace-only, or cannot be
            parsed as a float / HH:MM / HH:MM:SS token.
    """
    t = str(token).strip()
    if not t:
        raise ValueError("empty time token")
    if ":" in t:
        parts = t.split(":")
        if len(parts) == 2:
            try:
                hh = float(parts[0])
                mm = float(parts[1])
            except ValueError as exc:
                raise ValueError(
                    f"invalid HH:MM token '{t}': {exc}"
                ) from exc
            return hh + (mm / 60.0)
        if len(parts) == 3:
            try:
                hh = float(parts[0])
                mm = float(parts[1])
                ss = float(parts[2])
            except ValueError as exc:
                raise ValueError(
                    f"invalid HH:MM:SS token '{t}': {exc}"
                ) from exc
            return hh + (mm / 60.0) + (ss / 3600.0)
        raise ValueError(f"invalid HH:MM(:SS) token '{t}'")
    try:
        return float(t)
    except ValueError as exc:
        raise ValueError(f"invalid time token '{t}': {exc}") from exc


def parse_hydrograph_text(text: str) -> Tuple[np.ndarray, np.ndarray]:
    """Parse a hydrograph text block into ``(times_s, values)`` arrays.

    Format: ``"t1,v1;t2,v2;..."`` (commas OR ``=`` between time and
    value; ``;`` or ``\\n`` between entries). Time tokens follow
    :func:`parse_time_hours` semantics; values are parsed as ``float``.

    The returned arrays are 1-D ``float64``, sorted by time, and
    de-duplicated (entries whose times differ by less than ``1e-9``
    seconds collapse, keeping the last value).

    Args:
        text: Raw text from a workbench hydrograph editor.

    Returns:
        Tuple ``(times_s, values)`` where ``times_s`` is in **seconds**
        (NOT hours — the caller wanted seconds for the SWE solver).

    Raises:
        ValueError: If ``text`` is empty, contains no parseable entries,
            or any entry is malformed (missing separator, bad time
            token, or non-numeric value).
    """
    raw = str(text or "").strip()
    if not raw:
        raise ValueError("empty hydrograph text")

    pairs = []
    chunks = raw.replace("\n", ";").split(";")
    for chunk in chunks:
        c = chunk.strip()
        if not c:
            continue
        if "," in c:
            a, b = c.split(",", 1)
        elif "=" in c:
            a, b = c.split("=", 1)
        else:
            raise ValueError(
                f"hydrograph entry '{c}' must use ',' or '=' "
                "between time and value"
            )
        th = parse_time_hours(a.strip())
        try:
            vv = float(b.strip())
        except ValueError as exc:
            raise ValueError(
                f"invalid value in hydrograph entry '{c}': {exc}"
            ) from exc
        pairs.append((th * 3600.0, vv))

    if not pairs:
        raise ValueError("hydrograph text contained no parseable entries")

    pairs.sort(key=lambda x: x[0])

    uniq_t: list[float] = []
    uniq_v: list[float] = []
    for ti, vi in pairs:
        if uniq_t and abs(ti - uniq_t[-1]) < 1.0e-9:
            uniq_v[-1] = vi
        else:
            uniq_t.append(ti)
            uniq_v.append(vi)

    return (
        np.asarray(uniq_t, dtype=np.float64),
        np.asarray(uniq_v, dtype=np.float64),
    )
