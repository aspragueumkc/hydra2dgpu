#!/usr/bin/env python3
"""Rainfall + NRCS CN helpers for SWE2D and lumped workflows.

This module is intentionally lightweight and numpy-only so it can be used from
plugin UI code without introducing backend coupling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def _to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _normalize_text(value) -> str:
    return str(value or "").strip().lower()


def _parse_time_to_seconds(value) -> Optional[float]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text) * 3600.0
    except Exception:
        pass

    parts = text.split(":")
    if len(parts) not in (2, 3):
        return None
    try:
        hh = float(parts[0])
        mm = float(parts[1])
        ss = float(parts[2]) if len(parts) == 3 else 0.0
        return hh * 3600.0 + mm * 60.0 + ss
    except Exception:
        return None


def _convert_depth_to_mm(value: float, units: str) -> float:
    u = _normalize_text(units)
    if "in" in u:
        return value * 25.4
    if "cm" in u:
        return value * 10.0
    if "m" in u and "mm" not in u:
        return value * 1000.0
    return value


def _convert_intensity_to_mmph(value: float, units: str) -> float:
    u = _normalize_text(units)
    if "in" in u:
        return value * 25.4
    if "cm" in u:
        return value * 10.0
    if "m" in u and "mm" not in u:
        return value * 1000.0
    return value


@dataclass
class Hyetograph:
    """Piecewise-linear cumulative rainfall curve in mm."""

    times_s: np.ndarray
    cumulative_mm: np.ndarray

    def depth_between_mm(self, t0_s: float, t1_s: float) -> float:
        t0 = float(max(0.0, t0_s))
        t1 = float(max(t0, t1_s))
        if self.times_s.size == 0:
            return 0.0

        c0 = float(np.interp(t0, self.times_s, self.cumulative_mm, left=0.0, right=float(self.cumulative_mm[-1])))
        c1 = float(np.interp(t1, self.times_s, self.cumulative_mm, left=0.0, right=float(self.cumulative_mm[-1])))
        return max(0.0, c1 - c0)


@dataclass
class Gauge:
    gauge_id: str
    x: float
    y: float
    hyetograph_id: str


def build_hyetograph(rows: Iterable[Dict[str, object]]) -> Optional[Hyetograph]:
    parsed: List[Tuple[float, float, str, str]] = []
    for row in rows:
        t_s = _parse_time_to_seconds(row.get("Time"))
        if t_s is None:
            continue
        value = _to_float(row.get("Value"), default=0.0)
        units = str(row.get("units") or row.get("Units") or "mm/hr")
        value_type = _normalize_text(row.get("value_type") or row.get("ValueType") or "intensity")
        parsed.append((t_s, value, value_type, units))

    if not parsed:
        return None

    parsed.sort(key=lambda x: x[0])
    times = np.array([p[0] for p in parsed], dtype=np.float64)

    if times[0] > 0.0:
        times = np.insert(times, 0, 0.0)
        first = parsed[0]
        parsed.insert(0, (0.0, first[1], first[2], first[3]))

    mode = parsed[0][2]
    if mode in ("depth", "depth_inc", "increment", "incremental", "incremental_depth"):
        inc = np.array([_convert_depth_to_mm(p[1], p[3]) for p in parsed], dtype=np.float64)
        cumulative = np.cumsum(np.maximum(inc, 0.0))
    elif mode in ("depth_cum", "cumulative", "cumulative_depth"):
        cumulative = np.array([_convert_depth_to_mm(p[1], p[3]) for p in parsed], dtype=np.float64)
        cumulative = np.maximum.accumulate(np.maximum(cumulative, 0.0))
    else:
        # Intensity/rate modes are interpreted as stepwise values over each interval.
        cumulative = np.zeros(times.shape[0], dtype=np.float64)
        for i in range(1, times.shape[0]):
            dt_h = max(0.0, (times[i] - times[i - 1]) / 3600.0)
            mmph = _convert_intensity_to_mmph(parsed[i - 1][1], parsed[i - 1][3])
            cumulative[i] = cumulative[i - 1] + max(0.0, mmph * dt_h)

    return Hyetograph(times_s=times, cumulative_mm=cumulative)


def assign_cells_to_nearest_gauge(
    cell_x: np.ndarray,
    cell_y: np.ndarray,
    gauges: Sequence[Gauge],
) -> Optional[np.ndarray]:
    if not gauges:
        return None

    gx = np.array([g.x for g in gauges], dtype=np.float64)
    gy = np.array([g.y for g in gauges], dtype=np.float64)
    out = np.zeros(cell_x.shape[0], dtype=np.int32)

    for i in range(cell_x.shape[0]):
        dx = gx - float(cell_x[i])
        dy = gy - float(cell_y[i])
        out[i] = int(np.argmin(dx * dx + dy * dy))
    return out


def scs_retention_mm(curve_number: np.ndarray) -> np.ndarray:
    cn = np.clip(np.asarray(curve_number, dtype=np.float64), 1.0, 100.0)
    return np.maximum((25400.0 / cn) - 254.0, 0.0)


def scs_cumulative_excess_mm(cumulative_rain_mm: np.ndarray, curve_number: np.ndarray, ia_ratio: float = 0.2) -> np.ndarray:
    p = np.maximum(np.asarray(cumulative_rain_mm, dtype=np.float64), 0.0)
    s = scs_retention_mm(curve_number)
    ia = float(ia_ratio) * s
    pe = np.zeros_like(p)
    wet = p > ia
    pe[wet] = ((p[wet] - ia[wet]) ** 2.0) / np.maximum(p[wet] + (1.0 - float(ia_ratio)) * s[wet], 1.0e-12)
    return np.maximum(pe, 0.0)


class SCSCurveNumberLoss:
    """Stateful incremental excess calculator for event simulations."""

    def __init__(self, curve_number: np.ndarray, ia_ratio: float = 0.2):
        self.curve_number = np.clip(np.asarray(curve_number, dtype=np.float64), 1.0, 100.0)
        self.ia_ratio = float(ia_ratio)
        self.cumulative_rain_mm = np.zeros(self.curve_number.shape[0], dtype=np.float64)
        self.cumulative_excess_mm = np.zeros(self.curve_number.shape[0], dtype=np.float64)

    def step(self, rain_mm: np.ndarray) -> np.ndarray:
        r = np.maximum(np.asarray(rain_mm, dtype=np.float64), 0.0)
        self.cumulative_rain_mm += r
        new_excess = scs_cumulative_excess_mm(self.cumulative_rain_mm, self.curve_number, ia_ratio=self.ia_ratio)
        inc_excess = np.maximum(new_excess - self.cumulative_excess_mm, 0.0)
        self.cumulative_excess_mm = new_excess
        return inc_excess


class ThiessenRainCNForcing:
    """Maps rain gages to cells (Thiessen nearest gage) and applies CN losses."""

    def __init__(
        self,
        cell_to_gauge: np.ndarray,
        gauge_hyetographs: Dict[int, Hyetograph],
        curve_number: np.ndarray,
        ia_ratio: float = 0.2,
    ):
        self.cell_to_gauge = np.asarray(cell_to_gauge, dtype=np.int32)
        self.gauge_hyetographs = dict(gauge_hyetographs)
        self.cn_model = SCSCurveNumberLoss(curve_number=np.asarray(curve_number, dtype=np.float64), ia_ratio=ia_ratio)

    def step_net_rainfall_mps(self, t0_s: float, t1_s: float) -> Tuple[np.ndarray, Dict[str, float]]:
        dt_s = max(1.0e-9, float(t1_s) - float(t0_s))
        rain_mm = np.zeros(self.cell_to_gauge.shape[0], dtype=np.float64)

        unique_gauges = np.unique(self.cell_to_gauge)
        for gi in unique_gauges:
            hy = self.gauge_hyetographs.get(int(gi))
            if hy is None:
                continue
            depth_mm = hy.depth_between_mm(t0_s, t1_s)
            rain_mm[self.cell_to_gauge == int(gi)] = max(0.0, depth_mm)

        excess_mm = self.cn_model.step(rain_mm)
        rate_mps = (excess_mm / 1000.0) / dt_s

        stats = {
            "rain_mm_mean": float(np.mean(rain_mm)) if rain_mm.size else 0.0,
            "excess_mm_mean": float(np.mean(excess_mm)) if excess_mm.size else 0.0,
            "rain_mm_max": float(np.max(rain_mm)) if rain_mm.size else 0.0,
            "excess_mm_max": float(np.max(excess_mm)) if excess_mm.size else 0.0,
        }
        return rate_mps, stats


def runoff_depth_mm_from_event_rain_mm(rain_depth_mm: float, curve_number: float, ia_ratio: float = 0.2) -> float:
    p = max(0.0, float(rain_depth_mm))
    cn = float(np.clip(curve_number, 1.0, 100.0))
    s = float(scs_retention_mm(np.array([cn], dtype=np.float64))[0])
    ia = float(ia_ratio) * s
    if p <= ia:
        return 0.0
    return float(((p - ia) ** 2.0) / max(p + (1.0 - float(ia_ratio)) * s, 1.0e-12))


def composite_curve_number(area_cn_pairs: Sequence[Tuple[float, float]]) -> float:
    if not area_cn_pairs:
        return 75.0
    num = 0.0
    den = 0.0
    for area, cn in area_cn_pairs:
        a = max(0.0, float(area))
        c = float(np.clip(cn, 1.0, 100.0))
        num += a * c
        den += a
    if den <= 0.0:
        return 75.0
    return num / den


def time_of_concentration_hours_velocity_method(segments: Sequence[Tuple[float, float]]) -> float:
    """NRCS velocity method in direct form: Tc = sum(L / V).

    Input segments are (length_m, velocity_mps) tuples.
    """
    total_s = 0.0
    for length_m, velocity_mps in segments:
        l = max(0.0, float(length_m))
        v = max(1.0e-6, float(velocity_mps))
        total_s += l / v
    return total_s / 3600.0
