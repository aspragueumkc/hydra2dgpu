#!/usr/bin/env python3
"""Rainfall + NRCS CN helpers for SWE2D and lumped workflows.

This module is intentionally lightweight and numpy-only so it can be used from
plugin UI code without introducing backend coupling.
"""

from __future__ import annotations
import logging

logger = logging.getLogger(__name__)

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def _to_float(value, default: float = 0.0) -> float:
    """to float"""
    try:
        return float(value)
    except Exception:
        return float(default)


def _normalize_text(value) -> str:
    """normalize text"""
    return str(value or "").strip().lower()


def _parse_time_to_seconds(value) -> Optional[float]:
    """parse time to seconds"""
    text = str(value or "").strip()
    if not text:
        return None

    lower = text.lower()
    for suffix, mult in (("minutes", 60.0), ("minute", 60.0), ("mins", 60.0), ("min", 60.0), ("m", 60.0),
                         ("hours", 3600.0), ("hour", 3600.0), ("hrs", 3600.0), ("hr", 3600.0), ("h", 3600.0),
                         ("seconds", 1.0), ("second", 1.0), ("secs", 1.0), ("sec", 1.0), ("s", 1.0)):
        if lower.endswith(suffix):
            raw = lower[: -len(suffix)].strip()
            try:
                return float(raw) * mult
            except Exception:
                return None

    try:
        return float(text) * 3600.0
    except Exception:
        pass

    # MM:SS or HH:MM:SS clock format
    parts = text.split(":")
    if len(parts) in (2, 3):
        try:
            hh = float(parts[0])
            mm = float(parts[1])
            ss = float(parts[2]) if len(parts) == 3 else 0.0
            return hh * 3600.0 + mm * 60.0 + ss
        except Exception:
            return None
    return None


def _convert_depth_to_mm(value: float, units: str) -> float:
    """convert depth to mm"""
    u = _normalize_text(units)
    if "in" in u:
        return value * 25.4
    if "cm" in u:
        return value * 10.0
    if "m" in u and "mm" not in u:
        return value * 1000.0
    return value


def _convert_intensity_to_mmph(value: float, units: str) -> float:
    """convert intensity to mmph"""
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
        """
        depth between mm.

        Parameters
        ----------
        t0_s : float
            Description of t0_s.
        t1_s : float
            Description of t1_s.

        Returns
        -------
        float
        """
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
    """
    build hyetograph.

    Parameters
    ----------
    rows : Iterable[Dict[str, object]]
        Description of rows.

    Returns
    -------
    Optional[Hyetograph]
    """
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


def inspect_hyetograph_rows(rows: Iterable[Dict[str, object]]) -> Dict[str, object]:
    """Return diagnostics describing how hyetograph rows will be interpreted."""
    parsed: List[Tuple[float, float, str, str]] = []
    total_rows = 0
    numeric_time_tokens = 0
    clock_time_tokens = 0
    suffixed_time_tokens = 0
    invalid_time_tokens = 0

    for row in rows:
        total_rows += 1
        t_raw = row.get("Time")
        t_s = _parse_time_to_seconds(t_raw)
        txt = str(t_raw or "").strip().lower()
        if isinstance(t_raw, (int, float)):
            numeric_time_tokens += 1
        elif ":" in txt:
            clock_time_tokens += 1
        elif any(txt.endswith(s) for s in ("minutes", "minute", "mins", "min", "m", "hours", "hour", "hrs", "hr", "h", "seconds", "second", "secs", "sec", "s")):
            suffixed_time_tokens += 1
        if t_s is None:
            invalid_time_tokens += 1
            continue

        value = _to_float(row.get("Value"), default=0.0)
        units = str(row.get("units") or row.get("Units") or "mm/hr")
        value_type = _normalize_text(row.get("value_type") or row.get("ValueType") or "intensity")
        parsed.append((t_s, value, value_type, units))

    out: Dict[str, object] = {
        "n_rows": int(total_rows),
        "n_valid": int(len(parsed)),
        "n_invalid_time": int(invalid_time_tokens),
        "numeric_time_tokens": int(numeric_time_tokens),
        "clock_time_tokens": int(clock_time_tokens),
        "suffixed_time_tokens": int(suffixed_time_tokens),
        "mode": "unknown",
        "units": "unknown",
        "t_start_s": 0.0,
        "t_end_s": 0.0,
        "dt_median_s": 0.0,
        "total_depth_mm": 0.0,
        "warnings": [],
    }

    if not parsed:
        out["warnings"] = ["no valid Time/Value rows"]
        return out

    parsed.sort(key=lambda x: x[0])
    times = np.array([p[0] for p in parsed], dtype=np.float64)
    values = np.array([p[1] for p in parsed], dtype=np.float64)
    mode = str(parsed[0][2])
    units = str(parsed[0][3])

    out["mode"] = mode
    out["units"] = units
    out["t_start_s"] = float(times[0])
    out["t_end_s"] = float(times[-1])
    if times.size > 1:
        out["dt_median_s"] = float(np.median(np.diff(times)))

    warnings: List[str] = []
    if mode in ("depth", "depth_inc", "increment", "incremental", "incremental_depth"):
        inc = np.array([_convert_depth_to_mm(p[1], p[3]) for p in parsed], dtype=np.float64)
        out["total_depth_mm"] = float(np.sum(np.maximum(inc, 0.0)))
    elif mode in ("depth_cum", "cumulative", "cumulative_depth"):
        cum = np.array([_convert_depth_to_mm(p[1], p[3]) for p in parsed], dtype=np.float64)
        out["total_depth_mm"] = float(np.max(np.maximum(cum, 0.0)))
    else:
        total = 0.0
        for i in range(1, times.size):
            dt_h = max(0.0, (times[i] - times[i - 1]) / 3600.0)
            mmph = _convert_intensity_to_mmph(values[i - 1], units)
            total += max(0.0, mmph * dt_h)
        out["total_depth_mm"] = float(total)

    if numeric_time_tokens > 0 and clock_time_tokens == 0 and suffixed_time_tokens == 0:
        warnings.append("numeric Time interpreted as hours; use HH:MM or explicit min/hr suffix to avoid ambiguity")

    if out["dt_median_s"] > 0.0 and out["dt_median_s"] < 10.0:
        warnings.append("very small time interval detected; verify Time units")

    out["warnings"] = warnings
    return out


def assign_cells_to_nearest_gauge(
    cell_x: np.ndarray,
    cell_y: np.ndarray,
    gauges: Sequence[Gauge],
) -> Optional[np.ndarray]:
    """
    assign cells to nearest gauge.

    Parameters
    ----------
    cell_x : np.ndarray
        Description of cell_x.
    cell_y : np.ndarray
        Description of cell_y.
    gauges : Sequence[Gauge]
        Description of gauges.

    Returns
    -------
    Optional[np.ndarray]
    """
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
    """
    scs retention mm.

    Parameters
    ----------
    curve_number : np.ndarray
        Description of curve_number.

    Returns
    -------
    np.ndarray
    """
    cn = np.clip(np.asarray(curve_number, dtype=np.float64), 1.0, 100.0)
    return np.maximum((25400.0 / cn) - 254.0, 0.0)


def scs_cumulative_excess_mm(cumulative_rain_mm: np.ndarray, curve_number: np.ndarray, ia_ratio: float = 0.2) -> np.ndarray:
    """
    scs cumulative excess mm.

    Parameters
    ----------
    cumulative_rain_mm : np.ndarray
        Description of cumulative_rain_mm.
    curve_number : np.ndarray
        Description of curve_number.
    ia_ratio : float
        Description of ia_ratio.

    Returns
    -------
    np.ndarray
    """
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
        """
        step.

        Parameters
        ----------
        rain_mm : np.ndarray
            Description of rain_mm.

        Returns
        -------
        np.ndarray
        """
        r = np.maximum(np.asarray(rain_mm, dtype=np.float64), 0.0)
        self.cumulative_rain_mm += r
        new_excess = scs_cumulative_excess_mm(self.cumulative_rain_mm, self.curve_number, ia_ratio=self.ia_ratio)
        inc_excess = np.maximum(new_excess - self.cumulative_excess_mm, 0.0)
        self.cumulative_excess_mm = new_excess
        return inc_excess

    def preview_step(self, rain_mm: np.ndarray) -> np.ndarray:
        """Compute incremental excess without mutating cumulative CN state."""
        r = np.maximum(np.asarray(rain_mm, dtype=np.float64), 0.0)
        preview_cum_rain = self.cumulative_rain_mm + r
        new_excess = scs_cumulative_excess_mm(preview_cum_rain, self.curve_number, ia_ratio=self.ia_ratio)
        return np.maximum(new_excess - self.cumulative_excess_mm, 0.0)


class ThiessenRainCNForcing:
    """Maps rain gages to cells (Thiessen nearest gage) and applies infiltration losses.

    ``infiltration_method`` controls which loss model is applied:

    * ``"scs_cn"``  – NRCS SCS Curve Number (default, original behaviour)
    * ``"none"``    – no infiltration; raw rainfall depth becomes runoff directly
    """

    #: Recognised infiltration method identifiers.
    INFILTRATION_METHODS = ("none", "scs_cn")

    def __init__(
        self,
        cell_to_gauge: np.ndarray,
        gauge_hyetographs: Dict[int, Hyetograph],
        curve_number: np.ndarray,
        ia_ratio: float = 0.2,
        infiltration_method: str = "scs_cn",
    ):
        self.cell_to_gauge = np.asarray(cell_to_gauge, dtype=np.int32)
        self.gauge_hyetographs = dict(gauge_hyetographs)
        self.infiltration_method = str(infiltration_method).lower().strip()
        self.curve_number = np.clip(np.asarray(curve_number, dtype=np.float64).ravel(), 1.0, 100.0)
        self.ia_ratio = float(ia_ratio)
        self.cn_model = SCSCurveNumberLoss(curve_number=self.curve_number, ia_ratio=self.ia_ratio)
        self._preprocessed_cache: Optional[Dict[str, np.ndarray]] = None

    def _build_preprocessed_cache(self) -> Dict[str, np.ndarray]:
        """build preprocessed cache"""
        cache = self._preprocessed_cache
        if cache is not None:
            return cache

        times_all: List[np.ndarray] = []
        for hy in self.gauge_hyetographs.values():
            t = np.asarray(getattr(hy, "times_s", []), dtype=np.float64).ravel()
            if t.size:
                times_all.append(t)
        if not times_all:
            times = np.asarray([0.0], dtype=np.float64)
        else:
            times = np.unique(np.concatenate(times_all))
            if times.size == 0 or float(times[0]) > 0.0:
                times = np.insert(times, 0, 0.0)
        times = np.asarray(times, dtype=np.float64)

        gauge_ids = sorted({int(g) for g in np.unique(self.cell_to_gauge).tolist() if int(g) >= 0})
        gauge_cum: Dict[int, np.ndarray] = {}
        for gi in gauge_ids:
            hy = self.gauge_hyetographs.get(int(gi))
            if hy is None:
                gauge_cum[int(gi)] = np.zeros(times.shape[0], dtype=np.float64)
                continue
            th = np.asarray(hy.times_s, dtype=np.float64).ravel()
            ch = np.asarray(hy.cumulative_mm, dtype=np.float64).ravel()
            if th.size == 0 or ch.size == 0:
                gauge_cum[int(gi)] = np.zeros(times.shape[0], dtype=np.float64)
                continue
            gauge_cum[int(gi)] = np.interp(times, th, ch, left=0.0, right=float(ch[-1]))

        key_to_group: Dict[Tuple[object, ...], int] = {}
        group_cum: List[np.ndarray] = []
        cell_group_idx = np.full(self.cell_to_gauge.shape[0], -1, dtype=np.int32)

        zeros_curve = np.zeros(times.shape[0], dtype=np.float64)
        for i in range(self.cell_to_gauge.shape[0]):
            gi = int(self.cell_to_gauge[i])
            if gi < 0:
                key: Tuple[object, ...] = ("off",)
            elif self.infiltration_method == "none":
                key = ("none", gi)
            else:
                key = ("scs_cn", gi, float(self.curve_number[i]))

            gid = key_to_group.get(key)
            if gid is None:
                if gi < 0:
                    curve = zeros_curve.copy()
                else:
                    rain_curve = np.asarray(gauge_cum.get(gi, zeros_curve), dtype=np.float64)
                    if self.infiltration_method == "none":
                        curve = rain_curve.copy()
                    else:
                        cn_curve = np.full(rain_curve.shape, float(self.curve_number[i]), dtype=np.float64)
                        curve = scs_cumulative_excess_mm(
                            rain_curve,
                            cn_curve,
                            ia_ratio=self.ia_ratio,
                        )
                gid = len(group_cum)
                key_to_group[key] = gid
                group_cum.append(np.asarray(curve, dtype=np.float64))
            cell_group_idx[i] = int(gid)

        if not group_cum:
            group_cum_arr = np.zeros((1, times.shape[0]), dtype=np.float64)
            cell_group_idx = np.zeros(self.cell_to_gauge.shape[0], dtype=np.int32)
        else:
            group_cum_arr = np.vstack(group_cum).astype(np.float64, copy=False)

        cache = {
            "times_s": times,
            "group_cumulative_excess_mm": group_cum_arr,
            "cell_group_idx": cell_group_idx,
        }
        self._preprocessed_cache = cache
        return cache

    @staticmethod
    def _interp_group_cumulative_mm(times_s: np.ndarray, group_cum_mm: np.ndarray, t_s: float) -> np.ndarray:
        """interp group cumulative mm"""
        t = float(t_s)
        if times_s.size <= 1:
            return np.asarray(group_cum_mm[:, 0], dtype=np.float64)
        if t <= float(times_s[0]):
            return np.asarray(group_cum_mm[:, 0], dtype=np.float64)
        if t >= float(times_s[-1]):
            return np.asarray(group_cum_mm[:, -1], dtype=np.float64)
        j = int(np.searchsorted(times_s, t, side="right") - 1)
        j = max(0, min(j, int(times_s.size) - 2))
        t0 = float(times_s[j])
        t1 = float(times_s[j + 1])
        if t1 <= t0:
            return np.asarray(group_cum_mm[:, j], dtype=np.float64)
        w = (t - t0) / (t1 - t0)
        return (1.0 - w) * group_cum_mm[:, j] + w * group_cum_mm[:, j + 1]

    def build_native_preprocessed_payload(self) -> Dict[str, np.ndarray]:
        """Return preprocessed excess-hyetograph payload compatible with native API.

        The native backend currently accepts rainfall+CN inputs. We provide
        precomputed cumulative excess as pseudo-rainfall and force identity loss
        (CN=100, Ia=0) so runtime infiltration is not re-applied.
        """
        cache = self._build_preprocessed_cache()
        times = np.asarray(cache["times_s"], dtype=np.float64).ravel()
        group_cum = np.asarray(cache["group_cumulative_excess_mm"], dtype=np.float64)
        cell_group_idx = np.asarray(cache["cell_group_idx"], dtype=np.int32).ravel()

        n_groups = int(group_cum.shape[0])
        nt = int(times.size)
        if n_groups <= 0 or nt <= 0:
            return {
                "cell_gage_idx": np.full(cell_group_idx.shape[0], -1, dtype=np.int32),
                "gage_offsets": np.asarray([0], dtype=np.int32),
                "hg_time_s": np.asarray([0.0], dtype=np.float64),
                "hg_cum_mm": np.asarray([0.0], dtype=np.float64),
                "cn": np.full(cell_group_idx.shape[0], 100.0, dtype=np.float64),
                "ia_ratio": np.asarray([0.0], dtype=np.float64),
            }

        gage_offsets = np.arange(0, (n_groups + 1) * nt, nt, dtype=np.int32)
        hg_time_s = np.tile(times, n_groups).astype(np.float64, copy=False)
        hg_cum_mm = group_cum.reshape(-1).astype(np.float64, copy=False)
        cn = np.full(cell_group_idx.shape[0], 100.0, dtype=np.float64)
        ia_ratio = np.asarray([0.0], dtype=np.float64)
        return {
            "cell_gage_idx": cell_group_idx.astype(np.int32, copy=False),
            "gage_offsets": gage_offsets,
            "hg_time_s": hg_time_s,
            "hg_cum_mm": hg_cum_mm,
            "cn": cn,
            "ia_ratio": ia_ratio,
        }

    def _window_rain_mm(self, t0_s: float, t1_s: float) -> Tuple[np.ndarray, float]:
        """window rain mm"""
        dt_s = max(1.0e-9, float(t1_s) - float(t0_s))
        rain_mm = np.zeros(self.cell_to_gauge.shape[0], dtype=np.float64)

        unique_gauges = np.unique(self.cell_to_gauge)
        for gi in unique_gauges:
            hy = self.gauge_hyetographs.get(int(gi))
            if hy is None:
                continue
            depth_mm = hy.depth_between_mm(t0_s, t1_s)
            rain_mm[self.cell_to_gauge == int(gi)] = max(0.0, depth_mm)

        return rain_mm, dt_s

    def step_net_rainfall_mps(self, t0_s: float, t1_s: float, mutate_state: bool = True) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        step net rainfall mps.

        Parameters
        ----------
        t0_s : float
            Description of t0_s.
        t1_s : float
            Description of t1_s.
        mutate_state : bool
            Description of mutate_state.

        Returns
        -------
        Tuple[np.ndarray, Dict[str, float]]
        """
        rain_mm, dt_s = self._window_rain_mm(t0_s, t1_s)
        cache = self._build_preprocessed_cache()
        times = np.asarray(cache["times_s"], dtype=np.float64)
        group_cum = np.asarray(cache["group_cumulative_excess_mm"], dtype=np.float64)
        cell_group = np.asarray(cache["cell_group_idx"], dtype=np.int32)

        g0 = self._interp_group_cumulative_mm(times, group_cum, float(t0_s))
        g1 = self._interp_group_cumulative_mm(times, group_cum, float(t1_s))
        group_inc = np.maximum(g1 - g0, 0.0)
        excess_mm = np.maximum(group_inc[cell_group], 0.0)

        rate_mps = (excess_mm / 1000.0) / dt_s

        stats = {
            "rain_mm_mean": float(np.mean(rain_mm)) if rain_mm.size else 0.0,
            "excess_mm_mean": float(np.mean(excess_mm)) if excess_mm.size else 0.0,
            "rain_mm_max": float(np.max(rain_mm)) if rain_mm.size else 0.0,
            "excess_mm_max": float(np.max(excess_mm)) if excess_mm.size else 0.0,
        }
        return rate_mps, stats


def runoff_depth_mm_from_event_rain_mm(rain_depth_mm: float, curve_number: float, ia_ratio: float = 0.2) -> float:
    """
    runoff depth mm from event rain mm.

    Parameters
    ----------
    rain_depth_mm : float
        Description of rain_depth_mm.
    curve_number : float
        Description of curve_number.
    ia_ratio : float
        Description of ia_ratio.

    Returns
    -------
    float
    """
    p = max(0.0, float(rain_depth_mm))
    cn = float(np.clip(curve_number, 1.0, 100.0))
    s = float(scs_retention_mm(np.array([cn], dtype=np.float64))[0])
    ia = float(ia_ratio) * s
    if p <= ia:
        return 0.0
    return float(((p - ia) ** 2.0) / max(p + (1.0 - float(ia_ratio)) * s, 1.0e-12))


def composite_curve_number(area_cn_pairs: Sequence[Tuple[float, float]]) -> float:
    """
    composite curve number.

    Parameters
    ----------
    area_cn_pairs : Sequence[Tuple[float, float]]
        Description of area_cn_pairs.

    Returns
    -------
    float
    """
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
