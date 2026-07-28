import unittest
"""Tests for in-memory results rendering path."""
import numpy as np
from swe2d.results.data import SWE2DResultsData
from swe2d.results.queries import load_timeseries_from_live
from swe2d.results.queries import load_profile_from_live


def test_load_timeseries_from_live_empty():
    data = SWE2DResultsData()
    result = load_timeseries_from_live(data, "run_1", 0)
    assert result == {}


def _make_data_with_line_ts():
    data = SWE2DResultsData()
    data._live_run_id = "run_1"
    # Two mesh snapshots at t=0 and t=1
    h0 = np.array([1.0, 2.0], dtype=np.float64)
    h1 = np.array([1.2, 2.2], dtype=np.float64)
    zeros = np.zeros_like(h0)
    data.set_live_snapshot_timesteps([
        (0.0, h0, zeros, zeros),
        (1.0, h1, zeros, zeros),
    ])
    sample_map = [{
        "line_id": 0,
        "line_name": "Line A",
        "cell_idx": np.array([0, 1], dtype=np.int32),
    }]

    def fake_sampler(sm, t, h, hu, hv, cell_bed):
        ts_rows, prof_rows = [], []
        for m in sm:
            idx = np.asarray(m["cell_idx"], dtype=np.int32)
            ts_rows.append({
                "line_id": int(m["line_id"]),
                "line_name": m["line_name"],
                "depth": float(np.mean(h[idx])),
                "velocity": 0.0,
                "wse": float(np.mean(h[idx])),
                "bed": 0.0,
                "flow": 0.0,
                "wet_frac": 1.0,
                "fr": 0.0,
            })
        return ts_rows, prof_rows

    data.populate_live_line_metrics(
        sample_map=sample_map,
        sample_callback=fake_sampler,
        cell_solver_z=np.zeros_like(h0),
    )
    return data


def test_load_timeseries_from_live_with_rows():
    data = _make_data_with_line_ts()
    result = load_timeseries_from_live(data, "run_1", 0)
    assert "t_s" in result
    assert len(result["t_s"]) == 2
    np.testing.assert_almost_equal(result["t_s"], [0.0, 1.0])
    np.testing.assert_almost_equal(result["depth"], [1.5, 1.7])


def test_load_profile_from_live_empty():
    data = SWE2DResultsData()
    result = load_profile_from_live(data, "run_1", 0, 0.0)
    assert result == {}


def test_load_profile_from_live_with_rows():
    data = SWE2DResultsData()
    data._live_run_id = "run_1"
    h0 = np.array([1.0, 2.0], dtype=np.float64)
    h1 = np.array([1.2, 2.2], dtype=np.float64)
    zeros = np.zeros_like(h0)
    data.set_live_snapshot_timesteps([
        (0.0, h0, zeros, zeros),
        (1.0, h1, zeros, zeros),
    ])
    sample_map = [{
        "line_id": 0,
        "line_name": "Line A",
        "cell_idx": np.array([0, 1], dtype=np.int32),
        "station": np.array([0.0, 5.0], dtype=np.float64),
    }]

    def fake_sampler(sm, t, h, hu, hv, cell_bed):
        ts_rows, prof_rows = [], []
        for m in sm:
            idx = np.asarray(m["cell_idx"], dtype=np.int32)
            prof_rows.append({
                "line_id": int(m["line_id"]),
                "line_name": m["line_name"],
                "station": np.asarray(m["station"], dtype=np.float64),
                "depth": h[idx].astype(np.float64),
                "velocity": np.zeros_like(h[idx], dtype=np.float64),
                "wse": h[idx].astype(np.float64),
                "bed": np.zeros_like(h[idx], dtype=np.float64),
                "flow_qn": np.zeros_like(h[idx], dtype=np.float64),
                "fr": np.zeros_like(h[idx], dtype=np.float64),
                "wet": np.ones_like(h[idx], dtype=np.int32),
            })
        return ts_rows, prof_rows

    data.populate_live_line_metrics(
        sample_map=sample_map,
        sample_callback=fake_sampler,
        cell_solver_z=np.zeros_like(h0),
    )

    result = load_profile_from_live(data, "run_1", 0, 1.0)
    assert "station" in result
    np.testing.assert_almost_equal(result["station"], [0.0, 5.0])
    np.testing.assert_almost_equal(result["depth"], [1.2, 2.2])

    result0 = load_profile_from_live(data, "run_1", 0, 0.0)
    np.testing.assert_almost_equal(result0["depth"], [1.0, 2.0])

class _PytestStyleWrapper(unittest.TestCase):
    """Auto-generated wrapper for module-level test functions.

    Created by tools/wrap_pytest_style.py so that pytest-style tests
    (def test_* at module level) become visible to `python3 -m unittest`.
    Each module-level test is attached as a staticmethod so it can be
    discovered and run as a unittest TestCase.
    """
__wrapped_funcs = []
for _name, _obj in list(globals().items()):
    if _name.startswith("test_") and callable(_obj) and not isinstance(_obj, type):
        setattr(_PytestStyleWrapper, _name, staticmethod(_obj))
        __wrapped_funcs.append(_name)
for _name in __wrapped_funcs:
    del globals()[_name]
del _name, _obj, __wrapped_funcs
