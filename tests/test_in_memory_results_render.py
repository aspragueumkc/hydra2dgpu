"""Tests for in-memory results rendering path."""
import numpy as np
from swe2d.results.data import SWE2DResultsData
from swe2d.results.queries import load_timeseries_from_live
from swe2d.results.queries import load_profile_from_live


def test_load_timeseries_from_live_empty():
    data = SWE2DResultsData()
    result = load_timeseries_from_live(data, "run_1", 0)
    assert result == {}


def test_load_timeseries_from_live_with_rows():
    data = SWE2DResultsData()
    data.append_line_snapshot({"t_s": 0.0, "line_id": 0, "depth_m": 1.0,
                               "velocity_ms": 0.5, "wse_m": 11.0,
                               "bed_m": 10.0, "flow_cms": 5.0,
                               "run_id": "run_1"})
    data.append_line_snapshot({"t_s": 1.0, "line_id": 0, "depth_m": 1.2,
                               "velocity_ms": 0.6, "wse_m": 11.2,
                               "bed_m": 10.0, "flow_cms": 6.0,
                               "run_id": "run_1"})
    result = load_timeseries_from_live(data, "run_1", 0)
    assert "t_s" in result
    assert len(result["t_s"]) == 2
    np.testing.assert_almost_equal(result["t_s"], [0.0, 1.0])


def test_load_profile_from_live_empty():
    data = SWE2DResultsData()
    result = load_profile_from_live(data, "run_1", 0, 0.0)
    assert result == {}


def test_load_profile_from_live_with_rows():
    data = SWE2DResultsData()
    raw_bytes = np.array([[0.0, 11.0, 10.0, 1.0],
                          [1.0, 11.2, 10.0, 1.2],
                          [2.0, 11.5, 10.0, 1.5]], dtype=np.float64).tobytes()
    data.append_line_profile_snapshot({"t_s": 1.0, "line_id": 0,
                                       "data": raw_bytes,
                                       "run_id": "run_1"})
    result = load_profile_from_live(data, "run_1", 0, 1.0)
    assert "dist_m" in result
    assert len(result["dist_m"]) == 3
    np.testing.assert_almost_equal(result["depth_m"], [1.0, 1.2, 1.5])
