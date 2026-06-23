"""Tests for swe2d.results.timestep_service — pure data logic, no Qt."""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from swe2d.results.timestep_service import (
    compute_timestep_union,
    time_sec_to_frame_idx,
    frame_idx_to_time_sec,
    load_timesteps,
    load_line_timesteps,
    load_coupling_for_run,
)


# ---------------------------------------------------------------------------
# compute_timestep_union
# ---------------------------------------------------------------------------

class TestComputeTimestepUnion(unittest.TestCase):
    def test_empty_list_returns_empty_array(self):
        result = compute_timestep_union([])
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.float64)
        self.assertEqual(result.size, 0)

    def test_single_array(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        result = compute_timestep_union([arr])
        np.testing.assert_array_equal(result, arr)

    def test_union_of_multiple_arrays(self):
        a = np.array([1.0, 3.0, 5.0], dtype=np.float64)
        b = np.array([2.0, 4.0, 6.0], dtype=np.float64)
        result = compute_timestep_union([a, b])
        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
        np.testing.assert_array_equal(result, expected)

    def test_deduplicates_overlapping_values(self):
        a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        b = np.array([2.0, 3.0, 4.0], dtype=np.float64)
        result = compute_timestep_union([a, b])
        expected = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        np.testing.assert_array_equal(result, expected)

    def test_sorts_result(self):
        a = np.array([5.0, 1.0, 3.0], dtype=np.float64)
        b = np.array([4.0, 2.0, 0.0], dtype=np.float64)
        result = compute_timestep_union([a, b])
        expected = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        np.testing.assert_array_equal(result, expected)

    def test_preserves_float64_dtype(self):
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.array([3.0, 4.0], dtype=np.float64)
        result = compute_timestep_union([a, b])
        self.assertEqual(result.dtype, np.float64)


# ---------------------------------------------------------------------------
# time_sec_to_frame_idx
# ---------------------------------------------------------------------------

class TestTimeSecToFrameIdx(unittest.TestCase):
    def test_empty_timesteps_returns_zero(self):
        idx = time_sec_to_frame_idx(10.0, np.empty(0, dtype=np.float64))
        self.assertEqual(idx, 0)

    def test_exact_match(self):
        ts = np.array([0.0, 5.0, 10.0, 15.0], dtype=np.float64)
        idx = time_sec_to_frame_idx(10.0, ts)
        self.assertEqual(idx, 2)

    def test_nearest_rounds_down(self):
        ts = np.array([0.0, 5.0, 10.0, 15.0], dtype=np.float64)
        idx = time_sec_to_frame_idx(6.0, ts)
        self.assertEqual(idx, 1)

    def test_nearest_rounds_up(self):
        ts = np.array([0.0, 5.0, 10.0, 15.0], dtype=np.float64)
        idx = time_sec_to_frame_idx(7.0, ts)
        self.assertEqual(idx, 1)

    def test_below_range_returns_first(self):
        ts = np.array([5.0, 10.0, 15.0], dtype=np.float64)
        idx = time_sec_to_frame_idx(0.0, ts)
        self.assertEqual(idx, 0)

    def test_above_range_returns_last(self):
        ts = np.array([5.0, 10.0, 15.0], dtype=np.float64)
        idx = time_sec_to_frame_idx(100.0, ts)
        self.assertEqual(idx, 2)

    def test_single_element(self):
        ts = np.array([42.0], dtype=np.float64)
        idx = time_sec_to_frame_idx(100.0, ts)
        self.assertEqual(idx, 0)


# ---------------------------------------------------------------------------
# frame_idx_to_time_sec
# ---------------------------------------------------------------------------

class TestFrameIdxToTimeSec(unittest.TestCase):
    def test_empty_timesteps_returns_zero(self):
        val = frame_idx_to_time_sec(0, np.empty(0, dtype=np.float64))
        self.assertEqual(val, 0.0)

    def test_valid_index(self):
        ts = np.array([0.0, 5.0, 10.0, 15.0], dtype=np.float64)
        val = frame_idx_to_time_sec(2, ts)
        self.assertEqual(val, 10.0)

    def test_negative_index_clamps_to_first(self):
        ts = np.array([5.0, 10.0, 15.0], dtype=np.float64)
        val = frame_idx_to_time_sec(-1, ts)
        self.assertEqual(val, 5.0)

    def test_overshoot_index_clamps_to_last(self):
        ts = np.array([5.0, 10.0, 15.0], dtype=np.float64)
        val = frame_idx_to_time_sec(999, ts)
        self.assertEqual(val, 15.0)

    def test_first_index(self):
        ts = np.array([5.0, 10.0, 15.0], dtype=np.float64)
        val = frame_idx_to_time_sec(0, ts)
        self.assertEqual(val, 5.0)

    def test_last_index(self):
        ts = np.array([5.0, 10.0, 15.0], dtype=np.float64)
        val = frame_idx_to_time_sec(2, ts)
        self.assertEqual(val, 15.0)

    def test_single_element(self):
        ts = np.array([42.0], dtype=np.float64)
        val = frame_idx_to_time_sec(0, ts)
        self.assertEqual(val, 42.0)


# ---------------------------------------------------------------------------
# load_timesteps — DB-backed
# ---------------------------------------------------------------------------

class TestLoadTimesteps(unittest.TestCase):
    @patch("swe2d.results.timestep_service._resolve_ts_table")
    @patch("swe2d.results.timestep_service._open_ro")
    def test_shared_schema(self, mock_open_ro, mock_resolve):
        conn = MagicMock()
        mock_open_ro.return_value = conn
        mock_resolve.return_value = ("swe2d_line_results_ts", True)

        cur = MagicMock()
        conn.execute.return_value = cur
        cur.fetchall.return_value = [(0.0,), (5.0,), (10.0,)]

        result = load_timesteps("/test.gpkg", "run_abc")
        expected = np.array([0.0, 5.0, 10.0], dtype=np.float64)
        np.testing.assert_array_equal(result, expected)

        conn.execute.assert_called_once()
        call_sql = conn.execute.call_args[0][0]
        self.assertIn("run_id", call_sql.lower())
        self.assertIn("WHERE", call_sql)
        conn.close.assert_called_once()

    @patch("swe2d.results.timestep_service._resolve_ts_table")
    @patch("swe2d.results.timestep_service._open_ro")
    def test_legacy_schema(self, mock_open_ro, mock_resolve):
        conn = MagicMock()
        mock_open_ro.return_value = conn
        mock_resolve.return_value = ("swe2d_line_results_ts_run_abc", False)

        cur = MagicMock()
        conn.execute.return_value = cur
        cur.fetchall.return_value = [(1.0,), (2.0,)]

        result = load_timesteps("/test.gpkg", "run_abc")
        expected = np.array([1.0, 2.0], dtype=np.float64)
        np.testing.assert_array_equal(result, expected)

        conn.execute.assert_called_once()
        call_sql = conn.execute.call_args[0][0]
        self.assertNotIn("run_id", call_sql.lower())
        conn.close.assert_called_once()

    @patch("swe2d.results.timestep_service._resolve_ts_table")
    @patch("swe2d.results.timestep_service._open_ro")
    def test_no_table_returns_empty(self, mock_open_ro, mock_resolve):
        conn = MagicMock()
        mock_open_ro.return_value = conn
        mock_resolve.return_value = ("", False)

        result = load_timesteps("/test.gpkg", "run_abc")
        self.assertEqual(result.size, 0)
        conn.close.assert_called_once()

    @patch("swe2d.results.timestep_service._resolve_ts_table")
    @patch("swe2d.results.timestep_service._open_ro")
    def test_open_failure_returns_empty(self, mock_open_ro, mock_resolve):
        mock_open_ro.return_value = None
        result = load_timesteps("/test.gpkg", "run_abc")
        self.assertEqual(result.size, 0)
        mock_resolve.assert_not_called()

    @patch("swe2d.results.timestep_service._resolve_ts_table")
    @patch("swe2d.results.timestep_service._open_ro")
    def test_exception_returns_empty(self, mock_open_ro, mock_resolve):
        conn = MagicMock()
        mock_open_ro.return_value = conn
        mock_resolve.side_effect = Exception("DB error")

        result = load_timesteps("/test.gpkg", "run_abc")
        self.assertEqual(result.size, 0)
        conn.close.assert_called_once()


# ---------------------------------------------------------------------------
# load_line_timesteps — DB-backed
# ---------------------------------------------------------------------------

class TestLoadLineTimesteps(unittest.TestCase):
    @patch("swe2d.results.timestep_service._open_ro")
    def test_shared_schema(self, mock_open_ro):
        conn = MagicMock()
        mock_open_ro.return_value = conn

        # First query: sqlite_master to find tables
        cur_tables = MagicMock()
        cur_tables.fetchall.return_value = [("swe2d_line_results_ts",)]

        # Second query: timesteps for line
        cur_ts = MagicMock()
        cur_ts.fetchall.return_value = [(0.0,), (5.0,)]

        conn.execute.side_effect = [cur_tables, cur_ts]

        result = load_line_timesteps("/test.gpkg", 3)
        expected = np.array([0.0, 5.0], dtype=np.float64)
        np.testing.assert_array_equal(result, expected)

        # First call should be sqlite_master, second should filter by line_id
        self.assertIn("sqlite_master", conn.execute.call_args_list[0][0][0].lower())
        self.assertIn("line_id", conn.execute.call_args_list[1][0][0].lower())
        conn.close.assert_called_once()

    @patch("swe2d.results.timestep_service._open_ro")
    def test_open_failure_returns_empty(self, mock_open_ro):
        mock_open_ro.return_value = None
        result = load_line_timesteps("/test.gpkg", 3)
        self.assertEqual(result.size, 0)

    @patch("swe2d.results.timestep_service._open_ro")
    def test_empty_tables_returns_empty(self, mock_open_ro):
        conn = MagicMock()
        mock_open_ro.return_value = conn
        cur = MagicMock()
        cur.fetchall.return_value = []
        conn.execute.return_value = cur
        result = load_line_timesteps("/test.gpkg", 3)
        self.assertEqual(result.size, 0)
        conn.close.assert_called_once()

    @patch("swe2d.results.timestep_service._open_ro")
    def test_union_from_multiple_legacy_tables(self, mock_open_ro):
        conn = MagicMock()
        mock_open_ro.return_value = conn

        cur_tables = MagicMock()
        cur_tables.fetchall.return_value = [
            ("swe2d_line_results_ts_run_a",),
            ("swe2d_line_results_ts_run_b",),
        ]
        cur_ts_a = MagicMock()
        cur_ts_a.fetchall.return_value = [(1.0,), (3.0,)]
        cur_ts_b = MagicMock()
        cur_ts_b.fetchall.return_value = [(2.0,), (3.0,)]

        conn.execute.side_effect = [cur_tables, cur_ts_a, cur_ts_b]

        result = load_line_timesteps("/test.gpkg", 3)
        expected = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        np.testing.assert_array_equal(result, expected)
        conn.close.assert_called_once()


# ---------------------------------------------------------------------------
# load_coupling_for_run — DB-backed
# ---------------------------------------------------------------------------

class TestLoadCouplingForRun(unittest.TestCase):
    @patch("swe2d.results.timestep_service._find_prefixed_or_default_table")
    @patch("swe2d.results.timestep_service._open_ro")
    def test_returns_records(self, mock_open_ro, mock_find_table):
        conn = MagicMock()
        mock_open_ro.return_value = conn
        mock_find_table.return_value = "swe2d_coupling_results"

        cur = MagicMock()
        conn.execute.return_value = cur
        cur.fetchall.return_value = [
            (0.0, "structure", "flow", "obj1", "Weir A", 1.5),
            (5.0, "structure", "flow", "obj2", "Orifice B", 2.3),
        ]

        result = load_coupling_for_run("/test.gpkg", "run_abc")
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["object_name"], "Weir A")
        self.assertEqual(result[1]["value"], 2.3)
        conn.close.assert_called_once()

    @patch("swe2d.results.timestep_service._find_prefixed_or_default_table")
    @patch("swe2d.results.timestep_service._open_ro")
    def test_no_table_returns_empty(self, mock_open_ro, mock_find_table):
        conn = MagicMock()
        mock_open_ro.return_value = conn
        mock_find_table.return_value = ""

        result = load_coupling_for_run("/test.gpkg", "run_abc")
        self.assertEqual(result, [])
        conn.close.assert_called_once()

    @patch("swe2d.results.timestep_service._open_ro")
    def test_open_failure_returns_empty(self, mock_open_ro):
        mock_open_ro.return_value = None
        result = load_coupling_for_run("/test.gpkg", "run_abc")
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
