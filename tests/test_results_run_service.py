"""Tests for swe2d.results.run_service — pure data logic, no Qt."""

import unittest
from unittest.mock import patch

from swe2d.results.run_service import (
    RunRecord,
    collect_runs_from_gpkg,
    filter_run_keys_by_source,
    merge_run_records,
    next_color,
    remove_selected_runs,
)


# ---------------------------------------------------------------------------
# next_color
# ---------------------------------------------------------------------------

class TestNextColor(unittest.TestCase):
    def test_next_color_returns_tuple(self):
        c = next_color(0)
        self.assertIsInstance(c, tuple)
        self.assertEqual(3, len(c))
        self.assertTrue(all(isinstance(v, int) for v in c))

    def test_next_color_cycles_through_palette(self):
        c0 = next_color(0)
        c1 = next_color(1)
        self.assertNotEqual(c0, c1)
        self.assertEqual(next_color(0), next_color(10))
        self.assertEqual(next_color(1), next_color(11))

    def test_next_color_all_entries_unique(self):
        seen = {next_color(i) for i in range(10)}
        self.assertEqual(len(seen), 10)


# ---------------------------------------------------------------------------
# collect_runs_from_gpkg
# ---------------------------------------------------------------------------

class TestCollectRunsFromGpkg(unittest.TestCase):
    @patch("swe2d.results.run_service.discover_line_result_runs")
    def test_collect_runs(self, mock_discover):
        mock_discover.return_value = [
            {"run_id": "run_a", "has_profile": True},
            {"run_id": "run_b", "has_profile": False},
        ]
        result = collect_runs_from_gpkg("/data/test.gpkg")
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].run_id, "run_a")
        self.assertEqual(result[0].gpkg_path, "/data/test.gpkg")
        self.assertTrue(result[0].has_profile)
        self.assertIn("test.gpkg", result[0].label)
        self.assertEqual(result[1].run_id, "run_b")
        self.assertFalse(result[1].has_profile)

    @patch("swe2d.results.run_service.discover_line_result_runs")
    def test_collect_runs_empty_path(self, mock_discover):
        result = collect_runs_from_gpkg("")
        mock_discover.assert_not_called()
        self.assertEqual(result, [])

    @patch("swe2d.results.run_service.discover_line_result_runs")
    def test_collect_runs_skips_empty_run_id(self, mock_discover):
        mock_discover.return_value = [
            {"run_id": "", "has_profile": True},
            {"run_id": "valid", "has_profile": False},
        ]
        result = collect_runs_from_gpkg("/data/test.gpkg")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].run_id, "valid")

    @patch("swe2d.results.run_service.discover_line_result_runs")
    def test_snapshot_suffix(self, mock_discover):
        mock_discover.return_value = [
            {"run_id": "swe2d_snapshot_abc", "has_profile": False},
            {"run_id": "Snapshot_manual", "has_profile": False},
            {"run_id": "regular_run", "has_profile": False},
        ]
        result = collect_runs_from_gpkg("/data/test.gpkg")
        self.assertIn("[snapshot]", result[0].label)
        self.assertIn("[snapshot]", result[1].label)
        self.assertNotIn("[snapshot]", result[2].label)

    @patch("swe2d.results.run_service.discover_line_result_runs")
    def test_initial_color_is_black(self, mock_discover):
        mock_discover.return_value = [
            {"run_id": "r1", "has_profile": False},
        ]
        result = collect_runs_from_gpkg("/data/test.gpkg")
        self.assertEqual(result[0].color, (0, 0, 0))


# ---------------------------------------------------------------------------
# filter_run_keys_by_source
# ---------------------------------------------------------------------------

class TestFilterRunKeysBySource(unittest.TestCase):
    def test_filters_by_gpkg_path(self):
        keys = {"/a/b.gpkg::run1", "/a/b.gpkg::run2", "/c/d.gpkg::run3"}
        result = filter_run_keys_by_source(keys, "/a/b.gpkg")
        self.assertEqual(result, {"/a/b.gpkg::run1", "/a/b.gpkg::run2"})

    def test_empty_keys(self):
        self.assertEqual(filter_run_keys_by_source(set(), "/x.gpkg"), set())

    def test_no_match(self):
        keys = {"/a/b.gpkg::run1"}
        self.assertEqual(filter_run_keys_by_source(keys, "/none.gpkg"), set())


# ---------------------------------------------------------------------------
# remove_selected_runs
# ---------------------------------------------------------------------------

class TestRemoveSelectedRuns(unittest.TestCase):
    def setUp(self):
        self.records = [
            RunRecord(run_id="run1", gpkg_path="/a.gpkg", color=(255, 0, 0)),
            RunRecord(run_id="run2", gpkg_path="/a.gpkg", color=(0, 255, 0)),
            RunRecord(run_id="run3", gpkg_path="/b.gpkg", color=(0, 0, 255)),
        ]

    def test_removes_by_key(self):
        remaining, paths = remove_selected_runs(
            self.records, {self.records[0].key}, ["/b.gpkg"]
        )
        self.assertEqual(len(remaining), 2)
        self.assertEqual(remaining[0].run_id, "run2")
        self.assertEqual(remaining[1].run_id, "run3")

    def test_drops_orphaned_manual_path(self):
        remaining, paths = remove_selected_runs(
            self.records, {self.records[2].key}, ["/b.gpkg"]
        )
        self.assertEqual(paths, [])

    def test_preserves_path_with_remaining_runs(self):
        records = self.records + [
            RunRecord(run_id="run4", gpkg_path="/b.gpkg", color=(0, 0, 0)),
        ]
        remaining, paths = remove_selected_runs(
            records, {self.records[2].key}, ["/b.gpkg"]
        )
        self.assertEqual(paths, ["/b.gpkg"])

    def test_reassigns_colors_compactly(self):
        remaining, _ = remove_selected_runs(
            self.records, {self.records[0].key}, []
        )
        for rec in remaining:
            self.assertNotEqual(rec.color, (0, 0, 0))

    def test_no_selected_returns_same(self):
        remaining, paths = remove_selected_runs(self.records, set(), [])
        self.assertEqual(len(remaining), 3)
        self.assertEqual(paths, [])

    def test_does_not_mutate_input_list(self):
        orig_len = len(self.records)
        remove_selected_runs(self.records, {self.records[0].key}, [])
        self.assertEqual(len(self.records), orig_len)


# ---------------------------------------------------------------------------
# merge_run_records
# ---------------------------------------------------------------------------

class TestMergeRunRecords(unittest.TestCase):
    def test_merges_base_and_manual(self):
        base = [RunRecord(run_id="r1", gpkg_path="/a.gpkg", color=(0, 0, 0))]
        manual = [RunRecord(run_id="r2", gpkg_path="/b.gpkg", color=(0, 0, 0))]
        result = merge_run_records(base, manual, set(), set(), ["/b.gpkg"])
        self.assertEqual(len(result), 2)

    def test_deduplicates(self):
        base = [RunRecord(run_id="r1", gpkg_path="/a.gpkg", color=(0, 0, 0))]
        manual = [RunRecord(run_id="r1", gpkg_path="/a.gpkg", color=(0, 0, 0))]
        result = merge_run_records(base, manual, set(), set(), [])
        self.assertEqual(len(result), 1)

    def test_assigns_colors(self):
        result = merge_run_records(
            [RunRecord(run_id="r1", gpkg_path="/a.gpkg", color=(0, 0, 0))],
            [RunRecord(run_id="r2", gpkg_path="/b.gpkg", color=(0, 0, 0))],
            set(), set(), ["/b.gpkg"],
        )
        self.assertNotEqual(result[0].color, (0, 0, 0))
        self.assertNotEqual(result[1].color, (0, 0, 0))

    def test_filters_base_by_keys(self):
        base = [
            RunRecord(run_id="r1", gpkg_path="/a.gpkg", color=(0, 0, 0)),
            RunRecord(run_id="r2", gpkg_path="/a.gpkg", color=(0, 0, 0)),
        ]
        result = merge_run_records(base, [], {"/a.gpkg::r1"}, set(), [])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].run_id, "r1")

    def test_filters_manual_by_keys(self):
        manual = [
            RunRecord(run_id="r1", gpkg_path="/b.gpkg", color=(0, 0, 0)),
            RunRecord(run_id="r2", gpkg_path="/b.gpkg", color=(0, 0, 0)),
        ]
        result = merge_run_records(
            [], manual, set(), {"/b.gpkg::r1"}, ["/b.gpkg"],
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].run_id, "r1")

    def test_stale_base_keys_fall_back(self):
        base = [RunRecord(run_id="r1", gpkg_path="/a.gpkg", color=(0, 0, 0))]
        result = merge_run_records(base, [], {"/a.gpkg::nonexistent"}, set(), [])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].run_id, "r1")

    def test_stale_manual_keys_fall_back(self):
        manual = [RunRecord(run_id="r1", gpkg_path="/b.gpkg", color=(0, 0, 0))]
        result = merge_run_records(
            [], manual, set(), {"/b.gpkg::nonexistent"}, ["/b.gpkg"],
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].run_id, "r1")

    def test_does_not_mutate_input_lists(self):
        base = [RunRecord(run_id="r1", gpkg_path="/a.gpkg", color=(0, 0, 0))]
        manual = [RunRecord(run_id="r2", gpkg_path="/b.gpkg", color=(0, 0, 0))]
        orig_base_len = len(base)
        orig_manual_len = len(manual)
        merge_run_records(base, manual, set(), set(), ["/b.gpkg"])
        self.assertEqual(len(base), orig_base_len)
        self.assertEqual(len(manual), orig_manual_len)

    def test_manual_paths_controls_which_manual_records_included(self):
        manual = [
            RunRecord(run_id="r1", gpkg_path="/b.gpkg", color=(0, 0, 0)),
            RunRecord(run_id="r2", gpkg_path="/c.gpkg", color=(0, 0, 0)),
        ]
        result = merge_run_records([], manual, set(), set(), ["/b.gpkg"])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].gpkg_path, "/b.gpkg")


# ---------------------------------------------------------------------------
# RunRecord behaviour
# ---------------------------------------------------------------------------

class TestRunRecord(unittest.TestCase):
    def test_key_format(self):
        rec = RunRecord(run_id="my_run", gpkg_path="/p.gpkg", color=(1, 2, 3))
        self.assertEqual(rec.key, "/p.gpkg::my_run")

    def test_display_label_falls_back_to_run_id(self):
        rec = RunRecord(run_id="my_run", gpkg_path="/p.gpkg", color=(1, 2, 3))
        self.assertEqual(rec.display_label(), "my_run")
        rec.label = "Custom"
        self.assertEqual(rec.display_label(), "Custom")


if __name__ == "__main__":
    unittest.main()
