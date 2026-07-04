"""Tests for _overlay_selected_key on SWE2DResultsData and integration."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestOverlaySelected(unittest.TestCase):
    """Unit tests for _overlay_selected_key on SWE2DResultsData."""

    def test_overlay_selected_key_starts_empty(self):
        from swe2d.results.data import SWE2DResultsData
        data = SWE2DResultsData()
        self.assertEqual(data._overlay_selected_key, "")

    def test_remove_runs_clears_overlay_selected_if_matched(self):
        from swe2d.results.data import SWE2DResultsData
        from swe2d.results.run_service import RunRecord
        data = SWE2DResultsData()
        rec = RunRecord(
            run_id="r1", gpkg_path="/tmp/test.gpkg",
            color=(31, 119, 180), enabled=True,
        )
        data._run_records = [rec]
        data._overlay_selected_key = rec.key
        data.remove_runs({rec.key})
        self.assertEqual(data._overlay_selected_key, "")

    def test_remove_runs_preserves_overlay_selected_if_not_matched(self):
        from swe2d.results.data import SWE2DResultsData
        from swe2d.results.run_service import RunRecord
        data = SWE2DResultsData()
        rec1 = RunRecord(
            run_id="r1", gpkg_path="/tmp/a.gpkg",
            color=(31, 119, 180), enabled=True,
        )
        rec2 = RunRecord(
            run_id="r2", gpkg_path="/tmp/b.gpkg",
            color=(255, 127, 14), enabled=True,
        )
        data._run_records = [rec1, rec2]
        data._overlay_selected_key = rec2.key
        data.remove_runs({rec1.key})
        self.assertEqual(data._overlay_selected_key, rec2.key)

    def test_overlay_selected_run_returns_selected_or_first_enabled(self):
        from swe2d.results.data import SWE2DResultsData
        from swe2d.results.run_service import RunRecord
        data = SWE2DResultsData()
        rec1 = RunRecord(run_id="r1", gpkg_path="/tmp/a.gpkg", color=(1,2,3), enabled=True)
        rec2 = RunRecord(run_id="r2", gpkg_path="/tmp/b.gpkg", color=(4,5,6), enabled=True)
        data._run_records = [rec1, rec2]

        # No selection -> first enabled
        self.assertIs(data.overlay_selected_run(), rec1)

        # Selection -> selected run
        data._overlay_selected_key = rec2.key
        self.assertIs(data.overlay_selected_run(), rec2)

    def test_state_persists_overlay_selected_key(self):
        from swe2d.results.data import SWE2DResultsData
        data = SWE2DResultsData()
        data._overlay_selected_key = "path::run1"
        state = data.save_data_state()
        self.assertIn("overlay_selected_key", state)
        self.assertEqual(state["overlay_selected_key"], "path::run1")

        data2 = SWE2DResultsData()
        data2.restore_data_state(state)
        self.assertEqual(data2._overlay_selected_key, "path::run1")

    # --- Source-inspection tests for overlay_selected usage in controller/UI ---

    def test_sync_high_perf_uses_overlay_selected_run(self):
        """sync_high_perf_overlay_data must use the overlay-selected run,
        not just first_enabled_record()."""
        import inspect
        from swe2d.workbench.controllers.overlay_controller import OverlayController
        src = inspect.getsource(OverlayController.sync_high_perf_overlay_data)
        self.assertIn(
            "overlay_selected_run",
            src,
            "sync_high_perf_overlay_data must use overlay_selected_run() "
            "instead of first_enabled_record()",
        )

    def test_load_mesh_snapshot_uses_overlay_selected_run(self):
        """load_mesh_snapshot_for_overlay must use overlay-selected run."""
        import inspect
        from swe2d.workbench.controllers.overlay_controller import OverlayController
        src = inspect.getsource(OverlayController.load_mesh_snapshot_for_overlay)
        self.assertIn(
            "overlay_selected_run",
            src,
            "load_mesh_snapshot_for_overlay must use overlay_selected_run()",
        )

    def test_run_list_has_overlay_select_handler(self):
        """ResultsToolbox must wire a double-click handler for overlay selection."""
        import inspect
        from swe2d.workbench.views.results_controls import ResultsToolbox
        src = inspect.getsource(ResultsToolbox)
        self.assertIn(
            "_on_run_double_clicked",
            src,
            "ResultsToolbox must have a _on_run_double_clicked method "
            "for selecting the overlay run",
        )

    def test_rebuild_run_list_shows_overlay_indicator(self):
        """_rebuild_run_list must mark the overlay-selected run visually."""
        import inspect
        from swe2d.workbench.views.results_controls import ResultsToolbox
        src = inspect.getsource(ResultsToolbox._rebuild_run_list)
        self.assertIn(
            "overlay_selected_key",
            src,
            "_rebuild_run_list must check overlay_selected_key to mark "
            "the overlay-active run",
        )
