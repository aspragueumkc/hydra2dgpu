"""Tests for overlay_selected_run behaviour in overlay_controller.

Tasks 1-5 added _overlay_selected_key field with accessors, persistence,
and cleanup. Task 6 verifies sync_high_perf_overlay_data and
load_mesh_snapshot_for_overlay use overlay_selected_run() instead of
first_enabled_record() / enabled_overlay_targets()[0].
"""

import unittest


class TestOverlaySelected(unittest.TestCase):
    """High-level source-inspection tests for overlay_selected usage."""

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
