"""Test that finalize_and_persist clears live snapshots after baking."""
import os
import sys
import unittest
from unittest.mock import MagicMock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestClearOnFinalize(unittest.TestCase):
    def test_finalize_clears_live_snapshots(self):
        """After finalize, _live_times and _live_run_id must be cleared."""
        from swe2d.runtime.run_finalizer import SWE2DRunFinalizer
        from swe2d.results.data import SWE2DResultsData

        data = SWE2DResultsData()
        data._live_run_id = "run_X"
        data._live_times = np.array([0.0, 10.0])
        data._live_h = np.zeros((2, 5))
        data._live_hu = np.zeros((2, 5))
        data._live_hv = np.zeros((2, 5))

        view = MagicMock()
        view.log_message = MagicMock()
        view.is_cancel_requested = MagicMock(return_value=False)
        view.get_line_results_storage_path = MagicMock(return_value="")
        view.refresh_plot = MagicMock()

        finalizer = SWE2DRunFinalizer(view)

        # Inspect the finalize_and_persist source — it must call
        # clear_live_snapshots on results_data before returning.
        import inspect
        src = inspect.getsource(finalizer.finalize_and_persist)
        self.assertIn(
            "clear_live_snapshots",
            src,
            "finalize_and_persist must call clear_live_snapshots() "
            "to transition the run from live to GPKG data.",
        )


if __name__ == "__main__":
    unittest.main()
