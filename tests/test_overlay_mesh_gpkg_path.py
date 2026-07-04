"""Regression test for the overlay-mesh GPKG path bug.

Bug: load_mesh_snapshot_for_overlay used data.gpkg_path (the model
GeoPackage) instead of the per-run GeoPackage from the enabled
RunRecord.  When the model GPKG does not contain the baked mesh for
the displayed run (e.g. when results were loaded from a different
GPKG via "Add Results"), the overlay silently fails or renders the
wrong mesh.

Fix: load the GPKG path from the enabled RunRecord returned by
enabled_overlay_targets() — same source used by sync_high_perf_overlay_data.
"""
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestOverlayMeshGpkgPath(unittest.TestCase):
    """load_mesh_snapshot_for_overlay must read from the per-run GPKG."""

    def test_uses_per_run_gpkg_not_data_gpkg_path(self):
        from swe2d.results.data import SWE2DResultsData
        from swe2d.results.run_service import RunRecord

        data = SWE2DResultsData()
        # Simulate a run loaded from a results GPKG that is DIFFERENT from
        # the model GPKG.  data.gpkg_path is set to the model GPKG (the
        # "overarching" path set by show_results_panel), but the actual
        # results live in a separate file.
        data.set_gpkg_path("/path/to/model.gpkg")
        data._run_records.append(
            RunRecord(
                run_id="rid",
                gpkg_path="/path/to/results.gpkg",
                color=(31, 119, 180),
                enabled=True,
            )
        )

        # enabled_overlay_targets() must return the per-run GPKG, not
        # data.gpkg_path.
        targets = data.enabled_overlay_targets()
        self.assertEqual(targets, [("/path/to/results.gpkg", "rid")])

    def test_load_mesh_snapshot_for_overlay_passes_per_run_gpkg(self):
        """Inspect the source code path used by load_mesh_snapshot_for_overlay.

        The method currently reads gpkg from data.gpkg_path.  After the fix
        it must read from the enabled RunRecord's gpkg_path.
        """
        from swe2d.workbench.controllers.overlay_controller import OverlayController
        import inspect

        src = inspect.getsource(OverlayController.load_mesh_snapshot_for_overlay)
        # The fix replaces `data.gpkg_path` with the per-run GPKG derived
        # from enabled_overlay_targets().  The old lookup must be gone.
        self.assertNotIn(
            'getattr(data, "gpkg_path"',
            src,
            "load_mesh_snapshot_for_overlay still reads data.gpkg_path — "
            "the overlay would render from the model GPKG instead of the "
            "individual results GPKG.",
        )


if __name__ == "__main__":
    unittest.main()