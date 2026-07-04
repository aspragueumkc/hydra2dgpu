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
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestOverlayMeshGpkgPath(unittest.TestCase):
    """load_mesh_snapshot_for_overlay must read from the per-run GPKG."""

    def test_enabled_overlay_targets_returns_per_run_gpkg(self):
        """enabled_overlay_targets() must return each run's own GPKG, not an
        overarching 'data.gpkg_path'."""
        from swe2d.results.data import SWE2DResultsData
        from swe2d.results.run_service import RunRecord

        data = SWE2DResultsData()
        data._run_records.append(
            RunRecord(
                run_id="rid",
                gpkg_path="/path/to/results.gpkg",
                color=(31, 119, 180),
                enabled=True,
            )
        )

        targets = data.enabled_overlay_targets()
        self.assertEqual(targets, [("/path/to/results.gpkg", "rid")])

    def test_load_mesh_snapshot_for_overlay_passes_per_run_gpkg(self):
        """Inspect the source code path used by load_mesh_snapshot_for_overlay.

        The method must read from the enabled RunRecord's gpkg_path, not from
        an overarching data.gpkg_path (which no longer exists).
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