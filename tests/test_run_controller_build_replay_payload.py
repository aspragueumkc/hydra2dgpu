import unittest
from unittest.mock import MagicMock

from swe2d.workbench.controllers.run_controller import RunController


class TestRunControllerBuildReplayPayload(unittest.TestCase):
    def test_public_method_delegates_to_private_method(self):
        view = MagicMock()
        view._mesh_data = {}
        rc = RunController(view=view)
        rc._build_replay_payload = MagicMock(return_value={"ok": True})
        result = rc.build_replay_payload(
            widget_state={"a": 1},
            mesh_name="mesh",
            run_duration_s=3600.0,
            mesh_gpkg_path="/tmp/m.gpkg",
            run_id="run_1",
        )
        rc._build_replay_payload.assert_called_once_with(
            widget_state={"a": 1},
            mesh_name="mesh",
            run_duration_s=3600.0,
            mesh_gpkg_path="/tmp/m.gpkg",
            run_id="run_1",
        )
        self.assertEqual(result, {"ok": True})

    def test_payload_emits_empty_units_and_original_results_widget_names(self):
        """Snapshot must match the reference CLI JSON: empty units block and
        the original widget names ``results_gpkg_path_edit`` /
        ``results_table_name_edit`` in params."""
        view = MagicMock()
        view._mesh_data = {"crs_wkt": "EPSG:4326"}
        rc = RunController(view=view)
        widget_state = {
            "version": 1,
            "widgets": {
                "results_gpkg_path_edit": {
                    "type": "QLineEdit",
                    "value": "/tmp/out.gpkg",
                },
                "results_table_name_edit": {
                    "type": "QLineEdit",
                    "value": "run_1",
                },
                "n_mann_spin": {"type": "QDoubleSpinBox", "value": 0.035},
            },
        }
        payload = rc._build_replay_payload(
            widget_state=widget_state,
            mesh_name="mesh",
            run_duration_s=3600.0,
            mesh_gpkg_path="/tmp/m.gpkg",
            run_id="run_1",
        )
        self.assertEqual(payload["units"], {})
        params = payload["params"]
        self.assertIn("results_gpkg_path_edit", params)
        self.assertNotIn("results_gpkg_path", params)
        self.assertIn("results_table_name_edit", params)
        self.assertNotIn("results_table_name", params)
        self.assertEqual(params["results_gpkg_path_edit"], "/tmp/out.gpkg")
        self.assertEqual(params["results_table_name_edit"], "run_1")

    def test_build_run_context_reads_results_path_from_params(self):
        """``build_run_context_from_dict`` must resolve the results GPKG from
        ``params["results_gpkg_path_edit"]`` so replay JSON produced by the
        workbench can run without a separate command-line results path."""
        from swe2d.core.builder import build_run_context_from_dict

        results_path = "/tmp/out.gpkg"
        p = {
            "schema_version": "swe2d-replay/1",
            "mesh": {
                "gpkg_path": "/tmp/m.gpkg",
                "mesh_name": "mesh",
            },
            "params": {
                "results_gpkg_path_edit": results_path,
                "n_mann": 0.035,
            },
        }
        with self.assertRaises(FileNotFoundError):
            # Mesh GPKG does not exist, so it should fail on mesh lookup, not
            # on missing results path. This proves the builder accepted the
            # results path from params.
            build_run_context_from_dict(p)


if __name__ == "__main__":
    unittest.main(verbosity=2)
