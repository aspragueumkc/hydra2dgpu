"""Tests for swe2d.results.structure_service — pure data logic, no Qt."""

import unittest
from unittest.mock import MagicMock, PropertyMock, patch
import numpy as np

from swe2d.results.structure_service import (
    filter_structure_records,
    load_bound_layer_name,
    load_line_geometry,
    load_structure_overlay_data,
    load_structure_records,
    resolve_structure_profile_overlays,
)


# ---------------------------------------------------------------------------
# filter_structure_records (pure, no mocking needed)
# ---------------------------------------------------------------------------

class TestFilterStructureRecords(unittest.TestCase):
    def test_filters_by_metric_threshold(self):
        records = [
            {"object_id": "s1", "metric": "flow", "value": 5.0},
            {"object_id": "s2", "metric": "flow", "value": 15.0},
            {"object_id": "s3", "metric": "flow", "value": 3.0},
        ]
        result = filter_structure_records(records, "value", 10.0)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["object_id"], "s2")

    def test_empty_records(self):
        self.assertEqual(filter_structure_records([], "value", 0.0), [])

    def test_all_records_below_threshold(self):
        records = [
            {"object_id": "s1", "value": 1.0},
            {"object_id": "s2", "value": 2.0},
        ]
        self.assertEqual(filter_structure_records(records, "value", 10.0), [])

    def test_threshold_zero_keeps_all(self):
        records = [
            {"object_id": "s1", "value": 0.0},
            {"object_id": "s2", "value": 5.0},
        ]
        result = filter_structure_records(records, "value", 0.0)
        self.assertEqual(len(result), 2)

    def test_missing_metric_column(self):
        records = [{"object_id": "s1"}]
        self.assertEqual(filter_structure_records(records, "value", 0.0), [])

    def test_does_not_mutate_input(self):
        records = [{"object_id": "s1", "value": 5.0}]
        orig = list(records)
        filter_structure_records(records, "value", 10.0)
        self.assertEqual(records, orig)


# ---------------------------------------------------------------------------
# load_bound_layer_name (SQLite mocking)
# ---------------------------------------------------------------------------

class TestLoadBoundLayerName(unittest.TestCase):
    @patch("swe2d.results.structure_service.open_ro")
    def test_returns_layer_name(self, mock_open_ro):
        mock_conn = MagicMock()
        mock_open_ro.return_value = mock_conn
        mock_conn.execute.return_value.fetchone.return_value = ("my_struct_layer",)

        result = load_bound_layer_name("/data/test.gpkg", "hydraulic_structures", "swe2d_structures")
        self.assertEqual(result, "my_struct_layer")

    @patch("swe2d.results.structure_service.table_exists")
    @patch("swe2d.results.structure_service.open_ro")
    def test_returns_default_on_missing_table(self, mock_open_ro, mock_table_exists):
        mock_conn = MagicMock()
        mock_open_ro.return_value = mock_conn
        mock_table_exists.return_value = False

        result = load_bound_layer_name("/data/test.gpkg", "hydraulic_structures", "swe2d_structures")
        self.assertEqual(result, "swe2d_structures")

    @patch("swe2d.results.structure_service.open_ro")
    def test_returns_default_on_no_row(self, mock_open_ro):
        mock_conn = MagicMock()
        mock_open_ro.return_value = mock_conn
        mock_conn.execute.return_value.fetchone.return_value = None

        result = load_bound_layer_name("/data/test.gpkg", "hydraulic_structures", "fallback")
        self.assertEqual(result, "fallback")

    @patch("swe2d.results.structure_service.open_ro")
    def test_returns_default_on_empty_path(self, mock_open_ro):
        result = load_bound_layer_name("", "role", "default")
        mock_open_ro.assert_not_called()
        self.assertEqual(result, "default")

    @patch("swe2d.results.structure_service.open_ro")
    def test_returns_default_when_conn_none(self, mock_open_ro):
        mock_open_ro.return_value = None
        result = load_bound_layer_name("/data/test.gpkg", "role", "default")
        self.assertEqual(result, "default")


# ---------------------------------------------------------------------------
# load_structure_records (SQLite mocking)
# ---------------------------------------------------------------------------

class TestLoadStructureRecords(unittest.TestCase):
    @patch("swe2d.results.structure_service._find_prefixed_or_default_table")
    @patch("swe2d.results.structure_service.open_ro")
    def test_returns_records(self, mock_open_ro, mock_find_table):
        mock_conn = MagicMock()
        mock_open_ro.return_value = mock_conn
        mock_find_table.return_value = "swe2d_coupling_results"
        mock_conn.execute.return_value.fetchall.return_value = [
            ("0.0", "structure", "flow", "s1", "Node1", "5.0"),
            ("0.0", "structure", "flow", "s2", "Node2", "3.0"),
        ]

        result = load_structure_records("/data/test.gpkg", "run1")
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["object_id"], "s1")
        self.assertEqual(result[0]["value"], "5.0")
        self.assertEqual(result[0]["component"], "structure")
        self.assertEqual(result[1]["object_id"], "s2")

    @patch("swe2d.results.structure_service._find_prefixed_or_default_table")
    @patch("swe2d.results.structure_service.open_ro")
    def test_returns_empty_when_no_table(self, mock_open_ro, mock_find_table):
        mock_conn = MagicMock()
        mock_open_ro.return_value = mock_conn
        mock_find_table.return_value = ""

        result = load_structure_records("/data/test.gpkg", "run1")
        self.assertEqual(result, [])

    @patch("swe2d.results.structure_service.open_ro")
    def test_returns_empty_when_conn_none(self, mock_open_ro):
        mock_open_ro.return_value = None
        result = load_structure_records("/data/test.gpkg", "run1")
        self.assertEqual(result, [])

    @patch("swe2d.results.structure_service._find_prefixed_or_default_table")
    @patch("swe2d.results.structure_service.open_ro")
    def test_empty_run_id(self, mock_open_ro, mock_find_table):
        mock_conn = MagicMock()
        mock_open_ro.return_value = mock_conn
        mock_find_table.return_value = "swe2d_coupling_results"
        mock_conn.execute.return_value.fetchall.return_value = []

        result = load_structure_records("/data/test.gpkg", "")
        self.assertEqual(result, [])

    @patch("swe2d.results.structure_service._find_prefixed_or_default_table")
    @patch("swe2d.results.structure_service.open_ro")
    def test_empty_gpkg_path(self, mock_open_ro, mock_find_table):
        result = load_structure_records("", "run1")
        self.assertEqual(result, [])


# ---------------------------------------------------------------------------
# load_line_geometry (QGIS mocking)
# ---------------------------------------------------------------------------

class TestLoadLineGeometry(unittest.TestCase):
    def _make_feature_mock(self, line_id_val, name_val, vertices):
        """Create a feature mock with controlled __getitem__, geometry()."""
        def getitem(key):
            d = {"line_id": line_id_val, "name": name_val}
            return d.get(key, MagicMock())

        ft = MagicMock()
        ft.__getitem__.side_effect = getitem

        # Build a controlled geometry mock
        raw_mock = MagicMock()
        raw_mock.vertexCount.return_value = len(vertices)
        raw_mock.xAt.side_effect = [v[0] for v in vertices]
        raw_mock.yAt.side_effect = [v[1] for v in vertices]

        geom_mock = MagicMock()
        geom_mock.isEmpty.return_value = False
        # constGet returns the raw geometry
        geom_mock.constGet.return_value = raw_mock

        ft.geometry.return_value = geom_mock
        return ft

    @patch("swe2d.results.structure_service.QgsVectorLayer")
    def test_returns_vertex_array(self, mock_vl_cls):
        mock_layer = MagicMock()
        mock_vl_cls.return_value = mock_layer
        mock_layer.isValid.return_value = True
        mock_layer.fields.return_value.names.return_value = ["line_id", "name"]

        ft = self._make_feature_mock(1, "LineA", [(0.0, 0.0), (10.0, 5.0), (20.0, 10.0)])
        mock_layer.getFeatures.return_value = [ft]

        result = load_line_geometry("/data/test.gpkg", 1, "LineA")
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (3, 2))
        np.testing.assert_array_almost_equal(result, [[0.0, 0.0], [10.0, 5.0], [20.0, 10.0]])

    @patch("swe2d.results.structure_service.QgsVectorLayer")
    def test_matches_by_id(self, mock_vl_cls):
        mock_layer = MagicMock()
        mock_vl_cls.return_value = mock_layer
        mock_layer.isValid.return_value = True
        mock_layer.fields.return_value.names.return_value = ["line_id"]

        ft = self._make_feature_mock(42, "Wrong", [(0.0, 0.0), (5.0, 2.0)])
        mock_layer.getFeatures.return_value = [ft]

        result = load_line_geometry("/data/test.gpkg", 42, "wrong_name")
        self.assertEqual(result.shape, (2, 2))

    @patch("swe2d.results.structure_service.QgsVectorLayer")
    def test_returns_empty_on_invalid_layer(self, mock_vl_cls):
        mock_layer = MagicMock()
        mock_vl_cls.return_value = mock_layer
        mock_layer.isValid.return_value = False

        result = load_line_geometry("/data/test.gpkg", 1, "LineA")
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.size, 0)

    def test_returns_empty_on_empty_path(self):
        result = load_line_geometry("", 1, "LineA")
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.size, 0)

    def test_returns_empty_on_negative_id(self):
        result = load_line_geometry("/data/test.gpkg", -1, "LineA")
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.size, 0)


# ---------------------------------------------------------------------------
# resolve_structure_profile_overlays (complex mocking)
# ---------------------------------------------------------------------------

class TestResolveStructureProfileOverlays(unittest.TestCase):
    def _make_ft(self, sid, crest):
        """Create a feature mock for the structure layer."""
        def getitem(key):
            d = {"structure_id": sid, "crest_elev": crest}
            return d.get(key, MagicMock())

        ft = MagicMock()
        ft.__getitem__.side_effect = getitem

        # geometry: intersection -> inter_geom -> centroid -> inter_centroid
        inter_centroid = MagicMock()
        inter_centroid.isEmpty.return_value = False

        inter_geom = MagicMock()
        inter_geom.isEmpty.return_value = False
        inter_geom.centroid.return_value = inter_centroid

        ft_geom = MagicMock()
        ft_geom.isEmpty.return_value = False
        ft_geom.intersection.return_value = inter_geom
        ft_geom.centroid.return_value = inter_centroid
        ft_geom.nearestPoint.return_value = MagicMock()

        ft.geometry.return_value = ft_geom
        return ft

    @patch("swe2d.results.structure_service.load_structure_records")
    @patch("swe2d.results.structure_service._load_profile_line_geom")
    @patch("swe2d.results.structure_service.load_bound_layer_name")
    @patch("swe2d.results.structure_service.QgsVectorLayer")
    def test_returns_overlays_for_runs(
        self, mock_vl_cls, mock_bound_layer,
        mock_line_geom, mock_struct_records,
    ):
        mock_struct_records.side_effect = [
            # First call for line-id discovery
            [
                {"object_id": "s1", "object_name": "Struct1", "value": 5.0, "component": "structure", "metric": "flow", "t_s": 0.0},
                {"object_id": "s2", "object_name": "Struct2", "value": 3.0, "component": "structure", "metric": "flow", "t_s": 0.0},
            ],
            # Per-run calls
            [
                {"object_id": "s1", "object_name": "Struct1", "value": 5.0, "component": "structure", "metric": "flow", "t_s": 0.0},
                {"object_id": "s2", "object_name": "Struct2", "value": 3.0, "component": "structure", "metric": "flow", "t_s": 0.0},
            ],
            [
                {"object_id": "s3", "object_name": "Struct3", "value": 2.0, "component": "structure", "metric": "flow", "t_s": 0.0},
            ],
        ]

        mock_line_geom.return_value.isEmpty.return_value = False
        # Make lineLocatePoint return a sensible station value
        mock_line_geom.return_value.lineLocatePoint.return_value = 50.0

        mock_bound_layer.return_value = "swe2d_structures"

        mock_layer = MagicMock()
        mock_vl_cls.return_value = mock_layer
        mock_layer.isValid.return_value = True
        mock_layer.fields.return_value.names.return_value = ["structure_id", "crest_elev"]

        ft = self._make_ft("s1", 105.0)
        mock_layer.getFeatures.return_value = [ft]

        result = resolve_structure_profile_overlays(
            "/data/test.gpkg", ["run1", "run2"],
        )
        self.assertIsInstance(result, dict)
        self.assertIn("run1", result)
        self.assertIn("run2", result)
        self.assertEqual(len(result["run1"]), 2)
        self.assertEqual(len(result["run2"]), 1)

    @patch("swe2d.results.structure_service.load_structure_records")
    @patch("swe2d.results.structure_service._load_profile_line_geom")
    @patch("swe2d.results.structure_service.load_bound_layer_name")
    @patch("swe2d.results.structure_service.QgsVectorLayer")
    def test_overlay_dict_has_expected_keys(
        self, mock_vl_cls, mock_bound_layer,
        mock_line_geom, mock_struct_records,
    ):
        mock_struct_records.return_value = [
            {"object_id": "s1", "object_name": "Struct1", "value": 5.0, "component": "structure", "metric": "flow", "t_s": 0.0},
        ]
        mock_line_geom.return_value.isEmpty.return_value = False
        mock_line_geom.return_value.lineLocatePoint.return_value = 50.0

        mock_bound_layer.return_value = "swe2d_structures"

        mock_layer = MagicMock()
        mock_vl_cls.return_value = mock_layer
        mock_layer.isValid.return_value = True
        mock_layer.fields.return_value.names.return_value = ["structure_id", "crest_elev"]

        ft = self._make_ft("s1", 105.0)
        mock_layer.getFeatures.return_value = [ft]

        result = resolve_structure_profile_overlays("/data/test.gpkg", ["run1"])
        overlays = result["run1"]
        self.assertGreater(len(overlays), 0)
        overlay = overlays[0]
        expected_keys = {"run_id", "run_label", "object_id", "flow_cms", "station_m", "elev_m", "placement"}
        self.assertSetEqual(set(overlay.keys()), expected_keys)

    def test_returns_empty_for_empty_run_ids(self):
        result = resolve_structure_profile_overlays("/data/test.gpkg", [])
        self.assertEqual(result, {})


# ---------------------------------------------------------------------------
# load_structure_overlay_data (QGIS mocking, queries dependency)
# ---------------------------------------------------------------------------

class TestLoadStructureOverlayData(unittest.TestCase):
    def _make_ft(self, sid, crest):
        """Create a feature mock for the structure layer."""
        def getitem(key):
            d = {"structure_id": sid, "crest_elev": crest}
            return d.get(key, MagicMock())

        ft = MagicMock()
        ft.__getitem__.side_effect = getitem

        # geometry: intersection -> inter_geom -> centroid -> inter_centroid
        inter_centroid = MagicMock()
        inter_centroid.isEmpty.return_value = False

        inter_geom = MagicMock()
        inter_geom.isEmpty.return_value = False
        inter_geom.centroid.return_value = inter_centroid

        ft_geom = MagicMock()
        ft_geom.isEmpty.return_value = False
        ft_geom.intersection.return_value = inter_geom
        ft_geom.centroid.return_value = inter_centroid
        ft_geom.nearestPoint.return_value = MagicMock()

        ft.geometry.return_value = ft_geom
        return ft

    @patch("swe2d.results.structure_service.load_structure_records")
    @patch("swe2d.results.structure_service._load_profile_line_geom")
    @patch("swe2d.results.structure_service.load_bound_layer_name")
    @patch("swe2d.results.structure_service.QgsVectorLayer")
    @patch("swe2d.results.queries.find_nearest_timestep")
    @patch("swe2d.results.queries.load_structure_flows_at_time")
    def test_returns_records_for_multiple_runs(
        self, mock_load_flows, mock_find_tstep,
        mock_vl_cls, mock_bound_layer,
        mock_line_geom, mock_struct_records,
    ):
        mock_struct_records.return_value = [
            {"object_id": "s1", "object_name": "Struct1", "value": 5.0, "component": "structure", "metric": "flow", "t_s": 0.0},
            {"object_id": "s2", "object_name": "Struct2", "value": 3.0, "component": "structure", "metric": "flow", "t_s": 0.0},
        ]
        mock_line_geom.return_value.isEmpty.return_value = False
        mock_line_geom.return_value.lineLocatePoint.return_value = 50.0
        mock_bound_layer.return_value = "swe2d_structures"
        mock_find_tstep.return_value = 0.0
        mock_load_flows.return_value = [
            {"object_id": "s1", "value": 5.0},
            {"object_id": "s2", "value": 3.0},
        ]

        mock_layer = MagicMock()
        mock_vl_cls.return_value = mock_layer
        mock_layer.isValid.return_value = True
        mock_layer.fields.return_value.names.return_value = ["structure_id", "crest_elev"]

        ft = self._make_ft("s1", 105.0)
        ft2 = self._make_ft("s2", 100.0)
        mock_layer.getFeatures.return_value = [ft, ft2]

        result = load_structure_overlay_data(
            "/data/test.gpkg",
            ["run1", "run2"],
            t_sec=0.0,
        )
        self.assertIsInstance(result, list)
        # Returns one record per structure feature per run
        # run1: s1, s2 (2 records)
        # run2: s1, s2 (2 records)
        # Total: 4 records
        self.assertEqual(len(result), 4)
        # Verify all expected object_ids are present
        object_ids = {r["object_id"] for r in result}
        self.assertEqual(object_ids, {"s1", "s2"})

    @patch("swe2d.results.structure_service.load_structure_records")
    @patch("swe2d.results.structure_service._load_profile_line_geom")
    @patch("swe2d.results.structure_service.load_bound_layer_name")
    @patch("swe2d.results.structure_service.QgsVectorLayer")
    @patch("swe2d.results.queries.find_nearest_timestep")
    @patch("swe2d.results.queries.load_structure_flows_at_time")
    def test_returns_empty_for_empty_inputs(
        self, mock_load_flows, mock_find_tstep,
        mock_vl_cls, mock_bound_layer,
        mock_line_geom, mock_struct_records,
    ):
        mock_line_geom.return_value.isEmpty.return_value = False
        mock_bound_layer.return_value = "swe2d_structures"

        # Test with empty gpkg_path
        result = load_structure_overlay_data("", ["run1"], 0.0)
        self.assertEqual(result, [])

        # Test with empty run_ids
        result = load_structure_overlay_data("/data/test.gpkg", [], 0.0)
        self.assertEqual(result, [])

    @patch("swe2d.results.structure_service.load_structure_records")
    @patch("swe2d.results.structure_service._load_profile_line_geom")
    @patch("swe2d.results.structure_service.load_bound_layer_name")
    @patch("swe2d.results.structure_service.QgsVectorLayer")
    @patch("swe2d.results.queries.find_nearest_timestep")
    @patch("swe2d.results.queries.load_structure_flows_at_time")
    def test_station_less_mode_when_line_geom_missing(
        self, mock_load_flows, mock_find_tstep,
        mock_vl_cls, mock_bound_layer,
        mock_line_geom, mock_struct_records,
    ):
        mock_struct_records.return_value = [
            {"object_id": "s1", "object_name": "Struct1", "value": 5.0, "component": "structure", "metric": "flow", "t_s": 0.0},
        ]
        # Line geometry is None (not found)
        mock_line_geom.return_value = None

        result = load_structure_overlay_data(
            "/data/test.gpkg",
            ["run1"],
            t_sec=0.0,
        )
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["object_id"], "s1")
        self.assertEqual(result[0]["flow_cms"], 5.0)
        self.assertTrue(np.isnan(result[0]["station_m"]))
        self.assertTrue(np.isnan(result[0]["elev_m"]))
        self.assertEqual(result[0]["placement"], "unplaced")

    @patch("swe2d.results.structure_service.load_structure_records")
    @patch("swe2d.results.structure_service._load_profile_line_geom")
    @patch("swe2d.results.structure_service.load_bound_layer_name")
    @patch("swe2d.results.structure_service.QgsVectorLayer")
    @patch("swe2d.results.queries.find_nearest_timestep")
    @patch("swe2d.results.queries.load_structure_flows_at_time")
    def test_unplaced_structures_when_no_matching_geometry(
        self, mock_load_flows, mock_find_tstep,
        mock_vl_cls, mock_bound_layer,
        mock_line_geom, mock_struct_records,
    ):
        mock_struct_records.return_value = [
            {"object_id": "s1", "object_name": "Struct1", "value": 5.0, "component": "structure", "metric": "flow", "t_s": 0.0},
            {"object_id": "s2", "object_name": "Struct2", "value": 3.0, "component": "structure", "metric": "flow", "t_s": 0.0},
        ]
        mock_line_geom.return_value.isEmpty.return_value = False
        mock_line_geom.return_value.lineLocatePoint.return_value = 50.0
        mock_bound_layer.return_value = "swe2d_structures"
        mock_find_tstep.return_value = 0.0
        mock_load_flows.return_value = [
            {"object_id": "s1", "value": 5.0},
        ]
        # Only s1 is in the structure layer, s2 is not

        mock_layer = MagicMock()
        mock_vl_cls.return_value = mock_layer
        mock_layer.isValid.return_value = True
        mock_layer.fields.return_value.names.return_value = ["structure_id", "crest_elev"]

        ft = self._make_ft("s1", 105.0)
        mock_layer.getFeatures.return_value = [ft]

        result = load_structure_overlay_data(
            "/data/test.gpkg",
            ["run1"],
            t_sec=0.0,
        )
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["object_id"], "s1")
        self.assertEqual(result[0]["flow_cms"], 5.0)

    @patch("swe2d.results.structure_service.load_structure_records")
    @patch("swe2d.results.structure_service._load_profile_line_geom")
    @patch("swe2d.results.structure_service.load_bound_layer_name")
    @patch("swe2d.results.structure_service.QgsVectorLayer")
    @patch("swe2d.results.queries.find_nearest_timestep")
    @patch("swe2d.results.queries.load_structure_flows_at_time")
    def test_skips_runs_with_no_structure_records(
        self, mock_load_flows, mock_find_tstep,
        mock_vl_cls, mock_bound_layer,
        mock_line_geom, mock_struct_records,
    ):
        mock_struct_records.side_effect = [
            # run1 has records
            [{"object_id": "s1", "object_name": "Struct1", "value": 5.0, "component": "structure", "metric": "flow", "t_s": 0.0}],
            # run2 has no records
            [],
        ]
        mock_line_geom.return_value.isEmpty.return_value = False
        mock_line_geom.return_value.lineLocatePoint.return_value = 50.0
        mock_bound_layer.return_value = "swe2d_structures"
        mock_find_tstep.return_value = 0.0
        mock_load_flows.return_value = [
            {"object_id": "s1", "value": 5.0},
        ]

        mock_layer = MagicMock()
        mock_vl_cls.return_value = mock_layer
        mock_layer.isValid.return_value = True
        mock_layer.fields.return_value.names.return_value = ["structure_id", "crest_elev"]

        ft = self._make_ft("s1", 105.0)
        mock_layer.getFeatures.return_value = [ft]

        result = load_structure_overlay_data(
            "/data/test.gpkg",
            ["run1", "run2"],
            t_sec=0.0,
        )
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["object_id"], "s1")
        # run2 should be skipped


if __name__ == "__main__":
    unittest.main()
