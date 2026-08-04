"""Characterization tests for ``swe2d.workbench.services.graph_editor_service``.

Pattern P1 (pure service round-trip) per
``docs/specs/2026-08-02-gui-test-coverage-design.md`` §3-§4: real temp model
GPKGs built through the production creation path, real CSVs in tmpdirs, and
mutation verified by production readback.  No mocks.

Note on table names: the production model-GPKG path writes the graph tables
under their display names (``SWE2D_Hyetographs``/``SWE2D_Hydrographs``), while
the service queries lowercase ``swe2d_*`` names.  SQLite identifiers are
case-insensitive, so this matches — the round-trips below prove it.
"""

import os
import sqlite3
import tempfile
import unittest

from tests.qgis_real_env import (
    ensure_qgis_app,
    make_temp_model_gpkg,
    make_temp_results_gpkg,
    requires_qgis,
)

from swe2d.workbench.services import graph_editor_service as ges


@requires_qgis
class TestGpkgRoundTrip(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ensure_qgis_app()

    def test_hyetograph_round_trip_exact_series_and_metadata(self):
        series = [(0.0, 0.0), (300.0, 12.5), (600.0, 3.25), (3600.0, 0.0)]
        with make_temp_model_gpkg() as gpkg:
            ges.save_hyetograph(gpkg, "hyeto_A", series,
                                value_type="intensity", units="mm/hr")
            ges.save_hyetograph(gpkg, "hyeto_B", [(0.0, 1.0), (60.0, 2.0)])

            graphs = ges.load_graphs(gpkg)
        self.assertEqual(set(graphs["hyetographs"]), {"hyeto_A", "hyeto_B"})
        entry = graphs["hyetographs"]["hyeto_A"]
        self.assertEqual(entry["data"], series)
        self.assertEqual(entry["value_type"], "intensity")
        self.assertEqual(entry["units"], "mm/hr")
        self.assertEqual(graphs["hyetographs"]["hyeto_B"]["data"],
                         [(0.0, 1.0), (60.0, 2.0)])
        self.assertEqual(graphs["hydrographs"], {})

    def test_hyetograph_save_replaces_existing_data(self):
        with make_temp_model_gpkg() as gpkg:
            ges.save_hyetograph(gpkg, "h1", [(0.0, 1.0), (1.0, 2.0)])
            ges.save_hyetograph(gpkg, "h1", [(0.0, 9.0)])
            graphs = ges.load_graphs(gpkg)
        self.assertEqual(graphs["hyetographs"]["h1"]["data"], [(0.0, 9.0)])

    def test_hydrograph_round_trip_exact_series_and_metadata(self):
        series = [(0.0, 0.0), (120.0, 4.5), (240.0, 1.5)]
        with make_temp_model_gpkg() as gpkg:
            ges.save_hydrograph(gpkg, "inflow_1", series,
                                bc_type=2, description="north inflow")
            graphs = ges.load_graphs(gpkg)
        entry = graphs["hydrographs"]["inflow_1"]
        self.assertEqual(entry["data"], series)
        self.assertEqual(entry["bc_type"], 2)
        self.assertEqual(entry["description"], "north inflow")
        self.assertEqual(graphs["hyetographs"], {})

    def test_load_graphs_empty_gpkg_returns_empty_dicts(self):
        with make_temp_model_gpkg() as gpkg:
            graphs = ges.load_graphs(gpkg)
        self.assertEqual(graphs, {"hyetographs": {}, "hydrographs": {}})

    def test_load_graphs_without_graph_tables_raises_loudly(self):
        # A results GPKG has no swe2d_hyetographs/swe2d_hydrographs tables;
        # the service must fail loudly, not return empty data.
        with make_temp_results_gpkg() as gpkg:
            with self.assertRaises(sqlite3.OperationalError):
                ges.load_graphs(gpkg)

    def test_list_graph_ids_groups_by_table_and_sorts(self):
        with make_temp_model_gpkg() as gpkg:
            ges.save_hyetograph(gpkg, "b_hyeto", [(0.0, 1.0)])
            ges.save_hyetograph(gpkg, "a_hyeto", [(0.0, 2.0)])
            ges.save_hydrograph(gpkg, "hydro_1", [(0.0, 3.0)])
            ids = ges.list_graph_ids(gpkg)
        self.assertEqual(ids, {"hyetographs": ["a_hyeto", "b_hyeto"],
                               "hydrographs": ["hydro_1"]})

    def test_list_graph_ids_empty_gpkg_returns_empty_lists(self):
        with make_temp_model_gpkg() as gpkg:
            ids = ges.list_graph_ids(gpkg)
        self.assertEqual(ids, {"hyetographs": [], "hydrographs": []})

    def test_delete_graph_removes_from_load_and_list(self):
        with make_temp_model_gpkg() as gpkg:
            ges.save_hyetograph(gpkg, "keep_h", [(0.0, 1.0)])
            ges.save_hyetograph(gpkg, "drop_h", [(0.0, 2.0)])
            ges.save_hydrograph(gpkg, "drop_q", [(0.0, 3.0)])
            ges.save_hydrograph(gpkg, "keep_q", [(0.0, 4.0)])

            ges.delete_graph(gpkg, "swe2d_hyetographs", "drop_h")
            ges.delete_graph(gpkg, "swe2d_hydrographs", "drop_q")

            graphs = ges.load_graphs(gpkg)
            ids = ges.list_graph_ids(gpkg)
        self.assertEqual(set(graphs["hyetographs"]), {"keep_h"})
        self.assertEqual(set(graphs["hydrographs"]), {"keep_q"})
        self.assertEqual(ids, {"hyetographs": ["keep_h"],
                               "hydrographs": ["keep_q"]})

    def test_delete_graph_missing_id_is_silent_noop(self):
        # FINDING: deleting a non-existent id raises nothing and reports
        # nothing — production silently no-ops (no rowcount check).  This
        # test characterizes the real contract; see task report.
        with make_temp_model_gpkg() as gpkg:
            ges.save_hyetograph(gpkg, "real", [(0.0, 1.0)])
            ges.delete_graph(gpkg, "swe2d_hyetographs", "does_not_exist")
            graphs = ges.load_graphs(gpkg)
        self.assertEqual(set(graphs["hyetographs"]), {"real"})


@requires_qgis
class TestCsvParsing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ensure_qgis_app()

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory(prefix="hydra_test_csv_")

    def tearDown(self):
        self._tmpdir.cleanup()

    def _write_csv(self, text):
        path = os.path.join(self._tmpdir.name, "graph.csv")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return path

    def test_csv_columns_returns_headers(self):
        path = self._write_csv("time_min,discharge,note\n0,1.5,a\n60,2.5,b\n")
        self.assertEqual(ges.csv_columns(path), ["time_min", "discharge", "note"])

    def test_csv_columns_empty_file_returns_empty_list(self):
        # Contract: DictReader.fieldnames is None on an empty file → [].
        path = self._write_csv("")
        self.assertEqual(ges.csv_columns(path), [])

    def test_parse_csv_returns_float_pairs_in_order(self):
        path = self._write_csv(
            "time_min,discharge,note\n0,1.5,a\n60,2.5,b\n120,0.25,c\n"
        )
        data = ges.parse_csv(path, "time_min", "discharge")
        self.assertEqual(data, [(0.0, 1.5), (60.0, 2.5), (120.0, 0.25)])
        for t, v in data:
            self.assertIsInstance(t, float)
            self.assertIsInstance(v, float)

    def test_parse_csv_header_only_returns_empty(self):
        path = self._write_csv("time,value\n")
        self.assertEqual(ges.parse_csv(path, "time", "value"), [])

    def test_parse_csv_empty_file_raises_missing_column(self):
        path = self._write_csv("")
        with self.assertRaises(ValueError) as ctx:
            ges.parse_csv(path, "time", "value")
        msg = str(ctx.exception)
        self.assertIn("time", msg)
        self.assertIn("<none>", msg)

        path = self._write_csv("time,value\n0,1.0\n")
        with self.assertRaises(ValueError) as ctx:
            ges.parse_csv(path, "time", "nonexistent")
        msg = str(ctx.exception)
        self.assertIn("nonexistent", msg)
        self.assertIn("time", msg)  # available columns listed

    def test_parse_csv_nonfloat_row_raises_naming_row(self):
        path = self._write_csv("time,value\n0,1.0\nbogus,2.0\n60,3.0\n")
        with self.assertRaises(ValueError) as ctx:
            ges.parse_csv(path, "time", "value")
        self.assertIn("row 3", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
