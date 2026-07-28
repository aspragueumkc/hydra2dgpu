"""Round-trip tests for pipe-cell (per-sub-cell) coupling persistence.

Verifies that pipe-cell velocity / depth / flow / head data written by
``persist_baked_pipe_cell_ts`` can be read back from the GeoPackage via
direct SQLite query.
"""

import os
import sqlite3
import tempfile
import unittest

import numpy as np

from swe2d.services.gpkg_persistence_service import (
    load_baked_pipe_cell_ts,
    persist_baked_pipe_cell_ts,
)


class TestPipeCellCouplingOutput(unittest.TestCase):
    """Tests for pipe-cell baked timeseries persistence and round-trip."""

    def test_gpkg_pipe_cell_roundtrip(self):
        """pipe-cell velocity/depth/flow/head persist and round-trip correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gpkg = os.path.join(tmpdir, "test.gpkg")
            # Create minimal GPKG with required OGC tables
            conn = sqlite3.connect(gpkg)
            conn.execute(
                "CREATE TABLE gpkg_spatial_ref_sys (srs_name TEXT)"
            )
            conn.commit()
            conn.close()

            items = []
            for metric, vals in [
                ("velocity", [0.5, 0.6, 0.7]),
                ("depth", [0.1, 0.2, 0.3]),
                ("flow", [0.05, 0.06, 0.07]),
                ("head", [1.1, 1.2, 1.3]),
            ]:
                items.append(
                    {
                        "link_id": "L1",
                        "cell_sub_idx": 0,
                        "metric": metric,
                        "times": np.array([0.0, 1.0], dtype=np.float64),
                        "values": np.array([vals[0], vals[1]], dtype=np.float64),
                    }
                )

            persist_baked_pipe_cell_ts(gpkg, "run1", items, log_fn=None)

            # Verify via direct sqlite3 query
            conn = sqlite3.connect(gpkg)
            cur = conn.execute(
                "SELECT metric, n_timesteps FROM swe2d_baked_pipe_cell_ts WHERE run_id='run1'"
            ).fetchall()
            conn.close()

            self.assertEqual(
                len(cur), 4, f"Expected 4 rows (4 metrics), got {len(cur)}"
            )
            for metric, nt in cur:
                self.assertEqual(
                    nt, 2, f"Expected 2 timesteps for {metric}, got {nt}"
                )

    def test_gpkg_pipe_cell_roundtrip_multiple_links_and_cells(self):
        """Multiple links and sub-cells round-trip correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gpkg = os.path.join(tmpdir, "test2.gpkg")
            conn = sqlite3.connect(gpkg)
            conn.execute(
                "CREATE TABLE gpkg_spatial_ref_sys (srs_name TEXT)"
            )
            conn.commit()
            conn.close()

            items = []
            # Two links, 2 sub-cells each, 3 timesteps
            for link_id in ["L1", "L2"]:
                for sub_idx in range(2):
                    for metric in ["depth", "flow"]:
                        times = np.array([0.0, 30.0, 60.0], dtype=np.float64)
                        base = hash(link_id + str(sub_idx) + metric) % 100 / 100.0
                        values = np.array(
                            [base, base + 0.1, base + 0.2], dtype=np.float64
                        )
                        items.append(
                            {
                                "link_id": link_id,
                                "cell_sub_idx": sub_idx,
                                "metric": metric,
                                "times": times,
                                "values": values,
                            }
                        )

            persist_baked_pipe_cell_ts(gpkg, "run_multi", items, log_fn=None)

            conn = sqlite3.connect(gpkg)
            rows = conn.execute(
                "SELECT link_id, cell_sub_idx, metric, n_timesteps, times_blob, values_blob "
                "FROM swe2d_baked_pipe_cell_ts WHERE run_id='run_multi' "
                "ORDER BY link_id, cell_sub_idx, metric"
            ).fetchall()
            conn.close()

            self.assertEqual(
                len(rows), 8, f"Expected 8 rows (2 links × 2 cells × 2 metrics), got {len(rows)}"
            )
            for link_id, cell_sub_idx, metric, nt, times_blob, values_blob in rows:
                self.assertEqual(nt, 3, f"Expected 3 timesteps for {link_id}#{cell_sub_idx}/{metric}")
                times = np.frombuffer(times_blob, dtype=np.float64)
                values = np.frombuffer(values_blob, dtype=np.float64)
                self.assertEqual(len(times), 3)
                self.assertEqual(len(values), 3)
                # Verify times are preserved
                np.testing.assert_array_equal(
                    times,
                    np.array([0.0, 30.0, 60.0], dtype=np.float64),
                )


    def test_gpkg_pipe_cell_geometry_roundtrip(self):
        """Pipe-cell geometry columns (invert, width, height, shape) round-trip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gpkg = os.path.join(tmpdir, "test_geom.gpkg")
            conn = sqlite3.connect(gpkg)
            conn.execute("CREATE TABLE gpkg_spatial_ref_sys (srs_name TEXT)")
            conn.commit()
            conn.close()

            items = []
            for i, (metric, vals) in enumerate([
                ("velocity", [0.5, 0.6]),
                ("depth", [0.1, 0.2]),
            ]):
                items.append({
                    "link_id": "L1",
                    "cell_sub_idx": i,
                    "metric": metric,
                    "times": np.array([0.0, 1.0], dtype=np.float64),
                    "values": np.array(vals, dtype=np.float64),
                    "cell_invert": 1.5 + i,
                    "cell_width": 2.0 + i,
                    "cell_height": 3.0 + i,
                    "cell_shape_type": i,
                })

            persist_baked_pipe_cell_ts(gpkg, "run_geom", items, log_fn=None)

            conn = sqlite3.connect(gpkg)
            loaded = load_baked_pipe_cell_ts(conn, "run_geom")
            conn.close()

            self.assertEqual(len(loaded), 2)
            loaded.sort(key=lambda d: d["cell_sub_idx"])
            for i, item in enumerate(loaded):
                self.assertEqual(item["link_id"], "L1")
                self.assertEqual(item["cell_sub_idx"], i)
                self.assertAlmostEqual(item["cell_invert"], 1.5 + i)
                self.assertAlmostEqual(item["cell_width"], 2.0 + i)
                self.assertAlmostEqual(item["cell_height"], 3.0 + i)
                self.assertEqual(item["cell_shape_type"], i)

    def test_gpkg_pipe_cell_legacy_fallback(self):
        """Loading from an old GeoPackage without geometry columns falls back to defaults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gpkg = os.path.join(tmpdir, "test_legacy.gpkg")
            conn = sqlite3.connect(gpkg)
            conn.execute("CREATE TABLE gpkg_spatial_ref_sys (srs_name TEXT)")
            conn.execute("""
                CREATE TABLE swe2d_baked_pipe_cell_ts (
                    run_id TEXT,
                    link_id TEXT,
                    cell_sub_idx INTEGER,
                    metric TEXT,
                    n_timesteps INTEGER,
                    times_blob BLOB,
                    values_blob BLOB,
                    PRIMARY KEY (run_id, link_id, cell_sub_idx, metric))
            """)
            times = np.array([0.0, 1.0], dtype=np.float64)
            values = np.array([0.5, 0.6], dtype=np.float64)
            conn.execute(
                "INSERT INTO swe2d_baked_pipe_cell_ts VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("run_legacy", "L9", 3, "depth", len(times), times.tobytes(), values.tobytes()),
            )
            conn.commit()

            loaded = load_baked_pipe_cell_ts(conn, "run_legacy")
            conn.close()

            self.assertEqual(len(loaded), 1)
            item = loaded[0]
            self.assertEqual(item["link_id"], "L9")
            self.assertEqual(item["cell_sub_idx"], 3)
            self.assertEqual(item["metric"], "depth")
            np.testing.assert_array_equal(item["times"], times)
            np.testing.assert_array_equal(item["values"], values)
            self.assertAlmostEqual(item["cell_invert"], 0.0)
            self.assertAlmostEqual(item["cell_width"], 1.0)
            self.assertAlmostEqual(item["cell_height"], 1.0)
            self.assertEqual(item["cell_shape_type"], 0)

    def test_build_pipe_cell_items_includes_geometry(self):
        """build_pipe_cell_items must carry through per-cell geometry fields."""
        from swe2d.results.data import SWE2DResultsData

        rd = SWE2DResultsData()
        rd.init_pipe_cell_storage([("L1", 0, "depth")])
        rd.append_pipe_cell_snapshot({
            "link_id": "L1",
            "cell_sub_idx": 0,
            "metric": "depth",
            "t_s": 0.0,
            "value": 1.5,
            "cell_invert": 10.0,
            "cell_width": 2.0,
            "cell_height": 2.5,
            "cell_shape_type": 1,
        })

        items = rd.build_pipe_cell_items()
        self.assertEqual(len(items), 1)
        item = items[0]
        self.assertEqual(item["link_id"], "L1")
        self.assertEqual(item["cell_sub_idx"], 0)
        self.assertEqual(item["metric"], "depth")
        self.assertAlmostEqual(item["cell_invert"], 10.0)
        self.assertAlmostEqual(item["cell_width"], 2.0)
        self.assertAlmostEqual(item["cell_height"], 2.5)
        self.assertEqual(item["cell_shape_type"], 1)


if __name__ == "__main__":
    unittest.main()
