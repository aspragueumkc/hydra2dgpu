"""Tests for gpkg_service.load_mesh_snapshot."""
import os
import sqlite3
import tempfile
import unittest

import numpy as np

from swe2d.workbench.services.gpkg_service import load_mesh_snapshot


def create_test_gpkg(gpkg_path, table_name, run_id, t_s, h, hu, hv):
    """Create a test GPKG with mesh results data."""
    conn = sqlite3.connect(gpkg_path)
    cur = conn.cursor()
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS "{table_name}" (
            run_id TEXT,
            t_s REAL,
            cell_id INTEGER,
            h REAL,
            hu REAL,
            hv REAL
        )
    """
    )
    for i, (h_val, hu_val, hv_val) in enumerate(zip(h, hu, hv)):
        cur.execute(
            f'INSERT INTO "{table_name}" (run_id, t_s, cell_id, h, hu, hv) VALUES (?, ?, ?, ?, ?, ?)',
            (run_id, t_s, i, h_val, hu_val, hv_val),
        )
    conn.commit()
    conn.close()


class TestLoadMeshSnapshot(unittest.TestCase):
    def test_service_imports(self):
        self.assertIsNotNone(load_mesh_snapshot)

    def test_nonexistent_gpkg_returns_none(self):
        result = load_mesh_snapshot("/nonexistent/path.gpkg", "run1", 1.0)
        self.assertIsNone(result)

    def test_empty_gpkg_returns_none(self):
        with tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False) as f:
            gpkg_path = f.name
        try:
            result = load_mesh_snapshot(gpkg_path, "run1", 1.0)
            self.assertIsNone(result)
        finally:
            os.unlink(gpkg_path)

    def test_returns_correct_data(self):
        with tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False) as f:
            gpkg_path = f.name
        try:
            h = [1.0, 2.0, 3.0]
            hu = [0.1, 0.2, 0.3]
            hv = [0.0, 0.0, 0.0]
            create_test_gpkg(gpkg_path, "swe2d_mesh_results", "run1", 1.0, h, hu, hv)
            result = load_mesh_snapshot(gpkg_path, "run1", 1.0)
            self.assertIsNotNone(result)
            self.assertIn("h", result)
            self.assertIn("hu", result)
            self.assertIn("hv", result)
            self.assertIn("t_s", result)
            self.assertIn("cell_count", result)
            np.testing.assert_array_almost_equal(result["h"], h)
            np.testing.assert_array_almost_equal(result["hu"], hu)
            np.testing.assert_array_almost_equal(result["hv"], hv)
            self.assertEqual(result["t_s"], 1.0)
            self.assertEqual(result["cell_count"], 3)
        finally:
            os.unlink(gpkg_path)

    def test_nearest_timestep_match(self):
        with tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False) as f:
            gpkg_path = f.name
        try:
            h = [1.0, 2.0]
            hu = [0.1, 0.2]
            hv = [0.0, 0.0]
            # Store at t=1.0 and t=3.0
            create_test_gpkg(gpkg_path, "swe2d_mesh_results", "run1", 1.0, h, hu, hv)
            create_test_gpkg(gpkg_path, "swe2d_mesh_results", "run1", 3.0, h, hu, hv)
            # Request t=2.0, should get t=1.0 (nearest)
            result = load_mesh_snapshot(gpkg_path, "run1", 2.0)
            self.assertEqual(result["t_s"], 1.0)
        finally:
            os.unlink(gpkg_path)

    def test_prefixed_table_name(self):
        with tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False) as f:
            gpkg_path = f.name
        try:
            h = [1.0]
            hu = [0.1]
            hv = [0.0]
            # Use a prefixed table name like "extra_swe2d_mesh_results"
            create_test_gpkg(gpkg_path, "extra_swe2d_mesh_results", "run1", 1.0, h, hu, hv)
            result = load_mesh_snapshot(gpkg_path, "run1", 1.0)
            self.assertIsNotNone(result)
        finally:
            os.unlink(gpkg_path)


if __name__ == "__main__":
    unittest.main()
