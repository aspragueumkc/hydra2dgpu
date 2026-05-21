import os
import sqlite3
import tempfile
import unittest

from swe2d.results.velocity_layer import VelocityVectorBuilder


class TestVelocityVectorBuilder(unittest.TestCase):
    def _make_db(self) -> str:
        fd, path = tempfile.mkstemp(suffix=".gpkg")
        os.close(fd)
        conn = sqlite3.connect(path)
        try:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE swe2d_mesh_results (
                    run_id TEXT,
                    t_s REAL,
                    cell_id INTEGER,
                    h REAL,
                    hu REAL,
                    hv REAL,
                    PRIMARY KEY (run_id, t_s, cell_id)
                )
                """
            )
            conn.commit()
        finally:
            conn.close()
        return path

    def test_load_snapshot_falls_back_to_cell_momentum(self):
        db = self._make_db()
        try:
            conn = sqlite3.connect(db)
            try:
                conn.execute(
                    "INSERT INTO swe2d_mesh_results(run_id, t_s, cell_id, h, hu, hv) VALUES (?, ?, ?, ?, ?, ?)",
                    ("run_a", 10.0, 0, 2.0, 6.0, 8.0),
                )
                conn.commit()
            finally:
                conn.close()

            builder = VelocityVectorBuilder()
            snap = builder.load_snapshot(db, "run_a", 10.0, t_tol=0.1)
            self.assertIsNotNone(snap)
            assert snap is not None
            self.assertEqual(snap.source, "cell_momentum")
            self.assertAlmostEqual(float(snap.hu[0]), 6.0, places=9)
            self.assertAlmostEqual(float(snap.hv[0]), 8.0, places=9)
        finally:
            os.remove(db)

    def test_load_snapshot_reconstructs_from_face_flux(self):
        db = self._make_db()
        try:
            conn = sqlite3.connect(db)
            try:
                conn.execute(
                    "INSERT INTO swe2d_mesh_results(run_id, t_s, cell_id, h, hu, hv) VALUES (?, ?, ?, ?, ?, ?)",
                    ("run_b", 20.0, 7, 2.0, 0.0, 0.0),
                )
                conn.execute(
                    "INSERT INTO swe2d_mesh_results(run_id, t_s, cell_id, h, hu, hv) VALUES (?, ?, ?, ?, ?, ?)",
                    ("run_b", 20.0, 8, 2.0, 5.0, 6.0),
                )
                conn.execute(
                    """
                    CREATE TABLE swe2d_face_flux_results (
                        run_id TEXT,
                        t_s REAL,
                        cell_id INTEGER,
                        nx REAL,
                        ny REAL,
                        flux_n REAL,
                        face_length REAL
                    )
                    """
                )
                conn.execute(
                    "INSERT INTO swe2d_face_flux_results(run_id, t_s, cell_id, nx, ny, flux_n, face_length) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    ("run_b", 20.0, 7, 1.0, 0.0, 3.0, 1.0),
                )
                conn.execute(
                    "INSERT INTO swe2d_face_flux_results(run_id, t_s, cell_id, nx, ny, flux_n, face_length) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    ("run_b", 20.0, 7, 0.0, 1.0, 4.0, 1.0),
                )
                conn.commit()
            finally:
                conn.close()

            builder = VelocityVectorBuilder()
            snap = builder.load_snapshot(db, "run_b", 20.0, t_tol=0.1)
            self.assertIsNotNone(snap)
            assert snap is not None
            self.assertEqual(snap.source, "face_flux_reconstruction")
            self.assertAlmostEqual(float(snap.hu[0]), 3.0, places=9)
            self.assertAlmostEqual(float(snap.hv[0]), 4.0, places=9)
            # Cell 8 has no face rows, so raw cell momentum stays in place.
            self.assertAlmostEqual(float(snap.hu[1]), 5.0, places=9)
            self.assertAlmostEqual(float(snap.hv[1]), 6.0, places=9)

            vecs = builder.build_vectors(
                snapshot=snap,
                cell_xy={7: (0.0, 0.0), 8: (1.0, 0.0)},
                stride=1,
                min_depth=1.0e-6,
                min_speed=0.0,
            )
            by_id = {int(v["cell_id"]): v for v in vecs}
            self.assertIn(7, by_id)
            self.assertAlmostEqual(float(by_id[7]["u"]), 1.5, places=9)
            self.assertAlmostEqual(float(by_id[7]["v"]), 2.0, places=9)
        finally:
            os.remove(db)

    def test_build_streamline_traces_from_snapshot(self):
        db = self._make_db()
        try:
            conn = sqlite3.connect(db)
            try:
                for j in range(3):
                    for i in range(3):
                        cid = j * 3 + i
                        conn.execute(
                            "INSERT INTO swe2d_mesh_results(run_id, t_s, cell_id, h, hu, hv) VALUES (?, ?, ?, ?, ?, ?)",
                            ("run_c", 30.0, cid, 1.0, 1.0, 0.0),
                        )
                conn.commit()
            finally:
                conn.close()

            builder = VelocityVectorBuilder()
            snap = builder.load_snapshot(db, "run_c", 30.0, t_tol=0.1)
            self.assertIsNotNone(snap)
            assert snap is not None

            cell_xy = {cid: (float(cid % 3), float(cid // 3)) for cid in range(9)}
            traces = builder.build_streamline_traces(
                snapshot=snap,
                cell_xy=cell_xy,
                seed_count=6,
                max_steps=6,
                step_len_factor=0.4,
                min_depth=1.0e-6,
                min_speed=0.01,
                seed_stride=1,
            )
            self.assertGreater(len(traces), 0)

            first = traces[0]
            points = first.get("points", [])
            self.assertGreaterEqual(len(points), 3)
            self.assertGreater(float(first.get("length", 0.0)), 0.0)
            self.assertGreater(float(first.get("mean_speed", 0.0)), 0.1)
        finally:
            os.remove(db)


if __name__ == "__main__":
    unittest.main()
