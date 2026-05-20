import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from swe3d_geometry_ingest import (  # noqa: E402
    PatchGridSpec,
    apply_instance_transform,
    build_static_geometry_tensors,
    load_obj_mesh,
    write_solid_voxels_obj,
)


_CUBE_OBJ = """
# unit cube
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
v 0 0 1
v 1 0 1
v 1 1 1
v 0 1 1
f 1 2 3 4
f 5 8 7 6
f 1 5 6 2
f 2 6 7 3
f 3 7 8 4
f 5 1 4 8
""".strip()


_OPEN_BOX_OBJ = """
# unit box with open top (non-airtight)
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
v 0 0 1
v 1 0 1
v 1 1 1
v 0 1 1
f 1 2 3 4
f 1 5 6 2
f 2 6 7 3
f 3 7 8 4
f 5 1 4 8
""".strip()


class TestSWE3DGeometryIngest(unittest.TestCase):
    def _write_obj(self, payload: str) -> str:
        handle = tempfile.NamedTemporaryFile(mode="w", suffix=".obj", delete=False)
        handle.write(str(payload))
        handle.close()
        self.addCleanup(lambda: os.path.exists(handle.name) and os.remove(handle.name))
        return handle.name

    def _write_cube_obj(self) -> str:
        return self._write_obj(_CUBE_OBJ)

    def test_load_obj_mesh_triangulates_faces(self):
        obj_path = self._write_cube_obj()
        vertices, faces = load_obj_mesh(obj_path)

        self.assertEqual(vertices.shape, (8, 3))
        # 6 quad faces -> 12 triangles after fan triangulation.
        self.assertEqual(faces.shape, (12, 3))
        self.assertTrue(np.all(faces >= 0))

    def test_mesh_voxelization_blocks_interior_cells(self):
        obj_path = self._write_cube_obj()
        vertices, faces = load_obj_mesh(obj_path)

        world_vertices = apply_instance_transform(
            vertices,
            translation_xyz=(2.0, 2.0, 2.0),
            scale_xyz=(2.0, 2.0, 2.0),
            yaw_deg=0.0,
        )

        spec = PatchGridSpec(
            nx=8,
            ny=8,
            nz=8,
            dx=1.0,
            dy=1.0,
            dz=1.0,
            origin_x=0.0,
            origin_y=0.0,
            origin_z=0.0,
        )

        phi, ax, ay, az, diag = build_static_geometry_tensors(
            spec=spec,
            mesh_items=[(world_vertices, faces)],
            terrain_elevation=None,
        )

        n_cells = spec.nx * spec.ny * spec.nz
        self.assertEqual(phi.size, n_cells)
        self.assertEqual(ax.size, n_cells)
        self.assertEqual(ay.size, n_cells)
        self.assertEqual(az.size, n_cells)

        self.assertGreater(int(diag["solid_cells"]), 0)
        self.assertLess(int(diag["solid_cells"]), n_cells)

        # Center at (2.5, 2.5, 2.5) should be inside translated/scaled cube [2,4]^3.
        idx_inside = 2 + spec.nx * (2 + spec.ny * 2)
        self.assertLess(phi[idx_inside], 0.5)

        idx_outside = 0
        self.assertGreater(phi[idx_outside], 0.5)

        self.assertTrue(np.all((ax >= 0.0) & (ax <= 1.0)))
        self.assertTrue(np.all((ay >= 0.0) & (ay <= 1.0)))
        self.assertTrue(np.all((az >= 0.0) & (az <= 1.0)))

    def test_terrain_elevation_blocks_lower_layers(self):
        spec = PatchGridSpec(
            nx=4,
            ny=3,
            nz=4,
            dx=1.0,
            dy=1.0,
            dz=1.0,
            origin_x=0.0,
            origin_y=0.0,
            origin_z=0.0,
        )

        # z-centers are 0.5, 1.5, 2.5, 3.5 so first three layers are solid.
        terrain = np.full((spec.ny, spec.nx), 2.6, dtype=np.float64)
        phi, ax, ay, az, diag = build_static_geometry_tensors(
            spec=spec,
            mesh_items=[],
            terrain_elevation=terrain,
        )

        expected_solid = 3 * spec.nx * spec.ny
        self.assertEqual(int(diag["solid_cells"]), expected_solid)
        self.assertEqual(int(diag["terrain_solid_cells"]), expected_solid)
        self.assertEqual(int(diag["mesh_solid_cells"]), 0)

        self.assertTrue(np.all((phi >= 0.0) & (phi <= 1.0)))
        self.assertTrue(np.all((ax >= 0.0) & (ax <= 1.0)))
        self.assertTrue(np.all((ay >= 0.0) & (ay <= 1.0)))
        self.assertTrue(np.all((az >= 0.0) & (az <= 1.0)))

    def test_non_airtight_mesh_with_outside_point_seed(self):
        obj_path = self._write_obj(_OPEN_BOX_OBJ)
        vertices, faces = load_obj_mesh(obj_path)

        world_vertices = apply_instance_transform(
            vertices,
            translation_xyz=(2.0, 2.0, 2.0),
            scale_xyz=(2.0, 2.0, 2.0),
            yaw_deg=0.0,
        )

        spec = PatchGridSpec(
            nx=8,
            ny=8,
            nz=8,
            dx=1.0,
            dy=1.0,
            dz=1.0,
            origin_x=0.0,
            origin_y=0.0,
            origin_z=0.0,
        )

        phi_seed, _, _, _, diag_seed = build_static_geometry_tensors(
            spec=spec,
            mesh_items=[
                {
                    "vertices": world_vertices,
                    "faces": faces,
                    "outside_point": (1.5, 3.0, 3.0),
                }
            ],
            terrain_elevation=None,
        )

        idx_inside = 2 + spec.nx * (2 + spec.ny * 2)
        self.assertLess(phi_seed[idx_inside], 0.5)
        self.assertGreaterEqual(int(diag_seed.get("mesh_seed_instances_requested", 0)), 1)

    def test_fractional_cutcell_reconstruction_produces_intermediate_values(self):
        obj_path = self._write_cube_obj()
        vertices, faces = load_obj_mesh(obj_path)

        world_vertices = apply_instance_transform(
            vertices,
            translation_xyz=(2.25, 2.25, 2.25),
            scale_xyz=(2.0, 2.0, 2.0),
            yaw_deg=0.0,
        )

        spec = PatchGridSpec(
            nx=8,
            ny=8,
            nz=8,
            dx=1.0,
            dy=1.0,
            dz=1.0,
            origin_x=0.0,
            origin_y=0.0,
            origin_z=0.0,
        )

        phi, ax, ay, az, diag = build_static_geometry_tensors(
            spec=spec,
            mesh_items=[(world_vertices, faces)],
            terrain_elevation=None,
            cutcell_samples_per_axis=3,
        )

        frac_phi = np.where((phi > 1.0e-6) & (phi < 1.0 - 1.0e-6))[0]
        frac_ax = np.where((ax > 1.0e-6) & (ax < 1.0 - 1.0e-6))[0]
        frac_ay = np.where((ay > 1.0e-6) & (ay < 1.0 - 1.0e-6))[0]
        frac_az = np.where((az > 1.0e-6) & (az < 1.0 - 1.0e-6))[0]

        self.assertGreater(frac_phi.size, 0, "Expected fractional phi values near cut-cell boundaries")
        self.assertGreater(frac_ax.size, 0, "Expected fractional ax values near cut-cell boundaries")
        self.assertGreater(frac_ay.size, 0, "Expected fractional ay values near cut-cell boundaries")
        self.assertGreater(frac_az.size, 0, "Expected fractional az values near cut-cell boundaries")

        idx_solid_core = 3 + spec.nx * (3 + spec.ny * 3)
        idx_far_outside = 0
        self.assertLess(phi[idx_solid_core], 0.5)
        self.assertGreater(phi[idx_far_outside], 0.9)
        self.assertGreaterEqual(float(diag.get("cutcell_samples_per_axis", 0.0)), 3.0)

    def test_favor1981_porosity_mode_changes_face_area_reconstruction(self):
        obj_path = self._write_cube_obj()
        vertices, faces = load_obj_mesh(obj_path)

        world_vertices = apply_instance_transform(
            vertices,
            translation_xyz=(2.15, 2.10, 2.20),
            scale_xyz=(2.0, 2.0, 2.0),
            yaw_deg=27.0,
        )

        spec = PatchGridSpec(
            nx=8,
            ny=8,
            nz=8,
            dx=1.0,
            dy=1.0,
            dz=1.0,
            origin_x=0.0,
            origin_y=0.0,
            origin_z=0.0,
        )

        _, ax_default, ay_default, az_default, _ = build_static_geometry_tensors(
            spec=spec,
            mesh_items=[(world_vertices, faces)],
            terrain_elevation=None,
            cutcell_samples_per_axis=3,
            obstacle_method="fractional_cutcell",
        )

        phi_favor, ax_favor, ay_favor, az_favor, diag_favor = build_static_geometry_tensors(
            spec=spec,
            mesh_items=[(world_vertices, faces)],
            terrain_elevation=None,
            cutcell_samples_per_axis=3,
            obstacle_method="favor1981_porosity",
        )

        self.assertTrue(np.all((phi_favor >= 0.0) & (phi_favor <= 1.0)))
        self.assertTrue(np.all((ax_favor >= 0.0) & (ax_favor <= 1.0)))
        self.assertTrue(np.all((ay_favor >= 0.0) & (ay_favor <= 1.0)))
        self.assertTrue(np.all((az_favor >= 0.0) & (az_favor <= 1.0)))

        frac_face = np.where((ax_favor > 1.0e-6) & (ax_favor < 1.0 - 1.0e-6))[0]
        self.assertGreater(frac_face.size, 0, "Expected FAVOR-like mode to produce fractional face-open values")

        diff_norm = max(
            float(np.max(np.abs(ax_favor - ax_default))),
            float(np.max(np.abs(ay_favor - ay_default))),
            float(np.max(np.abs(az_favor - az_default))),
        )
        self.assertGreater(diff_norm, 1.0e-6, "Expected FAVOR-like face reconstruction to differ from default pair-min mode")

        self.assertGreaterEqual(float(diag_favor.get("obstacle_method_favor1981", 0.0)), 1.0)

    def test_write_solid_voxels_obj_exports_faces(self):
        spec = PatchGridSpec(
            nx=2,
            ny=2,
            nz=2,
            dx=1.0,
            dy=1.0,
            dz=1.0,
            origin_x=0.0,
            origin_y=0.0,
            origin_z=0.0,
        )
        phi3 = np.ones((spec.nz, spec.ny, spec.nx), dtype=np.float64)
        phi3[0, 0, 0] = 0.0
        phi = phi3.ravel(order="C")

        obj_path = self._write_obj("")
        summary = write_solid_voxels_obj(
            spec=spec,
            phi=phi,
            file_path=obj_path,
            solid_threshold=0.5,
        )

        self.assertGreaterEqual(float(summary.get("solid_cells", 0.0)), 1.0)
        self.assertGreater(float(summary.get("vertices", 0.0)), 0.0)
        self.assertGreater(float(summary.get("faces", 0.0)), 0.0)

        with open(obj_path, "r", encoding="utf-8") as handle:
            payload = handle.read()
        self.assertIn("\nv ", "\n" + payload)
        self.assertIn("\nf ", "\n" + payload)


if __name__ == "__main__":
    unittest.main()
