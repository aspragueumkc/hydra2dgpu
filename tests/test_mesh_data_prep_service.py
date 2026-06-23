"""Tests for swe2d.workbench.mesh_data_prep_service.

Pure-Python mesh data preparation service extracted from
``SWE2DWorkbenchStudioDialog._apply_overlay_frame`` and its surrounding
reset / cache paths (Task 4 of
docs/STUDIO_GUI_FULL_MIGRATION_PLAN_2026-06-16.md).

The service prepares the numpy bundles that drive the high-perf canvas
overlay. It must have ZERO Qt or qgis imports so it can be unit-tested
in isolation.
"""
from __future__ import annotations

import ast
import inspect
import os
import unittest

import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVICE_PATH = os.path.join(
    REPO_ROOT, "swe2d", "workbench", "services", "mesh_data_prep_service.py"
)


def _load_module():
    from swe2d.workbench.services import mesh_data_prep_service
    return mesh_data_prep_service


class TestServiceExists(unittest.TestCase):
    def test_module_file_exists(self):
        self.assertTrue(
            os.path.isfile(SERVICE_PATH),
            f"mesh_data_prep_service.py not found at {SERVICE_PATH}",
        )

    def test_module_imports(self):
        mod = _load_module()
        self.assertIsNotNone(mod)

    def test_module_has_no_qt_imports(self):
        with open(SERVICE_PATH, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source)
        forbidden_top = {"PyQt5", "PyQt4", "PySide2", "PySide6", "qgis"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    self.assertNotIn(
                        top, forbidden_top,
                        f"forbidden import: {alias.name}",
                    )
            elif isinstance(node, ast.ImportFrom):
                top = (node.module or "").split(".")[0]
                self.assertNotIn(
                    top, forbidden_top,
                    f"forbidden from-import: {node.module}",
                )


class TestCreateEmptyOverlayArrays(unittest.TestCase):
    def setUp(self):
        self.mod = _load_module()
        self.fn = self.mod.create_empty_overlay_arrays

    def test_returns_dict(self):
        out = self.fn()
        self.assertIsInstance(out, dict)

    def test_cell_x_empty_float64(self):
        out = self.fn()
        self.assertEqual(out["cell_x"].dtype, np.float64)
        self.assertEqual(out["cell_x"].size, 0)

    def test_cell_y_empty_float64(self):
        out = self.fn()
        self.assertEqual(out["cell_y"].dtype, np.float64)
        self.assertEqual(out["cell_y"].size, 0)

    def test_cell_bed_empty_float64(self):
        out = self.fn()
        self.assertEqual(out["cell_bed"].dtype, np.float64)
        self.assertEqual(out["cell_bed"].size, 0)

    def test_node_x_empty_float64(self):
        out = self.fn()
        self.assertEqual(out["node_x"].dtype, np.float64)
        self.assertEqual(out["node_x"].size, 0)

    def test_node_y_empty_float64(self):
        out = self.fn()
        self.assertEqual(out["node_y"].dtype, np.float64)
        self.assertEqual(out["node_y"].size, 0)

    def test_cell_nodes_empty_int32(self):
        out = self.fn()
        self.assertEqual(out["cell_nodes"].dtype, np.int32)
        self.assertEqual(out["cell_nodes"].size, 0)

    def test_tri_to_cell_empty_int32(self):
        out = self.fn()
        self.assertEqual(out["tri_to_cell"].dtype, np.int32)
        self.assertEqual(out["tri_to_cell"].size, 0)

    def test_mesh_fingerprint_empty_string(self):
        out = self.fn()
        self.assertEqual(out["mesh_fingerprint"], "")

    def test_does_not_return_shared_state(self):
        """Two calls must not share the same underlying numpy arrays."""
        out1 = self.fn()
        out2 = self.fn()
        self.assertIsNot(out1["cell_x"], out2["cell_x"])


class TestPrepareOverlayArrays(unittest.TestCase):
    def setUp(self):
        self.mod = _load_module()
        self.fn = self.mod.prepare_overlay_arrays

    def test_cell_centroids_become_cell_x_cell_y(self):
        cx = np.array([0.5, 1.5], dtype=np.float64)
        cy = np.array([0.25, 0.75], dtype=np.float64)
        bed = np.array([0.0, 0.0], dtype=np.float64)
        out = self.fn(
            mesh_data=None,
            cell_centroids_x=cx,
            cell_centroids_y=cy,
            cell_bed=bed,
        )
        np.testing.assert_array_equal(out["cell_x"], cx)
        np.testing.assert_array_equal(out["cell_y"], cy)
        np.testing.assert_array_equal(out["cell_bed"], bed)

    def test_node_x_node_y_from_mesh_data(self):
        mesh = {
            "node_x": np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float64),
            "node_y": np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64),
            "cell_nodes": np.array(
                [[0, 1, 2], [0, 2, 3]], dtype=np.int32,
            ),
        }
        cx = np.array([0.5, 0.5], dtype=np.float64)
        cy = np.array([0.5, 0.5], dtype=np.float64)
        bed = np.array([0.0, 0.0], dtype=np.float64)
        out = self.fn(mesh, cx, cy, bed)
        np.testing.assert_array_equal(out["node_x"], mesh["node_x"])
        np.testing.assert_array_equal(out["node_y"], mesh["node_y"])

    def test_triangulation_when_face_offsets_present(self):
        """Quad cell with face_offsets should fan out into 2 triangles.

        The bridge stores ``cell_nodes`` flat (``.ravel()``); the
        service must match that contract so downstream consumers can
        rely on a flat int32 buffer.
        """
        mesh = {
            "node_x": np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float64),
            "node_y": np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64),
            "cell_face_offsets": np.array([0, 4], dtype=np.int32),
            "cell_face_nodes": np.array([0, 1, 2, 3], dtype=np.int32),
        }
        cx = np.array([0.5], dtype=np.float64)
        cy = np.array([0.5], dtype=np.float64)
        bed = np.array([0.0], dtype=np.float64)
        out = self.fn(mesh, cx, cy, bed)
        self.assertEqual(out["cell_nodes"].dtype, np.int32)
        self.assertEqual(out["cell_nodes"].size, 6)
        np.testing.assert_array_equal(
            out["cell_nodes"],
            np.array([0, 1, 2, 0, 2, 3], dtype=np.int32),
        )
        self.assertEqual(out["tri_to_cell"].size, 2)
        self.assertEqual(out["tri_to_cell"].dtype, np.int32)
        np.testing.assert_array_equal(out["tri_to_cell"], np.array([0, 0]))

    def test_falls_back_to_raw_cell_nodes(self):
        """Without face_offsets, raw cell_nodes is used and tri_to_cell is empty."""
        mesh = {
            "node_x": np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float64),
            "node_y": np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64),
            "cell_nodes": np.array(
                [[0, 1, 2], [0, 2, 3]], dtype=np.int32,
            ),
        }
        cx = np.array([0.5, 0.5], dtype=np.float64)
        cy = np.array([0.5, 0.5], dtype=np.float64)
        bed = np.array([0.0, 0.0], dtype=np.float64)
        out = self.fn(mesh, cx, cy, bed)
        np.testing.assert_array_equal(
            out["cell_nodes"],
            np.array([0, 1, 2, 0, 2, 3], dtype=np.int32),
        )
        self.assertEqual(out["tri_to_cell"].size, 0)

    def test_empty_face_offsets_falls_back(self):
        """If face_offsets is present but produces no triangles, fall back."""
        mesh = {
            "node_x": np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float64),
            "node_y": np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64),
            "cell_face_offsets": np.array([0, 3], dtype=np.int32),
            "cell_face_nodes": np.array([0, 1, 2], dtype=np.int32),
            "cell_nodes": np.array([[0, 1, 2]], dtype=np.int32),
        }
        cx = np.array([0.5], dtype=np.float64)
        cy = np.array([0.5], dtype=np.float64)
        bed = np.array([0.0], dtype=np.float64)
        out = self.fn(mesh, cx, cy, bed)
        np.testing.assert_array_equal(
            out["cell_nodes"],
            np.array([0, 1, 2], dtype=np.int32),
        )

    def test_empty_mesh_data_does_not_crash(self):
        cx = np.array([0.5, 0.5], dtype=np.float64)
        cy = np.array([0.5, 0.5], dtype=np.float64)
        bed = np.array([0.0, 0.0], dtype=np.float64)
        out = self.fn(None, cx, cy, bed)
        self.assertEqual(out["node_x"].size, 0)
        self.assertEqual(out["cell_x"].size, 2)

    def test_empty_dict_mesh_data_does_not_crash(self):
        cx = np.array([0.5, 0.5], dtype=np.float64)
        cy = np.array([0.5, 0.5], dtype=np.float64)
        bed = np.array([0.0, 0.0], dtype=np.float64)
        out = self.fn({}, cx, cy, bed)
        self.assertEqual(out["node_x"].size, 0)
        self.assertEqual(out["cell_x"].size, 2)


class TestOverlayFrameInputs(unittest.TestCase):
    def setUp(self):
        self.mod = _load_module()
        self.fn = self.mod.overlay_frame_inputs

    def test_returns_tuple_of_three(self):
        sentinel_image = object()
        frame = {"image": sentinel_image, "extent": (0, 1, 0, 1)}
        qimage, extent, opacity = self.fn(frame, default_opacity=0.5)
        self.assertIs(qimage, sentinel_image)
        self.assertEqual(tuple(extent), (0.0, 1.0, 0.0, 1.0))
        self.assertAlmostEqual(opacity, 0.5)

    def test_missing_image_returns_none(self):
        qimage, extent, opacity = self.fn({}, default_opacity=1.0)
        self.assertIsNone(qimage)
        self.assertEqual(tuple(extent), (0.0, 1.0, 0.0, 1.0))
        self.assertAlmostEqual(opacity, 1.0)

    def test_missing_extent_returns_unit_square(self):
        sentinel_image = object()
        frame = {"image": sentinel_image}
        _, extent, _ = self.fn(frame, default_opacity=1.0)
        self.assertEqual(tuple(extent), (0.0, 1.0, 0.0, 1.0))

    def test_extent_passed_through(self):
        sentinel_image = object()
        frame = {"image": sentinel_image, "extent": (1.0, 5.0, 2.0, 8.0)}
        _, extent, _ = self.fn(frame, default_opacity=1.0)
        self.assertEqual(tuple(extent), (1.0, 5.0, 2.0, 8.0))

    def test_default_opacity_is_one(self):
        sentinel_image = object()
        frame = {"image": sentinel_image, "extent": (0, 1, 0, 1)}
        _, _, opacity = self.fn(frame, default_opacity=1.0)
        self.assertAlmostEqual(opacity, 1.0)

    def test_opacity_is_coerced_to_float(self):
        sentinel_image = object()
        frame = {"image": sentinel_image, "extent": (0, 1, 0, 1)}
        _, _, opacity = self.fn(frame, default_opacity=0)
        self.assertIsInstance(opacity, float)
        self.assertAlmostEqual(opacity, 0.0)


class TestOverlayFrameIsValid(unittest.TestCase):
    def setUp(self):
        self.mod = _load_module()
        self.fn = self.mod.overlay_frame_is_valid

    def test_none_is_invalid(self):
        self.assertFalse(self.fn(None))

    def test_object_without_isNull_is_valid(self):
        """A non-None object without isNull() should be considered valid."""
        self.assertTrue(self.fn(object()))

    def test_object_with_isNull_false_is_valid(self):
        class FakeQImage:
            def isNull(self):
                return False
        self.assertTrue(self.fn(FakeQImage()))

    def test_object_with_isNull_true_is_invalid(self):
        class FakeQImage:
            def isNull(self):
                return True
        self.assertFalse(self.fn(FakeQImage()))

    def test_isNull_raising_is_treated_invalid(self):
        class FakeQImage:
            def isNull(self):
                raise RuntimeError("deleted")
        self.assertFalse(self.fn(FakeQImage()))


class TestPublicApiTyped(unittest.TestCase):
    def test_all_public_functions_have_annotations(self):
        mod = _load_module()
        for name in (
            "create_empty_overlay_arrays",
            "prepare_overlay_arrays",
            "overlay_frame_inputs",
            "overlay_frame_is_valid",
        ):
            fn = getattr(mod, name)
            sig = inspect.signature(fn)
            for pname, p in sig.parameters.items():
                self.assertNotEqual(
                    p.annotation, inspect.Parameter.empty,
                    f"{name}.{pname} is missing a type annotation",
                )
            self.assertNotEqual(
                sig.return_annotation, inspect.Signature.empty,
                f"{name} is missing a return type annotation",
            )


if __name__ == "__main__":
    unittest.main()
