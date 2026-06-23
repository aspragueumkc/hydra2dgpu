"""Tests for swe2d.workbench.mesh_render_service.

Pure-Python rendering service. Zero Qt imports. The service takes mesh
data + result data + mode + h_min as plain parameters and returns a
rendered image as a numpy array.
"""
import ast
import inspect
import os
import unittest

import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVICE_PATH = os.path.join(
    REPO_ROOT, "swe2d", "workbench", "services", "mesh_render_service.py"
)


def _make_simple_mesh_data():
    """4-node 2-triangle mesh on the unit square."""
    node_x = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float64)
    node_y = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
    cell_nodes = np.array(
        [[0, 1, 2], [0, 2, 3]],
        dtype=np.int32,
    )
    return {
        "node_x": node_x,
        "node_y": node_y,
        "cell_nodes": cell_nodes,
    }


def _make_simple_result_data(n_cells):
    """Depth/velocity values for a 2-cell mesh."""
    h = np.array([1.0, 2.0], dtype=np.float64) if n_cells == 2 else np.zeros(n_cells)
    hu = np.array([0.5, 1.0], dtype=np.float64) if n_cells == 2 else np.zeros(n_cells)
    hv = np.array([0.25, 0.5], dtype=np.float64) if n_cells == 2 else np.zeros(n_cells)
    return {"h": h, "hu": hu, "hv": hv}


def _make_face_offsets_mesh_data():
    """Mesh with cell_face_offsets + cell_face_nodes (1 quad cell, 4 nodes)."""
    node_x = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float64)
    node_y = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
    cell_face_offsets = np.array([0, 4], dtype=np.int32)
    cell_face_nodes = np.array([0, 1, 2, 3], dtype=np.int32)
    return {
        "node_x": node_x,
        "node_y": node_y,
        "cell_face_offsets": cell_face_offsets,
        "cell_face_nodes": cell_face_nodes,
    }


class TestServiceExists(unittest.TestCase):
    def test_module_imports(self):
        from swe2d.workbench.services import mesh_render_service
        self.assertIsNotNone(mesh_render_service)

    def test_render_function_exists(self):
        from swe2d.workbench.services.mesh_render_service import render_workbench_mesh_view
        self.assertIsNotNone(render_workbench_mesh_view)
        self.assertTrue(callable(render_workbench_mesh_view))

    def test_module_has_no_qt_imports(self):
        """The service must be pure Python — no Qt or qgis imports."""
        with open(SERVICE_PATH) as f:
            source = f.read()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertFalse(
                        alias.name.startswith("PyQt"),
                        f"Qt import found: {alias.name}",
                    )
                    self.assertFalse(
                        alias.name.startswith("qgis"),
                        f"qgis import found: {alias.name}",
                    )
            elif isinstance(node, ast.ImportFrom):
                if node.module is None:
                    continue
                self.assertFalse(
                    node.module.startswith("PyQt"),
                    f"Qt import found: from {node.module} ...",
                )
                self.assertFalse(
                    node.module.startswith("qgis"),
                    f"qgis import found: from {node.module} ...",
                )


class TestSignature(unittest.TestCase):
    def test_signature_takes_plain_dicts(self):
        from swe2d.workbench.services.mesh_render_service import render_workbench_mesh_view
        sig = inspect.signature(render_workbench_mesh_view)
        params = sig.parameters
        self.assertIn("mesh_data", params)
        self.assertIn("result_data", params)
        self.assertIn("mode", params)
        self.assertIn("h_min", params)

    def test_signature_is_typed(self):
        """All public parameters must carry type annotations (PEP 484)."""
        from swe2d.workbench.services.mesh_render_service import render_workbench_mesh_view
        sig = inspect.signature(render_workbench_mesh_view)
        for name in ("mesh_data", "result_data", "mode", "h_min"):
            p = sig.parameters[name]
            self.assertNotEqual(
                p.annotation, inspect.Parameter.empty,
                f"Parameter {name!r} is missing a type annotation",
            )


class TestReturnValue(unittest.TestCase):
    def test_returns_numpy_array(self):
        from swe2d.workbench.services.mesh_render_service import render_workbench_mesh_view
        mesh = _make_simple_mesh_data()
        img = render_workbench_mesh_view(mesh, None, mode="mesh")
        self.assertIsInstance(img, np.ndarray)

    def test_returns_uint8(self):
        from swe2d.workbench.services.mesh_render_service import render_workbench_mesh_view
        mesh = _make_simple_mesh_data()
        img = render_workbench_mesh_view(mesh, None, mode="mesh")
        self.assertEqual(img.dtype, np.uint8)

    def test_returns_3d_rgb_image(self):
        """Image must be shape (H, W, 3) for RGB."""
        from swe2d.workbench.services.mesh_render_service import render_workbench_mesh_view
        mesh = _make_simple_mesh_data()
        img = render_workbench_mesh_view(mesh, None, mode="mesh")
        self.assertEqual(img.ndim, 3)
        self.assertEqual(img.shape[2], 3)
        self.assertGreater(img.shape[0], 0)
        self.assertGreater(img.shape[1], 0)

    def test_image_is_non_trivially_colored(self):
        """A rendered image must have some non-background pixels."""
        from swe2d.workbench.services.mesh_render_service import render_workbench_mesh_view
        mesh = _make_simple_mesh_data()
        img = render_workbench_mesh_view(mesh, None, mode="mesh")
        self.assertGreater(int(img.size), 0)
        unique_colors = np.unique(img.reshape(-1, 3), axis=0)
        self.assertGreater(
            len(unique_colors), 1,
            "Rendered image is single-color — rendering failed",
        )


class TestMeshMode(unittest.TestCase):
    def test_mesh_mode_renders(self):
        from swe2d.workbench.services.mesh_render_service import render_workbench_mesh_view
        mesh = _make_simple_mesh_data()
        img = render_workbench_mesh_view(mesh, None, mode="mesh")
        self.assertIsInstance(img, np.ndarray)
        self.assertEqual(img.dtype, np.uint8)
        self.assertEqual(img.shape[2], 3)

    def test_mesh_mode_with_result_data_still_renders_mesh(self):
        """When mode='mesh', result_data should be ignored and the mesh drawn."""
        from swe2d.workbench.services.mesh_render_service import render_workbench_mesh_view
        mesh = _make_simple_mesh_data()
        result = _make_simple_result_data(n_cells=2)
        img = render_workbench_mesh_view(mesh, result, mode="mesh")
        self.assertIsInstance(img, np.ndarray)
        self.assertEqual(img.shape[2], 3)

    def test_face_offsets_mesh_renders(self):
        """Mesh with cell_face_offsets + cell_face_nodes must render too."""
        from swe2d.workbench.services.mesh_render_service import render_workbench_mesh_view
        mesh = _make_face_offsets_mesh_data()
        img = render_workbench_mesh_view(mesh, None, mode="mesh")
        self.assertIsInstance(img, np.ndarray)
        self.assertEqual(img.shape[2], 3)


class TestDepthMode(unittest.TestCase):
    def test_depth_mode_renders(self):
        from swe2d.workbench.services.mesh_render_service import render_workbench_mesh_view
        mesh = _make_simple_mesh_data()
        result = _make_simple_result_data(n_cells=2)
        img = render_workbench_mesh_view(mesh, result, mode="depth")
        self.assertIsInstance(img, np.ndarray)
        self.assertEqual(img.dtype, np.uint8)
        self.assertEqual(img.shape[2], 3)


class TestVelocityMode(unittest.TestCase):
    def test_velocity_mode_renders(self):
        from swe2d.workbench.services.mesh_render_service import render_workbench_mesh_view
        mesh = _make_simple_mesh_data()
        result = _make_simple_result_data(n_cells=2)
        img = render_workbench_mesh_view(mesh, result, mode="velocity")
        self.assertIsInstance(img, np.ndarray)
        self.assertEqual(img.dtype, np.uint8)
        self.assertEqual(img.shape[2], 3)

    def test_velocity_mode_h_min_threshold(self):
        """When h <= h_min, velocity must be zero. We test by setting very high h_min."""
        from swe2d.workbench.services.mesh_render_service import render_workbench_mesh_view
        mesh = _make_simple_mesh_data()
        result = _make_simple_result_data(n_cells=2)
        img_high_threshold = render_workbench_mesh_view(
            mesh, result, mode="velocity", h_min=1.0e6,
        )
        img_low_threshold = render_workbench_mesh_view(
            mesh, result, mode="velocity", h_min=1.0e-12,
        )
        self.assertIsInstance(img_high_threshold, np.ndarray)
        self.assertIsInstance(img_low_threshold, np.ndarray)
        self.assertFalse(
            np.array_equal(img_high_threshold, img_low_threshold),
            "h_min threshold had no effect on rendered velocity",
        )


class TestNoneMeshData(unittest.TestCase):
    def test_none_mesh_renders_placeholder(self):
        """When mesh_data is None, the service must return an image (placeholder)."""
        from swe2d.workbench.services.mesh_render_service import render_workbench_mesh_view
        img = render_workbench_mesh_view(None, None, mode="mesh")
        self.assertIsInstance(img, np.ndarray)
        self.assertEqual(img.dtype, np.uint8)
        self.assertEqual(img.shape[2], 3)


class TestInvalidMeshData(unittest.TestCase):
    def test_missing_cell_nodes_renders_placeholder(self):
        """When mesh_data lacks both face_offsets and cell_nodes, the service must not crash."""
        from swe2d.workbench.services.mesh_render_service import render_workbench_mesh_view
        mesh = {
            "node_x": np.array([0.0, 1.0, 0.5], dtype=np.float64),
            "node_y": np.array([0.0, 0.0, 1.0], dtype=np.float64),
        }
        img = render_workbench_mesh_view(mesh, None, mode="mesh")
        self.assertIsInstance(img, np.ndarray)
        self.assertEqual(img.dtype, np.uint8)
        self.assertEqual(img.shape[2], 3)


class TestHMinIsFloat(unittest.TestCase):
    def test_h_min_is_optional_with_default(self):
        from swe2d.workbench.services.mesh_render_service import render_workbench_mesh_view
        sig = inspect.signature(render_workbench_mesh_view)
        p = sig.parameters["h_min"]
        self.assertIsNot(p.default, inspect.Parameter.empty)
        self.assertIsInstance(p.default, float)


if __name__ == "__main__":
    unittest.main()
