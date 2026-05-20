import importlib
import os
import sys
import unittest
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from swe2d_backend import configure_swe3d_runtime  # noqa: E402
from swe2d_workbench_non_gui_runtime import upload_experimental_3d_obj_geometry  # noqa: E402


@contextmanager
def _temporary_env(overrides):
    old = {}
    try:
        for key, value in (overrides or {}).items():
            old[key] = os.environ.get(key)
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        yield
    finally:
        for key, prev in old.items():
            if prev is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prev


class _Check:
    def __init__(self, checked):
        self._checked = bool(checked)

    def isChecked(self):
        return self._checked


class _BackendStub:
    def __init__(self):
        self.upload_calls = 0
        self.last_kwargs = None

    def supports_3d_patch_geometry_upload(self):
        return True

    def set_3d_patch_geometry(self, **kwargs):
        self.upload_calls += 1
        self.last_kwargs = dict(kwargs)


class _WorkbenchStub:
    def __init__(self):
        self.experimental_3d_obj_solids_chk = _Check(True)
        self.experimental_3d_obj_use_terrain_chk = _Check(True)
        self.logs = []

    def _log(self, message):
        self.logs.append(str(message))

    def _build_patch_spec_from_stats(self, patch_stats, swe3d_env_overrides):
        return SimpleNamespace(
            nx=int(patch_stats.get("nx", 0)),
            ny=int(patch_stats.get("ny", 0)),
            nz=int(patch_stats.get("nz", 0)),
            dx=1.0,
            dy=1.0,
            dz=1.0,
            origin_x=0.0,
            origin_y=0.0,
            origin_z=0.0,
        )

    def _build_patch_terrain_surface(self, spec):
        return np.zeros((int(spec.ny), int(spec.nx)), dtype=np.float64)

    def _experimental_3d_selected_obstacle_method(self):
        return "fractional_cutcell"

    def _experimental_3d_geometry_sanitize_options(self):
        return {"sanitize": False}


def _tensor_builder_with_diag(diag):
    def _build(spec, mesh_items, terrain_elevation, obstacle_method):
        n_cells = int(spec.nx * spec.ny * spec.nz)
        phi = np.ones(n_cells, dtype=np.float64)
        ax = np.ones(n_cells, dtype=np.float64)
        ay = np.ones(n_cells, dtype=np.float64)
        az = np.ones(n_cells, dtype=np.float64)
        out_diag = {
            "solid_cells": float(n_cells),
            "solid_fraction": float(diag.get("solid_fraction", 1.0)),
            "terrain_solid_cells": 0.0,
            "mesh_solid_cells": float(n_cells),
            "mesh_instances_used": 0.0,
            "mesh_seed_instances_requested": 0.0,
            "mesh_seed_instances_used": 0.0,
            "mesh_seed_leak_fallbacks": float(diag.get("mesh_seed_leak_fallbacks", 0.0)),
        }
        return phi, ax, ay, az, out_diag

    return _build


def _install_qgis_pyqt_stubs():
    if "qgis" in sys.modules:
        return

    qtcore_mod = ModuleType("qgis.PyQt.QtCore")
    qtcore_mod.Qt = SimpleNamespace()

    qtwidgets_mod = ModuleType("qgis.PyQt.QtWidgets")
    qtwidgets_mod.QDialog = type("QDialog", (), {})

    pyqt_mod = ModuleType("qgis.PyQt")
    pyqt_mod.QtCore = qtcore_mod
    pyqt_mod.QtWidgets = qtwidgets_mod

    qgis_mod = ModuleType("qgis")
    qgis_mod.PyQt = pyqt_mod

    sys.modules["qgis"] = qgis_mod
    sys.modules["qgis.PyQt"] = pyqt_mod
    sys.modules["qgis.PyQt.QtCore"] = qtcore_mod
    sys.modules["qgis.PyQt.QtWidgets"] = qtwidgets_mod


def _import_workbench_module():
    _install_qgis_pyqt_stubs()
    if "swe2d_workbench_qt" in sys.modules:
        return importlib.reload(sys.modules["swe2d_workbench_qt"])
    return importlib.import_module("swe2d_workbench_qt")


class TestSWE3DGeometryGuardrails(unittest.TestCase):
    def _run_upload(self, env, diag):
        wb = _WorkbenchStub()
        backend = _BackendStub()
        patch_stats = {"nx": 2, "ny": 2, "nz": 2, "n_cells": 8}

        with _temporary_env(env):
            upload_experimental_3d_obj_geometry(
                wb=wb,
                backend=backend,
                patch_stats=patch_stats,
                swe3d_env_overrides={},
                patch_grid_spec_cls=object,
                load_obj_mesh_fn=lambda *_args, **_kwargs: (None, None),
                apply_instance_transform_fn=lambda *_args, **_kwargs: None,
                build_static_geometry_tensors_fn=_tensor_builder_with_diag(diag),
                write_solid_voxels_obj_fn=None,
            )
        return wb, backend

    def test_non_strict_logs_violation_and_continues(self):
        wb, backend = self._run_upload(
            env={
                "BACKWATER_SWE3D_GEOM_STRICT": "0",
                "BACKWATER_SWE3D_GEOM_MAX_SOLID_FRACTION": "0.60",
                "BACKWATER_SWE3D_GEOM_MAX_SEED_LEAK_FALLBACKS": "0",
            },
            diag={"solid_fraction": 0.95, "mesh_seed_leak_fallbacks": 3},
        )

        self.assertEqual(backend.upload_calls, 1)
        self.assertTrue(
            any("3D geometry quality gate violation" in line for line in wb.logs),
            "Expected geometry quality violation log in non-strict mode.",
        )
        self.assertEqual(len(getattr(wb, "_swe3d_geom_gate_last_violations", [])), 2)
        self.assertEqual(
            int(getattr(wb, "_swe3d_geom_gate_last_metrics", {}).get("mesh_seed_leak_fallbacks", -1)),
            3,
        )

    def test_strict_raises_on_violation(self):
        patch_stats = {"nx": 2, "ny": 2, "nz": 2, "n_cells": 8}
        wb = _WorkbenchStub()
        backend = _BackendStub()

        with _temporary_env(
            {
                "BACKWATER_SWE3D_GEOM_STRICT": "1",
                "BACKWATER_SWE3D_GEOM_MAX_SOLID_FRACTION": "0.80",
                "BACKWATER_SWE3D_GEOM_MAX_SEED_LEAK_FALLBACKS": "0",
            }
        ):
            with self.assertRaises(RuntimeError):
                upload_experimental_3d_obj_geometry(
                    wb=wb,
                    backend=backend,
                    patch_stats=patch_stats,
                    swe3d_env_overrides={},
                    patch_grid_spec_cls=object,
                    load_obj_mesh_fn=lambda *_args, **_kwargs: (None, None),
                    apply_instance_transform_fn=lambda *_args, **_kwargs: None,
                    build_static_geometry_tensors_fn=_tensor_builder_with_diag(
                        {"solid_fraction": 0.92, "mesh_seed_leak_fallbacks": 2}
                    ),
                    write_solid_voxels_obj_fn=None,
                )

        self.assertTrue(bool(getattr(wb, "_swe3d_geom_gate_last_violations", [])))
        self.assertTrue(bool(getattr(wb, "_swe3d_geom_gate_last_config", {}).get("strict", False)))

    def test_strict_passes_within_thresholds(self):
        wb, backend = self._run_upload(
            env={
                "BACKWATER_SWE3D_GEOM_STRICT": "1",
                "BACKWATER_SWE3D_GEOM_MAX_SOLID_FRACTION": "0.99",
                "BACKWATER_SWE3D_GEOM_MAX_SEED_LEAK_FALLBACKS": "2",
            },
            diag={"solid_fraction": 0.50, "mesh_seed_leak_fallbacks": 1},
        )
        self.assertEqual(backend.upload_calls, 1)
        self.assertFalse(any("quality gate violation" in line for line in wb.logs))
        self.assertEqual(getattr(wb, "_swe3d_geom_gate_last_violations", []), [])
        self.assertAlmostEqual(
            float(getattr(wb, "_swe3d_geom_gate_last_config", {}).get("max_solid_fraction", -1.0)),
            0.99,
        )


class TestSWE3DRuntimeConfigGuardrails(unittest.TestCase):
    def test_configure_runtime_sets_geometry_gate_env(self):
        keys = {
            "BACKWATER_SWE3D_GEOM_STRICT": None,
            "BACKWATER_SWE3D_GEOM_MAX_SOLID_FRACTION": None,
            "BACKWATER_SWE3D_GEOM_MAX_SEED_LEAK_FALLBACKS": None,
        }
        with _temporary_env(keys):
            applied = configure_swe3d_runtime(
                geometry_gate_strict=True,
                geometry_gate_max_solid_fraction=0.975,
                geometry_gate_max_seed_leak_fallbacks=3,
            )
            self.assertTrue(applied["geometry_gate_strict"])
            self.assertAlmostEqual(applied["geometry_gate_max_solid_fraction"], 0.975)
            self.assertEqual(applied["geometry_gate_max_seed_leak_fallbacks"], 3)
            self.assertEqual(os.environ.get("BACKWATER_SWE3D_GEOM_STRICT"), "1")
            self.assertEqual(os.environ.get("BACKWATER_SWE3D_GEOM_MAX_SEED_LEAK_FALLBACKS"), "3")

    def test_configure_runtime_rejects_invalid_solid_fraction(self):
        with self.assertRaises(ValueError):
            configure_swe3d_runtime(geometry_gate_max_solid_fraction=1.5)


class TestSWE3DWorkbenchDelegateIntegration(unittest.TestCase):
    def test_workbench_delegate_enforces_strict_geometry_gate(self):
        workbench_module = _import_workbench_module()
        wb = _WorkbenchStub()
        backend = _BackendStub()
        patch_stats = {"nx": 2, "ny": 2, "nz": 2, "n_cells": 8}

        patches = {
            "PatchGridSpec": object,
            "load_obj_mesh": lambda *_args, **_kwargs: (None, None),
            "apply_instance_transform": lambda *_args, **_kwargs: None,
            "build_static_geometry_tensors": _tensor_builder_with_diag(
                {"solid_fraction": 0.95, "mesh_seed_leak_fallbacks": 1}
            ),
            "write_solid_voxels_obj": None,
        }
        previous = {}
        for key, value in patches.items():
            previous[key] = getattr(workbench_module, key, None)
            setattr(workbench_module, key, value)

        try:
            with _temporary_env(
                {
                    "BACKWATER_SWE3D_GEOM_STRICT": None,
                    "BACKWATER_SWE3D_GEOM_MAX_SOLID_FRACTION": None,
                    "BACKWATER_SWE3D_GEOM_MAX_SEED_LEAK_FALLBACKS": None,
                }
            ):
                configure_swe3d_runtime(
                    geometry_gate_strict=True,
                    geometry_gate_max_solid_fraction=0.80,
                    geometry_gate_max_seed_leak_fallbacks=0,
                )
                with self.assertRaises(RuntimeError):
                    workbench_module.SWE2DWorkbenchDialog._upload_experimental_3d_obj_geometry(
                        wb,
                        backend=backend,
                        patch_stats=patch_stats,
                        swe3d_env_overrides={},
                    )
        finally:
            for key, value in previous.items():
                setattr(workbench_module, key, value)

        self.assertEqual(backend.upload_calls, 1)
        self.assertTrue(any("3D geometry quality gate violation" in line for line in wb.logs))


if __name__ == "__main__":
    unittest.main()
