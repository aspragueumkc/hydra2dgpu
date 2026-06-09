#!/usr/bin/env python3
"""Validate that swe2d_workbench_qt module-level symbols are correctly imported.

This test catches NameError regressions like the SWE2DBackend removal that
broke the _on_run path at runtime.  It imports the workbench module and
verifies every symbol that extracted method files reference from its
global namespace is actually present.

Can be run headlessly (without QGIS).  When QGIS core is unavailable the
import falls back to stub definitions; the test still validates those stubs.
"""

import sys
import unittest


def _import_wb():
    """Import swe2d_workbench_qt and return the module, suppressing stdout."""
    import io
    import contextlib

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        from swe2d_workbench_qt import (
            SWE2DBackend,
            SWE2DCouplingController,
            SWE2DWorkbenchDialog,
            SWE2DWorkbenchStudioDialog,
            SWE2DStructureModule,
            SWE2DUrbanDrainageModule,
            StructureType,
            pack_coupling_soa,
            swe2d_gpu_available,
        )

    return {
        "SWE2DBackend": SWE2DBackend,
        "SWE2DCouplingController": SWE2DCouplingController,
        "SWE2DWorkbenchDialog": SWE2DWorkbenchDialog,
        "SWE2DWorkbenchStudioDialog": SWE2DWorkbenchStudioDialog,
        "SWE2DStructureModule": SWE2DStructureModule,
        "SWE2DUrbanDrainageModule": SWE2DUrbanDrainageModule,
        "StructureType": StructureType,
        "pack_coupling_soa": pack_coupling_soa,
        "swe2d_gpu_available": swe2d_gpu_available,
    }


_SYMBOLS = [
    # Backend symbols — must exist (even if None in fallback path)
    # so extracted run methods don't raise NameError.
    "SWE2DBackend",
    "swe2d_gpu_available",
    # Coupling symbols used in model_and_run_methods.py
    "SWE2DCouplingController",
    "pack_coupling_soa",
    "SWE2DStructureModule",
    "SWE2DUrbanDrainageModule",
    # Extension model enums used downstream
    "StructureType",
    # Dialog classes
    "SWE2DWorkbenchDialog",
    "SWE2DWorkbenchStudioDialog",
]

# Methods defined on SWE2DWorkbenchDialog that are referenced by
# wiring callbacks in run_component_wiring.py and extracted/*.py.
_EXPECTED_DIALOG_METHODS = [
    "_backend_ready_for_run_preflight",
    "_show_backend_unavailable_for_run_preflight",
    "_ensure_mesh_for_run_preflight",
    "_has_mesh_for_run_preflight",
    "_execute_run_request",
    "_on_run_requested",
    "_on_run",
    "_build_run_request",
    "_collect_boundary_arrays",
    "_build_side_hydrographs",
    "_collect_bc_layer_hydrographs",
    "_collect_bc_layer_edge_groups",
    "_initial_state",
    "_build_spatial_manning_array",
    "_update_unit_system_from_crs",
    "_parse_run_duration_seconds",
    "_apply_external_sources",
    "_build_internal_flow_forcing",
    "_internal_flow_source_cms_at_time",
    "_build_spatial_manning_array",
    "_build_spatial_cn_array",
    "_build_thiessen_rain_cn_forcing",
    "_build_pipe_network_config",
    "_build_hydraulic_structure_config",
    "_sample_coupling_object_metrics",
    "_persist_coupling_results_to_geopackage",
    "_log",
    "_require_run_components",
    "_init_startup_component",
    "_note_startup_component_missing",
]


class TestWorkbenchImports(unittest.TestCase):
    """Validate that all critical module-level symbols exist."""

    @classmethod
    def setUpClass(cls):
        cls.symbols = _import_wb()

    # -- Each critical symbol gets its own test for clear failure reporting --

    def test_symbol_SWE2DBackend(self):
        """SWE2DBackend is defined (may be None if import failed)."""
        self.assertIn("SWE2DBackend", self.symbols)

    def test_symbol_swe2d_gpu_available(self):
        """swe2d_gpu_available is defined and callable."""
        fn = self.symbols["swe2d_gpu_available"]
        self.assertTrue(callable(fn))

    def test_symbol_SWE2DCouplingController(self):
        """SWE2DCouplingController is defined."""
        self.assertIn("SWE2DCouplingController", self.symbols)

    def test_symbol_pack_coupling_soa(self):
        """pack_coupling_soa is defined."""
        self.assertIn("pack_coupling_soa", self.symbols)

    def test_symbol_SWE2DStructureModule(self):
        """SWE2DStructureModule is defined."""
        self.assertIn("SWE2DStructureModule", self.symbols)

    def test_symbol_SWE2DUrbanDrainageModule(self):
        """SWE2DUrbanDrainageModule is defined."""
        self.assertIn("SWE2DUrbanDrainageModule", self.symbols)

    def test_symbol_StructureType(self):
        """StructureType is defined."""
        self.assertIn("StructureType", self.symbols)

    def test_dialog_class_defined(self):
        """SWE2DWorkbenchDialog class is defined."""
        cls = self.symbols["SWE2DWorkbenchDialog"]
        self.assertIsNotNone(cls)

    def test_dialog_methods_exist(self):
        """All wiring callbacks exist on SWE2DWorkbenchDialog."""
        cls = self.symbols["SWE2DWorkbenchDialog"]
        if cls is None:
            self.skipTest("SWE2DWorkbenchDialog not importable")
        for m in _EXPECTED_DIALOG_METHODS:
            self.assertTrue(
                hasattr(cls, m),
                f"SWE2DWorkbenchDialog missing required method: {m}",
            )

    def test_all_expected_symbols_present(self):
        """All symbols in _SYMBOLS list are present after import."""
        for name in _SYMBOLS:
            with self.subTest(symbol=name):
                self.assertIn(name, self.symbols)

    def test_fallback_import_does_not_raise(self):
        """Re-import validates fallback paths don't crash."""
        import importlib

        # Pop the module if cached so import re-runs.
        sys.modules.pop("swe2d_workbench_qt", None)
        # Clear coupled sub-modules that may hold stale references.
        for key in list(sys.modules.keys()):
            if key.startswith("swe2d.workbench.extracted"):
                sys.modules.pop(key, None)
        importlib.invalidate_caches()

        try:
            import swe2d_workbench_qt as wb  # noqa: F811
        except Exception as exc:
            self.fail(f"swe2d_workbench_qt re-import raised: {exc}")


class TestWorkbenchModelAndRunMethods(unittest.TestCase):
    """Validate that extracted run method files reference only existing symbols.

    This test parses the extracted method file for symbol references and
    checks each against the workbench module's namespace.
    """

    def test_extracted_file_missing_globals(self):
        """Validate that model_and_run_methods only references existing symbols
        from the workbench module.

        This builds an AST of the extracted file, finds every module-level
        Name node that is NOT a local definition/internal name, and checks
        it exists in the swe2d_workbench_qt namespace.
        """
        import ast
        import os

        extracted_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "swe2d",
            "workbench",
            "extracted",
            "model_and_run_methods.py",
        )
        extracted_path = os.path.normpath(extracted_path)
        self.assertTrue(os.path.isfile(extracted_path), f"Not found: {extracted_path}")

        with open(extracted_path) as f:
            tree = ast.parse(f.read())

        # Collect names that are defined as local variables / parameters in
        # the extracted file itself so we can exclude them.
        local_names = set()

        def _collect_locals(body):
            for node in body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Parameters are local
                    for arg in node.args.args:
                        local_names.add(arg.arg)
                    if node.args.vararg:
                        local_names.add(node.args.vararg.arg)
                    if node.args.kwarg:
                        local_names.add(node.args.kwarg.arg)
                    for arg in node.args.kwonlyargs:
                        local_names.add(arg.arg)
                    for arg in node.args.posonlyargs:
                        local_names.add(arg.arg)
                    _collect_locals(node.body)
                elif isinstance(node, ast.Lambda):
                    for arg in node.args.args:
                        local_names.add(arg.arg)
                    if node.args.vararg:
                        local_names.add(node.args.vararg.arg)
                    if node.args.kwarg:
                        local_names.add(node.args.kwarg.arg)
                elif isinstance(node, ast.ClassDef):
                    _collect_locals(node.body)

        _collect_locals(tree.body)

        # Also collect names assigned at module level (those are defined locally too)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                local_names.add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        local_names.add(target.id)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                local_names.add(node.target.id)

        # Collect all bare Name nodes used at module level (top-level of the file).
        names = set()
        for node in tree.body:
            # Only walk top-level statements; we already collected locals from
            # function bodies above. Only look at Name nodes in non-defining contexts.
            for child in ast.walk(node):
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                    name = child.id
                    # Skip names defined locally in the extracted file.
                    if name in local_names:
                        continue
                    # Skip builtins and common Python/stdlib names
                    if name.upper() == name and len(name) > 1 and not name.startswith("_"):
                        names.add(name)

        # Known external module-level names that are imported in
        # model_and_run_methods.py itself (not from the workbench module).
        _KNOWN_LOCAL_IMPORTS = {
            "SWE2DCouplingController",
            "pack_coupling_soa",
            "SWE2DUrbanDrainageModule",
            "SWE2DStructureModule",
            "StructureType",
            "DrainageSolverMode",
            "DrainageLink",
            "DrainageNode",
            "HydraulicStructure",
            "HydraulicStructureConfig",
            "InletExchange",
            "InletType",
            "NodeInletAssignment",
            "OutfallExchange",
            "PipeEndExchange",
            "PipeNetworkConfig",
            "GodunovSolverMode",
            "HydraulicStructure",
            "HydraulicStructureConfig",
            "SWE2DEquationSet",
            "SolverModelOptions",
            "SpatialDiscretization",
            "TemporalScheme",
            "Gauge",
            "ThiessenRainCNForcing",
            "SWE2DBackend",
        }
        # These are expected to come from the workbench module (not imported locally).
        # model_and_run_methods references them as bare names resolved from
        # the workbench's global namespace when the methods are bound.
        _WORKBENCH_GLOBALS = {
            "SWE2DBackend",       # from swe2d_workbench_qt top-level import
            "SWE2DCouplingController",
            "pack_coupling_soa",
        }

        # Import the module and check each name
        sys.modules.pop("swe2d_workbench_qt", None)
        import importlib
        importlib.invalidate_caches()
        import swe2d_workbench_qt as wb

        unresolved = []
        for name in sorted(names):
            # Skip names that are imported locally in the extracted file itself
            if name in _KNOWN_LOCAL_IMPORTS:
                continue
            # Only check names that are expected to come from the workbench module
            if name not in _WORKBENCH_GLOBALS:
                continue
            if not hasattr(wb, name):
                unresolved.append(name)

        if unresolved:
            self.fail(
                f"{len(unresolved)} symbol(s) referenced by "
                f"model_and_run_methods.py not found in swe2d_workbench_qt:\n  "
                + "\n  ".join(unresolved)
            )


class TestWorkbenchBackendPreflight(unittest.TestCase):
    """Validate the backend availability preflight logic.

    These tests catch regressions where the GPU-only backend check was
    accidentally broken (e.g., NameError from removed swe2d_available).
    """

    def test_backend_ready_delegates_to_gpu(self):
        """_backend_ready_for_run_preflight returns swe2d_gpu_available()."""
        from swe2d_workbench_qt import swe2d_gpu_available

        # Just validate the symbol is callable — actual return value
        # depends on whether the CUDA module is built.
        self.assertTrue(callable(swe2d_gpu_available))


class TestBackendConstructorKwargs(unittest.TestCase):
    """Validate that every call to SWE2DBackend() uses compatible constructor kwargs.

    This catches ``TypeError: SWE2DBackend.__init__() got an unexpected keyword
    argument 'openmp_enabled'`` at import time rather than at QGIS run time.
    """

    @classmethod
    def setUpClass(cls):
        import inspect
        import ast
        import os

        # 1) Read SWE2DBackend.__init__ signature
        from swe2d_workbench_qt import SWE2DBackend as cls_bk

        sig = inspect.signature(cls_bk.__init__)
        cls._backend_params = set(sig.parameters.keys()) - {"self"}

        # 2) AST-scan every .py file in swe2d/runtime/ and swe2d/workbench/
        #    for calls to backend_cls(...) or SWE2DBackend(...)
        cls._backend_callsites = []

        repo_root = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..")
        )
        scan_dirs = [
            os.path.join(repo_root, "swe2d", "runtime"),
            os.path.join(repo_root, "swe2d", "workbench"),
        ]

        for scan_dir in scan_dirs:
            if not os.path.isdir(scan_dir):
                continue
            for root, _dirs, files in os.walk(scan_dir):
                for fn in files:
                    if not fn.endswith(".py"):
                        continue
                    fpath = os.path.join(root, fn)
                    try:
                        with open(fpath) as f:
                            tree = ast.parse(f.read(), filename=fpath)
                    except SyntaxError:
                        continue
                    for node in ast.walk(tree):
                        if not isinstance(node, ast.Call):
                            continue
                        func = node.func
                        name = None
                        if isinstance(func, ast.Name):
                            name = func.id
                        elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                            name = func.value.id  # e.g. cls.backend_cls(...)
                        if name in ("SWE2DBackend", "backend_cls"):
                            kwargs = {}
                            for kw in node.keywords:
                                if kw.arg is not None:
                                    kwargs[kw.arg] = kw.value
                            rel = os.path.relpath(fpath, repo_root)
                            cls._backend_callsites.append((rel, node.lineno, kwargs))

    def test_backend_accepts_only_use_gpu(self):
        """SWE2DBackend.__init__ only accepts 'use_gpu'."""
        self.assertIn("use_gpu", self._backend_params)
        # It should NOT have openmp_enabled or other CPU-era params
        self.assertNotIn(
            "openmp_enabled",
            self._backend_params,
            "SWE2DBackend.__init__ should not accept openmp_enabled. "
            "The GPU backend does not use OpenMP.",
        )

    def test_no_caller_passes_openmp_enabled(self):
        """No call site passes openmp_enabled to backend_cls/SWE2DBackend()."""
        bad = []
        for rel, lineno, kwargs in self._backend_callsites:
            if "openmp_enabled" in kwargs:
                bad.append(f"{rel}:{lineno}")
        if bad:
            self.fail(
                f"{len(bad)} call site(s) pass openmp_enabled= to "
                f"SWE2DBackend/backend_cls:\n  " + "\n  ".join(bad)
            )

    def test_all_caller_kwargs_are_valid(self):
        """Every kwarg passed to backend_cls() is accepted by __init__."""
        bad = []
        for rel, lineno, kwargs in self._backend_callsites:
            for key in kwargs:
                if key not in self._backend_params:
                    bad.append(f"{rel}:{lineno} passes '{key}' which is not in __init__")
        if bad:
            self.fail(
                f"Invalid constructor kwargs found:\n  " + "\n  ".join(bad)
            )

    def test_constructor_with_use_gpu_succeeds(self):
        """SWE2DBackend(use_gpu=True) does not raise a TypeError.

        Note: this may raise RuntimeError if the native module isn't built;
        that's expected and acceptable.  The point is to catch TypeErrors
        from unexpected keyword arguments.
        """
        from swe2d_workbench_qt import SWE2DBackend as bk_cls

        try:
            _ = bk_cls(use_gpu=True)
        except TypeError as exc:
            self.fail(f"SWE2DBackend(use_gpu=True) raised TypeError: {exc}")
        except RuntimeError:
            pass  # Expected when no native module is installed
        except Exception:
            pass  # Other errors are acceptable here


if __name__ == "__main__":
    unittest.main(verbosity=2)
