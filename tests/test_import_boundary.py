"""Runtime import-boundary tests.

These tests verify that the canonical CLI / runtime import path does NOT
load Qt (``qgis.PyQt``, ``qgis.gui``, or ``PyQt5``) at import time.  The CLI
is supposed to be Qt-free in the headless path — see
``docs/COMPREHENSIVE_REVIEW.md`` C-5 and ``docs/CLI_FIRST_REFACTOR_PLAN.md`` §2.2.

The existing ``tests/test_mvp_imports.py`` enforces the boundary at AST
parse time (no static ``from swe2d.workbench…`` lines in CLI source).
This file adds the complementary *runtime* check: even if a lazy import
slips in, we want to know whether it triggers Qt at module load.

The check is a *delta* on the set of Qt modules in ``sys.modules``: we
snapshot Qt modules right before the import under test, then assert that
no new Qt modules were added by that import.  This is robust against
test pollution — other tests in the same session (e.g. the builder tests
in ``test_run_context_builder.py``) install Qt mocks and therefore
populate ``sys.modules`` with Qt entries before this module runs.  The
absolute "no Qt in sys.modules" check would (and did) produce false
positives in that case.

Implemented as a ``unittest.TestCase`` (not pytest functions) because the
project root ``__init__.py`` is the QGIS plugin entry point and pytest's
package-discovery mode tries to import it during collection, which fails
in headless environments.  ``python -m unittest`` is the project's
canonical test runner — see ``.github/workflows/test.yml``.
"""

from __future__ import annotations

import importlib
import sys
import unittest


_QT_MODULE_PREFIXES = ("qgis.PyQt", "qgis.gui", "PyQt5")


def _qt_modules_loaded() -> list[str]:
    return [m for m in sys.modules if any(m == p or m.startswith(p + ".") for p in _QT_MODULE_PREFIXES)]


def _qt_modules_loaded_before() -> frozenset:
    """Snapshot the Qt modules currently in ``sys.modules``.

    Returned as a ``frozenset`` so it can be used as the ``before`` set
    in a delta check: ``after - before`` is the set of Qt modules the
    import under test newly introduced.
    """
    return frozenset(_qt_modules_loaded())


def _reset_swe2d_modules() -> None:
    """Drop every cached ``swe2d.*`` module so the import is observed fresh."""
    for mod in list(sys.modules):
        if mod == "swe2d" or mod.startswith("swe2d."):
            del sys.modules[mod]


class TestImportBoundary(unittest.TestCase):
    """Runtime Qt-import boundary checks for the CLI / runtime paths.

    Each test uses a *delta* check: it snapshots the Qt modules loaded
    before the import under test, performs the import, then asserts that
    no new Qt modules were added.  This is unaffected by Qt modules that
    other tests in the same session have already pulled into
    ``sys.modules`` (e.g. via ``tests.mocks.qgis_env``).
    """

    def setUp(self) -> None:
        _reset_swe2d_modules()

    def tearDown(self) -> None:
        _reset_swe2d_modules()

    def _assert_no_new_qt(self, before: frozenset, label: str) -> None:
        """Helper: fail if any new Qt module appeared during the import."""
        new_qt = set(_qt_modules_loaded()) - before
        self.assertFalse(
            new_qt,
            f"{label} must not import Qt at module load time, added: {sorted(new_qt)}",
        )

    def test_cli_does_not_import_qgis_gui(self):
        """``import swe2d.cli`` must not load Qt modules.

        The CLI's ``__init__.py`` is intentionally empty.  All CLI entry
        points (headless_runner, gpkg_adapter) must avoid eager Qt imports
        so ``python -m swe2d.cli …`` works without QGIS.
        """
        before = _qt_modules_loaded_before()
        importlib.import_module("swe2d.cli")
        self._assert_no_new_qt(before, "swe2d.cli")

    def test_cli_gpkg_adapter_does_not_import_qgis_gui(self):
        """``swe2d.cli.gpkg_adapter`` is a thin re-export from core and must not load PyQt GUI."""
        before = _qt_modules_loaded_before()
        importlib.import_module("swe2d.cli.gpkg_adapter")
        self._assert_no_new_qt(before, "swe2d.cli.gpkg_adapter")

    def test_cli_headless_runner_does_not_import_qgis_gui(self):
        """``swe2d.cli.headless_runner`` is the CLI's main entry; Qt must stay out."""
        before = _qt_modules_loaded_before()
        importlib.import_module("swe2d.cli.headless_runner")
        self._assert_no_new_qt(before, "swe2d.cli.headless_runner")

    def test_core_builder_does_not_import_qgis_gui(self):
        """``swe2d.core.builder`` is the canonical RunContext builder.

        It must remain free of ``qgis.PyQt`` / ``qgis.gui`` / ``PyQt5`` so
        the CLI can build a ``RunContext`` from a JSON spec without QGIS.
        It may import ``qgis.core`` (allowed) via the GPKG I/O helpers.
        """
        before = _qt_modules_loaded_before()
        importlib.import_module("swe2d.core.builder")
        self._assert_no_new_qt(before, "swe2d.core.builder")

    def test_core_package_does_not_import_qgis_gui(self):
        """``import swe2d.core`` must not pull in any Qt GUI modules.

        ``swe2d.core`` is the public, GUI-free API surface.  Eager imports of
        ``RunContext``, ``build_run_context``, ``execute_run``, etc. must not
        trigger ``qgis.PyQt`` / ``qgis.gui`` / ``PyQt5``.
        """
        before = _qt_modules_loaded_before()
        importlib.import_module("swe2d.core")
        self._assert_no_new_qt(before, "swe2d.core")

    def test_workbench_workers_package_does_not_eagerly_import_qt(self):
        """``swe2d.workbench.workers`` exposes both Qt and Qt-free symbols.

        ``RunContext`` is a plain dataclass and is re-exported eagerly from
        ``swe2d.core.run_context``.  ``SimulationWorker`` / ``PersistenceWorker`` /
        ``ComputeResult`` / ``SnapshotData`` all import Qt and must be loaded
        only on attribute access (PEP 562 ``__getattr__``).
        """
        before = _qt_modules_loaded_before()
        importlib.import_module("swe2d.workbench.workers")

        # RunContext is a Qt-free dataclass — it must be available immediately.
        run_ctx_cls = getattr(sys.modules["swe2d.workbench.workers"], "RunContext", None)
        self.assertIsNotNone(run_ctx_cls, "RunContext should be available eagerly")

        # Qt must not have been pulled in by the package import.
        self._assert_no_new_qt(before, "swe2d.workbench.workers package import")


if __name__ == "__main__":
    unittest.main()

