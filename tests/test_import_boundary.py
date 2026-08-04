"""Clean-subprocess import-boundary tests (MVP Rule 3).

Every service-layer module must be importable with ``qgis`` and ``PyQt5``
completely blocked.  The previous design installed fake qgis modules and
inspected ``sys.modules`` in-process — a mock that defeats the purpose,
since it asserts the service layer is Qt-free while polluting ``sys.modules``
with fakes.  This redesign spawns a clean ``sys.executable -c``
subprocess per module with a meta-path blocker that makes any ``qgis`` /
``PyQt5`` import raise ``ImportError``, then asserts the import succeeded and
no Qt module slipped into ``sys.modules`` anyway.

Service roots (``.opencode/rules/MVP_ARCHITECTURE.md`` Rule 3 + layer diagram):

- ``swe2d/runtime/``
- ``swe2d/boundary_and_forcing/``
- ``swe2d/mesh/``
- ``swe2d/results/``
- ``swe2d/workbench/services/``  (the ``*service*.py`` layer)

Packages are walked with ``pkgutil.iter_modules`` so new service modules are
checked automatically.  The CLI Qt-free surface (``swe2d.cli``, ``swe2d.core``)
is covered by the same mechanism — see ``docs/COMPREHENSIVE_REVIEW.md`` C-5.

The test itself needs no qgis, no mocks, and no harness imports — it only
spawns subprocesses.  Run with the project's canonical runner:

    python3 -m unittest -v tests.test_import_boundary
"""

from __future__ import annotations

import os
import pkgutil
import subprocess
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

_SUBPROCESS_TIMEOUT_S = 30
_MAX_WORKERS = 8

# Packages walked for the MVP-3 hard rule (Qt-free service layer).
_SERVICE_PACKAGE_ROOTS = (
    "swe2d.runtime",
    "swe2d.boundary_and_forcing",
    "swe2d.mesh",
    "swe2d.results",
    "swe2d.workbench.services",
)

# CLI / public-API modules that must also stay Qt-free so the headless CLI
# works without QGIS (docs/CLI_FIRST_REFACTOR_PLAN.md §2.2).  Not walked —
# the CLI surface is intentionally small and explicit.
_CLI_QT_FREE_MODULES = (
    "swe2d.cli",
    "swe2d.cli.gpkg_adapter",
    "swe2d.cli.headless_runner",
    "swe2d.core",
    "swe2d.core.builder",
    "swe2d.workbench.workers",  # Qt symbols are lazy via PEP 562 __getattr__
)

# Tier 2 — known Qt-bound modules sitting inside service roots.
#
# These are PRE-EXISTING MVP Rule 3 violations, documented here so the test
# suite stays green while making the violation set explicit and shrink-only:
# test_known_qt_bound_modules_still_fail asserts each of these DOES fail the
# blocked import.  Fixing a module (making it Qt-free) fails that test with
# instructions to shrink this list; adding a new Qt import to any other
# service module fails the tier-1 test.  The set cannot grow silently and
# cannot go stale.
_KNOWN_QT_BOUND = frozenset({
    # Top-level `from qgis.PyQt import QtCore` — a QObject playback
    # controller; belongs in workbench/views, not the results service layer.
    "swe2d.results.animation",
    # Top-level Qt import inside an `if True:` block (line ~999); a Qt
    # canvas-overlay renderer; belongs in workbench/views.
    "swe2d.results.high_perf_viewer",
    # Top-level `from qgis.PyQt.QtCore import QObject, pyqtSignal` — a
    # QObject batch runner; needs a Qt-free core + thin Qt wrapper split.
    "swe2d.workbench.services.batch_manager",
})

# Subprocess preamble: block qgis/PyQt5 at the meta-path level, import the
# target module, then verify nothing Qt-flavoured landed in sys.modules.
_CHECK_SCRIPT = """\
import importlib
import sys


class _QtBlocker:
    def find_spec(self, name, path=None, target=None):
        if name == "qgis" or name.startswith("qgis.") \\
                or name == "PyQt5" or name.startswith("PyQt5."):
            raise ImportError(f"MVP-3 blocked Qt import: {name}")
        return None


sys.meta_path.insert(0, _QtBlocker())
importlib.import_module(sys.argv[1])
qt = sorted(
    m for m in sys.modules
    if m == "qgis" or m.startswith("qgis.") or m.startswith("PyQt5")
)
if qt:
    print("QT MODULES LOADED:", qt, file=sys.stderr)
    sys.exit(1)
"""


def _walk_service_modules() -> list[str]:
    """Fully-qualified names of every module under the service roots."""
    names: list[str] = []
    for root in _SERVICE_PACKAGE_ROOTS:
        package = __import__(root, fromlist=["__path__"])
        names.extend(
            m.name for m in pkgutil.iter_modules(package.__path__, root + ".")
        )
    return sorted(names)


def _subprocess_env() -> dict:
    """Environment for the clean subprocess.

    Prepends the repo root and build/ to the inherited PYTHONPATH.  The
    inherited value MUST be preserved: the mamba env's activate.d scripts put
    QGIS's own ``share/qgis/python`` there, and clobbering it changes which
    modules are importable.  qgis/PyQt5 remain blocked by the meta-path
    hook regardless — being importable-but-blocked is exactly the tier-1
    scenario (a stray ``import qgis`` must fail loudly, not silently pass
    because qgis happened to be absent).
    """
    env = dict(os.environ)
    entries = [str(_REPO_ROOT), str(_REPO_ROOT / "build")]
    inherited = env.get("PYTHONPATH", "")
    if inherited:
        entries.append(inherited)
    env["PYTHONPATH"] = os.pathsep.join(entries)
    return env


def _check_module(module: str, env: dict) -> tuple[str, int, str]:
    """Run the blocked-import check for one module; return (name, rc, tail)."""
    proc = subprocess.run(
        [sys.executable, "-c", _CHECK_SCRIPT, module],
        capture_output=True,
        text=True,
        timeout=_SUBPROCESS_TIMEOUT_S,
        cwd=_REPO_ROOT,
        env=env,
    )
    output = (proc.stderr or proc.stdout).strip().splitlines()
    tail = output[-1] if output else ""
    return module, proc.returncode, tail


def _run_checks(modules: list[str]) -> dict[str, tuple[int, str]]:
    env = _subprocess_env()
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        results = list(pool.map(lambda m: _check_module(m, env), modules))
    return {name: (rc, tail) for name, rc, tail in results}


class TestImportBoundary(unittest.TestCase):
    """MVP Rule 3 enforcement via clean, qgis-blocked subprocesses."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._service_modules = _walk_service_modules()
        cls._tier1_modules = [
            m for m in cls._service_modules if m not in _KNOWN_QT_BOUND
        ]
        cls._results = _run_checks(cls._tier1_modules + list(_CLI_QT_FREE_MODULES))
        cls._known_results = _run_checks(sorted(_KNOWN_QT_BOUND))

    def test_service_layer_is_qt_free(self):
        """Every service-layer module imports with qgis/PyQt5 blocked."""
        for module in self._tier1_modules:
            rc, tail = self._results[module]
            with self.subTest(module=module):
                self.assertEqual(
                    rc,
                    0,
                    f"{module} must import cleanly with qgis/PyQt5 blocked "
                    f"(MVP Rule 3). Subprocess said: {tail}",
                )

    def test_cli_surface_is_qt_free(self):
        """The headless CLI / public API must not need Qt at import time."""
        for module in _CLI_QT_FREE_MODULES:
            rc, tail = self._results[module]
            with self.subTest(module=module):
                self.assertEqual(
                    rc,
                    0,
                    f"{module} must import cleanly with qgis/PyQt5 blocked "
                    f"(headless CLI contract). Subprocess said: {tail}",
                )

    def test_known_qt_bound_modules_still_fail(self):
        """Guardrail: the documented violation set is exact and shrink-only.

        Each module in ``_KNOWN_QT_BOUND`` must still FAIL the blocked
        import.  If one now passes, the module has been de-Qtified — remove
        it from ``_KNOWN_QT_BOUND`` so it joins tier 1.  If a NEW service
        module fails tier 1 instead, do NOT add it here — fix the import.
        """
        self.assertTrue(
            set(self._known_results) == _KNOWN_QT_BOUND,
            "guardrail must check exactly the _KNOWN_QT_BOUND set",
        )
        for module, (rc, tail) in self._known_results.items():
            with self.subTest(module=module):
                self.assertNotEqual(
                    rc,
                    0,
                    f"{module} is documented as Qt-bound but now imports "
                    f"cleanly with qgis blocked — remove it from "
                    f"_KNOWN_QT_BOUND so the tier-1 test covers it.",
                )
                self.assertIn(
                    "MVP-3 blocked Qt import",
                    tail,
                    f"{module} failed for an unexpected reason (not the Qt "
                    f"blocker): {tail}",
                )


if __name__ == "__main__":
    unittest.main()
