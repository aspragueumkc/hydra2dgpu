import unittest
"""Architectural boundary enforcement (AGENTS.md / PLANNING.md).

CLI must not import from swe2d.workbench. Pure-Python services must not
import Qt (any binding). GUI services must not import QtWidgets (any binding).
"""
import ast
import pathlib

import pytest


_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_CLI_DIR = _REPO_ROOT / "swe2d" / "cli"
_SHARED_SERVICES_DIRS = [
    _REPO_ROOT / "swe2d" / "services",
    _REPO_ROOT / "swe2d" / "runtime",
    _REPO_ROOT / "swe2d" / "results",
    _REPO_ROOT / "swe2d" / "mesh",
    _REPO_ROOT / "swe2d" / "boundary_and_forcing",
    _REPO_ROOT / "swe2d" / "extensions",
]
_GUI_SERVICES_DIR = _REPO_ROOT / "swe2d" / "workbench" / "services"


def _python_files(root: pathlib.Path):
    return sorted(p for p in root.rglob("*.py") if p.name != "__init__.py")


def _imports(source: str) -> list[str]:
    tree = ast.parse(source)
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            out.append(module)
    return out


@pytest.mark.parametrize("py_file", _python_files(_CLI_DIR), ids=lambda p: p.name)
def test_cli_does_not_import_workbench(py_file):
    offending = [m for m in _imports(py_file.read_text()) if m.startswith("swe2d.workbench")]
    assert not offending, f"{py_file.relative_to(_REPO_ROOT)} imports workbench: {offending}"


@pytest.mark.parametrize(
    "py_file",
    [p for d in _SHARED_SERVICES_DIRS for p in _python_files(d)],
    ids=lambda p: p.relative_to(_REPO_ROOT).as_posix(),
)
def test_shared_service_layer_does_not_import_pyqt5_widgets(py_file):
    imports = _imports(py_file.read_text())
    bad = [m for m in imports if _is_qtwidgets_import(m)]
    assert not bad, f"{py_file.relative_to(_REPO_ROOT)} imports QtWidgets: {bad}"


@pytest.mark.parametrize("py_file", _python_files(_GUI_SERVICES_DIR), ids=lambda p: p.name)
def test_gui_services_do_not_import_qtwidgets(py_file):
    imports = _imports(py_file.read_text())
    bad = [m for m in imports if _is_qtwidgets_import(m)]
    assert not bad, f"{py_file.relative_to(_REPO_ROOT)} imports QtWidgets: {bad}"


def _is_qtwidgets_import(m: str) -> bool:
    """Return True when an import module name resolves to QtWidgets under any binding.

    Phase 4.1 migrated runtime call sites to ``qgis.PyQt.*`` imports; this
    test must reject both the legacy ``PyQt5.QtWidgets`` form and the new
    ``qgis.PyQt.QtWidgets`` form for the service-layer QtWidgets ban.
    """
    if m == "PyQt5.QtWidgets" or m.startswith("PyQt5.QtWidgets."):
        return True
    if m == "qgis.PyQt.QtWidgets" or m.startswith("qgis.PyQt.QtWidgets."):
        return True
    return False

class _PytestStyleWrapper(unittest.TestCase):
    """Auto-generated wrapper for module-level test functions.

    Created by tools/wrap_pytest_style.py so that pytest-style tests
    (def test_* at module level) become visible to `python3 -m unittest`.
    Each module-level test is attached as a staticmethod so it can be
    discovered and run as a unittest TestCase.
    """
__wrapped_funcs = []
for _name, _obj in list(globals().items()):
    if _name.startswith("test_") and callable(_obj) and not isinstance(_obj, type):
        setattr(_PytestStyleWrapper, _name, staticmethod(_obj))
        __wrapped_funcs.append(_name)
for _name in __wrapped_funcs:
    del globals()[_name]
del _name, _obj, __wrapped_funcs
