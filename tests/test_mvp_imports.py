"""Architectural boundary enforcement (AGENTS.md / PLANNING.md).

CLI must not import from swe2d.workbench. Pure-Python services must not
import PyQt5. GUI services must not import PyQt5.QtWidgets.
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
    bad = [m for m in imports if m == "PyQt5.QtWidgets" or m.startswith("PyQt5.QtWidgets.")]
    assert not bad, f"{py_file.relative_to(_REPO_ROOT)} imports QtWidgets: {bad}"


@pytest.mark.parametrize("py_file", _python_files(_GUI_SERVICES_DIR), ids=lambda p: p.name)
def test_gui_services_do_not_import_qtwidgets(py_file):
    imports = _imports(py_file.read_text())
    bad = [m for m in imports if m == "PyQt5.QtWidgets" or m.startswith("PyQt5.QtWidgets.")]
    assert not bad, f"{py_file.relative_to(_REPO_ROOT)} imports QtWidgets: {bad}"