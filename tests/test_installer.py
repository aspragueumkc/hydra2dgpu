"""Tests for the pure-Python BackendInstaller service."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "qgis_plugin"))

from HYDRA2DGPU.installer import BackendInstaller


def test_env_dir_respects_override(monkeypatch, tmp_path):
    monkeypatch.setenv("HYDRA2DGPU_CACHE_DIR", str(tmp_path))
    inst = BackendInstaller(plugin_dir=".")
    assert inst.env_dir() == tmp_path / ".hydra2dgpu"


def test_wheel_name_format_includes_python_and_platform(monkeypatch):
    inst = BackendInstaller(plugin_dir=".")
    name = inst.wheel_name()
    assert name.startswith("hydra_swe2d-")
    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    assert py_tag in name, f"expected {py_tag} in {name}"
    assert name.endswith((".whl",)), name
    assert any(p in name for p in ("manylinux", "win_amd64"))


def test_wheel_url_uses_releases_download_path():
    """Regression: GitHub release assets live under /releases/download/<tag>/,
    NOT /releases/<tag>/ (the latter 404s). The default base must resolve to
    a URL pip can actually fetch (302 → asset, not 404).

    Fixture-free (uses unittest.mock) so it runs under both pytest and
    the wrap_pytest_style unittest shim.
    """
    from unittest.mock import patch
    import urllib.request as _urlrequest
    import HYDRA2DGPU.installer as _inst_mod

    class _FakeResp:
        def read(self):
            return (
                b'{"assets": [{"name": '
                b'"hydra_swe2d-0.3.0-cp312-cp312-manylinux_2_28_x86_64.whl"}]}'
            )
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    # Default base → the real GitHub download path (the API probe still runs,
    # so stub urlopen to make the test offline-safe).
    with patch.object(_urlrequest, "urlopen", return_value=_FakeResp()):
        inst = BackendInstaller(plugin_dir=".", version="0.3.0")
        url = inst.wheel_url()
    assert url.startswith(
        "https://github.com/aspragueumkc/hydra2dgpu/releases/download/"
    ), url
    assert url.endswith("manylinux_2_28_x86_64.whl"), url

    # Local QA mirror short-circuit keeps the old /v<tag>/ layout.
    with patch.object(_inst_mod, "GITHUB_RELEASES", "http://127.0.0.1:8765"):
        inst2 = BackendInstaller(plugin_dir=".")
        assert inst2.wheel_url().startswith("http://127.0.0.1:8765/v")


def test_site_packages_finds_windows_layout():
    """Regression: Windows venvs use <env>/Lib/site-packages (capital Lib),
    POSIX uses <env>/lib/pythonX.Y/site-packages. The old code only checked
    the POSIX path, so the installer failed on Windows with 'site-packages
    not found in created environment'.

    Fixture-free (tempfile) so it runs under both pytest and the
    wrap_pytest_style unittest shim.
    """
    import tempfile
    from pathlib import Path

    inst = BackendInstaller(plugin_dir=".")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        # Windows layout
        win = root / "win" / "Lib" / "site-packages"
        win.mkdir(parents=True)
        assert inst.site_packages(root / "win") == win
        # POSIX layout
        posix = root / "posix" / "lib" / "python3.12" / "site-packages"
        posix.mkdir(parents=True)
        assert inst.site_packages(root / "posix") == posix
        # Missing layout -> None (not a crash)
        assert inst.site_packages(root / "empty") is None


def test_real_python_prefers_python_exe_over_launcher():
    """Regression: inside OSGeo4W QGIS, sys._base_executable is the QGIS
    launcher (qgis-ltr-bin.exe). venv.create() would copy that launcher into
    the venv and 'python -m ensurepip' would crash (0xC0000005). The
    installer must resolve a real python.exe from base_prefix instead.

    Fixture-free (unittest.mock + tempfile) so it runs under both pytest
    and the wrap_pytest_style unittest shim.
    """
    import tempfile
    from unittest.mock import patch
    import HYDRA2DGPU.installer as _inst_mod
    from pathlib import Path

    inst = BackendInstaller(plugin_dir=".")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        prefix = root / "apps" / "Python312"
        (prefix).mkdir(parents=True)
        real_py = prefix / "python.exe"
        real_py.write_bytes(b"x")  # must exist for _real_python to pick it
        launcher = root / "bin" / "qgis-ltr-bin.exe"

        with patch.object(_inst_mod.sys, "_base_executable", str(launcher)), \
             patch.object(_inst_mod.sys, "executable", str(launcher)), \
             patch.object(_inst_mod.sys, "base_prefix", str(prefix)), \
             patch.object(_inst_mod.sys, "base_exec_prefix", str(prefix)):
            found = inst._real_python()
        assert found == real_py, f"expected {real_py}, got {found}"

        # Fallback: if no real python.exe exists, return sys.executable
        real_py.unlink()
        with patch.object(_inst_mod.sys, "_base_executable", str(launcher)), \
             patch.object(_inst_mod.sys, "executable", str(launcher)), \
             patch.object(_inst_mod.sys, "base_prefix", str(prefix)), \
             patch.object(_inst_mod.sys, "base_exec_prefix", str(prefix)):
            found2 = inst._real_python()
        assert found2 == Path(str(launcher))


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
