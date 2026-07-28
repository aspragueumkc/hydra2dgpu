import unittest
"""Verify the QGIS-repo packaging script produces a compliant zip."""
from __future__ import annotations
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _build():
    out = subprocess.run(
        ["python", "tools/package_plugin.py"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert out.returncode == 0, f"packager failed: {out.stderr}"


def test_zip_under_20mb():
    _build()
    zip_path = ROOT / "dist" / "HYDRA2DGPU.zip"
    assert zip_path.exists(), "packager must produce dist/HYDRA2DGPU.zip"
    mb = zip_path.stat().st_size / (1024 * 1024)
    assert mb < 20, f"zip {mb:.2f}MB exceeds QGIS 20MB limit"


def test_zip_excludes_binaries_and_cache():
    _build()
    with zipfile.ZipFile(ROOT / "dist" / "HYDRA2DGPU.zip") as zf:
        names = zf.namelist()
    banned = [n for n in names if n.endswith((".so", ".pyd", ".dll", ".dylib", ".pyc"))
              or "__pycache__" in n or "/.git/" in n or "/tests/" in n]
    assert not banned, f"banned entries in zip: {banned}"

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
