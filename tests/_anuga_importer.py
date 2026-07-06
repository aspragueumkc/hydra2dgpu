"""Helpers for importing ANUGA validation scripts as Python modules.

The ANUGA validation tests live in `reference/anuga_validation_tests/`
as plain Python files (not a Python package). This helper adds the
target directory to `sys.path` and imports the requested module,
giving us direct access to ANUGA's analytical and numerical setup
scripts without copying them.

Usage:
    from tests._anuga_importer import import_anuga_module
    analytical = import_anuga_module(
        'reference/anuga_validation_tests/analytical_exact/dam_break_wet/analytical_dam_break_wet.py'
    )
    h, u = analytical.vec_dam_break(x_array, t=0.5, h0=1.0, h1=5.0)
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parent.parent


def _resolve_repo_relative(path_like: str) -> Path:
    """Resolve a path relative to the repo root, or return as-is if absolute."""
    p = Path(path_like)
    if p.is_absolute():
        return p
    return _REPO_ROOT / p


def import_anuga_module(path_like: str, module_name: str | None = None) -> Any:
    """Import a Python file from the repo as a module.

    Parameters
    ----------
    path_like : str
        Path to the .py file, absolute or repo-relative. Must live under
        `reference/anuga_validation_tests/` (we add its parent to sys.path
        so ANUGA-internal relative imports resolve).
    module_name : str, optional
        Synthetic module name. Defaults to the file's stem.

    Returns
    -------
    module
        The imported Python module.
    """
    file_path = _resolve_repo_relative(path_like).resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"ANUGA module not found: {file_path}")
    parent_dir = str(file_path.parent)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    name = module_name or file_path.stem
    spec = importlib.util.spec_from_file_location(name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
