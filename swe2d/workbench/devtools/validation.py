"""Validation helpers for the patch builder.

The patch builder is destructive (it rewrites source files). Every patch
MUST be checked against this module's predicates before it is offered to
the user.

Predicates implemented:
    - ``enumerate_all_object_names`` — list every setObjectName("...") across
      a set of view files.
    - ``validate_object_name_unique`` — given a proposed new objectName and
      a set of existing ones, return whether the new name is unique (and if
      not, what the collision is).
    - ``validate_patch_compiles`` — given a proposed new source string,
      confirm ``compile()`` accepts it.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from swe2d.workbench.devtools.ast_patterns import scan_view_file


def enumerate_all_object_names(view_files: List[str]) -> Dict[str, List[str]]:
    """Scan every view file and return ``{object_name: [file_path, ...]}``.

    The same ``object_name`` may appear in more than one file (legitimate
    shared names like ``"run_btn"`` defined on multiple views). The caller
    is responsible for deciding whether a per-file uniqueness rule applies.
    """
    out: Dict[str, List[str]] = {}
    for fp in view_files:
        if not Path(fp).is_file():
            continue
        inv = scan_view_file(fp)
        for oname in inv.all_object_names():
            out.setdefault(oname, []).append(fp)
    return out


def validate_object_name_unique(
    new_name: str,
    existing: Dict[str, List[str]],
    *,
    ignore_file: Optional[str] = None,
) -> Tuple[bool, Optional[str]]:
    """Return ``(True, None)`` if *new_name* does not collide.

    If it does collide, return ``(False, conflicting_file)`` where
    ``conflicting_file`` is the first file already defining it (excluding
    *ignore_file*, which lets callers validate a rename within its own file).
    """
    if not new_name:
        return False, "objectName cannot be empty"
    if new_name not in existing:
        return True, None
    for fp in existing[new_name]:
        if ignore_file and Path(fp).resolve() == Path(ignore_file).resolve():
            continue
        return False, fp
    return True, None


def validate_patch_compiles(new_source: str, file_path: str) -> Tuple[bool, Optional[str]]:
    """Confirm the patched source parses as valid Python.

    Returns ``(True, None)`` on success, or ``(False, error_message)`` on
    a ``SyntaxError``.
    """
    try:
        ast.parse(new_source, filename=file_path)
        return True, None
    except SyntaxError as exc:
        return False, f"{file_path}:{exc.lineno}: {exc.msg}"


__all__ = [
    "enumerate_all_object_names",
    "validate_object_name_unique",
    "validate_patch_compiles",
]