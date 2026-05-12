#!/usr/bin/env python3
"""Audit project-owned Python files for docstring and type-hint coverage.

This utility performs a lightweight AST-based scan for:
- Missing function docstrings.
- Missing return annotations.
- Missing parameter annotations (excluding `self` and `cls`).

The script is intentionally conservative and dependency-free so it can run in
QGIS Python environments where extra lint packages may be unavailable.

Usage:
    python3 tools/python_style_audit.py
    python3 tools/python_style_audit.py --max-files 20
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


DEFAULT_EXCLUDES: Tuple[str, ...] = (
    "build/",
    "cpp/third_party/",
    ".git/",
    "__pycache__/",
)


@dataclass(frozen=True)
class Issue:
    """Represents a single style issue in a Python source file.

    Attributes:
        path: Workspace-relative file path.
        line: 1-based source line number.
        kind: Issue category name.
        detail: Human-readable issue detail.
    """

    path: str
    line: int
    kind: str
    detail: str


def _iter_python_files(root: Path, excludes: Sequence[str]) -> Iterable[Path]:
    """Yield project-owned Python files under `root`.

    Args:
        root: Repository root directory.
        excludes: Path prefixes to exclude from scanning.

    Yields:
        Python file paths not matching excluded prefixes.
    """
    for path in root.rglob("*.py"):
        rel = path.relative_to(root).as_posix()
        if any(rel.startswith(prefix) for prefix in excludes):
            continue
        yield path


def _audit_file(path: Path) -> List[Issue]:
    """Collect function-level docstring and annotation issues for a file.

    Args:
        path: Python source path.

    Returns:
        List of discovered issues.
    """
    issues: List[Issue] = []
    rel = path.as_posix()
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception as exc:
        issues.append(Issue(rel, 1, "parse_error", str(exc)))
        return issues

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        doc = ast.get_docstring(node)
        if not doc:
            issues.append(Issue(rel, node.lineno, "missing_docstring", node.name))

        if node.returns is None and node.name != "__init__":
            issues.append(Issue(rel, node.lineno, "missing_return_annotation", node.name))

        args = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
        for arg in args:
            if arg.arg in {"self", "cls"}:
                continue
            if arg.annotation is None:
                issues.append(
                    Issue(
                        rel,
                        node.lineno,
                        "missing_param_annotation",
                        f"{node.name}({arg.arg})",
                    )
                )

    return issues


def _summarize(issues: Sequence[Issue]) -> str:
    """Build a concise human-readable summary.

    Args:
        issues: Collected issues.

    Returns:
        Multi-line summary text.
    """
    by_kind = Counter(issue.kind for issue in issues)
    by_file = Counter(issue.path for issue in issues)

    lines = [
        f"Total issues: {len(issues)}",
        f"  missing_docstring: {by_kind.get('missing_docstring', 0)}",
        f"  missing_return_annotation: {by_kind.get('missing_return_annotation', 0)}",
        f"  missing_param_annotation: {by_kind.get('missing_param_annotation', 0)}",
        f"  parse_error: {by_kind.get('parse_error', 0)}",
        "Top files by issue count:",
    ]
    for path, count in by_file.most_common(12):
        lines.append(f"  {count:4d}  {path}")
    return "\n".join(lines)


def main() -> int:
    """Run the style audit CLI.

    Returns:
        Process exit code. Non-zero indicates issues were found.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Optional limit on number of scanned files (0 = all).",
    )
    args = parser.parse_args()

    root = Path(".").resolve()
    files = sorted(_iter_python_files(root, DEFAULT_EXCLUDES), key=lambda p: p.as_posix())
    if args.max_files > 0:
        files = files[: args.max_files]

    issues: List[Issue] = []
    for path in files:
        issues.extend(_audit_file(path))

    print(f"Scanned files: {len(files)}")
    print(_summarize(issues))

    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
