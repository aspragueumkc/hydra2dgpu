"""Detect tests that look like coverage but don't assert anything.

Spec: docs/specs/2026-07-26-test-discipline-design.md §3.2
Catches audit §4 row #3 (Test theatre).

Patterns flagged:
  - Pytest-style `def test_*` at module level (not in a TestCase class)
  - `assertTrue(True)` (or similar no-op) after a `with assertRaises(): pass`
  - Test methods whose only assertions are on `mock`/`MagicMock` objects

The temporary pytest-style allowlist was removed in Phase 4 T4.4.
"""
import ast
import os
import unittest


def _is_test_module(path):
    base = os.path.basename(path)
    return base.startswith("test_") and base.endswith(".py")


def _is_already_wrapped(source):
    """A file is wrapped if it has the `_PytestStyleWrapper` class.

    The wrap script adds this class at the bottom of pytest-style files.
    The module-level `def test_*` lines remain in the source (the wrap
    removes them from globals() at import time, not from the text), so
    the AST walker must skip files that already have the wrapper.
    """
    return "_PytestStyleWrapper" in source


def _walk_test_methods(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            yield node


def _has_after_context_noop(func):
    """Detect: `with self.assertRaises(): pass` followed by a tautological assert."""
    if len(func.body) < 2:
        return False
    for i, stmt in enumerate(func.body[:-1]):
        if not isinstance(stmt, ast.With):
            continue
        if len(stmt.body) != 1 or not isinstance(stmt.body[0], ast.Pass):
            continue
        nxt = func.body[i + 1]
        if (
            isinstance(nxt, ast.Expr)
            and isinstance(nxt.value, ast.Call)
            and isinstance(nxt.value.func, ast.Attribute)
            and nxt.value.func.attr in {
                "assertTrue", "assertEqual", "assertIsNone", "assertIsNotNone",
            }
        ):
            return True
    return False


def _has_only_mock_asserts(func):
    """A test asserting only on `mock`/`MagicMock` is theatre."""
    for stmt in ast.walk(func):
        if isinstance(stmt, ast.Assert):
            src = ast.unparse(stmt.test)
            if "mock" in src.lower() or "MagicMock" in src:
                return True
    return False


def _has_module_level_test(tree):
    """Pytest-style: `def test_*` at module level (not in a class)."""
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            return True
    return False


class TestNoTestTheatre(unittest.TestCase):
    def test_no_test_theatre(self):
        findings = []
        for root, dirs, files in os.walk("tests"):
            dirs[:] = [d for d in dirs if not d.startswith("__") and not d.startswith(".")]
            for fname in files:
                if not _is_test_module(os.path.join(root, fname)):
                    continue
                path = os.path.join(root, fname)
                with open(path) as f:
                    source = f.read()
                if _is_already_wrapped(source):
                    continue
                try:
                    tree = ast.parse(source, filename=path)
                except SyntaxError:
                    continue

                if _has_module_level_test(tree):
                    findings.append(
                        f"{path}: pytest-style `def test_*` at module level. "
                        "Wrap in `unittest.TestCase` or convert."
                    )
                for func in _walk_test_methods(tree):
                    if _has_after_context_noop(func):
                        findings.append(
                            f"{path}:{func.lineno}: tautological assert after "
                            "`with assertRaises(): pass`. Replace with a real assertion."
                        )
                    if _has_only_mock_asserts(func):
                        findings.append(
                            f"{path}:{func.lineno}: assertions only on mock objects. "
                            "Assert on the real code under test, not on the mock."
                        )
        if findings:
            self.fail(
                "Test theatre detected:\n\n"
                + "\n".join(f"  - {f}" for f in findings)
                + "\n\nFix or delete the offending test."
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)