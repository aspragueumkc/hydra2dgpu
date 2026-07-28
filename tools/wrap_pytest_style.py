#!/usr/bin/env python3
"""Wrap pytest-style `def test_*` at module level into unittest.TestCase.

Spec: docs/specs/2026-07-26-test-discipline-design.md §3.4 + §7.1 (PR #4)

Idempotent. Reads every tests/test_*.py, finds module-level `def test_*`,
appends a generated `_PytestStyleWrapper(unittest.TestCase)` class that
holds staticmethod references to each test function.

Usage:
    python3 tools/wrap_pytest_style.py           # wrap all
    python3 tools/wrap_pytest_style.py --check   # dry-run, exit 1 if any would be changed

Ponytail: pure-stdlib AST pass. No external dependencies.
"""
import argparse
import ast
import os
import sys

WRAPPER_TEMPLATE = '''
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
'''


def find_module_level_tests(tree):
    """Return names of module-level `def test_*` functions (not inside a class)."""
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            yield node.name


def already_wrapped(source):
    return "_PytestStyleWrapper" in source


def wrap_file(path, dry_run=False):
    with open(path) as f:
        source = f.read()
    if already_wrapped(source):
        return False
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as e:
        print(f"  SKIP (syntax error): {path}: {e}", file=sys.stderr)
        return False
    module_level = list(find_module_level_tests(tree))
    if not module_level:
        return False
    if not dry_run:
        # Ensure unittest is imported
        if "import unittest" not in source and "from unittest" not in source:
            source = "import unittest\n" + source
        source = source.rstrip() + "\n" + WRAPPER_TEMPLATE
        with open(path, "w") as f:
            f.write(source)
    print(f"  {'would wrap' if dry_run else 'wrapped'} {len(module_level)} tests in {path}")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="dry-run; exit 1 if any file would change")
    parser.add_argument("--path", default="tests", help="directory to scan (default: tests)")
    args = parser.parse_args()
    changed = 0
    for fname in sorted(os.listdir(args.path)):
        if not (fname.startswith("test_") and fname.endswith(".py")):
            continue
        path = os.path.join(args.path, fname)
        if wrap_file(path, dry_run=args.check):
            changed += 1
    if args.check and changed:
        print(f"\n{changed} file(s) would change. Re-run without --check to apply.")
        sys.exit(1)
    print(f"\n{changed} file(s) changed.")


if __name__ == "__main__":
    main()