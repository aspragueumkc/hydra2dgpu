#!/usr/bin/env python3
"""Find bare ``except … : pass`` blocks in the codebase.

Scans ``swe2d/`` (excluding ``tests/`` and ``__pycache__/``) for
``except`` blocks whose body is only ``pass``.  Results are classified
by exception type so you can see at a glance which ones are likely
intentional (``RuntimeError``, ``FileNotFoundError``, ``OSError``,
specific builtins) vs which are bare ``Exception`` swallows.

Usage::

    # Show everything
    python3 tools/find_bare_excepts.py

    # Only show bare ``except Exception: pass`` (most dangerous)
    python3 tools/find_bare_excepts.py --only-bare
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple

# Directories to scan relative to the repo root.
SCAN_DIRS = ["swe2d"]

# Exception types that are arguably fine to silently pass.
# These are the "skip" list from the fixer: specific exceptions
# raised by Qt/cPython internals for expected non-actionable cases.
SAFE_SKIP_TYPES: set[str] = {
    "RuntimeError",  # Qt deleted C++ wrapper
    "FileNotFoundError",  # temp-file cleanup race
    "OSError",  # file close / unlink
    "TypeError",  # data-format conversion
    "ValueError",  # data-format conversion
    "KeyError",  # dict lookup
}


def find_except_passes(
    root: Path,
    only_bare: bool = False,
) -> List[Tuple[str, int, str, str, str]]:
    """Return list of ``(rel_path, line_no, exc_type, line_text, next_line)``."""
    results: List[Tuple[str, int, str, str, str]] = []
    pattern = re.compile(r"(\s+)except\s+(\w+(?:\s*\([^)]*\))?)(?:\s+as\s+\w+)?:\s*$")

    for scan_dir in SCAN_DIRS:
        base = root / scan_dir
        if not base.is_dir():
            continue
        for pyfile in sorted(base.rglob("*.py")):
            rel = pyfile.relative_to(root)
            if "tests" in rel.parts or "__pycache__" in rel.parts:
                continue
            text = pyfile.read_text(encoding="utf-8")
            lines = text.splitlines()

            for lineno, line in enumerate(lines, 1):
                m = pattern.match(line)
                if not m:
                    continue
                # Check next line is only pass
                if lineno >= len(lines):
                    continue
                next_line = lines[lineno]
                if not re.match(r"^\s+pass\s*($|#)", next_line):
                    continue

                indent = m.group(1)
                exc_type = m.group(2).strip()
                # Strip trailing parentheses for tuple types
                exc_type = exc_type.split("(")[0].strip()

                if only_bare and exc_type not in ("Exception", "BaseException"):
                    continue

                results.append((str(rel), lineno, exc_type, line.strip(), next_line.strip()))

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find bare `except ... : pass` blocks in the codebase.",
    )
    parser.add_argument(
        "--only-bare",
        action="store_true",
        help="Only report bare ``except Exception: pass`` (skip specific exception types).",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Do NOT apply the SAFE_SKIP_TYPES filter.  Report EVERY except-pass.",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent.parent
    results = find_except_passes(repo, only_bare=args.only_bare)

    # Group for display
    by_type: dict[str, list] = {}
    for r in results:
        by_type.setdefault(r[2], []).append(r)

    total = len(results)
    safe_count = 0
    unsafe_count = 0

    print(f"\n{'=' * 70}")
    print(f"  Found {total} bare except-pass blocks in swe2d/")
    print(f"{'=' * 70}\n")

    for exc_type in sorted(by_type, key=lambda t: -len(by_type[t])):
        entries = by_type[exc_type]
        is_safe = exc_type in SAFE_SKIP_TYPES and not args.no_skip
        marker = "🟢 SAFE" if is_safe else "🔴 UNSAFE"
        if is_safe:
            safe_count += len(entries)
        else:
            unsafe_count += len(entries)

        print(f"  {marker}  {exc_type}  ({len(entries)} occurrences)")
        for rel, line, _et, raw, nxt in entries:
            print(f"       {rel}:{line}  {raw}")
        print()

    print(f"{'=' * 70}")
    print(f"  Total: {total}  |  Safe (specific types): {safe_count}"
          f"  |  Unsafe (bare Exception): {unsafe_count}")
    print(f"{'=' * 70}")

    if unsafe_count > 0:
        print("\n  Run: python3 tools/fix_bare_excepts.py")
        print("  to automatically add logging to unsafe blocks.\n")
    return 1 if unsafe_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
