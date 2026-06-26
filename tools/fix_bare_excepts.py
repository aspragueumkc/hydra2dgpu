#!/usr/bin/env python3
"""Add logging to bare ``except … : pass`` blocks found in ``swe2d/``.

Scans all Python files under ``swe2d/`` (excluding ``tests/`` and
``__pycache__/``) and replaces every ``except X:``\n``pass`` with a
logged version::

    # Before:
    except Exception:
        pass

    # After (file has a logger):
    except Exception as _e:
        logger.warning(f"[ERROR] Exception in {basename}: {_e}")

    # After (file has dialog._log):
    except Exception as _e:
        try:
            self._log(f"[ERROR] Exception in {basename}: {_e}")
        except Exception:
            pass

Exception types considered SAFE (are NOT touched):
  ``RuntimeError``      -- Qt deleted C++ wrapper, expected on stale references
  ``FileNotFoundError`` -- temp-file cleanup race, acceptable on concurrent delete
  ``OSError``           -- file close / unlink in error paths

Usage::

    # Preview changes without writing (dry-run)
    python3 tools/fix_bare_excepts.py --dry-run

    # Apply fixes
    python3 tools/fix_bare_excepts.py

    # Also fix the "safe" types listed above (not recommended)
    python3 tools/fix_bare_excepts.py --no-skip

Notes
-----
- Files that already have ``import logging`` or ``logger = …``  will have
  ``logger.warning()`` injected.
- Files without a logger but inside the workbench (``dialog._log`` /
  ``self._log`` available) will use that.
- Files with neither will have ``logging`` added at the top with a
  module-level ``logger = logging.getLogger(__name__)``.

- A comment like ``# intentional`` or ``# pass`` on the ``except`` or
  ``pass`` line is **NOT** sufficient to skip — we still add logging.
  Only the specific exception-type allowlist above skips.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import List, Tuple

# ── Configuration ───────────────────────────────────────────────────────────
# Directories to scan.
SCAN_DIRS = ["swe2d"]

# Exception types that will be left alone.  These are expected,
# non-actionable conditions from Qt / OS internals.
SAFE_SKIP_TYPES: set[str] = {
    "RuntimeError",
    "FileNotFoundError",
    "OSError",
}

# ── Heuristics for log mechanism availability ───────────────────────────────

HAS_LOG_PATTERNS: List[Tuple[str, str, str]] = [
    # (import_hint, first_arg_name, log_call_template)
    # File defines a class with ``self._log`` or ``dialog._log``
    # We can't fully AST-predict, so we'll check a few keywords.
]

WORKBENCH_LOG_FILES: set[str] = {
    # Files where self._log() is available (workbench dialog / controllers)
    "studio_dialog.py",
    "topology_controller.py",
    "mesh_controller.py",
    "run_controller.py",
    "overlay_controller.py",
    "topology_tab_view.py",
    "studio_results_panel.py",
    "unit_conversion_service.py",
    "widget_persistence_service.py",
    "studio_viewer_plot.py",
}

DIALOG_LOG_FILES: set[str] = {
    # Files where dialog._log() is available (callback-style views)
    "studio_results_panel.py",
    "studio_viewer_plot.py",
    "studio_tab_builder.py",
}

VIEW_LOG_FILES: set[str] = {
    # Files where self._log() is available
    "topology_tab_view.py",
    "mesh_controller.py",
    "topology_controller.py",
    "run_controller.py",
    "overlay_controller.py",
    "unit_conversion_service.py",
    "widget_persistence_service.py",
}


def _log_fn_for(filepath: Path) -> str | None:
    """Detect the best logging mechanism for a file.

    Returns a format string into which ``{basename}`` and ``{exc_type}``
    can be substituted, or ``None`` if we should add a logger.
    """
    name = filepath.name

    # Workbench files with self._log
    if name in VIEW_LOG_FILES:
        return 'self._log(f"[ERROR] {exc_type} in {basename}: {_e}")'

    # Files with dialog._log callback
    if name in DIALOG_LOG_FILES:
        return 'dialog._log(f"[ERROR] {exc_type} in {basename}: {_e}")'

    # Workbench services with self._log
    if name in WORKBENCH_LOG_FILES:
        return 'self._log(f"[ERROR] {exc_type} in {basename}: {_e}")'

    return None


def _has_logger_import(text: str) -> bool:
    return "import logging" in text and "logging.getLogger" in text


def _should_skip(filepath: Path) -> bool:
    """Skip files that are definitely out of scope."""
    for part in filepath.parts:
        if part in ("tests", "__pycache__", ".git"):
            return True
    return False


def fix_file(
    filepath: Path,
    dry_run: bool = False,
    no_skip: bool = False,
) -> Tuple[int, int]:
    """Fix bare except-pass blocks in a single file.

    Returns (fixed_count, skipped_count).
    """
    text = filepath.read_text(encoding="utf-8")
    orig = text
    basename = filepath.name
    pattern = re.compile(r"(\s+)except\s+(\w+)(?:\s+as\s+\w+)?:\s*\n\s+pass")

    skip_types: set[str] = set() if no_skip else SAFE_SKIP_TYPES
    log_fn = _log_fn_for(filepath)
    fixed = 0
    skipped = 0

    def repl(m: re.Match) -> str:
        nonlocal fixed, skipped
        indent = m.group(1)
        exc_type = m.group(2)

        if exc_type in skip_types:
            skipped += 1
            return m.group(0)

        fixed += 1
        msg = f"[ERROR] {exc_type} in {basename}"

        if log_fn is not None:
            # log_fn_template has {exc_type} and {basename} placeholders.
            # The template also contains literal {_e} which must survive .format()
            log_line = log_fn.replace("{exc_type}", exc_type).replace("{basename}", basename)
            return (
                f"{indent}except {exc_type} as _e:\n"
                f"{indent}    try:\n"
                f"{indent}        {log_line}\n"
                f"{indent}    except Exception:\n"
                f"{indent}        pass"
            )
        else:
            # Add a standard logger.warning call
            return (
                f"{indent}except {exc_type} as _e:\n"
                f"{indent}    logger.warning(f\"{msg}: {{_e}}\")"
            )

    new_text = re.sub(pattern, repl, text)

    # Add module-level logger if needed
    if fixed > 0 and log_fn is None and not _has_logger_import(new_text):
        marker = 'from __future__ import annotations'
        if marker in new_text:
            new_text = new_text.replace(
                marker,
                marker + '\nimport logging\n\nlogger = logging.getLogger(__name__)',
            )
        else:
            # Fallback: add after first import block
            new_text = new_text.replace(
                '"""',
                '"""\nimport logging\n\nlogger = logging.getLogger(__name__)\n',
                1,
            )

    if new_text == orig:
        return 0, 0

    if dry_run:
        print(f"  [DRY-RUN] {filepath.relative_to(filepath.parent.parent.parent)}"
              f"  — {fixed} fix(es), {skipped} skipped")
        return fixed, skipped

    filepath.write_text(new_text, encoding="utf-8")
    return fixed, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add logging to bare except-pass blocks in swe2d/.",
    )
    parser.add_argument("--dry-run", "-n", action="store_true",
                        help="Preview changes without writing.")
    parser.add_argument("--no-skip", action="store_true",
                        help="Also flag the 'safe' exception types "
                             "(RuntimeError, FileNotFoundError, OSError).")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent.parent
    total_fixed = 0
    total_skipped = 0
    touched_files = 0

    for scan_dir in SCAN_DIRS:
        base = repo / scan_dir
        if not base.is_dir():
            continue
        for pyfile in sorted(base.rglob("*.py")):
            if _should_skip(pyfile):
                continue
            f, s = fix_file(pyfile, dry_run=args.dry_run, no_skip=args.no_skip)
            if f or s:
                total_fixed += f
                total_skipped += s
                touched_files += 1

    if args.dry_run:
        print(f"\n  Dry-run summary: {total_fixed} fix(es) across {touched_files} file(s)")
        return

    print(f"  Fixed {total_fixed} bare except-pass block(s) across {touched_files} file(s).")
    if total_skipped:
        print(f"  Skipped (safe types): {total_skipped}")

    if total_fixed == 0:
        print("  (Nothing to do — codebase is clean!)")


if __name__ == "__main__":
    main()
