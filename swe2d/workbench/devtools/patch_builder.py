"""Patch builder — turn AST edits into unified-diff ``.patch`` files.

The builder never writes to source. It produces a patch and a target file
path. The user reviews the diff and applies it manually with ``git apply``.

The strategy is **line-level text replacement** driven by AST knowledge:

    1. ``ast.parse`` the source so we can find *which* lines contain a
       ``setObjectName("old")`` / ``QGroupBox("old")`` / etc.
    2. Read the source as text, locate the matched substring on that line,
       swap the string literal in place.
    3. Run ``difflib.unified_diff`` over the old/new line lists.

This avoids ``ast.unparse()`` entirely, which would mangle quote style,
blank lines, and comments. The AST is used only as a *locator*, not a
*re-renderer*.

Edit kinds supported in sprint 1:
    - rename ``setObjectName("old")`` -> ``setObjectName("new")``
    - rename ``QGroupBox("old")`` -> ``QGroupBox("new")``
    - rename ``toolbox.addItem(page, "old")`` -> ``toolbox.addItem(page, "new")``
    - rename ``_add_param_row(form, "old", ...)`` -> ``_add_param_row(form, "new", ...)``
    - rename ``addRow("old", widget)`` -> ``addRow("new", widget)``
"""

from __future__ import annotations

import ast
import difflib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from swe2d.workbench.devtools.ast_patterns import (
    LocatedNode,
    ViewFileInventory,
    scan_view_file,
)


# Match either single- or double-quoted Python string literals on a line.
_STRING_LITERAL_RE = re.compile(r"""(['"])([^'"\n]*?)\1""")


@dataclass(frozen=True)
class Edit:
    """A single string-rewrite edit.

    ``kind`` is the AST finder that located the original node, used for
    round-trip validation in tests.
    """
    kind: str
    file_path: str
    lineno: int
    old_value: str
    new_value: str

    def summary(self) -> str:
        return (
            f"{self.file_path}:{self.lineno}  "
            f"[{self.kind}]  {self.old_value!r} -> {self.new_value!r}"
        )


@dataclass
class PatchResult:
    """The output of a patch build."""
    file_path: str
    original_source: str
    new_source: str
    edits: List[Edit]
    patch_text: str

    def edit_count(self) -> int:
        return len(self.edits)


# ---------------------------------------------------------------------------
# Locator — given an Edit, find the (line_text, span) to rewrite.
# ---------------------------------------------------------------------------


def _find_replacement_span(line_text: str, old_value: str) -> Optional[Tuple[int, int]]:
    """Return ``(start, end)`` byte offsets within *line_text* covering
    the string literal whose contents equal *old_value*.

    The span includes the surrounding quotes so the caller can swap it
    cleanly without leaving behind mismatched delimiters.
    """
    for match in _STRING_LITERAL_RE.finditer(line_text):
        quote, contents = match.group(1), match.group(2)
        if contents == old_value:
            return match.span()
    return None


def _replace_on_line(
    source_lines: List[str],
    lineno: int,
    old_value: str,
    new_value: str,
) -> Tuple[List[str], bool]:
    """Mutate *source_lines* in place: replace the string literal
    matching *old_value* on *lineno* with *new_value*.

    Returns the new line list and a ``changed`` flag.
    """
    if lineno < 1 or lineno > len(source_lines):
        raise ValueError(
            f"edit target line {lineno} out of range "
            f"(file has {len(source_lines)} lines)"
        )
    line_idx = lineno - 1
    line = source_lines[line_idx]
    span = _find_replacement_span(line, old_value)
    if span is None:
        raise ValueError(
            f"line {lineno} does not contain {old_value!r} as a string literal"
        )
    start, end = span
    # Preserve the quote character that was on the line.
    quote = line[start]
    replacement = f"{quote}{new_value}{quote}"
    new_line = line[:start] + replacement + line[end:]
    new_lines = list(source_lines)
    new_lines[line_idx] = new_line
    return new_lines, True


def apply_edit_to_source(
    source: str,
    file_path: str,
    edits: List[Edit],
) -> str:
    """Apply *edits* to *source* and return the new source.

    Edits are applied top-down by lineno so each subsequent edit sees a
    stable line numbering. After patching, the source is verified to
    parse via ``ast.parse`` — any failure raises ``ValueError``.
    """
    # Validate first by walking the AST, so we fail fast on bad edits.
    tree = ast.parse(source, filename=file_path)
    _validate_edits(tree, edits, file_path)

    lines = source.splitlines(keepends=True)
    # Sort edits by ascending lineno so prior edits don't shift later line numbers.
    sorted_edits = sorted(edits, key=lambda e: e.lineno)
    for edit in sorted_edits:
        lines, _ = _replace_on_line(lines, edit.lineno, edit.old_value, edit.new_value)

    new_src = "".join(lines)
    try:
        ast.parse(new_src, filename=file_path)
    except SyntaxError as exc:
        raise ValueError(f"patched source failed to parse: {exc.msg}") from exc
    return new_src


def _validate_edits(tree: ast.Module, edits: List[Edit], file_path: str) -> None:
    """Confirm each edit's (kind, lineno, old_value) matches the AST.

    This runs *before* any text mutation so a bad edit fails cleanly.
    """
    by_kind = _index_nodes(tree)
    for edit in edits:
        nodes = by_kind.get(edit.kind, [])
        match = next(
            (
                n for n in nodes
                if n.lineno == edit.lineno
                and _extract_constant(n.node, edit.kind) == edit.old_value
            ),
            None,
        )
        if match is None:
            raise ValueError(
                f"edit target not found in AST: {edit.summary()} "
                f"(file: {file_path})"
            )


def _index_nodes(tree: ast.Module) -> dict:
    """Map ``kind -> [LocatedNode]`` so the validator doesn't re-walk the tree."""
    from swe2d.workbench.devtools import ast_patterns as P
    return {
        "setObjectName": P.find_set_object_name(tree),
        "QGroupBox": P.find_group_title(tree),
        "addItem": P.find_toolbox_add_item(tree),
        "_add_param_row": P.find_add_param_row(tree),
        "addRow": P.find_add_row_label(tree),
    }


def _extract_constant(call: ast.Call, kind: str) -> Optional[str]:
    """Return the current string value at the position *kind* expects."""
    idx = 1 if kind == "addItem" else 0
    if idx >= len(call.args):
        return None
    arg = call.args[idx]
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        return arg.value
    return None


# ---------------------------------------------------------------------------
# Top-level patch builders
# ---------------------------------------------------------------------------


def build_rename_patch(
    file_path: str,
    edits: List[Edit],
    relative_to: Optional[str] = None,
) -> PatchResult:
    """Read *file_path*, apply *edits*, and return a ``PatchResult``.

    The returned ``PatchResult.patch_text`` is a unified diff that the
    caller can write to a ``.patch`` file.

    Parameters
    ----------
    file_path : str
        Absolute or relative path to the source file to be patched.
    edits : list of Edit
        Edits to apply.
    relative_to : str, optional
        If provided, the diff headers are emitted as paths relative to
        this directory (e.g. the repo root), so the patch can be applied
        with ``git apply`` from that directory.  Defaults to the file path
        itself (absolute).
    """
    path = Path(file_path)
    original = path.read_text(encoding="utf-8")
    new_source = apply_edit_to_source(original, file_path, edits)
    if relative_to is None:
        old_label = new_label = str(path)
    else:
        old_label = new_label = str(path.relative_to(relative_to))
    patch_text = _make_unified_diff(
        original,
        new_source,
        old_label=old_label,
        new_label=new_label,
    )
    return PatchResult(
        file_path=str(path),
        original_source=original,
        new_source=new_source,
        edits=list(edits),
        patch_text=patch_text,
    )


def _make_unified_diff(
    old: str,
    new: str,
    *,
    old_label: str,
    new_label: str,
) -> str:
    """Produce a unified-diff string suitable for ``git apply``."""
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=old_label,
        tofile=new_label,
        n=3,
    )
    return "".join(diff)


def write_patch_file(patch: PatchResult, out_path: str) -> str:
    """Write ``patch.patch_text`` to *out_path* and return the path written.

    Creates parent directories if they don't exist. Refuses to overwrite
    existing files — the dev should never silently overwrite a manually
    curated patch.
    """
    out = Path(out_path)
    if out.exists():
        raise FileExistsError(
            f"refusing to overwrite existing patch file: {out_path}; "
            "delete it first or pick a different path"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(patch.patch_text, encoding="utf-8")
    return str(out)


def rename_in_file(
    file_path: str,
    kind: str,
    lineno: int,
    old_value: str,
    new_value: str,
) -> PatchResult:
    """Build a patch that renames one string constant in *file_path*."""
    edit = Edit(
        kind=kind,
        file_path=file_path,
        lineno=lineno,
        old_value=old_value,
        new_value=new_value,
    )
    return build_rename_patch(file_path, [edit])


__all__ = [
    "Edit",
    "PatchResult",
    "build_rename_patch",
    "rename_in_file",
    "write_patch_file",
    "apply_edit_to_source",
]