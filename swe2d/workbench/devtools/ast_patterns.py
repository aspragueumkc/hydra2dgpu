"""AST pattern recognition for the SWE2D view files.

The view files follow a small set of reproducible patterns. This module
provides targeted finders for each pattern so the patch builder does not
need to walk the entire AST.

Patterns recognised:

1. ``setObjectName("x")``                — assigns the widget's object name.
2. ``QtWidgets.QGroupBox("title")``      — creates a labelled group box.
3. ``some_toolbox.addItem(page, "title")`` — adds a toolbox / tab page.
4. ``form._add_param_row(form, "Label:", widget)`` — helper-based row add.
5. ``form.addRow("Label:", widget)``      — direct row add.
6. ``QCheckBox("text")``                  — standalone checkbox.

Each finder returns a list of ``LocatedNode`` records (file path, AST node,
source line, the matched string).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional


@dataclass(frozen=True)
class LocatedNode:
    """A located AST node plus its source-line context."""
    file_path: str
    node: ast.AST
    lineno: int
    col_offset: int
    matched: str  # the source text the finder matched

    def short_repr(self) -> str:
        return f"{self.file_path}:{self.lineno}:{self.col_offset}  {self.matched!r}"


@dataclass
class ViewFileInventory:
    """All recognised patterns in a single view file."""
    file_path: str
    set_object_names: List[LocatedNode] = field(default_factory=list)  # noqa: F821
    group_titles: List[LocatedNode] = field(default_factory=list)  # noqa: F821
    toolbox_add_items: List[LocatedNode] = field(default_factory=list)  # noqa: F821
    add_param_rows: List[LocatedNode] = field(default_factory=list)  # noqa: F821
    add_row_labels: List[LocatedNode] = field(default_factory=list)  # noqa: F821
    ast_tree: Optional[ast.Module] = None

    def all_object_names(self) -> List[str]:
        """Return the objectName string of every setObjectName node."""
        out: List[str] = []
        for loc in self.set_object_names:
            arg = loc.node.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                out.append(arg.value)
        return out


# ---------------------------------------------------------------------------
# Pattern finders
# ---------------------------------------------------------------------------


def _string_arg_value(node: ast.Call, idx: int = 0) -> Optional[str]:
    """Return the string value of positional argument *idx*, or ``None``."""
    if idx >= len(node.args):
        return None
    arg = node.args[idx]
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        return arg.value
    return None


def find_set_object_name(tree: ast.Module) -> List[LocatedNode]:
    """Find every ``widget.setObjectName("...")`` call."""
    out: List[LocatedNode] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "setObjectName"
        ):
            value = _string_arg_value(node)
            if value is None:
                continue
            out.append(
                LocatedNode(
                    file_path="<ast>",
                    node=node,
                    lineno=node.lineno,
                    col_offset=node.col_offset,
                    matched=f"setObjectName({value!r})",
                )
            )
    return out


def find_group_title(tree: ast.Module) -> List[LocatedNode]:
    """Find every ``QtWidgets.QGroupBox("title")`` constructor call."""
    out: List[LocatedNode] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "QGroupBox"
        ):
            value = _string_arg_value(node)
            if value is None:
                continue
            out.append(
                LocatedNode(
                    file_path="<ast>",
                    node=node,
                    lineno=node.lineno,
                    col_offset=node.col_offset,
                    matched=f"QGroupBox({value!r})",
                )
            )
    return out


def find_toolbox_add_item(tree: ast.Module) -> List[LocatedNode]:
    """Find every ``toolbox.addItem(page, "title")`` call (title is 2nd arg)."""
    out: List[LocatedNode] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "addItem"
        ):
            value = _string_arg_value(node, idx=1)
            if value is None:
                continue
            out.append(
                LocatedNode(
                    file_path="<ast>",
                    node=node,
                    lineno=node.lineno,
                    col_offset=node.col_offset,
                    matched=f"addItem(..., {value!r})",
                )
            )
    return out


def find_add_param_row(tree: ast.Module) -> List[LocatedNode]:
    """Find every ``self._add_param_row(form, "Label:", widget)`` call."""
    out: List[LocatedNode] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_add_param_row"
        ):
            value = _string_arg_value(node, idx=1)
            if value is None:
                continue
            out.append(
                LocatedNode(
                    file_path="<ast>",
                    node=node,
                    lineno=node.lineno,
                    col_offset=node.col_offset,
                    matched=f"_add_param_row(..., {value!r}, ...)",
                )
            )
    return out


def find_add_row_label(tree: ast.Module) -> List[LocatedNode]:
    """Find every ``layout.addRow("Label:", widget)`` call (label is 1st arg)."""
    out: List[LocatedNode] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "addRow"
        ):
            value = _string_arg_value(node, idx=0)
            if value is None:
                continue
            out.append(
                LocatedNode(
                    file_path="<ast>",
                    node=node,
                    lineno=node.lineno,
                    col_offset=node.col_offset,
                    matched=f"addRow({value!r}, ...)",
                )
            )
    return out


# ---------------------------------------------------------------------------
# Top-level scanner
# ---------------------------------------------------------------------------


def scan_view_file(file_path: str) -> ViewFileInventory:
    """Parse *file_path* once and run every finder against it.

    The returned inventory carries the parsed ``ast.Module`` so callers can
    patch nodes without re-parsing.
    """
    path = Path(file_path)
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(path))
    inv = ViewFileInventory(file_path=str(path), ast_tree=tree)
    inv.set_object_names = find_set_object_name(tree)
    inv.group_titles = find_group_title(tree)
    inv.toolbox_add_items = find_toolbox_add_item(tree)
    inv.add_param_rows = find_add_param_row(tree)
    inv.add_row_labels = find_add_row_label(tree)
    # Stamp file_path on every LocatedNode (finders leave "<ast>").
    for collection in (
        inv.set_object_names,
        inv.group_titles,
        inv.toolbox_add_items,
        inv.add_param_rows,
        inv.add_row_labels,
    ):
        for loc in collection:
            object.__setattr__(loc, "file_path", str(path))
    return inv


def all_callable_finders() -> List:
    """Return every finder function in this module (for tests / discovery)."""
    return [
        find_set_object_name,
        find_group_title,
        find_toolbox_add_item,
        find_add_param_row,
        find_add_row_label,
    ]


__all__ = [
    "LocatedNode",
    "ViewFileInventory",
    "scan_view_file",
    "find_set_object_name",
    "find_group_title",
    "find_toolbox_add_item",
    "find_add_param_row",
    "find_add_row_label",
    "all_callable_finders",
]