"""Static check for a known runtime bug in PGProfileWidget.refresh.

The profile viewer once referenced an undefined local ``t`` when loading
structure flows. This test parses the method and fails if any bare ``t``
name is loaded inside ``PGProfileWidget.refresh``.
"""

import ast
import inspect

import pytest

from swe2d.workbench.views import studio_viewer_profile_pg


def _find_method_node(module_ast, class_name: str, method_name: str):
    for node in ast.walk(module_ast):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return item
    return None


def test_refresh_method_does_not_use_undefined_t():
    """PGProfileWidget.refresh must not load a bare variable named 't'."""
    source = inspect.getsource(studio_viewer_profile_pg)
    module_ast = ast.parse(source)
    refresh = _find_method_node(module_ast, "PGProfileWidget", "refresh")
    assert refresh is not None, "refresh method not found"

    bad_nodes = [
        node for node in ast.walk(refresh)
        if isinstance(node, ast.Name) and node.id == "t" and isinstance(node.ctx, ast.Load)
    ]
    assert not bad_nodes, (
        f"PGProfileWidget.refresh uses undefined variable 't' at line(s) "
        f"{[node.lineno for node in bad_nodes]}"
    )
