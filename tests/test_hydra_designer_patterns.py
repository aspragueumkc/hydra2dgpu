"""Tests for hydra_designer AST pattern recognition.

These tests do NOT require a running QGIS instance — they operate purely
on AST and source strings. Run with::

    python -m pytest tests/test_hydra_designer_patterns.py -v
"""

from __future__ import annotations

import ast
from textwrap import dedent

from swe2d.workbench.devtools.ast_patterns import (
    find_add_param_row,
    find_add_row_label,
    find_group_title,
    find_set_object_name,
    find_toolbox_add_item,
    scan_view_file,
)


SAMPLE_VIEW = dedent(
    '''
    """Sample view for tests."""
    from qgis.PyQt import QtWidgets

    class SampleView(QtWidgets.QWidget):
        def _build_ui(self):
            self.cfl_spin = QtWidgets.QDoubleSpinBox()
            self.cfl_spin.setObjectName("cfl_spin")
            self.cfl_spin.setToolTip("Courant number")

            self.adaptive_chk = QtWidgets.QCheckBox("Adaptive")
            self.adaptive_chk.setObjectName("adaptive_chk")

            self.rain_chk = QtWidgets.QCheckBox("Enable rain")
            self.rain_chk.setObjectName("rain_chk")

            solver_group = QtWidgets.QGroupBox("Time Stepping")
            rain_group = QtWidgets.QGroupBox("Rain / Hydrology")

            form = QtWidgets.QFormLayout()
            form.addRow("CFL:", self.cfl_spin)
            self._add_param_row(form, "Adaptive:", self.adaptive_chk)

            tb = QtWidgets.QToolBox()
            tb.addItem(self, "Solver Parameters")
            tb.addItem(self, "Rain / Hydrology")
    '''
)


def _parse(src: str) -> ast.Module:
    return ast.parse(src, filename="<test>")


def test_find_set_object_name_picks_every_call():
    tree = _parse(SAMPLE_VIEW)
    matches = find_set_object_name(tree)
    names = [_extract(m.node, 0) for m in matches]
    assert names == ["cfl_spin", "adaptive_chk", "rain_chk"], names


def test_find_group_title_picks_every_qgroupbox():
    tree = _parse(SAMPLE_VIEW)
    matches = find_group_title(tree)
    titles = [_extract(m.node, 0) for m in matches]
    assert titles == ["Time Stepping", "Rain / Hydrology"], titles


def test_find_toolbox_add_item_picks_second_arg():
    tree = _parse(SAMPLE_VIEW)
    matches = find_toolbox_add_item(tree)
    titles = [_extract(m.node, 1) for m in matches]
    assert titles == ["Solver Parameters", "Rain / Hydrology"], titles


def test_find_add_param_row_picks_label_argument():
    tree = _parse(SAMPLE_VIEW)
    matches = find_add_param_row(tree)
    labels = [_extract(m.node, 1) for m in matches]
    assert labels == ["Adaptive:"], labels


def test_find_add_row_label_picks_first_string_arg():
    tree = _parse(SAMPLE_VIEW)
    matches = find_add_row_label(tree)
    labels = [_extract(m.node, 0) for m in matches]
    assert labels == ["CFL:"], labels


def test_scan_view_file_returns_inventory(tmp_path):
    f = tmp_path / "v.py"
    f.write_text(SAMPLE_VIEW, encoding="utf-8")
    inv = scan_view_file(str(f))
    assert inv.file_path == str(f)
    assert len(inv.set_object_names) == 3
    assert len(inv.group_titles) == 2
    assert len(inv.toolbox_add_items) == 2
    assert len(inv.add_param_rows) == 1
    assert len(inv.add_row_labels) == 1
    assert inv.all_object_names() == ["cfl_spin", "adaptive_chk", "rain_chk"]


def test_empty_file_no_crash(tmp_path):
    f = tmp_path / "empty.py"
    f.write_text("", encoding="utf-8")
    inv = scan_view_file(str(f))
    assert inv.set_object_names == []
    assert inv.group_titles == []


def _extract(call_node, idx: int):
    arg = call_node.args[idx]
    return getattr(arg, "value", None)