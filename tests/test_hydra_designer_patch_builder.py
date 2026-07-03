"""Tests for the patch builder.

Round-trip tests: parse → patch → unparse → diff → parse. The patched
source must compile, and the patch text must contain the expected
``-`` / ``+`` lines.
"""

from __future__ import annotations

import ast
import os
import re
import textwrap

import pytest

from swe2d.workbench.devtools.ast_patterns import scan_view_file
from swe2d.workbench.devtools.patch_builder import (
    Edit,
    PatchResult,
    apply_edit_to_source,
    build_rename_patch,
    rename_in_file,
    write_patch_file,
)
from swe2d.workbench.devtools.validation import (
    enumerate_all_object_names,
    validate_object_name_unique,
    validate_patch_compiles,
)


SAMPLE_VIEW = textwrap.dedent(
    '''
    """Sample view."""
    from qgis.PyQt import QtWidgets

    class V(QtWidgets.QWidget):
        def _build(self):
            self.cfl_spin = QtWidgets.QDoubleSpinBox()
            self.cfl_spin.setObjectName("cfl_spin")
            self.rain_chk = QtWidgets.QCheckBox("Enable rain")
            self.rain_chk.setObjectName("rain_chk")

            group = QtWidgets.QGroupBox("Time Stepping")
            tb = QtWidgets.QToolBox()
            tb.addItem(self, "Solver Parameters")
    '''
)


def _write_sample(tmp_path) -> str:
    f = tmp_path / "v.py"
    f.write_text(SAMPLE_VIEW, encoding="utf-8")
    return str(f)


# ---------------------------------------------------------------------------
# Pure-python: apply_edit_to_source
# ---------------------------------------------------------------------------


def test_apply_edit_changes_setobjectname(tmp_path):
    fp = _write_sample(tmp_path)
    inv = scan_view_file(fp)
    loc = inv.set_object_names[0]
    edit = Edit(
        kind="setObjectName",
        file_path=fp,
        lineno=loc.lineno,
        old_value="cfl_spin",
        new_value="courant_spin",
    )
    new_src = apply_edit_to_source(SAMPLE_VIEW, fp, [edit])
    assert "courant_spin" in new_src
    assert '"cfl_spin"' not in new_src
    # Must still parse.
    ast.parse(new_src)


def test_apply_edit_changes_qgroupbox_title(tmp_path):
    fp = _write_sample(tmp_path)
    inv = scan_view_file(fp)
    loc = inv.group_titles[0]
    edit = Edit(
        kind="QGroupBox",
        file_path=fp,
        lineno=loc.lineno,
        old_value="Time Stepping",
        new_value="Time Stepping (advanced)",
    )
    new_src = apply_edit_to_source(SAMPLE_VIEW, fp, [edit])
    assert '"Time Stepping (advanced)"' in new_src
    ast.parse(new_src)


def test_apply_edit_changes_additem_title(tmp_path):
    fp = _write_sample(tmp_path)
    inv = scan_view_file(fp)
    loc = inv.toolbox_add_items[0]
    edit = Edit(
        kind="addItem",
        file_path=fp,
        lineno=loc.lineno,
        old_value="Solver Parameters",
        new_value="Solver",
    )
    new_src = apply_edit_to_source(SAMPLE_VIEW, fp, [edit])
    assert '"Solver"' in new_src
    ast.parse(new_src)


def test_apply_edit_old_value_mismatch_raises(tmp_path):
    fp = _write_sample(tmp_path)
    inv = scan_view_file(fp)
    loc = inv.set_object_names[0]
    edit = Edit(
        kind="setObjectName",
        file_path=fp,
        lineno=loc.lineno,
        old_value="WRONG_OLD_VALUE",
        new_value="anything",
    )
    with pytest.raises(ValueError):
        apply_edit_to_source(SAMPLE_VIEW, fp, [edit])


def test_apply_multiple_edits(tmp_path):
    fp = _write_sample(tmp_path)
    inv = scan_view_file(fp)
    e1 = Edit(
        kind="setObjectName",
        file_path=fp,
        lineno=inv.set_object_names[0].lineno,
        old_value="cfl_spin",
        new_value="courant_spin",
    )
    e2 = Edit(
        kind="QGroupBox",
        file_path=fp,
        lineno=inv.group_titles[0].lineno,
        old_value="Time Stepping",
        new_value="Time Stepping (advanced)",
    )
    new_src = apply_edit_to_source(SAMPLE_VIEW, fp, [e1, e2])
    assert '"courant_spin"' in new_src
    assert '"Time Stepping (advanced)"' in new_src
    ast.parse(new_src)


# ---------------------------------------------------------------------------
# build_rename_patch
# ---------------------------------------------------------------------------


def test_build_rename_patch_returns_unified_diff(tmp_path):
    fp = _write_sample(tmp_path)
    patch = rename_in_file(
        file_path=fp,
        kind="setObjectName",
        lineno=scan_view_file(fp).set_object_names[0].lineno,
        old_value="cfl_spin",
        new_value="courant_spin",
    )
    assert isinstance(patch, PatchResult)
    assert patch.edit_count() == 1
    # Unified diff should mention - and + for the renamed line.
    assert '"cfl_spin"' in patch.patch_text
    assert '"courant_spin"' in patch.patch_text
    assert re.search(r"^-.*cfl_spin", patch.patch_text, re.MULTILINE)
    assert re.search(r"^\+.*courant_spin", patch.patch_text, re.MULTILINE)


def test_write_patch_file_creates_file(tmp_path):
    fp = _write_sample(tmp_path)
    patch = rename_in_file(
        file_path=fp,
        kind="setObjectName",
        lineno=scan_view_file(fp).set_object_names[0].lineno,
        old_value="cfl_spin",
        new_value="courant_spin",
    )
    out = tmp_path / "out.patch"
    written = write_patch_file(patch, str(out))
    assert written == str(out)
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert '"cfl_spin"' in text


def test_write_patch_file_refuses_overwrite(tmp_path):
    fp = _write_sample(tmp_path)
    patch = rename_in_file(
        file_path=fp,
        kind="setObjectName",
        lineno=scan_view_file(fp).set_object_names[0].lineno,
        old_value="cfl_spin",
        new_value="courant_spin",
    )
    out = tmp_path / "out.patch"
    out.write_text("EXISTING\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        write_patch_file(patch, str(out))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_enumerate_all_object_names(tmp_path):
    fp = _write_sample(tmp_path)
    out = enumerate_all_object_names([fp])
    assert out == {"cfl_spin": [fp], "rain_chk": [fp]}


def test_validate_object_name_unique_ok(tmp_path):
    fp = _write_sample(tmp_path)
    existing = enumerate_all_object_names([fp])
    ok, conflict = validate_object_name_unique("new_one", existing)
    assert ok is True
    assert conflict is None


def test_validate_object_name_unique_collision(tmp_path):
    fp = _write_sample(tmp_path)
    existing = enumerate_all_object_names([fp])
    ok, conflict = validate_object_name_unique("cfl_spin", existing)
    assert ok is False
    assert conflict == fp


def test_validate_object_name_unique_ignore_self(tmp_path):
    fp = _write_sample(tmp_path)
    existing = enumerate_all_object_names([fp])
    # Renaming cfl_spin → cfl_spin within its own file is fine.
    ok, conflict = validate_object_name_unique(
        "cfl_spin", existing, ignore_file=fp
    )
    assert ok is True


def test_validate_patch_compiles_ok():
    ok, err = validate_patch_compiles(SAMPLE_VIEW, "<test>")
    assert ok is True
    assert err is None


def test_validate_patch_compiles_syntax_error():
    ok, err = validate_patch_compiles("def broken(:\n    pass\n", "<test>")
    assert ok is False
    assert err is not None