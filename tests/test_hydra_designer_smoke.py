"""Smoke test: rename a real objectName in a real view file via patch_builder.

This guards against future refactors silently breaking the end-to-end flow.
"""

from __future__ import annotations

import os
import subprocess
import tempfile

import pytest

from swe2d.workbench.devtools.ast_patterns import scan_view_file
from swe2d.workbench.devtools.patch_builder import (
    Edit,
    build_rename_patch,
    rename_in_file,
)
from swe2d.workbench.devtools.validation import (
    enumerate_all_object_names,
    validate_object_name_unique,
)


REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir)
)
VIEW_FILE = os.path.join(
    REPO_ROOT, "swe2d", "workbench", "views", "model_tab_view.py"
)


def _find_object_name(file_path: str, object_name: str):
    inv = scan_view_file(file_path)
    for loc in inv.set_object_names:
        arg = loc.node.args[0]
        if getattr(arg, "value", None) == object_name:
            return loc
    return None


@pytest.mark.skipif(
    not os.path.isfile(VIEW_FILE),
    reason="model_tab_view.py not in repo",
)
def test_model_tab_view_scans_cleanly():
    inv = scan_view_file(VIEW_FILE)
    names = inv.all_object_names()
    # Sanity: we should find many widgets.
    assert len(names) > 30
    # No duplicates in a single file (assuming clean code).
    assert len(names) == len(set(names))


@pytest.mark.skipif(
    not os.path.isfile(VIEW_FILE),
    reason="model_tab_view.py not in repo",
)
def test_rename_cfl_spin_builds_valid_patch():
    """End-to-end: rename cfl_spin → courant_spin, validate the result."""
    loc = _find_object_name(VIEW_FILE, "cfl_spin")
    assert loc is not None, "cfl_spin must exist in model_tab_view.py"

    edit = Edit(
        kind="setObjectName",
        file_path=VIEW_FILE,
        lineno=loc.lineno,
        old_value="cfl_spin",
        new_value="courant_spin",
    )
    patch = build_rename_patch(VIEW_FILE, [edit], relative_to=REPO_ROOT)

    # 1. Patch contains both old and new.
    assert '"cfl_spin"' in patch.patch_text
    assert '"courant_spin"' in patch.patch_text

    # 2. Patched source compiles.
    import ast
    ast.parse(patch.new_source)

    # 3. The actual setObjectName call now uses the new value.
    #    (We don't assert '"cfl_spin"' is gone, because the file's docstring
    #    and comments may still mention the old name.)
    assert 'setObjectName("courant_spin")' in patch.new_source
    assert 'setObjectName("cfl_spin")' not in patch.new_source

    # 4. `git apply --check` would succeed if we had a git repo.
    #    We don't apply for real — we just confirm the patch is well-formed.
    #    Smoke: try `patch` command in dry-run mode if available.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".patch", delete=False, encoding="utf-8"
    ) as f:
        f.write(patch.patch_text)
        patch_path = f.name
    try:
        result = subprocess.run(
            ["patch", "--dry-run", "-p0", "-i", patch_path],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            timeout=30,
        )
        assert result.returncode == 0, (
            f"patch --dry-run failed:\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    finally:
        os.unlink(patch_path)


@pytest.mark.skipif(
    not os.path.isfile(VIEW_FILE),
    reason="model_tab_view.py not in repo",
)
def test_rename_collides_with_existing_objectname():
    """Trying to rename cfl_spin → rain_rate_spin must be detected as a collision."""
    loc = _find_object_name(VIEW_FILE, "cfl_spin")
    assert loc is not None

    existing = enumerate_all_object_names([VIEW_FILE])
    # No ``ignore_file`` here — we want to confirm that proposing
    # ``rain_rate_spin`` as a new name *fails* even though ``cfl_spin``
    # and ``rain_rate_spin`` are in the same file.  (Renaming within the
    # same file is fine; *adding* a new widget with an existing name is not.)
    ok, conflict = validate_object_name_unique("rain_rate_spin", existing)
    assert ok is False
    assert conflict == VIEW_FILE