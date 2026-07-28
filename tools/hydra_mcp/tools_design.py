"""Tier C design tools for the HYDRA MCP server (Phase 4).

These tools wrap the Hydra Designer ``swe2d.workbench.devtools`` modules so an
MCP client can propose, preview, and apply small source edits to the workbench
view files.  They are intentionally thin: all AST/patch logic lives in the
existing devtools modules.

All public functions return structured JSON
(``{"ok": true, ...}`` or ``{"ok": false, "error": ...}``) so the server can
forward them directly to the MCP client.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from swe2d.workbench.devtools import patch_builder
from swe2d.workbench.devtools import validation
from swe2d.workbench.devtools.patch_builder import Edit


_REPO_ROOT = Path(__file__).resolve().parents[2]
_VIEW_DIR = _REPO_ROOT / "swe2d" / "workbench" / "views"


def _view_files(view_files: Optional[Iterable[str]] = None) -> List[str]:
    """Return the list of view files to scan/edit.

    If *view_files* is not provided, default to every ``*.py`` in
    ``swe2d/workbench/views``.
    """
    if view_files is not None:
        return [str(Path(f).resolve()) for f in view_files if f]
    if not _VIEW_DIR.is_dir():
        return []
    return sorted(str(p.resolve()) for p in _VIEW_DIR.glob("*.py") if p.is_file())


def _edit_to_dict(edit: Edit) -> Dict[str, Any]:
    """Serialize an :class:`Edit` for JSON round-tripping."""
    return {
        "kind": edit.kind,
        "file_path": edit.file_path,
        "lineno": edit.lineno,
        "old_value": edit.old_value,
        "new_value": edit.new_value,
    }


def _dict_to_edit(data: Dict[str, Any]) -> Edit:
    """Rehydrate an :class:`Edit` from the JSON dict format."""
    return Edit(
        kind=data["kind"],
        file_path=data["file_path"],
        lineno=int(data["lineno"]),
        old_value=data["old_value"],
        new_value=data["new_value"],
    )


def design_rename_widget(
    old: str,
    new: str,
    view_files: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Propose a patch that renames a widget objectName.

    The new name is checked against all existing ``setObjectName`` values in
    the view files.  The old name must exist in exactly one file and appear
    exactly once there.
    """
    if not old:
        return {"ok": False, "error": "old name cannot be empty"}
    if not new:
        return {"ok": False, "error": "new name cannot be empty"}
    if old == new:
        return {"ok": False, "error": "old and new names are identical"}

    files = _view_files(view_files)
    if not files:
        return {"ok": False, "error": "no view files to scan"}

    existing = validation.enumerate_all_object_names(files)
    old_owners = existing.get(old, [])
    if not old_owners:
        return {
            "ok": False,
            "error": f"setObjectName({old!r}) not found in any view file",
        }
    if len(old_owners) > 1:
        return {
            "ok": False,
            "error": f"objectName {old!r} is defined in multiple files: {old_owners}",
        }

    ok, conflict = validation.validate_object_name_unique(
        new, existing, ignore_file=old_owners[0]
    )
    if not ok:
        return {
            "ok": False,
            "error": f"objectName {new!r} is already used: {conflict}",
        }

    try:
        patch = patch_builder.rename(old, new, files)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}

    return {
        "ok": True,
        "patch_text": patch.patch_text,
        "edits": [_edit_to_dict(e) for e in patch.edits],
        "file_path": patch.file_path,
    }


def design_relabel_widget(
    name: str,
    text: str,
    view_files: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Propose a patch that relabels a widget title/label.

    Matches ``QGroupBox("name")``, ``toolbox.addItem(page, "name")``,
    ``_add_param_row(..., "name", ...)`` and ``addRow("name", widget)``.
    """
    if not name:
        return {"ok": False, "error": "name cannot be empty"}
    if not text:
        return {"ok": False, "error": "text cannot be empty"}

    files = _view_files(view_files)
    if not files:
        return {"ok": False, "error": "no view files to scan"}

    try:
        patch = patch_builder.relabel(name, text, files)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}

    return {
        "ok": True,
        "patch_text": patch.patch_text,
        "edits": [_edit_to_dict(e) for e in patch.edits],
        "file_path": patch.file_path,
    }


def design_preview_patch(
    edits: List[Dict[str, Any]],
    view_files: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Return a unified diff for a list of proposed edits without writing files.

    *edits* is a list of dicts with keys ``kind``, ``file_path``, ``lineno``,
    ``old_value``, ``new_value``.  The diff is built per file and concatenated.
    """
    if not isinstance(edits, list) or not edits:
        return {"ok": False, "error": "edits must be a non-empty list"}

    allowed_files = set(_view_files(view_files))

    by_file: Dict[str, List[Edit]] = {}
    for raw in edits:
        try:
            edit = _dict_to_edit(raw)
        except (KeyError, TypeError, ValueError) as exc:
            return {"ok": False, "error": f"invalid edit {raw!r}: {exc}"}
        if edit.file_path not in allowed_files:
            return {
                "ok": False,
                "error": f"file path not in allowed view files: {edit.file_path}",
            }
        by_file.setdefault(edit.file_path, []).append(edit)

    patch_text = ""
    all_edits: List[Edit] = []
    for file_path, file_edits in sorted(by_file.items()):
        try:
            patch = patch_builder.build_rename_patch(file_path, file_edits)
        except (ValueError, FileNotFoundError) as exc:
            return {"ok": False, "error": f"{file_path}: {exc}"}
        patch_text += patch.patch_text
        all_edits.extend(patch.edits)

    return {
        "ok": True,
        "patch_text": patch_text,
        "edits": [_edit_to_dict(e) for e in all_edits],
    }


def design_apply_patch(
    diff: str,
    view_files: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Apply a patch returned by ``design_preview_patch`` to the view files.

    *diff* is either the JSON string returned by ``design_preview_patch`` or
    the raw ``patch_text`` field.  The safest form is the full JSON response,
    which carries the structured edit list; the function deserialises it and
    applies the edits through ``apply_edit_to_source``.
    """
    edits: List[Dict[str, Any]] = []
    if isinstance(diff, str):
        try:
            parsed = json.loads(diff)
        except json.JSONDecodeError:
            # If the caller passed the raw diff text, we cannot safely recover
            # the structured edits from the unified diff alone.
            return {
                "ok": False,
                "error": "diff must be the JSON response from design_preview_patch",
            }
        if not isinstance(parsed, dict):
            return {"ok": False, "error": "diff JSON must be an object"}
        edits = parsed.get("edits", [])
    elif isinstance(diff, dict):
        edits = diff.get("edits", [])
    else:
        return {"ok": False, "error": "diff must be a JSON string or dict"}

    if not edits:
        return {"ok": False, "error": "diff contains no edits"}

    preview = design_preview_patch(edits, view_files=view_files)
    if not preview["ok"]:
        return preview

    allowed_files = set(_view_files(view_files))
    by_file: Dict[str, List[Edit]] = {}
    for raw in edits:
        edit = _dict_to_edit(raw)
        if edit.file_path not in allowed_files:
            return {
                "ok": False,
                "error": f"file path not in allowed view files: {edit.file_path}",
            }
        by_file.setdefault(edit.file_path, []).append(edit)

    for file_path, file_edits in sorted(by_file.items()):
        path = Path(file_path)
        try:
            source = path.read_text(encoding="utf-8")
            new_source = patch_builder.apply_edit_to_source(
                source, file_path, file_edits
            )
            path.write_text(new_source, encoding="utf-8")
        except (ValueError, FileNotFoundError, OSError) as exc:
            return {"ok": False, "error": f"{file_path}: {exc}"}

    return {
        "ok": True,
        "files": sorted(by_file.keys()),
        "edit_count": sum(len(e) for e in by_file.values()),
    }


__all__ = [
    "design_rename_widget",
    "design_relabel_widget",
    "design_preview_patch",
    "design_apply_patch",
]
