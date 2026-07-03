"""Hydra Designer — runtime GUI editor for the SWE2D workbench.

Sprint 1 scope (read-only inspector + rename-to-patch):
    - widget_walker     — QWidget tree → lightweight dataclass
    - ast_patterns      — recognise setObjectName / QGroupBox(title) / addItem / etc.
    - validation        — objectName uniqueness across view files
    - patch_builder     — produce a unified-diff .patch from AST edits
    - inspector_dock    — persistent QTreeWidget showing the live widget tree
    - property_editor   — small dialog to rename / relabel / retitle, emits a patch
    - menu              — DevTools submenu registration (3 actions)

Sprint 2+ (move / add / delete) is intentionally not yet implemented.
The package is purely additive — it imports nothing from the rest of the
workbench at module load, so production users pay zero cost.
"""

from swe2d.workbench.devtools.widget_walker import (
    WidgetNode,
    walk_widget_tree,
    find_node_by_object_name,
)
from swe2d.workbench.devtools.ast_patterns import (
    ViewFileInventory,
    scan_view_file,
    find_set_object_name,
    find_group_title,
    find_toolbox_add_item,
    find_add_param_row,
    find_add_row_label,
)
from swe2d.workbench.devtools.validation import (
    validate_object_name_unique,
    validate_patch_compiles,
    enumerate_all_object_names,
)
from swe2d.workbench.devtools.patch_builder import (
    build_rename_patch,
    write_patch_file,
    apply_edit_to_source,
)

__all__ = [
    # widget_walker
    "WidgetNode",
    "walk_widget_tree",
    "find_node_by_object_name",
    # ast_patterns
    "ViewFileInventory",
    "scan_view_file",
    "find_set_object_name",
    "find_group_title",
    "find_toolbox_add_item",
    "find_add_param_row",
    "find_add_row_label",
    # validation
    "validate_object_name_unique",
    "validate_patch_compiles",
    "enumerate_all_object_names",
    # patch_builder
    "build_rename_patch",
    "write_patch_file",
    "apply_edit_to_source",
]