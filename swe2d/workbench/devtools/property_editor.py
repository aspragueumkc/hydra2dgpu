"""Property editor dialog — emit a ``.patch`` file from a rename.

The dialog is intentionally tiny. It shows the four string fields we can
patch in sprint 1:

    - objectName          (via setObjectName)
    - label               (via _add_param_row / addRow / QLabel)
    - tooltip             (via setToolTip)        — sprint 1.5 (skip for now)
    - group / page title  (via QGroupBox / addItem)

Sprint 1 scope is "rename-only". On "Generate Patch…", it:
    1. Asks the user where to save the patch (QFileDialog.getSaveFileName).
    2. Calls ``patch_builder.rename_in_file`` for the changed field.
    3. Validates the resulting source compiles.
    4. Validates the proposed objectName is unique (if it was changed).
    5. Writes the patch and shows a success message with the path.
"""

from __future__ import annotations

import os
from typing import Optional

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from swe2d.workbench.devtools.ast_patterns import scan_view_file
from swe2d.workbench.devtools.patch_builder import (
    Edit,
    build_rename_patch,
)
from swe2d.workbench.devtools.validation import (
    enumerate_all_object_names,
    validate_object_name_unique,
    validate_patch_compiles,
)
from swe2d.workbench.devtools.widget_walker import WidgetNode


# The directory under the project root that holds view files.
_VIEW_FILE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    "swe2d",
    "workbench",
    "views",
)


class PropertyEditorDialog(QDialog):
    """Tiny dialog: edit a few string fields, then "Generate Patch…".

    Parameters
    ----------
    node : WidgetNode
        The widget selected in the inspector tree.
    view_files : list of str
        Paths to the view files we will scan for the widget's source.
    """

    def __init__(
        self,
        node: WidgetNode,
        view_files: list,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("HydraDesignerPropertyEditor")
        self.setWindowTitle(f"Edit properties — {node.class_name}")
        self._node = node
        self._view_files = view_files

        form = QFormLayout()

        self._object_name_edit = QLineEdit(node.object_name)
        self._object_name_edit.setObjectName("prop_object_name")
        form.addRow("objectName:", self._object_name_edit)

        self._class_label = QLabel(node.class_name)
        self._class_label.setObjectName("prop_class_label")
        form.addRow("Class:", self._class_label)

        self._id_label = QLabel(str(node.widget_id))
        self._id_label.setObjectName("prop_id_label")
        form.addRow("Live id():", self._id_label)

        self._parent_label = QLabel(str(node.parent_id) if node.parent_id else "<root>")
        self._parent_label.setObjectName("prop_parent_label")
        form.addRow("Parent id:", self._parent_label)

        buttons = QDialogButtonBox()
        self._generate_btn = QPushButton("Generate Patch…")
        self._generate_btn.setObjectName("prop_generate_btn")
        self._generate_btn.clicked.connect(self._on_generate)
        buttons.addButton(self._generate_btn, QDialogButtonBox.AcceptRole)
        buttons.addButton(QDialogButtonBox.Cancel)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(QLabel(
            "Sprint 1: objectName only. Label/title/tooltip edits arrive in sprint 1.5."
        ))
        layout.addWidget(buttons)

    # ------------------------------------------------------------------
    # Internal slots
    # ------------------------------------------------------------------

    def _on_generate(self) -> None:
        new_name = self._object_name_edit.text().strip()
        old_name = self._node.object_name
        if new_name == old_name:
            QMessageBox.information(
                self, "No change",
                "objectName is unchanged; nothing to patch.",
            )
            return
        # Resolve which view file owns this objectName.
        inventory_by_file = {
            fp: scan_view_file(fp) for fp in self._view_files if os.path.isfile(fp)
        }
        owning_file = self._resolve_owner(inventory_by_file, old_name)
        if owning_file is None:
            QMessageBox.warning(
                self, "Not found",
                f"Could not locate setObjectName({old_name!r}) in any view file. "
                "The widget may be created in non-standard code.",
            )
            return

        # Validate uniqueness of the new name (excluding the file that owns it).
        existing = enumerate_all_object_names(self._view_files)
        ok, conflict = validate_object_name_unique(
            new_name, existing, ignore_file=owning_file
        )
        if not ok:
            QMessageBox.warning(
                self, "objectName collision",
                f"{new_name!r} is already defined in:\n  {conflict}\n\n"
                "Pick a different objectName.",
            )
            return

        # Find the AST node and build the edit.
        inv = inventory_by_file[owning_file]
        loc = next(
            (l for l in inv.set_object_names
             if self._arg_value(l.node, 0) == old_name),
            None,
        )
        if loc is None:
            QMessageBox.warning(
                self, "Not found",
                f"setObjectName({old_name!r}) matched file {owning_file} "
                "but no AST node was found at the recorded location.",
            )
            return

        edit = Edit(
            kind="setObjectName",
            file_path=owning_file,
            lineno=loc.lineno,
            old_value=old_name,
            new_value=new_name,
        )
        try:
            patch = build_rename_patch(owning_file, [edit])
        except ValueError as exc:
            QMessageBox.critical(self, "Patch failed", str(exc))
            return

        # Validate the patched source compiles.
        ok, err = validate_patch_compiles(patch.new_source, owning_file)
        if not ok:
            QMessageBox.critical(self, "Patch invalid", err or "syntax error")
            return

        # Ask the user where to save the patch.
        suggested = os.path.join(
            os.path.dirname(owning_file),
            f"hydra_designer_{os.path.basename(owning_file)}.patch",
        )
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save Hydra Designer patch", suggested, "Patch files (*.patch)"
        )
        if not out_path:
            return
        try:
            from swe2d.workbench.devtools.patch_builder import write_patch_file
            write_patch_file(patch, out_path)
        except FileExistsError as exc:
            QMessageBox.warning(self, "Patch exists", str(exc))
            return
        except OSError as exc:
            QMessageBox.critical(self, "Write failed", str(exc))
            return

        QMessageBox.information(
            self, "Patch written",
            f"Wrote {patch.edit_count()} edit(s) to:\n  {out_path}\n\n"
            "Review the diff, then apply with:\n  git apply " + out_path,
        )
        self.accept()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _arg_value(call, idx: int):
        if idx >= len(call.args):
            return None
        arg = call.args[idx]
        if hasattr(arg, "value"):
            return arg.value
        return None

    def _resolve_owner(self, inventory_by_file, object_name):
        for fp, inv in inventory_by_file.items():
            if object_name in inv.all_object_names():
                return fp
        return None


__all__ = ["PropertyEditorDialog"]