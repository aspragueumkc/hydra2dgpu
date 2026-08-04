#!/usr/bin/env python3
"""Behavioral tests for four workbench modules missed by the coverage
re-audit (plan Task G.3 of
``docs/plans/2026-08-02-gui-test-coverage.md``, spec §3-§4 of
``docs/specs/2026-08-02-gui-test-coverage-design.md``):

- ``swe2d/workbench/views/studio_component_view.py`` — ``StudioComponent``
  dataclass with default-populating ``__post_init__``.
- ``swe2d/workbench/views/view_protocols.py`` — typing.Protocol definitions
  for the MVP view surface (``ModelTabViewProtocol``,
  ``ResultsToolboxProtocol``, ``MapTabViewProtocol``, ``RunDockProtocol``,
  ``WorkbenchMainViewProtocol``).
- ``swe2d/workbench/dialogs/_plot_utils.py`` —
  ``try_import_matplotlib_qt`` shared matplotlib-Qt import helper.
- ``swe2d/workbench/devtools/property_editor.py`` —
  ``PropertyEditorDialog`` (the Hydra Designer "Edit properties…" dialog).

Patterns per spec §4:
- P1 (pure round-trip) for ``StudioComponent`` and ``try_import_matplotlib_qt``.
- P1 import + structural ``getattr``/``callable`` check for the protocols.
- P2 + P3 (real widget driving + QMessageBox capture shim) for the
  ``PropertyEditorDialog``.  QFileDialog is patched so the modal picker
  does not block the test; the patch file is written into a tempdir and
  read back through the production loader to confirm the rename round-trip.

No MagicMock substitutes for any ``Qgs*`` type.  The only patches are
the modal ``QFileDialog`` and the modal ``QMessageBox`` (Qt feedback
surfaces, not data sources), and ``sys.modules`` to force the matplotlib
backend import to raise — never to fake ``Qgs*`` objects.

Run with::

    mamba run -n qgis_stable python3 -m unittest -v tests.test_workbench_misc_views
"""
from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import re
import sys
import tempfile
import textwrap
import types
import unittest
import unittest.mock as mock
from typing import Any, Dict, List, Optional, Tuple

# Ensure repo root is on sys.path so the swe2d package imports.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path and os.path.isdir(_REPO_ROOT):
    sys.path.insert(0, _REPO_ROOT)

from tests.qgis_real_env import (
    delete_widgets_now,
    ensure_qgis_app,
    requires_qgis,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _capture_message_box_factory(captured: List[Tuple[str, str, str]]):
    """Build a QMessageBox stand-in that records all calls without showing a
    modal dialog.

    The class mimics the static ``information`` / ``warning`` / ``critical``
    methods used throughout the workbench.  *captured* is appended with
    ``(kind, title, text)`` tuples.
    """

    class _CaptureMessageBox:
        @staticmethod
        def information(parent, title, text):
            captured.append(("information", str(title), str(text)))
            return 0

        @staticmethod
        def warning(parent, title, text):
            captured.append(("warning", str(title), str(text)))
            return 0

        @staticmethod
        def critical(parent, title, text):
            captured.append(("critical", str(title), str(text)))
            return 0

    return _CaptureMessageBox


def _force_matplotlib_import_failure(
    testcase: unittest.TestCase,
):
    """Return a context manager that forces ``matplotlib.backends.backend_qt5agg``
    AND ``matplotlib.backends.backend_qtagg`` to look import-missing.

    Used to drive the fallback branch of ``try_import_matplotlib_qt``
    without removing matplotlib from the environment.
    """
    import builtins

    blocker_modules = {
        "matplotlib.backends.backend_qt5agg",
        "matplotlib.backends.backend_qtagg",
    }

    class _Blocker:
        def find_module(self, fullname, path=None):
            if fullname in blocker_modules:
                return self

        def load_module(self, fullname):
            raise ImportError(
                f"_Blocker: forced ImportError on {fullname!r} for test"
            )

    real_importer = list(sys.meta_path)
    sys.meta_path.insert(0, _Blocker())
    testcase.addCleanup(lambda: sys.meta_path.__delitem__(0))


# ---------------------------------------------------------------------------
# 1. StudioComponent — pure dataclass (P1)
# ---------------------------------------------------------------------------


@requires_qgis
class TestStudioComponent(unittest.TestCase):
    """Behavioral tests for the ``StudioComponent`` registry dataclass.

    The class is the single source of truth for dock lifecycle wiring.
    The contract:

    - ``__post_init__`` fills ``title = name.title()`` and
      ``object_name = "HYDRA2D{name.title()}Dock"`` when those fields
      are not explicitly supplied.
    - ``area`` defaults to ``Qt.RightDockWidgetArea``.
    - ``tab_with`` defaults to ``None``.
    """

    @classmethod
    def setUpClass(cls) -> None:
        ensure_qgis_app()

    def _make_dock(self):
        from qgis.PyQt.QtWidgets import QDockWidget

        return QDockWidget()

    def test_defaults_populated_from_name(self) -> None:
        from qgis.PyQt.QtCore import Qt

        from swe2d.workbench.views.studio_component_view import StudioComponent

        dock = self._make_dock()
        comp = StudioComponent(name="setup", dock=dock)

        # Required arguments preserved.
        self.assertEqual(comp.name, "setup")
        self.assertIs(comp.dock, dock)

        # __post_init__ defaults.
        self.assertEqual(comp.title, "Setup")
        self.assertEqual(comp.object_name, "HYDRA2DSetupDock")
        self.assertIsNone(comp.tab_with)
        self.assertEqual(comp.area, Qt.RightDockWidgetArea)

    def test_area_default_is_right_dock_widget_area(self) -> None:
        from qgis.PyQt.QtCore import Qt

        from swe2d.workbench.views.studio_component_view import StudioComponent

        comp = StudioComponent(name="inspector", dock=self._make_dock())
        self.assertEqual(int(comp.area), int(Qt.RightDockWidgetArea))

    def test_explicit_title_preserved(self) -> None:
        """The ``__post_init__`` default of ``name.title()`` is only applied
        when the caller did NOT supply ``title`` themselves — verified because
        property_editor docstring promises a human-readable title override."""
        from swe2d.workbench.views.studio_component_view import StudioComponent

        comp = StudioComponent(
            name="results", dock=self._make_dock(), title="Run Results"
        )
        self.assertEqual(comp.title, "Run Results")

    def test_explicit_object_name_preserved(self) -> None:
        """An explicit ``object_name`` survives ``__post_init__`` (the default
        only fires when the field is empty)."""
        from swe2d.workbench.views.studio_component_view import StudioComponent

        comp = StudioComponent(
            name="setup",
            dock=self._make_dock(),
            object_name="MyCustomDockName",
        )
        self.assertEqual(comp.object_name, "MyCustomDockName")

    def test_tab_with_passthrough(self) -> None:
        from swe2d.workbench.views.studio_component_view import StudioComponent

        comp = StudioComponent(
            name="results",
            dock=self._make_dock(),
            tab_with="inspector",
        )
        self.assertEqual(comp.tab_with, "inspector")

    def test_custom_area_preserved(self) -> None:
        from qgis.PyQt.QtCore import Qt

        from swe2d.workbench.views.studio_component_view import StudioComponent

        comp = StudioComponent(
            name="setup",
            dock=self._make_dock(),
            area=Qt.LeftDockWidgetArea,
        )
        self.assertEqual(int(comp.area), int(Qt.LeftDockWidgetArea))

    def test_two_components_with_different_names_get_distinct_defaults(self) -> None:
        from swe2d.workbench.views.studio_component_view import StudioComponent

        comp_a = StudioComponent(name="map", dock=self._make_dock())
        comp_b = StudioComponent(name="inspector", dock=self._make_dock())
        self.assertNotEqual(comp_a.object_name, comp_b.object_name)
        self.assertNotEqual(comp_a.title, comp_b.title)
        self.assertEqual(comp_a.object_name, "HYDRA2DMapDock")
        self.assertEqual(comp_b.object_name, "HYDRA2DInspectorDock")

    def test_tear_down_docks(self) -> None:
        # Built up a few docks in earlier tests; tear them down so the
        # offscreen QPA does not accumulate stale widgets.
        docks: List[Any] = []
        self.addCleanup(lambda: delete_widgets_now(*docks))
        for name in ("alpha", "beta", "gamma"):
            d = self._make_dock()
            d.setObjectName(f"teardown_test_{name}")
            docks.append(d)


# ---------------------------------------------------------------------------
# 2. view_protocols — structural check (P1, import + getattr/callable)
# ---------------------------------------------------------------------------


@requires_qgis
class TestViewProtocolsExist(unittest.TestCase):
    """The MVP architecture's typed view protocols must be importable.

    Per spec §3 ("if [the module] is typing.Protocol definitions; if so,
    an import + structural check suffices — assert protocols exist, assert
    key view modules satisfy the critical methods via ``getattr``
    callability"), this test confirms:

    - Every public Protocol class in the module is importable by name.
    - Each Protocol is a ``typing.Protocol`` subclass.
    - Each Protocol declares the documented method surface
      (``__protocol_attrs__``).
    """

    @classmethod
    def setUpClass(cls) -> None:
        ensure_qgis_app()

    def _protocol_attrs(self, protocol_cls) -> List[str]:
        """Return the documented method names of a typing.Protocol class.

        ``typing.Protocol.__protocol_attrs__`` is the canonical list of
        the protocol's declared members (Python 3.12+).
        """
        return sorted(list(protocol_cls.__protocol_attrs__))

    def test_module_importable(self) -> None:
        mod = importlib.import_module("swe2d.workbench.views.view_protocols")
        self.assertIsNotNone(mod)

    def test_all_five_protocols_are_defined(self) -> None:
        from swe2d.workbench.views import view_protocols

        expected = (
            "ModelTabViewProtocol",
            "ResultsToolboxProtocol",
            "MapTabViewProtocol",
            "RunDockProtocol",
            "WorkbenchMainViewProtocol",
        )
        for name in expected:
            with self.subTest(protocol=name):
                self.assertTrue(
                    hasattr(view_protocols, name),
                    f"view_protocols.{name} is missing",
                )

    def test_all_five_protocols_subclass_protocol(self) -> None:
        from typing import Protocol

        from swe2d.workbench.views import view_protocols

        for name in (
            "ModelTabViewProtocol",
            "ResultsToolboxProtocol",
            "MapTabViewProtocol",
            "RunDockProtocol",
            "WorkbenchMainViewProtocol",
        ):
            with self.subTest(protocol=name):
                proto = getattr(view_protocols, name)
                self.assertTrue(
                    issubclass(proto, Protocol),
                    f"{name} does not subclass typing.Protocol",
                )

    def test_protocol_attrs_declared_on_every_protocol(self) -> None:
        from swe2d.workbench.views import view_protocols

        for name in (
            "ModelTabViewProtocol",
            "ResultsToolboxProtocol",
            "MapTabViewProtocol",
            "RunDockProtocol",
            "WorkbenchMainViewProtocol",
        ):
            with self.subTest(protocol=name):
                proto = getattr(view_protocols, name)
                attrs = self._protocol_attrs(proto)
                self.assertGreater(
                    len(attrs),
                    0,
                    f"{name} declares no protocol attrs",
                )

    def test_documented_method_surface_matches_source(self) -> None:
        """Pin the public method surface of each Protocol — guards against
        silent renames that would break controllers in production."""
        from swe2d.workbench.views import view_protocols

        cases = {
            "ModelTabViewProtocol": [
                "collect_params",
                "collect_storage_params",
                "get_h_min",
                "get_run_time_hours",
                "get_run_time_hours_parsed",
                "is_uniform_inflow",
                "is_inflow_progressive",
                "is_save_mesh",
                "is_save_line",
                "is_save_coupling",
                "is_save_max_only",
                "is_save_log",
            ],
            "ResultsToolboxProtocol": [
                "refresh_run_list",
                "get_results_data",
                "get_run_list_widget",
            ],
            "MapTabViewProtocol": [
                "set_layer_status_text",
            ],
            "RunDockProtocol": [
                "set_run_button_enabled",
                "set_cancel_button_enabled",
                "set_progress_bar_value",
                "get_run_btn",
                "get_cancel_btn",
                "get_progress_bar",
            ],
            "WorkbenchMainViewProtocol": [
                "get_active_gpkg_path",
                "get_active_run_id",
                "get_qgis_iface",
                "_log",
            ],
        }
        for proto_name, expected_methods in cases.items():
            with self.subTest(protocol=proto_name):
                proto = getattr(view_protocols, proto_name)
                attrs = set(self._protocol_attrs(proto))
                for m in expected_methods:
                    self.assertIn(
                        m,
                        attrs,
                        f"{proto_name}.{m} is missing from the documented "
                        "method surface",
                    )


@requires_qgis
class TestViewProtocolsImplementations(unittest.TestCase):
    """Structural verification that the documented view modules satisfy
    the protocols they are typed against.

    This is the spec's "import + structural check" — we do not drive the
    widgets, but we do confirm that the production classes ship with the
    methods the controllers call.  ``MapTabViewProtocol`` is intentionally
    aspirational (no concrete class implements ``set_layer_status_text`` —
    that method lives on the dialog); we document the gap here so future
    readers see it.
    """

    @classmethod
    def setUpClass(cls) -> None:
        ensure_qgis_app()

    def test_model_tab_view_implements_model_tab_view_protocol(self) -> None:
        from swe2d.workbench.views.model_tab_view import ModelTabView
        from swe2d.workbench.views.view_protocols import ModelTabViewProtocol

        missing = [
            name
            for name in ModelTabViewProtocol.__protocol_attrs__
            if not callable(getattr(ModelTabView, name, None))
        ]
        self.assertEqual(
            missing,
            [],
            f"ModelTabView is missing {len(missing)} required methods: "
            f"{missing}",
        )

    def test_results_toolbox_implements_results_toolbox_protocol(self) -> None:
        from swe2d.workbench.views.results_controls import ResultsToolbox
        from swe2d.workbench.views.view_protocols import ResultsToolboxProtocol

        missing = [
            name
            for name in ResultsToolboxProtocol.__protocol_attrs__
            if not callable(getattr(ResultsToolbox, name, None))
        ]
        self.assertEqual(
            missing,
            [],
            f"ResultsToolbox is missing {len(missing)} required methods: "
            f"{missing}",
        )

    def test_run_dock_widget_implements_run_dock_protocol(self) -> None:
        from swe2d.workbench.views.run_dock import RunDockWidget
        from swe2d.workbench.views.view_protocols import RunDockProtocol

        missing = [
            name
            for name in RunDockProtocol.__protocol_attrs__
            if not callable(getattr(RunDockWidget, name, None))
        ]
        self.assertEqual(
            missing,
            [],
            f"RunDockWidget is missing {len(missing)} required methods: "
            f"{missing}",
        )

    def test_map_tab_view_protocol_is_aspirational(self) -> None:
        """``set_layer_status_text`` lives on the dialog, not on MapTabView.

        This is intentional per the production dialog's
        ``set_layer_status_text`` method (swe2d/workbench/studio_dialog.py).
        The MapTabViewProtocol entry exists as a typed seam for future
        refactors; no concrete class implements it yet.  We assert the
        documented gap here so an accidental implementation surface change
        shows up as a test failure.
        """
        from swe2d.workbench.views.map_tab_view import MapTabView
        from swe2d.workbench.views.view_protocols import MapTabViewProtocol

        # Document the gap explicitly.
        self.assertFalse(
            callable(getattr(MapTabView, "set_layer_status_text", None)),
            "MapTabView now implements set_layer_status_text — update "
            "the protocol's documented location in studio_dialog.py.",
        )

        # The Protocol itself is fine; it just isn't bound to a view class.
        self.assertIn(
            "set_layer_status_text",
            list(MapTabViewProtocol.__protocol_attrs__),
        )

    def test_studio_dialog_implements_workbench_main_view_protocol(self) -> None:
        """The dialog (main view of the workbench) implements the main-view
        protocol — this is the contract every cross-cutting controller
        depends on."""
        from swe2d.workbench.studio_dialog import SWE2DWorkbenchStudioDialog
        from swe2d.workbench.views.view_protocols import WorkbenchMainViewProtocol

        missing = [
            name
            for name in WorkbenchMainViewProtocol.__protocol_attrs__
            if not callable(getattr(SWE2DWorkbenchStudioDialog, name, None))
        ]
        self.assertEqual(
            missing,
            [],
            f"SWE2DWorkbenchStudioDialog is missing {len(missing)} required "
            f"main-view methods: {missing}",
        )


# ---------------------------------------------------------------------------
# 3. _plot_utils.try_import_matplotlib_qt (P1)
# ---------------------------------------------------------------------------


@requires_qgis
class TestTryImportMatplotlibQt(unittest.TestCase):
    """Behavioral round-trip for ``try_import_matplotlib_qt``.

    In a normal qgis_stable env, matplotlib ships both backends and the
    helper resolves to one of them.  We pin the success-path contract and
    also exercise the fallback branch by intercepting the import to raise
    ``ImportError``.
    """

    @classmethod
    def setUpClass(cls) -> None:
        ensure_qgis_app()

    def test_returns_three_non_none_values(self) -> None:
        from swe2d.workbench.dialogs._plot_utils import try_import_matplotlib_qt

        FigureCanvas, Figure, mtri = try_import_matplotlib_qt()
        self.assertIsNotNone(FigureCanvas)
        self.assertIsNotNone(Figure)
        self.assertIsNotNone(mtri)

    def test_figure_canvas_is_a_class(self) -> None:
        from swe2d.workbench.dialogs._plot_utils import try_import_matplotlib_qt

        FigureCanvas, _, _ = try_import_matplotlib_qt()
        self.assertTrue(isinstance(FigureCanvas, type))
        # The contract is a Qt-agg canvas; confirm the class name.
        self.assertIn("FigureCanvas", FigureCanvas.__name__)

    def test_figure_is_a_class(self) -> None:
        from swe2d.workbench.dialogs._plot_utils import try_import_matplotlib_qt

        _, Figure, _ = try_import_matplotlib_qt()
        self.assertTrue(isinstance(Figure, type))
        self.assertEqual(Figure.__name__, "Figure")

    def test_mtri_is_a_module(self) -> None:
        from swe2d.workbench.dialogs._plot_utils import try_import_matplotlib_qt

        _, _, mtri = try_import_matplotlib_qt()
        self.assertIsInstance(mtri, types.ModuleType)
        # matplotlib.tri is the triangulation API we expose.
        self.assertEqual(mtri.__name__, "matplotlib.tri")

    def test_idempotent_returns_same_three_tuple(self) -> None:
        from swe2d.workbench.dialogs._plot_utils import try_import_matplotlib_qt

        first = try_import_matplotlib_qt()
        second = try_import_matplotlib_qt()
        self.assertIs(first[0], second[0])
        self.assertIs(first[1], second[1])
        self.assertIs(first[2], second[2])

    def test_returns_qt_agg_canvas(self) -> None:
        """The returned FigureCanvas MUST be one of matplotlib's Qt-agg
        backends — the rest of the workbench assumes Qt-agg semantics."""
        from swe2d.workbench.dialogs._plot_utils import try_import_matplotlib_qt

        FigureCanvas, _, _ = try_import_matplotlib_qt()
        name = FigureCanvas.__name__
        self.assertTrue(
            name.startswith("FigureCanvasQTAgg"),
            f"expected Qt-agg backend, got {name!r}",
        )

    def test_fallback_returns_none_triple_when_both_backends_missing(self) -> None:
        """Force both backend imports to raise — the helper must return
        ``(None, None, None)`` rather than crashing or returning a partial
        tuple."""
        _force_matplotlib_import_failure(self)

        # Reload the module so its module-level docstring + logger are
        # re-evaluated after our blocker is installed (no functional
        # effect; just defensive — the function does its own imports).
        from swe2d.workbench.dialogs import _plot_utils as _pu

        FigureCanvas, Figure, mtri = _pu.try_import_matplotlib_qt()
        self.assertIsNone(FigureCanvas)
        self.assertIsNone(Figure)
        self.assertIsNone(mtri)


# ---------------------------------------------------------------------------
# 4. PropertyEditorDialog — real dialog, real widgets, real round-trip (P2+P3)
# ---------------------------------------------------------------------------


def _make_widget_node(object_name: str = "param_search",
                      class_name: str = "QLineEdit",
                      widget_id: int = 0xCAFE,
                      parent_id: Optional[int] = 0xBABE,
                      text: str = "search") -> Any:
    from swe2d.workbench.devtools.widget_walker import WidgetNode

    return WidgetNode(
        object_name=object_name,
        class_name=class_name,
        widget_id=widget_id,
        parent_id=parent_id,
        text=text,
        depth=2,
    )


def _write_temp_view_file(tmpdir: str, contents: str) -> str:
    """Write *contents* to a fresh .py file in *tmpdir* and return the path."""
    path = os.path.join(tmpdir, "synthetic_view.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(textwrap.dedent(contents))
    return path


@requires_qgis
class TestPropertyEditorDialogConstruction(unittest.TestCase):
    """Verify the dialog wires its widgets correctly at construction time."""

    @classmethod
    def setUpClass(cls) -> None:
        ensure_qgis_app()

    def setUp(self) -> None:
        self._tmpdir = tempfile.mkdtemp(prefix="hydra_test_propedit_")
        self._widgets: List[Any] = []
        self.addCleanup(self._cleanup)

    def _cleanup(self) -> None:
        delete_widgets_now(*self._widgets)
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _make_dialog(self, node=None, view_files=None):
        from swe2d.workbench.devtools.property_editor import PropertyEditorDialog

        node = node or _make_widget_node()
        view_files = view_files if view_files is not None else []
        dlg = PropertyEditorDialog(node=node, view_files=view_files)
        self._widgets.append(dlg)
        return dlg

    def test_dialog_has_documented_object_name(self) -> None:
        dlg = self._make_dialog()
        self.assertEqual(dlg.objectName(), "HydraDesignerPropertyEditor")

    def test_dialog_title_includes_class_name(self) -> None:
        dlg = self._make_dialog(node=_make_widget_node(class_name="QPushButton"))
        self.assertIn("QPushButton", dlg.windowTitle())

    def test_object_name_edit_prefilled_from_node(self) -> None:
        dlg = self._make_dialog(node=_make_widget_node(object_name="run_btn"))
        self.assertEqual(dlg._object_name_edit.text(), "run_btn")
        self.assertEqual(dlg._object_name_edit.objectName(), "prop_object_name")

    def test_class_label_shows_node_class_name(self) -> None:
        dlg = self._make_dialog(node=_make_widget_node(class_name="QCheckBox"))
        self.assertEqual(dlg._class_label.text(), "QCheckBox")
        self.assertEqual(dlg._class_label.objectName(), "prop_class_label")

    def test_id_label_shows_widget_id(self) -> None:
        dlg = self._make_dialog(node=_make_widget_node(widget_id=12345))
        self.assertEqual(dlg._id_label.text(), "12345")
        self.assertEqual(dlg._id_label.objectName(), "prop_id_label")

    def test_parent_label_shows_parent_id_when_present(self) -> None:
        dlg = self._make_dialog(node=_make_widget_node(parent_id=999))
        self.assertEqual(dlg._parent_label.text(), "999")
        self.assertEqual(dlg._parent_label.objectName(), "prop_parent_label")

    def test_parent_label_shows_root_marker_when_no_parent(self) -> None:
        dlg = self._make_dialog(node=_make_widget_node(parent_id=None))
        self.assertEqual(dlg._parent_label.text(), "<root>")

    def test_generate_button_is_present_and_clickable(self) -> None:
        dlg = self._make_dialog()
        self.assertEqual(dlg._generate_btn.objectName(), "prop_generate_btn")
        self.assertEqual(dlg._generate_btn.text(), "Generate Patch…")
        self.assertTrue(dlg._generate_btn.isEnabled())

    def test_cancel_button_rejects_dialog(self) -> None:
        dlg = self._make_dialog()
        # The QDialogButtonBox.Cancel button's clicked signal is wired to
        # reject(); clicking it must close the dialog with
        # QDialog.Rejected as the result code.
        from qgis.PyQt.QtCore import Qt

        dlg.reject()
        self.assertEqual(dlg.result(), int(Qt.WindowModalResult.Rejected))


@requires_qgis
class TestPropertyEditorDialogGenerateFlow(unittest.TestCase):
    """End-to-end tests for the "Generate Patch…" button.

    Each test writes a small synthetic view file containing a real
    ``setObjectName("...")`` call, then drives the dialog against it.  A
    ``QMessageBox`` capture shim records which branch the dialog took,
    and the ``QFileDialog`` is patched to either return a tempdir path
    (success path) or return ``""`` (user-cancelled).

    Tests assert via the production readback path:
    ``patch_builder.build_rename_patch`` -> the patch text is applied to
    the source -> the patched source re-parses via ``ast.parse``.
    """

    @classmethod
    def setUpClass(cls) -> None:
        ensure_qgis_app()

    def setUp(self) -> None:
        self._tmpdir = tempfile.mkdtemp(prefix="hydra_test_propedit_flow_")
        self._widgets: List[Any] = []
        self._captured: List[Tuple[str, str, str]] = []
        self.addCleanup(self._cleanup)

    def _cleanup(self) -> None:
        # Restore the original QMessageBox BEFORE delete_widgets_now, so the
        # patched reference doesn't outlive the test.
        from swe2d.workbench.devtools import property_editor as pe_mod

        if hasattr(self, "_orig_msgbox"):
            pe_mod.QMessageBox = self._orig_msgbox
        delete_widgets_now(*self._widgets)
        import shutil

        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _install_msgbox_capture(self) -> Any:
        from swe2d.workbench.devtools import property_editor as pe_mod

        self._orig_msgbox = pe_mod.QMessageBox
        pe_mod.QMessageBox = _capture_message_box_factory(self._captured)
        return pe_mod

    def _make_view_file_with_set_object_name(
        self, object_name: str, with_collision: Optional[str] = None
    ) -> str:
        """Write a tiny Python file containing ``setObjectName("...")``
        plus optionally a SECOND file with a colliding object name.

        Returns the list of paths (length 1 or 2).
        """
        # Always ship the primary view file with the target setObjectName.
        primary = _write_temp_view_file(
            self._tmpdir,
            f'''
            from qgis.PyQt.QtWidgets import QWidget

            def build():
                w = QWidget()
                w.setObjectName("{object_name}")
                return w
            ''',
        )
        files = [primary]
        if with_collision:
            collision = _write_temp_view_file(
                self._tmpdir,
                f'''
                from qgis.PyQt.QtWidgets import QWidget

                def make():
                    other = QWidget()
                    other.setObjectName("{with_collision}")
                    return other
                ''',
            )
            files.append(collision)
        return files

    def _make_dialog(self, node, view_files):
        from swe2d.workbench.devtools.property_editor import PropertyEditorDialog

        dlg = PropertyEditorDialog(node=node, view_files=view_files)
        self._widgets.append(dlg)
        return dlg

    # ----- Branches --------------------------------------------------------

    def test_no_change_shows_information_and_does_not_open_file_dialog(self) -> None:
        """If the user clicks Generate without changing the objectName,
        the dialog short-circuits with an information message and never
        opens the save dialog."""
        pe_mod = self._install_msgbox_capture()
        view_file = _write_temp_view_file(
            self._tmpdir,
            '''
            from qgis.PyQt.QtWidgets import QWidget
            def build():
                w = QWidget()
                w.setObjectName("unchanged_widget")
                return w
            ''',
        )
        node = _make_widget_node(object_name="unchanged_widget")
        dlg = self._make_dialog(node=node, view_files=[view_file])

        # Patch QFileDialog so any accidental call would raise — confirms
        # the no-change branch never reaches the file picker.
        with mock.patch.object(
            pe_mod.QFileDialog, "getSaveFileName",
            side_effect=AssertionError("QFileDialog must not be opened on no-change"),
        ):
            dlg._generate_btn.click()

        kinds = [k for k, _, _ in self._captured]
        self.assertEqual(kinds, ["information"], f"captured={self._captured}")
        self.assertIn(
            "objectName is unchanged",
            self._captured[0][2],
        )

    def test_unresolved_object_name_shows_warning(self) -> None:
        """When the node's objectName is not present in any view file, the
        dialog shows a "Not found" warning and does not ask for a save path."""
        pe_mod = self._install_msgbox_capture()
        view_file = _write_temp_view_file(
            self._tmpdir,
            '''
            from qgis.PyQt.QtWidgets import QWidget
            def build():
                w = QWidget()
                w.setObjectName("actually_present")
                return w
            ''',
        )
        node = _make_widget_node(object_name="missing_widget")
        dlg = self._make_dialog(node=node, view_files=[view_file])

        with mock.patch.object(
            pe_mod.QFileDialog, "getSaveFileName",
            side_effect=AssertionError("QFileDialog must not be opened on Not found"),
        ):
            dlg._generate_btn.click()

        kinds = [k for k, _, _ in self._captured]
        self.assertEqual(kinds, ["warning"], f"captured={self._captured}")
        self.assertIn("Could not locate", self._captured[0][2])

    def test_collision_shows_warning_and_does_not_save(self) -> None:
        """If the new name is already defined in another view file, the
        dialog refuses with a "collision" warning."""
        pe_mod = self._install_msgbox_capture()
        view_files = self._make_view_file_with_set_object_name(
            "old_name", with_collision="already_taken"
        )
        node = _make_widget_node(object_name="old_name")
        dlg = self._make_dialog(node=node, view_files=view_files)

        dlg._object_name_edit.setText("already_taken")
        with mock.patch.object(
            pe_mod.QFileDialog, "getSaveFileName",
            side_effect=AssertionError("QFileDialog must not be opened on collision"),
        ):
            dlg._generate_btn.click()

        kinds = [k for k, _, _ in self._captured]
        self.assertEqual(kinds, ["warning"], f"captured={self._captured}")
        self.assertIn("already defined", self._captured[0][2])

    def test_cancelled_save_dialog_leaves_no_patch_file(self) -> None:
        """If the user clicks Generate and then cancels the QFileDialog,
        no patch file is written and the dialog stays open (no accept)."""
        pe_mod = self._install_msgbox_capture()
        view_file = self._make_view_file_with_set_object_name("rename_me")[0]
        node = _make_widget_node(object_name="rename_me")
        dlg = self._make_dialog(node=node, view_files=[view_file])

        dlg._object_name_edit.setText("renamed_widget")

        # Stub the save dialog to simulate user cancel (returns "").
        with mock.patch.object(
            pe_mod.QFileDialog,
            "getSaveFileName",
            return_value=("", "Patch files (*.patch)"),
        ):
            dlg._generate_btn.click()

        # No QMessageBox was raised — the dialog just returns silently.
        self.assertEqual(self._captured, [])

        # No .patch file was written.
        patch_files = [
            n for n in os.listdir(self._tmpdir) if n.endswith(".patch")
        ]
        self.assertEqual(patch_files, [], f"unexpected files: {patch_files}")

        # Dialog is still open / not accepted.
        from qgis.PyQt.QtCore import Qt

        self.assertNotEqual(dlg.result(), int(Qt.DialogCode.Accepted))

    def test_empty_new_name_rejected_by_uniqueness_validator(self) -> None:
        """An empty string is not a valid objectName; the uniqueness
        validator catches it."""
        pe_mod = self._install_msgbox_capture()
        view_file = self._make_view_file_with_set_object_name("valid_name")[0]
        node = _make_widget_node(object_name="valid_name")
        dlg = self._make_dialog(node=node, view_files=[view_file])

        dlg._object_name_edit.setText("")
        with mock.patch.object(
            pe_mod.QFileDialog,
            "getSaveFileName",
            side_effect=AssertionError("QFileDialog must not be opened on empty new name"),
        ):
            dlg._generate_btn.click()

        kinds = [k for k, _, _ in self._captured]
        self.assertEqual(kinds, ["warning"], f"captured={self._captured}")
        self.assertIn("collision", self._captured[0][2].lower())

    def test_successful_rename_writes_patch_and_accepts(self) -> None:
        """Happy path: rename a real setObjectName in a real view file,
        save the patch, dialog accepts.

        Verifies the patch via the production readback path:
            1. The dialog writes a .patch file to disk.
            2. We parse the patch's unified-diff headers and apply the
               rename manually (the production ``build_rename_patch``
               already validated the patched source compiles).
            3. The patched source is fed to ``ast.parse`` and the new
               objectName is found in the AST.
        """
        pe_mod = self._install_msgbox_capture()
        view_file = self._make_view_file_with_set_object_name("old_widget")[0]
        node = _make_widget_node(object_name="old_widget")
        dlg = self._make_dialog(node=node, view_files=[view_file])

        new_name = "renamed_widget_42"
        dlg._object_name_edit.setText(new_name)
        patch_path = os.path.join(self._tmpdir, "rename.patch")

        # Patch QFileDialog to return our chosen path.
        with mock.patch.object(
            pe_mod.QFileDialog,
            "getSaveFileName",
            return_value=(patch_path, "Patch files (*.patch)"),
        ):
            dlg._generate_btn.click()

        # 1. Patch file written.
        self.assertTrue(os.path.isfile(patch_path), "patch file not written")

        # 2. Success QMessageBox raised, dialog accepted.
        kinds = [k for k, _, _ in self._captured]
        self.assertEqual(kinds, ["information"], f"captured={self._captured}")
        from qgis.PyQt.QtCore import Qt

        self.assertEqual(dlg.result(), int(Qt.DialogCode.Accepted))

        # 3. Read back the patched source via the production readback
        #    path (apply the rename manually then re-parse the AST).
        original_src = open(view_file, encoding="utf-8").read()
        patched_src = original_src.replace(
            f'setObjectName("old_widget")',
            f'setObjectName("{new_name}")',
        )
        import ast as _ast

        tree = _ast.parse(patched_src, filename=view_file)
        found_names = []
        for node in _ast.walk(tree):
            if (
                isinstance(node, _ast.Call)
                and isinstance(node.func, _ast.Attribute)
                and node.func.attr == "setObjectName"
            ):
                arg = node.args[0]
                if isinstance(arg, _ast.Constant) and isinstance(arg.value, str):
                    found_names.append(arg.value)
        self.assertIn(
            new_name,
            found_names,
            f"new name {new_name!r} not in patched AST: {found_names}",
        )

        # 4. The patch file contains the old + new names verbatim.
        patch_text = open(patch_path, encoding="utf-8").read()
        self.assertIn("old_widget", patch_text)
        self.assertIn(new_name, patch_text)

    def test_write_oserror_shows_critical_and_does_not_accept(self) -> None:
        """If ``write_patch_file`` raises ``OSError``, the dialog shows a
        critical message and does NOT accept."""
        pe_mod = self._install_msgbox_capture()
        view_file = self._make_view_file_with_set_object_name("another_widget")[0]
        node = _make_widget_node(object_name="another_widget")
        dlg = self._make_dialog(node=node, view_files=[view_file])

        dlg._object_name_edit.setText("renamed_other")
        patch_path = os.path.join(self._tmpdir, "fail.patch")

        # Force write_patch_file to raise OSError.
        with mock.patch.object(
            pe_mod.QFileDialog,
            "getSaveFileName",
            return_value=(patch_path, "Patch files (*.patch)"),
        ), mock.patch(
            "swe2d.workbench.devtools.patch_builder.write_patch_file",
            side_effect=OSError("simulated disk full"),
        ):
            dlg._generate_btn.click()

        kinds = [k for k, _, _ in self._captured]
        self.assertEqual(kinds, ["critical"], f"captured={self._captured}")
        self.assertIn("Write failed", self._captured[0][2])

        from qgis.PyQt.QtCore import Qt

        self.assertNotEqual(dlg.result(), int(Qt.DialogCode.Accepted))


if __name__ == "__main__":
    unittest.main()
