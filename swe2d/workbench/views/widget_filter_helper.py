"""Reusable View-layer helper: filterable widget registry.

A small registry pattern that lets any Studio tab view (or dock view)
register Qt widgets that should be hidden/shown when the user types in
a search field. The registry keeps the visible label(s), tooltip,
and objectName strings pre-lowered so ``apply_filter`` does no string
work in the hot path.

Why this exists
---------------

The model tab toolbox has a free-text filter (param_search) and a
"Show advanced parameters" toggle. The existing implementation
filtered only widgets registered via ``_add_param_row``. When the
Storage page was moved from ResultsToolbox into ModelTabView as the
"Output" page, those widgets were added via ``QFormLayout.addRow``
directly and bypassed the registry — the filter therefore saw no
Output widgets, even though they look like every other model
parameter.

This helper gives us a single, testable object that any view can
hold. To make a new widget filterable, the only thing the caller
needs is:

    self._filterable.add(widget, label_text=widget.text(), group=group)

or for widgets without a companion label (checkboxes that carry their
own text):

    self._filterable.add(
        widget,
        label_text=str(widget.text() or ""),
        tooltip=widget.toolTip(),
        group=group,
    )

The same call works for any QWidget subclass — there is no coupling
to ModelTabView, so the helper is reusable in TopologyTabView,
MapTabView, or any future view.

This is a *View-layer* helper because it touches QWidget.setVisible
directly. Per the MVP rules in ``.opencode/rules/MVP_ARCHITECTURE.md``,
Qt UI operations must stay in the View.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from qgis.PyQt import QtWidgets


class FilterableRowRegistry:
    """A registry of Qt widgets that participate in text/advanced filtering.

    Each entry is a tuple of ``(group, label_widget, control_widget, advanced)``
    where:

      * ``group`` is the QGroupBox (or None) whose visibility follows the
        "any visible child?" rule — a group is hidden only when no widget
        in it is currently visible.
      * ``label_widget`` is the QLabel that visually identifies the control
        (or None when the control embeds its own label, e.g. a QCheckBox
        whose text describes it).
      * ``control_widget`` is the QWidget whose visibility is toggled by
        the filter.
      * ``advanced`` flags widgets that should only appear when the
        "show advanced" toggle is on.

    The registry does not emit signals; callers connect their search
    box / advanced toggle to :meth:`apply_filter`.
    """

    def __init__(self) -> None:
        self._rows: List[Tuple[Optional[QtWidgets.QWidget],
                                Optional[QtWidgets.QLabel],
                                QtWidgets.QWidget,
                                bool]] = []

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def add(
        self,
        widget: QtWidgets.QWidget,
        *,
        label_widget: Optional[QtWidgets.QLabel] = None,
        label_text: str = "",
        tooltip: str = "",
        group: Optional[QtWidgets.QGroupBox] = None,
        advanced: bool = False,
    ) -> None:
        """Register a widget for filter participation.

        Parameters
        ----------
        widget:
            The control widget whose visibility is toggled.
        label_widget:
            Optional QLabel companion (used for setVisible on the label).
            Pass None when the widget is self-describing (e.g. a QCheckBox
            whose ``text()`` describes it).
        label_text, tooltip:
            Pre-lowered search strings cached at registration time so
            ``apply_filter`` does no string allocation per keystroke.
        group:
            Optional QGroupBox ancestor — its visibility follows the
            "any visible child?" rule.
        advanced:
            When True, the widget is only shown when the advanced toggle
            is on AND the search matches.
        """
        label = (label_text or "").lower().strip()
        tip = (tooltip or "").lower().strip()
        obj_name = (widget.objectName() or "").lower().strip()
        # Stash the pre-lowered search corpus on the widget itself so
        # ``apply_filter`` can read it without keeping a parallel dict.
        widget.setProperty("filter_search_blob", f"{label}\n{tip}\n{obj_name}")
        widget.setProperty("advanced", bool(advanced))
        self._rows.append((group, label_widget, widget, advanced))

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def apply_filter(
        self,
        text: str,
        *,
        show_advanced: bool,
    ) -> None:
        """Show/hide every registered widget based on ``text`` and ``show_advanced``.

        A widget is visible iff:
          * its pre-lowered search blob contains ``text`` (or ``text`` is empty), AND
          * it is not flagged advanced OR ``show_advanced`` is True.

        Groups are hidden iff no widget inside them is currently visible.

        Each registered widget has its ``filter_visible`` QProperty set
        to the filter's decision so tests can inspect the result without
        depending on Qt's ancestor-visibility chain (in headless test
        runs no page is actually shown, so ``widget.isVisible()`` may
        return False even when the filter says "show").
        """
        needle = (text or "").lower().strip()
        group_visibility: Dict[QtWidgets.QGroupBox, bool] = {}

        for group, label_widget, widget, advanced in self._rows:
            blob = str(widget.property("filter_search_blob") or "")
            matches = (not needle) or (needle in blob)
            visible = matches and (show_advanced or not advanced)

            widget.setProperty("filter_visible", bool(visible))
            widget.setVisible(visible)
            if label_widget is not None:
                label_widget.setProperty("filter_visible", bool(visible))
                label_widget.setVisible(visible)

            if group is not None:
                group_visibility[group] = group_visibility.get(group, False) or visible

        for group, visible in group_visibility.items():
            group.setProperty("filter_visible", bool(visible))
            group.setVisible(visible)

    def filter_visible(self, widget: QtWidgets.QWidget) -> bool:
        """Return the filter's last decision for ``widget`` (True = show).

        Returns True for widgets that were never registered (the filter
        does not touch them).
        """
        val = widget.property("filter_visible")
        if val is None:
            return True
        return bool(val)

    # ------------------------------------------------------------------
    # Introspection (mostly for tests)
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self):
        return iter(self._rows)

    def is_registered(self, widget: QtWidgets.QWidget) -> bool:
        """Return True if ``widget`` has been registered with this registry."""
        for _group, _label, w, _adv in self._rows:
            if w is widget:
                return True
        return False