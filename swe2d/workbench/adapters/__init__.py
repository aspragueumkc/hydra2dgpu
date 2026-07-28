"""Workbench adapters — bridges GUI widget state to the canonical builder.

Phase 1.B: the GUI adapter (:mod:`run_context_adapter`) translates live
widget values into a ``swe2d-run/2`` spec dict and delegates to the canonical
:func:`swe2d.runtime.run_context_builder.build_run_context`.
"""
from __future__ import annotations
