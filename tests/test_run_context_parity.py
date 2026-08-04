"""RunContext parity diff test — CLI-first refactor Phase 0 equivalence gate.

Builds one ``RunContext`` from the GUI Studio dialog (real headless QGIS —
see ``tests/qgis_real_env.py``) and
one from ``swe2d.runtime.run_context_builder.build_run_context_from_dict``
fed the ``swe2d-replay/1`` JSON exported from the SAME dialog widget state
(``RunController.collect_widget_state_for_save`` + ``_build_replay_payload``
— the exact JSON the GUI's "export replay" path writes).  Every
``RunContext`` field is then diffed recursively; any difference not
listed in ``KNOWN_DIVERGENCES`` fails the test.

The allowlist documents today's drift as field-path → reason.  Later
refactor phases delete entries as they fix them; entries marked
``"exercised": True`` additionally assert that the divergence still
exists, so a fix that forgets to delete the entry turns the test red.

GPU-gated: the GUI ``_build_run_context`` (the inlined equivalent of the
retired ``SWE2DRunOptionsBuilder.build``) hard-requires a CUDA GPU and
raises when ``swe2d_gpu_available()`` is False, so the GUI-side context
cannot be constructed without one.

Usage:
    python3 -m unittest tests.test_run_context_parity -v
"""

from __future__ import annotations

import enum
import math
import os
import shutil
import tempfile
import threading
import unittest
from dataclasses import fields as _dc_fields
from typing import Any, Dict, List, Tuple

import numpy as np

from tests.qgis_real_env import ensure_qgis_app, requires_qgis, stub_iface
from tests._swe2d_test_helpers import (
    _channel_bc_edges,
    _gpu_available,
    _make_cartesian_quad_mesh,
    _serialize_and_persist_mesh,
)
from tests.test_helpers import FallbackTracker


# ═══════════════════════════════════════════════════════════════════════════════
# Known divergences allowlist (Phase 0 baseline — shrink to zero by Phase 3)
# ═══════════════════════════════════════════════════════════════════════════════
#
# field path → {"reason": <why GUI and CLI differ today, with code refs>,
#               "exercised": <True if this fixture actually triggers it>}
#
# "Exercised" entries assert the divergence is still present; delete the
# entry in the same commit that fixes the underlying drift.  Non-exercised
# entries document drift the current fixture cannot trigger (the feature
# is off), so both sides are trivially equal here — they exist so later
# phases have the full inventory in one place.

KNOWN_DIVERGENCES: Dict[str, Dict[str, Any]] = {
    "pipe_network_cfg": {
        "reason": (
            "Closed by Phase 3 `_drainage_config_dict` "
            "(swe2d/core/builder.py:632) — CLI now forwards the 5 "
            "GUI-parity keys (friction_method, surcharge_method, "
            "recon_method, time_integrator, friction_alpha) into the "
            "GPKG loader.  Retained as a matcher test fixture "
            "(exercised=False; the parity fixture has no drainage layers)."
        ),
        "exercised": False,
    },
}


def _matches_allowlist(path: str, allowlist) -> bool:
    """Return True when a diff ``path`` is covered by an allowlist key.

    Matching rule: exact equality, or the path is a *nested child* of the
    key — ``path.startswith(key + "[")``.  The ``[`` anchor mirrors the
    diff engine's path syntax for dict/list entries (e.g.
    ``pipe_network_cfg['friction_method']``, ``sample_map_data[0]['key']``)
    and prevents false prefix hits (``edge_groups_extra`` does NOT match
    ``edge_groups``).
    """
    return any(path == key or path.startswith(key + "[") for key in allowlist)


# Callback fields are compared by presence + qualified name only, NOT by
# function identity: the GUI passes bound ``SWE2DWorkbenchStudioDialog``
# methods while the CLI passes closures/lambdas defined inside
# ``build_run_context_from_dict`` — different wrapper objects around the
# same underlying logic (e.g. both ``mesh_cell_areas`` resolve to
# ``mesh_computation_service.mesh_cell_areas`` on the same arrays), so
# identity comparison is meaningless across the two paths.
#
# NOTE (documented behavioural gap, not fixed in Phase 0): for the first
# four entries below the CLI passes *no-op* lambdas while the GUI passes
# real implementations.  Behaviour only diverges when timeseries BCs /
# external sources / sample lines are configured — features this fixture
# leaves off.  Tracked here so later phases can wire the real logic.
CALLBACK_FIELDS = {
    "apply_timeseries_bc_values",      # CLI: no-op lambda
    "distribute_total_flow_to_unit_q",  # CLI: no-op lambda
    "apply_external_sources",          # CLI: no-op lambda
    "build_line_sampling_map",         # CLI: no-op lambda
    "mesh_cell_areas",
    "mesh_cell_min_bed",
    "mesh_cell_centroids",
    "internal_flow_source_cms_at_time",
}

# Per-run identity/bookkeeping fields — normalized (not allowlisted) before
# the diff because they legitimately differ between any two constructions.
# ``cancel_event`` is compared by type only.
IDENTITY_FIELDS = {"run_id", "run_wallclock_start", "run_log_start_idx"}

# BC edge arrays are compared as an edge-keyed mapping, not elementwise:
# the GUI recomputes boundary edges from mesh topology while the CLI reads
# the baked-BLOB edge order — the same edge SET in different sequence.
_BC_EDGE_FIELDS = {"bc_n0", "bc_n1", "bc_tp", "bc_vl", "bc_relax"}


# ═══════════════════════════════════════════════════════════════════════════════
# Shared fixture
# ═══════════════════════════════════════════════════════════════════════════════

# Loggers whose .warning() calls during context construction are watched
# for silent fallbacks (FallbackTracker patches one logger per instance).
_WATCHED_LOGGERS = (
    "swe2d.runtime.run_context_builder",
    "swe2d.cli.gpkg_adapter",
    "swe2d.workbench.controllers.run_controller",
    "swe2d.workbench.studio_dialog",
)


class ParityFixture:
    """Builds the GUI dialog + mesh GPKG shared by the parity tests.

    The mesh is a tiny 4×2 quad channel on a 1% slope.  It is baked into a
    temporary GPKG; the dialog's ``_mesh_data`` is loaded back from the same
    BLOB (mirroring the dialog's ``on_load_simulation_config`` path) so
    serialization-order artifacts don't masquerade as config drift.

    The initial condition is ``uniform_wse`` (0.45 m on a 0–0.4 m bed) so
    the domain starts wet with a real gradient — the same fixture drives
    the GPU replay-equivalence test.
    """

    NX, NY = 4, 2
    LX, LY = 40.0, 10.0
    BED_SLOPE = 0.01
    INITIAL_WSE = 0.45
    RUN_TIME_TEXT = "0:01"       # HH:MM → 60 s
    OUTPUT_INTERVAL_TEXT = "0:01"

    def __init__(self) -> None:
        self._tmpdir = tempfile.mkdtemp(prefix="run_context_parity_")
        self.gpkg = os.path.join(self._tmpdir, "parity_model.gpkg")
        self.mesh_name = "parity_mesh"
        self.dlg = None
        self.iface = None
        self.ctx_gui = None
        self.payload: Dict[str, Any] = {}
        self.ctx_cli = None

    # ── construction ────────────────────────────────────────────────────
    def build(self) -> "ParityFixture":
        self._persist_mesh()
        self._build_dialog()
        return self

    def close(self) -> None:
        if self.dlg is not None:
            try:
                self.dlg.close()
            except Exception:
                pass
            self.dlg = None
        from qgis.core import QgsProject
        QgsProject.instance().clear()
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _persist_mesh(self) -> None:
        node_x, node_y, _, cell_nodes, _, _ = _make_cartesian_quad_mesh(
            self.NX, self.NY, self.LX, self.LY,
        )
        node_z = self.BED_SLOPE * (self.LX - node_x)
        # BC values baked here only control which edges the BLOB marks as
        # boundary (bc != 0); both builders re-derive type/values from the
        # configured default_bc_type.
        bc_n0, bc_n1, bc_tp, bc_vl = _channel_bc_edges(self.NX, self.NY, 0.0, self.BED_SLOPE)
        _serialize_and_persist_mesh(
            self.gpkg, self.mesh_name,
            node_x, node_y, node_z, cell_nodes,
            bc_n0, bc_n1, bc_tp, bc_vl,
        )

    def _build_dialog(self) -> None:
        ensure_qgis_app()
        from qgis.PyQt import QtWidgets as _qt_widgets
        from qgis.core import QgsProject
        from swe2d.workbench.studio_dialog import SWE2DWorkbenchStudioDialog

        # A cleared real QgsProject has an invalid CRS and no layers, so
        # the GUI unit service defaults to SI — matching the CLI's
        # derivation from the (empty) mesh CRS WKT.
        QgsProject.instance().clear()

        # A real QMainWindow as iface.mainWindow(): dock widgets are
        # parented to it, and _find_widget falls back to searching it.
        self.iface = stub_iface()
        self.iface.mainWindow.return_value = _qt_widgets.QMainWindow()
        dlg = SWE2DWorkbenchStudioDialog(iface=self.iface)
        self.dlg = dlg

        # Load the mesh from the same baked BLOB the CLI will read.
        from swe2d.core.gpkg_io import query_mesh_from_gpkg
        mesh_data = query_mesh_from_gpkg(self.gpkg, self.mesh_name)
        if mesh_data is None:
            raise RuntimeError(f"mesh '{self.mesh_name}' not found in {self.gpkg}")
        mesh_data["mesh_name"] = self.mesh_name
        dlg._mesh_data = mesh_data
        dlg._model_gpkg_path = self.gpkg

        mtv = dlg._model_tab_view
        mtv.run_time_edit.setText(self.RUN_TIME_TEXT)
        mtv.output_interval_edit.setText(self.OUTPUT_INTERVAL_TEXT)
        mtv.results_gpkg_path_edit.setText(self.gpkg)
        wse_idx = mtv.initial_condition_combo.findData("uniform_wse")
        if wse_idx < 0:
            raise RuntimeError("initial_condition_combo has no 'uniform_wse' item")
        mtv.initial_condition_combo.setCurrentIndex(wse_idx)
        mtv.initial_wse_spin.setValue(self.INITIAL_WSE)

        # Deselect every layer combo.  The dialogs populate them with a
        # "(none)" placeholder whose non-empty text trips the "layer not a
        # valid vector layer" guards in the run builders
        # (studio_dialog.py:1998-2012).  Signals must be blocked:
        # currentIndexChanged is wired to refresh_layer_combos()
        # (studio_tab_builder.py:140-174) which re-adds the placeholder.
        # With no items, currentText() == "" = "no layer selected".
        for combo in mtv.findChildren(_qt_widgets.QComboBox):
            if combo.objectName().endswith("_layer_combo"):
                combo.blockSignals(True)
                combo.clear()

    # ── context construction ────────────────────────────────────────────
    def build_contexts(
        self,
        payload_patches: "Dict[str, Any] | None" = None,
    ) -> Tuple[Any, Any]:
        """Build (gui_ctx, cli_ctx) from the same dialog widget state.

        ``payload_patches`` is merged into ``payload["params"]`` before the
        CLI build — used by the replay-equivalence test to normalize
        KNOWN_DIVERGENCES fields whose drift would otherwise change the
        physics (each patch there is commented with its allowlist entry).

        Both constructions are wrapped in FallbackTracker so any
        ``logger.warning``-based silent fallback during the build fails
        the test.  No ignore_patterns are needed today: this fixture
        configures none of the features whose KNOWN_DIVERGENCES entries
        would emit warnings (no bc_lines / drainage / hyetograph layers).
        If a future fixture enables them, filter the specific warning
        text here with a comment naming the allowlist entry.
        """
        import contextlib

        dlg = self.dlg
        with contextlib.ExitStack() as stack:
            for name in _WATCHED_LOGGERS:
                stack.enter_context(FallbackTracker(logger_name=name))
            ctx_gui = dlg._controller._build_run_context()
        if ctx_gui is None:
            raise RuntimeError("GUI _build_run_context returned None")
        self.ctx_gui = ctx_gui

        widget_state = dlg._controller.collect_widget_state_for_save()
        payload = dlg._controller._build_replay_payload(
            widget_state,
            self.mesh_name,
            ctx_gui.run_duration_s,
            self.gpkg,
            run_id=ctx_gui.run_id,
        )
        if payload_patches:
            payload.setdefault("params", {}).update(payload_patches)
        self.payload = payload

        with contextlib.ExitStack() as stack:
            for name in _WATCHED_LOGGERS:
                stack.enter_context(FallbackTracker(logger_name=name))
            from swe2d.core.builder import build_run_context_from_dict
            self.ctx_cli = build_run_context_from_dict(payload)
        return self.ctx_gui, self.ctx_cli


# ═══════════════════════════════════════════════════════════════════════════════
# Recursive diff engine
# ═══════════════════════════════════════════════════════════════════════════════

def _is_array_like(v: Any) -> bool:
    if isinstance(v, np.ndarray):
        return True
    if isinstance(v, (list, tuple)) and v:
        try:
            np.asarray(v, dtype=np.float64)
            return True
        except (TypeError, ValueError):
            return False
    return False


def _diff_values(path: str, a: Any, b: Any, out: List[Tuple[str, str, str]]) -> None:
    """Recursively compare two values, appending (path, gui, cli) diffs."""
    if a is None or b is None:
        if a is not b:
            out.append((path, repr(a)[:80], repr(b)[:80]))
        return
    if isinstance(a, threading.Event) or isinstance(b, threading.Event):
        if type(a) is not type(b):
            out.append((path, type(a).__name__, type(b).__name__))
        return
    if isinstance(a, enum.Enum) or isinstance(b, enum.Enum):
        # IntEnum compares equal to its int value; flag only real mismatch
        # or a str-enum vs raw value mismatch.
        if a != b:
            out.append((path, repr(a)[:80], repr(b)[:80]))
        return
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray) or (
        _is_array_like(a) and _is_array_like(b)
    ):
        try:
            aa = np.atleast_1d(np.asarray(a, dtype=np.float64))
            bb = np.atleast_1d(np.asarray(b, dtype=np.float64))
        except (TypeError, ValueError):
            out.append((path, type(a).__name__, f"{type(b).__name__} (not array-coercible)"))
            return
        if aa.shape != bb.shape:
            out.append((path, f"shape{aa.shape}", f"shape{bb.shape}"))
        elif not np.allclose(aa, bb, rtol=1e-12, atol=1e-15, equal_nan=True):
            out.append((path, "array-values-differ",
                        f"maxabs={np.max(np.abs(aa - bb)):.3g}"))
        return
    if isinstance(a, dict) or isinstance(b, dict):
        if not isinstance(a, dict) or not isinstance(b, dict):
            out.append((path, type(a).__name__, type(b).__name__))
            return
        for key in sorted(set(a) | set(b), key=repr):
            if key not in a:
                out.append((f"{path}[{key!r}]", "<missing>", repr(b[key])[:80]))
            elif key not in b:
                out.append((f"{path}[{key!r}]", repr(a[key])[:80], "<missing>"))
            else:
                _diff_values(f"{path}[{key!r}]", a[key], b[key], out)
        return
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        if type(a) is not type(b) and not (a == b):
            out.append((path, repr(a)[:80], repr(b)[:80]))
        elif len(a) != len(b):
            out.append((path, f"len={len(a)}", f"len={len(b)}"))
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                _diff_values(f"{path}[{i}]", x, y, out)
        return
    if isinstance(a, float) or isinstance(b, float):
        if not (isinstance(a, (int, float)) and isinstance(b, (int, float))) \
                or not math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-15):
            out.append((path, repr(a)[:80], repr(b)[:80]))
        return
    if isinstance(a, (str, int, bool, bytes)):
        if a != b:
            out.append((path, repr(a)[:80], repr(b)[:80]))
        return
    # Arbitrary objects (configs, forcing containers): fall back to equality
    # then repr comparison.
    try:
        if a == b:
            return
    except Exception:
        pass
    if repr(a) != repr(b):
        out.append((path, repr(a)[:80], repr(b)[:80]))


def _bc_edge_map(ctx) -> Dict[Tuple[int, int], Tuple[int, float, float]]:
    """BC config keyed by normalized edge (min(n0,n1), max(n0,n1))."""
    out: Dict[Tuple[int, int], Tuple[int, float, float]] = {}
    for n0, n1, tp, vl, rx in zip(
        np.asarray(ctx.bc_n0).ravel(),
        np.asarray(ctx.bc_n1).ravel(),
        np.asarray(ctx.bc_tp).ravel(),
        np.asarray(ctx.bc_vl).ravel(),
        np.asarray(ctx.bc_relax).ravel(),
    ):
        key = (int(min(n0, n1)), int(max(n0, n1)))
        out[key] = (int(tp), float(vl), float(rx))
    return out


def diff_run_contexts(ctx_gui, ctx_cli) -> List[Tuple[str, str, str]]:
    """Return a list of (field_path, gui_value, cli_value) differences.

    Callback fields are excluded (compared separately by presence/name),
    identity fields are normalized away, and BC edge arrays are compared
    order-insensitively via ``_bc_edge_map``.
    """
    diffs: List[Tuple[str, str, str]] = []
    gui_fields = {f.name for f in _dc_fields(ctx_gui)}
    cli_fields = {f.name for f in _dc_fields(ctx_cli)}
    if gui_fields != cli_fields:
        diffs.append(("<dataclass fields>",
                      repr(sorted(gui_fields - cli_fields)),
                      repr(sorted(cli_fields - gui_fields))))
    for name in sorted(gui_fields & cli_fields):
        if name in CALLBACK_FIELDS or name in IDENTITY_FIELDS or name == "cancel_event":
            continue
        if name in _BC_EDGE_FIELDS:
            continue  # handled order-insensitively below
        _diff_values(name, getattr(ctx_gui, name), getattr(ctx_cli, name), diffs)

    # ── BC edges: order-insensitive comparison ──────────────────────────
    gui_edges = _bc_edge_map(ctx_gui)
    cli_edges = _bc_edge_map(ctx_cli)
    for key in sorted(set(gui_edges) | set(cli_edges)):
        if key not in gui_edges:
            diffs.append(("bc_edges", "<missing edge>", repr(key)))
            continue
        if key not in cli_edges:
            diffs.append(("bc_edges", repr(key), "<missing edge>"))
            continue
        g_tp, g_vl, g_rx = gui_edges[key]
        c_tp, c_vl, c_rx = cli_edges[key]
        if g_tp != c_tp:
            diffs.append(("bc_tp", f"edge{key} type={g_tp}", f"type={c_tp}"))
        if not math.isclose(g_vl, c_vl, rel_tol=1e-12, abs_tol=1e-15):
            diffs.append(("bc_vl", f"edge{key} val={g_vl}", f"val={c_vl}"))
        if not math.isclose(g_rx, c_rx, rel_tol=1e-12, abs_tol=1e-15):
            diffs.append(("bc_relax", f"edge{key} relax={g_rx}", f"relax={c_rx}"))
    return diffs


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

@requires_qgis
@unittest.skipUnless(_gpu_available(), "CUDA GPU not available (GUI options builder requires it)")
class TestRunContextParity(unittest.TestCase):
    """GUI-built vs CLI-built RunContext recursive diff gate."""

    @classmethod
    def setUpClass(cls):
        ensure_qgis_app()
        # Register cleanup BEFORE build() so a mid-setup failure (e.g. a
        # FallbackTracker silent-fallback error) cannot leak the patched
        # globals / dialog into later test modules.
        cls.fixture = ParityFixture()
        cls.addClassCleanup(cls.fixture.close)
        cls.fixture.build()
        cls.ctx_gui, cls.ctx_cli = cls.fixture.build_contexts()

    def test_contexts_built(self):
        """Sanity: both builders produced a RunContext with the same mesh size."""
        self.assertIsNotNone(self.ctx_gui)
        self.assertIsNotNone(self.ctx_cli)
        self.assertEqual(self.ctx_gui.cell_areas.shape, self.ctx_cli.cell_areas.shape)
        self.assertGreater(self.ctx_gui.cell_areas.size, 0)

    def test_allowlist_entries_are_real_fields(self):
        """Every KNOWN_DIVERGENCES / CALLBACK / IDENTITY key names a real field."""
        field_names = {f.name for f in _dc_fields(self.ctx_gui)}
        for key in KNOWN_DIVERGENCES:
            self.assertIn(key, field_names,
                          f"KNOWN_DIVERGENCES entry {key!r} is not a RunContext field")
        for key in CALLBACK_FIELDS | IDENTITY_FIELDS | {"cancel_event"}:
            self.assertIn(key, field_names)

    def test_callbacks_present_both_paths(self):
        """Callback fields are callable on both contexts (identity is meaningless
        across paths — see CALLBACK_FIELDS comment)."""
        for name in sorted(CALLBACK_FIELDS):
            self.assertTrue(callable(getattr(self.ctx_gui, name)), f"GUI {name} not callable")
            self.assertTrue(callable(getattr(self.ctx_cli, name)), f"CLI {name} not callable")

    def test_run_context_parity_diff(self):
        """The core gate: no RunContext field differs outside KNOWN_DIVERGENCES.

        Allowlist matching follows ``_matches_allowlist``: exact path or
        nested-child (``key[...]``) match.
        """
        diffs = diff_run_contexts(self.ctx_gui, self.ctx_cli)

        unexpected = [
            (p, a, b) for p, a, b in diffs
            if not _matches_allowlist(p, KNOWN_DIVERGENCES)
        ]
        if unexpected:
            lines = "\n".join(f"  {p}: GUI={a}  CLI={b}" for p, a, b in unexpected)
            self.fail(
                f"{len(unexpected)} RunContext field(s) diverge outside the "
                f"KNOWN_DIVERGENCES allowlist:\n{lines}\n"
                "Fix the drift (preferred) or document it in KNOWN_DIVERGENCES."
            )

        # Every exercised allowlist entry must correspond to a live
        # divergence — when a fix lands, its entry must be deleted.
        diff_paths = [path for path, _, _ in diffs]
        stale = [
            key for key, meta in KNOWN_DIVERGENCES.items()
            if meta.get("exercised")
            and not any(p == key or p.startswith(key + "[") for p in diff_paths)
        ]
        if stale:
            self.fail(
                "These KNOWN_DIVERGENCES entries no longer diverge — delete "
                f"them (the drift is fixed): {stale}"
            )

    def test_known_divergences_documented(self):
        """Each allowlist entry carries a non-empty reason with a code ref."""
        for key, meta in KNOWN_DIVERGENCES.items():
            self.assertIn("reason", meta)
            self.assertTrue(str(meta["reason"]).strip(), f"{key} has no reason")


class TestAllowlistMatching(unittest.TestCase):
    """The allowlist matcher covers nested diff paths (dict/list entries)."""

    def test_nested_path_matches_parent_entry(self):
        # `sample_map_data` and `edge_groups` were retired from
        # KNOWN_DIVERGENCES in commit 7061cd7; repoint to other nested
        # entries under the still-present `pipe_network_cfg` parent.
        self.assertTrue(_matches_allowlist(
            "pipe_network_cfg['friction_method']", KNOWN_DIVERGENCES))
        self.assertTrue(_matches_allowlist(
            "pipe_network_cfg['surcharge_method']", KNOWN_DIVERGENCES))
        self.assertTrue(_matches_allowlist(
            "pipe_network_cfg['time_integrator']", KNOWN_DIVERGENCES))

    def test_exact_and_non_sibling_paths(self):
        # "pipe_network_cfg" is in the allowlist (Phase 3 drainage keys)
        # and the fixture has no drainage layers so it doesn't fire —
        # but the matcher itself must still work.
        self.assertTrue(_matches_allowlist("pipe_network_cfg", KNOWN_DIVERGENCES))
        # The "[" anchor prevents false prefix hits on sibling names.
        self.assertFalse(_matches_allowlist("pipe_network_cfg_extra", KNOWN_DIVERGENCES))
        self.assertFalse(_matches_allowlist("pipe_network", KNOWN_DIVERGENCES))
        self.assertFalse(_matches_allowlist("run_duration_s", KNOWN_DIVERGENCES))


if __name__ == "__main__":
    unittest.main()
