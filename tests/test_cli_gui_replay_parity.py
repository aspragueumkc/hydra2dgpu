"""CLI↔GUI replay-equivalence test — CLI-first refactor Phase 0 gate (GPU).

Runs ONE project configuration through both RunContext construction paths
and the shared executor, then asserts the final state arrays match:

* GUI path  — ``RunController._build_run_context()`` on the Studio dialog
  (real headless QGIS via ``tests.qgis_real_env``; same ``ParityFixture``
  as ``tests/test_run_context_parity.py``).
* CLI path  — ``build_run_context_from_dict()`` fed the ``swe2d-replay/1``
  JSON exported from the SAME dialog widget state.
 * Executor  — ``swe2d.core.executor.execute_run`` for BOTH contexts, so any result
   difference is attributable to context construction, not the executor.

The known GUI/CLI divergences (see ``KNOWN_DIVERGENCES`` in
``tests/test_run_context_parity.py``) would change the physics of this
run, so the CLI payload is normalized for exactly those fields — each
patch names its allowlist entry.  The remaining context diff after
normalization is asserted to stay within the non-physics-relevant
allowlist entries.  As later phases shrink the allowlist, the patches
below shrink with it.

GPU-gated: both contexts execute on the CUDA solver.

Usage:
    python3 -m unittest tests.test_cli_gui_replay_parity -v
"""

from __future__ import annotations

from typing import Any, Dict, List
import threading
import unittest

import numpy as np

from tests._swe2d_test_helpers import _gpu_available
from tests.qgis_real_env import ensure_qgis_app, requires_qgis
from tests.test_run_context_parity import (
    KNOWN_DIVERGENCES,
    ParityFixture,
    _matches_allowlist,
    diff_run_contexts,
)


# KNOWN_DIVERGENCES entries that must NOT affect this run's physics.  The
# CLI payload is patched to the GUI-side values for the fields whose drift
# would alter the trajectory; the remaining entries are either inert for
# this configuration (no rain → rain_mm_to_model_depth, no coupling →
# coupling_soa) or container-shape-only (cell_centroids, handled by the
# executor for both shapes).
_INERT_FOR_THIS_RUN = {"rain_mm_to_model_depth", "coupling_soa", "cell_centroids"}


@requires_qgis
@unittest.skipUnless(_gpu_available(), "CUDA GPU not available")
class TestCliGuiReplayParity(unittest.TestCase):
    """Final h/hu/hv from GUI-built and CLI-built RunContexts must match."""

    @classmethod
    def setUpClass(cls):
        ensure_qgis_app()
        # Register cleanup BEFORE build() so a mid-setup failure cannot
        # leak the patched globals / dialog into later test modules.
        cls.fixture = ParityFixture()
        cls.addClassCleanup(cls.fixture.close)
        cls.fixture.build()
        dlg = cls.fixture.dlg
        mtv = dlg._model_tab_view

        # Normalize the replay payload for the KNOWN_DIVERGENCES entries
        # whose drift would change this run's physics.  Values are read
        # from the live widgets (the SAME state the GUI context was built
        # from), not hand-written duplicates.
        #
        # dt_request / dt_fixed are no longer in the patch list: the
        # canonical builder now derives both from (adaptive_cfl_dt,
        # dt_cfg) consistently for GUI and CLI, so they agree on their
        # own.  The previous patch was a workaround for the lost
        # derivation (Phase 1.B refactor, commit 70561f9a).
        payload_patches = {
            # KNOWN_DIVERGENCES["bc_tp"]: widget_state stores the combo
            # currentIndex; the GUI applies currentData().  Patch to the
            # data value the GUI used.
            "default_bc_type": int(mtv.default_bc_type_combo.currentData()),
            # KNOWN_DIVERGENCES["h0"]: collect_params() omits the initial_*
            # widgets, so the exported JSON loses the initial condition.
            # Re-inject from the live widgets.
            "initial_wse": float(mtv.initial_wse_spin.value()),
            "initial_depth": float(mtv.initial_depth_spin.value()),
        }
        cls.ctx_gui, cls.ctx_cli = cls.fixture.build_contexts(
            payload_patches=payload_patches,
        )

    def test_context_diff_confined_to_inert_allowlist(self):
        """After payload normalization, the only remaining context diffs are
        KNOWN_DIVERGENCES entries that cannot affect this run's physics.

        Allowlist matching follows ``_matches_allowlist`` (exact path or
        nested-child ``key[...]`` match)."""
        diffs = diff_run_contexts(self.ctx_gui, self.ctx_cli)
        unexpected = [
            (p, a, b) for p, a, b in diffs
            if not _matches_allowlist(p, KNOWN_DIVERGENCES)
            or not _matches_allowlist(p, _INERT_FOR_THIS_RUN)
        ]
        if unexpected:
            lines = "\n".join(f"  {p}: GUI={a}  CLI={b}" for p, a, b in unexpected)
            self.fail(
                "Context diff after normalization is not confined to the "
                f"inert allowlist entries {sorted(_INERT_FOR_THIS_RUN)}:\n{lines}"
            )

    def test_replay_equivalence(self):
        """Both contexts produce matching final h/hu/hv on the GPU solver."""
        from swe2d.core.executor import execute_run
        from swe2d.core.sink_protocol import Sink

        class _TestSink(Sink):
            def __init__(self, log_capture):
                self._log_capture = log_capture
                self.snapshot_request_event = threading.Event()

            def log(self, message: str) -> None:
                self._log_capture.append(message)

            def progress(self, percent: float, diagnostics: Dict[str, Any]) -> None:
                pass

            def snapshot(self, fields: List[Any]) -> None:
                pass

            def finished(self, result: Dict[str, Any]) -> None:
                pass

            def failed(self, error: str) -> None:
                self._log_capture.append(f"[ERROR] {error}")

            def permutation(self, cell_perm, result) -> None:
                result.event.set()

            def request_snapshot(self) -> None:
                self.snapshot_request_event.set()

        gui_log: list = []
        cli_log: list = []
        result_gui = execute_run(self.ctx_gui, _TestSink(gui_log))
        result_cli = execute_run(self.ctx_cli, _TestSink(cli_log))

        self.assertTrue(
            result_gui.ok,
            "GUI-path run failed:\n" + "\n".join(gui_log[-30:]),
        )
        self.assertTrue(
            result_cli.ok,
            "CLI-path run failed:\n" + "\n".join(cli_log[-30:]),
        )
        self.assertAlmostEqual(
            result_gui.final_sim_time_s, result_cli.final_sim_time_s, places=9,
        )

        # The run must have actually computed something (fixture starts
        # wet on a slope) — a trivial all-dry comparison proves nothing.
        self.assertGreater(float(np.max(result_gui.h)), 1.0e-3,
                           "GUI run is bone dry — fixture is not exercising the solver")

        for name in ("h", "hu", "hv"):
            a = np.asarray(getattr(result_gui, name), dtype=np.float64)
            b = np.asarray(getattr(result_cli, name), dtype=np.float64)
            self.assertEqual(a.shape, b.shape, f"{name} shape mismatch")
            max_abs = float(np.max(np.abs(a - b))) if a.size else 0.0
            np.testing.assert_allclose(
                a, b, rtol=1e-9, atol=1e-12,
                err_msg=(
                    f"final {name} mismatch between GUI-path and CLI-path "
                    f"runs (max abs diff {max_abs:.3e})"
                ),
            )


if __name__ == "__main__":
    unittest.main()
