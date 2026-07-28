"""Smoke tests for CLI headless runner."""
import json
import os
import sys
import tempfile
import unittest

import numpy as np

# Make tests/_swe2d_test_helpers importable (repo-root style)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def test_sweep_expansion_simple():
    """_expand_sweep produces correct Cartesian product."""
    from swe2d.cli.batch_runner import _expand_sweep

    params = {
        "sweep": {
            "params.n_mann": [0.020, 0.030, 0.040],
        },
        "id_template": "n_{n_mann:.3f}",
        "params": {"duration_s": 3600},
    }
    expanded = _expand_sweep(params)
    assert len(expanded) == 3
    assert expanded[0]["params"]["n_mann"] == 0.020
    assert expanded[1]["params"]["n_mann"] == 0.030
    assert expanded[2]["params"]["n_mann"] == 0.040
    assert expanded[0]["id"] == "n_0.020"


def test_sweep_expansion_layer():
    """Sweep over a layer reference (string values)."""
    from swe2d.cli.batch_runner import _expand_sweep

    params = {
        "sweep": {
            "mannings_layer": ["landuse_a", "landuse_b"],
        },
        "id_template": "{mannings_layer}",
        "params": {"duration_s": 3600},
    }
    expanded = _expand_sweep(params)
    assert len(expanded) == 2
    assert expanded[0]["mannings_layer"] == "landuse_a"
    assert expanded[1]["mannings_layer"] == "landuse_b"


def test_mesh_persist_and_load_round_trip():
    """Full round trip: build small mesh, save to GPKG, load back via the CLI helper."""
    from tests._swe2d_test_helpers import _serialize_and_persist_mesh
    from swe2d.core.gpkg_io import query_mesh_from_gpkg

    mesh_data = {
        "node_x": np.array([0.0, 10.0, 5.0], dtype=np.float64),
        "node_y": np.array([0.0, 0.0, 10.0], dtype=np.float64),
        "node_z": np.array([5.0, 5.0, 4.0], dtype=np.float64),
        "cell_nodes": np.array([0, 1, 2], dtype=np.int32),
    }
    with tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False) as f:
        gpkg = f.name
    try:
        _serialize_and_persist_mesh(
            gpkg, "test",
            mesh_data["node_x"], mesh_data["node_y"], mesh_data["node_z"],
            mesh_data["cell_nodes"],
            np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float64),
        )
        loaded = query_mesh_from_gpkg(gpkg, "test")
        assert loaded is not None
        for k in ("node_x", "node_y", "node_z"):
            np.testing.assert_array_almost_equal(loaded[k], mesh_data[k])
        # cell_nodes may be returned as cell_face_nodes; check via cell_face_nodes too.
        if "cell_nodes" in loaded:
            np.testing.assert_array_equal(loaded["cell_nodes"], mesh_data["cell_nodes"])
    finally:
        if os.path.exists(gpkg):
            os.unlink(gpkg)


def test_mps_ensure_fallback_no_daemon():
    """_ensure_mps returns False gracefully when daemon is unavailable.

    This is the normal case on most systems (no MPS daemon running).
    The function must NOT raise; it logs a message and returns False.
    """
    from swe2d.cli.batch_runner import _ensure_mps, _stop_mps_if_we_started
    # Should return False since MPS is likely not available in CI/dev
    result = _ensure_mps()
    assert isinstance(result, bool)
    # Should not raise even if we didn't start it
    _stop_mps_if_we_started(False)
    _stop_mps_if_we_started(True)  # gracefully fails if we didn't start


def test_status_file_written_during_run():
    """_atomic_write_json creates a valid JSON status file."""
    from swe2d.cli.headless_runner import _atomic_write_json

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        payload = {"step": 42, "t": 10.5, "status": "running",
                    "wet_cells": 5000, "elapsed_s": 2.3}
        _atomic_write_json(path, payload)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["step"] == 42
        assert loaded["status"] == "running"
        assert loaded["wet_cells"] == 5000
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_status_file_types():
    """Status file payload has correct types."""
    from swe2d.cli.headless_runner import _atomic_write_json

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        payload = {"step": 0, "t": 0.0, "dt": 0.05, "wet_cells": -1,
                    "elapsed_s": 0.0, "status": "running"}
        _atomic_write_json(path, payload)
        with open(path) as f:
            loaded = json.load(f)
        assert isinstance(loaded["step"], int)
        assert isinstance(loaded["t"], (int, float))
        assert isinstance(loaded["status"], str)
    finally:
        if os.path.exists(path):
            os.unlink(path)


def _assert_one_second_run_reports_solver_step_and_diagnostics():
    """Status and callbacks expose solver context, not callback counts/percent."""
    from swe2d.cli.headless_runner import execute_run
    from tests._swe2d_test_helpers import (
        _make_cartesian_quad_mesh,
        _serialize_and_persist_mesh,
    )

    callbacks = []
    with tempfile.TemporaryDirectory() as tmp:
        gpkg = os.path.join(tmp, "status_mesh.gpkg")
        status_path = os.path.join(tmp, "status.json")
        mesh_name = "status_mesh"
        node_x, node_y, _, cell_nodes, _, _ = _make_cartesian_quad_mesh(
            2, 1, 2.0, 1.0,
        )
        _serialize_and_persist_mesh(
            gpkg,
            mesh_name,
            node_x,
            node_y,
            np.zeros_like(node_x),
            cell_nodes,
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float64),
        )

        execute_run(
            gpkg,
            {
                "mesh": mesh_name,
                "params": {
                    "run_duration_s": 1.0,
                    "output_interval_s": 1.0,
                    "dt_request": 0.1,
                    "dt_fixed": 0.1,
                    "initial_dt": 0.1,
                    "temporal_scheme": 1,
                    "save_mesh_results": False,
                },
            },
            progress_callback=lambda sim_time, diagnostics: callbacks.append(
                (sim_time, diagnostics)
            ),
            status_file_path=status_path,
            status_interval_s=0.0,
        )

        with open(status_path, encoding="utf-8") as status_file:
            payload = json.load(status_file)

    assert callbacks
    sim_time, diagnostics = callbacks[-1]
    assert set(diagnostics) == {"dt", "wet_cells", "elapsed_s"}
    assert sim_time >= 1.0
    assert diagnostics["dt"] > 0.0
    assert isinstance(diagnostics["wet_cells"], int)
    assert diagnostics["elapsed_s"] >= 0.0
    assert payload["step"] == len(callbacks) - 1
    assert payload["step_idx"] == payload["step"]
    assert payload["t"] == sim_time
    assert payload["dt"] == diagnostics["dt"]
    assert payload["wet_cells"] == diagnostics["wet_cells"]
    assert payload["status"] == "done"


class TestHeadlessStatusProgress(unittest.TestCase):
    """Unittest-discoverable wrapper for the real one-second CLI run."""

    def test_one_second_run_reports_solver_step_and_diagnostics(self):
        _assert_one_second_run_reports_solver_step_and_diagnostics()

class _PytestStyleWrapper(unittest.TestCase):
    """Auto-generated wrapper for module-level test functions.

    Created by tools/wrap_pytest_style.py so that pytest-style tests
    (def test_* at module level) become visible to `python3 -m unittest`.
    Each module-level test is attached as a staticmethod so it can be
    discovered and run as a unittest TestCase.
    """
__wrapped_funcs = []
for _name, _obj in list(globals().items()):
    if _name.startswith("test_") and callable(_obj) and not isinstance(_obj, type):
        setattr(_PytestStyleWrapper, _name, staticmethod(_obj))
        __wrapped_funcs.append(_name)
for _name in __wrapped_funcs:
    del globals()[_name]
del _name, _obj, __wrapped_funcs
