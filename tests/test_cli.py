"""Smoke tests for CLI headless runner."""
import json
import os
import tempfile
import time
import numpy as np

from swe2d.services.gpkg_persistence_service import (
    persist_mesh_to_geopackage,
)


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
    """Full round trip: build small mesh, save to GPKG, load back."""
    mesh_data = {
        "node_x": np.array([0.0, 10.0, 5.0], dtype=np.float64),
        "node_y": np.array([0.0, 0.0, 10.0], dtype=np.float64),
        "node_z": np.array([5.0, 5.0, 4.0], dtype=np.float64),
        "cell_nodes": np.array([0, 1, 2], dtype=np.int32),
    }
    with tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False) as f:
        gpkg = f.name
    try:
        persist_mesh_to_geopackage(gpkg, "test", mesh_data)
        from swe2d.cli.gpkg_adapter import query_mesh_from_gpkg
        loaded = query_mesh_from_gpkg(gpkg, "test")
        assert loaded is not None
        for k in ("node_x", "node_y", "node_z", "cell_nodes"):
            np.testing.assert_array_almost_equal(loaded[k], mesh_data[k])
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
