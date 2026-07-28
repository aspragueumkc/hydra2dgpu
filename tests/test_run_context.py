import unittest
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from swe2d.core.run_context import RunContext


def test_run_context_holds_arrays_and_cancel_event():
    ctx = RunContext(
        run_id="r1",
        run_wallclock_start="2026-01-01 00:00:00",
        run_log_start_idx=0,
        run_duration_s=10.0,
        output_interval_s=1.0,
        node_x=np.array([0.0, 1.0]),
        node_y=np.array([0.0, 0.0]),
        node_z=np.array([0.0, 0.0]),
        cell_nodes=np.array([[0, 1, 2]], dtype=np.int32),
        bc_n0=np.array([0], dtype=np.int32),
        bc_n1=np.array([1], dtype=np.int32),
        bc_tp=np.array([0], dtype=np.int32),
        bc_vl=np.array([0.0]),
        h0=np.array([1.0]),
        hu0=np.array([0.0]),
        hv0=np.array([0.0]),
    )
    assert ctx.run_id == "r1"
    assert ctx.node_x.size == 2
    assert ctx.cancel_event.is_set() is False


def _valid_run_context() -> RunContext:
    """Return the smallest context accepted by executor validation."""
    return RunContext(
        run_id="r1",
        run_wallclock_start="2026-01-01 00:00:00",
        run_log_start_idx=0,
        node_x=np.array([0.0, 1.0, 0.0]),
        node_y=np.array([0.0, 0.0, 1.0]),
        node_z=np.zeros(3),
        cell_nodes=np.array([[0, 1, 2]], dtype=np.int32),
        h0=np.zeros(1),
        hu0=np.zeros(1),
        hv0=np.zeros(1),
        cell_areas=np.ones(1),
        cell_centroids=np.array([[1.0 / 3.0, 1.0 / 3.0]]),
        mesh_cell_areas=lambda: np.ones(1),
        mesh_cell_min_bed=lambda: np.zeros(1),
        mesh_cell_centroids=lambda: (np.array([1.0 / 3.0]), np.array([1.0 / 3.0])),
        internal_flow_source_cms_at_time=lambda _forcing, _t_s: None,
    )


@pytest.mark.parametrize(
    "field_name",
    (
        "node_x",
        "node_y",
        "node_z",
        "cell_nodes",
        "h0",
        "hu0",
        "hv0",
        "cell_areas",
        "cell_centroids",
    ),
)
def test_validate_run_context_rejects_empty_required_array(field_name: str):
    from swe2d.core import executor

    ctx = replace(_valid_run_context(), **{field_name: np.empty(0)})

    with pytest.raises(ValueError, match=field_name):
        executor._validate_run_context(ctx)


@pytest.mark.parametrize(
    "field_name",
    (
        "mesh_cell_areas",
        "mesh_cell_min_bed",
        "mesh_cell_centroids",
        "internal_flow_source_cms_at_time",
    ),
)
def test_validate_run_context_rejects_default_required_callback(field_name: str):
    from swe2d.core import executor

    default_callback = RunContext.__dataclass_fields__[field_name].default
    ctx = replace(_valid_run_context(), **{field_name: default_callback})

    with pytest.raises(ValueError, match=field_name):
        executor._validate_run_context(ctx)


def test_execute_run_validates_context_before_runtime_imports():
    from swe2d.core.executor import execute_run

    with pytest.raises(ValueError, match="node_x"):
        execute_run(
            RunContext(
                run_id="r1",
                run_wallclock_start="2026-01-01 00:00:00",
                run_log_start_idx=0,
            ),
            object(),
        )


def test_run_context_defaults_and_immutability():
    ctx = RunContext(
        run_id="r1",
        run_wallclock_start="2026-01-01 00:00:00",
        run_log_start_idx=0,
    )
    assert ctx.node_x.size == 0
    assert ctx.node_y.size == 0
    assert ctx.cell_nodes.shape == (0, 3)
    with pytest.raises(FrozenInstanceError):
        ctx.run_id = "r2"


def test_from_widget_params_raises_without_mesh():
    """from_widget_params is now a thin normalizer: it must surface the
    canonical builder's validation errors (not silently return defaults).

    Either ``mesh_gpkg`` missing or ``mesh_name`` empty → canonical
    builder raises.  Verify both error paths.
    """
    import os
    import tempfile
    # Case 1: empty mesh_name with a valid GPKG path → ValueError("mesh_name required").
    with tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False) as tf:
        tmp_gpkg = tf.name
    try:
        with pytest.raises(ValueError, match="mesh_name"):
            RunContext.from_widget_params(
                {"n_mann_spin": 0.035}, mesh_name="", mesh_gpkg=tmp_gpkg,
            )
    finally:
        os.unlink(tmp_gpkg)


def test_from_widget_params_raises_when_mesh_gpkg_missing():
    """Mesh GPKG path that does not exist → ``FileNotFoundError``."""
    with pytest.raises(FileNotFoundError):
        RunContext.from_widget_params(
            {"n_mann_spin": 0.035},
            mesh_name="any_mesh",
            mesh_gpkg="/nonexistent/does_not_exist.gpkg",
        )

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
