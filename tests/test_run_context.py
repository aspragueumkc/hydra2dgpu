import numpy as np
import pytest
from dataclasses import FrozenInstanceError
from swe2d.workbench.workers.run_context import RunContext


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


def test_run_context_defaults_and_immutability():
    ctx = RunContext(
        run_id="r1",
        run_wallclock_start="2026-01-01 00:00:00",
        run_log_start_idx=0,
    )
    assert ctx.node_x.size == 0
    assert ctx.node_y.size == 0
    assert ctx.cell_nodes.shape == (0, 3)
    assert ctx.apply_timeseries_bc_values() is None
    assert ctx.apply_timeseries_bc_values(1, 2) is None
    assert ctx.distribute_total_flow_to_unit_q() is None
    assert ctx.distribute_total_flow_to_unit_q(1, 2, 3) is None
    with pytest.raises(FrozenInstanceError):
        ctx.run_id = "r2"
