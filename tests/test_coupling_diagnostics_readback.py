"""End-to-end coupling readback tests via the real headless CLI.

These tests build a small synthetic mesh, run the full CLI pipeline
(``swe2d.cli.headless_runner.execute_run``), then read the resulting GPKG
and assert the baked coupling records are present.  No mocks for non-GUI
code: the GPU kernels, ``SWE2DBackend``, ``SWE2DCouplingController``, and the
``swe2d_baked_coupling`` persistence layer are all exercised exactly as
``python -m swe2d.cli run`` would exercise them.

The pre-existing bug (``swe2d/runtime/coupling.py:1043`` read
``state["cell_flow"]`` while the C++ binding returns ``state["cell_Q"]``)
caused every ``drainage_link/flow`` row in the GPKG to be missing
entirely — the readback returned an empty ``link_flow`` array, the
sampling loop never appended any rows, and the GPKG ended up with no
drainage link coupling records at all.

GUI elements are never mocked because the CLI doesn't use them.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if os.path.join(_REPO_ROOT, "build") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "build"))

from tests._swe2d_test_helpers import (
    _channel_bc_edges,
    _gpu_available,
    _make_rect_mesh,
    _read_coupling_rows,
    _run_cli_coupling,
)


pytestmark = pytest.mark.skipif(
    not _gpu_available(),
    reason="CUDA GPU not available",
)


# Minimal mesh: 3 cells across × 1 row. Produces 8 boundary edges (4 inflow
# on the left + 4 normal-depth on the right when ny=1) plus 4 walls.
_NX, _NY = 3, 1
_LX, _LY = 30.0, 5.0
_S0 = 0.001
_Q_IN = 2.0  # m²/s unit-discharge inflow
_T_END = 10.0  # short run, but enough timesteps for several coupling snapshots


def _channel_mesh_arrays():
    """Return (node_x, node_y, node_z, cell_nodes, bc_n0, bc_n1, bc_tp, bc_vl)."""
    node_x, node_y, node_z, cell_nodes = _make_rect_mesh(_NX, _NY, _LX, _LY)
    bc_n0, bc_n1, bc_tp, bc_vl = _channel_bc_edges(_NX, _NY, _Q_IN, _S0)
    return node_x, node_y, node_z, cell_nodes, bc_n0, bc_n1, bc_tp, bc_vl


def test_cli_run_persists_drainage_link_flow_rows(tmp_path):
    """Regression: CLI run must persist drainage_link/flow rows.

    Pre-fix bug: ``swe2d/runtime/coupling.py:1043`` looked up
    ``state["cell_flow"]`` (wrong key).  The C++ binding returns
    ``state["cell_Q"]``, so ``out["link_flow"]`` was always empty and the
    sampling loop never appended ``drainage_link/flow`` rows.

    We assert that the rows exist with the expected shape (one per
    timestep).  Values may be zero — that depends on the 2D solver's
    inflow propagation and is a separate concern from the readback bug.
    """
    gpkg = str(tmp_path / "coupling_cli.gpkg")
    mesh_name = "tiny_channel"
    node_x, node_y, node_z, cell_nodes, bc_n0, bc_n1, bc_tp, bc_vl = _channel_mesh_arrays()

    _run_cli_coupling(
        gpkg, mesh_name,
        node_x, node_y, node_z, cell_nodes,
        bc_n0, bc_n1, bc_tp, bc_vl,
        params={},  # helper overrides duration_s and output intervals
        duration_s=_T_END,
        q_in=_Q_IN,
    )

    rows = _read_coupling_rows(gpkg)
    assert rows, "no baked coupling rows were written by the CLI run"

    # ── Every drainage_link/flow row exists with non-empty times/values ──
    # key shape: (run_id, component, object_id, metric)
    flow_keys = [k for k in rows if k[1] == "drainage_link" and k[3] == "flow"]
    assert flow_keys, (
        f"no drainage_link/flow rows — the cell_flow/cell_Q readback bug "
        f"is back. got keys={list(rows)[:6]}"
    )
    for key in flow_keys:
        times, values = rows[key]
        assert times.size > 0, f"empty times for {key}"
        assert values.size > 0, f"empty values for {key}"
        assert times.size == values.size, (
            f"times/values length mismatch for {key}: "
            f"{times.size} vs {values.size}"
        )


def test_cli_run_writes_expected_coupling_schema(tmp_path):
    """Sanity check: CLI run writes the full coupling schema (drainage rows)."""
    gpkg = str(tmp_path / "coupling_count.gpkg")
    mesh_name = "tiny_channel2"
    node_x, node_y, node_z, cell_nodes, bc_n0, bc_n1, bc_tp, bc_vl = _channel_mesh_arrays()

    _run_cli_coupling(
        gpkg, mesh_name,
        node_x, node_y, node_z, cell_nodes,
        bc_n0, bc_n1, bc_tp, bc_vl,
        params={},
        duration_s=_T_END,
        q_in=_Q_IN,
    )

    rows = _read_coupling_rows(gpkg)
    # key shape: (run_id, component, object_id, metric)
    components = {k[1] for k in rows}
    metrics_by_component = {}
    for _rid, comp, _oid, metric in rows:
        metrics_by_component.setdefault(comp, set()).add(metric)

    # Must have drainage rows (the regression we care about)
    assert "drainage_link" in components
    assert "drainage_node" in components
    assert "flow" in metrics_by_component["drainage_link"]
    assert "depth" in metrics_by_component["drainage_node"]


def test_cli_run_persists_nonzero_structure_flow(tmp_path):
    """Regression: CLI run with structures must persist nonzero structure flow.

    Pre-fix, structures were always zero in the GPKG even when configured
    correctly.  Two contributing bugs surfaced during this work:

    1. ``swe2d/extensions/extension_models.py::HydraulicStructureConfig``
       defaults to ``enabled=False`` and the bare-list form of
       ``build_structures_config_from_json`` never overrode it.
    2. ``swe2d/cli/gpkg_adapter.py::build_structures_config_from_json``
       left cfg.enabled at False in the bare-list path.

    With both fixes, a culvert between two cells with nonzero head
    differential must produce a nonzero ``structure/N/flow`` row in the GPKG.
    Requires the C++ build to enable the ``CULVERT_DIAG`` device-side print
    block so we can debug the HDS-5 secant solver when this test fails.
    """
    import os
    import sys

    # Build a tiny mesh: 4 cells along a channel (3×1 = 6 triangles but we
    # use polygon cells to keep the cell count small and stable).
    nx, ny = 4, 1
    stride = nx + 1
    n_cells = nx * ny
    cell_face_offsets = np.arange(0, n_cells * 4 + 1, 4, dtype=np.int32)
    xs = np.linspace(0.0, 40.0, nx + 1)
    ys = np.linspace(0.0, 5.0, ny + 1)
    X, Y = np.meshgrid(xs, ys)
    node_x = X.ravel().copy()
    node_y = Y.ravel().copy()
    node_z = np.zeros_like(node_x)
    cell_face_nodes = np.zeros(n_cells * 4, dtype=np.int32)
    for j in range(ny):
        for i in range(nx):
            c = j * nx + i
            n00 = j * stride + i
            n10 = n00 + 1
            n01 = n00 + stride
            n11 = n10 + stride
            cell_face_nodes[c * 4:c * 4 + 4] = [n00, n10, n11, n01]

    # Determine boundary edges with the same approach as the helper.
    from hydra_swe2d import swe2d_build_mesh_poly, swe2d_boundary_edges
    empty = np.empty(0, dtype=np.int32)
    emptyf = np.empty(0, dtype=np.float64)
    pm_empty = swe2d_build_mesh_poly(
        node_x, node_y, node_z, cell_face_offsets, cell_face_nodes,
        empty, empty, empty, emptyf,
    )
    ei, n0, n1, _tp, _vl, _c0 = swe2d_boundary_edges(pm_empty)
    bc_n0 = np.array(n0, dtype=np.int32).copy()
    bc_n1 = np.array(n1, dtype=np.int32).copy()
    bc_tp = np.zeros(ei.size, dtype=np.int32)
    bc_vl = np.zeros(ei.size, dtype=np.float64)
    for k in range(ei.size):
        a = int(n0[k]); b = int(n1[k])
        if a // stride == b // stride:  # horizontal
            bc_tp[k] = 2 if min(a, b) % stride == 0 else 7  # INFLOW / NORMAL_DEPTH
            bc_vl[k] = 2.0 if bc_tp[k] == 2 else 0.001
        else:
            bc_tp[k] = 1; bc_vl[k] = 0.0  # WALL

    # Place a culvert between cells 0 and 1.  Set an initial water depth so
    # the upstream cell is wet and the culvert has head differential.
    h0 = np.zeros(n_cells, dtype=np.float64)
    h0[0] = 2.0   # upstream cell: 2m of water
    h0[1] = 0.5   # downstream cell: 0.5m of water
    h0[2:] = 0.0

    structures_cfg = {
        "structures": [
            {
                "id": "culvert_0",
                "type": "culvert",
                "upstream_cell": 0,
                "downstream_cell": 1,
                "crest_elev": 0.0,
                "metadata": {
                    "diameter": 1.5,
                    "length": 5.0,
                    "roughness_n": 0.013,
                    "inlet_invert_elev": 0.0,
                    "outlet_invert_elev": 0.0,
                    "entrance_loss_k": 0.5,
                    "exit_loss_k": 1.0,
                    "culvert_code": 1,    # FHWA concrete pipe
                    "culvert_shape": 0,   # circular
                    "culvert_rise": 1.5,
                    "culvert_span": 1.5,
                    "culvert_barrels": 1,
                    "culvert_slope": 0.01,
                },
            },
        ],
    }

    gpkg = str(tmp_path / "struct_cli.gpkg")
    mesh_name = "tiny_struct"
    _run_cli_coupling(
        gpkg, mesh_name,
        node_x, node_y, node_z, cell_face_nodes,
        bc_n0, bc_n1, bc_tp, bc_vl,
        params={},
        duration_s=_T_END,
        q_in=_Q_IN,
        structures_cfg=structures_cfg,
        h0=h0,
    )

    rows = _read_coupling_rows(gpkg)
    struct_flow_keys = [
        k for k in rows
        if k[1] == "structure" and k[2] == "culvert_0" and k[3] == "flow"
    ]
    assert struct_flow_keys, (
        f"no structure/culvert_0/flow row in GPKG. Rows: "
        f"{[k for k in rows if k[1] == 'structure'][:6]}"
    )
    for key in struct_flow_keys:
        _, values = rows[key]
        assert values.size > 0
        # The culvert should produce nonzero flow for at least one timestep
        # given the head differential we set up (h_up=2m, h_down=0.5m).
        # If this fails, rebuild with CULVERT_DIAG=1 to see the device-side
        # diagnostic output from cpp/src/swe2d_gpu.cu:4137:
        #     cmake --build build --target hydra_swe2d
        #   with CULVERT_DIAG defined via:
        #     target_compile_options(hydra_swe2d PRIVATE
        #       $<$<COMPILE_LANGUAGE:CUDA>:-DCULVERT_DIAG=1>)
        # in CMakeLists.txt around line 130.
        assert np.any(np.abs(values) > 1e-3), (
            f"culvert flow is all zeros. Device-side CULVERT_DIAG prints\n"
            f"would show what inputs the kernel saw. Captured subprocess output:\n"
            f"{getattr(sys, 'last_captured', '')}"
            f"\nvalues={values[:5]}"
        )
    # Culvert diagnostic metrics should also be present in GPKG
    diag_metrics = {"inlet_control_flow", "outlet_control_flow", "orifice_cap",
                    "manning_cap", "embankment_flow", "available_head_up", "tailwater_depth"}
    struct_keys = {k for k in rows if k[1] == "structure" and k[2] == "culvert_0"}
    present = {k[3] for k in struct_keys}
    missing = diag_metrics - present
    assert not missing, f"culvert diag metrics missing from GPKG: {missing}"
    for mname in diag_metrics:
        match = [k for k in struct_keys if k[3] == mname]
        assert match, f"key for {mname} not found"
        _, vals = rows[match[0]]
        assert vals.size > 0
        if mname != "embankment_flow":
            assert np.any(np.abs(vals) > 1e-6), f"diag metric {mname} is all zeros"