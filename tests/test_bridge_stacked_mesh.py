import numpy as np

from swe2d.extensions.extension_models import HydraulicStructure, HydraulicStructureConfig, StructureType
from swe2d.mesh.bridge_stacked_mesh import (
    BridgeStackedGeometrySpec,
    bridge_specs_from_structure_config,
    build_bridge_stacked_plan,
)


def _toy_mesh(nx: int = 6, ny: int = 4):
    xs, ys = np.meshgrid(np.arange(nx + 1, dtype=float), np.arange(ny + 1, dtype=float))
    node_x = xs.ravel()
    node_y = ys.ravel()
    node_z = np.zeros_like(node_x)

    faces = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            n1 = n0 + 1
            n3 = n0 + (nx + 1)
            n2 = n3 + 1
            faces.append([n0, n1, n2, n3])

    flat_nodes = []
    offsets = [0]
    for f in faces:
        flat_nodes.extend(f)
        offsets.append(len(flat_nodes))

    return {
        "node_x": np.asarray(node_x, dtype=np.float64),
        "node_y": np.asarray(node_y, dtype=np.float64),
        "node_z": np.asarray(node_z, dtype=np.float64),
        "cell_face_offsets": np.asarray(offsets, dtype=np.int32),
        "cell_face_nodes": np.asarray(flat_nodes, dtype=np.int32),
    }


def test_bridge_stacked_plan_selects_cells_and_builds_layers():
    mesh = _toy_mesh()
    spec = BridgeStackedGeometrySpec(
        structure_id="B0",
        p0_xy=(1.0, 1.5),
        p1_xy=(5.0, 1.5),
        influence_width_m=2.0,
        deck_soffit_elev_m=3.0,
        deck_top_elev_m=4.0,
        model_top_elev_m=6.0,
        under_layers=3,
        over_layers=2,
    )

    plans = build_bridge_stacked_plan(mesh, [spec])
    assert len(plans) == 1
    plan = plans[0]

    assert plan.structure_id == "B0"
    assert plan.selected_cells.size > 0
    assert plan.layer_bottom_m.size == 5
    assert plan.layer_top_m.size == 5
    assert int(np.sum(plan.layer_role == 0)) == 3
    assert int(np.sum(plan.layer_role == 2)) == 2
    assert np.all(plan.opening_fraction >= 0.0)
    assert np.all(plan.opening_fraction <= 1.0)


def test_bridge_stacked_plan_blocks_pier_bands():
    mesh = _toy_mesh(nx=12, ny=3)
    spec = BridgeStackedGeometrySpec(
        structure_id="B1",
        p0_xy=(1.0, 1.5),
        p1_xy=(11.0, 1.5),
        influence_width_m=6.0,
        pier_count=2,
        pier_width_m=1.0,
        deck_soffit_elev_m=2.0,
        deck_top_elev_m=3.0,
        model_top_elev_m=4.0,
    )

    plans = build_bridge_stacked_plan(mesh, [spec])
    assert len(plans) == 1
    plan = plans[0]

    assert plan.effective_opening_width_m == 4.0
    assert np.any(plan.opening_fraction == 0.0)
    assert np.any(plan.opening_fraction == 1.0)


def test_bridge_specs_from_structure_config_reads_axis_and_stacked_fields():
    cfg = HydraulicStructureConfig(
        enabled=True,
        structures=[
            HydraulicStructure(
                structure_id="B2",
                structure_type=StructureType.BRIDGE,
                upstream_cell=0,
                downstream_cell=1,
                crest_elev=10.0,
                metadata={
                    "stacked_enabled": 1.0,
                    "axis_x0": 0.0,
                    "axis_y0": 1.0,
                    "axis_x1": 10.0,
                    "axis_y1": 1.0,
                    "influence_width_m": 8.0,
                    "under_layers": 4,
                    "over_layers": 2,
                    "pier_count": 2,
                    "pier_width": 1.0,
                    "deck_soffit_elev": 12.0,
                    "deck_top_elev": 13.0,
                    "model_top_elev": 16.0,
                },
            )
        ],
    )

    specs = bridge_specs_from_structure_config(cfg)
    assert len(specs) == 1
    s = specs[0]
    assert s.structure_id == "B2"
    assert s.p0_xy == (0.0, 1.0)
    assert s.p1_xy == (10.0, 1.0)
    assert s.influence_width_m == 8.0
    assert s.under_layers == 4
    assert s.over_layers == 2
    assert s.pier_count == 2
