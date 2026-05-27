import numpy as np

from swe2d.extensions.extension_models import HydraulicStructure, HydraulicStructureConfig, StructureType
from swe2d.runtime.bridge_stacked_runtime import build_bridge_stacked_plans_for_runtime


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


def test_build_bridge_stacked_plans_for_runtime_generates_plan_for_stacked_bridge():
    mesh = _toy_mesh()
    cfg = HydraulicStructureConfig(
        enabled=True,
        structures=[
            HydraulicStructure(
                structure_id="B0",
                structure_type=StructureType.BRIDGE,
                upstream_cell=0,
                downstream_cell=1,
                crest_elev=10.0,
                metadata={
                    "stacked_enabled": 1.0,
                    "axis_x0": 0.5,
                    "axis_y0": 1.5,
                    "axis_x1": 5.5,
                    "axis_y1": 1.5,
                    "influence_width_m": 2.0,
                    "deck_soffit_elev": 11.0,
                    "deck_top_elev": 12.0,
                    "model_top_elev": 14.0,
                    "under_layers": 2,
                    "over_layers": 1,
                },
            )
        ],
    )

    plans = build_bridge_stacked_plans_for_runtime(mesh, cfg)
    assert len(plans) == 1
    assert plans[0].selected_cells.size > 0


def test_build_bridge_stacked_plans_for_runtime_returns_empty_without_stacked_enabled():
    mesh = _toy_mesh()
    cfg = HydraulicStructureConfig(
        enabled=True,
        structures=[
            HydraulicStructure(
                structure_id="B1",
                structure_type=StructureType.BRIDGE,
                upstream_cell=0,
                downstream_cell=1,
                crest_elev=10.0,
                metadata={
                    "stacked_enabled": 0.0,
                    "axis_x0": 0.5,
                    "axis_y0": 1.5,
                    "axis_x1": 5.5,
                    "axis_y1": 1.5,
                },
            )
        ],
    )

    plans = build_bridge_stacked_plans_for_runtime(mesh, cfg)
    assert plans == []
