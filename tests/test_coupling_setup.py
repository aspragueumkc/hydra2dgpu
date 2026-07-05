"""Test for coupling-controller factory extracted from run_controller."""
import numpy as np
import pytest


def test_build_coupling_controller_none_when_no_configs():
    from swe2d.runtime.coupling import build_coupling_controller
    cc = build_coupling_controller(
        pipe_network_cfg=None,
        hydraulic_structures_cfg=None,
        cell_area=np.array([100.0], dtype=np.float64),
        cell_bed=np.array([0.0], dtype=np.float64),
        length_scale_si_to_model=1.0,
        bridge_cuda_coupling=False,
        bridge_stacked_coupling_mode="phase3_spatial",
        culvert_face_flux_mode="off",
        culvert_solver_mode="egl",
        drainage_gpu_method_mode="iterative",
        use_redistribution=False,
        log_fn=lambda msg: None,
    )
    assert cc is None


def test_build_coupling_controller_with_pipe_network():
    from swe2d.extensions.extension_models import (
        PipeNetworkConfig, DrainageNode, DrainageLink,
    )
    from swe2d.runtime.coupling import build_coupling_controller
    node = DrainageNode(node_id="n1", x=0.0, y=0.0, invert_elev=0.0, max_depth=5.0)
    link = DrainageLink(link_id="l1", from_node_id="n1", to_node_id="n1",
                       length=100.0, diameter=1.0, roughness_n=0.013)
    cfg = PipeNetworkConfig(nodes=[node], links=[link], enabled=True)
    cc = build_coupling_controller(
        pipe_network_cfg=cfg,
        hydraulic_structures_cfg=None,
        cell_area=np.array([100.0, 100.0], dtype=np.float64),
        cell_bed=np.array([0.0, 0.0], dtype=np.float64),
        length_scale_si_to_model=1.0,
        bridge_cuda_coupling=False,
        bridge_stacked_coupling_mode="phase3_spatial",
        culvert_face_flux_mode="off",
        culvert_solver_mode="egl",
        drainage_gpu_method_mode="iterative",
        use_redistribution=False,
        log_fn=lambda msg: None,
    )
    assert cc is not None
