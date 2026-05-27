import builtins
import importlib.util
import sys
import types

import numpy as np
import pytest

import swe2d.mesh.meshing as meshing
from swe2d.mesh.meshing import ConceptualModel, ConceptualRegion, generate_face_centric_mesh


def _simple_square_model() -> ConceptualModel:
    return ConceptualModel(
        nodes=[],
        arcs=[],
        regions=[
            ConceptualRegion(
                region_id=1,
                ring_xy=[(0.0, 0.0), (40.0, 0.0), (40.0, 20.0), (0.0, 20.0)],
                default_size=10.0,
                default_cell_type="triangular",
            )
        ],
        constraints=[],
        quad_edges=[],
    )


def test_mfem_post_opt_missing_module_non_strict_warns_and_returns_mesh(monkeypatch):
    model = _simple_square_model()
    orig_import = builtins.__import__

    def _fail_real_optimizer(*args, **kwargs):
        raise RuntimeError("mocked mesh-optimizer failure")

    def _guarded_import(name, *args, **kwargs):
        if name == "hydra_mfem_meshopt":
            raise ModuleNotFoundError("mocked missing hydra_mfem_meshopt")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(meshing, "optimize_with_mfem", _fail_real_optimizer)
    monkeypatch.setattr(builtins, "__import__", _guarded_import)

    with pytest.warns(RuntimeWarning, match="MFEM mesh-optimizer execution failed"):
        mesh = generate_face_centric_mesh(
            model,
            backend="structured",
            options={
                "post_opt_backend": "mfem_tmop",
                "mfem_post_opt_enable": True,
                "mfem_post_opt_strict": False,
            },
        )

    assert mesh.node_x.size > 0
    assert mesh.cell_face_offsets.size > 1


def test_mfem_post_opt_missing_module_strict_raises(monkeypatch):
    model = _simple_square_model()
    orig_import = builtins.__import__

    def _fail_real_optimizer(*args, **kwargs):
        raise RuntimeError("mocked mesh-optimizer failure")

    def _guarded_import(name, *args, **kwargs):
        if name == "hydra_mfem_meshopt":
            raise ModuleNotFoundError("mocked missing hydra_mfem_meshopt")
        return orig_import(name, *args, **kwargs)

    def _no_local_spec(*args, **kwargs):
        raise RuntimeError("mocked no local hydra_mfem_meshopt spec")

    monkeypatch.setattr(meshing, "optimize_with_mfem", _fail_real_optimizer)
    monkeypatch.setattr(builtins, "__import__", _guarded_import)
    monkeypatch.setattr(importlib.util, "spec_from_file_location", _no_local_spec)

    with pytest.raises(RuntimeError, match="MFEM mesh-optimizer execution failed"):
        generate_face_centric_mesh(
            model,
            backend="structured",
            options={
                "post_opt_backend": "mfem_tmop",
                "mfem_post_opt_enable": True,
                "mfem_post_opt_strict": True,
            },
        )


def test_mfem_post_opt_fake_module_path(monkeypatch):
    model = _simple_square_model()

    def _optimize_mesh_tmop(**kwargs):
        node_x = np.asarray(kwargs["node_x"], dtype=np.float64).copy()
        # Simulate an optimizer by applying a tiny deterministic perturbation.
        node_x += 1.0e-6
        return {
            "node_x": node_x,
            "node_y": np.asarray(kwargs["node_y"], dtype=np.float64),
            "cell_face_offsets": np.asarray(kwargs["cell_face_offsets"], dtype=np.int32),
            "cell_face_nodes": np.asarray(kwargs["cell_face_nodes"], dtype=np.int32),
        }

    fake_module = types.SimpleNamespace(optimize_mesh_tmop=_optimize_mesh_tmop)
    monkeypatch.setitem(sys.modules, "hydra_mfem_meshopt", fake_module)

    mesh = generate_face_centric_mesh(
        model,
        backend="structured",
        options={
            "post_opt_backend": "mfem_tmop",
            "mfem_post_opt_enable": True,
            "mfem_post_opt_strict": True,
        },
    )

    assert mesh.node_x.size > 0
    assert np.isfinite(mesh.node_x).all()
