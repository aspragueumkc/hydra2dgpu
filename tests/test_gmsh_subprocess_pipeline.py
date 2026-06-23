"""Test the GMSH topology meshing subprocess pipeline end-to-end.

Exercises the same code path as the GUI topology tab:
  topology_controller → subprocess.Popen → gmsh_subprocess_worker.py
  → _run_topology_mesh_job → GmshBackend.generate()

Run with:  python3 tools/gmsh_topology_mesher.py ...  (headless)
Or this test with the correct PYTHONPATH:
  PYTHONPATH=".:build" python3 -m unittest tests.test_gmsh_subprocess_pipeline
"""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


class TestGmshSubprocessPipeline(unittest.TestCase):
    """Test the GMSH subprocess path using the exact same payload format the topology controller sends."""

    _repo_root: str = ""

    @classmethod
    def setUpClass(cls):
        cls._repo_root = str(Path(__file__).resolve().parent.parent)
        # Check that the worker script exists
        cls._worker = os.path.join(cls._repo_root, "tools", "gmsh_subprocess_worker.py")
        if not os.path.exists(cls._worker):
            raise RuntimeError(f"Worker not found: {cls._worker}")

    def _build_conceptual_model(self):
        """Build the same ConceptualModel the topology controller would after conceptual_from_qgis_layers."""
        from swe2d.mesh.mesh_models import ConceptualModel, ConceptualRegion, ConceptualNode, ConceptualArc

        return ConceptualModel(
            nodes=[
                ConceptualNode(0, 0.0, 0.0),
                ConceptualNode(1, 100.0, 0.0),
                ConceptualNode(2, 100.0, 100.0),
                ConceptualNode(3, 0.0, 100.0),
            ],
            arcs=[
                ConceptualArc(0, 0, 1),
                ConceptualArc(1, 1, 2),
                ConceptualArc(2, 2, 3),
                ConceptualArc(3, 3, 0),
            ],
            constraints=[],
            quad_edges=[],
            regions=[
                ConceptualRegion(
                    region_id=0,
                    ring_xy=[[0, 0], [100, 0], [100, 100], [0, 100], [0, 0]],
                    default_size=10.0,
                    default_cell_type="triangle",
                ),
            ],
        )

    def _build_options(self):
        """Simulate _build_topology_meshing_options() from studio_dialog.py."""
        return {
            "gmsh_verbosity": 0,
            "gmsh_tri_algorithm": 6,
            "gmsh_min_angle_deg": 5.0,
            "gmsh_max_aspect_ratio": 20.0,
            "gmsh_max_non_orth_deg": 82.0,
            "gmsh_min_area_rel_bbox": 1e-14,
            "gmsh_quality_time_limit_s": 55.0,
        }

    def _run_via_subprocess(self, model, backend_name: str, options: dict) -> dict:
        """Run exactly as topology_controller.start_topology_mesh_async() does."""
        _pythonpath = os.pathsep.join([str(p) for p in [self._repo_root, os.path.join(self._repo_root, "build")]])

        in_fd, in_path = tempfile.mkstemp(suffix="_mesh_in.pkl")
        os.close(in_fd)
        out_fd, out_path = tempfile.mkstemp(suffix="_mesh_out.pkl")
        os.close(out_fd)
        err_fd, err_path = tempfile.mkstemp(suffix="_mesh_err.txt")
        os.close(err_fd)

        payload = {
            "conceptual": model,
            "backend_name": backend_name,
            "options": options,
        }
        with open(in_path, "wb") as f:
            pickle.dump(payload, f)

        env = dict(os.environ)
        _existing = env.get("PYTHONPATH", "")
        if _existing:
            _pythonpath = _existing + os.pathsep + _pythonpath
        env["PYTHONPATH"] = _pythonpath

        with open(err_path, "wb") as err_file:
            proc = subprocess.Popen(
                [sys.executable, self._worker, in_path, out_path],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=err_file,
            )

        for _ in range(120):
            if proc.poll() is not None:
                break
            time.sleep(0.5)
        else:
            proc.kill()
            with open(err_path) as f:
                stderr = f.read()
            self.fail(f"TIMEOUT (60s)\nstderr: {stderr[:2000]}")

        if proc.returncode != 0:
            with open(err_path) as f:
                stderr = f.read()
            self.fail(f"returncode={proc.returncode}\nstderr: {stderr[:2000]}")

        with open(out_path, "rb") as f:
            result = pickle.load(f)

        for p in [in_path, out_path, err_path]:
            try:
                os.remove(p)
            except OSError:
                pass

        return result

    def test_gmsh_backend_via_subprocess(self):
        """GMSH backend spawns worker, generates mesh, returns MeshResult."""
        model = self._build_conceptual_model()
        opts = self._build_options()
        result = self._run_via_subprocess(model, "gmsh", opts)
        self.assertTrue(result.get("ok"), result.get("error", ""))
        mesh = result["mesh"]
        self.assertIsNotNone(mesh.cell_nodes)
        self.assertGreater(mesh.cell_nodes.shape[0], 0)
        print(f"GMSH mesh: {mesh.cell_nodes.shape[0]} cells")

    def test_structured_backend_via_subprocess(self):
        """Structured backend spawns worker, generates mesh, returns MeshResult."""
        model = self._build_conceptual_model()
        opts = self._build_options()
        result = self._run_via_subprocess(model, "structured", opts)
        self.assertTrue(result.get("ok"), result.get("error", ""))
        mesh = result["mesh"]
        self.assertIsNotNone(mesh.cell_nodes)
        self.assertGreater(mesh.cell_nodes.shape[0], 0)
        print(f"Structured mesh: {mesh.cell_nodes.shape[0]} cells")


if __name__ == "__main__":
    unittest.main()


# ── QGIS-dependent test ──────────────────────────────────────────────
# Run from QGIS Python console or with QGIS_PYTHONPATH set:
#
#   from swe2d.workbench.studio_dialog import _studio_active_dialog
#   g = _studio_active_dialog._gpkg_path or "/path/to/your.gpkg"
#   import tests.test_gmsh_subprocess_pipeline as t
#   t.test_from_gpkg(g, "gmsh")
#

def load_from_gpkg(gpkg_path: str) -> "ConceptualModel":
    """Load topology layers from a GeoPackage and build a ConceptualModel.

    Requires QGIS (qgis.core).  Call from the QGIS Python console or
    a QGIS-aware test runner.
    """
    from qgis.core import QgsVectorLayer
    from swe2d.mesh.meshing import conceptual_from_qgis_layers

    def _layer(name: str):
        uri = f"{gpkg_path}|layername={name}"
        lyr = QgsVectorLayer(uri, name, "ogr")
        if not lyr.isValid():
            raise RuntimeError(f"Cannot load layer '{name}' from {gpkg_path}")
        return lyr

    return conceptual_from_qgis_layers(
        nodes_layer=_layer("swe2d_topo_nodes"),
        arcs_layer=_layer("swe2d_topo_arcs"),
        regions_layer=_layer("swe2d_topo_regions"),
        constraints_layer=_layer("swe2d_topo_constraints"),
        quad_edges_layer=_layer("swe2d_topo_quad_edges"),
    )


def test_from_gpkg(gpkg_path: str, backend: str = "gmsh") -> None:
    """Run the full subprocess pipeline with topology layers from a GPKG.

    Args:
        gpkg_path: Path to a GeoPackage with swe2d_topo_* layers.
        backend: "gmsh" or "structured"

    Usage from QGIS Python Console:

        gpkg = iface.activeLayer().source().split("|")[0]
        from tests.test_gmsh_subprocess_pipeline import test_from_gpkg
        test_from_gpkg(gpkg, "gmsh")
    """
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from swe2d.mesh.mesh_models import ConceptualModel

    print(f"Loading topology layers from: {gpkg_path}")
    model = load_from_gpkg(gpkg_path)

    regions = getattr(model, "regions", [])
    n_nodes = len(getattr(model, "nodes", []))
    n_arcs = len(getattr(model, "arcs", []))
    print(f"Model: {n_nodes} nodes, {n_arcs} arcs, {len(regions)} regions")

    _repo_root = str(Path(__file__).resolve().parent.parent)
    _worker = _repo_root + "/tools/gmsh_subprocess_worker.py"
    if not os.path.exists(_worker):
        print(f"ERROR: worker not found at {_worker}")
        return

    opts = {
        "gmsh_verbosity": 2,
        "gmsh_tri_algorithm": 6,
        "gmsh_min_angle_deg": 5.0,
        "gmsh_max_aspect_ratio": 20.0,
        "gmsh_max_non_orth_deg": 82.0,
        "gmsh_min_area_rel_bbox": 1e-14,
        "gmsh_quality_time_limit_s": 55.0,
    }

    in_fd, in_path = tempfile.mkstemp(suffix="_mesh_in.pkl")
    os.close(in_fd)
    out_fd, out_path = tempfile.mkstemp(suffix="_mesh_out.pkl")
    os.close(out_fd)
    err_fd, err_path = tempfile.mkstemp(suffix="_mesh_err.txt")
    os.close(err_fd)

    payload = {"conceptual": model, "backend_name": backend, "options": opts}
    with open(in_path, "wb") as f:
        pickle.dump(payload, f)

    _pythonpath = os.pathsep.join([_repo_root, os.path.join(_repo_root, "build")])
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    if existing:
        _pythonpath = existing + os.pathsep + _pythonpath
    env["PYTHONPATH"] = _pythonpath

    with open(err_path, "wb") as err_file:
        proc = subprocess.Popen(
            [_sys.executable, _worker, in_path, out_path],
            env=env, stdout=subprocess.DEVNULL, stderr=err_file,
        )

    print(f"Worker spawned (pid={proc.pid}), waiting...")
    for _ in range(120):
        if proc.poll() is not None:
            break
        time.sleep(0.5)
    else:
        print("TIMEOUT (60s) — killing")
        proc.kill()
        with open(err_path) as f:
            print("stderr:", f.read()[:2000])
        return

    print(f"Return code: {proc.returncode}")
    if proc.returncode != 0:
        with open(err_path) as f:
            print("stderr:", f.read()[:3000])
        return

    with open(out_path, "rb") as f:
        result = pickle.load(f)

    for p in [in_path, out_path, err_path]:
        try:
            os.remove(p)
        except OSError:
            pass

    if result.get("ok"):
        mesh = result["mesh"]
        print(f"SUCCESS: {mesh.cell_nodes.shape[0]} cells generated")
    else:
        print(f"FAILURE: {result.get('error', 'unknown')}")
        tb = result.get("traceback", "")
        if tb:
            print(tb[:2000])
