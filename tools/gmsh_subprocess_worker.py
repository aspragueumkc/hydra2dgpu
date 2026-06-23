"""GMSH subprocess worker — runs mesh generation in a clean Python child process.

Reads a pickled input dict, runs the mesh, writes pickled result.
Called via subprocess.Popen from QGIS to avoid Qt/threading conflicts.
"""
import pickle
import sys
import traceback

from swe2d.mesh.meshing import _run_topology_mesh_job


def main():
    in_path = sys.argv[1]
    out_path = sys.argv[2]

    with open(in_path, "rb") as f:
        data = pickle.load(f)

    try:
        mesh = _run_topology_mesh_job(
            conceptual=data["conceptual"],
            backend_name=data["backend_name"],
            options=data.get("options"),
        )
        result = {"ok": True, "mesh": mesh, "error": None}
    except Exception as exc:
        result = {"ok": False, "mesh": None, "error": str(exc), "traceback": traceback.format_exc()}

    with open(out_path, "wb") as f:
        pickle.dump(result, f)


if __name__ == "__main__":
    main()
