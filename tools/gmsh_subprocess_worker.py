"""GMSH subprocess worker — runs mesh generation in a clean Python child process.

Reads a pickled input dict, runs the mesh, writes pickled result.
Called via subprocess.Popen from QGIS to avoid Qt/threading conflicts.

Forwards Gmsh logger output to stderr so the parent process can tail it
and display in the UI log.
"""
import pickle
import sys
import threading
import time
import traceback

from swe2d.mesh.meshing import _run_topology_mesh_job


def _forward_gmsh_log_to_stderr(stop_event):
    """Poll gmsh.logger.get() and write new messages to stderr.

    Runs in a daemon thread alongside _run_topology_mesh_job.
    """
    try:
        import gmsh
    except ImportError:
        return
    seen = set()
    while not stop_event.is_set():
        try:
            msgs = list(gmsh.logger.get())
        except Exception:
            msgs = []
        for msg in msgs:
            msg_str = str(msg).strip()
            if msg_str and msg_str not in seen:
                seen.add(msg_str)
                print(msg_str, file=sys.stderr, flush=True)
        time.sleep(0.25)


def main():
    in_path = sys.argv[1]
    out_path = sys.argv[2]

    with open(in_path, "rb") as f:
        data = pickle.load(f)

    # Start a daemon thread that forwards Gmsh logger messages to stderr.
    # The parent process tails this stderr file and displays it in the UI.
    stop_event = threading.Event()
    log_thread = threading.Thread(
        target=_forward_gmsh_log_to_stderr,
        args=(stop_event,),
        daemon=True,
    )
    log_thread.start()

    try:
        mesh = _run_topology_mesh_job(
            conceptual=data["conceptual"],
            backend_name=data["backend_name"],
            options=data.get("options"),
        )
        result = {"ok": True, "mesh": mesh, "error": None}
    except Exception as exc:
        result = {"ok": False, "mesh": None, "error": str(exc), "traceback": traceback.format_exc()}
    finally:
        stop_event.set()
        log_thread.join(timeout=2.0)

    with open(out_path, "wb") as f:
        pickle.dump(result, f)


if __name__ == "__main__":
    main()
