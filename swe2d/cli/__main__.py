"""CLI entry point: python -m swe2d.cli run mesh.gpkg params.json [--results out.gpkg]"""
import argparse
import json
import sys
import os

from swe2d.cli.headless_runner import execute_run


def _write_error_status(status_file_path: str, exc: Exception) -> None:
    """Write a fatal-error status payload so the batch dialog sees it."""
    if not status_file_path:
        return
    import traceback
    try:
        with open(status_file_path, "w") as f:
            json.dump({
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                "step": 0,
                "t": 0.0,
                "elapsed_s": 0.0,
            }, f)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="HYDRA2DGPU headless runner")
    sub = parser.add_subparsers(dest="command")

    run_parser = sub.add_parser("run", help="Run a single simulation")
    run_parser.add_argument("mesh_gpkg", help="Path to mesh GeoPackage")
    run_parser.add_argument("params", help="Path to JSON params file, or JSON string")
    run_parser.add_argument("--results", "-r", default="", help="Results GeoPackage path")
    run_parser.add_argument("--progress", action="store_true", help="Print progress per step")
    run_parser.add_argument("--status-file-path", default="", help="Periodic JSON status file path (for optional QGIS progress monitoring)")
    run_parser.add_argument("--status-interval", type=float, default=5.0, help="Status file write interval in seconds (default 5.0)")

    batch_parser = sub.add_parser("batch", help="Run batch of simulations")
    batch_parser.add_argument("batch_json", help="Path to batch JSON file")
    batch_parser.add_argument("mesh_gpkg", help="Path to mesh GeoPackage")
    batch_parser.add_argument("--results", "-r", default="", help="Results GeoPackage path")
    batch_parser.add_argument("--max-workers", "-w", type=int, default=0, help="Max concurrent workers (0=auto)")

    replay_parser = sub.add_parser("replay", help="Replay a run from its replay JSON")
    replay_parser.add_argument("--replay-file", type=str, default="", help="Standalone replay JSON file path")
    replay_parser.add_argument("--progress", action="store_true", help="Print progress per step")

    args = parser.parse_args()

    if args.command == "run":
        try:
            params = _load_params(args.params)
        except Exception as exc:
            import traceback
            _write_error_status(args.status_file_path, exc)
            traceback.print_exc()
            sys.exit(1)
        try:
            results = execute_run(
                args.mesh_gpkg,
                params,
                results_gpkg=args.results or "",
                progress_callback=_make_progress(args.progress),
                status_file_path=args.status_file_path or None,
                status_interval_s=float(args.status_interval),
            )
            print(f"Run complete: {results['h'].size} cells, {len(results['diags'])} steps")
            if "max_results" in results:
                h_max = results['max_results']['max_h']
                print(f"Max tracking: h_max range [{h_max.min():.6f}, {h_max.max():.6f}]")
        except Exception as exc:
            import traceback
            _write_error_status(args.status_file_path, exc)
            traceback.print_exc()
            sys.exit(1)

    elif args.command == "batch":
        from swe2d.cli.batch_runner import run_batch
        run_batch(args.batch_json, args.mesh_gpkg, args.results, args.max_workers)

    elif args.command == "replay":
        from swe2d.cli.headless_runner import execute_replay
        try:
            results = execute_replay(
                args.replay_file,
                log_cb=print,
                progress_cb=_make_progress(args.progress),
            )

            class _TupleKeyEncoder(json.JSONEncoder):
                def default(self, o):
                    return str(o)

                def encode(self, o):
                    return super().encode(_normalize_keys(o))

            def _normalize_keys(obj):
                if isinstance(obj, dict):
                    return {str(k) if isinstance(k, tuple) else k: _normalize_keys(v) for k, v in obj.items()}
                if isinstance(obj, (list, tuple)):
                    return [_normalize_keys(v) for v in obj]
                return obj

            print(_TupleKeyEncoder(indent=2).encode(results))
        except Exception as exc:
            import traceback
            traceback.print_exc()
            sys.exit(1)

    else:
        parser.print_help()


def _load_params(param_source: str) -> dict:
    """Load params from JSON file or string."""
    s = str(param_source).strip()
    if os.path.isfile(s):
        with open(s) as f:
            return json.load(f)
    return json.loads(s)


def _make_progress(enabled: bool):
    if not enabled:
        return None
    def cb(pct, diag=None):
        print(f"  progress={pct}%", file=sys.stderr)
    return cb


if __name__ == "__main__":
    main()
