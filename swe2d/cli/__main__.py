"""CLI entry point: python -m swe2d.cli run mesh.gpkg params.json [--results out.gpkg]"""
import argparse
import json
import sys
import os

from swe2d.cli.headless_runner import execute_run


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

    args = parser.parse_args()

    if args.command == "run":
        params = _load_params(args.params)
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

    elif args.command == "batch":
        from swe2d.cli.batch_runner import run_batch
        run_batch(args.batch_json, args.mesh_gpkg, args.results, args.max_workers)

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
    def cb(t, diag):
        print(f"  t={t:.3f}s  dt={diag.get('dt', 0):.5f}  wet={diag.get('wet_cells', -1)}", file=sys.stderr)
    return cb


if __name__ == "__main__":
    main()
