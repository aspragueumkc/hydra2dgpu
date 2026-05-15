#!/usr/bin/env python3
"""
SWE2D GPU Drainage Coupling Benchmark Harness

Runs identical models with step vs iterative drainage methods and compares:
- Total coupling time
- GPU utilization
- Implicit iterations used
- Network convergence speed
"""

import os
import sys
import sqlite3
import argparse
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

def query_run_logs(gpkg_path: str, run_id: int) -> dict:
    """Extract coupling diagnostics from swe2d_run_logs table."""
    conn = sqlite3.connect(gpkg_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT log_text FROM swe2d_run_logs 
            WHERE run_id = ? ORDER BY id ASC
        """, (run_id,))
        rows = cursor.fetchall()
    finally:
        conn.close()
    
    if not rows:
        return {}
    
    # Parse log entries for coupling diagnostics
    diag_data = {
        "total_timesteps": 0,
        "coupling_timesteps": 0,
        "coupling_time_s": 0.0,
        "gpu_util_avg": 0.0,
        "native_iter_count": 0,
        "inactive_fastpath_count": 0,
        "max_implicit_iters": 0,
        "max_substeps": 0,
        "limiter_events_total": 0.0,
        "coupling_log_lines": 0,
    }
    
    coupling_times = []
    gpu_utils = []
    implicit_iter_counts = []
    
    for (log_text,) in rows:
        if log_text is None:
            continue
        
        diag_data["coupling_log_lines"] += 1
        
        # Try to parse as JSON (component sums line)
        try:
            data = json.loads(log_text)
            if "component_sums" in data:
                sums = data["component_sums"]
                
                # Coupling timing
                if "coupling_time_s" in sums:
                    diag_data["coupling_time_s"] += float(sums["coupling_time_s"])
                    coupling_times.append(float(sums["coupling_time_s"]))
                
                # GPU util
                if "gpu_frac" in sums:
                    gpu_utils.append(float(sums.get("gpu_frac", 0.0)))
                
                # Drainage diagnostics
                if "drainage_native_iterative" in sums:
                    if float(sums["drainage_native_iterative"]) > 0:
                        diag_data["native_iter_count"] += 1
                
                if "drainage_inactive_fastpath" in sums:
                    if float(sums["drainage_inactive_fastpath"]) > 0:
                        diag_data["inactive_fastpath_count"] += 1
                
                if "drainage_implicit_iters_used" in sums:
                    iters = int(float(sums["drainage_implicit_iters_used"]))
                    diag_data["max_implicit_iters"] = max(diag_data["max_implicit_iters"], iters)
                
                if "drainage_substeps_used" in sums:
                    steps = int(float(sums["drainage_substeps_used"]))
                    diag_data["max_substeps"] = max(diag_data["max_substeps"], steps)
                
                if "drainage_limiter_events" in sums:
                    diag_data["limiter_events_total"] += float(sums["drainage_limiter_events"])
                
                diag_data["coupling_timesteps"] += 1
            
            if "timestep_num" in data:
                diag_data["total_timesteps"] = max(diag_data["total_timesteps"], 
                                                   int(data["timestep_num"]))
        except (json.JSONDecodeError, ValueError, KeyError):
            # Not a JSON log line, skip
            pass
    
    # Compute averages
    if coupling_times:
        diag_data["coupling_time_avg_s"] = sum(coupling_times) / len(coupling_times)
    
    if gpu_utils:
        diag_data["gpu_util_avg"] = sum(gpu_utils) / len(gpu_utils)
    
    return diag_data


def compare_runs(step_run_id: int, iterative_run_id: int, gpkg_path: str) -> dict:
    """Compare two runs (step vs iterative methods)."""
    step_diag = query_run_logs(gpkg_path, step_run_id)
    iter_diag = query_run_logs(gpkg_path, iterative_run_id)
    
    comparison = {
        "step_run_id": step_run_id,
        "iterative_run_id": iterative_run_id,
        "timestamp": datetime.now().isoformat(),
        "step_method": step_diag,
        "iterative_method": iter_diag,
        "improvements": {},
    }
    
    # Calculate improvements
    if step_diag.get("coupling_time_s", 0) > 0:
        improvement_pct = ((step_diag["coupling_time_s"] - iter_diag.get("coupling_time_s", 0)) 
                          / step_diag["coupling_time_s"] * 100)
        comparison["improvements"]["coupling_time_reduction_pct"] = improvement_pct
    
    if step_diag.get("coupling_time_avg_s", 0) > 0:
        avg_improvement_pct = ((step_diag.get("coupling_time_avg_s", 0) 
                               - iter_diag.get("coupling_time_avg_s", 0)) 
                              / step_diag.get("coupling_time_avg_s", 1.0) * 100)
        comparison["improvements"]["avg_coupling_time_reduction_pct"] = avg_improvement_pct
    
    if step_diag.get("gpu_util_avg", 0) > 0:
        gpu_improvement = ((iter_diag.get("gpu_util_avg", 0) - step_diag["gpu_util_avg"]) 
                          / step_diag["gpu_util_avg"] * 100)
        comparison["improvements"]["gpu_util_improvement_pct"] = gpu_improvement
    
    comparison["improvements"]["native_iter_usage"] = iter_diag.get("native_iter_count", 0)
    comparison["improvements"]["inactive_fastpath_hits"] = iter_diag.get("inactive_fastpath_count", 0)
    comparison["improvements"]["max_implicit_iters"] = iter_diag.get("max_implicit_iters", 0)
    comparison["improvements"]["max_substeps"] = iter_diag.get("max_substeps", 0)
    
    return comparison


def format_comparison_report(comparison: dict) -> str:
    """Format comparison results as human-readable report."""
    lines = []
    lines.append("=" * 70)
    lines.append("SWE2D GPU DRAINAGE COUPLING A/B BENCHMARK")
    lines.append("=" * 70)
    lines.append(f"\nComparison Time: {comparison['timestamp']}")
    lines.append(f"Step Method Run ID: {comparison['step_run_id']}")
    lines.append(f"Iterative Method Run ID: {comparison['iterative_run_id']}")
    
    lines.append("\n" + "─" * 70)
    lines.append("STEP METHOD")
    lines.append("─" * 70)
    step = comparison["step_method"]
    lines.append(f"Total Coupling Time:     {step.get('coupling_time_s', 0):.3f} s")
    lines.append(f"Avg Coupling Time/Step:  {step.get('coupling_time_avg_s', 0):.6f} s")
    lines.append(f"GPU Utilization (avg):   {step.get('gpu_util_avg', 0)*100:.1f}%")
    lines.append(f"Coupling Timesteps:      {step.get('coupling_timesteps', 0)}")
    lines.append(f"Max Implicit Iters:      {step.get('max_implicit_iters', 0)}")
    lines.append(f"Max Substeps:            {step.get('max_substeps', 0)}")
    
    lines.append("\n" + "─" * 70)
    lines.append("ITERATIVE METHOD (NATIVE)")
    lines.append("─" * 70)
    it = comparison["iterative_method"]
    lines.append(f"Total Coupling Time:     {it.get('coupling_time_s', 0):.3f} s")
    lines.append(f"Avg Coupling Time/Step:  {it.get('coupling_time_avg_s', 0):.6f} s")
    lines.append(f"GPU Utilization (avg):   {it.get('gpu_util_avg', 0)*100:.1f}%")
    lines.append(f"Native Iter Uses:        {it.get('native_iter_count', 0)}")
    lines.append(f"Inactive Fastpath Hits:  {it.get('inactive_fastpath_count', 0)}")
    lines.append(f"Max Implicit Iters:      {it.get('max_implicit_iters', 0)}")
    lines.append(f"Max Substeps:            {it.get('max_substeps', 0)}")
    
    lines.append("\n" + "─" * 70)
    lines.append("IMPROVEMENTS (ITERATIVE vs STEP)")
    lines.append("─" * 70)
    imp = comparison["improvements"]
    lines.append(f"Total Coupling Time:     {imp.get('coupling_time_reduction_pct', 0):.1f}% faster")
    lines.append(f"Avg Coupling Time/Step:  {imp.get('avg_coupling_time_reduction_pct', 0):.1f}% faster")
    lines.append(f"GPU Utilization:         {imp.get('gpu_util_improvement_pct', 0):.1f}% change")
    
    lines.append("=" * 70)
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="SWE2D GPU Drainage Coupling Benchmark")
    parser.add_argument("--gpkg", required=True, help="Path to GeoPackage with swe2d_run_logs")
    parser.add_argument("--step-run", type=int, required=True, help="Run ID for step method")
    parser.add_argument("--iterative-run", type=int, required=True, help="Run ID for iterative method")
    parser.add_argument("--output", help="Output JSON file for detailed results")
    
    args = parser.parse_args()
    
    gpkg_path = Path(args.gpkg)
    if not gpkg_path.exists():
        print(f"Error: GeoPackage not found: {gpkg_path}", file=sys.stderr)
        return 1
    
    print(f"Analyzing runs from: {gpkg_path}")
    print(f"  Step method (run {args.step_run})")
    print(f"  Iterative method (run {args.iterative_run})")
    
    try:
        comparison = compare_runs(args.step_run, args.iterative_run, str(gpkg_path))
        print(format_comparison_report(comparison))
        
        if args.output:
            with open(args.output, "w") as f:
                json.dump(comparison, f, indent=2)
            print(f"\nDetailed results saved to: {args.output}")
        
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
