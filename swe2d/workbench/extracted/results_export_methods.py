from __future__ import annotations

import datetime
import os
import sqlite3
from typing import Dict, List, Optional

from swe2d_workbench_qt import *  # type: ignore F401,F403


def load_coupling_results_from_geopackage(
    self,
    gpkg_path: str,
    run_id: Optional[str] = None,
) -> tuple[str, List[Dict[str, object]]]:
    if not gpkg_path or not os.path.exists(gpkg_path):
        return "", []
    conn = sqlite3.connect(gpkg_path)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='swe2d_coupling_results'"
        )
        if cur.fetchone() is None:
            return "", []

        chosen = str(run_id or "").strip()
        if not chosen:
            cur.execute(
                """
                SELECT run_id FROM swe2d_coupling_results_runs
                ORDER BY datetime(created_utc) DESC, rowid DESC
                LIMIT 1
                """
            )
            row = cur.fetchone()
            if row is None:
                return "", []
            chosen = str(row[0])

        cur.execute(
            """
            SELECT t_s, component, object_id, object_name, metric, value
            FROM swe2d_coupling_results
            WHERE run_id = ?
            ORDER BY t_s ASC, component ASC, metric ASC, object_id ASC
            """,
            (chosen,),
        )
        rows: List[Dict[str, object]] = []
        for t_s, component, object_id, object_name, metric, value in cur.fetchall():
            rows.append(
                {
                    "t_s": float(t_s),
                    "component": str(component or ""),
                    "object_id": str(object_id or ""),
                    "object_name": str(object_name or ""),
                    "metric": str(metric or ""),
                    "value": float(value),
                }
            )
        return chosen, rows
    finally:
        conn.close()


def open_coupling_results_viewer(self) -> None:
    db_path = ""
    if self._coupling_results_latest_db_path and os.path.exists(self._coupling_results_latest_db_path):
        db_path = self._coupling_results_latest_db_path
    if not db_path:
        db_path = self._current_line_results_storage_path()
    if not db_path:
        self._log("No GeoPackage available for coupling results viewer.")
        return

    run_id = self._coupling_results_latest_run_id or None
    chosen, rows = self._load_coupling_results_from_geopackage(db_path, run_id=run_id)
    if not chosen or not rows:
        self._log("No drainage/structure coupling results found in GeoPackage yet.")
        return

    dlg = SWE2DCouplingResultsViewerDialog(
        records=rows,
        run_id=chosen,
        db_path=db_path,
        length_unit=self._length_unit_name,
        flow_unit_label=self._flow_unit_label(),
        parent=self,
    )
    dlg.exec()


def persist_mesh_results_to_geopackage(
    self,
    gpkg_path: str,
    run_id: str,
    mesh_rows: List[Dict[str, object]],
    interval_s: float,
) -> None:
    if not gpkg_path or not mesh_rows:
        return
    conn = sqlite3.connect(gpkg_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS swe2d_mesh_results_runs (
                run_id TEXT PRIMARY KEY,
                created_utc TEXT,
                interval_s REAL,
                row_count INTEGER
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS swe2d_mesh_results (
                run_id TEXT,
                t_s REAL,
                cell_id INTEGER,
                h REAL,
                hu REAL,
                hv REAL,
                PRIMARY KEY (run_id, t_s, cell_id)
            )
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_swe2d_mesh_results_run_t_cell "
            "ON swe2d_mesh_results(run_id, t_s, cell_id)"
        )
        cur.execute("DELETE FROM swe2d_mesh_results WHERE run_id = ?", (run_id,))
        cur.execute(
            """
            INSERT OR REPLACE INTO swe2d_mesh_results_runs
            (run_id, created_utc, interval_s, row_count)
            VALUES (?, ?, ?, ?)
            """,
            (
                str(run_id),
                datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
                float(interval_s),
                int(len(mesh_rows)),
            ),
        )
        cur.executemany(
            """
            INSERT OR REPLACE INTO swe2d_mesh_results
            (run_id, t_s, cell_id, h, hu, hv)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    str(run_id),
                    float(r.get("t_s", 0.0)),
                    int(r.get("cell_id", -1)),
                    float(r.get("h", 0.0)),
                    float(r.get("hu", 0.0)),
                    float(r.get("hv", 0.0)),
                )
                for r in mesh_rows
            ],
        )
        conn.commit()
        support = self._velocity_data_support_for_run(gpkg_path, run_id, "swe2d_mesh_results")
        if int(support.get("face_rows", 0)) > 0:
            self._log(
                "Velocity persistence check: both cell-centered and face-centered data are present "
                f"(run_id={run_id}, cell_rows={int(support.get('cell_rows', 0))}, "
                f"face_table={support.get('face_table')}, face_rows={int(support.get('face_rows', 0))})."
            )
        else:
            self._log(
                "Velocity persistence check: only cell-centered h/hu/hv rows were stored for this run; "
                "no face-centered flux rows were found in GeoPackage tables "
                "(swe2d_face_flux_results / swe2d_face_results / swe2d_flux_faces)."
            )
        self._log(
            f"Stored mesh snapshot results in GeoPackage: {gpkg_path} "
            f"(run_id={run_id}, rows={len(mesh_rows)})"
        )
    finally:
        conn.close()


def collect_run_log_metadata(self) -> Dict[str, object]:
    gate_cfg = dict(getattr(self, "_swe3d_geom_gate_last_config", {}) or {})
    gate_metrics = dict(getattr(self, "_swe3d_geom_gate_last_metrics", {}) or {})
    gate_violations = [str(v) for v in (getattr(self, "_swe3d_geom_gate_last_violations", []) or [])]

    if not gate_cfg:
        def _env_bool(name: str, default: bool) -> bool:
            raw = str(os.environ.get(name, "")).strip().lower()
            if not raw:
                return bool(default)
            return raw not in ("0", "false", "no", "off")

        def _env_float(name: str, default: float) -> float:
            try:
                return float(os.environ.get(name, str(default)))
            except Exception:
                return float(default)

        def _env_int(name: str, default: int) -> int:
            try:
                return int(os.environ.get(name, str(default)))
            except Exception:
                return int(default)

        gate_cfg = {
            "strict": _env_bool("BACKWATER_SWE3D_GEOM_STRICT", False),
            "max_solid_fraction": max(0.0, min(1.0, _env_float("BACKWATER_SWE3D_GEOM_MAX_SOLID_FRACTION", 0.98))),
            "max_seed_leak_fallbacks": max(0, _env_int("BACKWATER_SWE3D_GEOM_MAX_SEED_LEAK_FALLBACKS", 0)),
        }

    metadata: Dict[str, object] = {
        "swe3d_geometry_gate": {
            "strict": bool(gate_cfg.get("strict", False)),
            "max_solid_fraction": float(gate_cfg.get("max_solid_fraction", 0.98)),
            "max_seed_leak_fallbacks": int(gate_cfg.get("max_seed_leak_fallbacks", 0)),
            "violation_count": int(len(gate_violations)),
        }
    }
    if gate_metrics:
        metadata["swe3d_geometry_gate"]["metrics"] = gate_metrics
    if gate_violations:
        metadata["swe3d_geometry_gate"]["violations"] = gate_violations
    return metadata


def open_run_log_viewer(self) -> None:
    db_path = ""
    if self._run_log_latest_db_path and os.path.exists(self._run_log_latest_db_path):
        db_path = self._run_log_latest_db_path
    if not db_path:
        db_path = self._current_line_results_storage_path()
    if not db_path:
        self._log("No GeoPackage available for run log viewer.")
        return
    records = self._load_run_logs_from_geopackage(db_path)
    if not records:
        self._log("No saved run logs found in GeoPackage yet.")
        return
    dlg = SWE2DRunLogViewerDialog(
        records=records,
        run_id=self._run_log_latest_run_id,
        db_path=db_path,
        parent=self,
    )
    dlg.exec()


def export_mesh_to_hdf5(self) -> None:
    if self._mesh_data is None:
        self._on_generate_mesh()
    if self._mesh_data is None:
        return
    out_path, _ = QtWidgets.QFileDialog.getSaveFileName(
        self,
        "Save Mesh As HEC-RAS HDF5",
        "swe2d_mesh.hdf",
        "HEC-RAS HDF5 (*.hdf)",
    )
    if not out_path:
        return
    try:
        out_path = self._normalize_hecras_hdf_path(out_path)
        self._write_hecras_hdf5(out_path)
        n_nodes = int(self._mesh_data["node_x"].shape[0])
        self._log(f"Saved HEC-RAS HDF5 mesh: {out_path} (nodes={n_nodes})")
        self.layer_status_lbl.setText("Mesh saved to HEC-RAS HDF5.")
    except (OSError, RuntimeError, ValueError) as exc:
        QtWidgets.QMessageBox.critical(self, "HDF5 Export", f"Export failed:\n{exc}")


def export_results_to_hdf5(self) -> None:
    if self._mesh_data is None or not self._snapshot_timesteps:
        self._log("Run the model first (snapshots must be captured) to export HDF5 results.")
        return
    out_path, _ = QtWidgets.QFileDialog.getSaveFileName(
        self,
        "Save Results As HEC-RAS HDF5",
        "swe2d_results.hdf",
        "HEC-RAS HDF5 (*.hdf)",
    )
    if not out_path:
        return
    try:
        out_path = self._normalize_hecras_hdf_path(out_path)
        self._write_hecras_hdf5(out_path, timesteps=self._snapshot_timesteps)
        n_ts = len(self._snapshot_timesteps)
        self._log(f"Saved HEC-RAS HDF5 results: {out_path} ({n_ts} timesteps)")
        self.layer_status_lbl.setText(f"Results saved to HEC-RAS HDF5 ({n_ts} timesteps).")
    except (OSError, RuntimeError, ValueError) as exc:
        QtWidgets.QMessageBox.critical(self, "HDF5 Export", f"Export failed:\n{exc}")


def export_results_to_ugrid(self) -> None:
    if self._mesh_data is None or not self._snapshot_timesteps:
        self._log("Run the model first (snapshots must be captured) to export UGRID results.")
        return
    out_path, _ = QtWidgets.QFileDialog.getSaveFileName(
        self,
        "Save Results As UGRID NetCDF",
        "swe2d_results.nc",
        "UGRID NetCDF (*.nc)",
    )
    if not out_path:
        return
    if not out_path.lower().endswith(".nc"):
        out_path += ".nc"
    try:
        self._write_ugrid_nc(out_path, timesteps=self._snapshot_timesteps)
        n_ts = len(self._snapshot_timesteps)
        self._log(f"Saved UGRID NetCDF results: {out_path} ({n_ts} timesteps)")
        self.layer_status_lbl.setText(f"Results saved to UGRID NetCDF ({n_ts} timesteps).")
    except (OSError, RuntimeError, ValueError) as exc:
        QtWidgets.QMessageBox.critical(self, "UGRID Export", f"Export failed:\n{exc}")