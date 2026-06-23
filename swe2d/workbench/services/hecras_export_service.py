"""HEC-RAS HDF5 export service.

Provides HDF5 export of mesh results in a HEC-RAS 2D compatible format
readable by QGIS MDAL.  Pure Python + numpy + h5py — no Qt/PyQt5 imports.

Extracted from
``SWE2DWorkbenchStudioDialog._write_hecras_hdf5`` in
``swe2d.workbench.extracted.topology_and_io_methods`` (Task B2 of
the extracted-methods migration plan).
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

__all__ = ["write_hecras_hdf5"]


def _require_h5py():
    """Raise RuntimeError if h5py is not available."""
    try:
        import h5py as _h5py_local  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "h5py is not installed.  Run: pip install h5py"
        ) from exc
    return __import__("h5py")


def write_hecras_hdf5(
    path: str,
    mesh_data: dict,
    length_unit_name: str = "m",
    is_us_customary: bool = False,
    include_extra: bool = True,
    gravity: float = 9.81,
    h_min: float = 0.01,
    n_mann: float = 0.03,
    timesteps: Optional[list] = None,
    log_fn: Optional[Callable[[str], None]] = None,
    result_data: Optional[dict] = None,
    projection_wkt: str = 'LOCAL_CS["Unknown"]',
) -> None:
    """Write a HEC-RAS 2D compatible HDF5 file readable by QGIS MDAL.

    Parameters
    ----------
    path : str
        Output ``.h5`` file path.
    mesh_data : dict
        Mesh geometry dict with keys ``node_x``, ``node_y``,
        (optional) ``node_z``, and either ``cell_face_offsets`` +
        ``cell_face_nodes`` or ``cell_nodes`` (triangular).
    length_unit_name : str
        Unit label for length (e.g. ``"m"``, ``"ft"``).  Currently
        used only in metadata; the HEC-RAS format stores SI/USC at the
        file level via ``is_us_customary``.
    is_us_customary : bool
        If True, marks the file as US Customary units.
    include_extra : bool
        Whether to write extended outputs (momentum, qmag, wet mask,
        Froude number, Manning's n).
    gravity : float
        Gravitational acceleration in model units.
    h_min : float
        Minimum water depth for wet/dry discrimination.
    n_mann : float
        Manning's n fallback value (used when ``result_data`` does not
        contain per-cell Manning values).
    timesteps : list of (time_seconds, h, hu, hv) or None
        When supplied, results datasets are written; otherwise geometry
        only.
    log_fn : callable or None
        Optional logging function (e.g. ``self._log``).  Called with
        error messages.
    result_data : dict or None
        Optional result-data dict (may contain ``"n_mann_cell"``).
    projection_wkt : str
        CRS WKT string (e.g. from ``QgsCoordinateReferenceSystem.toWkt()``).
    """
    _h5py = _require_h5py()
    if mesh_data is None:
        raise RuntimeError("No mesh data available")

    node_x = mesh_data["node_x"]
    node_y = mesh_data["node_y"]
    node_z = mesh_data.get("node_z", np.zeros_like(node_x))

    # Build dense cell-vertex index array (HEC-RAS FacePoint Indexes,
    # -1 padded to maximum ring length).
    face_offsets = mesh_data.get("cell_face_offsets")
    face_nodes_arr = mesh_data.get("cell_face_nodes")
    cell_nodes_tri = mesh_data.get("cell_nodes")

    if face_offsets is not None and face_nodes_arr is not None:
        offsets = face_offsets.astype(np.int32)
        n_cells = int(offsets.size - 1)
        max_vp = int(max(offsets[i + 1] - offsets[i] for i in range(n_cells)))
        fp_idx = np.full((n_cells, max_vp), -1, dtype=np.int32)
        cell_cx = np.empty(n_cells, dtype=np.float64)
        cell_cy = np.empty(n_cells, dtype=np.float64)
        cell_solver_z = np.empty(n_cells, dtype=np.float64)
        for i in range(n_cells):
            s, e = int(offsets[i]), int(offsets[i + 1])
            ring = face_nodes_arr[s:e].astype(np.int32)
            fp_idx[i, : e - s] = ring
            cell_cx[i] = float(np.mean(node_x[ring]))
            cell_cy[i] = float(np.mean(node_y[ring]))
            cell_solver_z[i] = float(np.mean(node_z[ring]))
    else:
        tri = cell_nodes_tri.reshape(-1, 3).astype(np.int32)
        n_cells = tri.shape[0]
        fp_idx = tri
        cell_cx = np.mean(node_x[tri], axis=1)
        cell_cy = np.mean(node_y[tri], axis=1)
        cell_solver_z = np.mean(node_z[tri], axis=1)

    area_name = "Perimeter 1"

    with _h5py.File(path, "w") as f:
        f.attrs["File Type"] = np.bytes_(b"HEC-RAS Results")
        f.attrs["File Version"] = np.bytes_(b"HEC-RAS 7.0 April 2026")
        f.attrs["Units System"] = np.bytes_(
            b"US Customary" if is_us_customary else b"SI"
        )
        f.attrs["Projection"] = np.bytes_(projection_wkt.encode("utf-8"))

        # ---- Geometry ----
        geo = f.require_group("Geometry")
        geo.attrs["Complete Geometry"] = np.bytes_(b"True")
        geo.attrs["SI Units"] = np.bytes_(b"False" if is_us_customary else b"True")
        geo.attrs["Title"] = np.bytes_(b"Generated Geometry")
        geo.attrs["Version"] = np.bytes_(b"1.0")
        flow_areas_grp = geo.require_group("2D Flow Areas")

        attrs_dt = np.dtype(
            [
                ("Name", "S16"),
                ("Locked", np.uint8),
                ("Mann", np.float32),
                ("Multiple Face Mann n", np.uint8),
                ("Composite LC", np.uint8),
                ("Cell Vol Tol", np.float32),
                ("Cell Min Area Fraction", np.float32),
                ("Face Profile Tol", np.float32),
                ("Face Area Tol", np.float32),
                ("Face Conv Ratio", np.float32),
                ("Laminar Depth", np.float32),
                ("Min Face Length Ratio", np.float32),
                ("Spacing dx", np.float32),
                ("Spacing dy", np.float32),
                ("Shift dx", np.float32),
                ("Shift dy", np.float32),
                ("Cell Count", np.int32),
            ]
        )
        flow_areas_grp.create_dataset(
            "Attributes",
            data=np.array(
                [
                    (
                        area_name.encode(),
                        0,
                        np.float32(0.03),
                        0,
                        0,
                        np.float32(0.01),
                        np.float32(0.01),
                        np.float32(0.01),
                        np.float32(0.01),
                        np.float32(0.02),
                        np.float32(0.2),
                        np.float32(0.05),
                        np.float32(1.0),
                        np.float32(1.0),
                        np.float32(np.nan),
                        np.float32(np.nan),
                        n_cells,
                    )
                ],
                dtype=attrs_dt,
            ),
        )

        area_grp = flow_areas_grp.require_group(area_name)

        # Vertices ("FacePoints" in HEC-RAS 2D parlance)
        area_grp.create_dataset(
            "FacePoints Coordinate",
            data=np.column_stack([node_x, node_y]).astype(np.float64),
        )
        # Cell centroids
        area_grp.create_dataset(
            "Cells Center Coordinate",
            data=np.column_stack([cell_cx, cell_cy]).astype(np.float64),
        )
        # Solver-consistent bed elevation per cell.
        area_grp.create_dataset(
            "Cells Minimum Elevation",
            data=cell_solver_z.astype(np.float32),
        )
        if include_extra:
            if result_data is not None and "n_mann_cell" in result_data:
                n_face = np.asarray(result_data["n_mann_cell"], dtype=np.float64)[:n_cells]
            else:
                n_face = np.full(n_cells, float(n_mann), dtype=np.float64)
            area_grp.create_dataset("Cells Manning n", data=n_face.astype(np.float32))
        # Connectivity: nCells × maxVerts, -1 padded
        area_grp.create_dataset("Cells FacePoint Indexes", data=fp_idx)

        # ---- Results ----
        if timesteps:
            n_t = len(timesteps)
            times_hr = np.array([t / 3600.0 for t, *_ in timesteps], dtype=np.float32)

            ts_base = (
                "Results/Unsteady/Output/Output Blocks/"
                "Base Output/Unsteady Time Series"
            )
            ds_time = f.create_dataset(f"{ts_base}/Time", data=times_hr)
            ds_time.attrs["Number of actual Time Steps"] = np.array([n_t], dtype=np.int32)
            ds_time.attrs["Time"] = np.bytes_(b"Hours")

            # String time stamps (ddMONyyyy HH:MM:SS)
            stamps = []
            for t_s, *_ in timesteps:
                total_min = int(t_s / 60)
                hh, mm = divmod(total_min, 60)
                stamps.append(f"01JAN2000 {hh:02d}:{mm:02d}:00".encode())
            f.create_dataset(
                f"{ts_base}/Time Date Stamp",
                data=np.array(stamps, dtype="S26"),
            )

            depth_arr = np.zeros((n_t, n_cells), dtype=np.float32)
            wse_arr = np.zeros((n_t, n_cells), dtype=np.float32)
            vel_arr = np.zeros((n_t, n_cells), dtype=np.float32)
            vel_u_arr = np.zeros((n_t, n_cells), dtype=np.float32)
            vel_v_arr = np.zeros((n_t, n_cells), dtype=np.float32)
            if include_extra:
                mom_u_arr = np.zeros((n_t, n_cells), dtype=np.float32)
                mom_v_arr = np.zeros((n_t, n_cells), dtype=np.float32)
                qmag_arr = np.zeros((n_t, n_cells), dtype=np.float32)
                wet_arr = np.zeros((n_t, n_cells), dtype=np.float32)
                froude_arr = np.zeros((n_t, n_cells), dtype=np.float32)
                g = float(gravity)

            for ti, (_, h, hu, hv) in enumerate(timesteps):
                h_f = np.asarray(h, dtype=np.float64)[:n_cells]
                hu_f = np.asarray(hu, dtype=np.float64)[:n_cells]
                hv_f = np.asarray(hv, dtype=np.float64)[:n_cells]
                wet = (h_f > h_min)
                hmag = np.maximum(h_f, 1e-12)
                u = np.where(wet, hu_f / hmag, 0.0)
                v = np.where(wet, hv_f / hmag, 0.0)
                depth_arr[ti] = h_f.astype(np.float32)
                wse_arr[ti] = (h_f + cell_solver_z[:n_cells]).astype(np.float32)
                vel_arr[ti] = np.sqrt(u ** 2 + v ** 2).astype(np.float32)
                vel_u_arr[ti] = u.astype(np.float32)
                vel_v_arr[ti] = v.astype(np.float32)
                if include_extra:
                    mom_u_arr[ti] = hu_f.astype(np.float32)
                    mom_v_arr[ti] = hv_f.astype(np.float32)
                    qmag_arr[ti] = np.sqrt(hu_f ** 2 + hv_f ** 2).astype(np.float32)
                    wet_arr[ti] = wet.astype(np.float32)
                    froude_arr[ti] = np.where(wet, np.sqrt(u ** 2 + v ** 2) / np.sqrt(np.maximum(g * h_f, 1.0e-12)), 0.0).astype(np.float32)

            ar = f.require_group(f"{ts_base}/2D Flow Areas/{area_name}")
            ar.create_dataset("Depth", data=depth_arr)
            ar.create_dataset("Water Surface", data=wse_arr)
            ar.create_dataset("Cell Velocity - Magnitude", data=vel_arr)
            ar.create_dataset("Cell Velocity - X", data=vel_u_arr)
            ar.create_dataset("Cell Velocity - Y", data=vel_v_arr)
            # Alias names improve vector pairing across MDAL/QGIS versions.
            ar.create_dataset("Cell Velocity X", data=vel_u_arr)
            ar.create_dataset("Cell Velocity Y", data=vel_v_arr)
            ar.create_dataset("Velocity X", data=vel_u_arr)
            ar.create_dataset("Velocity Y", data=vel_v_arr)
            if include_extra:
                ar.create_dataset("Cell Momentum - X", data=mom_u_arr)
                ar.create_dataset("Cell Momentum - Y", data=mom_v_arr)
                ar.create_dataset("Unit Discharge - Magnitude", data=qmag_arr)
                ar.create_dataset("Wet Mask", data=wet_arr)
                ar.create_dataset("Cell Froude Number", data=froude_arr)

            # MDAL's HEC-RAS reader expects Summary Output to exist when a
            # Results tree is present, even if most summary datasets are not.
            f.require_group(
                "Results/Unsteady/Output/Output Blocks/"
                f"Base Output/Summary Output/2D Flow Areas/{area_name}"
            )
