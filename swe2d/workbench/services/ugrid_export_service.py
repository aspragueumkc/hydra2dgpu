"""UGRID NetCDF export service.

Provides netCDF4 export of mesh results following CF-1.8 + UGRID 1.0 conventions.
Pure Python + numpy + netCDF4 — no Qt/PyQt5 imports.

Extracted from
``SWE2DWorkbenchStudioDialog._write_ugrid_nc`` in
``swe2d.workbench.extracted.topology_and_io_methods`` (Task B2 of
the extracted-methods migration plan).
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

__all__ = ["write_ugrid_nc"]


def write_ugrid_nc(
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
    crs_wkt: str = 'LOCAL_CS["Unknown"]',
    epsg_code: Optional[int] = None,
) -> None:
    """Write a UGRID 1.0 NetCDF4 file readable by QGIS MDAL.

    The file follows the CF-1.8 + UGRID 1.0 conventions.  QGIS MDAL's
    UGRID driver natively pairs (velocity_u, velocity_v) into an arrow
    vector dataset without requiring any naming hacks.

    Parameters
    ----------
    path : str
        Output ``.nc`` file path.
    mesh_data : dict
        Mesh geometry dict with keys ``node_x``, ``node_y``,
        (optional) ``node_z``, and either ``cell_face_offsets`` +
        ``cell_face_nodes`` or ``cell_nodes`` (triangular).
    length_unit_name : str
        Unit label for length (e.g.  ``"m"``,  ``"ft"``).
    is_us_customary : bool
        If True, Manning unit string uses ft^(1/3).
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
        When supplied, result variables are written; otherwise topology
        only.
    log_fn : callable or None
        Optional logging function (e.g. ``self._log``).  Called with
        error messages.
    result_data : dict or None
        Optional result-data dict (may contain ``"n_mann_cell"``).
    crs_wkt : str
        CRS WKT string (e.g. from ``QgsCoordinateReferenceSystem.toWkt()``).
    epsg_code : int or None
        EPSG code for the CRS, if known.
    """
    # --- netCDF4 availability check ---
    try:
        import netCDF4 as _nc4_local  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "netCDF4 is unavailable (missing or binary-incompatible in current QGIS Python)."
            " Install a compatible netCDF4 build for this QGIS environment."
        ) from exc

    if mesh_data is None:
        raise RuntimeError("No mesh data available")

    node_x = mesh_data["node_x"]
    node_y = mesh_data["node_y"]
    node_z = mesh_data.get("node_z", np.zeros_like(node_x))

    # Build face→node connectivity (zero-based, row per face, -1 padded)
    face_offsets = mesh_data.get("cell_face_offsets")
    face_nodes_arr = mesh_data.get("cell_face_nodes")
    cell_nodes_tri = mesh_data.get("cell_nodes")

    if face_offsets is not None and face_nodes_arr is not None:
        offsets = face_offsets.astype(np.int32)
        n_cells = int(offsets.size - 1)
        max_vp = int(max(offsets[i + 1] - offsets[i] for i in range(n_cells)))
        face_node = np.full((n_cells, max_vp), -1, dtype=np.int32)
        cell_cx = np.empty(n_cells, dtype=np.float64)
        cell_cy = np.empty(n_cells, dtype=np.float64)
        cell_solver_z = np.empty(n_cells, dtype=np.float64)
        for i in range(n_cells):
            s, e = int(offsets[i]), int(offsets[i + 1])
            ring = face_nodes_arr[s:e].astype(np.int32)
            face_node[i, : e - s] = ring
            cell_cx[i] = float(np.mean(node_x[ring]))
            cell_cy[i] = float(np.mean(node_y[ring]))
            cell_solver_z[i] = float(np.mean(node_z[ring]))
    else:
        tri = cell_nodes_tri.reshape(-1, 3).astype(np.int32)
        n_cells = tri.shape[0]
        max_vp = 3
        face_node = tri
        cell_cx = np.mean(node_x[tri], axis=1)
        cell_cy = np.mean(node_y[tri], axis=1)
        cell_solver_z = np.mean(node_z[tri], axis=1)

    n_nodes = int(node_x.size)

    len_unit = length_unit_name if length_unit_name else "m"
    vel_unit = f"{len_unit} s-1"
    mom_unit = f"{len_unit}2 s-1"
    manning_unit = "s ft-1/3" if is_us_customary else "s m-1/3"

    import netCDF4 as _nc4_local  # noqa: F811 — re-imported for type narrowing

    with _nc4_local.Dataset(path, "w", format="NETCDF4") as ds:
        # Global attributes (CF + UGRID)
        ds.Conventions = "CF-1.8 UGRID-1.0"
        ds.title = "SWE2D HYDRA model results"
        ds.institution = "qgis-hydra-plugin"
        ds.history = "Created by swe2d_workbench_qt"
        ds.featureType = "mesh2D"

        # Dimensions
        ds.createDimension("node", n_nodes)
        ds.createDimension("face", n_cells)
        ds.createDimension("max_face_nodes", max_vp)
        if timesteps:
            ds.createDimension("time", len(timesteps))

        # ---- Mesh topology container variable ----
        mesh = ds.createVariable("mesh2d", "i4")
        mesh.cf_role = "mesh_topology"
        mesh.topology_dimension = 2
        mesh.node_coordinates = "node_x node_y"
        mesh.face_node_connectivity = "face_node"
        mesh.face_coordinates = "face_x face_y"

        # Node coordinates
        nx_var = ds.createVariable("node_x", "f8", ("node",))
        nx_var.standard_name = "projection_x_coordinate"
        nx_var.units = len_unit
        nx_var.mesh = "mesh2d"
        nx_var.location = "node"
        nx_var.grid_mapping = "crs"
        nx_var[:] = node_x.astype(np.float64)

        ny_var = ds.createVariable("node_y", "f8", ("node",))
        ny_var.standard_name = "projection_y_coordinate"
        ny_var.units = len_unit
        ny_var.mesh = "mesh2d"
        ny_var.location = "node"
        ny_var.grid_mapping = "crs"
        ny_var[:] = node_y.astype(np.float64)

        nz_var = ds.createVariable("node_z", "f8", ("node",))
        nz_var.standard_name = "altitude"
        nz_var.long_name = "bed elevation at node"
        nz_var.units = len_unit
        nz_var.mesh = "mesh2d"
        nz_var.location = "node"
        nz_var.grid_mapping = "crs"
        nz_var[:] = node_z.astype(np.float64)

        # Face centroid coordinates
        fx_var = ds.createVariable("face_x", "f8", ("face",))
        fx_var.standard_name = "projection_x_coordinate"
        fx_var.units = len_unit
        fx_var.mesh = "mesh2d"
        fx_var.location = "face"
        fx_var.grid_mapping = "crs"
        fx_var[:] = cell_cx.astype(np.float64)

        fy_var = ds.createVariable("face_y", "f8", ("face",))
        fy_var.standard_name = "projection_y_coordinate"
        fy_var.units = len_unit
        fy_var.mesh = "mesh2d"
        fy_var.location = "face"
        fy_var.grid_mapping = "crs"
        fy_var[:] = cell_cy.astype(np.float64)

        # Face bed elevation consistent with solver cell_zb.
        fz_var = ds.createVariable("face_z", "f8", ("face",))
        fz_var.long_name = "face bed elevation (mean vertex bed, solver-consistent)"
        fz_var.units = len_unit
        fz_var.mesh = "mesh2d"
        fz_var.location = "face"
        fz_var.grid_mapping = "crs"
        fz_var[:] = cell_solver_z.astype(np.float64)

        # Face→node connectivity (0-indexed as UGRID standard; -1 = fill)
        fn_var = ds.createVariable(
            "face_node", "i4", ("face", "max_face_nodes"),
            fill_value=-1,
        )
        fn_var.cf_role = "face_node_connectivity"
        fn_var.long_name = "face to node connectivity"
        fn_var.start_index = 0  # zero-based
        fn_var[:] = face_node

        # CRS variable
        crs_var = ds.createVariable("crs", "i4")
        crs_var.grid_mapping_name = "unknown"
        crs_var.crs_wkt = crs_wkt
        if epsg_code:
            crs_var.epsg_code = f"EPSG:{epsg_code}"

        # ---- Time-dependent results ----
        if timesteps:
            times_s = np.array([t for t, *_ in timesteps], dtype=np.float64)

            t_var = ds.createVariable("time", "f8", ("time",))
            t_var.standard_name = "time"
            t_var.long_name = "simulation time"
            t_var.units = "seconds since 2000-01-01 00:00:00"
            t_var.calendar = "proleptic_gregorian"
            t_var[:] = times_s

            depth_arr = np.zeros((len(timesteps), n_cells), dtype=np.float32)
            wse_arr = np.zeros((len(timesteps), n_cells), dtype=np.float32)
            vel_u_arr = np.zeros((len(timesteps), n_cells), dtype=np.float32)
            vel_v_arr = np.zeros((len(timesteps), n_cells), dtype=np.float32)
            vel_mag_arr = np.zeros((len(timesteps), n_cells), dtype=np.float32)
            if include_extra:
                mom_u_arr = np.zeros((len(timesteps), n_cells), dtype=np.float32)
                mom_v_arr = np.zeros((len(timesteps), n_cells), dtype=np.float32)
                qmag_arr = np.zeros((len(timesteps), n_cells), dtype=np.float32)
                wet_arr = np.zeros((len(timesteps), n_cells), dtype=np.float32)
                froude_arr = np.zeros((len(timesteps), n_cells), dtype=np.float32)
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
                vel_u_arr[ti] = u.astype(np.float32)
                vel_v_arr[ti] = v.astype(np.float32)
                vel_mag_arr[ti] = np.sqrt(u ** 2 + v ** 2).astype(np.float32)
                if include_extra:
                    mom_u_arr[ti] = hu_f.astype(np.float32)
                    mom_v_arr[ti] = hv_f.astype(np.float32)
                    qmag_arr[ti] = np.sqrt(hu_f ** 2 + hv_f ** 2).astype(np.float32)
                    wet_arr[ti] = wet.astype(np.float32)
                    froude_arr[ti] = np.where(wet, np.sqrt(u ** 2 + v ** 2) / np.sqrt(np.maximum(g * h_f, 1.0e-12)), 0.0).astype(np.float32)

            d_var = ds.createVariable(
                "water_depth", "f4", ("time", "face"), fill_value=np.float32(-9999.0)
            )
            d_var.standard_name = "water_depth"
            d_var.long_name = "water depth"
            d_var.units = len_unit
            d_var.mesh = "mesh2d"
            d_var.location = "face"
            d_var.coordinates = "face_x face_y"
            d_var.grid_mapping = "crs"
            d_var[:] = depth_arr

            w_var = ds.createVariable(
                "water_surface_elevation", "f4", ("time", "face"), fill_value=np.float32(-9999.0)
            )
            w_var.standard_name = "water_surface_elevation"
            w_var.long_name = "water surface elevation"
            w_var.units = len_unit
            w_var.mesh = "mesh2d"
            w_var.location = "face"
            w_var.coordinates = "face_x face_y"
            w_var.grid_mapping = "crs"
            w_var[:] = wse_arr

            u_var = ds.createVariable(
                "velocity_u", "f4", ("time", "face"), fill_value=np.float32(-9999.0)
            )
            u_var.standard_name = "eastward_water_velocity"
            u_var.long_name = "eastward component of velocity"
            u_var.units = vel_unit
            u_var.mesh = "mesh2d"
            u_var.location = "face"
            u_var.coordinates = "face_x face_y"
            u_var.grid_mapping = "crs"
            u_var[:] = vel_u_arr

            v_var = ds.createVariable(
                "velocity_v", "f4", ("time", "face"), fill_value=np.float32(-9999.0)
            )
            v_var.standard_name = "northward_water_velocity"
            v_var.long_name = "northward component of velocity"
            v_var.units = vel_unit
            v_var.mesh = "mesh2d"
            v_var.location = "face"
            v_var.coordinates = "face_x face_y"
            v_var.grid_mapping = "crs"
            v_var[:] = vel_v_arr

            vm_var = ds.createVariable(
                "velocity_magnitude", "f4", ("time", "face"), fill_value=np.float32(-9999.0)
            )
            vm_var.long_name = "velocity magnitude"
            vm_var.units = vel_unit
            vm_var.mesh = "mesh2d"
            vm_var.location = "face"
            vm_var.coordinates = "face_x face_y"
            vm_var.grid_mapping = "crs"
            vm_var[:] = vel_mag_arr

            if include_extra:
                mu_var = ds.createVariable(
                    "momentum_x", "f4", ("time", "face"), fill_value=np.float32(-9999.0)
                )
                mu_var.long_name = "x momentum per unit width"
                mu_var.units = mom_unit
                mu_var.mesh = "mesh2d"
                mu_var.location = "face"
                mu_var.coordinates = "face_x face_y"
                mu_var.grid_mapping = "crs"
                mu_var[:] = mom_u_arr

                mv_var = ds.createVariable(
                    "momentum_y", "f4", ("time", "face"), fill_value=np.float32(-9999.0)
                )
                mv_var.long_name = "y momentum per unit width"
                mv_var.units = mom_unit
                mv_var.mesh = "mesh2d"
                mv_var.location = "face"
                mv_var.coordinates = "face_x face_y"
                mv_var.grid_mapping = "crs"
                mv_var[:] = mom_v_arr

                qmag_var = ds.createVariable(
                    "unit_discharge_magnitude", "f4", ("time", "face"), fill_value=np.float32(-9999.0)
                )
                qmag_var.long_name = "unit discharge magnitude"
                qmag_var.units = mom_unit
                qmag_var.mesh = "mesh2d"
                qmag_var.location = "face"
                qmag_var.coordinates = "face_x face_y"
                qmag_var.grid_mapping = "crs"
                qmag_var[:] = qmag_arr

                wet_var = ds.createVariable(
                    "wet_mask", "f4", ("time", "face"), fill_value=np.float32(-9999.0)
                )
                wet_var.long_name = "wet mask"
                wet_var.units = "1"
                wet_var.mesh = "mesh2d"
                wet_var.location = "face"
                wet_var.coordinates = "face_x face_y"
                wet_var.grid_mapping = "crs"
                wet_var[:] = wet_arr

                fr_var = ds.createVariable(
                    "froude_number", "f4", ("time", "face"), fill_value=np.float32(-9999.0)
                )
                fr_var.long_name = "Froude number"
                fr_var.units = "1"
                fr_var.mesh = "mesh2d"
                fr_var.location = "face"
                fr_var.coordinates = "face_x face_y"
                fr_var.grid_mapping = "crs"
                fr_var[:] = froude_arr

        if include_extra:
            if result_data is not None and "n_mann_cell" in result_data:
                n_face = np.asarray(result_data["n_mann_cell"], dtype=np.float64)[:n_cells]
            else:
                n_face = np.full(n_cells, float(n_mann), dtype=np.float64)
            n_var = ds.createVariable("manning_n_face", "f4", ("face",), fill_value=np.float32(-9999.0))
            n_var.long_name = "Manning roughness at face"
            n_var.units = manning_unit
            n_var.mesh = "mesh2d"
            n_var.location = "face"
            n_var.coordinates = "face_x face_y"
            n_var.grid_mapping = "crs"
            n_var[:] = n_face.astype(np.float32)
