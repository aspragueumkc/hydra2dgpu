"""GeoTIFF export for high-perf overlay frames."""
from __future__ import annotations

import numpy as np

__all__ = ["export_overlay_grid_to_geotiff"]


def export_overlay_grid_to_geotiff(
    *,
    arr: np.ndarray,
    xmin: float,
    ymax: float,
    dx: float,
    dy: float,
    path: str,
    nodata: float = -9999.0,
    driver_name: str = "GTiff",
    options: tuple[str, ...] = ("COMPRESS=LZW", "TILED=YES", "BIGTIFF=YES"),
    crs_auth: str = "EPSG:4326",
    band_description: str = "",
) -> None:
    """Write a 2-D numpy array to a GeoTIFF file using GDAL.

    Parameters
    ----------
    arr : np.ndarray
        2-D array of values to write.
    xmin, ymax : float
        Upper-left corner in world coordinates.
    dx, dy : float
        Pixel width and height (map units).  ``dy`` is positive; the
        GeoTransform's north-south pixel step is stored as ``-dy``.
    path : str
        Output file path.
    nodata : float
        No-data value to register in the band.
    driver_name : str
        GDAL short driver name (default ``"GTiff"``).
    options : tuple[str, ...]
        Driver creation options passed to ``GDALDriver.Create``.
    crs_auth : str
        CRS authority identifier (e.g. ``"EPSG:4326"``).
    band_description : str
        Optional band description.
    """
    from osgeo import gdal, osr

    if gdal is None:
        raise RuntimeError("GDAL is not available. Cannot write GeoTIFF.")

    h, w = arr.shape
    driver = gdal.GetDriverByName(driver_name)
    if driver is None:
        raise RuntimeError(f"GDAL driver '{driver_name}' is not available.")

    ds = driver.Create(path, w, h, 1, gdal.GDT_Float64, options=list(options))
    if ds is None:
        raise RuntimeError("GDAL could not create output dataset.")

    gt = (xmin, dx, 0.0, ymax, 0.0, -dy)
    ds.SetGeoTransform(gt)

    srs = osr.SpatialReference()
    srs.SetFromUserInput(crs_auth)
    ds.SetProjection(srs.ExportToWkt())

    band = ds.GetRasterBand(1)
    band.WriteArray(arr)
    band.SetNoDataValue(nodata)
    if band_description:
        band.SetDescription(band_description)
    ds.FlushCache()
    ds = None
