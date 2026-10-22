"""The 0.25 degree benchmark grid, and conservative regridding onto it.

The benchmark grid is global at 0.25 degrees. Every forecast pipeline uses these exact
coordinate values, as do the evaluation datasets in ``eval-staging``, because downstream code
selects cells by exact label rather than by nearest neighbour.
"""

import numpy as np
import xarray as xr

#: ``(lon_min, lon_max, lat_min, lat_max)``
Extent = tuple[float, float, float, float]

#: Cell centres and cell edges (one more edge than centres) of the global grid
LONGITUDE = np.linspace(-180.0, 179.75, 1440)
LATITUDE = np.linspace(-90.0, 90.0, 721)
LONGITUDE_BNDS = np.linspace(-180.125, 179.875, 1441)
LATITUDE_BNDS = np.linspace(-90.125, 90.125, 722).clip(-90.0, 90.0)

#: Share of a target cell's source weight that may be missing before the cell is missing; this is
#: above floating point noise, and far below the weight of any real source cell
MISSING_TOLERANCE = 1e-9


def crop_grid(extent: Extent) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Crop the benchmark grid to ``extent`` = ``(lon_min, lon_max, lat_min, lat_max)``.

    The global arrays are sliced rather than rebuilt with ``linspace`` over the box, so the
    cropped values are bit-identical to the global grid.

    Returns
    -------
    longitude, latitude, longitude_bnds, latitude_bnds : numpy.ndarray
        Cell centres and cell edges (one more edge than centres) for the cropped region.
    """
    lon_min, lon_max, lat_min, lat_max = extent
    (lon_idx,) = np.nonzero((lon_min <= LONGITUDE) & (lon_max >= LONGITUDE))
    (lat_idx,) = np.nonzero((lat_min <= LATITUDE) & (lat_max >= LATITUDE))
    # A contiguous run of centres maps to the same run plus one in the edges.
    return (
        LONGITUDE[lon_idx],
        LATITUDE[lat_idx],
        LONGITUDE_BNDS[lon_idx[0] : lon_idx[-1] + 2],
        LATITUDE_BNDS[lat_idx[0] : lat_idx[-1] + 2],
    )


def cell_bounds(centers: np.ndarray, spacing: float) -> np.ndarray:
    """Cell edges (``len(centers) + 1``) for an ascending, uniformly spaced, cell-centred grid."""
    return np.concatenate(([centers[0] - spacing / 2.0], centers + spacing / 2.0))


def overlap_weights(source_bnds: np.ndarray, target_bnds: np.ndarray) -> np.ndarray:
    """Conservative (area-weighted) 1D weight matrix, shape ``(n_target, n_source)``.

    Row ``i`` is the share of target cell ``i`` covered by each source cell, normalised by the
    part actually covered. A target cell only partly covered by the source grid still gets a
    proper weighted average instead of a downward-biased sum.
    """
    src_lo, src_hi = source_bnds[:-1], source_bnds[1:]
    tgt_lo, tgt_hi = target_bnds[:-1], target_bnds[1:]
    overlap = np.minimum(tgt_hi[:, None], src_hi[None, :]) - np.maximum(
        tgt_lo[:, None], src_lo[None, :]
    )
    overlap = np.clip(overlap, 0.0, None)
    covered = overlap.sum(axis=1, keepdims=True)
    covered = np.where(covered > 0.0, covered, 1.0)  # a fully uncovered row stays zero
    return np.asarray(overlap / covered)


def regrid(field: xr.DataArray, target_extent: Extent) -> xr.DataArray:
    """Conservatively regrid a continuous field onto the benchmark grid cropped to the extent.

    The source grid is the field's own ``longitude`` and ``latitude`` coordinates, which must be
    ascending and uniformly spaced. Source cell edges are rebuilt from that spacing (centre +/-
    half spacing), which is exact for the uniform lat/lon grids of every source used here. Other
    dimensions (``level``, ``time``) are kept.

    Missing (NaN) source cells make only the target cells they overlap missing; the rest of the
    field is regridded as usual. A plain weighted sum would turn one NaN into a NaN everywhere.
    """
    source_longitude = field["longitude"].values
    source_latitude = field["latitude"].values
    source_longitude_spacing = float(np.diff(source_longitude).mean())
    source_latitude_spacing = float(np.diff(source_latitude).mean())
    target_lon, target_lat, target_lon_bnds, target_lat_bnds = crop_grid(target_extent)
    lon_weights = xr.DataArray(
        overlap_weights(cell_bounds(source_longitude, source_longitude_spacing), target_lon_bnds),
        dims=("_target_longitude", "longitude"),
        coords={"_target_longitude": target_lon, "longitude": source_longitude},
    )
    lat_weights = xr.DataArray(
        overlap_weights(cell_bounds(source_latitude, source_latitude_spacing), target_lat_bnds),
        dims=("_target_latitude", "latitude"),
        coords={"_target_latitude": target_lat, "latitude": source_latitude},
    )

    def apply_weights(x: xr.DataArray) -> xr.DataArray:
        return xr.dot(xr.dot(x, lon_weights, dim="longitude"), lat_weights, dim="latitude")

    missing = field.isnull()
    if missing.any():
        # Regrid with zeros in place of the missing cells, and drop the target cells that overlap
        # any of them (their share of missing weight is above the tolerance). Only the slices
        # (levels, times) that have missing cells need that overlap worked out.
        regridded = apply_weights(field.fillna(0.0))
        has_missing = missing.any(["longitude", "latitude"])
        share = apply_weights(missing.astype(float).where(has_missing, drop=True))
        affected = (share > MISSING_TOLERANCE).reindex_like(regridded, fill_value=False)
        regridded = regridded.where(~affected)
    else:
        regridded = apply_weights(field)
    return regridded.rename({"_target_longitude": "longitude", "_target_latitude": "latitude"})
