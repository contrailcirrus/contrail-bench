"""UK Met Office UM forecast source.

Differs from other forecast sources only in how raw ``air_temperature``/
``specific_humidity`` fields are obtained -- ``MetOfficeUM`` serves a fixed set
of 7 native cruise pressure levels, so :func:`fetch` vertically interpolates
(``contrailbench.pcr.interpolate_to_pressures``, log-linear in pressure) onto
each target flight level's ISA pressure before handing off to the shared
regrid/derive helper. Sources that can be queried at an arbitrary pressure
directly skip that step.
"""

import datetime
from collections.abc import Iterable

import numpy as np
import xarray as xr

from contrailbench import pcr
from contrailbench.datalib.metoffice import s3
from contrailbench.datalib.metoffice.ukmo import MetOfficeUM


def fetch(
    time: datetime.datetime,
    flight_levels: Iterable[int],
    *,
    extent: tuple[float, float, float, float] = pcr.CONUS_EXTENT,
    lead_hours: int | None = None,
    mirror_dir: str | None = None,
) -> xr.Dataset:
    """Compute PCR fields at many flight levels for a single time, from one met fetch.

    Parameters
    ----------
    time : datetime.datetime
        Target time.

    flight_levels : Iterable[int]
        Target flight levels. All interpolate from the same underlying met
        fetch (Met Office's 7 native cruise levels), so this fetch happens
        once regardless of how many flight levels are requested.

    extent : tuple[float, float, float, float], optional
        Benchmark-grid region to regrid onto. Defaults to
        :data:`contrailbench.pcr.CONUS_EXTENT`.

    lead_hours : int, optional
        ``None`` (default) reads the shortest-available lead. A fixed int
        reads instead from that lead's whole-window mirror.

    mirror_dir : str, optional
        Mirror output directory to read from (the ``--out-dir`` a prior
        ``contrailbench.datalib.metoffice.mirror`` run was pointed at).
        ``None`` (default) uses ``MetOfficeUM``'s own default.

    Returns
    -------
    xr.Dataset
        ``rhi``, ``sac`` and ``pcr`` (rhi_threshold=1.0 default) fields on the
        benchmark grid, with a ``flight_level`` dim. Extract one flight
        level's single-``level``-dim slice with
        :func:`contrailbench.pcr.extract_flight_level`.
    """
    flight_levels = list(flight_levels)
    target_pressures_hpa = [pcr.target_pressure_hpa(fl) for fl in flight_levels]

    met = MetOfficeUM(
        time=time,
        pressure_levels=s3.CRUISE_LEVELS_HPA,
        lead_hours=lead_hours,
        paths=mirror_dir,
    ).open_metdataset()
    ds = met.data.load()
    ds = pcr.interpolate_to_pressures(ds, target_pressures_hpa)

    native_longitude = ds["longitude"].values
    native_latitude = ds["latitude"].values
    lon_spacing = float(np.diff(native_longitude).mean())
    lat_spacing = float(np.diff(native_latitude).mean())

    return pcr.regrid_and_derive(
        ds,
        flight_levels=flight_levels,
        source_longitude=native_longitude,
        source_latitude=native_latitude,
        source_longitude_spacing=lon_spacing,
        source_latitude_spacing=lat_spacing,
        target_extent=extent,
    )
