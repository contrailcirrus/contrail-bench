"""Preprocess NCEP GFS forecasts into persistent contrail region (PCR) fields.

Reads the global 0.25 degree GFS forecasts from the public NOAA S3 bucket through pycontrails'
``GFSForecast`` datalib. Each valid hour is written at every lead given by ``--lead-offsets`` (its
shortest lead of 0 to 5 hours plus each offset; see ``src/forecast_runs.py``) and flight level,
from the initialisation that many hours earlier, to ``{epoch}_{flight_level}/{lead}.nc``. The shard
records that initialisation. The files are hourly for the first 120 hours, so every lead is
available.

The archive serves temperature and specific humidity on pressure levels 50 hPa apart. The pipeline
reads the six levels from 350 to 100 hPa, which bracket flight levels 270 to 440, so temperature
and humidity are interpolated (log-linear in pressure) onto each flight level's pressure, cropped
to the region, regridded onto the 0.25 degree benchmark grid, and ``rhi`` and ``sac`` derived
there. No humidity correction is applied.

The fields are cropped to the region plus ``CROP_PAD_DEGREES`` before regridding, so the edge cells
of the benchmark grid average over every source cell they overlap.

The datalib downloads the whole global GRIB file (about 0.5 GB) for each valid time and lead, and
keeps only the requested variables and levels. Each time uses a temporary cache that is removed
afterwards, so a worker's disk does not fill up.
"""

import datetime
import functools
import tempfile
from collections.abc import Iterator, Sequence

import numpy as np
import xarray as xr
from pycontrails.core.cache import DiskCacheStore
from pycontrails.datalib.gfs import GFSForecast

from src import cli, config, forecast_runs, pcr
from src.grid import Extent

# Pipeline parameters

#: Output directory name under ``forecasts/``
SOURCE = "ncep-gfs"

#: Pressure levels (hPa) read from the archive
GFS_LEVELS_HPA = (100, 150, 200, 250, 300, 350)

#: Degrees added around the region before regridding, so every source cell that overlaps an edge
#: target cell is kept (cell edges are 0.125 degrees beyond the edge centres)
CROP_PAD_DEGREES = 0.5


def compute_forecast(
    valid_time: datetime.datetime,
    init: datetime.datetime,
    flight_levels: Sequence[int],
    extent: Extent,
) -> xr.Dataset:
    """PCR fields at every flight level for one valid time, from one initialisation.

    Returns
    -------
    xr.Dataset
        ``rhi`` and ``sac`` on the benchmark grid with a ``flight_level`` dimension
    """
    pressures = [pcr.target_pressure_hpa(fl) for fl in flight_levels]
    pcr.bracketing_level_indices(np.array(sorted(GFS_LEVELS_HPA)), pressures)  # must cover them

    with tempfile.TemporaryDirectory() as tmp:
        met = GFSForecast(
            time=valid_time,
            variables=["air_temperature", "specific_humidity"],
            pressure_levels=list(GFS_LEVELS_HPA),
            forecast_time=init,
            grid=0.25,
            cachestore=DiskCacheStore(cache_dir=tmp),
        ).open_metdataset()
        ds = met.data.load()

    lon_min, lon_max, lat_min, lat_max = extent
    pad = CROP_PAD_DEGREES
    ds = ds.sel(
        longitude=slice(lon_min - pad, lon_max + pad), latitude=slice(lat_min - pad, lat_max + pad)
    )
    ds = pcr.interpolate_to_pressures(ds, pressures)

    return pcr.regrid_and_derive(ds, flight_levels, extent)


def preprocess_gfs(
    valid_time: datetime.datetime,
    flight_levels: Sequence[int],
    *,
    lead_offsets: Sequence[int],
    extent: Extent,
) -> Iterator[tuple[tuple[int, int], xr.Dataset]]:
    """The shard of each flight level and lead for one valid time, one lead at a time."""
    for lead in forecast_runs.leads(valid_time, lead_offsets):
        init = forecast_runs.init_for(valid_time, lead)
        batched = compute_forecast(valid_time, init, flight_levels, extent)
        for fl, shard in pcr.split_flight_levels(batched, flight_levels).items():
            yield (fl, lead), forecast_runs.record_init(shard, init, valid_time)


def main() -> None:
    """Program entrypoint."""
    parser = cli.shard_parser(__doc__)
    parser.add_argument(
        "--lead-offsets",
        type=forecast_runs.parse_offsets,
        default=config.LEAD_OFFSETS,
        help="hours added to each valid time's shortest lead, such as 0,12,24,36 (the default)",
    )
    parser.add_argument("--harness-threads", type=int, help="threads per Dataflow worker")
    args = parser.parse_args()

    produce = functools.partial(
        preprocess_gfs, lead_offsets=args.lead_offsets, extent=config.REGIONS[args.region]
    )
    job = cli.shard_job_for(args, SOURCE, lead_offsets=args.lead_offsets)
    cli.run(args, job, produce, "gfs")


if __name__ == "__main__":
    main()
