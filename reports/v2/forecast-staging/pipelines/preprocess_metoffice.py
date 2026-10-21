"""Preprocess Met Office Unified Model (UM) forecasts into persistent contrail region (PCR) fields.

Reads the global deterministic 10 km UM forecasts from the public Met Office S3 archive through
pycontrails' ``UM10`` datalib. A job for ``--run R`` writes one netCDF shard per valid hour and
flight level for the whole period, each from the newest ``R`` initialisation at or before that
hour (see ``src/forecast_runs.py``). The shard records that initialisation, so lead time is
worked out afterwards.

The archive serves relative humidity on pressure levels, and the datalib derives specific humidity
from it. The pipeline reads the nine levels from 350 to 150 hPa (25 hPa apart), which bracket
flight levels 270 to 440, so temperature and humidity are interpolated (log-linear in pressure)
onto each flight level's pressure, cropped to the region, regridded onto the 0.25 degree benchmark
grid, and ``rhi``, ``sac`` and ``pcr`` derived there. No humidity correction is applied.

The fields are cropped to the region plus ``CROP_PAD_DEGREES`` before regridding, so the edge cells
of the benchmark grid average over every source cell they overlap. Cropping exactly to the region
(a padding of 0) changes the values of the outermost ring of cells only.

The datalib fetches whole global fields (about 370 MB an hour) into a cache. Each time uses a
temporary cache that is removed afterwards, so a worker's disk does not fill up.
"""

import datetime
import functools
import tempfile
from collections.abc import Sequence

import numpy as np
import xarray as xr
from pycontrails.core.cache import DiskCacheStore
from pycontrails.datalib.ukmo import UM10

from src import cli, config, forecast_runs, pcr
from src.grid import Extent

# Pipeline parameters

#: Output directory name under ``forecasts/``
SOURCE = "metoffice-um"

#: Pressure levels (hPa) read from the archive
UM_LEVELS_HPA = (150, 175, 200, 225, 250, 275, 300, 325, 350)

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
        ``rhi``, ``sac`` and ``pcr`` on the benchmark grid with a ``flight_level`` dimension
    """
    pressures = [pcr.target_pressure_hpa(fl) for fl in flight_levels]
    pcr.bracketing_level_indices(np.array(UM_LEVELS_HPA), pressures)  # must cover them

    with tempfile.TemporaryDirectory() as tmp:
        met = UM10(
            time=valid_time,
            forecast_time=init,
            pressure_levels=list(UM_LEVELS_HPA),
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


def preprocess_metoffice(
    valid_time: datetime.datetime, flight_levels: Sequence[int], *, run: str, extent: Extent
) -> dict[int, xr.Dataset]:
    """One shard per flight level for one valid time, from the ``run`` initialisation."""
    init = forecast_runs.base_run(valid_time, run)
    batched = compute_forecast(valid_time, init, flight_levels, extent)
    return {
        fl: forecast_runs.record_init(shard, init, valid_time)
        for fl, shard in pcr.split_flight_levels(batched, flight_levels).items()
    }


def main() -> None:
    """Program entrypoint."""
    parser = cli.shard_parser(__doc__)
    parser.add_argument("--run", type=forecast_runs.parse_run, required=True, help="00Z, 06Z, ...")
    parser.add_argument("--harness-threads", type=int, help="threads per Dataflow worker")
    args = parser.parse_args()

    produce = functools.partial(
        preprocess_metoffice, run=args.run, extent=config.REGIONS[args.region]
    )
    job = cli.shard_job_for(args, SOURCE, run=args.run)
    cli.run(args, job, produce, f"metoffice-{args.run}")


if __name__ == "__main__":
    main()
