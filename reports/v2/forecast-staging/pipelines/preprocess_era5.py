"""Preprocess ERA5 reanalysis into persistent contrail region (PCR) fields.

Reads the daily ERA5 zarr stores (``t`` and ``q``: ERA5 model-level data interpolated to fixed
pressure levels, 134 to 416 hPa), interpolates onto each flight level's pressure, regrids onto the
0.25 degree benchmark grid, and derives ``rhi``, ``sac`` and ``pcr`` there (see
``src/pcr.py``). Writes one netCDF file per valid time and flight level.

ERA5 gets a humidity correction (pycontrails' latitude-dependent extension of the Teoh et al.
2022 bias correction, for data derived from model levels). HRES and IFS 50r1 get the same
correction; the Met Office and the product APIs get none.
"""

import datetime
import functools
from collections.abc import Sequence

import pandas as pd
import xarray as xr
from pycontrails.core.met import MetDataset
from pycontrails.models import sac

from src import cli, config, pcr, storage
from src.grid import Extent

# Pipeline parameters

#: Output directory name under ``forecasts/``
SOURCE = "ecmwf-ifs-era5"

#: Daily ERA5 zarr stores on pressure levels; ``{day}`` is ``YYYYmmdd``
TEMPLATE = "gs://contrails-301217-ecmwf-era5-zarr-v2/{day}_pl.zarr"


def compute_forecast(
    time: datetime.datetime, flight_levels: Sequence[int], extent: Extent, template: str
) -> xr.Dataset:
    """PCR fields (``rhi``, ``sac``, ``pcr``) for one time and every flight level, from one read.

    The stores have fixed pressure levels, so temperature and humidity are interpolated onto every
    flight level's pressure in one vectorized call, regridded onto the benchmark grid, and RHi and
    SAC derived there.
    """
    pressures = [pcr.target_pressure_hpa(fl) for fl in flight_levels]
    stamp = pd.Timestamp(time)
    lon_min, lon_max, lat_min, lat_max = extent
    # Variables and region are selected before anything is loaded: a whole global day is far
    # larger than the region needed. Latitude must be stored ascending.
    store = storage.open_zarr(template.format(day=f"{stamp:%Y%m%d}"))
    raw = (
        store[["t", "q"]]
        .sel(
            longitude=slice(lon_min, lon_max),
            latitude=slice(lat_min, lat_max),
            time=slice(stamp, stamp),
        )
        .load()
    )
    # pycontrails needs this dimension order; stores often have time first
    raw = raw.transpose("longitude", "latitude", "level", "time")
    met = MetDataset(raw, copy=False).standardize_variables(sac.SAC.ecmwf_met_variables())
    interpolated = pcr.interpolate_to_pressures(met.data, pressures)
    # The pressure levels are derived from model levels, hence "model"
    return pcr.regrid_and_derive(interpolated, flight_levels, extent, humidity_correction="model")


def preprocess_era5(
    time: datetime.datetime, flight_levels: Sequence[int], *, extent: Extent, template: str
) -> dict[int, xr.Dataset]:
    """One shard per flight level for one valid time."""
    return pcr.split_flight_levels(
        compute_forecast(time, flight_levels, extent, template), flight_levels
    )


def main() -> None:
    """Program entrypoint."""
    parser = cli.shard_parser(__doc__)
    args = parser.parse_args()

    produce = functools.partial(
        preprocess_era5, extent=config.REGIONS[args.region], template=TEMPLATE
    )
    job = cli.shard_job_for(args, SOURCE)
    cli.run(args, job, produce, "era5")


if __name__ == "__main__":
    main()
