"""Preprocess contrails.org forecasts.

contrails.org serves a CoCiP forecast driven by ECMWF HRES. Its API returns only the most recent
forecast for a valid time, so the forecasts at the other leads are read from the grids the product
is served from, one file per valid time, flight level and lead
(``gs://contrails-301217-api-preprocessor-prod/grids/default/{epoch}_{flight_level}/{lead}.nc``).

Each valid hour is written at every lead given by ``--lead-offsets`` (its shortest lead plus each
offset; see ``src/forecast_runs.py``) to ``{epoch}_{flight_level}/{lead}.nc``. The product has no
forecast at lead 0, so at the valid times that fall on an initialisation time its first lead is 6
hours. Each shard holds a binary ``pcr`` field, true where the energy forcing per metre is nonzero,
of either sign. That is how the product defines its region. The shard records the forecast's
initialisation, which is checked against the lead.

The grids are global, so each is cropped to the region.
"""

import datetime
import functools
from collections.abc import Iterator, Sequence

import numpy as np
import xarray as xr

from src import api, cli, config, forecast_runs, storage
from src import times as time_utils
from src.grid import Extent

# Pipeline parameters

#: Output directory name under ``forecasts/``
SOURCE = "contrails-org"

#: Bucket holding the product's grids
BUCKET = "contrails-301217-api-preprocessor-prod"

#: Prefix of the grids of the default aircraft class
PREFIX = "grids/default"


def fetch(valid_time: datetime.datetime, flight_level: int, lead: int) -> bytes:
    """The raw grid for one valid time, flight level and lead.

    Raises
    ------
    FileNotFoundError
        If the product has no such grid.
    """
    name = f"{PREFIX}/{time_utils.utc_epoch(valid_time)}_{flight_level}/{lead}.nc"
    blob = storage.gcs_client().bucket(BUCKET).blob(name)
    if not blob.exists():
        msg = f"no contrails.org grid gs://{BUCKET}/{name}"
        raise FileNotFoundError(msg)
    return bytes(blob.download_as_bytes())


def to_shard(
    contents: bytes, valid_time: datetime.datetime, lead: int, extent: Extent
) -> xr.Dataset:
    """A binary ``pcr`` field from a grid: energy forcing per metre != 0.

    Raises
    ------
    ValueError
        If the grid's own forecast reference time is not ``lead`` hours before ``valid_time``.
    """
    ds = api.load_netcdf(contents)
    init = forecast_runs.init_for(valid_time, lead)
    found = np.unique(ds["forecast_reference_time"].values)
    if found.size != 1 or found[0] != np.datetime64(init, "ns"):
        msg = (
            f"grid holds forecast reference time {found}, not {init} ({lead} h before {valid_time})"
        )
        raise ValueError(msg)

    ds = api.crop(ds, extent)
    pcr = ds["ef_per_m"] != 0
    pcr.attrs = {"long_name": "Persistent contrail region flag", "units": "1"}
    shard = ds.assign(pcr=pcr)[["pcr"]].drop_vars("forecast_reference_time")
    return forecast_runs.record_init(shard, init, valid_time)


def preprocess_contrails_org(
    valid_time: datetime.datetime,
    flight_levels: Sequence[int],
    *,
    lead_offsets: Sequence[int],
    extent: Extent,
) -> Iterator[tuple[tuple[int, int], xr.Dataset]]:
    """The shard of each flight level and lead for one valid time, one lead at a time."""
    for lead in forecast_runs.leads(valid_time, lead_offsets, has_analysis=False):
        for fl in flight_levels:
            yield (fl, lead), to_shard(fetch(valid_time, fl, lead), valid_time, lead, extent)


def main() -> None:
    """Program entrypoint."""
    parser = cli.shard_parser(__doc__)
    parser.add_argument(
        "--lead-offsets",
        type=forecast_runs.parse_offsets,
        default=config.LEAD_OFFSETS,
        help="hours added to each valid time's shortest lead, such as 0,12,24,36 (the default)",
    )
    args = parser.parse_args()

    produce = functools.partial(
        preprocess_contrails_org,
        lead_offsets=args.lead_offsets,
        extent=config.REGIONS[args.region],
    )
    job = cli.shard_job_for(args, SOURCE, lead_offsets=args.lead_offsets, has_analysis=False)
    cli.run(args, job, produce, "contrails-org")


if __name__ == "__main__":
    main()
