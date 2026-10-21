"""Preprocess contrails.org forecasts.

contrails.org serves a CoCiP forecast driven by ECMWF HRES. The API takes a valid time and flight
level and picks the HRES run itself, so there is no ``--run``: the shard records which run it was
(the ``forecast_reference_time`` coordinate). Each shard holds a binary ``pcr`` field, true where
the energy forcing per metre is nonzero, of either sign. That is how the product defines its
region.

The API has no bounding box, so each global response is cropped to the region.
"""

import datetime
import functools
from collections.abc import Sequence

import xarray as xr

from src import api, cli, config
from src.grid import Extent

# Pipeline parameters

#: Output directory name under ``forecasts/``
SOURCE = "contrails-org"

#: Secret Manager secret holding the API key
API_KEY_SECRET = "contrails-api-key"

#: API address
URL = "https://api.contrails.org/v1/grids"


def fetch(time: datetime.datetime, flight_level: int) -> bytes:
    """The raw API response for one time and flight level."""
    params = {
        "aircraft_class": "default",
        "flight_level": str(flight_level),
        "time": time.strftime("%Y-%m-%dT%H"),
        "units": "ef_per_m",
    }
    headers = {"x-api-key": api.get_secret(API_KEY_SECRET)}
    return api.get(URL, params, headers)


def to_shard(contents: bytes, extent: Extent) -> xr.Dataset:
    """A binary ``pcr`` field from an API response: energy forcing per metre != 0."""
    ds = api.crop(api.load_netcdf(contents), extent)
    pcr = ds["ef_per_m"] != 0
    pcr.attrs = {"long_name": "Persistent contrail region flag", "units": "1"}
    return ds.assign(pcr=pcr)[["pcr"]]


def preprocess_contrails_org(
    time: datetime.datetime, flight_levels: Sequence[int], *, extent: Extent
) -> dict[int, xr.Dataset]:
    """One shard per flight level for one valid time (one request per level)."""
    return {fl: to_shard(fetch(time, fl), extent) for fl in flight_levels}


def main() -> None:
    """Program entrypoint."""
    parser = cli.shard_parser(__doc__)
    args = parser.parse_args()

    produce = functools.partial(preprocess_contrails_org, extent=config.REGIONS[args.region])
    job = cli.shard_job_for(args, SOURCE)
    cli.run(args, job, produce, "contrails-org")


if __name__ == "__main__":
    main()
