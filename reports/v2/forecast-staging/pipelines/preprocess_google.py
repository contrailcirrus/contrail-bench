"""Preprocess Google contrail forecasts.

Google's API takes a valid time and returns the persistent contrail formation probability on a
global grid for every flight level. Each shard holds ``ppcr``, that probability scaled to 0 to 1,
cropped to the region.

This runs locally rather than on Beam, because the API's rate limit applies across the whole job:
requests go out in bursts, with the requests of a burst running concurrently.

The product is a reprocessed historical archive, not a forecast issued before the valid time: the
reference time recorded in a shard is when it was retrieved. Its lead time is undefined.
"""

import concurrent.futures
import datetime
import functools
import itertools
import time
from collections.abc import Sequence

import xarray as xr

from src import api, cli, config, shard_job
from src.grid import Extent

# Pipeline parameters

#: Output directory name under ``forecasts/``
SOURCE = "google"

#: Secret Manager secret holding the API key
API_KEY_SECRET = "google-contrails-api-key"

#: API address
URL = "https://contrails.googleapis.com/v2/grids"

#: The API allows a burst of this many requests, once a minute
BURST = 24
BURST_INTERVAL_S = 60.0


def fetch(valid_time: datetime.datetime) -> bytes:
    """The raw API response for one valid time (all flight levels)."""
    params = {"time": valid_time.isoformat(), "data": "persistent_formation_probability"}
    headers = {"x-goog-api-key": api.get_secret(API_KEY_SECRET)}
    return api.get(URL, params, headers)


def to_shards(
    contents: bytes, extent: Extent, flight_levels: Sequence[int]
) -> dict[int, xr.Dataset]:
    """Per-flight-level ``ppcr`` (probability, 0 to 1) shards from one API response."""
    ds = api.crop(api.load_netcdf(contents), extent)
    ppcr = ds["persistent_formation_probability"] / 100.0
    ppcr.attrs = {"long_name": "Persistent contrail region probability", "units": "1"}
    ds = ds.assign(ppcr=ppcr)
    return {fl: ds.sel(flight_level=[fl])[["ppcr"]] for fl in flight_levels}


def preprocess_google(
    valid_time: datetime.datetime, flight_levels: Sequence[int], *, extent: Extent
) -> dict[int, xr.Dataset]:
    """One shard per flight level for one valid time (one request)."""
    return to_shards(fetch(valid_time), extent, flight_levels)


def run_bursts(
    job: shard_job.ShardJob,
    produce: shard_job.Producer,
    times: Sequence[datetime.datetime],
    options: object = None,
    *,
    force: bool = False,
) -> tuple[int, int]:
    """Write ``times`` in rate-limited bursts, skipping the ones already done unless ``force``.

    Bursts start at least ``BURST_INTERVAL_S`` apart. A time that fails is skipped (and retried
    by the next run) if the job says so, otherwise the job fails. ``options`` is unused; it is
    there so this can stand in for :func:`shard_job.run`.

    Returns
    -------
    tuple[int, int]
        ``(already_done, run_now)``
    """
    todo = list(times) if force else shard_job.pending(job, times)

    def write(stamp: datetime.datetime) -> None:
        with shard_job.skipping_failures(job, stamp):
            shard_job.write_shards(job, stamp, produce(stamp, job.flight_levels))

    last_start: float | None = None
    for batch in itertools.batched(todo, BURST, strict=False):
        if last_start is not None:
            time.sleep(max(0.0, last_start + BURST_INTERVAL_S - time.monotonic()))
        last_start = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(batch)) as pool:
            list(pool.map(write, batch))
    return len(times) - len(todo), len(todo)


def main() -> None:
    """Program entrypoint."""
    parser = cli.shard_parser(__doc__, runner=False)
    args = parser.parse_args()

    produce = functools.partial(preprocess_google, extent=config.REGIONS[args.region])
    job = cli.shard_job_for(args, SOURCE)
    cli.run(args, job, produce, "google", run_job=run_bursts)


if __name__ == "__main__":
    main()
