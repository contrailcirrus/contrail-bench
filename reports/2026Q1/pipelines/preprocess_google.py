"""Preprocess Google forecasts.

No dataflow support because of issues with API rate limiting.
"""

import asyncio
import datetime
import itertools
import logging
from typing import Iterable

import aiohttp
import aiolimiter
import gcsfs
import numpy as np
import pandas as pd
import xarray as xr
from google.cloud import secretmanager

from pycontrails.utils import coroutines, temp

logging.basicConfig(
    level="WARNING",
    format="[%(asctime)s.%(msecs)03d] [%(levelname)s] [%(module)s] [%(funcName)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel("INFO")


# Pipeline parameters

#: Forecast times
TIMES = pd.date_range("2024-01-01 00:00", "2024-12-31 23:00", freq="1h").to_pydatetime().tolist()

#: Required flight levels
FLIGHT_LEVELS = list(range(270, 450, 10))

#: Limit size of forecast API bursts
GOOGLE_FORECAST_API_BURST_LIMIT = 24

#: Status codes on which to retry bursts of queries
GOOGLE_FORECAST_API_RETRY_ON = (503,)

#: Retry backoff (seconds)
GOOGLE_FORECAST_API_RETRY_BACKOFF = 60

#: GCP bucket for temporary assets
GCP_TMPDIR = "gs://contrails-301217-contrail-bench/tmp/2026Q1/google"


def get_secret(name: str) -> str:
    """Get secret from GCP secret manager.

    Parameters
    ----------
    name : str
        Name of secret    
        
    Returns
    -------
    str
        Value of secret

    """
    """Get secret from GCP secret manager."""
    client = secretmanager.SecretManagerServiceClient()
    project = "contrails-301217"
    full_name = f"projects/{project}/secrets/{name}/versions/latest"
    response = client.access_secret_version(name=full_name)
    return response.payload.data.decode()


async def get_forecast(
    time: datetime.datetime,
    key: str,
    session: aiohttp.ClientSession
) -> tuple[datetime.datetime, bytes]:
    """Get forecast at a single time.

    Parameters
    ----------
    time : datetime.datetime
        Target time

    key : str
        Google contrails API key

    session : aiohttp.ClientSession
        HTTP session

    Returns
    -------
    tuple[datetime.datetime], bytes
        Forecast time and contents as netcdf

    """
    url = "https://contrails.googleapis.com/v2/grids"
    params = {
        "time": time.isoformat(),
        "data": "persistent_formation_probability"
    }
    headers = {"x-goog-api-key": key}

    async with session.get(url, params=params, headers=headers) as resp:
        return time, await resp.read()


def write_to_gcs(time: datetime.datetime, contents: bytes) -> None:
    """Write forecast contents to GCS."""
    with temp.temp_file() as tmp:
        with open(tmp, "wb") as f:
            f.write(contents)
        ds = xr.load_dataset(tmp, engine="netcdf4", decode_timedelta=True)
        
    ppcr = ds["persistent_formation_probability"] / 100.0
    ppcr.attrs = {"long_name": "Persistent contrail region probability", "units": "1"}  # type: ignore
    ds["ppcr"] = ppcr

    fs = gcsfs.GCSFileSystem()
    for flight_level in FLIGHT_LEVELS:
        sink = f"{GCP_TMPDIR}/{int(time.timestamp())}_{flight_level}.nc"
        sl = ds.sel(flight_level=[flight_level])
        with temp.temp_file() as tmp:
            sl[["ppcr"]].to_netcdf(tmp)
            fs.put(tmp, sink)


async def submit_burst(
    times: Iterable[datetime.datetime],
    limiter: aiolimiter.AsyncLimiter
) -> bool:
    """Submit a single burst of requests.

    Parameters
    ----------
    times : list[datetime.datetime]
        Times included in burst

    limiter : aiolimiter.AsyncLimiter
        Burst rate limiter

    Returns
    -------
    bool
        Flag indicating whether the burst should be retried

    """
    async with limiter:
        try:
            key = get_secret("google-contrails-api-key")
            async with aiohttp.ClientSession(raise_for_status=True) as session:
                tasks = [get_forecast(time, key, session) for time in times]
                for result in asyncio.as_completed(tasks):
                    time, contents = await result
                    write_to_gcs(time, contents)

        except aiohttp.ClientResponseError as e:
            if e.status in GOOGLE_FORECAST_API_RETRY_ON:
                logger.warning(f"Burst failed ({e.status} {e.message}). Retrying after backoff.")
                await asyncio.sleep(GOOGLE_FORECAST_API_RETRY_BACKOFF)
                return True
            raise e

        return False


async def preprocess_google() -> None:
    """Preprocess Google forecasts."""
    # limit to a single burst per minute
    limiter = aiolimiter.AsyncLimiter(1)
    for times in itertools.batched(TIMES, GOOGLE_FORECAST_API_BURST_LIMIT):
        # Skip three dates missing from Google's backfill
        if times[0].date() in [
            datetime.date(2024, 4, 11),
            datetime.date(2024, 4, 12),
            datetime.date(2024, 6, 6)
        ]:
            continue
        logger.info(f"Submitting burst starting with {times[0]}")
        while await submit_burst(times, limiter):
            pass
        logger.info(f"Finished processing burst starting with {times[0]}")


def main() -> None:
    """Program entrypoint."""
    coroutines.run(preprocess_google())


if __name__ == "__main__":
    main()
