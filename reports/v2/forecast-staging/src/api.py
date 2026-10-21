"""Shared code for the product APIs (contrails.org and Google).

API keys, requests with retries, and reading and cropping the netCDF the APIs return.
"""

import functools
import io
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping

import xarray as xr

from src import config
from src.grid import Extent

LOGGER = logging.getLogger(__name__)

#: A request that fails with one of these statuses is retried, this many times, this many seconds
#: apart
RETRY_ON = (503,)
RETRIES = 2
RETRY_BACKOFF_S = 60.0

REQUEST_TIMEOUT_S = 60.0


class HttpError(RuntimeError):
    """An HTTP request failed; ``status`` is the response code."""

    def __init__(self, status: int, url: str) -> None:
        super().__init__(f"HTTP {status} from {url}")
        self.status = status


@functools.cache
def get_secret(name: str) -> str:
    """Get a secret from GCP Secret Manager (read once per process).

    Pipelines call this where the key is used, on the worker, so a key is never written into the
    job that is submitted.
    """
    from google.cloud import secretmanager

    client = secretmanager.SecretManagerServiceClient()
    response = client.access_secret_version(
        name=f"projects/{config.PROJECT}/secrets/{name}/versions/latest"
    )
    return response.payload.data.decode()


def get(url: str, params: Mapping[str, str], headers: Mapping[str, str]) -> bytes:
    """GET ``url`` with query ``params`` and return the body.

    A temporary failure (status in :data:`RETRY_ON`) is retried after a pause. Any other error
    status raises :class:`HttpError`.
    """
    request = urllib.request.Request(
        f"{url}?{urllib.parse.urlencode(params)}", headers=dict(headers)
    )
    attempt = 0
    while True:
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
                return bytes(response.read())
        except urllib.error.HTTPError as error:
            if error.code not in RETRY_ON or attempt >= RETRIES:
                raise HttpError(error.code, url) from error
            LOGGER.warning("%s; retrying after %.0f s", HttpError(error.code, url), RETRY_BACKOFF_S)
            time.sleep(RETRY_BACKOFF_S)
            attempt += 1


def load_netcdf(contents: bytes) -> xr.Dataset:
    """Read netCDF bytes into memory."""
    return xr.load_dataset(io.BytesIO(contents), engine="h5netcdf", decode_timedelta=True)


def crop(ds: xr.Dataset, extent: Extent) -> xr.Dataset:
    """Crop a global grid (ascending coordinates) to ``extent``.

    The APIs have no bounding-box parameter, and an uncropped file is many times larger than the
    benchmark needs.
    """
    lon_min, lon_max, lat_min, lat_max = extent
    return ds.sel(longitude=slice(lon_min, lon_max), latitude=slice(lat_min, lat_max))
