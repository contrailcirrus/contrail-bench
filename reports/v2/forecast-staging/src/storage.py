"""List and write shard files on local disk or Google Cloud Storage.

Paths are either local or ``gs://bucket/object`` URIs. Cloud libraries are imported inside the
functions that need them, so local runs do not require them.
"""

import functools
import io
import logging
import os
import tempfile
import threading
import time
from typing import Any

import xarray as xr

LOGGER = logging.getLogger(__name__)

#: Upload budget for one shard: a per-request timeout and an overall retry deadline
UPLOAD_TIMEOUT_S = 30
UPLOAD_DEADLINE_S = 60

_client: Any = None
_client_lock = threading.Lock()


def is_gcs_path(path: str) -> bool:
    """Whether ``path`` is a ``gs://`` URI rather than a local path."""
    return path.startswith("gs://")


def parse_gcs_path(path: str) -> tuple[str, str]:
    """Split ``gs://bucket/blob/name`` into ``(bucket, "blob/name")``."""
    bucket, _, blob = path.removeprefix("gs://").partition("/")
    return bucket, blob


def gcs_client() -> Any:
    """Process-wide Cloud Storage client (looking up credentials takes about 1.5 s)."""
    global _client  # noqa: PLW0603
    with _client_lock:
        if _client is None:
            from google.cloud import storage

            _client = storage.Client()
        return _client


@functools.lru_cache(maxsize=16)
def _open_zarr(path: str, thread: int) -> xr.Dataset:
    remote = path.startswith(("gs://", "s3://"))
    return xr.open_zarr(
        path, chunks=None, storage_options={"skip_instance_cache": True} if remote else None
    )


def open_zarr(path: str) -> xr.Dataset:
    """Open a zarr store lazily (metadata only), keeping the ones opened before.

    Opening a store costs about as much as reading an hour from it, and a job reads many hours of
    the same day. The calling thread is part of the cache key, so no filesystem object is shared
    between threads.
    """
    return _open_zarr(path, threading.get_ident())


def list_names(path: str) -> list[str]:
    """Names of the files directly under a local directory or ``gs://`` prefix.

    Returns an empty list if the directory does not exist or the prefix matches nothing.
    """
    if is_gcs_path(path):
        bucket, prefix = parse_gcs_path(path)
        if prefix and not prefix.endswith("/"):
            prefix = f"{prefix}/"
        return [blob.name[len(prefix) :] for blob in gcs_client().list_blobs(bucket, prefix=prefix)]
    return os.listdir(path) if os.path.isdir(path) else []


def write_netcdf_atomic(ds: xr.Dataset, path: str) -> None:
    """Write ``ds`` to a local or GCS path so a partial file never appears at ``path``.

    A GCS object only becomes visible once the upload completes, so a direct upload is already
    atomic. A local write goes to a temporary file in the same directory that is then renamed.
    Without that, a run killed mid-write would leave a truncated file that the existence check
    of a later run would take as finished.

    For GCS the dataset is serialized into memory first (a netCDF writer needs a seekable file and
    an upload stream is not) and uploaded in one call with bounded timeouts, so an unresponsive
    upload fails promptly instead of hanging a worker.
    """
    if is_gcs_path(path):
        from google.cloud.storage.retry import DEFAULT_RETRY

        buffer = io.BytesIO()
        ds.to_netcdf(buffer, engine="h5netcdf")
        started = time.monotonic()
        bucket, blob_name = parse_gcs_path(path)
        blob = gcs_client().bucket(bucket).blob(blob_name)
        nbytes = buffer.getbuffer().nbytes
        buffer.seek(0)
        blob.upload_from_file(
            buffer,
            rewind=True,
            timeout=UPLOAD_TIMEOUT_S,
            retry=DEFAULT_RETRY.with_deadline(UPLOAD_DEADLINE_S),
        )
        LOGGER.info(
            "gcs_write path=%s bytes=%d seconds=%.2f", path, nbytes, time.monotonic() - started
        )
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".nc.tmp")
    os.close(fd)
    try:
        ds.to_netcdf(tmp_path, engine="h5netcdf")
        os.replace(tmp_path, path)
    except BaseException:
        os.remove(tmp_path)
        raise
