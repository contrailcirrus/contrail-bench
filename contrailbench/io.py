"""I/O operations."""

import os
import tempfile

import fsspec
import xarray as xr
from fsspec.implementations.local import LocalFileSystem


def load_dataset(url: str) -> xr.Dataset:
    """Load NetCDF dataset.

    Parameters
    ----------
    url : str
        Location of netCDF dataset. Must be on a filesystem supported by fsspec.

    Returns
    -------
    xr.Dataset
        Contents of netCDF file as an in-memory dataset.
    """
    fs, path = fsspec.core.url_to_fs(url)
    with tempfile.NamedTemporaryFile(delete=True, delete_on_close=False) as f:
        f.close()
        fs.get(path, f.name)
        return xr.load_dataset(f.name)


def exists(url: str) -> bool:
    """Check whether a file exists at a URL.

    Parameters
    ----------
    url : str
        Location to check. Must be on a filesystem supported by fsspec.

    Returns
    -------
    bool
        True if a file already exists at ``url``.
    """
    fs, path = fsspec.core.url_to_fs(url)
    return fs.exists(path)


def write(url: str, data: bytes) -> None:
    """Write data to URL.

    Creates the parent directory first if it does not already exist --
    ``fsspec``'s local filesystem defaults to ``auto_mkdir=False``, so writing
    into a fresh directory would otherwise raise.

    On the local filesystem, writes atomically via a temp file created in
    ``url``'s own destination directory, followed by ``os.replace``.
    ``fsspec``'s local ``put`` is a bare ``shutil.copyfile`` directly to the
    destination path, so a process killed mid-write would otherwise leave a
    truncated file *at* ``url`` -- exactly what an existence-based resumability
    check would then wrongly treat as complete. Creating the temp file in the
    same directory (not the default ``/tmp``) guarantees it is on the same
    filesystem as the destination, which is what makes the final rename atomic
    per POSIX.

    On a remote filesystem (e.g. GCS), the object only becomes visible once
    the upload completes in full, so buffering through a local temp file
    followed by a single ``fs.put`` is already atomic there.

    Parameters
    ----------
    url : str
        Location where data is to be written. Must be on a filesystem supported by fsppec.

    data : bytes
        Data to be written.
    """
    fs, path = fsspec.core.url_to_fs(url)
    parent = os.path.dirname(path)
    if parent:
        fs.makedirs(parent, exist_ok=True)

    if isinstance(fs, LocalFileSystem):
        fd, tmp_path = tempfile.mkstemp(dir=parent or ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.replace(tmp_path, path)
        except BaseException:
            os.remove(tmp_path)
            raise
        return

    with tempfile.NamedTemporaryFile(delete=True, delete_on_close=False) as f:
        f.write(data)
        f.close()
        fs.put(f.name, path)
