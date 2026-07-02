"""I/O operations."""

import tempfile

import fsspec
import xarray as xr


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


def write(url: str, data: bytes) -> None:
    """Write data to URL.

    Will be atomic if the implementation of ``fsspec.put``
    is atomic for the target filesystem.

    Parameters
    ----------
    url : str
        Location where data is to be written. Must be on a filesystem supported by fsppec.

    data : bytes
        Data to be written.
    """
    fs, path = fsspec.core.url_to_fs(url)
    with tempfile.NamedTemporaryFile(delete=True, delete_on_close=False) as f:
        f.write(data)
        f.close()
        fs.put(f.name, path)
