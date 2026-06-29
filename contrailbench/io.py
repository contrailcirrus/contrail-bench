"""I/O operations."""

import tempfile

import fsspec
import xarray as xr


def load_dataset(url: str) -> xr.Dataset:
    """Load NetCDF dataset."""
    fs, path = fsspec.core.url_to_fs(url)
    with tempfile.NamedTemporaryFile(delete=True, delete_on_close=False) as f:
        f.close()
        fs.get(path, f.name)
        return xr.load_dataset(f.name)


def write(url: str, data: bytes) -> None:
    """Write data to URL.

    Will be atomic if the implementation of ``fsspec.put``
    is atomic for the target filesystem.
    """
    fs, path = fsspec.core.url_to_fs(url)
    with tempfile.NamedTemporaryFile(delete=True, delete_on_close=False) as f:
        f.write(data)
        f.close()
        fs.put(f.name, path)
