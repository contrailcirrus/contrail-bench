"""Regression tests for contrailbench.data.PCRStoreDataloader.

The bit-identity test guards the one genuinely new piece of arithmetic in the
whole port: the vectorized RHi-threshold sweep must match, value for value,
what the pre-port pipelines computed one threshold at a time via
``apply_rhi_threshold(rhi, sac, threshold) = (rhi > threshold) & (sac > 0)``.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from contrailbench import pcr
from contrailbench.data import PCRStoreDataloader


def _old_apply_rhi_threshold(
    rhi: xr.DataArray, sac: xr.DataArray, threshold: float
) -> xr.DataArray:
    """Verbatim copy of the pre-port per-threshold logic, kept here only as an
    independent reference for the bit-identity test below -- not imported from
    production code, since that code path is exactly what this port replaces."""
    return (rhi > threshold) & (sac > 0)


@pytest.fixture()
def synthetic_store(tmp_path):
    """A minimal on-disk store matching the real schema: `time`/`level` dims of
    size 1, `longitude`/`latitude`, `rhi`/`sac`/`pcr` variables."""
    rng = np.random.default_rng(0)
    longitude = np.linspace(-134, -63, 20)
    latitude = np.linspace(20, 50, 15)
    rhi = rng.uniform(0.5, 1.5, size=(len(longitude), len(latitude), 1, 1)).astype("float32")
    sac = rng.integers(0, 2, size=(len(longitude), len(latitude), 1, 1)).astype("float32")

    ds = xr.Dataset(
        {
            "rhi": (("longitude", "latitude", "level", "time"), rhi),
            "sac": (("longitude", "latitude", "level", "time"), sac),
            "pcr": (
                ("longitude", "latitude", "level", "time"),
                ((rhi > 1.0) & (sac > 0)).astype("float32"),
            ),
        },
        coords={
            "longitude": longitude,
            "latitude": latitude,
            "level": [287.0],
            "time": [pd.Timestamp("2024-09-01")],
        },
    )

    path = tmp_path / "store"
    path.mkdir()
    ts = 1725148800  # 2024-09-01T00:00:00Z
    ds.to_netcdf(path / f"{ts}_310.nc")
    return str(path), ds


def test_threshold_sweep_is_bit_identical_to_per_threshold_reference(synthetic_store):
    path, ds = synthetic_store
    loader = PCRStoreDataloader(path=path, rhi_thresholds=pcr.RHI_THRESHOLDS)

    result = loader.data(pd.Timestamp("2024-09-01"), 310, None)

    rhi = ds["rhi"].squeeze(("time", "level"), drop=True)
    sac = ds["sac"].squeeze(("time", "level"), drop=True)
    for threshold in pcr.RHI_THRESHOLDS:
        expected = _old_apply_rhi_threshold(rhi, sac, threshold)
        actual = result["pcr"].sel(rhi_threshold=threshold)
        assert np.array_equal(actual.values, expected.values), f"mismatch at threshold={threshold}"


def test_output_has_no_time_or_level_dims(synthetic_store):
    path, _ = synthetic_store
    loader = PCRStoreDataloader(path=path)
    result = loader.data(pd.Timestamp("2024-09-01"), 310, None)
    assert "time" not in result.dims
    assert "level" not in result.dims


def test_pcr_dimension_shape_matches_threshold_count(synthetic_store):
    path, _ = synthetic_store
    thresholds = [0.9, 1.0, 1.1]
    loader = PCRStoreDataloader(path=path, rhi_thresholds=thresholds)
    result = loader.data(pd.Timestamp("2024-09-01"), 310, None)
    assert result["pcr"].sizes["rhi_threshold"] == 3
    assert list(result["rhi_threshold"].values) == thresholds


def test_extent_crops_to_requested_bounding_box(synthetic_store):
    path, _ = synthetic_store
    loader = PCRStoreDataloader(path=path)
    full = loader.data(pd.Timestamp("2024-09-01"), 310, None)
    cropped = loader.data(pd.Timestamp("2024-09-01"), 310, (-100, -90, 25, 35))

    assert cropped.sizes["longitude"] < full.sizes["longitude"]
    assert cropped.sizes["latitude"] < full.sizes["latitude"]
    assert cropped["longitude"].max() <= -90
    assert cropped["longitude"].min() >= -100


def test_reads_real_metoffice_store_file():
    """Smoke test against an actual on-disk forecast file, not just synthetic data."""
    loader = PCRStoreDataloader(
        path="/home/jg931/contrails_org/contrail-bench/reports/jay_extension/data/metoffice"
    )
    result = loader.data(pd.Timestamp("2024-09-01T00:00:00"), 310, None)

    assert "time" not in result.dims
    assert "level" not in result.dims
    assert result["pcr"].sizes["rhi_threshold"] == len(pcr.RHI_THRESHOLDS)
    assert result["pcr"].dtype == bool
