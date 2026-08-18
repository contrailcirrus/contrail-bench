"""Regression tests for the netCDF-to-long-format notebook adapter."""

import datetime
import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from contrailbench.data import IAGOSDataloader, PCRStoreDataloader
from contrailbench.forecast import Forecast
from contrailbench.metrics import HitRate
from contrailbench.pcr import CONUS_EXTENT, PCR_FLIGHT_LEVELS
from contrailbench.pipelines._common import DATA_DIR
from contrailbench.pipelines.adapter import to_long_format


def test_raises_for_unknown_metric_prefix():
    ds = xr.Dataset({"iagos.observed_pcr_area": (("time",), [1.0])})
    with pytest.raises(ValueError, match="no variables found"):
        to_long_format(ds, "gruan")


def test_strips_metric_prefix_from_columns():
    ds = xr.Dataset(
        {
            "iagos.observed_pcr_area": (("rhi_threshold",), [1.0, 2.0]),
            "iagos.observed_pcr_area_in_forecast_pcr": (("rhi_threshold",), [0.5, 1.5]),
        },
        coords={"rhi_threshold": [0.9, 1.0]},
    )
    df = to_long_format(ds, "iagos")
    assert set(df.columns) >= {
        "observed_pcr_area",
        "observed_pcr_area_in_forecast_pcr",
        "rhi_threshold",
    }
    assert not any(c.startswith("iagos.") for c in df.columns)


def test_drops_rows_with_zero_coverage():
    ds = xr.Dataset(
        {
            "iagos.observed_pcr_area": (("time", "flight_level"), [[1.0, np.nan]]),
            "coverage": (("time", "flight_level"), [[1, 0]]),
        },
        coords={"time": [pd.Timestamp("2024-09-01")], "flight_level": [310, 320]},
    )
    df = to_long_format(ds, "iagos")
    assert len(df) == 1
    assert df["flight_level"].iloc[0] == 310


def test_matches_golden_schema_against_real_data():
    """The adapter's output schema/values must match the actual pre-port
    golden file for the same hour -- not just structurally, but exactly."""
    forecast_store = str(DATA_DIR / "metoffice-prefix-buggy")
    obs_store = str(DATA_DIR / "_obs_cache" / "iagos-prefix-buggy")
    golden_dir = str(DATA_DIR / "metoffice-iagos-contrailwatch-region")

    if not all(os.path.isdir(d) for d in (forecast_store, obs_store, golden_dir)):
        pytest.skip("local data mirror not present in this environment")

    time = datetime.datetime(2024, 10, 31, 0, 0)
    forecast = Forecast(
        PCRStoreDataloader(path=forecast_store),
        times=[time],
        flight_levels=PCR_FLIGHT_LEVELS,
        extent=CONUS_EXTENT,
    )
    result = forecast.evaluate(iagos=HitRate(IAGOSDataloader(path=obs_store)))

    adapted = to_long_format(result, "iagos")
    golden = pd.read_parquet(f"{golden_dir}/2024103100.pq")

    adapted_sorted = adapted.sort_values(["flight_level", "rhi_threshold"]).reset_index(drop=True)
    golden_sorted = golden.sort_values(["flight_level", "rhi_threshold"]).reset_index(drop=True)

    assert list(adapted_sorted["flight_level"]) == list(golden_sorted["flight_level"])
    assert np.allclose(
        adapted_sorted["observed_pcr_area_in_forecast_pcr"],
        golden_sorted["observed_pcr_area_in_forecast_pcr"],
        rtol=1e-9,
        atol=1e-6,
    )
    assert np.allclose(
        adapted_sorted["observed_pcr_area"],
        golden_sorted["observed_pcr_area"],
        rtol=1e-9,
        atol=1e-6,
    )
