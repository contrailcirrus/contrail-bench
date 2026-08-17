"""Regression tests for ragged-shard handling in _concatenate_fl_time.

Before this fix, xr.combine_nested crashed outright on any day with a missing
(time, flight_level) shard -- and missing shards are routine (partial source
coverage), not exceptional. These tests reproduce that exact ragged scenario
directly against _concatenate_fl_time (bypassing Beam) for speed, plus one
end-to-end test through evaluate_beam with a Dataloader that raises
FileNotFoundError for specific shards.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from apache_beam.options.pipeline_options import PipelineOptions

from contrailbench.data import Dataloader
from contrailbench.forecast import Forecast, _concatenate_fl_time
from contrailbench.metrics import Metric


def _shard(time, flight_level, value):
    ds = xr.Dataset(
        {"x": (("rhi_threshold",), np.array([value, value + 1], dtype="float64"))},
        coords={"rhi_threshold": [0.9, 1.0]},
    )
    return ds.assign_coords(time=pd.Timestamp(time), flight_level=flight_level)


TIMES = [pd.Timestamp("2024-09-01T00:00"), pd.Timestamp("2024-09-01T01:00")]
FLIGHT_LEVELS = [310, 320, 330]


def test_ragged_group_no_longer_raises():
    """The exact scenario that crashed the old xr.combine_nested implementation:
    one (time, flight_level) combination missing from the middle of the group."""
    shards = [_shard(t, fl, 1.0) for t in TIMES for fl in FLIGHT_LEVELS if (t, fl) != (TIMES[1], 320)]

    # this must not raise
    result = _concatenate_fl_time(shards, TIMES, FLIGHT_LEVELS)

    assert result.sizes["time"] == 2
    assert result.sizes["flight_level"] == 3


def test_missing_combination_is_nan_filled():
    shards = [_shard(t, fl, 1.0) for t in TIMES for fl in FLIGHT_LEVELS if (t, fl) != (TIMES[1], 320)]
    result = _concatenate_fl_time(shards, TIMES, FLIGHT_LEVELS)

    missing = result["x"].sel(time=TIMES[1], flight_level=320)
    assert bool(missing.isnull().all())

    present = result["x"].sel(time=TIMES[0], flight_level=310)
    assert not bool(present.isnull().any())


def test_coverage_variable_marks_present_and_missing_cells():
    shards = [_shard(t, fl, 1.0) for t in TIMES for fl in FLIGHT_LEVELS if (t, fl) != (TIMES[1], 320)]
    result = _concatenate_fl_time(shards, TIMES, FLIGHT_LEVELS)

    assert result["coverage"].sel(time=TIMES[1], flight_level=320).item() == 0
    assert result["coverage"].sel(time=TIMES[0], flight_level=310).item() == 1
    # 5 of 6 combinations present
    assert int(result["coverage"].sum()) == 5


def test_entirely_missing_flight_level_still_reindexed():
    """Not just a hole in the middle -- an entire flight level absent from
    every time in the group."""
    shards = [_shard(t, fl, 1.0) for t in TIMES for fl in FLIGHT_LEVELS if fl != 320]
    result = _concatenate_fl_time(shards, TIMES, FLIGHT_LEVELS)

    assert 320 in result["flight_level"].values
    assert bool(result["x"].sel(flight_level=320).isnull().all())
    assert int(result["coverage"].sel(flight_level=320).sum()) == 0


def test_complete_group_matches_previous_behavior():
    """No missing shards -- coverage is all 1s and no NaN is introduced."""
    shards = [_shard(t, fl, 1.0) for t in TIMES for fl in FLIGHT_LEVELS]
    result = _concatenate_fl_time(shards, TIMES, FLIGHT_LEVELS)

    assert int(result["coverage"].sum()) == len(TIMES) * len(FLIGHT_LEVELS)
    assert not bool(result["x"].isnull().any())


class _FlakyForecastLoader(Dataloader):
    """Raises FileNotFoundError for one specific (time, flight_level)."""

    def __init__(self, grid, missing):
        self.grid = grid
        self.missing = missing

    def data(self, time, flight_level, extent):
        if (pd.Timestamp(time), flight_level) == self.missing:
            raise FileNotFoundError("simulated missing forecast shard")
        longitude, latitude = self.grid
        pcr = xr.DataArray(
            np.ones((len(longitude), len(latitude), 2), dtype=bool),
            dims=("longitude", "latitude", "rhi_threshold"),
            coords={"longitude": longitude, "latitude": latitude, "rhi_threshold": [0.9, 1.0]},
        )
        return xr.Dataset({"pcr": pcr})


class _FixedObsLoader(Dataloader):
    def __init__(self, grid):
        self.grid = grid

    def data(self, time, flight_level, extent):
        longitude, latitude = self.grid
        return xr.Dataset(
            {
                "longitude": ("cell", [longitude[0]]),
                "latitude": ("cell", [latitude[0]]),
                "area": ("cell", [1.0]),
            }
        )


class _AreaMetric(Metric):
    def statistics(self, forecast, data):
        predicted = forecast["pcr"].sel(longitude=data["longitude"], latitude=data["latitude"])
        return xr.Dataset({"area_in_pcr": data["area"].where(predicted).sum("cell")})


def test_evaluate_beam_end_to_end_with_one_missing_shard(tmp_path):
    """A FileNotFoundError from the forecast Dataloader for one shard must not
    crash the whole day's pipeline -- the day's output file is still produced,
    with that one (time, flight_level) NaN-filled and coverage=0."""
    grid = (np.array([0.0, 5.0]), np.array([0.0, 5.0]))
    times = [pd.Timestamp("2024-09-01T00:00"), pd.Timestamp("2024-09-01T01:00")]
    flight_levels = [310, 320]
    missing = (times[1], 320)

    forecast = Forecast(
        _FlakyForecastLoader(grid, missing), times=times, flight_levels=flight_levels, extent=None
    )

    outputs = str(tmp_path / "outputs")
    intermediates = str(tmp_path / "intermediates")
    forecast.evaluate_beam(
        outputs,
        intermediates,
        PipelineOptions(runner="direct", direct_num_workers=1),
        metric=_AreaMetric(_FixedObsLoader(grid)),
    )

    day_file = f"{outputs}/20240901.nc"
    result = xr.open_dataset(day_file)
    assert result.sizes["time"] == 2
    assert result.sizes["flight_level"] == 2
    assert bool(result["metric.area_in_pcr"].sel(time=times[1], flight_level=320).isnull().all())
    assert result["coverage"].sel(time=times[1], flight_level=320).item() == 0
    assert result["coverage"].sel(time=times[0], flight_level=310).item() == 1
