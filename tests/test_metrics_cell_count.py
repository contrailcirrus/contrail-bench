"""Regression tests for HitRate's n_obs_cells/n_obs_cells_in_pcr statistics."""

import numpy as np
import xarray as xr

from contrailbench.data import IAGOSDataloader
from contrailbench.metrics import HitRate


def test_n_obs_cells_counts_all_observation_cells():
    forecast = xr.Dataset(
        {"pcr": (("longitude", "latitude", "rhi_threshold"), np.ones((3, 3, 2), dtype=bool))},
        coords={"longitude": [0, 1, 2], "latitude": [0, 1, 2], "rhi_threshold": [0.9, 1.0]},
    )
    data = xr.Dataset(
        {
            "longitude": ("cell", [0, 1, 2]),
            "latitude": ("cell", [0, 1, 2]),
            "area": ("cell", [1.0, 2.0, 3.0]),
        }
    )
    result = HitRate(IAGOSDataloader()).statistics(forecast, data)
    assert int(result["n_obs_cells"]) == 3


def test_n_obs_cells_in_pcr_varies_by_threshold():
    forecast = xr.Dataset(
        {
            "pcr": (
                ("longitude", "latitude", "rhi_threshold"),
                np.array(
                    [
                        [[True, False]],
                        [[True, False]],
                        [[True, True]],
                    ]
                ),
            )
        },
        coords={"longitude": [0, 1, 2], "latitude": [0], "rhi_threshold": [0.9, 1.0]},
    )
    data = xr.Dataset(
        {
            "longitude": ("cell", [0, 1, 2]),
            "latitude": ("cell", [0, 0, 0]),
            "area": ("cell", [1.0, 1.0, 1.0]),
        }
    )
    result = HitRate(IAGOSDataloader()).statistics(forecast, data)
    assert result["n_obs_cells_in_pcr"].sel(rhi_threshold=0.9).item() == 3
    assert result["n_obs_cells_in_pcr"].sel(rhi_threshold=1.0).item() == 1


def test_n_obs_cells_is_zero_for_empty_observations():
    forecast = xr.Dataset(
        {"pcr": (("longitude", "latitude", "rhi_threshold"), np.ones((2, 2, 1), dtype=bool))},
        coords={"longitude": [0, 1], "latitude": [0, 1], "rhi_threshold": [1.0]},
    )
    data = xr.Dataset(
        {
            "longitude": ("cell", np.array([], dtype="float64")),
            "latitude": ("cell", np.array([], dtype="float64")),
            "area": ("cell", np.array([], dtype="float64")),
        }
    )
    result = HitRate(IAGOSDataloader()).statistics(forecast, data)
    assert int(result["n_obs_cells"]) == 0
    assert int(result["n_obs_cells_in_pcr"].sel(rhi_threshold=1.0)) == 0
