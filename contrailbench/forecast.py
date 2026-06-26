"""Forecast evaluation."""

from collections.abc import Iterable

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import binary_dilation

from contrailbench.data import Dataloader
from contrailbench.metrics import Metric
from contrailbench.types import DatetimeLike


class Forecast:
    """PCR forecast evaluation."""

    def __init__(
        self,
        dataloader: Dataloader,
        times: Iterable[DatetimeLike],
        flight_levels: Iterable[int],
        extent: tuple[float, float, float, float] | None,
        buffers: Iterable[int],
    ) -> None:
        self.dataloader = dataloader
        self.times = [pd.to_datetime(t) for t in times]
        self.flight_levels = list(flight_levels)
        self.extent = extent
        self.buffers = list(buffers)

    def evaluate(self, path: str, **metrics: Metric) -> None:
        """Evaluate forecast against a list of metrics."""
        ds_list_list = []
        for time in self.times:
            ds_list = []
            for flight_level in self.flight_levels:
                ds_list.append(self._evaluate_shard(time, flight_level, metrics))
            ds_list_list.append(ds_list)

        ds = xr.combine_nested(  # noqa
            ds_list_list,
            concat_dim=["time", "flight_level"],
            compat="no_conflicts",
        )
        breakpoint()

    def _evaluate_shard(
        self, time: pd.Timestamp, flight_level: int, metrics: dict[str, Metric]
    ) -> xr.Dataset:
        """Run evaluation on a single shard."""
        forecast = self.dataloader.data(time, flight_level, self.extent)
        forecast = _apply_horizontal_buffers(forecast, self.buffers)

        ds_list = []
        for name, metric in metrics.items():
            data = metric.dataloader.data(time, flight_level, self.extent)
            statistics = metric.statistics(forecast, data)

            statistics = statistics.assign_coords(time=time, flight_level=flight_level)
            statistics = statistics.rename_vars({key: f"{name}.{key}" for key in statistics})
            ds_list.append(statistics)

        return xr.merge(ds_list, compat="identical")


def _apply_horizontal_buffers(forecast: xr.Dataset, buffers: list(int)) -> xr.Dataset:
    """Apply horizontal buffering to forecast PCR.

    This function mutates and returns the input Dataset.
    """

    pcr = forecast["pcr"]
    structure = np.array([[False, True, False], [True, True, True], [False, True, False]])

    pad = max(buffers)
    pad_left = pcr.values[-pad:, ...]
    pad_right = pcr.values[:pad, ...]
    padded = np.concat((pad_left, pcr.values, pad_right), axis=0)

    buffered = np.stack(
        [
            binary_dilation(padded, structure=structure, iterations=size) if size >= 1 else padded
            for size in buffers
        ],
        axis=-1,
    )
    buffered = buffered[pad:-pad, ...]

    new_dims = (*pcr.dims, "buffer")
    new_coords = pcr.coords.assign(buffer=buffers)
    out = xr.DataArray(buffered, dims=new_dims, coords=new_coords)

    forecast["pcr"] = out
    return forecast
