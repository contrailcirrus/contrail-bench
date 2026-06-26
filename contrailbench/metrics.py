"""Evaluation metrics."""

from abc import ABC, abstractmethod
from typing import override

import xarray as xr

from contrailbench.data import Dataloader


class Metric(ABC):
    """Base class for evaluation metrics.

    Implementing classes must implement the :meth:`data` and :meth:`statistics` methods
    """

    __slots__ = ("dataloader",)

    def __init__(self, dataloader: Dataloader) -> None:
        self.dataloader = dataloader

    @abstractmethod
    def statistics(self, forecast: xr.Dataset, data: xr.Dataset) -> xr.Dataset:
        """Compute required statistics from a shard of data."""


class FlightDistance(Metric):
    """Flight distance penalty metric."""

    @override
    def statistics(self, forecast: xr.Dataset, data: xr.Dataset) -> xr.Dataset:
        predicted = forecast["pcr"].sel(longitude=data["longitude"], latitude=data["latitude"])
        dist_pred = data["flight_distance"].where(predicted).sum("cell")
        dist_tot = data["flight_distance"].sum("cell")

        return xr.Dataset({"adsb_dist_in_forecast_pcr": dist_pred, "adsb_dist": dist_tot})


class HitRate(Metric):
    """Hit rate metric."""

    @override
    def statistics(self, forecast: xr.Dataset, data: xr.Dataset) -> xr.Dataset:
        predicted = forecast["pcr"].sel(longitude=data["longitude"], latitude=data["latitude"])
        area_pred = data["area"].where(predicted).sum("cell")
        area_tot = data["area"].sum("cell")

        return xr.Dataset(
            {"observed_pcr_area_in_forecast_pcr": area_pred, "observed_pcr_area": area_tot}
        )
