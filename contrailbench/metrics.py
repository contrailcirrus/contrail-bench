"""Evaluation metrics."""

from abc import ABC, abstractmethod
from typing import override

import xarray as xr

from contrailbench.data import Dataloader


class Metric(ABC):
    """Base class for evaluation metrics.

    This abstract base class assumes that statistics required by evaluation
    metrics can be calculated independently across flight levels and times,
    but does not otherwise restrict the format of forecast or evaluation
    data. Implementing classes should document the expected format of
    forecast and evaluation data in their implementation of the required
    :meth:`statistics` method.

    Parameters
    ----------
    dataloader : Dataloader
        Dataloader used to load evaluation data used by the metric.
    """

    __slots__ = ("dataloader",)

    #: Dataloader used to load evaluation data used by the metric
    dataloader: Dataloader

    def __init__(self, dataloader: Dataloader) -> None:
        self.dataloader = dataloader

    @abstractmethod
    def statistics(self, forecast: xr.Dataset, data: xr.Dataset) -> xr.Dataset:
        """Compute required statistics.

        Parameters
        ----------
        forecast : xr.Dataset
            Chunk of forecast data returned by a forecast dataloader for a single
            time, flight level and (optionally) geographic extent.

        data : xr.Dataset
            Chunk of evaluation data returned by ``self.dataloader`` for the same
            time, flight level and (optionally) geographic extent.

        Returns
        -------
        xr.Dataset
            Dataset containing statistics required by the implementing metric.
            No restriction is placed on the content of this dataset other than
            the requirement that neither ``"time"`` nor ``"flight_level"`` be
            present as coordinates.

        """
        # TODO: enforce this requirement and automatically attach time and flight_level to output?
        # TODO: validate inputs?


class FlightDistance(Metric):
    """Flight distance penalty metric."""

    @override
    def statistics(self, forecast: xr.Dataset, data: xr.Dataset) -> xr.Dataset:
        """Compute statistics for a PCR forecast flight distance penalty metric.

        Parameters
        ----------
        forecast : xr.Dataset
            PCR forecast. Must contain a boolean ``"pcr"`` variable with coordinates
            that include ``"longitude"`` and ``"latitude"`` indicating locations of
            forecast PCRs. Additional dimensions are permitted and allow evaluation
            to be vectorized across variants of the forecast.

        data : xr.Dataset
            Gridded flight distance dataset. Must contain ``"longitude"``, ``"latitude"``,
            and ``"flight_distance"`` as one-dimensional variables with dimension ``"cell"``,
            representing the total flight distance in individual grid cells. Values in
            ``"longitude"`` and ``"latitude"`` must be a subset of the longitude and latitude
            coordinates included in the PCR forecast. Grid cells with zero flight distance
            can be omitted from the dataset.

        Returns
        -------
        xr.Dataset
            Total flight distance (overall, ``"adsb_dist"``; and in forecast PCRs,
            ``"adsb_dist_in_forecast_pcr"``), summed over longitude and latitude.
        """
        predicted = forecast["pcr"].sel(longitude=data["longitude"], latitude=data["latitude"])
        dist_pred = data["flight_distance"].where(predicted).sum("cell")
        dist_tot = data["flight_distance"].sum("cell")

        return xr.Dataset({"adsb_dist_in_forecast_pcr": dist_pred, "adsb_dist": dist_tot})


class HitRate(Metric):
    """Compute statistics for a PCR forecast area-weighted hit rate metric.

    Parameters
    ----------
    forecast : xr.Dataset
        PCR forecast. Must contain a boolean ``"pcr"`` variable with coordinates
        that include ``"longitude"`` and ``"latitude"`` indicating locations of
        forecast PCRs. Additional dimensions are permitted and allow evaluation
        to be vectorized across variants of the forecast.

    data : xr.Dataset
        Gridded dataset of PCR observations. Must contain ``"longitude"``, ``"latitude"``,
        and ``"area"`` as one-dimensional variables with dimension ``"cell"``,
        representing the locations and areal extent of grid cells with observed evidence
        of a PCR. Values in ``"longitude"`` and ``"latitude"`` must be a subset of the
        longitude and latitude coordinates included in the PCR forecast.

    Returns
    -------
    xr.Dataset
        Observed PCR area (total, ``"observed_pcr_area"``; and in forecast PCRs,
        ``"observed_pcr_area_in_forecast_pcr"``), summed over longitude and latitude.
    """

    @override
    def statistics(self, forecast: xr.Dataset, data: xr.Dataset) -> xr.Dataset:
        predicted = forecast["pcr"].sel(longitude=data["longitude"], latitude=data["latitude"])
        area_pred = data["area"].where(predicted).sum("cell")
        area_tot = data["area"].sum("cell")

        return xr.Dataset(
            {"observed_pcr_area_in_forecast_pcr": area_pred, "observed_pcr_area": area_tot}
        )
