"""Data loading."""

from abc import ABC, abstractmethod
from typing import override

import numpy as np
import pandas as pd
import xarray as xr

from contrailbench.constants import radius_earth
from contrailbench.types import DatetimeLike


class Dataloader(ABC):
    """Base class for data loaders.

    Dataloaders are used by other classes to request data for a specific
    time, flight level, and (optionally) geographic extent.

    Implementing classes must implement the :meth:`data` method. This method
    is required to return data as an xarray Dataset but places no restrictions
    on the structure or content of the dataset.
    """

    @abstractmethod
    def data(
        self,
        time: DatetimeLike,
        flight_level: int,
        extent: tuple[float, float, float, float] | None,
    ) -> xr.Dataset:
        """Load a single shard of data.

        Parameters
        ----------
        time : DatetimeLike
            Requested time.

        flight_level : int
            Requested flight level.

        extent : tuple[float, float, float, float] | None
            Requested geographic area (optional). Elements represent,
            in order, the westward-most latitude, eastward-most latitude,
            southward-most longitude, and northward-most longitude of
            a bounding box.
        """


class ADSBDataloader(Dataloader):
    """Data loader for public ADSB evaluation dataset."""

    @override
    def data(
        self,
        time: DatetimeLike,
        flight_level: int,
        extent: tuple[float, float, float, float] | None,
    ) -> xr.Dataset:

        ts = int(pd.to_datetime(time).timestamp())
        df = pd.read_parquet(f"{self.root(time)}/{ts}_{flight_level}.pq")

        if extent is not None:
            lon_min, lon_max, lat_min, lat_max = extent
            df = df.loc[
                df["longitude"].between(lon_min, lon_max) & df["latitude"].between(lat_min, lat_max)
            ]

        lon = xr.DataArray(df["longitude"], dims="cell")
        lat = xr.DataArray(df["latitude"], dims="cell")
        dist = xr.DataArray(df["flight_distance"], dims="cell")

        return xr.Dataset(data_vars={"longitude": lon, "latitude": lat, "flight_distance": dist})

    @staticmethod
    def root(time: DatetimeLike) -> str:
        """Return GCS root of evaluation dataset.

        Parameters
        ----------
        time : DatetimeLike
            Requested time.

        Returns
        -------
        str
            GCS root.

        """
        year = pd.to_datetime(time).year
        if year == 2024:
            return "gs://contrailbench-public-data/v1/adsb"
        if year == 2025:
            return "gs://contrails-301217-contrail-bench/v2/evaluation-datasets/adsb"
        raise ValueError(f"ADSB evaluation data not available for {year}.")


class IAGOSDataloader(Dataloader):
    """Data loader for public IAGOS evaluation dataset."""

    @override
    def data(
        self,
        time: DatetimeLike,
        flight_level: int,
        extent: tuple[float, float, float, float] | None,
    ) -> xr.Dataset:

        ts = int(pd.to_datetime(time).timestamp())
        df = pd.read_parquet(f"{self.root(time)}/{ts}_{flight_level}.pq")

        if extent is not None:
            lon_min, lon_max, lat_min, lat_max = extent
            df = df.loc[
                df["longitude"].between(lon_min, lon_max) & df["latitude"].between(lat_min, lat_max)
            ]

        df = df.loc[df["pcr_distance"] > 0]
        lon = xr.DataArray(df["longitude"], dims="cell")
        lat = xr.DataArray(df["latitude"], dims="cell")
        area = (radius_earth * np.deg2rad(0.25)) ** 2 * np.cos(np.deg2rad(lat))

        return xr.Dataset(data_vars={"longitude": lon, "latitude": lat, "area": area})

    @staticmethod
    def root(time: DatetimeLike) -> str:
        """Return GCS root of evaluation dataset.

        Parameters
        ----------
        time : DatetimeLike
            Requested time.

        Returns
        -------
        str
            GCS root.

        """
        year = pd.to_datetime(time).year
        if year == 2024:
            return "gs://contrailbench-public-data/v1/iagos"
        if year == 2025:
            return "gs://contrails-301217-contrail-bench/v2/evaluation-datasets/iagos"
        raise ValueError(f"IAGOS evaluation data not available for {year}.")


class GRUANDataloader(Dataloader):
    """Data loader for public GRUAN evaluation dataset."""

    @override
    def data(
        self,
        time: DatetimeLike,
        flight_level: int,
        extent: tuple[float, float, float, float] | None,
    ) -> xr.Dataset:

        ts = int(pd.to_datetime(time).timestamp())
        df = pd.read_parquet(f"{self.root(time)}/{ts}_{flight_level}.pq")

        if extent is not None:
            lon_min, lon_max, lat_min, lat_max = extent
            df = df.loc[
                df["longitude"].between(lon_min, lon_max) & df["latitude"].between(lat_min, lat_max)
            ]

        df = df.loc[df["pcr_count"] > 0]
        lon = xr.DataArray(df["longitude"], dims="cell")
        lat = xr.DataArray(df["latitude"], dims="cell")
        area = (radius_earth * np.deg2rad(0.25)) ** 2 * np.cos(np.deg2rad(lat))

        return xr.Dataset(data_vars={"longitude": lon, "latitude": lat, "area": area})

    @staticmethod
    def root(time: DatetimeLike) -> str:
        """Return GCS root of evaluation dataset.

        Parameters
        ----------
        time : DatetimeLike
            Requested time.

        Returns
        -------
        str
            GCS root.

        """
        year = pd.to_datetime(time).year
        if year == 2024:
            return "gs://contrailbench-public-data/v1/gruan"
        if year == 2025:
            return "gs://contrails-301217-contrail-bench/v2/evaluation-datasets/gruan"
        raise ValueError(f"GRUAN evaluation data not available for {year}.")


class ContrailWatchDataloader(Dataloader):
    """Data loader for public ContrailWatch evaluation dataset."""

    @override
    def data(
        self,
        time: DatetimeLike,
        flight_level: int,
        extent: tuple[float, float, float, float] | None = None,
    ) -> xr.Dataset:

        ts = int(pd.to_datetime(time).timestamp())
        df = pd.read_parquet(f"{self.root(time)}/{ts}_{flight_level}.pq")

        if extent is not None:
            lon_min, lon_max, lat_min, lat_max = extent
            df = df.loc[
                df["longitude"].between(lon_min, lon_max) & df["latitude"].between(lat_min, lat_max)
            ]

        df = df.loc[df["attributed_flight_distance"] > 0]
        lon = xr.DataArray(df["longitude"], dims="cell")
        lat = xr.DataArray(df["latitude"], dims="cell")
        area = (radius_earth * np.deg2rad(0.25)) ** 2 * np.cos(np.deg2rad(lat))

        return xr.Dataset(data_vars={"longitude": lon, "latitude": lat, "area": area})

    @staticmethod
    def root(time: DatetimeLike) -> str:
        """Return GCS root of evaluation dataset.

        Parameters
        ----------
        time : DatetimeLike
            Requested time.

        Returns
        -------
        str
            GCS root.

        """
        year = pd.to_datetime(time).year
        if year == 2024:
            return "gs://contrailbench-public-data/v1/contrailwatch"
        if year == 2025:
            return "gs://contrails-301217-contrail-bench/v2/evaluation-datasets/contrailwatch"
        raise ValueError(f"ContrailWatch evaluation data not available for {year}.")


class ImperialAttributionsDataloader(Dataloader):
    """Data loader for public Imperial College attributions evaluation dataset."""

    @override
    def data(
        self,
        time: DatetimeLike,
        flight_level: int,
        extent: tuple[float, float, float, float] | None = None,
    ) -> xr.Dataset:

        ts = int(pd.to_datetime(time).timestamp())
        df = pd.read_parquet(f"{self.root(time)}/{ts}_{flight_level}.pq")

        if extent is not None:
            lon_min, lon_max, lat_min, lat_max = extent
            df = df.loc[
                df["longitude"].between(lon_min, lon_max) & df["latitude"].between(lat_min, lat_max)
            ]

        df = df.loc[df["attributed_flight_distance"] > 0]
        lon = xr.DataArray(df["longitude"], dims="cell")
        lat = xr.DataArray(df["latitude"], dims="cell")
        area = (radius_earth * np.deg2rad(0.25)) ** 2 * np.cos(np.deg2rad(lat))

        return xr.Dataset(data_vars={"longitude": lon, "latitude": lat, "area": area})

    @staticmethod
    def root(time: DatetimeLike) -> str:
        """Return GCS root of evaluation dataset.

        Parameters
        ----------
        time : DatetimeLike
            Requested time.

        Returns
        -------
        str
            GCS root.

        """
        year = pd.to_datetime(time).year
        if year == 2025:
            return (
                "gs://contrails-301217-contrail-bench/v2/evaluation-datasets/imperial-attributions"
            )
        raise ValueError(f"Imperial attributions evaluation data not available for {year}.")
