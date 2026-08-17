"""Data loading."""

from abc import ABC, abstractmethod
from collections.abc import Collection
from typing import override

import numpy as np
import pandas as pd
import xarray as xr

from contrailbench import io, pcr, time_utils
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

    #: Default path to GCS directory
    DEFAULT_PATH: str = "gs://contrailbench-public-data/v1/adsb"

    def __init__(self, path: str = DEFAULT_PATH) -> None:
        self.path = path

    @override
    def data(
        self,
        time: DatetimeLike,
        flight_level: int,
        extent: tuple[float, float, float, float] | None,
    ) -> xr.Dataset:

        ts = time_utils.to_utc_timestamp(pd.to_datetime(time))
        df = pd.read_parquet(f"{self.path}/{ts}_{flight_level}.pq")

        if extent is not None:
            lon_min, lon_max, lat_min, lat_max = extent
            df = df.loc[
                df["longitude"].between(lon_min, lon_max) & df["latitude"].between(lat_min, lat_max)
            ]

        lon = xr.DataArray(df["longitude"], dims="cell")
        lat = xr.DataArray(df["latitude"], dims="cell")
        dist = xr.DataArray(df["flight_distance"], dims="cell")

        return xr.Dataset(data_vars={"longitude": lon, "latitude": lat, "flight_distance": dist})


class IAGOSDataloader(Dataloader):
    """Data loader for public IAGOS evaluation dataset."""

    #: Default path to GCS directory
    DEFAULT_PATH: str = "gs://contrailbench-public-data/v1/iagos"

    def __init__(self, path: str = DEFAULT_PATH) -> None:
        self.path = path

    @override
    def data(
        self,
        time: DatetimeLike,
        flight_level: int,
        extent: tuple[float, float, float, float] | None,
    ) -> xr.Dataset:

        ts = time_utils.to_utc_timestamp(pd.to_datetime(time))
        df = pd.read_parquet(f"{self.path}/{ts}_{flight_level}.pq")

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


class GRUANDataloader(Dataloader):
    """Data loader for public GRUAN evaluation dataset."""

    #: Default path to GCS directory
    DEFAULT_PATH: str = "gs://contrailbench-public-data/v1/gruan"

    def __init__(self, path: str = DEFAULT_PATH) -> None:
        self.path = path

    @override
    def data(
        self,
        time: DatetimeLike,
        flight_level: int,
        extent: tuple[float, float, float, float] | None,
    ) -> xr.Dataset:

        ts = time_utils.to_utc_timestamp(pd.to_datetime(time))
        df = pd.read_parquet(f"{self.path}/{ts}_{flight_level}.pq")

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


class ContrailWatchDataloader(Dataloader):
    """Data loader for public ContrailWatch evaluation dataset."""

    #: Default path to GCS directory
    DEFAULT_PATH: str = "gs://contrailbench-public-data/v1/contrailwatch"

    def __init__(self, path: str = DEFAULT_PATH) -> None:
        self.path = path

    @override
    def data(
        self,
        time: DatetimeLike,
        flight_level: int,
        extent: tuple[float, float, float, float] | None = None,
    ) -> xr.Dataset:

        ts = time_utils.to_utc_timestamp(pd.to_datetime(time))
        df = pd.read_parquet(f"{self.path}/{ts}_{flight_level}.pq")

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


class PCRStoreDataloader(Dataloader):
    """Data loader for a preprocessed PCR forecast store.

    Reads per-(time, flight_level) netCDF files produced by a forecast
    preprocessing step -- continuous ``rhi`` and binary ``sac`` fields on the
    benchmark grid -- and sweeps a set of RHi thresholds to produce a boolean
    ``pcr`` field with an added ``rhi_threshold`` dimension. Vectorizing the
    sweep this way lets a single :meth:`Forecast.evaluate` call score every
    threshold at once, instead of one preprocessing pass per threshold.

    Parameters
    ----------
    path : str
        Location of the forecast store. Must be on a filesystem supported by
        fsspec.

    rhi_thresholds : Collection[float], optional
        RHi thresholds to sweep. Defaults to :data:`contrailbench.pcr.RHI_THRESHOLDS`.
    """

    def __init__(self, path: str, rhi_thresholds: Collection[float] = pcr.RHI_THRESHOLDS) -> None:
        self.path = path
        self.rhi_thresholds = list(rhi_thresholds)

    @override
    def data(
        self,
        time: DatetimeLike,
        flight_level: int,
        extent: tuple[float, float, float, float] | None,
    ) -> xr.Dataset:
        ts = time_utils.to_utc_timestamp(pd.to_datetime(time))
        ds = io.load_dataset(f"{self.path}/{ts}_{flight_level}.nc")

        # The stored file carries size-1 `time`/`level` dims (one fetch, one
        # flight level); a Dataloader shard is scoped to exactly one of each,
        # so both collapse to plain attributes of this shard rather than dims.
        ds = ds.squeeze(("time", "level"), drop=True)

        if extent is not None:
            lon_min, lon_max, lat_min, lat_max = extent
            ds = ds.sel(longitude=slice(lon_min, lon_max), latitude=slice(lat_min, lat_max))

        thresholds = xr.DataArray(
            self.rhi_thresholds, dims="rhi_threshold", coords={"rhi_threshold": self.rhi_thresholds}
        )
        pcr_swept = (ds["rhi"] > thresholds) & (ds["sac"] > 0)

        return ds.drop_vars("pcr").assign(pcr=pcr_swept)
