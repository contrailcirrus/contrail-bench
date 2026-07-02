"""Forecast evaluation."""

import datetime
import itertools
from collections.abc import Iterable, Iterator

import apache_beam as beam
import pandas as pd
import xarray as xr
from apache_beam.options.pipeline_options import PipelineOptions

from contrailbench import io
from contrailbench.data import Dataloader
from contrailbench.metrics import Metric
from contrailbench.types import DatetimeLike


class Forecast:
    """Load and evaluate a forecast.

    Parameters
    ----------
    dataloader: Dataloader
        Dataloader used to provide forecast data.

    times : Iterable[DatetimeLike]
        Times included in evaluation.

    flight_levels: Iterable[int]
        Flight levels included in evaluation.

    extent : tuple[float, float, float, float]
        Geographic area included in evaluation (optional). Elements represent,
        in order, the westward-most latitude, eastward-most latitude,
        southward-most longitude, and northward-most longitude of
        a bounding box.
    """

    __slots__ = (
        "dataloader",
        "extent",
        "flight_levels",
        "times",
    )

    #: Dataloader used to provide forecast data
    dataloader: Dataloader

    #: Times included in evaluation
    times: list[pd.Timestamp]

    #: Flight levels included in evaluation
    flight_levels: list[int]

    #: Longitude-latitude bounded box included in evaluation
    extent: tuple[float, float, float, float] | None

    def __init__(
        self,
        dataloader: Dataloader,
        times: Iterable[DatetimeLike],
        flight_levels: Iterable[int],
        extent: tuple[float, float, float, float] | None,
    ) -> None:
        self.dataloader = dataloader
        self.times = [pd.to_datetime(t) for t in times]
        self.flight_levels = list(flight_levels)
        self.extent = extent

    def evaluate(self, **metrics: Metric) -> xr.Dataset:
        """Evaluate forecast against a set of metrics.

        This function runs locally without any parallelization
        and is best suited for small-scale tests.

        Parameters
        ----------
        **metrics : Metric
            Metrics included in evaluation.

        Returns
        -------
        xr.Dataset
            Statistics computed by metrics used for evaluation.
            Names of variables representing computed statistics
            are prefixed by the name of the keyword argument
            assigned to the associated metric.
        """
        pcoll = itertools.product(self.times, self.flight_levels)
        ds_list = []
        for time, flight_level in pcoll:
            ds = self._evaluate_shard(time, flight_level, metrics)
            ds_list.append(ds)
        return _concatenate_fl_time(ds_list)

    def evaluate_beam(
        self, outputs: str, intermediates: str, options: PipelineOptions, **metrics: Metric
    ) -> None:
        """Evaluate forecast against a set of metrics.

        This function runs using an Apache Beam pipeline and is designed for
        parallel processing of large-scale datasets.

        Parameters
        ----------
        outputs : str
            Location where final pipeline outputs are saved. Any fsspec-supported filesystem can
            be used. Outputs are grouped by date and saved to netCDF files at
            ``<outputs>/YYYYmmmdd.nc``. See :math:`evaluate` for details about the format of
            output netCDFs.

        intermediates : str
            Location where intermediate pipeline outputs are cached. Any fsspec-supported filesystem
            can be used. Intermediate outputs are not intended to be accessed directly, but deletion
            after pipelines finish is the responsibility of the user.

        options : PipelineOptions
            Beam pipeline configuration options.

        **metrics : Metric
            Metrics included in evaluation.
        """
        pcoll = itertools.product(self.times, self.flight_levels)

        with beam.Pipeline(options=options) as pipeline:
            (
                pipeline
                | "Create PCollection" >> beam.Create(pcoll)
                | "Compute metrics" >> beam.ParDo(_Evaluate(self, metrics, intermediates))
                | "Group results" >> beam.GroupByKey()
                | "Save to GCS" >> beam.ParDo(_Concatenate(outputs))
            )

    def _evaluate_shard(
        self, time: pd.Timestamp, flight_level: int, metrics: dict[str, Metric]
    ) -> xr.Dataset:
        """Run evaluation on a single shard."""
        forecast = self.dataloader.data(time, flight_level, self.extent)
        ds_list = []
        for name, metric in metrics.items():
            data = metric.dataloader.data(time, flight_level, self.extent)
            statistics = metric.statistics(forecast, data)

            statistics = statistics.assign_coords(time=time, flight_level=flight_level)
            statistics = statistics.rename_vars({key: f"{name}.{key}" for key in statistics})
            ds_list.append(statistics)

        return xr.merge(ds_list, compat="identical")


class _Evaluate(beam.DoFn):
    """Run evaluation on a single shard of data."""

    def __init__(self, forecast: Forecast, metrics: dict[str, Metric], intermediates: str) -> None:
        self.forecast = forecast
        self.metrics = metrics
        self.intermediates = intermediates

    def process(self, element: tuple[pd.Timestamp, int]) -> Iterator[tuple[str, str]]:
        time, flight_level = element
        ds = self.forecast._evaluate_shard(time, flight_level, self.metrics)

        ts = int(time.timestamp())
        sink = f"{self.intermediates}/{ts}_{flight_level}.nc"
        io.write(sink, ds.to_netcdf())

        key = time.strftime("%Y%m%d")
        yield key, sink


class _Concatenate(beam.DoFn):
    """Concatenate and save shards."""

    def __init__(self, output: str) -> None:
        self.output = output

    def process(self, element: tuple[str, Iterable[str]]) -> None:
        key, paths = element
        ds_list = [io.load_dataset(path) for path in paths]
        ds = _concatenate_fl_time(ds_list)
        sink = f"{self.output}/{key}.nc"
        io.write(sink, ds.to_netcdf())


def _concatenate_fl_time(ds_list: list[xr.Dataset]) -> xr.Dataset:
    """Concatenate list of Datasets by flight level and time."""

    # itertools.groupby requires an iterable sorted by grouping key
    ds_list = sorted(ds_list, key=_time_key)
    nested = [sorted(g, key=_fl_key) for _, g in itertools.groupby(ds_list, key=_time_key)]

    ds = xr.combine_nested(
        nested,
        concat_dim=["time", "flight_level"],
        compat="no_conflicts",
    )

    # Let xarray determine appropriate units on write
    ds["time"].encoding.pop("units", None)

    return ds


def _time_key(ds: xr.Dataset) -> datetime.datetime:
    """Extract time from dataset for use as key."""
    return ds["time"].item()


def _fl_key(ds: xr.Dataset) -> int:
    """Extract flight level from dataset for use as key."""
    return ds["flight_level"].item()
