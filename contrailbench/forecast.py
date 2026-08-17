"""Forecast evaluation."""

import hashlib
import itertools
import json
from collections.abc import Collection, Iterable, Iterator

import apache_beam as beam
import pandas as pd
import xarray as xr
from apache_beam.options.pipeline_options import PipelineOptions

from contrailbench import io, time_utils
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
        return _concatenate_fl_time(ds_list, self.times, self.flight_levels)

    def evaluate_beam(
        self,
        outputs: str,
        intermediates: str,
        options: PipelineOptions,
        *,
        resume: bool = True,
        **metrics: Metric,
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

        resume : bool, optional
            Default ``True``. Skip days whose output file already exists (see
            :func:`pending_output_days`), and skip individual shards whose
            cached intermediate matches this run's configuration (see
            :class:`_Evaluate`). Pass ``False`` to force a full recompute.

        **metrics : Metric
            Metrics included in evaluation.
        """
        times = pending_output_days(self.times, outputs, resume=resume)
        pcoll = itertools.product(times, self.flight_levels)

        with beam.Pipeline(options=options) as pipeline:
            (
                pipeline
                | "Create PCollection" >> beam.Create(pcoll)
                | "Compute metrics" >> beam.ParDo(_Evaluate(self, metrics, intermediates, resume=resume))
                | "Group results" >> beam.GroupByKey()
                | "Save to GCS" >> beam.ParDo(_Concatenate(outputs, self))
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

    def __init__(
        self, forecast: Forecast, metrics: dict[str, Metric], intermediates: str, resume: bool = True
    ) -> None:
        self.forecast = forecast
        self.metrics = metrics
        self.intermediates = intermediates
        self.resume = resume
        self.signature = _shard_signature(forecast.dataloader, forecast.extent, metrics)

    def process(self, element: tuple[pd.Timestamp, int]) -> Iterator[tuple[str, str]]:
        time, flight_level = element
        ts = time_utils.to_utc_timestamp(time)
        sink = f"{self.intermediates}/{ts}_{flight_level}.nc"
        key = time.strftime("%Y%m%d")

        if self.resume and io.exists(sink) and self._cached_signature_matches(sink):
            yield key, sink
            return

        try:
            ds = self.forecast._evaluate_shard(time, flight_level, self.metrics)
        except FileNotFoundError:
            # Missing forecast or observation data for this (time, flight_level)
            # is routine, not exceptional -- partial source coverage is how
            # these arms actually run. Skip rather than crash the pipeline;
            # `_Concatenate` fills the gap with NaN and records it in `coverage`.
            return

        ds.attrs["contrailbench_signature"] = self.signature
        io.write(sink, ds.to_netcdf())
        yield key, sink

    def _cached_signature_matches(self, sink: str) -> bool:
        """Whether an existing shard was produced by this exact configuration.

        Existence alone isn't enough once one generic entry point can serve
        runs with different metrics/extent/dataloaders against the same
        `intermediates` path -- without this check, a stale shard from an
        earlier, differently-configured run would be silently reused instead
        of recomputed. Any read failure (corrupted/partial file) is treated as
        a miss, since recomputing is always safe and existence-based resume
        checks must never trust a file they haven't verified.
        """
        try:
            existing = io.load_dataset(sink)
        except Exception:
            return False
        return existing.attrs.get("contrailbench_signature") == self.signature


def _shard_signature(dataloader: Dataloader, extent: tuple[float, float, float, float] | None,
                      metrics: dict[str, Metric]) -> str:
    """Stable fingerprint of everything that affects a shard's content.

    Used to validate a cached intermediate before trusting it as a resume
    target -- two runs pointed at the same `intermediates` path with a
    different metric set, extent, or dataloader must not silently share
    shards, even though both would satisfy a bare existence check.
    """
    payload = {
        "dataloader": repr(dataloader),
        "extent": list(extent) if extent is not None else None,
        "metrics": {
            name: {"class": type(metric).__name__, "dataloader": repr(metric.dataloader)}
            for name, metric in metrics.items()
        },
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


def pending_output_days(
    times: Collection[pd.Timestamp], outputs: str, resume: bool = True
) -> list[pd.Timestamp]:
    """Restrict `times` to those whose day-output file doesn't exist yet.

    `_Concatenate` writes `{outputs}/{day}.nc` from only the shards seen in
    the current run, so resuming at anything finer than whole-day granularity
    would silently truncate a day's file to just the newly (re)computed
    hours. Redoing a whole day is the safe unit of resumability here --
    bounded to at most 24h x len(flight_levels) of rework, further reduced by
    `_Evaluate`'s own per-shard skip for hours already cached from a prior run
    of this same configuration.

    Parameters
    ----------
    times : Collection[pd.Timestamp]
        Candidate times, in order.

    outputs : str
        Output directory a completed day's file would be written to.

    resume : bool, optional
        Default ``True``. ``False`` returns ``times`` unchanged (always redo
        everything).

    Returns
    -------
    list[pd.Timestamp]
        `times`, excluding every time whose day already has a complete output
        file.
    """
    if not resume:
        return list(times)

    times = list(times)
    days_needed = {
        pd.Timestamp(t).strftime("%Y%m%d")
        for t in times
        if not io.exists(f"{outputs}/{pd.Timestamp(t):%Y%m%d}.nc")
    }
    return [t for t in times if pd.Timestamp(t).strftime("%Y%m%d") in days_needed]


class _Concatenate(beam.DoFn):
    """Concatenate and save shards."""

    def __init__(self, output: str, forecast: Forecast) -> None:
        self.output = output
        self.forecast = forecast

    def process(self, element: tuple[str, Iterable[str]]) -> None:
        key, paths = element
        ds_list = [io.load_dataset(path) for path in paths]
        day_times = [t for t in self.forecast.times if t.strftime("%Y%m%d") == key]
        ds = _concatenate_fl_time(ds_list, day_times, self.forecast.flight_levels)
        sink = f"{self.output}/{key}.nc"
        io.write(sink, ds.to_netcdf())


def _concatenate_fl_time(
    ds_list: list[xr.Dataset],
    expected_times: Collection[pd.Timestamp],
    expected_flight_levels: Collection[int],
) -> xr.Dataset:
    """Concatenate shards by flight level and time, tolerating missing shards.

    A (time, flight_level) combination can be legitimately absent -- partial
    forecast-source coverage, an observation gap -- and that must not crash
    the whole day's concatenation, as the previous nested-list-based
    implementation did on any ragged group. Missing combinations are
    reindexed to NaN and recorded in a `coverage` variable (1 = shard present,
    0 = filled), so partial coverage is visible in the output instead of
    silently absent.
    """
    expected_times = sorted(pd.Timestamp(t) for t in expected_times)
    expected_flight_levels = sorted(expected_flight_levels)

    if not ds_list:
        raise ValueError("cannot concatenate an empty shard list -- nothing to combine")

    present = {(pd.Timestamp(ds["time"].item()), int(ds["flight_level"].item())) for ds in ds_list}
    expanded = [ds.expand_dims(["time", "flight_level"]) for ds in ds_list]
    combined = xr.combine_by_coords(expanded, join="outer")
    ds = combined.reindex(time=expected_times, flight_level=expected_flight_levels)

    coverage = xr.DataArray(
        [[1 if (t, fl) in present else 0 for fl in expected_flight_levels] for t in expected_times],
        dims=("time", "flight_level"),
        coords={"time": expected_times, "flight_level": expected_flight_levels},
    )
    ds = ds.assign(coverage=coverage)

    # Let xarray determine appropriate units on write
    ds["time"].encoding.pop("units", None)

    return ds
