"""Preprocess ECMWF IFS 50r1 forecasts into persistent contrail region (PCR) fields.

The input is one zarr store per 00Z initialisation day, holding hourly leads on fixed pressure
levels. A Beam element is one initialisation day: it opens that day's store once, reads only the
temperature and humidity chunks and pressure levels needed, and writes a netCDF shard for each
valid hour and flight level (leads 0 to 23 of that init). 00Z is the only initialisation staged,
so there is no ``--run``.

Humidity gets the same correction as ERA5 and HRES (see ``src/pcr.py``).

A day whose store is missing stops the job unless ``--allow-incomplete-window`` is given, in which
case those days are skipped and the missing valid times stay unwritten.
"""

import datetime
import functools
import logging
import os
import re
from collections.abc import Sequence

import apache_beam as beam
import numpy as np
import pandas as pd
import xarray as xr
from apache_beam.options.pipeline_options import PipelineOptions

from src import cli, config, forecast_runs, pcr, shard_job, storage
from src.grid import Extent

LOGGER = logging.getLogger(__name__)

# Pipeline parameters

#: Output directory name under ``forecasts/``, and the one initialisation it holds
SOURCE = "ecmwf-ifs-hres-50r1"
RUN = "00Z"

#: One zarr store per initialisation day; ``{day}`` is ``YYYYmmdd``
TEMPLATE = "gs://contrails-301217-forecast-research-archive/ifs50r1/00h/{day}.zarr"

#: Attributes the stores must carry, as a check that they are what we think they are
EXPECTED_ATTRS = {"provider": "ECMWF", "dataset": "HRES", "product": "forecast"}


def available_days(template: str, days: Sequence[datetime.date]) -> list[datetime.date]:
    """The days of ``days`` that have a store, found from one listing, not one check per day."""
    if not storage.is_gcs_path(template):
        return [d for d in days if os.path.isdir(template.format(day=f"{d:%Y%m%d}"))]
    bucket, key = storage.parse_gcs_path(template)
    prefix, _, suffix = key.partition("{day}")
    pages = storage.gcs_client().list_blobs(bucket, prefix=prefix, delimiter="/")
    for _ in pages.pages:  # prefixes are filled in as the pages are read
        pass
    found = {
        m.group(1)
        for p in pages.prefixes
        if (m := re.fullmatch(re.escape(prefix) + r"(\d{8})" + re.escape(suffix) + "/", p))
    }
    return [d for d in days if f"{d:%Y%m%d}" in found]


def coordinate_slice(values: np.ndarray, low: float, high: float) -> slice:
    """A label slice from ``low`` to ``high`` that suits an ascending or descending coordinate."""
    if values[0] == values[-1]:
        msg = "coordinate has fewer than two distinct points"
        raise ValueError(msg)
    return slice(low, high) if values[0] < values[-1] else slice(high, low)


def netcdf_safe_attributes(ds: xr.Dataset) -> None:
    """Convert boolean attributes to integers, in place: netCDF cannot store booleans."""
    for attrs in [ds.attrs, *(variable.attrs for variable in ds.variables.values())]:
        for key, value in list(attrs.items()):
            if isinstance(value, bool | np.bool_):
                attrs[key] = np.int8(value)
            elif isinstance(value, np.ndarray) and value.dtype.kind == "b":
                attrs[key] = value.astype(np.int8)


def compute_forecast_day(
    day: datetime.date,
    times: Sequence[datetime.datetime],
    flight_levels: Sequence[int],
    extent: Extent,
    template: str,
) -> xr.Dataset:
    """PCR fields for ``times`` of one initialisation day and every flight level, from one store.

    Returns
    -------
    xr.Dataset
        ``rhi``, ``sac`` and ``pcr`` with ``time`` and ``flight_level`` dimensions

    Raises
    ------
    ValueError
        If the store is not what is expected: wrong metadata, missing variables or coordinates,
        levels that do not bracket the flight levels, or missing valid times.
    """
    uri = template.format(day=f"{day:%Y%m%d}")
    ds = xr.open_zarr(
        uri,
        chunks=None,
        consolidated=True,
        zarr_format=3,
        storage_options={"skip_instance_cache": True} if storage.is_gcs_path(uri) else None,
    )
    bad = {k: (ds.attrs.get(k), v) for k, v in EXPECTED_ATTRS.items() if ds.attrs.get(k) != v}
    if bad:
        ds.close()
        msg = f"unexpected IFS50r1 store metadata at {uri}: {bad}"
        raise ValueError(msg)
    absent = {"air_temperature", "specific_humidity"} - set(ds.data_vars)
    if absent:
        ds.close()
        msg = f"{uri} is missing PCR input fields: {sorted(absent)}"
        raise ValueError(msg)
    if "level" not in ds.coords or "time" not in ds.coords:
        ds.close()
        msg = f"{uri} must have pressure-level and time coordinates"
        raise ValueError(msg)

    pressures = [pcr.target_pressure_hpa(fl) for fl in flight_levels]
    levels = np.asarray(ds["level"].values, dtype=np.float64)
    if np.all(np.diff(levels) < 0):
        ds = ds.sortby("level")
        levels = levels[::-1]
    selected_levels = pcr.bracketing_level_indices(levels, pressures)

    wanted = pd.DatetimeIndex(times)
    absent_times = wanted.difference(pd.DatetimeIndex(pd.to_datetime(ds["time"].values)))
    if not absent_times.empty:
        ds.close()
        msg = f"{uri} lacks requested valid times: {absent_times.tolist()}"
        raise ValueError(msg)

    lon_min, lon_max, lat_min, lat_max = extent
    selected = ds[["air_temperature", "specific_humidity"]].isel(level=selected_levels)
    selected["air_temperature"].attrs.update({"standard_name": "air_temperature", "units": "K"})
    selected["specific_humidity"].attrs.update(
        {"standard_name": "specific_humidity", "units": "kg kg-1"}
    )
    selected = selected.sel(
        longitude=coordinate_slice(ds["longitude"].values, lon_min, lon_max),
        latitude=coordinate_slice(ds["latitude"].values, lat_min, lat_max),
        time=wanted,
    ).load()
    attrs = dict(ds.attrs)
    ds.close()

    # The regridder expects ascending coordinates
    selected = selected.sortby("longitude").sortby("latitude")
    selected = pcr.interpolate_to_pressures(selected, pressures)
    batched = pcr.regrid_and_derive(selected, flight_levels, extent, humidity_correction="pressure")
    # Provenance: the store's own metadata, and which store and levels this came from
    batched.attrs.update(attrs)
    batched.attrs.update(
        forecast_model_cycle="IFS50r1",
        input_store=uri,
        input_pressure_levels_hpa=",".join(map(str, levels[selected_levels].tolist())),
    )
    return batched


def preprocess_ifs50r1(
    day: datetime.date,
    times: Sequence[datetime.datetime],
    flight_levels: Sequence[int],
    *,
    extent: Extent,
    template: str,
) -> dict[datetime.datetime, dict[int, xr.Dataset]]:
    """Shards for ``times`` of one init day, as ``{valid time: {flight level: shard}}``."""
    batched = compute_forecast_day(day, times, flight_levels, extent, template)
    init = datetime.datetime.combine(day, datetime.time())
    shards = {}
    for time in times:
        one_time = batched.sel(time=[pd.Timestamp(time)])
        shards[time] = {}
        for fl, shard in pcr.split_flight_levels(one_time, flight_levels).items():
            out = forecast_runs.record_init(shard, init, time)
            netcdf_safe_attributes(out)  # a store's metadata can hold booleans
            shards[time][fl] = out
    return shards


class _WriteDay(beam.DoFn):
    """Beam wrapper: produce one initialisation day's shards and write them."""

    def __init__(self, job: shard_job.ShardJob, produce: functools.partial) -> None:
        self.job = job
        self.produce = produce

    def process(self, element: tuple[datetime.date, list[datetime.datetime]]) -> None:
        day, times = element
        with shard_job.skipping_failures(self.job, day):
            for time, shards in self.produce(day, times, self.job.flight_levels).items():
                shard_job.write_shards(self.job, time, shards)


def run_days(
    job: shard_job.ShardJob,
    produce: functools.partial,
    times: Sequence[datetime.datetime],
    options: PipelineOptions,
    *,
    force: bool = False,
    template: str = TEMPLATE,
    allow_incomplete: bool = False,
) -> tuple[int, int]:
    """Run ``job`` over ``times`` one initialisation day at a time, skipping times already done.

    Returns
    -------
    tuple[int, int]
        ``(already_done, run_now)``. Times of days with no store are not counted as run.
    """
    todo = list(times) if force else shard_job.pending(job, times)
    by_day: dict[datetime.date, list[datetime.datetime]] = {}
    for time in todo:
        by_day.setdefault(time.date(), []).append(time)

    have = set(available_days(template, list(by_day)))
    missing = sorted(set(by_day) - have)
    if missing:
        message = f"{len(missing)} day(s) have no store: {[d.isoformat() for d in missing]}"
        if not allow_incomplete:
            msg = f"{message}. Pass --allow-incomplete-window to skip them."
            raise SystemExit(msg)
        LOGGER.warning("%s; skipped", message)

    elements = [(day, by_day[day]) for day in sorted(have)]
    if elements:
        with beam.Pipeline(options=options) as pipeline:
            (
                pipeline
                | "Create days" >> beam.Create(elements)
                | "Write shards" >> beam.ParDo(_WriteDay(job, produce))
            )
    return len(times) - len(todo), sum(len(t) for _, t in elements)


def main() -> None:
    """Program entrypoint."""
    parser = cli.shard_parser(__doc__)
    parser.add_argument("--allow-incomplete-window", action="store_true")
    parser.add_argument("--harness-threads", type=int, help="threads per Dataflow worker")
    args = parser.parse_args()

    produce = functools.partial(
        preprocess_ifs50r1, extent=config.REGIONS[args.region], template=TEMPLATE
    )
    run_job = functools.partial(
        run_days, template=TEMPLATE, allow_incomplete=args.allow_incomplete_window
    )
    job = cli.shard_job_for(args, SOURCE, run=RUN)
    cli.run(args, job, produce, "ifs50r1", run_job=run_job)


if __name__ == "__main__":
    main()
