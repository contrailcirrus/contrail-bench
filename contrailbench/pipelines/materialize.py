"""CLI: warm a forecast source's preprocessed rhi/sac store for a source+region.

Replaces the per-source ``preprocess_<source>.py`` scripts -- the only
per-source logic left is each `sources.<name>.fetch`; everything else
(windowing, resumability, atomic writes) is generic.
"""

import argparse
import datetime
import os
import pathlib

import apache_beam as beam
import pandas as pd

from contrailbench import pcr, time_utils
from contrailbench.pipelines._common import DATA_DIR, parse_hours, pipeline_options
from contrailbench.sources import SOURCES

DEFAULT_START = pd.Timestamp("2024-09-01T00:00")
DEFAULT_END = pd.Timestamp("2024-12-31T23:00")


def store_dir(source: str, region: str = "conus", lead_hours: int | None = None) -> str:
    """Output directory for a source's preprocessed store.

    Matches the established ``data/{source}[-{region}][-lead{L:03d}]`` layout
    so this CLI can continue an existing (possibly partial) mirror without
    migrating data.

    Parameters
    ----------
    source : str
        Forecast source name (a key in :data:`contrailbench.sources.SOURCES`).

    region : str, optional
        Named region. ``"conus"`` (default) uses the bare source directory;
        any other region gets its own ``-{region}`` suffix.

    lead_hours : int, optional
        ``None`` (default) is the shortest-available-lead path. A fixed int
        gets its own ``-lead{L:03d}`` suffix.

    Returns
    -------
    str
    """
    name = source if region == "conus" else f"{source}-{region}"
    if lead_hours is not None:
        name = f"{name}-lead{lead_hours:03d}"
    return str(DATA_DIR / name)


def preprocess_forecast(
    time: datetime.datetime,
    flight_levels: list[int],
    source: str,
    *,
    extent: tuple[float, float, float, float],
    local_dir: str,
    lead_hours: int | None = None,
    mirror_dir: str | None = None,
) -> None:
    """Preprocess and save PCR fields at a single time, for many flight levels.

    One met fetch serves every flight level; the sink is written
    per-(time, flight_level), matching :class:`contrailbench.data.PCRStoreDataloader`'s
    expected schema.
    """
    fetch_kwargs = {}
    if lead_hours is not None:
        fetch_kwargs["lead_hours"] = lead_hours
    if mirror_dir is not None:
        fetch_kwargs["mirror_dir"] = mirror_dir

    batched = SOURCES[source].fetch(time, flight_levels, extent=extent, **fetch_kwargs)

    pathlib.Path(local_dir).mkdir(parents=True, exist_ok=True)
    for flight_level in flight_levels:
        out = pcr.extract_flight_level(batched, flight_level)
        sink = f"{local_dir}/{time_utils.to_utc_timestamp(time)}_{flight_level}.nc"
        pcr.write_netcdf_atomic(out, sink)


def main() -> None:
    """Program entrypoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, choices=sorted(SOURCES))
    parser.add_argument("--region", default="conus", choices=sorted(pcr.REGION_EXTENTS))
    parser.add_argument("--runner", required=True)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N times per contrailbench.pcr.limit_times's "
        "low-discrepancy order, instead of the full window.",
    )
    parser.add_argument(
        "--lead-hours",
        type=int,
        default=None,
        help="Preprocess a fixed lead's matched-validity-time intersection "
        "(contrailbench.pcr.LEAD_STUDY_TIMES) instead of the "
        "shortest-available-lead window.",
    )
    parser.add_argument("--mirror-dir", type=str, default=None)
    parser.add_argument("--start", type=pd.Timestamp, default=DEFAULT_START)
    parser.add_argument("--end", type=pd.Timestamp, default=DEFAULT_END)
    parser.add_argument(
        "--hours",
        type=str,
        default=None,
        help="Comma-separated UTC hours to restrict the forecast window to, "
        "e.g. '15,16,17,18,19,20,21,22'. Default (unset) keeps every hour.",
    )
    args = parser.parse_args()

    options = pipeline_options(
        args.runner, f"contrail-bench-materialize-{args.source}-{args.region}"
    )
    extent = pcr.REGION_EXTENTS[args.region]
    local_dir = store_dir(args.source, args.region, args.lead_hours)

    start = pd.Timestamp(args.start).to_pydatetime()
    end = pd.Timestamp(args.end).to_pydatetime()
    window_times = pcr.filter_hours(
        pd.date_range(start, end, freq="1h").to_pydatetime().tolist(), parse_hours(args.hours)
    )
    times = (
        list(pcr.LEAD_STUDY_TIMES)
        if args.lead_hours is not None
        else pcr.limit_times(window_times, args.limit)
    )
    existing = (
        [f"{local_dir}/{name}" for name in os.listdir(local_dir)]
        if os.path.isdir(local_dir)
        else []
    )
    times_to_run = pcr.pending_times(times, pcr.PCR_FLIGHT_LEVELS, existing, local_dir)
    print(
        f"{len(times) - len(times_to_run)}/{len(times)} hours already complete, "
        f"running {len(times_to_run)} remaining"
    )

    with beam.Pipeline(options=options) as pipeline:
        (
            pipeline
            | "Create PCollection" >> beam.Create(times_to_run)
            | "Preprocess forecasts"
            >> beam.Map(
                preprocess_forecast,
                flight_levels=pcr.PCR_FLIGHT_LEVELS,
                source=args.source,
                extent=extent,
                local_dir=local_dir,
                lead_hours=args.lead_hours,
                mirror_dir=args.mirror_dir,
            )
        )


if __name__ == "__main__":
    main()
