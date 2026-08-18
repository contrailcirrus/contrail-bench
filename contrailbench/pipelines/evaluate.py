"""CLI: evaluate a preprocessed forecast source against observation metrics.

Replaces the per-(source, observation-type, region) benchmark scripts -- one
generic entry point, parametrized by ``--source``/``--metrics``/``--region``,
since ``Forecast.evaluate_beam`` already scores every requested metric against
one forecast shard in a single pass.
"""

import argparse

import pandas as pd

from contrailbench import pcr
from contrailbench.data import (
    ADSBDataloader,
    ContrailWatchDataloader,
    GRUANDataloader,
    IAGOSDataloader,
    PCRStoreDataloader,
)
from contrailbench.forecast import Forecast
from contrailbench.metrics import FlightDistance, HitRate
from contrailbench.pipelines._common import DATA_DIR, parse_hours, pipeline_options
from contrailbench.pipelines.materialize import DEFAULT_END, DEFAULT_START, store_dir
from contrailbench.sources import SOURCES

#: obs name -> (Metric class, Dataloader class, default local obs-cache path)
METRICS = {
    "iagos": (HitRate, IAGOSDataloader, DATA_DIR / "_obs_cache" / "iagos"),
    "gruan": (HitRate, GRUANDataloader, DATA_DIR / "_obs_cache" / "gruan"),
    "contrailwatch": (HitRate, ContrailWatchDataloader, DATA_DIR / "_obs_cache" / "contrailwatch"),
    "adsb": (FlightDistance, ADSBDataloader, DATA_DIR / "_obs_cache" / "adsb"),
}


def output_dirs(
    source: str, metric_names: list[str], region: str, lead_hours: int | None = None
) -> tuple[str, str]:
    """(outputs, intermediates) directories for an evaluation run.

    Parameters
    ----------
    source : str
        Forecast source name.

    metric_names : list[str]
        Requested metric names, e.g. ``["iagos", "adsb"]``.

    region : str
        Named region.

    lead_hours : int, optional
        Fixed lead, if any -- gets its own ``-lead{L:03d}`` suffix.

    Returns
    -------
    tuple[str, str]
    """
    suffix = f"-lead{lead_hours:03d}" if lead_hours is not None else ""
    name = f"{source}-{'-'.join(metric_names)}-{region}{suffix}"
    return str(DATA_DIR / name), str(DATA_DIR / "_tmp" / name)


def main() -> None:
    """Program entrypoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, choices=sorted(SOURCES))
    parser.add_argument("--region", default="conus", choices=sorted(pcr.REGION_EXTENTS))
    parser.add_argument(
        "--metrics", required=True, help=f"comma-separated, from: {', '.join(sorted(METRICS))}"
    )
    parser.add_argument("--runner", required=True)
    parser.add_argument(
        "--lead-hours",
        type=int,
        default=None,
        help="Evaluate against a fixed lead's store instead of the shortest-available-lead store.",
    )
    parser.add_argument("--start", type=pd.Timestamp, default=DEFAULT_START)
    parser.add_argument("--end", type=pd.Timestamp, default=DEFAULT_END)
    parser.add_argument(
        "--hours",
        type=str,
        default=None,
        help="Comma-separated UTC hours to restrict the window to, "
        "e.g. '15,16,17,18,19,20,21,22'. Default (unset) keeps every hour.",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    metric_names = args.metrics.split(",")
    metrics = {}
    for name in metric_names:
        metric_cls, dataloader_cls, obs_path = METRICS[name]
        metrics[name] = metric_cls(dataloader_cls(path=str(obs_path)))

    extent = pcr.REGION_EXTENTS[args.region]
    forecast_dir = store_dir(args.source, args.region, args.lead_hours)

    start = pd.Timestamp(args.start).to_pydatetime()
    end = pd.Timestamp(args.end).to_pydatetime()
    times = pcr.filter_hours(
        pd.date_range(start, end, freq="1h").to_pydatetime().tolist(), parse_hours(args.hours)
    )

    forecast = Forecast(
        PCRStoreDataloader(path=forecast_dir),
        times=times,
        flight_levels=pcr.PCR_FLIGHT_LEVELS,
        extent=extent,
    )

    outputs, intermediates = output_dirs(args.source, metric_names, args.region, args.lead_hours)
    options = pipeline_options(args.runner, f"contrail-bench-evaluate-{args.source}-{args.region}")
    forecast.evaluate_beam(outputs, intermediates, options, resume=args.resume, **metrics)


if __name__ == "__main__":
    main()
