"""Constants shared by every pipeline: levels, regions, and where outputs go."""

# Pipeline parameters

#: Flight levels (hundreds of feet) produced by every pipeline
FLIGHT_LEVELS = tuple(range(270, 450, 10))

#: Engine efficiency for the Schmidt-Appleman criterion
ENGINE_EFFICIENCY = 0.3

#: Default period (valid times, UTC, inclusive), the same as ``eval-staging``; override with
#: ``--start`` and ``--end``
DEFAULT_START = "2025-01-01 00:00"
DEFAULT_END = "2025-12-31 23:00"

#: Regions as ``(lon_min, lon_max, lat_min, lat_max)``. ``global`` is the exact span of the
#: 0.25 degree benchmark grid.
REGIONS = {
    "global": (-180.0, 179.75, -90.0, 90.0),
    "shanwick": (-30.0, -10.0, 45.0, 61.0),
    "conus": (-134.0, -63.0, 20.0, 50.0),
}

# Where things go

#: GCP project
PROJECT = "contrails-301217"

#: GCP bucket prefix for all pipeline outputs
OUTPUT_ROOT = "gs://contrails-301217-contrail-bench/v2"


def output_dir(source: str, region: str, run: str | None = None, root: str | None = None) -> str:
    """Directory a pipeline writes to: ``<root>/forecasts/<source>[/<run>]/<region>``.

    For example ``forecasts/ecmwf-ifs-hres-49r1/00z/global``. Each file in it is named
    ``{epoch}_{flight_level}.nc``, where ``epoch`` is the valid time in Unix seconds. ``root``
    replaces :data:`OUTPUT_ROOT`, for local runs.
    """
    parts = [(root or OUTPUT_ROOT).rstrip("/"), "forecasts", source]
    if run:
        parts.append(run.lower())
    parts.append(region)
    return "/".join(parts)
