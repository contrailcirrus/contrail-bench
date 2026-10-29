"""Constants shared by every pipeline: levels, regions, and where outputs go."""

# Pipeline parameters

#: Flight levels (hundreds of feet) produced by every pipeline
FLIGHT_LEVELS = tuple(range(270, 450, 10))

#: Engine efficiency for the Schmidt-Appleman criterion
ENGINE_EFFICIENCY = 0.3

#: Lead offsets (hours) staged by the forecast pipelines: each valid time gets its shortest lead
#: (1 to 6 hours) plus each of these. They stop at 48 so that the longest lead, 54 hours, is within
#: the Met Office archive's hourly period.
LEAD_OFFSETS = tuple(range(0, 49, 6))

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


def output_dir(source: str, region: str, root: str | None = None) -> str:
    """Directory a pipeline writes to: ``<root>/forecasts/<source>/<region>``.

    For example ``forecasts/ecmwf-ifs-hres-49r1/global``. A source with leads has one directory
    per valid time and flight level, ``{epoch}_{flight_level}/{lead}.nc``, with ``epoch`` the valid
    time in Unix seconds and ``lead`` in hours; any other has ``{epoch}_{flight_level}.nc``.
    ``root`` replaces :data:`OUTPUT_ROOT`, for local runs.
    """
    return "/".join([(root or OUTPUT_ROOT).rstrip("/"), "forecasts", source, region])
