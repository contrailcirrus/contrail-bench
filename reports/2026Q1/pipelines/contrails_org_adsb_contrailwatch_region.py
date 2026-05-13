"""Benchmark Contrails.org forecast using ADSB trajectories."""

import argparse
import datetime
import itertools

import apache_beam as beam
import gcsfs
import numpy as np
import pandas as pd
import xarray as xr
from apache_beam.options.pipeline_options import PipelineOptions
from scipy.ndimage import binary_dilation

from pycontrails.utils import temp


# Pipeline parameters

#: Forecast times
TIMES = pd.date_range("2024-01-01 00:00", "2024-12-31 23:00", freq="1h").to_pydatetime().tolist()

#: Forecast flight levels
FLIGHT_LEVELS = list(range(270, 450, 10))

#: Geographic extent
EXTENT = (-134, -63, 20, 50)

#: Buffer sizes (grid cells)
BUFFERS = list(range(11))

#: GCP buckets for temporary Beam files
BEAM_TEMP = "gs://contrails-301217-tmp-10-day-ttl/contrail-bench/2026Q1/beam-tmp"
BEAM_STAGING = "gs://contrails-301217-tmp-10-day-ttl/contrail-bench/2026Q1/beam-staging"

#: GCP buckets for temporary assets
GCP_TMPDIR = "gs://contrails-301217-contrail-bench/tmp/2026Q1/contrails-org-adsb-contrailwatch-region"
GCP_FORECAST_TMPDIR = "gs://contrails-301217-contrail-bench/tmp/2026Q1/contrails-org"
GCP_ADSB_TMPDIR = "gs://contrails-301217-contrail-bench/tmp/2026Q1/adsb"

#: GCP bucket for permanent assets
GCP_ASSETS = "gs://contrails-301217-contrail-bench/2026Q1/contrails-org-adsb-contrailwatch-region"


def get_pipeline_options(runner: str) -> PipelineOptions:
    """Get pipeline options.

    Parameters
    ----------
    runner : str
        Which runner to provide pipeline options for.
        Must be one of "direct" and "dataflow".

    Returns
    -------
    PipelineOptions
        Beam pipeline options

    """
    if runner == "direct":
        return PipelineOptions(
            runner="direct",
            direct_num_workers=1
        )

    if runner == "dataflow":
        return PipelineOptions(
            runner="dataflow",
            job_name="contrail-bench-2025q1-contrails-org-adsb-contrailwatch-region",
            project="contrails-301217",
            region="us-east1",
            temp_location=BEAM_TEMP,
            staging_location=BEAM_STAGING,
            sdk_container_image="us-east1-docker.pkg.dev/contrails-301217/contrail-bench/2025q1:latest",
            sdk_location="container",
            save_main_session=False,
            service_account_email="contrail-bench-staging-sa@contrails-301217.iam.gserviceaccount.com",
            machine_type="e2-highmem-4",
            autoscaling_algorithm="NONE",
            num_workers=100
        )

    msg = f"Invalid pipeline option identifier {runner}"
    raise ValueError(msg)


def open_forecast(time: datetime.datetime, flight_level: int) -> xr.Dataset:
    """Open preprocessed forecast.

    Parameters
    ----------
    time : datetime.datetime
        Forecast time

    flight_level : int
        Forecast flight level

    Return
    ------
    xr.Dataset
        Binary PCR forecast

    """
    gcs_path = f"{GCP_FORECAST_TMPDIR}/{int(time.timestamp())}_{flight_level}.nc"
    with temp.temp_file() as tmp:
        gcsfs.GCSFileSystem().get(gcs_path, tmp)
        return xr.open_dataset(tmp, engine="netcdf4")


def open_adsb(time: datetime.datetime, flight_level: int) -> pd.DataFrame:
    """Open preprocessed ADSB flight distance.

    Parameters
    ----------
    time : datetime.datetime
        Target time

    flight_level : int
        Target flight level

    Return
    ------
    pd.DataFrame
        Preprocessed ADSB flight distance

    """
    gcs_path = f"{GCP_ADSB_TMPDIR}/{int(time.timestamp())}_{flight_level}.pq"
    return pd.read_parquet(gcs_path)


def apply_horizontal_buffer(pcr: xr.DataArray, size: int) -> xr.DataArray:
    """Apply horizontal buffering to forecast PCR.

    Parameters
    ----------
    pcr : xr.DataArray
        Binary PCR forecast

    size : int
        Size of buffer (forecast grid cells)

    Returns
    -------
    xr.DataArray
        Binary PCR forecast with horizontal buffers added

    """
    if size < 1:
        return pcr
    
    structure = np.array([[False, True, False], [True, True, True], [False, True, False]]).reshape((3, 3, 1, 1))

    pad_left = pcr.values[-size:,...]
    pad_right = pcr.values[:size,...]
    padded = np.concat((pad_left, pcr.values, pad_right), axis=0)
    buffered = binary_dilation(padded, structure=structure, iterations=size)
    buffered = buffered[size:-size,...]

    return xr.DataArray(
        buffered,
        dims=pcr.dims,
        coords=pcr.coords,
    )


def calculate_metrics(time: datetime.datetime, flight_level: int) -> tuple[str, str]:
    """Compute metrics at a single time and flight level.

    Parameters
    ----------
    time : datetime.datetime
        Target time

    flight_level: int
        Target flight level

    Returns
    -------
    tuple[str, str]
        Key-value pair containing string-formatted time (YYYYMMDDHH) as key
        and path to GCS output as values.

    """
    forecast = open_forecast(time, flight_level)
    adsb = open_adsb(time, flight_level)

    # restrict to contrailwatch region
    lon_min, lon_max, lat_min, lat_max = EXTENT
    adsb = adsb[
        adsb["longitude"].between(lon_min, lon_max) & 
        adsb["latitude"].between(lat_min, lat_max)
    ]

    pcr = forecast["pcr"].compute()
    target_lon = xr.DataArray(adsb["longitude"], dims="segment")
    target_lat = xr.DataArray(adsb["latitude"], dims="segment")
    dist = xr.DataArray(adsb["flight_distance"], dims="segment")
    dist_tot = dist.sum().item()

    records = []
    for buffer_size in BUFFERS:

        buffered = apply_horizontal_buffer(pcr, buffer_size)
        predicted = buffered.sel(longitude=target_lon, latitude=target_lat)
        dist_pred = dist.where(predicted).sum().item()

        records.append({
            "time": time,
            "flight_level": flight_level,
            "horizontal_buffer": buffer_size,
            "vertical_buffer_up": 0,
            "vertical_buffer_down": 0,
            "adsb_dist_in_forecast_pcr": dist_pred,
            "adsb_dist": dist_tot
        })

    df = pd.DataFrame.from_records(records)
    sink = f"{GCP_TMPDIR}/{int(time.timestamp())}_{flight_level}.pq"
    df.to_parquet(sink)

    key = time.strftime("%Y%m%d%H")
    return (key, sink)


def write_metrics(key: str, paths: list[str]) -> None:
    """Write results to GCS.

    Parameters
    ----------
    key : str
        String-formatted time used as grouping key (YYYYMMDDHH)

    paths : str
        List of GCS paths with per-flight-level files

    """
    df = pd.concat((pd.read_parquet(p) for p in sorted(paths)), ignore_index=True)
    sink = f"{GCP_ASSETS}/{key}.pq"
    df.to_parquet(sink)


def main() -> None:
    """Program entrypoint."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=str, required=True)
    args = parser.parse_args()
    
    options = get_pipeline_options(args.runner)
    pcoll = itertools.product(TIMES, FLIGHT_LEVELS)

    with beam.Pipeline(options=options) as pipeline:
        (
            pipeline
            | "Create PCollection" >> beam.Create(pcoll)
            | "Compute metrics" >> beam.MapTuple(calculate_metrics)
            | "Group by time" >> beam.GroupByKey()
            | "Save to GCS" >> beam.MapTuple(write_metrics)
        )


if __name__ == "__main__":
    main()

