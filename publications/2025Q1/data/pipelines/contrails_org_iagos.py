"""Benchmark Contrails.org forecast using IAGOS observations."""

import argparse
import datetime
import itertools
from typing import Any

import apache_beam as beam
import numpy as np
import pandas as pd
import xarray as xr
from apache_beam.options.pipeline_options import PipelineOptions
from scipy.ndimage import binary_dilation

from pycontrails.physics import constants

from preprocess_adsb import open_adsb
from preprocess_contrails_org import open_forecast
from preprocess_iagos import open_iagos


# Pipeline parameters

#: Forecast times
TIMES = pd.date_range("2024-06-01 00:00", "2024-06-07 23:00", freq="1h").to_pydatetime().tolist()

#: Forecast flight levels
FLIGHT_LEVELS = list(range(270, 450, 10))

#: Vertical resolution (m) used for bucketing observations and ADSB data
VERTICAL_RESOLUTION = 250.0

#: Buffer sizes (grid cells)
BUFFERS = list(range(11))

#: GCP buckets for temporary Beam files
BEAM_TEMP = "gs://contrails-301217-tmp-10-day-ttl/contrail-bench/2025Q1/beam-tmp"
BEAM_STAGING = "gs://contrails-301217-tmp-10-day-ttl/contrail-bench/2025Q1/beam-staging"

#: GCP bucket for permanent assets
GCP_ASSETS = "gs://contrails-301217-contrail-bench/2025Q1/contrails-org-iagos"


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
            job_name="contrail-bench-2025q1-contrails-org-iagos",
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
    buffered = buffered[size:-size]

    return xr.DataArray(
        buffered,
        dims=pcr.dims,
        coords=pcr.coords,
        attrs=pcr.attrs.update({"buffer_size": size})
    )


def calculate_pcr_area(forecast: xr.DataArray, observed: pd.DataFrame) -> tuple[float, float]:
    """Calculate observed PCR areas.

    Parameters
    ----------
    forecast : xr.DataArray
        PCR forecasts

    observed: pd.DataFrame
        Locations of PCR observations

    Returns
    -------
    tuple[float, float]
        Observed PCR area inside forecast PCRs and total observed PCR area

    """
    target_lon = xr.DataArray(observed["longitude"], dims="observed")
    target_lat = xr.DataArray(observed["latitude"], dims="observed")
    area = (constants.radius_earth * np.deg2rad(0.25))**2 * xr.DataArray(np.cos(np.deg2rad(observed["latitude"])), dims="observed")
    predicted = forecast.sel(longitude=target_lon, latitude=target_lat, method="nearest")
    return area.where(predicted).sum().item(), area.sum().item()


def calculate_flight_distance(forecast: xr.DataArray, adsb: pd.DataFrame) -> tuple[float, float]:
    """Calculate flight distance penalty.

    Parameters
    ----------
    forecast : xr.DataArray
        PCR forecasts

    adsb: pd.DataFrame
        Flight distance through forecast grid cells

    Returns
    -------
    tuple[float, float]
        Flight distance through PCRs and total flight distance

    """
    target_lon = xr.DataArray(adsb["longitude"], dims="segment")
    target_lat = xr.DataArray(adsb["latitude"], dims="segment")
    dist = xr.DataArray(adsb["flight_distance"], dims="segment")
    predicted = forecast.sel(longitude=target_lon, latitude=target_lat, method="nearest")
    return dist.where(predicted).sum().item(), dist.sum().item()


def calculate_metrics_horizontal_buffer(time: datetime.datetime, flight_level: int, buffer_size: int) -> tuple[str, dict[str, Any]]:
    """Compute hit and penalty metrics with horizontal buffering only.

    Parameters
    ----------
    time : datetime.datetime
        Target time

    flight_level: int
        Target flight level

    buffer_size: int
        Target buffer size

    Returns
    -------
    tuple[str, dict[str, Any]]
        Metrics with string-formatted time as key (YYYYMMDDHH)

    """
    forecast = open_forecast(time, flight_level)
    adsb = open_adsb(time, flight_level)
    iagos = open_iagos(time, flight_level)

    buffered = apply_horizontal_buffer(forecast["pcr"], buffer_size)
    area_pred, area_tot = calculate_pcr_area(buffered, iagos)
    dist_pred, dist_tot = calculate_flight_distance(buffered, adsb)

    return (time.strftime("%Y%m%d%H"), {
        "time": time,
        "flight_level": flight_level,
        "horizontal_buffer": buffer_size,
        "vertical_buffer_up": 0,
        "vertical_buffer_down": 0,
        "iagos_pcr_area_in_forecast_pcr": area_pred,
        "iagos_pcr_area": area_tot,
        "adsb_dist_in_forecast_pcr": dist_pred,
        "adsb_dist": dist_tot
    })


def write_metrics(key: str, results: list[dict[str, Any]]) -> None:
    """Write results to GCS.

    Parameters
    ----------
    key : str
        String-formatted time used as grouping key (YYYYMMDDHH)

    results : list[dict[str, Any]]
        Metrics for different flight levels and buffer sizes

    """
    df = pd.DataFrame.from_records(results)
    sink = f"{GCP_ASSETS}/{key}.pq"
    df.to_parquet(sink)


def main() -> None:
    """Program entrypoint."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=str, required=True)
    args = parser.parse_args()
    
    options = get_pipeline_options(args.runner)
    pcoll = itertools.product(TIMES, FLIGHT_LEVELS, BUFFERS)

    # compute metrics
    with beam.Pipeline(options=options) as pipeline:
        (
            pipeline
            | "Create PCollection" >> beam.Create(pcoll)
            | "Compute metrics" >> beam.MapTuple(calculate_metrics_horizontal_buffer)
            | "Group by time" >> beam.GroupByKey()
            | "Save to GCS" >> beam.MapTuple(write_metrics)
        )



if __name__ == "__main__":
    main()

