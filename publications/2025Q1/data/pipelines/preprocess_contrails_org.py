"""Preprocess Contrails.org forecasts."""

import argparse
import datetime
import itertools

import aiohttp
import apache_beam as beam
import gcsfs
import pandas as pd
import xarray as xr
from apache_beam.options.pipeline_options import PipelineOptions
from google.cloud import secretmanager

from pycontrails.utils import coroutines, temp


# Pipeline parameters

#: Forecast times
TIMES = pd.date_range("2024-06-01 00:00", "2024-06-07 23:00", freq="1h").to_pydatetime().tolist()

#: Forecast flight levels
FLIGHT_LEVELS = list(range(270, 450, 10))

#: GCP buckets for temporary Beam files
BEAM_TEMP = "gs://contrails-301217-tmp-10-day-ttl/contrail-bench/2025Q1/beam-tmp"
BEAM_STAGING = "gs://contrails-301217-tmp-10-day-ttl/contrail-bench/2025Q1/beam-staging"

#: GCP bucket for temporary assets
GCP_TMPDIR = "gs://contrails-301217-tmp-10-day-ttl/contrail-bench/2025Q1/contrails-org"


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
            job_name="contrail-bench-2025q1-preprocess-contrails-org",
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


def get_secret(name: str) -> str:
    """Get secret from GCP secret manager.

    Parameters
    ----------
    name : str
        Name of secret    
        
    Returns
    -------
    str
        Value of secret

    """
    """Get secret from GCP secret manager."""
    client = secretmanager.SecretManagerServiceClient()
    project = "contrails-301217"
    full_name = f"projects/{project}/secrets/{name}/versions/latest"
    response = client.access_secret_version(name=full_name)
    return response.payload.data.decode()


async def get_forecast(time: datetime.datetime, flight_level: int, sink: str) -> None:
    """Get forecast at a single level and time.

    Parameters
    ----------
    time : datetime.datetime
        Target time

    flight_level : int
        Target flight level

    key : str
        Contrails API key

    sink : str
        Path where forecast netcdf file should be saved

    """
    url = "https://api.contrails.org/v1/grids"
    params = {
        "aircraft_class": "default",
        "flight_level": str(flight_level),
        "time": time.strftime("%Y-%m-%dT%H"),
        "units": "ef_per_m"
    }
    headers = {"x-api-key": get_secret("contrails-api-key")}

    async with aiohttp.ClientSession(raise_for_status=True) as session:
        async with session.get(url, params=params, headers=headers) as resp:
            content = await resp.read()
            with open(sink, "wb") as f:
                f.write(content)


def preprocess_forecast(time: datetime.datetime, flight_level: int) -> None:
    """Preprocess and save forecast at a single level and time.

    Parameters
    ----------
    time : datetime.datetime
        Target time

    flight_level : int
        Target flight level

    """
    with temp.temp_file() as raw:
        coroutines.run(get_forecast(time, flight_level, raw))
        ds = xr.open_dataset(raw, engine="netcdf4", decode_timedelta=True)
        pcr = ds["ef_per_m"] != 0
        pcr.attrs = {"long_name": "Persistent contrail region flag", "units": "1"}
        ds["pcr"] = pcr
        fs = gcsfs.GCSFileSystem()
        sink = f"{GCP_TMPDIR}/{int(time.timestamp())}_{flight_level}.forecast.nc"
        with temp.temp_file() as tmp:
            ds[["pcr"]].to_netcdf(tmp)
            fs.put(tmp, sink)


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
    gcs_path = f"{GCP_TMPDIR}/{int(time.timestamp())}_{flight_level}.forecast.nc"
    with temp.temp_file() as tmp:
        gcsfs.GCSFileSystem().get(gcs_path, tmp)
        return xr.open_dataset(tmp, engine="netcdf4")


def main() -> None:
    """Program entrypoint."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=str, required=True)
    args = parser.parse_args()
    
    options = get_pipeline_options(args.runner)
    pcoll = itertools.product(TIMES, FLIGHT_LEVELS)
    
    # preprocess data
    with beam.Pipeline(options=options) as pipeline:
        (
            pipeline
            | "Create PCollection" >> beam.Create(pcoll)
            | "Preprocess forecasts" >> beam.MapTuple(preprocess_forecast)
        )


if __name__ == "__main__":
    main()

