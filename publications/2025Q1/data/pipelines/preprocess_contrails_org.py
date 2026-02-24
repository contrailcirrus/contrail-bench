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

from pycontrails.physics import units
from pycontrails.utils import coroutines, temp


# Pipeline parameters

#: Forecast times
TIMES = pd.date_range("2024-01-01 00:00", "2024-12-31 23:00", freq="1h").to_pydatetime().tolist()

#: Forecast flight levels
FLIGHT_LEVELS = list(range(270, 450, 10))

#: Path to /v0 forecast zarr store
V0_FORECAST_ZARR = "gs://contrails-301217-contrail-grid/v3/cocip-grid-hres.zarr"

#: Start time for continuous /v1 forecast
V1_FORECAST_START = datetime.datetime(2024, 6, 1, 0)

#: GCP buckets for temporary Beam files
BEAM_TEMP = "gs://contrails-301217-tmp-10-day-ttl/contrail-bench/2025Q1/beam-tmp"
BEAM_STAGING = "gs://contrails-301217-tmp-10-day-ttl/contrail-bench/2025Q1/beam-staging"

#: GCP bucket for temporary assets
GCP_TMPDIR = "gs://contrails-301217-contrail-bench/tmp/2025Q1/contrails-org"


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

    Uses the /v1 forecast starting June 1 2024 and the /v0 forecast earlier.

    Parameters
    ----------
    time : datetime.datetime
        Target time

    flight_level : int
        Target flight level

    sink : str
        Path where forecast netcdf file should be saved

    """
    if time < V1_FORECAST_START:
        get_v0_forecast(time, flight_level, sink)
        return

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


def get_v0_forecast(time: datetime.datetime, flight_level: int, sink: str) -> None:
    """Get v0 forecast at a single level and time.

    Parameters
    ----------
    time : datetime.datetime
        Target time

    flight_level : int
        Target flight level

    sink : str
        Path where forecast netcdf file should be saved

    """
    ds = xr.open_zarr(V0_FORECAST_ZARR)
    ds = ds[["ef_per_m"]]
    ds = ds.sel(time=[time], flight_level=[flight_level], aircraft_type="B738")
    ds = ds.drop("aircraft_type")
    ds = ds.assign_coords(flight_level=units.ft_to_pl(ds["flight_level"] * 100.0))
    ds = ds.rename(flight_level="level")
    ds.to_netcdf(sink)


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
        swcr = ds["ef_per_m"] >= 5e8
        swcr.attrs = {"long_name": "Strongly warming contrail region flag", "units": "1", "ef_threshold": "5e8 J/m"}
        ds["swcr"] = swcr
        fs = gcsfs.GCSFileSystem()
        sink = f"{GCP_TMPDIR}/{int(time.timestamp())}_{flight_level}.forecast.nc"
        with temp.temp_file() as tmp:
            ds[["pcr", "swcr"]].to_netcdf(tmp)
            fs.put(tmp, sink)


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

