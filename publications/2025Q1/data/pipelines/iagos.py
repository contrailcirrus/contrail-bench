"""Stage ContrailBench data for IAGOS."""

import argparse
import asyncio
import datetime
import warnings

import aiohttp
import apache_beam as beam
import gcsfs
import numpy as np
import pandas as pd
import xarray as xr
from apache_beam.options.pipeline_options import PipelineOptions
from google.cloud import secretmanager
from scipy.spatial import KDTree
from scipy.stats import binned_statistic_2d

from pycontrails.core import Fleet, JetA
from pycontrails.models import sac
from pycontrails.physics import constants, thermo, units
from pycontrails.utils import temp


# Pipeline parameters

#: Forecast times
TIMES = [datetime.datetime(2024, 6, 1, 10), datetime.datetime(2024, 6, 1, 11)]

#: Forecast flight levels
FLIGHT_LEVELS = [340, 350, 360]

#: Vertical resolution (m) used for bucketing observations and ADSB data
VERTICAL_RESOLUTION = 250.0

#: GCP buckets for temporary Beam files
BEAM_TEMP = "gs://contrails-301217-tmp-10-day-ttl/contrail-bench/2025Q1/beam-tmp"
BEAM_STAGING = "gs://contrails-301217-tmp-10-day-ttl/contrail-bench/2025Q1/beam-staging"

#: GCP bucket for temporary assets
GCP_TMPDIR = "gs://contrails-301217-tmp-10-day-ttl/contrail-bench/2025Q1/iagos"

#: GCP bucket for permanent assets
GCP_ASSETS = "gs://contrails-301217-contrail-bench/2025Q1/iagos"


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
            job_name="contrail-bench-2025q1-iagos",
            project="contrails-301217",
            region="us-east1",
            temp_location=BEAM_TEMP,
            staging_location=BEAM_STAGING,
            sdk_container_image="us-east1-docker.pkg.dev/contrails-301217/contrail-bench/2025q1-iagos:latest",
            sdk_location="container",
            save_main_session=False,
            service_account_email="contrail-bench-staging-sa@contrails-301217.iam.gserviceaccount.com",
            machine_type="e2-highmem-4",
            autoscaling_algorithm="NONE",
            num_workers=5
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
        asyncio.run(get_forecast(time, flight_level, raw))
        ds = xr.open_dataset(raw, engine="netcdf4", decode_timedelta=True)
        pcr = ds["ef_per_m"] != 0
        pcr.attrs = {"long_name": "Persistent contrail region flag", "units": "1"}
        ds["pcr"] = pcr
        fs = gcsfs.GCSFileSystem()
        sink = f"{GCP_TMPDIR}/{int(time.timestamp())}_{flight_level}.forecast.nc"
        with temp.temp_file() as tmp:
            ds[["pcr"]].to_netcdf(tmp)
            fs.put(tmp, sink)


def get_adsb(time: datetime.datetime) -> pd.DataFrame:
    """Get one hour of global ADSB data centered at target time.

    Parameters
    ----------
    time : datetime.datetime
        Target time

    Returns
    -------
    pd.DataFrame
        Unprocessed ADSB data

    """
    target = pd.Timestamp(time)
    start = target - pd.Timedelta(minutes=30)
    end = target + pd.Timedelta(minutes=30)

    gcs_path = "gs://contrails-301217-gaia/2024-Spire-Aireon/enhanced/accept/waypoints/{}-waypoints.pq"
    df = pd.read_parquet(gcs_path.format(start.floor("1d").strftime("%Y-%m-%d")))
    if end.floor("1d") != start.floor("1d"):
        df = pd.concat([df, pd.read_parquet(gcs_path.format(end.floor("1d").strftime("%Y-%m-%d")))], axis="index")

    df = df.sort_values(by=["flight_id", "timestamp"])

    # expand mask by 1 waypoint for segment length calculation
    mask = (df["timestamp"].between(start, end)).values
    mask[1:] |= mask[:-1]
    mask[:-1] |= mask[1:]
    df = df[mask].copy()

    return df


def preprocess_adsb(time: datetime.datetime) -> None:
    """Process and save ADSB data for one hour.

    Parameters
    ----------
    time : datetime.datetime
        Target time

    """
    df = get_adsb(time)

    # Resample from ~1 min to 4 s (equal to IAGOS sampling frequency).
    # Otherwise flights can cross entire grid cells between waypoints!
    # Refine time mask after resampling.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message="Method 'resample_and_fill'")
        warnings.filterwarnings("ignore", category=UserWarning, message="Empty flight found")
        fleet = Fleet(data=df, altitude_ft=df["altitude_baro"], time=df["timestamp"]).resample_and_fill("4s")
    target = pd.Timestamp(time)
    start = target - pd.Timedelta(minutes=30)
    end = target + pd.Timedelta(minutes=30)
    fleet = fleet.filter((fleet["time"] >= start) & (fleet["time"] < end))

    segment_length = fleet.segment_length()
    segment_length[np.isnan(segment_length)] = 0.0
    fleet["segment_length"] = segment_length
    fleet["altitude_ft"] = fleet.altitude_ft

    longitude = np.linspace(-180.0, 179.75, 1440)  # 0.25 degrees
    latitude = np.linspace(-80.0, 80.0, 641)  # 0.25 degrees
    longitude, latitude = np.meshgrid(longitude, latitude, indexing="ij")
    longitude = longitude.ravel()
    latitude = latitude.ravel()
    
    longitude_bnds = np.linspace(-180.125, 179.875, 1441)
    latitude_bnds = np.linspace(-80.125, 80.125, 642)

    for flight_level in FLIGHT_LEVELS:
        
        sink = f"{GCP_TMPDIR}/{int(time.timestamp())}_{flight_level}.adsb.pq"

        df = fleet.filter(
            (fleet["altitude_ft"] >= flight_level * 100.0 - VERTICAL_RESOLUTION) &
            (fleet["altitude_ft"] <= flight_level * 100.0 + VERTICAL_RESOLUTION)
        ).dataframe

        if len(df) == 0:
            out = pd.DataFrame(columns=["longitude", "latitude", "flight_distance"])
            out.to_parquet(sink)
            continue

        mask = df["longitude"] > longitude_bnds[-1]
        df.loc[mask, "longitude"] = df.loc[mask, "longitude"] - 360.0
        dist, _, _, _ = binned_statistic_2d(
            df["longitude"].values,
            df["latitude"].values,
            df["segment_length"].values,
            bins=[longitude_bnds, latitude_bnds],
            statistic="sum"
        )

        dist = dist.ravel()
        mask = dist > 0

        out = pd.DataFrame({"longitude": longitude[mask], "latitude": latitude[mask], "flight_distance": dist[mask]})
        out.to_parquet(sink)


def get_iagos(time: datetime.datetime) -> pd.DataFrame:
    """Get one hour of IAGOS data centered at target time.

    Parameters
    ----------
    time : datetime.datetime
        Target time

    Returns
    -------
    pd.DataFrame
        Unprocessed IAGOS data

    """
    target = pd.Timestamp(time)
    start = target - pd.Timedelta(minutes=30)
    end = target + pd.Timedelta(minutes=30)

    gcs_path = "gs://contrails-301217-iagos-v2/Processed/2024/waypoints.pq"
    df = pd.read_parquet(gcs_path)

    df = df.sort_values(by=["flight_id", "time"])

    mask = df["time"].between(start, end)
    df = df[mask].copy()

    return df


def preprocess_iagos(time: datetime.datetime) -> None:
    """Preprocess and save IAGOS data for one hour.

    Parameters
    ----------
    time : datetime.datetime
        Target time

    """
    df = get_iagos(time)

    engine_efficiency = 0.3
    fuel = JetA()

    air_temperature = df["air_temperature"].values
    specific_humidity = 1e-6 * df["h2o_gas_ppmv"].values * constants.R_d / constants.R_v
    air_pressure = df["pressure"].values
    G = sac.slope_mixing_line(specific_humidity, air_pressure, engine_efficiency, fuel.ei_h2o, fuel.q_fuel)
    T_sat_liquid_ = sac.T_sat_liquid(G)
    rh_crit_sac = sac.rh_critical_sac(air_temperature, T_sat_liquid_, G)
    rh = thermo.rh(specific_humidity, air_temperature, air_pressure)
    rhi = thermo.rhi(specific_humidity, air_temperature, air_pressure)

    pcr = (rh > rh_crit_sac) & (rhi > 1.0)
    df["pcr"] = pcr
    df["altitude_ft"] = units.m_to_ft(df["altitude_baro_m"])

    longitude = np.linspace(-180.0, 179.75, 1440)  # 0.25 degrees
    latitude = np.linspace(-80.0, 80.0, 641)  # 0.25 degrees
    longitude, latitude = np.meshgrid(longitude, latitude, indexing="ij")
    longitude = longitude.ravel()
    latitude = latitude.ravel()
    
    longitude_bnds = np.linspace(-180.125, 179.875, 1441)
    latitude_bnds = np.linspace(-80.125, 80.125, 642)
    
    for flight_level in FLIGHT_LEVELS:
        
        sink = f"{GCP_TMPDIR}/{int(time.timestamp())}_{flight_level}.iagos.pq"

        mask = df["altitude_ft"].between(
            flight_level * 100.0 - VERTICAL_RESOLUTION,
            flight_level * 100.0 + VERTICAL_RESOLUTION,
            inclusive="both"
        )
        subset = df[mask].copy()
        
        if len(subset) == 0:
            out = pd.DataFrame(columns=["longitude", "latitude", "pcr_count"])
            out.to_parquet(sink)
            continue

        mask = df["longitude"] > longitude_bnds[-1]
        subset.loc[mask, "longitude"] = subset.loc[mask, "longitude"] - 360.0
        count, _, _, _ = binned_statistic_2d(
            subset["longitude"].values,
            subset["latitude"].values,
            subset["pcr"].values,
            bins=[longitude_bnds, latitude_bnds],
            statistic="sum"
        )

        count = count.ravel()
        mask = count > 0

        out = pd.DataFrame({"longitude": longitude[mask], "latitude": latitude[mask], "pcr_count": count[mask]})
        out.to_parquet(sink)


def main() -> None:
    """Program entrypoint."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=str, required=True)
    args = parser.parse_args()
    
    options = get_pipeline_options(args.runner)
    
    with beam.Pipeline(options=options) as pipeline:
        pcoll = pipeline | "Create PCollection" >> beam.Create(TIMES)
        (
            pcoll
            | "Compute outer product" >> beam.FlatMap(lambda t: ((t, fl) for fl in FLIGHT_LEVELS))
            | "Fan out" >> beam.Reshuffle()
            | "Preprocess forecasts" >> beam.MapTuple(preprocess_forecast)
        )
        pcoll | "Preprocess ADSB data" >> beam.Map(preprocess_adsb)
        pcoll | "Preprocess IAGOS data" >> beam.Map(preprocess_iagos)


if __name__ == "__main__":
    main()

