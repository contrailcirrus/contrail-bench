"""Preprocess IAGOS observations."""

import argparse
import datetime

import apache_beam as beam
import numpy as np
import pandas as pd
from apache_beam.options.pipeline_options import PipelineOptions
from scipy.stats import binned_statistic_2d

from pycontrails.core import JetA
from pycontrails.models import sac
from pycontrails.physics import constants, thermo, units


# Pipeline parameters

#: Target times
TIMES = pd.date_range("2024-01-01 00:00", "2024-12-31 23:00", freq="1h").to_pydatetime().tolist()

#: Target flight levels
FLIGHT_LEVELS = list(range(270, 450, 10))

#: Vertical resolution (ft) used for bucketing observations and ADSB data
VERTICAL_RESOLUTION = 250.0

#: Engine efficiency for SAC calculation
ENGINE_EFFICIENCY = 0.3

#: GCP buckets for temporary Beam files
BEAM_TEMP = "gs://contrails-301217-tmp-10-day-ttl/contrail-bench/2025Q1/beam-tmp"
BEAM_STAGING = "gs://contrails-301217-tmp-10-day-ttl/contrail-bench/2025Q1/beam-staging"

#: GCP bucket for temporary assets
GCP_TMPDIR = "gs://contrails-301217-contrail-bench/tmp/2025Q1/iagos"


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
            job_name="contrail-bench-2025q1-preprocess-iagos",
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
    df = pd.read_parquet(gcs_path, columns=[
        "flight_id",
        "longitude",
        "latitude",
        "pressure",
        "altitude_baro_m",
        "time",
        "segment_length",
        "air_temperature",
        "air_temperature_std_err",
        "air_temperature_validity_flag",
        "h2o_gas_ppmv",
        "h2o_gas_ppmv_std_err",
        "h2o_gas_validity_flag",
    ])

    mask = df["time"].between(start, end)
    df = df[mask]
    df = df.sort_values(by=["flight_id", "time"])

    return df


def preprocess_iagos(time: datetime.datetime) -> None:
    """Preprocess and save IAGOS data for one hour.

    Parameters
    ----------
    time : datetime.datetime
        Target time

    """
    df = get_iagos(time)

    air_temperature = df["air_temperature"].values
    specific_humidity = 1e-6 * df["h2o_gas_ppmv"].values * constants.R_d / constants.R_v
    air_pressure = df["pressure"].values
    G = sac.slope_mixing_line(specific_humidity, air_pressure, ENGINE_EFFICIENCY, JetA.ei_h2o, JetA.q_fuel)
    T_sat_liquid_ = sac.T_sat_liquid(G)
    rh_crit_sac = sac.rh_critical_sac(air_temperature, T_sat_liquid_, G)
    rh = thermo.rh(specific_humidity, air_temperature, air_pressure)
    rhi = thermo.rhi(specific_humidity, air_temperature, air_pressure)

    pcr = (rh > rh_crit_sac) & (rhi > 1.0)
    quality_mask = (
        (df["h2o_gas_validity_flag"] == 0) &
        (df["air_temperature_validity_flag"] == 0)
    ).values
    df["pcr"] = pcr
    df["quality_mask"] = quality_mask
    df["altitude_ft"] = df["altitude_baro_m"]

    longitude = np.linspace(-180.0, 179.75, 1440)  # 0.25 degrees
    latitude = np.linspace(-80.0, 80.0, 641)  # 0.25 degrees
    longitude, latitude = np.meshgrid(longitude, latitude, indexing="ij")
    longitude = longitude.ravel()
    latitude = latitude.ravel()
    
    longitude_bnds = np.linspace(-180.125, 179.875, 1441)
    latitude_bnds = np.linspace(-80.125, 80.125, 642)
    
    for flight_level in FLIGHT_LEVELS:
        
        sink = f"{GCP_TMPDIR}/{int(time.timestamp())}_{flight_level}.pq"

        mask = df["quality_mask"] & df["altitude_ft"].between(
            flight_level * 100.0 - VERTICAL_RESOLUTION,
            flight_level * 100.0 + VERTICAL_RESOLUTION,
            inclusive="both"
        )
        subset = df[mask].copy()
        
        if len(subset) == 0:
            out = pd.DataFrame(columns=["longitude", "latitude", "pcr_distance", "total_distance"])
            out.to_parquet(sink)
            continue

        mask = df["longitude"] > longitude_bnds[-1]
        subset.loc[mask, "longitude"] = subset.loc[mask, "longitude"] - 360.0
        pcr_dist, _, _, _ = binned_statistic_2d(
            subset["longitude"].values,
            subset["latitude"].values,
            subset["segment_length"].where(subset["pcr"], other=0.0).values,
            bins=[longitude_bnds, latitude_bnds],
            statistic="sum"
        )
        total_dist, _, _, _ = binned_statistic_2d(
            subset["longitude"].values,
            subset["latitude"].values,
            subset["segment_length"].values,
            bins=[longitude_bnds, latitude_bnds],
            statistic="sum"
        )

        pcr_dist = pcr_dist.ravel()
        total_dist = total_dist.ravel()
        mask = (pcr_dist > 0) | (total_dist > 0)

        out = pd.DataFrame({
            "longitude": longitude[mask],
            "latitude": latitude[mask],
            "pcr_distance": pcr_dist[mask],
            "total_distance": total_dist[mask]
        })
        out.to_parquet(sink)


def main() -> None:
    """Program entrypoint."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=str, required=True)
    args = parser.parse_args()
    
    options = get_pipeline_options(args.runner)
    pcoll = TIMES

    preprocess_iagos(datetime.datetime(2024, 6, 1, 10))
    
    with beam.Pipeline(options=options) as pipeline:
        (
            pipeline
            | "Create PCollection" >> beam.Create(pcoll)
            | "Preprocess IAGOS data" >> beam.Map(preprocess_iagos)
        )


if __name__ == "__main__":
    main()

