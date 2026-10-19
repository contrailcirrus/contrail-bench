"""Preprocess IAGOS observations."""

import datetime

import apache_beam as beam
import numpy as np
import pandas as pd
from scipy.stats import binned_statistic_2d

from pycontrails.core import JetA
from pycontrails.models import sac
from pycontrails.physics import constants, thermo, units

from src.beam import get_pipeline_options
from src.config import TIMES, FLIGHT_LEVELS, VERTICAL_RESOLUTION, LONGITUDE, LONGITUDE_BNDS, LATITUDE, LATITUDE_BNDS, ENGINE_EFFICIENCY, GCP_IAGOS_SINK, shard_id


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

    gcs_path = "gs://contrails-301217-iagos-v2/Processed/2025/waypoints.pq"
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
    df["altitude_ft"] = units.m_to_ft(df["altitude_baro_m"])

    longitude, latitude = np.meshgrid(LONGITUDE, LATITUDE, indexing="ij")
    longitude = longitude.ravel()
    latitude = latitude.ravel()
    
    for flight_level in FLIGHT_LEVELS:
        
        shard = shard_id(time, flight_level)
        sink = f"{GCP_IAGOS_SINK}/{shard}.pq"

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

        mask = df["longitude"] > LONGITUDE_BNDS[-1]
        subset.loc[mask, "longitude"] = subset.loc[mask, "longitude"] - 360.0
        pcr_dist, _, _, _ = binned_statistic_2d(
            subset["longitude"].values,
            subset["latitude"].values,
            subset["segment_length"].where(subset["pcr"], other=0.0).values,
            bins=[LONGITUDE_BNDS, LATITUDE_BNDS],
            statistic="sum"
        )
        total_dist, _, _, _ = binned_statistic_2d(
            subset["longitude"].values,
            subset["latitude"].values,
            subset["segment_length"].values,
            bins=[LONGITUDE_BNDS, LATITUDE_BNDS],
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

    options = get_pipeline_options(runner="dataflow", id="iagos")
    pcoll = TIMES

    preprocess_iagos(datetime.datetime(2025, 6, 1, 10))
    
    with beam.Pipeline(options=options) as pipeline:
        (
            pipeline
            | "Create PCollection" >> beam.Create(pcoll)
            | "Preprocess IAGOS data" >> beam.Map(preprocess_iagos)
        )


if __name__ == "__main__":
    main()

