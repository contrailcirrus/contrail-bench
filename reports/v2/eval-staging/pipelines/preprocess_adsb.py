"""Preprocess ADSB telemetry."""

import datetime
import warnings

import apache_beam as beam
import numpy as np
import pandas as pd
from scipy.stats import binned_statistic_2d

from pycontrails.core import Fleet

from src.beam import get_pipeline_options
from src.config import TIMES, FLIGHT_LEVELS, VERTICAL_RESOLUTION, LONGITUDE, LATITUDE, LONGITUDE_BNDS, LATITUDE_BNDS, GCP_ADSB_SINK, shard_id


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

    gcs_template = "gs://contrails-301217-gaia-trajectories/{}-Spire-Aireon/enhanced/accept/waypoints/{}-waypoints.pq"

    gcs_path = gcs_template.format(start.floor("1D").year, start.floor("1D").strftime("%Y-%m-%d"))
    df = pd.read_parquet(gcs_path, columns=[
        "flight_id",
        "longitude",
        "latitude",
        "altitude_baro",
        "timestamp"
    ])
    if end.floor("1D") != start.floor("1D"):
        gcs_path = gcs_template.format(end.floor("1D").year, end.floor("1D").strftime("%Y-%m-%d"))
        df = pd.concat([
            df, 
            pd.read_parquet(gcs_path, columns=[
                "flight_id",
                "longitude",
                "latitude",
                "altitude_baro",
                "timestamp"
            ]),
        ], axis="index")

    df = df.sort_values(by=["flight_id", "timestamp"])

    # expand mask by 1 waypoint for segment length calculation
    mask = (df["timestamp"].between(start, end)).values.copy()
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

    # Resample from ~1 min to 10 s.
    # Otherwise flights can cross entire grid cells between waypoints!
    # Refine time mask after resampling.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message="Method 'resample_and_fill'")
        warnings.filterwarnings("ignore", category=UserWarning, message="Empty flight found")
        fleet = Fleet(data=df, altitude_ft=df["altitude_baro"], time=df["timestamp"]).resample_and_fill("10s")

    # Must compute segment length *before* filtering due to pycontrails bug
    segment_length = fleet.segment_length()
    segment_length[np.isnan(segment_length)] = 0.0
    fleet["segment_length"] = segment_length
    fleet["altitude_ft"] = fleet.altitude_ft
    
    # Can refine time mask now
    target = pd.Timestamp(time)
    start = target - pd.Timedelta(minutes=30)
    end = target + pd.Timedelta(minutes=30)
    fleet = fleet.filter((fleet["time"] >= start) & (fleet["time"] < end))

    longitude, latitude = np.meshgrid(LONGITUDE, LATITUDE, indexing="ij")
    longitude = longitude.ravel()
    latitude = latitude.ravel()
    
    for flight_level in FLIGHT_LEVELS:
        
        shard = shard_id(time, flight_level)
        sink = f"{GCP_ADSB_SINK}/{shard}.pq"

        df = fleet.filter(
            (fleet["altitude_ft"] >= flight_level * 100.0 - VERTICAL_RESOLUTION) &
            (fleet["altitude_ft"] <= flight_level * 100.0 + VERTICAL_RESOLUTION)
        ).dataframe

        if len(df) == 0:
            out = pd.DataFrame(columns=["longitude", "latitude", "flight_distance"])
            out.to_parquet(sink)
            continue

        mask = df["longitude"] > LONGITUDE_BNDS[-1]
        df.loc[mask, "longitude"] = df.loc[mask, "longitude"] - 360.0
        dist, _, _, _ = binned_statistic_2d(
            df["longitude"].values,
            df["latitude"].values,
            df["segment_length"].values,
            bins=[LONGITUDE_BNDS, LATITUDE_BNDS],
            statistic="sum"
        )

        dist = dist.ravel()
        mask = dist > 0

        out = pd.DataFrame({"longitude": longitude[mask], "latitude": latitude[mask], "flight_distance": dist[mask]})
        out.to_parquet(sink)


def main() -> None:
    """Program entrypoint."""

    options = get_pipeline_options(runner="dataflow", id="adsb")
    pcoll = TIMES
    
    with beam.Pipeline(options=options) as pipeline:
        (
            pipeline 
            | "Create PCollection" >> beam.Create(pcoll)
            | "Preprocess ADSB data" >> beam.Map(preprocess_adsb)
        )


if __name__ == "__main__":
    main()

