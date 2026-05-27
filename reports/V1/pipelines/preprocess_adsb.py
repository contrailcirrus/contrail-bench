"""Preprocess ADSB telemetry."""

import argparse
import datetime
import warnings

import apache_beam as beam
import numpy as np
import pandas as pd
from apache_beam.options.pipeline_options import PipelineOptions
from scipy.stats import binned_statistic_2d

from pycontrails.core import Fleet


# Pipeline parameters

#: GCS path to cleaned ADS-B data
GCS_TEMPLATE = "gs://contrails-301217-gaia-trajectories/2024-Spire-Aireon/enhanced/accept/waypoints/{}-waypoints.pq"

#: Target times
TIMES = pd.date_range("2024-01-01 00:00", "2024-12-31 23:00", freq="1h").to_pydatetime().tolist()

#: Target flight levels
FLIGHT_LEVELS = list(range(270, 450, 10))

#: Vertical resolution (m) used for bucketing segments
VERTICAL_RESOLUTION = 250.0

#: GCP buckets for temporary Beam files
BEAM_TEMP = "gs://contrails-301217-tmp-10-day-ttl/contrail-bench/2026Q1/beam-tmp"
BEAM_STAGING = "gs://contrails-301217-tmp-10-day-ttl/contrail-bench/2026Q1/beam-staging"

#: GCP bucket for temporary assets
GCP_TMPDIR = "gs://contrails-301217-contrail-bench/2026Q1/adsb"


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
            job_name="contrail-bench-2025q1-preprocess-adsb",
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

    df = pd.read_parquet(GCS_TEMPLATE.format(start.floor("1d").strftime("%Y-%m-%d")), columns=[
        "flight_id",
        "longitude",
        "latitude",
        "altitude_baro",
        "timestamp"
    ])
    if end.floor("1d") != start.floor("1d"):
        df = pd.concat([
            df, 
            pd.read_parquet(
                GCS_TEMPLATE.format(end.floor("1d").strftime("%Y-%m-%d")),
                columns=["flight_id", "longitude", "latitude", "altitude_baro", "timestamp"]
        )], axis="index")

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

    longitude = np.linspace(-180.0, 179.75, 1440)  # 0.25 degrees
    latitude = np.linspace(-80.0, 80.0, 641)  # 0.25 degrees
    longitude, latitude = np.meshgrid(longitude, latitude, indexing="ij")
    longitude = longitude.ravel()
    latitude = latitude.ravel()
    
    longitude_bnds = np.linspace(-180.125, 179.875, 1441)
    latitude_bnds = np.linspace(-80.125, 80.125, 642)

    for flight_level in FLIGHT_LEVELS:
        
        sink = f"{GCP_TMPDIR}/{int(time.timestamp())}_{flight_level}.pq"

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


def main() -> None:
    """Program entrypoint."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=str, required=True)
    args = parser.parse_args()
    
    options = get_pipeline_options(args.runner)
    pcoll = TIMES
    
    with beam.Pipeline(options=options) as pipeline:
        (
            pipeline 
            | "Create PCollection" >> beam.Create(pcoll)
            | "Preprocess ADSB data" >> beam.Map(preprocess_adsb)
        )


if __name__ == "__main__":
    main()

