"""Preprocess ContrailWatch attributions."""

import argparse
import datetime
import logging
import warnings

import apache_beam as beam
import numpy as np
import pandas as pd
from apache_beam.options.pipeline_options import PipelineOptions
from scipy.stats import binned_statistic_2d

from pycontrails.core import Fleet

logging.basicConfig(
    level="WARNING",
    format="[%(asctime)s.%(msecs)03d] [%(levelname)s] [%(module)s] [%(funcName)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel("INFO")


# Pipeline parameters

#: GCS path to cleaned ADSB data
GCS_TEMPLATE = "gs://contrails-301217-gaia-trajectories/2024-Spire-Aireon/enhanced/accept/waypoints/{}-waypoints.pq"

#: Output times
TIMES = pd.date_range("2024-05-01 00:00", "2024-12-31 23:00", freq="1h").to_pydatetime().tolist()

#: Output flight levels
FLIGHT_LEVELS = list(range(270, 450, 10))

#: Vertical resolution (m) used for bucketing attributions
VERTICAL_RESOLUTION = 250.0

#: GCP buckets for temporary Beam files
BEAM_TEMP = "gs://contrails-301217-tmp-10-day-ttl/contrail-bench/2025Q1/beam-tmp"
BEAM_STAGING = "gs://contrails-301217-tmp-10-day-ttl/contrail-bench/2025Q1/beam-staging"

#: GCP bucket for upstream assets
GCP_CONTRAILWATCH_TMPDIR = "gs://contrails-301217-contrail-bench/tmp/2025Q1/contrailwatch-raw"

#: GCP bucket for generated assets
GCP_TMPDIR = "gs://contrails-301217-contrail-bench/tmp/2025Q1/contrailwatch"


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
            job_name="contrail-bench-2025q1-preprocess-contrailwatch",
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
            num_workers=100,
        )

    msg = f"Invalid pipeline option identifier {runner}"
    raise ValueError(msg)


def get_attributions(time: datetime.datetime) -> pd.DataFrame:
    """Get DataFrame with attributed flight segments.

    Parameters
    ----------
    time : datetime.datetime
        Target time

    Returns
    -------
    pd.DataFrame
        Attributed flight segments. Includes the flight's ICAO address and callsign
        and the segment start and end time.
    
    """
    target = pd.Timestamp(time)
    start = target - pd.Timedelta(minutes=30)
    end = target + pd.Timedelta(minutes=30)
    
    # raw data is sharded by scheduled departure date
    # allow two day buffer before target date
    dates = pd.date_range(target.floor("1d") - pd.Timedelta(days=2), target.floor("1d"), freq="1d")
    df_list = [
        pd.read_parquet(f"{GCP_CONTRAILWATCH_TMPDIR}/{date.strftime('%Y-%m-%d')}.pq")
        for date in dates
    ]
    df_list = [df for df in df_list if not df.empty]
    df = pd.concat(df_list, axis="index")

    return df[(df["end"] >= start) & (df["start"] <= end)]
    

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
    columns = [
        "flight_id",
        "icao_address",
        "callsign",
        "longitude",
        "latitude",
        "altitude_baro",
        "timestamp"
    ]
        

    df = pd.read_parquet(GCS_TEMPLATE.format(start.floor("1d").strftime("%Y-%m-%d")), columns=columns)
    if end.floor("1d") != start.floor("1d"):
        df = pd.concat([
            df,
            pd.read_parquet(GCS_TEMPLATE.format(end.floor("1d").strftime("%Y-%m-%d")), columns=columns)
        ], axis="index")

    df = df.sort_values(by=["flight_id", "timestamp"])

    # expand mask by 1 waypoint for segment length calculation
    mask = (df["timestamp"].between(start, end)).values
    mask[1:] |= mask[:-1]
    mask[:-1] |= mask[1:]
    df = df[mask].copy()

    return df


def join(attributions: pd.DataFrame, adsb: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join attributed flight segments with ADSB trajectories.

    Parameters
    ----------
    attributions : pd.DataFrame
        Attributed flight segments

    adsb : pd.DataFrame
        ADSB trajectories

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Joined attributions and trajectories. After the join,
        - flight ids are attached to attributed segments that were matched to a trajectory
        - attributed segments that were not matched to a trajectory are dropped
        - trajectories not matched any attributed segments are dropped

    """
    # matches are based on (icao_address, callsign) as a proxy for flight identity.
    # only keep (icao_address, callsign) pairs that map to a single flight id
    fid_map = adsb.set_index(["icao_address", "callsign"])["flight_id"]
    fid_map = fid_map.drop_duplicates(ignore_index=False)
    count = fid_map.groupby(fid_map.index).size()
    unique = count[count == 1].index
    fid_map = fid_map.loc[unique]

    # assign flight ids to attributions
    attributions = attributions.set_index(["icao_address", "callsign"])
    attributions = attributions[attributions.index.isin(fid_map.index)]
    attributions["flight_id"] = attributions.index.map(fid_map)
    attributions = attributions.reset_index()

    # drop trajectories not matched to attributed segments
    adsb = adsb[adsb["flight_id"].isin(attributions["flight_id"])]

    return attributions, adsb


def _attributed_segments(trajectory: pd.DataFrame, attributions: pd.DataFrame) -> pd.DataFrame:
    """Keep only portions of a trajectory within attributed segments."""
    flight_id = trajectory.name  # set during groupby on flight id
    segments = attributions[attributions["flight_id"] == flight_id]
    mask = pd.Series(False, index=trajectory.index)
    for _, segment in segments.iterrows():
        mask |= trajectory["time"].between(segment["start"], segment["end"])
    return trajectory[mask]


def preprocess_contrailwatch(time: datetime.datetime) -> None:
    """Process and save ContrailWatch attributions for one hour.

    Parameters
    ----------
    time : datetime.datetime
        Target time

    """
    logger.info(f"Starting processing for {time}")

    attributions = get_attributions(time)
    df = get_adsb(time)
    attributions, df = join(attributions, df)

    if len(df) == 0:
        logger.warning(f"No attributed flight segments found for {time}")
        for flight_level in FLIGHT_LEVELS:
            sink = f"{GCP_TMPDIR}/{int(time.timestamp())}_{flight_level}.pq"
            out = pd.DataFrame(columns=["longitude", "latitude", "attributed_flight_distance"])
            out.to_parquet(sink)
        return

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

    # filter for attributed segments only
    attributed = fleet.dataframe.groupby("flight_id")[fleet.dataframe.columns].apply(
        lambda x: _attributed_segments(x, attributions)
    )

    longitude = np.linspace(-180.0, 179.75, 1440)  # 0.25 degrees
    latitude = np.linspace(-80.0, 80.0, 641)  # 0.25 degrees
    longitude, latitude = np.meshgrid(longitude, latitude, indexing="ij")
    longitude = longitude.ravel()
    latitude = latitude.ravel()
    
    longitude_bnds = np.linspace(-180.125, 179.875, 1441)
    latitude_bnds = np.linspace(-80.125, 80.125, 642)

    for flight_level in FLIGHT_LEVELS:
        
        sink = f"{GCP_TMPDIR}/{int(time.timestamp())}_{flight_level}.pq"

        df = attributed[attributed["altitude_ft"].between(
            flight_level * 100.0 - VERTICAL_RESOLUTION,
            flight_level * 100.0 + VERTICAL_RESOLUTION
        )]

        if len(df) == 0:
            out = pd.DataFrame(columns=["longitude", "latitude", "attributed_flight_distance"])
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

        out = pd.DataFrame({"longitude": longitude[mask], "latitude": latitude[mask], "attributed_flight_distance": dist[mask]})
        out.to_parquet(sink)
    
    logger.info(f"Finished processing for {time}")


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
            | "Preprocess attributions" >> beam.Map(preprocess_contrailwatch)
        )


if __name__ == "__main__":
    main()

