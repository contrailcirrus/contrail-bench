"""Preprocess ContrailWatch attributions."""

import datetime
import logging
import warnings

import apache_beam as beam
import numpy as np
import pandas as pd
from scipy.stats import binned_statistic_2d

from pycontrails.core import Fleet

from src.beam import get_pipeline_options
from src.config import TIMES, FLIGHT_LEVELS, VERTICAL_RESOLUTION, LONGITUDE, LATITUDE, LONGITUDE_BNDS, LATITUDE_BNDS, GCP_CONTRAILWATCH_SINK, shard_id

logging.basicConfig(
    level="WARNING",
    format="[%(asctime)s.%(msecs)03d] [%(levelname)s] [%(module)s] [%(funcName)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel("INFO")


def get_attributions(time: datetime.datetime) -> pd.DataFrame:
    """Get DataFrame with attributed flight segments.

    Parameters
    ----------
    time : datetime.datetime
        Target time

    Returns
    -------
    pd.DataFrame
        Attributed flight segments. Includes the flight's ICAO address and flight number
        and the segment start and end time.
    
    """
    target = pd.Timestamp(time)
    start = target - pd.Timedelta(minutes=30)
    end = target + pd.Timedelta(minutes=30)
    
    # raw data is sharded by scheduled departure date
    # allow two day buffer before target date
    contrailwatch_ingest = "gs://contrails-301217-contrail-bench/v2/obs-ingest/contrailwatch"
    dates = pd.date_range(target.floor("1D") - pd.Timedelta(days=2), target.floor("1D"), freq="1D")
    df_list = [
        pd.read_parquet(f"{contrailwatch_ingest}/{date.strftime('%Y-%m-%d')}.pq")
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
        "longitude",
        "latitude",
        "altitude_baro",
        "timestamp"
    ]

    gcs_template = "gs://contrails-301217-gaia-trajectories/{}-Spire-Aireon/enhanced/accept/waypoints/{}-waypoints.pq"

    df = pd.read_parquet(
        gcs_template.format(start.floor("1D").year, start.floor("1D").strftime("%Y-%m-%d")),
        columns=columns
    )
    if end.floor("1D") != start.floor("1D"):
        df = pd.concat([df, pd.read_parquet(
            gcs_template.format(end.floor("1D").year, end.floor("1D").strftime("%Y-%m-%d")),
            columns=columns
        )], axis="index")

    df = df.sort_values(by=["flight_id", "timestamp"])

    # expand mask by 1 waypoint for segment length calculation
    mask = (df["timestamp"].between(start, end)).values.copy()
    mask[1:] |= mask[:-1]
    mask[:-1] |= mask[1:]
    df = df[mask].copy()

    # join flight number from summary files
    prev = target - pd.Timedelta(days=1)
    columns = ["flight_id", "flight_number"]
    gcs_meta_template = "gs://contrails-301217-gaia-trajectories/{}-Spire-Aireon/enhanced/accept/summary/{}-summary.pq"
    meta = pd.concat([
        pd.read_parquet(
            gcs_meta_template.format(prev.floor("1D").year, prev.floor("1D").strftime("%Y-%m-%d")),
            columns=columns
        ),
        pd.read_parquet(
            gcs_meta_template.format(target.floor("1D").year, target.floor("1D").strftime("%Y-%m-%d")),
            columns=columns
        )
    ], axis="index")
    id_to_flight_number_map = meta[["flight_id", "flight_number"]].set_index("flight_id")["flight_number"]
    df["flight_number"] = df["flight_id"].map(id_to_flight_number_map)

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
    # matches are based on (icao_address, flight_number) as a proxy for flight identity.
    # only keep (icao_address, flight_number) pairs that map to a single flight id
    fid_map = adsb.set_index(["icao_address", "flight_number"])["flight_id"]
    fid_map = fid_map.drop_duplicates(ignore_index=False)
    count = fid_map.groupby(fid_map.index).size()
    unique = count[count == 1].index
    fid_map = fid_map.loc[unique]

    # assign flight ids to attributions
    attributions = attributions.set_index(["icao_address", "flight_number"])
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
            shard = shard_id(time, flight_level)
            sink = f"{GCP_CONTRAILWATCH_SINK}/{shard}.pq"
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

    longitude, latitude = np.meshgrid(LONGITUDE, LATITUDE, indexing="ij")
    longitude = longitude.ravel()
    latitude = latitude.ravel()
    
    for flight_level in FLIGHT_LEVELS:
        
        shard = shard_id(time, flight_level)
        sink = f"{GCP_CONTRAILWATCH_SINK}/{shard}.pq"

        df = attributed[attributed["altitude_ft"].between(
            flight_level * 100.0 - VERTICAL_RESOLUTION,
            flight_level * 100.0 + VERTICAL_RESOLUTION
        )]

        if len(df) == 0:
            out = pd.DataFrame(columns=["longitude", "latitude", "attributed_flight_distance"])
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

        out = pd.DataFrame({"longitude": longitude[mask], "latitude": latitude[mask], "attributed_flight_distance": dist[mask]})
        out.to_parquet(sink)
    
    logger.info(f"Finished processing for {time}")


def main() -> None:
    """Program entrypoint."""

    options = get_pipeline_options(runner="dataflow", id="contrailwatch")
    pcoll = TIMES

    with beam.Pipeline(options=options) as pipeline:
        (
            pipeline 
            | "Create PCollection" >> beam.Create(pcoll)
            | "Preprocess attributions" >> beam.Map(preprocess_contrailwatch)
        )


if __name__ == "__main__":
    main()

