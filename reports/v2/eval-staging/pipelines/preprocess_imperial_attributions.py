"""Preprocess Imperial attributions."""

import datetime
import logging

import apache_beam as beam
import gcsfs
import numpy as np
import pandas as pd
from scipy.stats import binned_statistic_2d

from src.beam import get_pipeline_options
from src.config import TIMES, FLIGHT_LEVELS, VERTICAL_RESOLUTION, LONGITUDE, LATITUDE, LONGITUDE_BNDS, LATITUDE_BNDS, GCP_IMPERIAL_ATT_SINK, shard_id

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
        Attributed flight segments. Include spatial and temporal coordinates
        of segment centerpoints.
    
    """
    target = pd.Timestamp(time)
    start = target - pd.Timedelta(minutes=30)
    end = target + pd.Timedelta(minutes=30)
    
    # data are sharded by hour and labeled by start time, so need to load two parquet files
    # to cover 30 minutes before to 30 minutes after the hour
    imperial_ingest = "gs://contrails-301217-contrail-bench/v2/obs-ingest/imperial-attributions/data/GOES-East/google_api"
    dates = pd.date_range(start.floor("1h"), end.floor("1h"), freq="1h")

    # a small number of hours are missing, so check before trying to open files
    fs = gcsfs.GCSFileSystem()
    paths = [f"{imperial_ingest}/{date.strftime("%Y%m%dT%H")}.parquet" for date in dates]
    paths = [path for path in paths if fs.exists(path)]

    # return an empty dataframe if no files are available
    if len(paths) == 0:
        return pd.DataFrame()

    df_list = [pd.read_parquet(path) for path in paths]
    df = pd.concat(df_list, axis="index")
    
    return df[df["time"].between(start, end)]


def preprocess_imperial_attributions(time: datetime.datetime) -> None:
    """Process and save Imperial attributions for one hour.

    Parameters
    ----------
    time : datetime.datetime
        Target time

    """
    logger.info(f"Starting processing for {time}")

    # Times and locations of Imperial attributions are centerpoints of attributed segments.
    # Unlike ContrailWatch attributions (which provide endpoints) we can't (and don't) resample
    # and instead treat attributions as evidence of a PCR in the grid cell that contains the
    # reported centerpoint.
    attributions = get_attributions(time)

    if len(attributions) == 0:
        logger.warning(f"No attributed flight segments found for {time}")
        for flight_level in FLIGHT_LEVELS:
            shard = shard_id(time, flight_level)
            sink = f"{GCP_IMPERIAL_ATT_SINK}/{shard}.pq"
            out = pd.DataFrame(columns=[
                "longitude",
                "latitude",
                "attributed_flight_distance_low",
                "attributed_flight_distance_medium",
                "attributed_flight_distance_high",
            ])
            out.to_parquet(sink)
        return

    longitude, latitude = np.meshgrid(LONGITUDE, LATITUDE, indexing="ij")
    longitude = longitude.ravel()
    latitude = latitude.ravel()
    
    for flight_level in FLIGHT_LEVELS:
        
        shard = shard_id(time, flight_level)
        sink = f"{GCP_IMPERIAL_ATT_SINK}/{shard}.pq"

        df = attributions[attributions["altitude_ft"].between(
            flight_level * 100.0 - VERTICAL_RESOLUTION,
            flight_level * 100.0 + VERTICAL_RESOLUTION
        )]

        if len(df) == 0:
            out = pd.DataFrame(columns=[
                "longitude",
                "latitude",
                "attributed_flight_distance_low",
                "attributed_flight_distance_medium",
                "attributed_flight_distance_high",
            ])
            out.to_parquet(sink)
            continue

        mask = df["longitude"] > LONGITUDE_BNDS[-1]
        df.loc[mask, "longitude"] = df.loc[mask, "longitude"] - 360.0
        
        dist_lo, _, _, _ = binned_statistic_2d(
            df["longitude"].values,
            df["latitude"].values,
            df["attributed_flight_distance_low"].values,
            bins=[LONGITUDE_BNDS, LATITUDE_BNDS],
            statistic="sum"
        )
        dist_lo = dist_lo.ravel()
        
        dist_md, _, _, _ = binned_statistic_2d(
            df["longitude"].values,
            df["latitude"].values,
            df["attributed_flight_distance_medium"].values,
            bins=[LONGITUDE_BNDS, LATITUDE_BNDS],
            statistic="sum"
        )
        dist_md = dist_md.ravel()
        
        dist_hi, _, _, _ = binned_statistic_2d(
            df["longitude"].values,
            df["latitude"].values,
            df["attributed_flight_distance_high"].values,
            bins=[LONGITUDE_BNDS, LATITUDE_BNDS],
            statistic="sum"
        )
        dist_hi = dist_hi.ravel()

        mask = np.maximum.reduce([dist_lo, dist_md, dist_hi]) > 0

        out = pd.DataFrame({
            "longitude": longitude[mask],
            "latitude": latitude[mask],
            "attributed_flight_distance_low": dist_lo[mask],
            "attributed_flight_distance_medium": dist_md[mask],
            "attributed_flight_distance_high": dist_hi[mask],
        })
        out.to_parquet(sink)
    
    logger.info(f"Finished processing for {time}")


def main() -> None:
    """Program entrypoint."""

    options = get_pipeline_options(runner="dataflow", id="imperial-attributions")
    pcoll = TIMES

    with beam.Pipeline(options=options) as pipeline:
        (
            pipeline 
            | "Create PCollection" >> beam.Create(pcoll)
            | "Preprocess attributions" >> beam.Map(preprocess_imperial_attributions)
        )


if __name__ == "__main__":
    main()

