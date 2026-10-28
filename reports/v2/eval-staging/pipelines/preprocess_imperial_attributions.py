"""Preprocess Imperial attributions."""

import datetime
import logging

import apache_beam as beam
import gcsfs
import numpy as np
import numpy.typing as npt
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
    return pd.concat(df_list, axis="index")


def gc_interp(
    lon1: npt.NDArray[np.floating],
    lat1: npt.NDArray[np.floating],
    alt1: npt.NDArray[np.floating],
    t1: npt.NDArray[np.datetime64],
    lon2: npt.NDArray[np.floating],
    lat2: npt.NDArray[np.floating],
    alt2: npt.NDArray[np.floating],
    t2: npt.NDArray[np.datetime64],
    frac: npt.NDArray[np.floating],
) -> tuple[
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.datetime64]
]:
    r"""Interpolate along great circles via SLERP.

    Extension of https://github.com/contrailcirrus/contrailopt/blob/main/contrailopt/slerp.py

    Parameters
    ----------
    lon1 : npt.NDArray[np.floating]
        Longitude of source, [:math:`\deg`].
    lat1 : npt.NDArray[np.floating]
        Latitude of source, [:math:`\deg`].
    alt1 : npt.NDArray[np.floating]
        Altitude at source.
    t1 : npt.NDArray[np.floating]
        Time at source.
    lon2 : npt.NDArray[np.floating]
        Longitude of destination, [:math:`\deg`].
    lat2 : npt.NDArray[np.floating]
        Latitude of destination, [:math:`\deg`].
    alt2 : npt.NDArray[np.floating]
        Altitude at destination.
    t2 : npt.NDArray[np.floating]
        Time at destination.
    frac : npt.NDArray[np.floating]
        Fractional position along the arc, 0 = source, 1 = dest.

    Returns
    -------
    lon : npt.NDArray[np.floating]
        Interpolated longitude, [:math:`\deg`].
    lat : npt.NDArray[np.floating]
        Interpolated latitude, [:math:`\deg`].
    alt : npt.NDArray[np.floating]
        Interpolated altitude.
    t : npt.NDArray[np.datetime64]
        Interpolated time.

    """
    lon1r = np.deg2rad(lon1)
    lat1r = np.deg2rad(lat1)
    lon2r = np.deg2rad(lon2)
    lat2r = np.deg2rad(lat2)

    cos_lat1 = np.cos(lat1r)
    cos_lat2 = np.cos(lat2r)

    # Central angle via haversine
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2) ** 2 + cos_lat1 * cos_lat2 * np.sin(dlon / 2) ** 2
    sigma = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

    sin_sigma = np.sin(sigma)
    # Guard against zero-length arcs: as sigma -> 0, we have A -> 1 - frac and  B -> frac
    zero_arc = sin_sigma < 1e-12
    safe_sin_sigma = np.where(zero_arc, 1.0, sin_sigma)  # avoid division by zero
    A = np.where(zero_arc, 1.0 - frac, np.sin((1.0 - frac) * sigma) / safe_sin_sigma)
    B = np.where(zero_arc, frac, np.sin(frac * sigma) / safe_sin_sigma)

    # Cartesian SLERP
    x = A * cos_lat1 * np.cos(lon1r) + B * cos_lat2 * np.cos(lon2r)
    y = A * cos_lat1 * np.sin(lon1r) + B * cos_lat2 * np.sin(lon2r)
    z = A * np.sin(lat1r) + B * np.sin(lat2r)

    lat_out = np.rad2deg(np.arctan2(z, np.sqrt(x**2 + y**2)))
    lon_out = np.rad2deg(np.arctan2(y, x))

    alt_out = alt1  + (alt2 - alt1) * frac
    t_out = t1 + (t2 - t1) * frac

    return lon_out, lat_out, alt_out, t_out


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

    # Resample from ~1min to ~10s.
    # Allows attributions to be spread across grid cells.
    dt = attributions["time_end"] - attributions["time_start"]
    n = np.ceil(dt / np.timedelta64(10, "s")).astype(int)
    frac = np.concat([np.linspace(0.5 / nn, 1.0 - 0.5 / nn, nn) for nn in n])

    lon1 = np.repeat(attributions["longitude_start"].values, n)
    lat1 = np.repeat(attributions["latitude_start"].values, n)
    alt1 = np.repeat(attributions["altitude_ft_start"].values, n)
    t1 = np.repeat(attributions["time_start"].values, n)
    
    lon2 = np.repeat(attributions["longitude_end"].values, n)
    lat2 = np.repeat(attributions["latitude_end"].values, n)
    alt2 = np.repeat(attributions["altitude_ft_end"].values, n)
    t2 = np.repeat(attributions["time_end"].values, n)

    lon, lat, alt, t = gc_interp(lon1, lat1, alt1, t1, lon2, lat2, alt2, t2, frac)
    len_lo = np.repeat(attributions["attributed_flight_distance_low"].values / n, n)
    len_md = np.repeat(attributions["attributed_flight_distance_medium"].values / n, n)
    len_hi = np.repeat(attributions["attributed_flight_distance_high"].values / n, n)

    # Filter by time after resampling
    target = pd.Timestamp(time)
    start = target - pd.Timedelta(minutes=30)
    end = target + pd.Timedelta(minutes=30)
    mask = (t >= start) & (t < end)
    lon = lon[mask]
    lat = lat[mask]
    alt = alt[mask]
    len_lo = len_lo[mask]
    len_md = len_md[mask]
    len_hi = len_hi[mask]

    longitude, latitude = np.meshgrid(LONGITUDE, LATITUDE, indexing="ij")
    longitude = longitude.ravel()
    latitude = latitude.ravel()
    
    for flight_level in FLIGHT_LEVELS:
        
        shard = shard_id(time, flight_level)
        sink = f"{GCP_IMPERIAL_ATT_SINK}/{shard}.pq"

        fl_mask = (
            (alt >= flight_level * 100.0 - VERTICAL_RESOLUTION) &
            (alt <= flight_level * 100.0 + VERTICAL_RESOLUTION)
        )

        if fl_mask.sum() == 0:
            out = pd.DataFrame(columns=[
                "longitude",
                "latitude",
                "attributed_flight_distance_low",
                "attributed_flight_distance_medium",
                "attributed_flight_distance_high",
            ])
            out.to_parquet(sink)
            continue

        fl_lon = lon[fl_mask]
        fl_lat = lat[fl_mask]
        fl_lo = len_lo[fl_mask]
        fl_md = len_md[fl_mask]
        fl_hi = len_hi[fl_mask]

        wrap_mask = fl_lon > LONGITUDE_BNDS[-1]
        fl_lon[wrap_mask] = fl_lon[wrap_mask] - 360.0
        
        dist_lo, _, _, _ = binned_statistic_2d(
            fl_lon,
            fl_lat,
            fl_lo,
            bins=[LONGITUDE_BNDS, LATITUDE_BNDS],
            statistic="sum"
        )
        dist_lo = dist_lo.ravel()
        
        dist_md, _, _, _ = binned_statistic_2d(
            fl_lon,
            fl_lat,
            fl_md,
            bins=[LONGITUDE_BNDS, LATITUDE_BNDS],
            statistic="sum"
        )
        dist_md = dist_md.ravel()
        
        dist_hi, _, _, _ = binned_statistic_2d(
            fl_lon,
            fl_lat,
            fl_hi,
            bins=[LONGITUDE_BNDS, LATITUDE_BNDS],
            statistic="sum"
        )
        dist_hi = dist_hi.ravel()

        write_mask = np.maximum.reduce([dist_lo, dist_md, dist_hi]) > 0

        out = pd.DataFrame({
            "longitude": longitude[write_mask],
            "latitude": latitude[write_mask],
            "attributed_flight_distance_low": dist_lo[write_mask],
            "attributed_flight_distance_medium": dist_md[write_mask],
            "attributed_flight_distance_high": dist_hi[write_mask],
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

