"""Preprocess GRUAN RS41 observations.

Profiles are read from a GCS bucket that contains a mirror of
https://www.gruan.org/data/file-archive/rs41-gdp1-at-lc/*/2025/*
"""

import itertools
import logging
import tempfile
import warnings
from concurrent.futures import ProcessPoolExecutor

import gcsfs
import numpy as np
import pandas as pd
import xarray as xr
from pycontrails import JetA
from pycontrails.models import pcr as pcr_mod
from pycontrails.physics import constants, thermo, units
from tqdm import tqdm

from src.config import TIMES, FLIGHT_LEVELS, VERTICAL_RESOLUTION, LONGITUDE, LATITUDE, LONGITUDE_BNDS, LATITUDE_BNDS, ENGINE_EFFICIENCY, GCP_GRUAN_SINK, shard_id

logging.basicConfig(
    level="WARNING",
    format="[%(asctime)s.%(msecs)03d] [%(levelname)s] [%(module)s] [%(funcName)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel("INFO")

#: Number of parallel workers
NUM_WORKERS = 8


def process_waypoints(ds: xr.Dataset) -> pd.DataFrame:
    """Compute binned PCR and total observation counts.

    Parameters
    ----------
    ds : xr.Dataset
        GRUAN profile

    Returns
    -------
    pd.DataFrame
        Binned counts

    """
    assert ds["press"].attrs["units"] == "hPa"
    assert ds["temp"].attrs["units"] == "K"

    rh_unit = ds["rh"].attrs["units"]
    assert rh_unit in ("percent", "1")

    if "cor_temp" in ds:
        ds = ds.rename(cor_temp="temp_corr", cor_rh="rh_corr")

    # ds is lazy despite having in-memory data, so load just the data we need
    vars = ["time", "lat", "lon", "press", "temp", "rh", "temp_corr", "rh_corr"]
    df = ds[vars].to_dataframe().reset_index()

    # Only keep rows within barometric altitude bounds
    # Can ignore warnings about disagreement with ISA at very high altitudes
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message="Pressure level is below")
        fl = units.pl_to_ft(df["press"]) / 100.0
    fl_bin = (fl / 10.0).round() * 10.0
    cond = (
        (fl >= FLIGHT_LEVELS[0])
        & (fl <= FLIGHT_LEVELS[-1])
        & ((fl - fl_bin).abs() * 100 <= VERTICAL_RESOLUTION)
    )

    df = df.assign(fl_bin=fl_bin)[cond]

    # Assign remaining rows to longitude, latitude and time bins
    # Need to explicitly drop nan longitude and latitudes, which break assignment logic
    df = df.dropna(subset=["lon", "lat"])
    
    lon = df["lon"].values.copy()
    mask = lon > LONGITUDE_BNDS[-1]
    lon[mask] = lon[mask] - 360.0
    lon_idx = np.searchsorted(LONGITUDE_BNDS, lon) - 1
    lon_bin = LONGITUDE[lon_idx]

    lat_idx = np.searchsorted(LATITUDE_BNDS, df["lat"]) - 1
    lat_bin = LATITUDE[lat_idx]

    time_hour = df["time"].dt.round("h")

    df = df.assign(lon_bin=lon_bin, lat_bin=lat_bin, time_hour=time_hour)

    T = df["temp"] + df["temp_corr"]
    rh = df["rh"] + df["rh_corr"]
    if rh_unit == "percent":
        rh = rh / 100.0  # percent -> fraction

    p = df["press"] * 100.0  # hPa -> Pa
    q = rh * constants.epsilon * thermo.e_sat_liquid(T) / p

    df["pcr"] = pcr_mod.pcr(T, q, p, ENGINE_EFFICIENCY, ei_h2o=JetA.ei_h2o, q_fuel=JetA.q_fuel)[0] == 1.0
    grouped = df.groupby(["time_hour", "lat_bin", "lon_bin", "fl_bin"])["pcr"]
    return grouped.agg(pcr_count="sum", total_count="count")


def read_and_process(id_filename: tuple[str, str], gfs: gcsfs.GCSFileSystem) -> pd.DataFrame:
    """Read and process a single sounding.

    Parameters
    ----------
    id_filename : tuple[str, str]
        Sounding ID and GCS netCDF path.

    gfs: GCSFileSystem
        GCS storage connection.

    Returns
    -------
    pd.DataFrame
        Processed sounding

    """
    sounding_id, rpath = id_filename
    
    with tempfile.TemporaryDirectory() as tmpdir:
        lpath = f"{tmpdir}/profile.nc"
        gfs.get(rpath, lpath)
        ds = xr.open_dataset(lpath)
        try:
            df = process_waypoints(ds)
        finally:
            ds.close()

    return df.assign(sounding_id=sounding_id)


def process_gruan(gfs: gcsfs.GCSFileSystem) -> pd.DataFrame:
    """Download and process GRUAN data from GCS netCDF files.

    Parameters
    ----------
    gfs: GCSFileSystem
        GCS storage connection

    Returns
    -------
    pd.DataFrame
        Processed sounding

    """
    filenames = gfs.glob("gs://contrails-301217-contrail-bench/v2/obs-ingest/gruan/RS41-GDP.1/*/2025/*")
    logger.info(f"Found {len(filenames)} netCDF files")

    # A single sounding can have multiple revisions in separate netCDF files
    # For each, keep only the latest revision
    sounding_ids: dict[str, tuple[int, str]] = {}  # sounding_id -> (revision, bytes)

    for f in filenames:
        assert f.endswith(".nc")
        sounding_id, parts = f.removesuffix(".nc").rsplit("_", 1)
        revision = int(parts.split("-")[1])  # parts is something like 1-000-001
        if sounding_id not in sounding_ids or revision > sounding_ids[sounding_id][0]:
            sounding_ids[sounding_id] = revision, f
            
    logger.info(f"Mirror contains {len(sounding_ids)} unique sounding IDs")
    id_filename_pairs = [(s, f) for s, (_, f) in sounding_ids.items()]
    
    # Need to set max_tasks_per_child to force starting of workers with `spawn` instead of `fork`
    # Using `spawn` results in gRPC errors
    with ProcessPoolExecutor(max_workers=NUM_WORKERS, max_tasks_per_child=999999) as pool:
        df_list = list(tqdm(
            pool.map(read_and_process, id_filename_pairs, itertools.repeat(gfs)),
            total=len(id_filename_pairs)
        ))
    df = pd.concat(df_list)

    # There could be multiple weather balloons in the same cell, so aggregate again
    # It's rare, but the same site can launch two balloons within an hour of one another
    return df.groupby(["time_hour", "lat_bin", "lon_bin", "fl_bin"]).sum()


def write_results(df: pd.DataFrame, gfs: gcsfs.GCSFileSystem) -> None:
    """Write preprocessed observations to GCS.

    Parameters
    ----------
    df : pd.DataFrame
        Binned PCR and total observation counts

    gfs : gcsfs.GCSFileSystem
        GCS storage connection

    """
    # Slightly reformat for IAGOS compatibility
    df = (
        df.reset_index(level=["lat_bin", "lon_bin"])
        .rename(columns={"lat_bin": "latitude", "lon_bin": "longitude"})
        .drop(columns="sounding_id")
    )

    # Option 2: Write each time-FL combination to a separate parquet for IAGOS compatibility
    grouped = df.groupby(level=["time_hour", "fl_bin"])
    with tempfile.TemporaryDirectory() as tmpdir:
        lpaths = []
        rpaths = []

        for time, flight_level in itertools.product(TIMES, FLIGHT_LEVELS):
            shard = shard_id(time, flight_level)
            lpath = f"{tmpdir}/{shard}.pq"
            rpath = f"{GCP_GRUAN_SINK}/{shard}.pq"
            
            try:
                group = grouped.get_group((time, flight_level))
            except KeyError:
                group = pd.DataFrame(columns=["longitude", "latitude", "pcr_count", "total_count"])    

            group.to_parquet(lpath, index=False)
            lpaths.append(lpath)
            rpaths.append(rpath)

        # Faster to upload the files concurrently
        gfs.put(lpaths, rpaths)


def main() -> None:
    """Program entrypoint."""
    gfs = gcsfs.GCSFileSystem()
    df = process_gruan(gfs)
    write_results(df, gfs)


if __name__ == "__main__":
    main()
