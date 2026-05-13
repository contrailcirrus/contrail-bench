"""Process 2024 GRUAN RS41 data for Contrail Bench ingestion.

GRUAN documentation:
https://www.gruan.org/gruan/editor/documents/gruan/GRUAN-TN-13_RS41-GDP_UserGuide_v1.0_2022-11-21.pdf

Paul's GRUAN download code:
https://github.com/contrailcirrus/sonde-analysis/blob/main/gruan/gruan_process_raw.py

pycontrails documentation:
https://py.contrails.org/notebooks/GRUAN.html
"""

import io
import itertools
import logging
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor

import gcsfs
import pandas as pd
import xarray as xr
from pycontrails import JetA
from pycontrails.models import pcr as pcr_mod
from pycontrails.physics import constants, thermo, units

logging.basicConfig(
    level="WARNING",
    format="[%(asctime)s.%(msecs)03d] [%(levelname)s] [%(module)s] [%(funcName)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel("INFO")


# Pipeline parameters

#: GCS blobs with 2024 RS41 profiles
GCS_PATTERN = "gs://contrails-301217-radiosonde-data/downloads/gruan/gruan_RS41*_2024_*.zip"

#: Target times
TIMES = pd.date_range("2024-01-01 00:00", "2024-12-31 23:00", freq="1h").to_pydatetime().tolist()

#: Target flight levels
FLIGHT_LEVELS = list(range(270, 450, 10))

#: Vertical resolution (ft) used for binning observations
VERTICAL_RESOLUTION = 250.0

#: Engine efficiency for SAC calculation
ENGINE_EFFICIENCY = 0.3

#: GCP bucket for temporary assets
GCP_TMPDIR = "gs://contrails-301217-contrail-bench/tmp/2026Q1/gruan"


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
    fl = units.pl_to_ft(df["press"]) / 100.0
    fl_bin = (fl / 10.0).round() * 10.0
    cond = (
        (fl >= FLIGHT_LEVELS[0])
        & (fl <= FLIGHT_LEVELS[-1])
        & ((fl - fl_bin).abs() * 100 <= VERTICAL_RESOLUTION)
    )

    df = df.assign(fl_bin=fl_bin)[cond].assign(
        lat_bin=lambda x: (x["lat"] / 0.25).round() * 0.25,
        lon_bin=lambda x: (x["lon"] / 0.25).round() * 0.25,
        time_hour=lambda x: x["time"].dt.round("h"),
    )

    T = df["temp"] + df["temp_corr"]
    rh = df["rh"] + df["rh_corr"]
    if rh_unit == "percent":
        rh = rh / 100.0  # percent -> fraction

    p = df["press"] * 100.0  # hPa -> Pa
    q = rh * constants.epsilon * thermo.e_sat_liquid(T) / p

    df["pcr"] = pcr_mod.pcr(T, q, p, ENGINE_EFFICIENCY, ei_h2o=JetA.ei_h2o, q_fuel=JetA.q_fuel)[0] == 1.0
    grouped = df.groupby(["time_hour", "lat_bin", "lon_bin", "fl_bin"])["pcr"]
    return grouped.agg(pcr_count="sum", total_count="count")


def read_and_process(id_bytes: tuple[str, bytes]) -> pd.DataFrame:
    """Read and process a single sounding.

    Parameters
    ----------
    id_bytes : tuple[str, bytes]
        Sounding ID and NetCDF contents.

    Returns
    -------
    pd.DataFrame
        Processed sounding

    """
    sounding_id, nc_bytes = id_bytes
    try:
        ds = xr.open_dataset(nc_bytes)
    except ValueError:
        logger.debug("Falling back to scipy engine")
        ds = xr.open_dataset(io.BytesIO(nc_bytes), engine="scipy")
    try:
        df = process_waypoints(ds)
    finally:
        ds.close()
    return df.assign(sounding_id=sounding_id)


def process_gruan_2024(gfs: gcsfs.GCSFileSystem) -> pd.DataFrame:
    """Download and process GRUAN 2024 data from GCS zip files."""
    all_dfs = []

    files = gfs.glob(GCS_PATTERN)
    logger.info(f"Found {len(files)} zip files")

    for zip_path in files:
        logger.info(f"Processing {zip_path}")
        zip_bytes = io.BytesIO(gfs.cat(zip_path))

        # A single sounding can have multiple revisions in the zip file
        # For each, keep only the latest revision
        sounding_ids: dict[str, tuple[int, bytes]] = {}  # sounding_id -> (revision, bytes)

        with zipfile.ZipFile(zip_bytes, "r") as zf:
            filenames = zf.namelist()
            for f in filenames:
                assert f.endswith(".nc")
                sounding_id, parts = f.removesuffix(".nc").rsplit("_", 1)
                revision = int(parts.split("-")[1])  # parts is something like 1-000-001
                if sounding_id not in sounding_ids or revision > sounding_ids[sounding_id][0]:
                    sounding_ids[sounding_id] = revision, zf.read(f)

            logger.info(f"Zip contains {len(sounding_ids)} unique sounding IDs")
            id_byte_pairs = [(s, b) for s, (_, b) in sounding_ids.items()]

        # Process each NetCDF file in parallel with a process pool
        # ThreadPoolExecutor doesn't provide much speedup
        with ProcessPoolExecutor(max_workers=8) as executor:
            dfs = list(executor.map(read_and_process, id_byte_pairs))

        df = pd.concat(dfs)
        all_dfs.append(df)

    df = pd.concat(all_dfs)

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
            lpath = f"{tmpdir}/{int(time.timestamp())}_{flight_level}.pq"
            rpath = f"{GCP_TMPDIR}/{int(time.timestamp())}_{flight_level}.pq"
            
            try:
                group = grouped.get_group((time, flight_level))
            except KeyError:
                group = pd.DataFrame(columns=["longitude", "latitude", "pcr_count", "total_count"])    

            group.to_parquet(lpath, index=False)
            lpaths.append(lpath)
            rpaths.append(rpath)

        # Faster to upload the files concurrently
        gfs.put(lpaths, rpaths)


if __name__ == "__main__":
    gfs = gcsfs.GCSFileSystem()
    df = process_gruan_2024(gfs)
    write_results(df, gfs)
