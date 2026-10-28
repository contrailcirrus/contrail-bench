"""Shared constants."""

import datetime

import numpy as np
import pandas as pd

#: Target times
TIMES = pd.date_range(
    "2025-01-01 00:00",
    "2025-12-31 23:00",
    freq="1h",
).to_pydatetime().tolist()

#: Target flight levels
FLIGHT_LEVELS = list(range(270, 450, 10))

#: Vertical resolution (ft) used for bucketing evaluation data
VERTICAL_RESOLUTION = 250.0

#: Target longitudes
LONGITUDE = np.linspace(-180.0, 179.75, 1440)  # 0.25 degrees

#: Target latitudes
LATITUDE = np.linspace(-90.0, 90.0, 721)  # 0.25 degrees
    
#: Longitude bounds
LONGITUDE_BNDS = np.linspace(-180.125, 179.875, 1441)

#: Latitude bounds
LATITUDE_BNDS = np.linspace(-90.125, 90.125, 722).clip(-90.0, 90.0)

#: Engine efficiency for SAC calculations
ENGINE_EFFICIENCY = 0.3

#: GCP path for ADSB evaluation dataset
GCP_ADSB_SINK = "gs://contrails-301217-contrail-bench/v2/evaluation-datasets/adsb"

#: GCP path for IAGOS evaluation dataset
GCP_IAGOS_SINK = "gs://contrails-301217-contrail-bench/v2/evaluation-datasets/iagos"

#: GCP path for GRUAN evaluation dataset
GCP_GRUAN_SINK = "gs://contrails-301217-contrail-bench/v2/evaluation-datasets/gruan"

#: GCP path for ContrailWatch evaluation dataset
GCP_CONTRAILWATCH_SINK = "gs://contrails-301217-contrail-bench/v2/evaluation-datasets/contrailwatch"

#: GCP path for Imperial attributions evaluation dataset
GCP_IMPERIAL_ATT_SINK = "gs://contrails-301217-contrail-bench/v2/evaluation-datasets/imperial-attributions"


def shard_id(time: datetime.datetime, flight_level: int) -> str:
    """Compute name of data shard. Assumes timezone-naive datetimes are UTC."""
    if time.tzinfo is None:
        time = time.replace(tzinfo=datetime.UTC)
    return f"{int(time.timestamp())}_{flight_level}"

