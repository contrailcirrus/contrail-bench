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

#: GCP path for ContrailWatch data
GCP_CONTRAILWATCH_SINK = "gs://contrails-301217-contrail-bench/v2/obs-ingest/contrailwatch"

def shard_id(time: datetime.datetime, flight_level: int) -> str:
    """Compute name of data shard. Assumes timezone-naive datetimes are UTC."""
    if time.tzinfo is None:
        time = time.replace(tzinfo=datetime.UTC)
    return f"{int(time.timestamp())}_{flight_level}"

