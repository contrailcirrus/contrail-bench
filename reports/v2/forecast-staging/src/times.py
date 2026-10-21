"""The valid times a pipeline covers.

Naive datetimes and timestamps are UTC by convention everywhere.
"""

import datetime
from collections.abc import Collection

import pandas as pd


def utc_epoch(time: datetime.datetime) -> int:
    """Unix seconds for ``time``, treating a naive value as UTC.

    ``datetime.timestamp()`` would read a naive value as host-local time, which is wrong on any
    non-UTC machine. An aware value keeps its own timezone. Works for ``pandas.Timestamp`` too.
    """
    if time.tzinfo is None:
        time = time.replace(tzinfo=datetime.UTC)
    return int(time.timestamp())


def parse_hours(text: str) -> tuple[int, ...]:
    """Hours of day from text such as ``"15,16,22"`` or ``"15-22"`` (or a mix); each in 0 to 23."""
    hours: list[int] = []
    for part in text.split(","):
        low, _, high = part.strip().partition("-")
        hours.extend(range(int(low), int(high or low) + 1))
    if not hours or any(not 0 <= h <= 23 for h in hours):
        msg = f"hours must be between 0 and 23: {text!r}"
        raise ValueError(msg)
    return tuple(sorted(set(hours)))


def valid_times(
    start: pd.Timestamp,
    end: pd.Timestamp,
    hours: Collection[int] | None = None,
) -> list[datetime.datetime]:
    """Hourly valid times (UTC) from ``start`` to ``end`` inclusive, within the given ``hours``."""
    times = pd.date_range(start, end, freq="1h").to_pydatetime().tolist()
    if hours is not None:
        keep = set(hours)
        times = [t for t in times if t.hour in keep]
    return times
