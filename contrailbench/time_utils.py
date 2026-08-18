"""Shared UTC-safe timestamp conversion.

Every naive ``datetime``/``pandas.Timestamp`` in this codebase is UTC-intended by
convention (forecast validity times, observation windows, etc.), but
``datetime.timestamp()`` on a naive value assumes *host-local* wall-clock time and
applies that zone's historical DST rule -- silently wrong on any non-UTC host.
:func:`to_utc_timestamp` is the single choke point every call site should convert
through instead of calling ``.timestamp()`` directly.
"""

from __future__ import annotations

import datetime


def to_utc_timestamp(dt: datetime.datetime) -> int:
    """UTC-safe epoch conversion for the naive datetimes used throughout this codebase.

    A naive ``dt`` is treated as UTC rather than rejected, matching the convention
    every caller already relies on. An aware ``dt`` is converted respecting its own
    ``tzinfo`` instead. Works for both plain ``datetime.datetime`` and
    ``pandas.Timestamp``.

    Parameters
    ----------
    dt : datetime.datetime
        Naive (UTC-intended) or aware datetime/``pandas.Timestamp``.

    Returns
    -------
    int
        Unix epoch seconds.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.UTC)
    return int(dt.timestamp())
