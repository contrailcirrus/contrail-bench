"""Shared UTC-safe timestamp conversion.

Every naive ``datetime``/``pandas.Timestamp`` in this codebase is UTC-intended by
convention (forecast validity times, observation windows, etc.). ``datetime.timestamp()``
on a naive value instead assumes *host-local* wall-clock time and applies the local
zone's historical DST rule for that date -- silently wrong on any non-UTC host. Found
2026-08-14 as the root cause of sink filenames encoding the wrong epoch on a UK-zoned
sandbox for any pre-2024-10-27 (BST) date. :func:`to_utc_timestamp` is the single choke
point every call site should convert through instead of calling ``.timestamp()`` directly.
"""

from __future__ import annotations

import datetime


def to_utc_timestamp(dt: datetime.datetime) -> int:
    """UTC-safe epoch conversion for the naive datetimes used throughout this codebase.

    A naive ``dt`` is treated as UTC, not rejected -- every existing caller already
    relies on "naive means UTC" as a convention (e.g. forecast validity times built via
    ``pd.date_range(...).to_pydatetime()``), so raising here would require localizing
    every one of them first, which is not the bug being fixed. An aware ``dt`` is
    converted respecting its own ``tzinfo`` instead, since it already unambiguously
    names an instant.

    Works for both plain ``datetime.datetime`` and ``pandas.Timestamp`` -- the latter is
    a ``datetime`` subclass and supports the same ``.replace(tzinfo=...)``/``.timestamp()``
    calls used here.

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
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return int(dt.timestamp())
