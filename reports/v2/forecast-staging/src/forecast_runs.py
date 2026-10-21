"""Which forecast run serves a valid time, and how to record it in a shard.

A forecast run is an initialisation at 00, 06, 12 or 18 UTC. A pipeline for run ``R`` writes one
shard per valid hour, each from the newest ``R`` initialisation at or before that hour, so every
run covers every hour with leads of 0 to 23 hours. Each shard records its own initialisation, and
the lead is worked out from that afterwards.
"""

import datetime
import re

import numpy as np
import xarray as xr

#: Initialisation times a run can have
RUNS = ("00Z", "06Z", "12Z", "18Z")


def parse_run(value: str) -> str:
    """Normalise a run given as ``00``, ``6``, ``00z`` or ``06Z`` to the :data:`RUNS` form.

    Raises
    ------
    ValueError
        If it is not one of the four runs.
    """
    match = re.fullmatch(r"(\d{1,2})[zZ]?", value.strip())
    if match:
        run = f"{int(match.group(1)):02d}Z"
        if run in RUNS:
            return run
    msg = f"run must be one of {RUNS}, not {value!r}"
    raise ValueError(msg)


def base_run(valid_time: datetime.datetime, run: str) -> datetime.datetime:
    """The newest ``run`` initialisation at or before ``valid_time``.

    ``run`` is in the form :func:`parse_run` returns, such as ``"06Z"``. For example the 06Z run
    serves 03:00 on 5 November from 06Z on 4 November (lead 21 h), and 07:00 on 5 November from
    06Z on 5 November (lead 1 h).
    """
    hour = int(run[:2])
    day = valid_time.replace(hour=hour, minute=0, second=0, microsecond=0)
    return day if day <= valid_time else day - datetime.timedelta(days=1)


def record_init(
    ds: xr.Dataset, init: datetime.datetime, valid_time: datetime.datetime
) -> xr.Dataset:
    """``ds`` with the forecast's initialisation recorded, so the lead can be derived later.

    Adds a scalar ``forecast_reference_time`` coordinate (the form downstream audits read) and the
    ``forecast_cycle``, ``forecast_lead_hours`` and ``forecast_valid_time`` attributes.
    """
    ds = ds.assign_coords(forecast_reference_time=np.datetime64(init, "ns"))
    ds.attrs.update(
        forecast_cycle=f"{init:%H}Z",
        forecast_lead_hours=int((valid_time - init).total_seconds() // 3600),
        forecast_valid_time=valid_time.isoformat(),
    )
    return ds
