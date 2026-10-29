"""Which forecast serves a valid time at a given lead, and how to record it in a shard.

Forecasts are initialised at 00, 06, 12 and 18 UTC, so a valid time has forecasts at leads
``shortest_lead``, ``shortest_lead + 6``, ``shortest_lead + 12`` and so on, where the shortest lead
is 1 to 6 hours (lead 0, the analysis step, is not used). A pipeline writes one shard per valid
hour, flight level and lead, each from the initialisation ``lead`` hours before the valid time.
Each shard records its own initialisation.
"""

import datetime
from collections.abc import Iterable

import numpy as np
import xarray as xr

from src import config

#: Hours between initialisations
CYCLE_HOURS = 6


def parse_offsets(value: str) -> tuple[int, ...]:
    """Lead offsets from ``"all"`` or text such as ``"0,12,24,36"``.

    ``"all"`` is :data:`config.LEAD_OFFSETS`.

    Raises
    ------
    ValueError
        If an offset is not a non-negative multiple of :data:`CYCLE_HOURS`.
    """
    if value.strip().lower() == "all":
        return config.LEAD_OFFSETS
    offsets = tuple(sorted({int(part) for part in value.split(",")}))
    if not offsets or any(o < 0 or o % CYCLE_HOURS for o in offsets):
        msg = f"offsets must be non-negative multiples of {CYCLE_HOURS}: {value!r}"
        raise ValueError(msg)
    return offsets


def shortest_lead(valid_time: datetime.datetime) -> int:
    """The shortest lead in hours (1 to 6) of any forecast for ``valid_time``.

    For example 06:00 and 00:00 are served at lead 6 by the 00Z and 18Z forecasts, and 07:00 at
    lead 1 by the 06Z forecast.
    """
    return (valid_time.hour - 1) % CYCLE_HOURS + 1


def leads(valid_time: datetime.datetime, offsets: Iterable[int]) -> list[int]:
    """The leads in hours for ``valid_time``: its shortest lead plus each offset."""
    first = shortest_lead(valid_time)
    return [first + offset for offset in offsets]


def init_for(valid_time: datetime.datetime, lead: int) -> datetime.datetime:
    """The initialisation ``lead`` hours before ``valid_time``.

    Raises
    ------
    ValueError
        If that is not a 00, 06, 12 or 18 UTC initialisation.
    """
    init = valid_time - datetime.timedelta(hours=lead)
    if init.hour % CYCLE_HOURS or init.minute or init.second or init.microsecond:
        msg = f"no forecast is initialised at {init}, {lead} h before {valid_time}"
        raise ValueError(msg)
    return init


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
