"""Which forecast serves a valid time at a given lead, and how to record it in a shard.

Forecasts are initialised at 00, 06, 12 and 18 UTC, so a valid time has forecasts at leads
``shortest_lead``, ``shortest_lead + 6``, ``shortest_lead + 12`` and so on, where the shortest lead
is 0 to 5 hours (0 when the valid time is an initialisation time). The forecast "no shorter than"
``L`` hours is the one at lead ``L`` to ``L + 5``, which is ``shortest_lead + L`` for a multiple of
6. A pipeline writes one shard per valid hour, flight level and lead, each from the initialisation
``lead`` hours before the valid time. Each shard records its own initialisation.
"""

import datetime
from collections.abc import Iterable

import numpy as np
import xarray as xr

#: Hours between initialisations
CYCLE_HOURS = 6


def parse_offsets(value: str) -> tuple[int, ...]:
    """Lead offsets from text such as ``"0,12,24,36"``.

    Raises
    ------
    ValueError
        If an offset is not a non-negative multiple of :data:`CYCLE_HOURS`.
    """
    offsets = tuple(sorted({int(part) for part in value.split(",")}))
    if not offsets or any(o < 0 or o % CYCLE_HOURS for o in offsets):
        msg = f"offsets must be non-negative multiples of {CYCLE_HOURS}: {value!r}"
        raise ValueError(msg)
    return offsets


def shortest_lead(valid_time: datetime.datetime) -> int:
    """The shortest lead in hours (0 to 5) of any forecast for ``valid_time``.

    For example 06:00 and 00:00 are served at lead 0 by the 06Z and 00Z forecasts, and 07:00 at
    lead 1 by the 06Z forecast.
    """
    return valid_time.hour % CYCLE_HOURS


def leads(
    valid_time: datetime.datetime, offsets: Iterable[int], *, has_analysis: bool = True
) -> list[int]:
    """The leads in hours for ``valid_time``: its shortest lead plus each offset.

    A source without the analysis step (lead 0) has no forecast at lead 0, so for the valid times
    that fall on an initialisation time a lead of 0 becomes the next one, :data:`CYCLE_HOURS`
    (``has_analysis=False``). A lead that this repeats is listed once.
    """
    first = shortest_lead(valid_time)
    found = [first + offset for offset in offsets]
    if not has_analysis:
        found = [lead or CYCLE_HOURS for lead in found]
    return list(dict.fromkeys(found))


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
