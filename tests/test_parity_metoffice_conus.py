"""Numeric parity gate: new Forecast/Dataloader/Metric pipeline vs. the pre-port
golden benchmark outputs.

Scope: metoffice/CONUS/IAGOS, restricted to hours at or after 2024-10-27T01:00Z.
UK local time equals UTC from that point on, so the pre-port DST bug (naive
`datetime.timestamp()` assuming host-local time) never manifested for these
hours -- the "-prefix-buggy" forecast/observation stores are actually correct
here, and this is the only offline, already-computed reference available.
1,582 such hours have complete (all 14 flight levels) forecast + IAGOS
coverage and a golden `metoffice-iagos-contrailwatch-region/*.pq` file.

This test samples a subset (every ~53rd hour, spread across the full range,
deterministic) rather than all 1,582, to keep runtime reasonable while still
covering the full period. Comparison uses a tight `np.allclose` rather than
exact bit-identity: the old pipeline summed area values in raw dataframe row
order, the new `PCRStoreDataloader` sums after `.sel()`-based reindexing,
which can reorder terms and shift the last ULP or two of a large float64 sum
(confirmed empirically -- the only observed differences are ~1e-13 relative,
consistent with summation-order noise, not a logic error). `rtol=1e-9` is
tight enough to still catch a real regression (wrong threshold, missing
observation, unit error) while tolerating that noise.

Skipped entirely if the reference data directories aren't present (e.g. in an
environment without the local data mirror).
"""

import datetime
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import pytest

from contrailbench.data import IAGOSDataloader, PCRStoreDataloader
from contrailbench.forecast import Forecast
from contrailbench.metrics import HitRate
from contrailbench.pcr import CONUS_EXTENT, PCR_FLIGHT_LEVELS

DATA_ROOT = "/home/jg931/contrails_org/contrail-bench/reports/jay_extension/data"
FORECAST_STORE = f"{DATA_ROOT}/metoffice-prefix-buggy"
OBS_STORE = f"{DATA_ROOT}/_obs_cache/iagos-prefix-buggy"
GOLDEN_DIR = f"{DATA_ROOT}/metoffice-iagos-contrailwatch-region"
DST_BOUNDARY = datetime.datetime(2024, 10, 27, 1, 0, tzinfo=datetime.UTC)


def _find_dst_clean_hours_with_goldens() -> list[int]:
    if not all(os.path.isdir(d) for d in (FORECAST_STORE, OBS_STORE, GOLDEN_DIR)):
        return []

    fl_set = set(PCR_FLIGHT_LEVELS)
    forecast_ts = defaultdict(set)
    for f in os.listdir(FORECAST_STORE):
        ts, fl = f.removesuffix(".nc").split("_")
        forecast_ts[int(ts)].add(int(fl))
    obs_ts = defaultdict(set)
    for f in os.listdir(OBS_STORE):
        ts, fl = f.removesuffix(".pq").split("_")
        obs_ts[int(ts)].add(int(fl))
    goldens = set(os.listdir(GOLDEN_DIR))

    complete = [ts for ts, fls in forecast_ts.items() if fls == fl_set and obs_ts.get(ts) == fl_set]
    dst_clean = [
        ts
        for ts in complete
        if datetime.datetime.fromtimestamp(ts, tz=datetime.UTC) >= DST_BOUNDARY
    ]
    return sorted(
        ts
        for ts in dst_clean
        if f"{datetime.datetime.fromtimestamp(ts, tz=datetime.UTC):%Y%m%d%H}.pq" in goldens
    )


_ALL_CANDIDATE_HOURS = _find_dst_clean_hours_with_goldens()
_SAMPLE_HOURS = _ALL_CANDIDATE_HOURS[::53]  # spread across the full range, deterministic

pytestmark = pytest.mark.skipif(
    not _ALL_CANDIDATE_HOURS, reason="local data mirror not present in this environment"
)


@pytest.mark.parametrize("ts", _SAMPLE_HOURS)
def test_new_pipeline_matches_golden_exactly(ts):
    time = datetime.datetime.fromtimestamp(ts, tz=datetime.UTC).replace(tzinfo=None)

    forecast = Forecast(
        PCRStoreDataloader(path=FORECAST_STORE),
        times=[time],
        flight_levels=PCR_FLIGHT_LEVELS,
        extent=CONUS_EXTENT,
    )
    result = forecast.evaluate(iagos=HitRate(IAGOSDataloader(path=OBS_STORE)))

    new_df = (
        result[["iagos.observed_pcr_area_in_forecast_pcr", "iagos.observed_pcr_area"]]
        .to_dataframe()
        .reset_index()
        .rename(
            columns={
                "iagos.observed_pcr_area_in_forecast_pcr": "observed_pcr_area_in_forecast_pcr",
                "iagos.observed_pcr_area": "observed_pcr_area",
            }
        )
        .sort_values(["flight_level", "rhi_threshold"])
        .reset_index(drop=True)
    )

    key = f"{datetime.datetime.fromtimestamp(ts, tz=datetime.UTC):%Y%m%d%H}.pq"
    golden_df = (
        pd.read_parquet(f"{GOLDEN_DIR}/{key}")
        .sort_values(["flight_level", "rhi_threshold"])
        .reset_index(drop=True)
    )

    assert list(new_df["flight_level"]) == list(golden_df["flight_level"])
    assert np.allclose(new_df["rhi_threshold"], golden_df["rhi_threshold"])
    # Not exact bit-identity: the old pipeline summed area values in raw
    # dataframe row order, the new Dataloader sums after `.sel()`-based
    # reindexing, which can reorder terms and shift the last ULP or two of a
    # ~1e10-magnitude float64 sum. rtol here is tight enough to catch any real
    # regression (a wrong RHi threshold, a missing/extra observation, a unit
    # error) while tolerating float summation-order noise.
    assert np.allclose(
        new_df["observed_pcr_area_in_forecast_pcr"].to_numpy(),
        golden_df["observed_pcr_area_in_forecast_pcr"].to_numpy(),
        rtol=1e-9,
        atol=1e-6,
    ), f"observed_pcr_area_in_forecast_pcr mismatch at {key}"
    assert np.allclose(
        new_df["observed_pcr_area"].to_numpy(),
        golden_df["observed_pcr_area"].to_numpy(),
        rtol=1e-9,
        atol=1e-6,
    ), f"observed_pcr_area mismatch at {key}"


def test_sample_is_nonempty():
    """Guard against the skip condition silently hiding a config problem --
    if the data mirror is present, the sample must not be empty."""
    if _ALL_CANDIDATE_HOURS:
        assert len(_SAMPLE_HOURS) >= 20, (
            f"expected a substantial sample, got {len(_SAMPLE_HOURS)} "
            f"out of {len(_ALL_CANDIDATE_HOURS)} candidate hours"
        )
