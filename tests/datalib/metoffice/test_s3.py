"""Tests for the Met Office S3 access layer.

``test_range_read_bit_identical_to_reference`` touches the network: on first run it
downloads a full reference object from the public, anonymous-access
``met-office-atmospheric-model-data`` bucket into a local cache
(``tests/fixtures/metoffice/.cache/``, gitignored), then range-reads the same object
and checks the two match. Skipped, not failed, if the bucket is unreachable.
"""

import datetime
import io
import os
import pathlib

import numpy as np
import pytest
import xarray as xr
from botocore.exceptions import BotoCoreError, ClientError

from contrailbench.datalib.metoffice import s3

FIXTURE_DIR = pathlib.Path(__file__).parents[2] / "fixtures" / "metoffice"
CACHE_DIR = FIXTURE_DIR / ".cache"

#: The run/validity/lead fetched as the reference object for the bit-identity check
REFERENCE_RUN = datetime.datetime(2026, 8, 3, 0, 0)
REFERENCE_VALIDITY = datetime.datetime(2026, 8, 3, 0, 0)
REFERENCE_LEAD_HOURS = 0


def _cached_reference_file(parameter: str) -> pathlib.Path:
    """Download (and locally cache) the full reference object for ``parameter``.

    Skips the test rather than failing it if the bucket can't be reached.
    """
    key = s3.object_key(REFERENCE_RUN, REFERENCE_VALIDITY, REFERENCE_LEAD_HOURS, parameter)
    cache_path = CACHE_DIR / pathlib.PurePosixPath(key).name
    if cache_path.exists():
        return cache_path

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        body = s3.filesystem().get_object(Bucket=s3.BUCKET, Key=key)["Body"].read()
    except (BotoCoreError, ClientError) as exc:
        pytest.skip(f"could not reach s3://{s3.BUCKET} to fetch reference object: {exc}")
    cache_path.write_bytes(body)
    return cache_path


class _CountingClient:
    """Wraps a boto3 S3 client, counting bytes actually returned by ``get_object``."""

    def __init__(self, client) -> None:
        self._client = client
        self.bytes_read = 0

    def get_object(self, **kwargs):
        response = self._client.get_object(**kwargs)
        body = response["Body"].read()
        self.bytes_read += len(body)
        response["Body"] = io.BytesIO(body)
        return response

    def __getattr__(self, name):
        return getattr(self._client, name)


def test_run_and_lead_for_validity_within_cycle():
    run, lead_hours = s3.run_and_lead_for_validity(datetime.datetime(2024, 9, 15, 3))
    assert run == datetime.datetime(2024, 9, 15, 0)
    assert lead_hours == 3


def test_run_and_lead_for_validity_on_cycle_boundary():
    run, lead_hours = s3.run_and_lead_for_validity(datetime.datetime(2024, 9, 15, 18))
    assert run == datetime.datetime(2024, 9, 15, 18)
    assert lead_hours == 0


def test_run_and_lead_for_validity_rejects_non_hourly():
    with pytest.raises(ValueError, match="must fall on the hour"):
        s3.run_and_lead_for_validity(datetime.datetime(2024, 9, 15, 3, 30))


def test_run_hours_for_lead_all_cycles_at_and_below_boundary():
    assert s3.run_hours_for_lead(54) == (0, 6, 12, 18)


def test_run_hours_for_lead_long_cycles_only_above_boundary():
    assert s3.run_hours_for_lead(55) == (0, 12)


def test_run_for_validity_at_lead_matches_shortest_lead_at_lead_zero():
    validity = datetime.datetime(2024, 9, 15, 18)
    expected_run, _ = s3.run_and_lead_for_validity(validity)
    assert s3.run_for_validity_at_lead(validity, 0) == expected_run


def test_run_for_validity_at_lead_solves_run_minus_lead():
    run = s3.run_for_validity_at_lead(datetime.datetime(2024, 9, 15, 12), 24)
    assert run == datetime.datetime(2024, 9, 14, 12)


def test_run_for_validity_at_lead_rejects_unreachable_run_hour():
    # validity 2024-09-15T06 at lead 72 implies run 2024-09-12T06, but only 00/12Z
    # runs reach lead 72.
    with pytest.raises(ValueError, match="does not reach that lead"):
        s3.run_for_validity_at_lead(datetime.datetime(2024, 9, 15, 6), 72)


def test_run_for_validity_at_lead_rejects_non_hourly():
    with pytest.raises(ValueError, match="must fall on the hour"):
        s3.run_for_validity_at_lead(datetime.datetime(2024, 9, 15, 3, 30), 24)


def test_available_validity_times_at_lead_counts_per_day():
    start = datetime.datetime(2024, 9, 1, 0)
    end = datetime.datetime(2024, 9, 7, 23)

    at_24 = s3.available_validity_times_at_lead(start, end, 24)
    at_72 = s3.available_validity_times_at_lead(start, end, 72)

    assert len(at_24) == 7 * 4
    assert len(at_72) == 7 * 2
    assert all(v.hour in (0, 6, 12, 18) for v in at_24)
    assert all(v.hour in (0, 12) for v in at_72)


def test_matched_validity_times_reduces_to_hour_0_and_12():
    start = datetime.datetime(2024, 9, 1, 0)
    end = datetime.datetime(2024, 9, 7, 23)

    matched = s3.matched_validity_times(start, end, (0, 24, 48, 72))

    assert len(matched) == 7 * 2
    assert all(v.hour in (0, 12) for v in matched)


def test_matched_validity_times_empty_leads():
    start = datetime.datetime(2024, 9, 1)
    end = datetime.datetime(2024, 9, 2)
    assert s3.matched_validity_times(start, end, ()) == []


def test_object_key_matches_plan_example():
    run = datetime.datetime(2026, 8, 3, 0, 0)
    validity = datetime.datetime(2026, 8, 3, 0, 0)
    key = s3.object_key(run, validity, 0, "temperature_on_pressure_levels")
    assert key == (
        "global-deterministic-10km/20260803T0000Z/"
        "20260803T0000Z-PT0000H00M-temperature_on_pressure_levels.nc"
    )


def test_object_key_lead_and_minutes_padding():
    run = datetime.datetime(2026, 8, 3, 0, 0)
    validity = datetime.datetime(2026, 8, 4, 0, 0)
    key = s3.object_key(run, validity, 24, "temperature_on_pressure_levels")
    assert key == (
        "global-deterministic-10km/20260803T0000Z/"
        "20260804T0000Z-PT0024H00M-temperature_on_pressure_levels.nc"
    )


def test_select_cruise_subset_crops_to_passed_in_extent():
    pressure_pa = np.asarray(s3.CRUISE_LEVELS_HPA, dtype=np.float64) * 100.0
    latitude = np.linspace(30.0, 70.0, 41)  # ascending, 1 degree steps
    longitude = np.linspace(-40.0, 0.0, 41)  # 1 degree steps
    data = np.zeros((len(pressure_pa), len(latitude), len(longitude)), dtype=np.float32)
    ds = xr.Dataset(
        {"air_temperature": (("pressure", "latitude", "longitude"), data)},
        coords={"pressure": pressure_pa, "latitude": latitude, "longitude": longitude},
    )

    shanwick_extent = (-30.0, -10.0, 45.0, 61.0)
    da = s3.select_cruise_subset(
        ds, "temperature_on_pressure_levels", extent=shanwick_extent
    )

    assert da["longitude"].values.min() >= -30.0
    assert da["longitude"].values.max() <= -10.0
    assert da["latitude"].values.min() >= 45.0
    assert da["latitude"].values.max() <= 61.0


def test_select_cruise_subset_defaults_to_conus_extent():
    pressure_pa = np.asarray(s3.CRUISE_LEVELS_HPA, dtype=np.float64) * 100.0
    latitude = np.linspace(10.0, 60.0, 51)
    longitude = np.linspace(-140.0, -50.0, 91)
    data = np.zeros((len(pressure_pa), len(latitude), len(longitude)), dtype=np.float32)
    ds = xr.Dataset(
        {"air_temperature": (("pressure", "latitude", "longitude"), data)},
        coords={"pressure": pressure_pa, "latitude": latitude, "longitude": longitude},
    )

    da = s3.select_cruise_subset(ds, "temperature_on_pressure_levels")

    lon_min, lon_max, lat_min, lat_max = s3.CONUS_EXTENT
    assert da["longitude"].values.min() >= lon_min
    assert da["longitude"].values.max() <= lon_max
    assert da["latitude"].values.min() >= lat_min
    assert da["latitude"].values.max() <= lat_max


@pytest.mark.parametrize(
    "parameter", ["temperature_on_pressure_levels", "relative_humidity_on_pressure_levels"]
)
def test_range_read_bit_identical_to_reference(parameter):
    fixture_path = _cached_reference_file(parameter)

    local_ds = xr.open_dataset(
        fixture_path, engine="h5netcdf", decode_times=True, decode_timedelta=True
    )
    expected = s3.select_cruise_subset(local_ds, parameter, key=str(fixture_path)).load()

    key = s3.object_key(REFERENCE_RUN, REFERENCE_VALIDITY, REFERENCE_LEAD_HOURS, parameter)
    counting_client = _CountingClient(s3.filesystem())
    actual = s3.fetch_pressure_level_field(
        counting_client,
        key,
        parameter,
        run=REFERENCE_RUN,
        validity=REFERENCE_VALIDITY,
        lead_hours=REFERENCE_LEAD_HOURS,
    )

    np.testing.assert_array_equal(actual.values, expected.values)
    assert list(actual.dims) == ["pressure", "latitude", "longitude"]
    assert actual.sizes["pressure"] == len(s3.CRUISE_LEVELS_HPA)

    whole_file_size = os.path.getsize(fixture_path)
    ratio = counting_client.bytes_read / whole_file_size
    print(
        f"\n[{parameter}] range-read bytes={counting_client.bytes_read} "
        f"whole-file bytes={whole_file_size} ratio={ratio:.4f}"
    )
    assert counting_client.bytes_read < whole_file_size
