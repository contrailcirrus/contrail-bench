"""Tests for the Met Office pycontrails datalib.

Builds tiny synthetic monthly Zarr stores + manifests directly (matching
``mirror.py``'s schema at a small 2x2 lat/lon, 2-3 hour scale) rather than reusing
``mirror.py``'s private helpers, so these tests don't couple to its internals.
"""

import datetime
import json
import pathlib

import numpy as np
import pytest
import xarray as xr
from pycontrails.physics import thermo

from contrailbench.datalib.metoffice import s3
from contrailbench.datalib.metoffice.ukmo import (
    DATASET,
    PROVIDER,
    MetOfficeMirrorNotFoundError,
    MetOfficeUM,
    suppress_unregistered_source_warnings,
)

LATITUDE = np.array([30.0, 31.0])
LONGITUDE = np.array([-100.0, -99.0])
PRESSURE_PA = np.asarray(s3.CRUISE_LEVELS_HPA, dtype=np.float64) * 100.0


def _write_store(
    tmp_path: pathlib.Path,
    year: int,
    month: int,
    hours: list[datetime.datetime],
    *,
    air_temperature: np.ndarray | None = None,
    relative_humidity: np.ndarray | None = None,
    latitude: np.ndarray = LATITUDE,
    pressure_pa: np.ndarray = PRESSURE_PA,
    variables: tuple[str, ...] = ("air_temperature", "relative_humidity"),
) -> pathlib.Path:
    """Write a small synthetic monthly Zarr store matching mirror.py's schema."""
    shape = (len(hours), len(pressure_pa), len(latitude), len(LONGITUDE))
    if air_temperature is None:
        air_temperature = np.full(shape, 220.0, dtype=np.float32)
    if relative_humidity is None:
        relative_humidity = np.full(shape, 0.8, dtype=np.float32)

    data = {}
    if "air_temperature" in variables:
        data["air_temperature"] = (
            ("time", "pressure", "latitude", "longitude"),
            air_temperature,
        )
    if "relative_humidity" in variables:
        data["relative_humidity"] = (
            ("time", "pressure", "latitude", "longitude"),
            relative_humidity,
        )

    ds = xr.Dataset(
        data,
        coords={
            "time": hours,
            "pressure": pressure_pa,
            "latitude": latitude,
            "longitude": LONGITUDE,
        },
    )
    store_path = tmp_path / f"{year:04d}-{month:02d}.zarr"
    ds.to_zarr(store_path, mode="w")
    return store_path


def _write_manifest(
    tmp_path: pathlib.Path, year: int, month: int, entries: dict[datetime.datetime, dict]
) -> pathlib.Path:
    manifest_path = tmp_path / f"{year:04d}-{month:02d}.manifest.json"
    manifest_path.write_text(json.dumps({h.isoformat(): entry for h, entry in entries.items()}))
    return manifest_path


def _ok_entries(hours: list[datetime.datetime]) -> dict[datetime.datetime, dict]:
    return {h: {"run": h.isoformat(), "lead_hours": 0, "status": "ok"} for h in hours}


def _write_lead_store(
    tmp_path: pathlib.Path,
    lead_hours: int,
    hours: list[datetime.datetime],
    *,
    air_temperature: np.ndarray | None = None,
    relative_humidity: np.ndarray | None = None,
) -> pathlib.Path:
    """Write a small synthetic whole-window, fixed-lead Zarr store."""
    shape = (len(hours), len(PRESSURE_PA), len(LATITUDE), len(LONGITUDE))
    if air_temperature is None:
        air_temperature = np.full(shape, 220.0, dtype=np.float32)
    if relative_humidity is None:
        relative_humidity = np.full(shape, 0.8, dtype=np.float32)

    ds = xr.Dataset(
        {
            "air_temperature": (("time", "pressure", "latitude", "longitude"), air_temperature),
            "relative_humidity": (
                ("time", "pressure", "latitude", "longitude"),
                relative_humidity,
            ),
        },
        coords={
            "time": hours,
            "pressure": PRESSURE_PA,
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
        },
    )
    store_path = tmp_path / f"lead{lead_hours:03d}.zarr"
    ds.to_zarr(store_path, mode="w")
    return store_path


def _write_lead_manifest(
    tmp_path: pathlib.Path, lead_hours: int, entries: dict[datetime.datetime, dict]
) -> pathlib.Path:
    manifest_path = tmp_path / f"lead{lead_hours:03d}.manifest.json"
    manifest_path.write_text(json.dumps({h.isoformat(): entry for h, entry in entries.items()}))
    return manifest_path


def test_open_metdataset_selects_requested_hour(tmp_path):
    hours = [datetime.datetime(2024, 9, 1, h) for h in (0, 1, 2)]
    shape = (len(hours), len(PRESSURE_PA), len(LATITUDE), len(LONGITUDE))
    air_temperature = np.stack(
        [np.full(shape[1:], marker, dtype=np.float32) for marker in (200.0, 210.0, 220.0)]
    )
    _write_store(tmp_path, 2024, 9, hours, air_temperature=air_temperature)
    _write_manifest(tmp_path, 2024, 9, _ok_entries(hours))

    mds = MetOfficeUM(hours[1], paths=tmp_path).open_metdataset()

    assert mds.data.sizes["time"] == 1
    np.testing.assert_allclose(mds.data["air_temperature"].values, 210.0)


def test_open_metdataset_rejects_descending_latitude(tmp_path):
    hours = [datetime.datetime(2024, 9, 1, 0)]
    _write_store(tmp_path, 2024, 9, hours, latitude=np.array([31.0, 30.0]))
    _write_manifest(tmp_path, 2024, 9, _ok_entries(hours))

    with pytest.raises(AssertionError, match="latitude"):
        MetOfficeUM(hours[0], paths=tmp_path).open_metdataset()


def test_specific_humidity_round_trips_to_relative_humidity(tmp_path):
    hours = [datetime.datetime(2024, 9, 1, 0)]
    t_value, rh_value = 215.0, 0.7
    _write_store(
        tmp_path,
        2024,
        9,
        hours,
        air_temperature=np.full(
            (1, len(PRESSURE_PA), len(LATITUDE), len(LONGITUDE)), t_value, dtype=np.float32
        ),
        relative_humidity=np.full(
            (1, len(PRESSURE_PA), len(LATITUDE), len(LONGITUDE)), rh_value, dtype=np.float32
        ),
    )
    _write_manifest(tmp_path, 2024, 9, _ok_entries(hours))

    mds = MetOfficeUM(hours[0], paths=tmp_path).open_metdataset()

    # MetDataset sorts `level` ascending, so index 0 is 150 hPa, not
    # CRUISE_LEVELS_HPA[0] (300 hPa) -- read the actual level back rather than
    # assuming which physical level ends up at a given index.
    level_pa = mds.data["level"].isel(level=0).compute().item() * 100.0
    expected_q = rh_value * thermo.q_sat_liquid(np.float64(t_value), np.float64(level_pa))
    q = (
        mds.data["specific_humidity"]
        .isel(level=0, latitude=0, longitude=0, time=0)
        .compute()
        .item()
    )
    assert np.isclose(q, expected_q, rtol=1e-4)

    recovered_rh = q / thermo.q_sat_liquid(np.float64(t_value), np.float64(level_pa))
    assert np.isclose(recovered_rh, rh_value, rtol=1e-4)


def test_hand_computed_rhi_matches_thermo_rhi_via_q(tmp_path):
    hours = [datetime.datetime(2024, 9, 1, 0)]
    t_value, rh_value = 210.0, 0.9
    _write_store(
        tmp_path,
        2024,
        9,
        hours,
        air_temperature=np.full(
            (1, len(PRESSURE_PA), len(LATITUDE), len(LONGITUDE)), t_value, dtype=np.float32
        ),
        relative_humidity=np.full(
            (1, len(PRESSURE_PA), len(LATITUDE), len(LONGITUDE)), rh_value, dtype=np.float32
        ),
    )
    _write_manifest(tmp_path, 2024, 9, _ok_entries(hours))

    mds = MetOfficeUM(hours[0], paths=tmp_path).open_metdataset()

    level_pa = mds.data["level"].isel(level=0).compute().item() * 100.0
    q = (
        mds.data["specific_humidity"]
        .isel(level=0, latitude=0, longitude=0, time=0)
        .compute()
        .item()
    )

    rhi_expected = (
        rh_value * thermo.e_sat_liquid(np.float64(t_value)) / thermo.e_sat_ice(np.float64(t_value))
    )
    rhi_actual = thermo.rhi(np.float64(q), np.float64(t_value), np.float64(level_pa))
    assert np.isclose(rhi_actual, rhi_expected, rtol=1e-4)


def test_open_metdataset_raises_on_missing_variable(tmp_path):
    hours = [datetime.datetime(2024, 9, 1, 0)]
    _write_store(tmp_path, 2024, 9, hours, variables=("air_temperature",))
    _write_manifest(tmp_path, 2024, 9, _ok_entries(hours))

    with pytest.raises(KeyError, match="relative_humidity"):
        MetOfficeUM(hours[0], paths=tmp_path).open_metdataset()


def test_download_dataset_raises_on_manifest_error_status(tmp_path):
    hours = [datetime.datetime(2024, 9, 1, 0)]
    _write_store(tmp_path, 2024, 9, hours)
    _write_manifest(tmp_path, 2024, 9, {hours[0]: {"status": "error", "error": "boom"}})

    with pytest.raises(MetOfficeMirrorNotFoundError, match="status='error'"):
        MetOfficeUM(hours[0], paths=tmp_path).open_metdataset()


def test_download_dataset_raises_on_missing_manifest_entry(tmp_path):
    hours = [datetime.datetime(2024, 9, 1, 0), datetime.datetime(2024, 9, 1, 1)]
    _write_store(tmp_path, 2024, 9, hours)
    _write_manifest(tmp_path, 2024, 9, _ok_entries([hours[0]]))  # hours[1] absent

    with pytest.raises(MetOfficeMirrorNotFoundError, match="no manifest entry"):
        MetOfficeUM(hours[1], paths=tmp_path).open_metdataset()


def test_download_dataset_raises_on_missing_month(tmp_path):
    with pytest.raises(MetOfficeMirrorNotFoundError, match="No manifest"):
        MetOfficeUM(datetime.datetime(2024, 9, 1, 0), paths=tmp_path).open_metdataset()


def test_provider_dataset_attrs_warn_without_suppression(tmp_path):
    hours = [datetime.datetime(2024, 9, 1, 0)]
    _write_store(tmp_path, 2024, 9, hours)
    _write_manifest(tmp_path, 2024, 9, _ok_entries(hours))

    mds = MetOfficeUM(hours[0], paths=tmp_path).open_metdataset()

    assert mds.attrs["provider"] == PROVIDER
    assert mds.attrs["dataset"] == DATASET
    with pytest.warns(UserWarning, match="Unknown provider"):
        assert mds.provider_attr == PROVIDER
    with pytest.warns(UserWarning, match="Unknown dataset"):
        assert mds.dataset_attr == DATASET


def test_suppress_unregistered_source_warnings_silences_them(tmp_path):
    hours = [datetime.datetime(2024, 9, 1, 0)]
    _write_store(tmp_path, 2024, 9, hours)
    _write_manifest(tmp_path, 2024, 9, _ok_entries(hours))

    mds = MetOfficeUM(hours[0], paths=tmp_path).open_metdataset()

    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with suppress_unregistered_source_warnings():
            _ = mds.provider_attr
            _ = mds.dataset_attr
    assert not caught


def test_open_metdataset_spans_month_boundary(tmp_path):
    sep_hours = [datetime.datetime(2024, 9, 30, 23)]
    oct_hours = [datetime.datetime(2024, 10, 1, 0)]
    shape = (1, len(PRESSURE_PA), len(LATITUDE), len(LONGITUDE))
    _write_store(
        tmp_path,
        2024,
        9,
        sep_hours,
        air_temperature=np.full(shape, 201.0, dtype=np.float32),
    )
    _write_store(
        tmp_path,
        2024,
        10,
        oct_hours,
        air_temperature=np.full(shape, 202.0, dtype=np.float32),
    )
    _write_manifest(tmp_path, 2024, 9, _ok_entries(sep_hours))
    _write_manifest(tmp_path, 2024, 10, _ok_entries(oct_hours))

    mds = MetOfficeUM((sep_hours[0], oct_hours[0]), paths=tmp_path).open_metdataset()

    assert mds.data.sizes["time"] == 2
    sep_val = (
        mds.data.sel(time=sep_hours[0])["air_temperature"]
        .isel(level=0, latitude=0, longitude=0)
        .compute()
        .item()
    )
    oct_val = (
        mds.data.sel(time=oct_hours[0])["air_temperature"]
        .isel(level=0, latitude=0, longitude=0)
        .compute()
        .item()
    )
    assert np.isclose(sep_val, 201.0)
    assert np.isclose(oct_val, 202.0)


def test_open_metdataset_rejects_unmirrored_pressure_level(tmp_path):
    """A store missing one of the *requested but otherwise-supported* cruise levels
    (as opposed to a level outside ``s3.CRUISE_LEVELS_HPA`` entirely, which
    ``parse_pressure_levels`` already rejects at construction time) must fail loudly
    in ``_process``, not silently select the wrong level."""
    hours = [datetime.datetime(2024, 9, 1, 0)]
    levels_missing_300 = PRESSURE_PA[1:]  # drops 300 hPa, keeps the other 6
    _write_store(tmp_path, 2024, 9, hours, pressure_pa=levels_missing_300)
    _write_manifest(tmp_path, 2024, 9, _ok_entries(hours))

    with pytest.raises(ValueError, match="expected exactly one 300"):
        MetOfficeUM(hours[0], pressure_levels=[300], paths=tmp_path).open_metdataset()


def test_open_metdataset_rejects_pressure_level_outside_supported_set(tmp_path):
    hours = [datetime.datetime(2024, 9, 1, 0)]
    _write_store(tmp_path, 2024, 9, hours)
    _write_manifest(tmp_path, 2024, 9, _ok_entries(hours))

    with pytest.raises(ValueError, match="not supported"):
        MetOfficeUM(hours[0], pressure_levels=[100], paths=tmp_path)


# -- Fixed-lead store support -------------------------------------------------


def test_create_cachepath_fixed_lead(tmp_path):
    t = datetime.datetime(2024, 9, 1, 0)
    um = MetOfficeUM(t, paths=tmp_path, lead_hours=24)
    assert um.create_cachepath(t) == str(tmp_path / "lead024.zarr")


def test_create_cachepath_default_unaffected(tmp_path):
    t = datetime.datetime(2024, 9, 1, 0)
    um = MetOfficeUM(t, paths=tmp_path)
    assert um.create_cachepath(t) == str(tmp_path / "2024-09.zarr")


def test_open_metdataset_reads_fixed_lead_store(tmp_path):
    # Spans a month boundary -- a fixed-lead store has no month structure at all,
    # so this must NOT need the two-store concatenation the shortest-lead path does.
    hours = [datetime.datetime(2024, 9, 30, 12), datetime.datetime(2024, 10, 1, 12)]
    shape = (len(hours), len(PRESSURE_PA), len(LATITUDE), len(LONGITUDE))
    air_temperature = np.stack(
        [np.full(shape[1:], marker, dtype=np.float32) for marker in (231.0, 232.0)]
    )
    _write_lead_store(tmp_path, 24, hours, air_temperature=air_temperature)
    _write_lead_manifest(tmp_path, 24, _ok_entries(hours))

    mds = MetOfficeUM(hours[1], paths=tmp_path, lead_hours=24).open_metdataset()

    assert mds.data.sizes["time"] == 1
    np.testing.assert_allclose(mds.data["air_temperature"].values, 232.0)


def test_download_dataset_fixed_lead_manifest_not_found(tmp_path):
    t = datetime.datetime(2024, 9, 1, 0)
    with pytest.raises(MetOfficeMirrorNotFoundError, match="lead 24h"):
        MetOfficeUM(t, paths=tmp_path, lead_hours=24).open_metdataset()


def test_download_dataset_fixed_lead_error_status(tmp_path):
    hours = [datetime.datetime(2024, 9, 1, 0)]
    _write_lead_store(tmp_path, 24, hours)
    _write_lead_manifest(tmp_path, 24, {hours[0]: {"status": "error", "error": "boom"}})

    with pytest.raises(MetOfficeMirrorNotFoundError, match="status='error'"):
        MetOfficeUM(hours[0], paths=tmp_path, lead_hours=24).open_metdataset()


def test_fixed_lead_zero_reads_its_own_store_not_month_store(tmp_path):
    """lead_hours=0 must never be conflated with lead_hours=None (truthiness bug)."""
    hour = datetime.datetime(2024, 9, 1, 0)

    # Month store (shortest-lead / lead_hours=None) has one value...
    _write_store(
        tmp_path,
        2024,
        9,
        [hour],
        air_temperature=np.full(
            (1, len(PRESSURE_PA), len(LATITUDE), len(LONGITUDE)), 200.0, dtype=np.float32
        ),
    )
    _write_manifest(tmp_path, 2024, 9, _ok_entries([hour]))

    # ...and the fixed lead-0 store has a different one.
    _write_lead_store(
        tmp_path,
        0,
        [hour],
        air_temperature=np.full(
            (1, len(PRESSURE_PA), len(LATITUDE), len(LONGITUDE)), 999.0, dtype=np.float32
        ),
    )
    _write_lead_manifest(tmp_path, 0, _ok_entries([hour]))

    mds = MetOfficeUM(hour, paths=tmp_path, lead_hours=0).open_metdataset()
    np.testing.assert_allclose(mds.data["air_temperature"].values, 999.0)
