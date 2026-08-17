"""Tests for the mirror.

Monkeypatches ``s3.filesystem``/``s3.fetch_pressure_level_field``/
``s3.open_pressure_level_field`` to synthetic in-memory data so these tests never
touch the network -- ``test_s3.py`` already covers the real S3 fetch path.
``_fetch_and_write_hour`` runs inside a ``ProcessPoolExecutor`` worker; on Linux the
default ``fork`` start method duplicates the parent's (already-patched) module state
at pool-creation time, so the patches apply inside worker processes too.
"""

import datetime
import json

import numpy as np
import pytest
import xarray as xr

from contrailbench.datalib.metoffice import mirror, s3

LATITUDE = np.array([30.0, 31.0])
LONGITUDE = np.array([-100.0, -99.0])
PRESSURE_PA = np.asarray(s3.CRUISE_LEVELS_HPA, dtype=np.float64) * 100.0


def _fake_field(parameter: str, validity: datetime.datetime | None) -> xr.DataArray:
    """A small, deterministic field keyed only by validity hour and parameter.

    ``validity`` is ``None`` for :func:`mirror._fetch_conus_coords`'s coordinate-only
    probe (it calls ``open_pressure_level_field`` without a validity kwarg) -- the
    actual values don't matter there, only the coordinates.
    """
    variable = s3.PARAMETER_VARIABLE[parameter]
    hour = 0.0 if validity is None else float(validity.hour)
    marker = hour + (100.0 if variable == "relative_humidity" else 0.0)
    data = np.full((len(PRESSURE_PA), len(LATITUDE), len(LONGITUDE)), marker, dtype=np.float32)
    return xr.DataArray(
        data,
        dims=("pressure", "latitude", "longitude"),
        coords={"pressure": PRESSURE_PA, "latitude": LATITUDE, "longitude": LONGITUDE},
    )


class _FakeClient:
    """Stand-in for a boto3 client -- never actually called since the fetch
    functions themselves are replaced below."""


@pytest.fixture(autouse=True)
def _patch_s3(monkeypatch):
    monkeypatch.setattr(s3, "filesystem", _FakeClient)

    def fake_open_field(fs, key, parameter, *, run=None, validity=None, lead_hours=None, extent=None):
        return _fake_field(parameter, validity)

    def fake_fetch_field(fs, key, parameter, *, run=None, validity=None, lead_hours=None, extent=None):
        return _fake_field(parameter, validity)

    monkeypatch.setattr(s3, "open_pressure_level_field", fake_open_field)
    monkeypatch.setattr(s3, "fetch_pressure_level_field", fake_fetch_field)


def _manifest(path) -> dict:
    return json.loads(path.read_text())


def test_mirror_fixed_lead_creates_sparse_store_and_manifest(tmp_path):
    start = datetime.datetime(2024, 9, 1, 0)
    end = datetime.datetime(2024, 9, 2, 23)

    mirror.mirror_fixed_lead(
        start, end, lead_hours=24, lead_family=(0, 24), out_dir=tmp_path, workers=2
    )

    store_path = tmp_path / "lead024.zarr"
    manifest_path = tmp_path / "lead024.manifest.json"
    assert store_path.exists()
    assert manifest_path.exists()

    manifest = _manifest(manifest_path)
    # Two days x 4 validity times/day (hour in {0,6,12,18}) at lead 24, per the
    # fixed-lead cadence arithmetic -- see s3.available_validity_times_at_lead.
    assert len(manifest) == 2 * 4
    assert all(entry["status"] == "ok" for entry in manifest.values())
    assert all(entry["lead_hours"] == 24 for entry in manifest.values())

    ds = xr.open_zarr(store_path)
    assert ds.sizes["time"] == 2 * 4
    assert not bool(np.isnan(ds["air_temperature"].values).any())


def test_mirror_fixed_lead_manifest_records_correct_run(tmp_path):
    start = datetime.datetime(2024, 9, 1, 0)
    end = datetime.datetime(2024, 9, 1, 23)

    mirror.mirror_fixed_lead(
        start, end, lead_hours=24, lead_family=(0, 24), out_dir=tmp_path, workers=1
    )

    manifest = _manifest(tmp_path / "lead024.manifest.json")
    validity = datetime.datetime(2024, 9, 1, 12).isoformat()
    assert manifest[validity]["run"] == datetime.datetime(2024, 8, 31, 12).isoformat()


def test_mirror_fixed_lead_is_resumable(tmp_path):
    start = datetime.datetime(2024, 9, 1, 0)
    end = datetime.datetime(2024, 9, 1, 23)

    mirror.mirror_fixed_lead(
        start, end, lead_hours=24, lead_family=(0, 24), out_dir=tmp_path, workers=1
    )
    manifest_path = tmp_path / "lead024.manifest.json"
    first_manifest = _manifest(manifest_path)
    assert len(first_manifest) == 4

    # Mark one entry as failed, simulating an interrupted prior run, then re-run.
    one_key = next(iter(first_manifest))
    first_manifest[one_key] = {"status": "error", "error": "simulated"}
    manifest_path.write_text(json.dumps(first_manifest))

    mirror.mirror_fixed_lead(
        start, end, lead_hours=24, lead_family=(0, 24), out_dir=tmp_path, workers=1
    )

    second_manifest = _manifest(manifest_path)
    assert all(entry["status"] == "ok" for entry in second_manifest.values())


def test_mirror_fixed_lead_store_naming_does_not_collide_with_month_store(tmp_path):
    (tmp_path / "2024-09.zarr").mkdir()
    (tmp_path / "2024-09.manifest.json").write_text("{}")

    start = datetime.datetime(2024, 9, 1, 0)
    end = datetime.datetime(2024, 9, 1, 23)
    mirror.mirror_fixed_lead(
        start, end, lead_hours=24, lead_family=(0, 24), out_dir=tmp_path, workers=1
    )

    assert (tmp_path / "lead024.zarr").exists()
    assert (tmp_path / "2024-09.zarr").exists()  # untouched


def test_mirror_defaults_to_conus_region_in_manifest(tmp_path):
    start = datetime.datetime(2024, 9, 1, 0)
    end = datetime.datetime(2024, 9, 1, 2)

    mirror.mirror(start, end, tmp_path, workers=1)

    manifest = _manifest(tmp_path / "2024-09.manifest.json")
    assert all(entry["region"] == "conus" for entry in manifest.values())


def test_mirror_records_shanwick_region_in_manifest(tmp_path):
    start = datetime.datetime(2024, 9, 1, 0)
    end = datetime.datetime(2024, 9, 1, 2)

    mirror.mirror(start, end, tmp_path, workers=1, region="shanwick")

    manifest = _manifest(tmp_path / "2024-09.manifest.json")
    assert all(entry["region"] == "shanwick" for entry in manifest.values())


def test_mirror_rejects_non_conus_region_at_default_out_dir():
    start = datetime.datetime(2024, 9, 1, 0)
    end = datetime.datetime(2024, 9, 1, 2)

    with pytest.raises(ValueError, match="out-dir"):
        mirror.mirror(start, end, mirror.DEFAULT_OUT_DIR, workers=1, region="shanwick")


def test_mirror_fixed_lead_records_shanwick_region_in_manifest(tmp_path):
    start = datetime.datetime(2024, 9, 1, 0)
    end = datetime.datetime(2024, 9, 1, 23)

    mirror.mirror_fixed_lead(
        start,
        end,
        lead_hours=24,
        lead_family=(0, 24),
        out_dir=tmp_path,
        workers=1,
        region="shanwick",
    )

    manifest = _manifest(tmp_path / "lead024.manifest.json")
    assert all(entry["region"] == "shanwick" for entry in manifest.values())


def test_mirror_hours_filter_restricts_times_to_process(tmp_path):
    start = datetime.datetime(2024, 9, 1, 0)
    end = datetime.datetime(2024, 9, 2, 23)

    mirror.mirror(start, end, tmp_path, workers=1, hours=range(15, 23))

    manifest = _manifest(tmp_path / "2024-09.manifest.json")
    captured_hours = {datetime.datetime.fromisoformat(k).hour for k in manifest}
    assert captured_hours == set(range(15, 23))
