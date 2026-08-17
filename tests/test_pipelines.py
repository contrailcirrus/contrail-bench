"""Regression tests for the materialize/evaluate CLI pipelines."""

import sys

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from contrailbench.pipelines import evaluate, materialize


class TestStoreDir:
    def test_conus_uses_bare_source_name(self):
        assert materialize.store_dir("metoffice", "conus").endswith("/metoffice")

    def test_non_conus_region_gets_suffix(self):
        assert materialize.store_dir("metoffice", "shanwick").endswith("/metoffice-shanwick")

    def test_fixed_lead_gets_suffix(self):
        assert materialize.store_dir("metoffice", "conus", 24).endswith("/metoffice-lead024")

    def test_region_and_lead_suffixes_compose(self):
        assert materialize.store_dir("metoffice", "shanwick", 48).endswith("/metoffice-shanwick-lead048")


class TestOutputDirs:
    def test_single_metric_conus(self):
        outputs, intermediates = evaluate.output_dirs("metoffice", ["iagos"], "conus")
        assert outputs.endswith("/metoffice-iagos-conus")
        assert intermediates.endswith("/_tmp/metoffice-iagos-conus")

    def test_multiple_metrics_joined(self):
        outputs, _ = evaluate.output_dirs("metoffice", ["iagos", "adsb"], "conus")
        assert outputs.endswith("/metoffice-iagos-adsb-conus")

    def test_lead_hours_suffix(self):
        outputs, _ = evaluate.output_dirs("metoffice", ["iagos"], "conus", lead_hours=72)
        assert outputs.endswith("/metoffice-iagos-conus-lead072")


class _FakeSource:
    """Stand-in for a real forecast source's module, avoiding any network/mirror
    dependency in materialize.py's own tests."""

    @staticmethod
    def fetch(time, flight_levels, *, extent, **kwargs):
        longitude = np.array([-10.0, 0.0, 10.0])
        latitude = np.array([-10.0, 0.0, 10.0])
        shape = (len(longitude), len(latitude), len(flight_levels))
        rhi = np.full(shape, 1.1, dtype="float32")
        sac = np.ones(shape, dtype="float32")
        return xr.Dataset(
            {
                "rhi": (("longitude", "latitude", "flight_level"), rhi),
                "sac": (("longitude", "latitude", "flight_level"), sac),
                "pcr": (("longitude", "latitude", "flight_level"), (rhi > 1.0).astype("float32")),
            },
            coords={
                "longitude": longitude,
                "latitude": latitude,
                "flight_level": list(flight_levels),
                "level": ("flight_level", [300.0] * len(flight_levels)),
            },
        )


def test_preprocess_forecast_writes_one_file_per_flight_level(tmp_path, monkeypatch):
    monkeypatch.setitem(materialize.SOURCES, "fake", _FakeSource)
    time = pd.Timestamp("2024-09-01T00:00").to_pydatetime()

    materialize.preprocess_forecast(
        time, [310, 320], "fake", extent=(-20, 20, -20, 20), local_dir=str(tmp_path)
    )

    from contrailbench import time_utils

    ts = time_utils.to_utc_timestamp(time)
    assert (tmp_path / f"{ts}_310.nc").exists()
    assert (tmp_path / f"{ts}_320.nc").exists()

    ds = xr.open_dataset(tmp_path / f"{ts}_310.nc")
    assert "level" in ds.dims
    assert ds.sizes["level"] == 1


def test_preprocess_forecast_passes_lead_hours_and_mirror_dir(tmp_path, monkeypatch):
    captured = {}

    class _CapturingSource:
        @staticmethod
        def fetch(time, flight_levels, *, extent, **kwargs):
            captured.update(kwargs)
            return _FakeSource.fetch(time, flight_levels, extent=extent)

    monkeypatch.setitem(materialize.SOURCES, "fake", _CapturingSource)
    time = pd.Timestamp("2024-09-01T00:00").to_pydatetime()

    materialize.preprocess_forecast(
        time,
        [310],
        "fake",
        extent=(-20, 20, -20, 20),
        local_dir=str(tmp_path),
        lead_hours=24,
        mirror_dir="/some/mirror",
    )

    assert captured == {"lead_hours": 24, "mirror_dir": "/some/mirror"}


def test_evaluate_cli_end_to_end_against_real_metoffice_store(tmp_path, monkeypatch):
    """Exercises evaluate.py's full argparse + wiring + Forecast.evaluate_beam
    path against real on-disk data (the -prefix-buggy stores, DST-clean post
    2024-10-27), without needing network access or a real forecast fetch."""
    data_root = "/home/jg931/contrails_org/contrail-bench/reports/jay_extension/data"
    forecast_store = f"{data_root}/metoffice-prefix-buggy"
    obs_store = f"{data_root}/_obs_cache/iagos-prefix-buggy"

    import os

    if not (os.path.isdir(forecast_store) and os.path.isdir(obs_store)):
        pytest.skip("local data mirror not present in this environment")

    outputs_dir = str(tmp_path / "outputs")
    intermediates_dir = str(tmp_path / "intermediates")

    monkeypatch.setattr(evaluate, "store_dir", lambda source, region, lead_hours: forecast_store)
    monkeypatch.setattr(
        evaluate, "output_dirs", lambda *a, **kw: (outputs_dir, intermediates_dir)
    )
    monkeypatch.setitem(
        evaluate.METRICS,
        "iagos",
        (evaluate.HitRate, evaluate.IAGOSDataloader, obs_store),
    )

    argv = [
        "evaluate.py",
        "--source", "metoffice",
        "--region", "conus",
        "--metrics", "iagos",
        "--runner", "direct",
        "--start", "2024-10-31T00:00",
        "--end", "2024-10-31T01:00",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    evaluate.main()

    output_file = f"{outputs_dir}/20241031.nc"
    assert os.path.exists(output_file)
    result = xr.open_dataset(output_file)
    assert "iagos.observed_pcr_area" in result.data_vars
    assert result.sizes["flight_level"] == 14
    assert result.sizes["rhi_threshold"] == 13
