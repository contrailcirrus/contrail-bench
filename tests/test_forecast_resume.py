"""Regression tests for Forecast.evaluate_beam's resumability."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from apache_beam.options.pipeline_options import PipelineOptions

from contrailbench.data import IAGOSDataloader, PCRStoreDataloader
from contrailbench.forecast import Forecast, _shard_signature, pending_output_days
from contrailbench.metrics import HitRate


def _direct_options() -> PipelineOptions:
    return PipelineOptions(runner="direct", direct_num_workers=1)


@pytest.fixture()
def tiny_store(tmp_path):
    """A minimal 2-time x 2-flight-level forecast + observation store."""
    longitude = np.array([-10.0, -5.0, 0.0, 5.0])
    latitude = np.array([-10.0, -5.0, 0.0, 5.0])
    times = [pd.Timestamp("2024-09-01T00:00"), pd.Timestamp("2024-09-01T01:00")]
    flight_levels = [310, 320]

    forecast_dir = tmp_path / "forecast"
    forecast_dir.mkdir()
    obs_dir = tmp_path / "obs"
    obs_dir.mkdir()

    for time in times:
        for fl in flight_levels:
            rhi = np.full((len(longitude), len(latitude), 1, 1), 1.1, dtype="float32")
            sac = np.ones((len(longitude), len(latitude), 1, 1), dtype="float32")
            ds = xr.Dataset(
                {
                    "rhi": (("longitude", "latitude", "level", "time"), rhi),
                    "sac": (("longitude", "latitude", "level", "time"), sac),
                    "pcr": (
                        ("longitude", "latitude", "level", "time"),
                        (rhi > 1.0).astype("float32"),
                    ),
                },
                coords={
                    "longitude": longitude,
                    "latitude": latitude,
                    "level": [287.0],
                    "time": [time],
                },
            )
            ts = int(time.timestamp())
            ds.to_netcdf(forecast_dir / f"{ts}_{fl}.nc")

            obs_df = pd.DataFrame(
                {"longitude": [0.0, 5.0], "latitude": [0.0, 5.0], "pcr_distance": [1.0, 1.0]}
            )
            obs_df.to_parquet(obs_dir / f"{ts}_{fl}.pq")

    return {
        "forecast_dir": str(forecast_dir),
        "obs_dir": str(obs_dir),
        "times": times,
        "flight_levels": flight_levels,
    }


def _build_forecast(tiny_store) -> Forecast:
    return Forecast(
        PCRStoreDataloader(path=tiny_store["forecast_dir"]),
        times=tiny_store["times"],
        flight_levels=tiny_store["flight_levels"],
        extent=None,
    )


class TestShardSignature:
    def test_stable_for_identical_configuration(self, tiny_store):
        forecast = _build_forecast(tiny_store)
        metrics = {"iagos": HitRate(IAGOSDataloader(path=tiny_store["obs_dir"]))}
        sig1 = _shard_signature(forecast.dataloader, forecast.extent, metrics)
        sig2 = _shard_signature(forecast.dataloader, forecast.extent, metrics)
        assert sig1 == sig2

    def test_differs_when_metric_set_changes(self, tiny_store):
        forecast = _build_forecast(tiny_store)
        sig_one_metric = _shard_signature(
            forecast.dataloader,
            forecast.extent,
            {"iagos": HitRate(IAGOSDataloader(path=tiny_store["obs_dir"]))},
        )
        sig_two_metrics = _shard_signature(
            forecast.dataloader,
            forecast.extent,
            {
                "iagos": HitRate(IAGOSDataloader(path=tiny_store["obs_dir"])),
                "iagos2": HitRate(IAGOSDataloader(path="/some/other/path")),
            },
        )
        assert sig_one_metric != sig_two_metrics

    def test_differs_when_extent_changes(self, tiny_store):
        forecast = _build_forecast(tiny_store)
        metrics = {"iagos": HitRate(IAGOSDataloader(path=tiny_store["obs_dir"]))}
        sig_no_extent = _shard_signature(forecast.dataloader, None, metrics)
        sig_with_extent = _shard_signature(forecast.dataloader, (-10, 10, -10, 10), metrics)
        assert sig_no_extent != sig_with_extent


class TestPendingOutputDays:
    def test_all_days_pending_when_no_outputs_exist(self, tmp_path):
        times = [pd.Timestamp("2024-09-01T00:00"), pd.Timestamp("2024-09-02T00:00")]
        result = pending_output_days(times, str(tmp_path / "outputs"))
        assert result == times

    def test_day_excluded_once_its_output_exists(self, tmp_path):
        outputs = tmp_path / "outputs"
        outputs.mkdir()
        (outputs / "20240901.nc").write_bytes(b"fake")

        times = [
            pd.Timestamp("2024-09-01T00:00"),
            pd.Timestamp("2024-09-01T01:00"),
            pd.Timestamp("2024-09-02T00:00"),
        ]
        result = pending_output_days(times, str(outputs))
        assert result == [pd.Timestamp("2024-09-02T00:00")]

    def test_resume_false_returns_everything_unchanged(self, tmp_path):
        outputs = tmp_path / "outputs"
        outputs.mkdir()
        (outputs / "20240901.nc").write_bytes(b"fake")
        times = [pd.Timestamp("2024-09-01T00:00")]
        assert pending_output_days(times, str(outputs), resume=False) == times


class TestEvaluateBeamResume:
    def test_second_run_recomputes_nothing(self, tiny_store, tmp_path, monkeypatch):
        outputs = str(tmp_path / "outputs")
        intermediates = str(tmp_path / "intermediates")
        forecast = _build_forecast(tiny_store)
        metrics = {"iagos": HitRate(IAGOSDataloader(path=tiny_store["obs_dir"]))}

        forecast.evaluate_beam(outputs, intermediates, _direct_options(), **metrics)

        call_count = 0
        original = Forecast._evaluate_shard

        def counting_evaluate_shard(self, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original(self, *args, **kwargs)

        monkeypatch.setattr(Forecast, "_evaluate_shard", counting_evaluate_shard)

        forecast2 = _build_forecast(tiny_store)
        forecast2.evaluate_beam(outputs, intermediates, _direct_options(), **metrics)

        assert call_count == 0, "second run with unchanged config should compute no shards"

    def test_mismatched_signature_forces_recompute(self, tiny_store, tmp_path, monkeypatch):
        outputs = str(tmp_path / "outputs")
        intermediates = str(tmp_path / "intermediates")
        forecast = _build_forecast(tiny_store)

        forecast.evaluate_beam(
            outputs,
            intermediates,
            _direct_options(),
            iagos=HitRate(IAGOSDataloader(path=tiny_store["obs_dir"])),
        )

        call_count = 0
        original = Forecast._evaluate_shard

        def counting_evaluate_shard(self, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original(self, *args, **kwargs)

        monkeypatch.setattr(Forecast, "_evaluate_shard", counting_evaluate_shard)

        # Different metric set (renamed key) -> different signature -> must recompute,
        # even though the outputs directory already has files from the run above.
        # Point outputs elsewhere since day-output existence would otherwise skip
        # the whole day regardless of shard-level signature.
        outputs2 = str(tmp_path / "outputs2")
        forecast2 = _build_forecast(tiny_store)
        forecast2.evaluate_beam(
            outputs2,
            intermediates,
            _direct_options(),
            iagos_v2=HitRate(IAGOSDataloader(path=tiny_store["obs_dir"])),
        )

        expected_shards = len(tiny_store["times"]) * len(tiny_store["flight_levels"])
        assert call_count == expected_shards, "changed metric set must invalidate cached shards"

    def test_resume_false_forces_full_recompute(self, tiny_store, tmp_path, monkeypatch):
        outputs = str(tmp_path / "outputs")
        intermediates = str(tmp_path / "intermediates")
        forecast = _build_forecast(tiny_store)
        metrics = {"iagos": HitRate(IAGOSDataloader(path=tiny_store["obs_dir"]))}

        forecast.evaluate_beam(outputs, intermediates, _direct_options(), **metrics)

        call_count = 0
        original = Forecast._evaluate_shard

        def counting_evaluate_shard(self, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original(self, *args, **kwargs)

        monkeypatch.setattr(Forecast, "_evaluate_shard", counting_evaluate_shard)

        forecast2 = _build_forecast(tiny_store)
        forecast2.evaluate_beam(outputs, intermediates, _direct_options(), resume=False, **metrics)

        expected_shards = len(tiny_store["times"]) * len(tiny_store["flight_levels"])
        assert call_count == expected_shards

    def test_corrupted_intermediate_is_recomputed_not_trusted(self, tiny_store, tmp_path):
        outputs = str(tmp_path / "outputs")
        intermediates = str(tmp_path / "intermediates")
        forecast = _build_forecast(tiny_store)
        metrics = {"iagos": HitRate(IAGOSDataloader(path=tiny_store["obs_dir"]))}

        forecast.evaluate_beam(outputs, intermediates, _direct_options(), **metrics)

        # Corrupt one intermediate shard.
        import os

        shard_files = [f for f in os.listdir(intermediates) if f.endswith(".nc")]
        assert shard_files
        with open(os.path.join(intermediates, shard_files[0]), "wb") as f:
            f.write(b"not a valid netcdf file")

        # Re-running (against a fresh outputs dir, so day-level skip doesn't
        # short-circuit before the per-shard check even runs) must not raise --
        # the corrupted shard is detected and recomputed, not trusted.
        outputs2 = str(tmp_path / "outputs2")
        forecast2 = _build_forecast(tiny_store)
        forecast2.evaluate_beam(outputs2, intermediates, _direct_options(), **metrics)

        with open(os.path.join(intermediates, shard_files[0]), "rb") as f:
            assert f.read(3) != b"not", "corrupted shard should have been overwritten"
