"""Tests for the shared PCR derivation helper.

Builds tiny synthetic fields/``MetDataset``s directly, matching the convention in
``tests/datalib/metoffice/test_ukmo.py``, rather than depending on real mirrored or
ARCO data.
"""

import datetime
import itertools
import os
import warnings

import numpy as np
import pytest
import xarray as xr

from contrailbench import pcr
from pycontrails.core.met import MetDataset
from pycontrails.physics import thermo


def test_crop_benchmark_grid_slices_global_grid_exactly():
    longitude, latitude, longitude_bnds, latitude_bnds = pcr.crop_benchmark_grid(
        (-134, -63, 20, 50)
    )

    assert longitude[0] >= -134
    assert longitude[-1] <= -63
    assert latitude[0] >= 20
    assert latitude[-1] <= 50

    # Cropped coordinates must be bit-identical elements of the global grid -- not
    # independently reconstructed -- since downstream code does an exact `.sel()`.
    assert np.all(np.isin(longitude, pcr.BENCHMARK_LONGITUDE))
    assert np.all(np.isin(latitude, pcr.BENCHMARK_LATITUDE))

    assert len(longitude_bnds) == len(longitude) + 1
    assert len(latitude_bnds) == len(latitude) + 1


def test_cell_bounds_centered_on_uniform_spacing():
    centers = np.array([10.0, 10.25, 10.5])
    bnds = pcr.cell_bounds(centers, 0.25)

    np.testing.assert_allclose(bnds, [9.875, 10.125, 10.375, 10.625])


def test_regrid_identity_when_source_matches_target_grid():
    """Source grid == target grid should regrid to (numerically) itself."""
    longitude, latitude, _, _ = pcr.crop_benchmark_grid((-100, -99.75, 30, 30.25))
    field = xr.DataArray(
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        dims=("longitude", "latitude"),
        coords={"longitude": longitude, "latitude": latitude},
    )

    regridded = pcr.regrid_to_benchmark_grid(
        field,
        source_longitude=longitude,
        source_latitude=latitude,
        source_longitude_spacing=0.25,
        source_latitude_spacing=0.25,
        target_extent=(-100, -99.75, 30, 30.25),
    )

    np.testing.assert_allclose(regridded.transpose("longitude", "latitude").values, field.values)


def test_regrid_conserves_mass_for_finer_source_grid():
    """A uniform-valued finer source field regrids to the same uniform value."""
    target_extent = (-100.0, -99.0, 30.0, 31.0)
    # Native grid 5x finer than the 0.25-degree benchmark grid, fully covering the
    # target extent with margin so boundary under-coverage doesn't leak in.
    spacing = 0.05
    source_longitude = np.arange(-100.5, -98.5 + spacing / 2, spacing)
    source_latitude = np.arange(29.5, 31.5 + spacing / 2, spacing)

    field = xr.DataArray(
        np.full((len(source_longitude), len(source_latitude)), 7.0),
        dims=("longitude", "latitude"),
        coords={"longitude": source_longitude, "latitude": source_latitude},
    )

    regridded = pcr.regrid_to_benchmark_grid(
        field,
        source_longitude=source_longitude,
        source_latitude=source_latitude,
        source_longitude_spacing=spacing,
        source_latitude_spacing=spacing,
        target_extent=target_extent,
    )

    np.testing.assert_allclose(regridded.values, 7.0, rtol=1e-10)


def test_regrid_preserves_spatial_gradient_direction():
    """A source field increasing eastward should regrid to a coarser but still
    monotonically increasing field, not something scrambled by a bookkeeping bug."""
    target_extent = (-100.0, -99.0, 30.0, 31.0)
    spacing = 0.05
    source_longitude = np.arange(-100.5, -98.5 + spacing / 2, spacing)
    source_latitude = np.arange(29.5, 31.5 + spacing / 2, spacing)

    lon_grid, _ = np.meshgrid(source_longitude, source_latitude, indexing="ij")
    field = xr.DataArray(
        lon_grid,
        dims=("longitude", "latitude"),
        coords={"longitude": source_longitude, "latitude": source_latitude},
    )

    regridded = pcr.regrid_to_benchmark_grid(
        field,
        source_longitude=source_longitude,
        source_latitude=source_latitude,
        source_longitude_spacing=spacing,
        source_latitude_spacing=spacing,
        target_extent=target_extent,
    )

    mean_by_lon = regridded.mean(dim="latitude")
    assert np.all(np.diff(mean_by_lon.values) > 0)


def _synthetic_met(
    air_temperature: float, specific_humidity: float, level_hpa: float
) -> MetDataset:
    longitude = np.array([-100.0, -99.75])
    latitude = np.array([30.0, 30.25])
    level = np.array([level_hpa])
    time = np.array([np.datetime64("2024-09-01T00:00")])

    ds = xr.Dataset(
        {
            "air_temperature": (
                ("longitude", "latitude", "level", "time"),
                np.full((2, 2, 1, 1), air_temperature),
            ),
            "specific_humidity": (
                ("longitude", "latitude", "level", "time"),
                np.full((2, 2, 1, 1), specific_humidity),
            ),
        },
        coords={"longitude": longitude, "latitude": latitude, "level": level, "time": time},
    )
    met = MetDataset(ds)
    met.attrs.update(
        provider="Met Office", dataset="UM-Global-Deterministic-10km", product="forecast"
    )
    return met


def test_compute_rhi_and_sac_matches_hand_computation():
    air_temperature = 215.0
    specific_humidity = 3e-4
    level_hpa = 225.0
    met = _synthetic_met(air_temperature, specific_humidity, level_hpa)

    rhi, sac_field = pcr.compute_rhi_and_sac(met)

    expected_rhi = thermo.rhi(specific_humidity, air_temperature, level_hpa * 100.0)
    np.testing.assert_allclose(rhi.values, expected_rhi, rtol=1e-6)

    assert set(np.unique(sac_field.values)) <= {0.0, 1.0}


def test_compute_rhi_and_sac_emits_no_unsuppressed_warning():
    met = _synthetic_met(215.0, 3e-4, 225.0)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pcr.compute_rhi_and_sac(met)

    assert not caught, [str(w.message) for w in caught]


def _synthetic_native_ds(
    levels: list[tuple[float, float, float]],
) -> tuple[xr.Dataset, np.ndarray, np.ndarray]:
    """Build a synthetic multi-level, native-grid (pre-regrid) ``air_temperature``/
    ``specific_humidity`` dataset, one distinguishable ``(T, q)`` pair per level.

    Parameters
    ----------
    levels : list[tuple[float, float, float]]
        ``(air_temperature, specific_humidity, level_hpa)`` per level, in the given
        order -- this is the order :func:`pcr.regrid_and_derive` must not silently
        scramble relative to its ``flight_levels`` argument.
    """
    longitude = np.array([-100.0, -99.75])
    latitude = np.array([30.0, 30.25])
    level = np.array([lvl for _, _, lvl in levels])
    time = np.array([np.datetime64("2024-09-01T00:00")])
    n = len(levels)

    def _field(values: list[float]) -> np.ndarray:
        return np.asarray(values).reshape(1, 1, n, 1) * np.ones((2, 2, n, 1))

    ds = xr.Dataset(
        {
            "air_temperature": (
                ("longitude", "latitude", "level", "time"),
                _field([t for t, _, _ in levels]),
            ),
            "specific_humidity": (
                ("longitude", "latitude", "level", "time"),
                _field([q for _, q, _ in levels]),
            ),
        },
        coords={"longitude": longitude, "latitude": latitude, "level": level, "time": time},
    )
    ds.attrs.update(
        provider="Met Office", dataset="UM-Global-Deterministic-10km", product="forecast"
    )
    return ds, longitude, latitude


#: Matches the synthetic grid's own extent exactly (source == target grid), the
#: same convention `test_regrid_identity_when_source_matches_target_grid` uses --
#: keeps every source cell in coverage so no cell falls outside the regrid target
#: and comes back NaN, which would mask real bugs behind coincidentally-matching NaNs.
_NATIVE_TARGET_EXTENT = (-100.0, -99.75, 30.0, 30.25)


def _regrid_and_derive_native(ds: xr.Dataset, flight_levels: list[int], longitude, latitude):
    return pcr.regrid_and_derive(
        ds,
        flight_levels=flight_levels,
        source_longitude=longitude,
        source_latitude=latitude,
        source_longitude_spacing=0.25,
        source_latitude_spacing=0.25,
        target_extent=_NATIVE_TARGET_EXTENT,
    )


def test_pcr_flight_levels_have_no_pressure_rounding_collisions():
    """`ERA5ARCO`'s `parse_pressure_levels` hard-fails on duplicate target pressures
    -- assert there aren't any at the real flight-level spacing, rather than assume."""
    pressures = [pcr.target_pressure_hpa(fl) for fl in pcr.PCR_FLIGHT_LEVELS]
    assert len(set(pressures)) == len(pcr.PCR_FLIGHT_LEVELS)


def test_extract_flight_level_reconstructs_single_level_schema():
    flight_levels = [310, 320]
    pressures = [pcr.target_pressure_hpa(fl) for fl in flight_levels]
    levels = [(215.0, 3e-4, pressures[0]), (210.0, 2e-4, pressures[1])]
    ds, longitude, latitude = _synthetic_native_ds(levels)

    batched = _regrid_and_derive_native(ds, flight_levels, longitude, latitude)
    extracted = pcr.extract_flight_level(batched, flight_levels[1])

    assert "flight_level" not in extracted.dims
    assert "level" in extracted.dims
    assert extracted.sizes["level"] == 1
    assert float(extracted["level"].item()) == pressures[1]


def test_interpolate_to_pressures_recovers_exact_input_at_native_levels():
    """Interpolating onto the same pressures the data is already defined at must
    return the input unchanged -- a log-linear interpolant passes through its own
    knots exactly."""
    ds, _, _ = _synthetic_native_ds([(215.0, 3e-4, 250.0), (205.0, 2e-4, 200.0)])

    interpolated = pcr.interpolate_to_pressures(ds, [250.0, 200.0])

    np.testing.assert_allclose(
        interpolated["air_temperature"].values, ds["air_temperature"].values
    )
    np.testing.assert_allclose(
        interpolated["specific_humidity"].values, ds["specific_humidity"].values
    )
    np.testing.assert_array_equal(interpolated["level"].values, [250.0, 200.0])


def test_interpolate_to_pressures_is_monotone_between_bracketing_levels():
    """A target pressure strictly between two native levels must interpolate to a
    value strictly between their two field values -- catches a sign/log error that
    otherwise wouldn't show up at the exact-knot case above."""
    ds, _, _ = _synthetic_native_ds([(210.0, 2e-4, 300.0), (230.0, 4e-4, 200.0)])

    interpolated = pcr.interpolate_to_pressures(ds, [250.0])

    value = float(interpolated["air_temperature"].isel(level=0, longitude=0, latitude=0, time=0))
    assert 210.0 < value < 230.0


def test_regrid_and_derive_matches_independent_per_level_calls():
    """Batched (one met fetch, many flight levels) must match calling the same
    derivation independently per flight level, bit-for-bit -- not just close."""
    flight_levels = [310, 320, 330]
    pressures = [pcr.target_pressure_hpa(fl) for fl in flight_levels]
    levels = [(215.0 + i, 3e-4 + i * 1e-5, p) for i, p in enumerate(pressures)]
    ds, longitude, latitude = _synthetic_native_ds(levels)

    batched = _regrid_and_derive_native(ds, flight_levels, longitude, latitude)

    for fl, level_spec in zip(flight_levels, levels, strict=True):
        single_ds, single_lon, single_lat = _synthetic_native_ds([level_spec])
        expected = _regrid_and_derive_native(single_ds, [fl], single_lon, single_lat)

        extracted = pcr.extract_flight_level(batched, fl)
        expected_single = pcr.extract_flight_level(expected, fl)

        np.testing.assert_array_equal(extracted["rhi"].values, expected_single["rhi"].values)
        np.testing.assert_array_equal(extracted["sac"].values, expected_single["sac"].values)
        np.testing.assert_array_equal(extracted["pcr"].values, expected_single["pcr"].values)


def test_regrid_and_derive_labels_correct_when_flight_levels_shuffled():
    """Regression test for the bug a positional (not value-based) level<->
    flight_level correspondence would have: `MetDataset` construction inside
    `regrid_and_derive` unconditionally sorts the `level` dim ascending by pressure,
    so passing `flight_levels` out of order must not silently swap which data ends
    up labeled which flight level."""
    flight_levels = [310, 320, 330]
    shuffled = [330, 310, 320]
    pressures = [pcr.target_pressure_hpa(fl) for fl in flight_levels]
    levels = [(215.0 + i, 3e-4 + i * 1e-5, p) for i, p in enumerate(pressures)]
    ds, longitude, latitude = _synthetic_native_ds(levels)

    batched = _regrid_and_derive_native(ds, shuffled, longitude, latitude)

    for fl, (air_temperature, specific_humidity, level_hpa) in zip(
        flight_levels, levels, strict=True
    ):
        extracted = pcr.extract_flight_level(batched, fl)
        expected_rhi = thermo.rhi(specific_humidity, air_temperature, level_hpa * 100.0)
        np.testing.assert_allclose(extracted["rhi"].values, expected_rhi, rtol=1e-6)
        assert float(extracted["level"].item()) == level_hpa


def test_utc_epoch_seconds_ignores_local_dst_state():
    """Found 2026-08-14: `int(time.timestamp())` on a naive datetime silently applies
    the *local* zone's *historical* DST rule for that date, producing filenames 1h off
    from the data's true UTC validity time on a non-UTC host for any date inside that
    zone's historical DST window (verified against the real `era5` mirror on a
    UK-zoned host: every mismatched file was dated before 2024-10-27, the 2024 UK
    DST-end date). `utc_epoch_seconds` must treat every naive `datetime` as UTC
    regardless of the host's local zone or the date's historical DST state --
    checked here against the exact known-bad case (a pre-2024-10-27 date) and its
    exact expected epoch, independent of `pytz`/host tzdata."""
    bst_period_time = datetime.datetime(2024, 10, 8, 12, 0)  # inside 2024 UK BST
    gmt_period_time = datetime.datetime(2024, 11, 13, 22, 0)  # after 2024 UK BST ended

    assert pcr.utc_epoch_seconds(bst_period_time) == 1728388800
    assert pcr.utc_epoch_seconds(gmt_period_time) == 1731535200


def test_pending_times_returns_everything_when_nothing_done():
    times = [datetime.datetime(2024, 9, 1, h) for h in range(3)]
    assert pcr.pending_times(times, [310, 320], [], "gs://bucket/arm") == times


def test_pending_times_excludes_fully_written_times():
    times = [datetime.datetime(2024, 9, 1, h) for h in range(3)]
    flight_levels = [310, 320]
    done = times[:2]
    existing = [
        f"gs://bucket/arm/{pcr.utc_epoch_seconds(t)}_{fl}.nc" for t in done for fl in flight_levels
    ]

    result = pcr.pending_times(times, flight_levels, existing, "gs://bucket/arm")

    assert result == times[2:]


def test_pending_times_keeps_partially_written_time():
    """A time with only some flight levels present must still be redone in full --
    `preprocess_forecast` writes all flight levels from one met fetch, so there's no
    cheaper way to backfill just the missing ones."""
    time = datetime.datetime(2024, 9, 1, 0)
    flight_levels = [310, 320, 330]
    existing = [f"gs://bucket/arm/{pcr.utc_epoch_seconds(time)}_310.nc"]

    result = pcr.pending_times([time], flight_levels, existing, "gs://bucket/arm")

    assert result == [time]


def test_pending_times_respects_extension_parameter():
    """`pending_times` is also reused for `.pq` observation sinks,
    not just `.nc` forecast sinks -- a `.nc`-only existing name must not count as
    done for a `.pq`-extension query, and vice versa."""
    time = datetime.datetime(2024, 9, 1, 0)
    flight_levels = [310]
    existing = [f"gs://bucket/arm/{pcr.utc_epoch_seconds(time)}_310.nc"]

    assert pcr.pending_times(
        [time], flight_levels, existing, "gs://bucket/arm", extension="pq"
    ) == [time]
    assert (
        pcr.pending_times([time], flight_levels, existing, "gs://bucket/arm", extension="nc") == []
    )


def test_write_netcdf_atomic_round_trips(tmp_path):
    ds = xr.Dataset({"x": (("i",), np.array([1.0, 2.0, 3.0]))})
    path = str(tmp_path / "out.nc")

    pcr.write_netcdf_atomic(ds, path)

    written = xr.open_dataset(path)
    np.testing.assert_array_equal(written["x"].values, ds["x"].values)
    # No leftover temp file from a successful write.
    assert list(tmp_path.iterdir()) == [tmp_path / "out.nc"]


def test_write_netcdf_atomic_leaves_nothing_on_failure(tmp_path, monkeypatch):
    """A kill/exception mid-write must not leave a truncated file at the target
    path -- that's exactly the corruption `pending_times`'s existence check can't
    detect. Simulates the failure via a raising `to_netcdf` rather than a real
    SIGKILL (the real-process kill test lives in the execution session's manual
    verification, not here)."""

    def _raise(self, *args, **kwargs):
        raise RuntimeError("simulated crash mid-write")

    monkeypatch.setattr(xr.Dataset, "to_netcdf", _raise)

    ds = xr.Dataset({"x": (("i",), np.array([1.0]))})
    path = str(tmp_path / "out.nc")

    with pytest.raises(RuntimeError, match="simulated crash mid-write"):
        pcr.write_netcdf_atomic(ds, path)

    assert not os.path.exists(path)
    assert list(tmp_path.iterdir()) == []


def test_kronecker_order_is_a_permutation_and_deterministic():
    n = 50
    order = pcr.kronecker_order(n)

    assert sorted(order) == list(range(n))
    assert pcr.kronecker_order(n) == order


def test_limit_times_none_returns_everything_in_order():
    times = [datetime.datetime(2024, 9, 1, h) for h in range(10)]
    assert pcr.limit_times(times, None) == times


def test_limit_times_is_deterministic_and_chronologically_ordered():
    times = [datetime.datetime(2024, 9, 1) + datetime.timedelta(hours=h) for h in range(2928)]

    result = pcr.limit_times(times, 127)

    assert len(result) == 127
    assert result == sorted(result)
    assert pcr.limit_times(times, 127) == result


def test_limit_times_nests_across_increasing_limits():
    """Every prefix's selection must be a subset of every larger prefix's -- this is
    what lets a progressive `--limit` ladder never redo a timestep an earlier,
    smaller pass already wrote."""
    times = [datetime.datetime(2024, 9, 1) + datetime.timedelta(hours=h) for h in range(2928)]

    selections = [set(pcr.limit_times(times, n)) for n in (127, 293, 586, 1464, 2928)]

    for smaller, larger in itertools.pairwise(selections):
        assert smaller <= larger


def test_filter_hours_none_is_a_no_op():
    times = [datetime.datetime(2024, 9, 1, h) for h in range(24)]
    assert pcr.filter_hours(times, None) == times


def test_filter_hours_restricts_to_subset():
    times = [datetime.datetime(2024, 9, 1, h) for h in range(24)]
    result = pcr.filter_hours(times, range(15, 23))
    assert result == [datetime.datetime(2024, 9, 1, h) for h in range(15, 23)]


def test_filter_hours_excludes_everything_when_no_hour_matches():
    times = [datetime.datetime(2024, 9, 1, h) for h in range(6)]
    assert pcr.filter_hours(times, {15, 16, 17}) == []
