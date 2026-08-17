"""Shared PCR derivation helper for the forecast arms.

Every new forecast source must call this one module so the comparison between
sources isolates the weather source, not the derivation. Every preprocess pipeline
follows the same order of operations:

1. Get ``air_temperature``/``specific_humidity`` on the target flight level's exact
   ISA pressure, at native horizontal resolution. Sources whose datalib only exposes
   a small fixed set of pressure levels (e.g. Met Office's 7, GFS's 4 in the cruise
   band) use :func:`interpolate_to_pressures` to get there; a source that can be
   queried at an arbitrary pressure directly (e.g. ARCO ERA5) skips that step.
2. :func:`regrid_to_benchmark_grid` -- conservative (area-weighted), horizontal-only
   regrid of those two *continuous* fields onto the shared 0.25 degree benchmark
   grid (:data:`BENCHMARK_LONGITUDE`/:data:`BENCHMARK_LATITUDE`, matching
   ``preprocess_adsb.py``/``preprocess_iagos.py`` exactly).
3. :func:`compute_rhi_and_sac` -- SAC and RHi computed *after* regridding, on the
   coarse grid. This ordering is deliberate: averaging a binary PCR outcome
   conservatively would produce a fractional coverage value with no principled
   binary cutoff, whereas averaging the continuous physical fields first keeps the
   final ``sac``/``pcr`` fields genuinely binary.

RHi threshold sweeping is deliberately *not* done here. ``sac`` is
threshold-independent; ``rhi`` is continuous. Downstream benchmark pipelines
recompute ``pcr = (rhi > threshold) & sac`` per swept threshold from these two
stored fields, mirroring how ``contrails_org``'s ``horizontal_buffer`` sweep and
``google``'s implied ``probability_threshold`` sweep both happen downstream of
preprocessing rather than inside it.
"""

from __future__ import annotations

import datetime
import os
import tempfile
import warnings
from collections.abc import Collection

import numpy as np
import xarray as xr

from contrailbench import time_utils
from contrailbench.datalib.metoffice import s3, ukmo
from pycontrails.core.met import MetDataset
from pycontrails.models import sac
from pycontrails.physics import thermo, units

#: Shared benchmark grid -- reused verbatim (not re-derived) from
#: ``preprocess_adsb.py``/``preprocess_iagos.py`` so coordinate values are
#: bit-identical to the existing arms. Downstream benchmark code does an *exact*
#: ``.sel()``, not ``.sel(method="nearest")``, against these coordinates.
BENCHMARK_LONGITUDE = np.linspace(-180.0, 179.75, 1440)
BENCHMARK_LATITUDE = np.linspace(-80.0, 80.0, 641)
BENCHMARK_LONGITUDE_BNDS = np.linspace(-180.125, 179.875, 1441)
BENCHMARK_LATITUDE_BNDS = np.linspace(-80.125, 80.125, 642)

#: CONUS bounding box, (lon_min, lon_max, lat_min, lat_max). Matches
#: ``contrailbench.datalib.metoffice.s3.CONUS_EXTENT`` and the ``EXTENT`` constant
#: duplicated across ``reports/jay_extension/pipelines/*_contrailwatch_region.py``.
CONUS_EXTENT = (-134, -63, 20, 50)

#: Shanwick OCA bounding box, (lon_min, lon_max, lat_min, lat_max).
SHANWICK_EXTENT = (-30.0, -10.0, 45.0, 61.0)

#: Named region extents, keyed for CLI/manifest use (e.g. ``mirror.py --region``).
REGION_EXTENTS: dict[str, tuple[float, float, float, float]] = {
    "conus": CONUS_EXTENT,
    "shanwick": SHANWICK_EXTENT,
}

#: Flight levels both PCR arms cover. Restricted to 310-440 (not the existing arms'
#: 270-440) because Met Office's 7 mirrored cruise levels (300-150 hPa) only bracket
#: this range -- FL270/280/290/300 fall outside the mirrored pressure band. ERA5 has
#: no such limit but is restricted to match, so the era5-vs-metoffice comparison
#: stays apples-to-apples.
PCR_FLIGHT_LEVELS = list(range(310, 450, 10))

#: RHi threshold sweep -- widened to 0.70-1.30 (0.90-1.30 was too narrow to
#: overlap the published-product arms' penalty range; ERA5's useful dynamic range
#: sits mostly below 1.0 since IFS clips supersaturation under ice cloud). Not
#: applied here -- exposed for downstream benchmark pipelines to recompute
#: ``pcr`` per threshold from stored ``rhi``/``sac``.
RHI_THRESHOLDS = tuple(np.round(np.linspace(0.70, 1.30, 13), 2))

#: Engine efficiency for SAC, matching ``preprocess_iagos.py``/``preprocess_gruan.py``
#: (and pycontrails' own ``SACParams`` default).
ENGINE_EFFICIENCY = 0.3

#: Leads (hours) studied for the forecast lead-time degradation study. T+0 is
#: mirrored fresh into its own store (not reused from the shortest-lead mirror) so
#: every lead in this family is handled by exactly one code path. Extend with
#: (120, 168) if the T+0-72 curves are still informative.
LEAD_STUDY_LEADS: tuple[int, ...] = (0, 24, 48, 72)

#: Intersection of validity times available at every lead in LEAD_STUDY_LEADS, over
#: the full Sep-Dec 2024 window -- 244 00Z/12Z-only validity times. Both
#: the T+0 baseline and every longer lead must be restricted to exactly this set, not
#: the full per-lead availability or the existing shortest-lead mirror's validity set.
#: Computed once at import time -- pure calendar arithmetic, no network I/O.
LEAD_STUDY_TIMES: tuple[datetime.datetime, ...] = tuple(
    s3.matched_validity_times(
        datetime.datetime(2024, 9, 1, 0, 0),
        datetime.datetime(2024, 12, 31, 23, 0),
        LEAD_STUDY_LEADS,
    )
)


def target_pressure_hpa(flight_level: int) -> int:
    """ISA pressure for a flight level, rounded to the nearest hPa.

    Rounded because ARCO ERA5's model-level-to-pressure-level interpolation
    (``pycontrails.datalib.ecmwf.model_levels.ml_to_pl``) only ever returns
    pressures rounded to the nearest hPa -- rounding here too, and using this same
    value as Met Office's vertical-interpolation target, keeps both arms evaluated
    at identical pressures rather than off by up to 0.5 hPa.

    Parameters
    ----------
    flight_level : int
        Flight level (hundreds of feet).

    Returns
    -------
    int
        ISA pressure, hPa.
    """
    return round(float(units.ft_to_pl(flight_level * 100.0)))


def interpolate_to_pressures(ds: xr.Dataset, target_pressures_hpa: list[float]) -> xr.Dataset:
    """Vertically interpolate (log-linear in pressure) onto several target pressures.

    Shared by every source whose datalib only exposes a small fixed set of pressure
    levels rather than an arbitrary one (Met Office's 7 native cruise levels, GFS's 4).
    One vectorized call replaces what would otherwise be one call per flight level --
    linear interpolation along an axis is elementwise per query point, so this is
    numerically identical to looping the same interpolation one target at a time.

    Parameters
    ----------
    ds : xr.Dataset
        Must have a ``level`` dim/coord in hPa, spanning every value in
        ``target_pressures_hpa``.

    target_pressures_hpa : list[float]
        Target pressures, hPa.

    Returns
    -------
    xr.Dataset
        ``ds`` with the ``level`` dim resampled to ``target_pressures_hpa``
        (size ``len(target_pressures_hpa)``).

    """
    attrs = ds.attrs
    log_level = np.log(ds["level"].values)
    log_targets = np.log(np.asarray(target_pressures_hpa, dtype=np.float64))
    interpolated = ds.assign_coords(level=log_level).interp(level=log_targets)
    return interpolated.assign_coords(level=list(target_pressures_hpa)).assign_attrs(attrs)


def crop_benchmark_grid(
    extent: tuple[float, float, float, float] = CONUS_EXTENT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Crop the global benchmark grid to a bounding box.

    Slices the global arrays rather than constructing an independent ``linspace``
    over the box, so the cropped coordinate values are bit-identical to the global
    grid (no float-precision misalignment with the existing arms).

    Parameters
    ----------
    extent : tuple[float, float, float, float]
        ``(lon_min, lon_max, lat_min, lat_max)``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ``(longitude, latitude, longitude_bnds, latitude_bnds)`` for the cropped
        region.
    """
    lon_min, lon_max, lat_min, lat_max = extent

    (lon_idx,) = np.nonzero((lon_min <= BENCHMARK_LONGITUDE) & (lon_max >= BENCHMARK_LONGITUDE))
    (lat_idx,) = np.nonzero((lat_min <= BENCHMARK_LATITUDE) & (lat_max >= BENCHMARK_LATITUDE))

    longitude = BENCHMARK_LONGITUDE[lon_idx]
    latitude = BENCHMARK_LATITUDE[lat_idx]
    # bnds has one more element than centers; a contiguous index range into centers
    # maps to the same contiguous range plus one into bnds.
    longitude_bnds = BENCHMARK_LONGITUDE_BNDS[lon_idx[0] : lon_idx[-1] + 2]
    latitude_bnds = BENCHMARK_LATITUDE_BNDS[lat_idx[0] : lat_idx[-1] + 2]

    return longitude, latitude, longitude_bnds, latitude_bnds


def cell_bounds(centers: np.ndarray, spacing: float) -> np.ndarray:
    """Cell edges for a uniform, ascending, cell-centred grid.

    Parameters
    ----------
    centers : np.ndarray
        Ascending, uniformly-spaced cell-centre coordinates.
    spacing : float
        Uniform spacing between centres.

    Returns
    -------
    np.ndarray
        ``len(centers) + 1`` ascending cell edges.
    """
    return np.concatenate(([centers[0] - spacing / 2.0], centers + spacing / 2.0))


def _overlap_weights(source_bnds: np.ndarray, target_bnds: np.ndarray) -> np.ndarray:
    """Conservative (area-weighted) 1D overlap-weight matrix.

    Row ``i`` gives the fraction of target cell ``i`` covered by each source cell,
    normalized by the *actual* covered fraction (not assumed to be 1) so a target
    cell only partially covered by source data still yields a valid weighted
    average rather than a downward-biased sum.

    Returns
    -------
    np.ndarray
        Shape ``(len(target_bnds) - 1, len(source_bnds) - 1)``.
    """
    src_lo, src_hi = source_bnds[:-1], source_bnds[1:]
    tgt_lo, tgt_hi = target_bnds[:-1], target_bnds[1:]

    overlap = np.minimum(tgt_hi[:, None], src_hi[None, :]) - np.maximum(
        tgt_lo[:, None], src_lo[None, :]
    )
    overlap = np.clip(overlap, 0.0, None)

    covered = overlap.sum(axis=1, keepdims=True)
    covered = np.where(covered > 0.0, covered, 1.0)  # avoid 0/0 for a fully-uncovered row
    return overlap / covered


def regrid_to_benchmark_grid(
    field: xr.DataArray,
    *,
    source_longitude: np.ndarray,
    source_latitude: np.ndarray,
    source_longitude_spacing: float,
    source_latitude_spacing: float,
    target_extent: tuple[float, float, float, float] = CONUS_EXTENT,
) -> xr.DataArray:
    """Conservatively regrid a continuous field onto the cropped benchmark grid.

    Source cell bounds are reconstructed analytically from known uniform spacing
    (``center +/- half_spacing``) since neither the Met Office mirror nor the ARCO
    ERA5 store carries ``latitude_bnds``/``longitude_bnds`` -- both sources are
    uniform lat/lon grids, so this is exact.

    Parameters
    ----------
    field : xr.DataArray
        Must have ``longitude``/``latitude`` dims matching ``source_longitude``/
        ``source_latitude``. Other dims (``level``, ``time``) are preserved.
    source_longitude, source_latitude : np.ndarray
        Ascending, uniformly-spaced native cell-centre coordinates.
    source_longitude_spacing, source_latitude_spacing : float
        Uniform native spacing, degrees.
    target_extent : tuple[float, float, float, float]
        ``(lon_min, lon_max, lat_min, lat_max)`` to crop the benchmark grid to.

    Returns
    -------
    xr.DataArray
        ``field`` regridded onto the cropped benchmark grid's ``longitude``/
        ``latitude`` coordinates.
    """
    target_lon, target_lat, target_lon_bnds, target_lat_bnds = crop_benchmark_grid(target_extent)

    source_lon_bnds = cell_bounds(source_longitude, source_longitude_spacing)
    source_lat_bnds = cell_bounds(source_latitude, source_latitude_spacing)

    lon_weights = xr.DataArray(
        _overlap_weights(source_lon_bnds, target_lon_bnds),
        dims=("_target_longitude", "longitude"),
        coords={"_target_longitude": target_lon, "longitude": source_longitude},
    )
    lat_weights = xr.DataArray(
        _overlap_weights(source_lat_bnds, target_lat_bnds),
        dims=("_target_latitude", "latitude"),
        coords={"_target_latitude": target_lat, "latitude": source_latitude},
    )

    regridded = xr.dot(field, lon_weights, dim="longitude")
    regridded = xr.dot(regridded, lat_weights, dim="latitude")
    return regridded.rename({"_target_longitude": "longitude", "_target_latitude": "latitude"})


def compute_rhi_and_sac(coarse_met: MetDataset) -> tuple[xr.DataArray, xr.DataArray]:
    """Compute RHi (continuous) and SAC (binary, threshold-independent) fields.

    Must be called *after* regridding (see module docstring). Wraps the SAC
    ``Model`` construction/eval to suppress two ``UserWarning``s that are expected,
    not spurious:

    - "Unknown provider"/"Unknown dataset" (via
      :func:`contrailbench.datalib.metoffice.ukmo.suppress_unregistered_source_warnings`)
      -- ``Model.eval()`` reads ``MetDataset.provider_attr``/``.dataset_attr``
      internally, which warns for an unrecognized provider (Met Office; ERA5's
      ``provider="ECMWF"`` is already recognized and never warns).
    - "Met data appears to have originated from ECMWF and no humidity scaling is
      enabled" (ERA5 only) -- this is pycontrails recommending a humidity-bias
      correction fitted against a reference dataset, which would break fairness:
      fitting one arm's humidity to the benchmark's own ground truth
      (IAGOS) while the other arms go uncorrected would no longer be a fair
      same-algorithm comparison.

    Parameters
    ----------
    coarse_met : MetDataset
        Must carry ``air_temperature`` and ``specific_humidity``, and
        ``provider``/``dataset``/``product`` attrs carried forward from the source
        (regridding does not preserve ``.attrs`` by default).

    Returns
    -------
    tuple[xr.DataArray, xr.DataArray]
        ``(rhi, sac)``.
    """
    with ukmo.suppress_unregistered_source_warnings(), warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=UserWarning, message="\nMet data appears to have originated"
        )
        sac_result = sac.SAC(
            coarse_met, params=sac.SACParams(engine_efficiency=ENGINE_EFFICIENCY)
        ).eval()

    rhi = thermo.rhi(
        coarse_met["specific_humidity"].data,
        coarse_met["air_temperature"].data,
        coarse_met["air_pressure"].data,
    )
    return rhi, sac_result["sac"].data


def regrid_and_derive(
    ds: xr.Dataset,
    *,
    flight_levels: list[int],
    source_longitude: np.ndarray,
    source_latitude: np.ndarray,
    source_longitude_spacing: float,
    source_latitude_spacing: float,
    target_extent: tuple[float, float, float, float] = CONUS_EXTENT,
) -> xr.Dataset:
    """Regrid and derive PCR fields for many flight levels from one met fetch.

    Shared by both arms' ``compute_forecast`` as a fetch-hoisting optimization, so
    a single met fetch covering ``len(flight_levels)`` pressure levels serves all of
    them, instead of one fetch per flight level. ``ds`` must already carry a
    ``level`` dim of size ``len(flight_levels)`` holding each flight level's ISA
    target pressure (see :func:`target_pressure_hpa`), in any order -- this function
    does not assume the two are positionally aligned (see below).

    **Why not just zip ``level`` and ``flight_levels`` by position:**
    ``MetDataset.__init__`` (inside :func:`compute_rhi_and_sac`, via
    ``MetDataset(coarse_met)``) unconditionally sorts the ``level`` dim ascending by
    pressure value, and ``pycontrails``' ``parse_pressure_levels`` does the same on
    the ERA5 side. Since ascending flight level is descending pressure, whatever
    order ``ds``'s ``level`` dim was built in cannot be trusted to still match
    ``flight_levels``'s order by the time this function's regrid/derive steps run.
    Reordering by *value* (via ``.sel``) before attaching flight-level labels is
    what makes this safe regardless of any internal resort.

    Parameters
    ----------
    ds : xr.Dataset
        Must have ``air_temperature``/``specific_humidity`` on a ``level`` dim (hPa)
        of size ``len(flight_levels)``, at native horizontal resolution.
    flight_levels : list[int]
        Flight levels ``ds``'s ``level`` dim covers, one ISA target pressure each
        (via :func:`target_pressure_hpa`). Order defines the returned dataset's
        ``flight_level`` dim order.
    source_longitude, source_latitude, source_longitude_spacing,
    source_latitude_spacing, target_extent
        Passed through to :func:`regrid_to_benchmark_grid`.

    Returns
    -------
    xr.Dataset
        ``rhi``/``sac``/``pcr`` fields (rhi_threshold=1.0 default) on the benchmark
        grid, with dims including ``flight_level`` (values = ``flight_levels``) in
        place of ``level``. The original pressures survive as a secondary
        coordinate ``level`` indexed by ``flight_level``. Extract one flight level's
        single-``level``-dim slice (matching the pre-batching per-call schema) with
        :func:`extract_flight_level`.
    """
    regridded = xr.Dataset(
        {
            "air_temperature": regrid_to_benchmark_grid(
                ds["air_temperature"],
                source_longitude=source_longitude,
                source_latitude=source_latitude,
                source_longitude_spacing=source_longitude_spacing,
                source_latitude_spacing=source_latitude_spacing,
                target_extent=target_extent,
            ),
            "specific_humidity": regrid_to_benchmark_grid(
                ds["specific_humidity"],
                source_longitude=source_longitude,
                source_latitude=source_latitude,
                source_longitude_spacing=source_longitude_spacing,
                source_latitude_spacing=source_latitude_spacing,
                target_extent=target_extent,
            ),
        }
    ).assign_attrs(ds.attrs)

    coarse_met = MetDataset(regridded)
    rhi, sac_field = compute_rhi_and_sac(coarse_met)
    default_pcr = (rhi > 1.0) & (sac_field > 0)

    out = xr.Dataset(
        {
            "rhi": rhi.astype("float32"),
            "sac": sac_field.astype("float32"),
            "pcr": default_pcr.astype("float32"),
        }
    )
    out["rhi"].attrs = {"long_name": "Relative humidity over ice", "units": "1"}
    out["sac"].attrs = {
        "long_name": "Schmidt-Appleman criterion satisfied flag",
        "units": "1",
    }
    out["pcr"].attrs = {
        "long_name": "Persistent contrail region flag (rhi_threshold=1.0 default)",
        "units": "1",
    }

    # Reorder by value (not position -- see docstring) so `flight_levels`'s order is
    # guaranteed, regardless of whatever order MetDataset's internal sort left the
    # `level` dim in.
    pressures = [target_pressure_hpa(fl) for fl in flight_levels]
    out = out.sel(level=pressures).rename(level="flight_level")
    return out.assign_coords(flight_level=list(flight_levels), level=("flight_level", pressures))


def extract_flight_level(batched: xr.Dataset, flight_level: int) -> xr.Dataset:
    """Extract one flight level's slice from :func:`regrid_and_derive`'s output.

    Inverse of the ``flight_level``-labeling step in :func:`regrid_and_derive`:
    reconstructs the single-flight-level, ``level``-dim schema each per-(time,
    flight_level) sink file expects (bit-identical to the pre-batching per-call
    output), from the batched, ``flight_level``-dim dataset.

    Parameters
    ----------
    batched : xr.Dataset
        Output of :func:`regrid_and_derive`.
    flight_level : int
        Flight level to extract. Must be one of the ``flight_levels`` ``batched``
        was built from.

    Returns
    -------
    xr.Dataset
        ``batched``, restricted to ``flight_level``, with the dim renamed back to
        ``level`` (size 1, value = that flight level's ISA target pressure).
    """
    sl = batched.sel(flight_level=[flight_level])
    pressure = float(sl["level"].item())
    return sl.drop_vars("level").rename(flight_level="level").assign_coords(level=[pressure])


def utc_epoch_seconds(time: datetime.datetime) -> int:
    """UTC-safe epoch conversion for the naive datetimes used throughout this codebase.

    Thin wrapper over :func:`contrailbench.time_utils.to_utc_timestamp`, which is now
    the canonical implementation shared by every pipeline (not just this module's own
    callers, :func:`pending_times` and ``preprocess_era5.py``'s sink-naming line). Kept
    here, rather than removed, so existing callers and tests don't need to change.
    """
    return time_utils.to_utc_timestamp(time)


def pending_times(
    times: Collection[datetime.datetime],
    flight_levels: Collection[int],
    existing_names: Collection[str],
    gcp_tmpdir: str,
    extension: str = "nc",
) -> list[datetime.datetime]:
    """Filter to the times still missing at least one flight level's output file.

    A single local ``DirectRunner`` process has no equivalent to
    Dataflow's autoscaling/per-element retry, so a killed/interrupted multi-hour run
    must be restartable without redoing already-written hours. A time counts as done
    only if *every* one of its flight levels' sink files is present -- a partially
    written time (e.g. killed mid-write) is treated as not done and is redone in
    full, since ``preprocess_forecast`` writes all of a time's flight levels from one
    met fetch and there is no cheaper way to redo just the missing ones.

    Parameters
    ----------
    times : Collection[datetime.datetime]
        Candidate times (e.g. a pipeline's full ``TIMES``).
    flight_levels : Collection[int]
        Flight levels each time must have a sink file for.
    existing_names : Collection[str]
        Sink file paths already present (e.g. one ``fsspec`` ``ls`` of the output
        prefix), in the same ``{gcp_tmpdir}/{unix_ts}_{flight_level}.{extension}``
        form ``preprocess_forecast`` writes.
    gcp_tmpdir : str
        Output prefix, matching the pipeline's ``GCP_TMPDIR``.
    extension : str, optional
        Sink file extension, without the leading dot. Defaults to ``"nc"``,
        unchanged from before this parameter existed (``preprocess_forecast``'s
        netCDF sinks). This same existence-check logic is also reused for the
        ``.pq`` observation-cache sinks written by
        ``preprocess_iagos.py``/``preprocess_adsb.py``.

    Returns
    -------
    list[datetime.datetime]
        ``times``, in order, excluding every time whose flight levels are all
        already present in ``existing_names``.
    """
    # `fsspec`-family `ls()` implementations (e.g. `gcsfs`) return paths with the
    # `gs://` scheme stripped, even when the listed directory was passed with it --
    # normalize both sides so this doesn't depend on which form the caller's listing
    # happened to come back in.
    def _strip_scheme(path: str) -> str:
        return path.split("://", 1)[-1]

    existing = {_strip_scheme(name) for name in existing_names}
    return [
        time
        for time in times
        if not all(
            _strip_scheme(f"{gcp_tmpdir}/{utc_epoch_seconds(time)}_{fl}.{extension}") in existing
            for fl in flight_levels
        )
    ]


def write_netcdf_atomic(ds: xr.Dataset, path: str) -> None:
    """Write ``ds`` to ``path`` atomically via temp-file-then-``os.replace``.

    GCS's upload API only makes an object visible on full completion (confirmed
    empirically: killing a real upload mid-transfer leaves no object at all, never a
    truncated one) -- local filesystem writes have no equivalent guarantee. A bare
    ``ds.to_netcdf(path)`` killed mid-write would leave a truncated file *at* ``path``,
    which :func:`pending_times`'s existence check would then wrongly treat as done.

    Writing to a temp file in ``path``'s own directory first means a kill leaves only
    the abandoned temp file -- ``path`` itself never exists until the write is
    complete and the rename (atomic within one filesystem, per POSIX) has happened.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to write.
    path : str
        Destination path. Parent directory must already exist.
    """
    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".nc.tmp")
    os.close(fd)
    try:
        ds.to_netcdf(tmp_path)
        os.replace(tmp_path, path)
    except BaseException:
        os.remove(tmp_path)
        raise


#: Golden ratio conjugate -- irrational, so `(i * PHI_INV) % 1.0` never repeats a
#: value for distinct integer `i`, which is what makes `kronecker_order` a genuine
#: permutation (no ties) rather than merely "usually" one.
PHI_INV = 0.6180339887498949


def kronecker_order(n: int) -> list[int]:
    """Deterministic low-discrepancy permutation of ``range(n)``.

    Sorting ``range(n)`` by the fractional part of ``i * PHI_INV`` (a Kronecker
    sequence) spreads any prefix of the result evenly across ``range(n)`` -- far more
    evenly than a chronological prefix or a fixed stride, and without a stride's
    tendency to lock onto a residue pattern modulo some other period in the data
    (e.g. a forecast's run cycle). Used by :func:`limit_times` to pick a
    representative subsample of a pipeline's ``TIMES`` for a partial/preview run.

    Parameters
    ----------
    n : int
        Length of the range to permute.

    Returns
    -------
    list[int]
        A permutation of ``range(n)``, most-representative-prefix-first.
    """
    return sorted(range(n), key=lambda i: (i * PHI_INV) % 1.0)


def limit_times(
    times: Collection[datetime.datetime], limit: int | None
) -> list[datetime.datetime]:
    """First ``limit`` times per :func:`kronecker_order`; ``None`` returns all.

    Because ``kronecker_order`` computes one fixed permutation for a given
    ``len(times)``, increasing ``limit`` only ever takes a longer prefix of that same
    permutation -- so successive calls with growing ``limit`` values *nest*: every
    time selected by a smaller ``limit`` is also selected by any larger one. Combined
    with :func:`pending_times`, a progressive ladder of runs (``--limit 127``, then
    ``--limit 293``, ...) never redoes a timestep an earlier pass already wrote.

    Parameters
    ----------
    times : Collection[datetime.datetime]
        Candidate times (e.g. a pipeline's full ``TIMES``), in order.
    limit : int | None
        How many to select. ``None`` selects all of ``times`` (order preserved).

    Returns
    -------
    list[datetime.datetime]
        The selected times, restored to ``times``'s original order.
    """
    times = list(times)
    if limit is None:
        return times
    order = kronecker_order(len(times))
    return [times[i] for i in sorted(order[:limit])]


def filter_hours(
    times: Collection[datetime.datetime], hours: Collection[int] | None
) -> list[datetime.datetime]:
    """Restrict a list of target validity times to an hour-of-day subset.

    Applies to the target validity-time list itself (e.g. a pipeline's ``TIMES``),
    not to raw observation rows -- observation loaders like ``get_iagos``/``get_adsb``
    fetch a +/-30 minute window around each target hour, so filtering their raw rows
    by hour-of-day directly is not quite equivalent to "which target hours are in
    scope".

    Parameters
    ----------
    times : Collection[datetime.datetime]
        Candidate times, in order.
    hours : Collection[int] | None
        UTC hours to keep (e.g. ``range(15, 23)`` for 1500-2200 UTC). ``None`` is a
        no-op -- every hour is kept (CONUS's existing all-hours behaviour).

    Returns
    -------
    list[datetime.datetime]
        The subset of ``times`` whose ``.hour`` is in ``hours``, order preserved.
    """
    if hours is None:
        return list(times)
    hour_set = set(hours)
    return [t for t in times if t.hour in hour_set]
