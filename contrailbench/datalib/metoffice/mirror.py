r"""Resumable, parallel mirror of Met Office cruise-level CONUS fields to Zarr.

One Zarr store per calendar month, populated hour-by-hour via independent
region writes so a run can be interrupted and resumed without re-fetching completed
hours. Iterates months oldest-first, since that is the order the rolling S3 archive
expires data.

Uses a **process** pool, not threads. ``h5py``/HDF5 serializes internally across
threads within one process regardless of how many are spawned (measured: zero
speedup from 8 threads); separate processes each get their own HDF5 instance and
scale close to linearly (measured: ~4x wall-clock speedup from 4 processes).

Run as a script::

    python -m contrailbench.datalib.metoffice.mirror \
        --start 2024-09-01T00:00 --end 2024-12-31T23:00 --out-dir data/metoffice

**Default window is September 2024 only**, not the full Sep-Dec window. This is
a local-machine throughput limitation, not a scope decision: the pre-2026 archive
files need ~19x more individual chunk reads than current files for the same CONUS
subset (measured directly), so even with process-level parallelism, one month is a
multi-hour run. Pass ``--end 2024-12-31T23:00`` explicitly to cover the rest of the
window once September has been validated end-to-end.
"""

import argparse
import dataclasses
import datetime
import json
import logging
import pathlib
from collections.abc import Collection
from concurrent.futures import ProcessPoolExecutor, as_completed

import dask.array as dask_array
import numpy as np
import pandas as pd
import xarray as xr
from botocore.client import BaseClient

from contrailbench import pcr
from contrailbench.datalib.metoffice import s3

logging.basicConfig(
    level="INFO",
    format="[%(asctime)s.%(msecs)03d] [%(levelname)s] [%(module)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

#: Parameters mirrored for every hour
MIRRORED_PARAMETERS = ("temperature_on_pressure_levels", "relative_humidity_on_pressure_levels")

#: Default period -- September 2024 only; see module docstring on why this is
#: narrower than the full Sep-Dec window
DEFAULT_START = datetime.datetime(2024, 9, 1, 0, 0)
DEFAULT_END = datetime.datetime(2024, 9, 30, 23, 0)

#: Default number of parallel fetch processes
DEFAULT_WORKERS = 8

#: Default mirror output directory. Named as a module constant (rather than left
#: inline in ``_parse_args``) so other modules -- notably the pycontrails datalib
#: (``ukmo.py``) -- can resolve the same default without duplicating the path
#: arithmetic.
DEFAULT_OUT_DIR = pathlib.Path(__file__).resolve().parents[3] / "data" / "metoffice"

#: Per-file global attrs captured into the manifest for provenance, at zero extra
#: fetch cost since the file is already open. Not enforced -- absence (e.g. for
#: hours mirrored before this was added) is not an error. Lets model-configuration
#: changes across the archive (e.g. a UM parametrization-suite or grid-config
#: change) be detected later, rather than assumed away.
PROVENANCE_ATTRS = ("um_version", "mosg__grid_version")


@dataclasses.dataclass
class _MirrorTarget:
    """One mirror run's worth of work: a store, its manifest, and a time axis.

    ``label`` is used only in log messages -- ``"2024-09"`` for a calendar-month
    target (shortest-lead), ``"lead024"`` for a fixed-lead target.
    ``lead_hours`` is ``None`` for the shortest-lead path (:func:`mirror`) and a
    fixed int for :func:`mirror_fixed_lead` -- ``is not None`` throughout, never
    truthiness, since ``lead_hours=0`` is a real fixed lead, not "unset."
    """

    label: str
    store_path: pathlib.Path
    manifest_path: pathlib.Path
    time_index: pd.DatetimeIndex
    times_to_process: list[datetime.datetime]
    lead_hours: int | None
    extent: tuple[float, float, float, float]
    region: str


def _month_hours(year: int, month: int) -> pd.DatetimeIndex:
    """Get every hourly timestamp in a calendar month."""
    start = pd.Timestamp(year=year, month=month, day=1)
    end = start + pd.offsets.MonthBegin(1)
    return pd.date_range(start, end, freq="1h", inclusive="left")


def _months_between(start: datetime.datetime, end: datetime.datetime) -> list[tuple[int, int]]:
    """Get (year, month) pairs from ``start`` to ``end``, oldest first."""
    months = []
    cursor = datetime.date(start.year, start.month, 1)
    end_marker = datetime.date(end.year, end.month, 1)
    while cursor <= end_marker:
        months.append((cursor.year, cursor.month))
        cursor = (pd.Timestamp(cursor) + pd.offsets.MonthBegin(1)).date()
    return months


def _load_manifest(path: pathlib.Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _write_manifest(path: pathlib.Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def _fetch_conus_coords(
    fs: BaseClient,
    first_hour: datetime.datetime,
    lead_hours: int | None = None,
    extent: tuple[float, float, float, float] = s3.CONUS_EXTENT,
) -> tuple:
    """Fetch native-grid pressure/latitude/longitude coordinate values.

    Reads them from one real file (lazily -- no data fetch), since the mirror's Zarr
    template needs to be created with the exact native-grid coordinate values, not
    invented ones.

    Parameters
    ----------
    lead_hours : int, optional
        ``None`` (default) resolves the shortest available lead for ``first_hour``.
        A fixed int resolves the run at that lead instead.
    extent : tuple[float, float, float, float], optional
        Region to crop to. Defaults to :data:`s3.CONUS_EXTENT`, unchanged from before
        this parameter existed.
    """
    if lead_hours is None:
        run, resolved_lead = s3.run_and_lead_for_validity(first_hour)
    else:
        run, resolved_lead = s3.run_for_validity_at_lead(first_hour, lead_hours), lead_hours
    key = s3.object_key(run, first_hour, resolved_lead, "temperature_on_pressure_levels")
    da = s3.open_pressure_level_field(fs, key, "temperature_on_pressure_levels", extent=extent)
    return (
        da["pressure"].values,
        da["latitude"].values,
        da["longitude"].values,
    )


def _create_store_template(
    store_path: pathlib.Path,
    time_index: pd.DatetimeIndex,
    pressure_pa: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
) -> None:
    """Create an uninitialized Zarr store sized for ``time_index``.

    Writes only structure/metadata (``compute=False``) -- no chunk data is written,
    so unfetched hours read back as NaN (the fill value) until region-written.
    ``time_index`` is the store's entire time axis: dense-hourly for a calendar
    month, or the sparse matched-validity-time set for a fixed lead -- this
    function doesn't care which.
    """
    shape = (len(time_index), len(pressure_pa), len(latitude), len(longitude))
    chunks = (1, len(pressure_pa), len(latitude), len(longitude))
    dims = ("time", "pressure", "latitude", "longitude")

    template = xr.Dataset(
        {
            name: (
                dims,
                dask_array.full(shape, np.nan, chunks=chunks, dtype=np.float32),
                {},
                {"_FillValue": np.nan},
            )
            for name in ("air_temperature", "relative_humidity")
        },
        coords={
            "time": time_index,
            "pressure": pressure_pa,
            "latitude": latitude,
            "longitude": longitude,
        },
    )
    template.to_zarr(store_path, mode="w-", compute=False)


def _fetch_and_write_hour(
    store_path: pathlib.Path,
    hour: datetime.datetime,
    time_index: int,
    lead_hours: int | None = None,
    *,
    extent: tuple[float, float, float, float] = s3.CONUS_EXTENT,
    region_name: str = "conus",
) -> dict:
    """Fetch one hour's cruise-level, region-cropped fields and region-write them.

    Runs in its own worker process (see :func:`_mirror_target`), so it builds its
    own S3 client rather than receiving one -- ``boto3`` clients aren't meant to
    cross a process boundary. The region write needs no lock: each hour occupies
    disjoint Zarr chunks (chunk size 1 along ``time``), so concurrent writes from
    different processes never touch the same file.

    Parameters
    ----------
    lead_hours : int, optional
        ``None`` (default) resolves the shortest available lead for ``hour``.
        A fixed int resolves the run at that lead instead -- the
        returned manifest entry's ``lead_hours`` is always the actually-used
        value, never ``None``.
    extent : tuple[float, float, float, float], optional
        Region to crop to. Defaults to :data:`s3.CONUS_EXTENT`, unchanged from before
        this parameter existed.
    region_name : str, optional
        Named region label recorded verbatim into the returned manifest entry.
        Named ``region_name``, not ``region``, to avoid
        shadowing this function's local Zarr write-region dict (below). Defaults to
        ``"conus"``, matching ``extent``'s default.
    """
    fs = s3.filesystem()
    if lead_hours is None:
        run, resolved_lead = s3.run_and_lead_for_validity(hour)
    else:
        run, resolved_lead = s3.run_for_validity_at_lead(hour, lead_hours), lead_hours

    values = {}
    provenance: dict[str, str] = {}
    for parameter in MIRRORED_PARAMETERS:
        key = s3.object_key(run, hour, resolved_lead, parameter)
        field = s3.fetch_pressure_level_field(
            fs, key, parameter, run=run, validity=hour, lead_hours=resolved_lead, extent=extent
        )
        values[parameter] = field.values

        for attr in PROVENANCE_ATTRS:
            value = field.attrs.get(attr)
            if value is None:
                continue
            if attr in provenance and provenance[attr] != value:
                logger.warning(
                    "%s disagrees across parameters for %s: %s vs %s",
                    attr,
                    hour,
                    provenance[attr],
                    value,
                )
            provenance[attr] = value

    hour_ds = xr.Dataset(
        {
            "air_temperature": (
                ("time", "pressure", "latitude", "longitude"),
                values["temperature_on_pressure_levels"][np.newaxis],
            ),
            "relative_humidity": (
                ("time", "pressure", "latitude", "longitude"),
                values["relative_humidity_on_pressure_levels"][np.newaxis],
            ),
        },
    )

    region = {
        "time": slice(time_index, time_index + 1),
        "pressure": slice(None),
        "latitude": slice(None),
        "longitude": slice(None),
    }
    hour_ds.to_zarr(store_path, region=region)

    return {
        "run": run.isoformat(),
        "lead_hours": resolved_lead,
        "region": region_name,
        "status": "ok",
        **provenance,
    }


def _mirror_target(target: _MirrorTarget, workers: int) -> None:
    """Mirror one target's requested times, parallel and resumable.

    Lead-agnostic: used for both a calendar-month, shortest-lead target
    and a whole-window, fixed-lead target -- the only difference is
    ``target.time_index``'s density and ``target.lead_hours``.
    """
    manifest = _load_manifest(target.manifest_path)
    pending = [
        t
        for t in target.times_to_process
        if manifest.get(t.isoformat(), {}).get("status") != "ok"
    ]

    logger.info(
        "%s: %d times requested, %d already captured, %d to fetch",
        target.label,
        len(target.times_to_process),
        len(target.times_to_process) - len(pending),
        len(pending),
    )

    if not target.store_path.exists():
        fs = s3.filesystem()
        first = target.times_to_process[0] if target.times_to_process else target.time_index[0]
        pressure_pa, latitude, longitude = _fetch_conus_coords(
            fs, first, target.lead_hours, extent=target.extent
        )
        _create_store_template(
            target.store_path, target.time_index, pressure_pa, latitude, longitude
        )
        logger.info("%s: created Zarr template at %s", target.label, target.store_path)

    if not pending:
        logger.info("%s: nothing to do", target.label)
        return

    completed = 0
    failed = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _fetch_and_write_hour,
                target.store_path,
                t,
                int(target.time_index.get_loc(pd.Timestamp(t))),
                target.lead_hours,
                extent=target.extent,
                region_name=target.region,
            ): t
            for t in pending
        }
        for future in as_completed(futures):
            t = futures[future]
            try:
                entry = future.result()
            except Exception as e:  # record and continue, don't abort the whole target
                failed += 1
                entry = {"status": "error", "error": str(e)}
                logger.warning("%s %s failed: %s", target.label, t, e)
            else:
                completed += 1

            # single-threaded here (the parent process's own loop) -- no lock needed
            manifest[t.isoformat()] = entry
            _write_manifest(target.manifest_path, manifest)

            # Log every completion (not just every 24th) -- this is a multi-hour,
            # background-run process, and a steady heartbeat is the cheapest way to
            # tell "still working" apart from "silently died" without polling the
            # manifest file.
            logger.info(
                "%s: %d/%d done (%d failed) -- last: %s",
                target.label,
                completed + failed,
                len(pending),
                failed,
                t.isoformat(),
            )

    logger.info(
        "%s: finished -- %d captured, %d failed this run", target.label, completed, failed
    )


def _resolve_region_extent(region: str, out_dir: pathlib.Path) -> tuple[float, float, float, float]:
    """Resolve ``region`` to an extent, enforcing the mirror/out_dir separation.

    Raises rather than silently proceeding if a non-``"conus"`` region is about to
    write into :data:`DEFAULT_OUT_DIR` -- that default is CONUS's own mirror
    location, and this mirror must never overwrite or extend it.
    """
    if region != "conus" and out_dir == DEFAULT_OUT_DIR:
        msg = (
            f"region={region!r} requires an explicit --out-dir distinct from the "
            f"CONUS default ({DEFAULT_OUT_DIR}) -- this "
            "mirror must not overwrite or extend the CONUS mirror."
        )
        raise ValueError(msg)
    return pcr.REGION_EXTENTS[region]


def mirror(
    start: datetime.datetime,
    end: datetime.datetime,
    out_dir: pathlib.Path,
    workers: int,
    region: str = "conus",
    *,
    hours: Collection[int] | None = None,
) -> None:
    """Mirror every hour from ``start`` to ``end`` inclusive, oldest month first.

    Shortest-available lead -- one store per calendar month.

    Parameters
    ----------
    region : str, optional
        Named region to crop to, a key of :data:`contrailbench.pcr.REGION_EXTENTS`.
        Defaults to ``"conus"``, unchanged from before this parameter existed. A
        non-``"conus"`` region requires ``out_dir`` to differ from
        :data:`DEFAULT_OUT_DIR`.
    hours : Collection[int] | None, optional
        UTC hours to restrict fetching to, e.g.
        ``range(15, 23)`` for 1500-2200 UTC. ``None`` (default) fetches every hour,
        unchanged from before this parameter existed. The store's time axis stays
        dense regardless -- unfetched hours are simply never region-written, so
        restricting ``hours`` costs no extra disk and a later, broader-``hours``
        run still resumes cleanly off the same manifest.
    """
    extent = _resolve_region_extent(region, out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for year, month in _months_between(start, end):
        month_hours = _month_hours(year, month)
        hours_to_process = [h.to_pydatetime() for h in month_hours if start <= h <= end]
        hours_to_process = pcr.filter_hours(hours_to_process, hours)
        target = _MirrorTarget(
            label=f"{year:04d}-{month:02d}",
            store_path=out_dir / f"{year:04d}-{month:02d}.zarr",
            manifest_path=out_dir / f"{year:04d}-{month:02d}.manifest.json",
            time_index=month_hours,
            times_to_process=hours_to_process,
            lead_hours=None,
            extent=extent,
            region=region,
        )
        _mirror_target(target, workers)


def mirror_fixed_lead(
    start: datetime.datetime,
    end: datetime.datetime,
    *,
    lead_hours: int,
    lead_family: Collection[int],
    out_dir: pathlib.Path,
    workers: int,
    region: str = "conus",
) -> None:
    """Mirror one fixed lead's matched-validity-time intersection.

    Unlike :func:`mirror`, this writes **one store spanning the whole window**,
    not one per month -- the sparse (2-4/day) validity set at a fixed long lead
    doesn't warrant month-partitioning, and a single store avoids month-boundary
    concatenation on the read side.

    Parameters
    ----------
    lead_hours : int
        The fixed lead (whole hours) to mirror.
    lead_family : Collection[int]
        Every lead being studied together (e.g. ``(0, 24, 48, 72)``) -- used only
        to compute the matched-validity-time intersection, not to
        select which leads this call mirrors.
    region : str, optional
        Named region to crop to, a key of :data:`contrailbench.pcr.REGION_EXTENTS`.
        Defaults to ``"conus"``, unchanged from before this parameter existed. A
        non-``"conus"`` region requires ``out_dir`` to differ from
        :data:`DEFAULT_OUT_DIR`.
    """
    extent = _resolve_region_extent(region, out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    validity_times = pd.DatetimeIndex(sorted(s3.matched_validity_times(start, end, lead_family)))
    target = _MirrorTarget(
        label=f"lead{lead_hours:03d}",
        store_path=out_dir / f"lead{lead_hours:03d}.zarr",
        manifest_path=out_dir / f"lead{lead_hours:03d}.manifest.json",
        time_index=validity_times,
        times_to_process=list(validity_times.to_pydatetime()),
        lead_hours=lead_hours,
        extent=extent,
        region=region,
    )
    _mirror_target(target, workers)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=pd.Timestamp, default=DEFAULT_START)
    parser.add_argument("--end", type=pd.Timestamp, default=DEFAULT_END)
    parser.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--lead-hours",
        type=int,
        default=None,
        help="Mirror one fixed lead's matched-validity-time intersection "
        "instead of the shortest-available-lead, month-partitioned mirror.",
    )
    parser.add_argument(
        "--lead-family",
        type=str,
        default="0,24,48,72",
        help="Comma-separated leads (hours) to intersect over when --lead-hours is "
        "given. Only used to compute the matched validity-time set.",
    )
    parser.add_argument(
        "--region",
        type=str,
        choices=sorted(pcr.REGION_EXTENTS),
        default="conus",
        help="Named region to crop to. A non-'conus' region "
        "requires --out-dir to differ from the CONUS default, to avoid overwriting "
        "or extending the CONUS mirror.",
    )
    parser.add_argument(
        "--hours",
        type=str,
        default=None,
        help="Comma-separated UTC hours to restrict the shortest-available-lead "
        "mirror to, e.g. '15,16,17,18,19,20,21,22' "
        "for 1500-2200 UTC. Default (unset) fetches every hour, unchanged from "
        "before this flag existed. Only applies to the shortest-available-lead "
        "path (mirror()), not --lead-hours.",
    )
    return parser.parse_args()


def main() -> None:
    """Program entrypoint."""
    args = _parse_args()
    start = pd.Timestamp(args.start).to_pydatetime()
    end = pd.Timestamp(args.end).to_pydatetime()
    if args.lead_hours is not None:
        lead_family = tuple(int(x) for x in args.lead_family.split(","))
        mirror_fixed_lead(
            start,
            end,
            lead_hours=args.lead_hours,
            lead_family=lead_family,
            out_dir=args.out_dir,
            workers=args.workers,
            region=args.region,
        )
    else:
        hours = tuple(int(x) for x in args.hours.split(",")) if args.hours else None
        mirror(start, end, args.out_dir, args.workers, region=args.region, hours=hours)


if __name__ == "__main__":
    main()
