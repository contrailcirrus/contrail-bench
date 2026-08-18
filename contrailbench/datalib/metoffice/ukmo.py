"""pycontrails datalib wrapping the Met Office mirror.

Structured like ``pycontrails.datalib.dwd.icon``/``ods``, but there is no remote
fetch at datalib-usage time: :mod:`contrailbench.datalib.metoffice.mirror` has
already populated local, monthly Zarr stores, so "download" here means confirming
the requested hours were captured (via the mirror's manifest), not fetching bytes
-- see :meth:`MetOfficeUM.download_dataset`.

**Humidity convention.** The mirror stores raw, water-referenced, fractional relative
humidity (``units: 1``). This datalib converts it to ``specific_humidity`` via
``q = RH_w * thermo.q_sat_liquid(T, p)``, which makes pycontrails' own
``thermo.rhi(q, T, p) = q*p / (epsilon*e_sat_ice(T))`` reduce to the ice-referenced
RHi conversion (``RH_w * e_sat_liquid(T) / e_sat_ice(T)``) that ISSR/SAC/PCR need,
without this module reimplementing saturation-vapor-pressure formulas itself.

**Scope.** Geopotential height and the ``flag`` variable are not mirrored, and
aren't needed: ``MetDataset`` derives ``air_pressure``/``altitude`` from the
``level`` (pressure) coordinate itself, and the 7 mirrored cruise levels (300-150
hPa) sit well above CONUS terrain, so the flag's below-surface-pressure condition
can't occur here. pycontrails' "Unknown provider"/"Unknown dataset" ``UserWarning``
comes from hardcoded tuples in pycontrails core, unrelated to this module's own
variable/unit validation (``metsource.parse_variables``); this module sets
``provider``/``dataset``/``product`` attrs as metadata regardless, and
:func:`suppress_unregistered_source_warnings` silences the warning for callers
that construct/run a pycontrails ``Model`` (e.g. ``ISSR``, ``SAC``, ``PCR``)
against this data.
"""

from __future__ import annotations

import contextlib
import json
import pathlib
import warnings
from collections.abc import Iterator
from datetime import datetime
from typing import Any

import numpy as np
import xarray as xr
from pycontrails.core import met_var
from pycontrails.core.met import MetDataset, MetVariable
from pycontrails.datalib._met_utils import metsource
from pycontrails.physics import thermo

from contrailbench.datalib.metoffice import mirror, s3

#: MetDataset.attrs values set by :meth:`MetOfficeUM.set_metadata`. Not recognized by
#: pycontrails core (see module docstring) -- cosmetic metadata, not validated.
PROVIDER = "Met Office"
DATASET = "UM-Global-Deterministic-10km"
PRODUCT = "forecast"


class MetOfficeMirrorNotFoundError(Exception):
    """Raised when requested data was never captured by the mirror.

    Distinguishes "you asked for data that doesn't exist yet/failed to fetch" from a
    downstream NaN silently flowing into ISSR/SAC and producing a wrong PCR field.
    """


@contextlib.contextmanager
def suppress_unregistered_source_warnings() -> Iterator[None]:
    """Suppress pycontrails' "Unknown provider"/"Unknown dataset" warnings.

    ``MetOfficeUM``'s ``provider``/``dataset`` attrs (:data:`PROVIDER`,
    :data:`DATASET`) aren't in pycontrails core's hardcoded, non-extensible
    ``supported`` tuples (see module docstring), so anything that reads
    ``MetDataset.provider_attr``/``.dataset_attr`` -- notably running a pycontrails
    ``Model`` (e.g. ``ISSR``, ``SAC``, ``PCR``) against this data -- will trigger a
    ``UserWarning``. Wrap that access in this context manager to silence it, the same
    way ``pycontrails.models.cocip.Cocip``'s docstring recommends for its own
    unrecognized-source case.

    Examples
    --------
    >>> with suppress_unregistered_source_warnings():  # doctest: +SKIP
    ...     pcr = PCR(met).eval()
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message="Unknown provider")
        warnings.filterwarnings("ignore", category=UserWarning, message="Unknown dataset")
        yield


class MetOfficeUM(metsource.MetDataSource):
    """pycontrails datalib for the UK Met Office ``global-deterministic-10km`` product.

    Reads from the local monthly Zarr stores produced by
    :mod:`contrailbench.datalib.metoffice.mirror`, not from S3 directly -- data must
    already be mirrored for the requested window (see :meth:`download_dataset`).

    Parameters
    ----------
    time : metsource.TimeInput
        Single datetime or ``(start, end)`` range. Parsed to hourly timesteps.
    variables : metsource.VariableInput, optional
        Requested variables. Defaults to
        ``[met_var.AirTemperature, met_var.SpecificHumidity]`` -- the two ISSR/SAC/PCR
        require. ``specific_humidity`` is always computed internally regardless of
        what's requested (see module docstring); pass ``met_var.RelativeHumidity``
        explicitly to also keep the raw mirrored value.
    pressure_levels : metsource.PressureLevelInput, optional
        Requested pressure levels in hPa. Defaults to
        :data:`contrailbench.datalib.metoffice.s3.CRUISE_LEVELS_HPA` (the 7 mirrored
        cruise levels).
    paths : str | pathlib.Path, optional
        Mirror output directory (i.e. the ``--out-dir`` passed to ``mirror.py``).
        Defaults to :data:`contrailbench.datalib.metoffice.mirror.DEFAULT_OUT_DIR`.
    grid : float, optional
        Not supported -- the mirror serves fixed native-grid data; regridding
        happens in the PCR pipeline, not this datalib. A non-``None`` value
        is ignored with a warning.
    lead_hours : int, optional
        ``None`` (default) reads from the shortest-lead, month-partitioned mirror
        -- unchanged default behavior. A fixed int
        reads instead from the single whole-window store
        ``{out_dir}/lead{lead_hours:03d}.zarr`` produced by
        :func:`contrailbench.datalib.metoffice.mirror.mirror_fixed_lead`. See
        :class:`contrailbench.datalib.metoffice.mirror._MirrorTarget` for why
        ``lead_hours=0`` is checked via ``is not None`` rather than truthiness.

    """

    __slots__ = ("_manifest_cache", "cachestore", "lead_hours", "out_dir")

    def __init__(
        self,
        time: metsource.TimeInput,
        *,
        variables: metsource.VariableInput | None = None,
        pressure_levels: metsource.PressureLevelInput = s3.CRUISE_LEVELS_HPA,
        paths: str | pathlib.Path | None = None,
        grid: float | None = None,
        lead_hours: int | None = None,
        **kwargs: Any,
    ) -> None:
        if grid is not None:
            warnings.warn(
                f"MetOfficeUM serves fixed native-grid data -- regridding happens in "
                f"the PCR pipeline, not this datalib. Ignoring grid={grid!r}."
            )
        self.grid = None

        self.pressure_levels = metsource.parse_pressure_levels(
            pressure_levels, supported=list(s3.CRUISE_LEVELS_HPA)
        )

        if variables is None:
            variables = [met_var.AirTemperature, met_var.SpecificHumidity]
        self.variables = metsource.parse_variables(variables, self.supported_variables)

        self.timesteps = metsource.parse_timesteps(time, freq="1h")

        self.out_dir = pathlib.Path(paths) if paths is not None else mirror.DEFAULT_OUT_DIR
        self.paths = str(self.out_dir)
        self.lead_hours = lead_hours

        # No download cache to maintain -- the mirror IS the durable cache. Setting
        # this to None makes the inherited `is_datafile_cached` return False
        # unconditionally (its first line), which makes `download()` always call
        # `download_dataset(self.timesteps)` -- exactly "validate what's requested
        # against the manifest," for free, with no override needed. See module
        # docstring for why this differs from ICON/ERA5's real download cache.
        self.cachestore = None
        # Keyed by (year, month) for the shortest-lead path, or ("lead", lead_hours)
        # for a fixed-lead store -- see _manifest.
        self._manifest_cache: dict[tuple[int, int] | tuple[str, int], dict[str, Any]] = {}

        del kwargs  # accepted only for MetDataSource ABC compatibility; unused

    @property
    def pressure_level_variables(self) -> list[MetVariable]:
        """Variables available from the mirrored store."""
        return [met_var.AirTemperature, met_var.SpecificHumidity, met_var.RelativeHumidity]

    @property
    def single_level_variables(self) -> list[MetVariable]:
        """Single-level variables available -- none; the mirror is pressure-level only."""
        return []

    def _manifest_path(self, t: datetime) -> pathlib.Path:
        """Manifest path covering ``t``.

        One whole-window manifest per fixed lead when :attr:`lead_hours`
        is set, else one per calendar month.
        """
        if self.lead_hours is not None:
            return self.out_dir / f"lead{self.lead_hours:03d}.manifest.json"
        return self.out_dir / f"{t.year:04d}-{t.month:02d}.manifest.json"

    def _manifest(self, t: datetime) -> dict[str, Any]:
        """Load (and cache) the manifest covering ``t``.

        Raises
        ------
        MetOfficeMirrorNotFoundError
            If no manifest exists for this month/lead -- mirror.py hasn't been run
            for it.
        """
        key = ("lead", self.lead_hours) if self.lead_hours is not None else (t.year, t.month)
        if key not in self._manifest_cache:
            manifest_path = self._manifest_path(t)
            if not manifest_path.exists():
                if self.lead_hours is not None:
                    msg = (
                        f"No manifest for lead {self.lead_hours}h at {manifest_path} -- "
                        "mirror.py has not been run for this lead. Run: "
                        f"python -m contrailbench.datalib.metoffice.mirror "
                        f"--lead-hours {self.lead_hours} --out-dir {self.out_dir}"
                    )
                else:
                    msg = (
                        f"No manifest for {t.year:04d}-{t.month:02d} at {manifest_path} -- "
                        "mirror.py has not been run for this month. Run: "
                        f"python -m contrailbench.datalib.metoffice.mirror "
                        f"--start {t.year:04d}-{t.month:02d}-01T00:00 --out-dir {self.out_dir}"
                    )
                raise MetOfficeMirrorNotFoundError(msg)
            self._manifest_cache[key] = json.loads(manifest_path.read_text())
        return self._manifest_cache[key]

    def download_dataset(self, times: list[datetime]) -> None:
        """Confirm every requested hour was captured by the mirror.

        Nothing is fetched or written here -- this only validates, against the
        relevant manifest (per-month for shortest-lead, per-lead for a fixed lead),
        that the requested hours are real data rather than NaN holes (unmirrored,
        or mirrored-and-failed).

        Raises
        ------
        MetOfficeMirrorNotFoundError
            If a requested hour's month/lead was never mirrored, the hour has no
            manifest entry, or its entry is not ``status: "ok"``.
        """
        for t in times:
            manifest = self._manifest(t)
            entry = manifest.get(t.isoformat())
            if entry is None:
                if self.lead_hours is not None:
                    msg = (
                        f"{t.isoformat()} was requested but has no manifest entry for "
                        f"lead {self.lead_hours}h -- mirror.py has not reached this "
                        f"hour. Run: python -m contrailbench.datalib.metoffice.mirror "
                        f"--lead-hours {self.lead_hours} --out-dir {self.out_dir}"
                    )
                else:
                    msg = (
                        f"{t.isoformat()} was requested but has no manifest entry for "
                        f"{t.year:04d}-{t.month:02d} -- mirror.py has not reached this "
                        f"hour. Run: python -m contrailbench.datalib.metoffice.mirror "
                        f"--start {t.year:04d}-{t.month:02d}-01T00:00 --out-dir {self.out_dir}"
                    )
                raise MetOfficeMirrorNotFoundError(msg)
            if entry.get("status") != "ok":
                msg = (
                    f"{t.isoformat()} is recorded with status={entry.get('status')!r} "
                    f"({entry.get('error')}) in {self._manifest_path(t)} -- this hour "
                    "was not captured; its Zarr slot is a NaN hole, not data. Re-run "
                    "mirror.py to retry it."
                )
                raise MetOfficeMirrorNotFoundError(msg)

    def create_cachepath(self, t: datetime) -> str:
        """Return the store path covering ``t``.

        Required by the ``MetDataSource`` ABC, but does not gate anything in this
        datalib's control flow (:attr:`cachestore` is always ``None``, so
        ``is_datafile_cached`` never reaches this). Many timesteps legitimately
        collapse to the same path -- the ABC contract doesn't require a bijection.
        """
        if self.lead_hours is not None:
            return str(self.out_dir / f"lead{self.lead_hours:03d}.zarr")
        return str(self.out_dir / f"{t.year:04d}-{t.month:02d}.zarr")

    def cache_dataset(self, dataset: xr.Dataset) -> None:
        """Not used -- the mirror is the write path, not this datalib.

        Unreachable in this class's control flow (:meth:`download_dataset` returns
        ``None`` rather than datasets to cache). Raises loudly rather than silently
        no-op-ing in case something calls it by mistake.
        """
        del dataset
        msg = (
            "MetOfficeUM does not write to the mirror -- run "
            "contrailbench.datalib.metoffice.mirror to populate it."
        )
        raise NotImplementedError(msg)

    def open_dataset(
        self,
        disk_paths: str | list[str] | pathlib.Path | list[pathlib.Path],
        **xr_kwargs: Any,
    ) -> xr.Dataset:
        """Open one or more monthly Zarr stores as a single dataset."""
        xr_kwargs.setdefault("engine", "zarr")
        xr_kwargs.setdefault("combine", "by_coords")
        return xr.open_mfdataset(disk_paths, **xr_kwargs)

    def open_metdataset(
        self,
        dataset: xr.Dataset | None = None,
        xr_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> MetDataset:
        """Open the requested window as a ``MetDataset``, from the local mirror."""
        if dataset is not None:
            msg = "Parameter 'dataset' is not supported for MetOfficeUM data"
            raise ValueError(msg)

        xr_kwargs = dict(xr_kwargs or {})
        self.download(**xr_kwargs)  # validates manifest coverage; fetches/writes nothing

        if self.lead_hours is not None:
            store_paths = [str(self.out_dir / f"lead{self.lead_hours:03d}.zarr")]
        else:
            months = sorted({(t.year, t.month) for t in self.timesteps})
            store_paths = [
                str(self.out_dir / f"{year:04d}-{month:02d}.zarr") for year, month in months
            ]

        raw_ds = self.open_dataset(store_paths, **xr_kwargs)
        raw_ds = raw_ds.sel(time=self.timesteps)
        if raw_ds.sizes["time"] != len(self.timesteps):
            msg = (
                f"expected {len(self.timesteps)} timesteps after selection, got "
                f"{raw_ds.sizes['time']} -- an hour was silently dropped"
            )
            raise AssertionError(msg)

        mds = self._process(raw_ds, **kwargs)
        self.set_metadata(mds)
        return mds

    def _process(self, raw_ds: xr.Dataset, **kwargs: Any) -> MetDataset:
        """Transform the raw mirrored schema into a pycontrails ``MetDataset``.

        Order matters: the ascending-latitude assertion must run before
        ``MetDataset`` construction, because ``MetDataset.__init__`` (with the
        default ``copy=True``) silently ``sortby``s every coordinate rather than
        erroring on a reversed one -- data mirrored north-south would otherwise
        pass through silently flipped rather than raising, despite this module's
        own ascending-latitude assertion having already run. Likewise, no manual
        transpose/dtype-cast/vertical-coordinate derivation is done here --
        ``MetDataset`` already does all of that.
        """
        latitude = raw_ds["latitude"].values
        if not np.all(np.diff(latitude) > 0):
            msg = (
                "expected ascending latitude ordering in the mirrored store "
                f"({self.out_dir}), got {latitude[:3]}...{latitude[-3:]}"
            )
            raise AssertionError(msg)

        pressure_pa = raw_ds["pressure"].values.astype(np.float64)
        target_pa = np.asarray(self.pressure_levels, dtype=np.float64) * 100.0
        level_indices = []
        for level_hpa, level_pa in zip(self.pressure_levels, target_pa, strict=True):
            matches = np.flatnonzero(np.isclose(pressure_pa, level_pa, atol=1e-3))
            if len(matches) != 1:
                msg = (
                    f"expected exactly one {level_hpa} hPa level in the mirrored "
                    f"store, found {len(matches)}"
                )
                raise ValueError(msg)
            level_indices.append(int(matches[0]))

        ds = raw_ds.isel(pressure=level_indices)
        ds = ds.rename(pressure="level")
        ds = ds.assign_coords(level=ds["level"] / 100.0)

        for required in ("air_temperature", "relative_humidity"):
            if required not in ds:
                msg = f"mirrored store is missing required variable '{required}'"
                raise KeyError(msg)

        # q computed regardless of what was requested -- PCR needs it, and this is
        # the conversion documented in the module docstring (RHi for free via
        # thermo.rhi once q is derived this way).
        level_pa = ds["level"] * 100.0
        q = ds["relative_humidity"] * thermo.q_sat_liquid(ds["air_temperature"], level_pa)
        ds = ds.assign(specific_humidity=q)

        keep = list(self.variable_standardnames)
        ds = ds[keep]

        return MetDataset(ds, **kwargs)

    def set_metadata(self, ds: xr.Dataset | MetDataset) -> None:
        """Set ``provider``/``dataset``/``product`` attrs.

        These values aren't in pycontrails core's hardcoded ``supported`` tuples
        (see module docstring) -- setting them here is metadata only and doesn't
        itself trigger any warning (the warning fires on *read*, via
        ``.provider_attr``/``.dataset_attr``, not on assignment). Callers reading
        those properties -- notably running an ``ISSR``/``SAC``/``PCR`` model
        against this data -- should wrap that in
        :func:`suppress_unregistered_source_warnings`.
        """
        ds.attrs.update(provider=PROVIDER, dataset=DATASET, product=PRODUCT)
