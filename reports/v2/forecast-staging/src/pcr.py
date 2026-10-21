"""PCR derivation shared by every pipeline that reads a forecast or reanalysis.

Every forecast source goes through this one module, so a comparison between sources isolates
the weather data and not the derivation. The order of operations is fixed:

1. Get ``air_temperature`` and ``specific_humidity`` on the exact pressure of each flight
   level, at native horizontal resolution. Sources that expose only a few pressure levels use
   :func:`interpolate_to_pressures` to get there.
2. :func:`src.grid.regrid` conservatively (area-weighted) regrids those two
   continuous fields onto the 0.25 degree benchmark grid.
3. RHi and the Schmidt-Appleman criterion (SAC) are derived *after* regridding. Averaging a binary
   outcome would give a fractional value with no principled cutoff, whereas averaging the
   continuous fields first keeps ``sac`` genuinely binary.

No threshold is applied beyond ``pcr`` (RHi above 1): ``sac`` does not depend on one and ``rhi`` is
continuous, so downstream code can flag ``(rhi > threshold) & sac`` for any threshold.
"""

import warnings
from collections.abc import Collection, Sequence

import numpy as np
import xarray as xr
from pycontrails.core.met import MetDataset
from pycontrails.models import sac
from pycontrails.models.humidity_scaling import (
    ExponentialBoostLatitudeCorrectionHumidityScaling,
    HumidityScaling,
)
from pycontrails.physics import thermo, units

from src import config, grid
from src.grid import Extent


def target_pressure_hpa(flight_level: int) -> int:
    """ISA pressure of a flight level (hundreds of feet), rounded to the nearest hPa.

    Rounded because pycontrails' model-level-to-pressure-level interpolation only returns
    whole-hPa pressures. Using the same rounded value as every source's vertical target keeps
    all arms evaluated at identical pressures, not off by up to 0.5 hPa.
    """
    return round(float(units.ft_to_pl(flight_level * 100.0)))


def bracketing_level_indices(levels_hpa: np.ndarray, targets_hpa: Collection[float]) -> list[int]:
    """Indices of the source levels needed to interpolate (log-pressure) onto the targets.

    Only these need to be read, which can be a small part of a large store.

    Raises
    ------
    ValueError
        If the levels are not strictly ascending, or do not bracket every target.
    """
    levels = np.asarray(levels_hpa, dtype=np.float64)
    if levels.ndim != 1 or len(levels) < 2 or not np.all(np.diff(levels) > 0):
        msg = f"pressure levels must be strictly ascending: {levels}"
        raise ValueError(msg)
    if min(targets_hpa) < levels[0] or max(targets_hpa) > levels[-1]:
        msg = (
            f"pressure levels {levels[0]} to {levels[-1]} hPa do not bracket the targets "
            f"{min(targets_hpa)} to {max(targets_hpa)} hPa"
        )
        raise ValueError(msg)
    indices: set[int] = set()
    for target in targets_hpa:
        indices.add(int(np.searchsorted(levels, target, side="right") - 1))
        indices.add(int(np.searchsorted(levels, target, side="left")))
    return sorted(indices)


def interpolate_to_pressures(ds: xr.Dataset, target_pressures_hpa: Sequence[float]) -> xr.Dataset:
    """Interpolate (log-linear in pressure) onto several target pressures in one call.

    For sources that expose only a few fixed pressure levels. Linear interpolation is
    elementwise per query point, so one vectorized call equals one call per target.

    Coordinates other than ``level`` that depend on it (a ``MetDataset``'s ``air_pressure`` and
    ``altitude``, computed from the source levels) are dropped first. Interpolating them would
    leave ``air_pressure`` above the target pressure, and RHi, which is proportional to pressure,
    too high by the same fraction (about 0.1% to 0.2% on 25 hPa levels). Wrapping the result in a
    ``MetDataset`` again recomputes them from the target levels.

    Parameters
    ----------
    ds : xarray.Dataset
        Has a ``level`` coordinate in hPa spanning every target pressure.
    target_pressures_hpa : list[float]
        Target pressures in hPa.

    Returns
    -------
    xarray.Dataset
        ``ds`` with ``level`` resampled to ``target_pressures_hpa``.
    """
    attrs = ds.attrs
    log_level = np.log(ds["level"].values)
    log_targets = np.log(np.asarray(target_pressures_hpa, dtype=np.float64))
    stale = [name for name, c in ds.coords.items() if "level" in c.dims and name != "level"]
    interpolated = ds.drop_vars(stale).assign_coords(level=log_level).interp(level=log_targets)
    return interpolated.assign_coords(level=list(target_pressures_hpa)).assign_attrs(attrs)


def _compute_rhi_and_sac(
    coarse_met: MetDataset, humidity_scaling: HumidityScaling | None = None
) -> tuple[xr.DataArray, xr.DataArray]:
    """Compute RHi (continuous) and SAC (binary, threshold-independent).

    Call this *after* regridding. If ``humidity_scaling`` is given it is applied to
    ``coarse_met`` first, modifying its ``specific_humidity`` in place, so both RHi and SAC
    reflect it.

    Parameters
    ----------
    coarse_met : pycontrails.MetDataset
        Carries ``air_temperature`` and ``specific_humidity`` and the source's ``provider``,
        ``dataset`` and ``product`` attributes, which regridding does not preserve.
    humidity_scaling : pycontrails.models.humidity_scaling.HumidityScaling, optional
        Bias correction to apply before deriving RHi and SAC.

    Returns
    -------
    tuple[xarray.DataArray, xarray.DataArray]
        ``(rhi, sac)``.
    """
    if humidity_scaling is not None:
        humidity_scaling.eval(coarse_met, copy_source=False)

    # Running a pycontrails model on a source it does not list (for example Met Office data)
    # warns about it, which is expected and not a problem here
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message="Unknown provider")
        warnings.filterwarnings("ignore", category=UserWarning, message="Unknown dataset")
        # SAC is given no humidity scaling: it applies one it is handed to its own copy of the
        # humidity, and `coarse_met` is already scaled above, so passing it would scale twice
        sac_result = sac.SAC(
            coarse_met, params=sac.SACParams(engine_efficiency=config.ENGINE_EFFICIENCY)
        ).eval()

    rhi = thermo.rhi(
        coarse_met["specific_humidity"].data,
        coarse_met["air_temperature"].data,
        coarse_met["air_pressure"].data,
    )
    return rhi, sac_result["sac"].data


def regrid_and_derive(
    ds: xr.Dataset,
    flight_levels: Sequence[int],
    extent: Extent,
    humidity_correction: str | None = None,
) -> xr.Dataset:
    """Regrid and derive PCR fields for many flight levels from one met fetch.

    One fetch covering every pressure level serves all flight levels, instead of one fetch
    per flight level. ``ds`` must already carry a ``level`` dimension holding each flight
    level's target pressure (:func:`target_pressure_hpa`), in any order.

    Levels are matched to flight levels by pressure *value*, not position. pycontrails'
    ``MetDataset`` sorts ``level`` ascending by pressure, which is descending flight level,
    so the incoming order cannot be trusted to still match ``flight_levels`` afterwards.

    Parameters
    ----------
    ds : xarray.Dataset
        ``air_temperature`` and ``specific_humidity`` on a ``level`` dimension (hPa) of size
        ``len(flight_levels)``, at native horizontal resolution, on ascending, uniformly spaced
        ``longitude`` and ``latitude`` coordinates.
    flight_levels : sequence of int
        Flight levels covered. Their order becomes the order of the output ``flight_level``
        dimension.
    extent : tuple[float, float, float, float]
        ``(lon_min, lon_max, lat_min, lat_max)`` of the output region.
    humidity_correction : str, optional
        Correct humidity with pycontrails' latitude-dependent extension of the Teoh et al. 2022
        bias correction. The value is the correction's ``level_type``: ``"model"`` for data
        derived from model levels, ``"pressure"`` for data on exact pressure levels. ``None``
        applies no correction.

    Returns
    -------
    xarray.Dataset
        ``rhi``, ``sac`` and ``pcr`` (RHi above 1 where SAC holds) on the benchmark grid with
        a ``flight_level`` dimension. The pressures remain as a ``level`` coordinate along
        ``flight_level``. Use :func:`split_flight_levels` for single-level files.
    """
    regridded = xr.Dataset(
        {
            "air_temperature": grid.regrid(ds["air_temperature"], extent),
            "specific_humidity": grid.regrid(ds["specific_humidity"], extent),
        }
    ).assign_attrs(ds.attrs)

    # A new instance per call: ``eval`` stores its source on the instance, which would race
    # between concurrent worker threads sharing one
    humidity_scaling = (
        ExponentialBoostLatitudeCorrectionHumidityScaling(
            params={"level_type": humidity_correction}
        )
        if humidity_correction
        else None
    )
    rhi, sac_field = _compute_rhi_and_sac(MetDataset(regridded), humidity_scaling)
    out = xr.Dataset(
        {
            "rhi": rhi.astype("float32"),
            "sac": sac_field.astype("float32"),
            "pcr": ((rhi > 1.0) & (sac_field > 0)).astype("float32"),
        }
    )
    out["rhi"].attrs = {"long_name": "Relative humidity over ice", "units": "1"}
    out["sac"].attrs = {"long_name": "Schmidt-Appleman criterion satisfied flag", "units": "1"}
    out["pcr"].attrs = {
        "long_name": "Persistent contrail region flag (rhi_threshold=1.0 default)",
        "units": "1",
    }

    pressures = [target_pressure_hpa(fl) for fl in flight_levels]
    out = out.sel(level=pressures).rename(level="flight_level")
    return out.assign_coords(flight_level=list(flight_levels), level=("flight_level", pressures))


def split_flight_levels(batched: xr.Dataset, flight_levels: Sequence[int]) -> dict[int, xr.Dataset]:
    """One dataset per flight level from :func:`regrid_and_derive`'s output.

    Gives the single-level schema the per-(time, flight level) files use: the dimension is
    renamed back to ``level`` (size 1, holding that level's pressure).
    """
    shards = {}
    for flight_level in flight_levels:
        one = batched.sel(flight_level=[flight_level])
        pressure = float(one["level"].item())
        shards[flight_level] = (
            one.drop_vars("level").rename(flight_level="level").assign_coords(level=[pressure])
        )
    return shards
