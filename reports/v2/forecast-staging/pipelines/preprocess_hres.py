"""Preprocess ECMWF HRES (IFS 49r1) forecasts into persistent contrail region (PCR) fields.

Reads the raw ECMWF dissemination GRIB files (model levels) from the delivery bucket, one per
valid time, and interpolates temperature and humidity onto each flight level's pressure with
pycontrails' ``ml_to_pl``. A job for ``--run R`` takes each valid hour from the newest ``R``
initialisation at or before it (see ``src/forecast_runs.py``). That run's file must exist: there
is no fallback to another run.

The delivery bucket is a rolling buffer of about a year, and its file names carry no year, so the
GRIB's own valid time is checked against the requested one.

One time needs about 11 GB of memory, so Dataflow workers run two threads, not one per core
(``--harness-threads``).
"""

import datetime
import functools
import tempfile
from collections.abc import Sequence

import numpy as np
import pandas as pd
import xarray as xr
from google.api_core.exceptions import NotFound
from pycontrails.datalib.ecmwf import HRESModelLevel
from pycontrails.datalib.ecmwf.hres import get_forecast_filename
from pycontrails.datalib.ecmwf.model_levels import ml_to_pl

from src import cli, config, forecast_runs, pcr, storage
from src.grid import Extent

# Pipeline parameters

#: Output directory name under ``forecasts/``
SOURCE = "ecmwf-ifs-hres-49r1"

#: Delivery bucket of the ECMWF HRES dissemination GRIB files (a rolling buffer, not an archive)
DELIVERY_BUCKET = "contrails-301217-ecmwf-hres-delivery-be"

#: Dissemination stream name of the files read
STREAM = "T1"


def fetch_grib(init: datetime.datetime, valid_time: datetime.datetime, sink: str) -> str:
    """Download the dissemination GRIB for one initialisation and valid time.

    Returns
    -------
    str
        The object's name in the delivery bucket.

    Raises
    ------
    FileNotFoundError
        If the file is not in the bucket.
    """
    name = get_forecast_filename(init, valid_time, cc=STREAM)
    try:
        storage.gcs_client().bucket(DELIVERY_BUCKET).blob(name).download_to_filename(sink)
    except NotFound as error:
        msg = f"no HRES file gs://{DELIVERY_BUCKET}/{name} (run {init:%H}Z, valid {valid_time})"
        raise FileNotFoundError(msg) from error
    return name


def check_grib_time(ds: xr.Dataset, valid_time: datetime.datetime, source: str) -> None:
    """Raise if the GRIB's own valid time is not ``valid_time``.

    The delivery bucket's file names carry no year, so a slot can hold another year's data after
    it is overwritten. ``ds`` is as cfgrib opens it: ``valid_time``, or else ``time`` plus ``step``.

    Raises
    ------
    ValueError
        If any valid time in the GRIB differs from ``valid_time``.
    """
    if "valid_time" in ds.coords:
        found = ds["valid_time"]
    else:  # cfgrib's own definition; with no step the field is an analysis, at ``time``
        found = ds["time"] + ds["step"] if "step" in ds.variables else ds["time"]
    expected = np.datetime64(pd.Timestamp(valid_time).to_datetime64(), "ns")
    times = np.unique(found.values.astype("datetime64[ns]"))
    if times.size != 1 or times[0] != expected:
        msg = f"{source} holds valid time {times}, not the requested {valid_time}"
        raise ValueError(msg)


def compute_forecast(
    valid_time: datetime.datetime,
    init: datetime.datetime,
    flight_levels: Sequence[int],
    extent: Extent,
) -> xr.Dataset:
    """PCR fields at every flight level for one valid time, from one GRIB.

    Returns
    -------
    xr.Dataset
        ``rhi``, ``sac`` and ``pcr`` on the benchmark grid with a ``flight_level`` dimension
    """
    pressures = [pcr.target_pressure_hpa(fl) for fl in flight_levels]

    # HRESModelLevel only accepts a 00Z or 12Z forecast_time, but a run can be 06Z or 18Z. It is
    # used here only for its processing methods, which do not read forecast_time, so a placeholder
    # (the valid time floored to 00Z or 12Z) satisfies the check. The real initialisation is
    # recorded on the output.
    placeholder = valid_time.replace(minute=0, second=0, microsecond=0)
    placeholder -= datetime.timedelta(hours=placeholder.hour % 12)
    hres = HRESModelLevel(
        time=valid_time,
        forecast_time=placeholder,
        variables=["air_temperature", "specific_humidity"],
        pressure_levels=pressures,
    )

    with tempfile.TemporaryDirectory() as tmp:
        raw = f"{tmp}/hres.grib"
        name = fetch_grib(init, valid_time, raw)

        # An unfiltered open fails because the scalar hybrid coordinate of ``lnsp`` (log surface
        # pressure) collides with the 28-level one, so the file is opened once per key. cfgrib
        # logs that collision as an error while it builds the "hybrid" group, then recovers by
        # dropping ``lnsp``, which the second open reads. The logged error is expected.
        ds_ml = xr.open_dataset(raw, engine="cfgrib", filter_by_keys={"typeOfLevel": "hybrid"})
        ds_lnsp = xr.open_dataset(raw, engine="cfgrib", filter_by_keys={"shortName": "lnsp"})
        check_grib_time(ds_ml, valid_time, name)
        ds_ml = ds_ml.rename(hybrid="model_level")
        ds_pl = ml_to_pl(ds_ml, target_pl=pressures, lnsp=ds_lnsp["lnsp"])

        # ml_to_pl drops the scalar ``time`` that cfgrib gives a single-message file, and the
        # processing below needs ``time`` as a dimension
        if "time" not in ds_pl.dims:
            ds_pl = ds_pl.expand_dims(time=[pd.Timestamp(valid_time).to_datetime64()])

        # Process as open_metdataset would: select variables, times and levels, then standardize
        # the variable names. The GRIB is global, so crop afterwards.
        met = hres._process_dataset(ds_pl)
        hres.set_metadata(met)
        lon_min, lon_max, lat_min, lat_max = extent
        met = met.downselect(bbox=[lon_min, lat_min, lon_max, lat_max])  # west, south, east, north
        ds = met.data.load()

    # The data is already on exact target pressures (from ml_to_pl), hence "pressure"
    return pcr.regrid_and_derive(ds, flight_levels, extent, humidity_correction="pressure")


def preprocess_hres(
    valid_time: datetime.datetime, flight_levels: Sequence[int], *, run: str, extent: Extent
) -> dict[int, xr.Dataset]:
    """One shard per flight level for one valid time, from the ``run`` initialisation."""
    init = forecast_runs.base_run(valid_time, run)
    batched = compute_forecast(valid_time, init, flight_levels, extent)
    return {
        fl: forecast_runs.record_init(shard, init, valid_time)
        for fl, shard in pcr.split_flight_levels(batched, flight_levels).items()
    }


def main() -> None:
    """Program entrypoint."""
    parser = cli.shard_parser(__doc__)
    parser.add_argument("--run", type=forecast_runs.parse_run, required=True, help="00Z, 06Z, ...")
    parser.add_argument(
        "--harness-threads", type=int, default=2, help="threads per Dataflow worker"
    )
    args = parser.parse_args()

    produce = functools.partial(preprocess_hres, run=args.run, extent=config.REGIONS[args.region])
    job = cli.shard_job_for(args, SOURCE, run=args.run)
    cli.run(args, job, produce, f"hres-{args.run}")


if __name__ == "__main__":
    main()
