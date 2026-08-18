"""Adapter from the new netCDF evaluation output to the pre-port long-format schema.

The analysis notebooks and ``compile_benchmarks.py`` consume per-hour
long-format DataFrames (one row per (time, flight_level, rhi_threshold)) with
columns like ``observed_pcr_area_in_forecast_pcr`` -- exactly what the old
per-source benchmark scripts wrote directly as parquet. ``Forecast.evaluate``/
``evaluate_beam`` instead produce a wide ``xr.Dataset`` (metric-prefixed
variables over ``time``/``flight_level``/``rhi_threshold`` dims). This module
is the one place that reshapes the new format back into the old one, so
downstream notebook/compile-script cells only need to change their loader
call, not their own logic.
"""

import pandas as pd
import xarray as xr


def to_long_format(ds: xr.Dataset, metric: str) -> pd.DataFrame:
    """Reshape one metric's output into the pre-port per-hour long format.

    Parameters
    ----------
    ds : xr.Dataset
        Output of :meth:`contrailbench.forecast.Forecast.evaluate` or a day
        file written by :meth:`contrailbench.forecast.Forecast.evaluate_beam`
        (e.g. via ``xr.open_dataset``). Must contain variables prefixed
        ``f"{metric}."``, over ``time``/``flight_level``/``rhi_threshold``
        dims (and a ``coverage`` variable, if present).

    metric : str
        The metric name prefix to extract, e.g. ``"iagos"``.

    Returns
    -------
    pd.DataFrame
        One row per (time, flight_level, rhi_threshold), with the metric
        prefix stripped from column names -- matching the schema the old
        per-source benchmark scripts wrote directly (see
        ``metoffice_iagos_contrailwatch_region.py::calculate_metrics``).
        Rows with ``coverage == 0`` (a shard that was missing and NaN-filled --
        see :func:`contrailbench.forecast._concatenate_fl_time`) are dropped,
        matching the old pipelines' silent-skip behavior for missing forecasts.
    """
    prefix = f"{metric}."
    metric_vars = [v for v in ds.data_vars if str(v).startswith(prefix)]
    if not metric_vars:
        msg = f"no variables found with prefix {prefix!r} in dataset"
        raise ValueError(msg)

    subset = ds[metric_vars]
    if "coverage" in ds.data_vars:
        subset = subset.where(ds["coverage"] == 1)

    df = subset.to_dataframe().reset_index()
    df = df.rename(columns={v: str(v).removeprefix(prefix) for v in metric_vars})
    df = df.dropna(subset=[str(v).removeprefix(prefix) for v in metric_vars], how="all")

    sort_cols = [c for c in ("time", "flight_level", "rhi_threshold") if c in df.columns]
    return (
        df.sort_values(sort_cols).reset_index(drop=True) if sort_cols else df.reset_index(drop=True)
    )
