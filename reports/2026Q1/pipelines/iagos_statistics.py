"""Benchmark Contrails.org forecast using IAGOS observations."""

import argparse
import datetime
import itertools

import apache_beam as beam
import pandas as pd
from apache_beam.options.pipeline_options import PipelineOptions


# Pipeline parameters

#: Forecast times
TIMES = pd.date_range("2024-01-01 00:00", "2024-12-31 01:00", freq="1h").to_pydatetime().tolist()

#: Forecast flight levels
FLIGHT_LEVELS = list(range(270, 450, 10))

#: GCP buckets for temporary Beam files
BEAM_TEMP = "gs://contrails-301217-tmp-10-day-ttl/contrail-bench/2026Q1/beam-tmp"
BEAM_STAGING = "gs://contrails-301217-tmp-10-day-ttl/contrail-bench/2026Q1/beam-staging"

#: GCP buckets for temporary assets
GCP_OBS_TMPDIR = "gs://contrails-301217-contrail-bench/2026Q1/iagos"
GCP_ADSB_TMPDIR = "gs://contrails-301217-contrail-bench/2026Q1/adsb"

#: GCP bucket for permanent assets
GCP_ASSETS = "gs://contrails-301217-contrail-bench/2026Q1/iagos-statistics"



def get_pipeline_options(runner: str) -> PipelineOptions:
    """Get pipeline options.

    Parameters
    ----------
    runner : str
        Which runner to provide pipeline options for.
        Must be one of "direct" and "dataflow".

    Returns
    -------
    PipelineOptions
        Beam pipeline options

    """
    if runner == "direct":
        return PipelineOptions(
            runner="direct",
            direct_num_workers=1
        )

    if runner == "dataflow":
        return PipelineOptions(
            runner="dataflow",
            job_name="contrail-bench-2025q1-iagos-statistics",
            project="contrails-301217",
            region="us-east1",
            temp_location=BEAM_TEMP,
            staging_location=BEAM_STAGING,
            sdk_container_image="us-east1-docker.pkg.dev/contrails-301217/contrail-bench/2025q1:latest",
            sdk_location="container",
            save_main_session=False,
            service_account_email="contrail-bench-staging-sa@contrails-301217.iam.gserviceaccount.com",
            machine_type="e2-highmem-4",
            autoscaling_algorithm="NONE",
            num_workers=100
        )

    msg = f"Invalid pipeline option identifier {runner}"
    raise ValueError(msg)


def open_adsb(time: datetime.datetime, flight_level: int) -> pd.DataFrame:
    """Open preprocessed ADSB flight distance.

    Parameters
    ----------
    time : datetime.datetime
        Target time

    flight_level : int
        Target flight level

    Return
    ------
    pd.DataFrame
        Preprocessed ADSB flight distance

    """
    gcs_path = f"{GCP_ADSB_TMPDIR}/{int(time.timestamp())}_{flight_level}.pq"
    return pd.read_parquet(gcs_path)


def count_observations(time: datetime.datetime, flight_level: int, extent: list[float] | None = None) -> tuple[str, dict]:
    """Count observations.

    Parameters
    ----------
    time : datetime.datetime
        Target time

    flight_level : int
        Target flight level

    Return
    ------
    list[tuple[str, int]]
        Labeled counts

    """
    gcs_path = f"{GCP_OBS_TMPDIR}/{int(time.timestamp())}_{flight_level}.pq"
    df = pd.read_parquet(gcs_path)

    if extent is not None:
        lon_min, lon_max, lat_min, lat_max = extent
        df = df[
            df["longitude"].between(lon_min, lon_max) & 
            df["latitude"].between(lat_min, lat_max)
        ]

    pcr_count = (df["pcr_distance"] > 0).sum()
    total_count = len(df)

    adsb = open_adsb(time, flight_level)
    adsb = adsb.set_index(["longitude", "latitude"])
    pcr_flight_m = adsb.reindex(df[df["pcr_distance"] > 0].set_index(["longitude", "latitude"]).index).sum().item()
    total_flight_m = adsb.reindex(df.set_index(["longitude", "latitude"]).index).sum().item()

    record = {
        "pcr_grid_cells": pcr_count,
        "total_grid_cells": total_count,
        "pcr_flight_m": pcr_flight_m,
        "total_flight_m": total_flight_m,
    }
    key = time.strftime("%Y%m%d")
    return (key, record)


def aggregate_statistics(key: str, records: list[dict]) -> tuple[str, dict]:
    """Aggregate statistics per day."""
    record = {
        k: sum(record[k] for record in records)
        for k in ["pcr_grid_cells", "total_grid_cells", "pcr_flight_m", "total_flight_m"]
    }
    return key, record


def combine_tuple_or_df(statistics: list[tuple[str, dict] | pd.DataFrame]) -> pd.DataFrame:
    """Combine statistics into a single dataframe.

    Note that statistics may be combined in multiple stages,
    so this function must handle tuples provided by the previous
    step as well as partial DataFrames containing multiple tuples.

    """
    df_list = []
    for item in statistics:
        if isinstance(item, pd.DataFrame):
            df_list.append(item)
            continue
        
        idx_str, data = item
        idx = pd.to_datetime(idx_str, format="%Y%m%d")
        df_list.append(pd.DataFrame(data, index=[idx]))

    return pd.concat(df_list, axis="index")


def write_to_gcs(df: pd.DataFrame, name: str) -> None:
    """Write results to GCS."""
    df = df.sort_index()
    sink = f"{GCP_ASSETS}/{name}.pq"
    df.to_parquet(sink)


def main() -> None:
    """Program entrypoint."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=str, required=True)
    args = parser.parse_args()
    
    options = get_pipeline_options(args.runner)
    pcoll = itertools.product(TIMES, FLIGHT_LEVELS)

    # compute metrics
    with beam.Pipeline(options=options) as pipeline:

        p1 = pipeline | "Create PCollection" >> beam.Create(pcoll)

        # Global
        (
            p1
            | "Count observations (global)" >> beam.MapTuple(count_observations)
            | "Group by day (global)" >> beam.GroupByKey()
            | "Aggregate statistics (global)" >> beam.MapTuple(aggregate_statistics)
            | "Combine (global)" >> beam.CombineGlobally(combine_tuple_or_df)
            | "Write to GCS (global)" >> beam.Map(write_to_gcs, name="daily")
        )

        # ContrailWatch Region
        (
            p1
            | "Count observations (contrailwatch)" >> beam.MapTuple(count_observations, extent=[-134, -63, 20, 50])
            | "Group by day (contrailwatch)" >> beam.GroupByKey()
            | "Aggregate statistics (contrailwatch)" >> beam.MapTuple(aggregate_statistics)
            | "Combine (contrailwatch)" >> beam.CombineGlobally(combine_tuple_or_df)
            | "Write to GCS (contrailwatch)" >> beam.Map(write_to_gcs, name="daily-contrailwatch")
        )


if __name__ == "__main__":
    main()

