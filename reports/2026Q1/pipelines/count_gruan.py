"""Benchmark Contrails.org forecast using IAGOS observations."""

import argparse
import datetime
import itertools

import apache_beam as beam
import gcsfs
import pandas as pd
from apache_beam.options.pipeline_options import PipelineOptions


# Pipeline parameters

#: Forecast times
TIMES = pd.date_range("2024-01-01 00:00", "2024-12-31 23:00", freq="1h").to_pydatetime().tolist()

#: Forecast flight levels
FLIGHT_LEVELS = list(range(270, 450, 10))

#: GCP buckets for temporary Beam files
BEAM_TEMP = "gs://contrails-301217-tmp-10-day-ttl/contrail-bench/2026Q1/beam-tmp"
BEAM_STAGING = "gs://contrails-301217-tmp-10-day-ttl/contrail-bench/2026Q1/beam-staging"

#: GCP buckets for temporary assets
GCP_OBS_TMPDIR = "gs://contrails-301217-contrail-bench/2026Q1/gruan"


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
            job_name="contrail-bench-2025q1-count-gruan",
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


def count_observations(time: datetime.datetime, flight_level: int) -> list[tuple[str, int]]:
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
    pcr_count = (df["pcr_count"] > 0).sum()
    total_count = len(df)
    return [("pcr", pcr_count), ("total", total_count)]


def write_to_gcs(key: str, count: int) -> None:
    """Write counts to GCS."""
    fs = gcsfs.GCSFileSystem()
    with fs.open(f"{GCP_OBS_TMPDIR}/{key}_count.txt", "w") as f:
        f.write(f"{count}")


def main() -> None:
    """Program entrypoint."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=str, required=True)
    args = parser.parse_args()
    
    options = get_pipeline_options(args.runner)
    pcoll = itertools.product(TIMES, FLIGHT_LEVELS)

    # compute metrics
    with beam.Pipeline(options=options) as pipeline:
        (
            pipeline
            | "Create PCollection" >> beam.Create(pcoll)
            | "Count observations" >> beam.FlatMapTuple(count_observations)
            | "Calculate sum" >> beam.CombinePerKey(sum)
            | "Write to GCS" >> beam.MapTuple(write_to_gcs)
        )



if __name__ == "__main__":
    main()

