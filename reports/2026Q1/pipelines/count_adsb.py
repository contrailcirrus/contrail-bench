"""Count cells with ADSB data."""

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
GCP_ADSB_TMPDIR = "gs://contrails-301217-contrail-bench/tmp/2026Q1/adsb"


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
            job_name="contrail-bench-2025q1-count-adsb",
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


def count_adsb(time: datetime.datetime, flight_level: int) -> list[tuple[str, int]]:
    """Count ADSB cells.

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
    gcs_path = f"{GCP_ADSB_TMPDIR}/{int(time.timestamp())}_{flight_level}.pq"
    df = pd.read_parquet(gcs_path)
    total = len(df)
    if time >= datetime.datetime(2024, 5, 1, 0, 0):
        goes = (df["longitude"].between(-135, -30) & df["latitude"].between(-50, 50)).sum()
    else:
        goes = 0
        
    total_flight_m = df["flight_distance"].sum().item()
    geos_invis_flight_m_north = df[df["latitude"] > 66.5]["flight_distance"].sum().item()
    geos_invis_flight_m_south = df[df["latitude"] < -66.5]["flight_distance"].sum().item()
    geos_invis_flight_m_other = df[df["latitude"].between(-66.5, 66.5) & df["longitude"].between(69.0, 86.0)]["flight_distance"].sum().item()

    return [
        ("goeseast", goes),
        ("total", total),
        ("total_flight_m", total_flight_m),
        ("geos_invis_flight_m_north", geos_invis_flight_m_north),
        ("geos_invis_flight_m_south", geos_invis_flight_m_south),
        ("geos_invis_flight_m_other", geos_invis_flight_m_other),
    ]


def write_to_gcs(key: str, count: int) -> None:
    """Write counts to GCS."""
    fs = gcsfs.GCSFileSystem()
    with fs.open(f"{GCP_ADSB_TMPDIR}/{key}_count.txt", "w") as f:
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
            | "Count ADSB cells" >> beam.FlatMapTuple(count_adsb)
            | "Calculate sum" >> beam.CombinePerKey(sum)
            | "Write to GCS" >> beam.MapTuple(write_to_gcs)
        )



if __name__ == "__main__":
    main()

