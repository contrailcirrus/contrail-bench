"""Beam pipeline options for local and Dataflow runs, and the Dataflow settings they use."""

import re

from apache_beam.options.pipeline_options import PipelineOptions

from src import config

REGION = "us-east1"
SDK_IMAGE = "us-east1-docker.pkg.dev/contrails-301217/contrail-bench/v2-forecast-staging:latest"
SERVICE_ACCOUNT = "contrail-bench-staging-sa@contrails-301217.iam.gserviceaccount.com"
MACHINE_TYPE = "e2-highmem-4"

#: GCP buckets for temporary Beam files
TEMP_LOCATION = "gs://contrails-301217-tmp-10-day-ttl/contrail-bench/v2-forecast-staging/beam-tmp"
STAGING_LOCATION = (
    "gs://contrails-301217-tmp-10-day-ttl/contrail-bench/v2-forecast-staging/beam-staging"
)

#: Upper bound on ``--num-workers``, as a guard against an expensive typo
MAX_WORKERS = 20

#: Dataflow job names are at most this long
MAX_JOB_NAME_LENGTH = 60


def job_name(*parts: str) -> str:
    """A valid Dataflow job name such as ``contrail-bench-v2-forecast-staging-hres-00z-global``.

    Lower-case letters, digits and single hyphens only, as Dataflow requires.
    """
    text = "-".join(["contrail-bench-v2-forecast-staging", *parts]).lower()
    return re.sub(r"[^a-z0-9]+", "-", text).strip("-")[:MAX_JOB_NAME_LENGTH].strip("-")


def get_pipeline_options(
    runner: str,
    name: str | None = None,
    num_workers: int = 1,
    *,
    harness_threads: int | None = None,
) -> PipelineOptions:
    """Get pipeline options.

    Parameters
    ----------
    runner : str
        Which runner to provide pipeline options for. Must be one of "direct" and "dataflow".

    name : str, optional
        Dataflow job name (see :func:`job_name`). Required by the Dataflow runner.

    num_workers : int
        Fixed number of Dataflow workers (no autoscaling), at most ``MAX_WORKERS``.

    harness_threads : int, optional
        Threads per Dataflow worker. Unset uses Dataflow's default (one per core); memory-hungry
        jobs can run out of memory unless this is lowered.

    Returns
    -------
    PipelineOptions
        Beam pipeline options
    """
    if runner == "direct":
        # Beam's default local runner is Prism, which does not run these pipelines reliably
        return PipelineOptions(flags=[], runner="FnApiRunner", direct_num_workers=1)

    if runner == "dataflow":
        if name is None:
            msg = "a Dataflow run needs a job name"
            raise ValueError(msg)
        if not 1 <= num_workers <= MAX_WORKERS:
            msg = f"--num-workers must be between 1 and {MAX_WORKERS}"
            raise ValueError(msg)
        options = {
            "runner": "dataflow",
            "job_name": name,
            "project": config.PROJECT,
            "region": REGION,
            "temp_location": TEMP_LOCATION,
            "staging_location": STAGING_LOCATION,
            "sdk_container_image": SDK_IMAGE,
            "sdk_location": "container",
            "save_main_session": False,
            "service_account_email": SERVICE_ACCOUNT,
            "machine_type": MACHINE_TYPE,
            "autoscaling_algorithm": "NONE",
            "num_workers": num_workers,
        }
        if harness_threads:
            options["number_of_worker_harness_threads"] = harness_threads
        return PipelineOptions(flags=[], **options)

    msg = f"Invalid pipeline option identifier {runner}"
    raise ValueError(msg)
