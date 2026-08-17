"""Shared CLI configuration for the materialize/evaluate pipelines."""

import os
import pathlib

from apache_beam.options.pipeline_options import PipelineOptions

#: Local data root. Not GCS: publishing to the public bucket is licence-gated
#: regardless, and per-file GCS upload latency made a full-window run's cost
#: prohibitive in this environment. A later bulk `gcloud storage cp -r -m`
#: handles moving this to GCS once/if that's actually needed.
#:
#: Overridable via ``CONTRAILBENCH_DATA_DIR`` -- this package can be checked
#: out as a git worktree separate from wherever the (gitignored, multi-GB)
#: data directory actually lives, so the path can't always be derived from
#: this file's own location.
DATA_DIR = pathlib.Path(
    os.environ.get(
        "CONTRAILBENCH_DATA_DIR",
        str(pathlib.Path(__file__).resolve().parents[2] / "reports" / "jay_extension" / "data"),
    )
)

BEAM_TEMP = "gs://contrails-301217-tmp-10-day-ttl/contrail-bench/2026Q1/beam-tmp"
BEAM_STAGING = "gs://contrails-301217-tmp-10-day-ttl/contrail-bench/2026Q1/beam-staging"


def pipeline_options(runner: str, job_name: str) -> PipelineOptions:
    """Beam pipeline options for the ``"direct"`` or ``"dataflow"`` runner.

    Parameters
    ----------
    runner : str
        Must be one of ``"direct"`` and ``"dataflow"``.

    job_name : str
        Dataflow job name. Ignored for the direct runner.

    Returns
    -------
    PipelineOptions
    """
    if runner == "direct":
        return PipelineOptions(runner="direct", direct_num_workers=1)

    if runner == "dataflow":
        return PipelineOptions(
            runner="dataflow",
            job_name=job_name,
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
            num_workers=100,
        )

    msg = f"Invalid pipeline option identifier {runner}"
    raise ValueError(msg)


def parse_hours(hours: str | None) -> tuple[int, ...] | None:
    """Parse a comma-separated ``--hours`` flag, e.g. ``"15,16,17"``."""
    return tuple(int(x) for x in hours.split(",")) if hours else None
