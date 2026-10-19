"""Beam utilities."""

from apache_beam.options.pipeline_options import PipelineOptions


def get_pipeline_options(runner: str, id: str | None = None) -> PipelineOptions:
    """Get pipeline options.

    Parameters
    ----------
    runner : str
        Which runner to provide pipeline options for.
        Must be one of "direct" and "dataflow".

    id : str
        Job ID. Only used by Dataflow runner.

    Returns
    -------
    PipelineOptions
        Beam pipeline options

    """
    if runner == "direct":
        return PipelineOptions(
            runner="FnApiRunner",  # don't use Prism
            direct_num_workers=1,
        )

    if runner == "dataflow":
        if not id:
            msg = "Job ID must be provided when using the Dataflow runner."
            raise ValueError(msg)

        return PipelineOptions(
            runner="dataflow",
            job_name=f"contrail-bench-v2-eval-staging-{id}",
            project="contrails-301217",
            region="us-east1",
            temp_location="gs://contrails-301217-tmp-10-day-ttl/contrail-bench/v2-eval-staging/beam-tmp",
            staging_location="gs://contrails-301217-tmp-10-day-ttl/contrail-bench/v2-eval-staging/beam-staging",
            sdk_container_image="us-east1-docker.pkg.dev/contrails-301217/contrail-bench/v2-eval-staging:latest",
            sdk_location="container",
            save_main_session=False,
            service_account_email="contrail-bench-staging-sa@contrails-301217.iam.gserviceaccount.com",
            machine_type="e2-highmem-4",
            autoscaling_algorithm="NONE",
            num_workers=100,
        )

    msg = f"Invalid pipeline option identifier {runner}"
    raise ValueError(msg)
