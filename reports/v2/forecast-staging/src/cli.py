"""Command line flags and the run flow common to every shard pipeline.

A pipeline's ``main`` is: build the parser with :func:`shard_parser`, add its own flags, build the
job with :func:`shard_job_for`, and hand the job and its producer to :func:`run`.
"""

import argparse
import sys
from collections.abc import Callable

import pandas as pd

from src import beam, config, shard_job
from src import times as time_utils


def shard_parser(description: str | None = None, *, runner: bool = True) -> argparse.ArgumentParser:
    """Parser with the flags every shard pipeline takes.

    ``runner=False`` leaves out the Beam flags (``--runner``, ``--num-workers``, ``--job-name``),
    for a pipeline that does not run on Beam.
    """
    parser = argparse.ArgumentParser(description=description)
    if runner:
        parser.add_argument("--runner", choices=["direct", "dataflow"], default="direct")
    parser.add_argument("--root", help=f"output root (default: {config.OUTPUT_ROOT})")
    parser.add_argument("--region", choices=sorted(config.REGIONS), default="global")
    parser.add_argument("--start", type=pd.Timestamp, default=config.DEFAULT_START)
    parser.add_argument("--end", type=pd.Timestamp, default=config.DEFAULT_END)
    parser.add_argument("--hours", default="0-23", help="UTC hours, such as 0-23, 15-22 or 0,12")
    parser.add_argument("--force", action="store_true", help="recompute files that already exist")
    parser.add_argument(
        "--skip-failures",
        action="store_true",
        help="log a failing time and carry on; a rerun retries it (default: fail the job)",
    )
    if runner:
        parser.add_argument("--num-workers", type=int, default=1, help="Dataflow workers (fixed)")
        parser.add_argument("--job-name", help="Dataflow job name")
    parser.add_argument(
        "--dry-run", action="store_true", help="say what would run and where, then stop"
    )
    return parser


def shard_job_for(
    args: argparse.Namespace,
    source: str,
    *,
    run: str | None = None,
) -> shard_job.ShardJob:
    """The job writing ``source``'s files, in the directory :func:`config.output_dir` gives."""
    directory = config.output_dir(source, args.region, run=run, root=args.root)
    return shard_job.ShardJob(directory, config.FLIGHT_LEVELS, skip_failures=args.skip_failures)


def run(
    args: argparse.Namespace,
    job: shard_job.ShardJob,
    produce: shard_job.Producer,
    name: str,
    run_job: Callable[..., tuple[int, int]] = shard_job.run,
) -> None:
    """Work out the times, then run ``job`` and say what happened.

    Parameters
    ----------
    name : str
        Short name for the Dataflow job, such as ``hres-00z``.

    run_job : callable
        Runs the job, called as ``run_job(job, produce, times, options, force=...)`` and returning
        ``(already_done, run_now)``. The default is the Beam driver, :func:`shard_job.run`.
    """
    runner = getattr(args, "runner", "direct")
    try:
        times = time_utils.valid_times(args.start, args.end, time_utils.parse_hours(args.hours))
        options = beam.get_pipeline_options(
            runner,
            getattr(args, "job_name", None) or beam.job_name(name, args.region),
            getattr(args, "num_workers", 1),
            harness_threads=getattr(args, "harness_threads", None),
        )
    except ValueError as error:
        sys.exit(str(error))
    if not times:
        sys.exit("the selected period and hours leave no times to process")

    if args.dry_run:
        todo = len(times) if args.force else len(shard_job.pending(job, times))
        print(f"{name}: {todo}/{len(times)} times to run -> {job.output_dir}")
        return
    done, ran = run_job(job, produce, times, options, force=args.force)
    print(f"{name}: {done}/{len(times)} times already complete, {ran} run -> {job.output_dir}")
