"""Jobs that write one netCDF file per (time, flight level): the shape of every shard pipeline.

A job is a directory of ``{epoch}_{flight_level}.nc`` files and a *producer*: a callable that,
given a valid time and the flight levels, returns the dataset for each file. The driver does the
rest: it skips times that are already complete (so an interrupted run resumes), runs the rest in a
Beam pipeline, and writes every file atomically. All data access lives in the producer, so a new
source only has to say how to build one time's files.
"""

import contextlib
import dataclasses
import datetime
import logging
from collections.abc import Callable, Iterator, Mapping, Sequence

import apache_beam as beam
import xarray as xr
from apache_beam.options.pipeline_options import PipelineOptions

from src import storage
from src import times as time_utils

LOGGER = logging.getLogger(__name__)

#: Builds one time's files: ``produce(time, flight_levels) -> {flight_level: dataset}``
Producer = Callable[[datetime.datetime, Sequence[int]], Mapping[int, xr.Dataset]]


@dataclasses.dataclass(frozen=True)
class ShardJob:
    """Where a job writes, and what it writes."""

    #: Output directory (local, or a ``gs://`` prefix)
    output_dir: str
    flight_levels: tuple[int, ...]
    #: Log a time that fails and carry on, instead of failing the job. Nothing is written for it,
    #: so it is picked up the next time the job runs. Without this, one bad hour fails the whole
    #: job once Beam's retries are used up, discarding the wait on every other hour.
    skip_failures: bool = False

    def path(self, time: datetime.datetime, flight_level: int) -> str:
        """The file for one time and flight level."""
        return f"{self.output_dir}/{time_utils.utc_epoch(time)}_{flight_level}.nc"


def pending(job: ShardJob, times: Sequence[datetime.datetime]) -> list[datetime.datetime]:
    """The times that still lack a file for at least one flight level.

    A time counts as done only if every flight level's file exists, so a partially written time is
    redone in full. This is what makes an interrupted run restartable.
    """
    existing = set(storage.list_names(job.output_dir))
    return [
        t
        for t in times
        if not all(f"{time_utils.utc_epoch(t)}_{fl}.nc" in existing for fl in job.flight_levels)
    ]


def write_shards(job: ShardJob, time: datetime.datetime, shards: Mapping[int, xr.Dataset]) -> None:
    """Write one time's files, atomically.

    Raises
    ------
    ValueError
        If ``shards`` does not hold exactly the job's flight levels. Nothing is written then.
    """
    if set(shards) != set(job.flight_levels):
        msg = (
            f"producer returned flight levels {sorted(shards)}, "
            f"expected {sorted(job.flight_levels)}"
        )
        raise ValueError(msg)
    for flight_level, dataset in shards.items():
        storage.write_netcdf_atomic(dataset, job.path(time, flight_level))


@contextlib.contextmanager
def skipping_failures(job: ShardJob, what: object) -> Iterator[None]:
    """Run the block; if it fails and the job skips failures, log that and carry on."""
    try:
        yield
    except Exception:
        if not job.skip_failures:
            raise
        LOGGER.exception("failed for %s; left unwritten so a rerun retries it", what)


class _WriteTime(beam.DoFn):
    """Beam wrapper: produce one time's files and write them."""

    def __init__(self, job: ShardJob, produce: Producer) -> None:
        self.job = job
        self.produce = produce

    def process(self, element: datetime.datetime) -> None:
        with skipping_failures(self.job, element):
            write_shards(self.job, element, self.produce(element, self.job.flight_levels))


def run(
    job: ShardJob,
    produce: Producer,
    times: Sequence[datetime.datetime],
    options: PipelineOptions,
    *,
    force: bool = False,
) -> tuple[int, int]:
    """Run ``job`` over ``times``, skipping the ones already done unless ``force``.

    Returns
    -------
    tuple[int, int]
        ``(already_done, run_now)``.
    """
    todo = list(times) if force else pending(job, times)
    if todo:
        with beam.Pipeline(options=options) as pipeline:
            (
                pipeline
                | "Create times" >> beam.Create(todo)
                | "Write shards" >> beam.ParDo(_WriteTime(job, produce))
            )
    return len(times) - len(todo), len(todo)
