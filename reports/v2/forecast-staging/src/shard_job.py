"""Jobs that write one netCDF file per (time, flight level), or per (time, flight level, lead).

A job is a directory of ``{epoch}_{flight_level}.nc`` files (``{epoch}_{flight_level}/{lead}.nc``
for a source with leads) and a *producer*: a callable that, given a valid time and the flight
levels, returns the dataset for each file. The driver does the
rest: it skips times that are already complete (so an interrupted run resumes), runs the rest in a
Beam pipeline, and writes every file atomically. All data access lives in the producer, so a new
source only has to say how to build one time's files.
"""

import contextlib
import dataclasses
import datetime
import logging
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence

import apache_beam as beam
import xarray as xr
from apache_beam.options.pipeline_options import PipelineOptions

from src import forecast_runs, storage
from src import times as time_utils

LOGGER = logging.getLogger(__name__)

#: A file's key: its flight level, or ``(flight_level, lead)`` for a job with leads
ShardKey = int | tuple[int, int]

#: Builds one time's files: ``produce(time, flight_levels)`` gives ``{key: dataset}``, or for a job
#: with leads an iterator of ``(key, dataset)`` so that each is written before the next is made
Producer = Callable[
    [datetime.datetime, Sequence[int]],
    Mapping[ShardKey, xr.Dataset] | Iterator[tuple[ShardKey, xr.Dataset]],
]


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
    #: Hours added to each valid time's shortest lead (see :mod:`src.forecast_runs`), one file per
    #: lead; ``None`` for a source with no leads
    lead_offsets: tuple[int, ...] | None = None

    def keys(self, time: datetime.datetime) -> set[ShardKey]:
        """The key of every file the job writes for ``time``."""
        if self.lead_offsets is None:
            return set(self.flight_levels)
        leads = forecast_runs.leads(time, self.lead_offsets)
        return {(fl, lead) for fl in self.flight_levels for lead in leads}

    def name(self, time: datetime.datetime, key: ShardKey) -> str:
        """The name of one file, relative to :attr:`output_dir`."""
        epoch = time_utils.utc_epoch(time)
        if isinstance(key, tuple):
            flight_level, lead = key
            return f"{epoch}_{flight_level}/{lead}.nc"
        return f"{epoch}_{key}.nc"

    def path(self, time: datetime.datetime, key: ShardKey) -> str:
        """The file for one time and flight level (and lead)."""
        return f"{self.output_dir}/{self.name(time, key)}"


def pending(job: ShardJob, times: Sequence[datetime.datetime]) -> list[datetime.datetime]:
    """The times that still lack a file for at least one flight level (and lead).

    A time counts as done only if every file exists, so a partially written time is redone in full.
    This is what makes an interrupted run restartable.
    """
    existing = set(storage.list_names(job.output_dir))
    return [t for t in times if not all(job.name(t, key) in existing for key in job.keys(t))]


def write_shards(
    job: ShardJob,
    time: datetime.datetime,
    shards: Mapping[ShardKey, xr.Dataset] | Iterable[tuple[ShardKey, xr.Dataset]],
) -> None:
    """Write one time's files, each atomically.

    ``shards`` is a mapping, or an iterable of ``(key, dataset)`` that is written as it is
    produced, so that a long time need not hold every dataset at once.

    Raises
    ------
    ValueError
        If ``shards`` does not hold exactly the job's files. For a mapping nothing is written
        then; an iterable is checked once it is exhausted, and a time left incomplete is redone
        by the next run.
    """
    expected = job.keys(time)
    if isinstance(shards, Mapping):
        _check_keys(expected, set(shards))
        items: Iterable[tuple[ShardKey, xr.Dataset]] = shards.items()
    else:
        items = shards
    written: set[ShardKey] = set()
    for key, dataset in items:
        storage.write_netcdf_atomic(dataset, job.path(time, key))
        written.add(key)
    _check_keys(expected, written)


def _check_keys(expected: set[ShardKey], found: set[ShardKey]) -> None:
    """Raise if a producer's files are not the job's."""
    if found != expected:
        msg = f"producer returned files {sorted(found)}, expected {sorted(expected)}"
        raise ValueError(msg)


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
