"""Regression tests for the shared UTC-safe timestamp conversion.

The bug this guards against: ``datetime.timestamp()`` on a naive value assumes
*host-local* wall-clock time and applies that zone's historical DST rule for the exact
date being converted. The tests below actually flip the host's system timezone
(``TZ`` + ``time.tzset()``) rather than just pinning expected epoch values against one
machine's zone -- that's what makes this the test that would have caught the original
bug, instead of merely re-confirming a fixed-zone expectation.
"""

import datetime
import time

import pandas as pd
import pytest

from contrailbench import time_utils

_NAIVE_CASES = [
    datetime.datetime(2024, 9, 1, 0, 0),  # would fall inside UK BST
    datetime.datetime(2024, 10, 8, 12, 0),  # inside 2024 UK BST
    datetime.datetime(2024, 11, 13, 22, 0),  # after 2024 UK BST ended
    datetime.datetime(2024, 1, 1, 0, 0),  # winter, arbitrary non-UK-relevant date
]


@pytest.fixture
def system_timezone(monkeypatch):
    """Temporarily set the process's local timezone, restoring it afterward.

    Unix-only (``time.tzset()`` has no effect/no equivalent on Windows), which is fine
    for this repo's sandbox.
    """

    def _set(tz_name: str) -> None:
        monkeypatch.setenv("TZ", tz_name)
        time.tzset()

    yield _set
    time.tzset()  # restore whatever TZ monkeypatch reverts to on teardown


@pytest.mark.parametrize("naive", _NAIVE_CASES)
def test_to_utc_timestamp_is_independent_of_host_timezone(system_timezone, naive):
    """Same naive (UTC-intended) datetime must convert identically regardless of the
    host's local timezone -- this is exactly the property the original bug violated."""
    system_timezone("UTC")
    utc_result = time_utils.to_utc_timestamp(naive)

    system_timezone("America/Los_Angeles")
    la_result = time_utils.to_utc_timestamp(naive)

    assert utc_result == la_result


def test_to_utc_timestamp_naive_datetime_matches_known_epoch():
    bst_period_time = datetime.datetime(2024, 10, 8, 12, 0)
    gmt_period_time = datetime.datetime(2024, 11, 13, 22, 0)

    assert time_utils.to_utc_timestamp(bst_period_time) == 1728388800
    assert time_utils.to_utc_timestamp(gmt_period_time) == 1731535200


@pytest.mark.parametrize("naive", _NAIVE_CASES)
def test_to_utc_timestamp_accepts_pandas_timestamp(system_timezone, naive):
    """`pandas.Timestamp` is the other naive-datetime type in use across the codebase
    (e.g. `pd.date_range(...).to_pydatetime()` yields plain `datetime`s, but plenty of
    call sites hold a `pd.Timestamp` directly) -- must behave identically to the
    equivalent plain `datetime.datetime`."""
    ts = pd.Timestamp(naive)

    system_timezone("UTC")
    utc_result = time_utils.to_utc_timestamp(ts)

    system_timezone("America/Los_Angeles")
    la_result = time_utils.to_utc_timestamp(ts)

    assert utc_result == la_result == time_utils.to_utc_timestamp(naive)


def test_to_utc_timestamp_aware_datetime_respects_its_own_offset():
    """An already-aware datetime must not be silently reassigned to UTC -- it already
    unambiguously names an instant via its own tzinfo."""
    aware_utc = datetime.datetime(2024, 10, 8, 12, 0, tzinfo=datetime.timezone.utc)
    naive_equivalent = datetime.datetime(2024, 10, 8, 12, 0)
    assert time_utils.to_utc_timestamp(aware_utc) == time_utils.to_utc_timestamp(
        naive_equivalent
    )

    minus_five = datetime.timezone(datetime.timedelta(hours=-5))
    aware_minus_five = datetime.datetime(2024, 10, 8, 7, 0, tzinfo=minus_five)
    # 07:00-05:00 is the same instant as 12:00 UTC.
    assert time_utils.to_utc_timestamp(aware_minus_five) == time_utils.to_utc_timestamp(
        naive_equivalent
    )
