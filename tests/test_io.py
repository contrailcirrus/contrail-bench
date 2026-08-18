"""Regression tests for contrailbench.io's write atomicity and mkdir safety."""

import os

import pytest

from contrailbench import io


def test_write_creates_missing_parent_directory(tmp_path):
    """fsspec's local filesystem defaults to auto_mkdir=False, so writing into a
    directory that doesn't exist yet must not raise."""
    dest = tmp_path / "fresh" / "nested" / "dir" / "out.bin"
    io.write(str(dest), b"hello")
    assert dest.read_bytes() == b"hello"


def test_write_round_trips(tmp_path):
    dest = tmp_path / "out.bin"
    io.write(str(dest), b"some bytes")
    assert dest.read_bytes() == b"some bytes"


def test_write_leaves_no_truncated_file_on_failure(tmp_path, monkeypatch):
    """A write that fails partway through must not leave anything at the destination
    path -- the whole point of writing to a same-directory temp file first and only
    then renaming into place."""
    dest = tmp_path / "out.bin"

    real_fdopen = os.fdopen

    def failing_fdopen(fd, mode):
        f = real_fdopen(fd, mode)
        real_write = f.write

        def bad_write(data):
            real_write(data)
            raise OSError("simulated failure mid-write")

        f.write = bad_write
        return f

    monkeypatch.setattr(os, "fdopen", failing_fdopen)

    with pytest.raises(OSError, match="simulated failure mid-write"):
        io.write(str(dest), b"partial data")

    assert not dest.exists()
    # no leaked temp file left behind in the destination directory either
    assert list(tmp_path.iterdir()) == []


def test_write_uses_same_directory_as_destination_for_temp_file(tmp_path, monkeypatch):
    """The temp file must be created in url's own directory, not the default tempdir --
    otherwise the final rename may not be atomic (different filesystem)."""
    dest = tmp_path / "sub" / "out.bin"
    dest.parent.mkdir()

    seen_dirs = []
    real_mkstemp = __import__("tempfile").mkstemp

    def spy_mkstemp(*args, **kwargs):
        seen_dirs.append(kwargs.get("dir"))
        return real_mkstemp(*args, **kwargs)

    monkeypatch.setattr("tempfile.mkstemp", spy_mkstemp)
    io.write(str(dest), b"data")

    assert seen_dirs == [str(dest.parent)]


def test_exists_true_after_write(tmp_path):
    dest = tmp_path / "out.bin"
    assert not io.exists(str(dest))
    io.write(str(dest), b"data")
    assert io.exists(str(dest))


def test_exists_false_for_missing_file(tmp_path):
    assert not io.exists(str(tmp_path / "does" / "not" / "exist.bin"))
