"""Regression tests for contrailbench.data's observation Dataloaders."""

import pytest

from contrailbench.data import (
    ADSBDataloader,
    ContrailWatchDataloader,
    GRUANDataloader,
    IAGOSDataloader,
)

_DATALOADER_CLASSES = [ADSBDataloader, IAGOSDataloader, GRUANDataloader, ContrailWatchDataloader]


@pytest.mark.parametrize("cls", _DATALOADER_CLASSES)
def test_default_path_matches_published_bucket(cls):
    """Constructing without an explicit path must preserve today's hardcoded default --
    this is what makes adding the `path` parameter a non-breaking change."""
    expected = {
        ADSBDataloader: "gs://contrailbench-public-data/v1/adsb",
        IAGOSDataloader: "gs://contrailbench-public-data/v1/iagos",
        GRUANDataloader: "gs://contrailbench-public-data/v1/gruan",
        ContrailWatchDataloader: "gs://contrailbench-public-data/v1/contrailwatch",
    }[cls]
    assert cls().path == expected


@pytest.mark.parametrize("cls", _DATALOADER_CLASSES)
def test_path_is_constructor_overridable(cls):
    """A custom path must be respected instead of the class default -- needed so a
    Dataloader can point at a private/local observation cache rather than only the
    public bucket."""
    custom = "gs://some-other-bucket/custom-path"
    assert cls(path=custom).path == custom
