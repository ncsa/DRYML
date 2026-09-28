"""Explicit collection gate for real Stage 5+7 qualification tests."""

from __future__ import annotations

import pytest


def pytest_addoption(parser):
    """Register the opt-in switch required before real qualification collection runs."""

    parser.addoption(
        "--stage5-7-qualification", action="store_true", default=False,
        help="run explicitly selected real Stage 5+7 TFDS/training qualification tests",
    )


def pytest_configure(config):
    """Declare the marker used for tests that may load fixtures or train models."""

    config.addinivalue_line(
        "markers", "stage5_7_qualification: real opt-in TFDS/training/GPU qualification",
    )


def pytest_collection_modifyitems(config, items):
    """Mark disabled real work as unrun rather than allowing accidental execution."""

    if config.getoption("--stage5-7-qualification"):
        return
    for item in items:
        if item.get_closest_marker("stage5_7_qualification"):
            item.add_marker(pytest.mark.skip(
                reason="UNRUN: pass --stage5-7-qualification after preparing caller-owned fixtures",
            ))
