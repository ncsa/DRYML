"""Explicit collection gate for real ML workflow qualification tests."""

from __future__ import annotations

import pytest


def pytest_addoption(parser):
    """Register the opt-in switch required before real qualification collection runs."""

    parser.addoption(
        "--ml-workflow-qualification", action="store_true", default=False,
        help="run explicitly selected real ML workflow TFDS/training qualification tests",
    )


def pytest_configure(config):
    """Declare the marker used for tests that may load fixtures or train models."""

    config.addinivalue_line(
        "markers", "ml_workflow_qualification: real opt-in TFDS/training/GPU qualification",
    )


def pytest_collection_modifyitems(config, items):
    """Mark disabled real work as unrun rather than allowing accidental execution."""

    if config.getoption("--ml-workflow-qualification"):
        return
    for item in items:
        if item.get_closest_marker("ml_workflow_qualification"):
            item.add_marker(pytest.mark.skip(
                reason="UNRUN: pass --ml-workflow-qualification after preparing caller-owned fixtures",
            ))
