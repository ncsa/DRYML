"""Fixtures for DRYML test-runner administration checks."""

import pytest

from tests.tools import test_buckets


@pytest.fixture(scope="session")
def maintained_test_nodeids() -> set[str]:
    """Return one maintained-node snapshot for this pytest session."""

    return test_buckets.collected_test_nodeids()
