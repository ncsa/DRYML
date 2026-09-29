"""Ray-category integration opt-in validation for existing deployments."""

from __future__ import annotations

import pytest

from tests.execute.conftest import integration_enabled, require_ray_integration, validate_existing_environment_opt_in


def pytest_configure(config: pytest.Config) -> None:
    """Validate every enabled external prerequisite before Ray tests collect."""
    del config
    validate_existing_environment_opt_in()
    if integration_enabled():
        require_ray_integration()
