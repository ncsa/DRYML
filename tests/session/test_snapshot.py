"""Public snapshot immutability and redaction contracts."""

from types import MappingProxyType

import pytest

from dryml.environments import EnvironmentRequirement
from dryml.session import SelectedWorldAllocation, SessionSnapshot
from dryml.worlds import ProcessAllocation


def test_snapshot_data_is_detached_immutable_and_redacted():
    """Public data removes credentials, environment values, paths, and URI tails."""

    snapshot = SessionSnapshot(
        "python", None, None, EnvironmentRequirement(),
        {"environment": False, "world": False, "runtime": False},
        {"api_token": "secret", "path": "/private/data", "url": "https://user:pass@example.test/a?token=x#fragment", "env": {"HOME": "/private/home"}},
        {}, object(), 0, "healthy",
    )
    data = snapshot.to_data()

    assert data["controls"]["api_token"] == "<redacted>"
    assert data["controls"]["path"] == "<local-path>"
    assert data["controls"]["url"] == "https://example.test/a"
    assert data["controls"]["env"]["HOME"] == "<redacted>"
    assert snapshot.controls["api_token"] == "<redacted>"
    assert snapshot.controls["env"]["HOME"] == "<redacted>"
    with pytest.raises(TypeError):
        snapshot.controls["x"] = "y"
    assert isinstance(snapshot.controls, MappingProxyType)


def test_snapshot_fields_do_not_expose_allocation_environment_values():
    """Redaction applies to direct snapshot allocation fields, not only data."""

    snapshot = SessionSnapshot(
        "managed", None, SelectedWorldAllocation("main", ProcessAllocation(0, 0, 0, env={"TOKEN": "secret"})), EnvironmentRequirement(),
        {"environment": True, "world": True, "runtime": True}, {}, {}, object(), 1, "healthy",
    )

    assert snapshot.allocation.process.env["TOKEN"] == "<redacted>"


def test_snapshot_redacts_direct_windows_paths_without_changing_file_uris():
    """Direct display fields hide native paths while file URI paths remain useful."""

    snapshot = SessionSnapshot(
        "python", None, None, EnvironmentRequirement(),
        {"environment": False, "world": False, "runtime": False},
        {"windows_path": r"C:\\private\\token.txt", "unc_path": r"\\\\server\\share\\secret.txt", "uri": "file:///private/token.txt", "note": "token=supplied-secret"},
        {}, object(), 0, "healthy",
    )

    data = snapshot.to_data()

    assert data["controls"]["windows_path"] == "<local-path>"
    assert data["controls"]["unc_path"] == "<local-path>"
    assert data["controls"]["uri"] == "file:///private/token.txt"
    assert data["controls"]["note"] == "token=<redacted>"


def test_snapshot_redacts_environment_diagnostics_in_direct_fields():
    """Requirement diagnostics are display data and cannot expose supplied secrets."""

    snapshot = SessionSnapshot(
        "python", None, None,
        EnvironmentRequirement(details={"source_path": "relative/private", "note": "token=supplied-secret"}, metadata={"api_token": "supplied-secret"}),
        {"environment": False, "world": False, "runtime": False},
        {}, {}, object(), 0, "healthy",
    )

    assert snapshot.environment.details["source_path"] == "<local-path>"
    assert snapshot.environment.details["note"] == "token=<redacted>"
    assert snapshot.environment.metadata["api_token"] == "<redacted>"
