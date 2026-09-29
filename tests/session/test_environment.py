"""Managed current-environment validation contracts."""

import pytest

from dryml.session import current, enforce_requirements, manage, require_env
from dryml.session import state


def test_environment_check_is_skipped_when_axis_is_disabled(session_runtime, monkeypatch):
    """Parity-only disabled environment checks do not inspect the interpreter."""

    manage(cpus=1)
    enforce_requirements(environment=False, world=True, runtime=True)
    monkeypatch.setattr(state, "inspect_current", lambda: (_ for _ in ()).throw(AssertionError("unexpected inspection")))

    require_env("missing-package>=1")


def test_environment_check_prevents_publication_before_effects(session_runtime, monkeypatch):
    """An enabled incompatible requirement leaves the current generation intact."""

    manage(cpus=1)
    enforce_requirements(environment=True, world=False, runtime=False)
    before = state.current()
    monkeypatch.setattr(state, "inspect_current", lambda: object())

    with pytest.raises(Exception, match="current environment"):
        require_env("missing-package>=1")
    assert state.current().generation == before.generation


def test_conflicting_environment_merge_is_atomic(session_runtime):
    """A contradictory merge leaves the prior requirement generation intact."""

    accepted = require_env("example-package>=1")
    with pytest.raises(Exception):
        require_env(excludes=("example-package",))
    after = current()

    assert after.generation == accepted.generation
    assert after.environment == accepted.environment
