"""Current-format U15 resume admission, independent of U16/U17 completion."""

from dataclasses import replace

import pytest

from dryml.core.repo import RepoLoadError
from dryml.managed import ManagedConfig
from dryml.managed.errors import ManagedRerunRequiredError
from tests.managed.u15_fixtures import Value, enter, seed, setup


def test_resume_restores_latest_keeps_run_and_changes_invocation(tmp_path):
    control, live, saved = setup(tmp_path)
    old = seed(control, live, saved)
    live.value = 91
    resumed, value, resuming = enter(control, live)
    assert (value, resuming) == (1, True)
    assert resumed.run_id == old.run_id and resumed.invocation_id != old.invocation_id
    assert resumed.checkpoint.graph_equal(saved)
    assert (Value.restores, Value.captures, Value.bodies) == (1, 0, 0)


def test_nonresumable_unfinished_requires_rerun_even_with_latest(tmp_path):
    control, live, saved = setup(tmp_path)
    old = seed(control, live, saved, descriptor=Value.fixed)
    with pytest.raises(ManagedRerunRequiredError, match="rerun_required"):
        enter(control, live, descriptor=Value.fixed)
    assert control.inspect(old.operation_id) == old and Value.restores == 0


def test_resume_never_falls_back_from_unavailable_latest_to_initial(tmp_path, monkeypatch):
    control, live, initial = setup(tmp_path)
    live.value = 2
    latest = control.state_repo.save(live)
    Value.captures = 0
    old = seed(control, live, latest, initial=initial)
    selected = []
    def missing(keys):
        keys = tuple(keys)
        assert initial.state.key not in keys
        selected.extend(keys)
        raise RepoLoadError("latest unavailable")
    monkeypatch.setattr(control.control_store, "read_states", missing)
    with pytest.raises(RepoLoadError):
        enter(control, live)
    assert control.inspect(old.operation_id) == old
    assert selected == [latest.state.key]
    assert (Value.restores, Value.captures, Value.bodies) == (0, 0, 0)


@pytest.mark.parametrize("phase", ["completed", "interrupted", "failed"])
def test_changed_arguments_require_explicit_rerun_before_effects(tmp_path, phase):
    control, live, saved = setup(tmp_path)
    old = seed(control, live, saved, state=phase)
    with pytest.raises(ManagedRerunRequiredError, match="argument_mismatch"):
        enter(control, live, amount=3)
    assert control.inspect(old.operation_id) == old
    assert Value.restores == 0


def test_resumed_failure_preserves_latest_for_fresh_resume(tmp_path):
    control, live, saved = setup(tmp_path)
    old = seed(control, live, saved)
    def fail(*args):
        raise ValueError("body seam failed")
    with pytest.raises(ValueError, match="body seam failed"):
        enter(control, live, execute=fail)
    failed = control.inspect(old.operation_id)
    assert failed.state == "failed" and failed.run_id == old.run_id
    live.value = 99
    current, value, resuming = enter(control, live)
    assert resuming and value == 1 and current.run_id == old.run_id
    assert len({old.invocation_id, failed.invocation_id, current.invocation_id}) == 3
    assert (Value.restores, Value.captures, Value.bodies) == (2, 0, 0)
