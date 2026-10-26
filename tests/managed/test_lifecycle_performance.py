"""Work-count guards for managed execution without cached authority checks."""

import pytest

from dryml.managed import ManagedConfig
from tests.conftest import _synthetic_environment_record
from tests.managed.u16_fixtures import Value, setup


@pytest.mark.parametrize("unavailable", [False, True])
def test_environment_observed_once_per_execution_and_never_on_replay(tmp_path, monkeypatch, unavailable):
    from dryml.environments import introspection

    observations = []
    environment = _synthetic_environment_record()

    def observe():
        observations.append(True)
        if unavailable:
            raise OSError("inventory unavailable")
        return environment

    monkeypatch.setattr(introspection, "inspect_current", observe)
    control, live = setup(tmp_path)

    def body(obj, ctx):
        obj.value += 1
        ctx.checkpoint()
        obj.value += 1

    monkeypatch.setattr(Value, "action", body)
    config = ManagedConfig(state_repo=control.state_repo)
    result = live.run(managed=config)
    assert len(observations) == 1 and Value.captures == 2
    capture = control.state_repo.get_snapshot_metadata(result.final_state_ref)
    assert capture.environment_status == ("unavailable" if unavailable else "known")
    assert capture.environment == (None if unavailable else environment)

    replay = live.run(managed=config)
    assert replay.execution.producer_invocation_id == result.execution.producer_invocation_id
    assert len(observations) == 1 and Value.captures == 2

    live.run(managed=ManagedConfig(state_repo=control.state_repo, rerun=True))
    assert len(observations) == 2 and Value.captures == 4


def test_safe_points_reread_authority_without_redecoding_stable_graphs(
        tmp_path, monkeypatch, fixed_managed_snapshot_environment):
    from dryml.managed import control as module

    control, live = setup(tmp_path, index="sqlite")
    decodes, reads = [], []
    decode, capture = module.decode_node_graph, module.ManagedControlStore._capture

    def counted_decode(*args, **kwargs):
        decodes.append(True)
        return decode(*args, **kwargs)

    def counted_capture(*args, **kwargs):
        reads.append(True)
        return capture(*args, **kwargs)

    monkeypatch.setattr(module, "decode_node_graph", counted_decode)
    monkeypatch.setattr(module.ManagedControlStore, "_capture", counted_capture)

    def body(obj, ctx):
        ctx.checkpoint()
        ctx._safe_point()
        previous_decodes, previous_reads = len(decodes), len(reads)
        for _ in range(5):
            ctx._safe_point()
        assert len(decodes) == previous_decodes
        assert len(reads) >= previous_reads + 5

    monkeypatch.setattr(Value, "action", body)
    result = live.run(managed=ManagedConfig(state_repo=control.state_repo))
    assert result.status.state == "completed" and decodes


def test_selected_final_without_capture_does_not_observe_environment(
        tmp_path, monkeypatch, fixed_snapshot_environment):
    from dryml.environments import introspection

    control, live = setup(tmp_path)
    saved = control.state_repo.save(live)
    monkeypatch.setattr(Value, "action", lambda obj, ctx: ctx.final_state(saved))
    monkeypatch.setattr(introspection, "inspect_current", lambda: pytest.fail("capture-free final observed environment"))
    result = live.run(managed=ManagedConfig(state_repo=control.state_repo))
    assert result.final_state_ref.graph_equal(saved)


def test_control_decode_reuse_rejects_changed_bytes_and_wrong_operation(
        tmp_path, monkeypatch, fixed_managed_snapshot_environment):
    from dryml.core._content_facts import _content_facts_scope
    from dryml.managed import ManagedControlError
    from dryml.managed.control import ControlSnapshot

    control, live = setup(tmp_path)
    monkeypatch.setattr(Value, "action", lambda obj, ctx: None)
    result = live.run(managed=ManagedConfig(state_repo=control.state_repo))
    operation = result.execution.operation_id
    payload = control.inspect(operation).to_bytes()
    with _content_facts_scope():
        first = ControlSnapshot.from_bytes(payload, operation_id=operation)
        assert ControlSnapshot.from_bytes(payload, operation_id=operation) is first
        with pytest.raises(ManagedControlError):
            ControlSnapshot.from_bytes(payload + b"!", operation_id=operation)
        with pytest.raises(ManagedControlError):
            ControlSnapshot.from_bytes(payload, operation_id="f" * 64)

        path = control._current_path(control._operation_path(operation))
        from pathlib import Path
        Path(path).write_bytes(payload + b"!")
        with pytest.raises(ManagedControlError):
            control.inspect(operation)
