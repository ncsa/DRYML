"""U6 managed default resolution across local and worker invocation boundaries."""

from __future__ import annotations

import contextvars
import threading
from pathlib import Path

import pytest

from dryml.core import Repo
from dryml.core.execute import core_worker_setup, current_context
from dryml.core.execute_codec import decode_outcome, encode_invocation, invoke_invocation
from dryml.core.object import Pickleable
from dryml.core.session import config as session_config
from dryml.core.store.dir import DirStore
from dryml.execute.models import WorkerSetupContext
from dryml.formats import make_envelope, semantic_id
from dryml.managed import ManagedConfig, managed_operation
from dryml.managed.defaults import _control_store_defaults, _current_control_store_default
from dryml.managed.errors import ManagedStoreError
from dryml.runtime import RuntimeContextSpec, RuntimeMode


class DefaultsValue(Pickleable):
    """Managed receiver that projects its resolved authority as portable paths."""

    @managed_operation()
    def observe(self, *, managed):
        """Return the selected state default and control Store paths."""

        return (
            str(managed.state_repo.default_store.base_dir),
            str(managed.control_store.base_dir),
        )


def _nested_observe(value, managed):
    """Call a managed method normally from an ordinary nested callable."""

    return value.observe(managed=managed)


def _nested_execute_observe(value):
    """Invoke an omitted-config managed call inside a transported callable."""

    return value.observe()


def _effect_then_request_missing_state(value, marker, missing_state):
    """Perform ordinary work before a worker-local resource request fails."""

    from dryml.core.store.store import Store

    Path(marker).write_text("ordinary-effect", encoding="ascii")
    return value.observe(managed=ManagedConfig(state_repo=Store.from_definition({
        "kind": "dir", "path": missing_state, "query_index": "none",
    })))


def _worker_setup_data(tmp_path, control_store):
    """Build one detached core worker setup with an explicit control role."""

    state = DirStore(tmp_path / "worker-state", query_index="none")
    repo = Repo(state)
    definition = repo.to_definition().to_data()
    repo.close(flush=False)
    state.close()
    payload = {
        "runtime": RuntimeContextSpec(RuntimeMode.INLINE).to_data(),
        "repo": definition,
        "role": "main",
        "replica": 0,
        "control_store": control_store.to_definition(),
    }
    return make_envelope(
        schema="dryml.core.execute.v1.1",
        kind="worker_setup",
        prefix="core_setup",
        payload=payload,
        semantic_id=semantic_id(
            "core_setup", "dryml.core.execute.v1.1", "worker_setup", payload,
            max_depth=64, max_nodes=65_536, max_entries=65_536,
        ),
        max_depth=64,
        max_nodes=65_536,
        max_entries=65_536,
    )


def _worker_context():
    """Return minimal trusted generic setup evidence for a local worker test."""

    return WorkerSetupContext(
        submission_id="managed-defaults", backend="subprocess", environment=None,
        allocation=None, native_grant={"kind": "subprocess"},
    )


@pytest.mark.parametrize("nested", (False, True))
@pytest.mark.parametrize("selection", ("omitted", "empty", "state", "control", "both"))
def test_managed_default_resolution_uses_invocation_state_and_worker_control(tmp_path, nested, selection):
    """Direct and nested calls share field-wise state/control precedence."""

    ambient = Repo(DirStore(tmp_path / "ambient"))
    explicit = Repo(DirStore(tmp_path / "explicit"))
    worker_control = DirStore(tmp_path / "worker-control")
    explicit_control = DirStore(tmp_path / "explicit-control")
    value = DefaultsValue(repo=ambient)
    kwargs = {
        "omitted": None,
        "empty": ManagedConfig(),
        "state": ManagedConfig(state_repo=explicit),
        "control": ManagedConfig(control_store=explicit_control),
        "both": ManagedConfig(state_repo=explicit, control_store=explicit_control),
    }
    expected_state = explicit if selection in {"state", "both"} else ambient
    expected_control = explicit_control if selection in {"control", "both"} else worker_control

    with session_config(repo=ambient), _control_store_defaults(worker_control):
        result = _nested_observe(value, kwargs[selection]) if nested else value.observe(managed=kwargs[selection])

    assert result == (str(expected_state.default_store.base_dir), str(expected_control.base_dir))


def test_omitted_state_honors_nested_selection_and_clearing_fails(tmp_path):
    """Defaults read the current Repo at invocation time rather than setup time."""

    first = Repo(DirStore(tmp_path / "first"))
    second = Repo(DirStore(tmp_path / "second"))
    control = DirStore(tmp_path / "control")
    value = DefaultsValue(repo=first)
    explicit_value = DefaultsValue(repo=first)

    with session_config(repo=first), _control_store_defaults(control):
        with session_config(repo=second):
            assert value.observe() == (str(second.default_store.base_dir), str(control.base_dir))
            assert explicit_value.observe(managed=ManagedConfig(state_repo=first)) == (
                str(first.default_store.base_dir), str(control.base_dir),
            )
        with session_config(repo=None):
            with pytest.raises(ManagedStoreError, match="no current Repo"):
                value.observe()


def test_worker_control_default_is_owner_bound_and_expires_after_setup(tmp_path):
    """Copied and inactive worker default contexts cannot retain control authority."""

    control = DirStore(tmp_path / "control")
    observed = []

    with _control_store_defaults(control):
        assert _current_control_store_default() is control
        copied = contextvars.copy_context()

        def read_default():
            with pytest.raises(RuntimeError, match="different thread"):
                _current_control_store_default()
            observed.append(True)

        thread = threading.Thread(target=lambda: copied.run(read_default))
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive()

    assert observed == [True]
    with pytest.raises(RuntimeError, match="inactive"):
        copied.run(_current_control_store_default)


def test_ordinary_managed_named_keyword_remains_callable_data(tmp_path):
    """The managed owner consumes only its own keyword, not ordinary call kwargs."""

    def ordinary(*, managed):
        """Return an ordinary keyword value named after the managed control."""

        return managed

    repo = Repo(DirStore(tmp_path / "state", query_index="none"))
    outcome = decode_outcome(
        invoke_invocation(encode_invocation(ordinary, (), {"managed": "ordinary-value"}, repo=repo), repo=repo),
        repo=repo,
    )

    assert outcome["success"]
    assert outcome["result"] == "ordinary-value"


@pytest.mark.parametrize("nested", (False, True))
def test_execute_omitted_defaults_read_the_invocation_repo_not_the_codec_repo(tmp_path, nested):
    """Managed Execute targets do not receive an injected root state configuration."""

    codec_repo = Repo(DirStore(tmp_path / "codec", query_index="none"))
    invocation_repo = Repo(DirStore(tmp_path / "invocation", query_index="none"))
    control = DirStore(tmp_path / "control", query_index="none")
    value = DefaultsValue(repo=codec_repo)
    codec_repo.save_object(value, deep_capture=True)
    target = _nested_execute_observe if nested else value.observe
    args = (value,) if nested else ()

    with session_config(repo=invocation_repo), _control_store_defaults(control):
        outcome = decode_outcome(
            invoke_invocation(encode_invocation(target, args, {}, repo=codec_repo), repo=codec_repo),
            repo=codec_repo,
        )

    assert outcome["success"]
    assert outcome["result"] == (str(invocation_repo.default_store.base_dir), str(control.base_dir))


def test_worker_setup_installs_its_control_default_before_managed_invocation(tmp_path):
    """Worker setup makes its selected control Store available without root injection."""

    control = DirStore(tmp_path / "control", query_index="none")
    data = _worker_setup_data(tmp_path, control)
    control.close()

    with core_worker_setup(_worker_context(), data):
        value = DefaultsValue(repo=current_context().repo)
        assert value.observe() == (
            str(current_context().repo.default_store.base_dir),
            str(current_context().control_store.base_dir),
        )


def test_dynamic_missing_resource_preserves_earlier_effect_and_skips_managed_mutation(tmp_path):
    """A demand-driven request fails only after preceding ordinary code has run."""

    repo = Repo(DirStore(tmp_path / "state", query_index="none"))
    value = DefaultsValue(repo=repo)
    repo.save_object(value, deep_capture=True)
    marker = tmp_path / "ordinary-effect"
    with session_config(repo=repo):
        outcome = decode_outcome(
            invoke_invocation(
                encode_invocation(
                    _effect_then_request_missing_state,
                    (value, str(marker), str(tmp_path / "missing")), {}, repo=repo,
                ),
                repo=repo,
            ),
            repo=repo,
        )

    assert not outcome["success"]
    assert outcome["reason"] == "StoreAuthorityError"
    assert marker.read_text(encoding="ascii") == "ordinary-effect"
    assert value.observe.status(state_repo=repo).state == "not_started"
