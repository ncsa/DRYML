"""Closed operation and argument identity tests for managed operations."""

from __future__ import annotations

import math

import pytest

import dryml.managed.identity as identity
from dryml.core import Object, Repo
from dryml.core.reference_values import ObjectId
from dryml.core.store.dir import DirStore
from dryml.managed import ManagedConfigError, argument_digest, managed_operation, operation_digest


class IdentityObject(Object):
    """Small stateless Object used to obtain exact Node metadata."""

    def __init__(self, value):
        self.value = value

    @managed_operation(resumable=True)
    def calculate(self, value=1, *, managed):
        return value


def test_operation_identity_uses_only_exact_object_ref_and_member(tmp_path):
    """Operation IDs are stable across config and separate from argument identity."""

    obj = IdentityObject(1, repo=Repo(DirStore(tmp_path / "store")))
    first = operation_digest(obj.definition, "calculate")
    second = operation_digest(obj.definition, "calculate")

    assert first == second
    assert len(first) == 64
    assert first != operation_digest(obj.definition, "other")
    with pytest.raises(ManagedConfigError):
        operation_digest(object(), "calculate")


def test_argument_digest_binds_defaults_and_excludes_instance_and_config(tmp_path):
    """Equivalent native calls share one typed argument digest."""

    obj = IdentityObject(1, repo=Repo(DirStore(tmp_path / "store")))
    descriptor = type(obj).__dict__["calculate"]

    positional = argument_digest(descriptor, obj, (), {})
    keyword = argument_digest(descriptor, obj, (), {"value": 1})
    configured = argument_digest(descriptor, obj, (), {"managed": None})

    assert positional == keyword == configured
    assert positional != argument_digest(descriptor, obj, (True,), {})
    assert positional != argument_digest(descriptor, obj, (1.0,), {})


def test_argument_identity_is_closed_typed_and_bounded(tmp_path):
    """Unsupported values and limits are rejected without representation hooks."""

    obj = IdentityObject(1, repo=Repo(DirStore(tmp_path / "store")))
    descriptor = type(obj).__dict__["calculate"]
    scalar = argument_digest(descriptor, obj, (ObjectId(),), {})
    assert scalar != argument_digest(descriptor, obj, (scalar,), {})
    assert argument_digest(descriptor, obj, (-0.0,), {}) != argument_digest(descriptor, obj, (0.0,), {})

    cyclic = []
    cyclic.append(cyclic)
    for value in (
        cyclic,
        {"wrong": object()},
        {1: "key", "mixed": "key"},
        {"nan": math.nan},
        {"call": lambda: None},
        {"set": {1}},
    ):
        with pytest.raises(ManagedConfigError):
            argument_digest(descriptor, obj, (value,), {})
    with pytest.raises(ManagedConfigError, match="64"):
        argument_digest(descriptor, obj, ("x" * (64 * 1024 + 1),), {})
    with pytest.raises(ManagedConfigError, match="256"):
        argument_digest(descriptor, obj, (1 << 256,), {})
    with pytest.raises(ManagedConfigError, match="1024"):
        argument_digest(descriptor, obj, ([None] * 1025,), {})
    nested = None
    for _ in range(17):
        nested = [nested]
    with pytest.raises(ManagedConfigError, match="depth"):
        argument_digest(descriptor, obj, (nested,), {})


def test_argument_container_order_and_live_objects_have_closed_identity(tmp_path):
    """Container spelling and live Object payload changes do not alter identities."""

    repo = Repo(DirStore(tmp_path / "store"))
    obj = IdentityObject(1, repo=repo)
    other = IdentityObject(2, repo=repo)
    descriptor = type(obj).__dict__["calculate"]

    assert argument_digest(descriptor, obj, ({"a": [1], "b": (2,)},), {}) == argument_digest(
        descriptor, obj, ({"b": (2,), "a": [1]},), {}
    )
    assert argument_digest(descriptor, obj, ([1, 2],), {}) != argument_digest(descriptor, obj, ((1, 2),), {})
    before = argument_digest(descriptor, obj, (other,), {})
    other.value = 99
    assert before == argument_digest(descriptor, obj, (other,), {})


def test_argument_digest_preserves_canonical_bytes_for_accepted_values(tmp_path):
    """Streaming accepted arguments preserves the prior canonical digest bytes."""

    obj = IdentityObject(1, repo=Repo(DirStore(tmp_path / "store")))
    descriptor = type(obj).__dict__["calculate"]

    assert argument_digest(descriptor, obj, ({"b": (2, b"x"), "a": [-0.0, None]},), {}) == (
        "5cc04cc4adaf2228e6442adcc46f98bbf7f295b78e6c3e52d90b9becae8a7b71"
    )
    assert argument_digest(descriptor, obj, ("ascii",), {}) == "7db0aef6cdb30ff5c7b810c5ed6310dd5e6ea91be4c4cd1aa03b61c32b128070"
    assert argument_digest(descriptor, obj, ({"snowman": "\u2603"},), {}) == (
        "89e42cf5dae9142abf8a8ec13827edf5716420ef4dda62c79704da6ea4ace19e"
    )


def test_argument_digest_checks_aggregate_preimage_before_emitting(tmp_path, monkeypatch):
    """Individually valid strings exceed the aggregate limit without joined output."""

    obj = IdentityObject(1, repo=Repo(DirStore(tmp_path / "store")))
    descriptor = type(obj).__dict__["calculate"]
    chunks = []
    original_write = identity._ArgumentEncoder._write

    def capture_write(self, data):
        chunks.append(len(data))
        original_write(self, data)

    monkeypatch.setattr(identity._ArgumentEncoder, "_write", capture_write)
    values = ["x" * (64 * 1024)] * 17

    argument_digest(descriptor, obj, (values[:15],), {})
    assert max(chunks) <= 64 * 1024
    chunks.clear()
    with pytest.raises(ManagedConfigError, match="preimage exceeds 1 MiB"):
        argument_digest(descriptor, obj, (values,), {})
    assert chunks == []


def test_argument_digest_reports_utf8_and_key_paths_with_bounded_errors(tmp_path):
    """Malformed text and large dictionary keys produce bounded managed errors."""

    obj = IdentityObject(1, repo=Repo(DirStore(tmp_path / "store")))
    descriptor = type(obj).__dict__["calculate"]

    for value in ("\ud800", {"\ud800": "value"}):
        with pytest.raises(ManagedConfigError, match="valid UTF-8"):
            argument_digest(descriptor, obj, (value,), {})
    with pytest.raises(ManagedConfigError) as error:
        argument_digest(descriptor, obj, ({"key" * (64 * 1024 // 3): object()},), {})
    assert len(str(error.value)) < 256


def test_projection_weakens_graph_state_but_not_quoted_configuration(tmp_path):
    """Graph selection changes share an operation; quoted recipe changes do not."""
    from pathlib import Path
    from dryml.core import Definition
    from tests.managed.test_u14_authority import Dataset, Experiment

    store = DirStore(tmp_path / "state", query_index="none")
    recipe = Definition(Dataset).concretize()
    states = []
    for payload in (b"one", b"two"):
        staging = store.create_local_state_staging()
        Path(staging, "data", "value").write_bytes(payload)
        states.append(store.install_state(recipe, staging, codec="Test", codec_version=1, interpretation={}).state)
    first = recipe.with_identity(ObjectId()).with_state(states[0])
    second = first.with_realization(states={"$": states[1]})
    a = Definition(Experiment, first).concretize()
    b = Definition(Experiment, second).concretize()
    assert operation_digest(a, "run") == operation_digest(b, "run")
    from dryml.core.utils.graph.path import GraphPath, Parameter
    configuration = Definition(IdentityObject, 1).concretize()
    quoted_a = configuration.lens(GraphPath((Parameter("value"),))).set(first.quote())
    quoted_b = configuration.lens(GraphPath((Parameter("value"),))).set(second.quote())
    assert operation_digest(quoted_a, "calculate") != operation_digest(quoted_b, "calculate")
