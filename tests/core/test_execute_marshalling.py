"""Focused whole-call authority and preflight proofs for core Execute."""

from __future__ import annotations

import dill
from pathlib import Path
import pytest

from dryml.core import ObjectRef, Repo, Serializable
from dryml.core.execute import CoreOptions, SharedDirStoreStrategy, prepare_shared_storage
from dryml.core.execute_codec import CoreCallCodecError, decode_outcome, encode_invocation, invoke_invocation
from dryml.core.signatures import Ref, ReferenceSelection
from dryml.core.store.dir import DirStore
from dryml.core.symbol import ImportRef


class SavedValue(Serializable):
    """Small durable input whose saved state is distinct from a later live mutation."""

    def __init__(self, value=0):
        self.value = value

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        (Path(dest_dir) / "value").write_text(str(self.value), encoding="ascii")

    def restore_state_from_dir_imp(self, src_dir, *, codec):
        self.value = int((Path(src_dir) / "value").read_text(encoding="ascii"))


class SavedBlob(Serializable):
    """State-only payload fixture proving request bytes do not contain saved data."""

    def __init__(self):
        self.payload = ""

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        (Path(dest_dir) / "payload").write_text(self.payload, encoding="ascii")

    def restore_state_from_dir_imp(self, src_dir, *, codec):
        self.payload = (Path(src_dir) / "payload").read_text(encoding="ascii")


def _read_value(value):
    return value.value


def _same(left, right):
    return left is right


def _blob_size(value):
    return len(value.payload)


def _reference_identity(value: Ref[ObjectRef]) -> Ref[ObjectRef]:
    return value


def _path_identity(value):
    """Return public path data after transport reconstructs the concrete path."""
    return str(value), value.name


_USER_HELPER_GLOBAL = 0
_USER_HELPER_REPO = None


def _user_helper(value, adjustment=1):
    """Use module state and a default that transport must snapshot."""
    return value + adjustment + _USER_HELPER_GLOBAL


def _call_user_helper(value):
    """Call a user-owned module helper captured by the transported target."""
    return _user_helper(value)


def _user_helper_with_repo():
    """Expose a module-level live resource capture to structural validation."""
    return _USER_HELPER_REPO


def _call_user_helper_with_repo():
    """Call a user-owned helper whose own capture graph is invalid."""
    return _user_helper_with_repo()


def _invoke(strategy, fn, args, *, repo, **kwargs):
    """Prepare and execute one codec call directly in an explicit worker Repo."""
    prepared = strategy.prepare(fn, args, {}, repo=repo, control_store=None, update_args=False, **kwargs)
    result = strategy.recover(
        strategy.invoke(prepared.invocation, repo=repo, update_args=False), prepared,
        repo=repo, args=args, kwargs={}, return_objects=False, update_args=False,
    )
    return prepared, result


def test_saved_authority_is_used_without_detecting_or_saving_later_mutation(tmp_path):
    """A known receipt crosses the boundary while unsaved caller payload stays local."""
    repo = Repo(DirStore(tmp_path / "state"))
    value = SavedValue(3, repo=repo)
    saved = repo.save_object(value, deep_capture=True)
    value.value = 9

    _, result = _invoke(SharedDirStoreStrategy(), _read_value, (value,), repo=repo)

    assert result == 3
    assert value.value == 9
    assert value.last_state_ref == saved


def test_pinned_reference_intent_and_diamond_aliases_cross_one_graph(tmp_path):
    """Exact State/Object authority and repeated ordinary aliases survive one memo."""
    store = DirStore(tmp_path / "state")
    repo = Repo(store)
    value = SavedValue(4, repo=repo)
    saved = repo.save_object(value, deep_capture=True)
    strategy = SharedDirStoreStrategy()

    _, reference = _invoke(
        strategy, _reference_identity, (saved,), repo=repo,
        selections={"value": ReferenceSelection(saved.object, store)},
    )
    _, diamond = _invoke(strategy, _same, (["shared"],) * 2, repo=repo)

    assert reference == saved.object
    assert diamond is True


def test_frozen_storage_snapshot_is_reused_and_rejects_unknown_selected_store(tmp_path, monkeypatch):
    """A prepared table is the sole Store-pin authority for one coordinator call."""
    store = DirStore(tmp_path / "state")
    repo = Repo(store)
    value = SavedValue(4, repo=repo)
    saved = repo.save_object(value, deep_capture=True)
    snapshot = prepare_shared_storage(CoreOptions(repo=repo))
    try:
        monkeypatch.setattr(repo, "to_definition", lambda: pytest.fail("unexpected second Repo export"))
        prepared = SharedDirStoreStrategy().prepare(
            _reference_identity, (saved,), {}, repo=repo, control_store=None,
            update_args=False, selections={"value": ReferenceSelection(saved.object, store)},
            _frozen_storage=snapshot.frozen_storage,
        )
        assert prepared.storage_setup == snapshot.storage_setup

        other = DirStore(tmp_path / "other")
        with pytest.raises(CoreCallCodecError, match="unsupported selected Store"):
            SharedDirStoreStrategy().prepare(
                _reference_identity, (saved,), {}, repo=repo, control_store=None,
                update_args=False, selections={"value": ReferenceSelection(saved.object, other)},
                _frozen_storage=snapshot.frozen_storage,
            )
    finally:
        snapshot.close()


def test_prepare_freezes_storage_once_without_activating_future_annotations(tmp_path, monkeypatch):
    """Preparation exports one storage snapshot but leaves annotations inert."""
    repo = Repo(DirStore(tmp_path / "state"))
    exports = []
    original = repo.to_definition
    monkeypatch.setattr(repo, "to_definition", lambda: exports.append(True) or original())
    namespace = {"Ref": Ref, "ObjectRef": ObjectRef}
    exec("def target(value: 'Ref[ObjectRef]'): return value", namespace)

    prepared = SharedDirStoreStrategy().prepare(
        namespace["target"], (1,), {},
        repo=repo, control_store=None, update_args=False,
    )

    assert prepared.invocation
    assert prepared.storage_setup["repo"]
    assert exports == [True]


def test_saved_payload_size_does_not_expand_the_invocation_envelope(tmp_path):
    """Only StateRef authority, not a live saved model payload, enters the request."""
    repo = Repo(DirStore(tmp_path / "state"))
    small = SavedBlob(repo=repo)
    small.payload = "x"
    repo.save_object(small, deep_capture=True)
    large = SavedBlob(repo=repo)
    large.payload = "x" * 200_000
    repo.save_object(large, deep_capture=True)
    strategy = SharedDirStoreStrategy()

    small_call = strategy.prepare(_blob_size, (small,), {}, repo=repo, control_store=None, update_args=False)
    large_call = strategy.prepare(_blob_size, (large,), {}, repo=repo, control_store=None, update_args=False)

    assert len(large_call.invocation) == len(small_call.invocation)


def test_explicit_invocation_and_result_limits_are_honored(tmp_path):
    """The strategy receives caller-selected budgets instead of a hidden codec cap."""
    repo = Repo(DirStore(tmp_path / "state"))
    strategy = SharedDirStoreStrategy()
    with pytest.raises(CoreCallCodecError, match="oversized call graph"):
        strategy.prepare(_same, ("x" * 100, "x" * 100), {}, repo=repo, control_store=None,
                         update_args=False, invocation_limit_bytes=32)
    prepared = strategy.prepare(_same, ("x", "x"), {}, repo=repo, control_store=None, update_args=False)
    with pytest.raises(CoreCallCodecError, match="oversized result"):
        strategy.invoke(prepared.invocation, repo=repo, update_args=False, result_limit_bytes=1)


def test_path_arguments_use_public_value_transport_instead_of_private_slots(tmp_path):
    """Path transport is independent of CPython's version-specific slot layout."""
    repo = Repo(DirStore(tmp_path / "state"))
    path = tmp_path / "marker"

    invocation = encode_invocation(_path_identity, (path,), {}, repo=repo)
    graph = dill.loads(invocation)
    path_nodes = [node for node in graph["nodes"] if node.get("tag") == "path"]

    assert path_nodes == [{"tag": "path", "kind": type(path).__name__, "value": str(path)}]
    outcome = decode_outcome(invoke_invocation(invocation, repo=repo), repo=repo)
    assert outcome["success"]
    assert outcome["result"] == (str(path), path.name)


def test_user_helper_globals_and_defaults_are_structural_snapshots(tmp_path, monkeypatch):
    """User helpers retain coordinator globals and defaults instead of worker imports."""
    repo = Repo(DirStore(tmp_path / "state"))
    monkeypatch.setitem(globals(), "_USER_HELPER_GLOBAL", 7)
    monkeypatch.setattr(_user_helper, "__defaults__", (11,))

    invocation = encode_invocation(_call_user_helper, (3,), {}, repo=repo)
    monkeypatch.setitem(globals(), "_USER_HELPER_GLOBAL", -100)
    monkeypatch.setattr(_user_helper, "__defaults__", (-200,))

    outcome = decode_outcome(invoke_invocation(invocation, repo=repo), repo=repo)
    assert outcome["success"]
    assert outcome["result"] == 21


def test_user_helper_live_repo_capture_is_rejected(tmp_path, monkeypatch):
    """User helper indirection cannot hide a live Repo from capture validation."""
    repo = Repo(DirStore(tmp_path / "state"))
    monkeypatch.setitem(globals(), "_USER_HELPER_REPO", repo)

    with pytest.raises(CoreCallCodecError, match="live core resource"):
        encode_invocation(_call_user_helper_with_repo, (), {}, repo=repo)


def test_unsupported_targets_and_unsaved_objects_fail_before_body_invocation(tmp_path):
    """Coordinator-detectable unsupported shapes never reach the workload body."""
    repo = Repo(DirStore(tmp_path / "state"))
    calls = []

    def target(value):
        calls.append(value)

    async def asynchronous():
        return None

    class AsyncCallable:
        async def __call__(self):
            return None

    class GeneratorCallable:
        def __call__(self):
            yield None

    def generator():
        yield None

    strategy = SharedDirStoreStrategy()
    with pytest.raises(CoreCallCodecError, match="async or generator"):
        strategy.prepare(asynchronous, (), {}, repo=repo, control_store=None, update_args=False)
    with pytest.raises(CoreCallCodecError, match="async or generator"):
        strategy.prepare(generator, (), {}, repo=repo, control_store=None, update_args=False)
    with pytest.raises(CoreCallCodecError, match="async or generator"):
        strategy.prepare(AsyncCallable(), (), {}, repo=repo, control_store=None, update_args=False)
    with pytest.raises(CoreCallCodecError, match="async or generator"):
        strategy.prepare(GeneratorCallable(), (), {}, repo=repo, control_store=None, update_args=False)
    with pytest.raises(CoreCallCodecError, match="unsaved live Object"):
        strategy.prepare(target, (SavedValue(),), {}, repo=repo, control_store=None, update_args=False)

    assert calls == []


def test_unknown_callable_modality_is_rejected_without_invoking_call(tmp_path):
    """
    Core Execute admits callable instances only with statically known sync
    roots.
    """

    repo = Repo(DirStore(tmp_path / "state"))
    calls = []

    class DeferredCallable:
        @property
        def __call__(self):
            calls.append("property")
            return lambda: None

    with pytest.raises(CoreCallCodecError, match="async or generator"):
        SharedDirStoreStrategy().prepare(
            DeferredCallable(), (), {}, repo=repo, control_store=None,
            update_args=False,
        )

    assert calls == []


def test_slots_captures_lower_saved_objects_and_reject_live_repo_globals(tmp_path):
    """Structural capture walks slots and globals instead of falling back to dill."""
    repo = Repo(DirStore(tmp_path / "state"))
    value = SavedValue(12, repo=repo)
    repo.save_object(value, deep_capture=True)

    class SlotCallable:
        __slots__ = ("value",)

        def __init__(self, captured):
            self.value = captured

        def __call__(self):
            return self.value.value

    prepared, result = _invoke(SharedDirStoreStrategy(), SlotCallable(value), (), repo=repo)
    assert result == 12
    assert prepared.invocation

    captured_repo = repo
    secret = "transport-secret-marker"

    def captures_repo():
        return captured_repo, secret

    with pytest.raises(CoreCallCodecError, match="live core resource") as error:
        encode_invocation(captures_repo, (), {}, repo=repo)
    assert secret not in str(error.value)


def test_nested_live_results_publish_before_result_transport(tmp_path):
    """Core Execute publishes a live Object hidden in a nested ordinary result graph."""
    repo = Repo(DirStore(tmp_path / "state"))
    def target():
        return {"nested": [SavedValue()]}

    invocation = encode_invocation(target, (), {}, repo=repo)
    from dryml.core.execute_codec import decode_outcome

    outcome = decode_outcome(invoke_invocation(invocation, repo=repo), repo=repo)
    assert outcome["success"]
    assert outcome["result"]["nested"][0].object_id is not None


def test_malformed_nodes_fail_before_symbol_resolution_without_sensitive_fields(tmp_path, monkeypatch):
    """Closed graph validation rejects unknown fields before imports or invocation."""
    repo = Repo(DirStore(tmp_path / "state"))
    secret = "transport-secret-marker"
    payload = dill.dumps({
        "version": 2, "root": 0,
        "nodes": [{"tag": "import", "module": "math", "qualname": None, "extra": secret}],
    }, protocol=5)
    monkeypatch.setattr(ImportRef, "resolve", lambda self: pytest.fail("symbol resolution ran"))

    with pytest.raises(CoreCallCodecError, match="malformed graph node") as error:
        invoke_invocation(payload, repo=repo)
    assert secret not in str(error.value)


def test_malformed_path_kind_fails_with_codec_error(tmp_path):
    """An unhashable path kind is rejected without leaking a raw TypeError."""
    repo = Repo(DirStore(tmp_path / "state"))
    payload = dill.dumps({
        "version": 2, "root": 0,
        "nodes": [{"tag": "path", "kind": [], "value": "marker"}],
    }, protocol=5)

    with pytest.raises(CoreCallCodecError, match="malformed path"):
        invoke_invocation(payload, repo=repo)
