from pathlib import Path

import pytest

pytestmark = pytest.mark.usefixtures("fixed_snapshot_environment")

from dryml.core import Object, Repo, Serializable
from dryml.core.object import Pickleable
from dryml.core.repo import RepoLoadError, RepoSaveError
from dryml.core.store.dir import DirStore
from dryml.core.utils.graph.path import GraphPath, Parameter, Index


class RestoreIntoValue(Serializable):
    def __init__(self, value):
        self.value = value

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        Path(dest_dir, "value").write_text(str(self.value), encoding="ascii")

    def restore_state_from_dir_imp(self, src_dir, *, codec):
        self.value = int(Path(src_dir, "value").read_text(encoding="ascii"))


class RestoreIntoPair(Object):
    def __init__(self, child):
        self.first = child
        self.second = child


class RestoreIntoIndependentPair(Object):
    def __init__(self, first, second):
        self.first = first
        self.second = second


class FailingRestoreIntoValue(RestoreIntoValue):
    fail = False

    def restore_state_from_dir_imp(self, src_dir, *, codec):
        super().restore_state_from_dir_imp(src_dir, codec=codec)
        if type(self).fail:
            raise RuntimeError("restore hook failed")


class SelectivelyFailingRestoreValue(RestoreIntoValue):
    failing_target = None

    def restore_state_from_dir_imp(self, src_dir, *, codec):
        super().restore_state_from_dir_imp(src_dir, codec=codec)
        if self is type(self).failing_target:
            raise RuntimeError("selected restore hook failed")


class RestoreIntoPickleable(Pickleable):
    def __init__(self, value):
        self.value = value


class RestoreIntoPickleableParent(Pickleable):
    def __init__(self, child, value):
        self.child = child
        self.value = value


class CountingEmbeddedValue(RestoreIntoValue):
    restores = 0

    def restore_state_from_dir_imp(self, src_dir, *, codec):
        type(self).restores += 1
        super().restore_state_from_dir_imp(src_dir, codec=codec)


class CountingEmbeddedParent(Serializable):
    observed_child_value = None

    def __init__(self, child):
        self.child = child

    def restore_state_from_dir_imp(self, src_dir, *, codec):
        type(self).observed_child_value = self.child[0].value


def test_targeted_restore_keeps_the_supplied_live_identity_and_forces_hook(tmp_path):
    repo = Repo(DirStore(tmp_path / "store"))
    obj = RestoreIntoValue(1, repo=repo)
    state = repo.save_object(obj)
    obj.value = 99

    with repo.reserve_state_graph(obj) as reservation:
        restored = repo.restore_state_ref_into(obj, state, reservation=reservation)

    assert restored is state
    assert obj.value == 1
    assert obj.last_state_ref is state


def test_targeted_restore_preserves_alias_identity_and_invalidates_after_hook_failure(tmp_path):
    repo = Repo(DirStore(tmp_path / "store"))
    child = FailingRestoreIntoValue(2, repo=repo)
    root = RestoreIntoPair(child, repo=repo)
    state = repo.save_object(root)
    child.value = 99

    repo.restore_state_ref_into(root, state)
    assert root.first is root.second is child
    assert child.value == 2

    FailingRestoreIntoValue.fail = True
    try:
        with pytest.raises(RepoLoadError, match="Targeted exact restore failed"):
            repo.restore_state_ref_into(root, state)
    finally:
        FailingRestoreIntoValue.fail = False

    assert root._restore_failed
    assert child._restore_failed
    assert root.last_state_ref is None
    assert child._last_state_hash is None
    with pytest.raises(RepoSaveError, match="invalidated"):
        repo.save_object(root)
    with pytest.raises(RepoLoadError, match="invalidated"):
        repo.restore_state_ref_into(root, state)
    assert child not in repo._all_live_candidates()
    with pytest.raises(RepoLoadError, match="fresh materialization"):
        repo.materialize_boundary((root,))

    fresh = Repo(DirStore(tmp_path / "store")).load_state_ref(state, reuse_live="never")
    assert not fresh._restore_failed
    assert fresh.first.value == 2
    assert fresh.first.object_id == child.object_id


def test_restore_failure_invalidates_equal_definition_siblings(tmp_path):
    repo = Repo(DirStore(tmp_path / "store"))
    first = SelectivelyFailingRestoreValue(2, repo=repo)
    second = SelectivelyFailingRestoreValue(2, repo=repo)
    root = RestoreIntoIndependentPair(first, second, repo=repo)
    state = repo.save_object(root)
    SelectivelyFailingRestoreValue.failing_target = second

    try:
        with pytest.raises(RepoLoadError, match="Targeted exact restore failed"):
            repo.restore_state_ref_into(root, state)
    finally:
        SelectivelyFailingRestoreValue.failing_target = None

    assert root._restore_failed
    assert first._restore_failed
    assert second._restore_failed
    with pytest.raises(RepoSaveError, match="invalidated"):
        repo.save_object(second)


def test_targeted_restore_preflight_failure_leaves_target_valid_and_pickle_payload_replaces(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    obj = RestoreIntoPickleable(1, repo=repo)
    state = repo.save_object(obj)
    obj.extra = "discard me"
    repo.restore_state_ref_into(obj, state)
    assert not hasattr(obj, "extra")
    assert obj.value == 1

    Path(store.state_data_path(state.state.key), "heavy.pkl").unlink()
    with pytest.raises(RepoLoadError, match="preflight is incomplete"):
        repo.restore_state_ref_into(obj, state)
    assert not obj._restore_failed


def test_targeted_restore_walks_embedded_reference_closure_once_before_parent_hook(tmp_path):
    repo = Repo(DirStore(tmp_path / "store"))
    child = CountingEmbeddedValue(1, repo=repo)
    seed = repo.save_object(child)
    root = CountingEmbeddedParent([seed, seed], repo=repo)
    state = repo.save_object(root)
    child = root.child[0]
    child.value = 99
    CountingEmbeddedValue.restores = 0
    CountingEmbeddedParent.observed_child_value = None

    repo.restore_state_ref_into(root, state)

    assert root.child[0] is root.child[1] is child
    assert child.value == 1
    assert CountingEmbeddedValue.restores == 1
    assert CountingEmbeddedParent.observed_child_value == 1


def test_pickleable_targeted_restore_preserves_live_graph_bindings(tmp_path):
    repo = Repo(DirStore(tmp_path / "store"))
    child = RestoreIntoValue(2, repo=repo)
    root = RestoreIntoPickleableParent(child, 1, repo=repo)
    state = repo.save_object(root)
    root.extra = "discard me"
    root.value = 99

    repo.restore_state_ref_into(root, state)

    assert root.child is child
    assert root.child.object_id == child.object_id
    assert root.value == 1
    assert not hasattr(root, "extra")
    assert repo.describe(root).object_projection_equal(state)


def test_full_restore_checks_original_projection_before_hooks(tmp_path):
    from dryml.core import ObjectId
    repo = Repo(DirStore(tmp_path / "store"))
    obj = CountingEmbeddedValue(1, repo=repo)
    saved = repo.save(obj)
    wrong = saved.replace_bindings(identities={"$": ObjectId()})
    obj.value = 99
    CountingEmbeddedValue.restores = 0
    with pytest.raises(ValueError, match="original supplied graph"):
        repo.restore(obj, wrong)
    assert CountingEmbeddedValue.restores == 0
    assert obj.value == 99 and not obj._restore_failed


def test_prepared_restore_handoff_is_one_shot_and_never_a_history_shortcut(tmp_path):
    repo = Repo(DirStore(tmp_path / "store"))
    obj = CountingEmbeddedValue(1, repo=repo)
    saved = repo.save(obj)
    obj.value = 99
    CountingEmbeddedValue.restores = 0
    prepared = repo.prepare_restore(obj, saved)
    assert obj.value == 99 and CountingEmbeddedValue.restores == 0
    with prepared:
        assert prepared.apply() is saved
    assert obj.value == 1 and CountingEmbeddedValue.restores == 1
    with pytest.raises(RuntimeError, match="consumed"):
        prepared.apply()
    obj.value = 55
    repo.restore(obj, saved)
    assert obj.value == 1 and CountingEmbeddedValue.restores == 2


@pytest.mark.parametrize("entrypoint", ("prepare", "restore"))
def test_full_restore_rebases_decoded_tied_set_states_and_lineages(
        tmp_path, monkeypatch, entrypoint):
    """Unordered path ordinals cannot transfer state or lineage between identities."""
    import builtins
    import json
    import subprocess
    import sys
    from datetime import datetime, timezone

    from dryml.core.node import _owned_nodes
    from dryml.core.node_codec import decode_node_graph, encode_node_graph
    from dryml.core.node_prepare import NodePreparation
    from dryml.core.snapshot_capture import install_lineage_fact
    from dryml.core.utils.graph import value as graph_values

    repo = Repo(DirStore(tmp_path / "store", query_index="none"))
    first = CountingEmbeddedValue(1, repo=repo)
    second = CountingEmbeddedValue(1, repo=repo)
    # Align this fixture's initial Node probes with the live set's ordinary repr
    # order so unrelated construction-path ordering cannot mask the restore case.
    occurrence_ranks = {
        child.definition._node_id: rank
        for rank, child in enumerate(sorted((first, second), key=repr))
    }
    monkeypatch.setattr(
        graph_values, "id", lambda token: occurrence_ranks.get(token, builtins.id(token)), raising=False,
    )
    root = RestoreIntoPair([{first, second}, first], repo=repo)
    expected_values = {first.object_id: 11, second.object_id: 22}
    expected_lineages = {
        first.object_id: datetime(2024, 1, 1, tzinfo=timezone.utc),
        second.object_id: datetime(2024, 1, 2, tzinfo=timezone.utc),
    }
    for child in (first, second):
        child.value = expected_values[child.object_id]
        install_lineage_fact(child, child.object_id, expected_lineages[child.object_id])
    saved = repo.save(root)
    packet = encode_node_graph(NodePreparation((saved,)))
    if entrypoint == "prepare":
        process = subprocess.run(
            [sys.executable, "-c", """
import json, sys
from dryml.core.node_codec import decode_node_graph, encode_node_graph
from dryml.core.node_prepare import NodePreparation
saved = decode_node_graph(json.load(sys.stdin))
json.dump(encode_node_graph(NodePreparation((saved,))), sys.stdout)
"""], input=json.dumps(packet), text=True, capture_output=True,
            check=True, timeout=30,
        )
        packet = json.loads(process.stdout)
    supplied = decode_node_graph(packet)
    receiver = repo.describe(root)
    live_members = graph_values.iter_set_members(receiver.parameters["child"][0])
    assert len({segment.fingerprint for segment, _ in live_members}) == 1
    assert len(live_members) == 2
    live_tokens = {node.object_id: node._occurrence for _, node in live_members}
    supplied_members = supplied.parameters["child"][0]
    assert all(node._occurrence is not live_tokens[node.object_id] for node in supplied_members)

    # Model the opposite fresh-token address order without changing live paths or
    # relying on allocator luck. Only the separately decoded occurrence probes vary.
    opposite_ranks = {
        node._occurrence: graph_values.id(live_members[1 - index][1]._occurrence)
        for index, (_, live) in enumerate(live_members)
        for node in supplied_members if node.object_id == live.object_id
    }
    occurrence_ranks.update(opposite_ranks)
    supplied_order = graph_values.iter_set_members(supplied_members)
    assert [node.object_id for _, node in supplied_order] == [node.object_id for _, node in reversed(live_members)]
    assert supplied.object_projection_equal(receiver)

    targets = {child.object_id: child for child in (first, second)}
    calls = []
    original_restore = CountingEmbeddedValue.restore_state_from_dir_imp

    def restore(target, directory, *, codec):
        calls.append(target.object_id)
        original_restore(target, directory, codec=codec)

    monkeypatch.setattr(CountingEmbeddedValue, "restore_state_from_dir_imp", restore)
    for child in targets.values():
        child.value = 99
        if entrypoint == "prepare":
            del child._lineage_fact_object_id
            del child._lineage_created_at
    if entrypoint == "prepare":
        with repo.prepare_restore(root, supplied) as prepared:
            assert calls == []
            assert all(child.value == 99 for child in targets.values())
            assert prepared.apply() is supplied
        with pytest.raises(RuntimeError, match="consumed"):
            prepared.apply()
    else:
        assert repo.restore(root, supplied) is supplied

    assert sorted(calls, key=str) == sorted(targets, key=str)
    assert {
        identity: (child.value, child._lineage_fact_object_id, child._lineage_created_at)
        for identity, child in targets.items()
    } == {
        identity: (expected_values[identity], identity, expected_lineages[identity])
        for identity in targets
    }
    for identity, child in targets.items():
        assert child.object_id == identity
        assert child._lineage_fact_object_id == identity
        assert child._lineage_created_at == expected_lineages[identity]
        assert child._construction_node._occurrence is live_tokens[identity]
    assert root.first is root.second
    assert root.first[1] is first and root.first[0] == {first, second}
    assert root.last_state_ref is supplied
    assert repo.describe(root).object_projection_equal(supplied)
    recaptured = repo.save(root)
    assert recaptured.exact_digest() == saved.exact_digest()

    first.value, second.value = 77, 88
    calls.clear()
    destination = next(path for path, node in _owned_nodes(receiver) if node.object_id == first.object_id)
    selected = next(node.state for _, node in _owned_nodes(supplied) if node.object_id == first.object_id)
    assert repo.apply_state(root, {destination: selected}) is None
    assert calls == [first.object_id]
    assert first.value == 11 and second.value == 88
    assert root.first is root.second and root.first[1] is first


def test_prepared_restore_is_owned_and_abandonment_releases_without_hooks(tmp_path):
    from threading import Thread
    repo = Repo(DirStore(tmp_path / "store"))
    obj = CountingEmbeddedValue(1, repo=repo)
    saved = repo.save(obj)
    CountingEmbeddedValue.restores = 0
    prepared = repo.prepare_restore(obj, saved)
    errors = []

    def attempt():
        try:
            prepared.apply()
        except RuntimeError as error:
            errors.append(str(error))

    thread = Thread(target=attempt)
    thread.start()
    thread.join(5)
    assert not thread.is_alive() and len(errors) == 1
    prepared.close()
    prepared.close()
    assert CountingEmbeddedValue.restores == 0 and not obj._restore_failed
    with repo.reserve_state_graph(obj):
        pass


def test_restore_adopts_s0_construction_after_s1_capture(tmp_path):
    from dryml.core import Definition
    from dryml.core.node_views import creation_cdef
    repo = Repo(DirStore(tmp_path / "store"))
    donor = CountingEmbeddedValue(1, repo=repo)
    s0 = repo.save(donor)
    root = Definition(CountingEmbeddedParent, [s0, s0]).build(repo=repo, cache="none")
    child = root.child[0]
    child.value = 2
    saved = repo.save(root)
    path = GraphPath((Parameter("child"), Index(0)))
    s1 = saved.at(path)
    assert s1.state.key != s0.state.key
    assert saved.construction_view.at(path).state.key == s0.state.key
    fresh = repo.load_state_ref(saved, reuse_live="never", cache="none")
    assert fresh.child[0] is fresh.child[1]
    assert fresh.child[0].value == 2
    assert fresh.child[0].object_id == child.object_id
    target = Definition(CountingEmbeddedParent, [s1, s1]).with_identity(saved.object_id).build(repo=repo, cache="none")
    assert repo.describe(target).creation_digest() != saved.creation_digest()
    identities = target.object_id, target.child[0].object_id
    repo.restore(target, saved)
    assert (target.object_id, target.child[0].object_id) == identities
    assert target.child[0] is target.child[1]
    recaptured = repo.save(target)
    assert creation_cdef(recaptured).exact_digest() == creation_cdef(saved).exact_digest()
    assert recaptured.construction_view.at(path).state.key == s0.state.key


def test_foreign_compatible_apply_keeps_donor_key_and_destination_creation(tmp_path):
    from dryml.core import Definition
    repo = Repo(DirStore(tmp_path / "store"))
    donor_child = repo.save(RestoreIntoValue(1, repo=repo))
    other_child = repo.save(RestoreIntoValue(1, repo=repo))
    donor = Definition(RestoreIntoPickleableParent, donor_child, 1).build(repo=repo, cache="none")
    destination = Definition(RestoreIntoPickleableParent, other_child, 1).build(repo=repo, cache="none")
    donor.value = 7
    selected = repo.save(donor).state
    original_creation = repo.describe(destination).creation_digest()
    assert original_creation != selected.key.cdef_digest
    identity, child = destination.object_id, destination.child
    child.value = 99
    repo.apply_state(destination, {GraphPath(): selected})
    assert destination.value == 7
    assert destination.child is child and child.value == 99
    assert destination.object_id == identity
    assert repo.describe(destination).state.key == selected.key
    captured = repo.save(destination)
    assert captured.state.key.cdef_digest == original_creation
    assert captured.state.key != selected.key


def test_missing_later_selection_preflights_before_earlier_hook(tmp_path):
    repo = Repo(DirStore(tmp_path / "store"))
    first = CountingEmbeddedValue(1, repo=repo)
    second = CountingEmbeddedValue(2, repo=repo)
    root = RestoreIntoIndependentPair(first, second, repo=repo)
    saved = repo.save(root)
    missing = saved.at(GraphPath((Parameter("second"),))).state
    Path(repo.default_store.state_data_path(missing.key), "value").unlink()
    first.value = second.value = 99
    CountingEmbeddedValue.restores = 0
    with pytest.raises(RepoLoadError, match="preflight is incomplete"):
        repo.apply_state(root, saved.state_spec())
    assert CountingEmbeddedValue.restores == 0
    assert first.value == second.value == 99
    assert not any(node._restore_failed for node in (root, first, second))


@pytest.mark.speed_heavy
@pytest.mark.parametrize("kind", ("xgb-regressor", "stats-logit"))
@pytest.mark.parametrize("store_kind", ("dir", "zip"))
def test_native_graph_restore_unfitted_fitted_invalidation_and_fresh_recovery(
        tmp_path, monkeypatch, kind, store_kind):
    """Native candidate isolation cannot rescue a Core-invalidated aliased graph."""
    import numpy as np
    from dryml.core.store.zip import ZipStore
    from tests.models.test_estimator_experiments import _arrays, _recipe

    destination = tmp_path / ("store" if store_kind == "dir" else "store.zip")
    store_cls = DirStore if store_kind == "dir" else ZipStore
    repo = Repo(store_cls(destination))
    model, _ = _recipe(kind)
    root = RestoreIntoPair(model, repo=repo)
    x, y = _arrays(kind)
    try:
        empty = repo.save_object(root, deep_capture=True)
        model.fit(x, y)
        fitted = repo.save_object(root, deep_capture=True)
        repo.default_store.commit()
        payload, metadata = model._accepted_fit.payload, model._accepted_fit.metadata
        expected = model(x)
        object_id = model.object_id
        repo.restore_state_ref_into(root, empty)
        assert root.first is root.second is model and model.object_id == object_id
        assert model._accepted_fit is None
        with pytest.raises(ValueError, match="unfitted"):
            model(x)
        repo.restore_state_ref_into(root, fitted)
        assert root.first is root.second is model and model.object_id == object_id
        assert model._accepted_fit.payload == payload
        assert model._accepted_fit.metadata == metadata
        np.testing.assert_array_equal(model(x), expected)
        prior = model._accepted_fit
        assert root.last_state_ref == fitted
        assert model._last_state_hash is not None
        for node in (root, model):
            assert any(candidate is node for candidate in repo._all_live_candidates())

        def reject(candidate):
            raise ValueError("entered native restore validation")

        monkeypatch.setattr(model, "_validate_native", reject)
        with pytest.raises(RepoLoadError, match="Targeted exact restore failed"):
            repo.restore_state_ref_into(root, fitted)
        assert model._accepted_fit is prior
        for node in (root, model):
            assert node._restore_failed
            assert node.last_state_ref is None and node._last_state_hash is None
            assert all(candidate is not node for candidate in repo._all_live_candidates())
            with pytest.raises(RepoSaveError, match="invalidated"):
                repo.save_object(node)
        with pytest.raises(RepoLoadError, match="invalidated"):
            repo.restore_state_ref_into(root, fitted)
        fresh_repo = Repo(store_cls(destination))
        try:
            fresh_empty = fresh_repo.load_state_ref(empty, reuse_live="never", cache="none")
            fresh = fresh_repo.load_state_ref(fitted, reuse_live="never", cache="none")
            assert fresh_empty.first._accepted_fit is None
            assert fresh is not root and fresh.first is not model
            assert fresh.first is fresh.second
            assert fresh.first.object_id == object_id and fresh.object_ref == fitted.object
            assert not fresh._restore_failed and not fresh.first._restore_failed
            assert fresh.last_state_ref == fitted
            assert fresh.first._accepted_fit.payload == payload
            assert fresh.first._accepted_fit.metadata == metadata
            np.testing.assert_array_equal(fresh.first(x), expected)
        finally:
            fresh_repo.close(flush=False)
    finally:
        repo.close(flush=False)
