"""Exact-load selection preserves one captured metadata authority per snapshot."""

import inspect
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest

import dryml.environments as envs
from dryml.core import (
    LineageMetadata, MetadataConflictError, Object, Repo, SaveRouting, Selector, Serializable,
    load_state_ref,
)
from dryml.core.repo import RepoLoadError
from dryml.core.store.dir import DirStore
from dryml.core.utils.graph.path import GraphPath


class EvidenceValue(Serializable):
    """Stateful value with observable construction and restore boundaries."""

    constructions = 0
    restores = 0

    def __init__(self, value):
        type(self).constructions += 1
        self.value = value

    def save_state_to_dir_imp(self, directory, *, codec):
        Path(directory, "value").write_text(str(self.value), encoding="ascii")

    def restore_state_from_dir_imp(self, directory, *, codec):
        type(self).restores += 1
        self.value = int(Path(directory, "value").read_text(encoding="ascii"))


class EvidenceRoot(Object):
    """Stateless parent used to place its stateful child in a routed Store."""

    def __init__(self, child):
        self.child = child


class StatelessEvidenceRoot(Object):
    """Stateless exact-reference root used to exercise unknown lineage facts."""

    def __init__(self, value):
        self.value = value


class StatelessSeedContainer(Object):
    """Stateless root holding materializing stateless exact-reference seeds."""

    def __init__(self, seeds):
        self.seeds = seeds


@pytest.fixture(autouse=True)
def synthetic_environment(monkeypatch):
    """Keep snapshot evidence stable without probing the host inventory."""

    monkeypatch.setattr(
        "dryml.environments.introspection.inspect_current",
        lambda: envs.EnvironmentRecord(
            python=envs.PythonRecord("3.12.0", "CPython"),
            platform=envs.PlatformRecord("Linux", "1", "v", "x86_64", "Linux-x86_64"),
            distributions={},
            dryml=envs.DrymlRuntimeRecord(),
        ),
    )


def test_exact_load_reinstalls_persisted_lineage_before_later_save(tmp_path):
    """A reopened exact identity retains its original creation fact on resave."""

    store = DirStore(tmp_path / "store")
    source = Repo(store)
    state = source.save_object(EvidenceValue(1, repo=source))
    expected = source.get_snapshot_metadata(state).lineages[GraphPath()].created_at
    assert expected is not None

    reopened = Repo(DirStore.open_existing(store.base_dir))
    loaded = reopened.load_state_ref(state, reuse_live="never")
    loaded.value = 2
    changed = reopened.save_object(loaded)

    assert changed != state
    assert reopened.get_snapshot_metadata(changed).lineages[GraphPath()].created_at == expected


def test_exact_load_rejects_conflicting_replica_evidence_before_construction(tmp_path):
    """Unqualified replicas disagreeing about one snapshot never pick by order."""

    first = DirStore(tmp_path / "first")
    second = DirStore(tmp_path / "second")
    source = Repo(first)
    state = source.save_object(EvidenceValue(1, repo=source))
    Repo(second).save_object(source.load_state_ref(state), store=second)

    EvidenceValue.constructions = 0
    for stores in ((first, second), (second, first)):
        with pytest.raises(MetadataConflictError, match="snapshot evidence"):
            Repo(stores).load_state_ref(state, reuse_live="never")

    assert EvidenceValue.constructions == 0


def test_exact_load_selectors_choose_root_and_routed_child_authority(tmp_path):
    """Root and independently routed child selectors remain explicit authority cuts."""

    root_store = DirStore(tmp_path / "root")
    child_store = DirStore(tmp_path / "child")
    writer = Repo(
        [root_store, child_store],
        save_routing=SaveRouting(
            ((Selector(EvidenceRoot), root_store), (Selector(EvidenceValue), child_store)),
        ),
    )
    state = writer.save_object(
        EvidenceRoot(EvidenceValue(3, repo=writer), repo=writer), deep_capture=True,
    )
    child_state = state.at(next(iter(state.object.objects)))
    reader = Repo([root_store, child_store])

    loaded = reader.load_state_ref(
        state,
        reuse_live="never",
        source_store=root_store,
        source_stores={child_state: child_store},
    )

    assert loaded.child.value == 3


def test_exact_load_rejects_conflicting_selected_child_lineage_before_hooks(tmp_path, monkeypatch):
    """A selected routed child cannot disagree with the enclosing lineage cut."""

    root_store = DirStore(tmp_path / "root")
    child_store = DirStore(tmp_path / "child")
    writer = Repo(
        [root_store, child_store],
        save_routing=SaveRouting(
            ((Selector(EvidenceRoot), root_store), (Selector(EvidenceValue), child_store)),
        ),
    )
    state = writer.save_object(
        EvidenceRoot(EvidenceValue(3, repo=writer), repo=writer), deep_capture=True,
    )
    child_path = next(iter(state.object.objects))
    child_state = state.at(child_path)
    metadata = child_store.read_snapshot_metadata(child_state.digest())
    fact = metadata.lineages[GraphPath()]
    conflicting = replace(
        metadata,
        lineages={
            GraphPath(): LineageMetadata(
                fact.object_ref, "known", fact.created_at + timedelta(seconds=1),
            ),
        },
    )
    original = child_store.read_snapshot_metadata
    monkeypatch.setattr(
        child_store,
        "read_snapshot_metadata",
        lambda digest: conflicting if digest == child_state.digest() else original(digest),
    )
    EvidenceValue.constructions = 0
    EvidenceValue.restores = 0

    with pytest.raises(MetadataConflictError, match="lineage evidence"):
        Repo([root_store, child_store]).load_state_ref(
            state,
            reuse_live="never",
            source_store=root_store,
            source_stores={child_state: child_store},
        )

    assert EvidenceValue.constructions == 0
    assert EvidenceValue.restores == 0


def test_exact_load_requires_explicit_choice_for_conflicting_child_replicas(tmp_path):
    """A routed child replica conflict is rejected until its source is selected."""

    root_store = DirStore(tmp_path / "root")
    child_store = DirStore(tmp_path / "child")
    replica_store = DirStore(tmp_path / "replica")
    writer = Repo(
        [root_store, child_store],
        save_routing=SaveRouting(
            ((Selector(EvidenceRoot), root_store), (Selector(EvidenceValue), child_store)),
        ),
    )
    state = writer.save_object(
        EvidenceRoot(EvidenceValue(3, repo=writer), repo=writer), deep_capture=True,
    )
    child_state = state.at(next(iter(state.object.objects)))
    replica = Repo(child_store).load_state_ref(child_state, reuse_live="never")
    Repo(replica_store).save_object(replica, store=replica_store)
    reader = Repo([root_store, child_store, replica_store])
    EvidenceValue.constructions = 0
    EvidenceValue.restores = 0

    with pytest.raises(MetadataConflictError, match="snapshot evidence"):
        reader.load_state_ref(state, reuse_live="never", source_store=root_store)

    assert EvidenceValue.constructions == 0
    assert EvidenceValue.restores == 0
    assert reader.load_state_ref(
        state,
        reuse_live="never",
        source_store=root_store,
        source_stores={child_state: child_store},
    ).child.value == 3


def test_exact_load_rejects_corrupt_selected_root_payload_without_replica_fallback(tmp_path):
    """A selected root's advertised local payload is not treated as routed absence."""

    first = DirStore(tmp_path / "first")
    second = DirStore(tmp_path / "second")
    source = Repo(first)
    child = EvidenceValue(3, repo=source)
    child_state = source.save_object(child)
    state = source.save_object(
        EvidenceRoot(child, repo=source), deep_capture=True,
    )
    child_path = next(iter(state.object.objects))
    assert state.at(child_path) == child_state
    Repo(second).save_object(
        source.load_state_ref(child_state, reuse_live="never"), store=second,
    )
    Path(first.open_local_state(state, child_path).handle, "data", "value").unlink()
    EvidenceValue.constructions = 0
    EvidenceValue.restores = 0

    with pytest.raises(RepoLoadError, match="local state"):
        Repo([second, first]).load_state_ref(
            state, reuse_live="never", source_store=first,
        )

    assert EvidenceValue.constructions == 0
    assert EvidenceValue.restores == 0


def test_exact_load_keeps_unknown_lineages_per_stateless_reference(tmp_path):
    """Independent stateless roots do not collide in the ObjectId lineage index."""

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    first = repo.save_object(StatelessEvidenceRoot("first", repo=repo))
    second = repo.save_object(StatelessEvidenceRoot("second", repo=repo))
    state = repo.save_object(StatelessSeedContainer([first, second], repo=repo))

    from dryml.core.materialization import build_exact_state_load_plan

    plan = build_exact_state_load_plan(Repo(store), state)

    assert None not in plan.lineage_facts_by_object_id
    assert set(plan.lineage_facts_by_reference) == {
        state.digest(), first.digest(), second.digest(),
    }
    assert Repo(store).load_state_ref(state, reuse_live="never").seeds[0].value == "first"


def test_exact_load_rejects_bad_selectors_before_construction_or_restore(tmp_path):
    """Disconnected, nonholding, and unrelated selectors fail before side effects."""

    source_store = DirStore(tmp_path / "source")
    empty_store = DirStore(tmp_path / "empty")
    detached_store = DirStore(tmp_path / "detached")
    source = Repo(source_store)
    state = source.save_object(EvidenceValue(1, repo=source))
    unrelated = source.save_object(EvidenceValue(2, repo=source))
    reader = Repo([source_store, empty_store])
    EvidenceValue.constructions = 0
    EvidenceValue.restores = 0

    with pytest.raises(ValueError, match="connected"):
        reader.load_state_ref(state, source_store=detached_store)
    with pytest.raises(RepoLoadError, match="selected source"):
        reader.load_state_ref(state, source_store=empty_store)
    with pytest.raises(ValueError, match="outside the exact StateRef closure"):
        reader.load_state_ref(state, source_stores={unrelated: source_store})

    assert EvidenceValue.constructions == 0
    assert EvidenceValue.restores == 0


def test_exact_load_rejects_selected_lineage_conflict_with_live_cache(tmp_path, monkeypatch):
    """Selected facts cannot retimestamp a cached exact identity before restore."""

    first = DirStore(tmp_path / "first")
    second = DirStore(tmp_path / "second")
    source = Repo(first)
    state = source.save_object(EvidenceValue(1, repo=source))
    Repo(second).save_object(source.load_state_ref(state), store=second)
    reader = Repo([first, second])
    cached = reader.load_state_ref(state, reuse_live="never", source_store=first)
    metadata = second.read_snapshot_metadata(state.digest())
    fact = metadata.lineages[GraphPath()]
    conflicting = replace(
        metadata,
        lineages={
            GraphPath(): LineageMetadata(
                fact.object_ref, "known", fact.created_at + timedelta(seconds=1),
            ),
        },
    )
    original = second.read_snapshot_metadata
    monkeypatch.setattr(
        second,
        "read_snapshot_metadata",
        lambda digest: conflicting if digest == state.digest() else original(digest),
    )
    EvidenceValue.constructions = 0
    EvidenceValue.restores = 0

    with pytest.raises(MetadataConflictError, match="Live cached identity"):
        reader.load_state_ref(state, reuse_live="greedy", source_store=second)

    assert EvidenceValue.constructions == 0
    assert EvidenceValue.restores == 0
    assert cached.value == 1


def test_targeted_restore_rejects_selected_lineage_conflict_for_uncached_target(tmp_path, monkeypatch):
    """Targeted restore cannot retimestamp an uncached supplied live target."""

    first = DirStore(tmp_path / "first")
    second = DirStore(tmp_path / "second")
    source = Repo(first)
    state = source.save_object(EvidenceValue(1, repo=source))
    Repo(second).save_object(source.load_state_ref(state, reuse_live="never"), store=second)
    reader = Repo([first, second])
    target = reader.load_state_ref(
        state, reuse_live="never", cache="none", source_store=first,
    )
    metadata = second.read_snapshot_metadata(state.digest())
    fact = metadata.lineages[GraphPath()]
    conflicting = replace(
        metadata,
        lineages={
            GraphPath(): LineageMetadata(
                fact.object_ref, "known", fact.created_at + timedelta(seconds=1),
            ),
        },
    )
    original = second.read_snapshot_metadata
    monkeypatch.setattr(
        second,
        "read_snapshot_metadata",
        lambda digest: conflicting if digest == state.digest() else original(digest),
    )
    EvidenceValue.restores = 0

    with pytest.raises(MetadataConflictError, match="Target live identity"):
        reader.restore_state_ref_into(target, state, source_store=second)

    assert EvidenceValue.restores == 0


def test_all_exact_load_entry_points_expose_and_forward_evidence_selectors(monkeypatch):
    """Public exact-load wrappers pass selector values through to Repo unchanged."""

    for callable_ in (
            Repo.load_state_ref, Repo.restore_state_ref_into, load_state_ref):
        parameters = inspect.signature(callable_).parameters
        assert "source_store" in parameters
        assert "source_stores" in parameters

    captured = {}
    repo = Repo()
    state_ref = object()
    source_store = object()
    source_stores = {object(): object()}
    monkeypatch.setattr(
        Repo,
        "load_state_ref",
        lambda self, state, **kwargs: captured.update(state=state, **kwargs) or state,
    )

    assert load_state_ref(
        state_ref,
        repo=repo,
        source_store=source_store,
        source_stores=source_stores,
    ) is state_ref
    assert captured == {
        "state": state_ref,
        "reuse_live": "matching",
        "cache": "weak",
        "source_store": source_store,
        "source_stores": source_stores,
    }
