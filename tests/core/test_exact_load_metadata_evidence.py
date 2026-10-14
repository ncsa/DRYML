"""Exact-load selection preserves one captured metadata authority per snapshot."""

import inspect
import hashlib
import sys
from collections import Counter
from contextlib import contextmanager
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from threading import Thread

import pytest

import dryml.environments as envs
from dryml.core import (
    Definition, LineageMetadata, MetadataConflictError, Object, Repo, SaveRouting,
    Selector, Serializable, load_state_ref,
)
from dryml.core.repo import RepoLoadError
from dryml.core.store.dir import DirStore
from dryml.core.utils.graph.path import GraphPath, Parameter


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


@contextmanager
def exact_work_counts():
    """Observe owning work without replacing acceleration-guarded callables."""
    counts = Counter()
    previous = sys.getprofile()

    def observe(frame, event, arg):
        if event == "call":
            name = frame.f_code.co_name
            if name in {"_read_snapshot", "_validate_local_state_dir", "validate_payload"}:
                counts[name] += 1
            if name == "_read_file":
                kind = frame.f_locals["record_type"].__name__
                counts[kind] += 1
            if name == "_read_json":
                counts["json_reads"] += 1
            if name == "graph_equal" and frame.f_back.f_code.co_name == "_validate_snapshot_locals":
                counts["path_definition_checks"] += 1
        if event == "c_call" and frame.f_code.co_name == "_validate_local_state_dir" and arg is hashlib.sha256:
            counts["definition_byte_hashes"] += 1
        if previous is not None:
            previous(frame, event, arg)

    sys.setprofile(observe)
    try:
        yield counts
    finally:
        sys.setprofile(previous)


@pytest.mark.parametrize("size", [1, 3, 7])
def test_exact_plan_work_budget(tmp_path, size, monkeypatch):
    """One bundle and K physical passes replace baseline 4/8 bundles and 1/9 passes."""
    import builtins
    from dryml.core.materialization import build_exact_state_load_plan

    store = DirStore(tmp_path / "store", query_index="none")
    writer = Repo(store)
    values = [EvidenceValue(i, repo=writer) for i in range(size)]
    root = values[0] if size == 1 else StatelessSeedContainer(values, repo=writer)
    state = writer.save_object(root, deep_capture=True)
    original_open = builtins.open
    payload_reads = []

    def observe_open(path, *args, **kwargs):
        if Path(path).name == "value" and Path(path).parent.name == "data":
            payload_reads.append(path)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", observe_open)
    assert store._can_reuse_exact_validation()
    with exact_work_counts() as counts:
        plan = build_exact_state_load_plan(Repo(store), state)
    assert len(plan.actions) == size
    assert counts["_read_snapshot"] == 1
    assert counts["validate_payload"] == size
    assert counts["_validate_local_state_dir"] == size
    assert counts["LocalStateManifest"] == size
    assert counts["DefinitionRecord"] == 2 * size + (size != 1)
    assert counts["definition_byte_hashes"] == size
    assert counts["json_reads"] == 3
    assert counts["path_definition_checks"] == size
    assert len(payload_reads) == size


def test_projected_eager_plan_checks_unrequested_corrupt_sibling(tmp_path):
    """Whole-snapshot promotion rejects sibling corruption before any hooks."""
    from dryml.core.materialization import build_exact_state_load_plan

    store = DirStore(tmp_path / "store", query_index="none")
    writer = Repo(store)
    state = writer.save_object(StatelessSeedContainer(
        [EvidenceValue(1, repo=writer), EvidenceValue(2, repo=writer)], repo=writer,
    ), deep_capture=True)
    first, second = state.states
    Path(store.open_local_state(state, second).handle, "data", "value").write_text("bad")
    with pytest.raises(RepoLoadError, match="local state"):
        build_exact_state_load_plan(Repo(store), state.at(first))
    deferred = build_exact_state_load_plan(Repo(store), state.at(first), _defer_payload=True)
    assert len(deferred.actions) == 1


def test_shared_physical_payload_is_hashed_once_for_distinct_paths(tmp_path):
    """Physical deduplication preserves both path actions and ObjectIds."""
    from dryml.core.materialization import build_exact_state_load_plan

    store = DirStore(tmp_path / "store", query_index="none")
    writer = Repo(store)
    state = writer.save_object(StatelessSeedContainer(
        [EvidenceValue(1, repo=writer), EvidenceValue(1, repo=writer)], repo=writer,
    ), deep_capture=True)
    assert len(state.states) == 2
    with exact_work_counts() as counts:
        plan = build_exact_state_load_plan(Repo(store), state)
    assert len(plan.actions) == 2
    assert plan.actions[0].object_id != plan.actions[1].object_id
    assert plan.actions[0].payload.handle == plan.actions[1].payload.handle
    assert counts["validate_payload"] == 1
    assert counts["LocalStateManifest"] == 1


def test_shared_payload_still_checks_every_definition_association(tmp_path):
    """Equal state hashes cannot authorize the wrong definition at a second path."""
    from dryml.core.materialization import build_exact_state_load_plan
    from dryml.records import GenericRecord, decode_record, encode_record

    store = DirStore(tmp_path / "store", query_index="none")
    writer = Repo(store)
    first, second = EvidenceValue(1, repo=writer), EvidenceValue(2, repo=writer)
    second.value = first.value
    state = writer.save_object(StatelessSeedContainer([first, second], repo=writer), deep_capture=True)
    assert len(set(state.states.values())) == 1
    directory = store.get_snapshot_directory(state)
    placement = decode_record(store._read_json(str(directory / "placement.json")))
    entries = [dict(item) for item in placement.data["local"]]
    entries[1]["directory"] = entries[0]["directory"]
    altered = encode_record(GenericRecord(placement.kind, placement.version, {
        **dict(placement.data), "local": entries,
    }))
    association = decode_record(store._read_json(str(directory / "snapshot.json")))
    store._write_json(str(directory / "placement.json"), altered)
    store._write_json(str(directory / "snapshot.json"), encode_record(GenericRecord(
        association.kind, association.version,
        {**dict(association.data), "placement_id": altered["id"]},
    )))
    with pytest.raises(RepoLoadError, match="definition does not match StateRef path"):
        build_exact_state_load_plan(Repo(store), state)


@pytest.mark.parametrize("sibling", ["state-ref.record", "metadata.json", "placement.json", "snapshot.json"])
def test_exact_plan_requires_every_bundle_sibling(tmp_path, sibling):
    """A fresh cut never substitutes previous bundle evidence for a missing file."""
    from dryml.core.materialization import build_exact_state_load_plan

    store = DirStore(tmp_path / "store", query_index="none")
    writer = Repo(store)
    state = writer.save_object(EvidenceValue(1, repo=writer))
    reader = Repo(store)
    plan = build_exact_state_load_plan(reader, state)
    Path(store.get_snapshot_directory(state), sibling).unlink()
    with pytest.raises(RepoLoadError, match="preflight is incomplete"):
        build_exact_state_load_plan(reader, state)
    assert Path(plan.actions[0].payload.handle).is_dir()


def test_exact_plan_retains_order_across_default_change_and_alias_views(tmp_path):
    """Fence deduplication neither changes captured order nor merges handle views."""
    from dryml.core.materialization import build_exact_state_load_plan

    first = DirStore(tmp_path / "first", query_index="none")
    writer = Repo(first)
    state = writer.save_object(EvidenceValue(1, repo=writer))
    alias = DirStore(first.base_dir, query_index="none")
    reader = Repo([first, alias])
    assert first.authority_fence_key() == alias.authority_fence_key()
    with reader._read_cut() as cut:
        failures = []

        def reorder():
            try:
                reader.set_default_store(alias)
            except BaseException as error:
                failures.append(error)

        thread = Thread(target=reorder)
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive() and not failures
        assert reader.stores[0] is alias
        with exact_work_counts() as counts:
            plan = build_exact_state_load_plan(reader, state, _read_cut=cut)
        assert plan.actions[0].store is first
        assert cut.stores == (first, alias)
        assert counts["_read_snapshot"] == 2
        assert len(cut.snapshots) == 2
    assert build_exact_state_load_plan(reader, state).actions[0].store is alias


@pytest.mark.parametrize("reader_name", [
    "read_state_ref_record", "read_snapshot_metadata", "open_local_state",
    "validate_local_state", "_read_snapshot", "_validate_local_state_dir",
])
def test_exact_reader_overrides_remain_authoritative(tmp_path, monkeypatch, reader_name):
    """An overridden built-in reader is called, never bypassed by private proof."""
    from dryml.core.materialization import build_exact_state_load_plan

    store = DirStore(tmp_path / "store", query_index="none")
    writer = Repo(store)
    state = writer.save_object(EvidenceValue(1, repo=writer))
    calls = []

    def reject(*args, **kwargs):
        calls.append(True)
        raise RuntimeError("override remains authoritative")

    monkeypatch.setattr(store, reader_name, reject)
    assert not store._can_reuse_exact_validation()
    with pytest.raises(RepoLoadError, match="override remains authoritative"):
        build_exact_state_load_plan(Repo(store), state)
    assert calls


def test_nested_independent_plan_does_not_inherit_outer_payload_proof(tmp_path):
    """Only explicitly joined internal work shares a cut; independent work is fresh."""
    from dryml.core.materialization import build_exact_state_load_plan

    store = DirStore(tmp_path / "store", query_index="none")
    writer = Repo(store)
    state = writer.save_object(EvidenceValue(1, repo=writer))
    reader = Repo(store)
    with reader._read_cut() as cut:
        with exact_work_counts() as counts:
            outer = build_exact_state_load_plan(reader, state, _read_cut=cut)
            joined = build_exact_state_load_plan(reader, state, _read_cut=cut)
            nested = build_exact_state_load_plan(reader, state)
        assert counts["_read_snapshot"] == 2
        assert counts["validate_payload"] == 2
        assert outer.actions[0].payload == joined.actions[0].payload == nested.actions[0].payload
        assert reader._topology_leases == 1
    assert reader._topology_leases == 0


def test_custom_lower_validator_keeps_supplied_manifest_and_none_return(tmp_path):
    """Custom subclasses preserve their old validator protocol without acceleration."""
    from dryml.core.materialization import build_exact_state_load_plan
    from dryml.core.store.records import LocalStateManifest

    class CustomDir(DirStore):
        calls = 0

        def _validate_local_state_dir(self, directory, manifest, *, defer_payload=False):
            assert isinstance(manifest, LocalStateManifest)
            self.calls += 1
            super()._validate_local_state_dir(directory, manifest, defer_payload=defer_payload)

    store = CustomDir(tmp_path / "store", query_index="none")
    writer = Repo(store)
    state = writer.save_object(EvidenceValue(1, repo=writer))
    assert not store._can_reuse_exact_validation()
    store.calls = 0
    with exact_work_counts() as counts:
        plan = build_exact_state_load_plan(Repo(store), state)
    assert len(plan.actions) == 1
    assert store.calls == 1
    assert counts["_read_snapshot"] == 4


def test_payload_validator_override_keeps_uncached_error(tmp_path, monkeypatch):
    """Lower manifest validation overrides cannot be bypassed by cut promotion."""
    from dryml.core.materialization import build_exact_state_load_plan
    from dryml.core.store.records import LocalStateManifest, StoreRecordError

    store = DirStore(tmp_path / "store", query_index="none")
    writer = Repo(store)
    state = writer.save_object(EvidenceValue(1, repo=writer))

    def reject(*args, **kwargs):
        raise StoreRecordError("overridden payload validator")

    monkeypatch.setattr(LocalStateManifest, "validate_payload", reject)
    assert not store._can_reuse_exact_validation()
    with pytest.raises(RepoLoadError, match="overridden payload validator"):
        build_exact_state_load_plan(Repo(store), state)


@pytest.mark.parametrize("component", ["manifest.record", "def.pkl"])
def test_corrupt_local_authority_rejects_before_hooks(tmp_path, component):
    """Bundle reuse does not bypass malformed local manifest/definition authority."""
    from dryml.core.materialization import build_exact_state_load_plan

    store = DirStore(tmp_path / "store", query_index="none")
    writer = Repo(store)
    state = writer.save_object(EvidenceValue(1, repo=writer))
    source = store.open_local_state(state, GraphPath())
    Path(source.handle, component).write_bytes(b"invalid authority")
    EvidenceValue.constructions = EvidenceValue.restores = 0
    with pytest.raises(RepoLoadError, match="local state"):
        build_exact_state_load_plan(Repo(store), state)
    assert EvidenceValue.constructions == EvidenceValue.restores == 0


def test_fallback_checks_record_target_before_custom_metadata_reader(tmp_path, monkeypatch):
    """Compatibility readers retain record admission and metadata error ordering."""
    from dryml.core.materialization import build_exact_state_load_plan
    from dryml.core.store.records import StateRefRecord

    store = DirStore(tmp_path / "store", query_index="none")
    writer = Repo(store)
    state = writer.save_object(EvidenceValue(1, repo=writer))
    other = writer.save_object(EvidenceValue(2, repo=writer))
    metadata_calls = []

    def metadata(digest):
        metadata_calls.append(digest)
        raise AssertionError("incompatible record must precede metadata")

    monkeypatch.setattr(store, "read_state_ref_record", lambda digest: StateRefRecord(other))
    monkeypatch.setattr(store, "read_snapshot_metadata", metadata)
    with pytest.raises(RepoLoadError, match="incompatible authority"):
        build_exact_state_load_plan(Repo(store), state)
    # Enclosing catalogue selection may inspect real compatible projections;
    # neither the requested record nor the unrelated saved graph is admitted.
    assert not metadata_calls


def test_fallback_enclosing_catalogue_does_not_require_direct_record_reader(tmp_path, monkeypatch):
    """Custom catalogue projections keep their original metadata-only selection path."""
    from dryml.core.materialization import build_exact_state_load_plan

    store = DirStore(tmp_path / "store", query_index="none")
    writer = Repo(store)
    state = writer.save_object(
        EvidenceRoot(EvidenceValue(1, repo=writer), repo=writer), deep_capture=True,
    )
    child = state.at(next(iter(state.states)))
    monkeypatch.setattr(store, "read_state_ref_record", lambda digest: None)
    assert not store._can_reuse_exact_validation()
    plan = build_exact_state_load_plan(Repo(store), child)
    assert len(plan.actions) == 1


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


def test_exact_load_accepts_projected_authority_from_enclosing_closure(tmp_path):
    """A closure snapshot can restore an exact descendant projection directly."""

    store = DirStore(tmp_path / "store")
    writer = Repo(store)
    state = writer.save_object(
        EvidenceRoot(EvidenceValue(3, repo=writer), repo=writer),
        deep_capture=True,
    )
    child_state = state.at(next(iter(state.object.objects)))

    assert store.read_state_ref_record(child_state.digest()) is None
    loaded = Repo(store).load_state_ref(child_state, reuse_live="never")

    assert loaded.value == 3
    container = Repo(store).load_or_build(
        Definition(EvidenceRoot, child_state), cache="none",
    )
    assert container.child.value == 3


def test_exact_load_accepts_stateless_projection_from_enclosing_closure(tmp_path):
    """Enclosing authority can select a stateless branch around saved state."""

    store = DirStore(tmp_path / "store")
    writer = Repo(store)
    branch = StatelessEvidenceRoot(EvidenceValue(4, repo=writer), repo=writer)
    state = writer.save_object(
        EvidenceRoot(branch, repo=writer), deep_capture=True,
    )
    branch_state = state.at(GraphPath((Parameter("child"),)))

    assert branch_state.object_id is None
    assert store.read_state_ref_record(branch_state.digest()) is None
    loaded = Repo(store).load_state_ref(branch_state, reuse_live="never")

    assert loaded.value.value == 4


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
