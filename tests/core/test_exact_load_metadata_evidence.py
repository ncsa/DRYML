"""Fresh Node load authority, detached metadata, and bounded shared preparation.

Store-3 snapshot JSON/cut-promotion details are retired. These tests exercise
their surviving obligations through Store-4 framed records, full StateKeys,
public readers, operation-local preparation, and independently protected loads.
"""

import inspect
import json
import hashlib
import os
import sys
from collections import Counter
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pytest

import dryml.environments as envs
from dryml.core import (
    Definition, MetadataConflictError, Object, Repo, SaveAnnotations, SaveRouting,
    Selector, Serializable, load_state_ref,
)
from dryml.core.repo import RepoLoadError
from dryml.core.store.dir import DirStore
from dryml.core.store.store import StoreAuthorityError
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


class StatefulEvidenceRoot(EvidenceValue):
    """Stateful parent with distinct independently routed child payloads."""

    def __init__(self, children, value=100):
        self.children = children
        self.value = value


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
    """Observe owning work without replacing acceleration-guarded callables.

    This shared test helper retains counters used by other maintained consumer
    tests. Current Node boundary counters below are separate from these retired
    snapshot-format observations; a zero here makes no authority/reuse claim.
    """
    counts = Counter()
    previous = sys.getprofile()

    def observe(frame, event, arg):
        if event == "return" and frame.f_code.co_name == "_read_snapshot" and arg is not None:
            counts["complete_bundles"] += 1
        if event == "call":
            name = frame.f_code.co_name
            if name in {"validate_payload", "_validate_payload"}:
                counts["validate_payload"] += 1
            if name == "_read_file":
                kind = frame.f_locals["record_type"].__name__
                counts[kind] += 1
            if name == "_read_json":
                counts["json_reads"] += 1
            if name in {"canonical_json_load_bytes", "canonical_json_bytes", "deep_freeze_json",
                        "decode_record", "decode_snapshot_metadata", "_decode_framed",
                        "decode_cdef_graph", "encode_cdef_graph", "_canonical_labels", "_freeze_json_fact",
                        "_thaw_json_fact", "_content_hash", "_primitive_size",
                        "_context_matches", "_graph_and_labels", "_role_paths",
                        "_eligible_graph_size", "_check_graph_eligibility", "_graph_content_size", "_array_content",
                        "_freeze_graph_content", "_graph_content_matches", "_thaw_graph_content",
                        "_realize_prepared_graph", "_build_graph_nodes", "_topology_projection"}:
                counts[name] += 1
            if name == "_digest" and frame.f_globals.get("__name__") == "dryml.core.store.records":
                counts["store_record_id"] += 1
            if name == "_canonical_bytes" and frame.f_globals.get("__name__") == "dryml.core.reference_values":
                counts["reference_identity_encode"] += 1
            if name == "dumps" and frame.f_globals.get("__name__") == "json":
                counts["json_dumps"] += 1
            if name == "load" and frame.f_globals.get("__name__") == "dill._dill":
                counts["dill_load"] += 1
            if name == "_matches_definition" and frame.f_back.f_code.co_name == "_validate_snapshot_locals":
                counts["path_definition_checks"] += 1
        if event == "c_call" and frame.f_code.co_name == "_validate_local_state_dir" and arg is hashlib.sha256:
            counts["definition_byte_hashes"] += 1
        if event == "c_call" and frame.f_code.co_name == "_read_json" and arg is os.lstat:
            counts["json_shape_checks"] += 1
        if event == "c_call" and frame.f_code.co_name == "_read_snapshot" and arg is os.scandir:
            counts["snapshot_inventory_reads"] += 1
        if previous is not None:
            previous(frame, event, arg)

    sys.setprofile(observe)
    try:
        yield counts
    finally:
        sys.setprofile(previous)


def test_node_load_groups_identical_bytes_and_rechecks_each_destination(tmp_path, monkeypatch):
    """One full-key read/header, but independent compatible destination checks."""
    from dryml.core import LocalState
    from tests.core.test_state_ref_restore_into import RestoreIntoIndependentPair
    store = DirStore(tmp_path / "store", query_index="none")
    repo = Repo(store)
    first, second = EvidenceValue(1, repo=repo), EvidenceValue(1, repo=repo)
    saved = repo.save(RestoreIntoIndependentPair(first, second, repo=repo))
    assert saved.at(GraphPath((Parameter("first"),))).state.key == saved.at(GraphPath((Parameter("second"),))).state.key
    counts = Counter()
    original_read = store.read_states
    original_creation = store.read_creation
    original_validate = LocalState.validate_for

    @contextmanager
    def read(keys):
        counts["keys"] += len(keys)
        with original_read(keys) as sources:
            yield sources

    def creation(digest):
        counts["headers"] += 1
        return original_creation(digest)

    def validate(state, destination, **kwargs):
        counts["destinations"] += 1
        return original_validate(state, destination, **kwargs)

    monkeypatch.setattr(store, "read_states", read)
    monkeypatch.setattr(store, "read_creation", creation)
    monkeypatch.setattr(LocalState, "validate_for", validate)
    EvidenceValue.restores = 0
    loaded = Repo(store).load_state_ref(saved, reuse_live="never")
    assert loaded.first is not loaded.second
    assert loaded.first.object_id != loaded.second.object_id
    assert EvidenceValue.restores == 2
    assert counts == {"keys": 1, "headers": 1, "destinations": 2}


def test_node_restore_hooks_hold_native_key_leases_without_writer_fences(tmp_path, monkeypatch):
    from dryml.core.store.retention import _held, _key_address
    store = DirStore(tmp_path / "store", query_index="none")
    repo = Repo(store)
    obj = EvidenceValue(1, repo=repo)
    saved = repo.save(obj)
    original = EvidenceValue.restore_state_from_dir_imp
    calls = []
    writer_depth = [0]
    original_writer = store.writer_lock

    @contextmanager
    def writer(*args, **kwargs):
        writer_depth[0] += 1
        try:
            with original_writer(*args, **kwargs) as result:
                yield result
        finally:
            writer_depth[0] -= 1

    def restore(target, directory, *, codec):
        assert _key_address(store, saved.state.key) in dict(_held())
        assert writer_depth[0] == 0
        with store.state_leases((saved.state.key,), exclusive=True, blocking=False) as acquired:
            assert not acquired
        calls.append(target)
        return original(target, directory, codec=codec)

    monkeypatch.setattr(store, "writer_lock", writer)
    monkeypatch.setattr(EvidenceValue, "restore_state_from_dir_imp", restore)
    repo.restore(obj, saved)
    assert calls == [obj]
    assert not _held()


def test_node_load_plan_does_not_retain_availability_between_uses(tmp_path):
    from dryml.core.materialization import build_exact_state_load_plan, execute_exact_state_load_plan
    store = DirStore(tmp_path / "store", query_index="none")
    repo = Repo(store)
    saved = repo.save(EvidenceValue(1, repo=repo))
    plan = build_exact_state_load_plan(repo, saved)
    assert plan.state_ref is saved
    assert plan.actions[0].state == saved.state
    assert not hasattr(plan.actions[0], "payload")
    Path(store.state_data_path(saved.state.key), "value").unlink()
    EvidenceValue.constructions = EvidenceValue.restores = 0
    with pytest.raises(RepoLoadError, match="preflight is incomplete"):
        execute_exact_state_load_plan(repo, plan, reuse_live="never")
    assert EvidenceValue.constructions == EvidenceValue.restores == 0


def test_node_key_lookup_does_not_inspect_unselected_sibling_variants(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store", query_index="none")
    repo = Repo(store)
    obj = EvidenceValue(1, repo=repo)
    first = repo.save(obj)
    obj.value = 2
    second = repo.save(obj)
    assert first.state.key.cdef_digest == second.state.key.cdef_digest
    Path(store.state_data_path(second.state.key), "value").unlink()
    monkeypatch.setattr(store, "inspect_states", lambda *a, **k: pytest.fail("sibling catalogue scanned"))
    loaded = Repo(store).load(first)
    assert loaded.value == 1


@pytest.mark.usefixtures("fixed_snapshot_environment")
def test_node_exact_load_carries_lineage_to_a_new_store(tmp_path):
    source, target = (DirStore(tmp_path / name, query_index="none") for name in ("source", "target"))
    writer = Repo(source)
    saved = writer.save(EvidenceValue(3, repo=writer))
    expected = writer.get_snapshot_metadata(saved).lineages[GraphPath()].created_at
    assert expected is not None
    reader = Repo((source, target))
    loaded = reader.load_state_ref(saved, reuse_live="never")
    copied = reader.save(loaded, store=target)
    assert reader.get_snapshot_metadata(copied, store=target).lineages[GraphPath()].created_at == expected


@contextmanager
def _read_counts(monkeypatch, stores):
    """Count fresh owning boundaries, independently of pure graph preparation."""
    counts = Counter()
    with monkeypatch.context() as patch:
        for store in stores:
            read, creation, authenticate = store.read_states, store.read_creation, store._authenticate_state_data

            @contextmanager
            def states(keys, _read=read):
                keys = tuple(keys)
                counts["keys"] += len(keys)
                with _read(keys) as sources:
                    yield sources

            def header(digest, _creation=creation):
                counts["headers"] += 1
                return _creation(digest)

            def payload(manifest, data, _authenticate=authenticate):
                counts["payloads"] += 1
                return _authenticate(manifest, data)

            patch.setattr(store, "read_states", states)
            patch.setattr(store, "read_creation", header)
            patch.setattr(store, "_authenticate_state_data", payload)
        yield counts


def _saved(tmp_path):
    store = DirStore(tmp_path / "store", query_index="none")
    repo = Repo(store)
    return store, repo.save(EvidenceValue(1, repo=repo))


def test_snapshot_json_cold_warm_characterization(tmp_path, monkeypatch, record_testsuite_property):
    """Every independent current-format preflight reads full keys afresh."""
    from dryml.core.materialization import build_exact_state_load_plan
    store, saved = _saved(tmp_path)
    reader = Repo(store)
    for phase in ("cold", "warm", "cleared"):
        if phase == "cleared":
            reader.clear_cache(strong=False, weak=False)
        EvidenceValue.constructions = EvidenceValue.restores = 0
        with _read_counts(monkeypatch, (store,)) as counts:
            plan = build_exact_state_load_plan(reader, saved)
        assert plan.state_ref is saved and len(plan.actions) == 1
        assert counts == {"keys": 1, "headers": 1, "payloads": 1}
        assert EvidenceValue.constructions == EvidenceValue.restores == 0
        assert not reader._validation_evidence._pending
        record_testsuite_property(f"u13_{phase}_fresh_read_ledger", dict(counts))


def test_json_warm_nested_public_metadata_is_detached(tmp_path):
    """Fresh captured annotation returns cannot mutate stored or cached facts."""
    store = DirStore(tmp_path / "store", query_index="none")
    writer = Repo(store)
    state = writer.save(EvidenceValue(1, repo=writer), annotations=SaveAnnotations(
        state={"nested": [{"values": [1, 2]}]},
    ))
    reader = Repo(store)
    for _ in range(2):
        metadata = reader.get_snapshot_metadata(state)
        metadata.captured_state_annotations["nested"][0]["values"].append(3)
        assert reader.get_snapshot_metadata(state).captured_state_annotations["nested"] == [{"values": [1, 2]}]


@pytest.mark.parametrize("damage", ["changed", "missing", "malformed", "oversized", "shape"])
def test_json_warm_reads_still_reject_changed_authority(tmp_path, damage):
    """A previously valid plan never certifies a damaged current manifest."""
    from dryml.core.materialization import build_exact_state_load_plan
    store, state = _saved(tmp_path)
    reader = Repo(store)
    build_exact_state_load_plan(reader, state)
    path = Path(store.state_data_path(state.state.key)).parent / "manifest.record"
    if damage == "changed":
        path.write_bytes(path.read_bytes().replace(b'"version":1', b'"version":99'))
    elif damage == "missing":
        path.unlink()
    elif damage == "shape":
        path.unlink()
        path.mkdir()
    elif damage == "oversized":
        path.write_bytes(b'{"value":"' + b"x" * 2_000_000 + b'"}')
    else:
        path.write_bytes(b'{"duplicate":1,"duplicate":2}')
    EvidenceValue.constructions = EvidenceValue.restores = 0
    for _ in range(2):
        with pytest.raises(RepoLoadError, match="preflight is incomplete"):
            reader.load_state_ref(state, reuse_live="never")
        assert not reader._validation_evidence._pending
    assert EvidenceValue.constructions == EvidenceValue.restores == 0


@pytest.mark.parametrize("size", [1, 3, 7])
def test_exact_plan_work_budget(tmp_path, size, monkeypatch, record_testsuite_property):
    """K distinct physical keys have K reads, independent of retained plan reuse."""
    from dryml.core.materialization import build_exact_state_load_plan
    store = DirStore(tmp_path / "store", query_index="none")
    writer = Repo(store)
    values = [EvidenceValue(i, repo=writer) for i in range(size)]
    root = values[0] if size == 1 else StatelessSeedContainer(values, repo=writer)
    state = writer.save(root)
    reader = Repo(store)
    for phase in ("cold", "warm", "cleared"):
        if phase == "cleared":
            reader.clear_cache(strong=False, weak=False)
        with _read_counts(monkeypatch, (store,)) as counts:
            plan = build_exact_state_load_plan(reader, state)
        assert len(plan.actions) == size
        assert counts == {"keys": size, "headers": size, "payloads": size}
        assert all(not hasattr(action, "payload") for action in plan.actions)
        record_testsuite_property(f"u13_k{size}_{phase}", dict(counts))


@pytest.mark.parametrize("size", [1, 3, 7])
def test_routed_stateful_parent_cut_work_budget(tmp_path, monkeypatch, size, record_testsuite_property):
    """Explicit key source choices retain routing and one validation per key."""
    from dryml.core.materialization import build_exact_state_load_plan
    from dryml.core.utils.graph.path import Index
    parent = DirStore(tmp_path / "parent", query_index="none")
    children = [DirStore(tmp_path / f"child-{i}", query_index="none") for i in range(size)]
    writer = Repo([parent, *children], save_routing=SaveRouting((
        (Selector(StatefulEvidenceRoot), parent),
        *((Selector(Definition(EvidenceValue, i)), store) for i, store in enumerate(children)),
    ), graph_mode="per-object"))
    state = writer.save(StatefulEvidenceRoot([EvidenceValue(i, repo=writer) for i in range(size)], repo=writer))
    paths = [GraphPath((Parameter("children"), Index(i))) for i in range(size)]
    selections = {state.state.key: parent, **{state.at(path).state.key: store for path, store in zip(paths, children)}}
    reader = Repo([parent, *children])
    with _read_counts(monkeypatch, (parent, *children)) as counts:
        plan = build_exact_state_load_plan(reader, state, source_store=parent, source_stores=selections)
    assert len(plan.actions) == size + 1
    assert {action.state.key: action.store for action in plan.actions} == selections
    assert counts == {"keys": size + 1, "headers": size + 1, "payloads": size + 1}
    assert reader._topology_leases == 0
    record_testsuite_property(f"u13_routed_k{size}", dict(counts))


def test_failed_aggregate_preflight_drops_all_pending_json(tmp_path):
    """A failed full-key preflight leaves no retained validity or live hooks."""
    store, state = _saved(tmp_path)
    reader = Repo(store)
    data = Path(store.state_data_path(state.state.key), "value")
    original = data.read_bytes()
    data.write_bytes(b"bad")
    EvidenceValue.constructions = EvidenceValue.restores = 0
    for _ in range(2):
        with pytest.raises(RepoLoadError):
            reader.load_state_ref(state, reuse_live="never")
        assert not reader._validation_evidence._pending
    assert EvidenceValue.constructions == EvidenceValue.restores == 0
    data.write_bytes(original)
    assert reader.load_state_ref(state, reuse_live="never").value == 1


@pytest.mark.parametrize("component", ["manifest.record", "def.pkl", "data/value"])
def test_delegated_probe_still_poisoned_by_advertised_parent_corruption(tmp_path, component):
    """A valid routed child does not hide corrupt selected parent authority."""
    parent = DirStore(tmp_path / "parent", query_index="none")
    child_store = DirStore(tmp_path / "child", query_index="none")
    writer = Repo([parent, child_store], save_routing=SaveRouting((
        (Selector(StatefulEvidenceRoot), parent), (Selector(EvidenceValue), child_store),
    ), graph_mode="per-object"))
    state = writer.save(StatefulEvidenceRoot([EvidenceValue(1, repo=writer)], repo=writer))
    directory = Path(parent.state_data_path(state.state.key))
    path = directory.parent.parent / component if component == "def.pkl" else directory.parent / component
    path.write_bytes(b"corrupt parent authority")
    EvidenceValue.constructions = EvidenceValue.restores = 0
    with pytest.raises(RepoLoadError, match="preflight is incomplete"):
        Repo([parent, child_store]).load_state_ref(state, source_store=parent, reuse_live="never")
    assert EvidenceValue.constructions == EvidenceValue.restores == 0


def test_projected_eager_plan_checks_unrequested_corrupt_sibling(tmp_path):
    """A selected full-key projection is independent of unrelated sibling data."""
    from dryml.core.utils.graph.path import Index
    store = DirStore(tmp_path / "store", query_index="none")
    writer = Repo(store)
    state = writer.save(StatelessSeedContainer([EvidenceValue(1, repo=writer), EvidenceValue(2, repo=writer)], repo=writer))
    first, second = (state.at(GraphPath((Parameter("seeds"), Index(i)))) for i in range(2))
    Path(store.state_data_path(second.state.key), "value").write_bytes(b"bad")
    assert Repo(store).load_state_ref(first, reuse_live="never").value == 1
    EvidenceValue.constructions = EvidenceValue.restores = 0
    with pytest.raises(RepoLoadError, match="preflight is incomplete"):
        Repo(store).load_state_ref(state, reuse_live="never")
    assert EvidenceValue.constructions == EvidenceValue.restores == 0


def test_shared_payload_still_checks_every_definition_association(tmp_path):
    """Same bytes in distinct creation buckets never authorize a wrong destination."""
    from dryml.core import LocalState
    from dryml.core.materialization import build_exact_state_load_plan
    store = DirStore(tmp_path / "store", query_index="none")
    writer = Repo(store)
    first, second = EvidenceValue(1, repo=writer), EvidenceValue(2, repo=writer)
    second.value = first.value
    a, b = writer.save(first), writer.save(second)
    assert a.state.key.codec_hash == b.state.key.codec_hash
    assert a.state.key.cdef_digest != b.state.key.cdef_digest
    wrong = b.replace_bindings(states={GraphPath(): LocalState(a.state.key, a.state.creation_cdef)})
    EvidenceValue.constructions = EvidenceValue.restores = 0
    with pytest.raises((ValueError, RepoLoadError)):
        build_exact_state_load_plan(Repo(store), wrong)
    assert EvidenceValue.constructions == EvidenceValue.restores == 0


@pytest.mark.parametrize("sibling", ["manifest", "creation", "capture", "block"])
def test_exact_plan_requires_every_bundle_sibling(tmp_path, sibling):
    """Current manifest, creation, capture and graph-block evidence is mandatory."""
    from dryml.core.materialization import build_exact_state_load_plan
    from dryml.core.store.records import NodeRootRecord
    store, state = _saved(tmp_path)
    reader = Repo(store)
    plan = build_exact_state_load_plan(reader, state)
    _, digest = store._node_metadata_address(state, "capture")
    record = store._read_file(store._node_path("graphs", digest), NodeRootRecord)
    data = Path(store.state_data_path(state.state.key))
    path = {"manifest": data.parent / "manifest.record", "creation": data.parent.parent / "def.pkl",
            "capture": Path(store._node_metadata_path("capture", digest)),
            "block": Path(store._node_path("definitions/blocks", record.block_digest))}[sibling]
    path.unlink()
    EvidenceValue.constructions = EvidenceValue.restores = 0
    error = StoreAuthorityError if sibling == "capture" else RepoLoadError
    message = "lacks captured metadata" if sibling == "capture" else "preflight is incomplete"
    with pytest.raises(error, match=message):
        reader.load_state_ref(state, reuse_live="never")
    assert not hasattr(plan.actions[0], "payload") and data.is_dir()
    assert EvidenceValue.constructions == EvidenceValue.restores == 0


@pytest.mark.parametrize("reader_name", ["read_creation", "read_states", "read_node", "_read_file"])
@pytest.mark.parametrize("phase", ["cold", "warm", "cleared"])
def test_exact_reader_overrides_remain_authoritative(tmp_path, monkeypatch, reader_name, phase):
    """Supported current-format public readers cannot be bypassed by pure facts."""
    from dryml.core.materialization import build_exact_state_load_plan
    store, state = _saved(tmp_path)
    reader = Repo(store)
    if phase != "cold":
        build_exact_state_load_plan(reader, state)
    if phase == "cleared":
        reader.clear_cache(strong=False, weak=False)
    calls = []

    def reject(*args, **kwargs):
        calls.append(True)
        raise StoreAuthorityError("custom current reader rejects authority")

    monkeypatch.setattr(store, reader_name, reject)
    EvidenceValue.constructions = EvidenceValue.restores = 0
    error = StoreAuthorityError if reader_name in {"read_node", "_read_file"} else RepoLoadError
    message = "custom current reader" if error is StoreAuthorityError else "preflight is incomplete"
    with pytest.raises(error, match=message):
        reader.load_state_ref(state, reuse_live="never")
    assert calls and reader._topology_leases == 0
    assert EvidenceValue.constructions == EvidenceValue.restores == 0


def test_nested_independent_plan_does_not_inherit_outer_payload_proof(tmp_path, monkeypatch):
    """Node plans retain preparation but never borrowed readers or payload proof."""
    from dryml.core.materialization import build_exact_state_load_plan
    store, state = _saved(tmp_path)
    reader = Repo(store)
    with _read_counts(monkeypatch, (store,)) as counts:
        outer = build_exact_state_load_plan(reader, state)
        nested = build_exact_state_load_plan(reader, state)
    assert counts == {"keys": 2, "headers": 2, "payloads": 2}
    assert outer.actions == nested.actions
    with reader._read_cut() as cut:
        with pytest.raises(ValueError, match="fresh full-key readers"):
            build_exact_state_load_plan(reader, state, _read_cut=cut)
    assert reader._topology_leases == 0


def test_absent_snapshot_is_never_retained(tmp_path):
    """An absent full key becomes visible after publication without cache clearing."""
    source, state = _saved(tmp_path)
    target = DirStore(tmp_path / "target", query_index="none")
    reader = Repo(target)
    query = reader.query().local_states().state_key(state.state.key)
    assert not query.exists() and not query.exists()
    writer = Repo((source, target))
    loaded = writer.load_state_ref(state, reuse_live="never", source_store=source)
    writer.save(loaded, store=target)
    assert query.one().key == state.state.key


@pytest.mark.parametrize("reader_name", ["read_states", "read_creation"])
@pytest.mark.parametrize("error_type", [KeyError, StoreAuthorityError])
def test_custom_local_errors_keep_routing_contract(tmp_path, monkeypatch, reader_name, error_type):
    """Current reader exceptions reject selected authority; missing keys are omissions."""
    from dryml.core.materialization import build_exact_state_load_plan
    store, state = _saved(tmp_path)
    calls = []

    def reject(*args, **kwargs):
        calls.append(True)
        raise error_type("custom current reader failure")

    monkeypatch.setattr(store, reader_name, reject)
    EvidenceValue.constructions = EvidenceValue.restores = 0
    with pytest.raises(RepoLoadError, match="preflight is incomplete"):
        build_exact_state_load_plan(Repo(store), state, source_store=store)
    assert calls and EvidenceValue.constructions == EvidenceValue.restores == 0


def test_custom_lower_validator_keeps_supplied_manifest_and_none_return(tmp_path, monkeypatch):
    """The current payload seam honors a successful custom None-returning validator."""
    from dryml.core.materialization import build_exact_state_load_plan
    from dryml.core.store.records import StateVariantManifest
    store, state = _saved(tmp_path)
    original = store._authenticate_state_data
    calls = []

    def validate(manifest, data):
        assert isinstance(manifest, StateVariantManifest) and manifest.key == state.state.key
        original(manifest, data)
        calls.append(manifest.key)

    monkeypatch.setattr(store, "_authenticate_state_data", validate)
    reader = Repo(store)
    for _ in range(2):
        assert build_exact_state_load_plan(reader, state).actions[0].state.key == state.state.key
    assert calls == [state.state.key, state.state.key]


def _archive(tmp_path):
    """Extract current immutable state via an explicitly committed archive."""
    from dryml.core.store.zip import ZipStore
    source, state = _saved(tmp_path)
    archive = tmp_path / "store.zip"
    import zipfile
    with zipfile.ZipFile(archive, "w") as output:
        for path in Path(source.base_dir).rglob("*"):
            if path.is_file():
                output.write(path, path.relative_to(source.base_dir))
    return ZipStore.open_existing(archive), state


def test_reopened_archive_borrows_content_but_revalidates_fresh_view(tmp_path, monkeypatch):
    """Reopened archive loads recheck extracted bytes and invalidate on close."""
    from dryml.core.materialization import build_exact_state_load_plan
    store, state = _archive(tmp_path)
    reader = Repo(store)
    try:
        for _ in range(2):
            with _read_counts(monkeypatch, (store,)) as counts:
                assert reader.load_state_ref(state, reuse_live="never").value == 1
            assert counts == {"keys": 1, "headers": 1, "payloads": 1}
        plan = build_exact_state_load_plan(reader, state)
        store.close()
        with pytest.raises((RuntimeError, RepoLoadError)):
            reader.load_state_ref(state, reuse_live="never")
        assert not hasattr(plan.actions[0], "payload")
    finally:
        store.close()


@pytest.mark.parametrize("reader_name", ["read_creation", "_read_file"])
@pytest.mark.parametrize("phase", ["cold", "warm"])
def test_zip_legacy_base_reader_override(tmp_path, monkeypatch, reader_name, phase):
    """Archive current reader delegation preserves custom successful readers."""
    from dryml.core.materialization import build_exact_state_load_plan
    store, state = _archive(tmp_path)
    reader = Repo(store)
    try:
        if phase == "warm":
            build_exact_state_load_plan(reader, state)
        original = getattr(DirStore, reader_name)
        calls = []

        def delegate(self, *args, **kwargs):
            calls.append(True)
            return original(self, *args, **kwargs)

        monkeypatch.setattr(DirStore, reader_name, delegate)
        for _ in range(2):
            calls.clear()
            assert reader.load_state_ref(state, reuse_live="never").value == 1
            assert calls
    finally:
        store.close()


@pytest.mark.parametrize("reader_name", ["read_creation", "_read_file"])
@pytest.mark.parametrize("phase", ["cold", "warm"])
def test_zip_rejecting_base_reader_override(tmp_path, monkeypatch, reader_name, phase):
    """A warm archive never bypasses rejecting current base reader delegates."""
    from dryml.core.materialization import build_exact_state_load_plan
    store, state = _archive(tmp_path)
    reader = Repo(store)
    try:
        if phase == "warm":
            build_exact_state_load_plan(reader, state)
        calls = []

        def reject(self, *args, **kwargs):
            calls.append(True)
            raise StoreAuthorityError("base reader rejects authority")

        monkeypatch.setattr(DirStore, reader_name, reject)
        EvidenceValue.constructions = EvidenceValue.restores = 0
        error = StoreAuthorityError if reader_name == "_read_file" else RepoLoadError
        message = "base reader rejects" if error is StoreAuthorityError else "preflight is incomplete"
        with pytest.raises(error, match=message):
            reader.load_state_ref(state, reuse_live="never")
        assert calls and EvidenceValue.constructions == EvidenceValue.restores == 0
    finally:
        store.close()


def test_payload_validator_override_keeps_uncached_error(tmp_path, monkeypatch):
    """Current full-key payload authentication is called even after preflight."""
    from dryml.core.materialization import build_exact_state_load_plan
    store, state = _saved(tmp_path)
    reader = Repo(store)
    build_exact_state_load_plan(reader, state)
    calls = []

    def reject(*args, **kwargs):
        calls.append(True)
        raise StoreAuthorityError("overridden payload validator")

    monkeypatch.setattr(store, "_authenticate_state_data", reject)
    with pytest.raises(RepoLoadError, match="preflight is incomplete"):
        build_exact_state_load_plan(reader, state)
    assert calls


@pytest.mark.parametrize("component", ["manifest.record", "def.pkl"])
def test_corrupt_local_authority_rejects_before_hooks(tmp_path, component):
    """Fresh Node preflight rejects malformed manifest/creation before hooks."""
    store, state = _saved(tmp_path)
    data = Path(store.state_data_path(state.state.key))
    path = data.parent.parent / component if component == "def.pkl" else data.parent / component
    path.write_bytes(b"invalid authority")
    EvidenceValue.constructions = EvidenceValue.restores = 0
    with pytest.raises(RepoLoadError, match="preflight is incomplete"):
        Repo(store).load_state_ref(state, reuse_live="never")
    assert EvidenceValue.constructions == EvidenceValue.restores == 0


def test_fallback_checks_record_target_before_custom_metadata_reader(tmp_path, monkeypatch):
    """A custom root reader's incompatible result precedes capture acceptance."""
    from dryml.core.materialization import build_exact_state_load_plan
    store, state = _saved(tmp_path)
    other = Definition(EvidenceValue, 2).concretize()
    monkeypatch.setattr(store, "read_node", lambda *a, **k: other)
    calls = []
    monkeypatch.setattr(store, "_read_node_capture", lambda *a: calls.append(True))
    with pytest.raises(StoreAuthorityError, match="incompatible exact authority"):
        build_exact_state_load_plan(Repo(store), state)
    assert not calls


def test_fallback_enclosing_catalogue_does_not_require_direct_record_reader(tmp_path, monkeypatch):
    """Embedded current state uses full-key authority without legacy catalogues."""
    from dryml.core.materialization import build_exact_state_load_plan
    store = DirStore(tmp_path / "store", query_index="none")
    writer = Repo(store)
    state = writer.save(EvidenceRoot(EvidenceValue(1, repo=writer), repo=writer))
    child = state.at(GraphPath((Parameter("child"),)))
    monkeypatch.setattr("dryml.core.query.authority.capture_store_projection", lambda *a, **kw: pytest.fail("load inventoried authority"))
    plan = build_exact_state_load_plan(Repo(store), child)
    assert len(plan.actions) == 1 and plan.actions[0].state.key == child.state.key


@pytest.mark.parametrize("phase", ["cold", "warm", "cleared"])
def test_exact_records_decoder_override_remains_authoritative(tmp_path, monkeypatch, phase):
    """Independent current cuts invoke the current Node decoder, even after warming."""
    import dryml.core.node_codec as codec
    from dryml.core.materialization import build_exact_state_load_plan
    store, state = _saved(tmp_path)
    reader = Repo(store)
    if phase != "cold":
        build_exact_state_load_plan(reader, state)
    if phase == "cleared":
        reader.clear_cache(strong=False, weak=False)
    calls = []

    def reject(*args, **kwargs):
        calls.append(True)
        raise StoreAuthorityError("current Node decoder rejects authority")

    monkeypatch.setattr(codec, "_decode_node_graph_views", reject)
    EvidenceValue.constructions = EvidenceValue.restores = 0
    for _ in range(2):
        with pytest.raises(StoreAuthorityError, match="current Node decoder"):
            build_exact_state_load_plan(reader, state)
    assert len(calls) == 2 and reader._topology_leases == 0
    assert EvidenceValue.constructions == EvidenceValue.restores == 0


@pytest.mark.parametrize("callable_object", [False, True])
def test_exact_records_decoder_delegate_keeps_uncached_policy(tmp_path, monkeypatch, callable_object):
    """Successful custom decoder functions/objects run in every fresh operation."""
    import dryml.core.node_codec as codec
    from dryml.core.materialization import build_exact_state_load_plan
    store, state = _saved(tmp_path)
    reader = Repo(store)
    original, calls = codec._decode_node_graph_views, []

    def delegate(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    class Decoder:
        __slots__ = ()

        def __call__(self, *args, **kwargs):
            return delegate(*args, **kwargs)

    for phase in ("cold", "warm", "cleared"):
        if phase == "cleared":
            reader.clear_cache(strong=False, weak=False)
        monkeypatch.setattr(codec, "_decode_node_graph_views", Decoder() if callable_object else delegate)
        calls.clear()
        plan = build_exact_state_load_plan(reader, state)
        assert calls and plan.state_ref is state


@pytest.mark.parametrize("budget", [0, 1024 * 1024])
def test_node_preparation_budget_never_rejects_valid_discovery(tmp_path, monkeypatch, budget):
    """Pure cache admission is optional; zero bytes cannot reject valid discovery."""
    from tests.core.test_local_state_discovery import saved_values
    store, reader, saved = saved_values(tmp_path)
    reader._validation_evidence.max_entry_bytes = budget
    monkeypatch.setattr(store, "_authenticate_state_data", lambda *a: pytest.fail("query read payload"))
    result = reader.query().local_states(origin=saved[0].state.creation_cdef).page_size(1).refresh(False).take(10)
    assert len(result) == 10
    assert result.diagnostic().cdef_blobs_decoded == (11 if budget == 0 else 1)


def test_caught_bundle_failure_still_poisoned_for_retention(tmp_path, monkeypatch):
    """A failed preparation cannot seed reusable validity for a later operation."""
    import dryml.core.node_codec as codec
    from dryml.core.materialization import build_exact_state_load_plan
    store, state = _saved(tmp_path)
    reader = Repo(store)
    with monkeypatch.context() as patch:
        patch.setattr(codec, "_decode_node_graph_views", lambda *a, **k: (_ for _ in ()).throw(StoreAuthorityError("bad block")))
        with pytest.raises(StoreAuthorityError, match="bad block"):
            build_exact_state_load_plan(reader, state)
    assert not reader._validation_evidence._pending
    assert reader.load_state_ref(state, reuse_live="never").value == 1


@pytest.mark.parametrize("defer", [False, True])
def test_routed_deferred_child_does_not_authorize_eager_promotion(tmp_path, defer):
    """Current Node preflight never borrows retired deferred snapshot validity."""
    from dryml.core.materialization import build_exact_state_load_plan
    store, state = _saved(tmp_path)
    Path(store.state_data_path(state.state.key), "value").write_bytes(b"bad")
    EvidenceValue.constructions = EvidenceValue.restores = 0
    with pytest.raises(RepoLoadError, match="preflight is incomplete"):
        build_exact_state_load_plan(Repo(store), state, _defer_payload=defer)
    assert EvidenceValue.constructions == EvidenceValue.restores == 0


def test_exact_plan_retains_order_across_default_change_and_alias_views(tmp_path):
    """Explicit current source selection survives a later default-order change."""
    from dryml.core.materialization import build_exact_state_load_plan, execute_exact_state_load_plan
    first, state = _saved(tmp_path)
    alias = DirStore(tmp_path / "replica", query_index="none")
    source = Repo(first)
    Repo(alias).save(source.load_state_ref(state, reuse_live="never"))
    reader = Repo([first, alias])
    plan = build_exact_state_load_plan(reader, state, source_store=first)
    reader.set_default_store(alias)
    assert reader.stores[0] is alias and plan.actions[0].store is first
    assert build_exact_state_load_plan(reader, state, source_store=alias).actions[0].store is alias
    assert execute_exact_state_load_plan(reader, plan, reuse_live="never").value == 1
    same_domain = DirStore.open_existing(first.base_dir, query_index="none")
    duplicate = Repo([first, same_domain])
    for selected in (first, same_domain):
        duplicate_plan = build_exact_state_load_plan(duplicate, state, source_store=selected)
        assert duplicate_plan.actions[0].store is selected
        assert execute_exact_state_load_plan(duplicate, duplicate_plan, reuse_live="never").value == 1


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
    writer = Repo([root_store, child_store], save_routing=SaveRouting(
        ((Selector(EvidenceRoot), root_store), (Selector(EvidenceValue), child_store))))
    state = writer.save_object(EvidenceRoot(EvidenceValue(3, repo=writer), repo=writer), deep_capture=True)
    child_state = state.at(GraphPath((Parameter("child"),)))
    reader = Repo([root_store, child_store])
    loaded = reader.load_state_ref(state, reuse_live="never", source_store=root_store,
                                  source_stores={child_state.state.key: child_store})
    assert loaded.child.value == 3


def test_exact_load_accepts_projected_authority_from_enclosing_closure(tmp_path):
    """A closure snapshot can restore an exact descendant projection directly."""
    store = DirStore(tmp_path / "store")
    writer = Repo(store)
    state = writer.save_object(EvidenceRoot(EvidenceValue(3, repo=writer), repo=writer), deep_capture=True)
    child_state = state.at(GraphPath((Parameter("child"),)))
    _, digest = store._node_metadata_address(child_state, "capture")
    assert store.read_node(digest, kind="saved") is None
    loaded = Repo(store).load_state_ref(child_state, reuse_live="never")
    assert loaded.value == 3
    container = Repo(store).load_or_build(Definition(EvidenceRoot, child_state), cache="none")
    assert container.child.value == 3


def test_exact_load_accepts_stateless_projection_from_enclosing_closure(tmp_path):
    """Enclosing authority can select a stateless branch around saved state."""
    store = DirStore(tmp_path / "store")
    writer = Repo(store)
    branch = StatelessEvidenceRoot(EvidenceValue(4, repo=writer), repo=writer)
    state = writer.save_object(EvidenceRoot(branch, repo=writer), deep_capture=True)
    branch_state = state.at(GraphPath((Parameter("child"),)))
    assert branch_state.object_id is None
    _, digest = store._node_metadata_address(branch_state, "capture")
    assert store.read_node(digest, kind="saved") is None
    loaded = Repo(store).load_state_ref(branch_state, reuse_live="never")
    assert loaded.value.value == 4


def _shift_node_lineage(monkeypatch, store, target):
    from dryml.core.store.records import NodeMetadataRecord, _json_bytes
    original = store._read_file

    def read(path, record_type):
        record = original(path, record_type)
        if isinstance(record, NodeMetadataRecord) and record.scope in {"lineage", "capture"}:
            data = json.loads(record.content)
            rows = data["lineages"] if record.scope == "capture" else [data]
            changed = False
            for row in rows:
                if row["identity"] == target.object_id.to_data():
                    row["created_at"] += 1
                    changed = True
            if changed:
                return replace(record, content=_json_bytes(data))
        return record

    monkeypatch.setattr(store, "_read_file", read)


def test_exact_load_rejects_conflicting_selected_child_lineage_before_hooks(tmp_path, monkeypatch):
    """A selected routed child cannot disagree with the enclosing lineage cut."""
    root_store = DirStore(tmp_path / "root")
    child_store = DirStore(tmp_path / "child")
    writer = Repo([root_store, child_store], save_routing=SaveRouting(
        ((Selector(EvidenceRoot), root_store), (Selector(EvidenceValue), child_store))))
    state = writer.save_object(EvidenceRoot(EvidenceValue(3, repo=writer), repo=writer), deep_capture=True)
    child_state = state.at(GraphPath((Parameter("child"),)))
    _shift_node_lineage(monkeypatch, child_store, child_state)
    EvidenceValue.constructions = EvidenceValue.restores = 0
    with pytest.raises(MetadataConflictError, match="lineage evidence"):
        Repo([root_store, child_store]).load_state_ref(state, reuse_live="never", source_store=root_store,
                                                    source_stores={child_state.state.key: child_store})
    assert EvidenceValue.constructions == EvidenceValue.restores == 0


def test_exact_load_requires_explicit_choice_for_conflicting_child_replicas(tmp_path):
    """A routed child replica conflict is rejected until its source is selected."""
    root_store = DirStore(tmp_path / "root")
    child_store = DirStore(tmp_path / "child")
    replica_store = DirStore(tmp_path / "replica")
    writer = Repo([root_store, child_store], save_routing=SaveRouting(
        ((Selector(EvidenceRoot), root_store), (Selector(EvidenceValue), child_store))))
    state = writer.save_object(EvidenceRoot(EvidenceValue(3, repo=writer), repo=writer), deep_capture=True)
    child_state = state.at(GraphPath((Parameter("child"),)))
    replica = Repo(child_store).load_state_ref(child_state, reuse_live="never")
    Repo(replica_store).save_object(replica, store=replica_store)
    reader = Repo([root_store, child_store, replica_store])
    EvidenceValue.constructions = EvidenceValue.restores = 0
    with pytest.raises(MetadataConflictError, match="snapshot evidence"):
        reader.load_state_ref(state, reuse_live="never")
    assert EvidenceValue.constructions == EvidenceValue.restores == 0
    assert reader.load_state_ref(state, reuse_live="never", source_store=root_store,
                                 source_stores={child_state.state.key: child_store}).child.value == 3


def test_exact_load_rejects_corrupt_selected_root_payload_without_replica_fallback(tmp_path):
    """A selected root's advertised local payload is not treated as routed absence."""
    first = DirStore(tmp_path / "first")
    second = DirStore(tmp_path / "second")
    source = Repo(first)
    child = EvidenceValue(3, repo=source)
    child_state = source.save_object(child)
    state = source.save_object(EvidenceRoot(child, repo=source), deep_capture=True)
    assert state.at(GraphPath((Parameter("child"),))).graph_equal(child_state)
    Repo(second).save_object(source.load_state_ref(child_state, reuse_live="never"), store=second)
    Path(first.state_data_path(child_state.state.key), "value").unlink()
    EvidenceValue.constructions = EvidenceValue.restores = 0
    with pytest.raises(RepoLoadError, match="preflight is incomplete"):
        Repo([second, first]).load_state_ref(state, reuse_live="never", source_store=first)
    assert EvidenceValue.constructions == EvidenceValue.restores == 0


def test_exact_load_keeps_unknown_lineages_per_stateless_reference(tmp_path):
    """Independent stateless roots do not collide in the ObjectId lineage index."""
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    first = repo.save_object(StatelessEvidenceRoot("first", repo=repo))
    second = repo.save_object(StatelessEvidenceRoot("second", repo=repo))
    state = repo.save_object(StatelessSeedContainer([first, second], repo=repo))
    from dryml.core.materialization import build_exact_state_load_plan
    plan = build_exact_state_load_plan(Repo(store), state)
    assert plan.actions == ()
    assert not hasattr(plan, "lineage_facts_by_object_id")
    assert not hasattr(plan, "lineage_facts_by_reference")
    for saved in (state, first, second):
        fact = repo.get_snapshot_metadata(saved).lineages[GraphPath()]
        assert fact.object_ref.object_id is None and fact.creation_status == "unknown"
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
    EvidenceValue.constructions = EvidenceValue.restores = 0
    with pytest.raises(ValueError, match="connected"):
        reader.load_state_ref(state, source_store=detached_store)
    with pytest.raises(RepoLoadError, match="preflight is incomplete"):
        reader.load_state_ref(state, source_store=empty_store)
    with pytest.raises(ValueError, match="outside the selected state closure"):
        reader.load_state_ref(state, source_stores={unrelated.state.key: source_store})
    assert EvidenceValue.constructions == EvidenceValue.restores == 0


def test_exact_load_rejects_selected_lineage_conflict_with_live_cache(tmp_path, monkeypatch):
    """Selected facts cannot retimestamp a cached exact identity before restore."""
    first = DirStore(tmp_path / "first")
    second = DirStore(tmp_path / "second")
    source = Repo(first)
    state = source.save_object(EvidenceValue(1, repo=source))
    Repo(second).save_object(source.load_state_ref(state), store=second)
    reader = Repo([first, second])
    cached = reader.load_state_ref(state, reuse_live="never", source_store=first)
    _shift_node_lineage(monkeypatch, second, state)
    EvidenceValue.constructions = EvidenceValue.restores = 0
    with pytest.raises(MetadataConflictError, match="Live cached identity"):
        reader.load_state_ref(state, reuse_live="greedy", source_store=second)
    assert EvidenceValue.constructions == EvidenceValue.restores == 0
    assert cached.value == 1


def test_targeted_restore_rejects_selected_lineage_conflict_for_uncached_target(tmp_path, monkeypatch):
    """Targeted restore cannot retimestamp an uncached supplied live target."""
    first = DirStore(tmp_path / "first")
    second = DirStore(tmp_path / "second")
    source = Repo(first)
    state = source.save_object(EvidenceValue(1, repo=source))
    Repo(second).save_object(source.load_state_ref(state, reuse_live="never"), store=second)
    reader = Repo([first, second])
    target = reader.load_state_ref(state, reuse_live="never", cache="none", source_store=first)
    _shift_node_lineage(monkeypatch, second, state)
    EvidenceValue.restores = 0
    with pytest.raises(MetadataConflictError, match="Target live identity"):
        reader.restore_state_ref_into(target, state, source_store=second)
    assert EvidenceValue.restores == 0


def test_all_exact_load_entry_points_expose_and_forward_evidence_selectors(monkeypatch):
    """Public exact-load wrappers pass selector values through to Repo unchanged."""
    for callable_ in (Repo.load_state_ref, Repo.restore_state_ref_into, load_state_ref):
        parameters = inspect.signature(callable_).parameters
        assert "source_store" in parameters
        assert "source_stores" in parameters
    captured = {}
    repo = Repo()
    state_ref, source_store = object(), object()
    source_stores = {object(): object()}
    monkeypatch.setattr(Repo, "load_state_ref", lambda self, state, **kwargs: captured.update(state=state, **kwargs) or state)
    assert load_state_ref(state_ref, repo=repo, source_store=source_store, source_stores=source_stores) is state_ref
    assert captured == {"state": state_ref, "reuse_live": "matching", "cache": "weak",
                        "source_store": source_store, "source_stores": source_stores}
