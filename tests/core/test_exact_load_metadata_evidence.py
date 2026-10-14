"""Exact-load selection preserves one captured metadata authority per snapshot."""

import inspect
import hashlib
import os
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
    """Observe owning work without replacing acceleration-guarded callables."""
    counts = Counter()
    previous = sys.getprofile()

    def observe(frame, event, arg):
        if event == "return" and frame.f_code.co_name == "_read_snapshot" and arg is not None:
            counts["complete_bundles"] += 1
        if event == "call":
            name = frame.f_code.co_name
            if name in {"_read_snapshot", "_validate_local_state_dir", "validate_payload"}:
                counts[name] += 1
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


def test_snapshot_json_cold_warm_characterization(tmp_path, monkeypatch, record_testsuite_property):
    """Complete independent preflights reread bytes and isolate mutable returns."""
    from dryml.core.materialization import build_exact_state_load_plan

    store = DirStore(tmp_path / "store", query_index="none")
    writer = Repo(store)
    state = writer.save_object(EvidenceValue(1, repo=writer))
    reader = Repo(store)
    assert store._can_reuse_exact_validation()
    import builtins
    original_open = builtins.open
    sibling_reads = []

    def observe_open(path, *args, **kwargs):
        if Path(path).name in {"placement.json", "metadata.json", "snapshot.json"}:
            sibling_reads.append(Path(path).name)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", observe_open)
    ledgers = []
    for phase in ("cold", "warm", "cleared"):
        if phase == "cleared":
            reader.clear_cache(strong=False, weak=False)
            assert reader._validation_evidence.accounted_bytes == 0
        sibling_reads.clear()
        with exact_work_counts() as counts:
            plan = build_exact_state_load_plan(reader, state)
        ledgers.append(counts)
        assert counts["json_reads"] == 3
        assert counts["json_shape_checks"] == 3
        assert counts["snapshot_inventory_reads"] == 1
        assert sorted(sibling_reads) == ["metadata.json", "placement.json", "snapshot.json"]
        assert counts["canonical_json_load_bytes"] == (0 if phase == "warm" else 3)
        assert counts["decode_snapshot_metadata"] == 1
        assert counts["decode_record"] >= 3
        assert counts["canonical_json_bytes"] > 0
        assert counts["deep_freeze_json"] > 0
        assert counts["validate_payload"] == 1
        assert counts["StateRefRecord"] == 1
        assert counts["DefinitionRecord"] == 2
        assert counts["dill_load"] == 8
        assert counts["decode_cdef_graph"] == (0 if phase == "warm" else 3)
        assert counts["encode_cdef_graph"] == (0 if phase == "warm" else 3)
        assert counts["store_record_id"] > 0 and counts["reference_identity_encode"] > 0
        assert counts["_graph_and_labels"] > 0
        assert counts["_role_paths"] > 0
        assert counts["_build_graph_nodes"] == 3
        assert counts["_realize_prepared_graph"] == (3 if phase == "warm" else 0)
        assert counts["_check_graph_eligibility"] == 3
        assert (counts["_graph_content_size"] == 0) == (phase == "warm")
        if phase != "warm":
            assert counts["_graph_content_matches"] == 0
        else:
            assert counts["_graph_content_matches"] > 0
        assert plan.state_ref == state
    for key in ("json_reads", "decode_snapshot_metadata", "decode_record",
                "canonical_json_bytes", "validate_payload",
                "StateRefRecord", "DefinitionRecord", "dill_load",
                "_build_graph_nodes", "_check_graph_eligibility"):
        assert ledgers[0][key] == ledgers[1][key]
    assert ledgers[0]["deep_freeze_json"] == ledgers[1]["deep_freeze_json"] + 3
    assert ledgers[1]["_graph_and_labels"] < ledgers[0]["_graph_and_labels"]
    assert ledgers[1]["_role_paths"] < ledgers[0]["_role_paths"]
    assert ledgers[1]["_graph_content_matches"] > 0
    assert ledgers[0] == ledgers[2]
    for phase, ledger in zip(("cold", "warm", "cleared"), ledgers):
        record_testsuite_property(f"u3_{phase}_ledger", repr(dict(sorted(ledger.items()))))
    assert len(reader._validation_evidence._resident) == 6
    assert not reader._validation_evidence._pending
    with reader._read_cut() as cut:
        first = cut.snapshot(store, state.digest())
        first[2].clear()
    with reader._read_cut() as cut:
        second = cut.snapshot(store, state.digest())
        assert len(second[2]) == 1
        assert second[0].state_ref == state


def test_json_warm_nested_public_metadata_is_detached(tmp_path):
    """Lists/maps in decoded annotation values never expose retained primitives."""
    from dryml.core import SaveAnnotations
    from dryml.core.materialization import build_exact_state_load_plan

    store = DirStore(tmp_path / "store", query_index="none")
    writer = Repo(store)
    state = writer.save_object(EvidenceValue(1, repo=writer), annotations=SaveAnnotations(
        state={"nested": [{"values": [1, 2]}]},
    ))
    reader = Repo(store)
    build_exact_state_load_plan(reader, state)
    with reader._read_cut() as cut:
        metadata = cut.metadata(store, state.digest())
        values = metadata.captured_state_annotations
        values["nested"][0]["values"].append(3)
    with exact_work_counts() as counts:
        with reader._read_cut() as cut:
            fresh = cut.metadata(store, state.digest())
    assert counts["canonical_json_load_bytes"] == 0
    assert fresh.captured_state_annotations["nested"] == [{"values": [1, 2]}]


@pytest.mark.parametrize("damage", ["changed", "missing", "malformed", "oversized", "shape"])
def test_json_warm_reads_still_reject_changed_authority(tmp_path, damage):
    """Hits cannot replace fresh bytes, canonical parser bounds or file shape."""
    from dryml.core.materialization import build_exact_state_load_plan

    store = DirStore(tmp_path / "store", query_index="none")
    writer = Repo(store)
    state = writer.save_object(EvidenceValue(1, repo=writer))
    reader = Repo(store)
    build_exact_state_load_plan(reader, state)
    path = Path(store._snapshot_path(state.digest()), "placement.json")
    if damage == "changed":
        path.write_bytes(path.read_bytes().replace(b'"version":1', b'"version":2'))
    elif damage == "missing":
        path.unlink()
    elif damage == "shape":
        path.unlink()
        path.mkdir()
    elif damage == "oversized":
        path.write_bytes(b'{"value":"' + b"x" * 4097 + b'"}')
    else:
        path.write_bytes(b'{"duplicate":1,"duplicate":2}')
    facility = reader._validation_evidence
    residents = tuple(facility._resident.values())
    for _ in range(2):
        with pytest.raises(RepoLoadError, match="preflight is incomplete"):
            build_exact_state_load_plan(reader, state)
        assert {id(fact) for fact in facility._resident.values()} == {id(fact) for fact in residents}
        assert not facility._pending


def test_json_content_context_change_misses_without_reader_override(tmp_path, monkeypatch):
    """Changing the existing parser implementation causes fresh eligible parsing."""
    import dryml.formats as formats
    from dryml.core.materialization import build_exact_state_load_plan

    store = DirStore(tmp_path / "store", query_index="none")
    writer = Repo(store)
    state = writer.save_object(EvidenceValue(1, repo=writer))
    reader = Repo(store)
    build_exact_state_load_plan(reader, state)
    original = formats.canonical_json_load_bytes
    calls = []

    def parser(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(formats, "canonical_json_load_bytes", parser)
    assert store._can_reuse_exact_validation()
    build_exact_state_load_plan(reader, state)
    assert len(calls) == 3
    build_exact_state_load_plan(reader, state)
    assert len(calls) == 3


def test_unknown_parser_identity_uses_uncached_existing_parser(tmp_path, monkeypatch):
    """An unretainable callable context is not a reason to reject valid input."""
    import dryml.formats as formats
    from dryml.core.materialization import build_exact_state_load_plan
    store = DirStore(tmp_path / "store", query_index="none")
    writer = Repo(store)
    state = writer.save_object(EvidenceValue(1, repo=writer))
    original = formats.canonical_json_load_bytes
    calls = []
    class Parser:
        __slots__ = ()
        def __call__(self, *args, **kwargs):
            calls.append(True)
            return original(*args, **kwargs)
    monkeypatch.setattr(formats, "canonical_json_load_bytes", Parser())
    reader = Repo(store)
    assert store._can_reuse_exact_validation()
    for _ in range(2):
        build_exact_state_load_plan(reader, state)
    assert len(calls) == 6
    assert all(fact.context[0] != "snapshot-json" for fact in reader._validation_evidence._resident.values())


def test_reopened_archive_borrows_content_but_revalidates_fresh_view(tmp_path):
    """Durable reopen avoids JSON parsing, never fresh frame/payload authority."""
    from dryml.core.materialization import build_exact_state_load_plan
    from dryml.core.store.zip import ZipStore
    from dryml.managed.storage import _validate_reopened_managed_state

    store = ZipStore(tmp_path / "store.zip")
    writer = Repo(store)
    try:
        state, report = writer.save_object(EvidenceValue(3, repo=writer), report_stores=True)
        store.commit()
        reader = Repo(store)
        assert store._can_reuse_exact_validation()
        with exact_work_counts() as cold:
            build_exact_state_load_plan(reader, state)
        facility = reader._validation_evidence
        with exact_work_counts() as warm:
            _validate_reopened_managed_state(reader, state, report, (store,))
        # Known Zip liveness wrappers and their Dir implementation both count.
        assert cold["json_reads"] == warm["json_reads"] == 6
        assert cold["canonical_json_load_bytes"] == 3
        assert warm["canonical_json_load_bytes"] == 0
        assert warm["validate_payload"] == 1
        assert warm["StateRefRecord"] > 0
        assert warm["dill_load"] > 0
        assert warm["_realize_prepared_graph"] == 3
        reader.clear_cache(strong=False, weak=False)
        assert facility.accounted_bytes == 0
        with exact_work_counts() as cleared:
            _validate_reopened_managed_state(reader, state, report, (store,))
        assert cleared["canonical_json_load_bytes"] == 3
        assert cleared["decode_cdef_graph"] == 3
        assert cleared["validate_payload"] == 1
        assert cleared["dill_load"] == warm["dill_load"]
        assert len(facility._resident) == 6 and not facility._pending
        assert facility.accounted_bytes == sum(fact.charge for fact in facility._resident.values())
        assert facility.high_water_bytes <= facility.max_bytes
        assert reader._topology_leases == 0
        assert all(store.base_dir.encode() not in fact.source for fact in facility._resident.values())
        reader.close(flush=False)
        assert not facility._resident
    finally:
        writer.close(flush=False)
        store.close()


def test_complete_snapshot_reads_under_forced_content_hash_collisions(tmp_path, monkeypatch):
    """Full owning reads distinguish component kind and bytes within four slots."""
    import dryml.core.repo as repo_module
    from dryml.core.materialization import build_exact_state_load_plan

    monkeypatch.setattr(repo_module, "_content_hash", lambda source: b"collision")
    store = DirStore(tmp_path / "store", query_index="none")
    writer = Repo(store)
    first = writer.save_object(EvidenceValue(1, repo=writer))
    second = writer.save_object(EvidenceValue(2, repo=writer))
    reader = Repo(store)
    build_exact_state_load_plan(reader, first)
    build_exact_state_load_plan(reader, second)
    assert len(reader._validation_evidence._resident) == 4
    with exact_work_counts() as counts:
        plan = build_exact_state_load_plan(reader, second)
    # The first four pending components fill this bucket on a cold cut. Later
    # graph candidates use fallback rather than exceeding the shared limit.
    assert counts["canonical_json_load_bytes"] == 0
    assert counts["json_reads"] == 3 and counts["validate_payload"] == 1
    assert plan.state_ref == second and plan.state_ref != first
    with exact_work_counts() as counts:
        plan = build_exact_state_load_plan(reader, first)
    assert counts["canonical_json_load_bytes"] > 0
    assert plan.state_ref == first
    assert reader._validation_evidence.high_water_entries <= 4


def test_caught_bundle_failure_still_poisoned_for_retention(tmp_path):
    """Catching an internal read error cannot admit its earlier parsed siblings."""
    from dryml.core.store.store import StoreAuthorityError
    store = DirStore(tmp_path / "store", query_index="none")
    writer = Repo(store)
    state = writer.save_object(EvidenceValue(1, repo=writer))
    Path(store._snapshot_path(state.digest()), "snapshot.json").write_bytes(b"malformed")
    reader = Repo(store)
    with reader._read_cut() as cut:
        with pytest.raises(StoreAuthorityError):
            cut.snapshot(store, state.digest())
        assert not cut.snapshots and not cut._pending
    assert not reader._validation_evidence._resident
    assert reader._validation_evidence.accounted_bytes == 0


def test_failed_aggregate_preflight_drops_all_pending_json(tmp_path):
    """Successful bundle parsing is not admitted before late payload validation."""
    from dryml.core.materialization import build_exact_state_load_plan

    store = DirStore(tmp_path / "store", query_index="none")
    writer = Repo(store)
    state = writer.save_object(EvidenceValue(1, repo=writer))
    Path(store.open_local_state(state, GraphPath()).handle, "data", "value").write_text("bad")
    reader = Repo(store)
    for _ in range(2):
        with exact_work_counts() as counts:
            with pytest.raises(RepoLoadError):
                build_exact_state_load_plan(reader, state)
        assert counts["canonical_json_load_bytes"] == 3
        assert not reader._validation_evidence._resident
        assert not reader._validation_evidence._pending
        assert reader._validation_evidence.accounted_bytes == 0


def test_failed_live_lineage_preflight_drops_pending_json(tmp_path):
    """Late cached-identity checks precede content admission even after warming."""
    from dryml.core.materialization import build_exact_state_load_plan

    store = DirStore(tmp_path / "store", query_index="none")
    writer = Repo(store)
    state = writer.save_object(EvidenceValue(1, repo=writer))
    reader = Repo(store)
    cached = reader.load_state_ref(state, reuse_live="never")
    reader.clear_cache(strong=False, weak=False)
    cached._lineage_created_at += timedelta(seconds=1)
    with exact_work_counts() as counts:
        with pytest.raises(MetadataConflictError, match="Live cached identity"):
            build_exact_state_load_plan(reader, state)
    assert counts["canonical_json_load_bytes"] == 3
    assert not reader._validation_evidence._resident
    assert not reader._validation_evidence._pending


def test_absent_snapshot_is_never_retained(tmp_path):
    """An absence can become successful authority without clearing a Repo."""
    store = DirStore(tmp_path / "store", query_index="none")
    reader = Repo(store)
    for _ in range(2):
        with reader._read_cut() as cut:
            assert cut.snapshot(store, "a" * 64) is None
    assert not reader._validation_evidence._resident
    assert reader._validation_evidence.accounted_bytes == 0


@pytest.mark.parametrize("size", [1, 3, 7])
def test_exact_plan_work_budget(tmp_path, size, monkeypatch, record_testsuite_property):
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
    reader = Repo(store)
    ledgers = []
    for phase in ("cold", "warm", "cleared"):
        if phase == "cleared":
            reader.clear_cache(strong=False, weak=False)
        payload_reads.clear()
        with exact_work_counts() as counts:
            plan = build_exact_state_load_plan(reader, state)
        assert len(plan.actions) == size
        assert counts["_read_snapshot"] == 1
        assert counts["validate_payload"] == size
        assert counts["_validate_local_state_dir"] == size
        assert counts["LocalStateManifest"] == size
        assert counts["DefinitionRecord"] == 2 * size + (size != 1)
        assert counts["definition_byte_hashes"] == size
        assert counts["json_reads"] == counts["json_shape_checks"] == 3
        assert counts["path_definition_checks"] == size
        assert len(payload_reads) == size
        assert counts["canonical_json_load_bytes"] == (0 if phase == "warm" else 3)
        ledgers.append(counts)
        record_testsuite_property(f"u5_k{size}_{phase}", repr(dict(sorted(counts.items()))))
    assert ledgers[0] == ledgers[2]


@pytest.mark.parametrize("size", [1, 3, 7])
def test_routed_stateful_parent_cut_work_budget(tmp_path, monkeypatch, size, record_testsuite_property):
    """Delegated child probes preserve parent promotion and successful content admission."""
    import builtins
    from dryml.core.materialization import build_exact_state_load_plan

    parent = DirStore(tmp_path / "parent", query_index="none")
    children = [DirStore(tmp_path / f"child-{i}", query_index="none") for i in range(size)]
    stores = [parent, *children]
    writer = Repo(stores, save_routing=SaveRouting((
        (Selector(StatefulEvidenceRoot), parent),
        *((Selector(Definition(EvidenceValue, i)), store) for i, store in enumerate(children)),
    ), graph_mode="per-object"))
    state = writer.save_object(StatefulEvidenceRoot(
        [EvidenceValue(i, repo=writer) for i in range(size)], repo=writer,
    ), deep_capture=True)
    child_paths = [path for path in state.states if path]
    expected_stores = {GraphPath(): parent, **dict(zip(child_paths, children))}
    assert len(state.states) == size + 1
    assert set(parent._read_snapshot(state.digest())[2]) == {GraphPath()}
    guarded = {store: tuple(getattr(store, name).__func__ for name in (
        "_read_snapshot", "_open_local_state_for_cut", "_validate_snapshot_locals",
    )) for store in stores}
    original_open = builtins.open
    payload_reads = []

    def observe_open(path, *args, **kwargs):
        if Path(path).name == "value" and Path(path).parent.name == "data":
            payload_reads.append(os.fspath(path))
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", observe_open)
    reader = Repo(stores)
    ledgers = []
    for phase in ("cold", "warm", "cleared"):
        if phase == "cleared":
            reader.clear_cache(strong=False, weak=False)
            assert reader._validation_evidence.accounted_bytes == 0
        payload_reads.clear()
        assert all(store._can_reuse_exact_validation() for store in stores)
        with exact_work_counts() as counts:
            with reader._read_cut() as cut:
                # A delegated-first probe must still promote the parent's locals.
                cut.open_local(parent, state, child_paths[-1])
                plan = build_exact_state_load_plan(reader, state, _read_cut=cut)
                joined = build_exact_state_load_plan(reader, state, _read_cut=cut)
                assert not cut._failed
                assert len(cut.snapshots) == len(cut.eager_snapshots) == size + 1
                if phase != "warm":
                    assert cut._pending
        assert plan.actions == joined.actions
        assert len(plan.actions) == size + 1
        for action in plan.actions:
            assert action.reference == state
            assert action.store is expected_stores[action.path]
            assert action.object_id == state.object.objects[action.path]
            assert action.state_hash == state.states[action.path] == action.payload.manifest.state_hash
            assert action.definition.graph_equal(state.object.at(action.path).definition)
            assert action.payload.eagerly_validated
        assert counts["complete_bundles"] == size + 1
        assert counts["json_reads"] == counts["json_shape_checks"] == 3 * (size + 1)
        assert counts["validate_payload"] == counts["_validate_local_state_dir"] == size + 1
        assert counts["LocalStateManifest"] == counts["definition_byte_hashes"] == size + 1
        assert counts["path_definition_checks"] == size + 1
        assert len(payload_reads) == len(set(payload_reads)) == size + 1
        assert counts["canonical_json_load_bytes"] == (0 if phase == "warm" else 3 * (size + 1))
        assert counts["decode_cdef_graph"] == (0 if phase == "warm" else counts["_build_graph_nodes"])
        assert (counts["_realize_prepared_graph"] > 0) == (phase == "warm")
        assert (counts["_graph_content_size"] == 0) == (phase == "warm")
        facility = reader._validation_evidence
        assert facility._resident and not facility._pending
        assert facility.accounted_bytes == sum(fact.charge for fact in facility._resident.values())
        assert reader._topology_leases == 0
        for store in stores:
            assert store._can_reuse_exact_validation()
            assert guarded[store] == tuple(getattr(store, name).__func__ for name in (
                "_read_snapshot", "_open_local_state_for_cut", "_validate_snapshot_locals",
            ))
        ledgers.append(counts)
        record_testsuite_property(f"routed_k{size}_{phase}", repr(dict(sorted(counts.items()))))
    assert ledgers[0] == ledgers[2]
    assert ledgers[1]["_graph_and_labels"] < ledgers[0]["_graph_and_labels"]
    assert ledgers[1]["_role_paths"] < ledgers[0]["_role_paths"]


@pytest.mark.parametrize("operation", ["plan", "load"])
def test_custom_legacy_validator_routes_per_object_children_uncached(tmp_path, operation):
    """Legacy eager validators keep generic per-object routing and fresh validation."""
    from dryml.core.materialization import build_exact_state_load_plan
    from dryml.core.store.records import LocalStateManifest

    class CustomDir(DirStore):
        calls = 0

        def _validate_local_state_dir(self, directory, manifest):
            assert isinstance(manifest, LocalStateManifest)
            self.calls += 1
            super()._validate_local_state_dir(directory, manifest)

    parent = CustomDir(tmp_path / "parent", query_index="none")
    child_store = DirStore(tmp_path / "child", query_index="none")
    writer = Repo([parent, child_store], save_routing=SaveRouting((
        (Selector(StatefulEvidenceRoot), parent), (Selector(EvidenceValue), child_store),
    ), graph_mode="per-object"))
    root = StatefulEvidenceRoot([EvidenceValue(1, repo=writer)], repo=writer)
    root.value = 123
    root.children[0].value = 456
    state = writer.save_object(root, deep_capture=True)
    child_path = next(path for path in state.states if path)
    assert set(parent._read_snapshot(state.digest())[2]) == {GraphPath()}
    assert child_store.read_state_ref_record(state.at(child_path).digest()) is not None
    assert not parent._can_reuse_exact_validation()
    assert child_store._can_reuse_exact_validation()
    reader = Repo([parent, child_store])
    ledgers, validations = [], []
    for phase in ("cold", "warm", "cleared"):
        if phase == "cleared":
            reader.clear_cache(strong=False, weak=False)
        parent.calls = 0
        with exact_work_counts() as counts:
            if operation == "plan":
                with reader._read_cut() as cut:
                    plan = build_exact_state_load_plan(reader, state, _read_cut=cut)
                    assert cut._failed and not cut._pending
                actions = {action.path: action for action in plan.actions}
                assert set(actions) == {GraphPath(), child_path}
                assert actions[GraphPath()].store is parent
                assert actions[child_path].store is child_store
                assert all(action.payload.eagerly_validated for action in plan.actions)
            else:
                restored = reader.load_state_ref(state, reuse_live="never", cache="none")
                assert restored.value == 123 and restored.children[0].value == 456
                assert restored.object_id == state.object.objects[GraphPath()]
                assert restored.children[0].object_id == state.object.objects[child_path]
        assert parent.calls > 0
        assert counts["validate_payload"] >= 2
        assert counts["canonical_json_load_bytes"] > 0
        facility = reader._validation_evidence
        assert not facility._resident and not facility._pending
        assert facility.accounted_bytes == 0 and reader._topology_leases == 0
        ledgers.append(counts)
        validations.append(parent.calls)
    assert ledgers[0] == ledgers[1] == ledgers[2]
    assert validations[0] == validations[1] == validations[2]


@pytest.mark.parametrize("reader_name", ["open_local_state", "_validate_local_state_dir"])
@pytest.mark.parametrize("error_type", [KeyError, StoreAuthorityError])
def test_custom_local_errors_keep_routing_contract(tmp_path, monkeypatch, reader_name, error_type):
    """Custom KeyError means missing placement; genuine authority errors reject routing."""
    from dryml.core.materialization import build_exact_state_load_plan

    parent = DirStore(tmp_path / "parent", query_index="none")
    writer = Repo(parent)
    child = EvidenceValue(3, repo=writer)
    child_state = writer.save_object(child)
    state = writer.save_object(EvidenceRoot(child, repo=writer), deep_capture=True)
    replica = DirStore.open_existing(parent.base_dir, query_index="none")
    assert replica.read_state_ref_record(child_state.digest()).state_ref == child_state

    original = getattr(parent, reader_name)
    parent_directory = os.path.normpath(parent.open_local_state(state, next(iter(state.states))).handle)
    child_directory = parent.open_local_state(child_state, GraphPath()).handle
    rejected = []

    def reject(target, *args, **kwargs):
        if target == state or (
                reader_name == "_validate_local_state_dir"
                and os.path.normpath(target) == parent_directory):
            rejected.append(target)
            raise error_type("custom reader local failure")
        return original(target, *args, **kwargs)

    monkeypatch.setattr(parent, reader_name, reject)
    reader = Repo([parent, replica])
    assert not parent._can_reuse_exact_validation()
    with reader._read_cut() as cut:
        if error_type is KeyError:
            plan = build_exact_state_load_plan(reader, state, source_store=parent, _read_cut=cut)
            assert len(plan.actions) == 1 and plan.actions[0].store is parent
            assert plan.actions[0].payload.handle == child_directory
            assert os.fspath(child_directory) != parent_directory
            assert plan.actions[0].payload.eagerly_validated
        else:
            with pytest.raises(RepoLoadError, match="custom reader local failure"):
                build_exact_state_load_plan(reader, state, source_store=parent, _read_cut=cut)
            assert not cut.snapshots
        assert cut._failed and not cut._pending
    assert rejected
    assert not reader._validation_evidence._resident
    assert not reader._validation_evidence._pending
    assert reader._validation_evidence.accounted_bytes == 0


@pytest.mark.parametrize("component", ["manifest.record", "def.pkl", "data/value"])
def test_delegated_probe_still_poisoned_by_advertised_parent_corruption(tmp_path, component):
    """Delegated-first eager probes cannot bypass the parent's advertised locals."""
    from dryml.core.materialization import build_exact_state_load_plan
    from dryml.core.store.store import StoreAuthorityError

    parent = DirStore(tmp_path / "parent", query_index="none")
    child_store = DirStore(tmp_path / "child", query_index="none")
    writer = Repo([parent, child_store], save_routing=SaveRouting((
        (Selector(StatefulEvidenceRoot), parent), (Selector(EvidenceValue), child_store),
    ), graph_mode="per-object"))
    state = writer.save_object(StatefulEvidenceRoot(
        [EvidenceValue(1, repo=writer)], repo=writer,
    ), deep_capture=True)
    child_path = next(path for path in state.states if path)
    assert child_store.open_local_state(state.at(child_path), GraphPath()).eagerly_validated
    local = parent.open_local_state(state, GraphPath())
    Path(local.handle, component).write_bytes(b"corrupt parent authority")
    reader = Repo([parent, child_store])
    with reader._read_cut() as cut:
        with pytest.raises(StoreAuthorityError):
            cut.open_local(parent, state, child_path)
        assert cut._failed and not cut.snapshots and not cut.eager_snapshots and not cut._pending
    with pytest.raises(RepoLoadError, match="local state"):
        build_exact_state_load_plan(reader, state, source_store=parent)
    assert not reader._validation_evidence._resident and not reader._validation_evidence._pending


def test_routed_deferred_child_does_not_authorize_eager_promotion(tmp_path):
    """Normal delegated restore routing keeps deferred and eager evidence distinct."""
    from dryml.core.materialization import build_exact_state_load_plan
    from tests.core.test_deferred_local_state import DeferredPayload

    parent = DirStore(tmp_path / "parent", query_index="none")
    child_store = DirStore(tmp_path / "child", query_index="none")
    writer = Repo([parent, child_store], save_routing=SaveRouting((
        (Selector(StatefulEvidenceRoot), parent), (Selector(DeferredPayload), child_store),
    ), graph_mode="per-object"))
    state = writer.save_object(StatefulEvidenceRoot(
        [DeferredPayload(repo=writer)], repo=writer,
    ), deep_capture=True)
    child_path = next(path for path in state.states if path)
    local = child_store.open_local_state(state.at(child_path), GraphPath())
    Path(local.handle, "data", "payload.bin").write_bytes(b"corrupt")
    reader = Repo([parent, child_store])
    with reader._read_cut() as cut:
        deferred = build_exact_state_load_plan(reader, state, _defer_payload=True, _read_cut=cut)
        assert not cut._failed and len(cut.snapshots) == 2 and not cut.eager_snapshots
        actions = {action.path: action for action in deferred.actions}
        assert actions[GraphPath()].payload.eagerly_validated
        assert not actions[child_path].payload.eagerly_validated
        assert actions[child_path].store is child_store
        with pytest.raises(RepoLoadError, match="payload is invalid"):
            build_exact_state_load_plan(reader, state, _read_cut=cut)
        assert cut._failed and not cut.locals and not cut.eager_snapshots and not cut._pending
    assert not reader._validation_evidence._resident


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
    "validate_local_state", "_read_snapshot", "_validate_local_state_dir", "read_definition_record",
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

        def _validate_local_state_dir(self, directory, manifest):
            assert isinstance(manifest, LocalStateManifest)
            self.calls += 1
            super()._validate_local_state_dir(directory, manifest)

    store = CustomDir(tmp_path / "store", query_index="none")
    writer = Repo(store)
    state = writer.save_object(EvidenceValue(1, repo=writer))
    assert not store._can_reuse_exact_validation()
    assert store.read_state_ref_record(state.digest()).state_ref == state
    assert store.read_snapshot_metadata(state.digest()).state_ref == state
    manifest = store.validate_local_state(state, GraphPath())
    source = store.open_local_state(state, GraphPath())
    assert source.manifest == manifest
    assert Path(source.handle, "data", "value").read_text(encoding="ascii") == "1"
    reader = Repo(store)
    for phase in ("cold", "warm", "cleared"):
        if phase == "cleared":
            reader.clear_cache(strong=False, weak=False)
        store.calls = 0
        with exact_work_counts() as counts:
            plan = build_exact_state_load_plan(reader, state)
        assert len(plan.actions) == 1
        assert store.calls == 1
        assert counts["_read_snapshot"] == 4
        assert counts["canonical_json_load_bytes"] == 12
        assert counts["validate_payload"] == 1
        assert not reader._validation_evidence._resident


def _override_legacy_zip_delegate(monkeypatch, reader_name, calls, *, reject=False):
    """Replace the actual DirStore delegate with its pre-preparation signature."""
    original = getattr(DirStore, reader_name)

    def called():
        calls.append(reader_name)
        if reject:
            raise RuntimeError("legacy base reader rejects authority")

    if reader_name == "_read_file":
        def reader(self, path, record_type):
            called()
            return original(self, path, record_type)
    elif reader_name == "_read_json":
        def reader(path):
            called()
            return original(path)
        reader = staticmethod(reader)
    else:
        def reader(self, digest, *, payloads=False, directory=None):
            called()
            return original(self, digest, payloads=payloads, directory=directory)
    monkeypatch.setattr(DirStore, reader_name, reader)


@pytest.mark.parametrize("reader_name", ["_read_file", "_read_json", "_read_snapshot"])
@pytest.mark.parametrize("phase", ["cold", "warm"])
def test_zip_legacy_base_reader_override(tmp_path, monkeypatch, reader_name, phase):
    """Public and exact archive reads honor legacy delegates, even after warming."""
    from dryml.core.materialization import build_exact_state_load_plan
    from dryml.core.store.zip import ZipStore

    store = ZipStore(tmp_path / "store.zip")
    try:
        writer = Repo(store)
        state = writer.save_object(EvidenceValue(1, repo=writer))
        reader = Repo(store)
        assert store._can_reuse_exact_validation()
        if phase == "warm":
            build_exact_state_load_plan(reader, state)
            assert reader._validation_evidence._resident
        retained = tuple(reader._validation_evidence._resident.values())
        calls = []
        _override_legacy_zip_delegate(monkeypatch, reader_name, calls)

        assert store.read_state_ref_record(state.digest()).state_ref == state
        assert store.read_snapshot_metadata(state.digest()).state_ref == state
        source = store.open_local_state(state, GraphPath())
        assert source.eagerly_validated
        assert calls
        assert not store._can_reuse_exact_validation()
        for _ in range(2):
            calls.clear()
            plan = build_exact_state_load_plan(reader, state)
            assert plan.state_ref == state
            assert len(plan.actions) == 1
            assert plan.actions[0].payload.handle == source.handle
            assert calls
            assert tuple(reader._validation_evidence._resident.values()) == retained
            assert not reader._validation_evidence._pending

        store.close()
        assert not Path(source.handle).exists()
        with pytest.raises(RuntimeError, match="closed"):
            store.read_state_ref_record(state.digest())
        with pytest.raises(RuntimeError, match="closed"):
            store._can_reuse_exact_validation()
    finally:
        store.close()


@pytest.mark.parametrize("reader_name", ["_read_file", "_read_json", "_read_snapshot"])
@pytest.mark.parametrize("phase", ["cold", "warm"])
def test_zip_rejecting_base_reader_override(tmp_path, monkeypatch, reader_name, phase):
    """Actual base delegates disable reuse and preserve custom rejecting errors."""
    from dryml.core.materialization import build_exact_state_load_plan
    from dryml.core.store.zip import ZipStore

    store = ZipStore(tmp_path / "store.zip")
    try:
        writer = Repo(store)
        state = writer.save_object(EvidenceValue(1, repo=writer))
        reader = Repo(store)
        assert store._can_reuse_exact_validation()
        if phase == "warm":
            build_exact_state_load_plan(reader, state)
            assert reader._validation_evidence._resident
        retained = tuple(reader._validation_evidence._resident.values())
        calls = []
        _override_legacy_zip_delegate(monkeypatch, reader_name, calls, reject=True)

        assert not store._can_reuse_exact_validation()
        with pytest.raises(RuntimeError, match="legacy base reader rejects authority"):
            store.read_state_ref_record(state.digest())
        assert calls
        for _ in range(2):
            calls.clear()
            with pytest.raises(RepoLoadError, match="legacy base reader rejects authority"):
                build_exact_state_load_plan(reader, state)
            assert calls
            assert tuple(reader._validation_evidence._resident.values()) == retained
            assert not reader._validation_evidence._pending
        assert reader._topology_leases == 0
    finally:
        store.close()


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


@pytest.mark.parametrize("override", ["alias", "code"])
@pytest.mark.parametrize("phase", ["cold", "warm", "cleared"])
def test_exact_records_decoder_override_remains_authoritative(tmp_path, monkeypatch, override, phase):
    """Eligible exact reads cannot bypass a records-only rejecting decoder."""
    import dryml.core.store.records as records
    from dryml.core.materialization import build_exact_state_load_plan
    from dryml.core.store.store import StoreAuthorityError
    from tests.core.test_store_records import _isolate_records_decoder_code, _reject_records_decoder

    store = DirStore(tmp_path / "store", query_index="none")
    writer = Repo(store)
    state = writer.save_object(EvidenceValue(1, repo=writer))
    original = (_isolate_records_decoder_code(monkeypatch) if override == "code"
                else records.decode_cdef_graph)
    reader = Repo(store)
    if phase != "cold":
        build_exact_state_load_plan(reader, state)
        assert reader._validation_evidence._resident
    if phase == "cleared":
        reader.clear_cache(strong=False, weak=False)
        assert reader._validation_evidence.accounted_bytes == 0
    retained = tuple(reader._validation_evidence._resident.values())
    _reject_records_decoder(monkeypatch, override, original)
    assert store._can_reuse_exact_validation()
    with pytest.raises(StoreAuthorityError, match="graph authority"):
        store.open_local_state(state, GraphPath())
    for _ in range(2):
        with pytest.raises(RepoLoadError, match="graph authority"):
            build_exact_state_load_plan(reader, state)
        assert {id(fact) for fact in reader._validation_evidence._resident.values()} == {
            id(fact) for fact in retained
        }
        assert not reader._validation_evidence._pending
    assert reader._topology_leases == 0


def test_exact_records_decoder_delegate_keeps_uncached_policy(tmp_path, monkeypatch):
    """A successful alias wrapper runs on every exact cut and can later reject."""
    import dryml.core.store.records as records
    from dryml.core.materialization import build_exact_state_load_plan

    store = DirStore(tmp_path / "store", query_index="none")
    writer = Repo(store)
    state = writer.save_object(EvidenceValue(1, repo=writer))
    reader = Repo(store)
    original = records.decode_cdef_graph
    calls = []
    rejecting = False

    def decoder(data):
        calls.append(True)
        if rejecting:
            raise ValueError("custom decoder now rejects graph")
        return original(data)

    for phase in ("cold", "warm", "cleared"):
        if phase == "warm":
            with monkeypatch.context() as patch:
                patch.setattr(records, "decode_cdef_graph", original)
                build_exact_state_load_plan(reader, state)
        if phase == "cleared":
            reader.clear_cache(strong=False, weak=False)
        monkeypatch.setattr(records, "decode_cdef_graph", decoder)
        for _ in range(2):
            calls.clear()
            with exact_work_counts() as counts:
                assert build_exact_state_load_plan(reader, state).state_ref == state
            assert len(calls) == 2
            # StateRef decoding owns a different decoder binding and stays eligible.
            assert counts["_realize_prepared_graph"] <= 1
        rejecting = True
        with pytest.raises(RepoLoadError, match="graph authority"):
            build_exact_state_load_plan(reader, state)
        rejecting = False


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
