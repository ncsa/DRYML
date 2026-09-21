"""U7 barrier-controlled overlap coverage for snapshot metadata authority."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from threading import Barrier, Event, Thread

import dryml.environments as envs
import pytest

from dryml.core import MetadataConflictError, Repo, SaveAnnotations, Serializable
from dryml.core.query import field
from dryml.core.store.dir import DirStore
from dryml.core.store.records import DefinitionRecord, LocalStateManifest
from dryml.core.metadata import LineageMetadata, SnapshotCapture
from dryml.core.reference_values import StateRef
from dryml.core.store.zip import ZipStore
from dryml.core.utils.graph.path import GraphPath


class ConcurrentPayload(Serializable):
    """Stateful fixture whose equal payloads produce one exact StateRef."""

    def __init__(self, value="same"):
        self.value = value

    def save_state_to_dir_imp(self, directory, *, codec):
        Path(directory, "value").write_text(self.value, encoding="ascii")


def test_concurrent_first_publications_select_one_complete_capture_for_replicas(tmp_path):
    """Two candidate captures cannot create a mixed or discarded replica record."""

    first = DirStore(tmp_path / "first")
    second = DirStore(tmp_path / "second")
    value = ConcurrentPayload("same")
    record = DefinitionRecord(value.definition)
    definition_bytes = record.to_bytes()
    payload = b"same"
    manifest = LocalStateManifest(
        value.state_codec, record.graph_hash, record.digest,
        hashlib.sha256(definition_bytes).hexdigest(),
        (("value", len(payload), hashlib.sha256(payload).hexdigest()),),
    )
    reference = StateRef(value.object_ref, {GraphPath(): manifest.state_hash})
    barrier = Barrier(2)
    candidates = [
        SnapshotCapture(
            {GraphPath(): LineageMetadata(reference.object, "unknown", None)},
            datetime(2020, 1, day, tzinfo=timezone.utc),
            envs.EnvironmentRecord(
                python=envs.PythonRecord(f"3.12.{day}", "CPython"),
                platform=envs.PlatformRecord("Linux", "1", "v", "x86_64", "Linux-x86_64"),
                distributions={},
                dryml=envs.DrymlRuntimeRecord(),
            ),
            "known", None, "unavailable", "incomplete", (),
        )
        for day in (1, 2)
    ]
    metadata = []
    errors = []

    def publish(candidate):
        stage = first.create_local_state_staging()
        try:
            Path(stage, "data", "value").write_bytes(payload)
            Path(stage, "def.pkl").write_bytes(definition_bytes)
            Path(stage, "manifest.record").write_bytes(manifest.to_bytes())
            source = first.prepare_local_state(stage, manifest)
            barrier.wait(timeout=10)
            metadata.append(first.publish_snapshot(
                reference, evidence=candidate, local_states={GraphPath(): source},
            ))
        except BaseException as error:
            errors.append(error)
        finally:
            first.discard_local_state_staging(stage)

    threads = [
        Thread(target=publish, args=(candidate,)) for candidate in candidates
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=20)
        assert not thread.is_alive()
    assert errors == []
    assert len(metadata) == 2 and metadata[0] == metadata[1]
    winner = first.read_snapshot_metadata(reference.digest())
    assert winner == metadata[0]
    selected = next(candidate for candidate in candidates if candidate.saved_at == winner.saved_at)
    assert winner.environment == selected.environment

    stage = second.create_local_state_staging()
    try:
        Path(stage, "data", "value").write_bytes(payload)
        Path(stage, "def.pkl").write_bytes(definition_bytes)
        Path(stage, "manifest.record").write_bytes(manifest.to_bytes())
        source = second.prepare_local_state(stage, manifest)
        assert second.publish_snapshot(
            reference, evidence=winner, local_states={GraphPath(): source},
        ) == winner
    finally:
        second.discard_local_state_staging(stage)
    assert second.read_snapshot_metadata(reference.digest()) == winner


def test_concurrent_lww_replace_and_delete_recreate_expose_complete_mappings_only(tmp_path):
    """Concurrent current-map operations retain whole mappings and frozen captures."""

    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    state = repo.save_object(
        ConcurrentPayload(repo=repo), annotations=SaveAnnotations(object={"captured": True}),
    )
    captured = repo.get_snapshot_metadata(state)
    start = Barrier(3)
    values = ({"writer": "left", "nested": [1]}, {"writer": "right", "nested": [2]})
    errors = []

    def replace_value(value):
        try:
            start.wait(timeout=10)
            repo.set_metadata(state.object, value)
        except BaseException as error:
            errors.append(error)

    writers = [Thread(target=replace_value, args=(value,)) for value in values]
    for writer in writers:
        writer.start()
    start.wait(timeout=10)
    for writer in writers:
        writer.join(timeout=10)
        assert not writer.is_alive()
    assert errors == []
    assert repo.get_metadata(state.object) in values
    assert repo.get_snapshot_metadata(state) == captured

    delete_start = Barrier(3)

    def delete():
        delete_start.wait(timeout=10)
        repo.delete_metadata(state.object)

    def recreate():
        delete_start.wait(timeout=10)
        repo.set_metadata(state.object, {"recreated": True})

    removers = [Thread(target=delete), Thread(target=recreate)]
    for worker in removers:
        worker.start()
    delete_start.wait(timeout=10)
    for worker in removers:
        worker.join(timeout=10)
        assert not worker.is_alive()
    assert repo.get_metadata(state.object) in (None, {"recreated": True})
    assert repo.get_snapshot_metadata(state) == captured


def test_distinct_zip_transactions_are_all_fenced_for_query_source_and_fork_reads(
        tmp_path, monkeypatch):
    """One archive path does not collapse independent buffered authority cuts."""

    archive = tmp_path / "store.zip"
    writer = ZipStore(archive)
    state = Repo(writer).save_object(
        ConcurrentPayload(repo=Repo(writer)),
        annotations=SaveAnnotations(object={"view": "initial"}, state={"view": "initial"}),
    )
    writer.commit()
    writer.close()
    first = ZipStore.open_existing(archive)
    second = ZipStore.open_existing(archive)
    target = DirStore(tmp_path / "target")
    first.write_metadata(state.object, {"view": "first"})
    second.write_metadata(state.object, {"view": "second"})
    active = {}
    maximum = [0]

    for store in (first, second):
        original = store.authority_read_fence

        @contextmanager
        def tracked(original=original, store=store):
            with original():
                key = id(store)
                active[key] = active.get(key, 0) + 1
                maximum[0] = max(maximum[0], len(active))
                try:
                    yield
                finally:
                    active[key] -= 1
                    if not active[key]:
                        del active[key]

        monkeypatch.setattr(store, "authority_read_fence", tracked)

    repo = Repo([first, second])
    with pytest.raises(MetadataConflictError):
        repo.references().exact(state.object).where(
            field("object", "view").eq("first")
        ).object_refs()
    assert maximum[0] == 2

    maximum[0] = 0
    repo.add_store(target)
    evidence = repo.reference_evidence(state.definition)
    assert evidence.states[0].state_ref == state
    assert maximum[0] == 2

    maximum[0] = 0
    fork = repo.fork_state_ref(
        state, store=target, source_store=first,
        copy_annotations=("object", "state"),
    )
    assert target.read_state_ref_record(fork.digest()).state_ref == fork
    assert maximum[0] == 2

    first.close()
    second.close()


def test_distinct_zip_transaction_query_cut_blocks_overlapping_mapping_write(
        tmp_path, monkeypatch):
    """A writer cannot interleave one mapping into a multi-transaction read cut."""

    archive = tmp_path / "store.zip"
    writer = ZipStore(archive)
    writer_repo = Repo(writer)
    state = writer_repo.save_object(
        ConcurrentPayload(repo=writer_repo),
        annotations=SaveAnnotations(object={"view": "initial"}),
    )
    writer.commit()
    writer.close()
    first = ZipStore.open_existing(archive)
    second = ZipStore.open_existing(archive)
    repo = Repo([first, second])
    entered = Event()
    release = Event()
    write_started = Event()
    write_completed = Event()
    results = []
    errors = []
    original_read = first.read_metadata

    def pause_read(target):
        if target == state.object and not entered.is_set():
            entered.set()
            assert release.wait(timeout=10)
        return original_read(target)

    def query():
        try:
            results.extend(repo.references().exact(state.object).where(
                field("object", "view").eq("initial")
            ).object_refs())
        except BaseException as error:
            errors.append(error)

    def mutate():
        write_started.set()
        first.write_metadata(state.object, {"view": "updated"})
        write_completed.set()

    monkeypatch.setattr(first, "read_metadata", pause_read)
    reader = Thread(target=query)
    reader.start()
    assert entered.wait(timeout=10)
    updater = Thread(target=mutate)
    updater.start()
    assert write_started.wait(timeout=10)
    assert not write_completed.is_set()
    release.set()
    reader.join(timeout=10)
    updater.join(timeout=10)

    assert not reader.is_alive() and not updater.is_alive()
    assert errors == []
    assert results == [state.object]
    assert first.read_metadata(state.object) == {"view": "updated"}
    first.close()
    second.close()
