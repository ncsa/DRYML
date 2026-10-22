"""Deterministic Store-v4 mutation cuts, crash evidence and lock boundaries."""

from contextlib import contextmanager
import multiprocessing
import os
from pathlib import Path
from threading import Event, Thread

import pytest

from dryml.core import Definition, Object, ObjectId
from dryml.core.node_prepare import NodePreparation
from dryml.core.node_views import CREATION_VIEW, view_digest
from dryml.core.store.authority import StoreCutChanged, StoreMutationError
from dryml.core.store.dir import DirStore
from dryml.core.store.records import MutationIntentRecord, NodeNameRecord
from dryml.core.store.store import StoreAuthorityError, StoreCapabilityError
from dryml.core.store.zip import ZipStore


class JournalObject(Object):
    def __init__(self, value=1):
        self.value = value


def creation(value=1):
    return Definition(JournalObject, value=value).concretize()


def install(store, payload=b"state"):
    stage = store.create_local_state_staging()
    Path(stage, "data", "value").write_bytes(payload)
    return store.install_state(creation(), stage, codec="Test", codec_version=1, interpretation={})


def intent_records(store):
    return tuple(MutationIntentRecord.from_bytes(p.read_bytes())
                 for p in sorted(Path(store.base_dir, "mutations").glob("*.record")))


def test_names_membership_and_declarations_share_journal_cut(tmp_path):
    store = DirStore(tmp_path / "store", query_index="none")
    identity = ObjectId()
    root = store.publish_node(creation().with_identity(identity))
    name = NodeNameRecord("main", "recipe", root.digest)
    before = store.authority_cut()
    assert store.compare_and_set_node_name(name, expected=None) == name
    assert store.authority_cut().epoch == before.epoch + 1
    stable = store.authority_cut()
    assert store.compare_and_set_node_name(name, expected=name) == name
    assert store.authority_cut() == stable
    assert store.read_node_name("main") == name
    assert not store.withdraw_node(root.digest)
    assert store.delete_node_name("main", expected=name)
    assert store.withdraw_node(root.digest)
    assert store.read_node(root.digest) is None
    assert store.read_declaration_anchor(identity).sources == ()
    assert store.read_creation(view_digest(creation(), CREATION_VIEW)) is None
    assert {scope for r in intent_records(store) for scope in r.intent["scopes"]} >= {
        "names", "membership", "declarations"}


def test_no_graph_work_or_hooks_under_writer_or_short_read_fences(tmp_path, monkeypatch):
    import dryml.core.node_codec as codec
    import dryml.core.node_views as views
    from dryml.core.state_values import LocalState
    store = DirStore(tmp_path / "store", query_index="none")
    active = [0]
    original_lock = store.writer_lock
    @contextmanager
    def tracked():
        with original_lock():
            active[0] += 1
            try:
                yield
            finally:
                active[0] -= 1
    monkeypatch.setattr(store, "writer_lock", tracked)
    def guard(original):
        def guarded(*args, **kwargs):
            assert active[0] == 0, "graph preparation or hook under writer fence"
            return original(*args, **kwargs)
        return guarded
    monkeypatch.setattr(NodePreparation, "__init__", guard(NodePreparation.__init__))
    monkeypatch.setattr(codec, "decode_node_graph", guard(codec.decode_node_graph))
    monkeypatch.setattr(codec, "encode_node_graph", guard(codec.encode_node_graph))
    monkeypatch.setattr(views, "view_equal", guard(views.view_equal))
    monkeypatch.setattr(LocalState, "validate_for", guard(LocalState.validate_for))
    monkeypatch.setattr(store, "_authenticate_state_data", guard(store._authenticate_state_data))
    store.install_creations((creation(), creation(2)))
    store.install_creation(creation())
    result = install(store)
    assert not install(store).installed
    root = creation().with_identity(ObjectId()).with_state(result.state)
    published = store.publish_node(root, kind="saved")
    store.publish_node(root, kind="saved")
    assert store.read_node(published.digest, kind="saved").object_id == root.object_id
    store.read_declaration_anchor(root.object_id)
    store.inspect_states(result.state.key.cdef_digest)
    with store.read_state(result.state.key, destination=creation(), codec="Test", codec_version=1):
        assert active[0] == 0
    name = NodeNameRecord("main", "saved", published.digest)
    store.compare_and_set_node_name(name, expected=None)
    renamed = store.rename_node_name("main", "renamed", expected=name)
    assert store.read_node_name("renamed") == renamed
    store.delete_node_name("renamed", expected=renamed)
    assert store.withdraw_node(published.digest, kind="saved")
    assert store.read_declaration_anchor(root.object_id).sources == ()


@pytest.mark.parametrize("boundary", ["intent", "named-intent", "pending-cut", "header", "variant", "membership", "name", "cut", "ack"])
def test_failure_boundaries_preserve_exact_replay_evidence(tmp_path, monkeypatch, boundary):
    import dryml.filesystem as fs
    store = DirStore(tmp_path / "store", query_index="none")
    root = store.publish_node(creation()) if boundary == "name" else None
    before = store.authority_cut()
    original_file, original_directory = fs.publish_file, fs.publish_directory
    fired = []
    def publish_file(source, target, *, replace=False):
        result = original_file(source, target, replace=replace)
        relative = str(Path(target).relative_to(store.base_dir))
        matches = ((boundary == "intent" and relative == "mutation-prepare.record")
                   or (boundary == "named-intent" and relative.startswith("mutations/"))
                   or (boundary == "pending-cut" and relative == "authority.record"
                       and store.authority_cut().epoch == before.epoch and store.authority_cut().pending)
                   or (boundary == "header" and relative.endswith("def.pkl"))
                   or (boundary == "membership" and relative.startswith("membership/"))
                   or (boundary == "name" and relative.startswith("names/"))
                   or (boundary == "cut" and relative == "authority.record"
                       and store.authority_cut().epoch > before.epoch))
        if matches and not fired:
            fired.append(True)
            raise OSError("injected " + boundary)
        return result
    def directory(source, target):
        result = original_directory(source, target)
        if boundary == "variant" and not fired:
            fired.append(True)
            raise OSError("injected variant")
        return result
    original_ack = __import__("dryml.core.store.authority", fromlist=["acknowledge"]).acknowledge
    def ack(*args):
        if boundary == "ack" and not fired:
            fired.append(True)
            raise OSError("injected ack")
        return original_ack(*args)
    with monkeypatch.context() as patch:
        patch.setattr(fs, "publish_file", publish_file)
        patch.setattr(fs, "publish_directory", directory)
        patch.setattr("dryml.core.store.authority.acknowledge", ack)
        with pytest.raises(StoreMutationError, match="injected") as raised:
            if boundary == "name":
                store.compare_and_set_node_name(NodeNameRecord("main", "recipe", root.digest), expected=None)
            elif boundary in {"membership", "ack"}:
                store.publish_node(creation().with_identity(ObjectId()))
            else:
                install(store)
    assert fired
    error = raised.value
    assert error.mutation_id and error.index_disposition == "repair_pending"
    reopened = DirStore(store.base_dir, query_index="none")
    records = reopened.recover_authority()
    assert records and all(r.intent["status"] == "classified" for r in records)
    record = next(r for r in records if r.intent["mutation_id"] == error.mutation_id)
    assert record.intent["projection"]
    cut = reopened.authority_cut()
    assert cut.epoch == before.epoch + int(boundary not in {"intent", "named-intent", "pending-cut"})
    assert reopened.recover_authority() == records
    assert reopened.authority_cut() == cut
    if boundary == "header":
        assert reopened.inspect_states(view_digest(creation(), CREATION_VIEW)) == ()
    if boundary == "variant":
        assert len(reopened.inspect_states(view_digest(creation(), CREATION_VIEW))) == 1


def test_intent_write_failure_before_visibility_has_no_effects(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store", query_index="none")
    before = store.authority_cut()
    original = store._atomic_write
    def fail(path, data):
        if str(path).endswith("mutation-prepare.record"):
            raise OSError("intent unavailable")
        return original(path, data)
    monkeypatch.setattr(store, "_atomic_write", fail)
    with pytest.raises(StoreMutationError, match="intent unavailable"):
        store.install_creation(creation())
    assert store.authority_cut() == before
    assert not Path(store.base_dir, "states").exists()


def test_old_repair_ack_cannot_acknowledge_later_writer(tmp_path, monkeypatch):
    import dryml.core.store.authority as authority
    store = DirStore(tmp_path / "store", query_index="none")
    original = authority.acknowledge
    def fail(*args):
        raise OSError("ack unavailable")
    with monkeypatch.context() as patch:
        patch.setattr(authority, "acknowledge", fail)
        for value in (1, 2):
            with pytest.raises(StoreMutationError):
                store.install_creation(creation(value))
    first, second = store.recover_authority()
    first_id, second_id = (r.intent["mutation_id"] for r in (first, second))
    with pytest.raises(StoreAuthorityError, match="Acknowledgement"):
        store.acknowledge_mutation(first_id, second.intent["epoch"])
    result = store.acknowledge_mutation(first_id, first.intent["epoch"])
    assert result.pending == (second_id,)
    assert original(store, first_id, first.intent["epoch"]) == result


def _spawn_writer(path, index, ready, release, results, boundary=None):
    import dryml.filesystem as fs
    store = DirStore(path, query_index="none")
    ready.put(index)
    assert release.wait(20)
    if boundary:
        original_file, original_directory = fs.publish_file, fs.publish_directory
        def file(source, target, *, replace=False):
            result = original_file(source, target, replace=replace)
            relative = str(Path(target).relative_to(store.base_dir))
            if ((boundary == "header" and relative.endswith("def.pkl"))
                    or (boundary == "membership" and relative.startswith("membership/"))
                    or (boundary == "name" and relative.startswith("names/"))):
                os._exit(71)
            return result
        def directory(source, target):
            original_directory(source, target)
            if boundary == "variant":
                os._exit(71)
        fs.publish_file, fs.publish_directory = file, directory
    if boundary == "membership":
        store.publish_node(creation().with_identity(ObjectId()))
    elif boundary == "name":
        root = store.publish_node(creation())
        store.compare_and_set_node_name(NodeNameRecord("main", "recipe", root.digest), expected=None)
    else:
        result = install(store, bytes([index]))
        results.put(result.state.key)


def test_spawned_sibling_writers_never_replace_bucket(tmp_path):
    store = DirStore(tmp_path / "store", query_index="none")
    context = multiprocessing.get_context("spawn")
    ready, results, release = context.Queue(), context.Queue(), context.Event()
    workers = [context.Process(target=_spawn_writer, args=(store.base_dir, i, ready, release, results))
               for i in (1, 2)]
    for worker in workers:
        worker.start()
    assert {ready.get(timeout=20) for _ in workers} == {1, 2}
    release.set()
    for worker in workers:
        worker.join(20)
        assert worker.exitcode == 0
    keys = {results.get(timeout=10) for _ in workers}
    assert {d.state.key for d in store.inspect_states(next(iter(keys)).cdef_digest)} == keys


@pytest.mark.parametrize("boundary", ["header", "variant", "membership", "name"])
def test_spawned_crash_leaves_inspectable_persisted_intent(tmp_path, boundary):
    store = DirStore(tmp_path / "store", query_index="none")
    context = multiprocessing.get_context("spawn")
    ready, results, release = context.Queue(), context.Queue(), context.Event()
    worker = context.Process(target=_spawn_writer,
                             args=(store.base_dir, 1, ready, release, results, boundary))
    worker.start()
    assert ready.get(timeout=20) == 1
    release.set()
    worker.join(20)
    assert worker.exitcode == 71
    reopened = DirStore(store.base_dir, query_index="none")
    assert reopened.authority_cut().pending
    recovered = reopened.recover_authority()
    assert recovered and recovered[-1].intent["confirmed"]
    assert recovered[-1].intent["epoch"] == reopened.authority_cut().epoch


def test_barrier_reader_never_observes_half_installed_variant(tmp_path, monkeypatch):
    import dryml.filesystem as fs
    store = DirStore(tmp_path / "store", query_index="none")
    install(store)
    entered, release, reading, done = Event(), Event(), Event(), Event()
    original = fs.publish_directory
    errors, values = [], []
    def pause(source, target):
        entered.set()
        assert release.wait(10)
        return original(source, target)
    def writer():
        try:
            install(store, b"sibling")
        except BaseException as error:
            errors.append(error)
    def reader():
        reading.set()
        try:
            values.extend(store.inspect_states(view_digest(creation(), CREATION_VIEW)))
        except BaseException as error:
            errors.append(error)
        finally:
            done.set()
    monkeypatch.setattr(fs, "publish_directory", pause)
    writing = Thread(target=writer)
    writing.start()
    assert entered.wait(10)
    read = Thread(target=reader)
    read.start()
    assert reading.wait(10) and not done.is_set()
    release.set()
    writing.join(10)
    read.join(10)
    assert not writing.is_alive() and not read.is_alive() and not errors
    assert len(values) == 2 and all(d.available for d in values)


def test_zip_failed_commit_keeps_old_archive_and_pending_buffer(tmp_path, monkeypatch):
    import dryml.filesystem as fs
    archive = tmp_path / "store.zip"
    store = ZipStore(archive)
    try:
        first = install(store)
        store.commit()
        baseline = archive.read_bytes()
        second = install(store, b"new")
        buffered_cut = store.authority_cut()
        original = fs.publish_file
        def fail(source, target, *, replace=False):
            if str(target) == str(archive):
                raise OSError("archive unavailable")
            return original(source, target, replace=replace)
        with monkeypatch.context() as patch:
            patch.setattr(fs, "publish_file", fail)
            with pytest.raises(OSError, match="archive unavailable"):
                store.commit()
        assert archive.read_bytes() == baseline and store._archive_dirty
        assert store.authority_cut() == buffered_cut
        assert intent_records(store)
        old = ZipStore(archive)
        try:
            assert {d.state.key for d in old.inspect_states(first.state.key.cdef_digest)} == {first.state.key}
        finally:
            old.close()
        store.commit()
        reopened = ZipStore(archive)
        try:
            assert {d.state.key for d in reopened.inspect_states(first.state.key.cdef_digest)} == {
                first.state.key, second.state.key}
        finally:
            reopened.close()
    finally:
        store.close()


def test_explicit_sqlite_policy_refuses_before_publication_effects(tmp_path):
    store = DirStore(tmp_path / "store", query_index="sqlite")
    cut = store.authority_cut()
    with pytest.raises(StoreCapabilityError, match="coverage"):
        store.publish_node(creation())
    assert store.authority_cut() == cut
    assert not Path(store.base_dir, "mutations").exists()


@pytest.mark.parametrize("boundary", ["header", "anchor", "rename"])
def test_exact_replay_completes_partial_work_without_preparation_or_hooks(tmp_path, monkeypatch, boundary):
    import dryml.filesystem as fs
    store = DirStore(tmp_path / "store", query_index="none")
    identity = ObjectId()
    named = None
    if boundary == "rename":
        root = store.publish_node(creation().with_identity(identity))
        named = NodeNameRecord("old", "recipe", root.digest)
        store.compare_and_set_node_name(named, expected=None)
    original = fs.publish_file
    def fail(source, target, *, replace=False):
        result = original(source, target, replace=replace)
        relative = str(Path(target).relative_to(store.base_dir))
        if ((boundary == "header" and relative.endswith("def.pkl"))
                or (boundary == "anchor" and relative.startswith("declarations/anchors/"))
                or (boundary == "rename" and relative == "names/new.record")):
            raise OSError("interrupted before remaining transitions")
        return result
    with monkeypatch.context() as patch:
        patch.setattr(fs, "publish_file", fail)
        with pytest.raises(StoreMutationError) as raised:
            if boundary == "header":
                install(store)
            elif boundary == "anchor":
                store.publish_node(creation().with_identity(identity))
            else:
                store.rename_node_name("old", "new", expected=named)
    mutation_id = raised.value.mutation_id
    record = next(r for r in store.pending_mutations() if r.intent["mutation_id"] == mutation_id)
    with monkeypatch.context() as patch:
        patch.setattr(NodePreparation, "__init__", lambda *a, **kw: pytest.fail("recovery prepared a graph"))
        patch.setattr(store, "_prepare_node_roots", lambda *a, **kw: pytest.fail("recovery encoded a graph"))
        store.recover_authority()
        cut = store.replay_mutation(mutation_id)
    assert not cut.pending
    if boundary == "header":
        assert len(store.inspect_states(view_digest(creation(), CREATION_VIEW))) == 1
    elif boundary == "anchor":
        path = next(t["path"] for t in record.intent["transitions"] if t["path"].startswith("membership/"))
        digest = Path(path).stem
        assert store.read_node(digest).object_id == identity
    else:
        assert store.read_node_name("old") is None
        assert store.read_node_name("new") == NodeNameRecord("new", "recipe", named.root_digest)


def test_replay_refuses_to_overwrite_later_name_writer(tmp_path, monkeypatch):
    import dryml.filesystem as fs
    store = DirStore(tmp_path / "store", query_index="none")
    roots = [store.publish_node(creation(i)) for i in (1, 2)]
    first, second = (NodeNameRecord("main", "recipe", r.digest) for r in roots)
    original = fs.publish_file
    def fail(source, target, *, replace=False):
        result = original(source, target, replace=replace)
        if str(target).endswith("names/main.record"):
            raise OSError("name visible but incomplete call")
        return result
    with monkeypatch.context() as patch:
        patch.setattr(fs, "publish_file", fail)
        with pytest.raises(StoreMutationError) as raised:
            store.compare_and_set_node_name(first, expected=None)
    store.compare_and_set_node_name(second, expected=first)
    before = store.authority_cut()
    with pytest.raises(StoreCutChanged):
        store.replay_mutation(raised.value.mutation_id)
    assert store.read_node_name("main") == second and store.authority_cut() == before


def test_prepared_batch_seam_leaves_exact_projection_pending_for_owner_ack(tmp_path):
    store = DirStore(tmp_path / "store", query_index="none")
    other = DirStore(store.base_dir, query_index="none")
    root = store.publish_node(creation())
    name = NodeNameRecord("main", "recipe", root.digest)
    with store._metadata_read() as captured:
        assert store.read_node(root.digest) is not None
        change = store._prepared_file(store._name_path("main"), name, mutable=True)
        facts = captured.batch([change], ("names",))
    with pytest.raises(StoreAuthorityError, match="different Store owner"):
        other.apply_prepared_mutation(facts)
    identity, cut = store.apply_prepared_mutation(facts)
    pending, = store.pending_mutations()
    assert pending.intent["mutation_id"] == identity
    assert pending.intent["epoch"] == cut.epoch
    from dryml.core.store.records import _decode_authority_bytes
    row, = pending.intent["projection"]
    assert NodeNameRecord.from_bytes(_decode_authority_bytes(row["value"])) == name
    assert store.acknowledge_mutation(identity, cut.epoch).pending == ()
    with pytest.raises(StoreCutChanged):
        store.apply_prepared_mutation(facts)


@pytest.mark.parametrize("damage", ["missing-authority", "authority-version", "intent-version", "missing-intent"])
def test_incompatible_or_missing_journal_authority_is_never_reinitialized(tmp_path, monkeypatch, damage):
    import dryml.core.store.authority as authority
    from dryml.core.store.records import AuthorityRecord
    from dryml.formats.envelope import encode_metadata_frame
    store = DirStore(tmp_path / "store", query_index="none")
    def fail(*args):
        raise OSError("retain pending")
    with monkeypatch.context() as patch:
        patch.setattr(authority, "acknowledge", fail)
        with pytest.raises(StoreMutationError):
            store.install_creation(creation())
    if damage.endswith("authority") or damage == "authority-version":
        path = Path(store.base_dir, "authority.record")
        record_type = AuthorityRecord
    else:
        path, = Path(store.base_dir, "mutations").glob("*.record")
        record_type = MutationIntentRecord
    if damage.startswith("missing"):
        path.unlink()
    else:
        data = record_type.from_bytes(path.read_bytes()).to_data()
        data["version"] = 2
        path.write_bytes(encode_metadata_frame(record_type.schema, 1, data))
    before = {str(p): p.read_bytes() for p in Path(store.base_dir).rglob("*") if p.is_file()}
    with pytest.raises(StoreAuthorityError):
        reopened = DirStore(store.base_dir, query_index="none")
        reopened.recover_authority()
    assert {str(p): p.read_bytes() for p in Path(store.base_dir).rglob("*") if p.is_file()} == before


def test_zip_ambiguous_archive_barrier_keeps_retry_identity_and_dirty_extraction(tmp_path, monkeypatch):
    import dryml.filesystem as fs
    from dryml.core.store.zip import ZipStoreCommitError
    store = ZipStore(tmp_path / "store.zip")
    try:
        install(store)
        store.commit()
        committed = store.committed_archive_cut
        install(store, b"new")
        original = fs.publish_file
        def fail(source, target, *, replace=False):
            result = original(source, target, replace=replace)
            if str(target) == store.archive_path:
                raise OSError("archive visible, barrier uncertain")
            return result
        with monkeypatch.context() as patch:
            patch.setattr(fs, "publish_file", fail)
            with pytest.raises(ZipStoreCommitError, match="barrier uncertain") as raised:
                store.commit()
        assert store._archive_dirty and store.committed_archive_cut == committed
        assert raised.value.committed_archive_cut == committed
        assert raised.value.durability == "buffered"
        store.commit()
        assert not store._archive_dirty
        assert store.committed_archive_cut == store.authority_cut()
    finally:
        store.close()


def test_failed_intent_keeps_complete_staging_retryable_without_capture(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store", query_index="none")
    stage = store.create_local_state_staging()
    Path(stage, "data", "value").write_bytes(b"prepared once")
    original = store._atomic_write
    def fail(path, data):
        if str(path).endswith("mutation-prepare.record"):
            raise OSError("intent unavailable")
        return original(path, data)
    def publish():
        return store.install_state(creation(), stage, codec="Test", codec_version=1, interpretation={})
    with monkeypatch.context() as patch:
        patch.setattr(store, "_atomic_write", fail)
        with pytest.raises(StoreMutationError):
            publish()
    assert Path(stage, "data", "value").read_bytes() == b"prepared once"
    assert {p.name for p in Path(stage).iterdir()} == {"data", "manifest.record", "availability.record"}
    result = publish()
    assert result.installed
    assert Path(store.state_data_path(result.state.key), "value").read_bytes() == b"prepared once"


def test_no_effect_reinstallation_failure_does_not_advance_epoch(tmp_path, monkeypatch):
    import dryml.filesystem as fs
    store = DirStore(tmp_path / "store", query_index="none")
    state = install(store).state
    Path(store.state_data_path(state.key), "value").unlink()
    Path(store.state_data_path(state.key)).rmdir()
    before = store.authority_cut()
    original = fs.publish_file
    def fail(source, target, *, replace=False):
        result = original(source, target, replace=replace)
        if str(target).endswith("mutation-prepare.record"):
            raise OSError("before reinstallation effects")
        return result
    with monkeypatch.context() as patch:
        patch.setattr(fs, "publish_file", fail)
        with pytest.raises(StoreMutationError):
            install(store)
    store.recover_authority()
    assert store.authority_cut().epoch == before.epoch
    assert not store.inspect_states(state.key.cdef_digest)[0].available


def test_cancellation_keeps_confirmed_publication_and_recovery_attributes(tmp_path, monkeypatch):
    import dryml.filesystem as fs
    store = DirStore(tmp_path / "store", query_index="none")
    original = fs.publish_file
    def interrupt(source, target, *, replace=False):
        result = original(source, target, replace=replace)
        if str(target).endswith("def.pkl"):
            raise KeyboardInterrupt("header visible")
        return result
    with monkeypatch.context() as patch:
        patch.setattr(fs, "publish_file", interrupt)
        with pytest.raises(KeyboardInterrupt) as raised:
            install(store)
    assert raised.value.mutation_id in store.authority_cut().pending
    assert any(path.endswith("def.pkl") for path in raised.value.confirmed_refs)
    assert raised.value.index_disposition == "repair_pending"
    store.replay_mutation(raised.value.mutation_id)
    assert len(store.inspect_states(view_digest(creation(), CREATION_VIEW))) == 1


def test_post_ack_header_error_reports_actual_cut_and_retained_exact_intent(tmp_path, monkeypatch):
    import dryml.filesystem as fs
    store = DirStore(tmp_path / "store", query_index="none")
    original = fs.publish_file
    def fail(source, target, *, replace=False):
        result = original(source, target, replace=replace)
        if Path(target).name == "authority.record":
            cut = store.authority_cut()
            if cut.epoch == 1 and not cut.pending:
                raise OSError("ack header visible but call failed")
        return result
    with monkeypatch.context() as patch:
        patch.setattr(fs, "publish_file", fail)
        with pytest.raises(StoreMutationError) as raised:
            store.install_creation(creation())
    assert raised.value.cut == store.authority_cut()
    assert raised.value.confirmed_refs
    records = intent_records(store)
    assert len(records) == 1 and records[0].intent["mutation_id"] == raised.value.mutation_id
    assert records[0].intent["status"] == "acknowledged"


def test_prepared_adapter_cannot_overwrite_unregistered_intent_bridge(tmp_path, monkeypatch):
    import dryml.filesystem as fs
    store = DirStore(tmp_path / "store", query_index="none")
    root = store.publish_node(creation())
    original = fs.publish_file
    def fail(source, target, *, replace=False):
        result = original(source, target, replace=replace)
        if Path(target).name == "mutation-prepare.record":
            raise OSError("intent durable, registration interrupted")
        return result
    with monkeypatch.context() as patch:
        patch.setattr(fs, "publish_file", fail)
        with pytest.raises(StoreMutationError):
            store.install_creation(creation(2))
    bridge = Path(store.base_dir, "mutation-prepare.record")
    before = bridge.read_bytes()
    name = NodeNameRecord("main", "recipe", root.digest)
    with store._metadata_read() as captured:
        store.read_node(root.digest)
        facts = captured.batch([store._prepared_file(store._name_path("main"), name, mutable=True)], ("names",))
    with pytest.raises(StoreCutChanged, match="registration"):
        store.apply_prepared_mutation(facts)
    assert bridge.read_bytes() == before
    assert store.read_node_name("main") is None
    records = store.recover_authority()
    assert records and records[-1].intent["status"] == "classified"
