"""Store-owned schema-10 projection completion, independent of Repo consumers."""

from pathlib import Path
from contextlib import contextmanager
import asyncio
import json
import multiprocessing
import sqlite3
from threading import Event, Thread

import pytest

from dryml.core import Definition, Object, ObjectId, Repo
from dryml.core.node_prepare import NodePreparation
from dryml.core.query.model import QueryIndexDirty, QueryIndexIncompatible, QueryIndexUnavailable, QueryStats
from dryml.core.query.sqlite import SQLiteQueryIndexConfig
from dryml.core.query.sqlite.schema import STORE_PROJECTION_FAMILIES
from dryml.core.store import authority
from dryml.core.store.authority import StoreMutationError, PreparedChange
from dryml.core.store.dir import DirStore
from dryml.core.store.records import NodeNameRecord, DefinitionRecord, MutationIntentRecord
from dryml.core.store.store import StoreAuthorityError, StoreCapabilityError
from dryml.core.store.zip import ZipStore


class IndexedObject(Object):
    def __init__(self, value=1):
        self.value = value


def creation(value=1):
    return Definition(IndexedObject, value=value).concretize()


def indexed_store(path):
    return DirStore(path, query_index=SQLiteQueryIndexConfig(journal_mode="delete"))


def install(store, payload=b"state"):
    stage = store.create_local_state_staging()
    Path(stage, "data", "value").write_bytes(payload)
    return store.install_state(creation(), stage, codec="Test", codec_version=1, interpretation={})


def values(index, family):
    return tuple(json.loads(row[2]) for row in index.projection_rows(family, refresh=False))


def authority_bytes(store):
    return {"store-format.record": Path(store.store_format_path).read_bytes(), **{
            str(p.relative_to(store.base_dir)): p.read_bytes()
            for area in ("states", "definitions", "graphs", "membership", "declarations", "names", "mutations")
            for p in Path(store.base_dir, area).rglob("*") if p.is_file() and "data" not in p.parts}}


def test_publication_is_visible_without_query_refresh(tmp_path):
    store = indexed_store(tmp_path / "store")
    root = store.publish_node(creation())
    name = NodeNameRecord("main", "recipe", root.digest)
    store.compare_and_set_node_name(name, expected=None)
    index = store.open_query_index()
    index.refresh(False)
    assert index.status().schema_version == 10
    assert index.status().state == "ready"
    assert not store.authority_cut().pending
    assert len(index.projection_rows("recipe_root", refresh=False)) == 1
    assert len(index.projection_rows("name", refresh=False)) == 1
    assert index.projection_rows("saved_root", refresh=False) == ()
    assert not tuple(Path(store.dryml_dir).glob("query-index.dirty*"))


def test_coalesced_prepared_batch_is_one_transaction_and_exact_ack(tmp_path, monkeypatch):
    store = indexed_store(tmp_path / "store")
    root = store.publish_node(creation())
    index = store.open_query_index()
    names = [NodeNameRecord(n, "recipe", root.digest) for n in ("a", "b")]
    with store._metadata_read() as captured:
        facts = captured.batch([store._prepared_file(store._name_path(n.name), n, mutable=True)
                                for n in names], ("names",))
    calls, original = [], index._run_write_transaction
    def write(operation):
        calls.append(True)
        return original(operation)
    monkeypatch.setattr(index, "_run_write_transaction", write)
    before = store.authority_cut()
    identity, cut = store.apply_prepared_mutation(facts)
    assert len(calls) == 1 and cut.epoch == before.epoch + 1 and not cut.pending
    assert {v["name"] for v in values(index, "name")} == {"a", "b"}
    assert store._read_file(authority._path(store, identity), MutationIntentRecord).intent["status"] == "acknowledged"


def test_noop_has_zero_authority_sql_and_dirty_churn(tmp_path):
    store = indexed_store(tmp_path / "store")
    root = store.publish_node(creation())
    name = NodeNameRecord("main", "recipe", root.digest)
    store.compare_and_set_node_name(name, expected=None)
    index = store.open_query_index()
    con = index._connections.connection(readonly=False)
    cut, generation, count = store.authority_cut(), index.current_generation(), con.total_changes
    before, mtime = authority_bytes(store), index.path.stat().st_mtime_ns
    store.publish_node(creation())
    store.compare_and_set_node_name(name, expected=name)
    store.rename_node_name("main", "main", expected=name)
    assert store.authority_cut() == cut and index.current_generation() == generation
    assert con.total_changes == count and index.path.stat().st_mtime_ns == mtime
    assert authority_bytes(store) == before and not store.query_index_is_dirty()


def test_root_membership_state_and_names_remain_separate_after_rebuild(tmp_path, monkeypatch):
    store = indexed_store(tmp_path / "store")
    dependency = store.install_creation(creation(2))
    one, two = install(store), install(store, b"other")
    identity = ObjectId()
    recipe = store.publish_node(creation().with_identity(identity))
    saved = [store.publish_node(creation().with_identity(identity).with_state(state.state), kind="saved")
             for state in (one, two)]
    index = store.open_query_index()
    assert len(values(index, "creation")) == 2
    assert len(values(index, "recipe_root")) == 1 and len(values(index, "saved_root")) == 2
    assert len(values(index, "manifest")) == len(values(index, "availability")) == 2
    assert len(values(index, "declaration")) == 1
    before = authority_bytes(store)
    monkeypatch.setattr(store, "_authenticate_state_data", lambda *a: pytest.fail("read payload"))
    index.rebuild()
    assert authority_bytes(store) == before
    assert {v["digest"] for v in values(index, "recipe_root")} == {recipe.digest}
    assert {v["digest"] for v in values(index, "saved_root")} == {r.digest for r in saved}
    assert dependency not in {v["digest"] for v in values(index, "recipe_root")}
    assert store.withdraw_node(saved[0].digest, kind="saved")
    assert len(values(index, "saved_root")) == 1
    assert len(values(index, "manifest")) == 2
    assert len(store.read_declaration_anchor(identity).sources) == 2
    assert index.validate(thorough=True).ok


@pytest.mark.parametrize("damage", ["missing", "corrupt", "schema9", "incomplete"])
def test_missing_stale_or_owned_incompatible_sidecar_rebuilds_metadata_only(tmp_path, monkeypatch, capsys, damage):
    store = indexed_store(tmp_path / "store")
    root = store.publish_node(creation())
    index = store.open_query_index()
    before = authority_bytes(store)
    index.close()
    if damage == "missing":
        index.path.unlink()
    elif damage == "corrupt":
        index.path.write_bytes(b"broken owned derived file")
    else:
        with sqlite3.connect(index.path) as con:
            if damage == "schema9":
                con.execute("PRAGMA user_version=9")
            else:
                con.execute("DELETE FROM store_projection_coverage WHERE family='availability'")
    assert index.status().state != "ready"
    index.refresh(False)
    with pytest.raises((QueryIndexDirty, QueryIndexIncompatible, sqlite3.Error)):
        index.projection_rows("recipe_root", refresh=False)
    monkeypatch.setattr(store, "_authenticate_state_data", lambda *a: pytest.fail("payload IO"))
    store.publish_node(creation())  # Authoritative no-op still owns index recovery.
    assert index.status().state == "ready"
    assert len(values(index, "recipe_root")) == 1
    assert authority_bytes(store) == before
    assert "rebuild" in capsys.readouterr().err


@pytest.mark.parametrize("damage", ["foreign", "future", "future-coverage-on-old-schema", "future-semantic-on-old-schema"])
def test_foreign_future_sidecar_refused_intact_before_effects(tmp_path, damage):
    store = indexed_store(tmp_path / "store")
    store.publish_node(creation())
    index = store.open_query_index()
    index.close()
    with sqlite3.connect(index.path) as con:
        if damage == "foreign":
            con.execute("UPDATE catalog_state SET store_key='other-store'")
        elif damage == "future":
            con.execute("PRAGMA user_version=11")
        else:
            con.execute("PRAGMA user_version=9")
            con.execute("UPDATE store_projection_cut SET contract_version=2" if damage == "future-coverage-on-old-schema"
                        else "UPDATE catalog_state SET canonical_version=canonical_version+1")
    before, data, cut = authority_bytes(store), index.path.read_bytes(), store.authority_cut()
    for action in (lambda: store.publish_node(creation(2)), index.rebuild):
        with pytest.raises(QueryIndexIncompatible):
            action()
    assert index.path.read_bytes() == data and authority_bytes(store) == before
    assert store.authority_cut() == cut


@pytest.mark.parametrize("phase", ["pre-sql", "post-sql", "pre-ack"])
def test_index_phase_failure_preserves_publication_and_repairs_without_capture(tmp_path, monkeypatch, phase):
    store = indexed_store(tmp_path / "store")
    store.publish_node(creation())
    index = store.open_query_index()
    original_write = index._run_write_transaction
    def fail(*args, **kwargs):
        raise OSError("injected " + phase)
    with monkeypatch.context() as patch:
        if phase == "pre-sql":
            patch.setattr(index, "_run_write_transaction", fail)
        elif phase == "post-sql":
            def commit(operation):
                result = original_write(operation)
                patch.setattr(index, "_confirm_store_projection", fail)
                return result
            patch.setattr(index, "_run_write_transaction", commit)
        else:
            patch.setattr(authority, "acknowledge", fail)
        with pytest.raises(StoreMutationError) as raised:
            store.publish_node(creation(2))
    error = raised.value
    assert error.confirmed_refs and error.mutation_id in store.authority_cut().pending
    assert error.cut.epoch == 2 and error.index_disposition == "repair_pending"
    assert error.recovery_attempt == "journal-replay-or-metadata-rebuild"
    with pytest.raises(QueryIndexDirty):
        values(index, "recipe_root")
    before = authority_bytes(store)
    monkeypatch.setattr(store, "_prepare_node_roots", lambda *a: pytest.fail("application capture"))
    monkeypatch.setattr(store, "_authenticate_state_data", lambda *a: pytest.fail("payload IO"))
    index.maintain_store()
    assert not store.authority_cut().pending and len(values(index, "recipe_root")) == 2
    after = authority_bytes(store)
    assert {k: v for k, v in after.items() if not k.startswith("mutations/")} == {
        k: v for k, v in before.items() if not k.startswith("mutations/")}


def test_ambiguous_sql_commit_requires_receipt_reread_before_ack(tmp_path, monkeypatch):
    store = indexed_store(tmp_path / "store")
    store.publish_node(creation())
    index, calls = store.open_query_index(), []
    original_write, original_ack = index._run_write_transaction, authority.acknowledge
    def commit(operation):
        result = original_write(operation)
        calls.append("committed")
        raise OSError("lost commit response")
    def ack(*args):
        assert calls == ["committed"]
        calls.append("ack")
        return original_ack(*args)
    monkeypatch.setattr(index, "_run_write_transaction", commit)
    monkeypatch.setattr(authority, "acknowledge", ack)
    store.publish_node(creation(2))
    assert calls == ["committed", "ack"] and len(values(index, "recipe_root")) == 2


def test_clean_incremental_work_never_inventories_unrelated_store_growth(tmp_path, monkeypatch):
    store = indexed_store(tmp_path / "store")
    store.publish_node(creation())
    index = store.open_query_index()
    counts, original = [], index._write_store_projection
    def write(con, prepared, **kwargs):
        counts.append(len(prepared.sources))
        return original(con, prepared, **kwargs)
    monkeypatch.setattr(index, "_write_store_projection", write)
    monkeypatch.setattr(authority._DetachedRead, "directory", lambda *a: pytest.fail("incremental inventory"))
    for number in range(2, 8):
        store.publish_node(creation(number))
    assert counts == [3] * 6  # block, compact root, independent membership
    assert len(values(index, "recipe_root")) == 7


@pytest.mark.parametrize("family", STORE_PROJECTION_FAMILIES)
def test_closed_coverage_families_are_guarded_or_explicitly_unavailable(tmp_path, family):
    store = indexed_store(tmp_path / "store")
    store.publish_node(creation())
    index = store.open_query_index()
    if family in {"requested_metadata", "lifecycle"}:
        with pytest.raises(QueryIndexUnavailable):
            values(index, family)
    else:
        values(index, family)
        with sqlite3.connect(index.path) as con:
            con.execute("DELETE FROM store_projection_coverage WHERE family=?", (family,))
        assert index.status().state == "dirty"
        with pytest.raises(QueryIndexDirty):
            values(index, "recipe_root")


def test_unmanaged_posting_change_trips_guard_and_rebuild_restores_it(tmp_path):
    store = indexed_store(tmp_path / "store")
    store.publish_node(creation())
    index = store.open_query_index()
    assert values(index, "facet")
    with sqlite3.connect(index.path) as con:
        con.execute("DELETE FROM store_projection_rows WHERE family='facet'")
    assert index.status().state == "dirty"
    with pytest.raises(QueryIndexDirty):
        values(index, "recipe_root")
    index.maintain_store()
    assert values(index, "facet") and index.validate(thorough=True).ok


def test_destination_conflict_is_persistent_and_precedes_authority_effects(tmp_path):
    store = indexed_store(tmp_path / "store")
    store.publish_node(creation())
    before, cut = authority_bytes(store), store.authority_cut()
    other = DirStore(store.base_dir, query_index=SQLiteQueryIndexConfig(path=tmp_path / "other.sqlite"))
    with pytest.raises(QueryIndexIncompatible, match="Conflicting"):
        other.publish_node(creation(2))
    assert authority_bytes(store) == before and store.authority_cut() == cut
    assert not (tmp_path / "other.sqlite").exists()


def test_cached_backend_rechecks_domain_binding_after_derived_config_loss(tmp_path):
    store = indexed_store(tmp_path / "store")
    store.publish_node(creation())
    index = store.open_query_index()
    Path(store.dryml_dir, "query-index-destination.record").unlink()
    other = DirStore(store.base_dir, query_index=SQLiteQueryIndexConfig(
        path=tmp_path / "new.sqlite", journal_mode="delete"))
    other.open_query_index()
    cut = store.authority_cut()
    with pytest.raises(QueryIndexIncompatible, match="Conflicting"):
        store.publish_node(creation(2))
    with pytest.raises(QueryIndexIncompatible, match="Conflicting"):
        index.maintain_store()
    assert store.authority_cut() == cut


@pytest.mark.parametrize("policy", ["none", "memory"])
def test_nonpersistent_policies_create_no_sidecar(tmp_path, policy):
    store = DirStore(tmp_path / "store", query_index=policy)
    store.publish_node(creation())
    assert store.open_query_index() is None and not Path(store.dryml_dir).exists()


def test_optional_unavailable_fallback_and_explicit_failure(tmp_path, monkeypatch):
    def unavailable():
        raise QueryIndexUnavailable("optional sqlite missing")
    monkeypatch.setattr("dryml.core.query.sqlite.require_sqlite", unavailable)
    automatic = DirStore(tmp_path / "auto")
    automatic.publish_node(creation())
    assert automatic.open_query_index() is None
    assert automatic.mutation_index_disposition == "nonpersistent:auto-backend-unavailable"
    explicit = DirStore(tmp_path / "explicit", query_index="sqlite")
    before = explicit.authority_cut()
    with pytest.raises(QueryIndexUnavailable):
        explicit.publish_node(creation())
    assert explicit.authority_cut() == before and not Path(explicit.base_dir, "mutations").exists()


def test_graph_work_never_runs_under_store_fence(tmp_path, monkeypatch):
    store = indexed_store(tmp_path / "store")
    held = [0]
    original_lock, prepare = store.writer_lock, NodePreparation.__init__
    @contextmanager
    def lock():
        with original_lock():
            held[0] += 1
            try:
                yield
            finally:
                held[0] -= 1
    def guarded(*args, **kwargs):
        assert held[0] == 0
        return prepare(*args, **kwargs)
    monkeypatch.setattr(store, "writer_lock", lock)
    monkeypatch.setattr(NodePreparation, "__init__", guarded)
    state = install(store)
    store.publish_node(creation().with_identity(ObjectId()).with_state(state.state), kind="saved")
    store.open_query_index().rebuild()


def test_legacy_consumers_refuse_without_inventory_or_authority_effects(tmp_path):
    store = indexed_store(tmp_path / "store")
    before = store.authority_cut()
    for action in (store.query, lambda: store.write_definition_record(DefinitionRecord(creation()))):
        with pytest.raises(StoreCapabilityError, match="not integrated|legacy Store consumer"):
            action()
    assert store.authority_cut() == before and not Path(store.base_dir, "mutations").exists()
    saved = Repo(store).save_object(IndexedObject())
    assert saved.state is None
    assert len(store.open_query_index().projection_rows("saved_root", refresh=False)) == 1
    assert not store.authority_cut().pending


def test_zip_projection_policy_stays_buffered_nonpersistent_until_archive_commit(tmp_path):
    archive = tmp_path / "store.zip"
    store = ZipStore(archive)
    try:
        receipt = install(store)
        store.publish_node(creation())
        assert receipt.durability == "buffered" and store.open_query_index() is None
        assert not archive.exists() and store.committed_archive_cut is None
        store.commit()
        assert store.committed_archive_cut == store.authority_cut()
    finally:
        store.close()


def test_old_detached_repair_cannot_ack_or_overwrite_later_writer(tmp_path, monkeypatch):
    import dryml.core.query.authority as projection
    store = indexed_store(tmp_path / "store")
    store.publish_node(creation())
    index = store.open_query_index()
    captured, release = Event(), Event()
    errors, old_records = [], []
    capture, ack = projection.capture_store_projection, authority.acknowledge
    def detached(*args, **kwargs):
        prepared = capture(*args, **kwargs)
        if __import__("threading").current_thread().name == "old-writer":
            old_records.extend(prepared.receipts)
            captured.set()
            assert release.wait(15)
        return prepared
    def fail_later(store, identity, epoch):
        if epoch == 3:
            raise OSError("later writer ack pending")
        return ack(store, identity, epoch)
    def older():
        try:
            store.publish_node(creation(2))
        except BaseException as error:
            errors.append(error)
        finally:
            index._connections.close_current()
    monkeypatch.setattr(projection, "capture_store_projection", detached)
    monkeypatch.setattr(authority, "acknowledge", fail_later)
    worker = Thread(target=older, name="old-writer")
    worker.start()
    try:
        assert captured.wait(10)
        with pytest.raises(StoreMutationError) as raised:
            store.publish_node(creation(3))
        later_id = raised.value.mutation_id
        assert store.authority_cut().pending == (later_id,)
        release.set()
        worker.join(15)
        assert len(errors) == 1 and isinstance(errors[0], StoreMutationError)
        assert errors[0].mutation_id == old_records[0].intent["mutation_id"]
        assert store.authority_cut().pending == (later_id,)
    finally:
        release.set()
        worker.join(15)
    monkeypatch.setattr(authority, "acknowledge", ack)
    index.maintain_store()
    assert len(values(index, "recipe_root")) == 3 and not store.authority_cut().pending


def test_public_ack_refuses_unincorporated_receipt(tmp_path, monkeypatch):
    store = indexed_store(tmp_path / "store")
    store.publish_node(creation())
    index = store.open_query_index()
    with monkeypatch.context() as patch:
        patch.setattr(index, "_run_write_transaction", lambda *a: (_ for _ in ()).throw(OSError("SQL unavailable")))
        with pytest.raises(StoreMutationError) as raised:
            store.publish_node(creation(2))
    error = raised.value
    with pytest.raises(StoreMutationError):
        store.acknowledge_mutation(error.mutation_id, error.cut.epoch)
    assert error.mutation_id in store.authority_cut().pending
    index.maintain_store()
    assert not store.authority_cut().pending


def test_async_tasks_never_share_connection_or_reader_lease(tmp_path):
    store = indexed_store(tmp_path / "store")
    store.publish_node(creation())
    manager = store.open_query_index()._connections
    async def scenario():
        entered, done = asyncio.Event(), asyncio.Event()
        connections = []
        async def first():
            with manager._read_lease() as (con, _):
                connections.append(con)
                con.execute("BEGIN")
                entered.set()
                await done.wait()
                assert con.in_transaction
                con.execute("ROLLBACK")
        async def second():
            await entered.wait()
            with manager._read_lease() as (con, _):
                assert not con.in_transaction
                connections.append(con)
                con.execute("BEGIN")
                assert len(manager._leases) == 2
                con.execute("ROLLBACK")
            done.set()
        await asyncio.gather(first(), second())
        assert connections[0] is not connections[1]
    asyncio.run(scenario())
    assert not manager._leases and all(not c.in_transaction for c in manager._connections.values())
    store.close()


def _spawn_indexed_writer(path, ready, release, result, value):
    store = indexed_store(path)
    ready.put(value)
    assert release.wait(15)
    try:
        install(store, bytes([value]))
        result.put((store.store_uuid, store.authority_cut().epoch))
    except BaseException as error:
        result.put(repr(error))
    finally:
        store.close()


def test_spawned_sibling_writers_preserve_complete_index_cut_and_full_keys(tmp_path):
    store = indexed_store(tmp_path / "store")
    store.install_creation(creation())
    context = multiprocessing.get_context("spawn")
    ready, result, release = context.Queue(), context.Queue(), context.Event()
    workers = [context.Process(target=_spawn_indexed_writer, args=(store.base_dir, ready, release, result, value))
               for value in (1, 2)]
    for worker in workers:
        worker.start()
    try:
        assert {ready.get(timeout=20) for _ in workers} == {1, 2}
        release.set()
        for worker in workers:
            worker.join(20)
            assert worker.exitcode == 0
        outcomes = [result.get(timeout=10) for _ in workers]
        # A competing detached cut may explicitly report retryable instability,
        # but complete successes must always leave a certified combined cut.
        assert all(isinstance(item, tuple) or "StoreMutationError" in item for item in outcomes)
        store.open_query_index().maintain_store()
        assert len(values(store.open_query_index(), "manifest")) == 2
        assert len(values(store.open_query_index(), "availability")) == 2
        assert not store.authority_cut().pending
    finally:
        release.set()
        for worker in workers:
            worker.join(5)
            if worker.is_alive():
                worker.terminate()
                worker.join(5)


def test_new_coverage_version_and_missing_guards_never_advertise_ready(tmp_path):
    store = indexed_store(tmp_path / "store")
    store.publish_node(creation())
    index = store.open_query_index()
    with sqlite3.connect(index.path) as con:
        con.execute("DROP TRIGGER store_projection_delete")
    assert index.status().state == "dirty"
    index.maintain_store()
    assert index.status().state == "ready"
    with sqlite3.connect(index.path) as con:
        con.execute("UPDATE store_projection_cut SET contract_version=2")
    before = index.path.read_bytes()
    with pytest.raises(QueryIndexIncompatible, match="Future"):
        store.publish_node(creation(2))
    assert index.path.read_bytes() == before


def test_shared_creation_block_decodes_once_and_retains_compatibility_prefilters(tmp_path, monkeypatch):
    import dryml.core.node_codec as codec
    from dryml.core.node_views import COMPATIBILITY_VIEW, view_digest
    store = indexed_store(tmp_path / "store")
    calls, decode = [], codec._decode_node_graph_views
    def shared(packet):
        calls.append(len(packet["views"]))
        return decode(packet)
    monkeypatch.setattr(codec, "_decode_node_graph_views", shared)
    roots = (creation(), creation(2))
    store.install_creations(roots)
    assert calls == [2]
    rows = values(store.open_query_index(), "creation")
    assert {row["compatibility_digest"] for row in rows} == {
        view_digest(root, COMPATIBILITY_VIEW) for root in roots}
    assert values(store.open_query_index(), "recipe_root") == ()


def test_typed_postings_preserve_roles_paths_and_overlay_correlation(tmp_path):
    from dryml.core.links import DefLink
    from dryml.core.cdef_graph import EdgeKind
    store = indexed_store(tmp_path / "store")
    root = Definition(IndexedObject, value={"left": creation(),
        "right": DefLink(EdgeKind.REF, creation(2))}).concretize()
    store.publish_node(root)
    index = store.open_query_index()
    edges, nodes = values(index, "edge"), values(index, "node")
    assert {edge["role"] for edge in edges} >= {"materialize", "ref"}
    assert all(edge["path"] for edge in edges)
    slots = {(node["table"], node["slot"]) for node in nodes}
    assert all((edge["table"], edge["owner"]) in slots and
               (edge["table"], edge["target"]) in slots for edge in edges)
    assert len({json.dumps(edge["path"]) for edge in edges}) >= 2


def test_sql_rollback_and_cancellation_never_advance_projection_coverage(tmp_path, monkeypatch):
    store = indexed_store(tmp_path / "store")
    store.publish_node(creation())
    index = store.open_query_index()
    original = index._write_store_projection
    def interrupted(con, prepared, **kwargs):
        original(con, prepared, **kwargs)
        raise KeyboardInterrupt("SQL transaction interrupted")
    with monkeypatch.context() as patch:
        patch.setattr(index, "_write_store_projection", interrupted)
        with pytest.raises(KeyboardInterrupt) as raised:
            store.publish_node(creation(2))
    assert raised.value.mutation_id in store.authority_cut().pending
    assert raised.value.confirmed_refs and raised.value.index_disposition == "repair_pending"
    assert all(not con.in_transaction for con in index._connections._connections.values())
    with sqlite3.connect(index.path) as con:
        assert con.execute("SELECT epoch FROM store_projection_cut").fetchone() == (1,)
        assert con.execute("SELECT COUNT(*) FROM store_projection_rows WHERE family='recipe_root'").fetchone() == (1,)
    index.maintain_store()
    assert len(values(index, "recipe_root")) == 2


def test_detectable_required_sqlite_failure_precedes_first_authority_effect(tmp_path, monkeypatch):
    from dryml.core.query.sqlite.connection import SQLiteConnectionManager
    store = indexed_store(tmp_path / "store")
    before = store.authority_cut()
    def unavailable(*args, **kwargs):
        raise OSError("required SQLite connection unavailable")
    monkeypatch.setattr(SQLiteConnectionManager, "_initialize_connection", unavailable)
    with pytest.raises(OSError, match="required SQLite"):
        store.publish_node(creation())
    assert store.authority_cut() == before
    assert not Path(store.base_dir, "mutations").exists()


def test_disabled_writer_makes_old_enabled_cut_stale_and_rebuild_is_visible(tmp_path, capsys):
    store = indexed_store(tmp_path / "store")
    store.publish_node(creation())
    index = store.open_query_index()
    disabled = DirStore(store.base_dir, query_index="none")
    disabled.publish_node(creation(2))
    assert index.status().state == "dirty"
    with pytest.raises(QueryIndexDirty):
        values(index, "recipe_root")
    before = authority_bytes(store)
    stats = QueryStats()
    index.refresh("auto", stats=stats)
    assert stats.store_scan_count == 1 and len(values(index, "recipe_root")) == 2
    assert authority_bytes(store) == before and "rebuild" in capsys.readouterr().err


def test_authority_classification_is_independent_of_derived_destination(tmp_path, monkeypatch):
    store = indexed_store(tmp_path / "store")
    store.publish_node(creation())
    index = store.open_query_index()
    with monkeypatch.context() as patch:
        patch.setattr(authority, "acknowledge", lambda *a: (_ for _ in ()).throw(OSError("ack pending")))
        with pytest.raises(StoreMutationError):
            store.publish_node(creation(2))
    with sqlite3.connect(index.path) as con:
        con.execute("PRAGMA user_version=11")
    records = store.recover_authority()
    assert records and all(record.intent["status"] == "classified" for record in records)
    with pytest.raises(QueryIndexIncompatible):
        index.maintain_store()


def test_prepared_batch_with_no_net_effect_has_zero_journal_or_sql_churn(tmp_path):
    store = indexed_store(tmp_path / "store")
    root = store.publish_node(creation())
    name = NodeNameRecord("main", "recipe", root.digest)
    store.compare_and_set_node_name(name, expected=None)
    index = store.open_query_index()
    cut, before, generation = store.authority_cut(), authority_bytes(store), index.current_generation()
    path = "names/main.record"
    with store._metadata_read() as captured:
        raw = authority.raw(store, path)
        facts = captured.batch([PreparedChange(path, raw, None), PreparedChange(path, None, raw)], ("names",))
    identity, result = store.apply_prepared_mutation(facts)
    assert identity is None and result == cut
    assert authority_bytes(store) == before and index.current_generation() == generation


def test_epoch_gap_between_replay_decision_and_detached_capture_cannot_certify_missing_rows(tmp_path, monkeypatch):
    import dryml.core.query.authority as projection
    store = indexed_store(tmp_path / "store")
    store.publish_node(creation())
    index = store.open_query_index()
    with monkeypatch.context() as patch:
        patch.setattr(index, "_run_write_transaction", lambda *a: (_ for _ in ()).throw(OSError("SQL unavailable")))
        with pytest.raises(StoreMutationError):
            store.publish_node(creation(2))
    capture, injected = projection.capture_store_projection, []
    def advance(*args, **kwargs):
        if not injected:
            injected.append(True)
            DirStore(store.base_dir, query_index="none").publish_node(creation(3))
        return capture(*args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(projection, "capture_store_projection", advance)
        with pytest.raises(QueryIndexDirty):
            index.maintain_store()
    assert store.authority_cut().epoch == 3 and index.status().state == "dirty"
    with sqlite3.connect(index.path) as con:
        assert con.execute("SELECT epoch FROM store_projection_cut").fetchone() == (1,)
    index.maintain_store()
    assert len(values(index, "recipe_root")) == 3


def test_reconcile_reports_visible_delta_even_when_exact_ack_fails(tmp_path, monkeypatch):
    store = indexed_store(tmp_path / "store")
    store.publish_node(creation())
    index = store.open_query_index()
    with monkeypatch.context() as patch:
        patch.setattr(index, "_run_write_transaction", lambda *a: (_ for _ in ()).throw(OSError("SQL unavailable")))
        with pytest.raises(StoreMutationError):
            store.publish_node(creation(2))
    with monkeypatch.context() as patch:
        patch.setattr(authority, "acknowledge", lambda *a: (_ for _ in ()).throw(OSError("ack unavailable")))
        report = index.reconcile()
    assert report.changed and not report.validated and report.issues
    assert report.generation_after > report.generation_before
    assert store.authority_cut().pending
    index.maintain_store()
    assert len(values(index, "recipe_root")) == 2


@pytest.mark.parametrize("missing", ["availability", "retired-anchor"])
def test_rebuild_refuses_incomplete_authority_and_invalidates_old_coverage(tmp_path, missing):
    store = indexed_store(tmp_path / "store")
    if missing == "availability":
        state = install(store).state
        path = Path(store.state_data_path(state.key)).parent / "availability.record"
    else:
        identity = ObjectId()
        root = store.publish_node(creation().with_identity(identity))
        store.withdraw_node(root.digest)
        path = Path(store._node_path("declarations/anchors", store._anchor_key(identity)))
    index = store.open_query_index()
    path.unlink()
    before, old_index = authority_bytes(store), index.path.read_bytes()
    with pytest.raises(StoreAuthorityError, match="Missing required projection authority"):
        index.rebuild()
    assert index.path.read_bytes() == old_index and authority_bytes(store) == before
    assert index.status().state == "dirty"
    with pytest.raises(QueryIndexDirty):
        values(index, "recipe_root")
