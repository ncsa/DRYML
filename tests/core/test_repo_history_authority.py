"""Current history uses published Node membership and exact Node-name CAS."""

import multiprocessing
from pathlib import Path

import pytest

from dryml.core import Repo, Serializable
from dryml.core.node_prepare import NodePreparation
from dryml.core.store.dir import DirStore
from dryml.core.store.records import NodeNameRecord
from dryml.core.store.store import StoreAliasConflictError, StoreAuthorityError

pytestmark = pytest.mark.usefixtures("fixed_snapshot_environment")


class HistoryValue(Serializable):
    """Persist one integer to distinguish immutable history snapshots."""

    def __init__(self, value):
        self.value = value

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        from pathlib import Path
        Path(dest_dir, "value.txt").write_text(str(self.value), encoding="ascii")

    def restore_state_from_dir_imp(self, src_dir, *, codec):
        from pathlib import Path
        self.value = int(Path(src_dir, "value.txt").read_text(encoding="ascii"))


def _name(saved, name="history"):
    return NodeNameRecord(name, "saved", NodePreparation((saved,)).digest(saved, domain="dryml-saved-root/v1"))


def _publish_current_state(store_path, initial, value, ready, start, results):
    """Publish competing immutable branches before one exact current-name CAS."""
    repo = Repo(DirStore.open_existing(store_path, query_index="memory"))
    try:
        live = repo.load_state_ref(initial, reuse_live="never", cache="none")
        live.value = value
        saved = repo.save_object(live)
        ready.set()
        if not start.wait(timeout=60):
            raise RuntimeError("start barrier timeout")
        try:
            repo.default_store.compare_and_set_node_name(_name(saved), expected=_name(initial))
        except StoreAliasConflictError:
            results.put(("conflict", _name(saved).root_digest))
        else:
            results.put(("success", _name(saved).root_digest))
    except BaseException as error:
        ready.set()
        results.put(("error", repr(error)))
    finally:
        repo.close(flush=False)


def test_store_state_alias_cas_requires_expected_current_target(tmp_path):
    """A current Node name advances only from its exact expected predecessor."""
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    live = HistoryValue(1)
    first = repo.save_object(live)
    live.value = 2
    second = repo.save_object(live)
    record = _name(first)
    store.compare_and_set_node_name(record, expected=None)
    with pytest.raises(StoreAliasConflictError):
        store.compare_and_set_node_name(_name(second), expected=None)
    assert store.read_node_name("history") == record
    store.compare_and_set_node_name(_name(second), expected=record)
    assert store.read_node_name("history") == _name(second)


def test_store_state_alias_cas_rejects_missing_or_wrong_scope_targets(tmp_path):
    """Missing and unpublished dependency-only roots cannot become named history."""
    store = DirStore(tmp_path / "store")
    with pytest.raises(StoreAuthorityError):
        store.compare_and_set_node_name(NodeNameRecord("history", "saved", "f" * 64), expected=None)
    assert store.read_node_name("history") is None
    definition = HistoryValue(1).definition
    store.install_creations((definition,))
    digest = NodePreparation((definition,)).digest(definition, domain="dryml-recipe-root/v1")
    with pytest.raises(StoreAuthorityError):
        store.compare_and_set_node_name(NodeNameRecord("history", "recipe", digest), expected=None)
    assert store.read_node_name("history") is None


def test_stale_history_writer_conflicts_and_retains_immutable_orphan(tmp_path):
    """Controlled real writers keep both saved branches and exactly one current name."""
    store_path = tmp_path / "store"
    owner = Repo(DirStore(store_path, query_index="memory"))
    initial = owner.save_object(HistoryValue(0))
    owner.default_store.compare_and_set_node_name(_name(initial), expected=None)
    context = multiprocessing.get_context("spawn")
    start, results = context.Event(), context.Queue()
    ready = [context.Event(), context.Event()]
    processes = [context.Process(target=_publish_current_state,
        args=(str(store_path), initial, value, ready[i], start, results)) for i, value in enumerate((1, 2))]
    for process in processes:
        process.start()
    try:
        for event in ready:
            assert event.wait(timeout=60)
        start.set()
        for process in processes:
            process.join(timeout=90)
            assert process.exitcode == 0
        outcomes = [results.get(timeout=5) for _ in processes]
    finally:
        start.set()
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
    assert sorted(status for status, _ in outcomes) == ["conflict", "success"], outcomes
    winner = next(digest for status, digest in outcomes if status == "success")
    assert owner.default_store.read_node_name("history").root_digest == winner
    for _, digest in outcomes:
        assert owner.default_store.read_node(digest, kind="saved") is not None
    assert len(tuple(Path(store_path, "membership", "saved").glob("*/*.record"))) == 3


@pytest.mark.parametrize("boundary", ("before", "after"))
def test_current_save_does_not_mask_interruption(tmp_path, monkeypatch, boundary):
    """Cancellation exposes exact pending authority; repair never captures again."""
    store = DirStore(tmp_path / "store", query_index="sqlite")
    repo = Repo(store)
    saved = repo.save_object(HistoryValue(1))
    record = _name(saved)
    original = store._atomic_write

    def interrupt(relative, content, *args, **kwargs):
        selected = str(relative).endswith("/names/history.record")
        if selected and boundary == "before":
            raise KeyboardInterrupt("history cancellation")
        result = original(relative, content, *args, **kwargs)
        if selected and boundary == "after":
            raise KeyboardInterrupt("history cancellation")
        return result

    monkeypatch.setattr(store, "_atomic_write", interrupt)
    with pytest.raises(KeyboardInterrupt):
        store.compare_and_set_node_name(record, expected=None)
    monkeypatch.setattr(store, "_atomic_write", original)
    for mutation in store.authority_cut().pending:
        store.replay_mutation(mutation)
    assert store.read_node(_name(saved).root_digest, kind="saved").graph_equal(saved)


def test_current_name_index_failure_preserves_committed_authority_and_pending_evidence(tmp_path, monkeypatch):
    """R79: committed names are partial failures until enabled coverage is repaired."""
    store = DirStore(tmp_path / "store", query_index="sqlite")
    saved = Repo(store).save_object(HistoryValue(1))
    original = store._complete_index_mutation
    monkeypatch.setattr(store, "_complete_index_mutation",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("history index failed")))
    with pytest.raises(OSError, match="history index failed"):
        store.compare_and_set_node_name(_name(saved), expected=None)
    assert store.read_node_name("history") == _name(saved)
    assert store.authority_cut().pending
    monkeypatch.setattr(store, "_complete_index_mutation", original)
    store.open_query_index().maintain_store()
    assert not store.authority_cut().pending
    assert store.read_node_name("history") == _name(saved)


def test_retention_cleanup_preserves_saved_and_independent_recipe_history_on_rebuild(tmp_path, monkeypatch):
    """Rebuild preserves independent memberships and unavailable state history."""
    store = DirStore(tmp_path / "store", query_index="sqlite")
    repo = Repo(store)
    live = HistoryValue(4, repo=repo)
    recipe = store.publish_node(live.definition)
    saved, report = repo.save_object(live, report_stores=True)
    saved_key = NodePreparation((saved,)).digest(saved, domain="dryml-saved-root/v1")
    publication, = report.publications
    assert [item.status for item in repo.release(publication.retention_token)] == ["removed"]

    def forbidden(*args, **kwargs):
        pytest.fail("historical inspection/index rebuild opened a payload")
    monkeypatch.setattr(store, "read_state", forbidden)
    monkeypatch.setattr(store, "_authenticate_state_data", forbidden)
    index = store.open_query_index()
    index.close()
    index.path.unlink()
    index.maintain_store()
    assert store.read_node(saved_key, kind="saved").graph_equal(saved)
    assert store.read_node(recipe.digest, kind="recipe").graph_equal(live.definition)
    descriptor, = repo.inspect_states(saved.state.key.cdef_digest)
    assert not descriptor.descriptor.available
