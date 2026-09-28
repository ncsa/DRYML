"""Focused current-history authority contracts for Store-selected Repo saves."""

import multiprocessing
from threading import Barrier, Thread

import pytest

from dryml.core import Repo, Serializable
from dryml.core.repo import RepoSaveError
from dryml.core.store.dir import DirStore
from dryml.core.store.records import StateAliasRecord
from dryml.core.store.store import StoreAliasConflictError, StoreAuthorityError


class HistoryValue(Serializable):
    """Small stateful value used to distinguish immutable history snapshots."""

    def __init__(self, value):
        self.value = value

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        from pathlib import Path

        Path(dest_dir, "value.txt").write_text(str(self.value), encoding="ascii")

    def restore_state_from_dir_imp(self, src_dir, *, codec):
        from pathlib import Path

        self.value = int(Path(src_dir, "value.txt").read_text(encoding="ascii"))


def _publish_current_state(store_path, initial, value, ready, start, results):
    """Publish one competing update from an isolated process-local reservation scope."""

    repo = Repo(DirStore(store_path, query_index="memory"))
    try:
        live = repo.load_state_ref(initial, reuse_live="never", cache="none")
        live.value = value
        ready.set()
        if not start.wait(timeout=30):
            results.put(("error", "start timeout"))
            return
        state = repo.save_object_if_current(
            live, alias="history", expected=initial, store=repo.default_store,
        )
        results.put(("success", state.digest()))
    except StoreAliasConflictError:
        results.put(("conflict", None))
    except BaseException as error:
        results.put(("error", repr(error)))
    finally:
        repo.close(flush=False)


def test_store_state_alias_cas_requires_expected_current_target(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    value = HistoryValue(1, repo=repo)
    first = repo.save_object(value, deep_capture=True)
    value.value = 2
    second = repo.save_object(value, deep_capture=True)
    record = StateAliasRecord("history", first.object, first.digest())

    assert store.compare_and_set_state_alias(record, expected_state_ref_digest=None) == record
    with pytest.raises(StoreAliasConflictError, match="expected"):
        store.compare_and_set_state_alias(
            StateAliasRecord("history", second.object, second.digest()),
            expected_state_ref_digest=None,
        )
    assert store.read_state_alias(first.object.digest(), "history") == record


def test_store_state_alias_cas_rejects_missing_or_wrong_scope_targets(tmp_path):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    first = repo.save_object(HistoryValue(1, repo=repo), deep_capture=True)
    second = repo.save_object(HistoryValue(2, repo=repo), deep_capture=True)

    with pytest.raises(StoreAuthorityError, match="same-Store"):
        store.compare_and_set_state_alias(
            StateAliasRecord("history", first.object, "f" * 64),
            expected_state_ref_digest=None,
        )
    with pytest.raises(StoreAuthorityError, match="same-Store"):
        store.compare_and_set_state_alias(
            StateAliasRecord("history", first.object, second.digest()),
            expected_state_ref_digest=None,
        )


def test_store_selected_history_declaration_is_serialized(tmp_path):
    store_path = tmp_path / "store"
    first = Repo(DirStore(store_path))
    second = Repo(DirStore(store_path))
    definition = HistoryValue(1).definition
    barrier = Barrier(2)
    results = []
    failures = []

    def declare(repo):
        try:
            barrier.wait(timeout=10)
            results.append(repo.get_or_declare_object_ref(definition, store=repo.default_store))
        except BaseException as error:
            failures.append(error)

    threads = [Thread(target=declare, args=(repo,)) for repo in (first, second)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not failures
    assert all(not thread.is_alive() for thread in threads)
    assert results[0] == results[1]
    assert len(first.default_store.iter_declaration_records()) == 1


def test_stale_history_writer_conflicts_and_retains_immutable_orphan(tmp_path):
    store_path = tmp_path / "store"
    owner = Repo(DirStore(store_path, query_index="memory"))
    initial = owner.save_object(HistoryValue(0, repo=owner), deep_capture=True)
    owner.set_state_alias("history", initial, store=owner.default_store)
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    results = context.Queue()
    ready = [context.Event(), context.Event()]
    processes = [
        context.Process(
            target=_publish_current_state,
            args=(str(store_path), initial, value, ready[index], start, results),
        )
        for index, value in enumerate((1, 2))
    ]
    for process in processes:
        process.start()
    try:
        for event in ready:
            assert event.wait(timeout=30)
        start.set()
        for process in processes:
            process.join(timeout=30)
            assert process.exitcode == 0
        outcomes = [results.get(timeout=1) for _ in processes]
    finally:
        start.set()
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)

    assert sorted(status for status, _ in outcomes) == ["conflict", "success"], outcomes
    winner_digest = next(digest for status, digest in outcomes if status == "success")
    assert owner.resolve_state_alias(initial.object, "history", store=owner.default_store).digest() == winner_digest
    assert len(owner.default_store.iter_state_ref_records()) == 3


def test_history_coordination_requires_one_writable_physical_store(tmp_path):
    first = DirStore(tmp_path / "first")
    second = DirStore(tmp_path / "second")
    repo = Repo([first, second])

    with pytest.raises(RepoSaveError, match="exactly one writable physical"):
        repo.get_or_declare_object_ref(HistoryValue(1).definition)


def test_current_save_reconciles_a_post_replacement_acknowledgement_error(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)
    original = store.compare_and_set_state_alias

    def replace_then_fail(record, *, expected_state_ref_digest):
        original(record, expected_state_ref_digest=expected_state_ref_digest)
        raise OSError("acknowledgement lost after replacement")

    monkeypatch.setattr(store, "compare_and_set_state_alias", replace_then_fail)
    state = repo.save_object_if_current(
        HistoryValue(1, repo=repo), alias="history", expected=None, store=store,
    )

    assert repo.resolve_state_alias(state.object, "history", store=store) == state


def test_current_save_does_not_mask_interruption(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store")
    repo = Repo(store)

    def interrupt(record, *, expected_state_ref_digest):
        raise KeyboardInterrupt

    monkeypatch.setattr(store, "compare_and_set_state_alias", interrupt)

    with pytest.raises(KeyboardInterrupt):
        repo.save_object_if_current(
            HistoryValue(1, repo=repo), alias="history", expected=None, store=store,
        )
