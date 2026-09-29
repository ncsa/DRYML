"""Store-local CAS publication behavior for ExperimentData."""

import multiprocessing

import pytest

from dryml.core import Repo, Serializable
from dryml.core.store.store import StoreAuthorityError
from dryml.core.store.dir import DirStore
from dryml.models import ExperimentData


class ConcurrentSubject(Serializable):
    """Small persisted subject shared by independent history writers."""


def _append(store_path, checkpoint, key, ready, start, results):
    """Publish one independently loaded row after a deterministic release."""

    repo = Repo(DirStore(store_path, query_index="memory"))
    try:
        history = ExperimentData.find(checkpoint.object_projection(), repo=repo)
        history.add_row(
            expected_artifacts=(), state_ref=checkpoint, prev_state_ref=None,
            time=key, examples_seen=key, row_key=f"row-{key}",
        )
        ready.set()
        if not start.wait(timeout=30):
            results.put("start timeout")
            return
        history.publish(repo=repo)
        results.put(None)
    except BaseException as error:
        results.put(repr(error))
    finally:
        repo.close(flush=False)


def test_stale_publishers_reload_reapply_and_preserve_both_rows(tmp_path):
    """Concurrent stale appends retain both rows through bounded CAS retry."""

    store_path = tmp_path / "store"
    owner = Repo(DirStore(store_path, query_index="memory"))
    checkpoint = owner.save_object(ConcurrentSubject(repo=owner), deep_capture=True)
    ExperimentData.get_or_create(checkpoint.object_projection(), repo=owner)
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    ready = [context.Event(), context.Event()]
    results = context.Queue()
    writers = [
        context.Process(
            target=_append,
            args=(str(store_path), checkpoint, key, ready[index], start, results),
        )
        for index, key in enumerate((1, 2))
    ]
    for writer in writers:
        writer.start()
    try:
        for event in ready:
            assert event.wait(timeout=30)
        start.set()
        for writer in writers:
            writer.join(timeout=30)
            assert writer.exitcode == 0
        assert [results.get(timeout=1) for _ in writers] == [None, None]
    finally:
        start.set()
        for writer in writers:
            if writer.is_alive():
                writer.terminate()
                writer.join(timeout=5)
    final = ExperimentData.find(checkpoint.object_projection(), repo=owner)
    assert set(final.data["row_key"]) == {"row-1", "row-2"}
    owner.close(flush=False)


def test_initialization_recovers_one_empty_snapshot_but_not_populated_history(tmp_path):
    """Adopt only a unique empty snapshot left before initial alias publication."""

    repo = Repo(DirStore(tmp_path / "store", query_index="memory"))
    checkpoint = repo.save_object(ConcurrentSubject(repo=repo), deep_capture=True)
    subject = checkpoint.object_projection()
    cdef = ExperimentData._definition(subject)
    reference = repo.get_or_declare_object_ref(cdef, store=repo.default_store)
    empty = repo.build_object_ref(reference, store=repo.default_store)
    repo.save_object(empty, store=repo.default_store)

    recovered = ExperimentData.get_or_create(subject, repo=repo)
    assert recovered.data.empty

    repo = Repo(DirStore(tmp_path / "populated", query_index="memory"))
    checkpoint = repo.save_object(ConcurrentSubject(repo=repo), deep_capture=True)
    subject = checkpoint.object_projection()
    cdef = ExperimentData._definition(subject)
    reference = repo.get_or_declare_object_ref(cdef, store=repo.default_store)
    populated = repo.build_object_ref(reference, store=repo.default_store)
    populated.add_row(
        expected_artifacts=(), state_ref=checkpoint, prev_state_ref=None,
        time=0, examples_seen=0,
    )
    repo.save_object(populated, store=repo.default_store)
    with pytest.raises(StoreAuthorityError, match="empty"):
        ExperimentData.get_or_create(subject, repo=repo)
