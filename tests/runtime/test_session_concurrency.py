import asyncio
import threading

import pytest

from dryml.runtime.errors import RuntimeTransitionError
from dryml.runtime.publication import PublicationService, SessionGeneration, publication


def test_writer_reentry_fails_before_recursive_state_acquisition():
    entered = threading.Event()
    release = threading.Event()
    failure = []

    def writer():
        with publication.writer():
            entered.set()
            try:
                publication.snapshot()
            except RuntimeTransitionError as exc:
                failure.append(exc)
            release.wait(timeout=2)

    thread = threading.Thread(target=writer)
    thread.start()
    assert entered.wait(timeout=2)
    with pytest.raises(RuntimeTransitionError, match="import-busy"):
        with publication.writer():
            pass
    release.set()
    thread.join(timeout=2)

    assert not thread.is_alive()
    assert len(failure) == 1


def test_async_tasks_project_the_same_process_generation_without_context_override():
    expected = publication.current()

    async def observe():
        await asyncio.sleep(0)
        return publication.current()

    async def collect():
        return await asyncio.gather(observe(), observe())

    observed = asyncio.run(collect())

    assert observed == [expected, expected]


def test_readers_overlap_while_writer_is_exclusive_and_reader_waits_only_for_writer():
    service = PublicationService()
    service.initialize(object())
    first_entered = threading.Event()
    second_entered = threading.Event()
    release_readers = threading.Event()

    def hold_reader(entered):
        with service.reader():
            entered.set()
            assert release_readers.wait(timeout=2)

    first = threading.Thread(target=hold_reader, args=(first_entered,))
    second = threading.Thread(target=hold_reader, args=(second_entered,))
    first.start()
    assert first_entered.wait(timeout=2)
    second.start()
    assert second_entered.wait(timeout=2)

    before = service.current()
    candidate = service.stage(before, SessionGeneration(before.number + 1, before.runtime))
    with pytest.raises(RuntimeTransitionError, match="import-busy"):
        service.commit(candidate)

    release_readers.set()
    first.join(timeout=2)
    second.join(timeout=2)
    assert not first.is_alive()
    assert not second.is_alive()

    writer_entered = threading.Event()
    release_writer = threading.Event()
    reader_entered = threading.Event()

    def hold_writer():
        with service.writer():
            writer_entered.set()
            assert release_writer.wait(timeout=2)

    def wait_for_writer():
        with service.reader():
            reader_entered.set()

    writer = threading.Thread(target=hold_writer)
    writer.start()
    assert writer_entered.wait(timeout=2)
    reader = threading.Thread(target=wait_for_writer)
    reader.start()
    assert not reader_entered.wait(timeout=0.1)
    release_writer.set()
    assert reader_entered.wait(timeout=2)
    writer.join(timeout=2)
    reader.join(timeout=2)
    assert not writer.is_alive()
    assert not reader.is_alive()


def test_writer_owner_cannot_reenter_reader_snapshot_or_lease():
    with publication.writer():
        with pytest.raises(RuntimeTransitionError, match="re-entry"):
            publication.snapshot()
        with pytest.raises(RuntimeTransitionError, match="re-entry"):
            with publication.lease():
                pass
