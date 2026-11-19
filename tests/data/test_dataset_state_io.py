"""Detached continuation publication and inert reading proofs."""

import numpy as np
import multiprocessing
import threading
from pathlib import Path

import pytest

import dryml.data as data


class _LegacyOSError(OSError):
    add_note = None


class _LegacyPermissionError(PermissionError):
    add_note = None


class _LegacyKeyboardInterrupt(KeyboardInterrupt):
    add_note = None


class _LegacyFileExistsError(FileExistsError):
    add_note = None


def private_values(private_value):
    yield np.uint8(0)


def test_record_and_train_state_repr_hide_contents_with_exact_byte_roundtrip(tmp_path):
    from dryml.core.tensor_spec import TensorSpec
    from dryml.models.train_spec import TrainState

    metadata_secret = "synthetic-private-initialization-7c53"
    payload_secret = b"synthetic-private-snapshot-8d64"
    dataset = data.Chain(
        data.ArrayDataset(np.frombuffer(payload_secret, dtype=np.uint8)),
        data.GeneratorDataset(private_values, metadata_secret,
                              spec=TensorSpec("uint8", shape=(), backend="numpy")),
    )
    with dataset.open() as traversal:
        traversal.enable_replay_recovery()
        record = traversal.capture_state()
    assert metadata_secret.encode() in record._metadata
    assert any(payload_secret in payload for payload in record._payloads)
    loaded = data.DatasetState.load(record.save(tmp_path / "private-cut"))
    assert loaded._metadata == record._metadata
    assert loaded._payloads == record._payloads
    loaded.validate(dataset)
    for state in (record, loaded):
        for display in (repr(state), repr(TrainState(dataset_state=state))):
            assert metadata_secret not in display
            assert payload_secret.decode() not in display
            assert "_metadata=" not in display and "_payloads=" not in display
        with dataset.open(state=state) as resumed:
            assert list(resumed.next_epoch()) == list(payload_secret) + [0]


def test_validate_type_hints_resolve_with_type_checking_names():
    from typing import get_type_hints
    import dryml.data.state as state_module
    from dryml.data.dataset import Dataset
    from dryml.data.native import PreparedDataset

    namespace = dict(vars(state_module), Dataset=Dataset, PreparedDataset=PreparedDataset)
    hints = get_type_hints(data.DatasetState.validate, globalns=namespace)
    assert hints == {"target": Dataset[state_module.T] | PreparedDataset,
                     "return": type(None)}


def _competing_writer(record, path, barrier, outcomes):
    import dryml.data.state as state_module
    record = data.DatasetState.load(record)
    original = state_module._write
    def synchronized(path, raw):
        original(path, raw)
        if path.name == "manifest.json":
            barrier.wait(timeout=10)
    state_module._write = synchronized
    try:
        record.save(path)
        outcomes.put("saved")
    except FileExistsError:
        outcomes.put("conflict")
    except BaseException as error:
        outcomes.put(repr(error))
    finally:
        state_module._write = original


def _record():
    with data.ArrayDataset(np.arange(12)).open() as traversal:
        next(traversal.next_epoch())
        return traversal.capture_state()


def retained_payload_factory(selection):
    from dryml.core.repo import manage_repo
    with manage_repo() as repo:
        payload = repo.load_state_ref(selection, reuse_live="never")
        yield from payload.array


@pytest.mark.usefixtures("fixed_snapshot_environment")
def test_external_selected_state_requires_ordinary_retention_and_never_gains_a_root(tmp_path, monkeypatch):
    from dryml.core import Repo
    from dryml.core.store.dir import DirStore
    from dryml.numpy import NumpyArray
    from dryml.core.tensor_spec import TensorSpec
    from dryml.core.links import DefLink
    from dryml.core.cdef_graph import EdgeKind
    from dryml.core.repo import default_repo, RepoLoadError
    store = DirStore(tmp_path / "store", query_index="none")
    repo = Repo(store)
    payload = NumpyArray((6,), "int64", repo=repo)
    payload.array = np.arange(6)
    saved, report = repo.save_object(payload, report_stores=True)
    selected = DefLink.finalized(EdgeKind.REF, saved)
    dataset = data.GeneratorDataset(retained_payload_factory, selected,
                                   spec=TensorSpec("int64", shape=(), backend="numpy"), repo=repo)
    with default_repo(repo), dataset.open() as traversal:
        traversal.enable_replay_recovery()
        next(traversal.next_epoch())
        record = traversal.capture_state()
        record.save(tmp_path / "continuation")
    loaded_payload = repo.load_state_ref(saved, reuse_live="never")
    assert loaded_payload.last_state_ref is not None
    equivalent = data.GeneratorDataset(retained_payload_factory, selected, spec=dataset.spec, repo=repo)
    with default_repo(repo), equivalent.open(state=data.DatasetState.load(tmp_path / "continuation")) as traversal:
        assert list(traversal.next_epoch()) == list(range(1, 6))
    with monkeypatch.context() as patch:
        patch.setattr(repo, "load_state_ref", lambda *a, **k: pytest.fail("inert reading restored a source Object"))
        loaded = data.DatasetState.load(tmp_path / "continuation")
        loaded.validate(equivalent)
    # The standalone record is not a retention root: release the actual owner and
    # ordinary reclamation must be possible without deleting the continuation.
    store.retention.release(report.publications[0].retention_token, (store,))
    assert store.delete_state(saved.state.key).status == "removed"
    assert data.DatasetState.load(tmp_path / "continuation").position == record.position
    with pytest.raises(RepoLoadError):
        repo.load_state_ref(saved, reuse_live="never")
    loaded.validate(equivalent)  # metadata compatibility is not future availability
    with default_repo(repo), equivalent.open(state=loaded) as traversal:
        with pytest.raises(RepoLoadError):
            traversal.next_epoch()


def test_load_validate_are_source_inert(tmp_path, monkeypatch):
    dataset = data.ArrayDataset(np.arange(8))
    with dataset.open() as traversal:
        view = traversal.next_epoch()
        assert next(view) == 0
        record = traversal.capture_state()
        record.save(tmp_path / "continuation")
    monkeypatch.setattr(data.ArrayDataset, "__init__", lambda *a, **k: (_ for _ in ()).throw(AssertionError("constructed")))
    loaded = data.DatasetState.load(tmp_path / "continuation")
    loaded.validate(dataset)
    with dataset.open(state=loaded) as resumed:
        assert list(resumed.next_epoch()) == list(range(1, 8))


def test_no_clobber_and_corrupt_inventory_does_not_touch_prior_record(tmp_path):
    record = _record()
    destination = record.save(tmp_path / "cut")
    before = {p.name: p.read_bytes() for p in destination.iterdir()}
    with pytest.raises(FileExistsError):
        record.save(destination)
    assert {p.name: p.read_bytes() for p in destination.iterdir()} == before
    assert (tmp_path / ".cut.dataset-state.lock").is_file()
    assert not list(tmp_path.glob(".cut.staging-*"))


@pytest.mark.parametrize("mode", ["thread", "process"])
def test_barrier_controlled_cooperating_name_publication_has_one_winner(tmp_path, monkeypatch, mode):
    import dryml.data.state as state_module
    record = _record()
    destination = tmp_path / "same"
    if mode == "process":
        source = record.save(tmp_path / "source")
        context = multiprocessing.get_context("spawn")
        barrier, outcomes = context.Barrier(2), context.Queue()
        workers = [context.Process(target=_competing_writer, args=(str(source), destination, barrier, outcomes)) for _ in range(2)]
    else:
        import queue
        barrier, outcomes = threading.Barrier(2), queue.Queue()
        original = state_module._write
        def synchronized(path, raw):
            original(path, raw)
            if path.name == "manifest.json":
                barrier.wait(timeout=10)
        monkeypatch.setattr(state_module, "_write", synchronized)
        def writer():
            try:
                record.save(destination)
                outcomes.put("saved")
            except FileExistsError:
                outcomes.put("conflict")
            except BaseException as error:
                outcomes.put(repr(error))
        workers = [threading.Thread(target=writer) for _ in range(2)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=15)
        assert not worker.is_alive()
    assert sorted([outcomes.get(timeout=2), outcomes.get(timeout=2)]) == ["conflict", "saved"]
    loaded = data.DatasetState.load(destination)
    assert loaded._metadata == record._metadata and loaded._payloads == record._payloads
    assert not list(tmp_path.glob(".same.staging-*"))


@pytest.mark.parametrize("failure", [PermissionError("denied"), KeyboardInterrupt("interrupted")])
def test_prepublication_failures_preserve_prior_and_remove_only_own_staging(tmp_path, monkeypatch, failure):
    import dryml.data.state as state_module
    record = _record()
    prior = record.save(tmp_path / "prior")
    before = {p.name: p.read_bytes() for p in prior.iterdir()}
    def fail(*args):
        raise failure
    monkeypatch.setattr(state_module, "_write", fail)
    with pytest.raises(type(failure)) as caught:
        record.save(tmp_path / "new")
    assert caught.value is failure
    assert not (tmp_path / "new").exists()
    assert not list(tmp_path.glob(".new.staging-*"))
    assert {p.name: p.read_bytes() for p in prior.iterdir()} == before


@pytest.mark.parametrize("legacy_notes", [False, True])
def test_short_write_and_cleanup_failure_preserve_primary_and_prior(tmp_path, monkeypatch, legacy_notes):
    import dryml.data.state as state_module
    record = _record()
    prior = record.save(tmp_path / "prior")
    before = {p.name: p.read_bytes() for p in prior.iterdir()}
    if legacy_notes:
        monkeypatch.setattr(state_module, "OSError", _LegacyOSError, raising=False)
    original = Path.open
    class ShortWriter:
        def __init__(self, file):
            self.file = file
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.file.close()
        def write(self, raw):
            return self.file.write(raw[:-1])
    def open_file(path, mode="r", *args, **kwargs):
        file = original(path, mode, *args, **kwargs)
        return ShortWriter(file) if mode == "xb" else file
    monkeypatch.setattr(Path, "open", open_file)
    monkeypatch.setattr(state_module.shutil, "rmtree", lambda path: (_ for _ in ()).throw(OSError("cleanup denied")))
    with pytest.raises(OSError, match="Short DatasetState write") as caught:
        record.save(tmp_path / "new")
    assert any("cleanup denied" in n for n in caught.value.__notes__)
    assert not (tmp_path / "new").exists()
    assert data.DatasetState.load(prior).position == record.position
    assert {p.name: p.read_bytes() for p in prior.iterdir()} == before


def test_inventory_validation_before_rename_and_interruption_after_rename(tmp_path, monkeypatch):
    import dryml.data.state as state_module
    record = _record()
    original_write = state_module._write
    def incomplete(path, raw):
        original_write(path, raw)
        if path.name == "manifest.json":
            (path.parent / "000000.bin").unlink()
    monkeypatch.setattr(state_module, "_write", incomplete)
    with pytest.raises(data.ContinuationFormatError):
        record.save(tmp_path / "incomplete")
    assert not (tmp_path / "incomplete").exists()
    monkeypatch.setattr(state_module, "_write", original_write)
    original_rename = state_module.os.rename
    def interrupted(source, target):
        original_rename(source, target)
        raise KeyboardInterrupt("after commit")
    monkeypatch.setattr(state_module.os, "rename", interrupted)
    with pytest.raises(KeyboardInterrupt) as caught:
        record.save(tmp_path / "committed")
    assert any("may have committed" in n for n in caught.value.__notes__)
    assert data.DatasetState.load(tmp_path / "committed").position == record.position
    with pytest.raises(FileExistsError):
        record.save(tmp_path / "committed")


@pytest.mark.parametrize("legacy_notes", [False, True])
def test_name_lock_cleanup_failure_does_not_replace_existing_destination_error(tmp_path, monkeypatch, legacy_notes):
    import dryml.data.state as state_module
    record = _record()
    destination = record.save(tmp_path / "prior")
    before = {p.name: p.read_bytes() for p in destination.iterdir()}
    if legacy_notes:
        monkeypatch.setattr(state_module, "FileExistsError", _LegacyFileExistsError, raising=False)
    original = state_module.FileLock.release
    def failed_release(lock):
        original(lock)
        raise OSError("lock cleanup failed")
    monkeypatch.setattr(state_module.FileLock, "release", failed_release)
    with pytest.raises(FileExistsError) as caught:
        record.save(destination)
    assert any("lock cleanup failed" in note for note in caught.value.__notes__)
    assert data.DatasetState.load(destination).position == record.position
    assert {p.name: p.read_bytes() for p in destination.iterdir()} == before
    assert not list(tmp_path.glob(".prior.staging-*"))


@pytest.mark.parametrize("error_type", [PermissionError, _LegacyPermissionError,
                                       KeyboardInterrupt, _LegacyKeyboardInterrupt])
@pytest.mark.parametrize("committed", [False, True])
def test_rename_failure_preserves_primary_cleanup_order_and_exact_bytes(
        tmp_path, monkeypatch, error_type, committed):
    import dryml.data.state as state_module

    record = _record()
    prior = record.save(tmp_path / "prior")
    before = {p.name: p.read_bytes() for p in prior.iterdir()}
    primary = error_type("rename interrupted")
    primary.__notes__ = ["existing diagnostic"]
    events = []
    rename, release, rmtree = state_module.os.rename, state_module.FileLock.release, state_module.shutil.rmtree

    def interrupted(source, target):
        events.append("rename")
        if committed:
            rename(source, target)
        raise primary

    def released(lock):
        events.append("release")
        release(lock)

    def removed(path):
        events.append("cleanup")
        assert path.parent == tmp_path and path.name.startswith(".new.staging-")
        rmtree(path)

    monkeypatch.setattr(state_module.os, "rename", interrupted)
    monkeypatch.setattr(state_module.FileLock, "release", released)
    monkeypatch.setattr(state_module.shutil, "rmtree", removed)
    with pytest.raises(error_type) as caught:
        record.save(tmp_path / "new")
    assert caught.value is primary
    assert primary.__notes__ == ["existing diagnostic",
        "DatasetState publication may have committed around rename; inspect the destination before retrying."]
    assert events == (["rename", "release"] if committed else ["rename", "release", "cleanup"])
    assert not list(tmp_path.glob(".new.staging-*"))
    assert {p.name: p.read_bytes() for p in prior.iterdir()} == before
    if committed:
        loaded = data.DatasetState.load(tmp_path / "new")
        assert (loaded._metadata, loaded._payloads) == (record._metadata, record._payloads)
    else:
        assert not (tmp_path / "new").exists()
    lock = state_module.FileLock(tmp_path / ".new.dataset-state.lock")
    assert lock.acquire(blocking=False)
    lock.release()


def test_missing_directory_keeps_filesystem_failure_and_incomplete_directory_is_format_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.DatasetState.load(tmp_path / "missing")
    (tmp_path / "partial").mkdir()
    with pytest.raises(data.ContinuationFormatError):
        data.DatasetState.load(tmp_path / "partial")


@pytest.mark.parametrize("mutation", ["extra", "missing", "corrupt", "subdirectory"])
def test_load_rejects_incomplete_or_uninventoried_directories(tmp_path, mutation):
    record = _record()
    path = record.save(tmp_path / "cut")
    if mutation == "extra":
        (path / "extra.bin").write_bytes(b"unexpected")
    elif mutation == "missing":
        (path / "000000.bin").unlink()
    elif mutation == "corrupt":
        (path / "000000.bin").write_bytes(b"wrong bytes")
    else:
        (path / "000000.bin").unlink()
        (path / "000000.bin").mkdir()
    with pytest.raises(data.ContinuationFormatError):
        data.DatasetState.load(path)
