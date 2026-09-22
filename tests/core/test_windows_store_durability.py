"""Windows Store durability policy with a mocked native move boundary."""

import builtins
import os
from pathlib import Path
import zipfile

import pytest

from dryml.core import Repo, Serializable
from dryml.core.store import _windows_durability
from dryml.core.store.dir import DirStore, _REMOVED_ENTRY_PREFIX
from dryml.core.store.records import DefinitionRecord
from dryml.core.store.store import StoreCapabilityError
from dryml.core.store.zip import ZipStore


class WindowsDurabilityPayload(Serializable):
    """Small stateful object used across direct and archive publication tests."""

    def __init__(self, value="value"):
        self.value = value

    def save_state_to_dir_imp(self, directory, *, codec):
        """Write one deterministic local-state payload file."""

        Path(directory, f"{_REMOVED_ENTRY_PREFIX}payload").write_text(
            self.value, encoding="ascii",
        )


def _emulate_windows_moves(monkeypatch):
    """Select Windows policy while implementing its native boundary on POSIX."""

    moves = []
    replace_existing = _windows_durability._MOVEFILE_REPLACE_EXISTING
    original_replace = os.replace
    original_rename = os.rename

    def move(source, destination, flags):
        moves.append((Path(source), Path(destination), flags))
        if flags & replace_existing:
            original_replace(source, destination)
        else:
            if os.path.lexists(destination):
                raise FileExistsError(destination)
            original_rename(source, destination)

    monkeypatch.setattr(_windows_durability, "_IS_WINDOWS", True)
    monkeypatch.setattr(_windows_durability, "_move_file_ex", move)
    return moves


def test_native_write_through_move_selects_exact_flags_and_propagates_failure(
        tmp_path, monkeypatch):
    """The adapter never requests cross-volume fallback or hides native errors."""

    observed = []
    monkeypatch.setattr(
        _windows_durability,
        "_move_file_ex",
        lambda source, destination, flags: observed.append(
            (source, destination, flags),
        ),
    )
    source = tmp_path / "source"
    destination = tmp_path / "destination"

    _windows_durability.move_file_write_through(
        source, destination, replace=False,
    )
    _windows_durability.move_file_write_through(
        source, destination, replace=True,
    )

    write_through = _windows_durability._MOVEFILE_WRITE_THROUGH
    replace_existing = _windows_durability._MOVEFILE_REPLACE_EXISTING
    assert observed == [
        (str(source), str(destination), write_through),
        (str(source), str(destination), write_through | replace_existing),
    ]

    def fail(*_args):
        raise PermissionError("native sharing denial")

    monkeypatch.setattr(_windows_durability, "_move_file_ex", fail)
    with pytest.raises(PermissionError, match="sharing denial"):
        _windows_durability.move_file_write_through(
            source, destination, replace=True,
        )


def test_native_adapter_uses_documented_extended_length_paths():
    """Drive, UNC, and already-prefixed names retain exact Win32 meaning."""

    convert = _windows_durability._extended_length_path
    assert convert(r"C:\store\record") == r"\\?\C:\store\record"
    assert convert(r"\\server\share\record") == (
        r"\\?\UNC\server\share\record"
    )
    assert convert(r"\\?\C:\store\record") == r"\\?\C:\store\record"


def test_windows_initialization_and_mkdir_use_write_through_directory_moves(
        tmp_path, monkeypatch):
    """Every new authoritative directory component is installed durably."""

    moves = _emulate_windows_moves(monkeypatch)
    root = tmp_path / "source" / "store"
    store = DirStore(root, query_index="none")
    store._makedirs_durable(os.fspath(root / "one" / "two"))

    write_through = _windows_durability._MOVEFILE_WRITE_THROUGH
    replace_existing = _windows_durability._MOVEFILE_REPLACE_EXISTING
    destinations = {destination: flags for _source, destination, flags in moves}
    for directory in (root.parent, root, root / "one", root / "one" / "two"):
        assert destinations[directory] == write_through
    assert destinations[Path(store.store_format_path)] == (
        write_through | replace_existing
    )
    assert all(flags & write_through for _source, _destination, flags in moves)
    with pytest.raises(StoreCapabilityError, match="Windows directory handles"):
        store._fsync_directory(store.base_dir)


def test_windows_mkdir_propagates_failure_after_destination_becomes_visible(
        tmp_path, monkeypatch):
    """A visible destination does not turn an uncertain barrier into success."""

    store = DirStore(tmp_path / "store", query_index="none")
    destination = Path(store.base_dir, "published-directory")
    original_rename = os.rename

    def move_then_fail(source, target, _flags):
        original_rename(source, target)
        raise OSError("write-through completion uncertain")

    monkeypatch.setattr(_windows_durability, "_IS_WINDOWS", True)
    monkeypatch.setattr(_windows_durability, "_move_file_ex", move_then_fail)
    with pytest.raises(OSError, match="completion uncertain"):
        store._makedirs_durable(os.fspath(destination))

    assert destination.is_dir()


def test_windows_snapshot_and_metadata_publication_preserve_order_and_handle_modes(
        tmp_path, monkeypatch):
    """Dirty authority precedes writes and staged files use writable handles."""

    store = DirStore(tmp_path / "store", query_index="sqlite")
    repo = Repo(store)
    moves = _emulate_windows_moves(monkeypatch)
    opened = []
    original_open = os.open

    def observe_open(path, flags, *args, **kwargs):
        opened.append((Path(path), flags))
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", observe_open)
    state = repo.save_object(WindowsDurabilityPayload(repo=repo))

    snapshot_index = next(
        index for index, (source, _destination, _flags) in enumerate(moves)
        if source.name.startswith("snapshot-")
    )
    dirty_indexes = [
        index for index, (_source, destination, _flags) in enumerate(moves)
        if destination.name.startswith("query-index.dirty.")
    ]
    dirty_before_snapshot = [
        index for index in dirty_indexes if index < snapshot_index
    ]
    assert dirty_before_snapshot
    assert not (
        moves[snapshot_index][2]
        & _windows_durability._MOVEFILE_REPLACE_EXISTING
    )
    staged_opens = [
        flags for path, flags in opened
        if any(part.startswith("snapshot-") for part in path.parts)
    ]
    assert staged_opens
    access_mode_mask = os.O_WRONLY | os.O_RDWR
    assert all(flags & access_mode_mask == os.O_RDWR for flags in staged_opens)

    moves.clear()
    repo.set_metadata(state.object, {"version": 1})
    destinations = [destination for _source, destination, _flags in moves]
    dirty_index = next(
        index for index, destination in enumerate(destinations)
        if destination.name.startswith("query-index.dirty.")
    )
    metadata_index = next(
        index for index, destination in enumerate(destinations)
        if "metadata" in destination.parts and destination.suffix == ".record"
    )
    assert dirty_index < metadata_index
    assert store.read_metadata(state.object) == {"version": 1}

    moves.clear()
    store.clear_query_index_dirty()
    assert not store.query_index_is_dirty()
    assert moves
    assert all(
        destination.name.startswith(_REMOVED_ENTRY_PREFIX)
        for _source, destination, _flags in moves
    )


def test_windows_logical_delete_retains_only_ignored_tombstone(
        tmp_path, monkeypatch):
    """Cleanup failure cannot restore a deleted authoritative metadata name."""

    store = DirStore(tmp_path / "store", query_index="none")
    target = WindowsDurabilityPayload().object_ref
    store.write_metadata(target, {"present": True})
    moves = _emulate_windows_moves(monkeypatch)
    original_unlink = os.unlink

    def retain_tombstone(path, *args, **kwargs):
        if Path(path).name.startswith(_REMOVED_ENTRY_PREFIX):
            raise PermissionError("retain tombstone")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", retain_tombstone)
    assert store.delete_metadata(target)

    metadata_parent = Path(store._metadata_path(target)).parent
    tombstones = tuple(metadata_parent.glob(f"{_REMOVED_ENTRY_PREFIX}*"))
    assert store.read_metadata(target) is None
    assert len(tombstones) == 1
    assert not tuple(metadata_parent.glob("*.record"))
    assert moves[-1][1] == tombstones[0]
    assert not (moves[-1][2] & _windows_durability._MOVEFILE_REPLACE_EXISTING)


def test_windows_zip_commit_flushes_writable_file_and_filters_tombstones(
        tmp_path, monkeypatch):
    """Archive replacement is write-through and excludes cleanup residue."""

    archive_path = tmp_path / "store.zip"
    store = ZipStore(archive_path)
    repo = Repo(store)
    target = repo.save_object(WindowsDurabilityPayload(repo=repo)).object
    store.write_metadata(target, {"present": True})
    moves = _emulate_windows_moves(monkeypatch)
    original_unlink = os.unlink

    with monkeypatch.context() as patch:
        def retain_tombstone(path, *args, **kwargs):
            if Path(path).name.startswith(_REMOVED_ENTRY_PREFIX):
                raise PermissionError("retain tombstone")
            return original_unlink(path, *args, **kwargs)

        patch.setattr(os, "unlink", retain_tombstone)
        assert store.delete_metadata(target)

    opened_modes = []

    def observe_open(path, mode="r", *args, **kwargs):
        opened_modes.append((Path(path), mode))
        return builtins.open(path, mode, *args, **kwargs)

    monkeypatch.setattr("dryml.core.store.zip.open", observe_open, raising=False)
    moves.clear()
    try:
        store.commit()
        archive_move = next(
            item for item in moves if item[1] == archive_path
        )
        expected_flags = (
            _windows_durability._MOVEFILE_WRITE_THROUGH
            | _windows_durability._MOVEFILE_REPLACE_EXISTING
        )
        assert archive_move[2] == expected_flags
        assert any(path.suffix == ".zip" and mode == "r+b" for path, mode in opened_modes)
        with zipfile.ZipFile(archive_path) as archive:
            names = archive.namelist()
            assert any(
                Path(name).name == f"{_REMOVED_ENTRY_PREFIX}payload"
                for name in names
            )
            assert all(
                not (
                    Path(name).name.startswith(_REMOVED_ENTRY_PREFIX)
                    and Path(name).parts[0] in {".dryml", "metadata"}
                )
                for name in names
            )
    finally:
        store.close()


@pytest.mark.skipif(
    not _windows_durability.is_windows(),
    reason="requires native Windows MoveFileExW and file sharing semantics",
)
def test_native_windows_dir_and_zip_store_round_trip(tmp_path):
    """Exercise direct overwrite/delete/reopen and buffered archive commit."""

    direct = DirStore(tmp_path / "nested" / "direct", query_index="sqlite")
    repo = Repo(direct)
    state = repo.save_object(WindowsDurabilityPayload(repo=repo))
    repo.set_metadata(state.object, {"version": 1})
    repo.set_metadata(state.object, {"version": 2})
    assert repo.get_metadata(state.object) == {"version": 2}

    reopened = DirStore(direct.base_dir, query_index="sqlite")
    reopened_repo = Repo(reopened)
    assert reopened.read_state_ref_record(state.digest()).state_ref == state
    assert reopened_repo.get_metadata(state.object) == {"version": 2}
    assert reopened_repo.delete_metadata(state.object)
    assert DirStore(
        direct.base_dir, query_index="none",
    ).read_metadata(state.object) is None

    archive_path = tmp_path / "store.zip"
    archive = ZipStore(archive_path)
    archive_record = DefinitionRecord(WindowsDurabilityPayload("zip").definition)
    archive.write_definition_record(archive_record)
    archive.commit()
    archive.close()
    committed = ZipStore.open_existing(archive_path)
    try:
        assert committed.read_definition_record(archive_record.digest) == archive_record
    finally:
        committed.close()
