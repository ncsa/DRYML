"""Format-4 Store durability policy at an emulated Windows native boundary."""

import os
from pathlib import Path
import zipfile

import pytest

import dryml.filesystem as filesystem
from dryml.filesystem import _windows as windows_backend
from dryml.core import Definition, Object
from dryml.core.query.model import QueryIndexError
from dryml.core.query.sqlite import SQLiteQueryIndexConfig
from dryml.core.query.sqlite.index import SQLiteStoreQueryIndex
from dryml.core.store.dir import DirStore, _REMOVED_ENTRY_PREFIX
from dryml.core.store.records import NodeNameRecord
from dryml.core.store.zip import ZipStore


class WindowsDurabilityPayload(Object):
    def __init__(self, value="value"):
        self.value = value


def creation():
    return Definition(WindowsDurabilityPayload).concretize()


def install(store):
    stage = store.create_local_state_staging()
    Path(stage, "data", _REMOVED_ENTRY_PREFIX + "payload").write_bytes(b"payload")
    return store.install_state(creation(), stage, codec="Test", codec_version=1, interpretation={})


def _emulate_windows_moves(monkeypatch):
    moves = []
    original_replace, original_rename = os.replace, os.rename
    def move(source, destination, flags):
        moves.append((Path(source), Path(destination), flags))
        if flags & windows_backend._MOVEFILE_REPLACE_EXISTING:
            original_replace(source, destination)
        else:
            if os.path.lexists(destination):
                raise FileExistsError(destination)
            original_rename(source, destination)
    monkeypatch.setattr(filesystem, "_backend", windows_backend)
    monkeypatch.setattr(windows_backend, "_move_file_ex", move)
    return moves


def test_windows_initialization_and_mkdir_use_write_through_directory_moves(tmp_path, monkeypatch):
    moves = _emulate_windows_moves(monkeypatch)
    root = tmp_path / "source" / "store"
    store = DirStore(root, query_index="none")
    filesystem.ensure_directory(root / "one" / "two")
    write_through = windows_backend._MOVEFILE_WRITE_THROUGH
    replace_existing = windows_backend._MOVEFILE_REPLACE_EXISTING
    destinations = {destination: flags for _source, destination, flags in moves}
    for directory in (root.parent, root, root / "one", root / "one" / "two"):
        assert destinations[directory] == write_through
    for record in (store.store_format_path, root / "authority.record"):
        assert destinations[Path(record)] == write_through | replace_existing
    assert all(flags & write_through for _source, _destination, flags in moves)
    with pytest.raises(AssertionError, match="write-through"):
        windows_backend.sync_directory(store.base_dir)


def test_windows_mkdir_propagates_failure_after_destination_becomes_visible(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store", query_index="none")
    destination = Path(store.base_dir, "published-directory")
    original_rename = os.rename
    def move_then_fail(source, target, _flags):
        original_rename(source, target)
        raise OSError("write-through completion uncertain")
    monkeypatch.setattr(filesystem, "_backend", windows_backend)
    monkeypatch.setattr(windows_backend, "_move_file_ex", move_then_fail)
    with pytest.raises(OSError, match="completion uncertain"):
        filesystem.ensure_directory(destination)
    assert destination.is_dir()


def test_windows_intent_precedes_variant_and_staged_flush_uses_writable_handles(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store", query_index="none")
    moves = _emulate_windows_moves(monkeypatch)
    opened = []
    original = os.open
    def observe_open(path, flags, *args, **kwargs):
        opened.append((Path(path), flags))
        return original(path, flags, *args, **kwargs)
    monkeypatch.setattr(os, "open", observe_open)
    result = install(store)
    variant = Path(store.state_data_path(result.state.key)).parent
    variant_index = next(i for i, (_s, d, _f) in enumerate(moves) if d == variant)
    assert any(d.parent.name == "mutations" for _s, d, _f in moves[:variant_index])
    assert any(d.name == "authority.record" for _s, d, _f in moves[:variant_index])
    assert moves[variant_index][2] == windows_backend._MOVEFILE_WRITE_THROUGH
    staged = [flags for path, flags in opened if ".staging" in path.parts]
    assert staged and all(flags & (os.O_WRONLY | os.O_RDWR) == os.O_RDWR for flags in staged)
    assert store.authority_cut().epoch == 1 and store.authority_cut().pending == ()
    assert all(flags & windows_backend._MOVEFILE_WRITE_THROUGH for _s, _d, flags in moves)


def test_windows_standalone_sqlite_dirty_and_activation_paths_use_native_moves(tmp_path, monkeypatch):
    moves = _emulate_windows_moves(monkeypatch)
    path, dirty_path = tmp_path / "standalone.sqlite", tmp_path / "standalone.dirty"
    index = SQLiteStoreQueryIndex(source_key="windows-standalone", path=path,
                                 config=SQLiteQueryIndexConfig(path=path, journal_mode="delete"), dirty_path=dirty_path)
    index.initialize_empty()
    index._mark_dirty()
    captured = index._dirty_markers()
    assert len(captured) == 1
    index._clear_dirty(captured, roots=(), clear_unscoped=True)
    assert not index._is_dirty()
    replacement_path = index._replacement_path()
    replacement = SQLiteStoreQueryIndex(source_key=index.source_key, path=replacement_path,
                                       config=SQLiteQueryIndexConfig(path=replacement_path, journal_mode="delete"))
    replacement.initialize_empty(generation=7)
    replacement.close()
    index._activate_replacement(replacement_path, quarantine_existing=False)
    assert index.current_generation() == 7
    assert any(destination == path for _source, destination, _flags in moves)
    index.close()


def test_windows_standalone_sqlite_native_failures_remain_query_index_errors(tmp_path, monkeypatch):
    path, dirty_path = tmp_path / "failure.sqlite", tmp_path / "failure.dirty"
    index = SQLiteStoreQueryIndex(source_key="windows-failure", path=path,
                                 config=SQLiteQueryIndexConfig(path=path, journal_mode="delete"), dirty_path=dirty_path)
    index.initialize_empty()
    replacement_path = index._replacement_path()
    replacement = SQLiteStoreQueryIndex(source_key=index.source_key, path=replacement_path,
                                       config=SQLiteQueryIndexConfig(path=replacement_path, journal_mode="delete"))
    replacement.initialize_empty(generation=9)
    replacement.close()
    marker = dirty_path.with_name(dirty_path.name + ".captured")
    marker.write_text("dirty\n", encoding="utf-8")
    monkeypatch.setattr(filesystem, "_backend", windows_backend)
    def fail(*args):
        raise PermissionError("native query-index denial")
    monkeypatch.setattr(windows_backend, "_move_file_ex", fail)
    with pytest.raises(QueryIndexError, match="dirty-marker publication"):
        index._mark_dirty()
    with pytest.raises(QueryIndexError, match="dirty-marker removal"):
        index._clear_dirty((marker,), roots=(), clear_unscoped=True)
    assert marker.exists()
    with pytest.raises(QueryIndexError, match="replacement activation"):
        index._activate_replacement(replacement_path, quarantine_existing=False)
    assert path.exists() and replacement_path.exists()
    index.close()


def test_windows_logical_name_delete_retains_only_ignored_tombstone(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store", query_index="none")
    root = store.publish_node(creation())
    name = NodeNameRecord("latest", "recipe", root.digest)
    store.compare_and_set_node_name(name, expected=None)
    moves = _emulate_windows_moves(monkeypatch)
    original = os.unlink
    def retain(path, *args, **kwargs):
        if Path(path).name.startswith(".dryml-removed-"):
            raise PermissionError("retain tombstone")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(os, "unlink", retain)
    assert store.delete_node_name("latest", expected=name)
    parent = Path(store.base_dir, "names")
    tombstones = tuple(parent.glob(".dryml-removed-*"))
    assert store.read_node_name("latest") is None
    assert len(tombstones) == 1 and not tuple(parent.glob("*.record"))
    name_moves = [m for m in moves if m[1].parent == parent]
    assert name_moves[-1][1] == tombstones[0]
    assert not name_moves[-1][2] & windows_backend._MOVEFILE_REPLACE_EXISTING
    assert store.read_node(root.digest) is not None


def test_windows_zip_commit_flushes_writable_file_and_filters_tombstones(tmp_path, monkeypatch):
    archive_path = tmp_path / "store.zip"
    store = ZipStore(archive_path)
    try:
        install(store)
        root = store.publish_node(creation())
        name = NodeNameRecord("latest", "recipe", root.digest)
        store.compare_and_set_node_name(name, expected=None)
        moves = _emulate_windows_moves(monkeypatch)
        original = os.unlink
        with monkeypatch.context() as patch:
            def retain(path, *args, **kwargs):
                if Path(path).name.startswith(".dryml-removed-"):
                    raise PermissionError("retain tombstone")
                return original(path, *args, **kwargs)
            patch.setattr(os, "unlink", retain)
            assert store.delete_node_name("latest", expected=name)
        moves.clear()
        store.commit()
        archive_move = next(m for m in moves if m[1] == archive_path)
        assert archive_move[2] == windows_backend._MOVEFILE_WRITE_THROUGH | windows_backend._MOVEFILE_REPLACE_EXISTING
        with zipfile.ZipFile(archive_path) as archive:
            names = archive.namelist()
            assert any(Path(n).name == _REMOVED_ENTRY_PREFIX + "payload" for n in names)
            assert not any(Path(n).name.startswith(".dryml-removed-") for n in names)
            assert "authority.record" in names
    finally:
        store.close()


@pytest.mark.skipif(os.name != "nt", reason="requires native Windows MoveFileExW and file sharing semantics")
def test_native_windows_dir_and_zip_store_round_trip(tmp_path):
    direct = DirStore(tmp_path / "nested" / "direct", query_index="none")
    state = install(direct)
    root = direct.publish_node(creation())
    name = NodeNameRecord("main", "recipe", root.digest)
    direct.compare_and_set_node_name(name, expected=None)
    reopened = DirStore(direct.base_dir, query_index="none")
    assert reopened.read_node_name("main") == name
    with reopened.read_state(state.state.key) as read:
        assert Path(read.data_directory, _REMOVED_ENTRY_PREFIX + "payload").read_bytes() == b"payload"
    assert reopened.delete_node_name("main", expected=name)
    archive_path = tmp_path / "store.zip"
    archive = ZipStore(archive_path)
    try:
        state = install(archive)
        archive.commit()
    finally:
        archive.close()
    committed = ZipStore(archive_path)
    try:
        with committed.read_state(state.state.key) as read:
            assert Path(read.data_directory, _REMOVED_ENTRY_PREFIX + "payload").read_bytes() == b"payload"
    finally:
        committed.close()
