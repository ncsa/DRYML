"""Focused Store-v4 authority, independent state buckets and borrowed reads."""

from pathlib import Path
import hashlib
from uuid import uuid4

import dill
import pytest

from dryml.core import Definition, Object, ObjectId
from dryml.core.node_views import CREATION_VIEW, view_digest
from dryml.core.state_values import StateKey
from dryml.core.store.dir import DirStore
from dryml.core.store.store import StoreAuthorityError
from dryml.core.store.records import (
    StoreFormatRecord, NodeRootRecord, NodeBlockRecord, StateVariantManifest,
    StateKeyRecord, StateAvailabilityRecord, DeclarationAnchorRecord, StoreRecordError,
)


class BucketObject(Object):
    def __init__(self, value=1):
        self.value = value


def creation(value=1):
    return Definition(BucketObject, value=value).concretize()


def install(store, root=None, payload=b"state"):
    root = creation() if root is None else root
    stage = store.create_local_state_staging()
    Path(stage, "data", "value").write_bytes(payload)
    return store.install_state(root, stage, codec="Test", codec_version=1,
                               interpretation={"encoding": "bytes"})


def test_format_four_has_persistent_fresh_uuid(tmp_path):
    first = DirStore(tmp_path / "first", query_index="none")
    second = DirStore(tmp_path / "second", query_index="none")
    assert first.store_uuid != second.store_uuid
    assert DirStore(first.base_dir, query_index="none").store_uuid == first.store_uuid
    assert first._read_file(first.store_format_path, StoreFormatRecord).format_version == 4


def test_idempotent_full_key_and_independent_creation_buckets(tmp_path, monkeypatch):
    import dryml.filesystem as fs
    publications = []
    original = fs.publish_directory
    def publish(source, destination):
        publications.append(destination)
        return original(source, destination)
    monkeypatch.setattr(fs, "publish_directory", publish)
    store = DirStore(tmp_path / "store", query_index="none")
    a = install(store)
    again = install(store, creation().with_identity(ObjectId()))
    b = install(store, creation(2))
    assert a.state.key == again.state.key
    assert a.state.key.codec_hash == b.state.key.codec_hash
    assert a.state.key.cdef_digest != b.state.key.cdef_digest
    for outcome in (a, b):
        key = outcome.state.key
        bucket = Path(store.base_dir, "states", key.cdef_digest[:2], key.cdef_digest)
        assert (bucket / "def.pkl").is_file()
        assert (bucket / key.codec_hash / "manifest.record").is_file()
        assert (bucket / key.codec_hash / "data" / "value").read_bytes() == b"state"
    assert again.installed is False
    assert a.installed and b.installed
    assert len(publications) == 2


def test_header_only_and_missing_private_data_preserve_history(tmp_path):
    store = DirStore(tmp_path / "store", query_index="none")
    digest = store.install_creation(creation())
    assert store.read_creation(digest).graph_equal(creation())
    assert store.inspect_states(digest) == ()
    outcome = install(store)
    key = outcome.state.key
    Path(store.state_data_path(key), "value").unlink()
    Path(store.state_data_path(key)).rmdir()
    historical, = store.inspect_states(digest)
    assert historical.state == outcome.state
    assert historical.available is False
    assert store.inspect_states(digest, available_only=True) == ()
    with pytest.raises(StoreAuthorityError, match="unavailable"):
        with store.read_state(key):
            pytest.fail("missing data must not be lent")


def test_carried_key_addressing_performs_no_graph_work(tmp_path, monkeypatch):
    from dryml.core.node_prepare import NodePreparation
    store = DirStore(tmp_path / "store", query_index="none")
    key = StateKey("a" * 64, "Test-" + "b" * 64)
    monkeypatch.setattr(NodePreparation, "__init__", lambda *_a, **_kw: pytest.fail("graph discovery"))
    monkeypatch.setattr(hashlib, "sha256", lambda *_a, **_kw: pytest.fail("hashing"))
    assert store.state_data_path(key) == str(Path(store.base_dir, "states", "aa", "a" * 64,
                                                key.codec_hash, "data"))


@pytest.mark.parametrize("version", [3, 5])
def test_incompatible_store_is_rejected_intact(tmp_path, version):
    root = tmp_path / "old"
    root.mkdir()
    original = b"DRYML-STORE-RECORD/store-format/1\n" + dill.dumps({
        "schema": "store-format", "version": 1, "format_version": version,
    }, protocol=5)
    Path(root, "store-format.record").write_bytes(original)
    with pytest.raises(StoreAuthorityError):
        DirStore(root, query_index="none")
    assert {p.name: p.read_bytes() for p in root.iterdir()} == {"store-format.record": original}


def authority_files(store):
    return {str(p.relative_to(store.base_dir)): p.read_bytes()
            for p in Path(store.base_dir).rglob("*")
            if p.is_file() and p.name != ".writer.lock" and ".staging" not in p.parts}


def test_batch_shares_creation_decode_and_reads_no_private_bytes(tmp_path, monkeypatch):
    import dryml.core.node_codec as codec
    import builtins
    store = DirStore(tmp_path / "store", query_index="none")
    first = install(store)
    install(store, payload=b"second")
    calls = []
    original = codec.decode_node_graph
    def decode(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    original_open = builtins.open
    def opened(path, *args, **kwargs):
        assert "/data/" not in str(path), "inspection opened private bytes"
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(codec, "decode_node_graph", decode)
    monkeypatch.setattr(builtins, "open", opened)
    monkeypatch.setattr(BucketObject, "__init__", lambda *_a, **_kw: pytest.fail("inspection constructed an Object"))
    descriptors = store.inspect_states(first.state.key.cdef_digest)
    assert len(descriptors) == 2 and all(d.available for d in descriptors)
    assert len(calls) == 1


def test_creation_headers_share_one_table_instead_of_subtree_exports(tmp_path):
    from dryml.core.node_prepare import NodePreparation
    store = DirStore(tmp_path / "store", query_index="none")
    leaf = creation()
    parent = Definition(BucketObject, value=[leaf, leaf]).concretize()
    digests = store.install_creations((parent, parent.parameters["value"][0]))
    headers = [NodeRootRecord.from_bytes(Path(store._bucket_path(d), "def.pkl").read_bytes()) for d in digests]
    assert headers[0].block_digest == headers[1].block_digest
    assert len(list(Path(store.base_dir, "definitions", "blocks").rglob("*.record"))) == 1
    for header, digest in zip(headers, digests):
        assert set(header.view) == {"table", "slot", "view", "key_format", "expected_key"}
        assert header.digest == digest
    restored = store.read_creation(digests[0])
    assert restored.parameters["value"][0] is restored.parameters["value"][1]
    assert NodePreparation((parent,)).exact_equal(parent, NodePreparation((restored,)), restored)


@pytest.mark.parametrize("damage", ["manifest-key", "codec-version", "interpretation", "trailing",
                                    "root-kind", "root-version", "identity", "domain", "blocks"])
def test_reopened_nested_authority_damage_is_refused_intact(tmp_path, damage):
    store = DirStore(tmp_path / "store", query_index="none")
    outcome = install(store)
    key = outcome.state.key
    variant = Path(store.state_data_path(key)).parent
    header = Path(store._bucket_path(key.cdef_digest), "def.pkl")
    from dryml.formats.envelope import encode_metadata_frame
    if damage in {"manifest-key", "codec-version", "interpretation", "trailing"}:
        target = variant / "manifest.record"
        data = StateVariantManifest.from_bytes(target.read_bytes()).to_data()
        if damage == "manifest-key":
            data["key"]["version"] = 2
        elif damage == "codec-version":
            data["codec_version"] += 1
        elif damage == "interpretation":
            data["interpretation"]["encoding"] = "changed"
        damaged = encode_metadata_frame(StateVariantManifest.schema, 1, data)
        if damage == "trailing":
            damaged += b"trailing"
    elif damage == "blocks":
        record = NodeRootRecord.from_bytes(header.read_bytes())
        target = Path(store._node_path("definitions/blocks", record.block_digest))
        data = NodeBlockRecord.from_bytes(target.read_bytes()).to_data()
        data["blocks"]["identity_version"] = 2
        damaged = encode_metadata_frame(NodeBlockRecord.schema, 1, data)
    else:
        target = header
        data = NodeRootRecord.from_bytes(header.read_bytes()).to_data()
        if damage == "root-kind":
            data["root_kind"] = "recipe"
        elif damage == "root-version":
            data["version"] = 2
        elif damage == "identity":
            data["identity_version"] = 2
        else:
            data["domain"] = "foreign/v1"
        damaged = encode_metadata_frame(NodeRootRecord.schema, 1, data)
    target.write_bytes(damaged)
    before = authority_files(store)
    reopened = DirStore(store.base_dir, query_index="none")
    with pytest.raises(StoreAuthorityError):
        reopened.inspect_states(key.cdef_digest)
    assert authority_files(store) == before
    with pytest.raises(StoreAuthorityError):
        install(reopened)
    assert authority_files(store) == before


def test_header_collision_never_replaces_bucket_or_sibling(tmp_path):
    store = DirStore(tmp_path / "store", query_index="none")
    a, b = install(store), install(store, creation(2))
    a_header = Path(store._bucket_path(a.state.key.cdef_digest), "def.pkl")
    a_header.write_bytes(Path(store._bucket_path(b.state.key.cdef_digest), "def.pkl").read_bytes())
    before = authority_files(store)
    with pytest.raises(StoreAuthorityError, match="collision|mismatch"):
        install(store)
    assert authority_files(store) == before


def test_state_use_checks_codec_configuration_and_bytes_before_lending(tmp_path):
    store = DirStore(tmp_path / "store", query_index="none")
    outcome = install(store)
    with store.read_state(outcome.state.key, destination=creation(), codec="Test", codec_version=1) as source:
        assert Path(source.data_directory, "value").read_bytes() == b"state"
        assert {p.name for p in Path(source.data_directory).iterdir()} == {"value"}
    for destination, codec in [(creation(), "Other"), (creation(2), "Test")]:
        with pytest.raises(StoreAuthorityError, match="incompatible"):
            with store.read_state(outcome.state.key, destination=destination, codec=codec, codec_version=1):
                pytest.fail("incompatible source lent")
    with pytest.raises(StoreAuthorityError, match="codec version"):
        with store.read_state(outcome.state.key, destination=creation(), codec="Test", codec_version=2):
            pytest.fail("incompatible codec version lent")
    Path(store.state_data_path(outcome.state.key), "value").write_bytes(b"wrong")
    assert store.inspect_states(outcome.state.key.cdef_digest)[0].available
    with pytest.raises(StoreAuthorityError, match="manifest"):
        with store.read_state(outcome.state.key):
            pytest.fail("corrupt source lent")


def test_read_sources_expire_but_detached_metadata_survives(tmp_path):
    store = DirStore(tmp_path / "store", query_index="none")
    outcome = install(store)
    with store.read_state(outcome.state.key) as source:
        assert Path(source.data_directory).is_dir()
    with pytest.raises(RuntimeError, match="expired"):
        source.data_directory
    with store.read_state(outcome.state.key) as source:
        detached = source.state
        store.close()
        with pytest.raises(RuntimeError, match="closed"):
            source.data_directory
    assert detached == outcome.state
    assert detached.creation_cdef.graph_equal(creation())


def test_independent_memberships_and_canonical_declaration_anchors(tmp_path):
    store = DirStore(tmp_path / "store", query_index="none")
    recipe = store.publish_node(creation())
    identity = ObjectId()
    variants = [install(store, payload=b) for b in (b"one", b"two", b"three")]
    records = [store.publish_node(creation().with_identity(identity).with_state(v.state), kind="saved")
               for v in variants]
    assert len({r.digest for r in records}) == 3
    assert len(list(Path(store.base_dir, "membership", "recipe").rglob("*.record"))) == 1
    anchor = store.read_declaration_anchor(identity)
    assert len(anchor.sources) == 3
    for record in records:
        assert store.read_node(record.digest, kind="saved").object_id == identity
    assert store.read_node(recipe.digest).object_id is None
    before = authority_files(store)
    with pytest.raises(StoreAuthorityError, match="projection conflicts"):
        store.publish_node(creation(2).with_identity(identity))
    assert authority_files(store) == before


@pytest.mark.parametrize("damage", ["missing", "nested-version", "source", "projection"])
def test_missing_or_incompatible_anchor_refuses_publication_before_membership(tmp_path, damage):
    from dryml.formats.envelope import encode_metadata_frame
    store = DirStore(tmp_path / "store", query_index="none")
    identity = ObjectId()
    original = store.publish_node(creation().with_identity(identity))
    replacement = creation().with_identity(identity).with_state(install(store).state)
    path = Path(store._node_path("declarations/anchors", store._anchor_key(identity)))
    if damage == "missing":
        path.unlink()
    else:
        data = DeclarationAnchorRecord.from_bytes(path.read_bytes()).to_data()
        if damage == "nested-version":
            data["projection"]["identity_version"] = 2
        elif damage == "source":
            data["sources"] = []
        else:
            data["projection"]["root_kind"] = "saved"
        path.write_bytes(encode_metadata_frame(DeclarationAnchorRecord.schema, 1, data))
    before = authority_files(store)
    with pytest.raises(StoreAuthorityError):
        store.read_node(original.digest)
    with pytest.raises(StoreAuthorityError):
        store.publish_node(replacement, kind="saved")
    assert authority_files(store) == before


def test_reinstallation_changes_incarnation_and_preserves_siblings(tmp_path):
    store = DirStore(tmp_path / "store", query_index="none")
    first, sibling = install(store), install(store, payload=b"other")
    foreign = install(store, creation(2))
    descriptor = next(d for d in store.inspect_states(first.state.key.cdef_digest) if d.state.key == first.state.key)
    Path(store.state_data_path(first.state.key), "value").unlink()
    Path(store.state_data_path(first.state.key)).rmdir()
    result = install(store)
    new = next(d for d in store.inspect_states(first.state.key.cdef_digest) if d.state.key == first.state.key)
    assert result.installed and new.incarnation != descriptor.incarnation
    assert Path(store.state_data_path(sibling.state.key), "value").read_bytes() == b"other"
    assert Path(store.state_data_path(foreign.state.key), "value").read_bytes() == b"state"


def test_zip_buffered_receipt_commit_failure_and_source_lifetime(tmp_path, monkeypatch):
    import dryml.filesystem as fs
    from dryml.core.store.zip import ZipStore
    archive = tmp_path / "store.zip"
    store = ZipStore(archive)
    try:
        result = install(store)
        assert result.durability == "buffered" and not archive.exists()
        original = fs.publish_file
        def fail(source, target, *, replace=False):
            if str(target) == str(archive):
                raise OSError("archive barrier failed")
            return original(source, target, replace=replace)
        with monkeypatch.context() as patch:
            patch.setattr(fs, "publish_file", fail)
            with pytest.raises(OSError, match="archive barrier"):
                store.commit()
        assert store._archive_dirty and not archive.exists()
        store.commit()
        with ZipStore(archive).read_state(result.state.key) as read:
            owner = read.store
            assert Path(read.data_directory, "value").read_bytes() == b"state"
            owner.close()
            with pytest.raises(RuntimeError, match="closed"):
                read.data_directory
    finally:
        store.close()


def test_legacy_nested_store_writers_and_query_coverage_refuse_before_effects(tmp_path):
    from dryml.core.store.records import DefinitionRecord
    from dryml.core.store.store import StoreCapabilityError
    store = DirStore(tmp_path / "store", query_index="sqlite")
    before = authority_files(store)
    for action in [lambda: store.write_definition_record(DefinitionRecord(creation())),
                   store.query]:
        with pytest.raises(StoreCapabilityError):
            action()
    assert authority_files(store) == before
    index = store.open_query_index()
    assert index.status().state == "missing"  # Binding alone is not coverage.
    assert {path: data for path, data in authority_files(store).items()
            if not path.startswith(".dryml/")} == before
    assert store.authority_cut().epoch == 0 and not store.authority_cut().pending


def test_state_key_record_is_closed_separate_version_one():
    key = StateKey("a" * 64, "Test-" + "b" * 64)
    record = StateKeyRecord(key)
    assert StateKeyRecord.from_bytes(record.to_bytes()).key == key
    for field, value in [("version", True), ("identity_version", 2), ("domain", "foreign/v1"),
                         ("kind", "local-state-manifest")]:
        data = record.to_data()
        data[field] = value
        with pytest.raises(StoreRecordError):
            StateKeyRecord.from_data(data)


def test_duplicate_semantic_creation_roots_share_header_even_with_distinct_slots(tmp_path):
    store = DirStore(tmp_path / "store", query_index="none")
    first, second = store.install_creations((creation(), creation()))
    assert first == second == view_digest(creation(), CREATION_VIEW)
    before = authority_files(store)
    assert store.install_creations((creation(), creation())) == (first, first)
    assert store.install_creation(creation()) == first
    assert authority_files(store) == before


def test_interrupted_variant_install_leaves_header_only_then_retry_succeeds(tmp_path, monkeypatch):
    import dryml.filesystem as fs
    store = DirStore(tmp_path / "store", query_index="none")
    def fail(source, destination):
        raise OSError("variant installation interrupted")
    with monkeypatch.context() as patch:
        patch.setattr(fs, "publish_directory", fail)
        with pytest.raises(OSError, match="interrupted"):
            install(store)
    digest = view_digest(creation(), CREATION_VIEW)
    assert store.read_creation(digest).graph_equal(creation())
    assert store.inspect_states(digest) == ()
    assert install(store).installed


def test_failed_reinstallation_never_advertises_prepared_data(tmp_path, monkeypatch):
    import dryml.filesystem as fs
    store = DirStore(tmp_path / "store", query_index="none")
    result = install(store)
    Path(store.state_data_path(result.state.key), "value").unlink()
    Path(store.state_data_path(result.state.key)).rmdir()
    original = fs.publish_directory
    def publish_then_fail(source, destination):
        original(source, destination)
        raise OSError("directory barrier failed after visibility")
    with monkeypatch.context() as patch:
        patch.setattr(fs, "publish_directory", publish_then_fail)
        with pytest.raises(OSError, match="after visibility"):
            install(store)
    historical, = store.inspect_states(result.state.key.cdef_digest)
    assert not historical.available
    assert Path(store.state_data_path(result.state.key), "value").is_file()
    assert install(store).installed
    assert store.inspect_states(result.state.key.cdef_digest)[0].available


def test_overlapping_variant_writers_share_header_and_preserve_both_children(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    stores = [DirStore(tmp_path / "store", query_index="none") for _ in range(2)]
    ready = Barrier(2)
    def writer(index):
        stage = stores[index].create_local_state_staging()
        Path(stage, "data", "value").write_bytes(bytes([index]))
        ready.wait(timeout=10)
        return stores[index].install_state(creation(), stage, codec="Test", codec_version=1,
                                           interpretation={"encoding": "bytes"})
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(writer, range(2)))
    descriptors = stores[0].inspect_states(results[0].state.key.cdef_digest)
    assert len(descriptors) == 2 and all(d.available for d in descriptors)
    assert {d.state.key for d in descriptors} == {r.state.key for r in results}


def test_zip_failure_after_variant_visibility_keeps_buffer_dirty(tmp_path, monkeypatch):
    import dryml.filesystem as fs
    from dryml.core.store.zip import ZipStore
    store = ZipStore(tmp_path / "store.zip")
    try:
        install(store)
        store.commit()
        original = fs.publish_directory
        def publish_then_fail(source, destination):
            original(source, destination)
            raise OSError("variant barrier failed after visibility")
        with monkeypatch.context() as patch:
            patch.setattr(fs, "publish_directory", publish_then_fail)
            with pytest.raises(OSError, match="after visibility"):
                install(store, payload=b"new sibling")
        assert store._archive_dirty
        assert len(store.inspect_states(view_digest(creation(), CREATION_VIEW))) == 2
    finally:
        store.close()


def test_borrowed_sources_refuse_serialization_and_changed_store_uuid(tmp_path):
    store = DirStore(tmp_path / "store", query_index="none")
    state = install(store).state
    with store.read_state(state.key) as source:
        with pytest.raises(TypeError, match="Borrowed"):
            dill.dumps(source)
        Path(store.store_format_path).write_bytes(StoreFormatRecord().to_bytes())
        with pytest.raises(StoreAuthorityError, match="UUID changed"):
            source.data_directory


def test_codec_content_hash_collision_checks_exact_inventory_before_effects(tmp_path, monkeypatch):
    import dryml.core.store.records as records
    original = records._digest
    def collision(domain, value):
        return "f" * 64 if domain == "dryml-codec-content/v1" else original(domain, value)
    monkeypatch.setattr(records, "_digest", collision)
    store = DirStore(tmp_path / "store", query_index="none")
    install(store)
    before = authority_files(store)
    with pytest.raises(StoreAuthorityError, match="manifest collision"):
        install(store, payload=b"different")
    assert authority_files(store) == before


@pytest.mark.parametrize("interpretation,version", [({1: "not a JSON key"}, 1), ({}, True), ([], 1)])
def test_malformed_codec_interpretation_fails_before_authority_effects(tmp_path, interpretation, version):
    store = DirStore(tmp_path / "store", query_index="none")
    before = authority_files(store)
    stage = store.create_local_state_staging()
    with pytest.raises(StoreAuthorityError):
        store.install_state(creation(), stage, codec="Test", codec_version=version, interpretation=interpretation)
    assert authority_files(store) == before


def test_invalid_embedded_state_evidence_cannot_publish_creation_header(tmp_path):
    from dryml.core.state_values import LocalState
    store = DirStore(tmp_path / "store", query_index="none")
    forged = LocalState(StateKey("0" * 64, "Test-" + "1" * 64), creation())
    parent = Definition(BucketObject, value=creation().with_state(forged)).concretize()
    before = authority_files(store)
    with pytest.raises(StoreAuthorityError, match="selected state"):
        store.install_creation(parent)
    assert authority_files(store) == before


def test_zip_file_like_is_explicit_v4_read_source_without_durable_publication(tmp_path):
    from io import BytesIO
    from dryml.core.store.zip import ZipStore
    from dryml.core.store.store import StoreCapabilityError
    archive = tmp_path / "store.zip"
    writer = ZipStore(archive)
    try:
        state = install(writer).state
        assert writer.publication_capabilities.durability == "buffered"
        with pytest.raises(StoreCapabilityError, match="buffered"):
            writer.publication_capabilities.require_durable("complete mutation")
        writer.commit()
    finally:
        writer.close()
    source = ZipStore(BytesIO(archive.read_bytes()))
    try:
        assert source.publication_capabilities.durability == "read-only"
        with source.read_state(state.key) as read:
            assert Path(read.data_directory, "value").read_bytes() == b"state"
        with pytest.raises(StoreCapabilityError, match="writable"):
            source.create_local_state_staging()
    finally:
        source.close()


@pytest.mark.parametrize("damage", ["header", "manifest", "availability", "availability-key", "data-type"])
def test_missing_required_records_or_wrong_data_type_refuse_without_repair(tmp_path, damage):
    store = DirStore(tmp_path / "store", query_index="none")
    state = install(store).state
    data = Path(store.state_data_path(state.key))
    if damage == "header":
        Path(store._bucket_path(state.key.cdef_digest), "def.pkl").unlink()
    elif damage == "data-type":
        Path(data, "value").unlink()
        data.rmdir()
        data.write_bytes(b"not a directory")
    elif damage == "availability-key":
        Path(data.parent, "availability.record").write_bytes(StateAvailabilityRecord(
            StateKey(state.key.cdef_digest, "Test-" + "0" * 64), str(uuid4())).to_bytes())
    else:
        Path(data.parent, damage + ".record").unlink()
    before = authority_files(store)
    with pytest.raises(StoreAuthorityError):
        store.inspect_states(state.key.cdef_digest)
    with pytest.raises(StoreAuthorityError):
        install(store)
    assert authority_files(store) == before


def test_archive_commit_excludes_unpublished_codec_staging(tmp_path):
    import zipfile
    from dryml.core.store.zip import ZipStore
    store = ZipStore(tmp_path / "store.zip")
    try:
        install(store)
        stage = store.create_local_state_staging()
        Path(stage, "data", "unfinished").write_bytes(b"not published")
        store.commit()
        with zipfile.ZipFile(store.archive_path) as archive:
            assert not any(name.startswith(".staging/") for name in archive.namelist())
        assert Path(stage, "data", "unfinished").is_file()
    finally:
        store.close()


def test_archive_replacement_during_open_does_not_rebase_old_extraction(tmp_path, monkeypatch):
    import os
    from dryml.core.store.zip import ZipStore, ZipStoreConflictError
    archive, replacement = tmp_path / "store.zip", tmp_path / "replacement.zip"
    for path, payload in [(archive, b"old"), (replacement, b"replacement")]:
        writer = ZipStore(path)
        try:
            install(writer, payload=payload)
            writer.commit()
        finally:
            writer.close()
    expected = replacement.read_bytes()
    original = ZipStore._extract_if_present
    def extract_then_replace(self):
        original(self)
        if self.archive_path == str(archive):
            os.replace(replacement, archive)
    monkeypatch.setattr(ZipStore, "_extract_if_present", extract_then_replace)
    reader = ZipStore(archive)
    try:
        install(reader, payload=b"local change")
        with pytest.raises(ZipStoreConflictError, match="changed since open"):
            reader.commit()
        assert archive.read_bytes() == expected
    finally:
        reader.close()


def test_short_manifest_write_cannot_report_installed_state(tmp_path, monkeypatch):
    import io
    store = DirStore(tmp_path / "store", query_index="none")
    original = io.open
    class ShortWriter:
        def __init__(self, stream):
            self.stream = stream
        def __enter__(self):
            self.stream.__enter__()
            return self
        def __exit__(self, *args):
            return self.stream.__exit__(*args)
        def write(self, payload):
            return self.stream.write(payload[:4])
    def opened(path, mode="r", *args, **kwargs):
        stream = original(path, mode, *args, **kwargs)
        return ShortWriter(stream) if not isinstance(path, int) and Path(path).name == "manifest.record" and mode == "wb" else stream
    monkeypatch.setattr(io, "open", opened)
    with pytest.raises(OSError, match="incomplete"):
        install(store)
    assert store.inspect_states(view_digest(creation(), CREATION_VIEW)) == ()


def test_unencodable_anchor_is_refused_before_any_publication_authority(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store", query_index="none")
    before = authority_files(store)
    def oversized(self):
        raise StoreRecordError("Oversized declaration anchor")
    monkeypatch.setattr(DeclarationAnchorRecord, "to_bytes", oversized)
    with pytest.raises(StoreRecordError, match="Oversized"):
        store.publish_node(creation().with_identity(ObjectId()))
    assert authority_files(store) == before
