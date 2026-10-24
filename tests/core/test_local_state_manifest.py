from pathlib import Path

import hashlib
import pytest

from dryml.core import Definition, Object
from dryml.core.store.dir import DirStore
from dryml.core.store.records import DefinitionRecord, LocalStateManifest, StoreRecordError
from dryml.core.store.store import StoreAuthorityError


class ManifestObject(Object):
    pass


def _stage(store, record, payload=b"payload"):
    stage = Path(store.base_dir) / ".staging" / "state"
    data = stage / "data" / "nested"
    data.mkdir(parents=True)
    (data / "value.bin").write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    definition_bytes = record.to_bytes()
    manifest = LocalStateManifest("Codec1", record.graph_hash, record.digest, hashlib.sha256(definition_bytes).hexdigest(), (("nested/value.bin", len(payload), digest),))
    (stage / "def.pkl").write_bytes(definition_bytes)
    (stage / "manifest.record").write_bytes(manifest.to_bytes())
    return stage, manifest


def test_local_state_preparation_validates_manifest_and_owned_staging(tmp_path):
    store = DirStore(tmp_path / "store")
    stage = Path(store.create_local_state_staging())
    (stage / "data" / "value").write_bytes(b"payload")
    result = store.install_state(Definition(ManifestObject).concretize(), stage, codec="Codec1", codec_version=1, interpretation={})
    with store.read_state(result.state.key) as source:
        assert source.manifest.key == result.state.key
        assert source.manifest.version == 1
        assert source.manifest.deferred_paths == ()
        assert Path(source.data_directory, "value").read_bytes() == b"payload"


def test_empty_data_root_is_a_valid_complete_local_state(tmp_path):
    store = DirStore(tmp_path / "store")
    stage = Path(store.create_local_state_staging())
    result = store.install_state(Definition(ManifestObject).concretize(), stage, codec="Codec1", codec_version=1, interpretation={})
    with store.read_state(result.state.key) as source:
        assert Path(source.data_directory).is_dir()
        assert source.manifest.files == ()


def test_manifest_rejects_empty_nested_directories_and_extra_files(tmp_path):
    store = DirStore(tmp_path / "store")
    record = DefinitionRecord(ManifestObject().definition)
    stage = Path(store.create_local_state_staging())
    (stage / "data" / "empty").mkdir()
    with pytest.raises((StoreAuthorityError, StoreRecordError), match="empty nested"):
        store.install_state(Definition(ManifestObject).concretize(), stage, codec="Codec1", codec_version=1, interpretation={})

    with pytest.raises(StoreRecordError, match="codec"):
        LocalStateManifest("not-valid!", record.graph_hash, record.digest, "0" * 64, ())


def test_manifest_rejects_symlinked_payload_entries(tmp_path):
    store = DirStore(tmp_path / "store")
    stage = Path(store.create_local_state_staging())
    (stage / "data" / "value").write_bytes(b"payload")
    (stage / "data" / "link").symlink_to(stage / "data" / "value")
    with pytest.raises((StoreAuthorityError, StoreRecordError), match="unsupported entry"):
        store.install_state(Definition(ManifestObject).concretize(), stage, codec="Codec1", codec_version=1, interpretation={})


def test_manifest_rejects_reencoded_or_modified_definition_bytes(tmp_path):
    store = DirStore(tmp_path / "store")
    stage = Path(store.create_local_state_staging())
    result = store.install_state(Definition(ManifestObject).concretize(), stage, codec="Codec1", codec_version=1, interpretation={})
    key = result.state.key
    header = Path(store.base_dir, "states", key.cdef_digest[:2], key.cdef_digest, "def.pkl")
    encoded = header.read_bytes()
    for changed in (encoded + b"\n", b"unsupported-definition" + encoded):
        header.write_bytes(changed)
        with pytest.raises(StoreAuthorityError):
            with store.read_state(key):
                pytest.fail("malformed creation bytes were admitted")
        assert header.read_bytes() == changed


def test_v3_deferred_paths_must_be_sorted_unique_and_in_the_payload_inventory(tmp_path):
    store = DirStore(tmp_path / "store")
    record = DefinitionRecord(ManifestObject().definition)
    stage, manifest = _stage(store, record)
    files = manifest.files

    for deferred_paths in (
            ("nested/value.bin", "nested/value.bin"),
            ("../nested/value.bin",),
            ("missing",),
    ):
        with pytest.raises(StoreRecordError):
            LocalStateManifest(
                manifest.codec, manifest.graph_hash, manifest.definition_digest,
                manifest.definition_file_digest, files, deferred_paths,
            )

    sorted_files = (("another", 0, "0" * 64), *files)
    with pytest.raises(StoreRecordError, match="sorted"):
        LocalStateManifest(
            manifest.codec, manifest.graph_hash, manifest.definition_digest,
            manifest.definition_file_digest, sorted_files,
            ("nested/value.bin", "another"),
        )

    deferred = LocalStateManifest(
        manifest.codec, manifest.graph_hash, manifest.definition_digest,
        manifest.definition_file_digest, files, ("nested/value.bin",),
    )
    (stage / "data" / "extra").write_bytes(b"extra")
    with pytest.raises(StoreRecordError, match="empty nested|exactly match"):
        deferred.validate_payload(stage / "data", defer_payload=True)
    (stage / "data" / "extra").unlink()
    (stage / "data" / "nested" / "value.bin").unlink()
    with pytest.raises(StoreRecordError, match="empty nested|exactly match"):
        deferred.validate_payload(stage / "data", defer_payload=True)
    (stage / "data" / "nested").rmdir()
    (stage / "data" / "nested").symlink_to(stage / "data")
    with pytest.raises(StoreRecordError, match="unsupported directory"):
        deferred.validate_payload(stage / "data", defer_payload=True)
