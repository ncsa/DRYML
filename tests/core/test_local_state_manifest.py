from pathlib import Path

import hashlib
import dill
import pytest

from dryml.core import Object
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


def test_local_state_install_validates_manifest_and_direct_path(tmp_path):
    store = DirStore(tmp_path / "store")
    record = DefinitionRecord(ManifestObject().definition)
    stage, manifest = _stage(store, record)

    store.install_local_state(stage, manifest)

    assert Path(store.open_local_state(record.graph_hash, manifest.state_hash)).is_dir()


def test_empty_data_root_is_a_valid_complete_local_state(tmp_path):
    store = DirStore(tmp_path / "store")
    record = DefinitionRecord(ManifestObject().definition)
    stage = Path(store.create_local_state_staging())
    definition_bytes = record.to_bytes()
    manifest = LocalStateManifest("Codec1", record.graph_hash, record.digest, hashlib.sha256(definition_bytes).hexdigest(), ())
    (stage / "def.pkl").write_bytes(definition_bytes)
    (stage / "manifest.record").write_bytes(manifest.to_bytes())

    store.install_local_state(stage, manifest)

    assert Path(store.open_local_state(record.graph_hash, manifest.state_hash)).is_dir()


def test_manifest_rejects_empty_nested_directories_and_extra_files(tmp_path):
    store = DirStore(tmp_path / "store")
    record = DefinitionRecord(ManifestObject().definition)
    stage, manifest = _stage(store, record)
    (stage / "data" / "empty").mkdir()

    with pytest.raises(StoreAuthorityError, match="empty nested"):
        store.install_local_state(stage, manifest)

    with pytest.raises(StoreRecordError, match="codec"):
        LocalStateManifest("not-valid!", record.graph_hash, record.digest, "0" * 64, ())


def test_manifest_rejects_symlinked_payload_entries(tmp_path):
    store = DirStore(tmp_path / "store")
    record = DefinitionRecord(ManifestObject().definition)
    stage, manifest = _stage(store, record)
    (stage / "data" / "link").symlink_to(stage / "data" / "nested" / "value.bin")

    with pytest.raises(StoreAuthorityError, match="unsupported file"):
        store.install_local_state(stage, manifest)


def test_manifest_rejects_reencoded_or_modified_definition_bytes(tmp_path):
    store = DirStore(tmp_path / "store")
    record = DefinitionRecord(ManifestObject().definition)
    stage, manifest = _stage(store, record)

    # The graph record still decodes to valid logical authority, but byte-level
    # local-state authority must retain the exact adjacent def.pkl payload.
    encoded = record.to_bytes()
    prefix = b"DRYML-STORE-RECORD/definition/1\n"
    (stage / "def.pkl").write_bytes(prefix + dill.dumps(dill.loads(encoded[len(prefix):]), protocol=4))
    with pytest.raises(StoreAuthorityError, match="definition file bytes"):
        store.install_local_state(stage, manifest)

    (stage / "def.pkl").write_bytes(encoded + b"\n")
    with pytest.raises(StoreAuthorityError, match="definition file bytes"):
        store.install_local_state(stage, manifest)
