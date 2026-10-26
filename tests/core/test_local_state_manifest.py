from pathlib import Path

import pytest

from dryml.core import Definition, Object
from dryml.core.store.dir import DirStore
from dryml.core.store.records import StateVariantManifest, StoreRecordError
from dryml.core.store.store import StoreAuthorityError


class ManifestObject(Object):
    pass


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
    stage = Path(store.create_local_state_staging())
    (stage / "data" / "empty").mkdir()
    with pytest.raises((StoreAuthorityError, StoreRecordError), match="empty nested"):
        store.install_state(Definition(ManifestObject).concretize(), stage, codec="Codec1", codec_version=1, interpretation={})

    with pytest.raises(StoreRecordError, match="codec"):
        StateVariantManifest.content_hash("not-valid!", 1, {}, ())


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
