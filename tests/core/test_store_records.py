from pathlib import Path

import hashlib

import dill
import pytest

from dryml.core import Definition, Object
from dryml.core.store.records import (
    DefinitionRecord, LocalStateManifest, StoreFormatRecord, StoreRecordError,
)


class RecordObject(Object):
    pass


class MappingRecordObject(Object):
    def __init__(self, config):
        self.config = config


def test_definition_record_round_trips_and_recomputes_all_digest_fields():
    record = DefinitionRecord(RecordObject().definition)

    decoded = DefinitionRecord.from_bytes(record.to_bytes())

    assert decoded.digest == record.digest
    assert decoded.definition.graph_equal(record.definition)
    data = record.to_data()
    data["graph_hash"] = "0" * 64
    with pytest.raises(StoreRecordError, match="hash fields"):
        DefinitionRecord.from_data(data)


def test_definition_record_round_trip_preserves_mapping_identity():
    definition = Definition(
        MappingRecordObject,
        config={"batch_size": 32, "seed": 7, "dtype": "float32"},
    ).concretize()

    decoded = DefinitionRecord.from_bytes(DefinitionRecord(definition).to_bytes())

    assert decoded.definition == definition
    assert decoded.definition.stable_hash() == definition.stable_hash()


def test_record_codecs_reject_unknown_version_and_trailing_bytes():
    payload = StoreFormatRecord().to_bytes()
    with pytest.raises(StoreRecordError, match="trailing"):
        StoreFormatRecord.from_bytes(payload + b"trailing")
    data = StoreFormatRecord().to_data()
    data["version"] = 0
    with pytest.raises(StoreRecordError, match="Unsupported"):
        StoreFormatRecord.from_data(data)


def test_v2_local_state_manifest_fixture_preserves_identity_and_is_eager(tmp_path):
    payload = b"v2 payload"
    digest = hashlib.sha256(payload).hexdigest()
    record = DefinitionRecord(RecordObject().definition)
    data = {
        "schema": "local-state-manifest",
        "version": 2,
        "codec": "Codec1",
        "graph_hash": record.graph_hash,
        "definition_digest": record.digest,
        "definition_file_digest": "0" * 64,
        "local_digest": hashlib.sha256(
            b'dryml-local-state-manifest-v2\0{"codec":"Codec1","files":[["payload",10,"'
            + digest.encode("ascii") + b'"]]}'
        ).hexdigest(),
        "files": [{"path": "payload", "size": len(payload), "digest": digest}],
    }
    fixture = b"DRYML-STORE-RECORD/local-state-manifest/1\n" + dill.dumps(data, protocol=5)

    manifest = LocalStateManifest.from_bytes(fixture)
    payload_path = tmp_path / "payload"
    payload_path.write_bytes(payload)

    assert manifest.version == 2
    assert manifest.deferred_paths == ()
    assert manifest.state_hash == f"Codec1-{data['local_digest']}"
    manifest.validate_payload(tmp_path, defer_payload=True)
    payload_path.write_bytes(b"v2 corrupt")
    with pytest.raises(StoreRecordError, match="exactly match"):
        manifest.validate_payload(tmp_path, defer_payload=True)


def test_json_fact_representation_is_detached_and_conservatively_charged():
    """Owned-size estimation covers all prospective tagged containers and leaves."""
    import sys
    from dryml.formats import canonical_json_load_bytes
    from dryml.core._content_facts import _primitive_size, _freeze_json_fact, _thaw_json_fact

    parsed = canonical_json_load_bytes(b'{"a":[{"nested":[1,true,null,2.5]},"text"],"b":[]}', max_depth=64)
    prepared = _freeze_json_fact(parsed)
    def owned_size(value):
        size = sys.getsizeof(value)
        return size + sum(owned_size(item) for item in value) if type(value) is tuple else size
    assert _primitive_size(parsed) >= owned_size(prepared)
    first, second = _thaw_json_fact(prepared), _thaw_json_fact(prepared)
    assert first == second == parsed
    assert first is not second and first["a"] is not second["a"]
    assert first["a"][0] is not second["a"][0]
    with pytest.raises(TypeError):
        first["a"][0]["nested"] = ()


def test_valid_large_json_uses_uncached_parser_working_memory(tmp_path):
    """Retention bounds are not input rejection or a decoding-memory ceiling."""
    from dryml.core import Repo
    from dryml.core.store.dir import DirStore

    path = tmp_path / "metadata.json"
    path.write_bytes(b"[" + b",".join([b'"' + b"x" * 100 + b'"'] * 25000) + b"]")
    repo = Repo()
    with repo._read_cut() as cut:
        value = DirStore._read_json(str(path), cut=cut, kind="metadata.json")
        assert len(value) == 25000
        assert repo._validation_evidence.accounted_bytes == 0
    assert not repo._validation_evidence._resident
