from pathlib import Path

import hashlib

import dill
import pytest

from dryml.core import Object
from dryml.core.store.records import (
    DefinitionRecord, LocalStateManifest, StoreFormatRecord, StoreRecordError,
)


class RecordObject(Object):
    pass


def test_definition_record_round_trips_and_recomputes_all_digest_fields():
    record = DefinitionRecord(RecordObject().definition)

    decoded = DefinitionRecord.from_bytes(record.to_bytes())

    assert decoded.digest == record.digest
    assert decoded.definition.graph_equal(record.definition)
    data = record.to_data()
    data["graph_hash"] = "0" * 64
    with pytest.raises(StoreRecordError, match="hash fields"):
        DefinitionRecord.from_data(data)


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
