from pathlib import Path

import pytest

from dryml.core import Object
from dryml.core.store.records import DefinitionRecord, StoreFormatRecord, StoreRecordError


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
