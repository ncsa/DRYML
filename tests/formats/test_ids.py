import pytest

from dryml.formats.errors import ContentIDError
from dryml.formats.ids import (
    content_id,
    is_content_id,
    parse_content_id,
    stable_hash,
    validate_id_prefix,
    validate_schema_version,
)


def test_stable_hash_is_stable_under_dict_ordering():
    assert stable_hash({"b": 2, "a": 1}) == stable_hash({"a": 1, "b": 2})


def test_content_id_is_stable_under_dict_ordering():
    first = content_id("envrec", 1, {"b": 2, "a": 1})
    second = content_id("envrec", 1, {"a": 1, "b": 2})
    assert first == second
    assert first.startswith("envrec-v1-")


def test_content_id_changes_when_data_or_schema_version_changes():
    first = content_id("envrec", 1, {"value": 1})
    assert first != content_id("envrec", 1, {"value": 2})
    assert first != content_id("envrec", 2, {"value": 1})


def test_parse_content_id_roundtrips_valid_ids():
    raw = content_id("record", 3, {"value": 1})
    parts = parse_content_id(raw)

    assert parts.prefix == "record"
    assert parts.schema_version == 3
    assert len(parts.digest) == 64
    assert parts.raw == raw
    assert is_content_id(raw, prefix="record")
    assert not is_content_id(raw, prefix="spec")


@pytest.mark.parametrize("prefix", ["EnvRec", "1env", "env-rec", ""])
def test_invalid_prefixes_reject(prefix):
    with pytest.raises(ContentIDError, match="prefix"):
        validate_id_prefix(prefix)


@pytest.mark.parametrize("schema_version", [0, -1, True, "1"])
def test_invalid_schema_versions_reject(schema_version):
    with pytest.raises(ContentIDError, match="schema version"):
        validate_schema_version(schema_version)


@pytest.mark.parametrize(
    "value",
    [
        "envrec-v0-" + "0" * 64,
        "EnvRec-v1-" + "0" * 64,
        "envrec-v1-" + "0" * 63,
        "envrec-v1-" + "g" * 64,
    ],
)
def test_invalid_content_ids_reject(value):
    with pytest.raises(ContentIDError, match="invalid content ID"):
        parse_content_id(value)


def test_future_prefixes_parse_as_content_ids():
    for prefix in ["record", "spec", "world", "runtime", "repr", "op", "annotation", "blob"]:
        assert parse_content_id(content_id(prefix, 1, {"x": prefix})).prefix == prefix
