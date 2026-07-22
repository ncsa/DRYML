import pytest

from dryml.formats.errors import ReferenceParseError
from dryml.formats.ids import content_id
from dryml.formats.refs import (
    format_cdef_id,
    format_ref_cdef,
    is_cdef_id,
    is_literal_escape,
    is_ref_cdef,
    is_reserved_ref,
    literal_escape,
    parse_cdef_id,
    parse_ref_cdef,
    parse_reserved_ref,
    unwrap_literal_escape,
)


def test_valid_cdef_ids_parse_and_format():
    raw = format_cdef_id("abcdef0123456789", schema_version=4)
    parsed = parse_cdef_id(raw)

    assert raw == "cdef-v4-abcdef0123456789"
    assert parsed.schema_version == 4
    assert parsed.digest == "abcdef0123456789"
    assert parsed.raw == raw
    assert is_cdef_id(raw)


@pytest.mark.parametrize("value", ["cdef-v0-abcdef0123456789", "cdef-v4-abc", "cdef-v4-ABCDEF0123456789"])
def test_invalid_cdef_ids_reject(value):
    with pytest.raises(ReferenceParseError, match="invalid CDef ID"):
        parse_cdef_id(value)


def test_valid_ref_cdef_values_parse_and_format():
    cdef_id = "cdef-v4-abcdef0123456789"
    raw = format_ref_cdef(cdef_id)
    parsed = parse_ref_cdef(raw)

    assert raw == "ref(cdef-v4-abcdef0123456789)"
    assert parsed.raw == cdef_id
    assert is_ref_cdef(raw)


@pytest.mark.parametrize("value", ["ref(record-v1-abc)", "ref(cdef-v4-abc)", "ref(cdef-v4-ABCDEF0123456789)"])
def test_invalid_ref_values_reject(value):
    with pytest.raises(ReferenceParseError, match="invalid CDef ref"):
        parse_ref_cdef(value)


def test_parse_reserved_ref_handles_cdef_and_ref_cdef():
    cdef = parse_reserved_ref("cdef-v4-abcdef0123456789")
    ref = parse_reserved_ref("ref(cdef-v4-abcdef0123456789)")

    assert cdef.kind == "cdef"
    assert cdef.target == "cdef-v4-abcdef0123456789"
    assert ref.kind == "ref_cdef"
    assert ref.target == "cdef-v4-abcdef0123456789"


def test_generic_content_ids_parse():
    for prefix in [
        "record",
        "spec",
        "env",
        "envreq",
        "world",
        "worldreq",
        "runtime",
        "repr",
        "op",
        "annotation",
        "blob",
        "envrec",
        "envspec",
        "envlock",
    ]:
        raw = content_id(prefix, 1, {"prefix": prefix})
        parsed = parse_reserved_ref(raw)
        assert parsed.kind == "content_id"
        assert parsed.prefix == prefix
        assert parsed.target == raw


@pytest.mark.parametrize(
    "value",
    [
        "record-v1-not-a-valid-digest",
        "spec-v1-xyz",
        "env-v0-" + "0" * 64,
        "world-v1-" + "0" * 63,
        "repr-v1-" + "g" * 64,
        "envrec-v1-not-a-valid-digest",
    ],
)
def test_malformed_reserved_content_refs_reject(value):
    with pytest.raises(ReferenceParseError, match="invalid reserved content reference"):
        parse_reserved_ref(value)


def test_malformed_non_reserved_content_like_strings_remain_ordinary():
    assert parse_reserved_ref("ordinary-v1-not-a-reserved-ref") is None


def test_literal_escape_prevents_ref_interpretation():
    escaped = literal_escape("cdef-v4-abcdef0123456789")
    parsed = parse_reserved_ref(escaped)

    assert escaped == {"$literal": "cdef-v4-abcdef0123456789"}
    assert is_literal_escape(escaped)
    assert parsed.kind == "literal"
    assert parsed.target is None
    assert unwrap_literal_escape(escaped) == "cdef-v4-abcdef0123456789"


def test_literal_escape_rejects_ambiguous_shapes():
    with pytest.raises(ReferenceParseError, match="only \\$literal"):
        parse_reserved_ref({"$literal": "x", "extra": True})


def test_literal_escape_can_wrap_non_string_json_values():
    escaped = literal_escape({"a": [1, True]})
    assert escaped == {"$literal": {"a": [1, True]}}
    assert unwrap_literal_escape(escaped) == {"a": [1, True]}


def test_unreserved_values_return_none_or_false():
    assert parse_reserved_ref("ordinary") is None
    assert parse_reserved_ref({"not_literal": "cdef-v4-abcdef0123456789"}) is None
    assert not is_reserved_ref("ordinary")
