"""Current Store framing and dependency-light retained content contracts."""

import pytest

from dryml.core.store.records import StoreFormatRecord, StoreRecordError


def test_record_codecs_reject_unknown_version_and_trailing_bytes():
    payload = StoreFormatRecord().to_bytes()
    with pytest.raises(StoreRecordError, match="trailing"):
        StoreFormatRecord.from_bytes(payload + b"trailing")
    data = StoreFormatRecord().to_data()
    data["version"] = 0
    with pytest.raises(StoreRecordError, match="Unsupported"):
        StoreFormatRecord.from_data(data)


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
