from collections.abc import Mapping

import pytest


def test_v1_1_canonical_codec_is_closed_and_duplicate_aware():
    from dryml.formats import CanonicalJSONError, canonical_json_bytes, canonical_json_loads

    assert canonical_json_bytes({"b": 1, "a": [True]}) == b'{"a":[true],"b":1}'
    with pytest.raises(CanonicalJSONError, match="duplicate"):
        canonical_json_loads('{"value":1,"value":2}')
    with pytest.raises(CanonicalJSONError, match="finite"):
        canonical_json_bytes({"value": float("nan")})


def test_canonical_decoding_returns_a_deeply_immutable_projection():
    from dryml.formats import canonical_json_load_bytes

    decoded = canonical_json_load_bytes(b'{"nested":{"items":[1,2]}}')

    assert isinstance(decoded, Mapping)
    assert decoded["nested"]["items"] == (1, 2)
    with pytest.raises(TypeError):
        decoded["nested"]["new"] = True
