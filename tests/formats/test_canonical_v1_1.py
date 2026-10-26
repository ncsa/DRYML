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


def test_encoder_projection_detaches_once_with_identical_validation_bounds():
    from collections import UserDict
    from types import MappingProxyType
    from dryml.formats import CanonicalJSONError, deep_freeze_json, json_ready

    source = UserDict({"nested": MappingProxyType({"items": [1, True, None, -0.0]})})
    ready = json_ready(source)
    assert ready == {"nested": {"items": [1, True, None, -0.0]}}
    ready["nested"]["items"].append("changed")
    assert len(source["nested"]["items"]) == 4
    for project in (deep_freeze_json, json_ready):
        for bounds in ({"max_depth": 1}, {"max_nodes": 3}, {"max_entries": 2}, {"max_string": 2}):
            with pytest.raises(CanonicalJSONError):
                project(source, **bounds)
        cycle = []
        cycle.append(cycle)
        with pytest.raises(CanonicalJSONError, match="cyclic"):
            project(cycle)
