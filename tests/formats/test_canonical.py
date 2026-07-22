from types import MappingProxyType

import pytest

from dryml.formats.canonical import (
    canonical_json_bytes,
    canonical_json_dumps,
    canonical_json_load_bytes,
    canonical_json_loads,
    deep_freeze_json,
    json_ready,
)
from dryml.formats.errors import CanonicalJSONError, DrymlFormatError


def test_errors_preserve_context_defensively():
    context = {"code": "x"}
    err = DrymlFormatError("boom", context=context)
    context["code"] = "mutated"
    assert str(err) == "boom"
    assert err.context == {"code": "x"}


def test_equivalent_dict_orderings_produce_identical_dumps_and_bytes():
    left = {"b": [2, 1], "a": {"y": 2, "x": 1}}
    right = {"a": {"x": 1, "y": 2}, "b": [2, 1]}

    assert canonical_json_dumps(left) == canonical_json_dumps(right)
    assert canonical_json_bytes(left) == canonical_json_bytes(right)


def test_deep_freeze_canonicalizes_nested_containers():
    frozen = deep_freeze_json({"items": {"b", "a"}, "nested": [1, {"x": True}]})

    assert canonical_json_dumps(frozen) == '{"items":["a","b"],"nested":[1,{"x":true}]}'
    with pytest.raises(TypeError):
        frozen["nested"] = ()
    with pytest.raises((AttributeError, TypeError)):
        frozen["nested"][1]["x"] = False


def test_mapping_proxy_converts_to_json_ready_dict():
    ready = json_ready(MappingProxyType({"b": 2, "a": 1}))
    assert ready == {"a": 1, "b": 2}
    assert isinstance(ready, dict)


def test_non_string_mapping_keys_raise_canonical_error():
    with pytest.raises(CanonicalJSONError, match="mapping keys must be strings"):
        deep_freeze_json({1: "integer-key"})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_floats_raise_canonical_error(value):
    with pytest.raises(CanonicalJSONError, match="floats must be finite"):
        canonical_json_dumps({"value": value})


def test_arbitrary_objects_are_rejected():
    with pytest.raises(CanonicalJSONError, match="not JSON serializable"):
        canonical_json_dumps({"bad": object()})


def test_load_helpers_roundtrip_json_data():
    data = {"b": [2, 1], "a": {"x": True}}
    text = canonical_json_dumps(data)
    raw = canonical_json_bytes(data)

    assert canonical_json_loads(text) == {"a": {"x": True}, "b": [2, 1]}
    assert canonical_json_load_bytes(raw) == {"a": {"x": True}, "b": [2, 1]}


@pytest.mark.parametrize("text", ["NaN", "Infinity", "-Infinity", '{"value": NaN}'])
def test_load_helpers_reject_non_finite_constants(text):
    with pytest.raises(CanonicalJSONError, match="non-finite"):
        canonical_json_loads(text)


def test_load_helpers_reject_duplicate_object_keys():
    with pytest.raises(CanonicalJSONError, match="duplicate key"):
        canonical_json_loads('{"a":1,"a":2}')
