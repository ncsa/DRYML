"""Closed codec, aliasing, large integer and host-only native evidence."""

import builtins
import json

import numpy as np
import pytest

from dryml.data.continuation import HostBuffer, encode_value, decode_value, rebuild_native
from dryml.data import ContinuationFormatError, UnsupportedContinuationError


def test_closed_value_grammar_aliases_and_pcg64_128_bit_state():
    array = np.arange(7)
    value = [None, True, 2**128 - 1, -2**127, float("nan"), complex(2, -1), "hello", b"bytes",
             np.float32(2), {"first": array, "again": array}, np.random.default_rng(7).bit_generator.state]
    metadata, payloads = encode_value(value)
    decoded = decode_value(metadata, payloads)
    assert decoded[2] == value[2] and decoded[3] == value[3]
    assert np.isnan(decoded[4]) and decoded[5] == value[5]
    assert decoded[9]["first"] is decoded[9]["again"]
    assert decoded[9]["first"] is not array
    first, second = np.random.default_rng(), np.random.default_rng()
    first.bit_generator.state = decoded[-1]
    second.bit_generator.state = value[-1]
    np.testing.assert_equal(first.integers(0, 100, 12), second.integers(0, 100, 12))


def test_host_only_decode_blocks_optional_imports_and_rejects_tags(monkeypatch):
    array = np.arange(4, dtype="int64")
    value = HostBuffer("torch", array.dtype.str, array.shape, "C", "cpu", array.tobytes())
    metadata, payloads = encode_value(value)
    original = builtins.__import__
    def imports(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "tensorflow", "jax", "jaxlib"}:
            raise ImportError("backend blocked")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", imports)
    assert decode_value(metadata, payloads) == value
    with pytest.raises(ImportError, match="backend blocked"):
        rebuild_native(decode_value(metadata, payloads))
    with pytest.raises(ContinuationFormatError):
        decode_value(metadata.replace(b'"native"', b'"mystery"'), payloads)


@pytest.mark.parametrize("backend", ["torch", "tensorflow", "jax"])
def test_registered_dense_native_buffers_rebuild_exact_values_and_aliases(backend):
    array = np.arange(6, dtype=np.float32).reshape(2, 3)
    if backend == "torch":
        native = pytest.importorskip("torch").as_tensor(array)
    elif backend == "tensorflow":
        native = pytest.importorskip("tensorflow").convert_to_tensor(array)
    else:
        native = pytest.importorskip("jax.numpy").asarray(array)
    metadata, payloads = encode_value({"first": native, "alias": native})
    decoded = decode_value(metadata, payloads)
    assert type(decoded["first"]) is HostBuffer
    assert decoded["first"] is decoded["alias"]
    rebuilt = rebuild_native(decoded)
    assert rebuilt["first"] is rebuilt["alias"]
    actual = np.asarray(rebuilt["first"])
    assert actual.dtype == array.dtype
    np.testing.assert_array_equal(actual, array)


@pytest.mark.parametrize("value", [object(), iter([1]), np.array([object()], dtype=object), {1: "bad"}])
def test_unsupported_live_values_never_get_encoded(value):
    with pytest.raises(UnsupportedContinuationError):
        encode_value(value)


@pytest.mark.parametrize("metadata,payloads", [
    (b'["ref",0]', ()), (b'["array",0,"O",[1],0]', (b"12345678",)),
    (b'["array",0,"<i8",[-1],0]', (b"",)), (b'["array",0,"<i8",[4],0]', (b"",)),
    (b'["int",true]', ()), (b'["none",0]', ()), (b'["none"]', (b"extra",)),
    (b'["dict",0,[["x",["none"]],["x",["none"]]]]', ()),
])
def test_malformed_tags_bounds_and_payloads_reject_before_reconstruction(metadata, payloads):
    with pytest.raises(ContinuationFormatError):
        decode_value(metadata, payloads)
