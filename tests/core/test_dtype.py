import pickle

import numpy as np
import pytest

from dryml.core2.dtype import DType, normalize_dtype
from dryml.core2.utils.stable_hash import stable_hash_function


def test_dtype_equality_hash_and_pickle():
    d1 = DType("float", 32)
    d2 = normalize_dtype("float32")
    d3 = normalize_dtype(np.float32)
    d4 = normalize_dtype(np.dtype("float32"))

    assert d1 == d2 == d3 == d4
    assert hash(d1) == hash(d2) == hash(d3) == hash(d4)
    assert pickle.loads(pickle.dumps(d1)) == d1


def test_dtype_name_and_str():
    assert DType("float", 32).name == "float32"
    assert DType("int", 64).name == "int64"
    assert DType("bool").name == "bool"
    assert DType("string").name == "string"

    assert str(DType("float", 32)) == "float32"
    assert str(DType("bool")) == "bool"


@pytest.mark.parametrize(
    ("kind", "bits", "name"),
    [
        ("int", 8, "int8"),
        ("int", 16, "int16"),
        ("int", 32, "int32"),
        ("int", 64, "int64"),
        ("uint", 8, "uint8"),
        ("uint", 16, "uint16"),
        ("uint", 32, "uint32"),
        ("uint", 64, "uint64"),
        ("float", 16, "float16"),
        ("float", 32, "float32"),
        ("float", 64, "float64"),
        ("bfloat", 16, "bfloat16"),
        ("complex", 64, "complex64"),
        ("complex", 128, "complex128"),
    ],
)
def test_dtype_names(kind, bits, name):
    assert DType(kind, bits).name == name


def test_bool_string_object_do_not_take_bits():
    with pytest.raises(ValueError):
        DType("bool", 8)

    with pytest.raises(ValueError):
        DType("string", 8)

    with pytest.raises(ValueError):
        DType("object", 8)


@pytest.mark.parametrize("kind", ["int", "uint", "float", "bfloat", "complex"])
def test_numeric_dtypes_require_positive_bits(kind):
    with pytest.raises(ValueError):
        DType(kind, None)

    with pytest.raises(ValueError):
        DType(kind, 0)

    with pytest.raises(ValueError):
        DType(kind, -1)


def test_invalid_dtype_kind():
    with pytest.raises(ValueError):
        DType("weird", 32)


def test_normalize_dtype_from_canonical_strings():
    assert normalize_dtype("float32") == DType("float", 32)
    assert normalize_dtype("float64") == DType("float", 64)
    assert normalize_dtype("int32") == DType("int", 32)
    assert normalize_dtype("uint8") == DType("uint", 8)
    assert normalize_dtype("bool") == DType("bool")
    assert normalize_dtype("string") == DType("string")
    assert normalize_dtype("object") == DType("object")
    assert normalize_dtype("bfloat16") == DType("bfloat", 16)
    assert normalize_dtype("complex128") == DType("complex", 128)


def test_normalize_dtype_rejects_unknown_string():
    with pytest.raises((TypeError, ValueError)):
        normalize_dtype("madeup32")


def test_normalize_dtype_string_returns_dtype():
    d = normalize_dtype("float32")
    assert isinstance(d, DType)
    assert d == DType("float", 32)


def test_normalize_dtype_accepts_dtype_instance():
    d = DType("float", 32)
    assert normalize_dtype(d) is d


def test_dtype_not_equal_to_raw_string():
    assert DType("float", 32) != "float32"


def test_dtype_hashing():
    stable_hash_function(DType("float", 32))
