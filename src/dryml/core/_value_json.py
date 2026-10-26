"""Private closed semantic-leaf codecs shared by Node and expression transport.

These codecs carry values, not legacy graph-reference envelopes. They never
construct Objects, read state, or open Stores; explicit type tags may resolve
their importable value types during decoding.
"""

from __future__ import annotations

import base64
import binascii
import json
import math
import sys
from collections.abc import Mapping
from enum import Enum

import numpy as np

from .cardinality import Cardinality, CardinalityKind
from .config import ConfigRef
from .dtype import DType
from .factory import FactorySpec
from .freeze import FrozenDict, FrozenList, FrozenNDArray, FrozenSet, FrozenTuple
from .reference_values import ObjectId
from .symbol import ImportRef, SourceSpec, resolve_symbol
from .tensor_spec import Dim, Dynamic, TensorSpec


_VALUE_GRAMMAR_VERSION = 3


class ReferenceJSONCodecError(ValueError):
    """Unsupported or malformed semantic value in a private transport codec."""


def _encoded_type(value: type, tag: str) -> dict[str, object]:
    module = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None)
    if (
            type(module) is not str
            or type(qualname) is not str
            or module == "__main__"
            or "<locals>" in qualname):
        raise ReferenceJSONCodecError(f"Reference JSON cannot encode local {tag} values.")
    resolved = sys.modules.get(module)
    for part in qualname.split("."):
        resolved = getattr(resolved, part, None)
    if resolved is not value:
        raise ReferenceJSONCodecError(
            f"Reference JSON {tag} import authority does not identify the encoded type."
        )
    return {"type": tag, "module": module, "qualname": qualname}


def _value_to_json(value: object, *, grammar: int) -> dict[str, object]:
    if grammar != _VALUE_GRAMMAR_VERSION:
        raise ReferenceJSONCodecError("Unsupported semantic value grammar.")
    if isinstance(value, ImportRef):
        return {"type": "import_ref", "module": value.module, "qualname": value.qualname}
    if isinstance(value, SourceSpec):
        return {
            "type": "source_spec",
            "kind": value.kind,
            "source": value.source,
            "name": value.name,
            "imports": {
                name: _value_to_json(reference, grammar=grammar)
                for name, reference in (value.imports or {}).items()
            },
        }
    if isinstance(value, FactorySpec):
        return {
            "type": "factory_spec",
            "target": _value_to_json(value.target, grammar=grammar),
            "args": [_value_to_json(item, grammar=grammar) for item in value.args],
            "kwargs": {
                key: _value_to_json(item, grammar=grammar)
                for key, item in value.kwargs.items()
            },
        }
    if isinstance(value, TensorSpec):
        def encode_dim(dimension):
            return "dynamic" if isinstance(dimension, Dim) else dimension

        return {
            "type": "tensor_spec",
            "dtype": str(value.dtype),
            "shape": None if value.shape is None else [encode_dim(item) for item in value.shape],
            "batch": encode_dim(value.batch) if value.batch is not None else None,
            "backend": None if value.backend is None else value.backend.value,
            "layout": value.layout.value,
            "axis_names": None if value.axis_names is None else list(value.axis_names),
            "batch_axis_name": value.batch_axis_name,
            "ragged_rank": value.ragged_rank,
            "row_splits_dtype": None if value.row_splits_dtype is None else str(value.row_splits_dtype),
            "sparse_format": value.sparse_format,
        }
    if isinstance(value, DType):
        if grammar < 3:
            raise ReferenceJSONCodecError(
                f"Reference JSON tag 'dtype' is not valid in v{grammar}."
            )
        if type(value.kind) is not str or (
                value.bits is not None and type(value.bits) is not int):
            raise ReferenceJSONCodecError("Reference JSON contains an invalid DType.")
        return {"type": "dtype", "kind": value.kind, "bits": value.bits}
    if isinstance(value, Cardinality):
        if value.kind is CardinalityKind.FINITE:
            if type(value.value) is not int or value.value < 0:
                raise ReferenceJSONCodecError("Reference JSON contains an invalid Cardinality.")
            kind, encoded = "finite", str(value.value)
        elif value.kind is CardinalityKind.INFINITE:
            kind, encoded = "infinite", None
        elif value.kind is CardinalityKind.UNKNOWN:
            kind, encoded = "unknown", None
        else:
            raise ReferenceJSONCodecError("Reference JSON contains an invalid Cardinality.")
        return {"type": "cardinality", "kind": kind, "value": encoded}
    if grammar >= 3 and isinstance(value, ConfigRef):
        return {
            "type": "config_ref",
            "key": value.key,
            "has_default": value.has_default,
            "default": _value_to_json(value.default, grammar=grammar) if value.has_default else None,
        }
    if grammar >= 3 and isinstance(value, ObjectId):
        return {
            "type": "object_id",
            "value": _value_to_json(value.to_data(), grammar=grammar),
        }
    if grammar >= 3 and isinstance(value, Enum):
        record = _encoded_type(type(value), "enum")
        record["member"] = value.name
        return record
    if grammar >= 3 and isinstance(value, type):
        module = getattr(value, "__module__", "")
        if module != "dryml.core" and not module.startswith("dryml.core."):
            raise ReferenceJSONCodecError(
                "Reference JSON supports naked types only from dryml.core."
            )
        return _encoded_type(value, "core_type")
    if value is None:
        return {"type": "null"}
    if type(value) is bool:
        return {"type": "bool", "value": value}
    if type(value) is int:
        return {"type": "int", "value": str(value)}
    if type(value) is float and math.isfinite(value):
        return {"type": "float", "value": value}
    if type(value) is str:
        return {"type": "str", "value": value}
    if grammar >= 3 and type(value) is bytes:
        return {"type": "bytes", "value": base64.b64encode(value).decode("ascii")}
    if grammar >= 3 and isinstance(value, np.generic):
        dtype = value.dtype
        if dtype.hasobject or dtype.fields is not None:
            raise ReferenceJSONCodecError(
                "Reference JSON cannot encode object or structured NumPy scalars."
            )
        return {
            "type": "numpy_scalar",
            "dtype": dtype.str,
            "bytes": base64.b64encode(value.tobytes()).decode("ascii"),
        }
    if isinstance(value, FrozenNDArray):
        if value.dtype.hasobject or value.dtype.fields is not None:
            raise ReferenceJSONCodecError(
                "Reference JSON cannot encode object or structured frozen arrays."
            )
        return {
            "type": "frozen_ndarray",
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "bytes": base64.b64encode(value.tobytes(order="C")).decode("ascii"),
        }
    if isinstance(value, FrozenList):
        return {
            "type": "frozen_list",
            "items": [_value_to_json(item, grammar=grammar) for item in value],
        }
    if isinstance(value, FrozenTuple):
        return {
            "type": "frozen_tuple",
            "items": [_value_to_json(item, grammar=grammar) for item in value],
        }
    if isinstance(value, FrozenSet):
        items = [_value_to_json(item, grammar=grammar) for item in value]
        items.sort(key=_json_sort_key)
        return {"type": "frozen_set", "items": items}
    if isinstance(value, FrozenDict):
        if grammar < 3:
            if not all(type(key) is str for key in value):
                raise ReferenceJSONCodecError(
                    "Legacy reference JSON has a non-string frozen mapping key."
                )
            items = [
                [key, _value_to_json(item, grammar=grammar)]
                for key, item in value.items()
            ]
        else:
            items = [
                [_value_to_json(key, grammar=grammar), _value_to_json(item, grammar=grammar)]
                for key, item in value.items()
            ]
        return {"type": "frozen_dict", "items": items}
    if isinstance(value, frozenset):
        items = [_value_to_json(item, grammar=grammar) for item in value]
        items.sort(key=_json_sort_key)
        return {"type": "frozenset", "items": items}
    if grammar >= 3 and isinstance(value, set):
        items = [_value_to_json(item, grammar=grammar) for item in value]
        items.sort(key=_json_sort_key)
        return {"type": "set", "items": items}
    if isinstance(value, list):
        return {"type": "list", "items": [_value_to_json(item, grammar=grammar) for item in value]}
    if isinstance(value, tuple):
        return {"type": "tuple", "items": [_value_to_json(item, grammar=grammar) for item in value]}
    if isinstance(value, Mapping):
        if grammar < 3:
            if not all(type(key) is str for key in value):
                raise ReferenceJSONCodecError(
                    "Legacy reference JSON has a non-string mapping key."
                )
            items = [
                [key, _value_to_json(item, grammar=grammar)]
                for key, item in value.items()
            ]
        else:
            if not all(type(key) in {str, int} for key in value):
                raise ReferenceJSONCodecError(
                    "Reference JSON mapping keys must be exact strings or integers."
                )
            items = [
                [_value_to_json(key, grammar=grammar), _value_to_json(item, grammar=grammar)]
                for key, item in value.items()
            ]
        return {"type": "dict", "items": items}
    raise ReferenceJSONCodecError(
        "Reference data contains unsupported portable value "
        f"{type(value).__module__}.{type(value).__qualname__}."
    )


def _json_sort_key(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _decode_integer(value: object) -> int:
    if type(value) is not str or not value or value == "-0" or value.startswith("+"):
        raise ReferenceJSONCodecError("Reference JSON integer is not canonical decimal text.")
    digits = value[1:] if value.startswith("-") else value
    if not digits.isdigit() or (len(digits) > 1 and digits.startswith("0")):
        raise ReferenceJSONCodecError("Reference JSON integer is not canonical decimal text.")
    return int(value)


def _decode_imported_type(
        value: Mapping, tag: str, fields: set[str],
) -> type:
    if set(value) != fields:
        raise ReferenceJSONCodecError(f"Reference JSON {tag} record has invalid fields.")
    try:
        target = resolve_symbol(ImportRef(value["module"], value["qualname"]))
    except Exception as error:
        raise ReferenceJSONCodecError(f"Reference JSON {tag} target cannot be resolved.") from error
    if not isinstance(target, type):
        raise ReferenceJSONCodecError(f"Reference JSON {tag} target is not a type.")
    return target


def _decode_mapping_items(value: object, *, grammar: int, frozen: bool) -> object:
    if not isinstance(value, list):
        raise ReferenceJSONCodecError("Reference JSON mapping record has invalid items.")
    pairs = []
    seen = set()
    for entry in value:
        if not isinstance(entry, list) or len(entry) != 2:
            raise ReferenceJSONCodecError("Reference JSON mapping entry is malformed.")
        if grammar < 3:
            key = entry[0]
            if type(key) is not str:
                raise ReferenceJSONCodecError("Legacy reference JSON mapping key is invalid.")
        else:
            key = _value_from_json(entry[0], grammar=grammar)
            if type(key) not in {str, int}:
                raise ReferenceJSONCodecError("Reference JSON mapping key is invalid.")
        if key in seen:
            raise ReferenceJSONCodecError("Reference JSON mapping repeats a key.")
        seen.add(key)
        pairs.append((key, _value_from_json(entry[1], grammar=grammar)))
    return FrozenDict(pairs) if frozen else dict(pairs)


def _value_from_json(value: object, *, grammar: int) -> object:
    if grammar != _VALUE_GRAMMAR_VERSION:
        raise ReferenceJSONCodecError("Unsupported reference JSON value grammar.")
    if not isinstance(value, Mapping) or not isinstance(value.get("type"), str):
        raise ReferenceJSONCodecError("Reference JSON value is not a closed tagged tree.")
    tag = value["type"]
    if grammar == 1 and tag not in {
            "null", "bool", "str", "float", "int", "list", "tuple",
            "import_ref", "source_spec", "factory_spec", "tensor_spec",
            "frozen_ndarray", "dict",
    }:
        raise ReferenceJSONCodecError(f"Reference JSON tag {tag!r} is not valid in v1.")
    if grammar == 2 and tag in {
            "bytes", "numpy_scalar", "set", "config_ref", "object_id",
            "object_ref", "state_ref", "enum", "core_type", "dtype",
    }:
        raise ReferenceJSONCodecError(f"Reference JSON tag {tag!r} is not valid in v2.")
    if tag == "null" and set(value) == {"type"}:
        return None
    if tag == "bool" and set(value) == {"type", "value"} and type(value["value"]) is bool:
        return value["value"]
    if tag == "str" and set(value) == {"type", "value"} and type(value["value"]) is str:
        return value["value"]
    if tag == "float" and set(value) == {"type", "value"} and type(value["value"]) is float and math.isfinite(value["value"]):
        return value["value"]
    if tag == "int" and set(value) == {"type", "value"}:
        return _decode_integer(value["value"])
    if tag == "bytes" and set(value) == {"type", "value"} and type(value["value"]) is str:
        try:
            return base64.b64decode(value["value"], validate=True)
        except (binascii.Error, ValueError) as error:
            raise ReferenceJSONCodecError("Reference JSON bytes record is malformed.") from error
    if tag in {"list", "tuple"} and set(value) == {"type", "items"}:
        if not isinstance(value["items"], list):
            raise ReferenceJSONCodecError("Reference JSON sequence has invalid items.")
        items = [_value_from_json(item, grammar=grammar) for item in value["items"]]
        return items if tag == "list" else tuple(items)
    if tag in {"frozen_list", "frozen_tuple"} and set(value) == {"type", "items"}:
        if not isinstance(value["items"], list):
            raise ReferenceJSONCodecError("Reference JSON frozen sequence has invalid items.")
        items = [_value_from_json(item, grammar=grammar) for item in value["items"]]
        if tag == "frozen_list":
            return FrozenList(items)
        return FrozenTuple(items)
    if tag in {"frozen_set", "frozenset", "set"} and set(value) == {"type", "items"}:
        if not isinstance(value["items"], list):
            raise ReferenceJSONCodecError("Reference JSON set record has invalid items.")
        items = [_value_from_json(item, grammar=grammar) for item in value["items"]]
        try:
            if tag == "frozen_set":
                result = FrozenSet(items)
            elif tag == "frozenset":
                result = frozenset(items)
            else:
                result = set(items)
        except TypeError as error:
            raise ReferenceJSONCodecError("Reference JSON set member is unhashable.") from error
        if len(result) != len(items):
            raise ReferenceJSONCodecError("Reference JSON set repeats a member.")
        return result
    if tag == "frozen_dict" and set(value) == {"type", "items"}:
        return _decode_mapping_items(value["items"], grammar=grammar, frozen=True)
    if tag == "dict" and set(value) == {"type", "items"}:
        return _decode_mapping_items(value["items"], grammar=grammar, frozen=False)
    if tag == "import_ref" and set(value) == {"type", "module", "qualname"}:
        try:
            return ImportRef(value["module"], value["qualname"])
        except (TypeError, ValueError) as error:
            raise ReferenceJSONCodecError("Reference JSON import reference is invalid.") from error
    if tag == "source_spec" and set(value) == {"type", "kind", "source", "name", "imports"}:
        if not isinstance(value["imports"], Mapping):
            raise ReferenceJSONCodecError("Reference JSON source imports are invalid.")
        try:
            imports = {
                name: _value_from_json(reference, grammar=grammar)
                for name, reference in value["imports"].items()
            }
            if not all(
                    type(name) is str and isinstance(reference, ImportRef)
                    for name, reference in imports.items()):
                raise ValueError("invalid imports")
            return SourceSpec(value["kind"], value["source"], value["name"], imports)
        except (TypeError, ValueError) as error:
            raise ReferenceJSONCodecError("Reference JSON source specification is invalid.") from error
    if tag == "factory_spec" and set(value) == {"type", "target", "args", "kwargs"}:
        if not isinstance(value["args"], list) or not isinstance(value["kwargs"], Mapping):
            raise ReferenceJSONCodecError("Reference JSON factory fields are invalid.")
        try:
            target = _value_from_json(value["target"], grammar=grammar)
            args = tuple(_value_from_json(item, grammar=grammar) for item in value["args"])
            kwargs = FrozenDict(
                (key, _value_from_json(item, grammar=grammar))
                for key, item in value["kwargs"].items()
            )
            if not all(type(key) is str for key in kwargs):
                raise ValueError("invalid keyword")
            if grammar == 1:
                return FactorySpec(target, *args, **dict(kwargs.items()))
            return FactorySpec._from_symbolic_parts(target, args, kwargs)
        except (TypeError, ValueError) as error:
            raise ReferenceJSONCodecError("Reference JSON factory specification is invalid.") from error
    if tag == "tensor_spec" and set(value) == {
        "type", "dtype", "shape", "batch", "backend", "layout", "axis_names",
        "batch_axis_name", "ragged_rank", "row_splits_dtype", "sparse_format",
    }:
        def decode_dim(dimension):
            if dimension == "dynamic":
                return Dynamic
            if type(dimension) is int:
                return dimension
            raise ValueError("invalid dimension")

        try:
            shape = value["shape"]
            if shape is not None:
                if not isinstance(shape, list):
                    raise ValueError("invalid shape")
                shape = tuple(decode_dim(item) for item in shape)
            batch = value["batch"]
            if batch is not None:
                batch = decode_dim(batch)
            axis_names = value["axis_names"]
            if axis_names is not None and not isinstance(axis_names, list):
                raise ValueError("invalid axis names")
            return TensorSpec(
                value["dtype"], shape=shape, batch=batch, backend=value["backend"],
                layout=value["layout"], axis_names=axis_names,
                batch_axis_name=value["batch_axis_name"], ragged_rank=value["ragged_rank"],
                row_splits_dtype=value["row_splits_dtype"], sparse_format=value["sparse_format"],
            )
        except (TypeError, ValueError) as error:
            raise ReferenceJSONCodecError("Reference JSON tensor specification is invalid.") from error
    if tag == "dtype" and set(value) == {"type", "kind", "bits"}:
        kind, bits = value["kind"], value["bits"]
        if type(kind) is str and (bits is None or type(bits) is int):
            try:
                return DType(kind, bits)
            except ValueError as error:
                raise ReferenceJSONCodecError("Reference JSON DType is invalid.") from error
        raise ReferenceJSONCodecError("Reference JSON DType is invalid.")
    if tag == "cardinality" and set(value) == {"type", "kind", "value"}:
        kind, encoded = value["kind"], value["value"]
        if kind == "finite" and type(encoded) is str:
            count = _decode_integer(encoded)
            if count >= 0:
                return Cardinality.finite(count)
        elif kind == "infinite" and encoded is None:
            return Cardinality.INFINITE
        elif kind == "unknown" and encoded is None:
            return Cardinality.UNKNOWN
        raise ReferenceJSONCodecError("Reference JSON Cardinality is invalid.")
    if tag == "config_ref" and set(value) == {"type", "key", "has_default", "default"}:
        if type(value["has_default"]) is not bool:
            raise ReferenceJSONCodecError("Reference JSON ConfigRef is invalid.")
        try:
            if value["has_default"]:
                return ConfigRef(
                    value["key"], _value_from_json(value["default"], grammar=grammar),
                )
            if value["default"] is not None:
                raise ValueError("unexpected default")
            return ConfigRef(value["key"])
        except (TypeError, ValueError) as error:
            raise ReferenceJSONCodecError("Reference JSON ConfigRef is invalid.") from error
    if tag == "object_id" and set(value) == {"type", "value"}:
        decoded = _value_from_json(value["value"], grammar=grammar)
        try:
            return ObjectId.from_data(decoded)
        except (TypeError, ValueError) as error:
            raise ReferenceJSONCodecError(f"Reference JSON {tag} is invalid.") from error
    if tag == "core_type":
        target = _decode_imported_type(
            value, "core type", {"type", "module", "qualname"},
        )
        module = getattr(target, "__module__", "")
        if module != "dryml.core" and not module.startswith("dryml.core."):
            raise ReferenceJSONCodecError("Reference JSON core type is outside dryml.core.")
        return target
    if tag == "enum" and set(value) == {"type", "module", "qualname", "member"}:
        target = _decode_imported_type(
            value, "Enum", {"type", "module", "qualname", "member"},
        )
        if not issubclass(target, Enum) or type(value["member"]) is not str:
            raise ReferenceJSONCodecError("Reference JSON Enum record is invalid.")
        try:
            return target[value["member"]]
        except KeyError as error:
            raise ReferenceJSONCodecError("Reference JSON Enum member is invalid.") from error
    if tag == "numpy_scalar" and set(value) == {"type", "dtype", "bytes"}:
        if type(value["dtype"]) is not str or type(value["bytes"]) is not str:
            raise ReferenceJSONCodecError("Reference JSON NumPy scalar fields are invalid.")
        try:
            dtype = np.dtype(value["dtype"])
            raw = base64.b64decode(value["bytes"], validate=True)
            if dtype.hasobject or dtype.fields is not None or dtype.str != value["dtype"]:
                raise ValueError("unsupported dtype")
            if len(raw) != dtype.itemsize:
                raise ValueError("byte length")
            return np.frombuffer(raw, dtype=dtype)[0]
        except (binascii.Error, TypeError, ValueError) as error:
            raise ReferenceJSONCodecError("Reference JSON NumPy scalar is malformed.") from error
    if tag == "frozen_ndarray" and set(value) == {"type", "dtype", "shape", "bytes"}:
        if type(value["dtype"]) is not str or not isinstance(value["shape"], list) or type(value["bytes"]) is not str:
            raise ReferenceJSONCodecError("Reference JSON frozen array fields are invalid.")
        if any(type(size) is not int or size < 0 for size in value["shape"]):
            raise ReferenceJSONCodecError("Reference JSON frozen array shape is invalid.")
        try:
            dtype = np.dtype(value["dtype"])
            raw = base64.b64decode(value["bytes"], validate=True)
            if dtype.hasobject or dtype.fields is not None or dtype.str != value["dtype"]:
                raise ValueError("unsupported dtype")
            if math.prod(value["shape"]) * dtype.itemsize != len(raw):
                raise ValueError("byte length")
            array = np.frombuffer(raw, dtype=dtype).reshape(tuple(value["shape"]))
            return FrozenNDArray.from_array(array)
        except (binascii.Error, TypeError, ValueError) as error:
            raise ReferenceJSONCodecError("Reference JSON frozen array is malformed.") from error
    raise ReferenceJSONCodecError(f"Reference JSON value tag {tag!r} is unknown or malformed.")
