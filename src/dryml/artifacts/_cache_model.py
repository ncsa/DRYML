"""Private logical tree and manifest helpers for cached Datasets."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from dryml.core.backend import Backend
from dryml.core.tensor_spec import Dim, Layout, SpecTree, TensorSpec, is_spec_tree


_FORMAT = "dryml.cached-dataset"
_VERSION = 2


class CacheIntegrityError(ValueError):
    """Report invalid, incompatible, or corrupt CachedDataset content.

    Args:
        message: Bounded diagnostic describing the rejected metadata or payload.

    This error identifies a failed integrity check; it does not repair content,
    fall back to another codec, or recompute from the retained source.
    """


def flatten_spec(spec: SpecTree) -> tuple[TensorSpec, ...]:
    """Validate a dense SpecTree and return its leaves in structural order."""

    if not is_spec_tree(spec):
        raise ValueError("CachedDataset source spec is not a valid SpecTree.")
    leaves: list[TensorSpec] = []

    def visit(value: object) -> None:
        if isinstance(value, TensorSpec):
            if value.layout is not Layout.DENSE:
                raise ValueError("CachedDataset supports only dense TensorSpec leaves.")
            if value.dtype.kind == "object":
                raise ValueError("CachedDataset does not support object dtype leaves.")
            leaves.append(value)
        elif isinstance(value, Mapping):
            for key, item in value.items():
                if type(key) not in (str, int, bool):
                    raise TypeError("CachedDataset dictionary keys must be str, int, or bool.")
                visit(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)
        else:
            raise ValueError("CachedDataset source spec is malformed.")

    visit(spec)
    return tuple(leaves)


def numpy_spec(spec: SpecTree) -> SpecTree:
    """Return the NumPy-output form of a validated logical SpecTree."""

    def visit(value: object):
        if isinstance(value, TensorSpec):
            return replace(value, backend=Backend.numpy)
        if isinstance(value, dict):
            return {key: visit(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(visit(item) for item in value)
        if isinstance(value, list):
            return [visit(item) for item in value]
        raise ValueError("CachedDataset source spec is malformed.")

    flatten_spec(spec)
    return visit(spec)


def normalize_value(value: object, spec: SpecTree) -> tuple[object, tuple[object, ...]]:
    """Require exact container structure and return normalized leaves in order."""

    leaves: list[object] = []

    def visit(item: object, expected: object):
        if isinstance(expected, TensorSpec):
            leaves.append(item)
            return item
        if isinstance(expected, dict):
            if not isinstance(item, dict) or tuple(item.keys()) != tuple(expected.keys()):
                raise ValueError("CachedDataset yield dictionary does not match its spec.")
            return {key: visit(item[key], child) for key, child in expected.items()}
        if isinstance(expected, tuple):
            if not isinstance(item, tuple) or len(item) != len(expected):
                raise ValueError("CachedDataset yield tuple does not match its spec.")
            return tuple(visit(child, child_spec) for child, child_spec in zip(item, expected))
        if isinstance(expected, list):
            if not isinstance(item, list) or len(item) != len(expected):
                raise ValueError("CachedDataset yield list does not match its spec.")
            return [visit(child, child_spec) for child, child_spec in zip(item, expected)]
        raise ValueError("CachedDataset source spec is malformed.")

    return visit(value, spec), tuple(leaves)


def rebuild_value(spec: SpecTree, leaves: tuple[object, ...]) -> object:
    """Rebuild one exact container tree from leaves in structural order."""

    iterator = iter(leaves)

    def visit(expected: object):
        if isinstance(expected, TensorSpec):
            return next(iterator)
        if isinstance(expected, dict):
            return {key: visit(child) for key, child in expected.items()}
        if isinstance(expected, tuple):
            return tuple(visit(child) for child in expected)
        if isinstance(expected, list):
            return [visit(child) for child in expected]
        raise CacheIntegrityError("CachedDataset manifest has an invalid logical tree.")

    result = visit(spec)
    try:
        next(iterator)
    except StopIteration:
        return result
    raise CacheIntegrityError("CachedDataset chunk has too many leaf values.")


def spec_to_data(spec: SpecTree) -> dict[str, object]:
    """Encode a SpecTree in a bounded JSON-compatible representation."""

    def dim(value: object):
        return "dynamic" if value is Dim.DYNAMIC else value

    def visit(value: object):
        if isinstance(value, TensorSpec):
            return {
                "kind": "tensor", "dtype": value.dtype.name,
                "shape": None if value.shape is None else [dim(item) for item in value.shape],
                "batch": dim(value.batch) if value.batch is not None else None,
                "layout": value.layout.value,
                "axis_names": None if value.axis_names is None else list(value.axis_names),
                "batch_axis_name": value.batch_axis_name,
            }
        if isinstance(value, dict):
            items = []
            for key, child in value.items():
                if type(key) is str:
                    encoded = ["str", key]
                elif type(key) is bool:
                    encoded = ["bool", key]
                elif type(key) is int:
                    encoded = ["int", key]
                else:
                    raise ValueError("CachedDataset dictionary key is invalid.")
                items.append([encoded, visit(child)])
            return {"kind": "dict", "items": items}
        if isinstance(value, tuple):
            return {"kind": "tuple", "items": [visit(item) for item in value]}
        if isinstance(value, list):
            return {"kind": "list", "items": [visit(item) for item in value]}
        raise ValueError("CachedDataset source spec is malformed.")

    return visit(spec)


def spec_from_data(data: object) -> SpecTree:
    """Decode and validate a JSON logical SpecTree representation."""

    def dim(value: object):
        return Dim.DYNAMIC if value == "dynamic" else value

    def visit(value: object):
        if not isinstance(value, dict) or not isinstance(value.get("kind"), str):
            raise CacheIntegrityError("CachedDataset manifest tree is malformed.")
        kind = value["kind"]
        if kind == "tensor":
            fields = {"kind", "dtype", "shape", "batch", "layout", "axis_names", "batch_axis_name"}
            if set(value) != fields:
                raise CacheIntegrityError("CachedDataset tensor manifest is malformed.")
            shape = value["shape"]
            if shape is not None and not isinstance(shape, list):
                raise CacheIntegrityError("CachedDataset tensor shape is malformed.")
            try:
                return TensorSpec(
                    value["dtype"], shape=None if shape is None else tuple(dim(item) for item in shape),
                    batch=None if value["batch"] is None else dim(value["batch"]),
                    backend=Backend.numpy, layout=value["layout"],
                    axis_names=None if value["axis_names"] is None else tuple(value["axis_names"]),
                    batch_axis_name=value["batch_axis_name"],
                )
            except (TypeError, ValueError) as error:
                raise CacheIntegrityError("CachedDataset tensor manifest is invalid.") from error
        if kind in {"dict", "tuple", "list"}:
            if set(value) != {"kind", "items"} or not isinstance(value["items"], list):
                raise CacheIntegrityError("CachedDataset container manifest is malformed.")
            if kind == "dict":
                result = {}
                for item in value["items"]:
                    if not isinstance(item, list) or len(item) != 2 or not isinstance(item[0], list) or len(item[0]) != 2:
                        raise CacheIntegrityError("CachedDataset dictionary manifest is malformed.")
                    tag, key = item[0]
                    if tag == "str" and type(key) is str:
                        pass
                    elif tag == "bool" and type(key) is bool:
                        pass
                    elif tag == "int" and type(key) is int:
                        pass
                    else:
                        raise CacheIntegrityError("CachedDataset dictionary key is malformed.")
                    if key in result:
                        raise CacheIntegrityError("CachedDataset dictionary has duplicate keys.")
                    result[key] = visit(item[1])
                return result
            values = [visit(item) for item in value["items"]]
            return tuple(values) if kind == "tuple" else values
        raise CacheIntegrityError("CachedDataset manifest tree kind is unsupported.")

    result = visit(data)
    flatten_spec(result)
    return result


__all__ = ["CacheIntegrityError", "_FORMAT", "_VERSION", "flatten_spec", "numpy_spec", "normalize_value", "rebuild_value", "spec_from_data", "spec_to_data"]
