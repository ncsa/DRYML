"""Versioned, topology-checked local state for experimental JAX Objects.

The codec stores host array values and typed JAX key data, never live device,
compiled, graph, or NNX GraphDef objects.  Callers reconstruct a runtime from
their definition first, then validate a complete saved tree against that runtime
before installing any of its values.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


STATE_VERSION = 2

_LEAF_FIELDS = {
    "path", "alias", "key", "key_impl", "dtype", "shape",
    "data_dtype", "data_shape",
}


class JaxStateError(ValueError):
    """Raise when an experimental JAX state envelope is malformed or incompatible.

    The error is raised before a caller installs any local state, allowing the
    repository restore boundary to invalidate the whole graph rather than expose
    a partly restored JAX object.
    """


def write_tree_state(dest_dir: str, name: str, value) -> None:
    """Write one JAX pytree as versioned host-array payloads.

    Args:
        dest_dir: Existing Store-owned local-state directory.
        name: Stable local owner payload name.
        value: Pytree whose leaves are JAX arrays, typed keys, or scalar arrays.

    Raises:
        TypeError: If a leaf cannot be represented as a dense host array.

    Side Effects:
        Writes a JSON topology envelope and one NumPy payload per distinct leaf
        identity. Typed JAX keys are saved as key data plus their implementation.
    """

    import jax

    directory = Path(dest_dir)
    paths_and_leaves, treedef = jax.tree_util.tree_flatten_with_path(value)
    aliases: dict[int, int] = {}
    leaves = []
    for path, leaf in paths_and_leaves:
        marker = id(leaf)
        first_occurrence = marker not in aliases
        alias = aliases.setdefault(marker, len(aliases))
        key = _is_typed_key(jax, leaf)
        data = jax.random.key_data(leaf) if key else np.asarray(leaf)
        if data.dtype.hasobject:
            raise TypeError("JAX state leaves must be dense non-object arrays.")
        record = {
            "path": _path_text(path),
            "alias": alias,
            "key": key,
            "key_impl": _key_impl_name(jax, leaf) if key else None,
            "dtype": str(getattr(leaf, "dtype", data.dtype)),
            "shape": list(getattr(leaf, "shape", data.shape)),
            "data_dtype": str(data.dtype),
            "data_shape": list(data.shape),
        }
        leaves.append(record)
        if first_occurrence:
            np.save(directory / f"{name}-{alias}.npy", data, allow_pickle=False)
    _write_json(directory / f"{name}.json", {
        "version": STATE_VERSION,
        "tree": str(treedef),
        "leaves": leaves,
    })


def read_tree_state(src_dir: str, name: str, template):
    """Validate and restore one JAX pytree against a reconstructed template.

    Args:
        src_dir: Store-owned local-state payload directory.
        name: Stable local owner payload name.
        template: Current runtime pytree defining the accepted topology and leaf
            dtype/shape/key contracts.

    Returns:
        A pytree with the template's exact topology and restored array values.

    Raises:
        JaxStateError: If the state version, topology, paths, aliases, dtype, or
            shape is incompatible or a payload is malformed.

    Side Effects:
        Reads payload files only. It does not mutate ``template``.
    """

    return _restore_tree_payload(_read_tree_payload(src_dir, name), template)


def _read_tree_payload(src_dir: str, name: str):
    """Read and validate a tree payload without retaining its source directory."""

    directory = Path(src_dir)
    envelope = _read_json(directory / f"{name}.json")
    if set(envelope) != {"version", "tree", "leaves"} or envelope["version"] != STATE_VERSION:
        raise JaxStateError("Unsupported JAX state envelope version.")
    records = envelope["leaves"]
    if not isinstance(envelope["tree"], str) or not isinstance(records, list):
        raise JaxStateError("Malformed JAX state tree envelope.")

    payloads = {}
    signatures = {}
    for record in records:
        if (
            not isinstance(record, dict)
            or set(record) != _LEAF_FIELDS
            or type(record["alias"]) is not int
            or record["alias"] < 0
            or type(record["key"]) is not bool
            or not isinstance(record["path"], list)
            or not all(isinstance(part, str) for part in record["path"])
            or not isinstance(record["dtype"], str)
            or not isinstance(record["data_dtype"], str)
            or not isinstance(record["shape"], list)
            or not isinstance(record["data_shape"], list)
            or not all(type(size) is int and size >= 0 for size in (*record["shape"], *record["data_shape"]))
            or (record["key_impl"] is not None and not isinstance(record["key_impl"], str))
        ):
            raise JaxStateError("Malformed JAX state leaf record.")
        alias = record["alias"]
        signature = (
            record["key"], record["key_impl"], record["dtype"], tuple(record["shape"]),
            record["data_dtype"], tuple(record["data_shape"]),
        )
        if alias in signatures and signatures[alias] != signature:
            raise JaxStateError("Aliased JAX state leaves have incompatible metadata.")
        signatures[alias] = signature
        if alias in payloads:
            continue
        payload_path = directory / f"{name}-{alias}.npy"
        try:
            data = np.load(payload_path, allow_pickle=False)
        except (OSError, ValueError) as error:
            raise JaxStateError(f"Cannot read JAX state payload {payload_path.name!r}.") from error
        if str(data.dtype) != record["data_dtype"] or list(data.shape) != record["data_shape"]:
            raise JaxStateError("JAX state payload dtype or shape does not match its envelope.")
        payloads[alias] = data
    return envelope, payloads


def _restore_tree_payload(payload, template):
    """Validate an in-memory tree payload against a runtime template and restore it."""

    import jax

    envelope, payloads = payload
    paths_and_template, treedef = jax.tree_util.tree_flatten_with_path(template)
    records = envelope["leaves"]
    if envelope["tree"] != str(treedef) or len(records) != len(paths_and_template):
        raise JaxStateError("JAX state tree topology does not match the reconstructed runtime.")

    template_aliases: dict[int, int] = {}
    restored_by_alias = {}
    restored_leaves = []
    for record, (path, leaf) in zip(records, paths_and_template):
        _validate_record(record, path, leaf, template_aliases, jax)
        alias = record["alias"]
        if alias not in restored_by_alias:
            data = payloads[alias]
            restored_by_alias[alias] = (
                jax.random.wrap_key_data(data, impl=_key_impl(jax, leaf))
                if record["key"] else jax.numpy.asarray(data, dtype=getattr(leaf, "dtype", None))
            )
        restored_leaves.append(restored_by_alias[alias])
    return jax.tree_util.tree_unflatten(treedef, restored_leaves)


def write_owner_envelope(dest_dir: str, name: str, **fields) -> None:
    """Write small non-array local owner metadata beside a tree payload.

    Args:
        dest_dir: Existing Store-owned local-state directory.
        name: Stable payload name.
        **fields: JSON-compatible owner fields in addition to the state version.

    Side Effects:
        Writes one versioned JSON file. It never serializes a live runtime object.
    """

    _write_json(Path(dest_dir) / f"{name}.json", {"version": STATE_VERSION, **fields})


def read_owner_envelope(src_dir: str, name: str, *, fields: set[str]) -> dict:
    """Read an exact versioned local owner envelope.

    Args:
        src_dir: Store-owned local-state payload directory.
        name: Stable payload name.
        fields: Required fields excluding the mandatory version field.

    Returns:
        The validated JSON object.

    Raises:
        JaxStateError: If the envelope is malformed, incomplete, or versioned
            incompatibly.
    """

    value = _read_json(Path(src_dir) / f"{name}.json")
    if set(value) != {"version", *fields} or value.get("version") != STATE_VERSION:
        raise JaxStateError("Unsupported JAX owner envelope.")
    return value


def _validate_record(record, path, leaf, aliases, jax) -> None:
    if not isinstance(record, dict) or set(record) != _LEAF_FIELDS or type(record["alias"]) is not int or record["alias"] < 0:
        raise JaxStateError("Malformed JAX state leaf record.")
    key = _is_typed_key(jax, leaf)
    data = jax.random.key_data(leaf) if key else np.asarray(leaf)
    marker = id(leaf)
    expected_alias = aliases.setdefault(marker, len(aliases))
    if (
        record["path"] != _path_text(path)
        or record["alias"] != expected_alias
        or record["key"] is not key
        or record["key_impl"] != (_key_impl_name(jax, leaf) if key else None)
        or record["dtype"] != str(getattr(leaf, "dtype", data.dtype))
        or record["shape"] != list(getattr(leaf, "shape", data.shape))
        or record["data_dtype"] != str(data.dtype)
        or record["data_shape"] != list(data.shape)
    ):
        raise JaxStateError("JAX state leaf topology, dtype, or shape does not match the runtime.")


def _is_typed_key(jax, value) -> bool:
    return hasattr(value, "dtype") and jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key)


def _key_impl(jax, value):
    return jax.random.key_impl(value) if _is_typed_key(jax, value) else None


def _key_impl_name(jax, value) -> str:
    """Return the stable public spelling for a typed PRNG key implementation."""

    return str(_key_impl(jax, value))


def _path_text(path) -> list[str]:
    """Encode JAX key paths deterministically without using them as authority."""

    return [f"{type(part).__name__}:{getattr(part, 'key', getattr(part, 'idx', getattr(part, 'name', part)))}" for part in path]


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")), encoding="utf-8")


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise JaxStateError(f"Cannot read JAX state envelope {path.name!r}.") from error
    if not isinstance(value, dict):
        raise JaxStateError("JAX state envelope must be a JSON object.")
    return value


__all__ = ["JaxStateError", "STATE_VERSION", "read_owner_envelope", "read_tree_state", "write_owner_envelope", "write_tree_state"]
