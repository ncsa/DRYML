"""Closed canonical identities for managed operations and ordinary arguments."""

from __future__ import annotations

import hashlib
import math

from dryml.core.reference_values import ObjectId, ObjectRef, StateRef

from .errors import ManagedConfigError

_OPERATION_DOMAIN = b"dryml-managed-operation-v1\x00"
_ARGUMENT_DOMAIN = b"dryml-managed-arguments-v1\x00"
_MAX_DEPTH = 16
_MAX_VALUES = 4096
_MAX_CONTAINER_ENTRIES = 1024
_MAX_PREIMAGE_BYTES = 1024 * 1024
_MAX_INT_BITS = 256
_MAX_TEXT_BYTES = 64 * 1024


def operation_digest(object_ref: ObjectRef, member: str) -> str:
    """Return the stable digest for one exact ObjectRef and managed member.

    Args:
        object_ref: Exact immutable ObjectRef of the method receiver graph.
        member: Exact non-empty Python member name recorded by the descriptor.

    Returns:
        A lowercase SHA-256 hexadecimal operation identifier.

    Raises:
        ManagedConfigError: If either input is outside the closed identity grammar.

    Side Effects:
        None. Arguments, stores, and config choices do not affect this identifier.
    """

    if type(object_ref) is not ObjectRef:
        raise ManagedConfigError(message="operation object_ref must be an exact ObjectRef")
    _validate_member(member)
    return _operation_digest_from_object_ref_digest(object_ref.digest(), member)


def _operation_digest_from_object_ref_digest(object_ref_digest: str, member: str) -> str:
    """Return the operation identity from already validated control components.

    This internal helper keeps control persistence on the same canonical encoding
    as :func:`operation_digest` without constructing an ObjectRef from a digest.
    """

    return hashlib.sha256(
        _OPERATION_DOMAIN + _atom(b"object-ref", object_ref_digest.encode("ascii")) + _atom(b"member", member.encode("utf-8"))
    ).hexdigest()


def argument_digest(descriptor: object, instance: object, args: tuple[object, ...], kwargs: dict[str, object]) -> str:
    """Bind and digest a managed call's ordinary arguments under the closed grammar.

    Args:
        descriptor: A managed declaration carrying its native author signature.
        instance: Receiver bound positionally as the first native parameter.
        args: Caller positional arguments excluding the receiver.
        kwargs: Caller keyword arguments, including an optional config ``managed``.

    Returns:
        A lowercase SHA-256 hexadecimal digest after defaults are normalized and
        the receiver and injected ``managed`` parameter are removed.

    Raises:
        ManagedConfigError: If binding fails or any value is unsupported, cyclic,
        or exceeds a fixed support bound.

    Side Effects:
        None. This never serializes user values or calls custom hooks.
    """

    signature = getattr(descriptor, "author_signature", None)
    instance_parameter = getattr(descriptor, "instance_parameter", None)
    if signature is None or type(instance_parameter) is not str:
        raise ManagedConfigError(message="argument digest requires a managed declaration")
    if type(args) is not tuple or type(kwargs) is not dict:
        raise ManagedConfigError(message="managed arguments must be native tuple and dict values")
    supplied = dict(kwargs)
    supplied.setdefault("managed", None)
    try:
        bound = signature.bind(instance, *args, **supplied)
        bound.apply_defaults()
    except TypeError as error:
        raise ManagedConfigError(message=f"managed arguments do not bind: {error}") from error
    values = [(name, value) for name, value in bound.arguments.items() if name not in {instance_parameter, "managed"}]
    encoder = _ArgumentEncoder()
    encoded = _atom(
        b"arguments",
        b"".join(
            _atom(b"parameter", _atom(b"name", name.encode("utf-8")) + encoder.encode(value, f"$.{name}", 0))
            for name, value in values
        ),
    )
    if len(encoded) > _MAX_PREIMAGE_BYTES:
        raise ManagedConfigError(message="managed argument preimage exceeds 1 MiB")
    return hashlib.sha256(_ARGUMENT_DOMAIN + encoded).hexdigest()


class _ArgumentEncoder:
    """One-use encoder enforcing the managed argument grammar and aggregate limits."""

    def __init__(self) -> None:
        self._values = 0
        self._active: set[int] = set()

    def encode(self, value: object, path: str, depth: int) -> bytes:
        """Encode one value without invoking user serialization or representation hooks."""

        if depth > _MAX_DEPTH:
            raise ManagedConfigError(message=f"{path}: maximum depth is {_MAX_DEPTH}")
        self._values += 1
        if self._values > _MAX_VALUES:
            raise ManagedConfigError(message=f"{path}: maximum aggregate values is {_MAX_VALUES}")
        value_type = type(value)
        if value is None:
            return _atom(b"none", b"")
        if value_type is bool:
            return _atom(b"bool", b"1" if value else b"0")
        if value_type is int:
            if value.bit_length() > _MAX_INT_BITS:
                raise ManagedConfigError(message=f"{path}: integers support at most {_MAX_INT_BITS} bits")
            return _atom(b"int", str(value).encode("ascii"))
        if value_type is float:
            if not math.isfinite(value):
                raise ManagedConfigError(message=f"{path}: floats must be finite")
            return _atom(b"float", value.hex().encode("ascii"))
        if value_type is str:
            return _atom(b"str", _bounded_text(value, path))
        if value_type is bytes:
            if len(value) > _MAX_TEXT_BYTES:
                raise ManagedConfigError(message=f"{path}: bytes support at most 64 KiB")
            return _atom(b"bytes", value)
        if value_type is ObjectId:
            return _atom(b"object-id", value.__stable_leaf_bytes__())
        if value_type is ObjectRef:
            return _atom(b"object-ref", value.digest().encode("ascii"))
        if value_type is StateRef:
            return _atom(b"state-ref", value.digest().encode("ascii"))
        if value_type in (list, tuple, dict):
            return self._encode_container(value, path, depth)
        object_ref = _live_object_ref(value)
        if object_ref is not None:
            return _atom(b"live-object", object_ref.digest().encode("ascii"))
        raise ManagedConfigError(message=f"{path}: unsupported argument type {value_type.__name__}")

    def _encode_container(self, value: list[object] | tuple[object, ...] | dict[object, object], path: str, depth: int) -> bytes:
        """Encode exact containers, preserving sequence kind and sorted string keys."""

        if len(value) > _MAX_CONTAINER_ENTRIES:
            raise ManagedConfigError(message=f"{path}: containers support at most {_MAX_CONTAINER_ENTRIES} entries")
        value_id = id(value)
        if value_id in self._active:
            raise ManagedConfigError(message=f"{path}: cyclic containers are unsupported")
        self._active.add(value_id)
        try:
            if type(value) is dict:
                entries = []
                for key in value:
                    if type(key) is not str:
                        raise ManagedConfigError(message=f"{path}: dictionary keys must be exact strings")
                for key in sorted(value):
                    self._count_value(f"{path}.<key>")
                    entries.append(_atom(b"key", _bounded_text(key, f"{path}.<key>")) + self.encode(value[key], f"{path}[{key!r}]", depth + 1))
                return _atom(b"dict", b"".join(entries))
            tag = b"list" if type(value) is list else b"tuple"
            return _atom(tag, b"".join(self.encode(item, f"{path}[{index}]", depth + 1) for index, item in enumerate(value)))
        finally:
            self._active.remove(value_id)

    def _count_value(self, path: str) -> None:
        """Count one dictionary key against the aggregate input support limit."""

        self._values += 1
        if self._values > _MAX_VALUES:
            raise ManagedConfigError(message=f"{path}: maximum aggregate values is {_MAX_VALUES}")


def _live_object_ref(value: object) -> ObjectRef | None:
    """Return an exact live ObjectRef without accepting arbitrary lookalikes."""

    from dryml.core.object import Object

    if not isinstance(value, Object):
        return None
    object_ref = Object.object_ref.__get__(value, type(value))
    if type(object_ref) is not ObjectRef:
        raise ManagedConfigError(message="live Object does not carry an exact ObjectRef")
    return object_ref


def _validate_member(member: str) -> None:
    """Validate the member component of an operation identity."""

    if type(member) is not str or not member or not member.isidentifier() or len(member.encode("utf-8")) > 255:
        raise ManagedConfigError(message="operation member must be a bounded Python identifier")


def _bounded_text(value: str, path: str) -> bytes:
    """Encode one exact string under the managed per-value size limit."""

    encoded = value.encode("utf-8")
    if len(encoded) > _MAX_TEXT_BYTES:
        raise ManagedConfigError(message=f"{path}: strings support at most 64 KiB")
    return encoded


def _atom(tag: bytes, payload: bytes) -> bytes:
    """Return an unambiguous length-delimited canonical atom."""

    return len(tag).to_bytes(2, "big") + tag + len(payload).to_bytes(8, "big") + payload


__all__ = ["argument_digest", "operation_digest"]
