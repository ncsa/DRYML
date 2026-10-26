"""Closed canonical identities for managed operations and ordinary arguments."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

from dryml.core.reference_values import ObjectId
from dryml.core.node import Node

from .errors import ManagedConfigError

_OPERATION_DOMAIN = b"dryml-managed-operation-v2\x00"
_ARGUMENT_DOMAIN = b"dryml-managed-arguments-v1\x00"
_MAX_DEPTH = 16
_MAX_VALUES = 4096
_MAX_CONTAINER_ENTRIES = 1024
_MAX_PREIMAGE_BYTES = 1024 * 1024
_MAX_INT_BITS = 256
_MAX_TEXT_BYTES = 64 * 1024


def operation_digest(receiver: Node, member: str) -> str:
    """Return canonical object-projection-plus-member identity within an authority.

    Args:
        receiver: Detached concrete Node receiver, with required ObjectIds bound.
        member: Exact non-empty Python member name recorded by the descriptor.

    Returns:
        A lowercase SHA-256 hexadecimal operation identifier.

    Raises:
        ManagedConfigError: If either input is outside the closed identity grammar.

    Side Effects:
        None. Arguments, stores, and config choices do not affect this identifier.
    """

    _receiver(receiver)
    _validate_member(member)
    return _operation_digest_from_projection(receiver.object_projection_digest(), member)


def _operation_digest_from_projection(projection_digest: str, member: str) -> str:
    """Return the operation identity from already validated control components.

    This internal helper keeps control persistence on the same canonical encoding
    as :func:`operation_digest` without constructing or realizing a receiver.
    """

    return hashlib.sha256(
        _OPERATION_DOMAIN + _atom(b"object-projection/v2", projection_digest.encode("ascii")) + _atom(b"member", member.encode("utf-8"))
    ).hexdigest()


def _receiver(receiver):
    from dryml.core.definition import ConcreteNode
    from dryml.core.node import _owned_nodes
    if not isinstance(receiver, ConcreteNode):
        raise ManagedConfigError(message="managed receiver must be detached concrete Node metadata")
    identities = set()
    for _, node in _owned_nodes(receiver):
        if node._stateful_role and node.object_id is None:
            raise ManagedConfigError(message="managed receiver requires bound stateful ObjectIds")
        if node.object_id is not None:
            if node.object_id in identities:
                raise ManagedConfigError(message="independent managed receiver Nodes cannot share an ObjectId")
            identities.add(node.object_id)
        if node.state is not None:
            try:
                node.state.validate_for(node, codec=node.state.codec)
            except (ValueError, TypeError) as error:
                raise ManagedConfigError(message="managed receiver has incompatible selected state metadata") from error


@dataclass(frozen=True, slots=True)
class AdmissionIdentity:
    """Detached metadata-only admission facts shared by direct and transport callers.

    receiver is concrete Node metadata, member is the selected declared identifier,
    and argument_digest is separate normalized ordinary-argument evidence. Invalid
    metadata raises ManagedConfigError. Construction computes pure identity facts;
    it never resolves symbols, constructs Objects, reads state, or opens Stores.
    """

    receiver: Node
    member: str
    argument_digest: str
    _operation_id: str = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "_operation_id", operation_digest(self.receiver, self.member))
        if (type(self.argument_digest) is not str or len(self.argument_digest) != 64
                or any(c not in "0123456789abcdef" for c in self.argument_digest)):
            raise ManagedConfigError(message="argument_digest must be a lowercase SHA-256 digest")

    @property
    def operation_id(self):
        """Return stable projection/member digest; arguments and payloads are excluded."""
        return self._operation_id

    @classmethod
    def from_call(cls, receiver, descriptor, args, kwargs):
        """Bind detached receiver and checked declaration to ordinary argument facts.

        receiver is a concrete Node; descriptor is the selected managed declaration;
        args/kwargs are native tuple/dict caller values. Returns AdmissionIdentity.
        Invalid declaration, binding or unsupported values raise ManagedConfigError.
        Binding uses the Node as an inert first parameter, with no realization/IO.
        Explicit signature authority must be normalized by its owning adapter before
        using the direct constructor; this method handles the closed ordinary grammar.
        """
        from .descriptor import ManagedOperation
        if not isinstance(descriptor, ManagedOperation):
            raise ManagedConfigError(message="admission requires selected managed declaration evidence")
        return cls(receiver, descriptor.member, argument_digest(descriptor, receiver, args, kwargs))


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
        raise ManagedConfigError(message="managed arguments do not bind") from error
    return _argument_digest_bound(
        tuple(
            (name, value)
            for name, value in bound.arguments.items()
            if name not in {
                instance_parameter,
                "managed",
                getattr(descriptor, "store_parameter", None),
                *getattr(descriptor, "_invocation_parameters", ()),
            }
        ),
    )


def _argument_digest_bound(values: tuple[tuple[str, object], ...]) -> str:
    """Digest one already-bound managed argument record without rebinding it.

    Args:
        values: Ordered native parameter/value pairs after defaults and injected
            controls have been removed. Callers retain source-category choices
            while replacing explicitly annotated slots with pinned authority.

    Returns:
        The stable closed-grammar SHA-256 digest for the supplied bound record.

    Raises:
        ManagedConfigError: If a retained name or value is outside the existing
            bounded managed identity grammar.

    Side Effects:
        None. This preserves the established encoder tags and never realizes a
        selected authority.
    """

    encoder = _ArgumentEncoder()
    budget = _PreimageBudget()
    budget.add(len(_ARGUMENT_DOMAIN))
    budget.add(_atom_header_size(b"arguments"))
    parameters = []
    arguments_size = 0
    for name, value in values:
        name_bytes = _parameter_name_bytes(name)
        budget.add(_atom_size(b"name", len(name_bytes)))
        value_size = encoder.measure(value, _parameter_path(name), 0, budget)
        parameter_size = _atom_size(b"name", len(name_bytes)) + value_size
        budget.add(_atom_header_size(b"parameter"))
        parameters.append((name_bytes, value, value_size, parameter_size))
        arguments_size += _atom_size(b"parameter", parameter_size)

    digest = hashlib.sha256()
    encoder.set_digest(digest)
    encoder._write(_ARGUMENT_DOMAIN)
    encoder._write(_atom_header(b"arguments", arguments_size))
    for name_bytes, value, value_size, parameter_size in parameters:
        encoder._write(_atom_header(b"parameter", parameter_size))
        encoder._write(_atom_header(b"name", len(name_bytes)))
        encoder._write(name_bytes)
        encoder.emit(value, value_size)
    return digest.hexdigest()


class _ArgumentEncoder:
    """One-use encoder enforcing the managed argument grammar and aggregate limits."""

    def __init__(self) -> None:
        self._values = 0
        self._active: set[int] = set()
        self._container_sizes: dict[int, int] = {}
        self._digest: hashlib._Hash | None = None

    def set_digest(self, digest: hashlib._Hash) -> None:
        """Set the SHA-256 sink after validation and preimage measurement succeed."""

        self._digest = digest

    def measure(self, value: object, path: str, depth: int, budget: _PreimageBudget) -> int:
        """Validate one value and return its exact canonical byte length."""

        if depth > _MAX_DEPTH:
            raise ManagedConfigError(message=f"{path}: maximum depth is {_MAX_DEPTH}")
        self._values += 1
        if self._values > _MAX_VALUES:
            raise ManagedConfigError(message=f"{path}: maximum aggregate values is {_MAX_VALUES}")
        value_type = type(value)
        if value is None:
            return self._leaf_size(b"none", 0, budget)
        if value_type is bool:
            return self._leaf_size(b"bool", 1, budget)
        if value_type is int:
            if value.bit_length() > _MAX_INT_BITS:
                raise ManagedConfigError(message=f"{path}: integers support at most {_MAX_INT_BITS} bits")
            return self._leaf_size(b"int", len(str(value).encode("ascii")), budget)
        if value_type is float:
            if not math.isfinite(value):
                raise ManagedConfigError(message=f"{path}: floats must be finite")
            return self._leaf_size(b"float", len(value.hex().encode("ascii")), budget)
        if value_type is str:
            return self._leaf_size(b"str", len(_bounded_text(value, path)), budget)
        if value_type is bytes:
            if len(value) > _MAX_TEXT_BYTES:
                raise ManagedConfigError(message=f"{path}: bytes support at most 64 KiB")
            return self._leaf_size(b"bytes", len(value), budget)
        if value_type is ObjectId:
            return self._leaf_size(b"object-id", len(value.__stable_leaf_bytes__()), budget)
        if isinstance(value, Node):
            return self._leaf_size(b"node-exact/v1", len(value.exact_digest().encode("ascii")), budget)
        if value_type in (list, tuple, dict):
            return self._measure_container(value, path, depth, budget)
        node = _live_identity_node(value)
        if node is not None:
            return self._leaf_size(b"live-node-object/v1", len(node.object_projection_digest().encode("ascii")), budget)
        raise ManagedConfigError(message=f"{path}: unsupported argument type")

    def _measure_container(
        self,
        value: list[object] | tuple[object, ...] | dict[object, object],
        path: str,
        depth: int,
        budget: _PreimageBudget,
    ) -> int:
        """Validate an exact container and account for its canonical byte length."""

        if len(value) > _MAX_CONTAINER_ENTRIES:
            raise ManagedConfigError(message=f"{path}: containers support at most {_MAX_CONTAINER_ENTRIES} entries")
        value_id = id(value)
        if value_id in self._active:
            raise ManagedConfigError(message=f"{path}: cyclic containers are unsupported")
        self._active.add(value_id)
        try:
            if type(value) is dict:
                for key in value:
                    if type(key) is not str:
                        raise ManagedConfigError(message=f"{path}: dictionary keys must be exact strings")
                budget.add(_atom_header_size(b"dict"))
                payload_size = 0
                for key in sorted(value):
                    self._count_value(f"{path}.<key>")
                    key_size = self._leaf_size(b"key", len(_bounded_text(key, f"{path}.<key>")), budget)
                    value_size = self.measure(value[key], f"{path}.<value>", depth + 1, budget)
                    payload_size += key_size + value_size
                size = _atom_header_size(b"dict") + payload_size
                self._container_sizes[value_id] = size
                return size
            tag = b"list" if type(value) is list else b"tuple"
            budget.add(_atom_header_size(tag))
            payload_size = sum(
                self.measure(item, f"{path}[{index}]", depth + 1, budget) for index, item in enumerate(value)
            )
            size = _atom_header_size(tag) + payload_size
            self._container_sizes[value_id] = size
            return size
        finally:
            self._active.remove(value_id)

    def emit(self, value: object, size: int) -> None:
        """Write a measured value's unchanged canonical bytes to the digest sink."""

        value_type = type(value)
        if value is None:
            self._emit_atom(b"none", b"")
        elif value_type is bool:
            self._emit_atom(b"bool", b"1" if value else b"0")
        elif value_type is int:
            self._emit_atom(b"int", str(value).encode("ascii"))
        elif value_type is float:
            self._emit_atom(b"float", value.hex().encode("ascii"))
        elif value_type is str:
            self._emit_atom(b"str", _bounded_text(value, "argument"))
        elif value_type is bytes:
            self._emit_atom(b"bytes", value)
        elif value_type is ObjectId:
            self._emit_atom(b"object-id", value.__stable_leaf_bytes__())
        elif isinstance(value, Node):
            self._emit_atom(b"node-exact/v1", value.exact_digest().encode("ascii"))
        elif value_type in (list, tuple, dict):
            self._emit_container(value, size)
        else:
            node = _live_identity_node(value)
            if node is None:
                raise ManagedConfigError(message="managed arguments changed while encoding")
            self._emit_atom(b"live-node-object/v1", node.object_projection_digest().encode("ascii"))

    def _emit_container(self, value: list[object] | tuple[object, ...] | dict[object, object], size: int) -> None:
        """Write a measured container without constructing a joined child payload."""

        tag = b"dict" if type(value) is dict else b"list" if type(value) is list else b"tuple"
        self._write(_atom_header(tag, size - _atom_header_size(tag)))
        if type(value) is dict:
            for key in sorted(value):
                key_bytes = _bounded_text(key, "dictionary key")
                self._emit_atom(b"key", key_bytes)
                self.emit(value[key], self._container_sizes.get(id(value[key]), 0))
            return
        for item in value:
            self.emit(item, self._container_sizes.get(id(item), 0))

    def _leaf_size(self, tag: bytes, payload_size: int, budget: _PreimageBudget) -> int:
        """Account for one leaf atom and return its encoded size."""

        size = _atom_size(tag, payload_size)
        budget.add(size)
        return size

    def _emit_atom(self, tag: bytes, payload: bytes) -> None:
        """Write one leaf atom in bounded header and payload chunks."""

        self._write(_atom_header(tag, len(payload)))
        self._write(payload)

    def _write(self, data: bytes) -> None:
        """Update the measured digest with one bounded canonical chunk."""

        if self._digest is None:
            raise RuntimeError("argument digest sink is not initialized")
        self._digest.update(data)

    def _count_value(self, path: str) -> None:
        """Count one dictionary key against the aggregate input support limit."""

        self._values += 1
        if self._values > _MAX_VALUES:
            raise ManagedConfigError(message=f"{path}: maximum aggregate values is {_MAX_VALUES}")


def _live_identity_node(value: object) -> Node | None:
    """Return retained live identity metadata without accepting arbitrary lookalikes."""

    from dryml.core.object import Object

    if not isinstance(value, Object):
        return None
    node = object.__getattribute__(value, "_identity_node")
    if not isinstance(node, Node):
        raise ManagedConfigError(message="live Object does not carry a runtime identity Node")
    return node


def _validate_member(member: str) -> None:
    """Validate the member component of an operation identity."""

    if type(member) is not str or not member or not member.isidentifier():
        raise ManagedConfigError(message="operation member must be a bounded Python identifier")
    try:
        member_bytes = member.encode("utf-8")
    except UnicodeError as error:
        raise ManagedConfigError(message="operation member must be valid UTF-8") from error
    if len(member_bytes) > 255:
        raise ManagedConfigError(message="operation member must be a bounded Python identifier")


def _bounded_text(value: str, path: str) -> bytes:
    """Encode one exact string under the managed per-value size limit."""

    if len(value) > _MAX_TEXT_BYTES:
        raise ManagedConfigError(message=f"{path}: strings support at most 64 KiB")
    try:
        encoded = value.encode("utf-8")
    except UnicodeError as error:
        raise ManagedConfigError(message=f"{path}: strings must be valid UTF-8") from error
    if len(encoded) > _MAX_TEXT_BYTES:
        raise ManagedConfigError(message=f"{path}: strings support at most 64 KiB")
    return encoded


class _PreimageBudget:
    """Incrementally enforce the complete SHA-256 preimage support limit."""

    def __init__(self) -> None:
        self._size = 0

    def add(self, size: int) -> None:
        """Count canonical bytes and reject before a joined representation exists."""

        self._size += size
        if self._size > _MAX_PREIMAGE_BYTES:
            raise ManagedConfigError(message="managed argument preimage exceeds 1 MiB")


def _parameter_name_bytes(name: object) -> bytes:
    """Encode a native parameter name without allowing raw UTF-8 errors to escape."""

    if type(name) is not str:
        raise ManagedConfigError(message="managed declaration has an invalid parameter name")
    try:
        return name.encode("utf-8")
    except UnicodeError as error:
        raise ManagedConfigError(message="managed declaration parameter names must be valid UTF-8") from error


def _parameter_path(name: object) -> str:
    """Return a bounded diagnostic path for a native parameter name."""

    if type(name) is str and len(name) <= 64 and name.isascii() and name.isidentifier():
        return f"$.{name}"
    return "$.<parameter>"


def _atom_header_size(tag: bytes) -> int:
    """Return the fixed canonical framing size for an atom tag."""

    return 2 + len(tag) + 8


def _atom_size(tag: bytes, payload_size: int) -> int:
    """Return the full canonical size of one atom without allocating its payload."""

    return _atom_header_size(tag) + payload_size


def _atom_header(tag: bytes, payload_size: int) -> bytes:
    """Return the fixed canonical framing for an atom payload of known size."""

    return len(tag).to_bytes(2, "big") + tag + payload_size.to_bytes(8, "big")


def _atom(tag: bytes, payload: bytes) -> bytes:
    """Return an unambiguous length-delimited canonical atom."""

    return _atom_header(tag, len(payload)) + payload


__all__ = ["AdmissionIdentity", "argument_digest", "operation_digest"]
