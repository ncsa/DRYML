from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .definition import ConcreteDefinition


V2_IDENTITY_VERSION = 2
_SUPPORTED_IDENTITY_VERSIONS = frozenset((V2_IDENTITY_VERSION,))


def new_node_id() -> object:
    """Allocate one private process-local CDef graph-node token.

    Returns:
        An opaque, hashable token used only by multiplicity-sensitive internal
        graph operations. The token is intentionally never persisted or
        included in structural identity.
    """

    return object()


def cdef_node_key(cdef: "ConcreteDefinition") -> object:
    """Return a CDef's private graph-node key for internal graph authority.

    Args:
        cdef: A current concrete definition.

    Returns:
        Its opaque process-local node token.

    Raises:
        TypeError: If ``cdef`` is not a concrete definition.
    """

    from .definition import ConcreteDefinition

    if not isinstance(cdef, ConcreteDefinition):
        raise TypeError(
            f"Expected ConcreteDefinition, got {type(cdef).__name__}."
        )
    return cdef._node_id


def same_cdef_node(
    left: "ConcreteDefinition", right: "ConcreteDefinition"
) -> bool:
    """Report whether two CDefs name the same private graph node.

    This deliberately differs from structural CDef equality and is restricted
    to graph, realization, and other multiplicity-sensitive internals.
    """

    return cdef_node_key(left) is cdef_node_key(right)


@dataclass(frozen=True, slots=True)
class CDefIdentityRecord:
    """Decoded persisted CDef identity data without constructor binding.

    Attributes:
        version: Exact identity format version. Only V2 is authoritative.
        cls: Persisted class or symbolic class reference.
        parameters: V2 semantic bound name/value record.
        stateful_role: Identity-neutral V2 ``Serializable`` role authority.
        stable_hash_cache: Optional cached digest from the serialized object.
    """

    version: int
    cls: Any
    stable_hash_cache: str | None = None
    parameters: Any = None
    stateful_role: bool = False


def validate_identity_version(version: int) -> int:
    """Return a supported CDef identity version or raise ``ValueError``.

    Args:
        version: Candidate persisted identity-version value.

    Returns:
        The validated version.

    Raises:
        ValueError: If the value is not a supported exact identity version.
    """

    if type(version) is not int or version not in _SUPPORTED_IDENTITY_VERSIONS:
        raise ValueError(
            "Unsupported ConcreteDefinition identity version "
            f"{version!r}; supported version is {V2_IDENTITY_VERSION}. "
            "This authority predates CDef V2 and cannot be loaded; recreate it "
            "with the current API."
        )
    return version


def decode_identity_record(state: Any) -> CDefIdentityRecord:
    """Decode one V2 CDef record without resolving symbols.

    Args:
        state: Pickle state emitted by the V2 CDef record codec.

    Returns:
        A normalized V2 identity record.

    Raises:
        TypeError: If the state has no recognized CDef record layout.
        ValueError: If a required field is missing or the version is unknown.
    """

    if not isinstance(state, dict):
        raise ValueError(
            "Unsupported pre-V2 ConcreteDefinition authority: expected a V2 "
            "mapping with identity_version=2, not a raw tuple/list record. "
            "Recreate the definition with the current API."
        )
    required = {"identity_version", "cls", "parameters", "stateful_role", "stable_hash_cache"}
    if set(state) != required:
        legacy_fields = sorted(set(state) & {"args", "kwargs"})
        detail = f"; legacy fields {legacy_fields!r}" if legacy_fields else ""
        raise ValueError(
            "Unsupported ConcreteDefinition authority: V2 requires exactly "
            f"{sorted(required)!r}{detail}. Recreate it with the current API."
        )
    version = validate_identity_version(state["identity_version"])
    if type(state["stateful_role"]) is not bool:
        raise TypeError("V2 ConcreteDefinition stateful_role must be a bool.")
    from .bound_args import decode_bound_arguments

    return CDefIdentityRecord(
        version,
        state["cls"],
        parameters=decode_bound_arguments(state["parameters"]),
        stable_hash_cache=state["stable_hash_cache"],
        stateful_role=state["stateful_role"],
    )


def stable_hash_domain(type_marker: str, version: int) -> str:
    """Return the structural-hash domain for the V2 CDef identity version.

    Args:
        type_marker: Historical concrete-definition type marker.
        version: Validated CDef identity version.

    Returns:
        A V2-specific marker.
    """

    version = validate_identity_version(version)
    return f"{type_marker}:identity-v{version}"


def same_cdef(left: "ConcreteDefinition", right: "ConcreteDefinition") -> bool:
    if left is right:
        return True
    if left.stable_hash() != right.stable_hash():
        return False
    try:
        return left == right
    except TypeError:
        return False
