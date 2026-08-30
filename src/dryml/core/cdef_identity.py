from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .definition import ConcreteDefinition


V1_IDENTITY_VERSION = 1
V2_IDENTITY_VERSION = 2
_SUPPORTED_IDENTITY_VERSIONS = frozenset(
    (V1_IDENTITY_VERSION, V2_IDENTITY_VERSION)
)


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
        version: Exact identity format version. Missing legacy data is V1.
        cls: Persisted class or symbolic class reference.
        args: Persisted legacy positional constructor surface.
        kwargs: Persisted legacy keyword constructor surface.
        parameters: Optional V2 semantic bound name/value record.
        stateful_role: Identity-neutral V2 ``Serializable`` role authority.
        stable_hash_cache: Optional cached digest from the serialized object.
    """

    version: int
    cls: Any
    args: Any = None
    kwargs: Any = None
    stable_hash_cache: str | None = None
    parameters: Any = None
    stateful_role: bool | None = None


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
            f"Unsupported ConcreteDefinition identity version {version!r}."
        )
    return version


def decode_identity_record(state: Any) -> CDefIdentityRecord:
    """Decode legacy or versioned CDef pickle state without resolving symbols.

    Args:
        state: Pickle state emitted by a legacy slotted CDef or a versioned
            CDef record.

    Returns:
        A normalized identity record. Legacy state with no version is V1.

    Raises:
        TypeError: If the state has no recognized CDef record layout.
        ValueError: If a required field is missing or the version is unknown.
    """

    if isinstance(state, (list, tuple)):
        if len(state) not in (3, 4):
            raise ValueError(
                "Legacy ConcreteDefinition state must contain cls, args, kwargs, and optional hash cache."
            )
        cls, args, kwargs = state[:3]
        stable_hash_cache = state[3] if len(state) == 4 else None
        return CDefIdentityRecord(
            V1_IDENTITY_VERSION,
            cls,
            args=args,
            kwargs=kwargs,
            stable_hash_cache=stable_hash_cache,
        )

    if isinstance(state, dict):
        if "cls" not in state:
            raise ValueError(
                "ConcreteDefinition state is missing required field 'cls'."
            )
        version = validate_identity_version(
            state.get("identity_version", V1_IDENTITY_VERSION)
        )
        if version == V2_IDENTITY_VERSION:
            if "parameters" not in state:
                raise ValueError(
                    "V2 ConcreteDefinition state is missing required field 'parameters'."
                )
            if "stateful_role" not in state:
                raise ValueError(
                    "V2 ConcreteDefinition state is missing required field 'stateful_role'."
                )
            if type(state["stateful_role"]) is not bool:
                raise TypeError(
                    "V2 ConcreteDefinition stateful_role must be a bool."
                )
            legacy_fields = [key for key in ("args", "kwargs") if key in state]
            if legacy_fields:
                raise ValueError(
                    "V2 ConcreteDefinition state cannot contain legacy fields: "
                    f"{legacy_fields!r}."
                )
            from .bound_args import decode_bound_arguments

            return CDefIdentityRecord(
                version,
                state["cls"],
                parameters=decode_bound_arguments(state["parameters"]),
                stable_hash_cache=state.get("stable_hash_cache"),
                stateful_role=state["stateful_role"],
            )
        required = ("args", "kwargs")
        missing = [key for key in required if key not in state]
        if missing:
            raise ValueError(
                f"ConcreteDefinition state is missing required fields: {missing!r}."
            )
        return CDefIdentityRecord(
            version,
            state["cls"],
            args=state["args"],
            kwargs=state["kwargs"],
            stable_hash_cache=state.get("stable_hash_cache"),
        )

    raise TypeError(
        f"Unsupported ConcreteDefinition pickle state {type(state).__name__}."
    )


def stable_hash_domain(type_marker: str, version: int) -> str:
    """Return the structural-hash domain for one CDef identity version.

    Args:
        type_marker: Historical concrete-definition type marker.
        version: Validated CDef identity version.

    Returns:
        The unchanged marker for V1 or a distinct marker for V2.
    """

    version = validate_identity_version(version)
    if version == V1_IDENTITY_VERSION:
        return type_marker
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
