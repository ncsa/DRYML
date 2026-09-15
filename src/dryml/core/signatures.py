"""Explicit synchronous normalization for supported DRYML signatures.

This module owns the Ref/Mat vocabulary, flat annotation grammar, lossless
assertions, local activation, and read-only Repo-backed authority selection.
Materialization stays with Repo and its later whole-boundary admission seam.
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import inspect
import threading
import types
from collections.abc import Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Annotated, Any, Callable, get_args, get_origin, get_type_hints

from .bound_args import BoundArguments
from .cdef_graph import EdgeKind


class SignatureError(TypeError):
    """Raised for bounded signature grammar, binding, context, or delivery errors.

    Args:
        reason: Fixed diagnostic category describing the failed contract.
        slot: Optional parameter or return slot path.

    Errors never include caller value representations. Chained lower-level errors
    remain available as causes.
    """

    def __init__(self, reason: str, slot: str | None = None) -> None:
        self.reason, self.slot = reason, slot
        super().__init__(f"Signature normalization failed: {reason}{'' if slot is None else f' at {slot}' }.")


@dataclass(frozen=True, slots=True)
class _Role:
    """Passive immutable metadata attached to a top-level role annotation."""

    name: str


class _RoleVocabulary:
    """One callable/subscriptable Ref or Mat vocabulary singleton."""

    def __init__(self, name: str, kind: EdgeKind) -> None:
        self._name, self._kind = name, kind

    @property
    def kind(self) -> EdgeKind:
        """Return the edge kind asserted by value calls."""

        return self._kind

    def __call__(self, target: Any):
        """Create a lossless assertion without freezing, selecting, or saving.

        Args:
            target: Original caller value retained until normalization.

        Returns:
            An unresolved ``DefLink`` assertion.
        """

        from .links import DefLink

        return DefLink.assertion(self._kind, target)

    def __getitem__(self, target: Any) -> Any:
        """Lower one subscription to passive ``Annotated`` role metadata.

        Args:
            target: Flat target, nullable target, or same-role union.

        Returns:
            An annotation that remains inert until explicit activation.
        """

        return Annotated[target, _Role(self._name.lower())]

    def __repr__(self) -> str:
        """Return the stable public vocabulary spelling."""

        return self._name


class AutoRef:
    """Annotation-only marker for automatic selection via ``Ref[AutoRef]``.

    Use the class itself, not an instance. At an activated Ref boundary, it
    selects a CDef, ObjectRef, or StateRef from the supplied value without saving
    or materializing it. Unsupported inputs raise ``SignatureError``.
    """


Ref = _RoleVocabulary("Ref", EdgeKind.REF)
"""Callable/subscriptable reference assertion and annotation vocabulary."""
Mat = _RoleVocabulary("Mat", EdgeKind.MATERIALIZE)
"""Callable/subscriptable materializing assertion and annotation vocabulary."""


@dataclass(frozen=True, slots=True)
class _Slot:
    """Validated flat role metadata for one parameter or return boundary."""

    role: str
    targets: tuple[Any, ...]
    nullable: bool
    mode: str

    @property
    def explicit(self) -> bool:
        """Return whether the author declared an explicit Ref/Mat role."""

        return self.mode != "default"


@dataclass(frozen=True, slots=True)
class ReferenceSelection:
    """Pin one signature slot to an exact ObjectRef and optional declaration Store.

    Args:
        object_ref: Exact identity allowed for a CDef-to-ObjectRef strengthening.
        store: Optional connected Store required to carry that declaration.

    A selection is caller-owned control, not authority by itself. The ObjectRef
    must match the slot's complete CDef topology and occur in valid declaration
    authority during the Repo evidence cut. Materializing delivery revalidates a
    selected Store's connection, declaration, and ClaimRecord before effects.
    """

    object_ref: Any
    store: Any = None


@dataclass(slots=True)
class _Lease:
    """Lifetime bit held separately from immutable context options."""

    active: bool = True


@dataclass(frozen=True, slots=True)
class _ContextConfig:
    """Immutable caller-owned controls carried by one active lease."""

    repo: Any
    cache: Any
    reuse_live: Any
    selections: Mapping[Any, Any]
    owner: tuple[int, int | None]
    lease: _Lease


@dataclass(frozen=True, slots=True)
class _Controls:
    """Resolved explicit or ambient controls retained by one boundary."""

    repo: Any
    cache: Any
    reuse_live: Any
    selections: Mapping[Any, Any]
    borrowed_owner: tuple[int, int | None] | None = None
    borrowed_lease: _Lease | None = None


_ACTIVE_CONTEXT: contextvars.ContextVar[_ContextConfig | None] = contextvars.ContextVar(
    "dryml_signature_context", default=None
)


def _execution_identity() -> tuple[int, int | None]:
    """Return current thread/task identity without retaining either object."""

    try:
        task = asyncio.current_task()
    except RuntimeError:
        task = None
    return threading.get_ident(), None if task is None else id(task)


def _active_context() -> _ContextConfig | None:
    """Return valid ambient controls or fail before metadata/normalization work."""

    config = _ACTIVE_CONTEXT.get()
    if config is not None and (not config.lease.active or config.owner != _execution_identity()):
        raise SignatureError("borrowed context is unavailable")
    return config


class _SignatureContext(AbstractContextManager[_ContextConfig]):
    """Borrow caller-owned controls in one originating thread/task scope."""

    def __init__(self, repo: Any, cache: Any, reuse_live: Any, selections: Mapping[Any, Any]) -> None:
        self._config = _ContextConfig(
            repo,
            cache,
            reuse_live,
            MappingProxyType(dict(selections)),
            _execution_identity(),
            _Lease(),
        )
        self._token: contextvars.Token[_ContextConfig | None] | None = None

    def __enter__(self) -> _ContextConfig:
        """Install controls and return their immutable snapshot."""

        self._token = _ACTIVE_CONTEXT.set(self._config)
        return self._config

    def __exit__(self, exc_type, exc, traceback) -> None:
        """Invalidate and reset context in ``finally`` behavior."""

        self._config.lease.active = False
        assert self._token is not None
        _ACTIVE_CONTEXT.reset(self._token)


def signature_context(*, repo: Any = None, cache: Any = None, reuse_live: Any = None,
                      selections: Mapping[Any, Any] | None = None) -> AbstractContextManager[_ContextConfig]:
    """Borrow explicit controls outside target keyword arguments.

    Args:
        repo: Caller-owned Repo authority; it is never opened or closed here.
        cache: Optional existing Repo cache policy.
        reuse_live: Optional existing Repo live-reuse policy.
        selections: Exact selection controls keyed by parameter name,
            ``"return"``, ``(variadic_name, index)``, or
            ``(variadic_name, keyword)``.

    Returns:
        A context manager exposing an immutable snapshot valid only in the
        originating thread and task while entered.

    Raises:
        SignatureError: If ``selections`` is not a mapping.

    Side Effects:
        Installs and resets a ContextVar. It never saves or realizes.
    """

    if selections is None:
        selections = {}
    if not isinstance(selections, Mapping):
        raise SignatureError("selection controls must be a mapping")
    return _SignatureContext(repo, cache, reuse_live, selections)


def _is_union(annotation: Any) -> bool:
    """Return whether an annotation uses a supported union spelling."""

    import typing

    return get_origin(annotation) in (types.UnionType, typing.Union)


def _contains_role(annotation: Any) -> bool:
    """Return whether a role occurs below a top-level annotation boundary."""

    if get_origin(annotation) is Annotated:
        return any(isinstance(item, _Role) for item in get_args(annotation)[1:])
    return any(_contains_role(item) for item in get_args(annotation))


def _reject_object_subclass(annotation: Any, slot: str) -> None:
    """Reject Object subclasses while leaving ordinary non-DRYML hints inert."""

    from .object import Object

    if isinstance(annotation, type) and annotation is not Object and issubclass(annotation, Object):
        raise SignatureError("Object subclass annotations are unsupported", slot)


def _validate_target(target: Any, role: str, slot: str) -> None:
    """Validate one supported role category without selection or realization."""

    from .definition import ConcreteDefinition, Definition
    from .object import Object
    from .quoted import QuotedDef, SelectorSpec
    from .reference_values import ObjectRef, StateRef
    from .selector import Selector

    supported = (
        Definition, ConcreteDefinition, ObjectRef, StateRef, Object, AutoRef,
        QuotedDef, Selector, SelectorSpec,
    )
    if target not in supported:
        _reject_object_subclass(target, slot)
        raise SignatureError("role target is unsupported", slot)
    if target is Object and role != "mat":
        raise SignatureError("Object is only a materializing target", slot)
    if target is AutoRef and role != "ref":
        raise SignatureError("AutoRef is only a reference target", slot)
    if target in (QuotedDef, Selector, SelectorSpec) and role != "ref":
        raise SignatureError("quotation targets are only reference data", slot)


def _parse_slot(annotation: Any, slot: str) -> _Slot:
    """Parse a top-level role annotation into flat mode metadata."""

    if annotation is inspect.Signature.empty:
        return _Slot("mat", (), False, "default")
    branches = get_args(annotation) if _is_union(annotation) else (annotation,)
    role_branches = [branch for branch in branches if get_origin(branch) is Annotated]
    if not role_branches:
        if _contains_role(annotation):
            raise SignatureError("nested role annotation is unsupported", slot)
        return _Slot("mat", (), False, "default")

    roles: set[str] = set()
    members: list[Any] = []
    nullable = False
    for branch in branches:
        if branch is type(None):
            nullable = True
            continue
        if get_origin(branch) is not Annotated:
            raise SignatureError("outer role union supports only role branches and None", slot)
        args = get_args(branch)
        metadata = [item for item in args[1:] if isinstance(item, _Role)]
        if len(args) != 2 or len(metadata) != 1:
            raise SignatureError("role annotation metadata is invalid", slot)
        roles.add(metadata[0].name)
        target = args[0]
        if _contains_role(target):
            raise SignatureError("nested role annotation is unsupported", slot)
        target_members = get_args(target) if _is_union(target) else (target,)
        nullable = nullable or type(None) in target_members
        for member in target_members:
            if member is not type(None) and member not in members:
                members.append(member)
    if len(roles) != 1:
        raise SignatureError("mixed Ref/Mat unions are unsupported", slot)
    role = next(iter(roles))
    if not members:
        raise SignatureError("standalone None role target is unsupported", slot)
    for member in members:
        if get_origin(member) is not None:
            raise SignatureError("container role target is unsupported", slot)
        _validate_target(member, role, slot)
    if AutoRef in members:
        if role != "ref" or len(members) != 1:
            raise SignatureError("AutoRef cannot be combined with another target", slot)
        mode = "automatic"
    elif len(members) == 1:
        mode = "exact"
    else:
        from .definition import ConcreteDefinition, Definition
        from .reference_values import ObjectRef, StateRef

        if not all(member in (Definition, ConcreteDefinition, ObjectRef, StateRef) for member in members):
            raise SignatureError("role union targets are incomparable", slot)
        mode = "union"
    return _Slot(role, tuple(members), nullable, mode)


@dataclass(frozen=True, slots=True)
class SignaturePlan:
    """Immutable compiled target snapshot for argument and return boundaries.

    Args:
        target: Synchronous callable or constructor class selected at compilation.
        signature: Immutable effective Python invocation signature.
        slots: Parsed argument slots keyed by effective parameter name.
        return_slot: Parsed return slot, or ``None`` for constructor mode.
        constructor: Whether the plan represents class construction.

    The plan binds once, validates flat roles, selects authority through only
    explicitly supplied controls, and creates local one-shot ``BoundaryPlan``
    values. Official wrappers deliberately supply valid ambient controls. The
    plan never opens a Repo, saves, acquires a claim, or materializes structures.
    """

    target: Callable[..., Any]
    signature: inspect.Signature
    slots: Mapping[str, _Slot]
    return_slot: _Slot | None
    constructor: bool = False

    def prepare_constructor_args(self, args: tuple[Any, ...], kwargs: Mapping[str, Any], *,
                                 repo: Any = None, cache: Any = None,
                                 reuse_live: Any = None,
                                 selections: Mapping[Any, Any] | None = None) -> "BoundaryPlan":
        """Prepare and normalize one new constructor call exactly once.

        Args:
            args: Positional constructor arguments before the class hook runs.
            kwargs: Keyword constructor arguments before the class hook runs.
            repo: Explicit caller-owned Repo authority for later seams.
            cache: Optional Repo cache policy.
            reuse_live: Optional Repo live-reuse policy.
            selections: Optional exact selections keyed by ordinary parameter
                names or variadic ``(name, index-or-key)`` occurrences.

        Returns:
            An argument boundary whose records include prepared defaults and
            finalized compatible assertions.

        Raises:
            SignatureError: If this is not a constructor plan, the preparation
                hook returns an invalid call shape, or ordinary binding fails.
            Exception: Propagates preparation-hook and Repo metadata failures
                without changing their type.

        Side Effects:
            Calls the target class's ``__prepare_args__`` hook once and may read
            explicitly supplied Repo metadata during selection. It never
            materializes, saves, claims, or replays persisted constructor records.
            Explicit roles validate finalized structural links as fresh input;
            unannotated constructor slots retain compatible canonical links.
        """

        if not self.constructor:
            raise SignatureError("constructor preparation requires a constructor plan")
        prepared = self.target.__prepare_args__(*args, **dict(kwargs))
        if not isinstance(prepared, tuple) or len(prepared) != 2:
            raise SignatureError("constructor preparation returned an invalid call shape")
        prepared_args, prepared_kwargs = prepared
        if not isinstance(prepared_args, tuple) or not isinstance(prepared_kwargs, Mapping):
            raise SignatureError("constructor preparation returned an invalid call shape")
        return self.prepare_args(
            prepared_args,
            prepared_kwargs,
            repo=repo,
            cache=cache,
            reuse_live=reuse_live,
            selections=selections,
        )

    def prepare_args(self, args: tuple[Any, ...], kwargs: Mapping[str, Any], *, repo: Any = None,
                     cache: Any = None, reuse_live: Any = None,
                     selections: Mapping[Any, Any] | None = None) -> "BoundaryPlan":
        """Bind and normalize one call's arguments without realization.

        Args:
            args: Positional target arguments.
            kwargs: Keyword target arguments.
            repo: Explicit caller-owned Repo authority for later seams.
            cache: Optional Repo cache policy.
            reuse_live: Optional Repo live-reuse policy.
            selections: Optional exact selections keyed by ordinary parameter
                names or variadic ``(name, index-or-key)`` occurrences.

        Returns:
            An argument-mode boundary with authority and canonical views.

        Raises:
            SignatureError: If binding, controls, or assertions are invalid.
            Exception: Propagates explicitly requested Repo metadata failures
                without changing their type.

        Side Effects:
            Selection may read explicitly supplied Repo metadata. Preparation does
            not realize, save, claim, reserve, or consult ambient controls.
        """

        controls = _explicit_controls(repo, cache, reuse_live, selections)
        return self._prepare_args(args, kwargs, controls)

    def _prepare_args(self, args: tuple[Any, ...], kwargs: Mapping[str, Any],
                      controls: _Controls) -> "BoundaryPlan":
        """Bind using already resolved explicit or deliberately borrowed controls."""

        try:
            bound = self.signature.bind(*args, **dict(kwargs))
            bound.apply_defaults()
        except TypeError as error:
            raise SignatureError("argument binding failed") from error
        controls = _validate_controls(
            controls, _selection_keys(self.signature, bound.arguments)
        )
        authority, canonical = [], []
        for name, value in bound.arguments.items():
            normalized, canonical_value = _normalize_parameter_value(
                value, self.signature.parameters[name], self.slots[name], name, controls,
                persist_role=self.constructor, preserve_finalized_links=False,
            )
            authority.append((name, normalized))
            canonical.append((name, canonical_value))
        return BoundaryPlan(
            self, "args", BoundArguments(authority), BoundArguments(canonical), controls
        )

    def _prepare_ambient_args(self, args: tuple[Any, ...],
                              kwargs: Mapping[str, Any]) -> "BoundaryPlan":
        """Prepare arguments while deliberately borrowing the active context.

        This private seam is for owning local invocation adapters such as
        ``function`` and ``Method``. It retains the ambient lease on the returned
        boundary instead of copying its Repo authority into explicit controls.
        """

        return self._prepare_args(args, kwargs, _ambient_controls())

    def prepare_bound(self, arguments: BoundArguments, *, partial: bool = False, repo: Any = None,
                      cache: Any = None, reuse_live: Any = None,
                      selections: Mapping[Any, Any] | None = None) -> "BoundaryPlan":
        """Normalize an existing semantic record without rebinding or defaults.

        Args:
            arguments: Existing core semantic argument record.
            partial: Whether omitted constructor/query fields are permitted.
            repo: Explicit caller-owned Repo authority for later seams.
            cache: Optional Repo cache policy.
            reuse_live: Optional Repo live-reuse policy.
            selections: Optional exact selections keyed by ordinary parameter
                names or variadic ``(name, index-or-key)`` occurrences.

        Returns:
            An argument-mode boundary preserving supplied semantic names.

        Raises:
            SignatureError: If names, controls, or assertions are invalid.
            Exception: Propagates explicitly requested Repo metadata failures
                without changing their type.

        Side Effects:
            Selection may read explicitly supplied Repo metadata. Preparation does
            not bind, apply defaults, realize, save, or consult ambient controls.
            Finalized links in constructor records replay without fresh assertion
            validation.
        """

        controls = _explicit_controls(repo, cache, reuse_live, selections)
        return self._prepare_bound(arguments, partial=partial, controls=controls)

    def _prepare_bound(self, arguments: BoundArguments, *, partial: bool,
                       controls: _Controls) -> "BoundaryPlan":
        """Normalize one bound record using already resolved controls."""

        if not isinstance(arguments, BoundArguments):
            raise SignatureError("bound arguments must be a BoundArguments record")
        names, expected = set(arguments), set(self.signature.parameters)
        if not names <= expected or (not partial and names != expected):
            raise SignatureError("bound argument names do not match the signature")
        controls = _validate_controls(
            controls, _selection_keys(self.signature, arguments)
        )
        authority, canonical = [], []
        for name, value in arguments.items():
            normalized, canonical_value = _normalize_parameter_value(
                value, self.signature.parameters[name], self.slots[name], name, controls,
                persist_role=self.constructor,
                preserve_finalized_links=self.constructor,
            )
            authority.append((name, normalized))
            canonical.append((name, canonical_value))
        return BoundaryPlan(
            self, "args", BoundArguments(authority), BoundArguments(canonical), controls
        )

    def prepare_return(self, value: Any, *, repo: Any = None, cache: Any = None,
                       reuse_live: Any = None, selections: Mapping[Any, Any] | None = None) -> "BoundaryPlan":
        """Normalize one fresh return boundary without realization.

        Args:
            value: The already-produced target result.
            repo: Explicit caller-owned Repo authority for later seams.
            cache: Optional Repo cache policy.
            reuse_live: Optional Repo live-reuse policy.
            selections: Optional exact selection keyed by ``"return"``.

        Returns:
            A return-mode boundary with authority and canonical views.

        Raises:
            SignatureError: If controls or a return assertion are invalid.
            Exception: Propagates explicitly requested Repo metadata failures
                without changing their type.

        Side Effects:
            Selection may read explicitly supplied Repo metadata. Preparation does
            not realize, save, or consult ambient controls.
        """

        controls = _validate_controls(
            _explicit_controls(repo, cache, reuse_live, selections), {"return"}
        )
        return self._prepare_return(value, controls)

    def _prepare_return(self, value: Any, controls: _Controls) -> "BoundaryPlan":
        """Normalize one return using already resolved controls."""

        if self.return_slot is None:
            return BoundaryPlan(self, "return", value, value, controls)
        authority, canonical = _normalize_value(
            value, self.return_slot, "return", controls, selection_key="return"
        )
        return BoundaryPlan(self, "return", authority, canonical, controls)

    def _prepare_ambient_return(self, value: Any) -> "BoundaryPlan":
        """Prepare a return while deliberately borrowing the active context.

        This private integration seam validates return-only selections and
        retains the borrowed context lease for one-shot delivery.
        """

        controls = _validate_controls(_ambient_controls(), {"return"})
        return self._prepare_return(value, controls)


@dataclass(slots=True)
class BoundaryPlan:
    """Invocation-owned normalized boundary with one-shot synchronous delivery.

    Args:
        plan: Compiled signature that created this boundary.
        mode: Either ``"args"`` or ``"return"``.
        authority: Selected, non-realized values.
        canonical: Finalized canonical values safe for later core persistence.
        controls: Immutable explicit or deliberately borrowed normalization
            controls, including retained selection Store provenance.

    Selection is complete before this object is returned. U2 retains controls and
    selected source authority for U3's Repo-owned materialization admission.
    """

    plan: SignaturePlan
    mode: str
    authority: Any
    canonical: Any
    controls: _Controls
    _owner: tuple[int, int | None] = field(default_factory=_execution_identity, init=False, repr=False)
    _delivered: bool = field(default=False, init=False, repr=False)

    @property
    def selections(self) -> Mapping[Any, ReferenceSelection | None]:
        """Return immutable, validated selections retained for Repo admission.

        Returns:
            A mapping keyed by ordinary parameter names, ``"return"``, or
            variadic ``(parameter_name, occurrence)`` pairs. Values retain any
            caller-selected declaration Store provenance.

        Side Effects:
            None. The mapping is runtime-only and is not persisted or encoded.
        """

        return self.controls.selections

    def deliver_args(self, *, extra_mat_roots: tuple[Any, ...] = (),
                     reservation: Any = None,
                     reserved_live: Mapping[str, Any] | None = None) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Deliver normalized arguments once through the original signature shape.

        Returns:
            Positional and keyword values for target invocation.

        Args:
            extra_mat_roots: Additional selected materializing authorities that
                must share this boundary's aggregate preflight but are not
                delivered to the target.
            reservation: Optional active Repo state-graph reservation retained by
                a higher-level lifecycle owner.
            reserved_live: Optional exact-reference digest to live-object mapping
                admitted through ``reservation`` for overlap reuse.

        Raises:
            SignatureError: If mode, ownership, or one-shot state is invalid.
            Exception: Propagates Repo admission, constructor, reservation,
                restoration, and cache failures without changing their type.

        Side Effects:
            Materializing slots may cause Repo to claim, reserve, construct,
            restore, reuse, or cache Objects. Ref slots have no realization
            effects, and delivery never saves automatically.
        """

        self._consume("args")
        values = self._deliver_materializing_values(
            dict(self.authority.items()), extra_mat_roots=extra_mat_roots,
            reservation=reservation, reserved_live=reserved_live,
        )
        args, kwargs = [], {}
        for parameter in self.plan.signature.parameters.values():
            value = values[parameter.name]
            if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD):
                args.append(value)
            elif parameter.kind is parameter.VAR_POSITIONAL:
                args.extend(value)
            elif parameter.kind is parameter.KEYWORD_ONLY:
                kwargs[parameter.name] = value
            else:
                kwargs.update(value)
        return tuple(args), kwargs

    def deliver_return(self, *, reservation: Any = None,
                       reserved_live: Mapping[str, Any] | None = None) -> Any:
        """Deliver one normalized result exactly once.

        Returns:
            The local normalized return value.

        Raises:
            SignatureError: If mode, ownership, or one-shot state is invalid.
            Exception: Propagates Repo admission, constructor, reservation,
                restoration, and cache failures without changing their type.

        Args:
            reservation: Optional active Repo state-graph reservation retained by
                a higher-level lifecycle owner.
            reserved_live: Optional exact-reference digest to live-object mapping
                admitted through ``reservation`` for overlap reuse.

        Side Effects:
            A materializing return may cause Repo to claim, reserve, construct,
            restore, reuse, or cache Objects. Delivery never saves automatically.
        """

        self._consume("return")
        return self._deliver_materializing_values(
            {"return": self.authority}, reservation=reservation,
            reserved_live=reserved_live,
        )["return"]

    def _deliver_materializing_values(self, values: dict[str, Any], *,
                                      extra_mat_roots: tuple[Any, ...] = (),
                                      reservation: Any = None,
                                      reserved_live: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Delegate every Mat slot to one Repo-owned aggregate admission.

        Args:
            values: Selected argument or return values keyed by signature slot.
            extra_mat_roots: Selected materializing roots preflighted but not
                delivered through this boundary.
            reservation: Optional active state-graph reservation for reuse.
            reserved_live: Optional exact-reference digest to retained live-object
                mapping used only with ``reservation``.

        Returns:
            The same mapping with only materializing slots realized.

        Raises:
            SignatureError: If a materializing slot has no caller-supplied Repo
                authority.

        Side Effects:
            Repo may construct, restore, reserve, claim, or cache selected roots.
            Ref slots remain metadata-only and never enter this method's Repo call.
        """

        names = [
            name for name in values
            if (self.plan.return_slot if name == "return" else self.plan.slots[name]).role == "mat"
        ]
        if not names and not extra_mat_roots:
            return values
        from .definition import ConcreteDefinition, Definition
        from .links import DefLink
        from .object import Object
        from .reference_values import ObjectRef, StateRef

        def requires_repo(value: Any, seen: set[int]) -> bool:
            """Return whether a value can trigger Repo-owned realization."""

            if id(value) in seen:
                return False
            seen.add(id(value))
            if isinstance(value, Object):
                return False
            if isinstance(value, (Definition, ConcreteDefinition, ObjectRef, StateRef)):
                return True
            if isinstance(value, DefLink):
                return value.kind is EdgeKind.MATERIALIZE and requires_repo(value.target, seen)
            if isinstance(value, Mapping):
                return any(requires_repo(item, seen) for item in value.values())
            if isinstance(value, (list, tuple, set, frozenset)):
                return any(requires_repo(item, seen) for item in value)
            return False

        roots = tuple(values[name] for name in names) + tuple(extra_mat_roots)
        if not any(requires_repo(value, set()) for value in roots):
            return values
        if self.controls.repo is None:
            raise SignatureError("materializing delivery requires a repo")
        materialize = getattr(self.controls.repo, "materialize_boundary", None)
        if not callable(materialize):
            raise SignatureError("repo does not support aggregate materialization")
        declaration_stores = {}
        for key, selection in self.controls.selections.items():
            if selection is None or selection.store is None:
                continue
            name = key[0] if isinstance(key, tuple) else key
            if name not in names:
                continue
            slot = self.plan.return_slot if name == "return" else self.plan.slots[name]
            if slot.role != "mat":
                continue
            digest = selection.object_ref.digest()
            previous = declaration_stores.get(digest)
            if previous is not None and previous is not selection.store:
                previous_key = previous.authority_fence_key()
                selected_key = selection.store.authority_fence_key()
                if previous_key != selected_key:
                    raise SignatureError(
                        "materializing identity has conflicting declaration Store selections",
                        _selection_label(key),
                    )
                continue
            declaration_stores[digest] = selection.store
        realized = materialize(
            roots,
            cache=self.controls.cache,
            reuse_live=self.controls.reuse_live,
            reservation=reservation,
            reserved_live=reserved_live,
            declaration_stores=MappingProxyType(declaration_stores),
        )
        if len(realized) != len(roots):
            raise SignatureError("repo aggregate materialization returned an invalid result")
        values.update(zip(names, realized))
        return values

    def _consume(self, expected: str) -> None:
        """Validate delivery mode, execution ownership, and one-shot state."""

        if self.mode != expected:
            raise SignatureError("boundary delivery mode is invalid")
        if self._owner != _execution_identity():
            raise SignatureError("boundary belongs to another thread or task")
        if self.controls.borrowed_lease is not None and (
            not self.controls.borrowed_lease.active
            or self.controls.borrowed_owner != _execution_identity()
        ):
            raise SignatureError("borrowed context is unavailable")
        if self._delivered:
            raise SignatureError("boundary has already been delivered")
        self._delivered = True


def _selection_keys(signature: inspect.Signature,
                    arguments: Mapping[str, Any]) -> set[Any]:
    """Return exact ordinary and variadic occurrence keys for one bound record."""

    valid: set[Any] = set()
    for name, value in arguments.items():
        parameter = signature.parameters[name]
        if parameter.kind is parameter.VAR_POSITIONAL:
            if not isinstance(value, tuple):
                raise SignatureError("variadic positional value must be a tuple", name)
            valid.update((name, index) for index in range(len(value)))
        elif parameter.kind is parameter.VAR_KEYWORD:
            if not isinstance(value, Mapping) or any(
                not isinstance(key, str) for key in value
            ):
                raise SignatureError("variadic keyword value must be a mapping", name)
            valid.update((name, key) for key in value)
        else:
            valid.add(name)
    return valid


def _selection_label(key: Any) -> str:
    """Render one bounded semantic selection key without payload values."""

    if isinstance(key, tuple) and len(key) == 2 and isinstance(key[0], str):
        occurrence = key[1]
        if isinstance(occurrence, int):
            return f"{key[0]}[{occurrence}]"
        if isinstance(occurrence, str):
            return f"{key[0]}[{occurrence!r}]"
    return key if isinstance(key, str) else "<selection>"


def _validate_controls(controls: _Controls, valid: set[Any]) -> _Controls:
    """Reject malformed paths and values before normalization or evidence reads."""

    if not isinstance(controls.selections, Mapping):
        raise SignatureError("selection controls must be a mapping")
    if any(key not in valid for key in controls.selections):
        raise SignatureError("selection controls are invalid")
    selections = {
        key: _selection_value(value, _selection_label(key))
        for key, value in controls.selections.items()
    }
    return replace(controls, selections=MappingProxyType(selections))


def _snapshot_selections(selections: Mapping[Any, Any] | None) -> Mapping[Any, Any]:
    """Copy one caller mapping into an immutable control snapshot."""

    if selections is None:
        return MappingProxyType({})
    if not isinstance(selections, Mapping):
        raise SignatureError("selection controls must be a mapping")
    return MappingProxyType(dict(selections))


def _explicit_controls(repo: Any, cache: Any, reuse_live: Any,
                       selections: Mapping[Any, Any] | None) -> _Controls:
    """Resolve only explicitly supplied compiled-API controls."""

    return _Controls(repo, cache, reuse_live, _snapshot_selections(selections))


def _ambient_controls() -> _Controls:
    """Deliberately borrow valid ambient controls for official local helpers."""

    ambient = _active_context()
    if ambient is None:
        return _Controls(None, None, None, MappingProxyType({}))
    return _Controls(
        ambient.repo,
        ambient.cache,
        ambient.reuse_live,
        ambient.selections,
        ambient.owner,
        ambient.lease,
    )


def _normalize_parameter_value(value: Any, parameter: inspect.Parameter,
                               slot: _Slot, name: str, controls: _Controls,
                               *, persist_role: bool,
                               preserve_finalized_links: bool = False) -> tuple[Any, Any]:
    """Normalize each explicitly annotated variadic occurrence independently."""

    if slot.explicit and parameter.kind is parameter.VAR_POSITIONAL:
        authority, canonical = [], []
        for index, item in enumerate(value):
            selected, frozen = _normalize_value(
                item,
                slot,
                f"{name}[{index}]",
                controls,
                persist_role=persist_role,
                preserve_finalized_links=preserve_finalized_links,
                selection_key=(name, index),
            )
            authority.append(selected)
            canonical.append(frozen)
        return tuple(authority), tuple(canonical)
    if slot.explicit and parameter.kind is parameter.VAR_KEYWORD:
        if not isinstance(value, Mapping):
            raise SignatureError("variadic keyword value must be a mapping", name)
        authority, canonical = {}, {}
        for key, item in value.items():
            selected, frozen = _normalize_value(
                item,
                slot,
                f"{name}[{key!r}]",
                controls,
                persist_role=persist_role,
                preserve_finalized_links=preserve_finalized_links,
                selection_key=(name, key),
            )
            authority[key] = selected
            canonical[key] = frozen
        return authority, canonical
    return _normalize_value(
        value,
        slot,
        name,
        controls,
        persist_role=persist_role,
        preserve_finalized_links=preserve_finalized_links,
        selection_key=name,
    )


def _normalize_value(value: Any, slot: _Slot, name: str, controls: _Controls,
                     *, persist_role: bool = False,
                     preserve_finalized_links: bool = False,
                     selection_key: Any = None) -> tuple[Any, Any]:
    """Validate assertions, select authority, and finalize compatible links."""

    from .links import DefLink

    asserted = False
    if isinstance(value, DefLink):
        if value.is_finalized:
            # Decoded and structural links already carry their edge authority.
            # Bound replay and unannotated constructor structure stay inert;
            # fresh explicit constructor roles still enforce their exact edge.
            if preserve_finalized_links or (persist_role and not slot.explicit):
                return value, value
            expected = EdgeKind.REF if slot.role == "ref" else EdgeKind.MATERIALIZE
            if value.kind is not expected:
                raise SignatureError("value assertion conflicts with the declared role", name)
            value, asserted = value.target, True
        else:
            expected = EdgeKind.REF if slot.role == "ref" else EdgeKind.MATERIALIZE
            if value.kind is not expected:
                raise SignatureError("value assertion conflicts with the declared role", name)
            if value.target is None:
                reason = (
                    "nullable role accepts only plain None"
                    if slot.nullable
                    else "None does not satisfy this role target"
                )
                raise SignatureError(reason, name)
            value, asserted = value.target, True
    if value is None and slot.targets and not slot.nullable:
        raise SignatureError("None does not satisfy this role target", name)
    selected = _select_authority(
        value,
        slot,
        name,
        controls,
        selection_key=name if selection_key is None else selection_key,
    )
    if persist_role and slot.role == "ref" and slot.mode == "exact":
        from .definition import Definition
        from .quoted import QuotedDef, SelectorSpec
        from .selector import Selector

        # Definition and selector roles carry expression data, not graph edges.
        # Quoting happens here, after the one owning signature has selected it.
        if slot.targets[0] is Definition and isinstance(selected, Definition):
            return selected, DefLink.finalized(EdgeKind.REF, QuotedDef(selected))
        if slot.targets[0] is Selector and isinstance(selected, Selector):
            return selected, DefLink.finalized(EdgeKind.REF, SelectorSpec(selected))
        if selected is None or isinstance(selected, (QuotedDef, SelectorSpec)):
            return selected, selected
    if asserted or (persist_role and slot.role == "ref" and selected is not None):
        return selected, DefLink.finalized(
            EdgeKind.REF if slot.role == "ref" else EdgeKind.MATERIALIZE,
            selected,
        )
    return selected, selected


class _SelectionUnavailable(Exception):
    """Private control flow for a union member that cannot be strengthened."""


class _SelectionAmbiguous(Exception):
    """Private control flow for an authority choice with multiple identities."""


def _select_authority(value: Any, slot: _Slot, name: str,
                      controls: _Controls, *, selection_key: Any) -> Any:
    """Choose one role-compatible authority without realizing or saving it."""

    from .quoted import QuotedDef, SelectorSpec

    if slot.role == "mat" and isinstance(value, (QuotedDef, SelectorSpec)):
        raise SignatureError("quoted input cannot implicitly materialize", name)
    if not slot.targets:
        return value
    if value is None:
        return None
    if slot.mode == "automatic":
        try:
            return _select_automatic(value, name)
        except _SelectionUnavailable as error:
            raise SignatureError("requested authority is unavailable", name) from error
    if slot.mode == "exact":
        try:
            return _select_exact(
                value,
                slot.targets[0],
                name,
                controls,
                selection_key=selection_key,
            )
        except _SelectionUnavailable as error:
            raise SignatureError("requested authority is unavailable", name) from error
        except _SelectionAmbiguous as error:
            raise SignatureError("requested authority is ambiguous", name) from error
    definition, cdef, object_ref, state_ref = _authority_types()
    ranks = {
        state_ref: 3,
        object_ref: 2,
        cdef: 1,
        definition: 0,
    }
    for target in sorted(slot.targets, key=lambda item: ranks[item], reverse=True):
        try:
            return _select_exact(
                value, target, name, controls, selection_key=selection_key
            )
        except _SelectionUnavailable:
            continue
        except _SelectionAmbiguous as error:
            raise SignatureError("requested authority is ambiguous", name) from error
    raise SignatureError("requested authority is unavailable", name)


def _authority_types() -> tuple[Any, Any, Any, Any]:
    """Return authority types in relaxation order without import-time cycles."""

    from .definition import ConcreteDefinition, Definition
    from .reference_values import ObjectRef, StateRef

    return Definition, ConcreteDefinition, ObjectRef, StateRef


def _select_automatic(value: Any, name: str) -> Any:
    """Select graph-aware Ref authority from supplied local information only."""

    from .cdef_graph import has_stateful_materialization
    from .definition import ConcreteDefinition
    from .object import Object
    from .reference_values import ObjectRef, StateRef

    if isinstance(value, (ConcreteDefinition, ObjectRef, StateRef)):
        return value
    if not isinstance(value, Object):
        raise _SelectionUnavailable()
    cdef = value.definition
    if not has_stateful_materialization(cdef):
        return cdef
    receipt = value.last_state_ref
    if receipt is not None and receipt.object == value.object_ref:
        return receipt
    return value.object_ref


def _select_exact(value: Any, target: Any, name: str,
                  controls: _Controls, *, selection_key: Any) -> Any:
    """Apply one exact authority or quotation conversion under source guards."""

    from .definition import ConcreteDefinition, Definition
    from .object import Object
    from .quoted import QuotedDef, SelectorSpec
    from .reference_values import ObjectRef, StateRef
    from .selector import Selector

    if target is QuotedDef:
        return value if isinstance(value, QuotedDef) else QuotedDef(
            _select_exact(
                value, Definition, name, controls, selection_key=selection_key
            )
        )
    if target is Selector:
        if isinstance(value, SelectorSpec):
            return value.selector
        if isinstance(value, Selector):
            return value
        return Selector(_select_exact(
            value, Definition, name, controls, selection_key=selection_key
        ))
    if target is SelectorSpec:
        if isinstance(value, SelectorSpec):
            return value
        selector = value if isinstance(value, Selector) else Selector(
            _select_exact(
                value, Definition, name, controls, selection_key=selection_key
            )
        )
        return SelectorSpec(selector)
    if isinstance(value, QuotedDef):
        if target is Definition:
            return value.value
        raise _SelectionUnavailable()
    if isinstance(value, SelectorSpec):
        if target is Selector:
            return value.selector
        raise _SelectionUnavailable()
    if target is Definition:
        if isinstance(value, Definition):
            return value
        if isinstance(value, (ConcreteDefinition, ObjectRef, StateRef)):
            return value.definition.thaw()
        if isinstance(value, Object):
            return value.definition.thaw()
        raise _SelectionUnavailable()
    if target is Object:
        # Object is a delivery category, not an authority rung.  Preserve the
        # selected value for Repo's aggregate admission, which owns any required
        # realization while direct live Objects retain their unsaved payload.
        if isinstance(value, (Definition, ConcreteDefinition, Object, ObjectRef, StateRef)):
            return value
        raise _SelectionUnavailable()
    if target is ConcreteDefinition:
        return _as_cdef(value, name)
    if target is ObjectRef:
        return _as_object_ref(
            value, name, controls, selection_key=selection_key
        )
    if target is StateRef:
        return _as_state_ref(
            value, name, controls, selection_key=selection_key
        )
    raise _SelectionUnavailable()


def _as_cdef(value: Any, name: str) -> Any:
    """Relax or guarded-strengthen one value to a complete CDef."""

    from .definition import ConcreteDefinition, Definition
    from .object import Object
    from .reference_values import ObjectRef, StateRef

    if isinstance(value, ConcreteDefinition):
        return value
    if isinstance(value, (ObjectRef, StateRef)):
        return value.definition
    if isinstance(value, Object):
        return value.definition
    if not isinstance(value, Definition):
        raise _SelectionUnavailable()
    try:
        cdef = value.concretize()
        supplied = value.parameters
    except Exception as error:
        raise _SelectionUnavailable() from error
    if set(supplied) != set(cdef.parameters) or any(
        supplied[key] != cdef.parameters[key] for key in supplied
    ):
        raise _SelectionUnavailable()
    return cdef


def _selection_value(selection: Any, name: str) -> ReferenceSelection | None:
    """Normalize one accepted slot selection shape before evidence reads."""

    from .reference_values import ObjectRef

    if selection is None:
        return None
    if isinstance(selection, ReferenceSelection):
        result = selection
    elif isinstance(selection, ObjectRef):
        result = ReferenceSelection(selection)
    elif isinstance(selection, tuple) and len(selection) == 2:
        result = ReferenceSelection(selection[0], selection[1])
    else:
        raise SignatureError("selection control is invalid", name)
    if not isinstance(result.object_ref, ObjectRef):
        raise SignatureError("selection control is invalid", name)
    return result


def _evidence_for(cdef: Any, name: str, controls: _Controls):
    """Read one required Repo authority cut without opening or modifying a Repo."""

    if controls.repo is None:
        raise _SelectionUnavailable()
    evidence = getattr(controls.repo, "reference_evidence", None)
    if not callable(evidence):
        raise SignatureError("repo does not support reference evidence", name)
    return evidence(cdef)


def _as_object_ref(value: Any, name: str, controls: _Controls, *,
                   selection_key: Any, evidence: Any = None) -> Any:
    """Relax or select a declared exact ObjectRef without acquiring a claim."""

    from .definition import ConcreteDefinition
    from .object import Object
    from .reference_values import ObjectRef, StateRef

    selection = controls.selections.get(selection_key)
    if isinstance(value, StateRef):
        result = value.object
        if selection is not None and selection.object_ref != result:
            raise SignatureError("selection topology does not match the slot", name)
        return result
    if isinstance(value, ObjectRef):
        if selection is not None and selection.object_ref != value:
            raise SignatureError("selection topology does not match the slot", name)
        return value
    if isinstance(value, Object):
        result = value.object_ref
        if selection is not None and selection.object_ref != result:
            raise SignatureError("selection topology does not match the slot", name)
        return result
    cdef = _as_cdef(value, name)
    if not isinstance(cdef, ConcreteDefinition):
        raise _SelectionUnavailable()
    if selection is not None and not selection.object_ref.definition.graph_equal(cdef):
        raise SignatureError("selection topology does not match the slot", name)
    evidence = _evidence_for(cdef, name, controls) if evidence is None else evidence
    candidates = list(evidence.declarations)
    if selection is not None:
        candidates = [
            candidate for candidate in candidates
            if candidate.object_ref == selection.object_ref
            and (selection.store is None or selection.store in candidate.stores)
        ]
    if not candidates:
        raise _SelectionUnavailable()
    if len(candidates) != 1:
        raise _SelectionAmbiguous()
    return candidates[0].object_ref


def _as_state_ref(value: Any, name: str, controls: _Controls, *,
                  selection_key: Any) -> Any:
    """Select existing exact StateRef authority without snapshot publication."""

    from .cdef_graph import has_stateful_materialization
    from .object import Object
    from .reference_values import ObjectRef, StateRef

    selection = controls.selections.get(selection_key)
    if isinstance(value, StateRef):
        if not has_stateful_materialization(value.definition):
            raise _SelectionUnavailable()
        if selection is not None and selection.object_ref != value.object:
            raise SignatureError("selection topology does not match the slot", name)
        return value
    if isinstance(value, Object):
        if not has_stateful_materialization(value.definition):
            raise _SelectionUnavailable()
        receipt = value.last_state_ref
        if receipt is None or receipt.object != value.object_ref:
            raise _SelectionUnavailable()
        if selection is not None and selection.object_ref != receipt.object:
            raise SignatureError("selection topology does not match the slot", name)
        return receipt
    if isinstance(value, ObjectRef):
        reference = _as_object_ref(
            value, name, controls, selection_key=selection_key
        )
        evidence = _evidence_for(reference.definition, name, controls)
    else:
        cdef = _as_cdef(value, name)
        evidence = _evidence_for(cdef, name, controls)
        reference = _as_object_ref(
            value,
            name,
            controls,
            selection_key=selection_key,
            evidence=evidence,
        )
    if not isinstance(reference, ObjectRef) or not has_stateful_materialization(reference.definition):
        raise _SelectionUnavailable()
    candidates = [
        candidate.state_ref for candidate in evidence.states
        if candidate.state_ref.object == reference
    ]
    if not candidates:
        raise _SelectionUnavailable()
    if len(candidates) != 1:
        raise _SelectionAmbiguous()
    return candidates[0]


def _annotation_target(target: Callable[..., Any], constructor: bool) -> Callable[..., Any]:
    """Return the callable whose annotations govern one effective invocation."""

    if constructor:
        if not inspect.isclass(target):
            raise SignatureError("constructor mode requires a class target")
        return target.__init__
    if inspect.isfunction(target) or inspect.ismethod(target):
        return target
    call = getattr(target, "__call__", None)
    if not callable(call):
        raise SignatureError("target is not a supported callable")
    return call


def compile_signature(target: Callable[..., Any], *, constructor: bool = False,
                      annotation_namespace: Mapping[str, Any] | None = None) -> SignaturePlan:
    """Compile one synchronous function, method, callable instance, or constructor.

    Args:
        target: Callable target, or a class when ``constructor`` is ``True``.
        constructor: Compile the class construction signature and ignore returns.
        annotation_namespace: Optional trusted names for annotation activation.

    Returns:
        An immutable target-signature snapshot.

    Raises:
        SignatureError: If target form, annotation activation, or flat grammar is
            unsupported.

    Side Effects:
        Resolves trusted annotations only at this explicit activation boundary.
    """

    if not callable(target) or (inspect.isclass(target) and not constructor):
        raise SignatureError("target is not a supported callable")
    annotation_target = _annotation_target(target, constructor)
    if inspect.iscoroutinefunction(annotation_target) or inspect.isgeneratorfunction(annotation_target) or inspect.isasyncgenfunction(annotation_target):
        raise SignatureError("async and generator targets are unsupported")
    try:
        signature = inspect.signature(annotation_target if constructor else target)
        if constructor:
            parameters = tuple(signature.parameters.values())
            if parameters and parameters[0].name == "self":
                signature = signature.replace(parameters=parameters[1:])
        hints = get_type_hints(annotation_target, globalns=None if annotation_namespace is None else dict(annotation_namespace), include_extras=True)
    except Exception as error:
        raise SignatureError("trusted annotation activation failed") from error
    slots = MappingProxyType({
        name: _parse_slot(hints.get(name, parameter.annotation), name)
        for name, parameter in signature.parameters.items()
    })
    return_slot = None if constructor else _parse_slot(hints.get("return", signature.return_annotation), "return")
    return SignaturePlan(target, signature, slots, return_slot, constructor)


def _discover_target() -> Callable[..., Any]:
    """Infer one immediate caller target only from a unique code-object binding."""

    frame = inspect.currentframe()
    caller = None if frame is None or frame.f_back is None else frame.f_back.f_back
    try:
        if caller is None:
            raise SignatureError("immediate caller is unavailable")
        code, candidates = caller.f_code, []
        for value in (*caller.f_locals.values(), *caller.f_globals.values()):
            if inspect.ismethod(value) and value.__func__.__code__ is code:
                candidates.append(value)
            elif inspect.isfunction(value) and value.__code__ is code:
                candidates.append(value)
        receiver = caller.f_locals.get("self")
        if receiver is not None:
            for cls in type(receiver).__mro__:
                for name, value in cls.__dict__.items():
                    if inspect.isfunction(value) and value.__code__ is code:
                        candidates.append(getattr(receiver, name))
        unique = {id(candidate): candidate for candidate in candidates}
        if len(unique) != 1:
            raise SignatureError("immediate caller target is ambiguous or unavailable")
        return next(iter(unique.values()))
    finally:
        del caller
        del frame


def normalize_args(*args: Any, **kwargs: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Discover and normalize the unique immediate caller's arguments.

    Args:
        *args: Arguments supplied to the immediate caller.
        **kwargs: Keyword arguments supplied to the immediate caller.

    Returns:
        The normalized ``(args, kwargs)`` pair.

    Raises:
        SignatureError: If caller discovery, binding, or validation fails.
        Exception: Propagates Repo admission, construction, restoration, and
            reservation failures without changing their type.

    Side Effects:
        Deliberately borrows active ``signature_context`` controls. Mat delivery
        may realize through Repo; this helper never saves automatically.
    """

    plan = compile_signature(_discover_target())
    return plan._prepare_ambient_args(args, kwargs).deliver_args()


def normalize_return(value: Any, **kwargs: Any) -> Any:
    """Discover and normalize one immediate caller return value.

    Args:
        value: Result selected by the immediate caller body.
        **kwargs: Rejected unless empty; retained only to surface accidental use.

    Returns:
        The normalized return value.

    Raises:
        SignatureError: If caller discovery, return validation, or keyword use
            is invalid.
        Exception: Propagates Repo admission, construction, restoration, and
            reservation failures without changing their type.

    Side Effects:
        Deliberately borrows active ``signature_context`` controls. A Mat return
        may realize through Repo; this helper never saves automatically or rolls
        back effects already performed by the caller.
    """

    if kwargs:
        raise SignatureError("return normalization does not accept target keywords")
    plan = compile_signature(_discover_target())
    return plan._prepare_ambient_return(value).deliver_return()


def function(target: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap one synchronous target with explicit argument and return activation.

    Args:
        target: Function, bound method, or synchronous callable instance.

    Returns:
        A wrapper retaining the original effective signature and metadata.

    Raises:
        SignatureError: If target compilation fails before invocation.
        Exception: Propagates target, Repo admission, constructor, restoration,
            and reservation failures without changing their type.

    Side Effects:
        Deliberately borrows active ``signature_context`` controls for each input
        and return boundary. Mat slots may realize through Repo. The wrapper
        invokes the target once, never saves or opens a Repo, and never consumes
        target keywords as controls.
    """

    plan = compile_signature(target)

    @functools.wraps(target)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        call_args, call_kwargs = plan._prepare_ambient_args(args, kwargs).deliver_args()
        result = target(*call_args, **call_kwargs)
        return plan._prepare_ambient_return(result).deliver_return()

    wrapped.__signature__ = plan.signature
    wrapped.__dryml_signature_plan__ = plan
    return wrapped
