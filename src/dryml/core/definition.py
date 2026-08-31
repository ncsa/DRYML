from __future__ import annotations

from copy import deepcopy
from inspect import isclass
import numpy as np
import sys
#import weakref
from collections.abc import Mapping
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

from .utils.stable_hash import stable_int_hash, stable_hash_function
from .cdef_identity import (
    V2_IDENTITY_VERSION,
    new_node_id,
    decode_identity_record,
    validate_identity_version,
)
from .utils.types import is_nonclass_callable
from .utils.general import get_class_str
from .utils.graph import GraphCtx, GraphTransformer, GraphMatcher
from .types import is_pod
from .freeze import FrozenDict, FrozenTuple
from .errors import PathAccessError
from .policies import CachePolicy
from .canonical import (
    to_canonical,
    thaw_value,
    NodeKind,
    matching_container_family,
    node_kind)
from .symbol import ImportRef, SourceSpec, maybe_symbol_ref, resolve_symbol
from .bound_args import BoundArguments, validate_canonical_bound_arguments

# Special value to skip args
SKIP_ARGS = object()


class DefInterface(ABC):
    """Common read, exact-construction, and matching interface for definitions.

    Implementations expose a class reference and constructor representation.
    ``Definition`` is a partial expression, while ``ConcreteDefinition`` is an
    exact identity that can be materialized through a repository.
    """
    @property
    @abstractmethod
    def cls(self):
        ...

    @property
    @abstractmethod
    def args(self):
        ...

    @property
    @abstractmethod
    def kwargs(self):
        ...

    @abstractmethod
    def concretize(self, repo: "Repo | None"=None) -> Any:
        ...

    def categorical(self, recursive=False):
        """Return a categorical form with unique arguments removed.

        Args:
            recursive: Whether to strip unique arguments from nested definitions
                as well as this definition.

        Returns:
            A transformed definition expression.

        Raises:
            ValueError: If a definition required for categorization has no class.
        """
        return categorical_definition(self, recursive=recursive)

    def build(
            self, *,
            repo: "Repo | None" = None,
            cache: "CachePolicy" = "weak") -> Object:
        """Freshly materialize this structural definition through a repository.

        Args:
            repo: Repository used to resolve links and persisted objects.
            cache: Cache policy for the resulting object.

        Returns:
            The loaded or newly materialized runtime object.

        Raises:
            TypeError: If this definition cannot be concretized or materialized.

        Side Effects:
            May create a runtime object and populate repository caches. It does
            not change this immutable definition.
        """
        from dryml.runtime import materialization_admission
        from .repo import manage_repo
        from .session import _construction_config

        with materialization_admission(operation="definition_build"):
            with manage_repo(repo=repo) as sub_repo:
                concrete_def = self.concretize(repo=sub_repo)
                with _construction_config():
                    return sub_repo.load_or_build(concrete_def, cache=cache)

    def match(self, other_def, *, strict: bool=False, verbose: bool=False, **sel_kwargs) -> bool:
        """Test this definition against another definition using selector rules.

        Args:
            other_def: Candidate definition or object to test.
            strict: Whether matching requires exact rather than selector
                semantics.
            verbose: Whether matching diagnostics are emitted.
            **sel_kwargs: Selector policy options, including ``cls_policy``.

        Returns:
            ``True`` when the candidate satisfies the applicable selector or
            exact-definition semantics.

        Raises:
            TypeError: If strict matching encounters an unsupported selector.
        """
        from .selector import Selector

        sel_kwargs.pop("full_diagnostic", None)
        sel_kwargs.pop("output_stream", None)
        sel_kwargs.pop("cls_str_compare", None)
        if "class_match" in sel_kwargs and "cls_policy" not in sel_kwargs:
            sel_kwargs["cls_policy"] = sel_kwargs.pop("class_match")
        if isinstance(self, Definition):
            return Selector(self, strict=strict, **sel_kwargs).matches(other_def, verbose=verbose)
        from .query.query import _query_match

        class_match = sel_kwargs.pop("cls_policy", sel_kwargs.pop("class_match", "selector"))
        return _query_match(self, other_def, strict=strict, class_match=class_match)

    def __call__(self, other_def, *, strict: bool=False, verbose: bool=False, **sel_kwargs):
        return self.match(other_def, strict=strict, verbose=verbose, **sel_kwargs)

    def stable_hash(self):
        """Return the deterministic hash for this definition representation.

        Returns:
            A stable hexadecimal content hash.
        """
        return stable_hash_function(self)


@dataclass(frozen=True, slots=True, init=False, eq=False)
class Definition(DefInterface, Mapping):
    """Immutable, partial structural graph expression.

    Values supplied at construction and update boundaries are deeply frozen so
    user-owned containers cannot mutate the Definition after creation. A
    Definition preserves omission: its ``parameters`` mapping contains only
    supplied values and never applies constructor defaults. It can therefore
    represent partial selector or search-space intent as well as a construction
    recipe. ``concretize()`` produces a fully bound exact V2
    ``ConcreteDefinition`` when its class and values are materializable.

    Args:
        *args: Optional class or symbol reference followed by positional
            constructor values. Use ``SKIP_ARGS`` after a class to preserve
            keyword-only partial intent.
        **kwargs: Supplied constructor values, retained as immutable fields.

    Raises:
        ValueError: If the leading value is not a supported class, callable, or
            symbol reference, or ``SKIP_ARGS`` is used incorrectly.
    """

    _cls: Callable[..., Any] | type | ImportRef | SourceSpec | None
    _args: FrozenTuple | None
    _kwargs: FrozenDict[str, Any]
    _stable_hash_cache: str | None = field(default=None, init=False, repr=False, compare=False, hash=False)

    def __init__(self, *args, **kwargs):
        """Create an immutable partial expression from supplied call spelling.

        Args:
            *args: Class/symbol reference and optional positional values.
            **kwargs: Explicit keyword values.

        Raises:
            ValueError: If the class or ``SKIP_ARGS`` form is invalid.

        Side Effects:
            Deeply freezes supplied definition values; caller-owned containers
            cannot subsequently mutate this expression.
        """
        object.__setattr__(self, "_stable_hash_cache", None)
        if len(args) > 0:
            if not callable(args[0]) and not isclass(args[0]) and not isinstance(args[0], (ImportRef, SourceSpec)):
                if args[0] is SKIP_ARGS and len(args) == 1:
                    object.__setattr__(self, "_cls", None)
                    object.__setattr__(self, "_args", None)
                    object.__setattr__(self, "_kwargs", self._freeze_kwargs(kwargs))
                else:
                    raise ValueError("First positional argument must be a class, callable, or symbol reference.")
            if len(args) > 1 and args[1] is SKIP_ARGS:
                if len(args) > 2:
                    raise ValueError("SKIP_ARGS must be the only positional argument besides the class.")

                object.__setattr__(self, "_cls", self._freeze_value(args[0]))
                object.__setattr__(self, "_args", None)
                object.__setattr__(self, "_kwargs", self._freeze_kwargs(kwargs))

            else:
                object.__setattr__(self, "_cls", self._freeze_value(args[0]))
                object.__setattr__(self, "_args", FrozenTuple(self._freeze_value(v) for v in args[1:]))
                object.__setattr__(self, "_kwargs", self._freeze_kwargs(kwargs))
        else:
            object.__setattr__(self, "_cls", None)
            object.__setattr__(self, "_args", FrozenTuple())
            object.__setattr__(self, "_kwargs", self._freeze_kwargs(kwargs))

    @staticmethod
    def _freeze_value(value):
        from .canonical import freeze_def_value

        return freeze_def_value(value)

    @classmethod
    def _freeze_kwargs(cls, kwargs):
        return FrozenDict({k: cls._freeze_value(v) for k, v in kwargs.items()})

    @property
    def cls(self):
        return self._cls

    @property
    def args(self):
        return self._args

    @property
    def kwargs(self):
        return self._kwargs

    @property
    def skip_args(self) -> bool:
        if self._args is None:
            return True
        else:
            return False

    @property
    def parameters(self) -> FrozenDict[str, Any]:
        """Return the immutable record of supplied semantic parameters.

        A live class is partially bound without applying defaults, so positional
        fields receive their declared names. Symbolic and callable definitions
        retain explicit keyword fields without resolution; positional values on
        those definitions are rejected because their semantic names are unknown.

        Returns:
            The supplied parameter names mapped to their frozen definition
            values. Omitted fields are not present.

        Raises:
            TypeError: If a safely available live class rejects the supplied
                partial constructor call, or unresolved positional values would
                require class resolution to name.
        """

        if self._cls is None or not isclass(self._cls):
            if self._args:
                raise TypeError(
                    "Semantic parameters for an unresolved definition require "
                    "keyword spelling or SKIP_ARGS; positional names are unknown."
                )
            return self._kwargs
        from .bound_args import bind_partial_arguments
        from .arg_roles import apply_bound_arg_roles

        args = () if self._args is None else tuple(self._args)
        bound_args = bind_partial_arguments(self._cls, args, self._kwargs)
        bound_args = apply_bound_arg_roles(self._cls, bound_args)
        return FrozenDict(
            (name, self._freeze_value(value))
            for name, value in bound_args.items()
        )

    def __getattr__(self, name: str) -> Any:
        """Return a supplied non-reserved semantic parameter by name.

        Args:
            name: Requested Python attribute name.

        Returns:
            The frozen supplied value for ``name``.

        Raises:
            AttributeError: If ``name`` is not a supplied semantic parameter.
        """

        try:
            return self.parameters[name]
        except KeyError as error:
            raise AttributeError(
                f"{type(self).__name__!s} object has no attribute {name!r}"
            ) from error

    # --- mapping interface (for view compatibility) ---
    def __getitem__(self, k: str) -> Any:
        if k == "cls":
            if self._cls is None:
                raise KeyError("cls")
            return self._cls
        if k == "args":
            if self._args is None:
                raise KeyError("args")
            return self._args
        if k == "kwargs":
            return self.kwargs
        raise KeyError(k)

    def __iter__(self) -> Iterator[str]:
        if self._cls is not None:
            yield "cls"
        if self._args is not None:
            yield "args"
        yield "kwargs"

    def __len__(self) -> int:
        return 1 + (1 if self._args is not None else 0) + (1 if self.cls is not None else 0)

    def __eq__(self, rhs):
        if type(self) is not type(rhs):
            return False
        return _structural_value_equal(self.cls, rhs.cls) and _structural_value_equal(self.args, rhs.args) and _structural_value_equal(self.kwargs, rhs.kwargs)

    def __hash__(self) -> int:
        return stable_int_hash(self.stable_hash())

    def stable_hash(self) -> str:
        cached = self._stable_hash_cache
        if cached is not None:
            return cached
        value = stable_hash_function(self)
        object.__setattr__(self, "_stable_hash_cache", value)
        return value

    def __ne__(self, rhs):
        return not self.__eq__(rhs)

    def __repr__(self):
        arg_elements = []
        arg_str = ""
        if self._cls is not None:
            arg_elements.append(f"{self._cls}")
        else:
            arg_elements.append("_")
        if self._args is not None:
            for arg in self._args:
                arg_elements.append(arg)
        for key, v in self.kwargs.items():
            arg_elements.append(f"{key}={v}")
        arg_str = ", ".join(map(str, arg_elements))
        return f"{type(self).__name__}({arg_str})"

    def _pickle_getstate(self):
        # Only structural data; drop ephemeral links
        return {
            'cls': self._cls,
            'args': self._args,
            'kwargs': self._kwargs
        }

    def _pickle_setstate(self, state):
        object.__setattr__(self, "_cls", self._freeze_value(state['cls']))
        object.__setattr__(self, "_args", None if state['args'] is None else FrozenTuple(self._freeze_value(v) for v in state['args']))
        object.__setattr__(self, "_kwargs", self._freeze_kwargs(dict(state['kwargs'])))
        object.__setattr__(self, "_stable_hash_cache", None)

    def __deepcopy__(self, memo):
        """
        Return self because Definition is immutable.
        """
        return self

    def thaw(self, memo: dict|None=None) -> Any:
        """Return this immutable expression without materializing it.

        Args:
            memo: Unused compatibility memo argument.

        Returns:
            This ``Definition`` instance.
        """
        return self

    def freeze(self):
        """Return a quoted Definition snapshot for expression-as-data use."""

        return self.quote()

    def concretize(self, repo: "Repo | None"=None) -> Any:
        """Convert this expression to a fully bound canonical V2 CDef.

        Args:
            repo: Optional repository used to resolve supported definition
                values during canonicalization.

        Returns:
            An exact ``ConcreteDefinition`` whose semantic record includes all
            effective declared defaults.

        Raises:
            TypeError: If required arguments are missing or a bound value cannot
                be canonicalized.

        Side Effects:
            May run the class preparation hook once and resolve values required
            for exact canonicalization.
        """
        return to_canonical(self, repo=repo)

    def with_cls(self, cls) -> "Definition":
        args = (SKIP_ARGS,) if self.args is None else tuple(self.args)
        return Definition(cls, *args, **dict(self.kwargs))

    def with_args(self, *args) -> "Definition":
        if self.cls is None:
            return Definition(*args, **dict(self.kwargs))
        return Definition(self.cls, *args, **dict(self.kwargs))

    def with_arg(self, index, value) -> "Definition":
        if self.args is None:
            raise IndexError("Cannot update positional args on a skip-args Definition.")
        args = list(self.args)
        args[index] = value
        return self.with_args(*args)

    def with_kwargs(self, **kwargs) -> "Definition":
        merged = dict(self.kwargs)
        merged.update(kwargs)
        args = (SKIP_ARGS,) if self.args is None else tuple(self.args)
        if self.cls is None:
            return Definition(*args, **merged)
        return Definition(self.cls, *args, **merged)

    def with_kwarg(self, key, value) -> "Definition":
        return self.with_kwargs(**{key: value})

    def without_kwarg(self, key) -> "Definition":
        kwargs = dict(self.kwargs)
        kwargs.pop(key)
        args = (SKIP_ARGS,) if self.args is None else tuple(self.args)
        if self.cls is None:
            return Definition(*args, **kwargs)
        return Definition(self.cls, *args, **kwargs)

    def at(self, path) -> "DefinitionLens":
        return DefinitionLens(self, path)

    def ref(self):
        from .links import Ref
        return Ref(self)

    def mat(self):
        from .links import Mat
        return Mat(self)

    def quote(self):
        from .quoted import QuotedDef
        return QuotedDef(self)

    def as_selector(self, **policy):
        from .selector import Selector
        return Selector(self, **policy)

    def as_space(self):
        from .search_space import SearchSpace
        return SearchSpace.from_def(self)


# Python 3.10's frozen-slots dataclass transform replaces custom pickle methods.
Definition.__getstate__ = Definition._pickle_getstate
Definition.__setstate__ = Definition._pickle_setstate


@dataclass(frozen=True, slots=True)
class DefinitionLens:
    definition: Definition
    path: Any

    def set(self, value: Any) -> Definition:
        from .canonical import freeze_def_value
        from .utils.graph.value import replace_subtree

        return replace_subtree(self.definition, self.path, freeze_def_value(value))


@dataclass(frozen=True, slots=True, init=False)
class ConcreteDefinition(DefInterface, Mapping):
    """Immutable exact canonical identity for a materializable object.

    CDef V2 identities persist a fully bound immutable
    ``parameters`` name/value record, including declared defaults. Equivalent
    positional and keyword calls therefore have one identity. Pre-V2 authority
    is rejected before it can construct a CDef.

    Semantic inspection through attributes, ``parameters``, ``graph_path()``,
    hashing, and graph/query processing reads the stored record without class
    resolution. Existing API names win attribute collisions; use
    ``parameters[name]`` for every V2 constructor name. ``args`` and ``kwargs``
    are call projections, not identity or structural inspection
    surfaces, and can resolve/import the current class.

    Args:
        cdef_cls: Live class for a new exact constructor call.
        args: Positional constructor values before preparation and binding.
        kwargs: Keyword constructor values before preparation and binding.

    Raises:
        TypeError: If the call cannot be fully bound or canonicalized.
    """

    cls: type | ImportRef | SourceSpec
    _args: FrozenTuple[Any, ...] = field(default_factory=lambda: FrozenTuple())
    _kwargs: FrozenDict[str, Any] = field(default_factory=lambda: FrozenDict({}))
    _stable_hash_cache: str | None = field(default=None, init=False, repr=False, compare=False, hash=False)
    _identity_version: int = field(default=V2_IDENTITY_VERSION, init=False, repr=False, compare=False, hash=False)
    _bound_args: BoundArguments = field(init=False, repr=False, compare=False, hash=False)
    _node_id: object = field(default_factory=new_node_id, init=False, repr=False, compare=False, hash=False)
    _stateful_role: bool = field(default=False, init=False, repr=False, compare=False, hash=False)

    def __init__(self, cdef_cls, args=(), kwargs=None) -> None:
        """Create a validated V2 exact identity from a public call surface.

        Args:
            cdef_cls: Live class for the exact constructor call.
            args: Positional constructor arguments before preparation.
            kwargs: Keyword constructor arguments before preparation.

        Raises:
            TypeError: If the class or call cannot be prepared, fully bound, or
                canonicalized into a V2 identity.

        Direct construction is a V2 factory.
        """

        from .canonical import to_canonical

        if kwargs is None:
            kwargs = {}
        if not isinstance(kwargs, Mapping):
            raise TypeError("ConcreteDefinition kwargs must be a mapping.")
        result = to_canonical(Definition(cdef_cls, *args, **dict(kwargs)))
        self._copy_record(result)

    def _copy_record(self, record: "ConcreteDefinition") -> None:
        """Populate this shell from a private, already-validated identity."""

        object.__setattr__(self, "cls", record.cls)
        object.__setattr__(self, "_args", record._args)
        object.__setattr__(self, "_kwargs", record._kwargs)
        object.__setattr__(self, "_identity_version", record.identity_version)
        object.__setattr__(self, "_bound_args", record._bound_args)
        object.__setattr__(self, "_stable_hash_cache", record._stable_hash_cache)
        object.__setattr__(self, "_node_id", record._node_id)
        object.__setattr__(self, "_stateful_role", record._stateful_role)

    @property
    def args(self) -> FrozenTuple[Any, ...]:
        """Return the compatibility positional call surface.

        The accessor resolves the current class and projects the persisted semantic record;
        this accessor can therefore import a backend or raise a current
        signature error and is not an identity or inspection surface.
        """

        from .materialization import project_cdef_call

        args, _ = project_cdef_call(self)
        return FrozenTuple(args)

    @property
    def kwargs(self) -> FrozenDict[str, Any]:
        """Return the compatibility keyword call surface.

        The accessor uses the current-signature projection as materialization and may
        resolve or import the referenced class.
        """

        from .materialization import project_cdef_call

        _, kwargs = project_cdef_call(self)
        return FrozenDict(kwargs)

    @property
    def identity_version(self) -> int:
        """Return the exact persisted CDef identity format version."""

        return self._identity_version

    @property
    def parameters(self) -> FrozenDict[str, Any]:
        """Return this CDef's immutable persisted semantic record.

        Returns:
            The canonical parameter-name/value mapping persisted with a V2
            CDef. Values are returned without resolving the CDef class.

        """
        return self._bound_args.as_frozen_dict()

    def __getattr__(self, name: str) -> Any:
        """Return a non-reserved V2 semantic parameter by name.

        Args:
            name: Requested Python attribute name.

        Returns:
            The canonical persisted value for ``name``.

        Raises:
            AttributeError: If ``name`` is not an effective V2 parameter.
        """

        try:
            return self._bound_args[name]
        except KeyError:
            pass
        raise AttributeError(f"{type(self).__name__!s} object has no attribute {name!r}")

    @classmethod
    def _from_persisted_record(
            cls,
            cdef_cls,
            *,
            identity_version: int,
            parameters: BoundArguments,
            stateful_role: bool | None = None,
            stable_hash_cache: str | None = None) -> "ConcreteDefinition":
        """Hydrate validated V2 authority without class resolution."""

        validate_identity_version(identity_version)
        result = cls._from_bound_record(cdef_cls, parameters, stateful_role=stateful_role)
        if stable_hash_cache is not None:
            computed_hash = stable_hash_function(
                result,
                reuse_validated_cdef_hashes=True,
            )
            if stable_hash_cache != computed_hash:
                raise ValueError("V2 ConcreteDefinition hash cache does not match its identity record.")
        object.__setattr__(result, "_stable_hash_cache", stable_hash_cache)
        return result

    @classmethod
    def _from_bound_record(
            cls,
            cdef_cls,
            bound_args: BoundArguments,
            *,
            stateful_role: bool | None = None) -> "ConcreteDefinition":
        """Create a validated private V2 identity from an already-bound record.

        Args:
            cdef_cls: Canonical class reference for the exact definition.
            bound_args: Fully canonical semantic constructor record.

        Returns:
            A V2 CDef whose identity is the supplied name/value record.

        Raises:
            TypeError: If the record is not recursively canonical.

        This is intentionally private: callers must use the binding pipeline in
        ``canonical`` rather than manufacture public exact identities.
        """

        bound_args = validate_canonical_bound_arguments(bound_args)
        if stateful_role is None:
            from .object import Serializable

            stateful_role = isinstance(cdef_cls, type) and issubclass(cdef_cls, Serializable)
        if type(stateful_role) is not bool:
            raise TypeError("CDef stateful role must be a bool.")
        result = object.__new__(cls)
        object.__setattr__(result, "cls", cdef_cls)
        object.__setattr__(result, "_args", FrozenTuple())
        object.__setattr__(result, "_kwargs", FrozenDict({}))
        object.__setattr__(result, "_stable_hash_cache", None)
        object.__setattr__(result, "_node_id", new_node_id())
        object.__setattr__(result, "_bound_args", bound_args)
        object.__setattr__(result, "_identity_version", V2_IDENTITY_VERSION)
        object.__setattr__(result, "_stateful_role", stateful_role)
        return result

    def __hash__(self) -> int:
        return stable_int_hash(self.stable_hash())

    def stable_hash(self) -> str:
        cached = self._stable_hash_cache
        if cached is not None:
            return cached
        value = stable_hash_function(self)
        object.__setattr__(self, "_stable_hash_cache", value)
        return value

    def _pickle_getstate(self):
        """Serialize one named V2 identity record."""

        return {
            "identity_version": self.identity_version,
            "cls": self.cls,
            "parameters": self._bound_args.as_frozen_dict(),
            "stateful_role": self._stateful_role,
            "stable_hash_cache": self._stable_hash_cache,
        }

    def _pickle_setstate(self, state):
        """Restore only a V2 record without resolving or binding classes."""

        record = decode_identity_record(state)
        restored = type(self)._from_persisted_record(
            record.cls,
            identity_version=record.version,
            parameters=record.parameters,
            stateful_role=record.stateful_role,
            stable_hash_cache=record.stable_hash_cache,
        )
        self._copy_record(restored)

    def __getitem__(self, k: str) -> Any:
        """Return an immutable V2 record field by name.

        Args:
            k: ``"cls"`` or ``"parameters"``.

        Returns:
            The stored canonical record field.

        Raises:
            KeyError: If ``k`` is not a V2 record field.
        """
        if k == "cls": return self.cls
        if k == "parameters": return self.parameters
        raise KeyError(k)

    def __iter__(self) -> Iterator[str]:
        yield from ("cls", "parameters")

    def __len__(self) -> int:
        return 2

    def __repr__(self):
        from .cdef_codec import render_cdef_repr

        return render_cdef_repr(self)

    def __eq__(self, rhs):
        if type(self) is not type(rhs):
            return False
        left_parameters = self.parameters
        right_parameters = rhs.parameters
        return (
            _structural_value_equal(self.cls, rhs.cls)
            and len(left_parameters) == len(right_parameters)
            and all(
                key in right_parameters
                and _structural_value_equal(value, right_parameters[key])
                for key, value in left_parameters.items()
            )
        )

    def __ne__(self, rhs):
        return not self.__eq__(rhs)


    def __deepcopy__(self, memo):
        """
        Return this immutable identity record without rebinding its call surface.
        """
        return self

    def __copy__(self):
        """Return this immutable identity record without rebinding it."""

        return self

    def copy(self):
        return deepcopy(self)

    def copy_graph(self) -> "ConcreteDefinition":
        """Return an independently keyed copy of this complete CDef graph.

        Returns:
            A graph-isomorphic CDef root with fresh private node tokens.

        Side Effects:
            Does not resolve symbols, materialize objects, or mutate this
            immutable CDef graph.
        """

        from .cdef_codec import copy_cdef_graph

        return copy_cdef_graph(self)

    def graph_equal(self, other: object) -> bool:
        """Compare rooted CDef topology while ignoring private node tokens.

        Args:
            other: Candidate concrete-definition root.

        Returns:
            ``True`` only when classes, structural values, typed edges, edge
            kinds, and shared-versus-independent node topology correspond.
        """

        from .cdef_codec import cdef_graph_equal

        return cdef_graph_equal(self, other)

    def graph_hash(self) -> str:
        """Return a deterministic digest of this CDef's rooted graph topology.

        Returns:
            A token-free graph digest that distinguishes sharing from
            independent structurally equal nodes.
        """

        from .cdef_codec import cdef_graph_hash

        return cdef_graph_hash(self)

    def freeze(self):
        """Return a non-materializing canonical reference to this CDef.

        Returns:
            A ``Ref`` that preserves this exact identity without materializing
            the referenced object.
        """

        return self.ref()

    def ref(self):
        from .links import Ref
        return Ref(self)

    def mat(self):
        from .links import Mat
        return Mat(self)

    def thaw(self, memo: dict | None = None) -> Any:
        """Return a Definition surface for this exact identity.

        Args:
            memo: Optional memo used while thawing a definition graph.

        Returns:
            A partial ``Definition`` or thawed canonical value suitable for
            editing or later concretization.

        Raises:
            TypeError: If the current class signature cannot accept a persisted
                V2 semantic record.

        Side Effects:
            V2 call projection resolves the current class and may import its
            backend. It does not execute the constructor, preparation hook, or
            declared defaults.
        """
        return thaw_value(self, memo=memo)

    def concretize(self, repo: "Repo | None"=None) -> Any:
        """Return this already exact identity unchanged.

        Args:
            repo: Unused compatibility repository argument.

        Returns:
            This ``ConcreteDefinition`` instance.
        """
        return self

    def graph_path(self, path: Any = "$") -> Any:
        """Resolve a V2 semantic graph path without materializing this CDef.

        Args:
            path: A typed ``GraphPath`` or textual graph path using semantic
                ``Parameter`` segments.

        Returns:
            The canonical value addressed by ``path``.

        Raises:
            QueryPathError: If the path is malformed or cannot be resolved
            under V2 semantic path rules.
        """

        from .utils.graph.value import get_subtree

        return get_subtree(self, path)

    def at(self, path: Any) -> DefinitionLens:
        """Return a copy-on-write lens rooted at this exact CDef graph.

        Args:
            path: A typed or textual path into this CDef's semantic values.

        Returns:
            A lens whose ``set`` operation reconstructs only ancestors on the
            selected path and preserves untouched private descendant nodes.
        """

        return DefinitionLens(self, path)


# Preserve the versioned record codec on Python 3.10; see the Definition note.
ConcreteDefinition.__getstate__ = ConcreteDefinition._pickle_getstate
ConcreteDefinition.__setstate__ = ConcreteDefinition._pickle_setstate


def freeze(value: Any) -> Any:
    """Return a non-materializing immutable graph wrapper for a definition value.

    Args:
        value: An ``Object``, ``Definition``, or ``ConcreteDefinition``.

    Returns:
        A quoted definition for partial expressions or a ``Ref`` for exact
        identities and objects.

    Raises:
        TypeError: If ``value`` is not a supported DRYML definition value.
    """

    from .object import Object

    if isinstance(value, Object):
        return value.definition.freeze()
    if isinstance(value, ConcreteDefinition):
        return value.freeze()
    if isinstance(value, Definition):
        return value.freeze()
    raise TypeError(
        "dryml.freeze expects Object, ConcreteDefinition, or Definition; "
        f"got {type(value).__name__}."
    )


def _structural_value_equal(left: Any, right: Any) -> bool:
    from .freeze import FrozenDict, FrozenList, FrozenNDArray, FrozenSet, FrozenTuple
    from .links import DefLink
    from .quoted import QuotedDef, SelectorSpec
    from .selector import Selector

    if left is right:
        return True
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        if not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray):
            return False
        return left.shape == right.shape and left.dtype == right.dtype and bool(np.array_equal(left, right))
    if type(left) is not type(right):
        return False
    if isinstance(left, (Definition, ConcreteDefinition)):
        return left == right
    if isinstance(left, DefLink):
        return left.kind is right.kind and _structural_value_equal(left.target, right.target)
    if isinstance(left, QuotedDef):
        return _structural_value_equal(left.value, right.value)
    if isinstance(left, SelectorSpec):
        return _structural_value_equal(left.selector, right.selector)
    if isinstance(left, Selector):
        return left.strict == right.strict and left.cls_policy == right.cls_policy and _structural_value_equal(left.root, right.root)
    if isinstance(left, (FrozenList, FrozenTuple, tuple, list)):
        return len(left) == len(right) and all(_structural_value_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, (FrozenDict, dict)):
        if len(left) != len(right) or tuple(left.keys()) != tuple(right.keys()):
            return False
        return all(_structural_value_equal(left[k], right[k]) for k in left.keys())
    if isinstance(left, (FrozenSet, frozenset, set)):
        if len(left) != len(right):
            return False
        unmatched = list(right)
        for item in left:
            for idx, other in enumerate(unmatched):
                if _structural_value_equal(item, other):
                    unmatched.pop(idx)
                    break
            else:
                return False
        return True
    result = left == right
    if isinstance(result, np.ndarray):
        return bool(np.all(result))
    return bool(result)


def get_path(obj_or_def, path):
    from .object import Object
    if len(path) == 0:
        return obj_or_def

    key = path[0]
    if key is None:
        return obj_or_def

    new_path = path[1:]
    if isinstance(obj_or_def, Object):
        if key == 'cls':
            value = obj_or_def.definition.cls
        elif key == 'args':
            value = obj_or_def.definition.args
        elif key == 'kwargs':
            value = obj_or_def.definition.kwargs
        else:
            raise PathAccessError(path)
    else:
        try:
            value = obj_or_def[key]
        except (KeyError, IndexError, ValueError, TypeError, PathAccessError) as e:
            if type(e) is PathAccessError:
                raise PathAccessError(path)
            else:
                raise PathAccessError(path)


    try:
        return get_path(value, new_path)
    except (KeyError, IndexError, ValueError, PathAccessError) as e:
        if type(e) is PathAccessError:
            raise PathAccessError(path)
        else:
            raise PathAccessError(path)


def render_path(path, key):
    path = ["root",] + path
    if key is not None:
        path = path + [key,]

    return "/".join(map(str, path))


# ----------------------------------------------------------------------
# Categorical definition
# ----------------------------------------------------------------------

class CategoricalDefinitionTransformer(GraphTransformer):
    """
    Strip unique args recursively (or only at the root when recursive=False).
    """

    def __init__(self, recursive: bool = True):
        super().__init__()
        self.recursive = recursive

    def transform(self, obj: Any, ctx: GraphCtx | None = None) -> Any:
        if ctx is not None and (not self.recursive) and ctx.path:
            # Match the intended current behavior: only operate at the root.
            return obj
        return super().transform(obj, ctx)

    def is_atomic(self, obj: Any, ctx: GraphCtx) -> bool:
        return is_pod(obj) or isinstance(obj, type)

    def memo_key(self, obj: Any, ctx: GraphCtx):
        if isinstance(obj, Definition):
            return id(obj)
        return None

    def transform_atomic(self, obj: Any, ctx: GraphCtx) -> Any:
        return obj

    def dispatch(self, obj: Any, ctx: GraphCtx) -> Any:
        from .freeze import FrozenDict, FrozenList, FrozenSet, FrozenTuple

        if isinstance(obj, FrozenDict):
            return {k: self.transform(v, ctx.child(k if isinstance(k, (str, int)) else str(k))) for k, v in obj.items()}
        if isinstance(obj, FrozenList):
            return [self.transform(v, ctx.child(i)) for i, v in enumerate(obj)]
        if isinstance(obj, FrozenTuple):
            return tuple(self.transform(v, ctx.child(i)) for i, v in enumerate(obj))
        if isinstance(obj, FrozenSet):
            return {self.transform(v, ctx.child(f"<set:{i}>")) for i, v in enumerate(obj)}
        return super().dispatch(obj, ctx)

    def should_track_cycle(self, obj: Any, ctx: GraphCtx) -> bool:
        from .definition import Definition

        return super().should_track_cycle(obj, ctx) or isinstance(obj, Definition)

    def transform_other(self, obj: Any, ctx: GraphCtx) -> Any:
        from .definition import Definition

        if isinstance(obj, Definition):
            if obj.cls is None:
                raise ValueError(
                    f"Cannot categorical-ify Definition with missing cls at {ctx.path_str()}"
                )

            defn_args = (
                self.transform(obj.args, ctx.child("args"))
                if obj.args is not None
                else None
            )
            defn_kwargs = self.transform(obj.kwargs, ctx.child("kwargs"))

            temp_args = defn_args if obj.args is not None else tuple()
            live_cls = resolve_symbol(obj.cls)
            new_args, new_kwargs = live_cls.__strip_unique_args__(
                *temp_args,
                **defn_kwargs,
            )

            new_defn_args = [live_cls]
            if obj.args is not None:
                new_defn_args.extend(new_args)
            else:
                from .definition import SKIP_ARGS
                new_defn_args.append(SKIP_ARGS)

            return Definition(*new_defn_args, **new_kwargs)

        raise TypeError(
            f"Cannot categorical-ify object of type {type(obj).__name__} at {ctx.path_str()}"
        )


def categorical_definition(defn, recursive=True, memo=None):
    from .definition import ConcreteDefinition
    from .object import Object

    if memo is None:
        memo = {}

    if isinstance(defn, Object):
        root = thaw_concrete(defn.definition, memo=memo)
    elif isinstance(defn, ConcreteDefinition):
        root = thaw_concrete(defn, memo=memo)
    else:
        root = deepcopy(defn)

    return CategoricalDefinitionTransformer(recursive=recursive).transform(root)


class SelectorMatcher(GraphMatcher):
    def __init__(
        self,
        *,
        strict: bool = True,
        cls_str_compare: bool = False,
        verbose: bool = False,
        full_diagnostic: bool = False,
        output_stream=sys.stderr,
    ):
        self.strict = strict
        self.cls_str_compare = cls_str_compare
        self.verbose = verbose
        self.full_diagnostic = full_diagnostic
        self.output_stream = output_stream

    # ------------------------------------------------------------------
    # small helpers
    # ------------------------------------------------------------------

    def _render(self, ctx: GraphCtx) -> str:
        if hasattr(ctx.path, "legacy_tuple"):
            return render_path(list(ctx.path.legacy_tuple()), None)
        return render_path(list(ctx.path), None)

    def _print(self, ctx: GraphCtx, msg: str) -> None:
        if self.verbose:
            self.output_stream.write(f"[{self._render(ctx)}]: {msg}\n")

    def _normalize_dryml(self, x: Any):
        from .object import Object
        if isinstance(x, Object):
            return x.definition
        return x

    def _is_structural(self, x: Any) -> bool:
        return node_kind(x) not in {
            NodeKind.POD,
            NodeKind.TYPE,
            NodeKind.IDENTITY_VALUE,
            NodeKind.REFERENCE_VALUE,
            NodeKind.IMPORT_REF,
            NodeKind.SOURCE_SPEC,
            NodeKind.UNSUPPORTED,
        }

    # ------------------------------------------------------------------
    # graph matcher hooks
    # ------------------------------------------------------------------

    def memo_key(self, selector: Any, target: Any, ctx: GraphCtx):
        if self._is_structural(selector) or self._is_structural(target):
            return (id(selector), id(target))
        return None

    def should_track_cycle(self, selector: Any, target: Any, ctx: GraphCtx) -> bool:
        return self._is_structural(selector) or self._is_structural(target)

    # ------------------------------------------------------------------
    # dispatch
    # ------------------------------------------------------------------

    def dispatch(self, selector: Any, target: Any, ctx: GraphCtx) -> bool:
        from .object import Object

        dryml_obj_types = (Object, Definition, ConcreteDefinition)

        if isinstance(target, Definition):
            if self.strict and target.skip_args:
                raise TypeError(
                    f"Definitions which skip args aren't allowed in strict mode {self._render(ctx)}"
                )

        # class comparisons
        if isclass(selector) and isclass(target):
            if self.strict:
                condition = selector is target
                if not condition:
                    self._print(ctx, "Classes differ")
                return condition
            else:
                condition = issubclass(target, selector)
                if not condition:
                    self._print(
                        ctx,
                        f"Classes not subclass: {get_class_str(target)} "
                        f"is not a subclass of {get_class_str(selector)}",
                    )
                return condition

        selector_ref = maybe_symbol_ref(selector, functions=False)
        target_ref = maybe_symbol_ref(target, functions=False)
        if selector_ref is not None and target_ref is not None:
            if not self.strict:
                try:
                    selector_obj = resolve_symbol(selector_ref)
                    target_obj = resolve_symbol(target_ref)
                except Exception:
                    selector_obj = None
                    target_obj = None
                if isclass(selector_obj) and isclass(target_obj):
                    condition = issubclass(target_obj, selector_obj)
                    if not condition:
                        self._print(
                            ctx,
                            f"Classes not subclass: {get_class_str(target_obj)} "
                            f"is not a subclass of {get_class_str(selector_obj)}",
                        )
                    return condition

            condition = selector_ref == target_ref
            if not condition:
                self._print(ctx, "Symbol refs differ")
            return condition

        if self.cls_str_compare and isinstance(selector, str) and isclass(target):
            condition = selector == get_class_str(target)
            if not condition:
                self._print(
                    ctx,
                    f"Class string comparison failed: {selector} != {get_class_str(target)}",
                )
            return condition

        # centralized container compatibility
        family = matching_container_family(selector, target)
        if family is not None:
            if family == "tuple":
                return self.match_tuple(selector, target, ctx)
            if family == "list":
                return self.match_list(selector, target, ctx)
            if family == "set":
                return self.match_set(selector, target, ctx)
            if family == "dict":
                return self.match_dict(selector, target, ctx)

            raise TypeError(f"Unhandled container family {family!r} in selector_match")

        # ndarray
        if isinstance(selector, np.ndarray) and isinstance(target, np.ndarray):
            if selector.shape != target.shape:
                self._print(
                    ctx,
                    f"Mismatched array shapes {selector.shape} != {target.shape}",
                )
                return False
            condition = np.all(target == selector)
            if not condition:
                self._print(ctx, "Unequal arrays")
            return condition

        # dryml object / definition
        if isinstance(selector, dryml_obj_types) and isinstance(target, dryml_obj_types):
            return self.match_dryml(selector, target, ctx)

        # callable selector
        if is_nonclass_callable(selector):
            if self.strict:
                raise TypeError(
                    f"Callable selectors are not allowed in strict mode {self._render(ctx)}"
                )
            condition = selector(target)
            if not condition:
                self._print(ctx, "Callable test failed")
            return condition

        return self.match_other(selector, target, ctx)

    # ------------------------------------------------------------------
    # container overrides with diagnostics / selector semantics
    # ------------------------------------------------------------------

    def match_tuple(self, selector, target, ctx: GraphCtx) -> bool:
        if len(selector) != len(target):
            self._print(
                ctx,
                f"Container lengths don't match. {len(selector)} in selector, "
                f"{len(target)} in target",
            )
            return False

        compare_failed = False
        for i, (sel_v, tgt_v) in enumerate(zip(selector, target)):
            res = self.match(sel_v, tgt_v, ctx.child(i))
            if not res:
                compare_failed = True
                if not self.full_diagnostic:
                    return False
        return not compare_failed

    def match_list(self, selector, target, ctx: GraphCtx) -> bool:
        if len(selector) != len(target):
            self._print(
                ctx,
                f"Container lengths don't match. {len(selector)} in selector, "
                f"{len(target)} in target",
            )
            return False

        compare_failed = False
        for i, (sel_v, tgt_v) in enumerate(zip(selector, target)):
            res = self.match(sel_v, tgt_v, ctx.child(i))
            if not res:
                compare_failed = True
                if not self.full_diagnostic:
                    return False
        return not compare_failed

    def match_set(self, selector, target, ctx: GraphCtx) -> bool:
        if len(selector) != len(target):
            self._print(
                ctx,
                f"Set lengths don't match. {len(selector)} in selector, "
                f"{len(target)} in target",
            )
            return False

        selector_items = list(selector)
        target_items = list(target)
        edges = [
            [j for j, tgt_v in enumerate(target_items) if self.match(sel_v, tgt_v, ctx.child(i))]
            for i, sel_v in enumerate(selector_items)
        ]
        order = sorted(range(len(selector_items)), key=lambda idx: len(edges[idx]))
        matched_to_selector: dict[int, int] = {}

        def augment(sel_idx: int, seen: set[int]) -> bool:
            for tgt_idx in edges[sel_idx]:
                if tgt_idx in seen:
                    continue
                seen.add(tgt_idx)
                if tgt_idx not in matched_to_selector or augment(matched_to_selector[tgt_idx], seen):
                    matched_to_selector[tgt_idx] = sel_idx
                    return True
            return False

        for sel_idx in order:
            if not augment(sel_idx, set()):
                self._print(ctx.child(sel_idx), "No matching set element found")
                return False

        return True

    def match_dict(self, selector, target, ctx: GraphCtx) -> bool:
        # Selector semantics: only keys mentioned in selector must match.
        compare_failed = False

        for k in selector.keys():
            child_ctx = ctx.child(k if isinstance(k, (str, int)) else str(k))
            if k not in target:
                compare_failed = True
                self._print(child_ctx, "Key missing in target")
                if not self.full_diagnostic:
                    return False
                continue

            res = self.match(selector[k], target[k], child_ctx)
            if not res:
                compare_failed = True
                if not self.full_diagnostic:
                    return False

        return not compare_failed

    # ------------------------------------------------------------------
    # dryml-specific matching
    # ------------------------------------------------------------------

    def match_dryml(self, selector, target, ctx: GraphCtx) -> bool:
        sel_def = self._normalize_dryml(selector)
        tgt_def = self._normalize_dryml(target)
        if isinstance(sel_def, Definition):
            from .arg_roles import apply_definition_arg_roles

            sel_def = apply_definition_arg_roles(sel_def)

        compare_failed = False

        if sel_def.cls is not None:
            if not self.match(sel_def.cls, tgt_def.cls, ctx.child("cls")):
                compare_failed = True
                if not self.full_diagnostic:
                    return False

        if isinstance(sel_def, Definition) and isinstance(tgt_def, ConcreteDefinition) and tgt_def._bound_args is not None:
            # V2 records have no raw invocation fields.  Bind only the values
            # supplied by the soft selector; defaults remain unconstrained.
            for name, child in sel_def.parameters.items():
                child_ctx = ctx.child(f"parameters[{name!r}]")
                if name not in tgt_def.parameters:
                    compare_failed = True
                    self._print(child_ctx, "Semantic parameter missing in target")
                    if not self.full_diagnostic:
                        return False
                    continue
                if not self.match(child, tgt_def.parameters[name], child_ctx):
                    compare_failed = True
                    if not self.full_diagnostic:
                        return False
            return not compare_failed

        if sel_def.args is not None:
            if not self.match(sel_def.args, tgt_def.args, ctx.child("args")):
                compare_failed = True
                if not self.full_diagnostic:
                    return False

        if not self.match(sel_def.kwargs, tgt_def.kwargs, ctx.child("kwargs")):
            compare_failed = True
            if not self.full_diagnostic:
                return False

        return not compare_failed

    # ------------------------------------------------------------------
    # plain values
    # ------------------------------------------------------------------

    def match_other(self, selector: Any, target: Any, ctx: GraphCtx) -> bool:
        if type(selector) is not type(target):
            self._print(ctx, "Type mismatch")
            return False

        condition = selector == target
        if not condition:
            self._print(ctx, "Values differ")
        return condition


def selector_match(
    selector,
    target,
    strict=True,
    cls_str_compare=False,
    verbose=False,
    full_diagnostic=False,
    output_stream=sys.stderr,
):
    return SelectorMatcher(
        strict=strict,
        cls_str_compare=cls_str_compare,
        verbose=verbose,
        full_diagnostic=full_diagnostic,
        output_stream=output_stream,
    ).match(selector, target)


def concretize_func(
    obj: Any,
    path: list[str | int] | None = None,
    repo: "Repo | None" = None,
) -> Any:
    return to_canonical(obj, repo=repo, path=path)


def thaw_concrete(cdef_or_obj: Any, memo: dict|None =None) -> Any:
    return thaw_value(cdef_or_obj, memo=memo)
