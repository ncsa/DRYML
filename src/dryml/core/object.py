from __future__ import annotations

from typing import TYPE_CHECKING
from threading import Lock
from abc import ABCMeta, update_abstractmethods
import inspect
import types

import os
from contextlib import contextmanager

from .utils.general import pickle_save, pickle_load
from .definition import Definition

if TYPE_CHECKING:
    from .repo import RevisionType


def in_definition_mode() -> bool:
    from .session import current_object_mode

    return current_object_mode() in {"definition", "concrete", "selector", "space"}


def definition_mode_concrete() -> bool:
    from .session import current_object_mode

    return current_object_mode() == "concrete"


@contextmanager
def selector_mode(enabled: bool = True):
    from .session import config

    with config(object_mode="selector" if enabled else "fresh"):
        yield


@contextmanager
def space_mode(enabled: bool = True):
    from .session import config

    with config(object_mode="space" if enabled else "fresh"):
        yield


@contextmanager
def definition_mode(enabled: bool = True, *, concrete: bool = False):
    from .session import config

    if not enabled:
        with config(object_mode="fresh"):
            yield
        return

    object_mode = "concrete" if concrete else "definition"
    with config(object_mode=object_mode):
        yield


_CLASS_TRANSFORMERS = "__dryml_class_transformers__"
_CLASS_VALIDATORS = "__dryml_class_validators__"


class _AbstractObjectAdmissionError(TypeError):
    """Raised when a live DRYML construction selects an abstract class."""


def _register_class_transformer(owner: type, callback) -> None:
    """Register one owner-local class-finalization transformer.

    The private protocol lets higher-level declaration owners transform their
    descriptors before standard ABC finalization without core importing them.
    """

    callbacks = owner.__dict__.get(_CLASS_TRANSFORMERS, ())
    if not isinstance(callbacks, tuple):
        raise TypeError("DRYML class transformer registration must be a tuple.")
    type.__setattr__(owner, _CLASS_TRANSFORMERS, (*callbacks, callback))


def _register_class_validator(owner: type, callback) -> None:
    """Register one owner-local non-mutating class-finalization validator."""

    callbacks = owner.__dict__.get(_CLASS_VALIDATORS, ())
    if not isinstance(callbacks, tuple):
        raise TypeError("DRYML class validator registration must be a tuple.")
    type.__setattr__(owner, _CLASS_VALIDATORS, (*callbacks, callback))


def _collect_class_finalizers(cls: type, slot: str) -> tuple:
    """Return class-local callbacks in base-to-derived C3 order by identity."""

    callbacks = []
    seen = set()
    for owner in reversed(cls.__mro__):
        registered = owner.__dict__.get(slot, ())
        if not isinstance(registered, tuple):
            raise TypeError("DRYML class finalizer registration must be a tuple.")
        for callback in registered:
            if id(callback) not in seen:
                seen.add(id(callback))
                callbacks.append(callback)
    return tuple(callbacks)


def _admit_materialization_class(cls: type) -> None:
    """Reject an abstract class before any DRYML runtime construction effect."""

    if inspect.isabstract(cls):
        members = ", ".join(sorted(cls.__abstractmethods__))
        raise _AbstractObjectAdmissionError(
            f"Cannot materialize abstract Object {cls.__qualname__}; "
            f"missing abstract members: {members}."
        )


class Dryml(ABCMeta):
    """Capture Object construction calls and apply the active repository mode."""

    def __new__(mcls, name, bases, namespace, **kwargs):
        """Finalize statically proven declaration descriptors after class creation.

        The hook is owner-neutral: core only follows a native wrapper's direct
        ``__wrapped__`` chain and invokes a private protocol supplied by the
        declaration owner.  It never imports or interprets managed policy.
        """

        cls = super().__new__(mcls, name, bases, namespace, **kwargs)
        for member_name, member in tuple(cls.__dict__.items()):
            outer = member
            candidate = member
            seen: set[int] = set()
            while type(candidate) is types.FunctionType and id(candidate) not in seen:
                seen.add(id(candidate))
                candidate = candidate.__dict__.get("__wrapped__")
                if candidate is None:
                    break
                hook = type(candidate).__dict__.get(
                    "__dryml_finalize_hidden_member__"
                )
                if hook is not None:
                    replacement = hook(candidate, cls, member_name, outer)
                    if replacement is None:
                        raise TypeError("declaration finalization returned no descriptor")
                    type.__setattr__(cls, member_name, replacement)
                    break
        for callback in _collect_class_finalizers(cls, _CLASS_TRANSFORMERS):
            callback(cls)
        update_abstractmethods(cls)
        for callback in _collect_class_finalizers(cls, _CLASS_VALIDATORS):
            callback(cls)
        return cls

    def __call__(dryml_cls, /, *args, repo=None, __cdef__=None, **kwargs):
        """Create, describe, concretize, or load an Object construction call.

        Args:
            *args: Runtime constructor positional arguments.
            repo: Optional repository for canonicalization and materialization.
            __cdef__: Internal exact identity used during reconstruction.
            **kwargs: Runtime constructor keyword arguments. ``cls`` remains a
                normal user keyword rather than colliding with this metaclass.

        Returns:
            An Object, Definition, ConcreteDefinition, Selector, or SearchSpace
            according to the active object mode.

        Raises:
            TypeError: If constructor binding or runtime initialization fails.

        Side Effects:
            May canonicalize arguments, access Store state, populate repository
            caches, allocate a workspace, and initialize a runtime object.
        """

        from .session import _construction_object_mode, get_config

        session_config = get_config()
        object_mode = _construction_object_mode()
        active_repo = repo if repo is not None else session_config.repo

        if __cdef__ is None and object_mode == "definition":
            defn = dryml_cls.defn(*args, **kwargs)
            return defn

        if __cdef__ is None and object_mode == "concrete":
            return dryml_cls.defn(*args, **kwargs).concretize(repo=active_repo)

        if __cdef__ is None and object_mode == "selector":
            return dryml_cls.defn(*args, **kwargs).as_selector()

        if __cdef__ is None and object_mode == "space":
            return dryml_cls.defn(*args, **kwargs).as_space()

        _admit_materialization_class(dryml_cls)

        from dryml.runtime import materialization_admission
        from .session import _construction_config
        from .repo_plan import realization_scope

        with materialization_admission(operation="direct_object_construction"):
            if __cdef__ is None and object_mode == "load_or_build":
                from .repo import manage_repo

                with manage_repo(repo=active_repo) as sub_repo:
                    _cache_runtime_object_args(sub_repo, args, kwargs)
                    cdef = Definition(dryml_cls, *args, **kwargs).concretize(repo=sub_repo)
                    return sub_repo.load_or_build(cdef, cache=session_config.cache)

            from .repo import default_repo, manage_repo
            with realization_scope(), _construction_config(), manage_repo(repo=active_repo) as sub_repo:
                if __cdef__ is None:
                    # First-time construction from a soft Definition
                    _cache_runtime_object_args(sub_repo, args, kwargs)
                    defn = Definition(dryml_cls, *args, **kwargs)
                    cdef = defn.concretize(repo=sub_repo)

                    from .materialization import project_cdef_call

                    canonical_args, canonical_kwargs = project_cdef_call(cdef, cls=dryml_cls)
                    rt_args = sub_repo._load_structural(canonical_args)
                    rt_kwargs = sub_repo._load_structural(canonical_kwargs)

                else:
                    # Reconstruction from an existing ConcreteDefinition
                    cdef = __cdef__
                    rt_args = args
                    rt_kwargs = kwargs

                # Run pre-init check
                dryml_cls.__pre_init__()

                # Resolve host/runtime-specific config leaves after identity has been
                # computed, but before the user initializer receives its arguments.
                rt_args = sub_repo.resolve_config(rt_args)
                rt_kwargs = sub_repo.resolve_config(rt_kwargs)

                # Actual object allocation
                obj = dryml_cls.__new__(dryml_cls)

                # This reservation is framework runtime state, not payload.
                # Save and exact restore use it to prevent concurrent mutation
                # of one live object during a state transition.
                obj._save_load_reservation = Lock()
                # A completed StateRef is runtime metadata, not saved payload.
                obj._last_state_ref = None
                # A failed in-place restore has no rollback boundary. The object
                # must be replaced by a fresh exact load before further state IO.
                obj._restore_failed = False

                # Attach the definition to the object.
                obj.__cdef__ = cdef

                # Set the workspace
                if isinstance(obj, WorkspaceCapable):
                    from .cdef_identity import cdef_node_key
                    from .repo_plan import current_realization_scope

                    scope = current_realization_scope()
                    ws = sub_repo.workspace_manager.alloc(
                        cdef.stable_hash(),
                        scope=scope,
                        node_key=cdef_node_key(cdef),
                    )
                    os.makedirs(ws.path(), exist_ok=True)
                    obj.__ws__ = ws
                else:
                    obj.__ws__ = None

                # Initialize with runtime (built) args while exposing the construction
                # repo to code that consults get_default_repo().
                with default_repo(sub_repo):
                    obj.__init__(*rt_args, **rt_kwargs)

                from .repo_plan import _NodeBindings, attach_runtime_binding

                memo = _NodeBindings()
                _collect_runtime_objects(rt_args, memo)
                _collect_runtime_objects(rt_kwargs, memo)
                memo[cdef] = obj
                bound = inspect.signature(dryml_cls.__init__).bind(obj, *rt_args, **rt_kwargs)
                parameters = dict(bound.arguments)
                parameters.pop("self", None)
                attach_runtime_binding(sub_repo, cdef, obj, memo, parameters)


        return obj


def _cache_runtime_object_args(repo, args, kwargs) -> None:
    """Seed repo weak cache for runtime Object values before Definition snapshotting."""

    seen: set[int] = set()

    def visit(value):
        oid = id(value)
        if oid in seen:
            return
        seen.add(oid)
        if isinstance(value, Object):
            repo.cache_weak(value)
            return
        if isinstance(value, dict):
            for child in value.values():
                visit(child)
            return
        if isinstance(value, (list, tuple, set, frozenset)):
            for child in value:
                visit(child)

    for arg in args:
        visit(arg)
    for value in kwargs.values():
        visit(value)


def _collect_runtime_objects(value, memo) -> None:
    """Seed an exact private-node memo from caller-supplied runtime Objects."""

    if isinstance(value, Object):
        if value.definition in memo:
            return
        memo[value.definition] = value
        for bound in getattr(value, "_runtime_bindings", {}).values():
            if isinstance(bound, Object):
                _collect_runtime_objects(bound, memo)
        return
    if isinstance(value, dict):
        for item in value.values():
            _collect_runtime_objects(item, memo)
        return
    if isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            _collect_runtime_objects(item, memo)


class Object(metaclass=Dryml):
    """Base type for DRYML definition-controlled runtime Objects.

    Subclasses may declare normal Python ABC obligations. Abstract subclasses can
    still produce inert definitions and selectors, but every live construction or
    materialization rejects them before framework allocation or user hooks.
    """

    __ws__: WorkspaceHandle | None
    __cdef__: ConcreteDefinition

    @classmethod
    def defn(cls, *args, **kwargs) -> "Definition":
        from .definition import Definition
        return Definition(cls, *args, **kwargs)

    # Alias for defn
    d = defn

    @classmethod
    def __pre_init__(cls):
        pass

    def __init__(self):
        # Optional sanity assertions (can be turned off later)
        assert hasattr(self, "__cdef__"), "__cdef__ must be set by Dryml.__call__ before __init__"
        assert hasattr(self, "__ws__"), "__ws__ must be set by Dryml.__call__ before __init__"

    @property
    def workspace(self) -> str:
        if self.__ws__ is None:
            raise RuntimeError("This object has no workspace")
        return self.__ws__.path()

    @property
    def definition(self) -> "ConcreteDefinition":
        # Get a `Definition` object for this particular object.
        return self.__cdef__

    @property
    def object_id(self):
        """Return this Serializable receiver's durable ObjectId, if any."""

        return getattr(self, "_object_id", None)

    @property
    def object_ref(self):
        """Return the completed immutable exact identity for this live graph."""

        return self._object_ref

    @property
    def last_state_ref(self) -> "StateRef | None":
        """Return this object's last fully published top-level StateRef receipt.

        Returns:
            The immutable exact StateRef most recently published for this object
            as a top-level graph root or requested by an exact top-level load, or
            ``None`` when no such receipt has completed.

        Raises:
            None. This read-only property never performs Store access.

        Side Effects:
            None. The receipt is runtime metadata and does not establish that the
            current live state still matches the returned snapshot.
        """

        return self._last_state_ref

    def graph_at(self, path="$"):
        """Return retained realization evidence at a typed graph path.

        Args:
            path: A ``GraphPath``-compatible path rooted at this Object.

        Returns:
            The receiver, exact bound Object/reference, or a defensive
            runtime-form non-Object value.

        Raises:
            GraphPathError: If the path was not present in the completed
                realization evidence.
        """

        from .utils.graph.path import normalize_path, GraphPathError
        from .repo_plan import _copy_runtime_value

        normalized = normalize_path(path)
        try:
            value = self._runtime_projection[normalized]
        except KeyError as error:
            raise GraphPathError(
                f"No completed runtime binding at {normalized!s}."
            ) from error
        return value if isinstance(value, Object) else _copy_runtime_value(value)

    def __hash__(self):
        # Objects are hashable through through its `ConcreteDefinition`
        return hash(self.definition)

    def __repr__(self):
        return f"<{self.definition.cls} at {hex(id(self))}>(args={self.definition.args}, kwargs={self.definition.kwargs})"

    def save(
            self,
            repo=None,
            main=True,
            *,
            store=None,
            alias: str | None = None,
            deep_capture: bool = False,
            match_mode: str | None = None,
            graph_mode: str | None = None,
            report_stores: bool = False,
            source_store=None, source_stores=None, annotations=None):
        """Publish this graph as an immutable exact StateRef.

        Args:
            repo: Repository or Store authority used for publication.
            main: Whether to update the target Store's structural main reference
                after StateRef publication succeeds.
            store: Optional explicit whole-graph closure Store. It bypasses
                routing, placement, and replication for this save.
            alias: Optional object alias to update after StateRef publication.
            deep_capture: Whether to serialize every owned Serializable node.
            match_mode: Optional ``"first"`` or ``"all"`` routing override.
            graph_mode: Optional ``"per-object"`` or ``"closure"`` placement
                override local to this save.
            report_stores: Whether to also return the selected Store report.
            source_store: Optional connected Store selecting existing complete root
                capture evidence independently of destination routing.
            source_stores: Optional mapping from exact root or embedded StateRefs
                to connected complete source Stores.
            annotations: Optional SaveAnnotations whole-map replacements for the
                root current ObjectRef and/or StateRef mappings.

        Returns:
            The published ``StateRef``, or it paired with a ``StoreReport``.

        Raises:
            RepoSaveError: If routing, immutable publication, or a later
                derived update fails. Its report records completed, failed, and
                unattempted Store work; completed Store work is not rolled back
                because cross-Store saves are not transactional.
            TypeError: If a supplied mode has the wrong type.
            ValueError: If a supplied mode is unsupported.
            MetadataConflictError: If completed source or destination snapshot
                evidence conflicts before publication.

        Side Effects:
            Publishes immutable StateRef authority and installs that StateRef as
            this top-level object's last-state receipt before later derived-index,
            main-reference, or alias updates. A later update failure propagates
            while the completed receipt remains available. Explicit annotations use
            Store-local LWW current updates and never rewrite captured annotations.

        Concurrency:
            The delegated Repo save retains one configuration snapshot and keeps
            completed authority after a partial cross-Store failure; inspect the
            RepoSaveError report when publication does not complete.
        """
        from dryml.runtime import materialization_admission
        from .repo import save_object

        with materialization_admission(operation="object_save"):
            return save_object(
                self, repo=repo, main=main, store=store, alias=alias,
                deep_capture=deep_capture, match_mode=match_mode,
                graph_mode=graph_mode,
                source_store=source_store, source_stores=source_stores,
                annotations=annotations,
                report_stores=report_stores,
            )

    def save_state_to_dir(self, dest_dir: str, *, codec: str) -> None:
        """Run every applicable local-state writer in MRO order.

        Args:
            dest_dir: Empty framework-provided payload directory.
            codec: Validated opaque state codec identifier.

        Side Effects:
            Invokes each class-local ``save_state_to_dir_imp`` hook with the
            unchanged codec. Framework metadata is deliberately not written to
            ``dest_dir``.
        """
        from dryml.runtime import materialization_admission

        with materialization_admission(operation="object_save_state"):
            for cls in type(self).__mro__:
                hook = cls.__dict__.get("save_state_to_dir_imp")
                if hook is not None:
                    hook(self, dest_dir, codec=codec)

    def save_state_to_dir_imp(self, dest_dir: str, *, codec: str) -> None:
        """Write this class's local payload contribution, if any.

        Args:
            dest_dir: Framework-provided empty payload directory.
            codec: Validated opaque codec selected by the Serializable class.
        """
        pass

    def restore_state_from_dir(self, src_dir: str, *, codec: str) -> None:
        from dryml.runtime import materialization_admission

        with materialization_admission(operation="object_restore_state"):
            for cls in type(self).__mro__:
                hook = cls.__dict__.get("restore_state_from_dir_imp")
                if hook is not None:
                    hook(self, src_dir, codec=codec)

    def restore_state_from_dir_imp(self, src_dir: str, *, codec: str) -> None:
        pass


class Serializable(Object):
    """Object with a codec-selected immutable local-state payload.

    Subclasses may set ``state_codec`` to a 1-32 character ASCII alphanumeric
    identifier. DRYML validates and forwards that opaque value unchanged to
    every applicable state hook; it does not assign codec semantics.
    """

    state_codec = "pkl"


class Pickleable(Serializable):
    _HEAVY_EXCLUDE = {
        "__cdef__",
        "__ws__",
        "definition",
        "_runtime_bindings",
        "_runtime_projection",
        "_object_ref",
        "_object_id",
        "_last_state_hash",
        "_last_state_ref",
        "_save_load_reservation",
        "_restore_failed",
        "_store_affinity",
        "_realization_scope",
    }

    def save_state_to_dir_imp(self, dest_dir: str, *, codec: str) -> None:
        bindings = self._graph_binding_fields()
        heavy_state = {k: v for k, v in self.__dict__.items()
                       if k not in self._HEAVY_EXCLUDE and k not in bindings}

        # Save the entire object as a pickle
        pickle_save(heavy_state, os.path.join(dest_dir, "heavy.pkl"))

    def restore_state_from_dir_imp(self, src_dir: str, *, codec: str) -> None:
        """Replace this object's non-framework payload from its checkpoint.

        Args:
            src_dir: Checkpoint payload directory containing ``heavy.pkl``.
            codec: Opaque selected codec, accepted for hook compatibility.

        Side Effects:
            Removes current ordinary payload attributes absent from the saved
            payload, then installs the checkpoint payload. Framework identity,
            runtime bindings, reservations, and invalidation state remain live.
        """

        heavy_state = pickle_load(os.path.join(src_dir, "heavy.pkl"))
        bindings = self._graph_binding_fields()
        for key in tuple(self.__dict__):
            if key not in self._HEAVY_EXCLUDE and key not in bindings and key not in heavy_state:
                del self.__dict__[key]
        self.__dict__.update({
            key: value for key, value in heavy_state.items() if key not in bindings
        })

    def _graph_binding_fields(self) -> set[str]:
        """Return direct payload fields that retain live graph object identity."""

        from .utils.graph.value import iter_value_edges

        graph_nodes = {
            id(value) for value in getattr(self, "_runtime_projection", {}).values()
            if isinstance(value, Object)
        }

        def retains_graph_node(value, seen):
            value_id = id(value)
            if value_id in graph_nodes:
                return True
            if value_id in seen:
                return False
            seen.add(value_id)
            return any(retains_graph_node(edge.value, seen) for edge in iter_value_edges(value))

        return {
            key for key, value in self.__dict__.items()
            if key not in self._HEAVY_EXCLUDE and retains_graph_node(value, set())
        }


class Compute(Object):
    __compute_reqs__ = "plain"
    # Define the components 
    @classmethod
    def __pre_init__(cls):
        assert hasattr(cls, "__compute_reqs__"), "classes which inherit Compute must define a __compute_reqs__ attribute listing their compute requirements"
        from ..context import context_check
        context_check(cls.__compute_reqs__)


class WorkspaceCapable:
    """Opt-in: this object gets a workspace."""
    pass
