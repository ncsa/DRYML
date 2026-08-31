from __future__ import annotations

from typing import TYPE_CHECKING
from threading import Lock

import uuid
import time
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


class Dryml(type):
    """Capture Object construction calls and apply the active repository mode."""

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
                    rt_args = sub_repo.load_object(canonical_args, build_missing=True)
                    rt_kwargs = sub_repo.load_object(canonical_kwargs, build_missing=True)

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
                attach_runtime_binding(sub_repo, cdef, obj, memo)


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
    # Base type for using CreationControl metaclass.
    # Provides basic implementations for all methods used
    # In the CreationControl process

    __ws__: WorkspaceHandle | None
    __cdef__: ConcreteDefinition

    @classmethod
    def __prepare_args__(cls, *args, **kwargs):
        # __prepare_args__ should be an idempotent function
        return args, kwargs

    @classmethod
    def __strip_unique_args__(cls, *args, **kwargs):
        # __strip_unique_args__ should be an idempotent function
        return args, kwargs

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
    def location(self) -> str:
        repo = getattr(self, "_store_affinity", None)
        if repo is not None:
            return repo.object_dir(self.definition)
        raise RuntimeError("Object has no retained Store affinity.")


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
            *,
            main=True,
            store=None,
            alias: str | None = None,
            deep_capture: bool = False,
            federated: bool = False,
            report_stores: bool = False):
        """Publish this graph as an immutable exact StateRef.

        Args:
            repo: Repository or Store authority used for publication.
            main: Whether to update the target Store's structural main reference
                after StateRef publication succeeds.
            store: Optional selected target Store.
            alias: Optional object alias to update after StateRef publication.
            deep_capture: Whether to serialize every owned Serializable node.
            federated: Whether validated dependency states may remain external.
            report_stores: Whether to also return the selected Store report.

        Returns:
            The published ``StateRef``, or it paired with a ``StoreReport``.

        Raises:
            StoreAuthorityError: If checkpoint or graph publication cannot be
                completed atomically.
        """
        from dryml.runtime import materialization_admission
        from .repo import save_object

        with materialization_admission(operation="object_save"):
            return save_object(
                self, repo=repo, main=main, store=store, alias=alias,
                deep_capture=deep_capture, federated=federated,
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

    def load(self, repo=None, revision: RevisionType|str|None = None):
        from dryml.runtime import materialization_admission
        from .repo_graph import manage_revision
        from .repo import load_object
        with materialization_admission(operation="object_load"):
            revision = manage_revision(self, revision)
            load_object(self, repo=repo, revision=revision)
            
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
        "_save_load_reservation",
        "_store_affinity",
        "_realization_scope",
    }

    def save_state_to_dir_imp(self, dest_dir: str, *, codec: str) -> None:
        # Grab all heavy-state data
        heavy_state = {k: v for k, v in self.__dict__.items()
                       if k not in self._HEAVY_EXCLUDE}

        # Save the entire object as a pickle
        pickle_save(heavy_state, os.path.join(dest_dir, "heavy.pkl"))

    def restore_state_from_dir_imp(self, src_dir: str, *, codec: str) -> None:
        # heavy-state data is stored in heavy.pkl
        heavy_state = pickle_load(os.path.join(src_dir, "heavy.pkl"))

        self.__dict__.update(heavy_state)


class UniqueID(Object):
    # Mixing in this class adds a `uid` keyword argument which is
    # initialized automatically if not provided.
    @classmethod
    def __prepare_args__(cls, *args, **kwargs):
        args, kwargs = super().__prepare_args__(*args, **kwargs)
        kwargs.setdefault("uid", str(uuid.uuid4()))
        return args, kwargs

    @classmethod
    def __strip_unique_args__(cls, *args, **kwargs):
        args, kwargs = super().__strip_unique_args__(*args, **kwargs)
        kwargs = kwargs.copy()
        if 'uid' in kwargs:
            del kwargs['uid']
        return args, kwargs

    def __init__(self, *args, uid=None, **kwargs):
        super().__init__(*args, **kwargs)
        # unique ID
        self.uid = uid


class Metadata(Object):
    # Mixing in this class adds a `metadata` keyword argument which is
    # used to store a basic 'description', and 'creation_time' metadata
    # along with any other metadata the user wishes to store.
    @classmethod
    def __prepare_args__(cls, *args, **kwargs):
        args, kwargs = super().__prepare_args__(*args, **kwargs)
        if 'metadata' not in kwargs:
            kwargs['metadata'] = {
            }
        if 'description' not in kwargs['metadata']:
            kwargs['metadata']['description'] = ""
        if 'creation_time' not in kwargs['metadata']:
            kwargs['metadata']['creation_time'] = time.time()
        return args, kwargs

    @classmethod
    def __strip_unique_args__(cls, *args, **kwargs):
        args, kwargs = super().__strip_unique_args__(*args, **kwargs)
        kwargs = kwargs.copy()
        if 'metadata' in kwargs:
            del kwargs['metadata']
        return args, kwargs

    def __init__(self, *args, metadata=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.metadata = metadata


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
