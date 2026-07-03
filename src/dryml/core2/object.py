from __future__ import annotations

from typing import TYPE_CHECKING

import uuid
import time
import os
from contextlib import contextmanager

from .utils.general import pickle_save, pickle_load, revision_path
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
    # Support metaclass to enable capture of input arguments

    def __call__(cls, *args, repo=None, __cdef__=None, **kwargs):
        from .session import get_config

        session_config = get_config()
        object_mode = session_config.object_mode
        active_repo = repo if repo is not None else session_config.repo

        if __cdef__ is None and object_mode == "definition":
            defn = cls.defn(*args, **kwargs)
            return defn

        if __cdef__ is None and object_mode == "concrete":
            return cls.defn(*args, **kwargs).concretize(repo=active_repo)

        if __cdef__ is None and object_mode == "selector":
            return cls.defn(*args, **kwargs).as_selector()

        if __cdef__ is None and object_mode == "space":
            return cls.defn(*args, **kwargs).as_space()

        if __cdef__ is None and object_mode == "load_or_build":
            from .repo import manage_repo

            with manage_repo(repo=active_repo) as sub_repo:
                _cache_runtime_object_args(sub_repo, args, kwargs)
                cdef = Definition(cls, *args, **kwargs).concretize(repo=sub_repo)
                return sub_repo.load_or_build(cdef, cache=session_config.cache)

        from .repo import default_repo, manage_repo
        with manage_repo(repo=active_repo) as sub_repo:
            if __cdef__ is None:
                # First-time construction from a soft Definition
                _cache_runtime_object_args(sub_repo, args, kwargs)
                defn = Definition(cls, *args, **kwargs)
                cdef = defn.concretize(repo=sub_repo)

                rt_args = sub_repo.load_object(cdef.args, build_missing=True)
                rt_kwargs = sub_repo.load_object(cdef.kwargs, build_missing=True)

            else:
                # Reconstruction from an existing ConcreteDefinition
                cdef = __cdef__
                rt_args = args
                rt_kwargs = kwargs

            # Run pre-init check
            cls.__pre_init__()

            # Resolve host/runtime-specific config leaves after identity has been
            # computed, but before the user initializer receives its arguments.
            rt_args = sub_repo.resolve_config(rt_args)
            rt_kwargs = sub_repo.resolve_config(rt_kwargs)

            # Actual object allocation
            obj = cls.__new__(cls)

            # Attach the definition to the object.
            obj.__cdef__ = cdef

            # Set the workspace
            if isinstance(obj, WorkspaceCapable):
                ws = sub_repo.workspace_manager.alloc(cdef.stable_hash())
                os.makedirs(ws.path(), exist_ok=True)
                obj.__ws__ = ws
            else:
                obj.__ws__ = None

            # Initialize with runtime (built) args while exposing the construction
            # repo to code that consults get_default_repo().
            from .session import config
            with config(object_mode="fresh"), default_repo(sub_repo):
                obj.__init__(*rt_args, **rt_kwargs)


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
        from .repo import get_default_repo
        return get_default_repo().location(self)


    @property
    def definition(self) -> "ConcreteDefinition":
        # Get a `Definition` object for this particular object.
        return self.__cdef__

    def __hash__(self):
        # Objects are hashable through through its `ConcreteDefinition`
        return hash(self.definition)

    def __repr__(self):
        return f"<{self.definition.cls} at {hex(id(self))}>(args={self.definition.args}, kwargs={self.definition.kwargs})"

    def save(
            self,
            repo=None,
            main=True,
            revision: RevisionType|str|None = None,
            *,
            store=None,
            alias: str | None = None,
            record_policy=None,
            record_options=None,
            options=None):
        from .repo import save_object
        save_object(
            self,
            repo=repo,
            main=main,
            revision=revision,
            store=store,
            alias=alias,
            record_policy=record_policy,
            record_options=record_options,
            options=options,
        )

    def save_state_to_dir(self, dest_dir: str, revision: str|None = None):
        pickle_save(self.definition, os.path.join(dest_dir, 'def.pkl'))
        self.save_state_to_dir_imp(dest_dir, revision)

    def save_state_to_dir_imp(self, dest_dir: str, revision: str|None = None):
        pass

    def load(self, repo=None, revision: RevisionType|str|None = None):
        from .repo_graph import manage_revision
        from .repo import load_object
        revision = manage_revision(self, revision)
        load_object(self, repo=repo, revision=revision)
            
    def restore_state_from_dir(self, src_dir: str, revision: str|None = None):
        loaded_def = pickle_load(os.path.join(src_dir, "def.pkl"))
        assert loaded_def == self.definition, f"Loaded definition {loaded_def} doesn't match expected definition {self.definition}"
        self.restore_state_from_dir_imp(src_dir, revision=revision)

    def restore_state_from_dir_imp(self, src_dir: str, revision: str|None = None):
        pass


class Serializable(Object):
    """Object with repo-backed persisted state by default."""


class Pickleable(Serializable):
    _HEAVY_EXCLUDE = {
        "__cdef__",
        "__ws__",
        "definition",
    }

    def save_state_to_dir_imp(self, dest_dir: str, revision: str|None=None):
        # Grab all heavy-state data
        heavy_state = {k: v for k, v in self.__dict__.items()
                       if k not in self._HEAVY_EXCLUDE}

        # Save the entire object as a pickle
        rev_path = revision_path("heavy", "pkl", dest_dir, revision=revision)
        pickle_save(
            heavy_state,
            rev_path
            )

    def restore_state_from_dir_imp(self, src_dir: str, revision: str|None):
        # heavy-state data is stored in heavy.pkl
        heavy_state = pickle_load(
            revision_path("heavy", "pkl", src_dir, revision=revision))

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
        from dryml.runtime import require_allocation_for_legacy_compute_reqs
        require_allocation_for_legacy_compute_reqs(cls.__compute_reqs__, f"initialize compute object {cls.__name__}")


class WorkspaceCapable:
    """Opt-in: this object gets a workspace."""
    pass
