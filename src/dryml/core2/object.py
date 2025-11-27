from __future__ import annotations

from functools import cached_property
import uuid
import time
import os
import inspect

from dryml.core2.util import pickle_save, pickle_load, _validate_init_sig
from dryml.core2.definition import Definition, \
    deepcopy_skip_definition_object
from copy import deepcopy


class Dryml(type):
    # Support metaclass to enable capture of input arguments

    def __call__(cls, *args, repo=None, **kwargs):
        from dryml.core2.repo import manage_repo

        with manage_repo(repo=repo) as sub_repo:
            # Build initial Definition
            defn = Definition(
                cls,
                *args,
                repo=sub_repo,
                *kwargs
            )

            # Concretize pass
            cdef = sub_repo.concretize_definition(
                defn)

            rt_args = sub_repo.load_object(cdef.args, build_missing=True)
            rt_kwargs = sub_repo.load_object(cdef.kwargs, build_missing=True)

        # Create the object initially
        obj = cls.__new__(cls)

        # Initialize object with 
        obj.__init__(*rt_args, **rt_kwargs)
        # Deepcopy the concrete definition so it's version of the arguments
        # is true to how it was originally called.
        obj.__cdef__ = deepcopy(cdef)

        return obj


class Object(metaclass=Dryml):
    # Base type for using CreationControl metaclass.
    # Provides basic implementations for all methods used
    # In the CreationControl process

    @classmethod
    def __prepare_args__(cls, *args, **kwargs):
        # __prepare_args__ should be an idempotent function
        return args, kwargs

    @classmethod
    def __strip_unique_args__(cls, *args, **kwargs):
        # __strip_unique_args__ should be an idempotent function
        return args, kwargs

    @classmethod
    def defn(cls, *args, repo=None, **kwargs) -> "Definition":
        from dryml.core2.definition import Definition
        return Definition(cls, *args, repo=repo, **kwargs)

    # Alias for defn
    d = defn

    def __init__(self):
        pass

    @cached_property
    def definition(self) -> "ConcreteDefinition":
        # Get a `Definition` object for this particular object.
        return self.__cdef__

    def __hash__(self):
        # Objects are hashable through through its `ConcreteDefinition`
        return hash(self.definition)

    def __repr__(self):
        return f"<{self.definition.cls} at {hex(id(self))}>(args={self.definition.args}, kwargs={self.definition.kwargs})"

    def save(self, repo=None):
        from dryml.core2.repo import save_object
        save_object(self, repo=repo)

    def save_to_dir(self, dest_dir: str):
        pickle_save(self.definition, os.path.join(dest_dir, 'def.pkl'))
        self.save_imp(dest_dir)

    def save_imp(self, dest_dir: str):
        pass

    def load(self, repo=None):
        from dryml.core2.repo import load_object
        load_object(self, repo=repo)
            

    def load_from_dir(self, src_dir: str):
        loaded_def = pickle_load(os.path.join(src_dir, "def.pkl"))
        assert loaded_def == self.definition, f"Loaded definition {loaded_def} doesn't match expected definition {self.definition}"
        self.load_imp(src_dir)

    def load_imp(self, src_dir: str):
        pass


class Pickleable(Object):
    def save_imp(self, dest_dir: str):
        # Grab all heavy-state data
        heavy_state = {}
        for key in self.__dict__:
            if key not in self.__orig_keys__:
                heavy_state[key] = getattr(self, key)

        # Save the entire object as a pickle
        pickle_save(heavy_state, os.path.join(dest_dir, "heavy.pkl"))

    def load_imp(self, src_dir: str):
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
