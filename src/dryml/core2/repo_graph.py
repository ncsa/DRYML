from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .definition import ConcreteDefinition, Definition
from .freeze import (
    FrozenDict,
    FrozenList,
    FrozenNDArray,
    FrozenSet,
    FrozenTuple,
)
from .object import Object
from .policies import CachePolicy, InstancePolicy
from .types import is_pod
from .utils.graph import GraphCtx, GraphTransformer, GraphVisitor


# ----------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------

def _path_part_from_key(k: Any) -> str | int:
    return k if isinstance(k, (str, int)) else str(k)


def manage_revision(obj: Any, revision: RevisionMapType | str | None):
    if revision is None:
        return {}
    elif isinstance(revision, str):
        if isinstance(obj, Object):
            return {obj.definition: revision}
        elif isinstance(obj, ConcreteDefinition):
            return {obj: revision}
        else:
            raise ValueError("When revision is a string, manage_revision must get a clear object or definition to create the revision dictionary.")
    else:
        return revision


# ----------------------------------------------------------------------
# Save visitor
# ----------------------------------------------------------------------

class RepoSaveVisitor(GraphVisitor):
    def __init__(self, repo: "Repo", *, store=None, revision=None):
        self.repo = repo
        self.store = store
        self.revision = revision
        self.saved_objs: dict[int, set[ConcreteDefinition]] = {}

    def is_atomic(self, obj: Any, ctx: GraphCtx) -> bool:
        return is_pod(obj) or isinstance(obj, (np.ndarray, FrozenNDArray))

    def visit_atomic(self, obj: Any, ctx: GraphCtx) -> None:
        return None

    def dispatch(self, obj: Any, ctx: GraphCtx) -> None:
        # domain-priority dispatch first, since some frozen types are tagged
        if isinstance(obj, Object):
            self.visit_object(obj, ctx)
            return

        if isinstance(obj, ConcreteDefinition):
            self.visit_concrete_definition(obj, ctx)
            return

        if isinstance(obj, FrozenList):
            self.visit_frozen_list(obj, ctx)
            return
        if isinstance(obj, FrozenTuple):
            self.visit_frozen_tuple(obj, ctx)
            return
        if isinstance(obj, FrozenSet):
            self.visit_frozen_set(obj, ctx)
            return
        if isinstance(obj, FrozenDict):
            self.visit_frozen_dict(obj, ctx)
            return

        if isinstance(obj, Definition):
            self.visit_definition(obj, ctx)
            return

        return super().dispatch(obj, ctx)

    def visit_object(self, obj: Object, ctx: GraphCtx) -> None:
        self._save_single_object(obj, ctx)

    def visit_concrete_definition(self, obj: ConcreteDefinition, ctx: GraphCtx) -> None:
        linked_obj = self.repo.get_cached(obj)
        if linked_obj is None:
            raise RepoSaveError(
                f"Definition of object {obj} is not reachable in this repo!"
            )
        self.visit(linked_obj, ctx)

    def visit_definition(self, obj: Definition, ctx: GraphCtx) -> None:
        raise RepoSaveError("Plain Definitions aren't allowed here.")

    def visit_frozen_tuple(self, obj: FrozenTuple, ctx: GraphCtx) -> None:
        for i, v in enumerate(obj):
            self.visit(v, ctx.child(i))

    def visit_frozen_list(self, obj: FrozenList, ctx: GraphCtx) -> None:
        for i, v in enumerate(obj):
            self.visit(v, ctx.child(i))

    def visit_frozen_set(self, obj: FrozenSet, ctx: GraphCtx) -> None:
        for i, v in enumerate(obj):
            self.visit(v, ctx.child(i))

    def visit_frozen_dict(self, obj: FrozenDict, ctx: GraphCtx) -> None:
        for k, v in obj.items():
            self.visit(v, ctx.child(_path_part_from_key(k)))

    def visit_tuple(self, obj: tuple[Any, ...], ctx: GraphCtx) -> None:
        for i, v in enumerate(obj):
            self.visit(v, ctx.child(i))

    def visit_list(self, obj: list[Any], ctx: GraphCtx) -> None:
        for i, v in enumerate(obj):
            self.visit(v, ctx.child(i))

    def visit_set(self, obj: set[Any], ctx: GraphCtx) -> None:
        for i, v in enumerate(obj):
            self.visit(v, ctx.child(i))

    def visit_dict(self, obj: dict[Any, Any], ctx: GraphCtx) -> None:
        for k, v in obj.items():
            self.visit(v, ctx.child(_path_part_from_key(k)))

    def visit_other(self, obj: Any, ctx: GraphCtx) -> None:
        raise RepoSaveError(
            f"Cannot save object of type {type(obj).__name__} at {ctx.path_str()}!"
        )

    def _save_single_object(self, obj: Object, ctx: GraphCtx) -> None:
        cdef = obj.definition
        store = self.store
        if store is None:
            if cdef in self.repo.obj_default_store:
                store = self.repo.obj_default_store[cdef]
            else:
                store = self.repo.default_store

        self.visit(cdef.args, ctx.child("args"))
        self.visit(cdef.kwargs, ctx.child("kwargs"))

        if cdef in self.repo.strong_obj_cache:
            if obj is not self.repo.strong_obj_cache[cdef]:
                raise ValueError(
                    f"We already have a different object with definition: {cdef}"
                )
        else:
            self.repo.pin(obj)

        if store is None:
            raise RepoSaveError("No store available to save object!")

        if id(store) not in self.saved_objs:
            self.saved_objs[id(store)] = set()

        if cdef not in self.saved_objs[id(store)]:
            revision_str = self.revision.get(cdef, None)
            store.save_object(obj, revision=revision_str)
            self.saved_objs[id(store)].add(cdef)
            self.repo._num_saves += 1


# ----------------------------------------------------------------------
# Add-objects visitor
# ----------------------------------------------------------------------

class RepoAddObjectsVisitor(GraphVisitor):
    def __init__(self, repo: "Repo", *, store=None):
        self.repo = repo
        self.store = store

    def is_atomic(self, obj: Any, ctx: GraphCtx) -> bool:
        # Treat arrays as leaves. The current implementation raises for these,
        # but that is usually not what you want in object definitions.
        return is_pod(obj) or isinstance(obj, (np.ndarray, FrozenNDArray))

    def visit_atomic(self, obj: Any, ctx: GraphCtx) -> None:
        return None

    def dispatch(self, obj: Any, ctx: GraphCtx) -> None:
        if isinstance(obj, Object):
            self.visit_object(obj, ctx)
            return

        if isinstance(obj, ConcreteDefinition):
            self.visit_concrete_definition(obj, ctx)
            return

        if isinstance(obj, FrozenList):
            self.visit_frozen_list(obj, ctx)
            return
        if isinstance(obj, FrozenTuple):
            self.visit_frozen_tuple(obj, ctx)
            return
        if isinstance(obj, FrozenSet):
            self.visit_frozen_set(obj, ctx)
            return
        if isinstance(obj, FrozenDict):
            self.visit_frozen_dict(obj, ctx)
            return

        if isinstance(obj, Definition):
            self.visit_definition(obj, ctx)
            return

        return super().dispatch(obj, ctx)

    def visit_object(self, obj: Object, ctx: GraphCtx) -> None:
        cdef = obj.definition
        self.visit(cdef.args, ctx.child("args"))
        self.visit(cdef.kwargs, ctx.child("kwargs"))

        if cdef in self.repo.strong_obj_cache and (obj is not self.repo.strong_obj_cache[cdef]):
            raise KeyError(f"Repo already has a different object matching {cdef}!")

        self._add_object_single(obj)

    def visit_concrete_definition(self, obj: ConcreteDefinition, ctx: GraphCtx) -> None:
        linked_obj = self.repo.get_cached(obj)
        if linked_obj is None:
            from .repo import _global_repo

            if self.repo is not _global_repo:
                linked_obj = _global_repo.get_cached(obj)

        if linked_obj is None:
            raise KeyError(f"No object linked to definition {obj} found in repo!")

        self.visit(linked_obj, ctx)

    def visit_definition(self, obj: Definition, ctx: GraphCtx) -> None:
        raise ValueError("Plain Definitions aren't allowed here.")

    def visit_frozen_tuple(self, obj: FrozenTuple, ctx: GraphCtx) -> None:
        for i, v in enumerate(obj):
            self.visit(v, ctx.child(i))

    def visit_frozen_list(self, obj: FrozenList, ctx: GraphCtx) -> None:
        for i, v in enumerate(obj):
            self.visit(v, ctx.child(i))

    def visit_frozen_set(self, obj: FrozenSet, ctx: GraphCtx) -> None:
        for i, v in enumerate(obj):
            self.visit(v, ctx.child(i))

    def visit_frozen_dict(self, obj: FrozenDict, ctx: GraphCtx) -> None:
        for k, v in obj.items():
            self.visit(v, ctx.child(_path_part_from_key(k)))

    def visit_tuple(self, obj: tuple[Any, ...], ctx: GraphCtx) -> None:
        for i, v in enumerate(obj):
            self.visit(v, ctx.child(i))

    def visit_list(self, obj: list[Any], ctx: GraphCtx) -> None:
        for i, v in enumerate(obj):
            self.visit(v, ctx.child(i))

    def visit_set(self, obj: set[Any], ctx: GraphCtx) -> None:
        for i, v in enumerate(obj):
            self.visit(v, ctx.child(i))

    def visit_dict(self, obj: dict[Any, Any], ctx: GraphCtx) -> None:
        for k, v in obj.items():
            self.visit(v, ctx.child(_path_part_from_key(k)))

    def visit_other(self, obj: Any, ctx: GraphCtx) -> None:
        raise TypeError(
            f"Unsupported type {type(obj).__name__} found when adding objects to repo at {ctx.path_str()}."
        )

    def _add_object_single(self, obj: Object) -> None:
        self.repo.pin(obj)
        if self.store is not None:
            self.repo.obj_default_store[obj.definition] = self.store
        else:
            if self.repo.default_store is not None:
                self.repo.obj_default_store[obj.definition] = self.repo.default_store


# ----------------------------------------------------------------------
# Realize transformer
# ----------------------------------------------------------------------

@dataclass(slots=True)
class RepoRealizeConfig:
    instance: InstancePolicy = "reuse"
    restore_state: bool = True
    build_missing: bool = False
    reuse_weak: bool = True
    cache: CachePolicy = "weak"
    revision: dict | None = None


class RepoRealizeTransformer(GraphTransformer):
    def __init__(self, repo: "Repo", config: RepoRealizeConfig):
        self.repo = repo
        self.config = config

    def is_atomic(self, obj: Any, ctx: GraphCtx) -> bool:
        return is_pod(obj)

    def transform_atomic(self, obj: Any, ctx: GraphCtx) -> Any:
        return obj

    def memo_key(self, obj: Any, ctx: GraphCtx):
        # Match the current important semantic: memoize materialized objects by cdef.
        if isinstance(obj, ConcreteDefinition):
            return obj
        return None

    def dispatch(self, obj: Any, ctx: GraphCtx) -> Any:
        # domain-priority dispatch first
        if isinstance(obj, FrozenNDArray):
            return self.transform_frozen_ndarray(obj, ctx)

        if isinstance(obj, np.ndarray):
            return self.transform_ndarray(obj, ctx)

        if isinstance(obj, ConcreteDefinition):
            return self.transform_concrete_definition(obj, ctx)

        if isinstance(obj, Definition):
            return self.transform_definition(obj, ctx)

        if isinstance(obj, Object):
            return self.transform_object(obj, ctx)

        if isinstance(obj, FrozenList):
            return self.transform_frozen_list(obj, ctx)
        if isinstance(obj, FrozenTuple):
            return self.transform_frozen_tuple(obj, ctx)
        if isinstance(obj, FrozenSet):
            return self.transform_frozen_set(obj, ctx)
        if isinstance(obj, FrozenDict):
            return self.transform_frozen_dict(obj, ctx)

        return super().dispatch(obj, ctx)

    def transform_frozen_ndarray(self, obj: FrozenNDArray, ctx: GraphCtx) -> Any:
        if hasattr(obj, "thaw"):
            return obj.thaw()
        return np.array(obj, copy=True)

    def transform_ndarray(self, obj: np.ndarray, ctx: GraphCtx) -> Any:
        return np.array(obj, copy=True)

    def transform_concrete_definition(self, obj: ConcreteDefinition, ctx: GraphCtx) -> Any:
        revision = manage_revision(obj, self.config.revision)
        return self.repo._materialize_cdef(
            obj,
            revision,
            instance=self.config.instance,
            restore_state=self.config.restore_state,
            build_missing=self.config.build_missing,
            reuse_weak=self.config.reuse_weak,
            cache=self.config.cache,
            memo=ctx.memo,
            path=list(ctx.path),
        )

    def transform_definition(self, obj: Definition, ctx: GraphCtx) -> Any:
        cdef = obj.concretize(repo=self.repo)
        revision = manage_revision(cdef, self.config.revision)
        return self.repo._materialize_cdef(
            cdef,
            revision,
            instance=self.config.instance,
            restore_state=self.config.restore_state,
            build_missing=self.config.build_missing,
            reuse_weak=self.config.reuse_weak,
            cache=self.config.cache,
            memo=ctx.memo,
            path=list(ctx.path),
        )

    def transform_object(self, obj: Object, ctx: GraphCtx) -> Any:
        revision = manage_revision(obj.definition, self.config.revision)

        if self.config.instance == "new":
            return self.repo._materialize_cdef(
                obj.definition,
                revision,
                instance="new",
                restore_state=self.config.restore_state,
                build_missing=self.config.build_missing,
                reuse_weak=self.config.reuse_weak,
                cache=self.config.cache,
                memo=ctx.memo,
                path=list(ctx.path),
            )

        if self.repo.get_cached(obj.definition, reuse_weak=self.config.reuse_weak) is None:
            self.repo.cache_weak(obj)

        return obj

    def transform_frozen_tuple(self, obj: FrozenTuple, ctx: GraphCtx) -> Any:
        return tuple(self.transform(v, ctx.child(i)) for i, v in enumerate(obj))

    def transform_frozen_list(self, obj: FrozenList, ctx: GraphCtx) -> Any:
        return [self.transform(v, ctx.child(i)) for i, v in enumerate(obj)]

    def transform_frozen_set(self, obj: FrozenSet, ctx: GraphCtx) -> Any:
        out = set()
        for i, v in enumerate(obj):
            out.add(self.transform(v, ctx.child(i)))
        return out

    def transform_frozen_dict(self, obj: FrozenDict, ctx: GraphCtx) -> Any:
        out = {}
        for k, v in obj.items():
            rk = self.transform(k, ctx.child("<key>"))
            rv = self.transform(v, ctx.child(_path_part_from_key(k)))
            out[rk] = rv
        return out

    def transform_tuple(self, obj: tuple[Any, ...], ctx: GraphCtx) -> Any:
        return tuple(self.transform(v, ctx.child(i)) for i, v in enumerate(obj))

    def transform_list(self, obj: list[Any], ctx: GraphCtx) -> Any:
        return [self.transform(v, ctx.child(i)) for i, v in enumerate(obj)]

    def transform_set(self, obj: set[Any], ctx: GraphCtx) -> Any:
        out = set()
        for i, v in enumerate(obj):
            out.add(self.transform(v, ctx.child(i)))
        return out

    def transform_dict(self, obj: dict[Any, Any], ctx: GraphCtx) -> Any:
        out = {}
        for k, v in obj.items():
            rk = self.transform(k, ctx.child("<key>"))
            rv = self.transform(v, ctx.child(_path_part_from_key(k)))
            out[rk] = rv
        return out

    def transform_other(self, obj: Any, ctx: GraphCtx) -> Any:
        raise RepoLoadError(
            f"Cannot realize type {type(obj).__name__} at {ctx.path_str()}"
        )
