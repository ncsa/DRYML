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
from .canonical import (
    NodeKind,
    is_canonical_value,
    is_runtime_leaf,
    iter_value_children,
    map_dict_items,
    matching_container_family,
    _path_part_from_key,
    node_kind)


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
        return is_runtime_leaf(obj)

    def visit_atomic(self, obj: Any, ctx: GraphCtx) -> None:
        return None

    def dispatch(self, obj: Any, ctx: GraphCtx) -> None:
        kind = node_kind(obj)

        if kind is NodeKind.OBJECT:
            self.visit_object(obj, ctx)
            return

        if kind is NodeKind.CONCRETE_DEFINITION:
            self.visit_concrete_definition(obj, ctx)
            return

        if kind is NodeKind.DEFINITION:
            raise RepoSaveError("Plain Definitions aren't allowed here.")

        if kind in {
            NodeKind.LIST,
            NodeKind.TUPLE,
            NodeKind.SET,
            NodeKind.DICT,
            NodeKind.FROZEN_LIST,
            NodeKind.FROZEN_TUPLE,
            NodeKind.FROZEN_SET,
            NodeKind.FROZEN_DICT,
        }:
            for part, child in iter_value_children(obj):
                self.visit(child, ctx.child(part))
            return

        raise RepoSaveError(
            f"Cannot save object of type {type(obj).__name__} at {ctx.path_str()}!"
        )

    def visit_object(self, obj: Object, ctx: GraphCtx) -> None:
        self._save_single_object(obj, ctx)

    def visit_concrete_definition(self, obj: ConcreteDefinition, ctx: GraphCtx) -> None:
        linked_obj = self.repo.get_cached(obj)
        if linked_obj is None:
            raise RepoSaveError(
                f"Definition of object {obj} is not reachable in this repo!"
            )
        self.visit(linked_obj, ctx)

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
        return is_runtime_leaf(obj)

    def visit_atomic(self, obj: Any, ctx: GraphCtx) -> None:
        return None

    def dispatch(self, obj: Any, ctx: GraphCtx) -> None:
        kind = node_kind(obj)

        if kind is NodeKind.OBJECT:
            self.visit_object(obj, ctx)
            return

        if kind is NodeKind.CONCRETE_DEFINITION:
            self.visit_concrete_definition(obj, ctx)
            return

        if kind is NodeKind.DEFINITION:
            raise ValueError("Plain Definitions aren't allowed here.")

        if kind in {
            NodeKind.LIST,
            NodeKind.TUPLE,
            NodeKind.SET,
            NodeKind.DICT,
            NodeKind.FROZEN_LIST,
            NodeKind.FROZEN_TUPLE,
            NodeKind.FROZEN_SET,
            NodeKind.FROZEN_DICT,
        }:
            for part, child in iter_value_children(obj):
                self.visit(child, ctx.child(part))
            return

        raise TypeError(
            f"Unsupported type {type(obj).__name__} found when adding objects to repo at {ctx.path_str()}."
        )

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
        return node_kind(obj) in {NodeKind.POD, NodeKind.TYPE}

    def transform_atomic(self, obj: Any, ctx: GraphCtx) -> Any:
        return obj

    def memo_key(self, obj: Any, ctx: GraphCtx):
        if node_kind(obj) is NodeKind.CONCRETE_DEFINITION:
            return obj
        return None

    def should_track_cycle(self, obj: Any, ctx: GraphCtx) -> bool:
        return node_kind(obj) in {
            NodeKind.LIST,
            NodeKind.TUPLE,
            NodeKind.SET,
            NodeKind.DICT,
            NodeKind.FROZEN_LIST,
            NodeKind.FROZEN_TUPLE,
            NodeKind.FROZEN_SET,
            NodeKind.FROZEN_DICT,
            NodeKind.DEFINITION,
            NodeKind.OBJECT,
        }

    def dispatch(self, obj: Any, ctx: GraphCtx) -> Any:
        kind = node_kind(obj)

        if kind is NodeKind.FROZEN_NDARRAY:
            return obj.thaw() if hasattr(obj, "thaw") else np.array(obj, copy=True)

        if kind is NodeKind.NDARRAY:
            return np.array(obj, copy=True)

        if kind is NodeKind.CONCRETE_DEFINITION:
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

        if kind is NodeKind.DEFINITION:
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

        if kind is NodeKind.OBJECT:
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

        if kind is NodeKind.FROZEN_TUPLE:
            return tuple(self.transform(v, ctx.child(p)) for p, v in iter_value_children(obj))

        if kind is NodeKind.FROZEN_LIST:
            return [self.transform(v, ctx.child(p)) for p, v in iter_value_children(obj)]

        if kind is NodeKind.FROZEN_SET:
            return {self.transform(v, ctx.child(p)) for p, v in iter_value_children(obj)}

        if kind is NodeKind.TUPLE:
            return tuple(self.transform(v, ctx.child(p)) for p, v in iter_value_children(obj))

        if kind is NodeKind.LIST:
            return [self.transform(v, ctx.child(p)) for p, v in iter_value_children(obj)]

        if kind is NodeKind.SET:
            return {self.transform(v, ctx.child(p)) for p, v in iter_value_children(obj)}

        if kind in {NodeKind.DICT, NodeKind.FROZEN_DICT}:
            return map_dict_items(
                obj,
                key_fn=lambda k: self.transform(k, ctx.child("<key>")),
                value_fn=lambda k, v: self.transform(v, ctx.child(_path_part_from_key(k))),
            )

        raise RepoLoadError(
            f"Cannot realize type {type(obj).__name__} at {ctx.path_str()}"
        )
