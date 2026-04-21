from __future__ import annotations

from typing import Any

from .definition import ConcreteDefinition
from .object import Object
from .utils.graph import GraphCtx, GraphVisitor
from .canonical import (
    NodeKind,
    is_runtime_leaf,
    iter_value_children,
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


class RepoStructuralVisitor(GraphVisitor):
    def __init__(self, repo):
        self.repo = repo

    def is_atomic(self, obj, ctx):
        return is_runtime_leaf(obj)

    def visit_atomic(self, obj, ctx):
        return None

    def dispatch(self, obj: Any, ctx: GraphCtx) -> None:
        from .repo import RepoSaveError
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


# ----------------------------------------------------------------------
# Save visitor
# ----------------------------------------------------------------------

class RepoSaveVisitor(RepoStructuralVisitor):
    def __init__(self, repo: "Repo", *, store=None, revision=None):
        super().__init__(repo)
        self.store = store
        self.revision = revision
        self.saved_objs: dict[int, set[ConcreteDefinition]] = {}

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

class RepoAddObjectsVisitor(RepoStructuralVisitor):
    def __init__(self, repo: "Repo", *, store=None):
        super().__init__(repo)
        self.store = store

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
