from __future__ import annotations

from typing import Any

from .definition import ConcreteDefinition
from .object import Object
from .policies import RepoGraphOptions
from .utils.graph import GraphCtx, GraphVisitor
from .canonical import (
    NodeKind,
    is_runtime_leaf,
    iter_value_children,
    node_kind)


# ----------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------

RevisionMapType = dict[ConcreteDefinition, str]


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


def _as_graph_options(options: RepoGraphOptions | None, **kwargs) -> RepoGraphOptions:
    if options is not None:
        if kwargs:
            raise TypeError("Cannot pass graph option kwargs when options is provided.")
        return options
    return RepoGraphOptions(**kwargs)


class RepoStructuralVisitor(GraphVisitor):
    def __init__(self, repo):
        self.repo = repo

    def is_atomic(self, obj, ctx):
        return is_runtime_leaf(obj)

    def visit_atomic(self, obj, ctx):
        return None

    def dispatch(self, obj: Any, ctx: GraphCtx) -> None:
        from .repo import RepoGraphError
        kind = node_kind(obj)

        if kind is NodeKind.OBJECT:
            self.visit_object(obj, ctx)
            return

        if kind is NodeKind.CONCRETE_DEFINITION:
            self.visit_concrete_definition(obj, ctx)
            return

        if kind is NodeKind.DEFINITION:
            raise RepoGraphError("Plain Definitions aren't allowed here.")

        if kind is NodeKind.FUNCTION:
            raise RepoGraphError("Plain functions aren't allowed here.")

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

        raise RepoGraphError(
            f"Unexpected object of type {type(obj).__name__} at {ctx.path_str()}!"
        )


class RepoObjectGraphVisitor(RepoStructuralVisitor):
    def __init__(self, repo, *, options: RepoGraphOptions | None = None, **kwargs):
        super().__init__(repo)
        self.options = _as_graph_options(options, **kwargs)
        if self.options.order not in ("pre", "post"):
            raise ValueError("Repo graph order must be 'pre' or 'post'.")
        if self.options.missing not in ("raise", "skip", "load"):
            raise ValueError("Repo graph missing policy must be 'raise', 'skip', or 'load'.")
        self._seen_cdefs: set[ConcreteDefinition] = set()
        self._load_memo: dict[ConcreteDefinition, Object] = {}

    def _should_visit_graph_object(self, ctx: GraphCtx) -> bool:
        return self.options.include_root or bool(ctx.path)

    def _resolve_cdef(self, cdef: ConcreteDefinition, ctx: GraphCtx) -> Object | None:
        obj = self.repo.get_cached(cdef, reuse_weak=self.options.load.reuse_weak)
        if obj is not None:
            return obj

        if self.options.missing == "skip":
            return None

        if self.options.missing == "load":
            load = self.options.load
            revision = manage_revision(cdef, load.revision)
            return self.repo._materialize_cdef(
                cdef,
                revision,
                instance=load.instance,
                restore_state=load.restore_state,
                build_missing=load.build_missing,
                reuse_weak=load.reuse_weak,
                cache=load.cache,
                memo=self._load_memo,
                path=list(ctx.path),
            )

        from .repo import RepoGraphError

        raise RepoGraphError(
            f"Definition {cdef} is not reachable as a live object in this repo at {ctx.path_str()}."
        )

    def visit_object(self, obj: Object, ctx: GraphCtx) -> None:
        cdef = obj.definition
        if self.options.dedupe:
            if cdef in self._seen_cdefs:
                return
            self._seen_cdefs.add(cdef)

        should_apply = self._should_visit_graph_object(ctx)
        if self.options.order == "pre" and should_apply:
            self.visit_graph_object(obj, ctx)

        self.visit(cdef.args, ctx.child("args"))
        self.visit(cdef.kwargs, ctx.child("kwargs"))

        if self.options.order == "post" and should_apply:
            self.visit_graph_object(obj, ctx)

    def visit_concrete_definition(self, obj: ConcreteDefinition, ctx: GraphCtx) -> None:
        linked_obj = self._resolve_cdef(obj, ctx)
        if linked_obj is not None:
            self.visit(linked_obj, ctx)

    def visit_graph_object(self, obj: Object, ctx: GraphCtx) -> None:
        pass


class RepoGraphCollectVisitor(RepoObjectGraphVisitor):
    def __init__(self, repo, *, options: RepoGraphOptions | None = None, **kwargs):
        super().__init__(repo, options=options, **kwargs)
        self.objects: list[Object] = []

    def visit_graph_object(self, obj: Object, ctx: GraphCtx) -> None:
        self.objects.append(obj)


class RepoGraphApplyVisitor(RepoObjectGraphVisitor):
    def __init__(self, repo, func, *, options: RepoGraphOptions | None = None, **kwargs):
        super().__init__(repo, options=options, **kwargs)
        self.func = func
        self.results: dict[ConcreteDefinition, Any] = {}

    def visit_graph_object(self, obj: Object, ctx: GraphCtx) -> None:
        self.results[obj.definition] = self.func(obj)


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
            from .repo import RepoSaveError

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
            self.repo.set_object_store(obj, self.store)
        elif obj.definition not in self.repo.obj_default_store:
            if self.repo.default_store is not None:
                self.repo.set_object_store(obj, self.repo.default_store)
