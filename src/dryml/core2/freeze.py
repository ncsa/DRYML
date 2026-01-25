from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Mapping
from collections.abc import Mapping as ABCMapping
import numpy as np


# -----------------------------
# Frozen container tags
# -----------------------------

class FrozenList(tuple):
    """Immutable representation of a Python list (tagged for thaw to list)."""
    __slots__ = ()
    def __new__(cls, items: Iterable[Any]):
        return super().__new__(cls, tuple(items))


#class FrozenTuple(tuple):
#    """Tagged tuple for symmetric thaw handling."""
#    __slots__ = ()
#    def __new__(cls, items: Iterable[Any]):
#        return super().__new__(cls, tuple(items))


class FrozenSet(frozenset):
    """Immutable representation of a Python set (tagged for thaw to set)."""
    __slots__ = ()
    def __new__(cls, items: Iterable[Any]):
        return super().__new__(cls, items)


class FrozenDict(ABCMapping):
    """
    Immutable mapping, not a dict subclass (important: boltons.remap default_enter
    will build a plain dict for dictlikes, avoiding 'can’t populate an immutable dict').
    """
    __slots__ = ("_items", "_dict")

    def __init__(self, items: Mapping[Any, Any] | Iterable[tuple[Any, Any]]):
        if isinstance(items, ABCMapping):
            items = items.items()
        items_t = tuple(items)
        self._items = items_t
        self._dict = dict(items_t)

    def __len__(self) -> int:
        return len(self._dict)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._dict)

    def __getitem__(self, k: Any) -> Any:
        return self._dict[k]

    def items(self):
        return self._dict.items()

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()

    def __repr__(self) -> str:
        return f"FrozenDict({self._dict!r})"


class FrozenNDArray(np.ndarray):
    """ndarray subclass tag; leaf-hashed like ndarray; thawed per policy."""
    __slots__ = ("_frozen_writeable",)

    @staticmethod
    def from_array(a: np.ndarray, *, copy: bool = True, writeable: bool = False) -> "FrozenNDArray":
        arr = np.array(a, copy=copy)
        try:
            arr.setflags(write=writeable)
        except Exception:
            pass
        out = arr.view(FrozenNDArray)
        out._frozen_writeable = bool(writeable)
        return out

    def thaw(self):
        return np.array(self, copy=True)


frozen_container_types = (FrozenList, tuple, FrozenSet, FrozenDict)


# -----------------------------
# Deep freeze / thaw
# -----------------------------

def deep_freeze(
    x: Any,
    *,
    path: tuple[str, ...] = (),
    memo: dict[int, Any] | None = None,
    stack: set[int] | None = None,
    # numpy policy
    copy_numpy: bool = True,
    freeze_numpy_writeable: bool = False,
    # DRYML policy (duck-typed to avoid import cycles)
    allow_concrete_definition: bool = True,
    allow_definition: bool = False,
    allow_object: bool = False,
) -> Any:
    """
    Deep-freeze supported containers into tagged immutable equivalents.

    - list      -> FrozenList
    - dict/mapping -> FrozenDict
    - set       -> FrozenSet
    - frozenset -> frozenset (already immutable)
    - ndarray   -> FrozenNDArray (copied; writeability configurable)
    - POD scalars pass through

    Preserves aliasing within DAGs via memo (for mutable containers).
    Rejects cycles.
    """
    if memo is None:
        memo = {}
    if stack is None:
        stack = set()

    # DRYML awareness (duck typing)
    tname = type(x).__name__
    if tname == "ConcreteDefinition":
        if not allow_concrete_definition:
            raise FreezeError(path, x, "ConcreteDefinition not allowed in deep_freeze")
        return x
    if tname == "Definition":
        if not allow_definition:
            raise FreezeError(path, x, "Definition not allowed in deep_freeze (concretize first)")
        return x
    if tname == "Object":
        if not allow_object:
            raise FreezeError(path, x, "Object not allowed in deep_freeze (use its ConcreteDefinition)")
        return x

    if _is_pod(x):
        return x

    if isinstance(x, np.ndarray):
        oid = id(x)
        if oid in memo:
            return memo[oid]
        if oid in stack:
            raise CycleError(path)
        stack.add(oid)
        out = FrozenNDArray.from_array(x, copy=copy_numpy, writeable=freeze_numpy_writeable)
        memo[oid] = out
        stack.remove(oid)
        return out

    if isinstance(x, dict) or isinstance(x, ABCMapping):
        oid = id(x)
        if oid in memo:
            return memo[oid]
        if oid in stack:
            raise CycleError(path)
        stack.add(oid)

        tmp_items: list[tuple[Any, Any]] = []
        for k, v in (x.items() if hasattr(x, "items") else x):
            fk = deep_freeze(
                k, path=path + ("<key>",),
                memo=memo, stack=stack,
                copy_numpy=copy_numpy, freeze_numpy_writeable=freeze_numpy_writeable,
                allow_concrete_definition=allow_concrete_definition,
                allow_definition=allow_definition,
                allow_object=allow_object,
            )
            fv = deep_freeze(
                v, path=path + (str(k),),
                memo=memo, stack=stack,
                copy_numpy=copy_numpy, freeze_numpy_writeable=freeze_numpy_writeable,
                allow_concrete_definition=allow_concrete_definition,
                allow_definition=allow_definition,
                allow_object=allow_object,
            )
            tmp_items.append((fk, fv))

        out = FrozenDict(tmp_items)
        memo[oid] = out
        stack.remove(oid)
        return out

    if isinstance(x, list):
        oid = id(x)
        if oid in memo:
            return memo[oid]
        if oid in stack:
            raise CycleError(path)
        stack.add(oid)

        items = [
            deep_freeze(
                v, path=path + (str(i),),
                memo=memo, stack=stack,
                copy_numpy=copy_numpy, freeze_numpy_writeable=freeze_numpy_writeable,
                allow_concrete_definition=allow_concrete_definition,
                allow_definition=allow_definition,
                allow_object=allow_object,
            )
            for i, v in enumerate(x)
        ]
        out = FrozenList(items)
        memo[oid] = out
        stack.remove(oid)
        return out

    if isinstance(x, tuple):
        items = (
            deep_freeze(
                v, path=path + (str(i),),
                memo=memo, stack=stack,
                copy_numpy=copy_numpy, freeze_numpy_writeable=freeze_numpy_writeable,
                allow_concrete_definition=allow_concrete_definition,
                allow_definition=allow_definition,
                allow_object=allow_object,
            )
            for i, v in enumerate(x)
        )
        return tuple(items)

    if isinstance(x, set):
        oid = id(x)
        if oid in memo:
            return memo[oid]
        if oid in stack:
            raise CycleError(path)
        stack.add(oid)
        items = (
            deep_freeze(
                v, path=path + ("<set>",),
                memo=memo, stack=stack,
                copy_numpy=copy_numpy, freeze_numpy_writeable=freeze_numpy_writeable,
                allow_concrete_definition=allow_concrete_definition,
                allow_definition=allow_definition,
                allow_object=allow_object,
            )
            for v in x
        )
        out = FrozenSet(items)
        memo[oid] = out
        stack.remove(oid)
        return out

    if isinstance(x, frozenset):
        items = (
            deep_freeze(
                v, path=path + ("<frozenset>",),
                memo=memo, stack=stack,
                copy_numpy=copy_numpy, freeze_numpy_writeable=freeze_numpy_writeable,
                allow_concrete_definition=allow_concrete_definition,
                allow_definition=allow_definition,
                allow_object=allow_object,
            )
            for v in x
        )
        return frozenset(items)

    raise FreezeError(path, x)


def deep_thaw(
    x: Any,
    *,
    memo: dict[int, Any] | None = None,
    # numpy policy
    thaw_numpy_copy: bool = True,
    thaw_numpy_writeable: bool = True,
    # DRYML policy
    thaw_concrete_definition_to_definition: bool = False,
) -> Any:
    """
    Reverse deep_freeze, reconstructing original container types as closely as possible.

    - FrozenList  -> list
    - FrozenDict  -> dict
    - FrozenSet   -> set
    - FrozenNDArray -> np.ndarray
    """
    if memo is None:
        memo = {}

    # DRYML awareness (duck type)
    tname = type(x).__name__
    if tname == "ConcreteDefinition":
        if not thaw_concrete_definition_to_definition:
            return x
        from .definition import Definition
        cls = x["cls"]
        args = deep_thaw(x.get("args", ()), memo=memo,
                        thaw_numpy_copy=thaw_numpy_copy,
                        thaw_numpy_writeable=thaw_numpy_writeable,
                        thaw_concrete_definition_to_definition=thaw_concrete_definition_to_definition)
        kwargs = deep_thaw(x.get("kwargs", {}), memo=memo,
                          thaw_numpy_copy=thaw_numpy_copy,
                          thaw_numpy_writeable=thaw_numpy_writeable,
                          thaw_concrete_definition_to_definition=thaw_concrete_definition_to_definition)
        return Definition(cls, *args, **kwargs)

    if _is_pod(x):
        return x

    if isinstance(x, np.ndarray):
        arr = np.array(x, copy=thaw_numpy_copy)
        if thaw_numpy_writeable:
            try:
                arr.setflags(write=True)
            except Exception:
                pass
        return arr

    if isinstance(x, FrozenDict):
        oid = id(x)
        if oid in memo:
            return memo[oid]
        out: dict[Any, Any] = {}
        memo[oid] = out
        for k, v in x.items():
            out[deep_thaw(k, memo=memo,
                          thaw_numpy_copy=thaw_numpy_copy,
                          thaw_numpy_writeable=thaw_numpy_writeable,
                          thaw_concrete_definition_to_definition=thaw_concrete_definition_to_definition)] = \
                deep_thaw(v, memo=memo,
                          thaw_numpy_copy=thaw_numpy_copy,
                          thaw_numpy_writeable=thaw_numpy_writeable,
                          thaw_concrete_definition_to_definition=thaw_concrete_definition_to_definition)
        return out

    if isinstance(x, FrozenList):
        oid = id(x)
        if oid in memo:
            return memo[oid]
        out_list: list[Any] = []
        memo[oid] = out_list
        out_list.extend(deep_thaw(v, memo=memo,
                                 thaw_numpy_copy=thaw_numpy_copy,
                                 thaw_numpy_writeable=thaw_numpy_writeable,
                                 thaw_concrete_definition_to_definition=thaw_concrete_definition_to_definition)
                        for v in x)
        return out_list

    if isinstance(x, tuple):
        oid = id(x)
        if oid in memo:
            return memo[oid]
        out_tup = tuple(deep_thaw(v, memo=memo,
                                 thaw_numpy_copy=thaw_numpy_copy,
                                 thaw_numpy_writeable=thaw_numpy_writeable,
                                 thaw_concrete_definition_to_definition=thaw_concrete_definition_to_definition)
                        for v in x)
        memo[oid] = out_tup
        return out_tup

    if isinstance(x, FrozenSet):
        oid = id(x)
        if oid in memo:
            return memo[oid]
        out_set: set[Any] = set()
        memo[oid] = out_set
        out_set.update(deep_thaw(v, memo=memo,
                                thaw_numpy_copy=thaw_numpy_copy,
                                thaw_numpy_writeable=thaw_numpy_writeable,
                                thaw_concrete_definition_to_definition=thaw_concrete_definition_to_definition)
                       for v in x)
        return out_set

    if isinstance(x, frozenset):
        return frozenset(deep_thaw(v, memo=memo,
                                  thaw_numpy_copy=thaw_numpy_copy,
                                  thaw_numpy_writeable=thaw_numpy_writeable,
                                  thaw_concrete_definition_to_definition=thaw_concrete_definition_to_definition)
                         for v in x)

    # Plain containers (if they slipped through)
    if isinstance(x, dict):
        return {deep_thaw(k, memo=memo): deep_thaw(v, memo=memo) for k, v in x.items()}
    if isinstance(x, list):
        return [deep_thaw(v, memo=memo) for v in x]
    if isinstance(x, tuple):
        return tuple(deep_thaw(v, memo=memo) for v in x)
    if isinstance(x, set):
        return {deep_thaw(v, memo=memo) for v in x}

    return x
