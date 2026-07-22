from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Iterable
import random

from .utils.graph.path import GraphPath
from .utils.graph.value import get_subtree, replace_subtree


@dataclass(frozen=True, slots=True)
class ParameterBinding:
    path: GraphPath
    par: Any


@dataclass(frozen=True, slots=True)
class SearchSpace:
    """Generative interpretation of a Definition template containing Par values."""

    template: Any
    params: tuple[ParameterBinding, ...]

    @classmethod
    def from_def(cls, defn: Any) -> "SearchSpace":
        from .definition import Definition
        from .params import Par
        from .utils.graph.value import iter_value_edges

        if not isinstance(defn, Definition):
            raise TypeError(f"SearchSpace.from_def requires Definition, got {type(defn).__name__}.")
        params: list[ParameterBinding] = []

        def visit(value: Any, path: GraphPath) -> None:
            if isinstance(value, Par):
                params.append(ParameterBinding(path, value))
                return
            for edge in iter_value_edges(value):
                visit(edge.value, path.child(edge.segment))

        visit(defn, GraphPath())
        return cls(defn, tuple(params))

    def sample(self, rng=None):
        rng_obj = rng if rng is not None else random.Random()
        out = self.template
        for binding in self.params:
            if binding.par.generator is None:
                raise ValueError(f"Parameter at {binding.path!s} has no generator.")
            out = replace_subtree(out, binding.path, binding.par.generator.sample(rng_obj))
        return out

    def grid(self) -> Iterable[Any]:
        grids = []
        for binding in self.params:
            if binding.par.generator is None:
                raise ValueError(f"Parameter at {binding.path!s} has no finite grid generator.")
            grids.append(binding.par.generator.grid())
        for combo in product(*grids):
            out = self.template
            for binding, value in zip(self.params, combo):
                out = replace_subtree(out, binding.path, value)
            yield out

    def support_selector(self):
        from .selector import Selector

        out = self.template
        for binding in self.params:
            out = replace_subtree(out, binding.path, binding.par)
        return Selector(out)


def space(defn: Any) -> SearchSpace:
    return SearchSpace.from_def(defn)
