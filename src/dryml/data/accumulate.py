"""Streaming post-transition carry over observation-first Accumulator Methods."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TypeVar

from dryml.core.cardinality import Cardinality
from dryml.methods import Accumulator, Method
from .dataset import Dataset

T = TypeVar("T")
StateT = TypeVar("StateT")


class Accumulate(Dataset[StateT]):
    """Emit one post-transition carry per source observation, without finalizing.

    Args:
        src: Re-iterable Dataset of observations of type ``T``.
        initial_state: Unary Method ``(T) -> StateT`` allocating fresh carry from
            the first real observation. Empty input never calls it.
        accumulator: Observation-first Method ``(T, StateT) -> StateT``. The
            first observation also passes through this transition exactly once.

    Carry belongs to each live occurrence, not either Method. Explicit sections
    retain carry; Repeat reopening this occurrence and implicit full passes reset
    it. Mutable supported emissions are detached from live carry; immutable and
    native-functional leaves share safely without host conversion. Prepared plans
    retain both invokers, and preparation never acquires a source or calls a body.
    Capture validates/detaches carry on request; unsupported dynamic values raise
    UnsupportedContinuationError without invalidating live accepted progress or
    older records. Restoration installs carry without replaying transitions.

    Raises:
        TypeError: Invalid Dataset, Method, or Accumulator roles.
        ImplementationSelectionError: Incompatible declared role specs during
            preparation, before source acquisition, or runtime role validation.
        Exception: Source, inference, transition, and cleanup failures propagate.
    """

    _stream_operator = "accumulate"

    def __init__(self, src: Dataset[T], *, initial_state: Method,
                 accumulator: Accumulator[T, StateT]) -> None:
        """Retain roles and infer carry metadata without running either body."""
        if not isinstance(src, Dataset):
            raise TypeError("Accumulate src must be a Dataset.")
        if not isinstance(initial_state, Method):
            raise TypeError("Accumulate initial_state must be a Method.")
        if not isinstance(accumulator, Accumulator):
            raise TypeError("Accumulate accumulator must be an Accumulator.")
        self.src = src
        self.initial_state = initial_state
        self.accumulator = accumulator
        super().__init__(spec=initial_state.infer_output_spec(src.spec))

    def __iter__(self) -> Iterator[StateT]:
        """Return an independent flattened physical pass with lazy acquisition.

        Source-inert preparation selects both roles; selection/inference errors
        propagate before consumption. Unqualified custom operator compositions
        raise NotImplementedError. Close the returned cursor to release an
        unfinished pass. Closure alone does not certify root EOF.
        """
        graph = self.method_graph()
        graph.learn()
        return graph.iterator()

    def __len__(self) -> Cardinality:
        """Return source yield cardinality without acquiring or initializing it."""
        return self.src.yield_cardinality()


__all__ = ["Accumulate"]
