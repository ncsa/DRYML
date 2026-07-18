"""Capability and cursor protocol for exact Dataset pipeline resume."""

from __future__ import annotations

import inspect
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from dryml.core2.definition import ConcreteDefinition, Definition
from dryml.core2.links import DefLink
from dryml.core2.symbol import resolve_symbol
from dryml.core2.tensor_spec import as_tensor_spec, unbatch_spec_tree
from dryml.core2.utils.recurse import iter_leaves

from .dataset import Dataset
from .source import ArrayDataset, GeneratorDataset, _tree_index
from .structural import Shuffle


class ResumeMode(str, Enum):
    """Pipeline continuation guarantee available after durable interruption."""

    EXACT = "exact"
    REPLAY = "replay"
    NONE = "none"


@dataclass(frozen=True, slots=True)
class DatasetResumeCapability:
    """Whole-pipeline resume result produced without materializing its source."""

    mode: ResumeMode
    stages: tuple[str, ...]
    diagnostic: str
    checkpoint_schema: str | None = None

    @property
    def resumable(self) -> bool:
        """Return whether durable cursor restoration is exact."""

        return self.mode is ResumeMode.EXACT


class ResumableDatasetIterator(Iterator[Any]):
    """Iterator whose complete state can be committed at an element boundary."""

    def checkpoint(self) -> Mapping[str, Any]:
        """Return an opaque pickle-compatible exact continuation state."""

        raise NotImplementedError


def dataset_resume_capability(dataset: Dataset | ConcreteDefinition | Definition) -> DatasetResumeCapability:
    """Inspect every known stateful stage without constructing the Dataset.

    Exact capability is conservative. Unknown stages are non-resumable unless
    their class publishes ``__dryml_dataset_resume_capability__``. Replay-only
    means a fresh iterator exists but does not satisfy exact continuation.
    """

    cdef = _as_cdef(dataset)
    return _capability(cdef)


def dataset_definition_metadata(
    dataset: Dataset | ConcreteDefinition | Definition,
):
    """Return element spec and cardinality from definition data only.

    The helper deliberately supports only Dataset nodes whose metadata contract
    is statically derivable. Callers must supply explicit cache metadata for an
    opaque custom node rather than constructing it during definition loading.
    """

    from dryml.core2.cardinality import Cardinality

    cdef = _as_cdef(dataset)
    cls = resolve_symbol(cdef.cls)
    if issubclass(cls, ArrayDataset):
        arrays = _definition_argument(cdef, "arrays")
        spec = _argument_or_default(cdef, "spec")
        batched = _argument_or_default(cdef, "batched")
        if spec is None:
            spec = as_tensor_spec(arrays, batched=batched)
            if batched:
                spec = unbatch_spec_tree(spec)
        lengths = tuple(len(leaf) for leaf in iter_leaves(arrays))
        if not lengths or len(set(lengths)) != 1:
            raise ValueError("ArrayDataset definition has inconsistent leading lengths")
        return spec, Cardinality.finite(lengths[0])
    if issubclass(cls, GeneratorDataset):
        spec = _argument_or_default(cdef, "spec")
        if spec is None:
            raise ValueError("GeneratorDataset cache definitions require explicit element metadata")
        return spec, _argument_or_default(cdef, "cardinality")
    if issubclass(cls, Shuffle):
        return dataset_definition_metadata(_source_cdef(cdef))
    raise ValueError(
        f"Dataset metadata for {cls.__module__}.{cls.__qualname__} is not statically derivable; "
        "pass spec and cardinality explicitly"
    )


def open_resumable_dataset(
    dataset: Dataset,
    state: Mapping[str, Any] | None = None,
) -> ResumableDatasetIterator:
    """Open an exact iterator or reject a pipeline lacking a restore contract."""

    if isinstance(dataset, ArrayDataset):
        return _ArrayIterator(dataset, state)
    if isinstance(dataset, Shuffle):
        return _ShuffleIterator(dataset, state)
    provider = getattr(dataset, "__dryml_open_resumable__", None)
    if provider is not None:
        iterator = provider(state)
        if not isinstance(iterator, ResumableDatasetIterator):
            if not hasattr(iterator, "checkpoint"):
                raise TypeError("custom resumable iterator must expose checkpoint()")
        return iterator
    raise TypeError(
        f"Dataset stage {type(dataset).__name__} has no exact checkpoint/restore implementation"
    )


class _ArrayIterator(ResumableDatasetIterator):
    def __init__(self, dataset: ArrayDataset, state: Mapping[str, Any] | None):
        self.dataset = dataset
        self.index = 0
        if state is not None:
            _require_state(state, "array", {"kind", "index"})
            index = state["index"]
            if type(index) is not int or index < 0 or index > dataset._length:
                raise ValueError("array resume cursor is outside the source")
            self.index = index

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.dataset._length:
            raise StopIteration
        value = _tree_index(self.dataset.arrays, self.index)
        self.index += 1
        return value

    def checkpoint(self) -> Mapping[str, Any]:
        return {"kind": "array", "index": self.index}


class _ShuffleIterator(ResumableDatasetIterator):
    def __init__(self, dataset: Shuffle, state: Mapping[str, Any] | None):
        self.dataset = dataset
        if state is None:
            self.source = open_resumable_dataset(dataset.src)
            self.rng = np.random.default_rng(seed=dataset.seed)
            self.buffer = []
            for _ in range(dataset.buffer_size):
                try:
                    self.buffer.append(next(self.source))
                except StopIteration:
                    break
            return
        _require_state(state, "shuffle", {"kind", "source", "rng", "buffer"})
        if not isinstance(state["source"], Mapping) or not isinstance(state["rng"], Mapping):
            raise ValueError("shuffle resume state is malformed")
        if not isinstance(state["buffer"], list) or len(state["buffer"]) > dataset.buffer_size:
            raise ValueError("shuffle resume buffer is malformed")
        self.source = open_resumable_dataset(dataset.src, state["source"])
        self.rng = np.random.default_rng()
        try:
            self.rng.bit_generator.state = dict(state["rng"])
        except Exception as exc:
            raise ValueError("shuffle RNG resume state is incompatible") from exc
        self.buffer = list(state["buffer"])

    def __iter__(self):
        return self

    def __next__(self):
        if not self.buffer:
            raise StopIteration
        index = int(self.rng.integers(0, len(self.buffer))) if len(self.buffer) > 1 else 0
        value = self.buffer.pop(index)
        try:
            self.buffer.append(next(self.source))
        except StopIteration:
            pass
        return value

    def checkpoint(self) -> Mapping[str, Any]:
        return {
            "kind": "shuffle",
            "source": self.source.checkpoint(),
            "rng": self.rng.bit_generator.state,
            "buffer": list(self.buffer),
        }


def _capability(cdef: ConcreteDefinition) -> DatasetResumeCapability:
    cls = resolve_symbol(cdef.cls)
    name = f"{cls.__module__}.{cls.__qualname__}"
    declared = inspect.getattr_static(cls, "__dryml_dataset_resume_capability__", None)
    if isinstance(declared, staticmethod):
        declared = declared.__func__(cdef)
    elif isinstance(declared, classmethod):
        declared = declared.__func__(cls, cdef)
    elif callable(declared) and not isinstance(declared, DatasetResumeCapability):
        declared = declared(cdef)
    if declared is not None:
        if not isinstance(declared, DatasetResumeCapability):
            raise TypeError("dataset resume capability provider returned an invalid value")
        source = _optional_source_cdef(cdef)
        if source is not None:
            child = _capability(source)
            stages = (*child.stages, *declared.stages)
            if declared.mode is ResumeMode.EXACT and child.mode is not ResumeMode.EXACT:
                return DatasetResumeCapability(
                    child.mode,
                    stages,
                    f"{name} cannot resume exactly because {child.diagnostic}",
                )
            if declared.mode is ResumeMode.NONE:
                return DatasetResumeCapability(
                    ResumeMode.NONE,
                    stages,
                    declared.diagnostic,
                )
            return DatasetResumeCapability(
                declared.mode,
                stages,
                declared.diagnostic,
                declared.checkpoint_schema,
            )
        return declared
    if issubclass(cls, ArrayDataset):
        return DatasetResumeCapability(
            ResumeMode.EXACT,
            (name,),
            "indexed source exposes an exact durable row cursor",
            "dryml.dataset-pipeline.v1",
        )
    if issubclass(cls, GeneratorDataset):
        return DatasetResumeCapability(
            ResumeMode.REPLAY,
            (name,),
            "source can replay from the beginning but has no exact durable cursor",
        )
    if issubclass(cls, Shuffle):
        source = _source_cdef(cdef)
        child = _capability(source)
        if child.mode is ResumeMode.EXACT:
            return DatasetResumeCapability(
                ResumeMode.EXACT,
                (*child.stages, name),
                "source cursor, shuffle RNG, and shuffle buffer are checkpointed",
                "dryml.dataset-pipeline.v1",
            )
        return DatasetResumeCapability(
            child.mode,
            (*child.stages, name),
            f"shuffle cannot resume exactly because {child.diagnostic}",
        )
    return DatasetResumeCapability(
        ResumeMode.NONE,
        (name,),
        f"stateful or unknown stage {name} has no checkpoint contract",
    )


def _source_cdef(cdef: ConcreteDefinition) -> ConcreteDefinition:
    value = _definition_argument(cdef, "src")
    if isinstance(value, DefLink):
        value = value.target
    if not isinstance(value, ConcreteDefinition):
        raise TypeError("dataset pipeline source is not a concrete definition")
    return value


def _optional_source_cdef(cdef: ConcreteDefinition) -> ConcreteDefinition | None:
    try:
        return _source_cdef(cdef)
    except TypeError:
        return None


def _definition_argument(definition: ConcreteDefinition, name: str) -> Any:
    if name in definition.kwargs:
        return definition.kwargs[name]
    cls = resolve_symbol(definition.cls)
    parameters = tuple(inspect.signature(cls.__init__).parameters.values())
    positional = tuple(
        parameter
        for parameter in parameters
        if parameter.name != "self" and parameter.kind in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }
    )
    for index, parameter in enumerate(positional):
        if parameter.name == name and definition.args is not None and index < len(definition.args):
            return definition.args[index]
    raise TypeError(f"dataset definition has no {name!r} argument")


def _argument_or_default(definition: ConcreteDefinition, name: str) -> Any:
    try:
        return _definition_argument(definition, name)
    except TypeError:
        cls = resolve_symbol(definition.cls)
        parameter = inspect.signature(cls.__init__).parameters.get(name)
        if parameter is None or parameter.default is inspect.Parameter.empty:
            raise
        return parameter.default


def _as_cdef(value: Dataset | ConcreteDefinition | Definition) -> ConcreteDefinition:
    if isinstance(value, ConcreteDefinition):
        return value
    if isinstance(value, Definition):
        return value.concretize()
    if isinstance(value, Dataset):
        return value.definition
    raise TypeError("dataset resume inspection requires a Dataset or definition")


def _require_state(state: Mapping[str, Any], kind: str, fields: set[str]) -> None:
    if not isinstance(state, Mapping) or set(state) != fields or state.get("kind") != kind:
        raise ValueError(f"{kind} resume state is malformed")


__all__ = [
    "DatasetResumeCapability",
    "ResumableDatasetIterator",
    "ResumeMode",
    "dataset_resume_capability",
    "dataset_definition_metadata",
    "open_resumable_dataset",
]
