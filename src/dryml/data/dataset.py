from __future__ import annotations

from typing import Generic, Iterator, TypeVar

from dryml.core import Object
from dryml.core.backend import discover_backends
from dryml.core.cardinality import Cardinality
from dryml.core.tensor_spec import SpecTree
from dryml.methods import ImplementationSelectionError


T = TypeVar("T")


class Dataset(Object, Generic[T]):
    """
    Base iterable dataset.

    Notes
    -----
    - A Dataset should be re-iterable: calling iter(ds) multiple times should
      produce fresh iterators.
    - `spec` describes what one yielded element looks like.
    """

    def __init__(self, spec: SpecTree | None = None):
        super().__init__()
        self._spec = spec

    @property
    def spec(self) -> SpecTree:
        if self._spec is None:
            raise ValueError(f"{type(self).__name__} has no known spec.")
        return self._spec

    def __iter__(self) -> Iterator[T]:
        raise NotImplementedError

    def peek(self) -> T:
        """
        Return one element from the dataset without mutating long-term dataset
        state, assuming the dataset is re-iterable.
        """
        it = iter(self)
        try:
            return next(it)
        except StopIteration as e:
            raise ValueError("Cannot peek an empty dataset.") from e

    def __len__(self) -> Cardinality:
        """
        Override in subclasses when cardinality is known.
        """
        raise NotImplementedError("Subclasses must implement their lengths")


class Map(Dataset):
    """Dataset node that applies one selected Method callable per source element.

    A complete source specification selects the callable before source
    consumption. When backend alone is unknown, the iterator contributes one
    value at most to complete that constraint. Other selection failures propagate
    without source consumption.
    """

    def __init__(self, src: Dataset, *methods):
        if not methods:
            raise ValueError("Map requires at least one Method.")

        if len(methods) == 1:
            method = methods[0]
        else:
            from dryml.data.methods import Pipe
            method = Pipe(*methods)

        self.src = src
        self.method = method
        super().__init__(spec=method.infer_output_spec(src.spec))

    def __iter__(self):
        try:
            implementation = self.method.find_implementation(
                input_spec=self.src.spec,
            )
        except ImplementationSelectionError as error:
            if (
                error.reason != "unknown_traits"
                or error.unknown_traits != ("backend",)
            ):
                raise
            it = iter(self.src)
            try:
                first = next(it)
            except StopIteration:
                return
            backends = discover_backends(first)
            if len(backends) != 1:
                if len(backends) > 1:
                    raise ImplementationSelectionError("conflict")
                raise error
            implementation = self.method.find_implementation(
                input_spec=self.src.spec,
                backend=next(iter(backends)),
            )
            yield implementation(first)
        else:
            it = iter(self.src)
        for item in it:
            yield implementation(item)

    def __len__(self) -> Cardinality:
        return self.src.__len__()
