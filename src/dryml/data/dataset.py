from __future__ import annotations

from typing import Generic, Iterator, TypeVar

from dryml.core import Object
from dryml.core.cardinality import Cardinality
from dryml.core.tensor_spec import SpecTree


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
    """Dataset node that applies one Method to each source element."""

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
        it = iter(self.src)
        try:
            first = next(it)
        except StopIteration:
            return

        impl, first_out = self.method.bind_first(first, input_spec=self.src.spec)
        yield first_out
        for item in it:
            yield impl(item)

    def __len__(self) -> Cardinality:
        return self.src.__len__()
