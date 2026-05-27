from __future__ import annotations

from typing import Any, Generic, Iterator, TypeVar

from dryml.core2 import Object
from dryml.core2.cardinality import Cardinality
from dryml.core2.utils.recurse import iter_leaves, map_leaf_groups
from dryml.core2.tensor_spec import as_tensor_spec, Dim, DimLike, SpecHint, SpecTree, TensorSpec


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

    def _detect_spec(self, hint: SpecHint):
        num_samples = hint.samples
        it = iter(self)
        batched = (hint.batch_mode.value == "batched")
        specs = []
        for _ in range(num_samples):
            element = next(it)
            el_spec = as_tensor_spec(element, batched=batched)
            specs.append(el_spec)

        spec_leaves_list = [ list(iter_leaves(spec)) for spec in specs ]

        if len(set(map(len, spec_leaves_list))) > 1:
            raise ValueError("Dataset yields elements with an inconsistent structure!")

        def _normalize_specs(spec_list):
            ranks = [ spec.rank for spec in spec_list ]
            if len(set(ranks)) > 1:
                raise ValueError("Inconsistent Tensor dimension count")

            rank = ranks.pop()

            def dim_process(dim_list) -> DimLike:
                dim_set = set(dim_list)
                if len(dim_set) > 1:
                    return Dim.DYNAMIC
                else:
                    return dim_set.pop()

            batch_dims = [ spec.batch for spec in spec_list ]
            batch = dim_process(batch_dims)

            shape = None
            if rank is not None:
                shape = []
                for dim in range(rank):
                    dim_vals = [ spec.shape[dim] for spec in spec_list ]
                    shape.append(dim_process(dim_vals))

            def uniform_value(val_list) -> Any:
                val_set = set(val_list)
                if len(val_set) > 1:
                    raise ValueError("Not all tensors have the same property values")
                return val_set.pop()

            dtype = uniform_value([spec.dtype for spec in spec_list])
            layout = uniform_value([spec.layout for spec in spec_list])
            ragged_rank = uniform_value([spec.ragged_rank for spec in spec_list])
            row_splits_dtype = uniform_value([spec.row_splits_dtype for spec in spec_list])
            sparse_format = uniform_value([spec.sparse_format for spec in spec_list])
            axis_names = uniform_value([spec.axis_names for spec in spec_list])
            batch_axis_name = uniform_value([spec.batch_axis_name for spec in spec_list])

            backends = [ spec.backend for spec in spec_list if spec.backend is not None ]
            backend = uniform_value(backends) if backends else None

            return TensorSpec(
                dtype=dtype,
                shape=shape,
                batch=batch,
                backend=backend,
                layout=layout,
                ragged_rank=ragged_rank,
                row_splits_dtype=row_splits_dtype,
                sparse_format=sparse_format,
                axis_names=axis_names,
                batch_axis_name=batch_axis_name)

        return map_leaf_groups(specs, _normalize_specs)


class Map(Dataset):
    """Dataset node that applies one transform to each source element."""

    def __init__(self, src: Dataset, *transforms):
        if not transforms:
            raise ValueError("Map requires at least one transform.")

        if len(transforms) == 1:
            transform = transforms[0]
        else:
            from dryml.data.transforms.elementwise import Pipe
            transform = Pipe(*transforms)

        self.src = src
        self.transform = transform
        super().__init__(spec=transform.infer_output_spec(src.spec))

    def __iter__(self):
        it = iter(self.src)
        try:
            first = next(it)
        except StopIteration:
            return

        impl, first_out = self.transform.bind_first(first, input_spec=self.src.spec)
        yield first_out
        for item in it:
            yield impl(item)

    def __len__(self) -> Cardinality:
        return self.src.__len__()


class ElementwiseDataset(Map):
    """Compatibility alias for the public Map dataset node."""


class StructuralDataset(Dataset):
    """Dataset node for one-input transforms that may change structure/cardinality."""

    def __init__(self, src: Dataset, transform):
        self.src = src
        self.transform = transform
        super().__init__(spec=transform.infer_output_spec(src.spec))

    def __iter__(self):
        yield from self.transform.iter_dataset(self.src)

    def __len__(self) -> Cardinality:
        if hasattr(self.transform, "infer_cardinality"):
            return self.transform.infer_cardinality(self.src.__len__())
        return Cardinality.UNKNOWN


class CombineDataset(Dataset):
    """Dataset node for transforms that consume multiple source datasets."""

    def __init__(self, sources: tuple[Dataset, ...], transform):
        self.sources = tuple(sources)
        self.transform = transform
        super().__init__(spec=transform.infer_output_spec(*(src.spec for src in self.sources)))

    def __iter__(self):
        yield from self.transform.iter_datasets(*self.sources)

    def __len__(self) -> Cardinality:
        if hasattr(self.transform, "infer_cardinality"):
            return self.transform.infer_cardinality(*(src.__len__() for src in self.sources))
        return Cardinality.UNKNOWN
