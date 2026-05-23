from __future__ import annotations

from typing import Generic, Iterator, TypeVar

from dryml.core2 import Object
from dryml.core2.cardinality import Cardinality
from dryml.core2.utils.recurse import iter_leaves, map_leaf_groups
from dryml.core2.tensor_spec import as_tensor_spec, Dim, DimLike


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

    def __init__(self):
        super().__init__()

    @property
    def spec(self) -> TensorSpec:
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
        from dryml.core2.tensor_spec import TensorSpec
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
            num_dims = [ len(spec.shape) for spec in spec_list ]
            if len(set(num_dims)) > 1:
                raise ValueError("Inconsistent Tensor dimension count")

            num_dims = num_dims.pop()

            def dim_process(dim_list) -> DimLike:
                dim_set = set(dim_list)
                if len(dim_set) > 1:
                    return Dim.DYNAMIC
                else:
                    return dim_set.pop()

            batch_dims = [ spec.batch for spec in spec_list ]
            batch = dim_process(batch_dims)

            shape = []
            for dim in range(num_dims):
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
            backend = uniform_value(backends)

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
