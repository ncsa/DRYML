from __future__ import annotations

from dryml.code import Method
from dryml.core2.tensor_spec import SpecTree


class Transform(Method):
    """Base class for dataset transforms with spec inference."""

    def infer_output_spec(self, *input_specs: SpecTree) -> SpecTree:
        raise NotImplementedError(
            f"{type(self).__name__}.infer_output_spec is not implemented."
        )


class ElementwiseTransform(Transform):
    """Transform applied independently to each dataset element."""


class StructuralTransform(Transform):
    """Transform that consumes one dataset and may change structure/cardinality."""

    def iter_dataset(self, src):
        raise NotImplementedError(
            f"{type(self).__name__}.iter_dataset is not implemented."
        )


class CombineTransform(Transform):
    """Transform that consumes multiple source datasets."""

    def iter_datasets(self, *sources):
        raise NotImplementedError(
            f"{type(self).__name__}.iter_datasets is not implemented."
        )
